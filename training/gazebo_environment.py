#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gazebo强化学习环境
集成Gazebo仿真与强化学习框架，为人形机器人提供训练环境

功能：
1. 基于Gazebo的机器人控制环境
2. 支持多种强化学习算法
3. 实时仿真控制和传感器反馈
4. 训练进度监控和模型保存
"""

import sys
import os
import logging
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, ClassVar
from dataclasses import dataclass, field
from enum import Enum
import math

# 导入强化学习库
try:
    import gymnasium as gym
    from gymnasium import spaces
    RL_LIBS_AVAILABLE = True
except ImportError as e:
    RL_LIBS_AVAILABLE = False
    print(f"警告: 强化学习库不可用: {e}")

# 导入仿真接口（优先使用统一仿真接口）
try:
    from hardware.unified_simulation import UnifiedSimulation
    UNIFIED_SIMULATION_AVAILABLE = True
    # 向后兼容：检查GazeboSimulation是否也可用
    try:
        from hardware.gazebo_simulation import GazeboSimulation
        GAZEBO_SIMULATION_AVAILABLE = True
    except ImportError:
        GAZEBO_SIMULATION_AVAILABLE = False
    GAZEBO_AVAILABLE = True  # 统一接口可用
except ImportError as e:
    UNIFIED_SIMULATION_AVAILABLE = False
    # 回退到原始Gazebo仿真
    try:
        from hardware.gazebo_simulation import GazeboSimulation
        GAZEBO_SIMULATION_AVAILABLE = True
        GAZEBO_AVAILABLE = True
    except ImportError as e2:
        GAZEBO_SIMULATION_AVAILABLE = False
        GAZEBO_AVAILABLE = False
        print(f"警告: Gazebo仿真不可用（统一接口和原始接口都不可用）: {e2}")

# 导入机器人控制器
try:
    from hardware.robot_controller import RobotJoint, JointState, HardwareManager
    ROBOT_CONTROLLER_AVAILABLE = True
except ImportError as e:
    ROBOT_CONTROLLER_AVAILABLE = False
    print(f"警告: 机器人控制器不可用: {e}")

logger = logging.getLogger(__name__)


class RobotTaskType(Enum):
    """机器人任务类型"""
    STAND_UP = "stand_up"  # 站立
    WALK = "walk"  # 行走
    BALANCE = "balance"  # 平衡
    REACH_TARGET = "reach_target"  # 到达目标位置
    PICK_AND_PLACE = "pick_and_place"  # 拾取和放置
    COMPLEX_MOTION = "complex_motion"  # 复杂动作


@dataclass
class GazeboEnvironmentConfig:
    """Gazebo环境配置"""
    
    # 常量定义
    IMU_DIMENSION: ClassVar[int] = 10  # IMU数据维度：加速度(3) + 角速度(3) + 姿态(4)
    POSITION_DIMENSION: ClassVar[int] = 3  # 位置/速度维度：x, y, z
    ORIENTATION_DIMENSION: ClassVar[int] = 4  # 姿态四元数维度：qx, qy, qz, qw
    
    # 任务配置
    task_type: RobotTaskType = RobotTaskType.STAND_UP
    max_steps: int = 1000
    target_position: Optional[List[float]] = None  # 目标位置 [x, y, z]
    
    # Gazebo配置
    gazebo_world: str = "empty.world"
    robot_model: str = "humanoid"
    gui_enabled: bool = True
    physics_timestep: float = 0.001
    simulation_speed: float = 1.0  # 仿真速度倍率
    use_real_time_sync: bool = False  # 是否使用实时同步（启用时会sleep）
    ros_master_uri: str = "http://localhost:11311"  # ROS主节点URI
    
    # 奖励配置
    reward_scale: float = 1.0
    success_reward: float = 100.0
    step_penalty: float = -0.01
    fall_penalty: float = -50.0
    energy_penalty_coef: float = -0.001  # 能量消耗惩罚系数
    balance_penalty_coef: float = -2.0  # 平衡惩罚系数（姿态角惩罚）
    height_reward_coef: float = 10.0  # 高度奖励系数
    forward_reward_coef: float = 5.0  # 前进奖励系数
    target_distance_reward_coef: float = 2.0  # 目标距离奖励系数
    
    # 奖励阈值和细节参数
    stability_angle_threshold: float = 0.1  # 稳定性角度阈值（弧度），小于此值视为稳定
    stability_reward: float = 0.5  # 稳定性奖励
    lateral_penalty_coef: float = -0.5  # 侧向移动惩罚系数
    walking_target_height: float = 0.5  # 行走任务目标高度（米）
    walking_height_penalty_coef: float = -2.0  # 行走高度惩罚系数
    distance_reward_smoothing: float = 0.1  # 距离奖励平滑项（避免除零）
    success_distance_threshold: float = 0.1  # 成功到达目标的距离阈值（米）
    fall_height_threshold: float = 0.2  # 摔倒高度阈值（米）
    excessive_tilt_threshold: float = math.pi / 4  # 倾斜过大阈值（弧度），π/4≈45度
    excessive_tilt_penalty_multiplier: float = 0.5  # 倾斜过大惩罚乘子（相对于摔倒惩罚）
    
    # 观察空间配置
    include_joint_positions: bool = True
    include_joint_velocities: bool = True
    include_imu_data: bool = True
    include_camera_data: bool = False  # 相机数据会增加观察维度
    
    # 机器人关节配置
    num_joints: int = 25  # 关节数量
    joint_names: Optional[List[str]] = None  # 关节名称列表，如果为None则使用joint_0, joint_1, ...
    joint_position_limits: Tuple[float, float] = (-np.pi/2, np.pi/2)  # 关节位置限制（弧度）
    joint_velocity_limits: Tuple[float, float] = (-5.0, 5.0)  # 关节速度限制（弧度/秒）
    
    # 真实数据配置（当Gazebo不可用时使用）
    simulated_acceleration_range: Tuple[float, float] = (-0.1, 0.1)  # 模拟加速度范围 (m/s²)
    simulated_angular_velocity_range: Tuple[float, float] = (-0.05, 0.05)  # 模拟角速度范围 (rad/s)
    simulated_robot_base_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.5])  # 模拟机器人基准位置 [x, y, z]
    simulated_robot_position_variance: float = 0.1  # 模拟机器人位置变化方差
    simulated_robot_height_variance: float = 0.05  # 模拟机器人高度变化方差
    simulated_joint_effort_range: Tuple[float, float] = (-5.0, 5.0)  # 模拟关节力矩范围
    
    def __post_init__(self):
        """初始化后处理"""
        if self.target_position is None:
            self.target_position = [0.0, 0.0, 0.5]  # 默认目标位置
        
        # 初始化关节名称
        if self.joint_names is None:
            self.joint_names = [f"joint_{i}" for i in range(self.num_joints)]
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典
        
        返回:
            dict: 配置字典
        """
        # 注意：枚举类型需要特殊处理
        return {
            "task_type": self.task_type.value,
            "max_steps": self.max_steps,
            "target_position": self.target_position,
            "gazebo_world": self.gazebo_world,
            "robot_model": self.robot_model,
            "gui_enabled": self.gui_enabled,
            "physics_timestep": self.physics_timestep,
            "simulation_speed": self.simulation_speed,
            "use_real_time_sync": self.use_real_time_sync,
            "ros_master_uri": self.ros_master_uri,
            "reward_scale": self.reward_scale,
            "success_reward": self.success_reward,
            "step_penalty": self.step_penalty,
            "fall_penalty": self.fall_penalty,
            "energy_penalty_coef": self.energy_penalty_coef,
            "balance_penalty_coef": self.balance_penalty_coef,
            "height_reward_coef": self.height_reward_coef,
            "forward_reward_coef": self.forward_reward_coef,
            "target_distance_reward_coef": self.target_distance_reward_coef,
            "stability_angle_threshold": self.stability_angle_threshold,
            "stability_reward": self.stability_reward,
            "lateral_penalty_coef": self.lateral_penalty_coef,
            "walking_target_height": self.walking_target_height,
            "walking_height_penalty_coef": self.walking_height_penalty_coef,
            "distance_reward_smoothing": self.distance_reward_smoothing,
            "success_distance_threshold": self.success_distance_threshold,
            "fall_height_threshold": self.fall_height_threshold,
            "excessive_tilt_threshold": self.excessive_tilt_threshold,
            "excessive_tilt_penalty_multiplier": self.excessive_tilt_penalty_multiplier,
            "include_joint_positions": self.include_joint_positions,
            "include_joint_velocities": self.include_joint_velocities,
            "include_imu_data": self.include_imu_data,
            "include_camera_data": self.include_camera_data,
            "num_joints": self.num_joints,
            "joint_names": self.joint_names,
            "joint_position_limits": self.joint_position_limits,
            "joint_velocity_limits": self.joint_velocity_limits,
            "simulated_acceleration_range": self.simulated_acceleration_range,
            "simulated_angular_velocity_range": self.simulated_angular_velocity_range,
            "simulated_robot_base_position": self.simulated_robot_base_position,
            "simulated_robot_position_variance": self.simulated_robot_position_variance,
            "simulated_robot_height_variance": self.simulated_robot_height_variance,
            "simulated_joint_effort_range": self.simulated_joint_effort_range
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GazeboEnvironmentConfig":
        """从字典创建配置
        
        参数:
            config_dict: 配置字典
            
        返回:
            GazeboEnvironmentConfig: 配置实例
        """
        # 复制字典以避免修改原始数据
        data = config_dict.copy()
        
        # 处理枚举类型
        if "task_type" in data and isinstance(data["task_type"], str):
            data["task_type"] = RobotTaskType(data["task_type"])
        
        # 创建配置实例
        config = cls(**data)
        return config


class GazeboRobotEnvironment(gym.Env):
    """Gazebo机器人强化学习环境
    
    基于Gazebo仿真和gym.Env的机器人控制环境
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config: Optional[GazeboEnvironmentConfig] = None):
        """初始化Gazebo机器人环境
        
        参数:
            config: 环境配置
        """
        super(GazeboRobotEnvironment, self).__init__()
        
        self.config = config or GazeboEnvironmentConfig()
        self.task_type = self.config.task_type
        
        # 初始化随机数生成器（用于可复现性）
        self.np_random = np.random.RandomState()
        self.seed()
        
        # 初始化Gazebo仿真
        self.gazebo = None
        self._init_gazebo()
        
        # 定义观察空间和动作空间
        self._define_spaces()
        
        # 环境状态
        self.current_step = 0
        self.max_steps = self.config.max_steps
        self.episode_reward = 0.0
        self.episode_success = False
        
        # 机器人状态
        self.joint_states: Dict[str, JointState] = {}
        self.imu_data: Optional[Dict[str, float]] = None
        # 使用配置的基准位置作为初始位置
        self.robot_position: List[float] = self.config.simulated_robot_base_position.copy()
        self.robot_orientation: List[float] = [0.0, 0.0, 0.0, 1.0]  # 单位四元数
        
        # 目标状态
        self.target_position = np.array(self.config.target_position, dtype=np.float32)
        
        logger.info(f"Gazebo机器人环境初始化完成，任务类型: {self.task_type.value}")
    
    def _init_gazebo(self):
        """初始化Gazebo仿真"""
        if not GAZEBO_AVAILABLE:
            logger.warning("Gazebo仿真不可用，使用模拟模式")
            self.gazebo = None
            return
        
        try:
            # 优先使用统一仿真接口
            if UNIFIED_SIMULATION_AVAILABLE:
                self.gazebo = UnifiedSimulation(
                    engine="gazebo",
                    engine_config={
                    "ros_master_uri": self.config.ros_master_uri,
                    "gazebo_world": self.config.gazebo_world,
                    "robot_model": self.config.robot_model,
                    "gui_enabled": self.config.gui_enabled,
                    "physics_timestep": self.config.physics_timestep,
                    "simulation_mode": True
                },
                    gui_enabled=self.config.gui_enabled,
                    simulation_mode=True
                )
                logger.info("使用统一Gazebo仿真接口")
            elif GAZEBO_SIMULATION_AVAILABLE:
                # 回退到原始Gazebo仿真
                self.gazebo = GazeboSimulation(
                    ros_master_uri=self.config.ros_master_uri,
                    gazebo_world=self.config.gazebo_world,
                    robot_model=self.config.robot_model,
                    gui_enabled=self.config.gui_enabled,
                    physics_timestep=self.config.physics_timestep,
                    simulation_mode=True
                )
                logger.info("使用原始Gazebo仿真接口（向后兼容）")
            else:
                logger.warning("没有可用的Gazebo仿真接口，使用模拟模式")
                self.gazebo = None
                return
            
            # 连接到Gazebo
            connected = self.gazebo.connect()
            if connected:
                logger.info("Gazebo仿真连接成功")
            else:
                logger.warning("Gazebo仿真连接失败，使用模拟模式")
                self.gazebo = None
        except Exception as e:
            logger.error(f"Gazebo仿真初始化失败: {e}")
            self.gazebo = None
    
    def _define_spaces(self):
        """定义观察空间和动作空间"""
        
        # 动作空间：关节目标位置（归一化到[-1, 1]）
        num_joints = self.config.num_joints
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(num_joints,), 
            dtype=np.float32
        )
        
        # 观察空间：关节位置、速度、IMU数据等
        obs_dim = 0
        
        if self.config.include_joint_positions:
            obs_dim += num_joints  # 关节位置
        
        if self.config.include_joint_velocities:
            obs_dim += num_joints  # 关节速度
        
        if self.config.include_imu_data:
            obs_dim += GazeboEnvironmentConfig.IMU_DIMENSION  # IMU数据：加速度(3)、角速度(3)、姿态(4)
        
        # 如果包含相机数据，维度会非常大，这里先不包含
        # 可以考虑使用特征提取网络处理相机数据
        
        # 目标位置（相对位置）
        obs_dim += GazeboEnvironmentConfig.POSITION_DIMENSION  # 目标相对位置 [dx, dy, dz]
        
        # 机器人速度
        obs_dim += GazeboEnvironmentConfig.POSITION_DIMENSION  # 线速度 [vx, vy, vz]
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        logger.debug(f"观察空间维度: {obs_dim}, 动作空间维度: {num_joints}")
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """设置随机种子
        
        参数:
            seed: 随机种子，如果为None则使用随机种子
            
        返回:
            list: 包含种子的列表
        """
        if seed is not None:
            self.np_random.seed(seed)
            # 同时设置Python random模块的种子（用于向后兼容）
            import random
            random.seed(seed)
            np.random.seed(seed)
        else:
            # 使用随机种子
            seed = np.random.randint(0, 2**31 - 1)
            self.np_random.seed(seed)
            import random
            random.seed(seed)
            np.random.seed(seed)
        
        logger.debug(f"环境随机种子设置为: {seed}")
        return [seed]
    
    def reset(self, **kwargs):
        """重置环境状态
        
        返回:
            observation: 初始观察
        """
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_success = False
        
        # 重置Gazebo仿真
        if self.gazebo is not None:
            try:
                self.gazebo.reset()
                # 只有在需要实时同步时才sleep
                if self.config.use_real_time_sync:
                    time.sleep(0.1)  # 等待重置完成
            except Exception as e:
                logger.error(f"Gazebo重置失败: {e}")
        
        # 获取初始状态
        self._update_robot_state()
        
        # 生成观察
        observation = self._get_observation()
        
        return observation
    
    def step(self, action: np.ndarray):
        """执行一步动作
        
        参数:
            action: 动作向量
            
        返回:
            observation: 新的观察
            reward: 奖励值
            done: 是否结束
            info: 附加信息
        """
        self.current_step += 1
        
        # 执行动作（控制机器人关节）
        self._apply_action(action)
        
        # 更新仿真（如果是Gazebo仿真）
        if self.gazebo is not None:
            try:
                self.gazebo.step()
                # 只有在需要实时同步时才sleep
                if self.config.use_real_time_sync:
                    time.sleep(self.config.physics_timestep * self.config.simulation_speed)
            except Exception as e:
                logger.error(f"Gazebo步进失败: {e}")
        else:
            # 模拟仿真步进
            if self.config.use_real_time_sync:
                time.sleep(0.01)
        
        # 更新机器人状态
        self._update_robot_state()
        
        # 计算奖励
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # 检查是否结束
        done = self._check_done()
        
        # 获取观察
        observation = self._get_observation()
        
        # 收集信息
        info = {
            "step": self.current_step,
            "reward": reward,
            "episode_reward": self.episode_reward,
            "success": self.episode_success,
            "robot_position": self.robot_position.copy(),
            "target_position": self.target_position.tolist(),
            "distance_to_target": np.linalg.norm(self.target_position - np.array(self.robot_position))
        }
        
        return observation, reward, done, info
    
    def _apply_action(self, action: np.ndarray):
        """应用动作到机器人
        
        参数:
            action: 动作向量（归一化到[-1, 1]）
        """
        # 验证动作维度
        if len(action) != self.config.num_joints:
            logger.warning(f"动作维度不匹配: 预期{self.config.num_joints}, 实际{len(action)}")
            # 裁剪或填充动作向量
            if len(action) > self.config.num_joints:
                action = action[:self.config.num_joints]
            else:
                # 填充零
                action = np.pad(action, (0, self.config.num_joints - len(action)), mode='constant')
        
        # 将归一化的动作转换为关节目标角度（弧度）
        # 使用配置的关节位置限制进行线性映射
        joint_min, joint_max = self.config.joint_position_limits
        joint_targets = (action + 1.0) / 2.0 * (joint_max - joint_min) + joint_min
        
        if self.gazebo is not None:
            # 如果有Gazebo仿真，发送关节控制命令
            try:
                # 使用配置的关节名称映射
                joint_dict = {}
                for i in range(self.config.num_joints):
                    joint_name = self.config.joint_names[i]
                    joint_dict[joint_name] = float(joint_targets[i])
                
                self.gazebo.set_joint_positions(joint_dict)
            except Exception as e:
                logger.error(f"应用关节控制失败: {e}")
        else:
            # 模拟模式：更新关节状态
            for i in range(self.config.num_joints):
                joint_name = self.config.joint_names[i]
                if joint_name not in self.joint_states:
                    self.joint_states[joint_name] = JointState(
                        name=joint_name,
                        position=0.0,
                        velocity=0.0,
                        effort=0.0
                    )
                self.joint_states[joint_name].position = float(joint_targets[i])
    
    def _update_robot_state(self):
        """更新机器人状态"""
        if self.gazebo is not None:
            self._update_from_gazebo()
        else:
            self._update_simulated_state()
    
    def _update_from_gazebo(self):
        """从Gazebo仿真获取真实状态"""
        try:
            # 获取关节状态
            joint_states = self.gazebo.get_joint_states()
            self.joint_states = joint_states
            
            # 获取IMU数据
            imu_data = self.gazebo.get_imu_data()
            self.imu_data = imu_data
            
            # 获取机器人位置（简化）
            # 实际应从Gazebo获取机器人位姿
            # 暂时使用配置的基准位置，后续应从Gazebo获取真实位姿
            self.robot_position = self.config.simulated_robot_base_position.copy()
            
        except Exception as e:
            logger.error(f"获取机器人状态失败: {e}")
    
    def _update_simulated_state(self):
        """更新模拟状态（无Gazebo时使用）"""
        # 使用可复现的随机数生成器
        rng = self.np_random
        
        # 获取配置参数
        joint_min, joint_max = self.config.joint_position_limits
        vel_min, vel_max = self.config.joint_velocity_limits
        effort_min, effort_max = self.config.simulated_joint_effort_range
        acc_min, acc_max = self.config.simulated_acceleration_range
        ang_min, ang_max = self.config.simulated_angular_velocity_range
        base_pos = self.config.simulated_robot_base_position
        pos_var = self.config.simulated_robot_position_variance
        height_var = self.config.simulated_robot_height_variance
        
        # 模拟关节状态
        for i in range(self.config.num_joints):
            joint_name = self.config.joint_names[i]
            self.joint_states[joint_name] = JointState(
                name=joint_name,
                position=rng.uniform(joint_min, joint_max),
                velocity=rng.uniform(vel_min, vel_max),
                effort=rng.uniform(effort_min, effort_max)
            )
        
        # 模拟IMU数据
        self.imu_data = {
            "acceleration": [rng.uniform(acc_min, acc_max) for _ in range(3)],
            "angular_velocity": [rng.uniform(ang_min, ang_max) for _ in range(3)],
            "orientation": [0.0, 0.0, 0.0, 1.0]  # 单位四元数
        }
        
        # 模拟机器人位置（基于基准位置的随机小幅度变化）
        self.robot_position = [
            base_pos[0] + rng.uniform(-pos_var, pos_var),
            base_pos[1] + rng.uniform(-pos_var, pos_var),
            base_pos[2] + rng.uniform(-height_var, height_var)
        ]
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察
        
        返回:
            observation: 观察向量
        """
        observation = []
        
        # 关节位置
        if self.config.include_joint_positions:
            joint_positions = []
            for i in range(self.config.num_joints):
                joint_name = self.config.joint_names[i]
                # 安全获取关节位置，提供缺省值
                joint_state = self.joint_states.get(joint_name)
                if joint_state is None:
                    # 创建缺省关节状态
                    joint_state = JointState(joint_name, 0.0, 0.0, 0.0)
                    self.joint_states[joint_name] = joint_state
                # 限制关节位置在合理范围内
                pos = np.clip(joint_state.position, 
                             self.config.joint_position_limits[0],
                             self.config.joint_position_limits[1])
                joint_positions.append(pos)
            observation.extend(joint_positions)
        
        # 关节速度
        if self.config.include_joint_velocities:
            joint_velocities = []
            for i in range(self.config.num_joints):
                joint_name = self.config.joint_names[i]
                joint_state = self.joint_states.get(joint_name)
                if joint_state is None:
                    joint_state = JointState(joint_name, 0.0, 0.0, 0.0)
                    self.joint_states[joint_name] = joint_state
                # 限制关节速度在合理范围内
                vel = np.clip(joint_state.velocity,
                             self.config.joint_velocity_limits[0],
                             self.config.joint_velocity_limits[1])
                joint_velocities.append(vel)
            observation.extend(joint_velocities)
        
        # IMU数据
        if self.config.include_imu_data:
            if self.imu_data:
                # 加速度（安全获取）
                accel = self.imu_data.get("acceleration", [0.0, 0.0, 0.0])
                # 限制加速度在合理范围内
                accel = np.clip(accel, -20.0, 20.0).tolist()
                observation.extend(accel)
                
                # 角速度（安全获取）
                gyro = self.imu_data.get("angular_velocity", [0.0, 0.0, 0.0])
                gyro = np.clip(gyro, -10.0, 10.0).tolist()
                observation.extend(gyro)
                
                # 姿态四元数（安全获取）
                orientation = self.imu_data.get("orientation", [0.0, 0.0, 0.0, 1.0])
                # 归一化四元数
                orient_norm = np.linalg.norm(orientation)
                if orient_norm > 0:
                    orientation = (np.array(orientation) / orient_norm).tolist()
                observation.extend(orientation)
            else:
                # 无IMU数据，提供缺省值
                observation.extend([0.0, 0.0, 0.0])  # 加速度
                observation.extend([0.0, 0.0, 0.0])  # 角速度
                observation.extend([0.0, 0.0, 0.0, 1.0])  # 姿态
        
        # 目标相对位置
        target_rel = self.target_position - np.array(self.robot_position)
        # 限制相对位置在合理范围内
        target_rel = np.clip(target_rel, -10.0, 10.0)
        observation.extend(target_rel.tolist())
        
        # 机器人速度（简化，使用上一帧的位置差）
        # 这里简化为零向量，实际应计算真实速度
        observation.extend([0.0, 0.0, 0.0])
        
        # 转换为numpy数组
        obs_array = np.array(observation, dtype=np.float32)
        
        # 维度校验
        expected_dim = self.observation_space.shape[0]
        actual_dim = obs_array.shape[0]
        if expected_dim != actual_dim:
            logger.error(f"观察维度不匹配: 预期{expected_dim}, 实际{actual_dim}")
            # 调整维度以匹配预期
            if actual_dim < expected_dim:
                # 填充零
                obs_array = np.pad(obs_array, (0, expected_dim - actual_dim), mode='constant')
            else:
                # 裁剪
                obs_array = obs_array[:expected_dim]
        
        return obs_array
    
    def _compute_orientation_angles(self) -> Tuple[float, float, float]:
        """计算机器人姿态角（俯仰、横滚、偏航）
        
        从IMU数据的四元数转换为欧拉角（弧度）
        
        返回:
            pitch: 俯仰角（前后倾斜，绕x轴）
            roll: 横滚角（左右倾斜，绕y轴）
            yaw: 偏航角（绕z轴）
        """
        if not self.imu_data or "orientation" not in self.imu_data:
            return 0.0, 0.0, 0.0
        
        q = self.imu_data["orientation"]
        if len(q) != 4:
            return 0.0, 0.0, 0.0
        
        # 四元数转换为欧拉角（Z-Y-X顺序，即偏航-俯仰-横滚）
        # 使用标准转换公式
        w, x, y, z = q[3], q[0], q[1], q[2]
        
        # 计算俯仰角（pitch）
        sin_pitch = 2.0 * (w * x + y * z)
        cos_pitch = 1.0 - 2.0 * (x * x + y * y)
        pitch = math.atan2(sin_pitch, cos_pitch)
        
        # 计算横滚角（roll）
        sin_roll = 2.0 * (w * y - z * x)
        if abs(sin_roll) >= 1.0:
            roll = math.copysign(math.pi / 2, sin_roll)
        else:
            roll = math.asin(sin_roll)
        
        # 计算偏航角（yaw）
        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(sin_yaw, cos_yaw)
        
        return pitch, roll, yaw
    
    def _calculate_reward(self) -> float:
        """计算奖励
        
        返回:
            reward: 奖励值
        """
        reward = 0.0
        
        # 基础步数惩罚
        reward += self.config.step_penalty
        
        # 计算姿态角（用于平衡惩罚）
        pitch, roll, yaw = self._compute_orientation_angles()
        
        # 平衡惩罚：惩罚较大的俯仰和横滚角
        balance_penalty = (pitch ** 2 + roll ** 2) * self.config.balance_penalty_coef
        reward += balance_penalty
        
        # 任务相关奖励
        if self.task_type == RobotTaskType.STAND_UP:
            # 站立任务：奖励高度和稳定性
            height = self.robot_position[2]
            reward += height * self.config.height_reward_coef
            
            # 额外奖励：保持直立（小角度）
            if (abs(pitch) < self.config.stability_angle_threshold and 
                abs(roll) < self.config.stability_angle_threshold):
                reward += self.config.stability_reward
        
        elif self.task_type == RobotTaskType.WALK:
            # 行走任务：奖励前进距离和稳定性
            forward_distance = self.robot_position[1]  # 假设y轴是前进方向
            reward += forward_distance * self.config.forward_reward_coef
            
            # 惩罚侧向移动（x轴）和高度变化（z轴）
            lateral_penalty = abs(self.robot_position[0]) * self.config.lateral_penalty_coef
            height_penalty = abs(self.robot_position[2] - self.config.walking_target_height) * self.config.walking_height_penalty_coef
            reward += lateral_penalty + height_penalty
        
        elif self.task_type == RobotTaskType.REACH_TARGET:
            # 到达目标：奖励接近目标
            distance = np.linalg.norm(self.target_position - np.array(self.robot_position))
            # 使用指数衰减奖励，越近奖励越高
            distance_reward = self.config.target_distance_reward_coef / (distance + self.config.distance_reward_smoothing)
            reward += distance_reward
            
            # 如果到达目标
            if distance < self.config.success_distance_threshold:
                reward += self.config.success_reward
                self.episode_success = True
        
        # 能量消耗惩罚（关节速度的平方和）
        energy_penalty = 0.0
        for joint_state in self.joint_states.values():
            energy_penalty += joint_state.velocity ** 2
        reward += energy_penalty * self.config.energy_penalty_coef
        
        # 摔倒惩罚（如果高度太低或倾斜角度过大）
        if self.robot_position[2] < self.config.fall_height_threshold:
            reward += self.config.fall_penalty
        
        # 倾斜过大惩罚
        if (abs(pitch) > self.config.excessive_tilt_threshold or 
            abs(roll) > self.config.excessive_tilt_threshold):
            reward += self.config.fall_penalty * self.config.excessive_tilt_penalty_multiplier
        
        return reward * self.config.reward_scale
    
    def _check_done(self) -> bool:
        """检查是否结束
        
        返回:
            done: 是否结束
        """
        # 达到最大步数
        if self.current_step >= self.max_steps:
            return True
        
        # 任务成功
        if self.episode_success:
            return True
        
        # 摔倒（高度太低）
        if self.robot_position[2] < self.config.fall_height_threshold:
            return True
        
        # Gazebo连接丢失
        if self.gazebo is None and not GAZEBO_AVAILABLE:
            # 模拟模式，继续训练
            return False
        
        return False
    
    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            # 如果Gazebo GUI启用，Gazebo会自己渲染
            if self.gazebo is not None and self.config.gui_enabled:
                return True
            else:
                # 打印当前状态
                print(f"Step: {self.current_step}, Reward: {self.episode_reward:.2f}, "
                      f"Position: {self.robot_position}, Success: {self.episode_success}")
                return True
        elif mode == 'rgb_array':
            # 返回RGB图像数组
            # 这里返回一个空数组作为占位
            return np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            raise ValueError(f"不支持的渲染模式: {mode}")
    
    def close(self):
        """关闭环境"""
        if self.gazebo is not None:
            try:
                self.gazebo.disconnect()
            except Exception as e:
                logger.error(f"Gazebo断开连接失败: {e}")
        
        logger.info("Gazebo机器人环境已关闭")


# 环境注册函数
def register_gazebo_environments():
    """注册Gazebo环境到gym"""
    if not RL_LIBS_AVAILABLE:
        return
    
    try:
        # 注册各种Gazebo环境
        env_specs = [
            ("Gazebo-HumanoidStand-v0", RobotTaskType.STAND_UP),
            ("Gazebo-HumanoidWalk-v0", RobotTaskType.WALK),
            ("Gazebo-HumanoidBalance-v0", RobotTaskType.BALANCE),
            ("Gazebo-HumanoidReach-v0", RobotTaskType.REACH_TARGET),
        ]
        
        for env_id, task_type in env_specs:
            # 创建环境工厂函数
            def make_env(env_id=env_id, task_type=task_type):
                config = GazeboEnvironmentConfig(task_type=task_type)
                return GazeboRobotEnvironment(config)
            
            # 注册环境
            try:
                gym.register(
                    id=env_id,
                    entry_point=make_env,
                    max_episode_steps=1000,
                    reward_threshold=500.0,
                )
                logger.info(f"已注册环境: {env_id}")
            except Exception as e:
                logger.warning(f"注册环境 {env_id} 失败: {e}")
    
    except Exception as e:
        logger.error(f"注册Gazebo环境失败: {e}")


# 自动注册环境
if RL_LIBS_AVAILABLE:
    register_gazebo_environments()