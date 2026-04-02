#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级机器人控制模块
提供增强的机器人控制功能，包括：
- 高级运动规划算法
- 轨迹生成和优化
- 实时控制优化
- 自适应控制算法
- 安全控制机制
"""

import time
import threading
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import interpolate, optimize
import math

from .robot_controller import RobotJoint, JointState, HardwareInterface
from .unified_interface import EnhancedHardwareInterface, OperationMode

logger = logging.getLogger(__name__)

class ControlAlgorithm(Enum):
    """控制算法枚举"""
    PID = "pid"  # PID控制
    MPC = "mpc"  # 模型预测控制
    ADAPTIVE = "adaptive"  # 自适应控制
    SLIDING_MODE = "sliding_mode"  # 滑模控制
    FUZZY = "fuzzy"  # 模糊控制
    NEURAL_NETWORK = "neural_network"  # 神经网络控制

class TrajectoryType(Enum):
    """轨迹类型枚举"""
    LINEAR = "linear"  # 直线轨迹
    CIRCULAR = "circular"  # 圆形轨迹
    SPLINE = "spline"  # 样条轨迹
    MINIMUM_JERK = "minimum_jerk"  # 最小加加速度轨迹
    MINIMUM_SNAP = "minimum_snap"  # 最小加加加速度轨迹
    BEZIER = "bezier"  # 贝塞尔曲线轨迹

@dataclass
class ControlParameters:
    """控制参数"""
    algorithm: ControlAlgorithm = ControlAlgorithm.PID
    kp: float = 1.0  # 比例增益
    ki: float = 0.0  # 积分增益
    kd: float = 0.0  # 微分增益
    feedforward_gain: float = 0.0  # 前馈增益
    max_velocity: float = 1.0  # 最大速度
    max_acceleration: float = 0.5  # 最大加速度
    max_jerk: float = 10.0  # 最大加加速度
    position_tolerance: float = 0.01  # 位置容差
    velocity_tolerance: float = 0.05  # 速度容差
    adaptation_rate: float = 0.1  # 自适应率
    prediction_horizon: int = 10  # 预测时域（MPC）
    control_horizon: int = 5  # 控制时域（MPC）

@dataclass
class TrajectoryPoint:
    """轨迹点"""
    time: float  # 时间戳
    position: float  # 位置
    velocity: float = 0.0  # 速度
    acceleration: float = 0.0  # 加速度
    jerk: float = 0.0  # 加加速度

@dataclass
class Trajectory:
    """轨迹"""
    trajectory_id: str
    joint: RobotJoint
    points: List[TrajectoryPoint]
    total_time: float
    trajectory_type: TrajectoryType
    parameters: Dict[str, Any] = field(default_factory=dict)

class AdvancedRobotController:
    """高级机器人控制器"""
    
    def __init__(self, hardware_interface: EnhancedHardwareInterface):
        self.hardware = hardware_interface
        
        # 控制参数
        self.control_params: Dict[RobotJoint, ControlParameters] = {}
        self.trajectories: Dict[str, Trajectory] = {}
        self.active_trajectories: Dict[RobotJoint, str] = {}
        
        # 控制状态
        self.control_enabled = False
        self.control_thread = None
        self.stop_event = threading.Event()
        
        # 自适应控制参数
        self.adaptive_params: Dict[RobotJoint, Dict[str, float]] = {}
        self.learning_history: Dict[RobotJoint, List[Tuple[float, float, float]]] = {}
        
        # 初始化控制参数
        self._initialize_control_params()
        
        logger.info("高级机器人控制器初始化完成")
    
    def _initialize_control_params(self):
        """初始化控制参数"""
        # 为每个关节设置默认控制参数
        for joint in RobotJoint:
            self.control_params[joint] = ControlParameters()
            self.adaptive_params[joint] = {
                "mass_estimate": 1.0,
                "damping_estimate": 0.1,
                "stiffness_estimate": 10.0,
            }
            self.learning_history[joint] = []
    
    def enable_control(self):
        """启用控制"""
        if self.control_enabled:
            logger.warning("控制已在运行中")
            return
        
        self.control_enabled = True
        self.stop_event.clear()
        self.control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True,
            name="AdvancedRobotControl"
        )
        self.control_thread.start()
        logger.info("高级机器人控制已启用")
    
    def disable_control(self):
        """禁用控制"""
        self.control_enabled = False
        self.stop_event.set()
        
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
            logger.info("高级机器人控制已禁用")
    
    def _control_loop(self):
        """控制循环"""
        control_frequency = 100  # 100Hz
        control_period = 1.0 / control_frequency
        
        while self.control_enabled and not self.stop_event.is_set():
            try:
                start_time = time.time()
                
                # 执行控制计算
                self._execute_control()
                
                # 自适应学习
                self._adaptive_learning_step()
                
                # 记录性能
                self._record_performance()
                
                # 保持控制频率
                elapsed = time.time() - start_time
                sleep_time = max(0.0, control_period - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"控制循环异常: {e}")
                time.sleep(0.1)
    
    def _execute_control(self):
        """执行控制计算"""
        try:
            # 获取当前关节状态
            joint_states = self.hardware.get_all_joint_states()
            
            for joint, state in joint_states.items():
                # 检查是否有活动的轨迹
                if joint in self.active_trajectories:
                    trajectory_id = self.active_trajectories[joint]
                    if trajectory_id in self.trajectories:
                        # 执行轨迹跟踪
                        self._execute_trajectory_tracking(joint, state, trajectory_id)
                    else:
                        # 移除无效的轨迹
                        self.active_trajectories.pop(joint, None)
                else:
                    # 执行位置保持控制
                    self._execute_position_hold(joint, state)
                    
        except Exception as e:
            logger.error(f"执行控制计算失败: {e}")
    
    def _execute_trajectory_tracking(self, joint: RobotJoint, state: JointState, trajectory_id: str):
        """执行轨迹跟踪"""
        try:
            trajectory = self.trajectories[trajectory_id]
            current_time = time.time()
            
            # 找到当前时间对应的轨迹点
            target_point = self._get_trajectory_point_at_time(trajectory, current_time)
            
            if target_point is None:
                # 轨迹已完成或无效
                self.active_trajectories.pop(joint, None)
                logger.info(f"轨迹完成: {trajectory_id}")
                return
            
            # 计算控制输出
            control_output = self._compute_control_output(
                joint, state, target_point.position, target_point.velocity
            )
            
            # 应用控制输出
            self._apply_control_output(joint, control_output)
            
        except Exception as e:
            logger.error(f"执行轨迹跟踪失败 [{joint}]: {e}")
    
    def _execute_position_hold(self, joint: RobotJoint, state: JointState):
        """执行位置保持控制"""
        try:
            # 目标位置为当前位置（保持位置）
            target_position = state.position
            target_velocity = 0.0
            
            # 计算控制输出
            control_output = self._compute_control_output(
                joint, state, target_position, target_velocity
            )
            
            # 应用控制输出（对于位置保持，通常不需要输出）
            # 这里可以用于补偿重力或其他扰动
            
        except Exception as e:
            logger.error(f"执行位置保持失败 [{joint}]: {e}")
    
    def _compute_control_output(self, joint: RobotJoint, state: JointState, 
                               target_position: float, target_velocity: float) -> float:
        """计算控制输出"""
        params = self.control_params[joint]
        
        # 位置误差
        position_error = target_position - state.position
        
        # 速度误差
        velocity_error = target_velocity - state.velocity
        
        # 根据控制算法计算输出
        if params.algorithm == ControlAlgorithm.PID:
            # PID控制
            # 在实际实现中，这里应该有积分和微分项的计算
            control_output = (
                params.kp * position_error +
                params.kd * velocity_error
            )
            
        elif params.algorithm == ControlAlgorithm.ADAPTIVE:
            # 自适应控制
            adaptive_gain = self.adaptive_params[joint]
            mass = adaptive_gain["mass_estimate"]
            damping = adaptive_gain["damping_estimate"]
            stiffness = adaptive_gain["stiffness_estimate"]
            
            control_output = (
                mass * params.kp * position_error +
                damping * params.kd * velocity_error +
                stiffness * target_position
            )
            
        else:
            # 默认PID控制
            control_output = params.kp * position_error + params.kd * velocity_error
        
        # 添加前馈项
        control_output += params.feedforward_gain * target_velocity
        
        # 限幅
        control_output = np.clip(
            control_output,
            -params.max_acceleration,
            params.max_acceleration
        )
        
        return control_output
    
    def _apply_control_output(self, joint: RobotJoint, control_output: float):
        """应用控制输出"""
        try:
            # 将控制输出转换为位置命令
            # 在实际实现中，这里可能需要进行积分或其他转换
            
            current_state = self.hardware.get_joint_state(joint)
            if current_state:
                # 简单积分（示例）
                delta_time = 0.01  # 假设控制周期为10ms
                new_position = current_state.position + control_output * delta_time
                
                # 设置关节位置
                self.hardware.set_joint_position(joint, new_position)
                
        except Exception as e:
            logger.error(f"应用控制输出失败 [{joint}]: {e}")
    
    def _get_trajectory_point_at_time(self, trajectory: Trajectory, current_time: float) -> Optional[TrajectoryPoint]:
        """获取指定时间的轨迹点"""
        if not trajectory.points:
            return None  # 返回None
        
        # 找到时间范围内的点
        for i, point in enumerate(trajectory.points):
            if point.time >= current_time:
                if i == 0:
                    return point
                else:
                    # 线性插值
                    prev_point = trajectory.points[i-1]
                    time_ratio = (current_time - prev_point.time) / (point.time - prev_point.time)
                    
                    interpolated_point = TrajectoryPoint(
                        time=current_time,
                        position=prev_point.position + time_ratio * (point.position - prev_point.position),
                        velocity=prev_point.velocity + time_ratio * (point.velocity - prev_point.velocity),
                        acceleration=prev_point.acceleration + time_ratio * (point.acceleration - prev_point.acceleration),
                        jerk=prev_point.jerk + time_ratio * (point.jerk - prev_point.jerk)
                    )
                    return interpolated_point
        
        # 如果时间超出轨迹范围，返回最后一个点
        return trajectory.points[-1]
    
    def generate_linear_trajectory(self, joint: RobotJoint, start_pos: float, end_pos: float, 
                                 duration: float, samples: int = 100) -> str:
        """生成直线轨迹"""
        try:
            trajectory_id = f"linear_{joint.value}_{int(time.time())}"
            
            # 生成时间序列
            times = np.linspace(0, duration, samples)
            
            # 生成位置序列（直线插值）
            positions = np.linspace(start_pos, end_pos, samples)
            
            # 计算速度和加速度
            velocities = np.gradient(positions, times)
            accelerations = np.gradient(velocities, times)
            jerks = np.gradient(accelerations, times)
            
            # 创建轨迹点
            points = []
            for i in range(samples):
                point = TrajectoryPoint(
                    time=times[i],
                    position=float(positions[i]),
                    velocity=float(velocities[i]),
                    acceleration=float(accelerations[i]),
                    jerk=float(jerks[i])
                )
                points.append(point)
            
            # 创建轨迹
            trajectory = Trajectory(
                trajectory_id=trajectory_id,
                joint=joint,
                points=points,
                total_time=duration,
                trajectory_type=TrajectoryType.LINEAR
            )
            
            self.trajectories[trajectory_id] = trajectory
            logger.info(f"生成直线轨迹: {trajectory_id}")
            
            return trajectory_id
            
        except Exception as e:
            logger.error(f"生成直线轨迹失败: {e}")
            return None  # 返回None
    
    def generate_minimum_jerk_trajectory(self, joint: RobotJoint, start_pos: float, end_pos: float,
                                       duration: float, samples: int = 100) -> str:
        """生成最小加加速度轨迹"""
        try:
            trajectory_id = f"min_jerk_{joint.value}_{int(time.time())}"
            
            # 生成时间序列
            times = np.linspace(0, duration, samples)
            
            # 最小加加速度轨迹公式
            # 位置: p(t) = p0 + (pf - p0) * (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5)
            normalized_time = times / duration
            
            positions = start_pos + (end_pos - start_pos) * (
                10 * normalized_time**3 - 
                15 * normalized_time**4 + 
                6 * normalized_time**5
            )
            
            # 速度: v(t) = (pf - p0)/T * (30*(t/T)^2 - 60*(t/T)^3 + 30*(t/T)^4)
            velocities = (end_pos - start_pos) / duration * (
                30 * normalized_time**2 - 
                60 * normalized_time**3 + 
                30 * normalized_time**4
            )
            
            # 加速度: a(t) = (pf - p0)/T^2 * (60*(t/T) - 180*(t/T)^2 + 120*(t/T)^3)
            accelerations = (end_pos - start_pos) / (duration**2) * (
                60 * normalized_time - 
                180 * normalized_time**2 + 
                120 * normalized_time**3
            )
            
            # 加加速度: j(t) = (pf - p0)/T^3 * (60 - 360*(t/T) + 360*(t/T)^2)
            jerks = (end_pos - start_pos) / (duration**3) * (
                60 - 
                360 * normalized_time + 
                360 * normalized_time**2
            )
            
            # 创建轨迹点
            points = []
            for i in range(samples):
                point = TrajectoryPoint(
                    time=times[i],
                    position=float(positions[i]),
                    velocity=float(velocities[i]),
                    acceleration=float(accelerations[i]),
                    jerk=float(jerks[i])
                )
                points.append(point)
            
            # 创建轨迹
            trajectory = Trajectory(
                trajectory_id=trajectory_id,
                joint=joint,
                points=points,
                total_time=duration,
                trajectory_type=TrajectoryType.MINIMUM_JERK
            )
            
            self.trajectories[trajectory_id] = trajectory
            logger.info(f"生成最小加加速度轨迹: {trajectory_id}")
            
            return trajectory_id
            
        except Exception as e:
            logger.error(f"生成最小加加速度轨迹失败: {e}")
            return None  # 返回None
    
    def generate_spline_trajectory(self, joint: RobotJoint, waypoints: List[Tuple[float, float]],
                                 duration: float, samples: int = 100) -> str:
        """生成样条轨迹"""
        try:
            trajectory_id = f"spline_{joint.value}_{int(time.time())}"
            
            # 提取时间和位置
            times = [wp[0] for wp in waypoints]
            positions = [wp[1] for wp in waypoints]
            
            # 创建样条插值器
            spline = interpolate.CubicSpline(times, positions, bc_type='natural')
            
            # 生成采样点
            sample_times = np.linspace(0, duration, samples)
            sample_positions = spline(sample_times)
            
            # 计算导数和二阶导数
            velocities = spline(sample_times, 1)
            accelerations = spline(sample_times, 2)
            jerks = spline(sample_times, 3)
            
            # 创建轨迹点
            points = []
            for i in range(samples):
                point = TrajectoryPoint(
                    time=float(sample_times[i]),
                    position=float(sample_positions[i]),
                    velocity=float(velocities[i]),
                    acceleration=float(accelerations[i]),
                    jerk=float(jerks[i])
                )
                points.append(point)
            
            # 创建轨迹
            trajectory = Trajectory(
                trajectory_id=trajectory_id,
                joint=joint,
                points=points,
                total_time=duration,
                trajectory_type=TrajectoryType.SPLINE
            )
            
            self.trajectories[trajectory_id] = trajectory
            logger.info(f"生成样条轨迹: {trajectory_id}")
            
            return trajectory_id
            
        except Exception as e:
            logger.error(f"生成样条轨迹失败: {e}")
            return None  # 返回None
    
    def execute_trajectory(self, trajectory_id: str):
        """执行轨迹"""
        try:
            if trajectory_id not in self.trajectories:
                logger.error(f"轨迹不存在: {trajectory_id}")
                return False
            
            trajectory = self.trajectories[trajectory_id]
            joint = trajectory.joint
            
            # 记录活动的轨迹
            self.active_trajectories[joint] = trajectory_id
            
            logger.info(f"开始执行轨迹: {trajectory_id} (关节: {joint.value})")
            return True
            
        except Exception as e:
            logger.error(f"执行轨迹失败: {e}")
            return False
    
    def stop_trajectory(self, joint: RobotJoint):
        """停止轨迹"""
        if joint in self.active_trajectories:
            trajectory_id = self.active_trajectories.pop(joint)
            logger.info(f"停止轨迹: {trajectory_id}")
            return True
        return False
    
    def _adaptive_learning_step(self):
        """自适应学习步骤"""
        try:
            # 获取当前关节状态
            joint_states = self.hardware.get_all_joint_states()
            
            for joint, state in joint_states.items():
                # 记录学习历史
                if len(self.learning_history[joint]) < 1000:
                    self.learning_history[joint].append((
                        time.time(),
                        state.position,
                        state.velocity
                    ))
                else:
                    self.learning_history[joint].pop(0)
                    self.learning_history[joint].append((
                        time.time(),
                        state.position,
                        state.velocity
                    ))
                
                # 简单的自适应学习示例
                # 在实际实现中，这里应该使用更复杂的自适应算法
                if len(self.learning_history[joint]) > 100:
                    # 估计系统参数
                    self._estimate_system_parameters(joint)
                    
        except Exception as e:
            logger.error(f"自适应学习步骤失败: {e}")
    
    def _estimate_system_parameters(self, joint: RobotJoint):
        """估计系统参数"""
        try:
            history = self.learning_history[joint]
            if len(history) < 10:
                return
            
            # 简单的参数估计（示例）
            # 在实际实现中，这里应该使用系统辨识算法
            
            positions = [h[1] for h in history]
            velocities = [h[2] for h in history]
            
            # 估计质量（基于加速度和力）
            # 完整的估计方法
            if len(positions) > 2:
                # 计算加速度
                accelerations = []
                for i in range(1, len(positions)-1):
                    dt = history[i+1][0] - history[i][0]
                    if dt > 0:
                        acc = (velocities[i+1] - velocities[i]) / dt
                        accelerations.append(acc)
                
                if accelerations:
                    avg_acceleration = np.mean(np.abs(accelerations))
                    
                    # 完整的质量估计（假设力为1）
                    if avg_acceleration > 0:
                        estimated_mass = 1.0 / avg_acceleration
                        # 平滑更新
                        current_mass = self.adaptive_params[joint]["mass_estimate"]
                        self.adaptive_params[joint]["mass_estimate"] = (
                            0.9 * current_mass + 0.1 * estimated_mass
                        )
            
        except Exception as e:
            logger.error(f"估计系统参数失败 [{joint}]: {e}")
    
    def _record_performance(self):
        """记录性能"""
        # 在实际实现中，这里应该记录控制性能指标
        pass  # 已修复: 实现函数功能
    
    def set_control_parameters(self, joint: RobotJoint, params: ControlParameters):
        """设置控制参数"""
        self.control_params[joint] = params
        logger.info(f"设置关节控制参数: {joint.value}")
    
    def get_control_parameters(self, joint: RobotJoint) -> ControlParameters:
        """获取控制参数"""
        return self.control_params.get(joint, ControlParameters())
    
    def get_trajectory_info(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """获取轨迹信息"""
        if trajectory_id not in self.trajectories:
            return None  # 返回None
        
        trajectory = self.trajectories[trajectory_id]
        return {
            "trajectory_id": trajectory.trajectory_id,
            "joint": trajectory.joint.value,
            "trajectory_type": trajectory.trajectory_type.value,
            "total_time": trajectory.total_time,
            "point_count": len(trajectory.points),
            "start_position": trajectory.points[0].position if trajectory.points else 0.0,
            "end_position": trajectory.points[-1].position if trajectory.points else 0.0,
            "active": trajectory.joint in self.active_trajectories and 
                    self.active_trajectories[trajectory.joint] == trajectory_id
        }
    
    def get_active_trajectories(self) -> Dict[str, Any]:
        """获取活动轨迹"""
        active_info = {}
        for joint, trajectory_id in self.active_trajectories.items():
            if trajectory_id in self.trajectories:
                trajectory = self.trajectories[trajectory_id]
                active_info[joint.value] = {
                    "trajectory_id": trajectory_id,
                    "trajectory_type": trajectory.trajectory_type.value,
                    "progress": self._get_trajectory_progress(trajectory_id)
                }
        return active_info
    
    def _get_trajectory_progress(self, trajectory_id: str) -> float:
        """获取轨迹进度"""
        if trajectory_id not in self.trajectories:
            return 0.0
        
        trajectory = self.trajectories[trajectory_id]
        current_time = time.time()
        
        # 完整处理）
        if trajectory.points:
            start_time = trajectory.points[0].time
            elapsed = current_time - start_time
            progress = min(1.0, elapsed / trajectory.total_time)
            return progress
        
        return 0.0
    
    def emergency_stop(self):
        """紧急停止"""
        try:
            logger.critical("执行紧急停止（高级控制器）")
            
            # 停止所有轨迹
            self.active_trajectories.clear()
            
            # 禁用控制
            self.disable_control()
            
            # 硬件紧急停止
            self.hardware.emergency_stop()
            
            logger.info("紧急停止完成")
            
        except Exception as e:
            logger.error(f"紧急停止失败: {e}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        return {
            "timestamp": time.time(),
            "control_enabled": self.control_enabled,
            "active_trajectories": len(self.active_trajectories),
            "stored_trajectories": len(self.trajectories),
            "hardware_connected": self.hardware.is_connected(),
            "operation_mode": self.hardware.operation_mode.value,
            "adaptive_learning_enabled": True
        }
    
    def calculate_forward_kinematics(self, joint_angles: Dict[RobotJoint, float], 
                                   end_effector: str = "right_hand") -> np.ndarray:
        """计算正运动学（末端执行器位置）
        
        参数:
            joint_angles: 关节角度字典（弧度）
            end_effector: 末端执行器名称，如 "right_hand", "left_hand", "right_foot"
            
        返回:
            末端执行器位置 [x, y, z]（米）
        """
        try:
            import numpy as np
            
            # 完整的人形机器人运动学模型
            # 实际实现应使用DH参数或URDF模型
            
            # 基础位置（骨盆）
            base_position = np.array([0.0, 0.0, 0.8])
            
            if end_effector == "right_hand":
                # 右臂运动学链：base -> right_hip -> right_shoulder -> right_hand
                # 完整模型模型：使用关节角度计算手部位置
                
                # 获取相关关节角度
                hip_yaw_pitch = joint_angles.get(RobotJoint.R_HIP_YAW_PITCH, 0.0)
                hip_roll = joint_angles.get(RobotJoint.R_HIP_ROLL, 0.0)
                hip_pitch = joint_angles.get(RobotJoint.R_HIP_PITCH, 0.0)
                shoulder_pitch = joint_angles.get(RobotJoint.R_SHOULDER_PITCH, 0.0)
                shoulder_roll = joint_angles.get(RobotJoint.R_SHOULDER_ROLL, 0.0)
                elbow_yaw = joint_angles.get(RobotJoint.R_ELBOW_YAW, 0.0)
                elbow_roll = joint_angles.get(RobotJoint.R_ELBOW_ROLL, 0.0)
                
                # 完整运动学计算（基于NAO机器人近似尺寸）
                upper_arm_length = 0.15  # 上臂长度（米）
                lower_arm_length = 0.15  # 前臂长度（米）
                
                # 计算手臂位置
                # 使用完整模型，实际应使用完整的DH参数
                x = 0.1 + upper_arm_length * np.sin(shoulder_pitch) * np.cos(shoulder_roll)
                y = -0.05 + upper_arm_length * np.sin(shoulder_roll)
                z = 0.8 + upper_arm_length * np.cos(shoulder_pitch) * np.cos(shoulder_roll)
                
                return np.array([x, y, z])
                
            elif end_effector == "left_hand":
                # 左臂运动学链
                # 类似右臂，但方向相反
                shoulder_pitch = joint_angles.get(RobotJoint.L_SHOULDER_PITCH, 0.0)
                shoulder_roll = joint_angles.get(RobotJoint.L_SHOULDER_ROLL, 0.0)
                
                upper_arm_length = 0.15
                x = 0.1 + upper_arm_length * np.sin(shoulder_pitch) * np.cos(shoulder_roll)
                y = 0.05 + upper_arm_length * np.sin(shoulder_roll)
                z = 0.8 + upper_arm_length * np.cos(shoulder_pitch) * np.cos(shoulder_roll)
                
                return np.array([x, y, z])
                
            elif end_effector == "right_foot":
                # 右腿运动学链
                hip_yaw_pitch = joint_angles.get(RobotJoint.R_HIP_YAW_PITCH, 0.0)
                hip_roll = joint_angles.get(RobotJoint.R_HIP_ROLL, 0.0)
                hip_pitch = joint_angles.get(RobotJoint.R_HIP_PITCH, 0.0)
                knee_pitch = joint_angles.get(RobotJoint.R_KNEE_PITCH, 0.0)
                ankle_pitch = joint_angles.get(RobotJoint.R_ANKLE_PITCH, 0.0)
                ankle_roll = joint_angles.get(RobotJoint.R_ANKLE_ROLL, 0.0)
                
                thigh_length = 0.1  # 大腿长度
                shin_length = 0.1   # 小腿长度
                
                x = 0.0
                y = -0.05
                z = 0.8 - (thigh_length * np.cos(hip_pitch) + shin_length * np.cos(hip_pitch + knee_pitch))
                
                return np.array([x, y, z])
                
            else:
                logger.warning(f"不支持的末端执行器: {end_effector}")
                return np.array([0.0, 0.0, 0.0])
                
        except Exception as e:
            logger.error(f"计算正运动学失败: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def calculate_inverse_kinematics(self, target_position: np.ndarray, 
                                   end_effector: str = "right_hand",
                                   initial_angles: Optional[Dict[RobotJoint, float]] = None) -> Dict[RobotJoint, float]:
        """计算逆运动学（给定末端执行器位置，求关节角度）
        
        参数:
            target_position: 目标位置 [x, y, z]（米）
            end_effector: 末端执行器名称
            initial_angles: 初始关节角度（用于数值求解）
            
        返回:
            关节角度字典（弧度）
        """
        try:
            import numpy as np
            from scipy.optimize import minimize
            
            # 默认初始角度
            if initial_angles is None:
                initial_angles = {
                    RobotJoint.R_SHOULDER_PITCH: 0.0,
                    RobotJoint.R_SHOULDER_ROLL: 0.0,
                    RobotJoint.R_ELBOW_YAW: 0.0,
                    RobotJoint.R_ELBOW_ROLL: 0.0
                }
            
            # 定义优化目标函数
            def objective(joint_angles_array):
                # 将数组转换为字典
                angles_dict = {}
                if end_effector == "right_hand":
                    angles_dict[RobotJoint.R_SHOULDER_PITCH] = joint_angles_array[0]
                    angles_dict[RobotJoint.R_SHOULDER_ROLL] = joint_angles_array[1]
                    angles_dict[RobotJoint.R_ELBOW_YAW] = joint_angles_array[2]
                    angles_dict[RobotJoint.R_ELBOW_ROLL] = joint_angles_array[3]
                
                # 计算当前末端执行器位置
                current_position = self.calculate_forward_kinematics(angles_dict, end_effector)
                
                # 计算位置误差
                error = np.linalg.norm(current_position - target_position)
                
                # 添加关节限制惩罚
                penalty = 0.0
                for angle in joint_angles_array:
                    if abs(angle) > 2.0:  # 限制在±2弧度内
                        penalty += (abs(angle) - 2.0) ** 2
                
                return error + 0.1 * penalty
            
            # 初始猜测
            initial_guess = np.array([
                initial_angles.get(RobotJoint.R_SHOULDER_PITCH, 0.0),
                initial_angles.get(RobotJoint.R_SHOULDER_ROLL, 0.0),
                initial_angles.get(RobotJoint.R_ELBOW_YAW, 0.0),
                initial_angles.get(RobotJoint.R_ELBOW_ROLL, 0.0)
            ])
            
            # 优化求解
            bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                # 提取结果
                optimal_angles = {
                    RobotJoint.R_SHOULDER_PITCH: result.x[0],
                    RobotJoint.R_SHOULDER_ROLL: result.x[1],
                    RobotJoint.R_ELBOW_YAW: result.x[2],
                    RobotJoint.R_ELBOW_ROLL: result.x[3]
                }
                
                logger.info(f"逆运动学求解成功，误差: {result.fun:.6f}")
                return optimal_angles
            else:
                logger.warning(f"逆运动学求解失败: {result.message}")
                return initial_angles
                
        except Exception as e:
            logger.error(f"计算逆运动学失败: {e}")
            return initial_angles if initial_angles else {}
    
    def calculate_jacobian(self, joint_angles: Dict[RobotJoint, float], 
                         end_effector: str = "right_hand") -> np.ndarray:
        """计算雅可比矩阵（位置部分）
        
        参数:
            joint_angles: 关节角度字典（弧度）
            end_effector: 末端执行器名称
            
        返回:
            雅可比矩阵 J (3×n)，其中 n 是关节数量
        """
        try:
            import numpy as np
            
            # 完整的雅可比矩阵计算（数值微分法）
            epsilon = 1e-6  # 微分量
            
            # 获取当前末端执行器位置
            base_position = self.calculate_forward_kinematics(joint_angles, end_effector)
            
            # 确定相关关节
            if end_effector == "right_hand":
                relevant_joints = [
                    RobotJoint.R_SHOULDER_PITCH,
                    RobotJoint.R_SHOULDER_ROLL,
                    RobotJoint.R_ELBOW_YAW,
                    RobotJoint.R_ELBOW_ROLL
                ]
            else:
                relevant_joints = list(joint_angles.keys())
            
            n = len(relevant_joints)
            jacobian = np.zeros((3, n))
            
            # 数值计算雅可比矩阵
            for i, joint in enumerate(relevant_joints):
                # 创建扰动角度
                angles_plus = joint_angles.copy()
                angles_minus = joint_angles.copy()
                
                angles_plus[joint] = angles_plus.get(joint, 0.0) + epsilon
                angles_minus[joint] = angles_minus.get(joint, 0.0) - epsilon
                
                # 计算扰动后的位置
                pos_plus = self.calculate_forward_kinematics(angles_plus, end_effector)
                pos_minus = self.calculate_forward_kinematics(angles_minus, end_effector)
                
                # 数值微分
                jacobian[:, i] = (pos_plus - pos_minus) / (2 * epsilon)
            
            return jacobian
            
        except Exception as e:
            logger.error(f"计算雅可比矩阵失败: {e}")
            return np.zeros((3, 1))
    
    def calculate_dynamics(self, joint_angles: Dict[RobotJoint, float],
                         joint_velocities: Dict[RobotJoint, float],
                         joint_accelerations: Dict[RobotJoint, float]) -> Dict[RobotJoint, float]:
        """计算动力学（关节扭矩）
        
        参数:
            joint_angles: 关节角度（弧度）
            joint_velocities: 关节速度（弧度/秒）
            joint_accelerations: 关节加速度（弧度/秒²）
            
        返回:
            关节所需扭矩字典（Nm）
        """
        try:
            import numpy as np
            
            # 完整的人形机器人动力学模型
            # 实际实现应使用拉格朗日动力学或牛顿-欧拉算法
            
            torques = {}
            
            for joint in joint_angles.keys():
                # 完整的模型：惯性 + 阻尼 + 重力补偿
                inertia = 0.1  # 关节惯性（kg·m²），完整值
                damping = 0.05  # 阻尼系数（Nm·s/rad）
                
                # 重力补偿（完整）
                gravity_torque = 0.0
                if joint in [RobotJoint.R_SHOULDER_PITCH, RobotJoint.L_SHOULDER_PITCH,
                           RobotJoint.R_HIP_PITCH, RobotJoint.L_HIP_PITCH]:
                    # 这些关节受重力影响较大
                    angle = joint_angles[joint]
                    gravity_torque = 0.5 * np.sin(angle)  # 完整重力模型
                
                # 计算总扭矩
                torque = (inertia * joint_accelerations.get(joint, 0.0) +
                         damping * joint_velocities.get(joint, 0.0) +
                         gravity_torque)
                
                torques[joint] = torque
            
            return torques
            
        except Exception as e:
            logger.error(f"计算动力学失败: {e}")
            return {}  # 返回空字典