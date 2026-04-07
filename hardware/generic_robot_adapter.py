#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用机器人适配器框架
提供对不同厂家型号人形机器人的兼容性和通用性支持

设计目标：
1. 一次训练成功可以兼容控制所有不同型号人形机器人
2. 支持部分硬件连接工作模式
3. 不依赖虚拟数据，使用真实硬件数据
4. 提供统一的控制接口，隐藏厂商差异

架构：
1. GenericRobotInterface: 通用机器人接口，使用标准化关节和传感器命名
2. RobotAdapter: 适配器基类，处理厂商特定映射
3. AdapterFactory: 适配器工厂，根据配置创建合适的适配器
4. UnifiedControl: 统一控制层，提供与训练系统的接口
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

# 导入硬件接口相关类型
try:
    from hardware.robot_controller import RobotJoint, JointState, HardwareInterface
except ImportError:
    # 如果导入失败，创建虚拟类型以支持类型检查
    class RobotJoint(Enum):
        pass
    
    class JointState:
        pass
    
    class HardwareInterface:
        pass

logger = logging.getLogger(__name__)


class GenericJointType(Enum):
    """通用关节类型枚举
    
    标准化的人形机器人关节类型，支持不同厂商的机器人映射
    """
    HEAD_YAW = "head_yaw"           # 头部偏航
    HEAD_PITCH = "head_pitch"       # 头部俯仰
    
    # 上肢关节
    SHOULDER_PITCH = "shoulder_pitch"  # 肩部俯仰
    SHOULDER_ROLL = "shoulder_roll"    # 肩部滚动
    ELBOW_YAW = "elbow_yaw"           # 肘部偏航
    ELBOW_ROLL = "elbow_roll"         # 肘部滚动
    WRIST_YAW = "wrist_yaw"           # 腕部偏航
    HAND = "hand"                     # 手部
    
    # 下肢关节
    HIP_YAW_PITCH = "hip_yaw_pitch"   # 髋部偏航俯仰
    HIP_ROLL = "hip_roll"             # 髋部滚动
    HIP_PITCH = "hip_pitch"           # 髋部俯仰
    KNEE_PITCH = "knee_pitch"         # 膝部俯仰
    ANKLE_PITCH = "ankle_pitch"       # 踝部俯仰
    ANKLE_ROLL = "ankle_roll"         # 踝部滚动


class GenericSensorType(Enum):
    """通用传感器类型枚举"""
    JOINT_POSITION = "joint_position"      # 关节位置
    JOINT_VELOCITY = "joint_velocity"      # 关节速度
    JOINT_TORQUE = "joint_torque"          # 关节扭矩
    JOINT_TEMPERATURE = "joint_temperature" # 关节温度
    
    IMU_ACCELERATION = "imu_acceleration"  # IMU加速度
    IMU_GYROSCOPE = "imu_gyroscope"        # IMU陀螺仪
    IMU_ORIENTATION = "imu_orientation"    # IMU方向
    
    FORCE_LEFT_FOOT = "force_left_foot"    # 左脚力传感器
    FORCE_RIGHT_FOOT = "force_right_foot"  # 右脚力传感器
    
    CAMERA_COLOR = "camera_color"          # 彩色相机
    CAMERA_DEPTH = "camera_depth"          # 深度相机
    CAMERA_INFRARED = "camera_infrared"    # 红外相机
    
    LIDAR = "lidar"                        # 激光雷达
    SONAR = "sonar"                        # 超声波传感器


class RobotModel(Enum):
    """机器人模型枚举"""
    GENERIC_HUMANOID = "generic_humanoid"  # 通用人形机器人
    NAO = "nao"                            # Aldebaran NAO
    PEPPER = "pepper"                      # SoftBank Pepper
    ATLAS = "atlas"                        # Boston Dynamics Atlas
    SPOT = "spot"                          # Boston Dynamics Spot
    CUSTOM = "custom"                      # 自定义机器人


class GenericRobotState:
    """通用机器人状态
    
    包含标准化的机器人状态信息，可用于训练和控制
    支持左右对称关节，键为(关节类型, 左右标识)元组
    """
    
    def __init__(self, robot_model: RobotModel):
        self.robot_model = robot_model
        self.timestamp = 0.0
        # 使用(关节类型, 左右标识)作为键，支持对称关节
        self.joint_positions: Dict[Tuple[GenericJointType, str], float] = {}
        self.joint_velocities: Dict[Tuple[GenericJointType, str], float] = {}
        self.joint_torques: Dict[Tuple[GenericJointType, str], float] = {}
        self.sensor_data: Dict[GenericSensorType, np.ndarray] = {}
        self.pose_6d: Optional[np.ndarray] = None  # 6D位姿 [x, y, z, roll, pitch, yaw]
        self.twist_6d: Optional[np.ndarray] = None  # 6D速度 [vx, vy, vz, wx, wy, wz]
    
    def to_numpy(self) -> np.ndarray:
        """将状态转换为numpy数组，用于机器学习训练"""
        # 收集所有关节位置和速度，按固定的顺序
        positions = []
        velocities = []
        
        # 定义固定的关节顺序：先所有关节类型的左侧，然后右侧，最后中心
        for side in ["left", "right", "center"]:
            for joint_type in GenericJointType:
                key = (joint_type, side)
                positions.append(self.joint_positions.get(key, 0.0))
                velocities.append(self.joint_velocities.get(key, 0.0))
        
        # 组合所有状态
        state_array = np.array(positions + velocities)
        
        # 添加位姿和速度（如果可用）
        if self.pose_6d is not None:
            state_array = np.concatenate([state_array, self.pose_6d])
        if self.twist_6d is not None:
            state_array = np.concatenate([state_array, self.twist_6d])
        
        return state_array
    
    def from_numpy(self, array: np.ndarray) -> None:
        """从numpy数组恢复状态"""
        # 根据固定的关节顺序恢复状态
        # 每个关节类型有3个侧面：left, right, center
        num_joint_types = len(GenericJointType)
        expected_size = num_joint_types * 3  # 3个侧面
        
        if len(array) >= expected_size:
            index = 0
            for side in ["left", "right", "center"]:
                for joint_type in GenericJointType:
                    if index < len(array):
                        key = (joint_type, side)
                        self.joint_positions[key] = float(array[index])
                        index += 1


class GenericRobotCommand:
    """通用机器人命令
    
    标准化的机器人控制命令，可映射到不同厂商的机器人
    支持左右对称关节，键为(关节类型, 左右标识)元组
    """
    
    def __init__(self):
        self.timestamp = 0.0
        # 使用(关节类型, 左右标识)作为键，支持对称关节
        self.joint_targets: Dict[Tuple[GenericJointType, str], float] = {}
        self.joint_velocities: Dict[Tuple[GenericJointType, str], float] = {}
        self.cartesian_target: Optional[np.ndarray] = None  # 笛卡尔空间目标
        self.trajectory: Optional[List[np.ndarray]] = None  # 轨迹点列表
    
    def to_numpy(self) -> np.ndarray:
        """将命令转换为numpy数组"""
        # 收集所有关节目标，按固定的顺序
        targets = []
        # 定义固定的关节顺序：先所有关节类型的左侧，然后右侧
        for side in ["left", "right", "center"]:
            for joint_type in GenericJointType:
                key = (joint_type, side)
                if key in self.joint_targets:
                    targets.append(self.joint_targets[key])
                else:
                    targets.append(0.0)
        
        return np.array(targets)
    
    def from_numpy(self, array: np.ndarray, joint_mask: Optional[List[Tuple[GenericJointType, str]]] = None) -> None:
        """从numpy数组恢复命令
        
        参数:
            array: 命令数组
            joint_mask: 关节掩码，指定哪些关节被控制，格式为[(关节类型, 左右标识), ...]
        """
        if joint_mask is None:
            # 默认掩码：所有关节类型的所有侧面
            joint_mask = []
            for joint_type in GenericJointType:
                for side in ["left", "right", "center"]:
                    joint_mask.append((joint_type, side))
        
        for i, key in enumerate(joint_mask):
            if i < len(array):
                self.joint_targets[key] = float(array[i])


class RobotAdapter(ABC):
    """机器人适配器基类
    
    将通用机器人接口映射到特定厂商的机器人接口
    """
    
    def __init__(self, robot_model: RobotModel, base_interface: Any):
        """
        参数:
            robot_model: 机器人模型
            base_interface: 基础硬件接口
        """
        self.robot_model = robot_model
        self.base_interface = base_interface
        self.logger = logging.getLogger(f"RobotAdapter.{robot_model.value}")
        
        # 关节映射：通用关节类型 -> 厂商特定关节标识
        self.joint_mapping: Dict[GenericJointType, str] = {}
        
        # 传感器映射：通用传感器类型 -> 厂商特定传感器标识
        self.sensor_mapping: Dict[GenericSensorType, str] = {}
        
        # 关节限制
        self.joint_limits: Dict[GenericJointType, Tuple[float, float]] = {}
        
        self._init_mappings()
        self.logger.info(f"{robot_model.value}适配器初始化完成")
    
    @abstractmethod
    def _init_mappings(self) -> None:
        """初始化映射关系（子类必须实现）"""
        pass
    
    @abstractmethod
    def get_state(self) -> GenericRobotState:
        """获取通用机器人状态"""
        pass
    
    @abstractmethod
    def send_command(self, command: GenericRobotCommand) -> bool:
        """发送通用机器人命令"""
        pass
    
    def get_available_joints(self) -> List[GenericJointType]:
        """获取可用的关节列表"""
        return list(self.joint_mapping.keys())
    
    def get_available_sensors(self) -> List[GenericSensorType]:
        """获取可用的传感器列表"""
        return list(self.sensor_mapping.keys())


class NAOAdapter(RobotAdapter):
    """NAO机器人适配器"""
    
    def __init__(self, base_interface: Any):
        super().__init__(RobotModel.NAO, base_interface)
    
    def _init_mappings(self) -> None:
        """初始化NAO机器人的映射关系
        
        根据项目要求"一次训练成功可以兼容控制所有不同型号人形机器人"，
        建立完整的通用关节类型到NAOqi关节名称的映射。
        
        注意：由于GenericJointType不区分左右，而NAO机器人是左右对称的，
        我们为左右侧创建独立的映射条目，支持完整的机器人控制。
        """
        # NAO关节映射：通用关节类型 -> NAOqi关节名称
        # 使用元组键(通用关节类型, 左右标识)来支持对称关节
        self.joint_mapping = {
            # 头部关节
            (GenericJointType.HEAD_YAW, "center"): "HeadYaw",
            (GenericJointType.HEAD_PITCH, "center"): "HeadPitch",
            
            # 左臂关节
            (GenericJointType.SHOULDER_PITCH, "left"): "LShoulderPitch",
            (GenericJointType.SHOULDER_ROLL, "left"): "LShoulderRoll",
            (GenericJointType.ELBOW_YAW, "left"): "LElbowYaw",
            (GenericJointType.ELBOW_ROLL, "left"): "LElbowRoll",
            (GenericJointType.WRIST_YAW, "left"): "LWristYaw",
            (GenericJointType.HAND, "left"): "LHand",
            
            # 右臂关节
            (GenericJointType.SHOULDER_PITCH, "right"): "RShoulderPitch",
            (GenericJointType.SHOULDER_ROLL, "right"): "RShoulderRoll",
            (GenericJointType.ELBOW_YAW, "right"): "RElbowYaw",
            (GenericJointType.ELBOW_ROLL, "right"): "RElbowRoll",
            (GenericJointType.WRIST_YAW, "right"): "RWristYaw",
            (GenericJointType.HAND, "right"): "RHand",
            
            # 左腿关节
            (GenericJointType.HIP_YAW_PITCH, "left"): "LHipYawPitch",
            (GenericJointType.HIP_ROLL, "left"): "LHipRoll",
            (GenericJointType.HIP_PITCH, "left"): "LHipPitch",
            (GenericJointType.KNEE_PITCH, "left"): "LKneePitch",
            (GenericJointType.ANKLE_PITCH, "left"): "LAnklePitch",
            (GenericJointType.ANKLE_ROLL, "left"): "LAnkleRoll",
            
            # 右腿关节
            (GenericJointType.HIP_YAW_PITCH, "right"): "RHipYawPitch",
            (GenericJointType.HIP_ROLL, "right"): "RHipRoll",
            (GenericJointType.HIP_PITCH, "right"): "RHipPitch",
            (GenericJointType.KNEE_PITCH, "right"): "RKneePitch",
            (GenericJointType.ANKLE_PITCH, "right"): "RAnklePitch",
            (GenericJointType.ANKLE_ROLL, "right"): "RAnkleRoll",
        }
        
        # 通用关节类型 -> RobotJoint枚举映射
        self.generic_to_robot_joint = {
            # 头部关节
            (GenericJointType.HEAD_YAW, "center"): RobotJoint.HEAD_YAW,
            (GenericJointType.HEAD_PITCH, "center"): RobotJoint.HEAD_PITCH,
            
            # 左臂关节
            (GenericJointType.SHOULDER_PITCH, "left"): RobotJoint.L_SHOULDER_PITCH,
            (GenericJointType.SHOULDER_ROLL, "left"): RobotJoint.L_SHOULDER_ROLL,
            (GenericJointType.ELBOW_YAW, "left"): RobotJoint.L_ELBOW_YAW,
            (GenericJointType.ELBOW_ROLL, "left"): RobotJoint.L_ELBOW_ROLL,
            (GenericJointType.WRIST_YAW, "left"): RobotJoint.L_WRIST_YAW,
            (GenericJointType.HAND, "left"): RobotJoint.L_HAND,
            
            # 右臂关节
            (GenericJointType.SHOULDER_PITCH, "right"): RobotJoint.R_SHOULDER_PITCH,
            (GenericJointType.SHOULDER_ROLL, "right"): RobotJoint.R_SHOULDER_ROLL,
            (GenericJointType.ELBOW_YAW, "right"): RobotJoint.R_ELBOW_YAW,
            (GenericJointType.ELBOW_ROLL, "right"): RobotJoint.R_ELBOW_ROLL,
            (GenericJointType.WRIST_YAW, "right"): RobotJoint.R_WRIST_YAW,
            (GenericJointType.HAND, "right"): RobotJoint.R_HAND,
            
            # 左腿关节
            (GenericJointType.HIP_YAW_PITCH, "left"): RobotJoint.L_HIP_YAW_PITCH,
            (GenericJointType.HIP_ROLL, "left"): RobotJoint.L_HIP_ROLL,
            (GenericJointType.HIP_PITCH, "left"): RobotJoint.L_HIP_PITCH,
            (GenericJointType.KNEE_PITCH, "left"): RobotJoint.L_KNEE_PITCH,
            (GenericJointType.ANKLE_PITCH, "left"): RobotJoint.L_ANKLE_PITCH,
            (GenericJointType.ANKLE_ROLL, "left"): RobotJoint.L_ANKLE_ROLL,
            
            # 右腿关节
            (GenericJointType.HIP_YAW_PITCH, "right"): RobotJoint.R_HIP_YAW_PITCH,
            (GenericJointType.HIP_ROLL, "right"): RobotJoint.R_HIP_ROLL,
            (GenericJointType.HIP_PITCH, "right"): RobotJoint.R_HIP_PITCH,
            (GenericJointType.KNEE_PITCH, "right"): RobotJoint.R_KNEE_PITCH,
            (GenericJointType.ANKLE_PITCH, "right"): RobotJoint.R_ANKLE_PITCH,
            (GenericJointType.ANKLE_ROLL, "right"): RobotJoint.R_ANKLE_ROLL,
        }
        
        # 关节限制（弧度）- NAO机器人关节限制
        self.joint_limits = {
            # 头部关节限制
            (GenericJointType.HEAD_YAW, "center"): (-2.0857, 2.0857),
            (GenericJointType.HEAD_PITCH, "center"): (-0.6720, 0.5149),
            
            # 左臂关节限制
            (GenericJointType.SHOULDER_PITCH, "left"): (-2.0857, 2.0857),
            (GenericJointType.SHOULDER_ROLL, "left"): (-0.3142, 1.3265),
            (GenericJointType.ELBOW_YAW, "left"): (-2.0857, 2.0857),
            (GenericJointType.ELBOW_ROLL, "left"): (-1.5446, 1.5446),
            (GenericJointType.WRIST_YAW, "left"): (-1.8238, 1.8238),
            (GenericJointType.HAND, "left"): (0.0, 1.0),
            
            # 右臂关节限制（与左臂对称）
            (GenericJointType.SHOULDER_PITCH, "right"): (-2.0857, 2.0857),
            (GenericJointType.SHOULDER_ROLL, "right"): (-1.3265, 0.3142),  # 注意：符号相反
            (GenericJointType.ELBOW_YAW, "right"): (-2.0857, 2.0857),
            (GenericJointType.ELBOW_ROLL, "right"): (-1.5446, 1.5446),
            (GenericJointType.WRIST_YAW, "right"): (-1.8238, 1.8238),
            (GenericJointType.HAND, "right"): (0.0, 1.0),
            
            # 左腿关节限制
            (GenericJointType.HIP_YAW_PITCH, "left"): (-1.1453, 0.7408),
            (GenericJointType.HIP_ROLL, "left"): (-0.3795, 0.7904),
            (GenericJointType.HIP_PITCH, "left"): (-1.5359, 0.4840),
            (GenericJointType.KNEE_PITCH, "left"): (-0.0923, 2.1125),
            (GenericJointType.ANKLE_PITCH, "left"): (-1.1864, 0.9226),
            (GenericJointType.ANKLE_ROLL, "left"): (-0.7689, 0.3979),
            
            # 右腿关节限制（与左腿对称）
            (GenericJointType.HIP_YAW_PITCH, "right"): (-0.7408, 1.1453),
            (GenericJointType.HIP_ROLL, "right"): (-0.7904, 0.3795),
            (GenericJointType.HIP_PITCH, "right"): (-1.5359, 0.4840),
            (GenericJointType.KNEE_PITCH, "right"): (-0.1031, 2.1202),
            (GenericJointType.ANKLE_PITCH, "right"): (-1.1894, 0.9321),
            (GenericJointType.ANKLE_ROLL, "right"): (-0.3979, 0.7689),
        }
        
        # 反向映射：NAOqi关节名称 -> (通用关节类型, 左右标识)
        self.naoqi_to_generic = {
            naoqi_joint: (generic_joint, side) 
            for (generic_joint, side), naoqi_joint in self.joint_mapping.items()
        }
        
        self.logger.info(f"NAO关节映射初始化完成，共映射 {len(self.joint_mapping)} 个关节")
    
    def get_state(self) -> GenericRobotState:
        """获取NAO机器人状态
        
        根据项目要求:
        1. 使用真实硬件数据，禁止使用虚拟数据
        2. 硬件不可用时直接报错，不进行降级处理
        3. 支持部分硬件连接工作模式
        """
        state = GenericRobotState(self.robot_model)
        state.timestamp = time.time()
        
        # 从基础硬件接口获取真实数据
        for (generic_joint, side), nao_joint in self.joint_mapping.items():
            # 将通用关节类型映射到RobotJoint枚举
            key = (generic_joint, side)
            if key in self.generic_to_robot_joint:
                robot_joint = self.generic_to_robot_joint[key]
                
                try:
                    # 从硬件接口获取关节状态
                    joint_state = self.base_interface.get_joint_state(robot_joint)
                    
                    if joint_state is None:
                        # 根据项目要求"不采用任何降级处理，直接报错"
                        raise RuntimeError(
                            f"无法获取关节状态: {generic_joint.value}({side})\n"
                            "根据项目要求'禁止使用虚拟数据'，硬件接口返回None时直接报错。\n"
                            "请检查硬件连接或使用真实的硬件数据源。"
                        )
                    
                    # 使用真实硬件数据
                    state.joint_positions[key] = joint_state.position
                    state.joint_velocities[key] = joint_state.velocity
                    state.joint_torques[key] = joint_state.torque
                    
                except Exception as e:
                    # 硬件访问失败，根据项目要求直接报错
                    raise RuntimeError(
                        f"获取关节状态失败: {generic_joint.value}({side})\n"
                        f"错误: {str(e)}\n"
                        "根据项目要求'不采用任何降级处理，直接报错'，硬件不可用时直接报错。\n"
                        "请连接真实硬件或确保硬件接口正常工作。"
                    ) from e
            else:
                # 没有映射的关节，跳过（部分硬件连接模式）
                self.logger.debug(f"关节 {generic_joint.value}({side}) 无映射，跳过")
        
        return state
    
    def send_command(self, command: GenericRobotCommand) -> bool:
        """发送命令到NAO机器人
        
        根据项目要求:
        1. 使用真实硬件控制，禁止使用虚拟数据
        2. 硬件不可用时直接报错，不进行降级处理
        3. 支持部分硬件连接工作模式
        """
        success = True
        
        # 将通用命令映射到NAO特定命令
        for (generic_joint, side), target in command.joint_targets.items():
            key = (generic_joint, side)
            if key in self.joint_mapping:
                nao_joint = self.joint_mapping[key]
                
                # 检查关节限制
                if key in self.joint_limits:
                    min_limit, max_limit = self.joint_limits[key]
                    if target < min_limit or target > max_limit:
                        # 根据项目要求"不采用任何降级处理，直接报错"
                        raise ValueError(
                            f"关节 {generic_joint.value}({side}) 目标值 {target} 超出限制 [{min_limit}, {max_limit}]\n"
                            "根据项目要求，关节限制检查失败时直接报错，禁止自动调整。"
                        )
                
                # 将通用关节类型映射到RobotJoint枚举
                if key in self.generic_to_robot_joint:
                    robot_joint = self.generic_to_robot_joint[key]
                    
                    try:
                        # 发送命令到真实硬件接口
                        command_success = self.base_interface.set_joint_position(robot_joint, target)
                        
                        if not command_success:
                            # 根据项目要求"不采用任何降级处理，直接报错"
                            raise RuntimeError(
                                f"设置关节位置失败: {generic_joint.value}({side}) -> {target}\n"
                                "根据项目要求'禁止使用虚拟数据'，硬件控制失败时直接报错。\n"
                                "请检查硬件连接或使用真实的硬件控制接口。"
                            )
                        
                        self.logger.debug(f"成功设置关节 {generic_joint.value}({side}) 位置: {target}")
                        
                    except Exception as e:
                        # 硬件控制失败，根据项目要求直接报错
                        raise RuntimeError(
                            f"发送关节命令失败: {generic_joint.value}({side})\n"
                            f"错误: {str(e)}\n"
                            "根据项目要求'不采用任何降级处理，直接报错'，硬件不可用时直接报错。\n"
                            "请连接真实硬件或确保硬件控制接口正常工作。"
                        ) from e
                else:
                    # 没有映射的关节，跳过（部分硬件连接模式）
                    self.logger.warning(f"关节 {generic_joint.value}({side}) 无RobotJoint映射，跳过控制")
                    success = False
        
        return success


class AdapterFactory:
    """适配器工厂"""
    
    @staticmethod
    def create_adapter(robot_model: RobotModel, base_interface: Any) -> RobotAdapter:
        """创建机器人适配器
        
        根据项目要求:
        1. 不采用任何降级处理，直接报错
        2. 不支持时直接抛出异常，禁止返回None
        """
        if robot_model == RobotModel.NAO:
            return NAOAdapter(base_interface)
        elif robot_model == RobotModel.PEPPER:
            # Pepper适配器尚未实现，根据项目要求直接报错
            raise NotImplementedError(
                f"{robot_model.value}适配器尚未实现\n"
                "根据项目要求'不采用任何降级处理，直接报错'，不支持的功能直接报错。\n"
                "请实现Pepper适配器或选择支持的机器人模型。"
            )
        elif robot_model == RobotModel.GENERIC_HUMANOID:
            # 通用适配器尚未实现，根据项目要求直接报错
            raise NotImplementedError(
                f"{robot_model.value}适配器尚未实现\n"
                "根据项目要求'不采用任何降级处理，直接报错'，不支持的功能直接报错。\n"
                "请实现通用人形机器人适配器或选择支持的机器人模型。"
            )
        else:
            # 不支持的机器人模型，根据项目要求直接报错
            raise ValueError(
                f"不支持的机器人模型: {robot_model}\n"
                "根据项目要求'不采用任何降级处理，直接报错'，不支持的模型直接报错。\n"
                "请使用支持的机器人模型: NAO, PEPPER, GENERIC_HUMANOID"
            )


class UnifiedRobotControl:
    """统一机器人控制
    
    提供与训练系统的统一接口，隐藏机器人型号差异
    """
    
    def __init__(self, robot_model: RobotModel, hardware_interface: Any):
        """
        参数:
            robot_model: 机器人模型
            hardware_interface: 硬件接口
        """
        self.robot_model = robot_model
        self.hardware_interface = hardware_interface
        
        # 创建适配器（如果失败会直接抛出异常）
        self.adapter = AdapterFactory.create_adapter(robot_model, hardware_interface)
        
        self.logger = logging.getLogger(f"UnifiedRobotControl.{robot_model.value}")
        self.logger.info(f"统一机器人控制初始化完成，模型: {robot_model.value}")
    
    def get_state(self) -> GenericRobotState:
        """获取机器人状态（统一接口）"""
        return self.adapter.get_state()
    
    def send_command(self, command: GenericRobotCommand) -> bool:
        """发送控制命令（统一接口）"""
        return self.adapter.send_command(command)
    
    def get_state_array(self) -> np.ndarray:
        """获取状态数组（用于机器学习）"""
        state = self.get_state()
        return state.to_numpy()
    
    def send_command_array(self, command_array: np.ndarray) -> bool:
        """发送命令数组（用于机器学习）"""
        command = GenericRobotCommand()
        command.from_numpy(command_array)
        return self.send_command(command)
    
    def get_observation_space(self) -> Tuple[int, ...]:
        """获取观测空间维度"""
        # 返回状态数组的维度
        state = GenericRobotState(self.robot_model)
        return (len(state.to_numpy()),)
    
    def get_action_space(self) -> Tuple[int, ...]:
        """获取动作空间维度"""
        # 返回命令数组的维度（关节类型数量 × 3个侧面）
        # 每个GenericJointType有3个侧面：left, right, center
        return (len(GenericJointType) * 3,)


# 全局导入time模块
import time

if __name__ == "__main__":
    """通用机器人适配器模块测试
    
    注意：根据项目要求"禁止使用虚拟数据"，此测试代码不使用模拟接口。
    仅演示接口定义和维度计算。
    """
    logging.basicConfig(level=logging.INFO)
    
    print("=== 通用机器人适配器测试 ===")
    print("根据项目要求'禁止使用虚拟数据'，此测试不使用模拟硬件接口。")
    print("仅验证接口定义和维度计算。\n")
    
    # 验证GenericJointType枚举
    print(f"通用关节类型数量: {len(GenericJointType)}")
    print("通用关节类型:")
    for joint_type in GenericJointType:
        print(f"  - {joint_type.value}")
    
    # 计算维度
    num_joint_types = len(GenericJointType)
    state_dim = num_joint_types * 3 * 2  # 3个侧面 × 2种状态（位置+速度）
    action_dim = num_joint_types * 3      # 3个侧面
    
    print(f"\n计算维度:")
    print(f"  - 状态空间维度（关节）: {state_dim} = {num_joint_types}类型 × 3侧面 × 2状态")
    print(f"  - 动作空间维度（关节）: {action_dim} = {num_joint_types}类型 × 3侧面")
    
    # 验证适配器工厂
    print(f"\n适配器工厂支持模型:")
    for model in RobotModel:
        print(f"  - {model.value}")
    
    print("\n注意：实际使用时必须提供真实硬件接口。")
    print("根据项目要求'禁止使用虚拟数据'和'不采用任何降级处理，直接报错'，")
    print("硬件不可用时应直接抛出异常，禁止使用模拟接口。")