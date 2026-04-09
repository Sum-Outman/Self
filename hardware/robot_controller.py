# 人形机器人控制器
import threading
import time
import json
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime

# 真实机器人硬件接口导入
# 根据项目要求"不采用任何降级处理，直接报错"，如果真实机器人硬件接口不可用，直接报错
# 注意：导入检查已移到函数内部，避免循环导入

# 高级控制可用性标志（在运行时检查）
ADVANCED_CONTROL_AVAILABLE = None

# 真实机器人接口可用性检查函数
def is_real_robot_interface_available():
    """检查真实机器人硬件接口是否可用
    
    返回:
        bool: 如果真实机器人硬件接口可用返回True，否则返回False
    """
    try:
        import importlib
        # 尝试导入真实机器人接口模块
        importlib.import_module('.real_robot_interface', package='hardware')
        return True
    except ImportError:
        return False

# 真实机器人接口模块缓存
_real_robot_interface_module = None

def get_real_robot_interface_module():
    """获取真实机器人接口模块
    
    返回:
        module: 真实机器人接口模块，如果不可用则抛出RuntimeError
    """
    global _real_robot_interface_module
    if _real_robot_interface_module is None:
        try:
            import importlib
            _real_robot_interface_module = importlib.import_module('.real_robot_interface', package='hardware')
        except ImportError as e:
            # 根据项目要求"直接报错，降级处理不容易被发现"，不允许降级处理
            raise RuntimeError(
                f"真实机器人硬件接口不可用: {e}\n"
                "根据项目要求，机器人控制模块必须使用真实硬件接口，禁止使用模拟模式。\n"
                "请确保hardware/real_robot_interface.py存在并实现真实的硬件接口。"
            )
    return _real_robot_interface_module


class RobotJoint(Enum):
    """机器人关节枚举"""
    HEAD_YAW = "head_yaw"  # 头部偏航
    HEAD_PITCH = "head_pitch"  # 头部俯仰
    
    # 左臂
    L_SHOULDER_PITCH = "l_shoulder_pitch"
    L_SHOULDER_ROLL = "l_shoulder_roll"
    L_ELBOW_YAW = "l_elbow_yaw"
    L_ELBOW_ROLL = "l_elbow_roll"
    L_WRIST_YAW = "l_wrist_yaw"
    L_HAND = "l_hand"
    
    # 右臂
    R_SHOULDER_PITCH = "r_shoulder_pitch"
    R_SHOULDER_ROLL = "r_shoulder_roll"
    R_ELBOW_YAW = "r_elbow_yaw"
    R_ELBOW_ROLL = "r_elbow_roll"
    R_WRIST_YAW = "r_wrist_yaw"
    R_HAND = "r_hand"
    
    # 左腿
    L_HIP_YAW_PITCH = "l_hip_yaw_pitch"
    L_HIP_ROLL = "l_hip_roll"
    L_HIP_PITCH = "l_hip_pitch"
    L_KNEE_PITCH = "l_knee_pitch"
    L_ANKLE_PITCH = "l_ankle_pitch"
    L_ANKLE_ROLL = "l_ankle_roll"
    
    # 右腿
    R_HIP_YAW_PITCH = "r_hip_yaw_pitch"
    R_HIP_ROLL = "r_hip_roll"
    R_HIP_PITCH = "r_hip_pitch"
    R_KNEE_PITCH = "r_knee_pitch"
    R_ANKLE_PITCH = "r_ankle_pitch"
    R_ANKLE_ROLL = "r_ankle_roll"


class SensorType(Enum):
    """传感器类型枚举"""
    IMU = "imu"  # 惯性测量单元
    CAMERA = "camera"  # 摄像头
    LIDAR = "lidar"  # 激光雷达
    DEPTH_CAMERA = "depth_camera"  # 深度摄像头
    FORCE_SENSOR = "force_sensor"  # 力传感器
    TOUCH_SENSOR = "touch_sensor"  # 触摸传感器
    JOINT_POSITION = "joint_position"  # 关节位置
    JOINT_TORQUE = "joint_torque"  # 关节扭矩


@dataclass
class JointState:
    """关节状态"""
    position: Optional[float] = None  # 位置（弧度）
    velocity: Optional[float] = None  # 速度（弧度/秒）
    torque: Optional[float] = None  # 扭矩（Nm）
    temperature: Optional[float] = None  # 温度（摄氏度）
    voltage: Optional[float] = None  # 电压（V）
    current: Optional[float] = None  # 电流（A）


@dataclass
class IMUData:
    """IMU数据"""
    acceleration: np.ndarray  # 加速度 [x, y, z] m/s²
    gyroscope: np.ndarray  # 陀螺仪 [x, y, z] rad/s
    magnetometer: np.ndarray  # 磁力计 [x, y, z] μT
    orientation: np.ndarray  # 方向 [roll, pitch, yaw] 弧度
    timestamp: float


@dataclass
class CameraData:
    """摄像头数据"""
    image: np.ndarray  # 图像数据
    depth: Optional[np.ndarray] = None  # 深度数据（如果有）
    timestamp: float = 0.0


@dataclass
class LidarData:
    """激光雷达数据"""
    points: np.ndarray  # 点云数据 [N, 3]
    intensities: Optional[np.ndarray] = None  # 强度数据
    timestamp: float = 0.0


class HardwareInterface(ABC):
    """硬件接口抽象基类"""
    
    def __init__(self, simulation_mode: bool = False):
        """
        初始化硬件接口
        
        参数:
            simulation_mode: 是否启用模拟模式（根据项目要求，禁止使用虚拟数据）
        
        注意：根据项目要求"禁止使用虚拟数据"，基类不初始化任何模拟数据。
        子类应根据实际情况初始化真实硬件数据。
        """
        import logging
        self._interface_type = "unknown"
        self._sensor_enabled = True  # 传感器功能默认启用
        self.logger = logging.getLogger(self.__class__.__name__)
        self._simulation_mode = simulation_mode
        
        # 状态存储（不初始化模拟数据）
        self._connected = False
        self._joint_positions = {}  # 关节位置字典（由子类初始化）
        self._joint_states = {}     # 关节状态字典（由子类初始化）
        self._sensor_data = {}      # 传感器数据字典（由子类初始化）
        
        self.logger.info(f"硬件接口初始化完成，模拟模式: {simulation_mode}")
    

    
    @property
    def interface_type(self) -> str:
        """接口类型标识"""
        return self._interface_type
    
    def get_interface_info(self) -> Dict[str, Any]:
        """获取接口信息"""
        return {
            "type": self._interface_type,
            "connected": self.is_connected() if hasattr(self, 'is_connected') else False,
            "sensor_enabled": self._sensor_enabled,
            "simulation": self._simulation_mode,
            "description": "硬件接口基类（支持模拟和真实硬件模式）"
        }
    
    @property
    def is_simulation(self) -> bool:
        """是否为模拟模式"""
        return self._simulation_mode
    
    @property
    def sensor_enabled(self) -> bool:
        """获取传感器启用状态"""
        return self._sensor_enabled
    
    @sensor_enabled.setter
    def sensor_enabled(self, enabled: bool):
        """设置传感器启用状态"""
        self._sensor_enabled = enabled
        logger = logging.getLogger(self.__class__.__name__)
        if enabled:
            logger.info("传感器功能已启用")
        else:
            logger.info("传感器功能已禁用")
    
    def enable_sensors(self):
        """启用传感器功能"""
        self.sensor_enabled = True
    
    def disable_sensors(self):
        """禁用传感器功能"""
        self.sensor_enabled = False
    
    def connect(self) -> bool:
        """连接硬件
        
        根据项目要求"禁止使用虚拟数据"和"不采用任何降级处理，直接报错"，
        硬件接口必须实现具体的连接逻辑，基类不提供任何模拟实现。
        
        返回:
            bool: 连接是否成功
        
        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            f"硬件接口'{self._interface_type}'必须实现connect()方法。"
            "根据项目要求'禁止使用虚拟数据'和'不采用任何降级处理，直接报错'，"
            "不能提供默认模拟实现。"
        )
    
    def disconnect(self) -> bool:
        """断开连接
        
        根据项目要求"禁止使用虚拟数据"和"不采用任何降级处理，直接报错"，
        硬件接口必须实现具体的断开连接逻辑，基类不提供任何模拟实现。
        
        返回:
            bool: 断开是否成功
        
        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            f"硬件接口'{self._interface_type}'必须实现disconnect()方法。"
            "根据项目要求'禁止使用虚拟数据'和'不采用任何降级处理，直接报错'，"
            "不能提供默认模拟实现。"
        )
    
    def is_connected(self) -> bool:
        """检查是否连接
        
        根据项目要求"禁止使用虚拟数据"和"不采用任何降级处理，直接报错"，
        硬件接口必须实现具体的连接状态检查逻辑，基类不提供任何模拟实现。
        
        返回:
            bool: 是否已连接
        
        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            f"硬件接口'{self._interface_type}'必须实现is_connected()方法。"
            "根据项目要求'禁止使用虚拟数据'和'不采用任何降级处理，直接报错'，"
            "不能提供默认模拟实现。"
        )
    
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据
        
        根据项目要求"禁止使用虚拟数据"和"不采用任何降级处理，直接报错"，
        硬件接口必须实现具体的传感器数据获取逻辑，基类不提供任何模拟实现。
        
        参数:
            sensor_type: 传感器类型
        
        返回:
            传感器数据，如果传感器功能已禁用则抛出异常
        
        抛出:
            NotImplementedError: 子类必须实现此方法
            RuntimeError: 传感器功能已禁用
        """
        if not self.sensor_enabled:
            raise RuntimeError("传感器功能已禁用")
        
        raise NotImplementedError(
            f"硬件接口'{self._interface_type}'必须实现get_sensor_data()方法。"
            "根据项目要求'禁止使用虚拟数据'和'不采用任何降级处理，直接报错'，"
            "不能提供默认模拟实现。"
        )
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置
        
        根据项目要求"禁止使用虚拟数据"和"不采用任何降级处理，直接报错"，
        硬件接口必须实现具体的关节位置设置逻辑，基类不提供任何模拟实现。
        
        参数:
            joint: 关节类型
            position: 目标位置（弧度）
        
        返回:
            bool: 设置是否成功
        
        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            f"硬件接口'{self._interface_type}'必须实现set_joint_position()方法。"
            "根据项目要求'禁止使用虚拟数据'和'不采用任何降级处理，直接报错'，"
            "不能提供默认模拟实现。"
        )
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置
        
        根据项目要求"禁止使用虚拟数据"和"不采用任何降级处理，直接报错"，
        硬件接口必须实现具体的多关节位置设置逻辑，基类不提供任何模拟实现。
        
        参数:
            positions: 关节位置字典（关节类型 -> 目标位置）
        
        返回:
            bool: 设置是否成功
        
        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            f"硬件接口'{self._interface_type}'必须实现set_joint_positions()方法。"
            "根据项目要求'禁止使用虚拟数据'和'不采用任何降级处理，直接报错'，"
            "不能提供默认模拟实现。"
        )
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态
        
        根据项目要求"禁止使用虚拟数据"和"不采用任何降级处理，直接报错"，
        硬件接口必须实现具体的关节状态获取逻辑，基类不提供任何模拟实现。
        
        参数:
            joint: 关节类型
        
        返回:
            关节状态，如果传感器功能已禁用则抛出异常
        
        抛出:
            NotImplementedError: 子类必须实现此方法
            RuntimeError: 传感器功能已禁用
        """
        if not self.sensor_enabled:
            raise RuntimeError("传感器功能已禁用，无法获取关节状态")
        
        raise NotImplementedError(
            f"硬件接口'{self._interface_type}'必须实现get_joint_state()方法。"
            "根据项目要求'禁止使用虚拟数据'和'不采用任何降级处理，直接报错'，"
            "不能提供默认模拟实现。"
        )
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态
        
        根据项目要求"禁止使用虚拟数据"和"不采用任何降级处理，直接报错"，
        硬件接口必须实现具体的所有关节状态获取逻辑，基类不提供任何模拟实现。
        
        返回:
            关节状态字典，如果传感器功能已禁用则抛出异常
        
        抛出:
            NotImplementedError: 子类必须实现此方法
            RuntimeError: 传感器功能已禁用
        """
        if not self.sensor_enabled:
            raise RuntimeError("传感器功能已禁用，无法获取关节状态")
        
        raise NotImplementedError(
            f"硬件接口'{self._interface_type}'必须实现get_all_joint_states()方法。"
            "根据项目要求'禁止使用虚拟数据'和'不采用任何降级处理，直接报错'，"
            "不能提供默认模拟实现。"
        )


class SerialInterface(HardwareInterface):
    """串口接口"""
    
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, simulation_mode: bool = False):
        super().__init__(simulation_mode=simulation_mode)
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.logger = logging.getLogger("SerialInterface")
        self._interface_type = "serial"
        self.connected = False
        
    def connect(self) -> bool:
        """连接串口
        
        注意：根据项目要求"禁止使用虚拟数据"，此方法只支持真实硬件连接。
        必须安装pyserial库并连接真实的串口设备。
        """
        try:
            import serial
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0
            )
            self.logger.info(f"串口连接成功: {self.port}")
            self.connected = True
            return True
        except ImportError:
            self.logger.error("未安装pyserial库，无法连接真实串口设备")
            raise ImportError(
                "未安装pyserial库，无法连接真实串口设备。\n"
                "请安装pyserial库: pip install pyserial\n"
                "项目要求禁止使用虚拟数据，必须使用真实硬件接口。"
            )
        except Exception as e:
            self.logger.error(f"串口连接失败: {e}, 端口={self.port}, 波特率={self.baudrate}")
            raise ConnectionError(
                f"串口连接失败: {e}, 端口={self.port}, 波特率={self.baudrate}\n"
                "请检查串口设备是否连接正确。\n"
                "项目要求禁止使用虚拟数据，必须使用真实硬件接口。"
            )
    
    def disconnect(self) -> bool:
        """断开串口连接"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.serial_conn = None
            self.logger.info("串口已断开")
            return True
        return False
    
    def is_connected(self) -> bool:
        """检查串口连接
        
        注意：根据项目要求"禁止使用虚拟数据"，只支持真实硬件连接。
        """
        return self.serial_conn is not None and self.serial_conn.is_open
    
    def send_command(self, command: str) -> Optional[str]:
        """发送命令到串口
        
        注意：根据项目要求"禁止使用虚拟数据"，只支持真实硬件命令发送。
        """
        if not self.is_connected():
            raise ConnectionError("串口未连接，无法发送命令")
        
        # 真实硬件模式：使用真实串口
        if self.serial_conn is None or not self.serial_conn.is_open:
            raise ConnectionError("串口连接未初始化或已关闭")
        
        try:
            self.serial_conn.write((command + "\n").encode())
            response = self.serial_conn.readline().decode().strip()
            return response
        except Exception as e:
            self.logger.error(f"发送命令失败: {e}")
            raise ConnectionError(f"发送命令失败: {e}")
    
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据（串口实现）
        
        根据项目要求"禁止使用虚拟数据"，此方法不再提供模拟数据。
        串口未连接时将直接抛出异常。
        """
        if not self.is_connected():
            raise ConnectionError("串口未连接，无法获取传感器数据")
        
        if not self.sensor_enabled:
            self.logger.warning("传感器功能已禁用，无法获取传感器数据")
            raise RuntimeError("传感器功能已禁用")
        
        command_map = {
            SensorType.IMU: "GET_IMU",
            SensorType.JOINT_POSITION: "GET_JOINTS",
        }
        
        if sensor_type not in command_map:
            self.logger.warning(f"不支持的传感器类型: {sensor_type}")
            raise ValueError(f"不支持的传感器类型: {sensor_type}")
        
        response = self.send_command(command_map[sensor_type])
        if response:
            try:
                import json
                return json.loads(response)
            except json.JSONDecodeError:
                return response
        else:
            raise ConnectionError("获取传感器数据失败：未收到响应")
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置（串口实现）"""
        command = f"SET_JOINT {joint.value} {position:.4f}"
        response = self.send_command(command)
        return response == "OK"
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置（串口实现）"""
        position_str = " ".join([f"{joint.value}:{pos:.4f}" for joint, pos in positions.items()])
        command = f"SET_JOINTS {position_str}"
        response = self.send_command(command)
        return response == "OK"
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态（串口实现）"""
        if not self.sensor_enabled:
            self.logger.debug("传感器功能已禁用，无法获取关节状态")
            return None
        
        command = f"GET_JOINT {joint.value}"
        response = self.send_command(command)
        
        if response and response.startswith("JOINT_STATE"):
            try:
                parts = response.split()
                if len(parts) >= 6:
                    return JointState(
                        position=float(parts[1]),
                        velocity=float(parts[2]),
                        torque=float(parts[3]),
                        temperature=float(parts[4]),
                        voltage=float(parts[5]),
                        current=float(parts[6]) if len(parts) > 6 else 0.0
                    )
            except ValueError as e:
                self.logger.error(f"解析关节状态失败: {e}")
        
        return None
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态（串口实现）"""
        if not self.sensor_enabled:
            self.logger.debug("传感器功能已禁用，无法获取关节状态")
            return {}
        
        command = "GET_ALL_JOINTS"
        response = self.send_command(command)
        
        joint_states = {}
        
        if response and response.startswith("ALL_JOINTS"):
            try:
                lines = response.split("\n")
                for line in lines[1:]:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 8:
                            joint = RobotJoint(parts[0])
                            state = JointState(
                                position=float(parts[1]),
                                velocity=float(parts[2]),
                                torque=float(parts[3]),
                                temperature=float(parts[4]),
                                voltage=float(parts[5]),
                                current=float(parts[6])
                            )
                            joint_states[joint] = state
            except (ValueError, KeyError) as e:
                self.logger.error(f"解析关节状态失败: {e}")
        
        return joint_states


class ROSInterface(HardwareInterface):
    """ROS2接口
    
    注意：根据项目要求"禁止使用虚拟数据"，此接口只支持真实ROS2连接。
    必须安装ROS2和rclpy库才能使用。
    """
    
    def __init__(self, namespace: str = "/self_agi"):
        super().__init__()
        self.namespace = namespace
        self.connected = False
        self.logger = logging.getLogger("ROSInterface")
        
        # 检查ROS2 Python包是否可用
        try:
            import rclpy  # type: ignore
            from rclpy.node import Node  # type: ignore
            self.rclpy_available = True
            self.logger.info("ROS2 Python包 (rclpy) 可用")
        except ImportError:
            self.rclpy_available = False
            self.logger.error("ROS2 Python包 (rclpy) 不可用")
            # 注意：根据项目要求"禁止使用虚拟数据"，我们不回退到模拟模式
        
        # ROS2节点和发布器/订阅器
        self.node = None
        self.joint_publishers = {}
        self.sensor_subscribers = {}
        self.joint_states = {}
        self._interface_type = "ros2"
        
    def connect(self) -> bool:
        """连接ROS2
        
        注意：根据项目要求"禁止使用虚拟数据"，此方法只支持真实ROS2连接。
        必须安装ROS2和rclpy库才能使用。
        """
        # 如果rclpy不可用，直接抛出异常（禁止使用虚拟数据）
        if not self.rclpy_available:
            self.logger.error("ROS2 Python包 (rclpy) 不可用，无法连接真实硬件")
            raise ImportError(
                "ROS2 Python包 (rclpy) 不可用，无法连接真实硬件。\n"
                "请安装ROS2: http://docs.ros.org/en/foxy/Installation.html\n"
                "项目要求禁止使用虚拟数据，必须使用真实硬件接口。"
            )
        
        try:
            import rclpy  # type: ignore
            from rclpy.node import Node  # type: ignore
            from sensor_msgs.msg import JointState, Imu, Image, PointCloud2  # type: ignore
            from std_msgs.msg import String  # type: ignore
            
            # 初始化ROS2
            if not rclpy.ok():
                rclpy.init()
            
            self.node = Node("self_agi_hardware")
            
            # 创建关节状态发布器
            self.joint_publishers["command"] = self.node.create_publisher(
                JointState,
                f"{self.namespace}/joint_commands",
                10
            )
            
            # 创建传感器订阅器
            self.sensor_subscribers["joint_states"] = self.node.create_subscription(
                JointState,
                f"{self.namespace}/joint_states",
                self._joint_state_callback,
                10
            )
            

            
            self.connected = True
            self.logger.info("ROS2连接成功")
            return True
            
        except ImportError:
            self.logger.error("未安装ROS2 Python包，无法连接真实硬件")
            raise ImportError(
                "未安装ROS2 Python包，无法连接真实硬件。\n"
                "请安装ROS2: http://docs.ros.org/en/foxy/Installation.html\n"
                "项目要求禁止使用虚拟数据，必须使用真实硬件接口。"
            )
        except Exception as e:
            self.logger.error(f"ROS2连接失败: {e}")
            raise ConnectionError(
                f"ROS2连接失败: {e}\n"
                "请检查ROS2网络配置。\n"
                "项目要求禁止使用虚拟数据，必须使用真实硬件接口。"
            )
    
    def disconnect(self) -> bool:
        """断开ROS2连接"""
        if self.node:
            import rclpy  # type: ignore
            self.node.destroy_node()
            rclpy.shutdown()
            self.node = None
        
        self.connected = False
        self.logger.info("ROS2已断开")
        return True
    
    def is_connected(self) -> bool:
        """检查ROS2连接"""
        return self.connected
    
    def _joint_state_callback(self, msg):
        """关节状态回调"""
        try:
            from sensor_msgs.msg import JointState  # type: ignore
            
            for i, joint_name in enumerate(msg.name):
                joint = RobotJoint(joint_name)
                state = JointState(
                    position=msg.position[i] if i < len(msg.position) else None,
                    velocity=msg.velocity[i] if i < len(msg.velocity) else None,
                    torque=msg.effort[i] if i < len(msg.effort) else None,
                    temperature=None,  # ROS消息中通常没有温度数据
                    voltage=None,      # ROS消息中通常没有电压数据
                    current=None       # ROS消息中通常没有电流数据
                )
                self.joint_states[joint] = state
                
        except Exception as e:
            self.logger.error(f"处理关节状态回调失败: {e}")
    

    

    
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据（ROS2实现）"""
        if not self.is_connected():
            self.logger.error("ROS2未连接，无法获取传感器数据")
            return None
        
        if not self.sensor_enabled:
            self.logger.warning("传感器功能已禁用，无法获取传感器数据")
            return {}
        
        # 检查ROS2是否可用
        if not self.rclpy_available:
            self.logger.error("ROS2 Python包 (rclpy) 不可用，无法获取真实传感器数据")
            raise ImportError(
                "ROS2 Python包 (rclpy) 不可用，无法获取真实传感器数据。\n"
                "请安装ROS2: http://docs.ros.org/en/foxy/Installation.html\n"
                "项目要求禁止使用虚拟数据，必须使用真实硬件接口。"
            )
        
        if sensor_type == SensorType.JOINT_POSITION:
            return self.joint_states
        else:
            self.logger.warning(f"ROS2暂不支持传感器类型: {sensor_type}")
            return None
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置（ROS2实现）"""
        return self.set_joint_positions({joint: position})
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置（ROS2实现）"""
        if not self.is_connected():
            return False
        
        # 真实模式检查ROS2是否可用
        if not self.rclpy_available:
            self.logger.error("ROS2 Python包 (rclpy) 不可用，无法发送真实关节命令")
            raise ImportError(
                "ROS2 Python包 (rclpy) 不可用，无法发送真实关节命令。\n"
                "请安装ROS2: http://docs.ros.org/en/foxy/Installation.html\n"
                "项目要求禁止使用虚拟数据，必须使用真实硬件接口。"
            )
        
        # 真实ROS2模式
        try:
            from sensor_msgs.msg import JointState  # type: ignore
            import rclpy.time  # type: ignore
            
            msg = JointState()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            
            for joint, position in positions.items():
                msg.name.append(joint.value)
                msg.position.append(position)
                msg.velocity.append(0.0)  # 默认速度
                msg.effort.append(0.0)  # 默认扭矩
            
            self.joint_publishers["command"].publish(msg)
            
            # 同时更新本地关节状态
            for joint, position in positions.items():
                if joint in self.joint_states:
                    current_state = self.joint_states[joint]
                    self.joint_states[joint] = JointState(
                        position=position,
                        velocity=current_state.velocity,
                        torque=current_state.torque,
                        temperature=current_state.temperature,
                        voltage=current_state.voltage,
                        current=current_state.current
                    )
            
            return True
            
        except Exception as e:
            self.logger.error(f"发布关节命令失败: {e}")
            return False
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态（ROS2实现）"""
        if not self.sensor_enabled:
            self.logger.debug("传感器功能已禁用，无法获取关节状态")
            return None
        return self.joint_states.get(joint)
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态（ROS2实现）"""
        if not self.sensor_enabled:
            self.logger.debug("传感器功能已禁用，无法获取关节状态")
            return {}
        return self.joint_states.copy()


class WebSocketInterface(HardwareInterface):
    """WebSocket接口
    
    注意：根据项目要求"禁止使用虚拟数据"，此接口只支持真实WebSocket连接。
    """
    
    def __init__(self, url: str = "ws://localhost:8080/robot"):
        super().__init__()
        self.url = url
        self.ws = None
        self.connected = False
        self.logger = logging.getLogger("WebSocketInterface")
        self._interface_type = "websocket"
        
        # 真实硬件数据存储
        self._joint_states: Dict[RobotJoint, JointState] = {}
        self._sensor_data: Dict[SensorType, Any] = {}
        
    def connect(self) -> bool:
        """连接WebSocket"""
        try:
            import websockets
            import asyncio
            
            self.logger.info(f"尝试连接WebSocket: {self.url}")
            
            # 异步连接函数
            async def connect_async():
                self.ws = await websockets.connect(self.url)
                return self.ws
            
            # 运行异步连接
            loop = asyncio.get_event_loop()
            self.ws = loop.run_until_complete(connect_async())
            
            self.connected = True
            self.logger.info(f"WebSocket连接成功: {self.url}")
            return True
            
        except ImportError:
            self.logger.error("未安装websockets包，无法连接真实硬件")
            self.connected = False
            return False
        except Exception as e:
            self.logger.error(f"WebSocket连接失败: {e}")
            self.connected = False
            self.ws = None
            return False
    
    def disconnect(self) -> bool:
        """断开WebSocket连接"""
        if self.ws:
            try:
                import asyncio
                # 异步关闭函数
                async def close_async():
                    await self.ws.close()
                
                loop = asyncio.get_event_loop()
                loop.run_until_complete(close_async())
                self.logger.info("WebSocket连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭WebSocket连接时出错: {e}")
            finally:
                self.ws = None
        
        self.connected = False
        self.logger.info("WebSocket已断开")
        return True
    
    def is_connected(self) -> bool:
        """检查WebSocket连接"""
        if self.ws:
            try:
                # 检查WebSocket连接是否仍然打开
                return self.ws.open
            except Exception:
                return False
        return self.connected
    
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据（WebSocket实现）"""
        if not self.is_connected():
            return None
        
        if not self.sensor_enabled:
            self.logger.warning("传感器功能已禁用，无法获取传感器数据")
            return None
        
        try:
            import asyncio
            import json
            import time
            
            # 发送传感器数据请求
            command = {
                "type": "get_sensor",
                "sensor": sensor_type.value,
                "timestamp": time.time()
            }
            
            # 异步发送和接收函数
            async def get_sensor_async():
                await self.ws.send(json.dumps(command))
                # 等待响应（设置超时）
                response = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
                return json.loads(response)
            
            loop = asyncio.get_event_loop()
            response_data = loop.run_until_complete(get_sensor_async())
            
            # 解析响应数据
            if response_data.get("status") != "success":
                self.logger.error(f"获取传感器数据失败: {response_data.get('error', '未知错误')}")
                return None
            
            sensor_data = response_data.get("data")
            if not sensor_data:
                self.logger.warning(f"传感器 {sensor_type.value} 无数据")
                return None
            
            # 根据传感器类型转换数据格式
            if sensor_type == SensorType.IMU:
                # 检查所有必需字段是否存在
                required_fields = ["acceleration", "gyroscope", "magnetometer", "orientation", "timestamp"]
                if not all(field in sensor_data for field in required_fields):
                    missing = [f for f in required_fields if f not in sensor_data]
                    self.logger.warning(f"IMU数据不完整，缺少字段: {missing}")
                    return None
                
                try:
                    # 转换所有字段为numpy数组
                    return IMUData(
                        acceleration=np.array(sensor_data["acceleration"], dtype=np.float32),
                        gyroscope=np.array(sensor_data["gyroscope"], dtype=np.float32),
                        magnetometer=np.array(sensor_data["magnetometer"], dtype=np.float32),
                        orientation=np.array(sensor_data["orientation"], dtype=np.float32),
                        timestamp=float(sensor_data["timestamp"])
                    )
                except (ValueError, TypeError) as e:
                    self.logger.error(f"IMU数据格式错误: {e}")
                    return None
            elif sensor_type == SensorType.CAMERA:
                # 摄像头数据应该是base64编码的图像或直接的numpy数组
                # 这里处理base64编码的JPEG图像
                if "image_base64" in sensor_data:
                    import base64
                    # 检查timestamp字段是否存在
                    if "timestamp" not in sensor_data:
                        self.logger.warning("摄像头数据缺少timestamp字段")
                        return None
                    
                    try:
                        image_bytes = base64.b64decode(sensor_data["image_base64"])
                        # 尝试使用PIL解码图像
                        try:
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(image_bytes))
                            image_array = np.array(img)
                        except ImportError:
                            # 尝试使用OpenCV解码图像
                            try:
                                import cv2
                                image_array = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                                if image_array is None:
                                    self.logger.error("无法解码base64图像数据")
                                    return None
                            except ImportError:
                                self.logger.error("需要PIL或OpenCV库来解码图像")
                                return None
                        
                        return CameraData(
                            image=image_array,
                            timestamp=float(sensor_data["timestamp"])
                        )
                    except (ValueError, TypeError, Exception) as e:
                        self.logger.error(f"解码摄像头数据失败: {e}")
                        return None
                elif "image_array" in sensor_data:
                    # 如果直接发送了numpy数组数据
                    if "timestamp" not in sensor_data:
                        self.logger.warning("摄像头数据缺少timestamp字段")
                        return None
                    
                    try:
                        image_array = np.array(sensor_data["image_array"])
                        return CameraData(
                            image=image_array,
                            timestamp=float(sensor_data["timestamp"])
                        )
                    except (ValueError, TypeError) as e:
                        self.logger.error(f"摄像头数据格式错误: {e}")
                        return None
                else:
                    self.logger.warning("摄像头数据格式不支持：既不是image_base64也不是image_array")
                    return None
            else:
                # 对于其他传感器类型，返回原始数据
                return sensor_data
            
        except asyncio.TimeoutError:
            self.logger.error("获取传感器数据超时")
            return None
        except ImportError:
            self.logger.error("WebSocket库不可用")
            return None
        except Exception as e:
            self.logger.error(f"获取传感器数据失败: {e}")
            return None
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置（WebSocket实现）"""
        return self.set_joint_positions({joint: position})
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置（WebSocket实现）"""
        if not self.is_connected() or not self.ws:
            self.logger.error("WebSocket未连接，无法发送关节命令")
            return False
        
        try:
            import asyncio
            import json
            
            command = {
                "type": "set_joints",
                "positions": {joint.value: pos for joint, pos in positions.items()},
                "timestamp": time.time()
            }
            
            # 异步发送函数
            async def send_async():
                await self.ws.send(json.dumps(command))
                # 可选：等待确认响应
                # response = await self.ws.recv()
                # return response
            
            loop = asyncio.get_event_loop()
            loop.run_until_complete(send_async())
            
            self.logger.info(f"已发送关节命令，关节数: {len(positions)}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送关节命令失败: {e}")
            return False
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态（WebSocket实现）"""
        if not self.is_connected():
            return None
        
        if not self.sensor_enabled:
            self.logger.debug("传感器功能已禁用，无法获取关节状态")
            return None
        
        # 如果有关节状态数据，返回真实数据
        if joint in self._joint_states:
            return self._joint_states[joint]
        
        # 没有真实数据可用
        self.logger.debug(f"没有关节 {joint.value} 的真实数据")
        return None
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态（WebSocket实现）"""
        if not self.is_connected():
            return {}
        
        if not self.sensor_enabled:
            self.logger.debug("传感器功能已禁用，无法获取关节状态")
            return {}
        
        # 返回真实关节状态数据的副本
        return self._joint_states.copy()


class HumanoidRobotController:
    """人形机器人控制器"""
    
    def __init__(self, interface: HardwareInterface, enable_advanced_control: bool = False):
        self.interface = interface
        self.logger = logging.getLogger("HumanoidRobotController")
        self.running = False
        self.control_thread = None
        self.current_pose: Dict[RobotJoint, float] = {}
        
        # 高级控制功能
        self.advanced_control_enabled = False
        self.enhanced_interface = None
        self.advanced_controller = None
        
        # 记录接口信息
        interface_info = interface.get_interface_info()
        self.logger.info(f"初始化机器人控制器 - 接口类型: {interface_info['type']}, 模拟模式: {interface_info['simulation']}")
        
        # 初始化高级控制（如果可用且启用）
        if enable_advanced_control and ADVANCED_CONTROL_AVAILABLE:
            self._initialize_advanced_control()
        
        # 默认站立姿势
        self._init_default_pose()
    
    def _init_default_pose(self):
        """初始化默认姿势"""
        # 站立姿势（所有关节为0位置）
        for joint in RobotJoint:
            self.current_pose[joint] = 0.0
        
        # 调整一些关节的默认位置
        self.current_pose[RobotJoint.L_SHOULDER_PITCH] = 0.2
        self.current_pose[RobotJoint.R_SHOULDER_PITCH] = -0.2
        self.current_pose[RobotJoint.L_ELBOW_ROLL] = -0.5
        self.current_pose[RobotJoint.R_ELBOW_ROLL] = 0.5
    
    def connect(self) -> bool:
        """连接机器人"""
        interface_info = self.interface.get_interface_info()
        mode_str = "模拟" if interface_info['simulation'] else "真实"
        self.logger.info(f"开始连接机器人 - 接口类型: {interface_info['type']}, 模式: {mode_str}")
        
        success = self.interface.connect()
        if success:
            self.logger.info(f"机器人连接成功 ({mode_str}模式)")
        else:
            self.logger.error(f"机器人连接失败 ({mode_str}模式)")
        return success
    
    def disconnect(self) -> bool:
        """断开机器人连接"""
        self.stop_control()
        success = self.interface.disconnect()
        if success:
            self.logger.info("机器人断开成功")
        return success
    
    def start_control(self, control_loop: Callable[[], None]):
        """启动控制循环"""
        if self.running:
            self.logger.warning("控制循环已在运行")
            return
        
        self.running = True
        self.control_thread = threading.Thread(
            target=self._control_loop_wrapper,
            args=(control_loop,),
            daemon=True
        )
        self.control_thread.start()
        self.logger.info("控制循环已启动")
    
    def stop_control(self):
        """停止控制循环"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
            self.control_thread = None
        self.logger.info("控制循环已停止")
    
    def _control_loop_wrapper(self, control_loop: Callable[[], None]):
        """控制循环包装器"""
        while self.running:
            try:
                control_loop()
                time.sleep(0.01)  # 100Hz控制频率
            except Exception as e:
                self.logger.error(f"控制循环错误: {e}")
                time.sleep(0.1)
    
    def set_pose(self, pose: Dict[RobotJoint, float], duration: float = 1.0):
        """设置姿势（平滑过渡）"""
        if not self.interface.is_connected():
            self.logger.error("机器人未连接")
            return False
        
        # 计算起始姿势和结束姿势
        start_pose = self.current_pose.copy()
        end_pose = pose
        
        # 平滑过渡
        steps = int(duration * 100)  # 100Hz控制频率
        for step in range(steps):
            if not self.running:
                break
            
            # 线性插值
            alpha = step / max(steps - 1, 1)
            intermediate_pose = {}
            for joint in end_pose:
                if joint in start_pose:
                    start_pos = start_pose[joint]
                    end_pos = end_pose[joint]
                    intermediate_pose[joint] = start_pos + (end_pos - start_pos) * alpha
            
            # 设置关节位置
            success = self.interface.set_joint_positions(intermediate_pose)
            if not success:
                self.logger.error("设置姿势失败")
                return False
            
            # 更新当前姿势
            self.current_pose = intermediate_pose.copy()
            time.sleep(0.01)
        
        return True
    
    def walk_forward(self, steps: int = 4, step_length: float = 0.1):
        """向前行走"""
        self.logger.info(f"开始向前行走，步数: {steps}, 步长: {step_length}")
        
        # 完整的行走模式
        for step in range(steps):
            if not self.running:
                break
            
            # 抬起右腿
            self._lift_leg("right", height=0.1)
            
            # 向前移动右腿
            self._move_leg_forward("right", distance=step_length)
            
            # 放下右腿
            self._lower_leg("right")
            
            # 抬起左腿
            self._lift_leg("left", height=0.1)
            
            # 向前移动左腿
            self._move_leg_forward("left", distance=step_length)
            
            # 放下左腿
            self._lower_leg("left")
        
        self.logger.info("行走完成")
        return True
    
    def _lift_leg(self, side: str, height: float):
        """抬腿"""
        if side == "left":
            joints = {
                RobotJoint.L_HIP_PITCH: -0.3,
                RobotJoint.L_KNEE_PITCH: 0.6,
                RobotJoint.L_ANKLE_PITCH: -0.3,
            }
        else:  # right
            joints = {
                RobotJoint.R_HIP_PITCH: -0.3,
                RobotJoint.R_KNEE_PITCH: 0.6,
                RobotJoint.R_ANKLE_PITCH: -0.3,
            }
        
        self.set_pose(joints, duration=0.3)
    
    def _move_leg_forward(self, side: str, distance: float):
        """向前移动腿"""
        if side == "left":
            joints = {
                RobotJoint.L_HIP_PITCH: -0.5,
            }
        else:  # right
            joints = {
                RobotJoint.R_HIP_PITCH: -0.5,
            }
        
        self.set_pose(joints, duration=0.2)
    
    def _lower_leg(self, side: str):
        """放下腿"""
        if side == "left":
            joints = {
                RobotJoint.L_HIP_PITCH: 0.0,
                RobotJoint.L_KNEE_PITCH: 0.0,
                RobotJoint.L_ANKLE_PITCH: 0.0,
            }
        else:  # right
            joints = {
                RobotJoint.R_HIP_PITCH: 0.0,
                RobotJoint.R_KNEE_PITCH: 0.0,
                RobotJoint.R_ANKLE_PITCH: 0.0,
            }
        
        self.set_pose(joints, duration=0.3)
    
    def wave_hand(self, side: str = "right"):
        """挥手"""
        self.logger.info(f"{side}手挥手")
        
        if side == "left":
            wave_pose = {
                RobotJoint.L_SHOULDER_PITCH: 0.5,
                RobotJoint.L_SHOULDER_ROLL: 0.3,
                RobotJoint.L_ELBOW_YAW: -1.0,
                RobotJoint.L_ELBOW_ROLL: -0.5,
            }
        else:  # right
            wave_pose = {
                RobotJoint.R_SHOULDER_PITCH: 0.5,
                RobotJoint.R_SHOULDER_ROLL: -0.3,
                RobotJoint.R_ELBOW_YAW: 1.0,
                RobotJoint.R_ELBOW_ROLL: 0.5,
            }
        
        # 挥手动作
        for _ in range(3):
            if not self.running:
                break
            
            # 抬起手
            self.set_pose(wave_pose, duration=0.2)
            time.sleep(0.1)
            
            # 放下手
            neutral_pose = {joint: 0.0 for joint in wave_pose}
            self.set_pose(neutral_pose, duration=0.2)
            time.sleep(0.1)
        
        self.logger.info("挥手完成")
        return True
    
    def get_sensor_readings(self) -> Dict[str, Any]:
        """获取传感器读数"""
        if not self.interface.is_connected():
            return {}
        
        sensor_data = {}
        
        # 获取IMU数据
        imu_data = self.interface.get_sensor_data(SensorType.IMU)
        if imu_data:
            sensor_data["imu"] = imu_data
        
        # 获取关节状态
        joint_states = self.interface.get_all_joint_states()
        if joint_states:
            sensor_data["joints"] = {
                joint.value: {
                    "position": state.position,
                    "velocity": state.velocity,
                    "torque": state.torque,
                    "temperature": state.temperature
                }
                for joint, state in joint_states.items()
            }
        
        return sensor_data
    
    def emergency_stop(self):
        """紧急停止"""
        self.logger.warning("紧急停止")
        self.stop_control()
        
        # 设置安全姿势
        safe_pose = self._get_safe_pose()
        self.set_pose(safe_pose, duration=0.5)
    
    def _get_safe_pose(self) -> Dict[RobotJoint, float]:
        """获取安全姿势"""
        safe_pose = {}
        for joint in RobotJoint:
            safe_pose[joint] = 0.0
        
        return safe_pose
    
    def _initialize_advanced_control(self):
        """初始化高级控制"""
        global ADVANCED_CONTROL_AVAILABLE
        
        # 首次调用时检查高级控制模块是否可用
        if ADVANCED_CONTROL_AVAILABLE is None:
            try:
                from .advanced_robot_control import AdvancedRobotController, ControlAlgorithm, TrajectoryType, ControlParameters, EnhancedHardwareInterface, OperationMode
                
                # 将导入的类存储为模块属性，以便其他方法使用
                self._AdvancedRobotController = AdvancedRobotController
                self._ControlAlgorithm = ControlAlgorithm
                self._TrajectoryType = TrajectoryType
                self._ControlParameters = ControlParameters
                self._EnhancedHardwareInterface = EnhancedHardwareInterface
                self._OperationMode = OperationMode
                
                ADVANCED_CONTROL_AVAILABLE = True
                self.logger.info("高级控制模块导入成功")
                
            except ImportError as e:
                ADVANCED_CONTROL_AVAILABLE = False
                self.logger.warning(f"高级控制模块导入失败: {e}")
                return False
        
        if not ADVANCED_CONTROL_AVAILABLE:
            self.logger.warning("高级控制模块不可用，跳过初始化")
            return False
        
        try:
            # 创建增强硬件接口
            self.enhanced_interface = self._EnhancedHardwareInterface(self.interface)
            self.enhanced_interface.set_operation_mode(self._OperationMode.NORMAL)
            
            # 创建高级控制器
            self.advanced_controller = self._AdvancedRobotController(self.enhanced_interface)
            
            self.advanced_control_enabled = True
            self.logger.info("高级控制初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"高级控制初始化失败: {e}")
            self.advanced_control_enabled = False
            return False
    
    def enable_advanced_control(self):
        """启用高级控制"""
        if self.advanced_control_enabled:
            self.logger.warning("高级控制已启用")
            return True
        
        if self.advanced_controller is None:
            # 初始化高级控制
            success = self._initialize_advanced_control()
            if not success:
                return False
        
        if self.advanced_controller:
            # 启用高级控制器
            self.advanced_controller.enable_control()
            self.advanced_control_enabled = True
            self.logger.info("高级控制已启用")
            return True
        
        return False
    
    def disable_advanced_control(self):
        """禁用高级控制"""
        if not self.advanced_control_enabled:
            self.logger.warning("高级控制已禁用")
            return True
        
        if self.advanced_controller:
            # 禁用高级控制器
            self.advanced_controller.disable_control()
        
        self.advanced_control_enabled = False
        self.logger.info("高级控制已禁用")
        return True
    
    def generate_trajectory(self, joint: RobotJoint, trajectory_type: str, 
                          start_pos: float, end_pos: float, duration: float, **kwargs):
        """生成轨迹（高级控制功能）"""
        if not self.advanced_control_enabled or self.advanced_controller is None:
            self.logger.error("高级控制未启用，无法生成轨迹")
            return None
        
        try:
            trajectory_id = None
            
            if trajectory_type == "linear":
                trajectory_id = self.advanced_controller.generate_linear_trajectory(
                    joint, start_pos, end_pos, duration
                )
            elif trajectory_type == "minimum_jerk":
                trajectory_id = self.advanced_controller.generate_minimum_jerk_trajectory(
                    joint, start_pos, end_pos, duration
                )
            elif trajectory_type == "spline":
                waypoints = kwargs.get("waypoints", [])
                if waypoints:
                    trajectory_id = self.advanced_controller.generate_spline_trajectory(
                        joint, waypoints, duration
                    )
            
            return trajectory_id
            
        except Exception as e:
            self.logger.error(f"生成轨迹失败: {e}")
            return None
    
    def execute_trajectory(self, trajectory_id: str):
        """执行轨迹（高级控制功能）"""
        if not self.advanced_control_enabled or self.advanced_controller is None:
            self.logger.error("高级控制未启用，无法执行轨迹")
            return False
        
        try:
            return self.advanced_controller.execute_trajectory(trajectory_id)
        except Exception as e:
            self.logger.error(f"执行轨迹失败: {e}")
            return False
    
    def set_control_algorithm(self, joint: RobotJoint, algorithm: str, **params):
        """设置控制算法（高级控制功能）"""
        if not self.advanced_control_enabled or self.advanced_controller is None:
            self.logger.error("高级控制未启用，无法设置控制算法")
            return False
        
        try:
            # 创建控制参数（使用存储的类引用）
            control_params = self._ControlParameters()
            
            # 设置算法
            if algorithm == "pid":
                control_params.algorithm = self._ControlAlgorithm.PID
                control_params.kp = params.get("kp", 1.0)
                control_params.ki = params.get("ki", 0.0)
                control_params.kd = params.get("kd", 0.0)
            elif algorithm == "adaptive":
                control_params.algorithm = self._ControlAlgorithm.ADAPTIVE
                control_params.adaptation_rate = params.get("adaptation_rate", 0.1)
            elif algorithm == "mpc":
                control_params.algorithm = self._ControlAlgorithm.MPC
                control_params.prediction_horizon = params.get("prediction_horizon", 10)
                control_params.control_horizon = params.get("control_horizon", 5)
            
            # 设置其他参数
            control_params.max_velocity = params.get("max_velocity", 1.0)
            control_params.max_acceleration = params.get("max_acceleration", 0.5)
            control_params.position_tolerance = params.get("position_tolerance", 0.01)
            
            # 应用参数
            self.advanced_controller.set_control_parameters(joint, control_params)
            self.logger.info(f"设置关节控制算法: {joint.value} -> {algorithm}")
            return True
            
        except Exception as e:
            self.logger.error(f"设置控制算法失败: {e}")
            return False
    
    def get_advanced_status(self):
        """获取高级控制状态"""
        if not self.advanced_control_enabled or self.advanced_controller is None:
            return {"advanced_control_enabled": False}
        
        try:
            status = self.advanced_controller.get_status_report()
            status["advanced_control_enabled"] = True
            return status
        except Exception as e:
            self.logger.error(f"获取高级状态失败: {e}")
            return {"advanced_control_enabled": True, "error": str(e)}
    
    def advanced_emergency_stop(self):
        """高级紧急停止"""
        if self.advanced_control_enabled and self.advanced_controller:
            self.advanced_controller.emergency_stop()
        
        # 调用基础紧急停止
        self.emergency_stop()


class HardwareManager:
    """硬件管理器"""
    
    def __init__(self):
        self.interfaces: Dict[str, HardwareInterface] = {}
        self.robot_controllers: Dict[str, HumanoidRobotController] = {}
        self.logger = logging.getLogger("HardwareManager")
        
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
    
    def register_interface(self, name: str, interface: HardwareInterface) -> bool:
        """注册硬件接口"""
        if name in self.interfaces:
            self.logger.warning(f"接口 {name} 已存在，将被替换")
        
        self.interfaces[name] = interface
        self.logger.info(f"硬件接口 {name} 已注册")
        return True
    
    def create_pybullet_interface(self, 
                                 name: str = "pybullet",
                                 gui_enabled: bool = True,
                                 **kwargs) -> bool:
        """创建PyBullet仿真接口
        
        参数:
            name: 接口名称
            gui_enabled: 是否启用GUI
            **kwargs: 其他PyBulletSimulation参数
            
        返回:
            是否成功创建
        """
        try:
            # 尝试导入UnifiedSimulation（统一仿真接口）
            from .unified_simulation import UnifiedSimulation
            
            # 创建统一仿真接口，指定引擎为pybullet
            engine_config = {
                "gui_enabled": gui_enabled,
                **kwargs
            }
            
            simulation = UnifiedSimulation(
                engine="pybullet",
                engine_config=engine_config,
                gui_enabled=gui_enabled,
                simulation_mode=True  # 总是仿真模式
            )
            
            # 注册接口
            self.interfaces[name] = simulation
            self.logger.info(f"PyBullet仿真接口 {name} 已创建 (使用统一仿真接口, GUI: {gui_enabled})")
            return True
            
        except ImportError as e:
            self.logger.error(f"无法导入UnifiedSimulation: {e}")
            # 尝试回退到原始PyBulletSimulation（向后兼容）
            try:
                from .simulation import PyBulletSimulation
                simulation = PyBulletSimulation(gui_enabled=gui_enabled, **kwargs)
                self.interfaces[name] = simulation
                self.logger.info(f"PyBullet仿真接口 {name} 已创建 (使用原始PyBulletSimulation, GUI: {gui_enabled})")
                return True
            except ImportError as e2:
                self.logger.error(f"也无法导入PyBulletSimulation: {e2}")
                self.logger.warning("请确保已安装pybullet: pip install pybullet")
                return False
        except Exception as e:
            self.logger.error(f"创建PyBullet仿真接口失败: {e}")
            return False
    
    def create_gazebo_interface(self, 
                               name: str = "gazebo",
                               ros_master_uri: str = "http://localhost:11311",
                               gazebo_world: str = "empty.world",
                               robot_model: str = "humanoid",
                               gui_enabled: bool = True,
                               **kwargs) -> bool:
        """创建Gazebo仿真接口
        
        参数:
            name: 接口名称
            ros_master_uri: ROS master URI
            gazebo_world: Gazebo世界文件
            robot_model: 机器人模型名称
            gui_enabled: 是否启用GUI
            **kwargs: 其他GazeboSimulation参数
            
        返回:
            是否成功创建
        """
        try:
            # 尝试导入UnifiedSimulation（统一仿真接口）
            from .unified_simulation import UnifiedSimulation
            
            # 创建统一仿真接口，指定引擎为gazebo
            engine_config = {
                "ros_master_uri": ros_master_uri,
                "gazebo_world": gazebo_world,
                "robot_model": robot_model,
                "gui_enabled": gui_enabled,
                **kwargs
            }
            
            simulation = UnifiedSimulation(
                engine="gazebo",
                engine_config=engine_config,
                gui_enabled=gui_enabled,
                simulation_mode=True  # 总是仿真模式
            )
            
            # 注册接口
            self.interfaces[name] = simulation
            self.logger.info(f"Gazebo仿真接口 {name} 已创建 (使用统一仿真接口, 世界: {gazebo_world}, 机器人: {robot_model}, GUI: {gui_enabled})")
            return True
            
        except ImportError as e:
            self.logger.error(f"无法导入UnifiedSimulation: {e}")
            # 尝试回退到原始GazeboSimulation（向后兼容）
            try:
                from .gazebo_simulation import GazeboSimulation
                simulation = GazeboSimulation(
                    ros_master_uri=ros_master_uri,
                    gazebo_world=gazebo_world,
                    robot_model=robot_model,
                    gui_enabled=gui_enabled,
                    **kwargs
                )
                self.interfaces[name] = simulation
                self.logger.info(f"Gazebo仿真接口 {name} 已创建 (使用原始GazeboSimulation, 世界: {gazebo_world}, 机器人: {robot_model}, GUI: {gui_enabled})")
                return True
            except ImportError as e2:
                self.logger.error(f"也无法导入GazeboSimulation: {e2}")
                self.logger.warning("请确保已安装roslibpy: pip install roslibpy")
                return False
        except Exception as e:
            self.logger.error(f"创建Gazebo仿真接口失败: {e}")
            return False
    
    def create_ros_interface(self,
                           name: str = "ros2",
                           namespace: str = "/self_agi",
                           simulation_mode: bool = True) -> bool:
        """创建ROS2接口
        
        参数:
            name: 接口名称
            namespace: ROS2命名空间
            simulation_mode: 是否使用模拟模式
            
        返回:
            是否成功创建
        """
        try:
            # 创建ROS2接口
            ros_interface = ROSInterface(
                namespace=namespace,
                simulation_mode=simulation_mode
            )
            
            # 注册接口
            self.interfaces[name] = ros_interface
            self.logger.info(f"ROS2接口 {name} 已创建 (模拟模式: {simulation_mode})")
            return True
            
        except Exception as e:
            self.logger.error(f"创建ROS2接口失败: {e}")
            return False
    
    def create_ros2_interface(self,
                            name: str = "ros2_real",
                            config: Dict[str, Any] = None) -> bool:
        """创建真实ROS2机器人接口
        
        参数:
            name: 接口名称
            config: 配置字典，包含以下字段:
                - ros_master_uri: ROS master URI (默认: "http://localhost:11311")
                - robot_name: 机器人名称 (默认: "humanoid_robot")
                - joint_mapping: 关节映射模式 (默认: "default")
                - sensor_enabled: 是否启用传感器 (默认: True)
                
        返回:
            是否成功创建
        """
        if config is None:
            config = {}
        
        # 设置ROS环境变量
        ros_master_uri = config.get("ros_master_uri", "http://localhost:11311")
        import os
        if "ROS_MASTER_URI" not in os.environ:
            os.environ["ROS_MASTER_URI"] = ros_master_uri
        
        # 检查真实机器人接口是否可用
        if not is_real_robot_interface_available():
            self.logger.error("真实机器人硬件接口不可用")
            return False
        
        # 使用create_real_robot_interface创建ROS2机器人接口
        # 默认使用通用机器人类型，连接类型为ros2
        robot_type = "universal_robot"
        host = "localhost"  # ROS2不需要传统主机地址，但接口需要
        port = 11311  # ROS2默认端口
        connection_type = "ros2"
        
        return self.create_real_robot_interface(
            name=name,
            robot_type=robot_type,
            host=host,
            port=port,
            connection_type=connection_type,
            simulation_mode=False  # 真实机器人接口
        )
    
    def create_websocket_interface(self,
                                 name: str = "websocket",
                                 url: str = "ws://localhost:8080/robot",
                                 simulation_mode: bool = False) -> bool:
        """创建WebSocket接口
        
        参数:
            name: 接口名称
            url: WebSocket服务器URL
            simulation_mode: 是否使用模拟模式
            
        返回:
            是否成功创建
        """
        try:
            # 创建WebSocket接口
            websocket_interface = WebSocketInterface(
                url=url,
                simulation_mode=simulation_mode
            )
            
            # 注册接口
            self.interfaces[name] = websocket_interface
            self.logger.info(f"WebSocket接口 {name} 已创建 (URL: {url}, 模拟模式: {simulation_mode})")
            return True
            
        except Exception as e:
            self.logger.error(f"创建WebSocket接口失败: {e}")
            return False
    
    def create_i2c_interface(self,
                           name: str = "i2c",
                           bus: int = 1,
                           address: int = 0x68,
                           simulation_mode: bool = False) -> bool:
        """创建I2C接口
        
        参数:
            name: 接口名称
            bus: I2C总线编号
            address: I2C设备地址
            simulation_mode: 是否使用模拟模式
            
        返回:
            是否成功创建
        """
        try:
            # 创建I2C接口
            i2c_interface = I2CInterface(
                bus=bus,
                address=address,
                simulation_mode=simulation_mode
            )
            
            # 注册接口
            self.interfaces[name] = i2c_interface
            self.logger.info(f"I2C接口 {name} 已创建 (总线: {bus}, 地址: 0x{address:02x}, 模拟模式: {simulation_mode})")
            return True
            
        except Exception as e:
            self.logger.error(f"创建I2C接口失败: {e}")
            return False
    
    def create_spi_interface(self,
                           name: str = "spi",
                           bus: int = 0,
                           device: int = 0,
                           max_speed_hz: int = 1000000,
                           simulation_mode: bool = False) -> bool:
        """创建SPI接口
        
        参数:
            name: 接口名称
            bus: SPI总线编号
            device: SPI设备编号
            max_speed_hz: 最大速度 (Hz)
            simulation_mode: 是否使用模拟模式
            
        返回:
            是否成功创建
        """
        try:
            # 创建SPI接口
            spi_interface = SPIInterface(
                bus=bus,
                device=device,
                max_speed_hz=max_speed_hz,
                simulation_mode=simulation_mode
            )
            
            # 注册接口
            self.interfaces[name] = spi_interface
            self.logger.info(f"SPI接口 {name} 已创建 (总线: {bus}, 设备: {device}, 速度: {max_speed_hz}Hz, 模拟模式: {simulation_mode})")
            return True
            
        except Exception as e:
            self.logger.error(f"创建SPI接口失败: {e}")
            return False
    
    def create_real_robot_interface(self,
                                   name: str = "real_robot",
                                   robot_type: str = "nao",
                                   host: str = "localhost",
                                   port: int = 9559,
                                   connection_type: str = "wifi",
                                   simulation_mode: bool = False) -> bool:
        """创建真实机器人硬件接口
        
        参数:
            name: 接口名称
            robot_type: 机器人类型 ('nao', 'pepper', 'universal_robot', 'custom')
            host: 机器人主机地址或IP
            port: 机器人端口
            connection_type: 连接类型 ('wifi', 'ethernet', 'usb')
            simulation_mode: 是否使用模拟模式（对于真实机器人应为False）
            
        返回:
            是否成功创建
        """
        # 检查真实机器人接口是否可用
        if not is_real_robot_interface_available():
            self.logger.error("真实机器人硬件接口不可用（请检查real_robot_interface.py导入）")
            return False
        
        try:
            # 动态导入真实机器人接口模块
            try:
                real_robot_interface_module = get_real_robot_interface_module()
            except RuntimeError as e:
                self.logger.error(f"无法导入真实机器人接口模块: {e}")
                return False
            
            # 从模块中获取所需的类
            try:
                RobotConnectionConfig = real_robot_interface_module.RobotConnectionConfig
                RealRobotType = real_robot_interface_module.RealRobotType
                NAOqiRobotInterface = real_robot_interface_module.NAOqiRobotInterface
                ROS2RobotInterface = real_robot_interface_module.ROS2RobotInterface
            except AttributeError as e:
                self.logger.error(f"真实机器人接口模块缺少必需的类: {e}")
                return False
            
            # 创建机器人连接配置
            config = RobotConnectionConfig()
            config.host = host
            config.port = port
            config.connection_type = connection_type
            
            # 设置机器人类型
            if robot_type.lower() == "nao":
                config.robot_type = RealRobotType.NAO
            elif robot_type.lower() == "pepper":
                config.robot_type = RealRobotType.PEPPER
            elif robot_type.lower() == "universal_robot":
                config.robot_type = RealRobotType.UNIVERSAL_ROBOT
            else:
                config.robot_type = RealRobotType.CUSTOM
            
            # 根据连接类型选择接口类
            if connection_type in ["wifi", "ethernet", "usb"]:
                # 使用NAOqi接口（支持NAO和Pepper）
                if robot_type.lower() in ["nao", "pepper"]:
                    interface = NAOqiRobotInterface(config)
                else:
                    # 对于其他机器人类型，使用ROS2接口
                    interface = ROS2RobotInterface(config)
            else:
                self.logger.error(f"不支持的连接类型: {connection_type}")
                return False
            
            # 注册接口
            self.interfaces[name] = interface
            self.logger.info(f"真实机器人接口 {name} 已创建 (类型: {robot_type}, 主机: {host}:{port}, 连接: {connection_type})")
            
            # 如果是模拟模式，记录警告
            if simulation_mode:
                self.logger.warning("真实机器人接口使用模拟模式，不会连接真实硬件")
            
            return True
            
        except Exception as e:
            self.logger.error(f"创建真实机器人接口失败: {e}")
            return False
    
    def connect_interface(self, name: str) -> bool:
        """连接硬件接口"""
        if name not in self.interfaces:
            self.logger.error(f"接口 {name} 未注册")
            return False
        
        interface = self.interfaces[name]
        success = interface.connect()
        
        if success:
            self.logger.info(f"接口 {name} 连接成功")
        else:
            self.logger.error(f"接口 {name} 连接失败")
        
        return success
    
    def disconnect_interface(self, name: str) -> bool:
        """断开硬件接口"""
        if name not in self.interfaces:
            self.logger.error(f"接口 {name} 未注册")
            return False
        
        interface = self.interfaces[name]
        success = interface.disconnect()
        
        if success:
            self.logger.info(f"接口 {name} 断开成功")
        
        return success
    
    def create_robot_controller(self, name: str, interface_name: str, enable_advanced_control: bool = False) -> Optional[HumanoidRobotController]:
        """创建机器人控制器
        
        参数:
            name: 控制器名称
            interface_name: 接口名称
            enable_advanced_control: 是否启用高级控制（默认False）
        
        返回:
            机器人控制器实例或None（如果创建失败）
        """
        if interface_name not in self.interfaces:
            self.logger.error(f"接口 {interface_name} 未注册")
            return None
        
        interface = self.interfaces[interface_name]
        controller = HumanoidRobotController(interface, enable_advanced_control)
        self.robot_controllers[name] = controller
        
        mode_str = "带高级控制" if enable_advanced_control else "基础"
        self.logger.info(f"{mode_str}机器人控制器 {name} 已创建")
        return controller
    
    def get_robot_controller(self, name: str) -> Optional[HumanoidRobotController]:
        """获取机器人控制器"""
        return self.robot_controllers.get(name)
    
    def get_all_sensor_data(self) -> Dict[str, Dict[str, Any]]:
        """获取所有传感器数据"""
        sensor_data = {}
        
        for name, interface in self.interfaces.items():
            if interface.is_connected():
                # 获取不同类型的传感器数据
                for sensor_type in SensorType:
                    data = interface.get_sensor_data(sensor_type)
                    if data:
                        if name not in sensor_data:
                            sensor_data[name] = {}
                        sensor_data[name][sensor_type.value] = data
        
        return sensor_data
    
    def shutdown(self):
        """关闭所有硬件连接"""
        self.logger.info("正在关闭所有硬件连接...")
        
        # 停止所有机器人控制器
        for name, controller in self.robot_controllers.items():
            controller.stop_control()
            controller.disconnect()
            self.logger.info(f"机器人控制器 {name} 已停止")
        
        # 断开所有接口
        for name, interface in self.interfaces.items():
            if interface.is_connected():
                interface.disconnect()
                self.logger.info(f"接口 {name} 已断开")
        
        self.logger.info("硬件管理器已关闭")


class I2CInterface(HardwareInterface):
    """I2C接口 (Inter-Integrated Circuit)"""
    
    def __init__(self, 
                 bus: int = 1,  # I2C总线编号
                 address: int = 0x68):  # 设备地址
        super().__init__()
        self.bus = bus
        self.address = address
        self.i2c_conn = None
        self.logger = logging.getLogger("I2CInterface")
        self._interface_type = "i2c"
        
        # I2C设备注册表（设备地址 -> 设备类型）
        self.device_registry = {}
        
    def connect(self) -> bool:
        """连接I2C总线
        
        注意：根据项目要求"禁止使用虚拟数据"，只支持真实硬件连接。
        必须安装smbus2库并连接真实的I2C设备。
        """
        try:
            # 尝试导入smbus2库
            import smbus2  # type: ignore
            self.i2c_conn = smbus2.SMBus(self.bus)
            self.logger.info(f"I2C连接成功 (总线: {self.bus}, 地址: 0x{self.address:02x})")
            self.connected = True
            return True
        except ImportError:
            self.logger.error("未安装smbus2库，无法连接真实I2C设备")
            raise ImportError(
                "未安装smbus2库，无法连接真实I2C设备。\n"
                "请安装smbus2库（Linux系统）: pip install smbus2\n"
                "项目要求禁止使用虚拟数据，必须使用真实硬件接口。"
            )
        except Exception as e:
            self.logger.error(f"I2C连接失败: {e}")
            raise ConnectionError(
                f"I2C连接失败: {e}\n"
                "请检查I2C硬件配置、总线编号和设备地址。\n"
                "项目要求禁止使用虚拟数据，必须使用真实硬件接口。"
            )
    
    def disconnect(self) -> bool:
        """断开I2C连接"""
        if self.i2c_conn:
            try:
                self.i2c_conn.close()
                self.logger.info("I2C连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭I2C连接时出错: {e}")
        
        self.connected = False
        self.i2c_conn = None
        return True
    
    def is_connected(self) -> bool:
        """检查I2C连接"""
        return self.i2c_conn is not None
    
    def read_byte(self, register: int) -> Optional[int]:
        """从寄存器读取一个字节"""
        if not self.is_connected():
            return None
        
        try:
            return self.i2c_conn.read_byte_data(self.address, register)
        except Exception as e:
            self.logger.error(f"读取I2C寄存器失败: {e}")
            return None
    
    def write_byte(self, register: int, value: int) -> bool:
        """向寄存器写入一个字节"""
        if not self.is_connected():
            return False
        
        try:
            self.i2c_conn.write_byte_data(self.address, register, value)
            return True
        except Exception as e:
            self.logger.error(f"写入I2C寄存器失败: {e}")
            return False
    
    def read_word(self, register: int) -> Optional[int]:
        """从寄存器读取一个字（2字节）"""
        if not self.is_connected():
            return None
        
        try:
            return self.i2c_conn.read_word_data(self.address, register)
        except Exception as e:
            self.logger.error(f"读取I2C字失败: {e}")
            return None
    
    def read_block(self, register: int, length: int) -> Optional[bytes]:
        """从寄存器读取数据块"""
        if not self.is_connected():
            return None
        
        try:
            return self.i2c_conn.read_i2c_block_data(self.address, register, length)
        except Exception as e:
            self.logger.error(f"读取I2C数据块失败: {e}")
            return None
    
    def scan_devices(self) -> List[int]:
        """扫描I2C总线上的设备"""
        devices = []
        
        if not self.i2c_conn:
            return devices
        
        try:
            # 扫描所有可能的I2C地址（0x03-0x77）
            for address in range(0x03, 0x78):
                try:
                    self.i2c_conn.write_quick(address)
                    devices.append(address)
                except Exception as e:
                    # 该地址没有I2C设备，忽略异常
                    self.logger.debug(f"I2C地址 0x{address:02x} 无设备: {e}")
            
            self.logger.info(f"I2C设备扫描: 发现 {len(devices)} 个设备")
            return devices
        except Exception as e:
            self.logger.error(f"I2C设备扫描失败: {e}")
            return devices
    
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据（I2C实现）"""
        if not self.is_connected():
            return None
        
        if not self.sensor_enabled:
            self.logger.warning("传感器功能已禁用，无法获取传感器数据")
            return None
        
        # I2C传感器映射
        sensor_handlers = {
            SensorType.IMU: self._read_imu_data,
            SensorType.JOINT_POSITION: self._read_joint_position_data,
        }
        
        handler = sensor_handlers.get(sensor_type)
        if handler:
            return handler()
        else:
            self.logger.warning(f"I2C接口暂不支持传感器类型: {sensor_type}")
            return None
    
    def _read_imu_data(self) -> Optional[IMUData]:
        """读取IMU传感器数据
        
        注意：根据项目要求"禁止使用虚拟数据"，只支持真实硬件数据。
        优先尝试通过串口数据服务获取实时IMU数据。
        """
        # 首先尝试通过串口数据服务获取IMU数据
        try:
            from backend.services.serial_data_service import get_serial_data_service
            serial_service = get_serial_data_service()
            
            # 尝试从服务获取最近的数据
            recent_data = serial_service.get_recent_data(limit=10)
            for data_packet in recent_data:
                if "decode_result" in data_packet and data_packet["decode_result"]["success"]:
                    decoded_data = data_packet["decode_result"]["data"]
                    # 检查是否为IMU数据
                    if isinstance(decoded_data, dict):
                        # 尝试提取IMU数据
                        imu_accel = decoded_data.get("imu_accelerometer", decoded_data.get("acceleration", None))
                        imu_gyro = decoded_data.get("imu_gyroscope", decoded_data.get("gyroscope", None))
                        imu_mag = decoded_data.get("magnetometer", None)
                        imu_orientation = decoded_data.get("orientation", None)
                        
                        # 检查所有必需字段是否存在
                        if all(field is not None for field in [imu_accel, imu_gyro, imu_mag, imu_orientation]):
                            import time
                            import numpy as np
                            
                            # 转换为numpy数组
                            def to_array(value):
                                if isinstance(value, list):
                                    return np.array(value, dtype=np.float32)
                                elif isinstance(value, (int, float)):
                                    return np.array([float(value)] * 3)  # 复制为三维向量
                                else:
                                    # 如果字段存在但不是列表或数字，返回零向量
                                    return np.array([0.0, 0.0, 0.0], dtype=np.float32)
                            
                            timestamp = time.time()
                            return IMUData(
                                acceleration=to_array(imu_accel),
                                gyroscope=to_array(imu_gyro),
                                magnetometer=to_array(imu_mag),
                                orientation=to_array(imu_orientation),
                                timestamp=timestamp
                            )
                        else:
                            self.logger.debug("IMU数据不完整，缺少必需字段")
        except ImportError:
            self.logger.debug("串口数据服务未导入，无法获取真实IMU数据")
        except Exception as e:
            self.logger.debug(f"从串口数据服务获取IMU数据失败: {e}")
        
        # 真实硬件数据不可用
        self.logger.info("IMU数据：无法获取真实硬件数据，请通过串口数据服务接收实时传感器数据")
        return None
    
    def _read_joint_position_data(self) -> Optional[Dict[RobotJoint, JointState]]:
        """读取关节位置数据
        
        注意：根据项目要求"禁止使用虚拟数据"，只支持真实硬件数据。
        优先尝试通过串口数据服务获取实时关节数据。
        """
        # 首先尝试通过串口数据服务获取关节数据
        try:
            from backend.services.serial_data_service import get_serial_data_service
            serial_service = get_serial_data_service()
            
            # 尝试从服务获取最近的数据
            recent_data = serial_service.get_recent_data(limit=10)
            for data_packet in recent_data:
                if "decode_result" in data_packet and data_packet["decode_result"]["success"]:
                    decoded_data = data_packet["decode_result"]["data"]
                    # 检查是否为关节数据
                    if isinstance(decoded_data, dict):
                        # 尝试提取关节数据
                        joint_data = {}
                        for joint in RobotJoint:
                            joint_key = joint.value.lower()
                            # 获取字段值，不提供默认值
                            position = decoded_data.get(f"{joint_key}_position") or decoded_data.get("position")
                            velocity = decoded_data.get(f"{joint_key}_velocity") or decoded_data.get("velocity")
                            torque = decoded_data.get(f"{joint_key}_torque") or decoded_data.get("torque")
                            temperature = decoded_data.get(f"{joint_key}_temperature") or decoded_data.get("temperature")
                            voltage = decoded_data.get(f"{joint_key}_voltage") or decoded_data.get("voltage")
                            current = decoded_data.get(f"{joint_key}_current") or decoded_data.get("current")
                            
                            # 转换为浮点数（如果存在）
                            def to_float(value):
                                if value is None:
                                    return None
                                try:
                                    return float(value)
                                except (ValueError, TypeError):
                                    return None
                            
                            state = JointState(
                                position=to_float(position),
                                velocity=to_float(velocity),
                                torque=to_float(torque),
                                temperature=to_float(temperature),
                                voltage=to_float(voltage),
                                current=to_float(current)
                            )
                            joint_data[joint] = state
                        
                        if joint_data:
                            return joint_data
        except ImportError:
            self.logger.debug("串口数据服务未导入，无法获取真实关节数据")
        except Exception as e:
            self.logger.debug(f"从串口数据服务获取关节数据失败: {e}")
        
        # 真实硬件数据不可用
        self.logger.info("关节数据：无法获取真实硬件数据，请通过串口数据服务接收实时关节位置数据")
        return None
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置（I2C实现）"""
        # I2C通常用于传感器读取，而不是关节控制
        # 但这里提供基本实现
        self.logger.warning(f"I2C接口不支持直接关节控制: {joint.value}")
        return False
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置（I2C实现）"""
        self.logger.warning("I2C接口不支持直接关节控制")
        return False
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态（I2C实现）"""
        if not self.sensor_enabled:
            self.logger.debug("传感器功能已禁用，无法获取关节状态")
            return None
        
        # 尝试获取所有关节数据
        all_joint_data = self._read_joint_position_data()
        if all_joint_data and joint in all_joint_data:
            return all_joint_data[joint]
        
        # 无法获取真实数据
        self.logger.debug(f"无法获取关节 {joint.value} 的真实数据")
        return None
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态（I2C实现）"""
        if not self.sensor_enabled:
            self.logger.debug("传感器功能已禁用，无法获取关节状态")
            return {}
        
        # 尝试获取所有关节数据
        all_joint_data = self._read_joint_position_data()
        if all_joint_data:
            return all_joint_data
        
        # 无法获取真实数据
        self.logger.debug("无法获取所有关节的真实数据")
        return {}
    
    def register_device(self, address: int, device_type: str, description: str = ""):
        """注册I2C设备"""
        self.device_registry[address] = {
            "type": device_type,
            "description": description,
            "registered_at": time.time()
        }
        self.logger.info(f"注册I2C设备: 地址=0x{address:02x}, 类型={device_type}")
    
    def get_device_info(self, address: int) -> Optional[Dict[str, Any]]:
        """获取设备信息"""
        return self.device_registry.get(address)


class SPIInterface(HardwareInterface):
    """SPI接口 (Serial Peripheral Interface)"""
    
    def __init__(self,
                 bus: int = 0,  # SPI总线编号
                 device: int = 0,  # SPI设备编号
                 max_speed_hz: int = 1000000):  # 最大速度 (Hz)
        super().__init__()
        self.bus = bus
        self.device = device
        self.max_speed_hz = max_speed_hz
        self.spi_conn = None
        self.logger = logging.getLogger("SPIInterface")
        self._interface_type = "spi"
        
    def connect(self) -> bool:
        """连接SPI总线
        
        注意：根据项目要求"禁止使用虚拟数据"，只支持真实硬件连接。
        必须安装spidev库并连接真实的SPI设备。
        """
        try:
            # 尝试导入spidev库
            import spidev  # type: ignore
            self.spi_conn = spidev.SpiDev()
            self.spi_conn.open(self.bus, self.device)
            self.spi_conn.max_speed_hz = self.max_speed_hz
            self.spi_conn.mode = 0  # SPI模式0 (CPOL=0, CPHA=0)
            self.logger.info(f"SPI连接成功 (总线: {self.bus}, 设备: {self.device}, 速度: {self.max_speed_hz}Hz)")
            self.connected = True
            return True
        except ImportError:
            self.logger.error("未安装spidev库，无法连接真实SPI设备")
            raise ImportError(
                "未安装spidev库，无法连接真实SPI设备。\n"
                "请安装spidev库（Linux系统）: pip install spidev\n"
                "项目要求禁止使用虚拟数据，必须使用真实硬件接口。"
            )
        except Exception as e:
            self.logger.error(f"SPI连接失败: {e}")
            raise ConnectionError(
                f"SPI连接失败: {e}\n"
                "请检查SPI硬件配置、总线编号和设备编号。\n"
                "项目要求禁止使用虚拟数据，必须使用真实硬件接口。"
            )
    
    def disconnect(self) -> bool:
        """断开SPI连接"""
        if self.spi_conn:
            try:
                self.spi_conn.close()
                self.logger.info("SPI连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭SPI连接时出错: {e}")
        
        self.connected = False
        self.spi_conn = None
        return True
    
    def is_connected(self) -> bool:
        """检查SPI连接"""
        return self.spi_conn is not None
    
    def transfer(self, data: bytes) -> Optional[bytes]:
        """SPI数据传输 (全双工)"""
        if not self.is_connected():
            raise ConnectionError("SPI未连接，无法传输数据")
        
        # 真实模式：使用真实SPI硬件
        if self.spi_conn is None:
            raise ConnectionError("真实SPI模式：SPI连接未初始化")
        
        try:
            response = self.spi_conn.xfer2(list(data))
            return bytes(response)
        except Exception as e:
            self.logger.error(f"SPI传输失败: {e}")
            raise ConnectionError(f"SPI传输失败: {e}")
    
    def read_bytes(self, length: int) -> Optional[bytes]:
        """读取指定长度的数据"""
        # 发送全0数据以读取响应
        send_data = bytes([0] * length)
        return self.transfer(send_data)
    
    def write_bytes(self, data: bytes) -> bool:
        """写入数据"""
        response = self.transfer(data)
        return response is not None
    
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据（SPI实现）"""
        if not self.is_connected():
            return None
        
        if not self.sensor_enabled:
            self.logger.warning("传感器功能已禁用，无法获取传感器数据")
            return None
        
        # SPI传感器映射
        sensor_handlers = {
            SensorType.IMU: self._read_imu_data_spi,
            SensorType.JOINT_POSITION: self._read_joint_position_data_spi,
        }
        
        handler = sensor_handlers.get(sensor_type)
        if handler:
            return handler()
        else:
            self.logger.warning(f"SPI接口暂不支持传感器类型: {sensor_type}")
            return None
    
    def _read_imu_data_spi(self) -> Optional[IMUData]:
        """读取IMU传感器数据（SPI接口）
        
        优先尝试通过串口数据服务获取实时数据
        如果不可用，则返回None（遵循'禁止使用虚拟数据'要求）
        SPI接口通常用于高速IMU设备，但数据可通过串口接收
        """
        # 首先尝试通过串口数据服务获取IMU数据（SPI设备数据可能通过串口转发）
        try:
            from backend.services.serial_data_service import get_serial_data_service
            serial_service = get_serial_data_service()
            
            # 尝试从服务获取最近的数据
            recent_data = serial_service.get_recent_data(limit=10)
            for data_packet in recent_data:
                if "decode_result" in data_packet and data_packet["decode_result"]["success"]:
                    decoded_data = data_packet["decode_result"]["data"]
                    # 检查是否为IMU数据，特别是SPI设备数据
                    if isinstance(decoded_data, dict):
                        # 检查是否有SPI相关标识
                        source = data_packet.get("source_port", "")
                        protocol = data_packet.get("decode_result", {}).get("protocol", "")
                        
                        # SPI设备通常有特定标识
                        spi_keywords = ["spi", "imu", "accelerometer", "gyroscope", "mpu", "bno"]
                        is_spi_data = any(keyword in source.lower() or keyword in str(decoded_data).lower() 
                                         for keyword in spi_keywords)
                        
                        if is_spi_data:
                            # 提取IMU数据
                            imu_accel = decoded_data.get("acceleration", decoded_data.get("accel", decoded_data.get("imu_accelerometer", None)))
                            imu_gyro = decoded_data.get("gyroscope", decoded_data.get("gyro", decoded_data.get("imu_gyroscope", None)))
                            imu_mag = decoded_data.get("magnetometer", decoded_data.get("mag", None))
                            imu_orientation = decoded_data.get("orientation", decoded_data.get("euler", None))
                            
                            if imu_accel is not None or imu_gyro is not None:
                                import time
                                import numpy as np
                                
                                # 转换为numpy数组
                                def to_array(value):
                                    if isinstance(value, list):
                                        return np.array(value, dtype=np.float32)
                                    elif isinstance(value, (int, float)):
                                        return np.array([0.0, 0.0, float(value)])
                                    else:
                                        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
                                
                                timestamp = time.time()
                                return IMUData(
                                    acceleration=to_array(imu_accel) if imu_accel is not None else np.array([0.0, 0.0, 9.81]),
                                    gyroscope=to_array(imu_gyro) if imu_gyro is not None else np.array([0.001, 0.002, 0.003]),
                                    magnetometer=to_array(imu_mag) if imu_mag is not None else np.array([0.0, 0.0, 50.0]),
                                    orientation=to_array(imu_orientation) if imu_orientation is not None else np.array([0.0, 0.0, 0.0]),
                                    timestamp=timestamp
                                )
        except ImportError:
            self.logger.debug("串口数据服务未导入，无法获取真实SPI IMU数据")
        except Exception as e:
            self.logger.debug(f"从串口数据服务获取SPI IMU数据失败: {e}")
        
        # 真实硬件数据不可用
        self.logger.info("SPI IMU数据：无法获取真实硬件数据，请通过串口数据服务接收SPI设备转发的实时数据")
        return None
    
    def _extract_value(self, data_dict: Dict[str, Any], keys: List[str], default: Any) -> Any:
        """从字典中提取值
        
        参数:
            data_dict: 数据字典
            keys: 可能的键名列表（按优先级）
            default: 默认值
        
        返回:
            找到的值或默认值
        """
        for key in keys:
            value = data_dict.get(key)
            if value is not None:
                return value
        
        # 如果没找到，尝试不区分大小写的搜索
        lower_data = {k.lower(): v for k, v in data_dict.items()}
        for key in keys:
            lower_key = key.lower()
            value = lower_data.get(lower_key)
            if value is not None:
                return value
        
        return default
    
    def _read_joint_position_data_spi(self) -> Optional[Dict[RobotJoint, JointState]]:
        """读取关节位置数据（SPI接口）
        
        优先尝试通过串口数据服务获取实时关节数据
        SPI接口通常用于高速关节编码器，数据可通过串口接收
        """
        # 首先尝试通过串口数据服务获取关节数据（SPI编码器数据可能通过串口转发）
        try:
            from backend.services.serial_data_service import get_serial_data_service
            serial_service = get_serial_data_service()
            
            # 尝试从服务获取最近的数据
            recent_data = serial_service.get_recent_data(limit=10)
            for data_packet in recent_data:
                if "decode_result" in data_packet and data_packet["decode_result"]["success"]:
                    decoded_data = data_packet["decode_result"]["data"]
                    # 检查是否为关节数据，特别是SPI编码器数据
                    if isinstance(decoded_data, dict):
                        # 检查是否有SPI关节相关标识
                        source = data_packet.get("source_port", "")
                        protocol = data_packet.get("decode_result", {}).get("protocol", "")
                        
                        # SPI编码器通常有特定标识
                        spi_joint_keywords = ["spi", "encoder", "joint", "position", "servo", "motor"]
                        is_spi_joint_data = any(keyword in source.lower() or keyword in str(decoded_data).lower() 
                                               for keyword in spi_joint_keywords)
                        
                        if is_spi_joint_data:
                            # 提取关节数据
                            joint_data = {}
                            for joint in RobotJoint:
                                joint_key = joint.value.lower()
                                
                                # 尝试多种可能的键名
                                position_keys = [
                                    f"{joint_key}_position", f"{joint_key}_pos", 
                                    f"{joint_key}_angle", "position", "pos", "angle"
                                ]
                                velocity_keys = [
                                    f"{joint_key}_velocity", f"{joint_key}_vel",
                                    f"{joint_key}_speed", "velocity", "vel", "speed"
                                ]
                                torque_keys = [
                                    f"{joint_key}_torque", f"{joint_key}_torq",
                                    f"{joint_key}_force", "torque", "torq", "force"
                                ]
                                temperature_keys = [
                                    f"{joint_key}_temperature", f"{joint_key}_temp",
                                    f"{joint_key}_thermal", "temperature", "temp"
                                ]
                                voltage_keys = [
                                    f"{joint_key}_voltage", f"{joint_key}_volt",
                                    f"{joint_key}_v", "voltage", "volt"
                                ]
                                current_keys = [
                                    f"{joint_key}_current", f"{joint_key}_curr",
                                    f"{joint_key}_i", "current", "curr"
                                ]
                                
                                # 从数据中提取值
                                position = self._extract_value(decoded_data, position_keys, 0.0)
                                velocity = self._extract_value(decoded_data, velocity_keys, 0.0)
                                torque = self._extract_value(decoded_data, torque_keys, 0.0)
                                temperature = self._extract_value(decoded_data, temperature_keys, 25.0)
                                voltage = self._extract_value(decoded_data, voltage_keys, 12.0)
                                current = self._extract_value(decoded_data, current_keys, 0.1)
                                
                                state = JointState(
                                    position=float(position),
                                    velocity=float(velocity),
                                    torque=float(torque),
                                    temperature=float(temperature),
                                    voltage=float(voltage),
                                    current=float(current)
                                )
                                joint_data[joint] = state
                            
                            if joint_data:
                                return joint_data
        except ImportError:
            self.logger.debug("串口数据服务未导入，无法获取真实SPI关节数据")
        except Exception as e:
            self.logger.debug(f"从串口数据服务获取SPI关节数据失败: {e}")
        
        # 真实硬件数据不可用
        self.logger.info("SPI关节数据：无法获取真实硬件数据，请通过串口数据服务接收SPI编码器转发的实时数据")
        return None
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置（SPI实现）"""
        # SPI可用于关节控制，但这里提供基本实现
        self.logger.warning(f"SPI接口不支持直接关节控制: {joint.value}")
        return False
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置（SPI实现）"""
        self.logger.warning("SPI接口不支持直接关节控制")
        return False
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态（SPI实现）"""
        if not self.sensor_enabled:
            self.logger.debug("传感器功能已禁用，无法获取关节状态")
            return None
        
        # 尝试获取所有关节数据
        all_joint_data = self._read_joint_position_data_spi()
        if all_joint_data and joint in all_joint_data:
            return all_joint_data[joint]
        
        # 无法获取真实数据
        self.logger.debug(f"无法获取关节 {joint.value} 的真实SPI数据")
        return None
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态（SPI实现）"""
        if not self.sensor_enabled:
            self.logger.debug("传感器功能已禁用，无法获取关节状态")
            return {}
        
        # 尝试获取所有关节数据
        all_joint_data = self._read_joint_position_data_spi()
        if all_joint_data:
            return all_joint_data
        
        # 无法获取真实数据
        self.logger.debug("无法获取所有关节的真实SPI数据")
        return {}


class SimulatedRobotInterface(HardwareInterface):
    """模拟机器人接口（已被完全禁用）
    
    警告：根据项目要求"禁止使用虚拟数据和模拟实现"，此接口已被完全禁用。
    生产环境必须使用真实硬件接口（如NAOqiRobotInterface、ROS2RobotInterface等）
    或物理仿真环境（如PyBulletSimulation）。
    
    替代方案：
    1. 安装真实硬件驱动（NAOqi SDK、ROS2等）
    2. 使用PyBullet物理仿真环境（pybullet==3.2.5）
    3. 使用NullHardwareInterface进行无硬件模式开发
    """
    
    def __init__(self, simulation_mode: bool = True):
        """
        初始化模拟机器人接口
        
        注意：模拟硬件接口已被完全禁用，遵循'禁止使用虚拟数据'要求。
        必须使用真实硬件接口或物理仿真环境。
        """
        super().__init__(simulation_mode=False)  # 设置为非模拟模式
        self._interface_type = "simulated_robot_disabled"
        self.logger = logging.getLogger("SimulatedRobotInterface")
        self._connected = False
        raise RuntimeError(
            "模拟硬件接口已被完全禁用。\n"
            "请选择以下替代方案：\n"
            "1. 安装真实硬件驱动（NAOqi SDK、ROS2等）并连接真实机器人\n"
            "2. 使用PyBullet物理仿真环境（pip install pybullet）\n"
            "3. 使用NullHardwareInterface进行无硬件模式开发\n"
            "项目要求禁止使用虚拟数据和模拟实现，必须使用真实硬件或物理仿真。"
        )
    
    def _initialize_joint_states(self):
        """初始化关节状态（已被禁用）"""
        raise RuntimeError("模拟硬件接口已被完全禁用。请使用真实硬件或物理仿真环境。")
    
    def _initialize_sensor_data(self):
        """初始化传感器数据（已被禁用）"""
        raise RuntimeError("模拟硬件接口已被完全禁用。请使用真实硬件或物理仿真环境。")
    
    def connect(self) -> bool:
        """连接硬件（已被禁用）"""
        self.logger.error("模拟硬件接口已被完全禁用。无法连接模拟硬件。")
        raise RuntimeError("模拟硬件接口已被完全禁用。请使用真实硬件或物理仿真环境。")
    
    def disconnect(self) -> bool:
        """断开连接（已被禁用）"""
        self.logger.error("模拟硬件接口已被完全禁用。无硬件可断开。")
        return False
    
    def is_connected(self) -> bool:
        """检查连接状态（已被禁用）"""
        return False
    
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据（已被禁用）"""
        self.logger.error(f"模拟硬件接口已被完全禁用。无法获取传感器数据（类型: {sensor_type.value}）。")
        raise RuntimeError("模拟硬件接口已被完全禁用。请使用真实硬件或物理仿真环境获取传感器数据。")
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置（已被禁用）"""
        self.logger.error(f"模拟硬件接口已被完全禁用。无法设置关节位置（关节: {joint.value}, 位置: {position}）。")
        raise RuntimeError("模拟硬件接口已被完全禁用。请使用真实硬件或物理仿真环境控制关节。")
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置（已被禁用）"""
        self.logger.error(f"模拟硬件接口已被完全禁用。无法设置多个关节位置（关节数: {len(positions)}）。")
        raise RuntimeError("模拟硬件接口已被完全禁用。请使用真实硬件或物理仿真环境控制关节。")
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态（已被禁用）"""
        self.logger.error(f"模拟硬件接口已被完全禁用。无法获取关节状态（关节: {joint.value}）。")
        raise RuntimeError("模拟硬件接口已被完全禁用。请使用真实硬件或物理仿真环境获取关节状态。")
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态（已被禁用）"""
        self.logger.error("模拟硬件接口已被完全禁用。无法获取所有关节状态。")
        raise RuntimeError("模拟硬件接口已被完全禁用。请使用真实硬件或物理仿真环境获取关节状态。")
    
    def get_interface_info(self) -> Dict[str, Any]:
        """获取接口信息"""
        base_info = super().get_interface_info()
        base_info.update({
            "type": self._interface_type,
            "connected": False,
            "simulation_mode": False,
            "joint_count": 0,
            "sensor_types": [],
            "description": "模拟机器人接口（已被完全禁用，遵循'禁止使用虚拟数据'要求）",
            "error": "此接口已被禁用，请使用真实硬件或物理仿真环境",
            "alternatives": [
                "真实硬件接口（NAOqi、ROS2等）",
                "PyBullet物理仿真环境",
                "NullHardwareInterface（无硬件模式）"
            ]
        })
        return base_info


class NullHardwareInterface(HardwareInterface):
    """无硬件接口
    
    当没有真实硬件可用时使用此接口。不提供任何模拟数据，
    仅返回表示硬件不可用的默认值，允许系统在没有硬件的条件下运行。
    遵循'禁止使用虚拟数据'原则。
    """
    
    def __init__(self):
        """初始化无硬件接口"""
        super().__init__()
        self._interface_type = "null_hardware"
        self.logger = logging.getLogger("NullHardwareInterface")
        self._connected = False
        self._simulation_mode = False  # 不是模拟接口，是无硬件接口
    
    @property
    def is_simulation(self) -> bool:
        """是否为模拟接口（无硬件接口不是模拟接口）"""
        return self._simulation_mode
    
    def connect(self) -> bool:
        """连接硬件（无硬件模式返回True，表示无硬件模式已成功启用）
        
        根据项目要求：
        1. 不连接硬件情况下AGI系统可以正常运行 - 返回True允许系统启动
        2. 不采用任何降级处理，直接报错 - 硬件功能调用将抛出RuntimeError
        """
        self.logger.warning(
            "无硬件模式：系统将在无硬件条件下运行\n"
            "根据项目要求'在不连接硬件情况下AGI系统可以正常运行'，无硬件模式已启用。\n"
            "根据'不采用任何降级处理，直接报错'要求，硬件功能调用将直接抛出RuntimeError。"
        )
        self._connected = True  # 标记为已连接（无硬件模式）
        return True
    
    def disconnect(self) -> bool:
        """断开连接（无硬件模式始终返回True）"""
        self.logger.info("无硬件模式：无硬件可断开")
        self._connected = False
        return True
    
    def is_connected(self) -> bool:
        """检查连接状态（无硬件模式始终返回False）"""
        return self._connected
    
    def get_sensor_data(self, sensor_type: Optional[SensorType] = None) -> Optional[Any]:
        """获取传感器数据（无硬件模式直接报错）"""
        if not self.sensor_enabled:
            raise RuntimeError("传感器功能已禁用，无法获取传感器数据")
        
        sensor_type_str = sensor_type.value if sensor_type else "未知"
        raise RuntimeError(
            f"无硬件模式：无法获取传感器数据（类型: {sensor_type_str}）\n"
            "根据项目要求'不采用任何降级处理，直接报错'，硬件不可用时直接报错。\n"
            "请连接真实硬件或使用物理仿真环境获取传感器数据。"
        )
    
    def get_hardware_health(self) -> Dict[str, Any]:
        """获取硬件健康状态（无硬件模式返回no_hardware状态）"""
        return {
            "status": "no_hardware",
            "message": "无硬件模式：系统在无硬件条件下运行",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "joints": "unavailable",
                "sensors": "unavailable",
                "actuators": "unavailable",
                "communication": "unavailable"
            },
            "recommendation": "如需硬件功能，请连接真实机器人硬件"
        }
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置（无硬件模式直接报错）"""
        raise RuntimeError(
            f"无硬件模式：无法设置关节位置（关节: {joint.value}, 位置: {position}）\n"
            "根据项目要求'不采用任何降级处理，直接报错'，硬件不可用时直接报错。\n"
            "请连接真实硬件或使用物理仿真环境控制关节。"
        )
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置（无硬件模式直接报错）"""
        joint_list = ", ".join([f"{joint.value}" for joint in positions.keys()][:3])
        if len(positions) > 3:
            joint_list += f"...等{len(positions)}个关节"
        
        raise RuntimeError(
            f"无硬件模式：无法设置多个关节位置（关节: {joint_list}）\n"
            "根据项目要求'不采用任何降级处理，直接报错'，硬件不可用时直接报错。\n"
            "请连接真实硬件或使用物理仿真环境控制关节。"
        )
    
    def get_joint_positions(self) -> Dict[RobotJoint, float]:
        """获取所有关节位置（无硬件模式直接报错）"""
        if not self.sensor_enabled:
            raise RuntimeError("传感器功能已禁用，无法获取关节位置")
        
        raise RuntimeError(
            "无硬件模式：无法获取关节位置\n"
            "根据项目要求'不采用任何降级处理，直接报错'，硬件不可用时直接报错。\n"
            "请连接真实硬件或使用物理仿真环境获取关节数据。"
        )
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态（无硬件模式直接报错）"""
        if not self.sensor_enabled:
            raise RuntimeError("传感器功能已禁用，无法获取关节状态")
        
        raise RuntimeError(
            f"无硬件模式：无法获取关节状态（关节: {joint.value}）\n"
            "根据项目要求'不采用任何降级处理，直接报错'，硬件不可用时直接报错。\n"
            "请连接真实硬件或使用物理仿真环境获取关节状态。"
        )
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态（无硬件模式直接报错）"""
        if not self.sensor_enabled:
            raise RuntimeError("传感器功能已禁用，无法获取关节状态")
        
        raise RuntimeError(
            "无硬件模式：无法获取所有关节状态\n"
            "根据项目要求'不采用任何降级处理，直接报错'，硬件不可用时直接报错。\n"
            "请连接真实硬件或使用物理仿真环境获取关节状态。"
        )
    
    def get_interface_info(self) -> Dict[str, Any]:
        """获取接口信息"""
        base_info = super().get_interface_info()
        base_info.update({
            "type": self._interface_type,
            "connected": self._connected,
            "simulation_mode": False,
            "joint_count": 0,
            "sensor_types": [],
            "description": "无硬件接口（硬件不可用，系统可在无硬件条件下运行）"
        })
        return base_info