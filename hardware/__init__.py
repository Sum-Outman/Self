"""
硬件接口模块
提供人形机器人控制、传感器集成和仿真环境接口

模块导出：
- HardwareInterface: 硬件接口抽象基类
- RobotJoint: 机器人关节枚举
- SensorType: 传感器类型枚举
- JointState: 关节状态数据类
- IMUData: IMU传感器数据类
- CameraData: 摄像头数据类
- LidarData: 激光雷达数据类
- SerialInterface: 串口硬件接口
- ROSInterface: ROS2硬件接口
- PyBulletSimulation: PyBullet物理仿真接口
- GazeboSimulation: Gazebo物理仿真接口
- UnifiedSimulation: 统一仿真接口（支持多种物理引擎）
- EnhancedHardwareInterface: 增强型硬件接口（带监控和安全功能）
- HardwareMonitor: 硬件监控器
"""

from .robot_controller import (
    HardwareInterface,
    RobotJoint,
    SensorType,
    JointState,
    IMUData,
    CameraData,
    LidarData,
    SerialInterface,
    ROSInterface
)

from .simulation import PyBulletSimulation
from .gazebo_simulation import GazeboSimulation
from .unified_simulation import UnifiedSimulation
from .hardware_monitor import HardwareMonitor, HardwareError, HardwareErrorLevel, HardwareErrorType

__all__ = [
    "HardwareInterface",
    "RobotJoint", 
    "SensorType",
    "JointState",
    "IMUData",
    "CameraData", 
    "LidarData",
    "SerialInterface",
    "ROSInterface",
    "PyBulletSimulation",
    "GazeboSimulation",
    "UnifiedSimulation",
    "HardwareMonitor",
    "HardwareError",
    "HardwareErrorLevel",
    "HardwareErrorType"
]

__version__ = "1.0.0"