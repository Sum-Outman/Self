"""
系统控制接口模块

功能：
- 串口通信：与外部硬件设备通信
- 硬件设备控制：电机、传感器、执行器等控制
- 传感器数据接入：读取各种传感器数据
- 系统状态监控：监控硬件状态和性能
- 设备管理：硬件设备的注册、发现和管理
"""

from .serial_controller import SerialController
from .hardware_manager import HardwareManager
from .sensor_interface import SensorInterface
from .motor_controller import MotorController
from .system_monitor import SystemMonitor
from .autonomous_mode_manager import AutonomousModeManager, get_autonomous_mode_manager, AutonomousState, GoalPriority

__all__ = [
    "SerialController",
    "HardwareManager",
    "SensorInterface",
    "MotorController",
    "SystemMonitor",
    "AutonomousModeManager",
    "get_autonomous_mode_manager",
    "AutonomousState",
    "GoalPriority",
]
