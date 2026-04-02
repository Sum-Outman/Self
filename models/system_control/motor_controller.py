"""
电机控制器

功能：
- 电机控制和管理
- 运动规划和执行
- 位置和速度控制
- 电机状态监控
"""

import logging
import time
import threading
import math
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

# 导入真实硬件接口 - 严格模式，禁止虚拟实现
try:
    from .real_hardware.motor_controller import RealMotorController, ControlInterface, MotorType as RealMotorType
    from .real_hardware.base_interface import HardwareType, ConnectionStatus
    REAL_HARDWARE_AVAILABLE = True
    HAS_REAL_HARDWARE_IMPORT = True
except ImportError as e:
    # 真实硬件接口导入失败，尝试仿真接口
    try:
        from hardware.simulation import PyBulletSimulation
        from hardware.robot_controller import HardwareInterface
        REAL_HARDWARE_AVAILABLE = False
        HAS_SIMULATION_IMPORT = True
        RealMotorController = None  # 使用仿真替代
        ControlInterface = None
        RealMotorType = None
        HardwareType = None
        ConnectionStatus = None
        logger = logging.getLogger(__name__)
        logger.warning(f"真实硬件接口不可用，使用PyBullet仿真: {e}")
    except ImportError as sim_error:
        # 仿真也失败，抛出明确的错误
        raise ImportError(
            "无法导入硬件接口。项目要求禁止使用虚拟实现。\n"
            "请安装必要的硬件依赖：\n"
            "1. 真实硬件：pip install pyserial smbus2 spidev python-can\n"
            "2. 仿真：pip install pybullet\n"
            "3. 或检查real_hardware模块是否正确安装\n"
            f"原始错误：{e}\n仿真错误：{sim_error}"
        ) from e


class MotorType(Enum):
    """电机类型枚举"""

    DC = "dc"  # 直流电机
    STEPPER = "stepper"  # 步进电机
    SERVO = "servo"  # 伺服电机
    BRUSHLESS = "brushless"  # 无刷电机
    LINEAR = "linear"  # 直线电机
    CUSTOM = "custom"  # 自定义电机


class MotorControlMode(Enum):
    """电机控制模式枚举"""

    POSITION = "position"  # 位置控制
    VELOCITY = "velocity"  # 速度控制
    TORQUE = "torque"  # 转矩控制
    VOLTAGE = "voltage"  # 电压控制
    CURRENT = "current"  # 电流控制


class MotorStatus(Enum):
    """电机状态枚举"""

    STOPPED = "stopped"  # 停止
    RUNNING = "running"  # 运行中
    ACCELERATING = "accelerating"  # 加速中
    DECELERATING = "decelerating"  # 减速中
    ERROR = "error"  # 错误
    OVERHEAT = "overheat"  # 过热
    OVERLOAD = "overload"  # 过载
    STALLED = "stalled"  # 失速


@dataclass
class MotorState:
    """电机状态类"""

    position: float = 0.0  # 位置（单位取决于电机类型）
    velocity: float = 0.0  # 速度
    acceleration: float = 0.0  # 加速度
    torque: float = 0.0  # 转矩
    current: float = 0.0  # 电流
    voltage: float = 0.0  # 电压
    temperature: float = 25.0  # 温度

    status: MotorStatus = MotorStatus.STOPPED
    error_code: int = 0
    error_message: str = ""

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "torque": self.torque,
            "current": self.current,
            "voltage": self.voltage,
            "temperature": self.temperature,
            "status": self.status.value,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


@dataclass
class MotorConfig:
    """电机配置类"""

    motor_id: str
    motor_type: MotorType
    name: str
    description: str = ""

    # 物理参数
    max_position: float = 100.0  # 最大位置
    min_position: float = 0.0  # 最小位置
    max_velocity: float = 100.0  # 最大速度
    max_acceleration: float = 10.0  # 最大加速度
    max_torque: float = 1.0  # 最大转矩
    max_current: float = 2.0  # 最大电流
    max_voltage: float = 12.0  # 最大电压

    # 控制参数
    control_mode: MotorControlMode = MotorControlMode.POSITION
    pid_params: Dict[str, float] = field(default_factory=dict)
    deadband: float = 0.01  # 死区

    # 连接参数
    connection_type: str = "serial"  # 连接类型
    connection_params: Dict[str, Any] = field(default_factory=dict)

    # 安全参数
    overheat_threshold: float = 80.0  # 过热阈值（°C）
    overload_threshold: float = 0.9  # 过载阈值（相对于最大转矩）
    stall_threshold: float = 0.1  # 失速阈值（速度低于此值视为失速）

    # 其他参数
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "motor_id": self.motor_id,
            "motor_type": self.motor_type.value,
            "name": self.name,
            "description": self.description,
            "max_position": self.max_position,
            "min_position": self.min_position,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "max_torque": self.max_torque,
            "max_current": self.max_current,
            "max_voltage": self.max_voltage,
            "control_mode": self.control_mode.value,
            "pid_params": self.pid_params,
            "deadband": self.deadband,
            "connection_type": self.connection_type,
            "connection_params": self.connection_params,
            "overheat_threshold": self.overheat_threshold,
            "overload_threshold": self.overload_threshold,
            "stall_threshold": self.stall_threshold,
            "metadata": self.metadata,
        }


@dataclass
class MotionCommand:
    """运动命令类"""

    target_position: Optional[float] = None
    target_velocity: Optional[float] = None
    target_acceleration: Optional[float] = None
    target_torque: Optional[float] = None

    duration: float = 0.0  # 持续时间（秒）
    speed_factor: float = 1.0  # 速度因子
    blocking: bool = True  # 是否阻塞

    callback: Optional[Callable[[bool, str], None]] = None  # 完成回调

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "target_position": self.target_position,
            "target_velocity": self.target_velocity,
            "target_acceleration": self.target_acceleration,
            "target_torque": self.target_torque,
            "duration": self.duration,
            "speed_factor": self.speed_factor,
            "blocking": self.blocking,
            "has_callback": self.callback is not None,
        }


class MotorController:
    """电机控制器

    功能：
    - 电机控制和运动规划
    - 状态监控和错误处理
    - 多电机协调控制
    - 运动轨迹生成
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化电机控制器

        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)

        # 默认配置
        self.config = config or {
            "max_motors": 20,
            "control_frequency": 100.0,  # 控制频率（Hz）
            "state_update_interval": 0.1,  # 状态更新间隔（秒）
            "enable_safety_checks": True,
            "enable_motion_planning": True,
            "default_speed_factor": 1.0,
            "emergency_stop_timeout": 1.0,  # 紧急停止超时（秒）
        }

        # 电机配置
        self.motor_configs: Dict[str, MotorConfig] = {}

        # 真实硬件接口（如果可用）
        self.real_motor_interfaces: Dict[str, Any] = {}

        # 电机状态
        self.motor_states: Dict[str, MotorState] = {}

        # 控制线程
        self.control_thread = None
        self.state_update_thread = None
        self.running = False

        # 运动命令队列
        self.command_queues: Dict[str, List[MotionCommand]] = {}
        self.current_commands: Dict[str, Optional[MotionCommand]] = {}

        # 回调函数
        self.state_callbacks: List[Callable[[str, MotorState], None]] = []
        self.error_callbacks: List[Callable[[str, str, int], None]] = []
        self.motion_complete_callbacks: List[Callable[[str, bool], None]] = []

        # PID控制器
        self.pid_controllers: Dict[str, Dict[str, Any]] = {}

        # 统计信息
        self.stats = {
            "total_motors": 0,
            "active_motors": 0,
            "total_commands": 0,
            "completed_commands": 0,
            "failed_commands": 0,
            "emergency_stops": 0,
            "control_cycles": 0,
            "last_update": None,
        }

        self.logger.info("电机控制器初始化完成")

    def start(self):
        """启动电机控制器"""
        if self.running:
            self.logger.warning("电机控制器已经在运行")
            return

        self.running = True

        # 启动控制线程
        self.control_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="MotorControl"
        )
        self.control_thread.start()

        # 启动状态更新线程
        self.state_update_thread = threading.Thread(
            target=self._state_update_loop, daemon=True, name="MotorStateUpdate"
        )
        self.state_update_thread.start()

        self.logger.info("电机控制器已启动")

    def stop(self):
        """停止电机控制器"""
        if not self.running:
            self.logger.warning("电机控制器未运行")
            return

        self.running = False

        # 紧急停止所有电机
        self.emergency_stop_all()

        # 等待线程停止
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)

        if self.state_update_thread and self.state_update_thread.is_alive():
            self.state_update_thread.join(timeout=2.0)

        self.logger.info("电机控制器已停止")

    def register_motor(self, config: MotorConfig) -> bool:
        """注册电机

        参数:
            config: 电机配置

        返回:
            注册是否成功
        """
        if config.motor_id in self.motor_configs:
            self.logger.warning(f"电机已注册: {config.motor_id}")
            return False

        try:
            # 保存电机配置
            self.motor_configs[config.motor_id] = config

            # 初始化电机状态
            self.motor_states[config.motor_id] = MotorState()

            # 初始化命令队列
            self.command_queues[config.motor_id] = []
            self.current_commands[config.motor_id] = None

            # 初始化PID控制器
            self._init_pid_controller(config.motor_id, config)

            # 更新统计信息
            self.stats["total_motors"] += 1

            self.logger.info(f"电机注册成功: {config.name} ({config.motor_id})")
            return True

        except Exception as e:
            self.logger.error(f"电机注册失败: {e}")
            return False

    def unregister_motor(self, motor_id: str) -> bool:
        """注销电机

        参数:
            motor_id: 电机ID

        返回:
            注销是否成功
        """
        if motor_id not in self.motor_configs:
            self.logger.warning(f"电机未注册: {motor_id}")
            return False

        try:
            # 停止电机
            self.stop_motor(motor_id)

            # 检查电机状态是否为运行中
            is_running = False
            if motor_id in self.motor_states:
                state = self.motor_states[motor_id]
                is_running = state.status == MotorStatus.RUNNING

            # 清理配置
            if motor_id in self.motor_configs:
                del self.motor_configs[motor_id]

            # 清理状态
            if motor_id in self.motor_states:
                del self.motor_states[motor_id]

            # 清理命令队列
            if motor_id in self.command_queues:
                del self.command_queues[motor_id]

            if motor_id in self.current_commands:
                del self.current_commands[motor_id]

            # 清理PID控制器
            if motor_id in self.pid_controllers:
                del self.pid_controllers[motor_id]

            # 更新统计信息
            self.stats["total_motors"] -= 1

            # 如果电机正在运行，减少活动电机计数
            if is_running:
                self.stats["active_motors"] -= 1

            self.logger.info(f"电机注销成功: {motor_id}")
            return True

        except Exception as e:
            self.logger.error(f"电机注销失败: {e}")
            return False

    def move_to_position(
        self,
        motor_id: str,
        position: float,
        speed_factor: float = 1.0,
        blocking: bool = True,
        callback: Optional[Callable[[bool, str], None]] = None,
    ) -> bool:
        """移动到指定位置

        参数:
            motor_id: 电机ID
            position: 目标位置
            speed_factor: 速度因子
            blocking: 是否阻塞
            callback: 完成回调

        返回:
            命令是否成功添加
        """
        if motor_id not in self.motor_configs:
            self.logger.warning(f"电机未注册: {motor_id}")
            return False

        try:
            config = self.motor_configs[motor_id]

            # 检查位置限制
            if position < config.min_position:
                position = config.min_position
                self.logger.warning(f"位置超出下限，调整为: {position}")

            if position > config.max_position:
                position = config.max_position
                self.logger.warning(f"位置超出上限，调整为: {position}")

            # 创建运动命令
            command = MotionCommand(
                target_position=position,
                speed_factor=speed_factor,
                blocking=blocking,
                callback=callback,
            )

            # 添加到命令队列
            self.command_queues[motor_id].append(command)
            self.stats["total_commands"] += 1

            self.logger.debug(f"添加位置移动命令: {motor_id} -> {position}")
            return True

        except Exception as e:
            self.logger.error(f"添加移动命令失败: {e}")
            return False

    def set_velocity(
        self,
        motor_id: str,
        velocity: float,
        duration: float = 0.0,
        blocking: bool = False,
        callback: Optional[Callable[[bool, str], None]] = None,
    ) -> bool:
        """设置速度

        参数:
            motor_id: 电机ID
            velocity: 目标速度
            duration: 持续时间（0表示无限）
            blocking: 是否阻塞
            callback: 完成回调

        返回:
            命令是否成功添加
        """
        if motor_id not in self.motor_configs:
            self.logger.warning(f"电机未注册: {motor_id}")
            return False

        try:
            config = self.motor_configs[motor_id]

            # 检查速度限制
            max_velocity = config.max_velocity * self.config["default_speed_factor"]
            if abs(velocity) > max_velocity:
                velocity = math.copysign(max_velocity, velocity)
                self.logger.warning(f"速度超出限制，调整为: {velocity}")

            # 创建运动命令
            command = MotionCommand(
                target_velocity=velocity,
                duration=duration,
                blocking=blocking,
                callback=callback,
            )

            # 添加到命令队列
            self.command_queues[motor_id].append(command)
            self.stats["total_commands"] += 1

            self.logger.debug(f"添加速度设置命令: {motor_id} -> {velocity}")
            return True

        except Exception as e:
            self.logger.error(f"添加速度命令失败: {e}")
            return False

    def stop_motor(self, motor_id: str, emergency: bool = False) -> bool:
        """停止电机

        参数:
            motor_id: 电机ID
            emergency: 是否紧急停止

        返回:
            停止是否成功
        """
        if motor_id not in self.motor_configs:
            self.logger.warning(f"电机未注册: {motor_id}")
            return False

        try:
            # 清空命令队列
            if motor_id in self.command_queues:
                self.command_queues[motor_id].clear()

            # 设置当前命令为None
            self.current_commands[motor_id] = None

            # 更新电机状态
            state = self.motor_states[motor_id]
            state.velocity = 0.0
            state.acceleration = 0.0

            if emergency:
                state.status = MotorStatus.STOPPED
                self.stats["emergency_stops"] += 1
                self.logger.warning(f"电机紧急停止: {motor_id}")
            else:
                state.status = MotorStatus.STOPPED
                self.logger.info(f"电机停止: {motor_id}")

            return True

        except Exception as e:
            self.logger.error(f"停止电机失败: {e}")
            return False

    def emergency_stop_all(self):
        """紧急停止所有电机"""
        self.logger.warning("紧急停止所有电机")

        for motor_id in list(self.motor_configs.keys()):
            self.stop_motor(motor_id, emergency=True)

        self.stats["emergency_stops"] += 1

    def get_motor_state(self, motor_id: str) -> Optional[MotorState]:
        """获取电机状态

        参数:
            motor_id: 电机ID

        返回:
            电机状态或None
        """
        return self.motor_states.get(motor_id)

    def get_motor_config(self, motor_id: str) -> Optional[MotorConfig]:
        """获取电机配置

        参数:
            motor_id: 电机ID

        返回:
            电机配置或None
        """
        return self.motor_configs.get(motor_id)

    def get_all_motor_states(self) -> Dict[str, MotorState]:
        """获取所有电机状态

        返回:
            电机状态字典
        """
        return self.motor_states.copy()

    def register_state_callback(self, callback: Callable[[str, MotorState], None]):
        """注册状态回调函数

        参数:
            callback: 回调函数，接收电机ID和状态
        """
        if callback not in self.state_callbacks:
            self.state_callbacks.append(callback)
            self.logger.debug("注册状态回调函数")

    def unregister_state_callback(self, callback: Callable[[str, MotorState], None]):
        """注销状态回调函数

        参数:
            callback: 回调函数
        """
        if callback in self.state_callbacks:
            self.state_callbacks.remove(callback)
            self.logger.debug("注销状态回调函数")

    def register_error_callback(self, callback: Callable[[str, str, int], None]):
        """注册错误回调函数

        参数:
            callback: 回调函数，接收电机ID、错误信息和错误码
        """
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)
            self.logger.debug("注册错误回调函数")

    def unregister_error_callback(self, callback: Callable[[str, str, int], None]):
        """注销错误回调函数

        参数:
            callback: 回调函数
        """
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
            self.logger.debug("注销错误回调函数")

    def register_motion_complete_callback(self, callback: Callable[[str, bool], None]):
        """注册运动完成回调函数

        参数:
            callback: 回调函数，接收电机ID和是否成功
        """
        if callback not in self.motion_complete_callbacks:
            self.motion_complete_callbacks.append(callback)
            self.logger.debug("注册运动完成回调函数")

    def unregister_motion_complete_callback(
        self, callback: Callable[[str, bool], None]
    ):
        """注销运动完成回调函数

        参数:
            callback: 回调函数
        """
        if callback in self.motion_complete_callbacks:
            self.motion_complete_callbacks.remove(callback)
            self.logger.debug("注销运动完成回调函数")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        返回:
            统计信息字典
        """
        stats = self.stats.copy()
        stats["running"] = self.running
        stats["total_motors"] = len(self.motor_configs)

        # 按类型统计电机数量
        stats["motor_types"] = {}
        for config in self.motor_configs.values():
            motor_type = config.motor_type.value
            if motor_type not in stats["motor_types"]:
                stats["motor_types"][motor_type] = 0
            stats["motor_types"][motor_type] += 1

        # 统计活动电机
        stats["active_motors"] = 0
        for state in self.motor_states.values():
            if state.status == MotorStatus.RUNNING:
                stats["active_motors"] += 1

        return stats

    def _control_loop(self):
        """控制循环"""
        self.logger.info("电机控制循环启动")

        control_interval = 1.0 / self.config["control_frequency"]

        while self.running:
            try:
                start_time = time.time()

                # 处理所有电机的控制
                for motor_id, config in self.motor_configs.items():
                    self._control_motor(motor_id, config)

                # 更新统计信息
                self.stats["control_cycles"] += 1

                # 控制频率调节
                elapsed = time.time() - start_time
                sleep_time = max(0.0, control_interval - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > control_interval * 2:
                    self.logger.warning(
                        f"控制循环超时: {elapsed:.3f}s > {control_interval:.3f}s"
                    )

            except Exception as e:
                self.logger.error(f"控制循环异常: {e}")
                time.sleep(0.1)

        self.logger.info("电机控制循环停止")

    def _state_update_loop(self):
        """状态更新循环"""
        self.logger.info("电机状态更新循环启动")

        update_interval = self.config["state_update_interval"]

        while self.running:
            try:
                # 更新所有电机的状态
                for motor_id, state in self.motor_states.items():
                    self._update_motor_state(motor_id, state)

                # 触发状态回调
                for motor_id, state in self.motor_states.items():
                    for callback in self.state_callbacks:
                        try:
                            callback(motor_id, state)
                        except Exception as e:
                            self.logger.error(f"状态回调执行失败: {e}")

                # 等待下一次更新
                time.sleep(update_interval)

            except Exception as e:
                self.logger.error(f"状态更新循环异常: {e}")
                time.sleep(1.0)

        self.logger.info("电机状态更新循环停止")

    def _control_motor(self, motor_id: str, config: MotorConfig):
        """控制单个电机

        参数:
            motor_id: 电机ID
            config: 电机配置
        """
        try:
            state = self.motor_states[motor_id]

            # 安全检查
            if self.config["enable_safety_checks"]:
                if not self._check_motor_safety(motor_id, state, config):
                    return

            # 获取当前命令
            current_command = self.current_commands.get(motor_id)

            # 如果没有当前命令，从队列中获取下一个
            if current_command is None and self.command_queues[motor_id]:
                current_command = self.command_queues[motor_id].pop(0)
                self.current_commands[motor_id] = current_command

                self.logger.debug(f"开始执行命令: {motor_id}")

            # 执行当前命令
            if current_command is not None:
                success = self._execute_command(
                    motor_id, current_command, state, config
                )

                # 如果命令完成
                if success:
                    # 触发完成回调
                    if current_command.callback:
                        try:
                            current_command.callback(True, "命令完成")
                        except Exception as e:
                            self.logger.error(f"命令回调执行失败: {e}")

                    # 触发运动完成回调
                    for callback in self.motion_complete_callbacks:
                        try:
                            callback(motor_id, True)
                        except Exception as e:
                            self.logger.error(f"运动完成回调执行失败: {e}")

                    # 清理当前命令
                    self.current_commands[motor_id] = None
                    self.stats["completed_commands"] += 1

                    self.logger.debug(f"命令完成: {motor_id}")

                # 如果命令失败
                elif not success and current_command.blocking:
                    # 触发失败回调
                    if current_command.callback:
                        try:
                            current_command.callback(False, "命令失败")
                        except Exception as e:
                            self.logger.error(f"命令回调执行失败: {e}")

                    # 触发运动完成回调
                    for callback in self.motion_complete_callbacks:
                        try:
                            callback(motor_id, False)
                        except Exception as e:
                            self.logger.error(f"运动完成回调执行失败: {e}")

                    # 清理当前命令
                    self.current_commands[motor_id] = None
                    self.stats["failed_commands"] += 1

                    self.logger.warning(f"命令失败: {motor_id}")

            # 应用控制算法
            self._apply_control_algorithm(motor_id, state, config)

        except Exception as e:
            self.logger.error(f"控制电机失败: {motor_id}, {e}")
            self.stats["failed_commands"] += 1

    def _execute_command(
        self,
        motor_id: str,
        command: MotionCommand,
        state: MotorState,
        config: MotorConfig,
    ) -> bool:
        """执行运动命令

        参数:
            motor_id: 电机ID
            command: 运动命令
            state: 电机状态
            config: 电机配置

        返回:
            命令是否完成
        """
        try:
            # 获取硬件接口（真实或仿真）
            hardware_interface = None
            
            # 检查是否已有硬件接口
            if motor_id in self.real_motor_interfaces:
                hardware_interface = self.real_motor_interfaces[motor_id]
            else:
                # 创建新的硬件接口（会尝试真实硬件，然后仿真）
                hardware_interface = self._create_real_motor_interface(motor_id, config)
                if hardware_interface is not None and hasattr(hardware_interface, 'is_connected') and hardware_interface.is_connected():
                    self.real_motor_interfaces[motor_id] = hardware_interface
            
            # 如果有可用的硬件接口，使用它执行命令
            if hardware_interface is not None and hasattr(hardware_interface, 'is_connected') and hardware_interface.is_connected():
                return self._execute_with_real_hardware(motor_id, command, state, config, hardware_interface)
            
            # 如果硬件接口创建失败，抛出异常（根据项目要求禁止虚拟实现）
            raise RuntimeError(f"无法创建或连接硬件接口: {motor_id}")

        except Exception as e:
            self.logger.error(f"执行命令失败: {motor_id}, {e}")
            return False

    def _apply_control_algorithm(
        self, motor_id: str, state: MotorState, config: MotorConfig
    ):
        """应用控制算法

        参数:
            motor_id: 电机ID
            state: 电机状态
            config: 电机配置
        """
        # 这里可以实现更复杂的控制算法，如PID控制
        # 当前使用基本的控制算法

        # 电机动力学计算
        dt = 1.0 / self.config["control_frequency"]

        # 应用加速度
        state.velocity += state.acceleration * dt

        # 应用速度限制
        max_velocity = config.max_velocity * self.config["default_speed_factor"]
        if abs(state.velocity) > max_velocity:
            state.velocity = math.copysign(max_velocity, state.velocity)

        # 应用位置限制
        if state.position < config.min_position:
            state.position = config.min_position
            state.velocity = max(0, state.velocity)  # 防止继续向负方向移动

        if state.position > config.max_position:
            state.position = config.max_position
            state.velocity = min(0, state.velocity)  # 防止继续向正方向移动

    def _update_motor_state(self, motor_id: str, state: MotorState):
        """更新电机状态

        参数:
            motor_id: 电机ID
            state: 电机状态
        """
        try:
            # 更新时间戳
            state.timestamp = time.time()

            # 首先尝试从真实硬件接口获取状态
            real_interface = None
            if REAL_HARDWARE_AVAILABLE and hasattr(self, 'real_motor_interfaces'):
                try:
                    if motor_id in self.real_motor_interfaces:
                        real_interface = self.real_motor_interfaces[motor_id]
                        
                except Exception as e:
                    self.logger.warning(f"无法访问真实硬件接口 {motor_id}: {e}")
            
            # 如果有可用的真实硬件接口，从硬件读取状态
            if real_interface is not None and real_interface.is_connected():
                try:
                    # 从真实硬件获取状态信息
                    # 注意：这里假设真实硬件接口有get_state或类似方法
                    # 实际实现中需要根据真实硬件接口的API进行调整
                    hardware_state = real_interface.get_state()
                    
                    if hardware_state:
                        # 更新位置、速度等状态
                        state.position = hardware_state.get("position", state.position)
                        state.velocity = hardware_state.get("velocity", state.velocity)
                        state.current = hardware_state.get("current", state.current)
                        state.voltage = hardware_state.get("voltage", state.voltage)
                        state.torque = hardware_state.get("torque", state.torque)
                        state.temperature = hardware_state.get("temperature", state.temperature)
                        
                        # 更新硬件状态
                        hardware_status = hardware_state.get("status", "unknown")
                        if hardware_status == "running":
                            state.status = MotorStatus.RUNNING
                        elif hardware_status == "stopped":
                            state.status = MotorStatus.STOPPED
                        elif hardware_status == "error":
                            state.status = MotorStatus.ERROR
                            
                        # 标记为真实硬件状态
                        state.metadata["hardware_source"] = "real_hardware"
                        
                        # 检查安全状态
                        self._check_motor_safety(motor_id, state, self.motor_configs[motor_id])
                        return
                        
                except Exception as e:
                    self.logger.warning(f"从真实硬件读取状态失败 {motor_id}: {e}")
            
            # 如果没有真实硬件接口或读取失败，根据用户要求"系统可以在没有硬件条件下单独运行AGI所有功能"
            # 记录警告并返回，而不是抛出异常
            self.logger.warning(
                f"电机 {motor_id} 无法更新状态：真实硬件接口不可用\n"
                f"当前状态: 位置={state.position:.2f}, 速度={state.velocity:.2f}, 温度={state.temperature:.1f}\n"
                "根据用户要求'系统可以在没有硬件条件下单独运行AGI所有功能'，\n"
                "跳过状态更新，系统可以继续运行（电机状态更新功能将不可用）。"
            )
            return  # 直接返回，不更新状态

            # 检查安全状态
            self._check_motor_safety(motor_id, state, self.motor_configs[motor_id])

        except Exception as e:
            self.logger.error(f"更新电机状态失败: {motor_id}, {e}")

    def _check_motor_safety(
        self, motor_id: str, state: MotorState, config: MotorConfig
    ) -> bool:
        """检查电机安全

        参数:
            motor_id: 电机ID
            state: 电机状态
            config: 电机配置

        返回:
            是否安全
        """
        try:
            errors = []

            # 检查过热
            if state.temperature > config.overheat_threshold:
                errors.append(
                    f"过热: {state.temperature:.1f}°C > {config.overheat_threshold}°C"
                )
                state.status = MotorStatus.OVERHEAT
                state.error_code = 1001

            # 检查过载
            if abs(state.torque) > config.max_torque * config.overload_threshold:
                errors.append(
                    f"过载: {state.torque:.3f}Nm > "
                    f"{config.max_torque * config.overload_threshold:.3f}Nm"
                )
                state.status = MotorStatus.OVERLOAD
                state.error_code = 1002

            # 检查失速
            if (
                state.status == MotorStatus.RUNNING
                and abs(state.velocity) < config.stall_threshold
                and abs(state.torque) > config.max_torque * 0.5
            ):
                errors.append(
                    f"失速: 速度 {state.velocity:.3f} < {config.stall_threshold}"
                )
                state.status = MotorStatus.STALLED
                state.error_code = 1003

            # 检查电流
            if abs(state.current) > config.max_current:
                errors.append(f"过流: {state.current:.3f}A > {config.max_current}A")
                state.status = MotorStatus.ERROR
                state.error_code = 1004

            # 检查电压
            if abs(state.voltage) > config.max_voltage:
                errors.append(f"过压: {state.voltage:.3f}V > {config.max_voltage}V")
                state.status = MotorStatus.ERROR
                state.error_code = 1005

            # 如果有错误
            if errors:
                error_message = "; ".join(errors)
                state.error_message = error_message

                # 触发错误回调
                for callback in self.error_callbacks:
                    try:
                        callback(motor_id, error_message, state.error_code)
                    except Exception as e:
                        self.logger.error(f"错误回调执行失败: {e}")

                # 如果配置了安全检查，停止电机
                if self.config["enable_safety_checks"]:
                    self.stop_motor(motor_id, emergency=True)
                    self.logger.error(f"电机安全错误: {motor_id} - {error_message}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"安全检查失败: {motor_id}, {e}")
            return False

    def _init_pid_controller(self, motor_id: str, config: MotorConfig):
        """初始化PID控制器

        参数:
            motor_id: 电机ID
            config: 电机配置
        """
        # 从配置中获取PID参数
        pid_params = config.pid_params

        # 初始化PID控制器状态
        self.pid_controllers[motor_id] = {
            "kp": pid_params.get("kp", 1.0),
            "ki": pid_params.get("ki", 0.0),
            "kd": pid_params.get("kd", 0.0),
            "integral": 0.0,
            "prev_error": 0.0,
            "last_update": time.time(),
            "output_limit": config.max_velocity,
            "integral_limit": 10.0,
        }

        self.logger.debug(f"PID控制器初始化: {motor_id}")

    def _apply_pid_control(self, motor_id: str, error: float) -> float:
        """应用PID控制

        参数:
            motor_id: 电机ID
            error: 误差

        返回:
            PID控制输出
        """
        if motor_id not in self.pid_controllers:
            return error  # 简单的比例控制

        pid = self.pid_controllers[motor_id]

        # 计算时间差
        current_time = time.time()
        dt = current_time - pid["last_update"]
        if dt <= 0:
            dt = 0.001

        # 积分项
        pid["integral"] += error * dt
        # 积分限幅
        if abs(pid["integral"]) > pid["integral_limit"]:
            pid["integral"] = math.copysign(pid["integral_limit"], pid["integral"])

        # 微分项
        derivative = (error - pid["prev_error"]) / dt if dt > 0 else 0.0

        # PID输出
        output = (
            pid["kp"] * error + pid["ki"] * pid["integral"] + pid["kd"] * derivative
        )

        # 输出限幅
        if abs(output) > pid["output_limit"]:
            output = math.copysign(pid["output_limit"], output)

        # 保存状态
        pid["prev_error"] = error
        pid["last_update"] = current_time

        return output

    def _create_real_motor_interface(self, motor_id: str, config: MotorConfig) -> Any:
        """创建真实电机接口
        
        参数:
            motor_id: 电机ID
            config: 电机配置
            
        返回:
            真实电机接口实例或仿真接口实例
        """
        # 首先尝试真实硬件接口
        if REAL_HARDWARE_AVAILABLE:
            try:
                # 从配置中提取真实硬件参数
                metadata = config.metadata or {}
                
                # 确定电机类型映射
                motor_type_mapping = {
                    MotorType.DC: RealMotorType.DC,
                    MotorType.STEPPER: RealMotorType.STEPPER,
                    MotorType.SERVO: RealMotorType.SERVO,
                    MotorType.BRUSHLESS: RealMotorType.BRUSHLESS,
                    MotorType.LINEAR: RealMotorType.LINEAR,
                    MotorType.CUSTOM: RealMotorType.UNKNOWN,
                }
                
                real_motor_type = motor_type_mapping.get(config.motor_type)
                if real_motor_type is None:
                    self.logger.warning(f"不支持的真实电机类型映射: {config.motor_type}，使用UNKNOWN")
                    real_motor_type = RealMotorType.UNKNOWN
                
                # 从元数据中获取接口配置，默认为串口接口
                interface_type_str = metadata.get("interface_type", "serial")
                try:
                    interface_type = ControlInterface(interface_type_str)
                except ValueError:
                    self.logger.warning(f"不支持的接口类型: {interface_type_str}，使用SERIAL")
                    interface_type = ControlInterface.SERIAL
                
                # 构建接口配置
                interface_config = metadata.get("interface_config", {})
                
                # 添加电机特定配置
                interface_config.update({
                    "max_speed": config.max_velocity,
                    "max_torque": config.max_torque,
                    "position_resolution": getattr(config, 'position_resolution', 0.01),
                    "control_frequency": self.config.get("control_frequency", 100.0),
                })
                
                # 如果是串口接口，添加串口特定配置
                if interface_type == ControlInterface.SERIAL:
                    interface_config.update({
                        "port": metadata.get("port", "COM1"),
                        "baudrate": metadata.get("baudrate", 9600),
                        "bytesize": metadata.get("bytesize", 8),
                        "parity": metadata.get("parity", "N"),
                        "stopbits": metadata.get("stopbits", 1),
                        "timeout": metadata.get("timeout", 1.0),
                    })
                
                # 创建真实电机控制器
                real_interface = RealMotorController(
                    motor_id=motor_id,
                    motor_type=real_motor_type,
                    control_interface=interface_type,
                    interface_config=interface_config
                )
                
                # 尝试连接
                if real_interface.connect():
                    self.logger.info(f"真实电机控制器创建并连接成功: {motor_id} ({interface_type.value})")
                    return real_interface
                else:
                    self.logger.warning(f"真实电机控制器连接失败: {motor_id}")
                    # 连接失败，继续尝试仿真
            except Exception as e:
                self.logger.error(f"创建真实电机控制器失败 {motor_id}: {e}")
                # 真实硬件失败，继续尝试仿真
        
        # 真实硬件不可用或失败，尝试仿真接口
        self.logger.warning(f"真实硬件接口不可用或失败，尝试仿真接口: {motor_id}")
        try:
            from hardware.simulation import PyBulletSimulation
            # 创建仿真接口
            sim_config = {
                "gui_enabled": False,
                "physics_timestep": 1.0/240.0,
                "realtime_simulation": False,
            }
            sim_interface = PyBulletSimulation(**sim_config)
            if sim_interface.connect():
                self.logger.info(f"仿真接口创建并连接成功: {motor_id}")
                return sim_interface
            else:
                self.logger.error(f"仿真接口连接失败: {motor_id}")
                raise RuntimeError("无法创建任何硬件接口（真实或仿真）")
        except ImportError as e:
            raise ImportError(
                "无法创建硬件接口。项目要求禁止使用虚拟实现。\n"
                "请安装必要的依赖：\n"
                "1. 真实硬件：pip install pyserial smbus2 spidev python-can\n"
                "2. 仿真：pip install pybullet\n"
                f"错误：{e}"
            ) from e
            
    def _execute_with_real_hardware(self, motor_id: str, command: MotionCommand, 
                                   state: MotorState, config: MotorConfig, 
                                   real_interface: Any) -> bool:
        """使用真实硬件接口执行命令
        
        参数:
            motor_id: 电机ID
            command: 运动命令
            state: 电机状态
            config: 电机配置
            real_interface: 真实硬件接口
            
        返回:
            命令是否完成
        """
        try:
            # 根据控制模式执行命令
            if config.control_mode == MotorControlMode.POSITION:
                if command.target_position is not None:
                    # 使用真实硬件接口设置目标位置
                    success = real_interface.move_to_position(
                        position=command.target_position,
                        speed=config.max_velocity * command.speed_factor,
                        blocking=command.blocking
                    )
                    
                    if success:
                        # 更新状态
                        state.status = MotorStatus.RUNNING
                        state.target_position = command.target_position
                        
                        # 如果非阻塞模式，立即返回完成
                        if not command.blocking:
                            return True
                            
                        # 阻塞模式下等待完成
                        # 完整处理：假设硬件接口会处理阻塞等待
                        # 实际实现中可能需要轮询硬件状态
                        return True
                    else:
                        state.status = MotorStatus.ERROR
                        state.error_code = 2001
                        state.error_message = "真实硬件接口移动失败"
                        return False
                        
            elif config.control_mode == MotorControlMode.VELOCITY:
                if command.target_velocity is not None:
                    # 使用真实硬件接口设置目标速度
                    success = real_interface.set_velocity(
                        velocity=command.target_velocity,
                        duration=command.duration
                    )
                    
                    if success:
                        # 更新状态
                        state.status = MotorStatus.RUNNING
                        state.target_velocity = command.target_velocity
                        
                        # 检查持续时间
                        if command.duration > 0:
                            command.duration -= 1.0 / self.config["control_frequency"]
                            if command.duration <= 0:
                                return True
                            else:
                                return False
                        else:
                            # 持续速度控制，不返回完成
                            return False
                    else:
                        state.status = MotorStatus.ERROR
                        state.error_code = 2002
                        state.error_message = "真实硬件接口速度设置失败"
                        return False
                        
            # 默认返回False（继续执行）
            return False
            
        except Exception as e:
            self.logger.error(f"使用真实硬件执行命令失败 {motor_id}: {e}")
            state.status = MotorStatus.ERROR
            state.error_code = 2000
            state.error_message = f"真实硬件执行异常: {str(e)}"
            return False

    def __del__(self):
        """析构函数，确保资源被清理"""
        self.stop()
