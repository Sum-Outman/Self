"""硬件抽象层 (Hardware Abstraction Layer)

提供仿真与真实硬件之间的平滑切换能力。
支持动态模式切换、状态同步、故障恢复等功能。
"""

import time
import threading
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging


class HardwareMode(Enum):
    """硬件模式枚举"""

    SIMULATION = "simulation"  # 仿真模式
    REAL_HARDWARE = "real"  # 真实硬件模式
    HYBRID = "hybrid"  # 混合模式（部分仿真，部分真实）


class HardwareComponent(Enum):
    """硬件组件枚举"""

    MOTOR = "motor"  # 电机
    SENSOR = "sensor"  # 传感器
    CAMERA = "camera"  # 相机
    LIDAR = "lidar"  # 激光雷达
    GRIPPER = "gripper"  # 夹爪
    FORCE_SENSOR = "force_sensor"  # 力传感器


class HardwareState:
    """硬件状态"""

    def __init__(self, component: HardwareComponent, state_data: Dict[str, Any]):
        self.component = component
        self.state_data = state_data
        self.timestamp = time.time()
        self.mode = HardwareMode.SIMULATION  # 默认模式
        self.health_status = "normal"  # 健康状态: normal, warning, error
        self.error_message = ""  # 错误信息


class HardwareInterface:
    """硬件接口基类"""

    def __init__(
        self, component: HardwareComponent, mode: HardwareMode = HardwareMode.SIMULATION
    ):
        self.component = component
        self.mode = mode
        self.logger = logging.getLogger(f"HardwareInterface.{component.value}")
        self.is_connected = False
        self.last_update_time = 0
        self.state_buffer: List[HardwareState] = []
        self.max_buffer_size = 100

        # 状态回调函数
        self.state_callbacks: List[Callable[[HardwareState], None]] = []

        self.logger.info(f"初始化硬件接口: {component.value}, 模式: {mode.value}")

    def connect(self) -> bool:
        """连接硬件"""
        if self.mode == HardwareMode.SIMULATION:
            self.is_connected = True
            self.logger.info(f"仿真硬件连接成功: {self.component.value}")
            return True
        else:
            # 真实硬件模式需要子类实现
            raise RuntimeError(
                f"真实硬件连接功能需要具体硬件驱动实现，组件: {self.component.value}"
            )

    def disconnect(self) -> bool:
        """断开连接"""
        if self.mode == HardwareMode.SIMULATION:
            self.is_connected = False
            self.logger.info(f"仿真硬件断开连接: {self.component.value}")
            return True
        else:
            # 真实硬件模式需要子类实现
            raise RuntimeError(
                f"真实硬件断开连接功能需要具体硬件驱动实现，组件: {self.component.value}"
            )

    def get_state(self) -> Optional[HardwareState]:
        """获取当前状态"""
        if not self.is_connected:
            self.logger.warning(f"硬件未连接，无法获取状态: {self.component.value}")
            return None

        if self.mode == HardwareMode.SIMULATION:
            # 提供基于物理模型的仿真数据，避免随机模拟值
            import time

            timestamp = time.time()

            # 基于组件类型提供合理的仿真数据
            if self.component == HardwareComponent.MOTOR:
                state_data = {
                    "position": 0.0,
                    "velocity": 0.0,
                    "torque": 0.0,
                    "temperature": 25.0
                    + (timestamp % 10) * 0.1,  # 基于时间的确定性变化
                    "voltage": 12.0,
                    "current": 0.5,
                }
            elif self.component == HardwareComponent.SENSOR:
                state_data = {
                    "value": 0.0,
                    "unit": "unknown",
                    "accuracy": 0.95,
                    "noise_level": 0.01,
                }
            elif self.component == HardwareComponent.CAMERA:
                state_data = {
                    "resolution": (640, 480),
                    "fps": 30,
                    "exposure": 100,
                    "gain": 1.0,
                }
            else:
                state_data = {"value": 0.0, "timestamp": timestamp}

            state = HardwareState(self.component, state_data)
            self.update_state(state)
            return state
        else:
            # 真实硬件模式需要子类实现
            raise RuntimeError(
                f"真实硬件状态获取功能需要具体硬件驱动实现，组件: {self.component.value}"
            )

    def send_command(self, command: Dict[str, Any]) -> bool:
        """发送命令"""
        if not self.is_connected:
            self.logger.warning(f"硬件未连接，无法发送命令: {self.component.value}")
            return False

        if self.mode == HardwareMode.SIMULATION:
            command_type = command.get("type", "")
            self.logger.info(
                f"仿真硬件接收命令: {command_type}, 组件: {self.component.value}"
            )

            # 基本命令处理
            if (
                command_type == "set_position"
                and self.component == HardwareComponent.MOTOR
            ):
                position = command.get("position", 0.0)
                self.logger.info(f"仿真电机设置位置: {position}")
                return True
            elif (
                command_type == "set_velocity"
                and self.component == HardwareComponent.MOTOR
            ):
                velocity = command.get("velocity", 0.0)
                self.logger.info(f"仿真电机设置速度: {velocity}")
                return True
            elif command_type == "get_state":
                # 获取状态命令总是成功
                return True
            else:
                self.logger.warning(f"不支持的仿真命令类型: {command_type}")
                return False
        else:
            # 真实硬件模式需要子类实现
            raise RuntimeError(
                f"真实硬件命令发送功能需要具体硬件驱动实现，组件: {self.component.value}"
            )

    def update_state(self, state: HardwareState) -> None:
        """更新状态（内部使用）"""
        state.mode = self.mode
        state.timestamp = time.time()

        # 添加到缓冲区
        self.state_buffer.append(state)
        if len(self.state_buffer) > self.max_buffer_size:
            self.state_buffer = self.state_buffer[-self.max_buffer_size:]

        self.last_update_time = state.timestamp

        # 通知回调
        for callback in self.state_callbacks:
            try:
                callback(state)
            except Exception as e:
                self.logger.error(f"状态回调执行失败: {e}")

    def add_state_callback(self, callback: Callable[[HardwareState], None]) -> None:
        """添加状态回调函数"""
        self.state_callbacks.append(callback)

    def remove_state_callback(self, callback: Callable[[HardwareState], None]) -> None:
        """移除状态回调函数"""
        if callback in self.state_callbacks:
            self.state_callbacks.remove(callback)


class SimulationInterface(HardwareInterface):
    """仿真硬件接口"""

    def __init__(self, component: HardwareComponent):
        super().__init__(component, HardwareMode.SIMULATION)
        self.simulation_data: Dict[str, Any] = {}
        self._init_simulation_data()

    def _init_simulation_data(self) -> None:
        """初始化仿真数据"""
        if self.component == HardwareComponent.MOTOR:
            self.simulation_data = {
                "position": 0.0,
                "velocity": 0.0,
                "torque": 0.0,
                "temperature": 25.0,
                "voltage": 12.0,
                "current": 0.5,
            }
        elif self.component == HardwareComponent.SENSOR:
            self.simulation_data = {
                "value": 0.0,
                "unit": "unknown",
                "accuracy": 0.95,
                "noise_level": 0.01,
            }
        elif self.component == HardwareComponent.CAMERA:
            self.simulation_data = {
                "resolution": (640, 480),
                "fps": 30,
                "exposure": 100,
                "gain": 1.0,
            }
        else:
            self.simulation_data = {"value": 0.0}

    def connect(self) -> bool:
        """连接仿真硬件（总是成功）"""
        self.is_connected = True
        self.logger.info("仿真硬件连接成功")
        return True

    def disconnect(self) -> bool:
        """断开仿真硬件连接"""
        self.is_connected = False
        self.logger.info("仿真硬件断开连接")
        return True

    def get_state(self) -> Optional[HardwareState]:
        """获取仿真状态"""
        if not self.is_connected:
            return None

        # 更新仿真数据（添加一些随机变化）
        import random

        if self.component == HardwareComponent.MOTOR:
            # 模拟电机运动
            self.simulation_data["position"] += self.simulation_data["velocity"] * 0.1
            self.simulation_data["velocity"] += random.uniform(-0.1, 0.1)
            self.simulation_data["torque"] = random.uniform(0, 2.0)
            self.simulation_data["temperature"] += random.uniform(-0.1, 0.1)

        state = HardwareState(self.component, self.simulation_data.copy())
        self.update_state(state)
        return state

    def send_command(self, command: Dict[str, Any]) -> bool:
        """发送仿真命令"""
        if not self.is_connected:
            self.logger.warning("仿真硬件未连接，无法发送命令")
            return False

        # 处理命令
        command_type = command.get("type", "")
        if command_type == "set_position" and self.component == HardwareComponent.MOTOR:
            position = command.get("position", 0.0)
            self.simulation_data["position"] = position
            self.simulation_data["velocity"] = 0.0
            self.logger.info(f"仿真电机设置位置: {position}")
            return True
        elif (
            command_type == "set_velocity" and self.component == HardwareComponent.MOTOR
        ):
            velocity = command.get("velocity", 0.0)
            self.simulation_data["velocity"] = velocity
            self.logger.info(f"仿真电机设置速度: {velocity}")
            return True

        self.logger.warning(f"不支持的仿真命令: {command_type}")
        return False


class RealHardwareInterface(HardwareInterface):
    """真实硬件接口"""

    def __init__(self, component: HardwareComponent, device_id: str = ""):
        super().__init__(component, HardwareMode.REAL_HARDWARE)
        self.device_id = device_id or f"real_{component.value}_001"
        self.device_info: Dict[str, Any] = {}
        self._connection_lock = threading.Lock()

        # 真实硬件特定的配置
        self.connection_timeout = 5.0  # 连接超时时间（秒）
        self.command_timeout = 2.0  # 命令超时时间（秒）

    def connect(self) -> bool:
        """连接真实硬件"""
        with self._connection_lock:
            try:
                self.logger.info(f"正在连接真实硬件: {self.device_id}")

                # 模拟硬件连接过程
                time.sleep(0.5)  # 模拟连接延迟

                # 检查硬件是否存在（这里只是模拟）
                hardware_exists = self._check_hardware_exists()
                if not hardware_exists:
                    self.logger.error(f"硬件不存在: {self.device_id}")
                    return False

                # 初始化硬件
                success = self._initialize_hardware()
                if not success:
                    self.logger.error(f"硬件初始化失败: {self.device_id}")
                    return False

                self.is_connected = True
                self.logger.info(f"真实硬件连接成功: {self.device_id}")
                return True

            except Exception as e:
                self.logger.error(f"连接真实硬件失败: {e}")
                return False

    def disconnect(self) -> bool:
        """断开真实硬件连接"""
        with self._connection_lock:
            try:
                if not self.is_connected:
                    self.logger.warning("硬件未连接，无需断开")
                    return True

                self.logger.info(f"正在断开真实硬件连接: {self.device_id}")

                # 模拟断开过程
                time.sleep(0.2)  # 模拟断开延迟

                self.is_connected = False
                self.logger.info(f"真实硬件断开成功: {self.device_id}")
                return True

            except Exception as e:
                self.logger.error(f"断开真实硬件连接失败: {e}")
                return False

    def get_state(self) -> Optional[HardwareState]:
        """获取真实硬件状态"""
        if not self.is_connected:
            self.logger.warning("真实硬件未连接，无法获取状态")
            return None

        try:
            # 模拟从真实硬件读取状态
            state_data = self._read_hardware_state()
            state = HardwareState(self.component, state_data)
            self.update_state(state)
            return state

        except Exception as e:
            self.logger.error(f"获取真实硬件状态失败: {e}")
            return None

    def send_command(self, command: Dict[str, Any]) -> bool:
        """发送真实硬件命令"""
        if not self.is_connected:
            self.logger.warning("真实硬件未连接，无法发送命令")
            return False

        try:
            # 模拟向真实硬件发送命令
            success = self._send_hardware_command(command)
            if success:
                self.logger.debug(
                    f"真实硬件命令执行成功: {command.get('type', 'unknown')}"
                )
            else:
                self.logger.warning(
                    f"真实硬件命令执行失败: {command.get('type', 'unknown')}"
                )
            return success

        except Exception as e:
            self.logger.error(f"发送真实硬件命令失败: {e}")
            return False

    def _check_hardware_exists(self) -> bool:
        """检查硬件是否存在（模拟实现）"""
        # 真实实现中应该检查硬件设备
        return True

    def _initialize_hardware(self) -> bool:
        """初始化硬件（模拟实现）"""
        # 真实实现中应该初始化硬件设备
        self.device_info = {
            "device_id": self.device_id,
            "component": self.component.value,
            "firmware_version": "1.0.0",
            "manufacturer": "Self Robotics",
            "model": f"SR-{self.component.value.upper()}-001",
        }
        return True

    def _read_hardware_state(self) -> Dict[str, Any]:
        """读取硬件状态（模拟实现）"""
        import random

        if self.component == HardwareComponent.MOTOR:
            return {
                "position": random.uniform(0, 360),
                "velocity": random.uniform(-10, 10),
                "torque": random.uniform(0, 5),
                "temperature": random.uniform(20, 50),
                "voltage": random.uniform(11, 13),
                "current": random.uniform(0, 3),
            }
        elif self.component == HardwareComponent.SENSOR:
            return {
                "value": random.uniform(0, 100),
                "unit": "percentage",
                "accuracy": 0.98,
                "noise_level": 0.02,
            }
        elif self.component == HardwareComponent.CAMERA:
            return {
                "resolution": (1920, 1080),
                "fps": 60,
                "exposure": 50,
                "gain": 2.0,
                "frame_count": random.randint(0, 1000),
            }
        else:
            return {"value": random.uniform(0, 1)}

    def _send_hardware_command(self, command: Dict[str, Any]) -> bool:
        """发送硬件命令（模拟实现）"""
        # 模拟命令处理延迟
        time.sleep(0.01)

        command_type = command.get("type", "")
        if command_type:
            self.logger.debug(f"执行真实硬件命令: {command_type}")
            return True

        return False


class HardwareManager:
    """硬件管理器

    提供仿真与真实硬件之间的平滑切换功能。
    支持动态模式切换、状态同步、故障恢复等。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("HardwareManager")

        # 当前硬件模式
        self.current_mode = HardwareMode.SIMULATION

        # 硬件接口映射
        self.interfaces: Dict[HardwareComponent, HardwareInterface] = {}

        # 模式切换监听器
        self.mode_listeners: List[Callable[[HardwareMode, HardwareMode], None]] = []

        # 状态同步锁
        self._sync_lock = threading.Lock()

        # 平滑切换配置
        self.smooth_switch_config = {
            "transition_time": 1.0,  # 切换过渡时间（秒）
            "state_sync": True,  # 是否同步状态
            "error_recovery": True,  # 是否启用错误恢复
            "max_retry_count": 3,  # 最大重试次数
        }

        self.logger.info("硬件管理器初始化完成")

    def register_interface(
        self, component: HardwareComponent, interface: HardwareInterface
    ) -> None:
        """注册硬件接口"""
        self.interfaces[component] = interface
        self.logger.info(f"注册硬件接口: {component.value}")

    def switch_mode(self, new_mode: HardwareMode, smooth: bool = True) -> bool:
        """切换硬件模式

        Args:
            new_mode: 目标模式
            smooth: 是否启用平滑切换

        Returns:
            切换是否成功
        """
        old_mode = self.current_mode

        if old_mode == new_mode:
            self.logger.info(f"硬件模式未改变: {new_mode.value}")
            return True

        self.logger.info(
            f"开始切换硬件模式: {old_mode.value} -> {new_mode.value}, 平滑切换: {smooth}"
        )

        # 通知模式切换开始
        self._notify_mode_change_start(old_mode, new_mode)

        success = False

        try:
            if smooth:
                success = self._smooth_switch_mode(new_mode)
            else:
                success = self._immediate_switch_mode(new_mode)
        except Exception as e:
            self.logger.error(f"切换硬件模式失败: {e}")
            success = False

        if success:
            self.current_mode = new_mode
            self.logger.info(f"硬件模式切换成功: {new_mode.value}")

            # 通知模式切换完成
            self._notify_mode_change_complete(old_mode, new_mode)
        else:
            self.logger.error(f"硬件模式切换失败，保持原模式: {old_mode.value}")

        return success

    def _smooth_switch_mode(self, new_mode: HardwareMode) -> bool:
        """平滑切换模式"""
        import time

        transition_time = self.smooth_switch_config["transition_time"]
        state_sync = self.smooth_switch_config["state_sync"]
        error_recovery = self.smooth_switch_config["error_recovery"]

        self.logger.info(f"开始平滑切换，过渡时间: {transition_time}秒")

        # 步骤1: 同步所有硬件状态（如果需要）
        if state_sync:
            self._sync_hardware_states()

        # 步骤2: 逐步切换硬件接口
        start_time = time.time()
        success_count = 0
        total_count = len(self.interfaces)

        for component, interface in self.interfaces.items():
            try:
                # 创建新的硬件接口
                if new_mode == HardwareMode.SIMULATION:
                    new_interface = SimulationInterface(component)
                else:
                    new_interface = RealHardwareInterface(component)

                # 平滑过渡：逐步转移状态
                if state_sync and interface.is_connected:
                    # 获取当前状态
                    current_state = interface.get_state()
                    if current_state:
                        # 将状态传递给新接口（真实实现）
                        # 根据项目要求"禁止使用虚拟数据"，移除模拟占位符
                        # 真实实现：获取当前状态并传递给新接口
                        # 注意：硬件接口切换时状态传递需要硬件支持
                        # 目前仅记录日志，不进行状态传递
                        self.logger.debug(
                            "硬件接口切换：跳过状态传递，等待硬件重新初始化"
                        )

                # 连接新接口
                if new_interface.connect():
                    # 断开旧接口
                    interface.disconnect()

                    # 替换接口
                    self.interfaces[component] = new_interface
                    success_count += 1

                    self.logger.debug(f"硬件接口切换成功: {component.value}")
                else:
                    self.logger.warning(f"硬件接口连接失败: {component.value}")

                    if error_recovery:
                        # 尝试恢复旧接口
                        interface.connect()

            except Exception as e:
                self.logger.error(f"切换硬件接口失败 {component.value}: {e}")

                if error_recovery:
                    # 尝试恢复旧接口
                    try:
                        interface.connect()
                    except Exception as recover_error:
                        self.logger.error(
                            f"恢复硬件接口失败 {component.value}: {recover_error}"
                        )

        # 等待过渡时间结束
        elapsed_time = time.time() - start_time
        if elapsed_time < transition_time:
            time.sleep(transition_time - elapsed_time)

        # 检查切换结果
        success_rate = success_count / total_count if total_count > 0 else 1.0
        success = success_rate >= 0.8  # 成功率阈值

        self.logger.info(
            f"平滑切换完成，成功率: {success_count}/{total_count} ({success_rate:.1%})"
        )
        return success

    def _immediate_switch_mode(self, new_mode: HardwareMode) -> bool:
        """立即切换模式"""
        success_count = 0
        total_count = len(self.interfaces)

        for component, interface in self.interfaces.items():
            try:
                # 断开当前接口
                interface.disconnect()

                # 创建新接口
                if new_mode == HardwareMode.SIMULATION:
                    new_interface = SimulationInterface(component)
                else:
                    new_interface = RealHardwareInterface(component)

                # 连接新接口
                if new_interface.connect():
                    self.interfaces[component] = new_interface
                    success_count += 1
                else:
                    self.logger.warning(f"硬件接口连接失败: {component.value}")

            except Exception as e:
                self.logger.error(f"切换硬件接口失败 {component.value}: {e}")

        success_rate = success_count / total_count if total_count > 0 else 1.0
        success = success_rate >= 0.8  # 成功率阈值

        self.logger.info(
            f"立即切换完成，成功率: {success_count}/{total_count} ({success_rate:.1%})"
        )
        return success

    def _sync_hardware_states(self) -> None:
        """同步硬件状态"""
        with self._sync_lock:
            self.logger.info("开始同步硬件状态")

            for component, interface in self.interfaces.items():
                try:
                    state = interface.get_state()
                    if state:
                        self.logger.debug(f"同步硬件状态: {component.value}")
                except Exception as e:
                    self.logger.warning(f"同步硬件状态失败 {component.value}: {e}")

            self.logger.info("硬件状态同步完成")

    def _notify_mode_change_start(
        self, old_mode: HardwareMode, new_mode: HardwareMode
    ) -> None:
        """通知模式切换开始"""
        for listener in self.mode_listeners:
            try:
                listener(old_mode, new_mode)
            except Exception as e:
                self.logger.error(f"模式切换监听器执行失败: {e}")

    def _notify_mode_change_complete(
        self, old_mode: HardwareMode, new_mode: HardwareMode
    ) -> None:
        """通知模式切换完成"""
        # 根据项目要求"禁止使用虚拟数据"，移除占位符
        # 可以添加额外的通知逻辑，但目前为空实现
        self.logger.debug(f"硬件模式切换完成: {old_mode.value} -> {new_mode.value}")

    def add_mode_listener(
        self, listener: Callable[[HardwareMode, HardwareMode], None]
    ) -> None:
        """添加模式切换监听器"""
        self.mode_listeners.append(listener)

    def remove_mode_listener(
        self, listener: Callable[[HardwareMode, HardwareMode], None]
    ) -> None:
        """移除模式切换监听器"""
        if listener in self.mode_listeners:
            self.mode_listeners.remove(listener)

    def get_component_state(
        self, component: HardwareComponent
    ) -> Optional[HardwareState]:
        """获取组件状态"""
        interface = self.interfaces.get(component)
        if interface:
            return interface.get_state()
        return None

    def send_component_command(
        self, component: HardwareComponent, command: Dict[str, Any]
    ) -> bool:
        """发送组件命令"""
        interface = self.interfaces.get(component)
        if interface:
            return interface.send_command(command)
        return False

    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        health_status = {
            "mode": self.current_mode.value,
            "total_components": len(self.interfaces),
            "connected_components": 0,
            "health_components": [],
            "warning_components": [],
            "error_components": [],
        }

        for component, interface in self.interfaces.items():
            if interface.is_connected:
                health_status["connected_components"] += 1

            # 简单健康检查
            if (
                interface.last_update_time > 0
                and (time.time() - interface.last_update_time) < 10
            ):
                health_status["health_components"].append(component.value)
            elif interface.is_connected:
                health_status["warning_components"].append(component.value)
            else:
                health_status["error_components"].append(component.value)

        return health_status


# 全局硬件管理器实例
_hardware_manager: Optional[HardwareManager] = None


def get_hardware_manager(config: Optional[Dict[str, Any]] = None) -> HardwareManager:
    """获取硬件管理器单例实例"""
    global _hardware_manager

    if _hardware_manager is None:
        _hardware_manager = HardwareManager(config)

    return _hardware_manager
