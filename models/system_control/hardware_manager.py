"""
硬件管理器

功能：
- 硬件设备管理和注册
- 设备发现和识别
- 设备状态监控
- 硬件资源分配
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass, field


class DeviceType(Enum):
    """设备类型枚举"""

    SENSOR = "sensor"  # 传感器
    ACTUATOR = "actuator"  # 执行器
    MOTOR = "motor"  # 电机
    CAMERA = "camera"  # 摄像头
    MICROPHONE = "microphone"  # 麦克风
    SPEAKER = "speaker"  # 扬声器
    DISPLAY = "display"  # 显示器
    INPUT_DEVICE = "input_device"  # 输入设备
    NETWORK = "network"  # 网络设备
    STORAGE = "storage"  # 存储设备
    UNKNOWN = "unknown"  # 未知设备


class DeviceStatus(Enum):
    """设备状态枚举"""

    ONLINE = "online"  # 在线
    OFFLINE = "offline"  # 离线
    ERROR = "error"  # 错误
    BUSY = "busy"  # 繁忙
    IDLE = "idle"  # 空闲
    MAINTENANCE = "maintenance"  # 维护中


@dataclass
class HardwareDevice:
    """硬件设备数据类"""

    device_id: str
    device_type: DeviceType
    name: str
    description: str = ""
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    version: str = ""

    # 连接信息
    connection_type: str = ""  # serial, usb, network, bluetooth, etc.
    connection_params: Dict[str, Any] = field(default_factory=dict)

    # 状态信息
    status: DeviceStatus = DeviceStatus.OFFLINE
    last_seen: float = 0.0
    error_count: int = 0
    error_message: str = ""

    # 能力信息
    capabilities: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    # 资源占用
    resource_usage: Dict[str, float] = field(default_factory=dict)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "name": self.name,
            "description": self.description,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "version": self.version,
            "connection_type": self.connection_type,
            "connection_params": self.connection_params,
            "status": self.status.value,
            "last_seen": self.last_seen,
            "error_count": self.error_count,
            "error_message": self.error_message,
            "capabilities": self.capabilities,
            "properties": self.properties,
            "resource_usage": self.resource_usage,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardwareDevice":
        """从字典创建"""
        device = cls(
            device_id=data["device_id"],
            device_type=DeviceType(data["device_type"]),
            name=data["name"],
            description=data.get("description", ""),
            manufacturer=data.get("manufacturer", ""),
            model=data.get("model", ""),
            serial_number=data.get("serial_number", ""),
            version=data.get("version", ""),
        )

        device.connection_type = data.get("connection_type", "")
        device.connection_params = data.get("connection_params", {})
        device.status = DeviceStatus(data.get("status", "offline"))
        device.last_seen = data.get("last_seen", 0.0)
        device.error_count = data.get("error_count", 0)
        device.error_message = data.get("error_message", "")
        device.capabilities = data.get("capabilities", [])
        device.properties = data.get("properties", {})
        device.resource_usage = data.get("resource_usage", {})
        device.metadata = data.get("metadata", {})

        return device


class HardwareManager:
    """硬件管理器

    功能：
    - 设备注册和管理
    - 设备状态监控
    - 设备发现和识别
    - 硬件资源管理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化硬件管理器

        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)

        # 默认配置
        self.config = config or {
            "auto_discovery": True,
            "discovery_interval": 60.0,  # 自动发现间隔（秒）- 优化为60秒
            "health_check_interval": 30.0,  # 健康检查间隔（秒）- 优化为30秒
            "device_timeout": 120.0,  # 设备超时时间（秒）- 适应新的发现间隔
            "max_error_count": 5,  # 最大错误次数
            "enable_logging": True,
            "log_level": "INFO",
        }

        # 设备注册表
        self.devices: Dict[str, HardwareDevice] = {}

        # 设备类型索引
        self.device_type_index: Dict[DeviceType, Set[str]] = {}

        # 回调函数
        self.device_callbacks = {
            "device_added": [],
            "device_removed": [],
            "device_status_changed": [],
            "device_error": [],
        }

        # 线程控制
        self.discovery_thread = None
        self.health_check_thread = None
        self.running = False

        # 统计信息
        self.stats = {
            "total_devices": 0,
            "online_devices": 0,
            "offline_devices": 0,
            "error_devices": 0,
            "discovery_count": 0,
            "last_discovery": None,
        }

        # 初始化设备类型索引
        for device_type in DeviceType:
            self.device_type_index[device_type] = set()

        self.logger.info("硬件管理器初始化完成")

    def start(self):
        """启动硬件管理器"""
        if self.running:
            self.logger.warning("硬件管理器已经在运行")
            return

        self.running = True

        # 启动设备发现线程
        if self.config["auto_discovery"]:
            self.discovery_thread = threading.Thread(
                target=self._discovery_loop, daemon=True, name="HardwareDiscovery"
            )
            self.discovery_thread.start()
            self.logger.info("设备发现线程启动")

        # 启动健康检查线程
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True, name="HardwareHealthCheck"
        )
        self.health_check_thread.start()
        self.logger.info("健康检查线程启动")

        self.logger.info("硬件管理器已启动")

    def stop(self):
        """停止硬件管理器"""
        if not self.running:
            self.logger.warning("硬件管理器未运行")
            return

        self.running = False

        # 等待线程停止
        if self.discovery_thread and self.discovery_thread.is_alive():
            self.discovery_thread.join(timeout=2.0)

        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=2.0)

        self.logger.info("硬件管理器已停止")

    def register_device(self, device: HardwareDevice) -> bool:
        """注册设备

        参数:
            device: 硬件设备

        返回:
            注册是否成功
        """
        if device.device_id in self.devices:
            self.logger.warning(f"设备已注册: {device.device_id}")
            return False

        try:
            # 添加到设备注册表
            self.devices[device.device_id] = device

            # 更新设备类型索引
            self.device_type_index[device.device_type].add(device.device_id)

            # 更新统计信息
            self.stats["total_devices"] += 1
            if device.status == DeviceStatus.ONLINE:
                self.stats["online_devices"] += 1
            elif device.status == DeviceStatus.OFFLINE:
                self.stats["offline_devices"] += 1
            elif device.status == DeviceStatus.ERROR:
                self.stats["error_devices"] += 1

            # 触发设备添加回调
            self._trigger_callbacks("device_added", device)

            self.logger.info(f"设备注册成功: {device.name} ({device.device_id})")
            return True

        except Exception as e:
            self.logger.error(f"设备注册失败: {e}")
            return False

    def unregister_device(self, device_id: str) -> bool:
        """注销设备

        参数:
            device_id: 设备ID

        返回:
            注销是否成功
        """
        if device_id not in self.devices:
            self.logger.warning(f"设备未注册: {device_id}")
            return False

        try:
            device = self.devices[device_id]

            # 从设备类型索引中移除
            self.device_type_index[device.device_type].discard(device_id)

            # 从设备注册表中移除
            del self.devices[device_id]

            # 更新统计信息
            self.stats["total_devices"] -= 1
            if device.status == DeviceStatus.ONLINE:
                self.stats["online_devices"] -= 1
            elif device.status == DeviceStatus.OFFLINE:
                self.stats["offline_devices"] -= 1
            elif device.status == DeviceStatus.ERROR:
                self.stats["error_devices"] -= 1

            # 触发设备移除回调
            self._trigger_callbacks("device_removed", device)

            self.logger.info(f"设备注销成功: {device_id}")
            return True

        except Exception as e:
            self.logger.error(f"设备注销失败: {e}")
            return False

    def update_device_status(
        self, device_id: str, status: DeviceStatus, error_message: str = ""
    ) -> bool:
        """更新设备状态

        参数:
            device_id: 设备ID
            status: 新状态
            error_message: 错误信息

        返回:
            更新是否成功
        """
        if device_id not in self.devices:
            self.logger.warning(f"设备未注册: {device_id}")
            return False

        try:
            device = self.devices[device_id]
            old_status = device.status

            # 更新设备状态
            device.status = status
            device.last_seen = time.time()

            if status == DeviceStatus.ERROR:
                device.error_count += 1
                device.error_message = error_message
            elif status == DeviceStatus.ONLINE:
                device.error_count = 0
                device.error_message = ""

            # 更新统计信息
            if old_status != status:
                # 减少旧状态计数
                if old_status == DeviceStatus.ONLINE:
                    self.stats["online_devices"] -= 1
                elif old_status == DeviceStatus.OFFLINE:
                    self.stats["offline_devices"] -= 1
                elif old_status == DeviceStatus.ERROR:
                    self.stats["error_devices"] -= 1

                # 增加新状态计数
                if status == DeviceStatus.ONLINE:
                    self.stats["online_devices"] += 1
                elif status == DeviceStatus.OFFLINE:
                    self.stats["offline_devices"] += 1
                elif status == DeviceStatus.ERROR:
                    self.stats["error_devices"] += 1

                # 触发状态变化回调
                self._trigger_callbacks(
                    "device_status_changed", device, old_status, status
                )

            # 触发错误回调
            if status == DeviceStatus.ERROR:
                self._trigger_callbacks("device_error", device, error_message)

            self.logger.debug(
                f"设备状态更新: {device_id} {old_status.value} -> {status.value}"
            )
            return True

        except Exception as e:
            self.logger.error(f"设备状态更新失败: {e}")
            return False

    def get_device(self, device_id: str) -> Optional[HardwareDevice]:
        """获取设备

        参数:
            device_id: 设备ID

        返回:
            设备对象或None
        """
        return self.devices.get(device_id)

    def get_devices_by_type(self, device_type: DeviceType) -> List[HardwareDevice]:
        """根据类型获取设备

        参数:
            device_type: 设备类型

        返回:
            设备列表
        """
        device_ids = self.device_type_index.get(device_type, set())
        return [
            self.devices[device_id]
            for device_id in device_ids
            if device_id in self.devices
        ]

    def get_all_devices(self) -> List[HardwareDevice]:
        """获取所有设备

        返回:
            设备列表
        """
        return list(self.devices.values())

    def discover_devices(self) -> List[HardwareDevice]:
        """发现新设备

        返回:
            发现的设备列表
        """
        discovered_devices = []

        try:
            # 发现串口设备
            serial_devices = self._discover_serial_devices()
            discovered_devices.extend(serial_devices)

            # 发现USB设备（需要平台特定实现）
            usb_devices = self._discover_usb_devices()
            discovered_devices.extend(usb_devices)

            # 发现网络设备
            network_devices = self._discover_network_devices()
            discovered_devices.extend(network_devices)

            # 更新统计信息
            self.stats["discovery_count"] += 1
            self.stats["last_discovery"] = time.time()

            self.logger.debug(f"发现 {len(discovered_devices)} 个潜在设备")

        except Exception as e:
            self.logger.error(f"设备发现失败: {e}")

        return discovered_devices

    def register_callback(self, event_type: str, callback: Callable):
        """注册回调函数

        参数:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.device_callbacks:
            if callback not in self.device_callbacks[event_type]:
                self.device_callbacks[event_type].append(callback)
                self.logger.debug(f"注册回调: {event_type}")
        else:
            self.logger.warning(f"未知的事件类型: {event_type}")

    def unregister_callback(self, event_type: str, callback: Callable):
        """注销回调函数

        参数:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.device_callbacks:
            if callback in self.device_callbacks[event_type]:
                self.device_callbacks[event_type].remove(callback)
                self.logger.debug(f"注销回调: {event_type}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        返回:
            统计信息字典
        """
        stats = self.stats.copy()
        stats["running"] = self.running
        stats["device_count_by_type"] = {}

        for device_type, device_ids in self.device_type_index.items():
            stats["device_count_by_type"][device_type.value] = len(device_ids)

        return stats

    def _discovery_loop(self):
        """设备发现循环"""
        self.logger.info("设备发现循环启动")

        while self.running:
            try:
                # 执行设备发现
                discovered_devices = self.discover_devices()

                # 统计新设备数量
                new_device_count = 0

                # 处理发现的设备
                for device in discovered_devices:
                    if device.device_id not in self.devices:
                        # 注册新设备
                        if self.register_device(device):
                            new_device_count += 1
                    else:
                        # 更新现有设备
                        existing_device = self.devices[device.device_id]
                        existing_device.last_seen = time.time()

                # 只有发现新设备时才记录
                if new_device_count > 0:
                    self.logger.info(f"发现 {new_device_count} 个新设备，已注册")
                elif discovered_devices:
                    # 有设备但都是已注册的，调试级别记录
                    self.logger.debug(
                        f"发现 {len(discovered_devices)} 个设备，均为已注册设备"
                    )
                else:
                    # 没有发现任何设备，调试级别记录
                    self.logger.debug("未发现任何设备")

                # 等待下一次发现
                time.sleep(self.config["discovery_interval"])

            except Exception as e:
                self.logger.error(f"设备发现循环异常: {e}")
                time.sleep(1.0)

        self.logger.info("设备发现循环停止")

    def _health_check_loop(self):
        """健康检查循环"""
        self.logger.info("健康检查循环启动")

        while self.running:
            try:
                current_time = time.time()
                timeout_threshold = self.config["device_timeout"]

                # 检查所有设备
                for device_id, device in self.devices.items():
                    # 检查设备超时
                    if (
                        device.status == DeviceStatus.ONLINE
                        and current_time - device.last_seen > timeout_threshold
                    ):
                        self.logger.warning(f"设备超时: {device_id}")
                        self.update_device_status(
                            device_id, DeviceStatus.OFFLINE, "设备响应超时"
                        )

                    # 检查错误计数
                    if device.error_count >= self.config["max_error_count"]:
                        self.logger.warning(
                            f"设备错误计数过高: {device_id} ({device.error_count})"
                        )
                        self.update_device_status(
                            device_id, DeviceStatus.ERROR, "错误计数过高"
                        )

                # 等待下一次健康检查
                time.sleep(self.config["health_check_interval"])

            except Exception as e:
                self.logger.error(f"健康检查循环异常: {e}")
                time.sleep(1.0)

        self.logger.info("健康检查循环停止")

    def _discover_serial_devices(self) -> List[HardwareDevice]:
        """发现串口设备"""
        discovered_devices = []

        try:
            import serial.tools.list_ports

            ports = serial.tools.list_ports.comports()
            for port in ports:
                # 创建设备对象
                device = HardwareDevice(
                    device_id=f"serial_{port.device}",
                    device_type=DeviceType.UNKNOWN,
                    name=port.name or port.device,
                    description=port.description or f"串口设备: {port.device}",
                    manufacturer=port.manufacturer or "未知",
                    serial_number=port.serial_number or "",
                    connection_type="serial",
                    connection_params={
                        "port": port.device,
                        "vid": port.vid,
                        "pid": port.pid,
                        "hwid": port.hwid,
                    },
                    status=DeviceStatus.ONLINE,
                    last_seen=time.time(),
                )

                # 根据描述猜测设备类型
                description_lower = device.description.lower()
                if any(word in description_lower for word in ["camera", "摄像头"]):
                    device.device_type = DeviceType.CAMERA
                elif any(word in description_lower for word in ["sensor", "传感器"]):
                    device.device_type = DeviceType.SENSOR
                elif any(word in description_lower for word in ["motor", "电机"]):
                    device.device_type = DeviceType.MOTOR

                discovered_devices.append(device)

        except ImportError:
            self.logger.warning("pyserial未安装，无法发现串口设备")
        except Exception as e:
            self.logger.error(f"串口设备发现失败: {e}")

        return discovered_devices

    def _discover_usb_devices(self) -> List[HardwareDevice]:
        """发现USB设备"""
        discovered_devices = []

        try:
            # 这里需要平台特定的USB发现实现
            # 对于Windows，可以使用pywin32或wmi
            # 对于Linux，可以使用pyusb或直接读取/sys/bus/usb
            # 完整的示例

            # 尝试使用platform模块获取基本信息
            import platform

            if platform.system() == "Windows":
                # Windows平台
                try:
                    import winreg

                    # 读取USB设备注册表
                    key_path = r"SYSTEM\CurrentControlSet\Enum\USB"
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                        i = 0
                        while True:
                            try:
                                device_id = winreg.EnumKey(key, i)

                                # 创建设备对象
                                device = HardwareDevice(
                                    device_id=f"usb_{device_id}",
                                    device_type=DeviceType.UNKNOWN,
                                    name=f"USB设备 {device_id}",
                                    description="USB设备",
                                    connection_type="usb",
                                    connection_params={"device_id": device_id},
                                    status=DeviceStatus.ONLINE,
                                    last_seen=time.time(),
                                )

                                discovered_devices.append(device)
                                i += 1
                            except OSError:
                                break

                except ImportError:
                    self.logger.warning("winreg不可用，无法发现USB设备")
                except Exception as e:
                    self.logger.debug(f"USB设备发现失败: {e}")

            elif platform.system() == "Linux":
                # Linux平台
                import os

                usb_path = "/sys/bus/usb/devices"
                if os.path.exists(usb_path):
                    for device_dir in os.listdir(usb_path):
                        if device_dir.startswith("usb"):
                            device_path = os.path.join(usb_path, device_dir)

                            # 读取设备信息
                            try:
                                with open(
                                    os.path.join(device_path, "product"), "r"
                                ) as f:
                                    product = f.read().strip()
                            except Exception:
                                product = "未知USB设备"

                            try:
                                with open(
                                    os.path.join(device_path, "manufacturer"), "r"
                                ) as f:
                                    manufacturer = f.read().strip()
                            except Exception:
                                manufacturer = "未知"

                            # 创建设备对象
                            device = HardwareDevice(
                                device_id=f"usb_{device_dir}",
                                device_type=DeviceType.UNKNOWN,
                                name=product,
                                description=f"USB设备: {product}",
                                manufacturer=manufacturer,
                                connection_type="usb",
                                connection_params={"device_dir": device_dir},
                                status=DeviceStatus.ONLINE,
                                last_seen=time.time(),
                            )

                            discovered_devices.append(device)

        except Exception as e:
            self.logger.debug(f"USB设备发现失败: {e}")

        return discovered_devices

    def _discover_network_devices(self) -> List[HardwareDevice]:
        """发现网络设备"""
        discovered_devices = []

        try:
            import socket

            # 获取本地网络接口
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)

            # 创建本地网络接口设备
            device = HardwareDevice(
                device_id=f"network_{hostname}",
                device_type=DeviceType.NETWORK,
                name=hostname,
                description=f"本地网络接口: {local_ip}",
                connection_type="network",
                connection_params={"hostname": hostname, "ip_address": local_ip},
                status=DeviceStatus.ONLINE,
                last_seen=time.time(),
                capabilities=["data_transfer", "communication"],
            )

            discovered_devices.append(device)

        except Exception as e:
            self.logger.debug(f"网络设备发现失败: {e}")

        return discovered_devices

    def _trigger_callbacks(self, event_type: str, *args, **kwargs):
        """触发回调函数

        参数:
            event_type: 事件类型
            *args: 位置参数
            **kwargs: 关键字参数
        """
        if event_type in self.device_callbacks:
            for callback in self.device_callbacks[event_type]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f"回调函数执行失败: {e}")

    def __del__(self):
        """析构函数，确保资源被清理"""
        self.stop()
