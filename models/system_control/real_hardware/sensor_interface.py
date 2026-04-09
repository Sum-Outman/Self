#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实传感器接口

支持多种传感器类型和数据采集
核心原则：真实优先，模拟后备
"""

import logging
import time
import threading
from typing import Dict, Any, List
from enum import Enum

from .base_interface import (
    RealHardwareInterface,
    HardwareType,
    ConnectionStatus,
    HardwareError,
    ConnectionError,
    OperationError,
)

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """传感器类型枚举"""

    TEMPERATURE = "temperature"  # 温度传感器
    HUMIDITY = "humidity"  # 湿度传感器
    PRESSURE = "pressure"  # 压力传感器
    LIGHT = "light"  # 光强传感器
    DISTANCE = "distance"  # 距离传感器
    MOTION = "motion"  # 运动传感器
    ACCELEROMETER = "accelerometer"  # 加速度计
    GYROSCOPE = "gyroscope"  # 陀螺仪
    MAGNETOMETER = "magnetometer"  # 磁力计
    GPS = "gps"  # GPS传感器
    CAMERA = "camera"  # 摄像头
    MICROPHONE = "microphone"  # 麦克风
    PROXIMITY = "proximity"  # 接近传感器
    TOUCH = "touch"  # 触摸传感器
    FORCE = "force"  # 力传感器
    FLOW = "flow"  # 流量传感器
    GAS = "gas"  # 气体传感器
    PH = "ph"  # pH传感器
    CONDUCTIVITY = "conductivity"  # 电导率传感器
    UNKNOWN = "unknown"  # 未知传感器


class SensorInterface(Enum):
    """传感器接口枚举"""

    I2C = "i2c"  # I2C接口
    SPI = "spi"  # SPI接口
    SERIAL = "serial"  # 串口
    ANALOG = "analog"  # 模拟接口
    DIGITAL = "digital"  # 数字接口
    USB = "usb"  # USB接口
    NETWORK = "network"  # 网络接口
    BLUETOOTH = "bluetooth"  # 蓝牙
    GPIO = "gpio"  # GPIO
    UNKNOWN = "unknown"  # 未知接口


class SensorData:
    """传感器数据类"""

    def __init__(
        self,
        sensor_type: SensorType,
        value: Any,
        unit: str = "",
        timestamp: float = None,
        accuracy: float = 1.0,
        metadata: Dict[str, Any] = None,
    ):
        """
        初始化传感器数据

        参数:
            sensor_type: 传感器类型
            value: 传感器值
            unit: 单位
            timestamp: 时间戳（秒）
            accuracy: 数据准确度（0.0-1.0）
            metadata: 元数据
        """
        self.sensor_type = sensor_type
        self.value = value
        self.unit = unit
        self.timestamp = timestamp or time.time()
        self.accuracy = max(0.0, min(1.0, accuracy))
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sensor_type": self.sensor_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "accuracy": self.accuracy,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorData":
        """从字典创建"""
        return cls(
            sensor_type=SensorType(data["sensor_type"]),
            value=data["value"],
            unit=data.get("unit", ""),
            timestamp=data.get("timestamp", time.time()),
            accuracy=data.get("accuracy", 1.0),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        return f"{self.sensor_type.value}: {self.value} {self.unit}"


class RealSensorInterface(RealHardwareInterface):
    """真实传感器接口

    支持多种传感器类型和数据采集接口
    """

    def __init__(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        sensor_interface: SensorInterface,
        interface_config: Dict[str, Any],
    ):
        """
        初始化真实传感器接口

        参数:
            sensor_id: 传感器唯一标识符
            sensor_type: 传感器类型
            sensor_interface: 传感器接口类型
            interface_config: 接口配置字典
        """
        super().__init__(
            hardware_type=HardwareType.SENSOR, interface_name=f"sensor_{sensor_id}"
        )

        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.sensor_interface = sensor_interface
        self.interface_config = interface_config

        # 传感器参数
        self.sampling_rate = interface_config.get("sampling_rate", 1.0)  # 采样率（Hz）
        self.resolution = interface_config.get("resolution", 0.01)  # 分辨率
        self.range_min = interface_config.get("range_min", -float("inf"))  # 量程下限
        self.range_max = interface_config.get("range_max", float("inf"))  # 量程上限
        self.calibration_offset = interface_config.get(
            "calibration_offset", 0.0
        )  # 校准偏移
        self.calibration_scale = interface_config.get(
            "calibration_scale", 1.0
        )  # 校准比例

        # 数据缓冲区
        self.data_buffer: List[SensorData] = []
        self.buffer_max_size = interface_config.get("buffer_size", 1000)
        self.last_sample_time = 0.0

        # 硬件接口实例
        self._hardware_interface = None
        self._interface_lock = threading.RLock()

        # 数据采集线程
        self._sampling_thread = None
        self._sampling_running = False

        # 初始化硬件接口
        self._init_hardware_interface()

        logger.info(
            f"初始化真实传感器接口: {sensor_id} "
            f"({sensor_type.value}, {sensor_interface.value})"
        )

    def _init_hardware_interface(self):
        """初始化硬件接口"""
        with self._interface_lock:
            try:
                if self.sensor_interface == SensorInterface.I2C:
                    self._init_i2c_interface()
                elif self.sensor_interface == SensorInterface.SPI:
                    self._init_spi_interface()
                elif self.sensor_interface == SensorInterface.SERIAL:
                    self._init_serial_interface()
                elif self.sensor_interface == SensorInterface.ANALOG:
                    self._init_analog_interface()
                elif self.sensor_interface == SensorInterface.DIGITAL:
                    self._init_digital_interface()
                elif self.sensor_interface == SensorInterface.USB:
                    self._init_usb_interface()
                elif self.sensor_interface == SensorInterface.NETWORK:
                    self._init_network_interface()
                elif self.sensor_interface == SensorInterface.BLUETOOTH:
                    self._init_bluetooth_interface()
                elif self.sensor_interface == SensorInterface.GPIO:
                    self._init_gpio_interface()
                else:
                    raise HardwareError(
                        f"不支持的传感器接口类型: {self.sensor_interface}"
                    )

                logger.info(f"传感器硬件接口初始化完成: {self.sensor_interface.value}")

            except Exception as e:
                logger.error(f"传感器硬件接口初始化失败: {e}")
                raise ConnectionError(f"传感器硬件接口初始化失败: {e}")

    def _init_i2c_interface(self):
        """初始化I2C接口"""
        import smbus2  # type: ignore

        bus_num = self.interface_config.get("bus", 1)
        address = self.interface_config.get("address", 0x40)

        try:
            self._i2c_bus = smbus2.SMBus(bus_num)
            self._i2c_address = address
            logger.info(f"I2C传感器连接成功: bus={bus_num}, address=0x{address:02x}")

        except ImportError:
            logger.warning("smbus2未安装，I2C功能不可用")
            self._i2c_bus = None
        except Exception as e:
            logger.error(f"I2C传感器连接失败: {e}")
            raise ConnectionError(f"I2C传感器连接失败: {e}")

    def _init_spi_interface(self):
        """初始化SPI接口"""
        try:
            import spidev  # type: ignore

            bus = self.interface_config.get("bus", 0)
            device = self.interface_config.get("device", 0)
            max_speed = self.interface_config.get("max_speed", 1000000)

            self._spi = spidev.SpiDev()
            self._spi.open(bus, device)
            self._spi.max_speed_hz = max_speed
            logger.info(f"SPI传感器连接成功: bus={bus}, device={device}")

        except ImportError:
            logger.warning("spidev未安装，SPI功能不可用")
            self._spi = None
        except Exception as e:
            logger.error(f"SPI传感器连接失败: {e}")
            raise ConnectionError(f"SPI传感器连接失败: {e}")

    def _init_serial_interface(self):
        """初始化串口接口"""
        try:
            import serial

            port = self.interface_config.get("port", "/dev/ttyUSB0")
            baudrate = self.interface_config.get("baudrate", 9600)
            timeout = self.interface_config.get("timeout", 1.0)

            self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
            logger.info(f"串口传感器连接成功: {port} @ {baudrate} baud")

        except ImportError:
            logger.warning("pyserial未安装，串口功能不可用")
            self._serial = None
        except Exception as e:
            logger.error(f"串口传感器连接失败: {e}")
            raise ConnectionError(f"串口传感器连接失败: {e}")

    def _init_analog_interface(self):
        """初始化模拟接口"""
        try:
            # 尝试导入ADC库
            import board  # type: ignore
            import busio  # type: ignore
            import adafruit_ads1x15.ads1115 as ADS  # type: ignore
            from adafruit_ads1x15.analog_in import AnalogIn  # type: ignore

            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS.ADS1115(i2c)

            channel = self.interface_config.get("channel", 0)
            if channel == 0:
                self._analog_channel = AnalogIn(ads, ADS.P0)
            elif channel == 1:
                self._analog_channel = AnalogIn(ads, ADS.P1)
            elif channel == 2:
                self._analog_channel = AnalogIn(ads, ADS.P2)
            elif channel == 3:
                self._analog_channel = AnalogIn(ads, ADS.P3)
            else:
                raise HardwareError(f"无效的模拟通道: {channel}")

            logger.info(f"模拟传感器接口初始化成功: channel={channel}")

        except ImportError as e:
            logger.error(f"模拟传感器库未安装: {e}")
            raise ImportError(
                "adafruit_ads1x15未安装。项目要求禁止使用模拟模式，必须安装硬件库。"
            ) from e
        except Exception as e:
            logger.error(f"模拟传感器接口初始化失败: {e}")
            raise RuntimeError(
                f"模拟传感器接口初始化失败: {e}。项目要求禁止使用模拟模式，必须确保模拟传感器硬件可用。"
            ) from e

    def _init_digital_interface(self):
        """初始化数字接口"""
        try:
            import RPi.GPIO as GPIO  # type: ignore

            pin = self.interface_config.get("pin", 17)
            pull_up_down = self.interface_config.get("pull_up_down", GPIO.PUD_UP)

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.IN, pull_up_down=pull_up_down)

            self._digital_pin = pin
            self._gpio_library = "RPi.GPIO"
            logger.info(f"数字传感器接口初始化成功: pin={pin}")

        except ImportError:
            try:
                import gpiozero  # type: ignore

                pin = self.interface_config.get("pin", 17)
                pull_up = self.interface_config.get("pull_up", True)

                self._digital_device = gpiozero.Button(pin, pull_up=pull_up)
                self._gpio_library = "gpiozero"
                logger.info(f"数字传感器接口初始化成功 (gpiozero): pin={pin}")

            except ImportError:
                logger.error("GPIO库未安装，数字接口功能不可用")
                raise ImportError(
                    "GPIO库（RPi.GPIO或gpiozero）未安装。项目要求禁止使用模拟模式，必须安装硬件库。"
                )

    def _init_usb_interface(self):
        """初始化USB接口"""
        try:
            import usb.core  # type: ignore
            import usb.util  # type: ignore

            vendor_id = self.interface_config.get("vendor_id", 0x1234)
            product_id = self.interface_config.get("product_id", 0x5678)

            self._usb_vendor_id = vendor_id
            self._usb_product_id = product_id
            self._usb_device = None

            logger.info(
                f"USB传感器接口配置: VID=0x{vendor_id:04x}, PID=0x{product_id:04x}"
            )

        except ImportError:
            logger.warning("PyUSB未安装，USB功能不可用")
            self._usb_device = None

    def _init_network_interface(self):
        """初始化网络接口"""
        self._network_host = self.interface_config.get("host", "192.168.1.100")
        self._network_port = self.interface_config.get("port", 8080)
        self._network_protocol = self.interface_config.get("protocol", "TCP")

        # 延迟到连接时建立socket
        self._network_socket = None
        logger.info(f"网络传感器接口配置: {self._network_host}:{self._network_port}")

    def _init_bluetooth_interface(self):
        """初始化蓝牙接口"""
        try:
            import bluetooth  # type: ignore

            address = self.interface_config.get("address", "")
            port = self.interface_config.get("port", 1)

            self._bluetooth_address = address
            self._bluetooth_port = port
            self._bluetooth_socket = None

            logger.info(f"蓝牙传感器接口配置: {address}:{port}")

        except ImportError:
            logger.warning("pybluez未安装，蓝牙功能不可用")
            self._bluetooth_socket = None

    def _init_gpio_interface(self):
        """初始化GPIO接口（用于简单传感器）"""
        self._init_digital_interface()  # 重用数字接口实现

    def connect(self) -> bool:
        """连接到传感器硬件"""
        with self._connection_lock:
            try:
                if self.connection_status == ConnectionStatus.CONNECTED:
                    logger.info("传感器已连接")
                    return True

                self.connection_status = ConnectionStatus.CONNECTING
                logger.info(f"正在连接传感器: {self.interface_name}")

                # 根据接口类型执行连接
                success = False

                if self.sensor_interface == SensorInterface.SERIAL:
                    success = self._connect_serial()
                elif self.sensor_interface == SensorInterface.I2C:
                    success = self._connect_i2c()
                elif self.sensor_interface == SensorInterface.SPI:
                    success = self._connect_spi()
                elif self.sensor_interface == SensorInterface.USB:
                    success = self._connect_usb()
                elif self.sensor_interface == SensorInterface.NETWORK:
                    success = self._connect_network()
                elif self.sensor_interface == SensorInterface.BLUETOOTH:
                    success = self._connect_bluetooth()
                else:
                    # 模拟、数字、GPIO接口在初始化时已连接
                    success = True

                if success:
                    self.connection_status = ConnectionStatus.CONNECTED
                    logger.info(f"传感器连接成功: {self.interface_name}")

                    # 启动数据采集线程
                    if self.sampling_rate > 0:
                        self._start_sampling()

                    return True
                else:
                    self.connection_status = ConnectionStatus.ERROR
                    logger.error(f"传感器连接失败: {self.interface_name}")
                    return False

            except Exception as e:
                self.connection_status = ConnectionStatus.ERROR
                self.last_error = str(e)
                logger.error(f"传感器连接异常: {e}")
                return False

    def _connect_serial(self) -> bool:
        """连接串口"""
        try:
            if hasattr(self, "_serial") and self._serial:
                if not self._serial.is_open:
                    self._serial.open()
                return self._serial.is_open
            return False
        except Exception as e:
            logger.error(f"串口连接失败: {e}")
            return False

    def _connect_i2c(self) -> bool:
        """连接I2C"""
        try:
            if hasattr(self, "_i2c_bus") and self._i2c_bus:
                # 发送测试字节
                self._i2c_bus.write_byte(self._i2c_address, 0x00)
                return True
        except Exception as e:
            logger.error(f"I2C连接失败: {e}")
            return False
        return False

    def _connect_spi(self) -> bool:
        """连接SPI"""
        try:
            if hasattr(self, "_spi") and self._spi:
                # 发送测试数据
                test_data = [0xAA, 0x55, 0x00]
                self._spi.xfer(test_data)
                return True
        except Exception as e:
            logger.error(f"SPI连接失败: {e}")
            return False
        return False

    def _connect_usb(self) -> bool:
        """连接USB"""
        try:
            if hasattr(self, "_usb_vendor_id"):
                import usb.core  # type: ignore
                import usb.util  # type: ignore

                # 查找设备
                dev = usb.core.find(
                    idVendor=self._usb_vendor_id, idProduct=self._usb_product_id
                )

                if dev is None:
                    logger.error("未找到USB设备")
                    return False

                # 配置设备
                if dev.is_kernel_driver_active(0):
                    dev.detach_kernel_driver(0)

                usb.util.claim_interface(dev, 0)
                self._usb_device = dev

                return True

        except ImportError:
            logger.error("PyUSB库未安装")
        except Exception as e:
            logger.error(f"USB连接失败: {e}")

        return False

    def _connect_network(self) -> bool:
        """连接网络"""
        import socket

        try:
            self._network_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._network_socket.settimeout(5.0)
            self._network_socket.connect((self._network_host, self._network_port))

            # 发送测试命令
            test_command = b"TEST\n"
            self._network_socket.send(test_command)

            response = self._network_socket.recv(1024)
            if response.startswith(b"OK"):
                return True
            else:
                self._network_socket.close()
                self._network_socket = None
                return False

        except Exception as e:
            logger.error(f"网络连接失败: {e}")
            if hasattr(self, "_network_socket") and self._network_socket:
                self._network_socket.close()
                self._network_socket = None
            return False

    def _connect_bluetooth(self) -> bool:
        """连接蓝牙"""
        try:
            import bluetooth

            self._bluetooth_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self._bluetooth_socket.connect(
                (self._bluetooth_address, self._bluetooth_port)
            )
            return True

        except ImportError:
            logger.error("pybluez未安装")
        except Exception as e:
            logger.error(f"蓝牙连接失败: {e}")

        return False

    def disconnect(self) -> bool:
        """断开与传感器硬件的连接"""
        with self._connection_lock:
            try:
                if self.connection_status == ConnectionStatus.DISCONNECTED:
                    return True

                logger.info(f"正在断开传感器连接: {self.interface_name}")

                # 停止数据采集
                self._stop_sampling()

                # 根据接口类型断开连接
                if self.sensor_interface == SensorInterface.SERIAL:
                    if (
                        hasattr(self, "_serial")
                        and self._serial
                        and self._serial.is_open
                    ):
                        self._serial.close()

                elif self.sensor_interface == SensorInterface.I2C:
                    if hasattr(self, "_i2c_bus"):
                        self._i2c_bus.close()

                elif self.sensor_interface == SensorInterface.SPI:
                    if hasattr(self, "_spi"):
                        self._spi.close()

                elif self.sensor_interface == SensorInterface.USB:
                    if hasattr(self, "_usb_device") and self._usb_device:
                        import usb.util

                        usb.util.release_interface(self._usb_device, 0)
                        self._usb_device = None

                elif self.sensor_interface == SensorInterface.NETWORK:
                    if hasattr(self, "_network_socket") and self._network_socket:
                        self._network_socket.close()
                        self._network_socket = None

                elif self.sensor_interface == SensorInterface.BLUETOOTH:
                    if hasattr(self, "_bluetooth_socket") and self._bluetooth_socket:
                        self._bluetooth_socket.close()
                        self._bluetooth_socket = None

                self.connection_status = ConnectionStatus.DISCONNECTED
                logger.info(f"传感器断开连接成功: {self.interface_name}")
                return True

            except Exception as e:
                self.connection_status = ConnectionStatus.ERROR
                self.last_error = str(e)
                logger.error(f"传感器断开连接异常: {e}")
                return False

    def is_connected(self) -> bool:
        """检查是否已连接到传感器硬件"""
        if self.connection_status != ConnectionStatus.CONNECTED:
            return False

        # 根据接口类型检查连接状态
        try:
            if self.sensor_interface == SensorInterface.SERIAL:
                return (
                    hasattr(self, "_serial") and self._serial and self._serial.is_open
                )

            elif self.sensor_interface == SensorInterface.I2C:
                if hasattr(self, "_i2c_bus"):
                    try:
                        self._i2c_bus.read_byte(self._i2c_address)
                        return True
                    except BaseException:
                        return False
                return False

            elif self.sensor_interface == SensorInterface.SPI:
                return hasattr(self, "_spi") and self._spi and self._spi.fd >= 0

            elif self.sensor_interface == SensorInterface.USB:
                return hasattr(self, "_usb_device") and self._usb_device is not None

            elif self.sensor_interface == SensorInterface.NETWORK:
                if hasattr(self, "_network_socket") and self._network_socket:
                    # 发送心跳包检查连接
                    try:
                        self._network_socket.settimeout(1.0)
                        self._network_socket.send(b"PING\n")
                        response = self._network_socket.recv(1024)
                        return response.startswith(b"PONG")
                    except BaseException:
                        return False
                return False

            elif self.sensor_interface == SensorInterface.BLUETOOTH:
                return (
                    hasattr(self, "_bluetooth_socket")
                    and self._bluetooth_socket is not None
                )

            else:
                # 模拟、数字、GPIO接口，只要状态是CONNECTED就认为是连接的
                return True

        except Exception as e:
            logger.warning(f"检查传感器连接状态失败: {e}")
            return False

    def get_hardware_info(self) -> Dict[str, Any]:
        """获取传感器硬件信息"""
        info = {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "sensor_interface": self.sensor_interface.value,
            "interface_config": self.interface_config,
            "connection_status": self.connection_status.value,
            "sampling_rate": self.sampling_rate,
            "resolution": self.resolution,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "calibration_offset": self.calibration_offset,
            "calibration_scale": self.calibration_scale,
            "buffer_size": len(self.data_buffer),
            "buffer_max_size": self.buffer_max_size,
            "last_sample_time": self.last_sample_time,
            "sampling_running": self._sampling_running,
            "performance_stats": self.get_performance_stats(),
        }

        # 添加接口特定信息
        if self.sensor_interface == SensorInterface.I2C:
            info["i2c_config"] = {
                "bus": self.interface_config.get("bus", 1),
                "address": self.interface_config.get("address", 0x40),
            }

        return info

    def execute_operation(self, operation: str, **kwargs) -> Any:
        """执行传感器操作"""
        if not self.is_connected():
            raise ConnectionError("传感器未连接")

        try:
            # 根据操作类型执行
            if operation == "read":
                return self.read_data(**kwargs)
            elif operation == "read_raw":
                return self.read_raw_data(**kwargs)
            elif operation == "read_buffer":
                return self.read_data_buffer(**kwargs)
            elif operation == "clear_buffer":
                return self.clear_data_buffer(**kwargs)
            elif operation == "calibrate":
                return self.calibrate(**kwargs)
            elif operation == "set_sampling_rate":
                return self.set_sampling_rate(**kwargs)
            elif operation == "start_sampling":
                return self.start_sampling(**kwargs)
            elif operation == "stop_sampling":
                return self.stop_sampling(**kwargs)
            elif operation == "get_sensor_info":
                return self.get_hardware_info()
            else:
                raise OperationError(f"不支持的传感器操作: {operation}")

        except (HardwareError, ConnectionError):
            raise
        except Exception as e:
            raise OperationError(f"传感器操作失败: {e}")

    def read_data(self, apply_calibration: bool = True) -> SensorData:
        """读取传感器数据"""
        try:
            # 读取原始数据
            raw_value = self._read_raw_value()

            # 应用校准
            if apply_calibration:
                calibrated_value = self._apply_calibration(raw_value)
            else:
                calibrated_value = raw_value

            # 限制范围
            limited_value = self._limit_range(calibrated_value)

            # 创建传感器数据对象
            sensor_data = SensorData(
                sensor_type=self.sensor_type,
                value=limited_value,
                unit=self._get_unit(),
                timestamp=time.time(),
                accuracy=self._estimate_accuracy(),
                metadata={
                    "raw_value": raw_value,
                    "calibrated": apply_calibration,
                    "sensor_id": self.sensor_id,
                },
            )

            # 添加到缓冲区
            self._add_to_buffer(sensor_data)
            self.last_sample_time = time.time()

            return sensor_data

        except Exception as e:
            logger.error(f"读取传感器数据失败: {e}")
            raise OperationError(f"读取传感器数据失败: {e}")

    def read_raw_data(self) -> Any:
        """读取原始传感器数据"""
        return self._read_raw_value()

    def read_data_buffer(self, max_samples: int = 100) -> List[SensorData]:
        """读取数据缓冲区"""
        with threading.Lock():
            if max_samples <= 0 or max_samples >= len(self.data_buffer):
                return self.data_buffer.copy()
            else:
                return self.data_buffer[-max_samples:].copy()

    def clear_data_buffer(self) -> bool:
        """清空数据缓冲区"""
        with threading.Lock():
            self.data_buffer.clear()
            logger.info("传感器数据缓冲区已清空")
            return True

    def calibrate(self, reference_value: float = None) -> bool:
        """校准传感器"""
        try:
            if reference_value is not None:
                # 使用参考值校准
                raw_value = self._read_raw_value()
                self.calibration_offset = reference_value - raw_value
                self.calibration_scale = 1.0
            else:
                # 自动校准（零点校准）
                samples = []
                for _ in range(10):
                    samples.append(self._read_raw_value())
                    time.sleep(0.1)

                avg_raw = sum(samples) / len(samples)
                self.calibration_offset = -avg_raw
                self.calibration_scale = 1.0

            logger.info(
                f"传感器校准完成: offset={                     self.calibration_offset}, scale={                     self.calibration_scale}"
            )
            return True

        except Exception as e:
            logger.error(f"传感器校准失败: {e}")
            return False

    def set_sampling_rate(self, rate: float) -> bool:
        """设置采样率"""
        if rate < 0:
            raise OperationError("采样率必须为非负数")

        old_rate = self.sampling_rate
        self.sampling_rate = rate

        # 如果采样率变化，重新启动采样线程
        if old_rate != rate and self._sampling_running:
            self._stop_sampling()
            if rate > 0:
                self._start_sampling()

        logger.info(f"采样率设置为: {rate} Hz")
        return True

    def start_sampling(self) -> bool:
        """启动数据采集"""
        if self._sampling_running:
            logger.warning("数据采集已在运行")
            return True

        if self.sampling_rate <= 0:
            logger.warning("采样率未设置或为0，无法启动数据采集")
            return False

        self._start_sampling()
        return True

    def stop_sampling(self) -> bool:
        """停止数据采集"""
        if not self._sampling_running:
            logger.warning("数据采集未运行")
            return True

        self._stop_sampling()
        return True

    def _read_raw_value(self) -> Any:
        """读取原始传感器值（子类实现）"""
        # 根据传感器类型和接口读取数据
        if self.sensor_interface == SensorInterface.I2C:
            return self._read_i2c_value()
        elif self.sensor_interface == SensorInterface.SPI:
            return self._read_spi_value()
        elif self.sensor_interface == SensorInterface.SERIAL:
            return self._read_serial_value()
        elif self.sensor_interface == SensorInterface.ANALOG:
            return self._read_analog_value()
        elif self.sensor_interface == SensorInterface.DIGITAL:
            return self._read_digital_value()
        elif self.sensor_interface == SensorInterface.GPIO:
            return self._read_gpio_value()
        elif self.sensor_interface == SensorInterface.UNKNOWN:
            # 未知接口，无法读取真实数据
            from .base_interface import OperationError

            raise OperationError(
                "传感器接口类型为UNKNOWN，无法读取真实数据。"
                f"传感器: {self.sensor_name}, 类型: {self.sensor_type.value}"
            )
        else:
            # 未支持的接口类型
            from .base_interface import OperationError

            raise OperationError(
                f"不支持的传感器接口类型: {self.sensor_interface.value}"
            )

    def _read_i2c_value(self) -> float:
        """读取I2C传感器值"""
        if not hasattr(self, "_i2c_bus") or not self._i2c_bus:
            raise OperationError("I2C接口未初始化")

        try:
            # 根据传感器类型读取数据
            if self.sensor_type == SensorType.TEMPERATURE:
                # 假设使用常见的温度传感器如LM75
                data = self._i2c_bus.read_word_data(self._i2c_address, 0x00)
                temperature = (data >> 8) | ((data & 0xFF) << 8)
                if temperature > 32767:
                    temperature -= 65536
                return temperature / 256.0

            elif self.sensor_type == SensorType.HUMIDITY:
                # 假设使用常见的湿度传感器如HTU21D
                self._i2c_bus.write_byte(self._i2c_address, 0xF5)  # 触发测量
                time.sleep(0.05)
                data = self._i2c_bus.read_i2c_block_data(self._i2c_address, 0x00, 3)
                raw_value = (data[0] << 8) | data[1]
                humidity = -6.0 + 125.0 * (raw_value / 65536.0)
                return max(0.0, min(100.0, humidity))

            else:
                # 通用I2C读取
                data = self._i2c_bus.read_byte(self._i2c_address)
                return float(data)

        except Exception as e:
            raise OperationError(f"I2C读取失败: {e}")

    def _read_spi_value(self) -> float:
        """读取SPI传感器值"""
        if not hasattr(self, "_spi") or not self._spi:
            raise OperationError("SPI接口未初始化")

        try:
            # 发送读取命令
            command = [0x01, 0x00, 0x00]  # 示例命令
            response = self._spi.xfer(command)

            # 解析响应
            if len(response) >= 3:
                value = (response[1] << 8) | response[2]
                return float(value)
            else:
                raise OperationError(f"SPI响应长度不足: {len(response)}")

        except Exception as e:
            raise OperationError(f"SPI读取失败: {e}")

    def _read_serial_value(self) -> str:
        """读取串口传感器值"""
        if not hasattr(self, "_serial") or not self._serial or not self._serial.is_open:
            raise OperationError("串口接口未初始化或未打开")

        try:
            # 发送读取命令
            command = "READ\n".encode()
            self._serial.write(command)

            # 读取响应
            response = self._serial.readline().decode().strip()

            # 尝试解析为数值
            try:
                return float(response)
            except ValueError:
                return response

        except Exception as e:
            raise OperationError(f"串口读取失败: {e}")

    def _read_analog_value(self) -> float:
        """读取模拟传感器值"""
        if not hasattr(self, "_analog_channel"):
            raise OperationError("模拟接口未初始化")

        try:
            if self._analog_channel == "simulated":
                # 模拟模式已禁用
                raise RuntimeError(
                    "模拟传感器模式已禁用。项目要求禁止使用虚拟数据，必须使用真实硬件。"
                )
            else:
                # 实际硬件读取
                pass

                return self._analog_channel.value

        except Exception as e:
            raise OperationError(f"模拟读取失败: {e}")

    def _read_digital_value(self) -> bool:
        """读取数字传感器值"""
        if not hasattr(self, "_gpio_library"):
            raise OperationError("数字接口未初始化")

        try:
            if self._gpio_library == "RPi.GPIO":
                import RPi.GPIO as GPIO

                return GPIO.input(self._digital_pin) == GPIO.HIGH
            elif self._gpio_library == "gpiozero":
                return self._digital_device.is_pressed
            elif self._gpio_library == "simulated":
                # 模拟模式已禁用
                raise RuntimeError(
                    "模拟数字传感器模式已禁用。项目要求禁止使用虚拟数据，必须使用真实硬件。"
                )
            else:
                raise OperationError("未知的GPIO库")

        except Exception as e:
            raise OperationError(f"数字读取失败: {e}")

    def _read_gpio_value(self) -> Any:
        """读取GPIO传感器值"""
        return self._read_digital_value()

    def _read_simulated_value(self) -> Any:
        """读取模拟传感器值（已禁用）"""
        raise RuntimeError(
            "模拟传感器值读取已禁用。项目要求禁止使用虚拟数据，必须使用真实硬件传感器。"
        )

    def _apply_calibration(self, raw_value: float) -> float:
        """应用校准"""
        return raw_value * self.calibration_scale + self.calibration_offset

    def _limit_range(self, value: float) -> float:
        """限制值在量程范围内"""
        if self.range_min <= value <= self.range_max:
            return value
        elif value < self.range_min:
            return self.range_min
        else:
            return self.range_max

    def _get_unit(self) -> str:
        """获取单位"""
        units = {
            SensorType.TEMPERATURE: "°C",
            SensorType.HUMIDITY: "%",
            SensorType.PRESSURE: "hPa",
            SensorType.LIGHT: "lux",
            SensorType.DISTANCE: "cm",
            SensorType.ACCELEROMETER: "m/s²",
            SensorType.GYROSCOPE: "rad/s",
            SensorType.MAGNETOMETER: "μT",
            SensorType.GPS: "度",
            SensorType.FLOW: "L/min",
            SensorType.PH: "pH",
            SensorType.CONDUCTIVITY: "μS/cm",
        }
        return units.get(self.sensor_type, "")

    def _estimate_accuracy(self) -> float:
        """估计数据准确度"""
        # 基于连接状态、采样率等因素估计准确度
        accuracy = 1.0

        if not self.is_connected():
            accuracy *= 0.3

        if self.sampling_rate > 100:  # 高采样率可能降低准确度
            accuracy *= 0.9

        # 真实数据准确度较低
        if self.sensor_interface == SensorInterface.UNKNOWN:
            accuracy *= 0.5

        return max(0.1, accuracy)

    def _add_to_buffer(self, sensor_data: SensorData):
        """添加数据到缓冲区"""
        with threading.Lock():
            self.data_buffer.append(sensor_data)

            # 限制缓冲区大小
            if len(self.data_buffer) > self.buffer_max_size:
                self.data_buffer = self.data_buffer[-self.buffer_max_size:]

    def _start_sampling(self):
        """启动数据采集线程"""
        if self._sampling_running:
            return

        self._sampling_running = True
        self._sampling_thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True,
            name=f"SensorSampling_{self.sensor_id}",
        )
        self._sampling_thread.start()

        logger.info(f"传感器数据采集启动: {self.sensor_id}")

    def _stop_sampling(self):
        """停止数据采集线程"""
        self._sampling_running = False

        if self._sampling_thread and self._sampling_thread.is_alive():
            self._sampling_thread.join(timeout=2.0)

        logger.info(f"传感器数据采集停止: {self.sensor_id}")

    def _sampling_loop(self):
        """数据采集循环"""
        logger.info(f"传感器数据采集循环启动: {self.sensor_id}")

        while self._sampling_running and self.is_connected():
            try:
                # 计算采样间隔
                sample_interval = (
                    1.0 / self.sampling_rate if self.sampling_rate > 0 else 1.0
                )

                # 读取数据
                sensor_data = self.read_data(apply_calibration=True)

                # 等待下一个采样点
                time.sleep(sample_interval)

            except OperationError as e:
                logger.warning(f"传感器数据采集失败: {e}")
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"传感器数据采集异常: {e}")
                time.sleep(1.0)

        logger.info(f"传感器数据采集循环停止: {self.sensor_id}")

    def __del__(self):
        """析构函数"""
        self._stop_sampling()
        self.disconnect()
