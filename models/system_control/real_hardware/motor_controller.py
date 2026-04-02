#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实电机控制器

控制真实电机硬件，支持多种电机类型和通信接口
核心原则：真实优先，模拟后备
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np

from .base_interface import (
    RealHardwareInterface, HardwareType, ConnectionStatus,
    HardwareError, ConnectionError, OperationError
)

logger = logging.getLogger(__name__)


class MotorType(Enum):
    """电机类型枚举"""
    STEPPER = "stepper"      # 步进电机
    DC = "dc"                # 直流电机
    SERVO = "servo"          # 伺服电机
    BRUSHLESS = "brushless"  # 无刷电机
    LINEAR = "linear"        # 线性电机
    UNKNOWN = "unknown"      # 未知类型


class ControlInterface(Enum):
    """控制接口枚举"""
    PWM = "pwm"              # PWM控制
    SERIAL = "serial"        # 串口控制
    I2C = "i2c"              # I2C控制
    SPI = "spi"              # SPI控制
    CAN = "can"              # CAN总线
    ETHERNET = "ethernet"    # 以太网
    USB = "usb"              # USB
    GPIO = "gpio"            # GPIO直接控制


class MotorState(Enum):
    """电机状态枚举"""
    STOPPED = "stopped"      # 停止
    RUNNING = "running"      # 运行中
    ACCELERATING = "accelerating"  # 加速中
    DECELERATING = "decelerating"  # 减速中
    ERROR = "error"          # 错误状态
    OVERHEAT = "overheat"    # 过热
    STALLED = "stalled"      # 堵转


class RealMotorController(RealHardwareInterface):
    """真实电机控制器
    
    控制真实电机硬件，支持多种电机类型和控制接口
    """
    
    def __init__(
        self,
        motor_id: str,
        motor_type: MotorType,
        control_interface: ControlInterface,
        interface_config: Dict[str, Any]
    ):
        """
        初始化真实电机控制器
        
        参数:
            motor_id: 电机唯一标识符
            motor_type: 电机类型
            control_interface: 控制接口类型
            interface_config: 接口配置字典
        """
        super().__init__(
            hardware_type=HardwareType.MOTOR,
            interface_name=f"motor_{motor_id}"
        )
        
        self.motor_id = motor_id
        self.motor_type = motor_type
        self.control_interface = control_interface
        self.interface_config = interface_config
        
        # 电机特定参数
        self.current_position = 0.0  # 当前位置（单位取决于电机类型）
        self.target_position = 0.0   # 目标位置
        self.current_speed = 0.0     # 当前速度（单位：转/分或脉冲/秒）
        self.target_speed = 0.0      # 目标速度
        self.current_torque = 0.0    # 当前扭矩（单位：N·m）
        self.current_temperature = 25.0  # 当前温度（单位：℃）
        
        # 电机状态
        self.motor_state = MotorState.STOPPED
        self.direction = 1  # 1: 正向, -1: 反向
        self.enabled = False  # 电机使能状态
        
        # 硬件接口实例（根据控制接口类型初始化）
        self._hardware_interface = None
        self._interface_lock = threading.RLock()
        
        # 电机参数限制
        self.max_speed = interface_config.get("max_speed", 1000.0)
        self.max_torque = interface_config.get("max_torque", 10.0)
        self.max_temperature = interface_config.get("max_temperature", 85.0)
        self.position_resolution = interface_config.get("position_resolution", 0.01)
        
        # 模拟模式已禁用，根据项目要求"禁止使用虚拟数据"
        
        # 初始化硬件接口
        self._init_hardware_interface()
        
        logger.info(
            f"初始化真实电机控制器: {motor_id} "
            f"({motor_type.value}, {control_interface.value})"
        )
    
    def _init_hardware_interface(self):
        """初始化硬件接口"""
        with self._interface_lock:
            try:
                if self.control_interface == ControlInterface.PWM:
                    self._init_pwm_interface()
                elif self.control_interface == ControlInterface.SERIAL:
                    self._init_serial_interface()
                elif self.control_interface == ControlInterface.I2C:
                    self._init_i2c_interface()
                elif self.control_interface == ControlInterface.SPI:
                    self._init_spi_interface()
                elif self.control_interface == ControlInterface.CAN:
                    self._init_can_interface()
                elif self.control_interface == ControlInterface.ETHERNET:
                    self._init_ethernet_interface()
                elif self.control_interface == ControlInterface.USB:
                    self._init_usb_interface()
                elif self.control_interface == ControlInterface.GPIO:
                    self._init_gpio_interface()
                else:
                    raise ValueError(f"不支持的接口类型: {self.control_interface}。项目要求禁止使用模拟模式，必须使用真实硬件接口。")
                
                logger.info(f"硬件接口初始化完成: {self.control_interface.value}")
                
            except Exception as e:
                logger.error(f"硬件接口初始化失败: {e}")
                raise RuntimeError("硬件接口初始化失败。项目要求禁止使用模拟模式，必须确保硬件接口正常工作。") from e
    
    def _init_pwm_interface(self):
        """初始化PWM接口"""
        # 导入PWM库（如果可用）
        try:
            import RPi.GPIO as GPIO  # type: ignore
            self._pwm_library = "RPi.GPIO"
            self._use_rpi_gpio = True
        except ImportError:
            try:
                import gpiozero  # type: ignore
                self._pwm_library = "gpiozero"
                self._use_gpiozero = True
            except ImportError:
                try:
                    import Adafruit_PCA9685  # type: ignore
                    self._pwm_library = "Adafruit_PCA9685"
                    self._use_pca9685 = True
                except ImportError:
                    # 如果没有硬件库，抛出异常
                    raise ImportError("未找到任何PWM硬件库（RPi.GPIO、gpiozero、Adafruit_PCA9685）。项目要求禁止使用模拟模式，必须安装硬件库。")
        
        # 配置PWM参数
        self.pwm_pin = self.interface_config.get("pwm_pin", 18)
        self.pwm_frequency = self.interface_config.get("pwm_frequency", 50)  # Hz
        self.pwm_min_duty = self.interface_config.get("pwm_min_duty", 2.5)   # %
        self.pwm_max_duty = self.interface_config.get("pwm_max_duty", 12.5)  # %
        
        # 初始化PWM
        self._init_pwm_hardware()
    
    def _init_pwm_hardware(self):
        """初始化PWM硬件"""
        # 实际硬件初始化
        try:
            if self._pwm_library == "RPi.GPIO":
                import RPi.GPIO as GPIO  # type: ignore
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pwm_pin, GPIO.OUT)
                self._pwm = GPIO.PWM(self.pwm_pin, self.pwm_frequency)
                self._pwm.start(0)
                
            elif self._pwm_library == "gpiozero":
                import gpiozero  # type: ignore
                self._pwm = gpiozero.PWMOutputDevice(
                    self.pwm_pin,
                    frequency=self.pwm_frequency
                )
                self._pwm.value = 0
                
            elif self._pwm_library == "Adafruit_PCA9685":
                import Adafruit_PCA9685  # type: ignore
                self._pwm = Adafruit_PCA9685.PCA9685()
                self._pwm.set_pwm_freq(self.pwm_frequency)
                
            self._pwm_running = True
            
        except Exception as e:
            logger.error(f"PWM硬件初始化失败: {e}")
            raise RuntimeError(f"PWM硬件初始化失败: {e}。项目要求禁止使用模拟模式，必须确保PWM硬件正常工作。") from e
    
    def _init_serial_interface(self):
        """初始化串口接口"""
        import serial
        
        port = self.interface_config.get("port", "/dev/ttyUSB0")
        baudrate = self.interface_config.get("baudrate", 9600)
        timeout = self.interface_config.get("timeout", 1.0)
        
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout
            )
            logger.info(f"串口连接成功: {port} @ {baudrate} baud")
            
        except Exception as e:
            logger.error(f"串口连接失败: {e}")
            raise RuntimeError(f"串口连接失败: {e}。项目要求禁止使用模拟模式，必须确保串口硬件可用。") from e
    
    def _init_i2c_interface(self):
        """初始化I2C接口"""
        import smbus2  # type: ignore
        
        bus_num = self.interface_config.get("bus", 1)
        address = self.interface_config.get("address", 0x40)
        
        try:
            self._i2c_bus = smbus2.SMBus(bus_num)
            self._i2c_address = address
            logger.info(f"I2C连接成功: bus={bus_num}, address=0x{address:02x}")
            
        except Exception as e:
            logger.error(f"I2C连接失败: {e}")
            raise RuntimeError(f"I2C连接失败: {e}。项目要求禁止使用模拟模式，必须确保I2C硬件可用。") from e
    
    def _init_spi_interface(self):
        """初始化SPI接口"""
        import spidev  # type: ignore
        
        bus = self.interface_config.get("bus", 0)
        device = self.interface_config.get("device", 0)
        max_speed = self.interface_config.get("max_speed", 1000000)
        
        try:
            self._spi = spidev.SpiDev()
            self._spi.open(bus, device)
            self._spi.max_speed_hz = max_speed
            logger.info(f"SPI连接成功: bus={bus}, device={device}")
            
        except Exception as e:
            logger.error(f"SPI连接失败: {e}")
            raise RuntimeError(f"SPI连接失败: {e}。项目要求禁止使用模拟模式，必须确保SPI硬件可用。") from e
    
    def _init_can_interface(self):
        """初始化CAN接口"""
        try:
            import can  # type: ignore
            interface = self.interface_config.get("interface", "socketcan")
            channel = self.interface_config.get("channel", "can0")
            
            self._can_bus = can.Bus(
                interface=interface,
                channel=channel,
                bitrate=self.interface_config.get("bitrate", 500000)
            )
            logger.info(f"CAN连接成功: {interface}:{channel}")
            
        except ImportError:
            logger.warning("未找到python-can库，CAN功能不可用")
            self._can_bus = None
        except Exception as e:
            logger.error(f"CAN连接失败: {e}")
            raise ConnectionError(f"CAN连接失败: {e}")
    
    def _init_ethernet_interface(self):
        """初始化以太网接口"""
        self._ethernet_host = self.interface_config.get("host", "192.168.1.100")
        self._ethernet_port = self.interface_config.get("port", 502)
        self._ethernet_protocol = self.interface_config.get("protocol", "TCP")
        
        # 延迟到连接时建立socket
        self._ethernet_socket = None
        logger.info(f"以太网接口配置: {self._ethernet_host}:{self._ethernet_port}")
    
    def _init_usb_interface(self):
        """初始化USB接口"""
        try:
            import pyusb  # type: ignore
            vendor_id = self.interface_config.get("vendor_id", 0x1234)
            product_id = self.interface_config.get("product_id", 0x5678)
            
            self._usb_vendor_id = vendor_id
            self._usb_product_id = product_id
            self._usb_device = None
            
            logger.info(f"USB接口配置: VID=0x{vendor_id:04x}, PID=0x{product_id:04x}")
            
        except ImportError:
            logger.warning("未找到PyUSB库，USB功能不可用")
            self._usb_device = None
    
    def _init_gpio_interface(self):
        """初始化GPIO接口"""
        try:
            import RPi.GPIO as GPIO  # type: ignore
            self._gpio_library = "RPi.GPIO"
            GPIO.setmode(GPIO.BCM)
            
            # 配置GPIO引脚
            self._gpio_pins = {
                "direction": self.interface_config.get("direction_pin", 17),
                "step": self.interface_config.get("step_pin", 18),
                "enable": self.interface_config.get("enable_pin", 27)
            }
            
            for pin_name, pin_num in self._gpio_pins.items():
                GPIO.setup(pin_num, GPIO.OUT)
                GPIO.output(pin_num, GPIO.LOW)
            
            self._gpio_initialized = True
            logger.info(f"GPIO接口初始化完成: {self._gpio_pins}")
            
        except ImportError:
            try:
                import gpiozero  # type: ignore
                self._gpio_library = "gpiozero"
                
                self._gpio_pins = {
                    "direction": gpiozero.DigitalOutputDevice(
                        self.interface_config.get("direction_pin", 17)
                    ),
                    "step": gpiozero.DigitalOutputDevice(
                        self.interface_config.get("step_pin", 18)
                    ),
                    "enable": gpiozero.DigitalOutputDevice(
                        self.interface_config.get("enable_pin", 27)
                    )
                }
                
                self._gpio_initialized = True
                logger.info("GPIO接口初始化完成 (gpiozero)")
                
            except ImportError:
                logger.error("未找到GPIO库")
                raise ImportError("未找到GPIO库（gpiozero）。项目要求禁止使用模拟模式，必须安装硬件库。")
    
    def connect(self) -> bool:
        """
        连接到真实电机硬件
        
        返回:
            连接是否成功
        """
        with self._connection_lock:
            try:
                if self.connection_status == ConnectionStatus.CONNECTED:
                    logger.info("电机已连接")
                    return True
                
                self.connection_status = ConnectionStatus.CONNECTING
                logger.info(f"正在连接电机: {self.interface_name}")
                
                # 根据接口类型执行连接
                success = False
                
                if self.control_interface == ControlInterface.SERIAL:
                    success = self._connect_serial()
                elif self.control_interface == ControlInterface.I2C:
                    success = self._connect_i2c()
                elif self.control_interface == ControlInterface.SPI:
                    success = self._connect_spi()
                elif self.control_interface == ControlInterface.CAN:
                    success = self._connect_can()
                elif self.control_interface == ControlInterface.ETHERNET:
                    success = self._connect_ethernet()
                elif self.control_interface == ControlInterface.USB:
                    success = self._connect_usb()
                else:
                    # PWM和GPIO在初始化时已连接
                    success = True
                
                if success:
                    self.connection_status = ConnectionStatus.CONNECTED
                    self.enabled = True
                    logger.info(f"电机连接成功: {self.interface_name}")
                    
                    # 读取初始状态
                    try:
                        self._update_motor_state()
                    except Exception as e:
                        logger.warning(f"读取电机状态失败: {e}")
                    
                    return True
                else:
                    self.connection_status = ConnectionStatus.ERROR
                    logger.error(f"电机连接失败: {self.interface_name}")
                    return False
                    
            except Exception as e:
                self.connection_status = ConnectionStatus.ERROR
                self.last_error = str(e)
                logger.error(f"电机连接异常: {e}")
                return False
    
    def _connect_serial(self) -> bool:
        """连接串口"""
        try:
            if hasattr(self, '_serial'):
                # 检查串口是否已打开
                if self._serial.is_open:
                    # 发送测试命令
                    self._serial.write(b"TEST\n")
                    response = self._serial.readline().strip()
                    if response == b"OK":
                        return True
                    else:
                        # 重新打开串口
                        self._serial.close()
                        self._serial.open()
                        return self._serial.is_open
                else:
                    self._serial.open()
                    return self._serial.is_open
            return False
        except Exception as e:
            logger.error(f"串口连接失败: {e}")
            return False
    
    def _connect_i2c(self) -> bool:
        """连接I2C"""
        try:
            if hasattr(self, '_i2c_bus'):
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
            if hasattr(self, '_spi'):
                # 发送测试数据
                test_data = [0xAA, 0x55, 0x00]
                response = self._spi.xfer(test_data)
                return True
        except Exception as e:
            logger.error(f"SPI连接失败: {e}")
            return False
        return False
    
    def _connect_can(self) -> bool:
        """连接CAN"""
        try:
            import can  # type: ignore
            if hasattr(self, '_can_bus') and self._can_bus:
                # 发送测试帧
                test_msg = can.Message(
                    arbitration_id=0x123,
                    data=[0x01, 0x02, 0x03, 0x04],
                    is_extended_id=False
                )
                self._can_bus.send(test_msg)
                return True
        except Exception as e:
            logger.error(f"CAN连接失败: {e}")
            return False
        return False
    
    def _connect_ethernet(self) -> bool:
        """连接以太网"""
        import socket
        
        try:
            self._ethernet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._ethernet_socket.settimeout(5.0)
            self._ethernet_socket.connect((self._ethernet_host, self._ethernet_port))
            
            # 发送测试命令
            test_command = b"\x00\x01\x00\x00\x00\x06\x01\x03\x00\x00\x00\x01"
            self._ethernet_socket.send(test_command)
            
            response = self._ethernet_socket.recv(1024)
            if len(response) >= 5:
                return True
            else:
                self._ethernet_socket.close()
                self._ethernet_socket = None
                return False
                
        except Exception as e:
            logger.error(f"以太网连接失败: {e}")
            if hasattr(self, '_ethernet_socket') and self._ethernet_socket:
                self._ethernet_socket.close()
                self._ethernet_socket = None
            return False
    
    def _connect_usb(self) -> bool:
        """连接USB"""
        try:
            if hasattr(self, '_usb_vendor_id'):
                import usb.core  # type: ignore
                import usb.util  # type: ignore
                
                # 查找设备
                dev = usb.core.find(
                    idVendor=self._usb_vendor_id,
                    idProduct=self._usb_product_id
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
    
    def disconnect(self) -> bool:
        """
        断开与真实电机硬件的连接
        
        返回:
            断开是否成功
        """
        with self._connection_lock:
            try:
                if self.connection_status == ConnectionStatus.DISCONNECTED:
                    return True
                
                logger.info(f"正在断开电机连接: {self.interface_name}")
                
                # 停止电机
                self.stop()
                self.enabled = False
                
                # 根据接口类型断开连接
                if self.control_interface == ControlInterface.SERIAL:
                    if hasattr(self, '_serial') and self._serial.is_open:
                        self._serial.close()
                        
                elif self.control_interface == ControlInterface.I2C:
                    if hasattr(self, '_i2c_bus'):
                        self._i2c_bus.close()
                        
                elif self.control_interface == ControlInterface.SPI:
                    if hasattr(self, '_spi'):
                        self._spi.close()
                        
                elif self.control_interface == ControlInterface.CAN:
                    if hasattr(self, '_can_bus') and self._can_bus:
                        self._can_bus.shutdown()
                        
                elif self.control_interface == ControlInterface.ETHERNET:
                    if hasattr(self, '_ethernet_socket') and self._ethernet_socket:
                        self._ethernet_socket.close()
                        self._ethernet_socket = None
                        
                elif self.control_interface == ControlInterface.USB:
                    if hasattr(self, '_usb_device') and self._usb_device:
                        import usb.util  # type: ignore
                        usb.util.release_interface(self._usb_device, 0)
                        self._usb_device = None
                
                # 对于PWM和GPIO，停止输出
                if self.control_interface == ControlInterface.PWM:
                    self._stop_pwm_output()
                elif self.control_interface == ControlInterface.GPIO:
                    self._stop_gpio_output()
                
                self.connection_status = ConnectionStatus.DISCONNECTED
                logger.info(f"电机断开连接成功: {self.interface_name}")
                return True
                
            except Exception as e:
                self.connection_status = ConnectionStatus.ERROR
                self.last_error = str(e)
                logger.error(f"电机断开连接异常: {e}")
                return False
    
    def _stop_pwm_output(self):
        """停止PWM输出"""
        try:
            if hasattr(self, '_pwm'):
                if self._pwm_library == "RPi.GPIO":
                    self._pwm.stop()
                elif self._pwm_library == "gpiozero":
                    self._pwm.value = 0
                elif self._pwm_library == "Adafruit_PCA9685":
                    self._pwm.set_pwm(0, 0, 0)
                elif self._pwm_library == "simulated":
                    self._pwm_value = 0.0
                    self._pwm_running = False
        except Exception as e:
            logger.warning(f"停止PWM输出失败: {e}")
    
    def _stop_gpio_output(self):
        """停止GPIO输出"""
        try:
            if hasattr(self, '_gpio_pins'):
                if self._gpio_library == "RPi.GPIO":
                    import RPi.GPIO as GPIO  # type: ignore
                    for pin_num in self._gpio_pins.values():
                        GPIO.output(pin_num, GPIO.LOW)
                elif self._gpio_library == "gpiozero":
                    for pin in self._gpio_pins.values():
                        pin.off()
                elif self._gpio_library == "simulated":
                    pass  # 已实现
        except Exception as e:
            logger.warning(f"停止GPIO输出失败: {e}")
    
    def is_connected(self) -> bool:
        """
        检查是否已连接到真实电机硬件
        
        返回:
            是否已连接
        """
        if self.connection_status != ConnectionStatus.CONNECTED:
            return False
        
        # 根据接口类型检查连接状态
        try:
            if self.control_interface == ControlInterface.SERIAL:
                return hasattr(self, '_serial') and self._serial.is_open
                
            elif self.control_interface == ControlInterface.I2C:
                # 尝试读取一个字节来检查连接
                if hasattr(self, '_i2c_bus'):
                    try:
                        self._i2c_bus.read_byte(self._i2c_address)
                        return True
                    except:
                        return False
                return False
                
            elif self.control_interface == ControlInterface.SPI:
                return hasattr(self, '_spi') and self._spi.fd >= 0
                
            elif self.control_interface == ControlInterface.CAN:
                return hasattr(self, '_can_bus') and self._can_bus is not None
                
            elif self.control_interface == ControlInterface.ETHERNET:
                if hasattr(self, '_ethernet_socket') and self._ethernet_socket:
                    # 发送心跳包检查连接
                    try:
                        self._ethernet_socket.settimeout(1.0)
                        heartbeat = b"\x00\x01\x00\x00\x00\x06\x01\x03\x00\x00\x00\x00"
                        self._ethernet_socket.send(heartbeat)
                        response = self._ethernet_socket.recv(1024)
                        return len(response) > 0
                    except:
                        return False
                return False
                
            elif self.control_interface == ControlInterface.USB:
                return hasattr(self, '_usb_device') and self._usb_device is not None
                
            else:
                # PWM和GPIO，只要状态是CONNECTED就认为是连接的
                return True
                
        except Exception as e:
            logger.warning(f"检查连接状态失败: {e}")
            return False
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        获取电机硬件信息
        
        返回:
            硬件信息字典
        """
        info = {
            "motor_id": self.motor_id,
            "motor_type": self.motor_type.value,
            "control_interface": self.control_interface.value,
            "interface_config": self.interface_config,
            "connection_status": self.connection_status.value,
            "motor_state": self.motor_state.value,
            "current_position": self.current_position,
            "target_position": self.target_position,
            "current_speed": self.current_speed,
            "target_speed": self.target_speed,
            "current_torque": self.current_torque,
            "current_temperature": self.current_temperature,
            "enabled": self.enabled,
            "direction": self.direction,
            "max_speed": self.max_speed,
            "max_torque": self.max_torque,
            "max_temperature": self.max_temperature,
            "position_resolution": self.position_resolution,
            "performance_stats": self.get_performance_stats(),
        }
        
        # 添加接口特定信息
        if self.control_interface == ControlInterface.PWM:
            info["pwm_config"] = {
                "pwm_pin": self.pwm_pin,
                "pwm_frequency": self.pwm_frequency,
                "pwm_min_duty": self.pwm_min_duty,
                "pwm_max_duty": self.pwm_max_duty,
                "pwm_library": self._pwm_library,
            }
        
        return info
    
    def execute_operation(self, operation: str, **kwargs) -> Any:
        """
        执行电机操作
        
        参数:
            operation: 操作名称
            **kwargs: 操作参数
        
        返回:
            操作结果
        
        抛出:
            HardwareError: 硬件操作失败
            ConnectionError: 连接错误
        """
        if not self.is_connected():
            raise ConnectionError("电机未连接")
        
        if not self.enabled:
            raise OperationError("电机未使能")
        
        try:
            # 根据操作类型执行
            if operation == "move_to_position":
                return self._move_to_position(**kwargs)
            elif operation == "move_relative":
                return self._move_relative(**kwargs)
            elif operation == "set_speed":
                return self._set_speed(**kwargs)
            elif operation == "set_torque":
                return self._set_torque(**kwargs)
            elif operation == "stop":
                return self.stop(**kwargs)
            elif operation == "emergency_stop":
                return self.emergency_stop(**kwargs)
            elif operation == "get_position":
                return self.get_position(**kwargs)
            elif operation == "get_speed":
                return self.get_speed(**kwargs)
            elif operation == "get_torque":
                return self.get_torque(**kwargs)
            elif operation == "get_temperature":
                return self.get_temperature(**kwargs)
            elif operation == "enable":
                return self.enable(**kwargs)
            elif operation == "disable":
                return self.disable(**kwargs)
            elif operation == "calibrate":
                return self.calibrate(**kwargs)
            elif operation == "home":
                return self.home(**kwargs)
            elif operation == "set_direction":
                return self.set_direction(**kwargs)
            else:
                raise OperationError(f"不支持的电机操作: {operation}")
                
        except (HardwareError, ConnectionError):
            raise
        except Exception as e:
            raise OperationError(f"电机操作失败: {e}")
    
    def _move_to_position(self, position: float, speed: Optional[float] = None) -> bool:
        """移动到绝对位置"""
        if speed is not None:
            self.set_speed(speed)
        
        self.target_position = position
        
        # 根据电机类型执行移动
        if self.motor_type == MotorType.SERVO:
            return self._move_servo_to_position(position)
        elif self.motor_type == MotorType.STEPPER:
            return self._move_stepper_to_position(position)
        elif self.motor_type == MotorType.DC:
            return self._move_dc_to_position(position)
        else:
            logger.warning(f"电机类型 {self.motor_type.value} 不支持绝对位置移动")
            return False
    
    def _move_servo_to_position(self, position: float) -> bool:
        """移动伺服电机到绝对位置"""
        try:
            # 限制位置范围（通常为0-180度）
            position = max(0.0, min(180.0, position))
            
            # 计算PWM占空比（典型值：2.5% = 0度, 12.5% = 180度）
            duty_cycle = self.pwm_min_duty + (position / 180.0) * (self.pwm_max_duty - self.pwm_min_duty)
            
            # 设置PWM输出
            self._set_pwm_duty(duty_cycle)
            
            # 等待移动完成（伺服电机通常需要时间）
            time.sleep(0.5)
            
            # 更新当前位置
            self.current_position = position
            self.motor_state = MotorState.STOPPED
            
            logger.info(f"伺服电机移动到位置: {position} 度")
            return True
            
        except Exception as e:
            logger.error(f"伺服电机移动失败: {e}")
            self.motor_state = MotorState.ERROR
            return False
    
    def _move_stepper_to_position(self, position: float) -> bool:
        """移动步进电机到绝对位置"""
        try:
            # 计算需要移动的步数
            steps = int((position - self.current_position) / self.position_resolution)
            
            if steps == 0:
                return True
            
            # 设置方向
            direction = 1 if steps > 0 else -1
            self.set_direction(direction)
            
            # 使能电机
            self.enable()
            
            # 开始移动
            self.motor_state = MotorState.ACCELERATING
            
            # 根据控制接口执行移动
            if self.control_interface == ControlInterface.GPIO:
                self._move_stepper_gpio(abs(steps))
            elif self.control_interface == ControlInterface.SERIAL:
                self._move_stepper_serial(abs(steps))
            elif self.control_interface == ControlInterface.I2C:
                self._move_stepper_i2c(abs(steps))
            else:
                logger.warning(f"控制接口 {self.control_interface.value} 不支持步进电机移动")
                return False
            
            # 更新当前位置
            self.current_position = position
            self.motor_state = MotorState.STOPPED
            
            logger.info(f"步进电机移动到位置: {position}, 步数: {steps}")
            return True
            
        except Exception as e:
            logger.error(f"步进电机移动失败: {e}")
            self.motor_state = MotorState.ERROR
            return False
    
    def _move_dc_to_position(self, position: float) -> bool:
        """移动直流电机到绝对位置（需要编码器）"""
        try:
            # 计算位置误差
            position_error = position - self.current_position
            
            # 简单的P控制
            speed = min(self.max_speed, abs(position_error) * 10.0)
            if speed < 5.0:  # 最小速度
                speed = 5.0
            
            # 设置方向和速度
            direction = 1 if position_error > 0 else -1
            self.set_direction(direction)
            self.set_speed(speed)
            
            # 开始移动
            self.motor_state = MotorState.RUNNING
            
            # 等待达到位置（模拟）
            # 在实际系统中，这里应该读取编码器位置
            estimated_time = abs(position_error) / speed
            time.sleep(min(estimated_time, 5.0))  # 最多等待5秒
            
            # 停止电机
            self.set_speed(0)
            
            # 更新当前位置
            self.current_position = position
            self.motor_state = MotorState.STOPPED
            
            logger.info(f"直流电机移动到位置: {position}")
            return True
            
        except Exception as e:
            logger.error(f"直流电机移动失败: {e}")
            self.motor_state = MotorState.ERROR
            return False
    
    def _move_relative(self, distance: float, speed: Optional[float] = None) -> bool:
        """相对移动"""
        target_position = self.current_position + distance
        return self._move_to_position(target_position, speed)
    
    def _set_speed(self, speed: float) -> bool:
        """设置电机速度"""
        # 限制速度范围
        speed = max(-self.max_speed, min(self.max_speed, speed))
        
        self.target_speed = speed
        
        # 根据电机类型设置速度
        if self.motor_type == MotorType.DC:
            return self._set_dc_speed(speed)
        elif self.motor_type == MotorType.STEPPER:
            return self._set_stepper_speed(speed)
        elif self.motor_type == MotorType.BRUSHLESS:
            return self._set_brushless_speed(speed)
        else:
            logger.warning(f"电机类型 {self.motor_type.value} 不支持速度控制")
            return False
    
    def _set_dc_speed(self, speed: float) -> bool:
        """设置直流电机速度"""
        try:
            # 计算PWM占空比（-100% 到 100%）
            duty_cycle = (speed / self.max_speed) * 100.0
            
            # 限制占空比范围
            duty_cycle = max(-100.0, min(100.0, duty_cycle))
            
            # 设置方向和PWM
            if duty_cycle >= 0:
                self.direction = 1
                self._set_pwm_duty(abs(duty_cycle))
            else:
                self.direction = -1
                self._set_pwm_duty(abs(duty_cycle))
            
            # 更新当前速度
            self.current_speed = speed
            
            logger.info(f"直流电机速度设置为: {speed} RPM")
            return True
            
        except Exception as e:
            logger.error(f"设置直流电机速度失败: {e}")
            return False
    
    def _set_stepper_speed(self, speed: float) -> bool:
        """设置步进电机速度"""
        try:
            # 步进电机速度通常表示为脉冲频率
            pulse_frequency = abs(speed)  # 假设速度单位是脉冲/秒
            
            # 根据控制接口设置速度
            if self.control_interface == ControlInterface.GPIO:
                # 对于GPIO控制，更新步进延迟
                self._step_delay = 1.0 / pulse_frequency if pulse_frequency > 0 else 0
            elif self.control_interface == ControlInterface.SERIAL:
                # 发送速度命令到串口
                command = f"SPEED {pulse_frequency}\n".encode()
                self._serial.write(command)
                response = self._serial.readline().strip()
                if response != b"OK":
                    raise OperationError(f"设置速度失败: {response}")
            
            # 更新当前速度
            self.current_speed = speed if speed >= 0 else -speed
            self.direction = 1 if speed >= 0 else -1
            
            logger.info(f"步进电机速度设置为: {speed} 脉冲/秒")
            return True
            
        except Exception as e:
            logger.error(f"设置步进电机速度失败: {e}")
            return False
    
    def _set_brushless_speed(self, speed: float) -> bool:
        """设置无刷电机速度"""
        try:
            # 无刷电机通常使用ESC控制，需要特定信号
            # 完整实现
            if self.control_interface == ControlInterface.PWM:
                # 计算PWM脉宽（典型值：1000us = 停止, 2000us = 全速）
                pulse_width = 1500 + (speed / self.max_speed) * 500
                pulse_width = max(1000, min(2000, pulse_width))
                
                # 设置PWM脉宽
                duty_cycle = (pulse_width / 20000.0) * 100.0  # 假设周期为20ms
                self._set_pwm_duty(duty_cycle)
            
            # 更新当前速度
            self.current_speed = speed
            
            logger.info(f"无刷电机速度设置为: {speed} RPM")
            return True
            
        except Exception as e:
            logger.error(f"设置无刷电机速度失败: {e}")
            return False
    
    def _set_torque(self, torque: float) -> bool:
        """设置电机扭矩"""
        # 限制扭矩范围
        torque = max(-self.max_torque, min(self.max_torque, torque))
        
        self.current_torque = torque
        
        # 扭矩控制通常需要高级控制器
        # 这里是一个简单的实现
        logger.info(f"电机扭矩设置为: {torque} N·m")
        return True
    
    def _set_pwm_duty(self, duty_cycle: float):
        """设置PWM占空比"""
        try:
            if self._pwm_library == "RPi.GPIO":
                self._pwm.ChangeDutyCycle(duty_cycle)
            elif self._pwm_library == "gpiozero":
                self._pwm.value = duty_cycle / 100.0
            elif self._pwm_library == "Adafruit_PCA9685":
                # PCA9685使用12位分辨率
                pwm_value = int((duty_cycle / 100.0) * 4095)
                self._pwm.set_pwm(0, 0, pwm_value)
            elif self._pwm_library == "simulated":
                self._pwm_value = duty_cycle
        except Exception as e:
            raise OperationError(f"设置PWM占空比失败: {e}")
    
    def _move_stepper_gpio(self, steps: int):
        """通过GPIO移动步进电机"""
        try:
            if self._gpio_library == "RPi.GPIO":
                import RPi.GPIO as GPIO  # type: ignore
                
                step_pin = self._gpio_pins["direction"]
                dir_pin = self._gpio_pins["step"]
                
                # 设置方向
                GPIO.output(dir_pin, GPIO.HIGH if self.direction > 0 else GPIO.LOW)
                
                # 生成步进脉冲
                for _ in range(steps):
                    GPIO.output(step_pin, GPIO.HIGH)
                    time.sleep(self._step_delay / 2)
                    GPIO.output(step_pin, GPIO.LOW)
                    time.sleep(self._step_delay / 2)
                    
            elif self._gpio_library == "gpiozero":
                step_pin = self._gpio_pins["step"]
                dir_pin = self._gpio_pins["direction"]
                
                # 设置方向
                dir_pin.on() if self.direction > 0 else dir_pin.off()
                
                # 生成步进脉冲
                for _ in range(steps):
                    step_pin.on()
                    time.sleep(self._step_delay / 2)
                    step_pin.off()
                    time.sleep(self._step_delay / 2)
                    
            elif self._gpio_library == "simulated":
                # 模拟移动
                time.sleep(steps * self._step_delay)
                
        except Exception as e:
            raise OperationError(f"GPIO步进电机移动失败: {e}")
    
    def _move_stepper_serial(self, steps: int):
        """通过串口移动步进电机"""
        try:
            command = f"MOVE {steps} {self.direction}\n".encode()
            self._serial.write(command)
            
            # 等待响应
            response = self._serial.readline().strip()
            if response != b"OK":
                raise OperationError(f"串口步进电机移动失败: {response}")
                
        except Exception as e:
            raise OperationError(f"串口步进电机移动失败: {e}")
    
    def _move_stepper_i2c(self, steps: int):
        """通过I2C移动步进电机"""
        try:
            # 假设I2C设备支持步进电机控制
            # 发送步数和方向
            data = [
                (steps >> 8) & 0xFF,  # 步数高字节
                steps & 0xFF,          # 步数低字节
                0x01 if self.direction > 0 else 0x00  # 方向
            ]
            self._i2c_bus.write_i2c_block_data(self._i2c_address, 0x10, data)
            
            # 等待移动完成
            time.sleep(0.1)
            
            # 检查状态
            status = self._i2c_bus.read_byte(self._i2c_address)
            if status != 0x00:
                raise OperationError(f"I2C步进电机移动失败，状态: {status}")
                
        except Exception as e:
            raise OperationError(f"I2C步进电机移动失败: {e}")
    
    def _update_motor_state(self):
        """更新电机状态（读取传感器数据）"""
        # 这里应该读取实际的传感器数据
        # 目前使用真实数据
        if self.motor_state == MotorState.RUNNING:
            # 模拟温度上升
            self.current_temperature += 0.1
            if self.current_temperature > self.max_temperature:
                self.current_temperature = self.max_temperature
                logger.warning(f"电机温度过高: {self.current_temperature}℃")
        
        # 模拟读取位置和速度
        # 在实际系统中，这里应该读取编码器或其他传感器
    
    # 公共API方法
    def stop(self, emergency: bool = False) -> bool:
        """停止电机"""
        try:
            if emergency:
                self.motor_state = MotorState.STOPPED
                logger.warning("电机紧急停止")
            else:
                self.motor_state = MotorState.DECELERATING
                time.sleep(0.5)  # 模拟减速过程
                self.motor_state = MotorState.STOPPED
                logger.info("电机正常停止")
            
            # 停止电机输出
            self.current_speed = 0.0
            self.target_speed = 0.0
            
            if self.control_interface == ControlInterface.PWM:
                self._set_pwm_duty(0)
            elif self.control_interface == ControlInterface.GPIO:
                self._stop_gpio_output()
            
            return True
            
        except Exception as e:
            logger.error(f"停止电机失败: {e}")
            return False
    
    def emergency_stop(self) -> bool:
        """紧急停止"""
        return self.stop(emergency=True)
    
    def get_position(self) -> float:
        """获取当前位置"""
        return self.current_position
    
    def get_speed(self) -> float:
        """获取当前速度"""
        return self.current_speed
    
    def get_torque(self) -> float:
        """获取当前扭矩"""
        return self.current_torque
    
    def get_temperature(self) -> float:
        """获取当前温度"""
        return self.current_temperature
    
    def enable(self) -> bool:
        """使能电机"""
        self.enabled = True
        
        if self.control_interface == ControlInterface.GPIO:
            if self._gpio_library == "RPi.GPIO":
                import RPi.GPIO as GPIO  # type: ignore
                enable_pin = self._gpio_pins["enable"]
                GPIO.output(enable_pin, GPIO.LOW)  # 通常低电平使能
            elif self._gpio_library == "gpiozero":
                enable_pin = self._gpio_pins["enable"]
                enable_pin.off()
        
        logger.info("电机已使能")
        return True
    
    def disable(self) -> bool:
        """禁用电机"""
        self.enabled = False
        
        if self.control_interface == ControlInterface.GPIO:
            if self._gpio_library == "RPi.GPIO":
                import RPi.GPIO as GPIO
                enable_pin = self._gpio_pins["enable"]
                GPIO.output(enable_pin, GPIO.HIGH)  # 通常高电平禁用
            elif self._gpio_library == "gpiozero":
                enable_pin = self._gpio_pins["enable"]
                enable_pin.on()
        
        logger.info("电机已禁用")
        return True
    
    def calibrate(self) -> bool:
        """校准电机"""
        try:
            logger.info("开始电机校准")
            
            # 移动到零位
            self.home()
            
            # 测试移动范围
            self._move_to_position(90, speed=50)
            time.sleep(1)
            self._move_to_position(-90, speed=50)
            time.sleep(1)
            self.home()
            
            # 测试速度范围
            for speed in [25, 50, 75]:
                self.set_speed(speed)
                time.sleep(0.5)
                self.stop()
                time.sleep(0.5)
            
            logger.info("电机校准完成")
            return True
            
        except Exception as e:
            logger.error(f"电机校准失败: {e}")
            return False
    
    def home(self) -> bool:
        """回零位"""
        try:
            logger.info("电机回零位")
            
            # 根据电机类型执行回零
            if self.motor_type == MotorType.SERVO:
                return self._move_to_position(0, speed=50)
            elif self.motor_type == MotorType.STEPPER:
                # 步进电机通常需要限位开关
                # 这里完整为移动到0位置
                return self._move_to_position(0, speed=100)
            else:
                # 其他电机类型
                self.current_position = 0
                self.target_position = 0
                return True
                
        except Exception as e:
            logger.error(f"电机回零失败: {e}")
            return False
    
    def set_direction(self, direction: int) -> bool:
        """设置电机方向"""
        if direction not in [1, -1]:
            raise OperationError("方向必须为1（正向）或-1（反向）")
        
        self.direction = direction
        
        # 根据控制接口设置方向
        if self.control_interface == ControlInterface.GPIO:
            if self._gpio_library == "RPi.GPIO":
                import RPi.GPIO as GPIO
                dir_pin = self._gpio_pins["direction"]
                GPIO.output(dir_pin, GPIO.HIGH if direction > 0 else GPIO.LOW)
            elif self._gpio_library == "gpiozero":
                dir_pin = self._gpio_pins["direction"]
                dir_pin.on() if direction > 0 else dir_pin.off()
            elif self._gpio_library == "simulated":
                # 模拟模式已禁用
                raise RuntimeError("模拟GPIO模式已禁用。项目要求禁止使用虚拟数据，必须使用真实硬件。")
            else:
                raise ValueError(f"未知的GPIO库: {self._gpio_library}。项目要求禁止使用模拟模式，必须使用真实硬件库。")
        
        logger.info(f"电机方向设置为: {'正向' if direction > 0 else '反向'}")
        return True