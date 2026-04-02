#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实硬件管理器

扩展硬件管理器以支持真实硬件接口
集成RealHardwareInterface及其子类
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Type, Union
from enum import Enum

from .base_interface import RealHardwareInterface, HardwareType, ConnectionStatus
from .motor_controller import RealMotorController, MotorType, ControlInterface
from .sensor_interface import RealSensorInterface, SensorType, SensorInterface
from ..hardware_manager import HardwareManager, HardwareDevice, DeviceType, DeviceStatus

logger = logging.getLogger(__name__)


class RealHardwareManager:
    """真实硬件管理器
    
    管理真实硬件接口实例，提供统一的硬件访问接口
    与现有的HardwareManager协同工作
    """
    
    def __init__(self, hardware_manager: Optional[HardwareManager] = None):
        """
        初始化真实硬件管理器
        
        参数:
            hardware_manager: 现有的硬件管理器实例
        """
        self.logger = logger
        
        # 引用现有的硬件管理器
        self.hardware_manager = hardware_manager
        
        # 真实硬件接口注册表
        self.real_interfaces: Dict[str, RealHardwareInterface] = {}
        
        # 设备ID到接口ID的映射
        self.device_to_interface: Dict[str, str] = {}
        
        # 接口ID到设备ID的映射
        self.interface_to_device: Dict[str, str] = {}
        
        # 接口类型注册表
        self.interface_types: Dict[HardwareType, Type[RealHardwareInterface]] = {}
        
        # 初始化接口类型注册表
        self._register_interface_types()
        
        # 自动连接线程
        self.auto_connect_thread = None
        self.running = False
        
        self.logger.info("真实硬件管理器初始化完成")
    
    def _register_interface_types(self):
        """注册接口类型"""
        self.interface_types = {
            HardwareType.MOTOR: RealMotorController,
            # 稍后添加其他类型
            HardwareType.SENSOR: RealSensorInterface,
            # HardwareType.ROBOT: NAOqiRealController,
            # HardwareType.ACTUATOR: RealActuatorController,
            # HardwareType.CAMERA: RealCameraInterface,
        }
        
        self.logger.debug(f"注册了 {len(self.interface_types)} 种接口类型")
    
    def register_interface_type(self, hardware_type: HardwareType, 
                               interface_class: Type[RealHardwareInterface]):
        """注册新的接口类型"""
        if not issubclass(interface_class, RealHardwareInterface):
            raise TypeError(f"接口类必须继承自RealHardwareInterface")
        
        self.interface_types[hardware_type] = interface_class
        self.logger.info(f"注册接口类型: {hardware_type.value} -> {interface_class.__name__}")
    
    def create_interface_from_device(self, device: HardwareDevice) -> Optional[RealHardwareInterface]:
        """
        从硬件设备创建真实硬件接口
        
        参数:
            device: 硬件设备
            
        返回:
            真实硬件接口实例或None
        """
        try:
            # 根据设备类型确定硬件类型
            hardware_type = self._map_device_type_to_hardware_type(device.device_type)
            if hardware_type not in self.interface_types:
                self.logger.warning(f"不支持此设备类型的接口: {device.device_type.value}")
                return None  # 返回None
            
            # 获取接口类
            interface_class = self.interface_types[hardware_type]
            
            # 根据设备类型创建接口配置
            interface_config = self._create_interface_config(device, hardware_type)
            
            # 创建接口实例
            interface = self._create_interface_instance(
                interface_class, device, hardware_type, interface_config
            )
            
            if interface:
                # 注册接口
                self._register_interface(interface, device.device_id)
                self.logger.info(f"为设备创建接口成功: {device.name} ({device.device_id})")
            
            return interface
            
        except Exception as e:
            self.logger.error(f"创建硬件接口失败: {e}")
            return None  # 返回None
    
    def _map_device_type_to_hardware_type(self, device_type: DeviceType) -> HardwareType:
        """映射设备类型到硬件类型"""
        mapping = {
            DeviceType.MOTOR: HardwareType.MOTOR,
            DeviceType.SENSOR: HardwareType.SENSOR,
            DeviceType.CAMERA: HardwareType.CAMERA,
            DeviceType.ACTUATOR: HardwareType.ACTUATOR,
            DeviceType.MICROPHONE: HardwareType.MICROPHONE,
            DeviceType.SPEAKER: HardwareType.SPEAKER,
            DeviceType.DISPLAY: HardwareType.DISPLAY,
            DeviceType.INPUT_DEVICE: HardwareType.INPUT_DEVICE,
            DeviceType.NETWORK: HardwareType.NETWORK,
            DeviceType.STORAGE: HardwareType.STORAGE,
            DeviceType.UNKNOWN: HardwareType.UNKNOWN,
        }
        
        return mapping.get(device_type, HardwareType.UNKNOWN)
    
    def _create_interface_config(self, device: HardwareDevice, 
                                hardware_type: HardwareType) -> Dict[str, Any]:
        """创建接口配置"""
        config = {
            "device_id": device.device_id,
            "device_name": device.name,
            "device_type": device.device_type.value,
            "connection_type": device.connection_type,
            **device.connection_params,
        }
        
        # 根据硬件类型添加特定配置
        if hardware_type == HardwareType.MOTOR:
            # 电机特定配置
            config.update({
                "motor_type": self._detect_motor_type(device),
                "control_interface": self._detect_control_interface(device),
                "max_speed": 1000.0,
                "max_torque": 10.0,
                "position_resolution": 0.01,
            })
        
        # 从设备属性中合并配置
        if device.properties:
            config.update(device.properties)
        
        return config
    
    def _detect_motor_type(self, device: HardwareDevice) -> MotorType:
        """检测电机类型"""
        description = device.description.lower()
        
        if any(word in description for word in ["servo", "伺服"]):
            return MotorType.SERVO
        elif any(word in description for word in ["stepper", "步进"]):
            return MotorType.STEPPER
        elif any(word in description for word in ["brushless", "无刷"]):
            return MotorType.BRUSHLESS
        elif any(word in description for word in ["dc", "直流"]):
            return MotorType.DC
        elif any(word in description for word in ["linear", "线性"]):
            return MotorType.LINEAR
        else:
            return MotorType.UNKNOWN
    
    def _detect_control_interface(self, device: HardwareDevice) -> ControlInterface:
        """检测控制接口"""
        connection_type = device.connection_type.lower()
        
        if connection_type == "serial":
            return ControlInterface.SERIAL
        elif connection_type == "i2c":
            return ControlInterface.I2C
        elif connection_type == "spi":
            return ControlInterface.SPI
        elif connection_type == "can":
            return ControlInterface.CAN
        elif connection_type == "network":
            return ControlInterface.ETHERNET
        elif connection_type == "usb":
            return ControlInterface.USB
        elif "pwm" in connection_type:
            return ControlInterface.PWM
        elif "gpio" in connection_type:
            return ControlInterface.GPIO
        else:
            # 默认使用串口
            return ControlInterface.SERIAL
    
    def _create_interface_instance(self, interface_class: Type[RealHardwareInterface],
                                  device: HardwareDevice, hardware_type: HardwareType,
                                  interface_config: Dict[str, Any]) -> Optional[RealHardwareInterface]:
        """创建接口实例"""
        try:
            if hardware_type == HardwareType.MOTOR:
                # 创建电机控制器
                motor_type = interface_config.get("motor_type", MotorType.UNKNOWN)
                control_interface = interface_config.get("control_interface", ControlInterface.SERIAL)
                
                # 确保motor_type和control_interface是枚举值
                if isinstance(motor_type, str):
                    motor_type = MotorType(motor_type)
                if isinstance(control_interface, str):
                    control_interface = ControlInterface(control_interface)
                
                interface = interface_class(
                    motor_id=device.device_id,
                    motor_type=motor_type,
                    control_interface=control_interface,
                    interface_config=interface_config
                )
                
                return interface
                
            else:
                # 对于其他硬件类型，使用通用构造函数
                interface = interface_class(
                    hardware_type=hardware_type,
                    interface_name=device.device_id,
                    **interface_config
                )
                
                return interface
                
        except Exception as e:
            self.logger.error(f"创建接口实例失败: {e}")
            return None  # 返回None
    
    def _register_interface(self, interface: RealHardwareInterface, device_id: str):
        """注册接口"""
        interface_id = f"{interface.hardware_type.value}_{device_id}"
        
        self.real_interfaces[interface_id] = interface
        self.device_to_interface[device_id] = interface_id
        self.interface_to_device[interface_id] = device_id
        
        self.logger.debug(f"注册接口: {interface_id} -> {device_id}")
    
    def get_interface(self, identifier: str) -> Optional[RealHardwareInterface]:
        """
        获取硬件接口
        
        参数:
            identifier: 设备ID或接口ID
            
        返回:
            硬件接口实例或None
        """
        # 首先检查是否是接口ID
        if identifier in self.real_interfaces:
            return self.real_interfaces[identifier]
        
        # 检查是否是设备ID
        if identifier in self.device_to_interface:
            interface_id = self.device_to_interface[identifier]
            return self.real_interfaces.get(interface_id)
        
        return None  # 返回None
    
    def get_interface_by_device(self, device_id: str) -> Optional[RealHardwareInterface]:
        """通过设备ID获取接口"""
        return self.get_interface(device_id)
    
    def get_all_interfaces(self) -> List[RealHardwareInterface]:
        """获取所有接口"""
        return list(self.real_interfaces.values())
    
    def get_interfaces_by_type(self, hardware_type: HardwareType) -> List[RealHardwareInterface]:
        """根据类型获取接口"""
        return [
            interface for interface in self.real_interfaces.values()
            if interface.hardware_type == hardware_type
        ]
    
    def connect_all(self) -> Dict[str, bool]:
        """连接所有接口"""
        results = {}
        
        for interface_id, interface in self.real_interfaces.items():
            try:
                if interface.is_connected():
                    results[interface_id] = True
                    self.logger.debug(f"接口已连接: {interface_id}")
                    continue
                
                success = interface.connect()
                results[interface_id] = success
                
                if success:
                    self.logger.info(f"接口连接成功: {interface_id}")
                    
                    # 更新硬件管理器中的设备状态
                    self._update_device_status_from_interface(interface)
                else:
                    self.logger.warning(f"接口连接失败: {interface_id}")
                    
            except Exception as e:
                results[interface_id] = False
                self.logger.error(f"接口连接异常: {interface_id}, 错误: {e}")
        
        return results
    
    def disconnect_all(self) -> Dict[str, bool]:
        """断开所有接口"""
        results = {}
        
        for interface_id, interface in self.real_interfaces.items():
            try:
                success = interface.disconnect()
                results[interface_id] = success
                
                if success:
                    self.logger.info(f"接口断开成功: {interface_id}")
                else:
                    self.logger.warning(f"接口断开失败: {interface_id}")
                    
            except Exception as e:
                results[interface_id] = False
                self.logger.error(f"接口断开异常: {interface_id}, 错误: {e}")
        
        return results
    
    def start_auto_connect(self, interval: float = 30.0):
        """启动自动连接"""
        if self.running:
            self.logger.warning("自动连接已运行")
            return
        
        self.running = True
        self.auto_connect_thread = threading.Thread(
            target=self._auto_connect_loop,
            args=(interval,),
            daemon=True,
            name="AutoConnect"
        )
        self.auto_connect_thread.start()
        
        self.logger.info(f"自动连接启动，间隔: {interval}秒")
    
    def stop_auto_connect(self):
        """停止自动连接"""
        if not self.running:
            self.logger.warning("自动连接未运行")
            return
        
        self.running = False
        
        if self.auto_connect_thread and self.auto_connect_thread.is_alive():
            self.auto_connect_thread.join(timeout=2.0)
        
        self.logger.info("自动连接停止")
    
    def _auto_connect_loop(self, interval: float):
        """自动连接循环"""
        self.logger.info("自动连接循环启动")
        
        while self.running:
            try:
                # 检查所有接口的连接状态
                for interface_id, interface in self.real_interfaces.items():
                    if not interface.is_connected():
                        self.logger.debug(f"尝试重新连接: {interface_id}")
                        
                        try:
                            success = interface.connect()
                            if success:
                                self.logger.info(f"重新连接成功: {interface_id}")
                                self._update_device_status_from_interface(interface)
                        except Exception as e:
                            self.logger.debug(f"重新连接失败: {interface_id}, 错误: {e}")
                
                # 等待下一次检查
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"自动连接循环异常: {e}")
                time.sleep(1.0)
        
        self.logger.info("自动连接循环停止")
    
    def _update_device_status_from_interface(self, interface: RealHardwareInterface):
        """根据接口状态更新设备状态"""
        if not self.hardware_manager:
            return
        
        # 获取对应的设备ID
        interface_id = f"{interface.hardware_type.value}_{interface.interface_name}"
        device_id = self.interface_to_device.get(interface_id)
        
        if not device_id:
            return
        
        # 获取设备
        device = self.hardware_manager.get_device(device_id)
        if not device:
            return
        
        # 根据接口状态更新设备状态
        if interface.is_connected():
            new_status = DeviceStatus.ONLINE
            error_message = ""
        else:
            new_status = DeviceStatus.OFFLINE
            error_message = "硬件接口断开连接"
        
        # 更新设备状态
        self.hardware_manager.update_device_status(
            device_id, new_status, error_message
        )
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_info = {
            "total_interfaces": len(self.real_interfaces),
            "connected_interfaces": 0,
            "disconnected_interfaces": 0,
            "interface_details": {},
            "errors": [],
        }
        
        for interface_id, interface in self.real_interfaces.items():
            try:
                is_connected = interface.is_connected()
                health_stats = interface.get_performance_stats()
                
                if is_connected:
                    health_info["connected_interfaces"] += 1
                else:
                    health_info["disconnected_interfaces"] += 1
                
                health_info["interface_details"][interface_id] = {
                    "connected": is_connected,
                    "hardware_type": interface.hardware_type.value,
                    "interface_name": interface.interface_name,
                    "performance_stats": health_stats,
                    "health_check": interface.health_check(),
                }
                
            except Exception as e:
                error_msg = f"接口健康检查失败 {interface_id}: {e}"
                health_info["errors"].append(error_msg)
                self.logger.error(error_msg)
        
        return health_info
    
    def execute_operation(self, device_id: str, operation: str, **kwargs) -> Any:
        """
        执行硬件操作
        
        参数:
            device_id: 设备ID
            operation: 操作名称
            **kwargs: 操作参数
            
        返回:
            操作结果
        """
        interface = self.get_interface(device_id)
        if not interface:
            raise ValueError(f"未找到设备对应的硬件接口: {device_id}")
        
        return interface.safe_execute(operation, **kwargs)
    
    def cleanup(self):
        """清理资源"""
        self.stop_auto_connect()
        self.disconnect_all()
        
        # 清理所有接口
        for interface in self.real_interfaces.values():
            try:
                interface.cleanup()
            except Exception as e:
                self.logger.warning(f"清理接口失败: {e}")
        
        self.logger.info("真实硬件管理器清理完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()
    
    def __del__(self):
        """析构函数"""
        self.cleanup()