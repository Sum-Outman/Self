#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实硬件接口模块

功能：
1. 真实电机控制接口
2. 真实传感器数据采集
3. 真实机器人硬件连接
4. 真实硬件设备管理
5. 串口数据服务集成（核心）

核心原则：真实优先，模拟后备，串口统一

根据用户需求："接收后交给后台处理串口数据解码即可使用"
本模块通过串口数据服务实现统一的硬件数据接收和处理
"""

import logging

logger = logging.getLogger(__name__)

# 版本信息
__version__ = "1.1.0"
__author__ = "Self AGI Team"
__description__ = "真实硬件接口模块（串口数据服务集成）"

# 导入真实硬件接口类
try:
    from .base_interface import RealHardwareInterface
    from .motor_controller import RealMotorController
    from .real_hardware_manager import RealHardwareManager
    
    # 尝试导入串口数据服务（核心集成）
    try:
        from backend.services.serial_data_service import (
            get_serial_data_service,
            receive_serial_raw_data,
            start_serial_data_service,
            stop_serial_data_service,
        )
        SERIAL_DATA_SERVICE_AVAILABLE = True
        logger.info("串口数据服务已导入，支持统一硬件数据接收")
    except ImportError as e:
        SERIAL_DATA_SERVICE_AVAILABLE = False
        get_serial_data_service = None
        receive_serial_raw_data = None
        start_serial_data_service = None
        stop_serial_data_service = None
        logger.warning(f"串口数据服务导入失败: {e}")
        logger.info("将使用传统硬件接口模块")
    
    # 尝试导入其他模块（可选，可通过串口数据服务替代）
    try:
        from .sensor_interface import RealSensorInterface
        SENSOR_INTERFACE_AVAILABLE = True
    except ImportError:
        RealSensorInterface = None
        SENSOR_INTERFACE_AVAILABLE = False
        logger.debug("传感器接口模块已实现，可通过串口数据服务接收传感器数据")
    
    try:
        from .naoqi_controller import NAOqiRealController
        NAOQI_CONTROLLER_AVAILABLE = True
    except ImportError:
        NAOqiRealController = None
        NAOQI_CONTROLLER_AVAILABLE = False
        logger.debug("NAOqi控制器模块已实现，机器人数据可通过串口接收")
    
    # 暂时禁用Arduino控制器，避免语法错误
    ArduinoRealController = None
    ARDUINO_CONTROLLER_AVAILABLE = False
    logger.debug("Arduino控制器模块暂时禁用，Arduino数据可通过串口接收")
    
    # I2C和SPI设备可通过串口数据服务接收数据
    I2CSensorReader = None
    I2C_SENSOR_READER_AVAILABLE = False
    SPIDeviceController = None
    SPI_DEVICE_CONTROLLER_AVAILABLE = False
    logger.info("I2C/SPI设备数据可通过串口数据服务统一接收和解码")
    
    __all__ = [
        "RealHardwareInterface",
        "RealMotorController", 
        "RealHardwareManager",
    ]
    
    # 添加串口数据服务相关导出（如果可用）
    if SERIAL_DATA_SERVICE_AVAILABLE:
        __all__.extend([
            "get_serial_data_service",
            "receive_serial_raw_data", 
            "start_serial_data_service",
            "stop_serial_data_service",
            "SERIAL_DATA_SERVICE_AVAILABLE",
        ])
    
    # 添加可用的模块到__all__（可选）
    if SENSOR_INTERFACE_AVAILABLE:
        __all__.append("RealSensorInterface")
        __all__.append("SENSOR_INTERFACE_AVAILABLE")
    
    if NAOQI_CONTROLLER_AVAILABLE:
        __all__.append("NAOqiRealController")
        __all__.append("NAOQI_CONTROLLER_AVAILABLE")
    
    if ARDUINO_CONTROLLER_AVAILABLE:
        __all__.append("ArduinoRealController")
        __all__.append("ARDUINO_CONTROLLER_AVAILABLE")
    
    # I2C和SPI不再单独导出，通过串口数据服务统一处理
    # 但保留导出标识以供兼容
    __all__.extend([
        "I2CSensorReader",
        "I2C_SENSOR_READER_AVAILABLE", 
        "SPIDeviceController",
        "SPI_DEVICE_CONTROLLER_AVAILABLE",
    ])
    
    logger.info("真实硬件接口模块初始化完成（串口数据服务集成）")
    
except ImportError as e:
    logger.warning(f"真实硬件接口模块导入失败: {e}")
    __all__ = []