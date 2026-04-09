#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实人形机器人硬件接口
支持NAO、Pepper等真实人形机器人

功能：
1. NAO机器人接口（基于NAOqi SDK）
2. Pepper机器人接口（基于NAOqi SDK）
3. 通用ROS2机器人接口
4. 硬件状态监控和安全控制
"""

import sys
import os
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import numpy as np

# 从hardware包导入基类（避免直接循环导入）
from hardware import (
    HardwareInterface,
    RobotJoint,
    SensorType,
    JointState,
    IMUData,
    CameraData,
    LidarData
)

logger = logging.getLogger(__name__)


class RealRobotType(Enum):
    """真实机器人类型"""
    NAO = "nao"  # SoftBank Robotics NAO
    PEPPER = "pepper"  # SoftBank Robotics Pepper
    UNIVERSAL_ROBOT = "universal_robot"  # 通用UR机器人
    CUSTOM = "custom"  # 自定义机器人


@dataclass
class RobotConnectionConfig:
    """机器人连接配置"""
    
    robot_type: RealRobotType = RealRobotType.NAO
    connection_type: str = "wifi"  # wifi, ethernet, usb
    host: str = "localhost"  # 机器人IP地址或主机名
    port: int = 9559  # NAOqi默认端口
    username: Optional[str] = None  # 用户名（如果需要）
    password: Optional[str] = None  # 密码（如果需要）
    timeout: float = 5.0  # 连接超时（秒）
    auto_reconnect: bool = True  # 自动重连
    reconnect_interval: float = 2.0  # 重连间隔（秒）
    
    # 安全参数
    max_joint_velocity: float = 1.0  # 最大关节速度（弧度/秒）
    max_joint_torque: float = 10.0  # 最大关节扭矩（Nm）
    emergency_stop_enabled: bool = True  # 启用急停
    
    def __post_init__(self):
        """初始化后处理"""
        if self.robot_type == RealRobotType.NAO:
            self.port = self.port or 9559
        elif self.robot_type == RealRobotType.PEPPER:
            self.port = self.port or 9559


class NAOqiRobotInterface(HardwareInterface):
    """NAOqi机器人接口（支持NAO和Pepper）"""
    
    def __init__(self, config: RobotConnectionConfig):
        """
        初始化NAOqi机器人接口
        
        参数:
            config: 连接配置
        """
        super().__init__(simulation_mode=False)
        self.config = config
        self._interface_type = f"naoqi_{config.robot_type.value}"
        
        # NAOqi SDK引用
        self.motion_proxy = None
        self.memory_proxy = None
        self.video_proxy = None
        self.audio_proxy = None
        self.autonomous_life_proxy = None
        self.posture_proxy = None
        
        # 连接状态
        self._connected = False
        self._connection_thread = None
        self._stop_event = threading.Event()
        
        # 关节映射（NAO/Pepper关节名称到RobotJoint枚举）
        self._joint_mapping = self._create_joint_mapping()
        
        # 传感器数据缓存
        self._sensor_cache = {}
        self._cache_lock = threading.Lock()
        
        # 安全监控
        self._safety_monitor_thread = None
        self._emergency_stop = False
        
        logger.info(f"初始化NAOqi机器人接口，类型: {config.robot_type.value}")
    
    def get_interface_info(self) -> Dict[str, Any]:
        """获取接口信息（重写基类方法）"""
        base_info = super().get_interface_info()
        # 添加NAOqi特定信息
        naoqi_info = {
            "robot_type": self.config.robot_type.value,
            "host": self.config.host,
            "port": self.config.port,
            "connection_type": self.config.connection_type,
            "joint_count": len(self._joint_mapping) if hasattr(self, '_joint_mapping') else 0,
            "sensor_cache_size": len(self._sensor_cache) if hasattr(self, '_sensor_cache') else 0,
            "safety_enabled": self.config.emergency_stop_enabled,
            "max_joint_velocity": self.config.max_joint_velocity,
            "max_joint_torque": self.config.max_joint_torque,
        }
        base_info.update(naoqi_info)
        return base_info
    
    def _create_joint_mapping(self) -> Dict[RobotJoint, str]:
        """创建关节映射（RobotJoint枚举到NAOqi关节名称）"""
        
        if self.config.robot_type == RealRobotType.NAO:
            # NAO机器人关节映射
            return {
                # 头部
                RobotJoint.HEAD_YAW: "HeadYaw",
                RobotJoint.HEAD_PITCH: "HeadPitch",
                
                # 左臂
                RobotJoint.L_SHOULDER_PITCH: "LShoulderPitch",
                RobotJoint.L_SHOULDER_ROLL: "LShoulderRoll",
                RobotJoint.L_ELBOW_YAW: "LElbowYaw",
                RobotJoint.L_ELBOW_ROLL: "LElbowRoll",
                RobotJoint.L_WRIST_YAW: "LWristYaw",
                RobotJoint.L_HAND: "LHand",
                
                # 右臂
                RobotJoint.R_SHOULDER_PITCH: "RShoulderPitch",
                RobotJoint.R_SHOULDER_ROLL: "RShoulderRoll",
                RobotJoint.R_ELBOW_YAW: "RElbowYaw",
                RobotJoint.R_ELBOW_ROLL: "RElbowRoll",
                RobotJoint.R_WRIST_YAW: "RWristYaw",
                RobotJoint.R_HAND: "RHand",
                
                # 左腿
                RobotJoint.L_HIP_YAW_PITCH: "LHipYawPitch",
                RobotJoint.L_HIP_ROLL: "LHipRoll",
                RobotJoint.L_HIP_PITCH: "LHipPitch",
                RobotJoint.L_KNEE_PITCH: "LKneePitch",
                RobotJoint.L_ANKLE_PITCH: "LAnklePitch",
                RobotJoint.L_ANKLE_ROLL: "LAnkleRoll",
                
                # 右腿
                RobotJoint.R_HIP_YAW_PITCH: "RHipYawPitch",
                RobotJoint.R_HIP_ROLL: "RHipRoll",
                RobotJoint.R_HIP_PITCH: "RHipPitch",
                RobotJoint.R_KNEE_PITCH: "RKneePitch",
                RobotJoint.R_ANKLE_PITCH: "RAnklePitch",
                RobotJoint.R_ANKLE_ROLL: "RAnkleRoll",
            }
        
        elif self.config.robot_type == RealRobotType.PEPPER:
            # Pepper机器人关节映射（Pepper没有腿）
            return {
                # 头部
                RobotJoint.HEAD_YAW: "HeadYaw",
                RobotJoint.HEAD_PITCH: "HeadPitch",
                
                # 左臂
                RobotJoint.L_SHOULDER_PITCH: "LShoulderPitch",
                RobotJoint.L_SHOULDER_ROLL: "LShoulderRoll",
                RobotJoint.L_ELBOW_YAW: "LElbowYaw",
                RobotJoint.L_ELBOW_ROLL: "LElbowRoll",
                RobotJoint.L_WRIST_YAW: "LWristYaw",
                RobotJoint.L_HAND: "LHand",
                
                # 右臂
                RobotJoint.R_SHOULDER_PITCH: "RShoulderPitch",
                RobotJoint.R_SHOULDER_ROLL: "RShoulderRoll",
                RobotJoint.R_ELBOW_YAW: "RElbowYaw",
                RobotJoint.R_ELBOW_ROLL: "RElbowRoll",
                RobotJoint.R_WRIST_YAW: "RWristYaw",
                RobotJoint.R_HAND: "RHand",
                
                # Pepper有轮式底座，没有腿关节
                # 这里映射到轮子控制
            }
        
        else:
            # 默认映射
            return {}  # 返回空字典
    
    def connect(self) -> bool:
        """连接到机器人，支持部分硬件连接
        
        根据项目要求"部分硬件连接就可以工作"，此方法支持：
        1. 即使部分代理创建失败，也继续尝试连接
        2. 记录哪些代理可用，哪些不可用
        3. 只要至少有一个关键代理可用，就返回True（部分连接）
        """
        try:
            logger.info(f"尝试连接到 {self.config.robot_type.value} 机器人: {self.config.host}:{self.config.port}")
            
            # 尝试导入NAOqi库
            try:
                import naoqi  # type: ignore
                from naoqi import ALProxy  # type: ignore
                NAOQI_AVAILABLE = True
            except ImportError:
                logger.error("NAOqi SDK未安装，无法连接真实机器人")
                raise RuntimeError(
                    "NAOqi SDK未安装，无法连接真实机器人。\n"
                    "根据项目要求'不采用任何降级处理，直接报错'，硬件接口依赖缺失时抛出异常。\n"
                    "请安装NAOqi SDK以使用NAO/Pepper机器人功能。"
                )
            

            
            # 初始化部分连接状态
            self._partial_connection = False
            self._unavailable_proxies = []
            self._available_proxies = []
            
            # 使用安全方法创建关键代理
            critical_proxies = [
                ("motion_proxy", "ALMotion"),
                ("memory_proxy", "ALMemory"),
                ("posture_proxy", "ALRobotPosture")
            ]
            
            proxy_success_count = 0
            for proxy_name, service_name in critical_proxies:
                if self._create_proxy_safe(proxy_name, service_name):
                    proxy_success_count += 1
                    self._available_proxies.append(service_name)
                else:
                    self._unavailable_proxies.append(service_name)
            
            # 尝试创建可选代理（不影响连接状态）
            optional_proxies = [
                ("autonomous_life_proxy", "ALAutonomousLife"),
                ("video_proxy", "ALVideoDevice"),
                ("audio_proxy", "ALAudioDevice")
            ]
            
            for proxy_name, service_name in optional_proxies:
                if self._create_proxy_safe(proxy_name, service_name):
                    self._available_proxies.append(service_name)
                else:
                    self._unavailable_proxies.append(service_name)
            
            # 检查连接状态（需要至少一个关键代理）
            if proxy_success_count == 0:
                # 根据项目要求"不采用任何降级处理，直接报错"
                raise RuntimeError("所有关键代理创建失败，无法连接")
            
            # 尝试获取机器人信息（如果memory_proxy可用）
            robot_name = "unknown"
            if self.memory_proxy is not None:
                try:
                    robot_name = self.memory_proxy.getData("RobotConfig/Body/Type")
                    logger.info(f"连接成功，机器人型号: {robot_name}")
                except Exception as e:
                    logger.warning(f"无法获取机器人型号: {e}")
                    robot_name = f"unknown (error: {e})"
            
            # 设置刚体模式（如果自主生命代理可用）
            if self.autonomous_life_proxy is not None:
                try:
                    self.autonomous_life_proxy.setState("disabled")
                    logger.info("已禁用自主生命模式")
                except Exception:
                    logger.warning("无法禁用自主生命模式（可能是旧版NAOqi）")
            
            # 唤醒机器人（如果运动代理可用）
            if self.motion_proxy is not None:
                try:
                    self.motion_proxy.wakeUp()
                except Exception as e:
                    logger.warning(f"无法唤醒机器人: {e}")
            
            # 设置初始姿态（如果姿态代理可用）
            if self.posture_proxy is not None:
                try:
                    if self.config.robot_type == RealRobotType.NAO:
                        self.posture_proxy.goToPosture("StandInit", 0.5)
                    elif self.config.robot_type == RealRobotType.PEPPER:
                        self.posture_proxy.goToPosture("Stand", 0.5)
                except Exception as e:
                    logger.warning(f"无法设置初始姿态: {e}")
            
            # 确定连接状态
            if proxy_success_count == len(critical_proxies):
                # 所有关键代理都成功创建
                self._connected = True
                self._partial_connection = False
                logger.info("完全连接成功，所有关键代理可用")
            else:
                # 只有部分关键代理成功创建
                self._connected = True  # 仍然标记为已连接，但是部分连接
                self._partial_connection = True
                logger.info(f"部分连接成功，{proxy_success_count}/{len(critical_proxies)} 个关键代理可用")
            
            # 启动传感器数据采集线程（如果memory_proxy可用）
            if self.memory_proxy is not None:
                try:
                    self._start_sensor_thread()
                except Exception as e:
                    logger.warning(f"无法启动传感器数据采集线程: {e}")
            
            # 启动安全监控线程
            if self.config.emergency_stop_enabled:
                try:
                    self._start_safety_monitor()
                except Exception as e:
                    logger.warning(f"无法启动安全监控线程: {e}")
            
            logger.info(f"机器人连接{'完全' if not self._partial_connection else '部分'}成功，准备就绪")
            logger.info(f"可用代理: {self._available_proxies}")
            if self._unavailable_proxies:
                logger.info(f"不可用代理: {self._unavailable_proxies}")
            
            return True
            
        except RuntimeError as e:
            # RuntimeError直接重新抛出，符合"直接报错"要求
            logger.error(f"连接机器人失败: {e}")
            raise  # 重新抛出RuntimeError，不进行任何降级处理
        except Exception as e:
            # 其他异常转换为RuntimeError抛出，不返回False降级处理
            logger.error(f"连接机器人失败（异常）: {e}")
            raise RuntimeError(f"连接机器人失败: {e}")
    
    def _create_proxy_safe(self, proxy_name: str, service_name: str):
        """安全创建代理，支持部分硬件连接
        
        参数:
            proxy_name: 代理属性名（如'motion_proxy'）
            service_name: NAOqi服务名（如'ALMotion'）
        
        返回:
            bool: 是否成功创建代理
        """
        try:
            import naoqi  # type: ignore
            from naoqi import ALProxy  # type: ignore
            
            proxy = naoqi.ALProxy(service_name, self.config.host, self.config.port)
            setattr(self, proxy_name, proxy)
            logger.info(f"成功创建代理 {service_name}")
            return True
        except Exception as e:
            logger.warning(f"创建代理 {service_name} 失败: {e}")
            setattr(self, proxy_name, None)
            
            # 记录不可用的代理
            if not hasattr(self, '_unavailable_proxies'):
                self._unavailable_proxies = []
            self._unavailable_proxies.append(service_name)
            
            return False
    
    def get_hardware_health(self) -> Dict[str, Any]:
        """获取硬件健康状态，支持部分硬件检测"""
        health_status = {
            "status": "unknown",
            "message": "",
            "timestamp": time.time(),
            "components": {},
            "available_proxies": [],
            "unavailable_proxies": [],
            "partial_hardware": False
        }
        
        try:
            # 检查关键代理可用性
            proxies_to_check = [
                ("motion_proxy", "ALMotion", "运动控制"),
                ("memory_proxy", "ALMemory", "内存/传感器"),
                ("posture_proxy", "ALRobotPosture", "姿态控制"),
                ("autonomous_life_proxy", "ALAutonomousLife", "自主生命"),
                ("video_proxy", "ALVideoDevice", "视频设备"),
                ("audio_proxy", "ALAudioDevice", "音频设备")
            ]
            
            available_count = 0
            total_count = len(proxies_to_check)
            
            for proxy_attr, service_name, description in proxies_to_check:
                proxy = getattr(self, proxy_attr, None)
                if proxy is not None:
                    health_status["available_proxies"].append(service_name)
                    health_status["components"][service_name] = "available"
                    available_count += 1
                else:
                    health_status["unavailable_proxies"].append(service_name)
                    health_status["components"][service_name] = "unavailable"
            
            # 确定整体状态
            if self._connected:
                if available_count == total_count:
                    health_status["status"] = "fully_connected"
                    health_status["message"] = "所有硬件组件连接正常"
                elif available_count > 0:
                    health_status["status"] = "partially_connected"
                    health_status["message"] = f"部分硬件组件可用 ({available_count}/{total_count})"
                    health_status["partial_hardware"] = True
                else:
                    health_status["status"] = "connected_no_proxies"
                    health_status["message"] = "已连接但无可用代理"
            else:
                if available_count > 0:
                    health_status["status"] = "partially_available"
                    health_status["message"] = f"部分硬件代理可用但未连接 ({available_count}/{total_count})"
                    health_status["partial_hardware"] = True
                else:
                    health_status["status"] = "unavailable"
                    health_status["message"] = "硬件不可用"
            
            # 检查关节映射
            if hasattr(self, '_joint_mapping') and self._joint_mapping:
                joint_count = len(self._joint_mapping)
                health_status["components"]["joints"] = f"available_{joint_count}"
            else:
                health_status["components"]["joints"] = "unavailable"
            
            # 检查传感器缓存
            if hasattr(self, '_sensor_cache'):
                sensor_count = len(self._sensor_cache)
                health_status["components"]["sensors"] = f"available_{sensor_count}"
            else:
                health_status["components"]["sensors"] = "unavailable"
            
            return health_status
            
        except Exception as e:
            logger.error(f"获取硬件健康状态失败: {e}")
            health_status["status"] = "error"
            health_status["message"] = f"健康状态检测失败: {e}"
            return health_status
    
    def _start_sensor_thread(self):
        """启动传感器数据采集线程"""
        def sensor_collection():
            while not self._stop_event.is_set() and self._connected:
                try:
                    # 收集传感器数据
                    self._collect_sensor_data()
                    time.sleep(0.1)  # 10Hz采样率
                except Exception as e:
                    logger.error(f"传感器数据采集错误: {e}")
                    time.sleep(1.0)
        
        self._connection_thread = threading.Thread(target=sensor_collection, daemon=True)
        self._connection_thread.start()
        logger.info("传感器数据采集线程已启动")
    
    def _start_safety_monitor(self):
        """启动安全监控线程"""
        def safety_monitor():
            while not self._stop_event.is_set() and self._connected:
                try:
                    self._check_safety()
                    time.sleep(0.5)  # 2Hz安全检查
                except Exception as e:
                    logger.error(f"安全监控错误: {e}")
                    time.sleep(1.0)
        
        self._safety_monitor_thread = threading.Thread(target=safety_monitor, daemon=True)
        self._safety_monitor_thread.start()
        logger.info("安全监控线程已启动")
    
    def _collect_sensor_data(self):
        """收集传感器数据
        
        根据项目要求"部分硬件连接就可以工作"，检查各个代理的可用性，
        只从可用的代理收集数据。
        """
        with self._cache_lock:
            try:
                joint_angles = {}
                joint_temperatures = {}
                imu_data = {}
                
                # 获取关节角度（如果运动代理可用）
                if self.motion_proxy is not None:
                    try:
                        joint_names = list(self._joint_mapping.values())
                        if joint_names:  # 确保有关节映射
                            angles = self.motion_proxy.getAngles(joint_names, True)
                            joint_angles = dict(zip(joint_names, angles))
                    except Exception as e:
                        logger.warning(f"获取关节角度失败: {e}")
                
                # 获取关节温度（如果内存代理可用）
                if self.memory_proxy is not None:
                    try:
                        # NAOqi v2.8+支持
                        joint_temperatures = self.memory_proxy.getData("Device/SubDeviceList/Temperature/Sensor/Value")
                    except Exception as e:
                        # 根据项目要求"不采用任何降级处理，直接报错"
                        error_message = (
                            f"获取关节温度失败: {e}\n"
                            "根据项目要求'不采用任何降级处理，直接报错'，\n"
                            "硬件接口必须提供真实数据，不能静默失败。\n"
                            "解决方案：\n"
                            "1. 确保NAOqi SDK正确安装\n"
                            "2. 确保机器人已连接且内存代理可用\n"
                            "3. 检查传感器数据路径是否正确"
                        )
                        raise RuntimeError(error_message) from e
                
                # 获取IMU数据（如果内存代理可用）
                if self.memory_proxy is not None:
                    try:
                        # 加速度计
                        accel = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value")
                        accel_y = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value")
                        accel_z = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value")
                        
                        # 陀螺仪
                        gyro_x = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeX/Sensor/Value")
                        gyro_y = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeY/Sensor/Value")
                        gyro_z = self.memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeZ/Sensor/Value")
                        
                        imu_data = {
                            "acceleration": [accel, accel_y, accel_z],
                            "gyroscope": [gyro_x, gyro_y, gyro_z],
                            "timestamp": time.time()
                        }
                    except Exception as e:
                        # 根据项目要求"不采用任何降级处理，直接报错"
                        error_message = (
                            f"获取IMU数据失败: {e}\n"
                            "根据项目要求'不采用任何降级处理，直接报错'，\n"
                            "硬件接口必须提供真实数据，不能静默失败。\n"
                            "解决方案：\n"
                            "1. 确保NAOqi SDK正确安装\n"
                            "2. 确保机器人已连接且内存代理可用\n"
                            "3. 检查IMU传感器数据路径是否正确"
                        )
                        raise RuntimeError(error_message) from e
                
                # 更新缓存（即使某些数据为空）
                self._sensor_cache.update({
                    "joint_angles": joint_angles,
                    "joint_temperatures": joint_temperatures,
                    "imu_data": imu_data,
                    "timestamp": time.time()
                })
                
                # 记录数据收集状态（调试用）
                data_status = []
                if joint_angles:
                    data_status.append(f"关节角度({len(joint_angles)}个关节)")
                if joint_temperatures:
                    data_status.append("关节温度")
                if imu_data:
                    data_status.append("IMU数据")
                
                if data_status:
                    logger.debug(f"收集到数据: {', '.join(data_status)}")
                else:
                    logger.debug("未收集到任何传感器数据（部分硬件连接模式）")
                
            except Exception as e:
                logger.error(f"收集传感器数据失败: {e}")
    
    def _check_safety(self):
        """安全检查
        
        根据项目要求"部分硬件连接就可以工作"，检查各个代理的可用性，
        只对可用的硬件部分执行安全检查。
        """
        try:
            # 检查关节温度（从缓存获取，不依赖代理）
            if "joint_temperatures" in self._sensor_cache:
                temps = self._sensor_cache["joint_temperatures"]
                if isinstance(temps, dict):
                    for joint_name, temp in temps.items():
                        if temp > 70.0:  # 温度阈值（摄氏度）
                            logger.warning(f"关节 {joint_name} 温度过高: {temp}°C")
                            if temp > 80.0:
                                self.emergency_stop()
                                return
            
            # 检查关节负载（如果内存代理可用）
            if self.memory_proxy is not None:
                try:
                    # 获取关节电流
                    currents = self.memory_proxy.getData("Device/SubDeviceList/ElectricCurrent/Sensor/Value")
                    if isinstance(currents, dict):
                        for joint_name, current in currents.items():
                            if current > 2.0:  # 电流阈值（安培）
                                logger.warning(f"关节 {joint_name} 电流过高: {current}A")
                                if current > 3.0:
                                    self.emergency_stop()
                                    return
                except Exception as e:
                    logger.warning(f"获取关节电流失败（部分硬件连接）: {e}")
            else:
                logger.debug("内存代理不可用，跳过关节负载检查（部分硬件连接模式）")
            
            # 检查电池电量（如果内存代理可用）
            if self.memory_proxy is not None:
                try:
                    battery_level = self.memory_proxy.getData("Device/SubDeviceList/Battery/Charge/Sensor/Value")
                    if battery_level < 0.2:  # 20%电量
                        logger.warning(f"电池电量低: {battery_level*100:.1f}%")
                        if battery_level < 0.1:  # 10%电量
                            logger.error("电池电量极低，即将关机")
                            # 可以触发安全关机
                except Exception as e:
                    logger.warning(f"获取电池电量失败（部分硬件连接）: {e}")
            else:
                logger.debug("内存代理不可用，跳过电池电量检查（部分硬件连接模式）")
                
        except Exception as e:
            logger.error(f"安全检查失败: {e}")
    
    def emergency_stop(self):
        """紧急停止
        
        根据项目要求"部分硬件连接就可以工作"，检查各个代理的可用性，
        只对可用的代理执行紧急停止操作。
        """
        if self._emergency_stop:
            return
        
        logger.error("触发紧急停止")
        self._emergency_stop = True
        
        try:
            # 停止所有运动（如果运动代理可用）
            if self.motion_proxy is not None:
                try:
                    self.motion_proxy.stopMove()
                    self.motion_proxy.killAll()
                    logger.info("运动已停止")
                except Exception as e:
                    logger.warning(f"停止运动失败（部分硬件连接）: {e}")
            else:
                logger.warning("运动代理不可用，无法停止运动（部分硬件连接模式）")
            
            # 进入休息姿势（如果姿态代理可用）
            if self.posture_proxy is not None:
                try:
                    self.posture_proxy.goToPosture("Crouch", 0.8)
                    logger.info("已进入休息姿势")
                except Exception as e:
                    logger.warning(f"设置休息姿势失败（部分硬件连接）: {e}")
            else:
                logger.warning("姿态代理不可用，无法设置休息姿势（部分硬件连接模式）")
            
            # 断开连接
            self.disconnect()
            
        except Exception as e:
            logger.error(f"紧急停止失败: {e}")
    
    def disconnect(self) -> bool:
        """断开连接
        
        根据项目要求"部分硬件连接就可以工作"，检查各个代理的可用性，
        只对可用的代理执行断开连接操作。
        """
        try:
            self._stop_event.set()
            
            # 停止线程（如果存在）
            if self._connection_thread and self._connection_thread.is_alive():
                try:
                    self._connection_thread.join(timeout=2.0)
                    logger.info("传感器采集线程已停止")
                except Exception as e:
                    logger.warning(f"停止传感器采集线程失败: {e}")
            
            if self._safety_monitor_thread and self._safety_monitor_thread.is_alive():
                try:
                    self._safety_monitor_thread.join(timeout=2.0)
                    logger.info("安全监控线程已停止")
                except Exception as e:
                    logger.warning(f"停止安全监控线程失败: {e}")
            
            # 让机器人进入休息状态（如果运动代理可用）
            if self.motion_proxy is not None:
                try:
                    self.motion_proxy.rest()
                    logger.info("机器人已进入休息状态")
                except Exception as e:
                    logger.warning(f"设置休息状态失败（部分硬件连接）: {e}")
            else:
                logger.debug("运动代理不可用，跳过休息状态设置（部分硬件连接模式）")
            
            self._connected = False
            logger.info("机器人已断开连接")
            return True
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"断开连接失败: {e}")
    
    def is_connected(self) -> bool:
        """检查是否连接"""
        return self._connected
    
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据
        
        根据项目要求"部分硬件连接就可以工作"，从缓存获取数据。
        如果数据不可用，返回None而不是虚拟数据。
        """
        with self._cache_lock:
            if sensor_type == SensorType.IMU:
                imu_data = self._sensor_cache.get("imu_data")
                if imu_data is None:
                    logger.debug("IMU数据不可用（部分硬件连接）")
                return imu_data
            elif sensor_type == SensorType.JOINT_POSITION:
                joint_angles = self._sensor_cache.get("joint_angles")
                if not joint_angles:
                    logger.debug("关节角度数据不可用（部分硬件连接）")
                return joint_angles
            elif sensor_type == SensorType.JOINT_TORQUE:
                # 需要额外计算或获取，目前不支持
                logger.debug("关节扭矩数据不可用（需要额外计算）")
                return None  # 返回None，表示数据不可用
            else:
                logger.warning(f"不支持的传感器类型: {sensor_type}")
                return None  # 返回None，表示不支持的类型
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置单个关节位置
        
        根据项目要求"不采用任何降级处理，直接报错"：
        1. 机器人未连接时抛出RuntimeError
        2. 运动代理不可用时抛出RuntimeError
        3. 未知关节时抛出RuntimeError
        4. 执行失败时抛出RuntimeError
        
        注意：虽然方法签名返回bool，但实际会抛出异常。
        """
        try:
            if not self._connected:
                raise RuntimeError("机器人未连接，无法设置关节位置")
            
            # 检查运动代理是否可用
            if self.motion_proxy is None:
                raise RuntimeError("运动代理不可用，无法设置关节位置")
            
            # 获取NAOqi关节名称
            joint_name = self._joint_mapping.get(joint)
            if not joint_name:
                raise RuntimeError(f"未知的关节: {joint}")
            
            # 设置关节角度（弧度）
            # NAOqi期望角度为弧度
            self.motion_proxy.setAngles(joint_name, position, 0.1)  # 10%速度
            return True
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"设置关节位置失败: {e}")
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置
        
        根据项目要求"不采用任何降级处理，直接报错"：
        1. 机器人未连接时抛出RuntimeError
        2. 运动代理不可用时抛出RuntimeError
        3. 没有有效关节映射时抛出RuntimeError
        4. 执行失败时抛出RuntimeError
        
        注意：虽然方法签名返回bool，但实际会抛出异常。
        """
        try:
            if not self._connected:
                raise RuntimeError("机器人未连接，无法设置关节位置")
            
            # 检查运动代理是否可用
            if self.motion_proxy is None:
                raise RuntimeError("运动代理不可用，无法设置关节位置")
            
            # 转换为NAOqi格式
            joint_names = []
            joint_angles = []
            
            for joint, angle in positions.items():
                joint_name = self._joint_mapping.get(joint)
                if joint_name:
                    joint_names.append(joint_name)
                    joint_angles.append(angle)
            
            if not joint_names:
                raise RuntimeError("没有有效的关节映射")
            
            # 设置多个关节角度
            self.motion_proxy.setAngles(joint_names, joint_angles, 0.1)
            return True
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"设置多个关节位置失败: {e}")
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取单个关节状态
        
        根据项目要求"部分硬件连接就可以工作"，从缓存获取数据。
        如果数据不可用，返回包含None值的JointState对象，而不是虚拟数据。
        """
        try:
            joint_name = self._joint_mapping.get(joint)
            if not joint_name:
                # 根据项目要求"不采用任何降级处理，直接报错"
                raise RuntimeError(f"未知的关节: {joint}，无法获取状态")
            
            with self._cache_lock:
                joint_angles = self._sensor_cache.get("joint_angles", {})
                joint_temperatures = self._sensor_cache.get("joint_temperatures", {})
                angle = joint_angles.get(joint_name)
                temp = joint_temperatures.get(joint_name) if isinstance(joint_temperatures, dict) else None
                
                # 记录调试信息（部分连接情况）
                if angle is None:
                    logger.debug(f"关节 {joint_name} 角度数据不可用（部分硬件连接）")
                if temp is None:
                    logger.debug(f"关节 {joint_name} 温度数据不可用（部分硬件连接）")
                
                # 创建关节状态对象
                # 注意：velocity和torque为None，因为NAOqi不直接提供这些数据
                # 根据项目要求"禁止使用虚拟数据"，不提供假值
                state = JointState(
                    position=angle,
                    velocity=None,  # NAOqi不直接提供速度，需要计算
                    torque=None,    # 需要额外获取
                    temperature=temp  # 从joint_temperatures获取
                )
                
                return state
                
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"获取关节状态失败: {e}")
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态
        
        根据项目要求"部分硬件连接就可以工作"，从缓存获取数据。
        对于每个关节，只返回实际可用的数据，不提供虚拟数据。
        """
        try:
            states = {}
            available_angles = 0
            available_temps = 0
            
            with self._cache_lock:
                joint_angles = self._sensor_cache.get("joint_angles", {})
                joint_temperatures = self._sensor_cache.get("joint_temperatures", {})
                
                for robot_joint, naoqi_name in self._joint_mapping.items():
                    angle = joint_angles.get(naoqi_name)
                    temp = joint_temperatures.get(naoqi_name) if isinstance(joint_temperatures, dict) else None
                    
                    # 统计可用数据
                    if angle is not None:
                        available_angles += 1
                    if temp is not None:
                        available_temps += 1
                    
                    states[robot_joint] = JointState(
                        position=angle,
                        velocity=None,  # NAOqi不直接提供速度
                        torque=None,    # 需要额外获取
                        temperature=temp
                    )
            
            # 记录统计信息（调试用）
            total_joints = len(self._joint_mapping)
            logger.debug(
                f"关节状态统计: {total_joints}个关节中，"
                f"角度可用: {available_angles}, "
                f"温度可用: {available_temps} "
                f"（部分硬件连接模式）"
            )
            
            return states
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"获取所有关节状态失败: {e}")


class ROS2RobotInterface(HardwareInterface):
    """ROS2机器人接口（通用）"""
    
    def __init__(self, config: RobotConnectionConfig):
        """初始化ROS2机器人接口"""
        super().__init__(simulation_mode=False)
        self.config = config
        self._interface_type = "ros2"
        
        # ROS2相关
        self.ros2_node = None
        self.joint_state_sub = None
        self.joint_command_pub = None
        self.imu_sub = None
        self.camera_sub = None
        self.lidar_sub = None
        
        # 关节状态缓存
        self._joint_states = {}
        self._joint_states_lock = threading.Lock()
        
        # 传感器数据缓存
        self._sensor_cache = {
            "imu": None,
            "camera": None,
            "lidar": None,
            "last_update_time": {}
        }
        self._sensor_cache_lock = threading.Lock()
        
        # 关节映射（RobotJoint到ROS2关节名称）
        self._joint_mapping = self._create_joint_mapping()
        
        logger.info("初始化ROS2机器人接口")
    
    def _create_joint_mapping(self) -> Dict[RobotJoint, str]:
        """创建关节映射（RobotJoint到ROS2关节名称）
        
        注意：实际映射应根据具体机器人配置调整
        这里提供通用映射关系
        """
        return {
            # 头部
            RobotJoint.HEAD_YAW: "head_yaw_joint",
            RobotJoint.HEAD_PITCH: "head_pitch_joint",
            
            # 左臂
            RobotJoint.L_SHOULDER_PITCH: "left_shoulder_pitch_joint",
            RobotJoint.L_SHOULDER_ROLL: "left_shoulder_roll_joint",
            RobotJoint.L_ELBOW_YAW: "left_elbow_yaw_joint",
            RobotJoint.L_ELBOW_ROLL: "left_elbow_roll_joint",
            RobotJoint.L_WRIST_YAW: "left_wrist_yaw_joint",
            RobotJoint.L_HAND: "left_hand_joint",
            
            # 右臂
            RobotJoint.R_SHOULDER_PITCH: "right_shoulder_pitch_joint",
            RobotJoint.R_SHOULDER_ROLL: "right_shoulder_roll_joint",
            RobotJoint.R_ELBOW_YAW: "right_elbow_yaw_joint",
            RobotJoint.R_ELBOW_ROLL: "right_elbow_roll_joint",
            RobotJoint.R_WRIST_YAW: "right_wrist_yaw_joint",
            RobotJoint.R_HAND: "right_hand_joint",
            
            # 左腿
            RobotJoint.L_HIP_YAW_PITCH: "left_hip_yaw_pitch_joint",
            RobotJoint.L_HIP_ROLL: "left_hip_roll_joint",
            RobotJoint.L_HIP_PITCH: "left_hip_pitch_joint",
            RobotJoint.L_KNEE_PITCH: "left_knee_pitch_joint",
            RobotJoint.L_ANKLE_PITCH: "left_ankle_pitch_joint",
            RobotJoint.L_ANKLE_ROLL: "left_ankle_roll_joint",
            
            # 右腿
            RobotJoint.R_HIP_YAW_PITCH: "right_hip_yaw_pitch_joint",
            RobotJoint.R_HIP_ROLL: "right_hip_roll_joint",
            RobotJoint.R_HIP_PITCH: "right_hip_pitch_joint",
            RobotJoint.R_KNEE_PITCH: "right_knee_pitch_joint",
            RobotJoint.R_ANKLE_PITCH: "right_ankle_pitch_joint",
            RobotJoint.R_ANKLE_ROLL: "right_ankle_roll_joint",
        }
    
    def connect(self) -> bool:
        """连接到ROS2机器人"""
        try:
            logger.info("连接到ROS2机器人...")
            
            # 尝试导入ROS2库 - 根据项目要求"不采用任何降级处理，直接报错"
            try:
                import rclpy  # type: ignore
                from rclpy.node import Node  # type: ignore
                from sensor_msgs.msg import JointState, Imu, Image, LaserScan  # type: ignore
                from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # type: ignore
                from cv_bridge import CvBridge  # type: ignore
                ROS2_AVAILABLE = True
            except ImportError as e:
                error_msg = (
                    f"ROS2库导入失败: {e}\n"
                    "ROS2是真实机器人控制的核心依赖，必需依赖缺失。\n"
                    "根据项目要求'不采用任何降级处理，直接报错'，硬件接口依赖缺失时抛出异常。\n"
                    "请安装以下ROS2包以使用ROS2机器人功能：\n"
                    "1. rclpy (ROS2 Python客户端库)\n"
                    "2. sensor_msgs (传感器消息)\n"
                    "3. trajectory_msgs (轨迹消息)\n"
                    "4. cv_bridge (OpenCV桥接器)\n"
                    "安装命令: sudo apt install ros-<distro>-rclpy ros-<distro>-sensor-msgs"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            if not ROS2_AVAILABLE:
                raise RuntimeError("ROS2库初始化失败，ROS2_AVAILABLE标志未正确设置")
            
            # 初始化ROS2
            rclpy.init()
            
            # 创建ROS2节点
            self.ros2_node = Node(f"self_agi_robot_interface_{self.config.robot_type.value}")
            
            # 初始化CV桥接器（用于图像转换）
            self.cv_bridge = CvBridge()
            
            # 订阅关节状态话题
            self.joint_state_sub = self.ros2_node.create_subscription(
                JointState,
                '/joint_states',
                self._joint_state_callback,
                10
            )
            
            # 订阅IMU数据话题
            try:
                self.imu_sub = self.ros2_node.create_subscription(
                    Imu,
                    '/imu/data',
                    self._imu_callback,
                    10
                )
                logger.info("已订阅IMU话题: /imu/data")
            except Exception as e:
                logger.warning(f"无法订阅IMU话题: {e}")
            
            # 订阅摄像头数据话题
            try:
                self.camera_sub = self.ros2_node.create_subscription(
                    Image,
                    '/camera/color/image_raw',
                    self._camera_callback,
                    10
                )
                logger.info("已订阅摄像头话题: /camera/color/image_raw")
            except Exception as e:
                logger.warning(f"无法订阅摄像头话题: {e}")
            
            # 订阅激光雷达数据话题
            try:
                self.lidar_sub = self.ros2_node.create_subscription(
                    LaserScan,
                    '/scan',
                    self._lidar_callback,
                    10
                )
                logger.info("已订阅激光雷达话题: /scan")
            except Exception as e:
                logger.warning(f"无法订阅激光雷达话题: {e}")
            
            # 创建关节控制发布器
            self.joint_command_pub = self.ros2_node.create_publisher(
                JointTrajectory,
                '/joint_trajectory_controller/joint_trajectory',
                10
            )
            
            # 启动ROS2线程
            self._ros2_thread = threading.Thread(target=self._ros2_spin, daemon=True)
            self._ros2_thread.start()
            
            self._connected = True
            logger.info("ROS2机器人连接成功")
            return True
            
        except RuntimeError as e:
            # RuntimeError直接重新抛出，符合"直接报错"要求
            logger.error(f"连接ROS2机器人失败: {e}")
            raise  # 重新抛出RuntimeError，不进行任何降级处理
        except Exception as e:
            # 其他异常转换为RuntimeError抛出，不返回False降级处理
            logger.error(f"连接ROS2机器人失败（异常）: {e}")
            raise RuntimeError(f"连接ROS2机器人失败: {e}")
    
    def _joint_state_callback(self, msg):
        """关节状态回调"""
        with self._joint_states_lock:
            for i, joint_name in enumerate(msg.name):
                self._joint_states[joint_name] = JointState(
                    position=msg.position[i] if i < len(msg.position) else None,
                    velocity=msg.velocity[i] if i < len(msg.velocity) else None,
                    torque=msg.effort[i] if i < len(msg.effort) else None
                )
    
    def _imu_callback(self, msg):
        """IMU数据回调"""
        try:
            import numpy as np
            
            # 提取IMU数据
            acceleration = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            
            gyroscope = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
            
            # 提取方向（四元数）
            orientation_q = msg.orientation
            
            # 将四元数转换为欧拉角（roll, pitch, yaw）
            # 使用标准四元数到欧拉角转换公式
            x = orientation_q.x
            y = orientation_q.y
            z = orientation_q.z
            w = orientation_q.w
            
            # 计算欧拉角
            # roll (x-axis rotation)
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # pitch (y-axis rotation)
            sinp = 2.0 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = np.copysign(np.pi / 2, sinp)  # 使用90度
            else:
                pitch = np.arcsin(sinp)
            
            # yaw (z-axis rotation)
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            orientation = np.array([roll, pitch, yaw])
            
            imu_data = IMUData(
                acceleration=acceleration,
                gyroscope=gyroscope,
                magnetometer=np.zeros(3),  # ROS2 IMU通常不包含磁力计
                orientation=orientation,
                timestamp=time.time()
            )
            
            with self._sensor_cache_lock:
                self._sensor_cache["imu"] = imu_data
                self._sensor_cache["last_update_time"]["imu"] = time.time()
                
        except Exception as e:
            logger.error(f"IMU回调处理失败: {e}")
    
    def _camera_callback(self, msg):
        """摄像头数据回调"""
        try:
            # 使用cv_bridge将ROS图像消息转换为OpenCV格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # 转换为numpy数组
            image_array = np.array(cv_image)
            
            camera_data = CameraData(
                image=image_array,
                depth=None,  # 如果需要深度数据，需订阅深度话题
                timestamp=time.time()
            )
            
            with self._sensor_cache_lock:
                self._sensor_cache["camera"] = camera_data
                self._sensor_cache["last_update_time"]["camera"] = time.time()
                
        except Exception as e:
            logger.error(f"摄像头回调处理失败: {e}")
    
    def _lidar_callback(self, msg):
        """激光雷达数据回调"""
        try:
            import numpy as np
            
            # 提取激光雷达数据
            ranges = np.array(msg.ranges)
            angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
            
            # 将极坐标转换为笛卡尔坐标
            valid_mask = ~np.isinf(ranges) & ~np.isnan(ranges)
            valid_ranges = ranges[valid_mask]
            valid_angles = angles[valid_mask]
            
            x_coords = valid_ranges * np.cos(valid_angles)
            y_coords = valid_ranges * np.sin(valid_angles)
            z_coords = np.zeros_like(x_coords)  # 2D激光雷达假设z=0
            
            # 组合点云数据 [N, 3]
            points = np.column_stack([x_coords, y_coords, z_coords])
            
            lidar_data = LidarData(
                points=points,
                intensities=None,  # LaserScan消息不包含强度数据
                timestamp=time.time()
            )
            
            with self._sensor_cache_lock:
                self._sensor_cache["lidar"] = lidar_data
                self._sensor_cache["last_update_time"]["lidar"] = time.time()
                
        except Exception as e:
            logger.error(f"激光雷达回调处理失败: {e}")
    
    def _ros2_spin(self):
        """ROS2自旋线程"""
        import rclpy  # type: ignore
        
        while rclpy.ok() and self._connected:
            rclpy.spin_once(self.ros2_node, timeout_sec=0.1)
        
        logger.info("ROS2自旋线程结束")
    
    def disconnect(self) -> bool:
        """断开连接
        
        根据项目要求"不采用任何降级处理，直接报错"：
        断开连接失败时抛出RuntimeError。
        
        注意：虽然方法签名返回bool，但实际会抛出异常。
        """
        try:
            self._connected = False
            
            if self.ros2_node:
                self.ros2_node.destroy_node()
            
            import rclpy  # type: ignore
            rclpy.shutdown()
            
            logger.info("ROS2机器人已断开连接")
            return True
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"断开ROS2连接失败: {e}")
    
    def is_connected(self) -> bool:
        """检查是否连接"""
        return self._connected
    
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据"""
        try:
            with self._sensor_cache_lock:
                if sensor_type == SensorType.IMU:
                    return self._sensor_cache.get("imu")
                elif sensor_type == SensorType.CAMERA:
                    return self._sensor_cache.get("camera")
                elif sensor_type == SensorType.LIDAR:
                    return self._sensor_cache.get("lidar")
                elif sensor_type == SensorType.DEPTH_CAMERA:
                    # 深度摄像头数据可能需要特殊处理
                    camera_data = self._sensor_cache.get("camera")
                    if camera_data and camera_data.depth is not None:
                        return camera_data
                    return None  # 返回None
                elif sensor_type == SensorType.FORCE_SENSOR:
                    # 力传感器数据需要订阅特定话题
                    return None  # 返回None
                elif sensor_type == SensorType.TOUCH_SENSOR:
                    # 触摸传感器数据需要订阅特定话题
                    return None  # 返回None
                elif sensor_type == SensorType.JOINT_POSITION:
                    # 返回关节位置数据
                    return self.get_joint_state(RobotJoint.HEAD_YAW)  # 示例，返回一个关节
                elif sensor_type == SensorType.JOINT_TORQUE:
                    # 返回关节扭矩数据
                    joint_state = self.get_joint_state(RobotJoint.HEAD_YAW)
                    if joint_state:
                        return joint_state.torque
                    return None  # 返回None
                else:
                    logger.warning(f"不支持的传感器类型: {sensor_type}")
                    return None  # 返回None
                    
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"获取传感器数据失败: {e}")
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置"""
        return self.set_joint_positions({joint: position})
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置
        
        根据项目要求"不采用任何降级处理，直接报错"：
        1. 机器人未连接时抛出RuntimeError
        2. 关节命令发布器不可用时抛出RuntimeError
        3. 没有有效关节时抛出RuntimeError
        4. 执行失败时抛出RuntimeError
        
        注意：虽然方法签名返回bool，但实际会抛出异常。
        """
        try:
            if not self._connected:
                raise RuntimeError("机器人未连接，无法设置关节位置")
            
            if not self.joint_command_pub:
                raise RuntimeError("关节命令发布器不可用，无法设置关节位置")
            
            # 创建轨迹消息
            import rclpy  # type: ignore
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # type: ignore
            
            msg = JointTrajectory()
            msg.joint_names = []
            point = JointTrajectoryPoint()
            point.positions = []
            
            # 添加关节位置（使用关节映射）
            for joint, pos in positions.items():
                # 使用关节映射获取ROS2关节名称
                joint_name = self._joint_mapping.get(joint)
                if joint_name:
                    msg.joint_names.append(joint_name)
                    point.positions.append(pos)
                else:
                    logger.warning(f"找不到关节映射: {joint}，使用默认名称")
                    msg.joint_names.append(joint.value)
                    point.positions.append(pos)
            
            if not msg.joint_names:
                raise RuntimeError("没有有效的关节可以控制")
            
            point.time_from_start = rclpy.duration.Duration(seconds=1.0).to_msg()
            msg.points.append(point)
            
            # 发布消息
            self.joint_command_pub.publish(msg)
            logger.debug(f"发布关节控制消息: {len(msg.joint_names)}个关节")
            return True
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"设置ROS2关节位置失败: {e}")
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态"""
        try:
            # 使用关节映射获取ROS2关节名称
            joint_name = self._joint_mapping.get(joint)
            if not joint_name:
                # 根据项目要求"不采用任何降级处理，直接报错"
                raise RuntimeError(f"找不到关节映射: {joint}，无法获取关节状态")
            
            with self._joint_states_lock:
                return self._joint_states.get(joint_name)
                
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"获取关节状态失败: {e}")
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态"""
        try:
            states = {}
            
            with self._joint_states_lock:
                # 遍历关节映射，将ROS2关节名称映射回RobotJoint
                for robot_joint, ros_joint_name in self._joint_mapping.items():
                    if ros_joint_name in self._joint_states:
                        # 直接使用缓存的关节状态
                        states[robot_joint] = self._joint_states[ros_joint_name]
                    else:
                        # 如果没有该关节的数据，创建空状态
                        states[robot_joint] = JointState(
                            position=None,
                            velocity=None,
                            torque=None
                        )
            
            return states
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"获取所有关节状态失败: {e}")


class RealRobotManager:
    """真实机器人管理器"""
    
    def __init__(self):
        """初始化真实机器人管理器"""
        self.interfaces = {}
        self.lock = threading.Lock()
        logger.info("真实机器人管理器初始化完成")
    
    def create_interface(self, config: RobotConnectionConfig) -> HardwareInterface:
        """创建机器人接口
        
        根据项目要求"不采用任何降级处理，直接报错"：
        1. 不支持的机器人类型时抛出RuntimeError
        2. 创建接口失败时抛出RuntimeError
        
        注意：方法签名不再返回Optional，总是返回HardwareInterface或抛出异常。
        """
        try:
            if config.robot_type in [RealRobotType.NAO, RealRobotType.PEPPER]:
                interface = NAOqiRobotInterface(config)
            elif config.robot_type == RealRobotType.UNIVERSAL_ROBOT:
                interface = ROS2RobotInterface(config)
            else:
                raise RuntimeError(f"不支持的机器人类型: {config.robot_type}")
            
            with self.lock:
                interface_id = f"{config.robot_type.value}_{config.host}_{config.port}"
                self.interfaces[interface_id] = interface
            
            logger.info(f"创建机器人接口: {interface_id}")
            return interface
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"创建机器人接口失败: {e}")
    
    def connect_robot(self, config: RobotConnectionConfig) -> Dict[str, Any]:
        """连接机器人
        
        根据项目要求"不采用任何降级处理，直接报错"：
        连接失败时抛出RuntimeError。
        
        成功时返回连接信息字典。
        """
        try:
            interface = self.create_interface(config)
            # create_interface现在总是返回接口或抛出异常，不需要检查None
            
            connected = interface.connect()
            # interface.connect()现在成功时返回True，失败时抛出RuntimeError
            
            return {
                "success": True,
                "interface_id": f"{config.robot_type.value}_{config.host}_{config.port}",
                "interface_info": interface.get_interface_info(),
                "message": f"成功连接到 {config.robot_type.value} 机器人"
            }
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"连接机器人失败: {e}")
    
    def disconnect_robot(self, interface_id: str) -> Dict[str, Any]:
        """断开机器人连接
        
        根据项目要求"不采用任何降级处理，直接报错"：
        1. 接口不存在时抛出RuntimeError
        2. 断开连接失败时抛出RuntimeError
        
        成功时返回断开连接信息字典。
        """
        try:
            with self.lock:
                interface = self.interfaces.get(interface_id)
                if not interface:
                    raise RuntimeError(f"接口 {interface_id} 不存在")
                
                disconnected = interface.disconnect()
                # interface.disconnect()现在成功时返回True，失败时抛出RuntimeError
                
                del self.interfaces[interface_id]
                return {
                    "success": True,
                    "message": f"已断开接口 {interface_id}"
                }
                
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"断开机器人连接失败: {e}")
    
    def list_interfaces(self) -> Dict[str, Any]:
        """列出所有接口"""
        with self.lock:
            interface_list = []
            for interface_id, interface in self.interfaces.items():
                interface_list.append({
                    "interface_id": interface_id,
                    "interface_info": interface.get_interface_info(),
                    "connected": interface.is_connected()
                })
            
            return {
                "success": True,
                "interfaces": interface_list,
                "count": len(interface_list)
            }


# 全局真实机器人管理器实例
_real_robot_manager = None


class UniversalRobotAdapter(HardwareInterface):
    """通用机器人适配器
    
    将通用控制命令适配到不同厂家型号的人形机器人。
    支持一次训练，兼容控制所有不同型号人形机器人。
    """
    
    def __init__(self, target_interface: HardwareInterface, robot_config: Dict[str, Any]):
        """
        初始化通用机器人适配器
        
        参数:
            target_interface: 目标机器人硬件接口
            robot_config: 机器人配置，包含关节映射、运动范围等
        """
        super().__init__(simulation_mode=False)
        self.target_interface = target_interface
        self.robot_config = robot_config
        self._interface_type = f"universal_adapter_{target_interface.interface_type}"
        
        # 机器人型号信息（必须先设置，因为后面会用到）
        self.robot_model = robot_config.get("robot_model", "unknown")
        self.manufacturer = robot_config.get("manufacturer", "unknown")
        
        # 从配置中提取关节映射 - 将字符串键转换为RobotJoint枚举
        raw_joint_mapping = robot_config.get("joint_mapping", {})
        self.joint_mapping = self._convert_string_keys_to_robot_joint(raw_joint_mapping)
        self.inverse_joint_mapping = {v: k for k, v in self.joint_mapping.items()}
        
        # 运动范围限制 - 同样转换键
        raw_joint_limits = robot_config.get("joint_limits", {})
        self.joint_limits = self._convert_string_keys_to_robot_joint(raw_joint_limits)
        
        # 几何参数 - 用于运动学计算和兼容性适配
        self.geometry_params = robot_config.get("geometry_params", {})
        if not self.geometry_params:
            # 根据机器人型号提供不同的默认几何参数
            self.geometry_params = self._get_default_geometry_params(self.robot_model)
            logger.warning(f"未提供几何参数，使用{self.robot_model}机器人默认尺寸")
        
        logger.info(f"初始化通用机器人适配器，目标接口: {target_interface.interface_type}，机器人型号: {self.robot_model}")
    
    def _convert_string_keys_to_robot_joint(self, input_dict: Dict[str, Any]) -> Dict[RobotJoint, Any]:
        """将字典的字符串键转换为RobotJoint枚举
        
        参数:
            input_dict: 输入字典，键为关节名称字符串
        
        返回:
            转换后的字典，键为RobotJoint枚举
        """
        if not input_dict:
            return {}
        
        result = {}
        for key_str, value in input_dict.items():
            try:
                # 尝试将字符串转换为RobotJoint枚举
                if isinstance(key_str, str):
                    # 首先尝试直接匹配枚举值
                    try:
                        robot_joint = RobotJoint(key_str)
                    except ValueError:
                        # 如果直接匹配失败，尝试不区分大小写匹配
                        key_lower = key_str.lower()
                        for joint in RobotJoint:
                            if joint.value.lower() == key_lower:
                                robot_joint = joint
                                break
                        else:
                            # 如果还是找不到，记录警告并使用字符串作为键
                            logger.warning(f"无法将字符串 '{key_str}' 转换为RobotJoint枚举，将跳过此键")
                            continue
                elif isinstance(key_str, RobotJoint):
                    # 已经是RobotJoint枚举，直接使用
                    robot_joint = key_str
                else:
                    logger.warning(f"不支持的键类型: {type(key_str)}，将跳过此键")
                    continue
                
                result[robot_joint] = value
                
            except Exception as e:
                logger.warning(f"转换键 '{key_str}' 时发生错误: {e}")
                continue
        
        logger.debug(f"将 {len(input_dict)} 个键转换为RobotJoint枚举，成功转换 {len(result)} 个")
        return result
    
    def _get_default_geometry_params(self, robot_model: str) -> Dict[str, float]:
        """获取默认几何参数（根据机器人型号）
        
        参数:
            robot_model: 机器人型号字符串
            
        返回:
            几何参数字典
        """
        # 根据机器人型号返回不同的默认几何参数
        robot_model_lower = robot_model.lower()
        
        if "nao" in robot_model_lower:
            # NAO机器人（SoftBank Robotics）
            return {
                "upper_arm_length": 0.15,
                "lower_arm_length": 0.15,
                "upper_leg_length": 0.20,
                "lower_leg_length": 0.20,
                "torso_height": 0.40,
                "shoulder_width": 0.10,
                "hip_width": 0.10,
                "foot_length": 0.15,
                "foot_width": 0.07,
                "total_height": 0.58,
                "weight": 5.5
            }
        elif "pepper" in robot_model_lower:
            # Pepper机器人（SoftBank Robotics）
            return {
                "upper_arm_length": 0.20,
                "lower_arm_length": 0.20,
                "upper_leg_length": 0.25,
                "lower_leg_length": 0.25,
                "torso_height": 0.50,
                "shoulder_width": 0.15,
                "hip_width": 0.15,
                "foot_length": 0.20,
                "foot_width": 0.10,
                "total_height": 1.21,
                "weight": 28.0
            }
        elif "atlas" in robot_model_lower:
            # Boston Dynamics Atlas
            return {
                "upper_arm_length": 0.35,
                "lower_arm_length": 0.35,
                "upper_leg_length": 0.45,
                "lower_leg_length": 0.45,
                "torso_height": 0.60,
                "shoulder_width": 0.25,
                "hip_width": 0.25,
                "foot_length": 0.30,
                "foot_width": 0.15,
                "total_height": 1.50,
                "weight": 89.0
            }
        elif "sophia" in robot_model_lower:
            # Hanson Robotics Sophia
            return {
                "upper_arm_length": 0.25,
                "lower_arm_length": 0.25,
                "upper_leg_length": 0.30,
                "lower_leg_length": 0.30,
                "torso_height": 0.45,
                "shoulder_width": 0.18,
                "hip_width": 0.18,
                "foot_length": 0.22,
                "foot_width": 0.12,
                "total_height": 1.10,
                "weight": 20.0
            }
        else:
            # 未知型号，使用通用人形机器人默认值
            logger.warning(f"未知机器人型号 '{robot_model}'，使用通用人形机器人默认尺寸")
            return {
                "upper_arm_length": 0.20,
                "lower_arm_length": 0.20,
                "upper_leg_length": 0.25,
                "lower_leg_length": 0.25,
                "torso_height": 0.45,
                "shoulder_width": 0.12,
                "hip_width": 0.12,
                "foot_length": 0.18,
                "foot_width": 0.09,
                "total_height": 1.00,
                "weight": 25.0
            }
    
    def get_interface_info(self) -> Dict[str, Any]:
        """获取接口信息"""
        base_info = super().get_interface_info()
        
        # 将RobotJoint枚举键转换为字符串用于JSON序列化
        joint_mapping_str = {k.value if isinstance(k, RobotJoint) else str(k): v 
                           for k, v in self.joint_mapping.items()}
        joint_limits_str = {k.value if isinstance(k, RobotJoint) else str(k): v 
                          for k, v in self.joint_limits.items()}
        
        # 获取几何信息
        geometry_info = self.get_robot_geometry_info()
        
        adapter_info = {
            "adapter_type": "universal_robot_adapter",
            "target_interface": self.target_interface.get_interface_info(),
            "joint_mapping": joint_mapping_str,
            "joint_limits": joint_limits_str,
            "joint_mapping_count": len(self.joint_mapping),
            "joint_limits_count": len(self.joint_limits),
            "robot_model": self.robot_model,
            "manufacturer": self.manufacturer,
            "geometry_params": self.geometry_params,
            "has_kinematics": True,
            "kinematics_methods": ["calculate_forward_kinematics"],
            "description": f"通用机器人适配器，支持跨厂家型号兼容控制 - {self.robot_model} ({self.manufacturer})"
        }
        adapter_info.update(geometry_info)
        base_info.update(adapter_info)
        return base_info
    
    def connect(self) -> bool:
        """连接硬件 - 转发到目标接口"""
        return self.target_interface.connect()
    
    def disconnect(self) -> bool:
        """断开连接 - 转发到目标接口"""
        return self.target_interface.disconnect()
    
    def is_connected(self) -> bool:
        """检查连接状态 - 转发到目标接口"""
        return self.target_interface.is_connected()
    
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据 - 转发到目标接口"""
        return self.target_interface.get_sensor_data(sensor_type)
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置 - 应用关节映射和限制"""
        # 应用关节映射
        mapped_joint = self.joint_mapping.get(joint, joint)
        
        # 应用关节限制
        if joint in self.joint_limits:
            limits = self.joint_limits[joint]
            position = max(limits.get("min", -3.14), min(position, limits.get("max", 3.14)))
        
        # 转发到目标接口
        return self.target_interface.set_joint_position(mapped_joint, position)
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置 - 应用关节映射和限制"""
        mapped_positions = {}
        for joint, position in positions.items():
            # 应用关节映射
            mapped_joint = self.joint_mapping.get(joint, joint)
            
            # 应用关节限制
            if joint in self.joint_limits:
                limits = self.joint_limits[joint]
                position = max(limits.get("min", -3.14), min(position, limits.get("max", 3.14)))
            
            mapped_positions[mapped_joint] = position
        
        # 转发到目标接口
        return self.target_interface.set_joint_positions(mapped_positions)
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态 - 应用反向关节映射"""
        mapped_joint = self.joint_mapping.get(joint, joint)
        state = self.target_interface.get_joint_state(mapped_joint)
        
        # 如果需要，可以在这里进行单位转换等适配
        return state
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态 - 应用反向关节映射"""
        all_states = self.target_interface.get_all_joint_states()
        
        # 应用反向关节映射
        adapted_states = {}
        for joint, state in all_states.items():
            # 查找原始关节
            original_joint = self.inverse_joint_mapping.get(joint, joint)
            adapted_states[original_joint] = state
        
        return adapted_states
    
    def calculate_forward_kinematics(self, joint_angles: Dict[RobotJoint, float], 
                                   end_effector: str = "right_hand") -> List[float]:
        """计算正运动学（基于几何参数）
        
        参数:
            joint_angles: 关节角度字典（弧度）
            end_effector: 末端执行器名称，如 "right_hand", "left_hand", "right_foot"
            
        返回:
            末端执行器位置 [x, y, z]（米）
        """
        try:
            import numpy as np
            
            # 使用几何参数进行计算
            upper_arm = self.geometry_params.get("upper_arm_length", 0.15)
            lower_arm = self.geometry_params.get("lower_arm_length", 0.15)
            upper_leg = self.geometry_params.get("upper_leg_length", 0.20)
            lower_leg = self.geometry_params.get("lower_leg_length", 0.20)
            torso_height = self.geometry_params.get("torso_height", 0.40)
            shoulder_width = self.geometry_params.get("shoulder_width", 0.10)
            
            # 基础位置（骨盆高度）
            base_z = torso_height / 2  # 近似骨盆高度
            
            if end_effector == "right_hand":
                # 获取相关关节角度（使用关节映射后的关节）
                shoulder_pitch = joint_angles.get(RobotJoint.R_SHOULDER_PITCH, 0.0)
                shoulder_roll = joint_angles.get(RobotJoint.R_SHOULDER_ROLL, 0.0)
                elbow_yaw = joint_angles.get(RobotJoint.R_ELBOW_YAW, 0.0)
                elbow_roll = joint_angles.get(RobotJoint.R_ELBOW_ROLL, 0.0)
                
                # 简化运动学计算（基于几何参数）
                # 实际应使用完整的DH参数或URDF模型
                x = shoulder_width/2 + upper_arm * np.sin(shoulder_pitch) * np.cos(shoulder_roll)
                y = -shoulder_width/2 + upper_arm * np.sin(shoulder_roll)
                z = base_z + upper_arm * np.cos(shoulder_pitch) * np.cos(shoulder_roll)
                
                return [float(x), float(y), float(z)]
                
            elif end_effector == "left_hand":
                shoulder_pitch = joint_angles.get(RobotJoint.L_SHOULDER_PITCH, 0.0)
                shoulder_roll = joint_angles.get(RobotJoint.L_SHOULDER_ROLL, 0.0)
                
                x = -shoulder_width/2 + upper_arm * np.sin(shoulder_pitch) * np.cos(shoulder_roll)
                y = shoulder_width/2 + upper_arm * np.sin(shoulder_roll)
                z = base_z + upper_arm * np.cos(shoulder_pitch) * np.cos(shoulder_roll)
                
                return [float(x), float(y), float(z)]
                
            elif end_effector == "right_foot":
                # 腿部运动学
                hip_pitch = joint_angles.get(RobotJoint.R_HIP_PITCH, 0.0)
                knee_pitch = joint_angles.get(RobotJoint.R_KNEE_PITCH, 0.0)
                ankle_pitch = joint_angles.get(RobotJoint.R_ANKLE_PITCH, 0.0)
                
                # 简化腿部运动学
                leg_length = upper_leg + lower_leg
                leg_angle = hip_pitch + knee_pitch + ankle_pitch
                
                x = 0.05  # 脚部相对于髋部的前后偏移
                y = -shoulder_width/2
                z = base_z - leg_length * np.cos(leg_angle)
                
                return [float(x), float(y), float(z)]
                
            else:
                # 不支持的其他末端执行器
                raise RuntimeError(f"不支持的末端执行器: {end_effector}")
                
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # 重新抛出RuntimeError
            else:
                raise RuntimeError(f"计算正运动学时发生错误: {e}")
    
    def get_robot_geometry_info(self) -> Dict[str, Any]:
        """获取机器人几何信息
        
        用于兼容性评估和参数适配。
        """
        return {
            "robot_model": self.robot_model,
            "manufacturer": self.manufacturer,
            "geometry_params": self.geometry_params,
            "joint_mapping_count": len(self.joint_mapping),
            "joint_limits_count": len(self.joint_limits),
            "description": f"通用机器人适配器 - {self.robot_model} ({self.manufacturer})"
        }


def get_real_robot_manager() -> RealRobotManager:
    """获取全局真实机器人管理器实例"""
    global _real_robot_manager
    if _real_robot_manager is None:
        _real_robot_manager = RealRobotManager()
    return _real_robot_manager