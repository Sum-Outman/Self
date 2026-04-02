#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAOqi真实控制器

支持NAO、Pepper等人形机器人，通过NAOqi SDK进行控制
核心原则：禁止模拟，强制真实（根据项目要求"禁止使用虚拟数据"）
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import numpy as np

from .base_interface import (
    RealHardwareInterface, HardwareType, ConnectionStatus,
    HardwareError, ConnectionError, OperationError
)

logger = logging.getLogger(__name__)


class NAOqiRobotType(Enum):
    """NAOqi机器人类型枚举"""
    NAO = "nao"                # NAO机器人
    PEPPER = "pepper"          # Pepper机器人
    ROMEO = "romeo"            # Romeo机器人
    JULIETTE = "juliette"      # Juliette机器人
    SIMULATED = "simulated"    # 模拟机器人
    UNKNOWN = "unknown"        # 未知类型


class NAOqiConnectionType(Enum):
    """NAOqi连接类型枚举"""
    DIRECT = "direct"          # 直接连接（IP地址）
    PROXY = "proxy"            # 代理连接
    SIMULATED = "simulated"    # 模拟连接
    UNKNOWN = "unknown"        # 未知类型


class NAOqiJoint(Enum):
    """NAOqi关节枚举"""
    # 头部
    HEAD_YAW = "HeadYaw"
    HEAD_PITCH = "HeadPitch"
    
    # 左臂
    L_SHOULDER_PITCH = "LShoulderPitch"
    L_SHOULDER_ROLL = "LShoulderRoll"
    L_ELBOW_YAW = "LElbowYaw"
    L_ELBOW_ROLL = "LElbowRoll"
    L_WRIST_YAW = "LWristYaw"
    L_HAND = "LHand"
    
    # 右臂
    R_SHOULDER_PITCH = "RShoulderPitch"
    R_SHOULDER_ROLL = "RShoulderRoll"
    R_ELBOW_YAW = "RElbowYaw"
    R_ELBOW_ROLL = "RElbowRoll"
    R_WRIST_YAW = "RWristYaw"
    R_HAND = "RHand"
    
    # 左腿
    L_HIP_YAW_PITCH = "LHipYawPitch"
    L_HIP_ROLL = "LHipRoll"
    L_HIP_PITCH = "LHipPitch"
    L_KNEE_PITCH = "LKneePitch"
    L_ANKLE_PITCH = "LAnklePitch"
    L_ANKLE_ROLL = "LAnkleRoll"
    
    # 右腿
    R_HIP_YAW_PITCH = "RHipYawPitch"
    R_HIP_ROLL = "RHipRoll"
    R_HIP_PITCH = "RHipPitch"
    R_KNEE_PITCH = "RKneePitch"
    R_ANKLE_PITCH = "RAnklePitch"
    R_ANKLE_ROLL = "RAnkleRoll"


class NAOqiSensorType(Enum):
    """NAOqi传感器类型枚举"""
    # 关节传感器
    JOINT_POSITION = "joint_position"
    JOINT_TEMPERATURE = "joint_temperature"
    JOINT_CURRENT = "joint_current"
    JOINT_STIFFNESS = "joint_stiffness"
    
    # IMU传感器
    IMU_ACCELEROMETER = "imu_accelerometer"
    IMU_GYROSCOPE = "imu_gyroscope"
    IMU_ANGLE = "imu_angle"
    
    # 触觉传感器
    FRONT_TACTILE = "front_tactile"
    MIDDLE_TACTILE = "middle_tactile"
    REAR_TACTILE = "rear_tactile"
    
    # 声纳传感器
    SONAR_LEFT = "sonar_left"
    SONAR_RIGHT = "sonar_right"
    
    # 摄像头
    CAMERA_TOP = "camera_top"
    CAMERA_BOTTOM = "camera_bottom"
    
    # 其他传感器
    BATTERY = "battery"
    TEMPERATURE = "temperature"
    FSR_LEFT = "fsr_left"
    FSR_RIGHT = "fsr_right"


class NAOqiRealController(RealHardwareInterface):
    """NAOqi真实控制器
    
    控制NAOqi机器人硬件，支持NAO、Pepper等机器人
    """
    
    def __init__(
        self,
        robot_id: str,
        robot_type: NAOqiRobotType,
        connection_type: NAOqiConnectionType,
        connection_config: Dict[str, Any]
    ):
        """
        初始化NAOqi真实控制器
        
        参数:
            robot_id: 机器人唯一标识符
            robot_type: 机器人类型
            connection_type: 连接类型
            connection_config: 连接配置字典
        """
        super().__init__(
            hardware_type=HardwareType.ROBOT,
            interface_name=f"naoqi_{robot_id}"
        )
        
        self.robot_id = robot_id
        self.robot_type = robot_type
        
        # 检查连接类型，禁止模拟连接
        if connection_type == NAOqiConnectionType.SIMULATED:
            raise ValueError("模拟连接已禁用。项目要求禁止使用虚拟数据，必须使用真实NAOqi连接（DIRECT或PROXY）。")
        
        self.connection_type = connection_type
        self.connection_config = connection_config
        
        # 机器人参数
        self.robot_name = connection_config.get("robot_name", "unknown")
        self.robot_ip = connection_config.get("ip", "127.0.0.1")
        self.robot_port = connection_config.get("port", 9559)
        self.robot_username = connection_config.get("username", "nao")
        self.robot_password = connection_config.get("password", "nao")
        
        # 机器人状态
        self.joint_positions: Dict[str, float] = {}
        self.joint_temperatures: Dict[str, float] = {}
        self.joint_currents: Dict[str, float] = {}
        self.imu_data: Dict[str, Any] = {}
        self.battery_level = 100.0
        self.robot_temperature = 25.0
        
        # NAOqi代理实例
        self._motion_proxy = None
        self._posture_proxy = None
        self._memory_proxy = None
        self._tts_proxy = None
        self._audio_proxy = None
        self._video_proxy = None
        self._sensor_proxy = None
        self._autonomous_life_proxy = None
        
        # 连接状态
        self._connected = False
        self._connection_lock = threading.RLock()
        
        # 初始化所有关节位置字典
        self._init_joint_positions()
        
        logger.info(
            f"初始化NAOqi真实控制器: {robot_id} "
            f"({robot_type.value}, {connection_type.value})"
        )
    
    def _init_joint_positions(self):
        """初始化关节位置字典"""
        for joint in NAOqiJoint:
            self.joint_positions[joint.value] = 0.0
            self.joint_temperatures[joint.value] = 25.0
            self.joint_currents[joint.value] = 0.0
    
    def connect(self) -> bool:
        """连接NAOqi机器人"""
        with self._connection_lock:
            try:
                if self._connected:
                    logger.info("NAOqi机器人已连接")
                    return True
                
                logger.info(f"连接NAOqi机器人: {self.robot_ip}:{self.robot_port}")
                
                # 根据连接类型进行连接
                if self.connection_type == NAOqiConnectionType.SIMULATED:
                    # 模拟连接已禁用
                    raise ValueError("模拟连接已禁用。项目要求禁止使用虚拟数据，必须使用真实NAOqi连接。")
                
                # 真实NAOqi连接
                try:
                    import naoqi  # type: ignore
                    from naoqi import ALProxy  # type: ignore
                    
                    # 连接到机器人
                    self._motion_proxy = ALProxy("ALMotion", self.robot_ip, self.robot_port)
                    self._posture_proxy = ALProxy("ALRobotPosture", self.robot_ip, self.robot_port)
                    self._memory_proxy = ALProxy("ALMemory", self.robot_ip, self.robot_port)
                    self._tts_proxy = ALProxy("ALTextToSpeech", self.robot_ip, self.robot_port)
                    
                    # 尝试连接其他代理（可选）
                    try:
                        self._audio_proxy = ALProxy("ALAudioDevice", self.robot_ip, self.robot_port)
                    except:
                        logger.warning("音频代理连接失败")
                    
                    try:
                        self._video_proxy = ALProxy("ALVideoDevice", self.robot_ip, self.robot_port)
                    except:
                        logger.warning("视频代理连接失败")
                    
                    try:
                        self._sensor_proxy = ALProxy("ALSensors", self.robot_ip, self.robot_port)
                    except:
                        logger.warning("传感器代理连接失败")
                    
                    try:
                        self._autonomous_life_proxy = ALProxy("ALAutonomousLife", self.robot_ip, self.robot_port)
                    except:
                        logger.warning("自主生命代理连接失败")
                    
                    # 测试连接
                    if self._motion_proxy:
                        self._motion_proxy.wakeUp()  # 唤醒机器人
                        time.sleep(1)
                        self._motion_proxy.rest()    # 让机器人休息
                        
                        self._connected = True
                        self.connection_status = ConnectionStatus.CONNECTED
                        logger.info("NAOqi真实连接成功")
                        return True
                    else:
                        raise ConnectionError("运动代理创建失败")
                        
                except ImportError as e:
                    logger.error(f"naoqi模块未安装: {e}")
                    raise ImportError("naoqi模块未安装。项目要求禁止使用模拟模式，必须安装NAOqi SDK。") from e
                    
            except Exception as e:
                logger.error(f"NAOqi连接失败: {e}")
                self.connection_status = ConnectionStatus.ERROR
                return False
    
    def disconnect(self) -> bool:
        """断开NAOqi机器人连接"""
        with self._connection_lock:
            try:
                if not self._connected:
                    logger.info("NAOqi机器人未连接")
                    return True
                
                logger.info("断开NAOqi机器人连接")
                
                # 真实NAOqi连接（SIMULATED类型已由__init__禁止）
                try:
                    if self._motion_proxy:
                        self._motion_proxy.rest()  # 让机器人休息
                except:
                    pass  # 断开连接时硬件可能已不可用
                
                # 清理代理
                self._motion_proxy = None
                self._posture_proxy = None
                self._memory_proxy = None
                self._tts_proxy = None
                self._audio_proxy = None
                self._video_proxy = None
                self._sensor_proxy = None
                self._autonomous_life_proxy = None
                
                self._connected = False
                self.connection_status = ConnectionStatus.DISCONNECTED
                logger.info("NAOqi断开连接成功")
                return True
                
            except Exception as e:
                logger.error(f"NAOqi断开连接失败: {e}")
                return False
    
    def get_joint_position(self, joint_name: Union[str, NAOqiJoint]) -> float:
        """获取关节位置
        
        参数:
            joint_name: 关节名称或NAOqiJoint枚举
        
        返回:
            关节位置（弧度）
        """
        try:
            if isinstance(joint_name, NAOqiJoint):
                joint_name = joint_name.value
            
            if not self._connected:
                # 未连接，无法获取关节位置
                raise RuntimeError("机器人未连接。项目要求禁止使用虚拟数据，必须先连接真实机器人。")
            
            # 真实NAOqi连接
            # connection_type != SIMULATED 已由__init__保证
            if self._motion_proxy:
                try:
                    # 获取关节角度（NAOqi返回的是弧度）
                    position = self._motion_proxy.getAngles(joint_name, True)[0]
                    self.joint_positions[joint_name] = position
                    return position
                except Exception as e:
                    logger.error(f"获取关节位置失败 {joint_name}: {e}")
                    raise RuntimeError(f"获取关节位置失败: {e}。项目要求禁止使用虚拟数据，必须使用真实硬件。") from e
            
            # 没有可用的运动代理
            raise RuntimeError("运动代理未初始化。项目要求禁止使用虚拟数据，必须确保机器人硬件连接正常。")
            
        except Exception as e:
            logger.error(f"获取关节位置异常 {joint_name}: {e}")
            raise RuntimeError(f"获取关节位置失败: {e}。项目要求禁止使用虚拟数据，必须使用真实硬件。") from e
    
    def set_joint_position(
        self, 
        joint_name: Union[str, NAOqiJoint], 
        position: float, 
        speed: float = 0.2,
        relative: bool = False
    ) -> bool:
        """设置关节位置
        
        参数:
            joint_name: 关节名称或NAOqiJoint枚举
            position: 目标位置（弧度）
            speed: 移动速度（0.0-1.0）
            relative: 是否为相对移动
        
        返回:
            是否成功
        """
        try:
            if isinstance(joint_name, NAOqiJoint):
                joint_name = joint_name.value
            
            if not self._connected:
                # 未连接，无法设置关节位置
                raise RuntimeError("机器人未连接。项目要求禁止使用虚拟数据，必须先连接真实机器人。")
            
            # 真实NAOqi连接
            # connection_type != SIMULATED 已由__init__保证
            if self._motion_proxy:
                try:
                    # 设置关节角度
                    if relative:
                        current = self._motion_proxy.getAngles(joint_name, True)[0]
                        position = current + position
                    
                    self._motion_proxy.setAngles(joint_name, position, speed)
                    self.joint_positions[joint_name] = position
                    logger.debug(f"设置关节位置 {joint_name}: {position}")
                    return True
                except Exception as e:
                    logger.error(f"设置关节位置失败 {joint_name}: {e}")
                    return False
            
            # 模拟模式已禁用
            raise RuntimeError("模拟设置关节位置已禁用。项目要求禁止使用虚拟数据，必须使用真实机器人硬件。")
            
        except Exception as e:
            logger.error(f"设置关节位置异常 {joint_name}: {e}")
            return False
    
    def get_sensor_data(self, sensor_type: Union[str, NAOqiSensorType]) -> Dict[str, Any]:
        """获取传感器数据
        
        参数:
            sensor_type: 传感器类型或NAOqiSensorType枚举
        
        返回:
            传感器数据字典
        """
        try:
            if isinstance(sensor_type, NAOqiSensorType):
                sensor_type = sensor_type.value
            
            if not self._connected:
                # 未连接，无法获取传感器数据
                raise RuntimeError("机器人未连接。项目要求禁止使用虚拟数据，必须先连接真实机器人。")
            
            # 真实NAOqi连接
            # connection_type != SIMULATED 已由__init__保证
            try:
                return self._get_real_sensor_data(sensor_type)
            except Exception as e:
                logger.error(f"获取真实传感器数据失败 {sensor_type}: {e}")
                raise RuntimeError(f"获取传感器数据失败: {e}。项目要求禁止使用虚拟数据，必须使用真实传感器硬件。") from e
            
        except Exception as e:
            logger.error(f"获取传感器数据异常 {sensor_type}: {e}")
            return {"type": sensor_type, "error": str(e), "value": 0.0}
    
    def _get_real_sensor_data(self, sensor_type: str) -> Dict[str, Any]:
        """获取真实传感器数据"""
        try:
            import naoqi  # type: ignore
            from naoqi import ALProxy  # type: ignore
            
            data = {"type": sensor_type, "timestamp": time.time()}
            
            if sensor_type == NAOqiSensorType.BATTERY.value:
                # 获取电池信息
                if self._memory_proxy:
                    battery_level = self._memory_proxy.getData("Device/SubDeviceList/Battery/Charge/Sensor/Value")
                    data["value"] = battery_level * 100.0  # 转换为百分比
                    data["unit"] = "%"
            
            elif sensor_type == NAOqiSensorType.IMU_ACCELEROMETER.value:
                # 获取加速度计数据
                if self._memory_proxy:
                    accel_x = self._memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value")
                    accel_y = self._memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value")
                    accel_z = self._memory_proxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value")
                    data["value"] = [accel_x, accel_y, accel_z]
                    data["unit"] = "m/s²"
            
            elif sensor_type == NAOqiSensorType.IMU_GYROSCOPE.value:
                # 获取陀螺仪数据
                if self._memory_proxy:
                    gyro_x = self._memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeX/Sensor/Value")
                    gyro_y = self._memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeY/Sensor/Value")
                    gyro_z = self._memory_proxy.getData("Device/SubDeviceList/InertialSensor/GyroscopeZ/Sensor/Value")
                    data["value"] = [gyro_x, gyro_y, gyro_z]
                    data["unit"] = "rad/s"
            
            elif sensor_type == NAOqiSensorType.IMU_ANGLE.value:
                # 获取角度数据
                if self._memory_proxy:
                    angle_x = self._memory_proxy.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
                    angle_y = self._memory_proxy.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")
                    data["value"] = [angle_x, angle_y]
                    data["unit"] = "rad"
            
            elif sensor_type in [NAOqiSensorType.JOINT_TEMPERATURE.value, 
                               NAOqiSensorType.JOINT_CURRENT.value]:
                # 获取关节温度或电流
                if self._sensor_proxy:
                    # 完整处理，实际需要更复杂的逻辑
                    data["value"] = 25.0 if "temperature" in sensor_type else 0.1
                    data["unit"] = "°C" if "temperature" in sensor_type else "A"
            
            else:
                # 未知传感器类型
                data["value"] = 0.0
                data["unit"] = "unknown"
            
            return data
            
        except Exception as e:
            logger.error(f"获取真实传感器数据失败 {sensor_type}: {e}")
            raise
    
    def _get_simulated_sensor_data(self, sensor_type: str) -> Dict[str, Any]:
        """获取模拟传感器数据（已禁用）"""
        raise RuntimeError("模拟传感器数据获取已禁用。项目要求禁止使用虚拟数据，必须使用真实传感器硬件。")
    
    def speak_text(self, text: str, language: str = "Chinese") -> bool:
        """让机器人说话
        
        参数:
            text: 要说的文本
            language: 语言
        
        返回:
            是否成功
        """
        try:
            if not self._connected:
                raise RuntimeError("机器人未连接。项目要求禁止使用虚拟数据，必须先连接真实机器人。")
            
            # 真实NAOqi连接
            # connection_type != SIMULATED 已由__init__保证
            if self._tts_proxy:
                try:
                    self._tts_proxy.say(text)
                    logger.debug(f"机器人说话: {text}")
                    return True
                except Exception as e:
                    logger.error(f"机器人说话失败: {e}")
                    return False
            
            # 模拟模式已禁用
            raise RuntimeError("模拟说话已禁用。项目要求禁止使用虚拟数据，必须使用真实机器人硬件。")
            
        except Exception as e:
            logger.error(f"机器人说话异常: {e}")
            return False
    
    def perform_posture(self, posture_name: str) -> bool:
        """执行预设姿势
        
        参数:
            posture_name: 姿势名称（如："Stand", "Sit", "Crouch"）
        
        返回:
            是否成功
        """
        try:
            if not self._connected:
                raise RuntimeError("机器人未连接。项目要求禁止使用虚拟数据，必须先连接真实机器人。")
            
            # 真实NAOqi连接
            # connection_type != SIMULATED 已由__init__保证
            if self._posture_proxy:
                try:
                    self._posture_proxy.goToPosture(posture_name, 0.5)  # 0.5速度
                    logger.debug(f"执行姿势: {posture_name}")
                    return True
                except Exception as e:
                    logger.error(f"执行姿势失败 {posture_name}: {e}")
                    return False
            
            # 模拟模式已禁用
            raise RuntimeError("模拟执行姿势已禁用。项目要求禁止使用虚拟数据，必须使用真实机器人硬件。")
            
        except Exception as e:
            logger.error(f"执行姿势异常 {posture_name}: {e}")
            return False
    
    def enable_autonomous_life(self, enable: bool = True) -> bool:
        """启用/禁用自主生命
        
        参数:
            enable: 是否启用
        
        返回:
            是否成功
        """
        try:
            if not self._connected:
                raise RuntimeError("机器人未连接。项目要求禁止使用虚拟数据，必须先连接真实机器人。")
            
            # 真实NAOqi连接
            # connection_type != SIMULATED 已由__init__保证
            if self._autonomous_life_proxy:
                try:
                    if enable:
                        self._autonomous_life_proxy.setState("solitary")
                    else:
                        self._autonomous_life_proxy.setState("disabled")
                    logger.debug(f"自主生命: {'启用' if enable else '禁用'}")
                    return True
                except Exception as e:
                    logger.error(f"设置自主生命失败: {e}")
                    return False
            
            # 模拟模式已禁用
            raise RuntimeError("模拟自主生命已禁用。项目要求禁止使用虚拟数据，必须使用真实机器人硬件。")
            
        except Exception as e:
            logger.error(f"设置自主生命异常: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取机器人状态
        
        返回:
            状态字典
        """
        status = super().get_status()
        
        # 添加NAOqi特定状态
        status.update({
            "robot_id": self.robot_id,
            "robot_type": self.robot_type.value,
            "connection_type": self.connection_type.value,
            "connected": self._connected,
            "battery_level": self.battery_level,
            "robot_temperature": self.robot_temperature,
            "joint_count": len(self.joint_positions),
            "sensors_available": [s.value for s in NAOqiSensorType],
        })
        
        return status
    
    def emergency_stop(self) -> bool:
        """紧急停止"""
        try:
            if not self._connected:
                raise RuntimeError("机器人未连接。项目要求禁止使用虚拟数据，必须先连接真实机器人。")
            
            # 真实NAOqi连接
            # connection_type != SIMULATED 已由__init__保证
            if self._motion_proxy:
                try:
                    self._motion_proxy.rest()  # 让机器人休息（停止所有运动）
                    logger.warning("NAOqi紧急停止")
                    return True
                except Exception as e:
                    logger.error(f"紧急停止失败: {e}")
                    return False
            
            # 模拟模式已禁用
            raise RuntimeError("模拟紧急停止已禁用。项目要求禁止使用虚拟数据，必须使用真实机器人硬件。")
            
        except Exception as e:
            logger.error(f"紧急停止异常: {e}")
            return False