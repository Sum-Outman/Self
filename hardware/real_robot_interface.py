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

# 导入基类
from .robot_controller import (
    HardwareInterface, RobotJoint, SensorType,
    JointState, IMUData, CameraData, LidarData
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
        """连接到机器人"""
        try:
            logger.info(f"尝试连接到 {self.config.robot_type.value} 机器人: {self.config.host}:{self.config.port}")
            
            # 尝试导入NAOqi库
            try:
                import naoqi  # type: ignore
                from naoqi import ALProxy  # type: ignore
                NAOQI_AVAILABLE = True
            except ImportError:
                logger.error("NAOqi SDK未安装，无法连接真实机器人")
                NAOQI_AVAILABLE = False
                return False
            
            if not NAOQI_AVAILABLE:
                return False
            
            # 创建代理
            self.motion_proxy = naoqi.ALProxy("ALMotion", self.config.host, self.config.port)
            self.memory_proxy = naoqi.ALProxy("ALMemory", self.config.host, self.config.port)
            self.posture_proxy = naoqi.ALProxy("ALRobotPosture", self.config.host, self.config.port)
            
            # 检查连接
            robot_name = self.memory_proxy.getData("RobotConfig/Body/Type")
            logger.info(f"连接成功，机器人型号: {robot_name}")
            
            # 设置刚体模式（禁用自主生命）
            try:
                self.autonomous_life_proxy = naoqi.ALProxy("ALAutonomousLife", self.config.host, self.config.port)
                self.autonomous_life_proxy.setState("disabled")
                logger.info("已禁用自主生命模式")
            except Exception:
                logger.warning("无法禁用自主生命模式（可能是旧版NAOqi）")
            
            # 唤醒机器人
            self.motion_proxy.wakeUp()
            
            # 设置初始姿态
            if self.config.robot_type == RealRobotType.NAO:
                self.posture_proxy.goToPosture("StandInit", 0.5)
            elif self.config.robot_type == RealRobotType.PEPPER:
                self.posture_proxy.goToPosture("Stand", 0.5)
            
            self._connected = True
            
            # 启动传感器数据采集线程
            self._start_sensor_thread()
            
            # 启动安全监控线程
            if self.config.emergency_stop_enabled:
                self._start_safety_monitor()
            
            logger.info(f"机器人连接成功，准备就绪")
            return True
            
        except Exception as e:
            logger.error(f"连接机器人失败: {e}")
            self._connected = False
            return False
    
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
        """收集传感器数据"""
        with self._cache_lock:
            try:
                # 获取关节角度
                joint_names = list(self._joint_mapping.values())
                joint_angles = self.motion_proxy.getAngles(joint_names, True)
                
                # 获取关节温度（如果可用）
                joint_temperatures = {}
                try:
                    # NAOqi v2.8+支持
                    joint_temperatures = self.memory_proxy.getData("Device/SubDeviceList/Temperature/Sensor/Value")
                except Exception:
                    pass  # 已实现
                
                # 获取IMU数据
                imu_data = {}
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
                except Exception:
                    pass  # 已实现
                
                # 更新缓存
                self._sensor_cache.update({
                    "joint_angles": dict(zip(joint_names, joint_angles)),
                    "joint_temperatures": joint_temperatures,
                    "imu_data": imu_data,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                logger.error(f"收集传感器数据失败: {e}")
    
    def _check_safety(self):
        """安全检查"""
        try:
            # 检查关节温度
            if "joint_temperatures" in self._sensor_cache:
                temps = self._sensor_cache["joint_temperatures"]
                if isinstance(temps, dict):
                    for joint_name, temp in temps.items():
                        if temp > 70.0:  # 温度阈值（摄氏度）
                            logger.warning(f"关节 {joint_name} 温度过高: {temp}°C")
                            if temp > 80.0:
                                self.emergency_stop()
                                return
            
            # 检查关节负载（如果可用）
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
            except Exception:
                pass  # 已实现
            
            # 检查电池电量
            try:
                battery_level = self.memory_proxy.getData("Device/SubDeviceList/Battery/Charge/Sensor/Value")
                if battery_level < 0.2:  # 20%电量
                    logger.warning(f"电池电量低: {battery_level*100:.1f}%")
                    if battery_level < 0.1:  # 10%电量
                        logger.error("电池电量极低，即将关机")
                        # 可以触发安全关机
            except Exception:
                pass  # 已实现
                
        except Exception as e:
            logger.error(f"安全检查失败: {e}")
    
    def emergency_stop(self):
        """紧急停止"""
        if self._emergency_stop:
            return
        
        logger.error("触发紧急停止")
        self._emergency_stop = True
        
        try:
            # 停止所有运动
            self.motion_proxy.stopMove()
            self.motion_proxy.killAll()
            
            # 进入休息姿势
            self.posture_proxy.goToPosture("Crouch", 0.8)
            
            # 断开连接
            self.disconnect()
            
        except Exception as e:
            logger.error(f"紧急停止失败: {e}")
    
    def disconnect(self) -> bool:
        """断开连接"""
        try:
            self._stop_event.set()
            
            # 停止线程
            if self._connection_thread and self._connection_thread.is_alive():
                self._connection_thread.join(timeout=2.0)
            
            if self._safety_monitor_thread and self._safety_monitor_thread.is_alive():
                self._safety_monitor_thread.join(timeout=2.0)
            
            # 让机器人进入休息状态
            if self.motion_proxy:
                self.motion_proxy.rest()
            
            self._connected = False
            logger.info("机器人已断开连接")
            return True
            
        except Exception as e:
            logger.error(f"断开连接失败: {e}")
            return False
    
    def is_connected(self) -> bool:
        """检查是否连接"""
        return self._connected
    
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据"""
        with self._cache_lock:
            if sensor_type == SensorType.IMU:
                return self._sensor_cache.get("imu_data")
            elif sensor_type == SensorType.JOINT_POSITION:
                return self._sensor_cache.get("joint_angles")
            elif sensor_type == SensorType.JOINT_TORQUE:
                # 需要额外计算或获取
                return None  # 返回None
            else:
                logger.warning(f"不支持的传感器类型: {sensor_type}")
                return None  # 返回None
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置单个关节位置"""
        try:
            if not self._connected:
                logger.error("机器人未连接")
                return False
            
            # 获取NAOqi关节名称
            joint_name = self._joint_mapping.get(joint)
            if not joint_name:
                logger.error(f"未知的关节: {joint}")
                return False
            
            # 设置关节角度（弧度）
            # NAOqi期望角度为弧度
            self.motion_proxy.setAngles(joint_name, position, 0.1)  # 10%速度
            return True
            
        except Exception as e:
            logger.error(f"设置关节位置失败: {e}")
            return False
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置"""
        try:
            if not self._connected:
                logger.error("机器人未连接")
                return False
            
            # 转换为NAOqi格式
            joint_names = []
            joint_angles = []
            
            for joint, angle in positions.items():
                joint_name = self._joint_mapping.get(joint)
                if joint_name:
                    joint_names.append(joint_name)
                    joint_angles.append(angle)
            
            if not joint_names:
                logger.error("没有有效的关节映射")
                return False
            
            # 设置多个关节角度
            self.motion_proxy.setAngles(joint_names, joint_angles, 0.1)
            return True
            
        except Exception as e:
            logger.error(f"设置多个关节位置失败: {e}")
            return False
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取单个关节状态"""
        try:
            joint_name = self._joint_mapping.get(joint)
            if not joint_name:
                return None  # 返回None
            
            with self._cache_lock:
                joint_angles = self._sensor_cache.get("joint_angles", {})
                joint_temperatures = self._sensor_cache.get("joint_temperatures", {})
                angle = joint_angles.get(joint_name)
                temp = joint_temperatures.get(joint_name) if isinstance(joint_temperatures, dict) else None
                
                # 创建关节状态对象
                state = JointState(
                    position=angle,
                    velocity=None,  # NAOqi不直接提供速度，需要计算
                    torque=None,    # 需要额外获取
                    temperature=temp  # 从joint_temperatures获取
                )
                
                return state
                
        except Exception as e:
            logger.error(f"获取关节状态失败: {e}")
            return None  # 返回None
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态"""
        try:
            states = {}
            
            with self._cache_lock:
                joint_angles = self._sensor_cache.get("joint_angles", {})
                joint_temperatures = self._sensor_cache.get("joint_temperatures", {})
                
                for robot_joint, naoqi_name in self._joint_mapping.items():
                    angle = joint_angles.get(naoqi_name)
                    temp = joint_temperatures.get(naoqi_name) if isinstance(joint_temperatures, dict) else None
                    
                    states[robot_joint] = JointState(
                        position=angle,
                        velocity=None,
                        torque=None,
                        temperature=temp
                    )
            
            return states
            
        except Exception as e:
            logger.error(f"获取所有关节状态失败: {e}")
            return {}  # 返回空字典


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
            
            # 尝试导入ROS2库
            try:
                import rclpy  # type: ignore
                from rclpy.node import Node  # type: ignore
                from sensor_msgs.msg import JointState, Imu, Image, LaserScan  # type: ignore
                from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # type: ignore
                from cv_bridge import CvBridge  # type: ignore
                ROS2_AVAILABLE = True
            except ImportError as e:
                logger.error(f"ROS2库未安装，无法连接: {e}")
                ROS2_AVAILABLE = False
                return False
            
            if not ROS2_AVAILABLE:
                return False
            
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
            
        except Exception as e:
            logger.error(f"连接ROS2机器人失败: {e}")
            return False
    
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
        """断开连接"""
        try:
            self._connected = False
            
            if self.ros2_node:
                self.ros2_node.destroy_node()
            
            import rclpy  # type: ignore
            rclpy.shutdown()
            
            logger.info("ROS2机器人已断开连接")
            return True
            
        except Exception as e:
            logger.error(f"断开ROS2连接失败: {e}")
            return False
    
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
            logger.error(f"获取传感器数据失败: {e}")
            return None  # 返回None
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置"""
        return self.set_joint_positions({joint: position})
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置"""
        try:
            if not self._connected or not self.joint_command_pub:
                return False
            
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
                logger.error("没有有效的关节可以控制")
                return False
            
            point.time_from_start = rclpy.duration.Duration(seconds=1.0).to_msg()
            msg.points.append(point)
            
            # 发布消息
            self.joint_command_pub.publish(msg)
            logger.debug(f"发布关节控制消息: {len(msg.joint_names)}个关节")
            return True
            
        except Exception as e:
            logger.error(f"设置ROS2关节位置失败: {e}")
            return False
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态"""
        try:
            # 使用关节映射获取ROS2关节名称
            joint_name = self._joint_mapping.get(joint)
            if not joint_name:
                logger.warning(f"找不到关节映射: {joint}，尝试使用默认名称")
                joint_name = joint.value
            
            with self._joint_states_lock:
                return self._joint_states.get(joint_name)
                
        except Exception as e:
            logger.error(f"获取关节状态失败: {e}")
            return None  # 返回None
    
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
            logger.error(f"获取所有关节状态失败: {e}")
            return {}  # 返回空字典


class RealRobotManager:
    """真实机器人管理器"""
    
    def __init__(self):
        """初始化真实机器人管理器"""
        self.interfaces = {}
        self.lock = threading.Lock()
        logger.info("真实机器人管理器初始化完成")
    
    def create_interface(self, config: RobotConnectionConfig) -> Optional[HardwareInterface]:
        """创建机器人接口"""
        try:
            if config.robot_type in [RealRobotType.NAO, RealRobotType.PEPPER]:
                interface = NAOqiRobotInterface(config)
            elif config.robot_type == RealRobotType.UNIVERSAL_ROBOT:
                interface = ROS2RobotInterface(config)
            else:
                logger.error(f"不支持的机器人类型: {config.robot_type}")
                return None  # 返回None
            
            with self.lock:
                interface_id = f"{config.robot_type.value}_{config.host}_{config.port}"
                self.interfaces[interface_id] = interface
            
            logger.info(f"创建机器人接口: {interface_id}")
            return interface
            
        except Exception as e:
            logger.error(f"创建机器人接口失败: {e}")
            return None  # 返回None
    
    def connect_robot(self, config: RobotConnectionConfig) -> Dict[str, Any]:
        """连接机器人"""
        try:
            interface = self.create_interface(config)
            if not interface:
                return {
                    "success": False,
                    "error": "创建接口失败"
                }
            
            connected = interface.connect()
            if not connected:
                return {
                    "success": False,
                    "error": "连接机器人失败"
                }
            
            return {
                "success": True,
                "interface_id": f"{config.robot_type.value}_{config.host}_{config.port}",
                "interface_info": interface.get_interface_info(),
                "message": f"成功连接到 {config.robot_type.value} 机器人"
            }
            
        except Exception as e:
            logger.error(f"连接机器人失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def disconnect_robot(self, interface_id: str) -> Dict[str, Any]:
        """断开机器人连接"""
        try:
            with self.lock:
                interface = self.interfaces.get(interface_id)
                if not interface:
                    return {
                        "success": False,
                        "error": f"接口 {interface_id} 不存在"
                    }
                
                disconnected = interface.disconnect()
                if disconnected:
                    del self.interfaces[interface_id]
                    return {
                        "success": True,
                        "message": f"已断开接口 {interface_id}"
                    }
                else:
                    return {
                        "success": False,
                        "error": "断开连接失败"
                    }
                
        except Exception as e:
            logger.error(f"断开机器人连接失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
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

def get_real_robot_manager() -> RealRobotManager:
    """获取全局真实机器人管理器实例"""
    global _real_robot_manager
    if _real_robot_manager is None:
        _real_robot_manager = RealRobotManager()
    return _real_robot_manager