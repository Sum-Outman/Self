#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一仿真环境接口

功能：
1. 提供统一的仿真环境接口，支持多种物理引擎后端（PyBullet、Gazebo）
2. 统一的方法签名和参数配置
3. 自动选择最佳可用仿真后端
4. 向后兼容现有代码

根据《重复文件和冗余代码整合计划》第一阶段创建
"""

import sys
import os
import logging
import time
import threading
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import math

# 导入硬件接口基类
from .robot_controller import (
    HardwareInterface, 
    RobotJoint, 
    SensorType,
    JointState,
    IMUData,
    CameraData,
    LidarData
)


class SimulationEngine(Enum):
    """仿真引擎类型"""
    PY_BULLET = "pybullet"
    GAZEBO = "gazebo"
    PURE_SIMULATION = "pure_simulation"  # 纯模拟模式，无物理引擎


class UnifiedSimulation(HardwareInterface):
    """统一仿真环境接口"""
    
    def __init__(self, 
                 engine: str = "pybullet",
                 engine_config: Optional[Dict[str, Any]] = None,
                 simulation_mode: bool = True,  # 总是仿真模式
                 gui_enabled: bool = True,
                 physics_timestep: float = 1.0/240.0,
                 realtime_simulation: bool = False):
        """
        初始化统一仿真环境
        
        参数:
            engine: 仿真引擎 ("pybullet", "gazebo", "pure_simulation")
            engine_config: 引擎特定配置
            simulation_mode: 总是True，因为这是仿真接口
            gui_enabled: 是否启用GUI可视化
            physics_timestep: 物理时间步长（秒）
            realtime_simulation: 是否实时仿真
        """
        super().__init__(simulation_mode=True)  # 总是仿真模式
        self.engine = engine
        self.engine_config = engine_config or {}
        self.gui_enabled = gui_enabled
        self.physics_timestep = physics_timestep
        self.realtime_simulation = realtime_simulation
        
        self.logger = logging.getLogger("UnifiedSimulation")
        
        # 后端仿真对象
        self.backend = None
        self.backend_type = None
        
        # 连接状态
        self.connected = False
        self.initialized = False
        
        # 选择并初始化后端
        self._initialize_backend()
        
        self._interface_type = f"unified_simulation_{engine}"
    
    def _initialize_backend(self):
        """初始化仿真后端"""
        try:
            if self.engine == SimulationEngine.PY_BULLET.value:
                self._initialize_pybullet_backend()
            elif self.engine == SimulationEngine.GAZEBO.value:
                self._initialize_gazebo_backend()
            elif self.engine == SimulationEngine.PURE_SIMULATION.value:
                self._initialize_pure_simulation_backend()
            else:
                self.logger.error(f"不支持的仿真引擎: {self.engine}")
                self._initialize_pure_simulation_backend()
                
        except Exception as e:
            self.logger.error(f"初始化仿真后端失败: {e}")
            self.logger.warning("将使用纯模拟模式")
            self._initialize_pure_simulation_backend()
    
    def _initialize_pybullet_backend(self):
        """初始化PyBullet后端"""
        try:
            import pybullet  # type: ignore
            
            self.backend_type = "pybullet"
            
            # 检查PyBullet是否可用
            self.logger.info("正在初始化PyBullet仿真后端...")
            
            # 这里可以创建PyBullet后端对象
            # 实际实现应封装原有PyBulletSimulation类的功能
            from .simulation import PyBulletSimulation
            
            # 传递配置参数
            pybullet_config = {
                "gui_enabled": self.gui_enabled,
                "physics_timestep": self.physics_timestep,
                "realtime_simulation": self.realtime_simulation,
                **self.engine_config
            }
            
            self.backend = PyBulletSimulation(
                gui_enabled=pybullet_config.get("gui_enabled", True),
                physics_timestep=pybullet_config.get("physics_timestep", 1.0/240.0),
                realtime_simulation=pybullet_config.get("realtime_simulation", False)
            )
            
            self.logger.info("PyBullet仿真后端初始化成功")
            
        except ImportError as e:
            self.logger.error(f"PyBullet不可用: {e}")
            self._initialize_pure_simulation_backend()
        except Exception as e:
            self.logger.error(f"初始化PyBullet后端失败: {e}")
            self._initialize_pure_simulation_backend()
    
    def _initialize_gazebo_backend(self):
        """初始化Gazebo后端"""
        try:
            # 检查roslibpy是否可用
            import roslibpy  # type: ignore
            
            self.backend_type = "gazebo"
            
            self.logger.info("正在初始化Gazebo仿真后端...")
            
            # 这里可以创建Gazebo后端对象
            # 实际实现应封装原有GazeboSimulation类的功能
            from .gazebo_simulation import GazeboSimulation
            
            # 传递配置参数
            gazebo_config = {
                "ros_master_uri": "http://localhost:11311",
                "gazebo_world": "empty.world",
                "robot_model": "humanoid",
                "gui_enabled": self.gui_enabled,
                "physics_timestep": self.physics_timestep,
                **self.engine_config
            }
            
            self.backend = GazeboSimulation(
                ros_master_uri=gazebo_config.get("ros_master_uri", "http://localhost:11311"),
                gazebo_world=gazebo_config.get("gazebo_world", "empty.world"),
                robot_model=gazebo_config.get("robot_model", "humanoid"),
                gui_enabled=gazebo_config.get("gui_enabled", True),
                physics_timestep=gazebo_config.get("physics_timestep", 0.001)
            )
            
            self.logger.info("Gazebo仿真后端初始化成功")
            
        except ImportError as e:
            self.logger.error(f"roslibpy不可用: {e}")
            self._initialize_pure_simulation_backend()
        except Exception as e:
            self.logger.error(f"初始化Gazebo后端失败: {e}")
            self._initialize_pure_simulation_backend()
    
    def _initialize_pure_simulation_backend(self):
        """初始化纯模拟后端"""
        self.backend_type = "pure_simulation"
        
        # 创建纯模拟后端
        class PureSimulationBackend:
            def __init__(self, logger, config):
                self.logger = logger
                self.config = config
                self.connected = False
                self.robot_joint_indices = {}
                self.robot_joint_limits = {}
                self.robot_sensors = {}
                self.joint_states = {}
                
            def connect(self):
                self.connected = True
                self.logger.info("纯模拟后端已连接")
                return True
                
            def disconnect(self):
                self.connected = False
                self.logger.info("纯模拟后端已断开")
                return True
                
            def get_joint_states(self):
                return self.joint_states.copy()
                
            def set_joint_position(self, joint_name, position, **kwargs):
                self.joint_states[joint_name] = {
                    "position": position,
                    "velocity": 0.0,
                    "effort": 0.0,
                    "timestamp": time.time()
                }
                return True
                
            def get_sensor_data(self, sensor_type=None):
                return {}  # 返回空字典
        
        self.backend = PureSimulationBackend(self.logger, self.engine_config)
        self.logger.info("纯模拟后端初始化成功")
    
    def connect(self) -> bool:
        """连接到仿真环境"""
        try:
            if self.backend is None:
                self.logger.error("仿真后端未初始化")
                return False
            
            success = self.backend.connect()
            if success:
                self.connected = True
                self.logger.info(f"{self.engine}仿真环境连接成功")
            else:
                self.logger.error(f"{self.engine}仿真环境连接失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"连接仿真环境失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        """断开仿真环境连接"""
        try:
            if self.backend is None or not self.connected:
                return True
            
            success = self.backend.disconnect()
            if success:
                self.connected = False
                self.logger.info(f"{self.engine}仿真环境已断开")
            
            return success
            
        except Exception as e:
            self.logger.error(f"断开仿真环境失败: {e}")
            return False
    
    def is_connected(self) -> bool:
        """检查是否连接"""
        return self.connected and hasattr(self.backend, 'connected') and self.backend.connected
    
    def get_joint_states(self) -> Dict[str, JointState]:
        """获取关节状态"""
        try:
            if not self.connected or self.backend is None:
                self.logger.warning("仿真环境未连接，返回空关节状态")
                return {}  # 返回空字典
            
            # 尝试调用后端方法
            if hasattr(self.backend, 'get_joint_states'):
                return self.backend.get_joint_states()
            else:
                # 向后兼容：调用get_sensor_data
                sensor_data = self.get_sensor_data()
                joint_states = {}
                for joint_name, state in sensor_data.get("joint_states", {}).items():
                    joint_states[joint_name] = JointState(**state)
                return joint_states
                
        except Exception as e:
            self.logger.error(f"获取关节状态失败: {e}")
            return {}  # 返回空字典
    
    def set_joint_position(self, 
                          joint_name: str, 
                          position: float,
                          velocity: Optional[float] = None,
                          effort: Optional[float] = None,
                          **kwargs) -> bool:
        """设置关节位置"""
        try:
            if not self.connected or self.backend is None:
                self.logger.warning("仿真环境未连接，无法设置关节位置")
                return False
            
            # 尝试调用后端方法
            if hasattr(self.backend, 'set_joint_position'):
                return self.backend.set_joint_position(joint_name, position, velocity=velocity, effort=effort, **kwargs)
            else:
                # 向后兼容：通过通用接口设置
                return False
                
        except Exception as e:
            self.logger.error(f"设置关节位置失败: {e}")
            return False
    
    def set_joint_positions(self, positions: Dict[str, float], **kwargs) -> bool:
        """批量设置关节位置
        
        参数:
            positions: 关节名称到目标位置的字典映射
            **kwargs: 其他参数（如速度、力矩限制等）
            
        返回:
            bool: 是否成功设置所有关节位置
        """
        try:
            if not self.connected or self.backend is None:
                self.logger.warning("仿真环境未连接，无法设置关节位置")
                return False
            
            # 优先尝试后端方法
            if hasattr(self.backend, 'set_joint_positions'):
                return self.backend.set_joint_positions(positions, **kwargs)
            
            # 如果没有批量设置方法，则逐个设置
            success = True
            for joint_name, position in positions.items():
                if not self.set_joint_position(joint_name, position, **kwargs):
                    self.logger.warning(f"设置关节 {joint_name} 位置失败")
                    success = False
            
            return success
                
        except Exception as e:
            self.logger.error(f"批量设置关节位置失败: {e}")
            return False
    
    def get_sensor_data(self, sensor_type: Optional[str] = None) -> Dict[str, Any]:
        """获取传感器数据"""
        try:
            if not self.connected or self.backend is None:
                self.logger.warning("仿真环境未连接，返回空传感器数据")
                return {}  # 返回空字典
            
            # 尝试调用后端方法
            if hasattr(self.backend, 'get_sensor_data'):
                return self.backend.get_sensor_data(sensor_type)
            else:
                # 返回基本真实数据
                return self._generate_simulated_sensor_data(sensor_type)
                
        except Exception as e:
            self.logger.error(f"获取传感器数据失败: {e}")
            return {}  # 返回空字典
    
    def get_imu_data(self) -> Dict[str, Any]:
        """获取IMU传感器数据
        
        返回:
            Dict[str, Any]: IMU数据，包含加速度、角速度、姿态等信息
        """
        try:
            # 调用get_sensor_data获取IMU数据
            sensor_data = self.get_sensor_data("imu")
            imu_data = sensor_data.get("imu", {})
            
            # 确保返回标准化的数据结构
            if not imu_data:
                # 生成基本的IMU数据
                current_time = time.time()
                imu_data = {
                    "acceleration": [0.0, 0.0, 9.81],
                    "angular_velocity": [0.0, 0.0, 0.0],  # 向后兼容使用angular_velocity而不是gyroscope
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "timestamp": current_time
                }
            elif "gyroscope" in imu_data and "angular_velocity" not in imu_data:
                # 兼容性处理：将gyroscope重命名为angular_velocity
                imu_data["angular_velocity"] = imu_data["gyroscope"]
            
            return imu_data
            
        except Exception as e:
            self.logger.error(f"获取IMU数据失败: {e}")
            current_time = time.time()
            return {
                "acceleration": [0.0, 0.0, 9.81],
                "angular_velocity": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "timestamp": current_time
            }
    
    def _generate_simulated_sensor_data(self, sensor_type: Optional[str] = None) -> Dict[str, Any]:
        """生成模拟传感器数据（纯模拟后端使用）"""
        current_time = time.time()
        
        # 基本IMU数据
        imu_data = {
            "acceleration": [0.0, 0.0, 9.81],
            "gyroscope": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "timestamp": current_time,
            "sensor_type": "imu",
            "sensor_id": "imu_simulated"
        }
        
        # 关节状态数据
        joint_states = {}
        standard_joints = [
            "head_yaw", "head_pitch",
            "shoulder_right", "elbow_right", "wrist_right",
            "shoulder_left", "elbow_left", "wrist_left",
            "hip_right", "knee_right", "ankle_right",
            "hip_left", "knee_left", "ankle_left",
            "torso"
        ]
        
        for joint in standard_joints:
            # 添加微小运动模拟
            breathing_motion = math.sin(current_time * 1.5 + abs(hash(joint)) % 10) * 0.002
            balance_adjustment = math.cos(current_time * 0.8 + abs(hash(joint)) % 7) * 0.001
            
            joint_states[joint] = {
                "position": 0.0 + breathing_motion + balance_adjustment,
                "velocity": 0.0,
                "effort": 0.0,
                "timestamp": current_time
            }
        
        # 构建完整传感器数据
        sensor_data = {
            "imu": imu_data,
            "joint_states": joint_states,
            "timestamp": current_time,
            "sensor_source": f"unified_simulation_{self.engine}",
            "is_simulated": True
        }
        
        # 如果指定了传感器类型，只返回该类型数据
        if sensor_type == "imu":
            return {"imu": imu_data}
        elif sensor_type == "joint_states":
            return {"joint_states": joint_states}
        
        return sensor_data
    
    def move_to_pose(self, target_pose: Dict[str, Any], duration: float = 1.0) -> bool:
        """移动到指定姿态"""
        try:
            if not self.connected or self.backend is None:
                return False
            
            # 尝试调用后端方法
            if hasattr(self.backend, 'move_to_pose'):
                return self.backend.move_to_pose(target_pose, duration)
            else:
                # 简单实现：逐个设置关节位置
                joint_positions = target_pose.get("joint_positions", {})
                success = True
                for joint_name, position in joint_positions.items():
                    if not self.set_joint_position(joint_name, position):
                        success = False
                return success
                
        except Exception as e:
            self.logger.error(f"移动到姿态失败: {e}")
            return False
    
    def step(self) -> bool:
        """执行一个仿真步进
        
        返回:
            bool: 是否成功执行步进
        """
        try:
            if not self.connected or self.backend is None:
                self.logger.warning("仿真环境未连接，无法执行步进")
                return False
            
            # 尝试调用后端方法
            if hasattr(self.backend, 'step'):
                return self.backend.step()
            elif hasattr(self.backend, 'stepSimulation'):
                # PyBullet风格
                return self.backend.stepSimulation()
            else:
                # 对于没有显式步进方法的后端，模拟时间流逝
                # 对于纯模拟后端，只需要更新内部时间
                if hasattr(self.backend, 'simulation_time'):
                    self.backend.simulation_time += self.physics_timestep
                return True
                
        except Exception as e:
            self.logger.error(f"执行仿真步进失败: {e}")
            return False
    
    def reset(self) -> bool:
        """重置仿真环境（reset_simulation的别名）"""
        return self.reset_simulation()
    
    def reset_simulation(self) -> bool:
        """重置仿真环境"""
        try:
            if not self.connected or self.backend is None:
                return False
            
            # 尝试调用后端方法
            if hasattr(self.backend, 'reset_simulation'):
                return self.backend.reset_simulation()
            else:
                # 简单重置：清除关节状态
                if hasattr(self.backend, 'joint_states'):
                    self.backend.joint_states.clear()
                return True
                
        except Exception as e:
            self.logger.error(f"重置仿真环境失败: {e}")
            return False
    
    def get_interface_type(self) -> str:
        """获取接口类型"""
        return self._interface_type
    
    def get_engine_info(self) -> Dict[str, Any]:
        """获取引擎信息"""
        return {
            "engine": self.engine,
            "backend_type": self.backend_type,
            "gui_enabled": self.gui_enabled,
            "physics_timestep": self.physics_timestep,
            "realtime_simulation": self.realtime_simulation,
            "connected": self.connected,
            "backend_available": self.backend is not None
        }


# 兼容性适配器：将原有接口转换为统一接口
class PyBulletCompatibilityAdapter(UnifiedSimulation):
    """PyBullet兼容性适配器"""
    def __init__(self, **kwargs):
        super().__init__(engine="pybullet", **kwargs)

class GazeboCompatibilityAdapter(UnifiedSimulation):
    """Gazebo兼容性适配器"""
    def __init__(self, **kwargs):
        super().__init__(engine="gazebo", **kwargs)