#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一硬件接口规范
提供硬件抽象层的统一接口，支持仿真和真实硬件的无缝切换

功能：
1. 统一的硬件状态管理
2. 统一的配置管理（JSON/YAML）
3. 统一的安全限制和校准
4. 统一的诊断和健康检查
5. 插件式硬件驱动架构
6. 向后兼容现有的HardwareInterface
"""

import sys
import os
import logging
import json
import yaml
import threading
import time
from typing import Dict, Any, List, Optional, Union, Type, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

# 导入基础硬件接口
from .robot_controller import HardwareInterface, RobotJoint, SensorType, JointState

logger = logging.getLogger(__name__)


class HardwareState(Enum):
    """硬件状态枚举"""
    
    UNINITIALIZED = "uninitialized"  # 未初始化
    INITIALIZING = "initializing"  # 初始化中
    CONNECTING = "connecting"  # 连接中
    CONNECTED = "connected"  # 已连接
    READY = "ready"  # 就绪
    RUNNING = "running"  # 运行中
    PAUSED = "paused"  # 已暂停
    ERROR = "error"  # 错误状态
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止
    DISCONNECTING = "disconnecting"  # 断开连接中
    DISCONNECTED = "disconnected"  # 已断开


class HardwareErrorCode(Enum):
    """硬件错误代码枚举"""
    
    NO_ERROR = 0  # 无错误
    CONNECTION_FAILED = 1001  # 连接失败
    COMMUNICATION_ERROR = 1002  # 通信错误
    SENSOR_ERROR = 1003  # 传感器错误
    ACTUATOR_ERROR = 1004  # 执行器错误
    OVER_TEMPERATURE = 1005  # 温度过高
    OVER_CURRENT = 1006  # 电流过大
    OVER_VOLTAGE = 1007  # 电压过高
    UNDER_VOLTAGE = 1008  # 电压过低
    POSITION_LIMIT = 1009  # 位置限制
    VELOCITY_LIMIT = 1010  # 速度限制
    TORQUE_LIMIT = 1011  # 扭矩限制
    CALIBRATION_ERROR = 1012  # 校准错误
    TIMEOUT_ERROR = 1013  # 超时错误
    UNKNOWN_ERROR = 1999  # 未知错误


@dataclass
class HardwareLimits:
    """硬件安全限制"""
    
    # 关节限制
    joint_position_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {joint_name: {"min": -1.57, "max": 1.57}}
    joint_velocity_limits: Dict[str, float] = field(default_factory=dict)  # {joint_name: 3.14} 弧度/秒
    joint_torque_limits: Dict[str, float] = field(default_factory=dict)  # {joint_name: 10.0} Nm
    
    # 系统限制
    max_temperature: float = 80.0  # 最高温度（摄氏度）
    max_current: float = 5.0  # 最大电流（安培）
    min_voltage: float = 10.0  # 最低电压（伏特）
    max_voltage: float = 30.0  # 最高电压（伏特）
    
    # 安全参数
    emergency_stop_timeout: float = 0.5  # 急停超时（秒）
    communication_timeout: float = 2.0  # 通信超时（秒）
    
    def validate_joint_position(self, joint_name: str, position: float) -> bool:
        """验证关节位置是否在限制范围内"""
        if joint_name in self.joint_position_limits:
            limits = self.joint_position_limits[joint_name]
            return limits.get("min", -float('inf')) <= position <= limits.get("max", float('inf'))
        return True  # 如果没有限制，则允许
    
    def validate_joint_velocity(self, joint_name: str, velocity: float) -> bool:
        """验证关节速度是否在限制范围内"""
        if joint_name in self.joint_velocity_limits:
            limit = self.joint_velocity_limits[joint_name]
            return abs(velocity) <= limit
        return True
    
    def validate_joint_torque(self, joint_name: str, torque: float) -> bool:
        """验证关节扭矩是否在限制范围内"""
        if joint_name in self.joint_torque_limits:
            limit = self.joint_torque_limits[joint_name]
            return abs(torque) <= limit
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class HardwareConfig:
    """硬件配置"""
    
    # 基本信息
    hardware_type: str = "unknown"
    hardware_model: str = "unknown"
    hardware_version: str = "1.0.0"
    
    # 连接配置
    connection_params: Dict[str, Any] = field(default_factory=dict)
    
    # 关节配置
    joint_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 传感器配置
    sensor_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 校准参数
    calibration_params: Dict[str, Any] = field(default_factory=dict)
    
    # 安全限制
    limits: HardwareLimits = field(default_factory=HardwareLimits)
    
    # 其他参数
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def save_to_file(self, filepath: str, format: str = "json") -> bool:
        """保存配置到文件"""
        try:
            data = asdict(self)
            # 处理嵌套的dataclass
            data["limits"] = self.limits.to_dict()
            
            if format.lower() == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format.lower() == "yaml":
                import yaml
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, allow_unicode=True)
            else:
                logger.error(f"不支持的格式: {format}")
                return False
            
            logger.info(f"配置已保存到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional["HardwareConfig"]:
        """从文件加载配置"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"配置文件不存在: {filepath}")
                return None  # 返回None
            
            # 根据文件扩展名确定格式
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext in ['.json', '.jsonc']:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif ext in ['.yaml', '.yml']:
                import yaml
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                logger.error(f"不支持的配置文件格式: {ext}")
                return None  # 返回None
            
            # 创建配置对象
            config = cls(
                hardware_type=data.get("hardware_type", "unknown"),
                hardware_model=data.get("hardware_model", "unknown"),
                hardware_version=data.get("hardware_version", "1.0.0"),
                connection_params=data.get("connection_params", {}),
                joint_configs=data.get("joint_configs", {}),
                sensor_configs=data.get("sensor_configs", {}),
                calibration_params=data.get("calibration_params", {}),
                additional_params=data.get("additional_params", {})
            )
            
            # 加载限制配置
            limits_data = data.get("limits", {})
            if limits_data:
                config.limits = HardwareLimits(**limits_data)
            
            logger.info(f"配置已从文件加载: {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return None  # 返回None


@dataclass
class HardwareDiagnostics:
    """硬件诊断信息"""
    
    # 状态信息
    state: HardwareState = HardwareState.UNINITIALIZED
    error_code: HardwareErrorCode = HardwareErrorCode.NO_ERROR
    error_message: str = ""
    
    # 性能指标
    uptime: float = 0.0  # 运行时间（秒）
    communication_latency: float = 0.0  # 通信延迟（毫秒）
    update_frequency: float = 0.0  # 更新频率（Hz）
    
    # 硬件指标
    temperature: float = 25.0  # 温度（摄氏度）
    voltage: float = 24.0  # 电压（伏特）
    current: float = 0.0  # 电流（安培）
    
    # 关节状态统计
    joint_errors: Dict[str, str] = field(default_factory=dict)
    joint_warnings: Dict[str, str] = field(default_factory=dict)
    
    # 时间戳
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "state": self.state.value,
            "error_code": self.error_code.value,
            "error_message": self.error_message,
            "uptime": self.uptime,
            "communication_latency": self.communication_latency,
            "update_frequency": self.update_frequency,
            "temperature": self.temperature,
            "voltage": self.voltage,
            "current": self.current,
            "joint_errors": self.joint_errors,
            "joint_warnings": self.joint_warnings,
            "timestamp": self.timestamp
        }
    
    def is_healthy(self) -> bool:
        """检查硬件是否健康"""
        return (
            self.state == HardwareState.READY or 
            self.state == HardwareState.RUNNING
        ) and self.error_code == HardwareErrorCode.NO_ERROR


# ============================================================================
# 增强型硬件接口功能（整合自enhanced_interface.py）
# ============================================================================

class OperationMode(Enum):
    """操作模式（来自enhanced_interface.py）"""
    NORMAL = "normal"        # 正常模式
    SAFE = "safe"           # 安全模式（限制功率和速度）
    RECOVERY = "recovery"   # 恢复模式（尝试自我恢复）
    MAINTENANCE = "maintenance"  # 维护模式
    EMERGENCY = "emergency"  # 紧急模式


@dataclass
class OperationLimits:
    """操作限制（来自enhanced_interface.py）"""
    max_velocity: float = 1.0  # 最大速度（归一化）
    max_acceleration: float = 0.5  # 最大加速度（归一化）
    max_torque: float = 0.8  # 最大扭矩（归一化）
    position_tolerance: float = 0.01  # 位置容差
    temperature_limit: float = 70.0  # 温度限制（摄氏度）
    power_limit: float = 0.8  # 功率限制（归一化）


@dataclass
class SafetyConfig:
    """安全配置（来自enhanced_interface.py）"""
    emergency_stop_on_error: bool = True
    auto_recovery_enabled: bool = True
    max_recovery_attempts: int = 3
    recovery_timeout: float = 10.0  # 恢复超时（秒）
    safety_margin: float = 0.1  # 安全裕度


class UnifiedHardwareInterface(HardwareInterface):
    """统一硬件接口（扩展自HardwareInterface）"""
    
    def __init__(self, 
                 config: Optional[HardwareConfig] = None,
                 simulation_mode: bool = False,
                 enable_safety: bool = True):
        """
        初始化统一硬件接口
        
        参数:
            config: 硬件配置
            simulation_mode: 是否为模拟模式
            enable_safety: 是否启用安全限制
        """
        super().__init__(simulation_mode=simulation_mode)
        
        # 配置和状态
        self.config = config or HardwareConfig()
        self._state = HardwareState.UNINITIALIZED
        self._error_code = HardwareErrorCode.NO_ERROR
        self._error_message = ""
        self._enable_safety = enable_safety
        
        # 诊断信息
        self.diagnostics = HardwareDiagnostics()
        self._start_time = time.time()
        self._update_count = 0
        self._last_update_time = time.time()
        
        # 线程安全
        self._lock = threading.RLock()
        self._state_listeners: List[Callable[[HardwareState], None]] = []
        
        # ============ 增强型硬件接口属性（来自enhanced_interface.py）============
        # 操作状态
        self.operation_mode = OperationMode.NORMAL
        self.operation_limits = OperationLimits()
        self.safety_config = SafetyConfig()
        
        # 故障恢复
        self.recovery_in_progress = False
        self.recovery_attempts = 0
        self.last_error_time = 0.0
        
        # 状态跟踪（增强功能）
        self.joint_command_history: Dict[RobotJoint, List[Tuple[float, float]]] = {}
        self.sensor_data_history: Dict[SensorType, List[Tuple[float, Any]]] = {}
        self.operation_log: List[Dict[str, Any]] = []
        
        # 锁（增强）
        self._operation_lock = threading.RLock()
        self._recovery_lock = threading.Lock()
        
        # 监控器（可选）
        self.monitor = None
        
        # 初始化历史记录
        self._initialize_history()
        # ============ 增强属性结束 ============
        
        # 更新接口类型
        self._interface_type = f"unified_{self.config.hardware_type}"
        
        logger.info(f"统一硬件接口初始化完成，类型: {self.config.hardware_type}，增强功能已集成")
    
    def _initialize_history(self):
        """初始化历史记录（来自enhanced_interface.py）"""
        # 初始化关节命令历史
        for joint in RobotJoint:
            self.joint_command_history[joint] = []
        
        # 初始化传感器数据历史
        for sensor_type in SensorType:
            self.sensor_data_history[sensor_type] = []
        
        logger.debug("硬件接口历史记录已初始化")
    
    @property
    def state(self) -> HardwareState:
        """获取硬件状态"""
        with self._lock:
            return self._state
    
    @state.setter
    def state(self, new_state: HardwareState):
        """设置硬件状态（触发状态监听器）"""
        with self._lock:
            old_state = self._state
            self._state = new_state
            self.diagnostics.state = new_state
        
        # 触发状态变化监听器
        if old_state != new_state:
            logger.info(f"硬件状态变化: {old_state.value} -> {new_state.value}")
            self._notify_state_listeners(new_state)
    
    @property
    def error_code(self) -> HardwareErrorCode:
        """获取错误代码"""
        with self._lock:
            return self._error_code
    
    @error_code.setter
    def error_code(self, code: HardwareErrorCode):
        """设置错误代码"""
        with self._lock:
            self._error_code = code
            self.diagnostics.error_code = code
    
    @property
    def error_message(self) -> str:
        """获取错误消息"""
        with self._lock:
            return self._error_message
    
    @error_message.setter
    def error_message(self, message: str):
        """设置错误消息"""
        with self._lock:
            self._error_message = message
            self.diagnostics.error_message = message
    
    def set_error(self, code: HardwareErrorCode, message: str = ""):
        """设置错误状态"""
        self.error_code = code
        self.error_message = message
        self.state = HardwareState.ERROR
        logger.error(f"硬件错误 [{code.value}]: {message}")
    
    def clear_error(self):
        """清除错误状态"""
        self.error_code = HardwareErrorCode.NO_ERROR
        self.error_message = ""
        if self.state == HardwareState.ERROR:
            self.state = HardwareState.READY
        logger.info("硬件错误已清除")
    
    def add_state_listener(self, listener: Callable[[HardwareState], None]):
        """添加状态监听器"""
        with self._lock:
            self._state_listeners.append(listener)
    
    def remove_state_listener(self, listener: Callable[[HardwareState], None]):
        """移除状态监听器"""
        with self._lock:
            if listener in self._state_listeners:
                self._state_listeners.remove(listener)
    
    def _notify_state_listeners(self, state: HardwareState):
        """通知状态监听器"""
        listeners = self._state_listeners.copy()
        for listener in listeners:
            try:
                listener(state)
            except Exception as e:
                logger.error(f"状态监听器执行失败: {e}")
    
    def _update_diagnostics(self):
        """更新诊断信息"""
        current_time = time.time()
        self.diagnostics.uptime = current_time - self._start_time
        self.diagnostics.timestamp = current_time
        
        # 计算更新频率
        time_since_last_update = current_time - self._last_update_time
        if time_since_last_update > 0:
            self.diagnostics.update_frequency = 1.0 / time_since_last_update
        
        self._last_update_time = current_time
        self._update_count += 1
    
    def connect(self) -> bool:
        """连接硬件（覆盖基类方法）"""
        try:
            self.state = HardwareState.CONNECTING
            
            # 调用具体的连接实现
            connected = self._connect_impl()
            
            if connected:
                self.state = HardwareState.CONNECTED
                logger.info("硬件连接成功")
                
                # 执行初始化
                initialized = self.initialize()
                if initialized:
                    self.state = HardwareState.READY
                    logger.info("硬件初始化完成，已就绪")
                else:
                    self.set_error(HardwareErrorCode.INITIALIZATION_ERROR, "硬件初始化失败")
                    return False
            else:
                self.set_error(HardwareErrorCode.CONNECTION_FAILED, "硬件连接失败")
                return False
            
            return True
            
        except Exception as e:
            self.set_error(HardwareErrorCode.UNKNOWN_ERROR, f"连接过程中发生异常: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """断开硬件连接（覆盖基类方法）"""
        try:
            self.state = HardwareState.DISCONNECTING
            
            # 调用具体的断开连接实现
            disconnected = self._disconnect_impl()
            
            if disconnected:
                self.state = HardwareState.DISCONNECTED
                logger.info("硬件已断开连接")
                return True
            else:
                logger.error("断开连接失败")
                return False
                
        except Exception as e:
            logger.error(f"断开连接过程中发生异常: {e}")
            self.state = HardwareState.ERROR
            return False
    
    def is_connected(self) -> bool:
        """检查是否连接（覆盖基类方法）"""
        return self.state in [HardwareState.CONNECTED, HardwareState.READY, HardwareState.RUNNING]
    
    def initialize(self) -> bool:
        """初始化硬件（校准、自检等）"""
        try:
            logger.info("开始硬件初始化...")
            
            # 1. 自检
            self.self_test()
            
            # 2. 校准（如果配置了校准参数）
            if self.config.calibration_params:
                self.calibrate()
            
            # 3. 移动到安全位置
            self.move_to_safe_pose()
            
            logger.info("硬件初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"硬件初始化失败: {e}")
            self.set_error(HardwareErrorCode.CALIBRATION_ERROR, f"初始化失败: {str(e)}")
            return False
    
    def self_test(self) -> Dict[str, Any]:
        """硬件自检"""
        test_results = {
            "passed": True,
            "tests": [],
            "errors": []
        }
        
        logger.info("执行硬件自检...")
        
        # 检查连接
        if not self.is_connected():
            test_results["passed"] = False
            test_results["errors"].append("硬件未连接")
            logger.error("自检失败: 硬件未连接")
            return test_results
        
        # 检查传感器（如果启用）
        if self.sensor_enabled:
            try:
                sensor_data = self.get_sensor_data(SensorType.IMU)
                if sensor_data:
                    test_results["tests"].append({"name": "IMU传感器", "status": "passed"})
                else:
                    test_results["tests"].append({"name": "IMU传感器", "status": "failed"})
                    test_results["errors"].append("IMU传感器无数据")
            except Exception as e:
                test_results["tests"].append({"name": "IMU传感器", "status": "error"})
                test_results["errors"].append(f"IMU传感器错误: {str(e)}")
        
        # 检查关节
        try:
            joint_states = self.get_all_joint_states()
            if joint_states:
                test_results["tests"].append({"name": "关节状态", "status": "passed"})
            else:
                test_results["tests"].append({"name": "关节状态", "status": "failed"})
                test_results["errors"].append("无法获取关节状态")
        except Exception as e:
            test_results["tests"].append({"name": "关节状态", "status": "error"})
            test_results["errors"].append(f"关节状态错误: {str(e)}")
        
        # 更新通过状态
        if test_results["errors"]:
            test_results["passed"] = False
            logger.warning(f"自检发现错误: {test_results['errors']}")
        else:
            logger.info("自检通过")
        
        return test_results
    
    def calibrate(self) -> bool:
        """校准硬件"""
        logger.info("开始硬件校准...")
        
        # 这里应该实现具体的校准逻辑
        # 例如：零点校准、传感器校准等
        
        logger.info("硬件校准完成（模拟）")
        return True
    
    def move_to_safe_pose(self) -> bool:
        """移动到安全姿势"""
        logger.info("移动到安全姿势...")
        
        try:
            # 获取安全姿势配置
            safe_pose = self._get_safe_pose()
            
            # 移动到安全姿势
            success = self.set_joint_positions(safe_pose)
            
            if success:
                logger.info("已移动到安全姿势")
                return True
            else:
                logger.error("移动到安全姿势失败")
                return False
                
        except Exception as e:
            logger.error(f"移动到安全姿势失败: {e}")
            return False
    
    def _get_safe_pose(self) -> Dict[RobotJoint, float]:
        """获取安全姿势（子类可以覆盖）"""
        # 默认返回所有关节为零的位置
        safe_pose = {}
        for joint in RobotJoint:
            safe_pose[joint] = 0.0
        return safe_pose
    
    def emergency_stop(self) -> bool:
        """紧急停止"""
        logger.error("执行紧急停止！")
        
        try:
            self.state = HardwareState.EMERGENCY_STOP
            
            # 停止所有运动
            success = self._emergency_stop_impl()
            
            if success:
                logger.info("紧急停止成功")
            else:
                logger.error("紧急停止失败")
            
            return success
            
        except Exception as e:
            logger.error(f"紧急停止过程中发生异常: {e}")
            return False
    
    def get_diagnostics(self) -> HardwareDiagnostics:
        """获取诊断信息"""
        self._update_diagnostics()
        return self.diagnostics
    
    def validate_joint_command(self, joint: RobotJoint, position: float) -> bool:
        """验证关节命令是否安全"""
        if not self._enable_safety:
            return True
        
        # 检查关节位置限制
        joint_name = joint.value
        if not self.config.limits.validate_joint_position(joint_name, position):
            logger.warning(f"关节 {joint_name} 位置超出限制: {position}")
            return False
        
        # 可以添加更多安全检查
        # 例如：速度限制、扭矩限制等
        
        return True
    
    # 以下方法提供默认实现（模拟模式）
    def _connect_impl(self) -> bool:
        """具体的连接实现（模拟模式默认实现）"""
        if self._simulation_mode:
            self.logger.info("模拟模式：连接实现")
            return True
        else:
            self.logger.error("真实硬件模式：连接方法必须由子类实现")
            return False
    
    def _disconnect_impl(self) -> bool:
        """具体的断开连接实现（模拟模式默认实现）"""
        if self._simulation_mode:
            self.logger.info("模拟模式：断开连接实现")
            return True
        else:
            self.logger.error("真实硬件模式：断开连接方法必须由子类实现")
            return False
    
    def _emergency_stop_impl(self) -> bool:
        """具体的紧急停止实现（模拟模式默认实现）"""
        if self._simulation_mode:
            self.logger.info("模拟模式：紧急停止实现")
            # 模拟紧急停止：将所有关节位置设为0
            for joint in RobotJoint:
                self.set_joint_position(joint, 0.0)
            return True
        else:
            self.logger.error("真实硬件模式：紧急停止方法必须由子类实现")
            return False
    
    # 以下方法继承自HardwareInterface，子类可以覆盖
    def get_sensor_data(self, sensor_type: SensorType) -> Optional[Any]:
        """获取传感器数据（子类可以覆盖）"""
        # 默认实现：调用基类方法
        return super().get_sensor_data(sensor_type)
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置（添加安全检查）"""
        # 安全检查
        if not self.validate_joint_command(joint, position):
            return False
        
        # 调用基类方法
        return super().set_joint_position(joint, position)
    
    def set_joint_positions(self, positions: Dict[RobotJoint, float]) -> bool:
        """设置多个关节位置（添加安全检查）"""
        # 安全检查每个关节
        for joint, position in positions.items():
            if not self.validate_joint_command(joint, position):
                return False
        
        # 调用基类方法
        return super().set_joint_positions(positions)
    
    def get_joint_state(self, joint: RobotJoint) -> Optional[JointState]:
        """获取关节状态（子类可以覆盖）"""
        return super().get_joint_state(joint)
    
    def get_all_joint_states(self) -> Dict[RobotJoint, JointState]:
        """获取所有关节状态（子类可以覆盖）"""
        return super().get_all_joint_states()

    # ============================================================================
    # 增强型硬件接口方法（来自enhanced_interface.py）
    # ============================================================================

    def set_operation_mode(self, mode: OperationMode):
        """设置操作模式（来自enhanced_interface.py）"""
        with self._operation_lock:
            old_mode = self.operation_mode
            self.operation_mode = mode
            
            # 根据模式调整限制
            if mode == OperationMode.SAFE:
                self.operation_limits.max_velocity = 0.5
                self.operation_limits.max_torque = 0.5
                self.operation_limits.power_limit = 0.6
            elif mode == OperationMode.RECOVERY:
                self.operation_limits.max_velocity = 0.2
                self.operation_limits.max_torque = 0.3
                self.operation_limits.power_limit = 0.4
            elif mode == OperationMode.EMERGENCY:
                self.operation_limits.max_velocity = 0.0
                self.operation_limits.max_torque = 0.0
                self.operation_limits.power_limit = 0.0
            else:  # NORMAL or MAINTENANCE
                self.operation_limits.max_velocity = 1.0
                self.operation_limits.max_torque = 0.8
                self.operation_limits.power_limit = 0.8
            
            logger.info(f"操作模式已更改: {old_mode.value} -> {mode.value}")
            
            # 记录操作日志
            self._log_operation(
                "set_operation_mode",
                {"old_mode": old_mode.value, "new_mode": mode.value}
            )

    def enable_monitoring(self):
        """启用硬件监控"""
        if self.monitor is None:
            try:
                from .hardware_monitor import HardwareMonitor, create_default_safety_rules
                self.monitor = HardwareMonitor(self)
                create_default_safety_rules(self.monitor)
                self.monitor.start_monitoring()
                logger.info("硬件监控已启用")
            except ImportError as e:
                logger.error(f"无法导入硬件监控模块: {e}")
                logger.warning("监控功能不可用，请在环境中安装必要依赖")
                self.monitor = None
            except Exception as e:
                logger.error(f"启用监控失败: {e}")
                self.monitor = None
        else:
            logger.info("硬件监控已启用")

    def disable_monitoring(self):
        """禁用硬件监控"""
        if self.monitor:
            try:
                self.monitor.stop_monitoring()
                logger.info("硬件监控已停止")
            except Exception as e:
                logger.error(f"停止监控失败: {e}")
            finally:
                self.monitor = None
        logger.info("硬件监控已禁用")

    def _log_operation(self, operation: str, details: Dict[str, Any]):
        """记录操作日志（来自enhanced_interface.py）"""
        log_entry = {
            "timestamp": time.time(),
            "operation": operation,
            "operation_mode": self.operation_mode.value,
            "connected": self.is_connected(),
            "details": details
        }
        
        self.operation_log.append(log_entry)
        
        # 限制日志大小
        if len(self.operation_log) > 10000:
            self.operation_log = self.operation_log[-10000:]

    def get_operation_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取操作日志（来自enhanced_interface.py）"""
        return self.operation_log[-limit:]

    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告（来自enhanced_interface.py）"""
        report = {
            "timestamp": time.time(),
            "interface_type": self._interface_type,
            "simulation_mode": self.is_simulation,
            "operation_mode": self.operation_mode.value,
            "connected": self.is_connected(),
            "sensor_enabled": self.sensor_enabled,
            "recovery_attempts": self.recovery_attempts,
            "recovery_in_progress": self.recovery_in_progress,
            "operation_log_summary": {
                "total_entries": len(self.operation_log),
                "recent_operations": [op["operation"] for op in self.operation_log[-10:]]
            }
        }
        
        # 如果监控器存在，添加监控报告
        if self.monitor:
            try:
                report["monitoring"] = self.monitor.get_status_report()
            except AttributeError:
                report["monitoring"] = {"enabled": True, "status": "monitor_available_but_no_status_method"}
            except Exception as e:
                report["monitoring"] = {"enabled": True, "status": f"error: {str(e)}"}
        
        return report

    # 增强的安全检查方法（需要时实现）
    def _check_safety_constraints(self, joint: RobotJoint, position: float) -> bool:
        """检查安全约束 - 基本位置限制检查"""
        # 定义关节位置限制（弧度）
        # 完整实现应根据具体机器人型号调整
        position_limits = {
            # 头部关节：-π/2 到 π/2
            RobotJoint.HEAD_YAW: (-1.57, 1.57),      # -90° 到 90°
            RobotJoint.HEAD_PITCH: (-1.57, 1.57),    # -90° 到 90°
            
            # 肩部关节：-π 到 π
            RobotJoint.L_SHOULDER_PITCH: (-3.14, 3.14),
            RobotJoint.L_SHOULDER_ROLL: (-3.14, 3.14),
            RobotJoint.R_SHOULDER_PITCH: (-3.14, 3.14),
            RobotJoint.R_SHOULDER_ROLL: (-3.14, 3.14),
            
            # 肘部关节：-π/2 到 π/2
            RobotJoint.L_ELBOW_YAW: (-1.57, 1.57),
            RobotJoint.L_ELBOW_ROLL: (-1.57, 1.57),
            RobotJoint.R_ELBOW_YAW: (-1.57, 1.57),
            RobotJoint.R_ELBOW_ROLL: (-1.57, 1.57),
            
            # 手腕关节：-π/2 到 π/2
            RobotJoint.L_WRIST_YAW: (-1.57, 1.57),
            RobotJoint.R_WRIST_YAW: (-1.57, 1.57),
            
            # 手部关节：0 到 1（抓握程度）
            RobotJoint.L_HAND: (0.0, 1.0),
            RobotJoint.R_HAND: (0.0, 1.0),
            
            # 腿部关节：根据人形机器人典型限制
            RobotJoint.L_HIP_YAW_PITCH: (-1.57, 1.57),
            RobotJoint.L_HIP_ROLL: (-0.79, 0.79),     # -45° 到 45°
            RobotJoint.L_HIP_PITCH: (-1.57, 1.57),
            RobotJoint.L_KNEE_PITCH: (0.0, 2.09),     # 0° 到 120°
            RobotJoint.L_ANKLE_PITCH: (-0.79, 0.79),
            RobotJoint.L_ANKLE_ROLL: (-0.79, 0.79),
            
            RobotJoint.R_HIP_YAW_PITCH: (-1.57, 1.57),
            RobotJoint.R_HIP_ROLL: (-0.79, 0.79),
            RobotJoint.R_HIP_PITCH: (-1.57, 1.57),
            RobotJoint.R_KNEE_PITCH: (0.0, 2.09),
            RobotJoint.R_ANKLE_PITCH: (-0.79, 0.79),
            RobotJoint.R_ANKLE_ROLL: (-0.79, 0.79),
        }
        
        # 获取该关节的限制
        if joint in position_limits:
            min_pos, max_pos = position_limits[joint]
            if position < min_pos or position > max_pos:
                logger.warning(f"安全约束检查失败: 关节 {joint.value} 位置 {position:.3f} 超出范围 [{min_pos:.3f}, {max_pos:.3f}]")
                return False
        
        # 默认通过检查
        return True

    def _apply_operation_limits(self, joint: RobotJoint, position: float) -> float:
        """应用操作限制 - 基本速度限制和位置平滑"""
        # 初始化位置历史记录（如果不存在）
        if not hasattr(self, '_last_positions'):
            self._last_positions = {}
            self._last_times = {}
            self._max_position_change = 0.1  # 最大位置变化（弧度/次）
            self._max_velocity = 1.0  # 最大速度（弧度/秒）
        
        current_time = time.time()
        joint_key = joint.value
        
        # 获取上次位置和时间
        last_position = self._last_positions.get(joint_key)
        last_time = self._last_times.get(joint_key)
        
        # 应用速度限制
        limited_position = position
        
        if last_position is not None and last_time is not None:
            # 计算时间差
            time_diff = current_time - last_time
            
            if time_diff > 0:
                # 计算请求的速度
                requested_velocity = abs(position - last_position) / time_diff
                
                # 如果速度超过限制，限制位置变化
                if requested_velocity > self._max_velocity:
                    max_change = self._max_velocity * time_diff
                    direction = 1 if position > last_position else -1
                    limited_position = last_position + direction * min(abs(position - last_position), max_change)
                    logger.debug(f"速度限制: 关节 {joint_key} 速度 {requested_velocity:.3f} rad/s > 限制 {self._max_velocity:.3f}, 限制位置变化")
        
        # 应用最大位置变化限制（即使没有历史数据）
        if last_position is not None:
            position_diff = limited_position - last_position
            if abs(position_diff) > self._max_position_change:
                direction = 1 if position_diff > 0 else -1
                limited_position = last_position + direction * self._max_position_change
                logger.debug(f"位置变化限制: 关节 {joint_key} 变化 {position_diff:.3f} > 限制 {self._max_position_change:.3f}")
        
        # 更新历史记录
        self._last_positions[joint_key] = limited_position
        self._last_times[joint_key] = current_time
        
        return limited_position

    # ============================================================================
    # 增强方法结束
    # ============================================================================


class UnifiedHardwareManager:
    """统一硬件管理器"""
    
    def __init__(self):
        self.interfaces: Dict[str, UnifiedHardwareInterface] = {}
        self.configs: Dict[str, HardwareConfig] = {}
        self.lock = threading.RLock()
        logger.info("统一硬件管理器初始化完成")
    
    def register_interface(self, 
                          name: str, 
                          interface: UnifiedHardwareInterface,
                          config: Optional[HardwareConfig] = None) -> bool:
        """注册硬件接口"""
        with self.lock:
            if name in self.interfaces:
                logger.warning(f"接口 {name} 已存在，将被替换")
            
            self.interfaces[name] = interface
            
            if config:
                self.configs[name] = config
            
            logger.info(f"硬件接口 {name} 已注册，类型: {interface.config.hardware_type}")
            return True
    
    def connect_interface(self, name: str) -> bool:
        """连接硬件接口"""
        with self.lock:
            interface = self.interfaces.get(name)
            if not interface:
                logger.error(f"接口 {name} 不存在")
                return False
            
            return interface.connect()
    
    def disconnect_interface(self, name: str) -> bool:
        """断开硬件接口连接"""
        with self.lock:
            interface = self.interfaces.get(name)
            if not interface:
                logger.error(f"接口 {name} 不存在")
                return False
            
            return interface.disconnect()
    
    def get_interface(self, name: str) -> Optional[UnifiedHardwareInterface]:
        """获取硬件接口"""
        with self.lock:
            return self.interfaces.get(name)
    
    def list_interfaces(self) -> Dict[str, Dict[str, Any]]:
        """列出所有接口信息"""
        with self.lock:
            result = {}
            for name, interface in self.interfaces.items():
                result[name] = {
                    "type": interface.config.hardware_type,
                    "model": interface.config.hardware_model,
                    "state": interface.state.value,
                    "error_code": interface.error_code.value,
                    "connected": interface.is_connected(),
                    "simulation": interface.is_simulation,
                    "diagnostics": interface.get_diagnostics().to_dict()
                }
            return result
    
    def emergency_stop_all(self) -> Dict[str, bool]:
        """紧急停止所有接口"""
        results = {}
        with self.lock:
            for name, interface in self.interfaces.items():
                try:
                    success = interface.emergency_stop()
                    results[name] = success
                except Exception as e:
                    logger.error(f"接口 {name} 紧急停止失败: {e}")
                    results[name] = False
        return results
    
    def load_config_from_file(self, name: str, filepath: str) -> bool:
        """从文件加载配置并创建接口"""
        try:
            config = HardwareConfig.load_from_file(filepath)
            if not config:
                return False
            
            # 根据配置创建接口
            interface = self._create_interface_from_config(config)
            if not interface:
                return False
            
            # 注册接口
            return self.register_interface(name, interface, config)
            
        except Exception as e:
            logger.error(f"从文件加载配置失败: {e}")
            return False
    
    def _create_interface_from_config(self, config: HardwareConfig) -> Optional[UnifiedHardwareInterface]:
        """根据配置创建硬件接口"""
        # 这里应该根据硬件类型创建相应的接口
        # 例如：nao、pepper、gazebo、pybullet等
        
        hardware_type = config.hardware_type.lower()
        
        if hardware_type in ["nao", "pepper"]:
            # 创建NAOqi接口
            from .real_robot_interface import NAOqiRobotInterface, RobotConnectionConfig
            # 需要从config.connection_params构建RobotConnectionConfig
            # 完整处理
            try:
                conn_config = RobotConnectionConfig(
                    robot_type=hardware_type,
                    host=config.connection_params.get("host", "localhost"),
                    port=config.connection_params.get("port", 9559)
                )
                interface = NAOqiRobotInterface(conn_config)
                interface.config = config  # 替换配置
                return interface
            except Exception as e:
                logger.error(f"创建NAOqi接口失败: {e}")
                return None  # 返回None
        
        elif hardware_type in ["gazebo", "pybullet"]:
            # 创建仿真接口（优先使用统一仿真接口）
            try:
                from .unified_simulation import UnifiedSimulation
                
                # 构建引擎配置
                engine_config = config.connection_params.copy()
                
                interface = UnifiedSimulation(
                    engine=hardware_type,  # "gazebo" 或 "pybullet"
                    engine_config=engine_config,
                    gui_enabled=config.connection_params.get("gui_enabled", True),
                    simulation_mode=True  # 总是仿真模式
                )
                
                interface.config = config
                logger.info(f"使用统一仿真接口创建{hardware_type}仿真环境")
                return interface
                
            except ImportError as e:
                logger.warning(f"统一仿真接口不可用: {e}，使用原始仿真模块（向后兼容）")
                # 回退到原始仿真模块
                from .simulation import PyBulletSimulation
                from .gazebo_simulation import GazeboSimulation
                
                if hardware_type == "gazebo":
                    interface = GazeboSimulation(
                        ros_master_uri=config.connection_params.get("ros_master_uri", "http://localhost:11311"),
                        gazebo_world=config.connection_params.get("gazebo_world", "empty.world"),
                        robot_model=config.connection_params.get("robot_model", "humanoid")
                    )
                else:
                    interface = PyBulletSimulation(
                        gui_enabled=config.connection_params.get("gui_enabled", True)
                    )
                
                interface.config = config
                logger.info(f"使用原始{hardware_type}仿真模块（向后兼容）")
                return interface
            except Exception as e:
                logger.error(f"创建{hardware_type}仿真接口失败: {e}")
                return None  # 返回None
        
        else:
            logger.error(f"不支持的硬件类型: {hardware_type}")
            return None  # 返回None


# 全局硬件管理器实例
_unified_hardware_manager = None

def get_unified_hardware_manager() -> UnifiedHardwareManager:
    """获取全局统一硬件管理器实例"""
    global _unified_hardware_manager
    if _unified_hardware_manager is None:
        _unified_hardware_manager = UnifiedHardwareManager()
    return _unified_hardware_manager


# ============================================================================
# 兼容层：EnhancedHardwareInterface（向后兼容性）
# ============================================================================

class EnhancedHardwareInterface:
    """增强型硬件接口（兼容层，包装UnifiedHardwareInterface）
    
    注意：为了保持向后兼容性，这个类提供了原始enhanced_interface.py的接口。
    新代码应该直接使用UnifiedHardwareInterface类。
    """
    
    def __init__(self, base_interface: HardwareInterface):
        """初始化增强接口
        
        参数:
            base_interface: 基础硬件接口，应该是一个UnifiedHardwareInterface实例
        """
        # 检查base_interface是否为UnifiedHardwareInterface
        if not isinstance(base_interface, UnifiedHardwareInterface):
            logger.warning(f"基础接口不是UnifiedHardwareInterface: {type(base_interface)}")
        
        self.base = base_interface
        self.monitor = None
        
        # 设置操作模式为NORMAL
        self.set_operation_mode(OperationMode.NORMAL)
        
        logger.info(f"增强型硬件接口（兼容层）已创建，基础接口: {base_interface._interface_type}")
    
    def set_operation_mode(self, mode: OperationMode):
        """设置操作模式（转发到基础接口）"""
        self.base.set_operation_mode(mode)
    
    def enable_monitoring(self):
        """启用监控（转发到基础接口）"""
        self.base.enable_monitoring()
    
    def disable_monitoring(self):
        """禁用监控（转发到基础接口）"""
        self.base.disable_monitoring()
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告（转发到基础接口）"""
        return self.base.get_status_report()
    
    def get_operation_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取操作日志（转发到基础接口）"""
        return self.base.get_operation_log(limit)
    
    def __getattr__(self, name):
        """转发已实现的方法到基础接口"""
        return getattr(self.base, name)
    
    def __repr__(self):
        return f"EnhancedHardwareInterface(base={self.base._interface_type})"


# 导出OperationMode以保持向后兼容性
__all__ = [
    "UnifiedHardwareInterface",
    "UnifiedHardwareManager", 
    "HardwareConfig",
    "HardwareLimits",
    "HardwareDiagnostics",
    "HardwareState",
    "HardwareErrorCode",
    "OperationMode",
    "OperationLimits",
    "SafetyConfig",
    "EnhancedHardwareInterface",  # 兼容导出
    "get_unified_hardware_manager"
]