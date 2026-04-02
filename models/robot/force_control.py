"""力控制系统

实现机器人力控制闭环，包括：
1. 阻抗控制
2. 导纳控制
3. 力/位混合控制
4. 自适应力控制
"""

import numpy as np
import time
import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging


class ForceControlType(Enum):
    """力控制类型"""
    IMPEDANCE = "impedance"          # 阻抗控制
    ADMITTANCE = "admittance"        # 导纳控制
    HYBRID = "hybrid"                # 力/位混合控制
    DIRECT_FORCE = "direct_force"    # 直接力控制


class ForceSensor:
    """力传感器模拟"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("ForceSensor")
        
        # 传感器参数
        self.params = {
            "range": self.config.get("range", [-100, 100]),  # 测量范围 (N)
            "resolution": self.config.get("resolution", 0.01),  # 分辨率 (N)
            "noise_level": self.config.get("noise_level", 0.1),  # 噪声水平
            "sampling_rate": self.config.get("sampling_rate", 1000),  # 采样率 (Hz)
            "bias": self.config.get("bias", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 零偏
        }
        
        # 状态变量
        self.force_torque = np.zeros(6)  # [Fx, Fy, Fz, Tx, Ty, Tz]
        self.is_calibrated = False
        self.calibration_matrix = np.eye(6)  # 校准矩阵
        self.last_update_time = 0
        
        # 数据缓冲区
        self.data_buffer: List[np.ndarray] = []
        self.max_buffer_size = 1000
        
        self.logger.info("力传感器初始化完成")
    
    def calibrate(self) -> bool:
        """校准传感器"""
        try:
            self.logger.info("开始力传感器校准")
            
            # 模拟校准过程
            time.sleep(0.5)
            
            # 设置零偏
            self.params["bias"] = [0.0] * 6
            
            # 设置校准矩阵
            self.calibration_matrix = np.eye(6)
            
            self.is_calibrated = True
            self.logger.info("力传感器校准完成")
            return True
            
        except Exception as e:
            self.logger.error(f"力传感器校准失败: {e}")
            return False
    
    def read_raw(self) -> np.ndarray:
        """读取原始数据"""
        # 模拟传感器读数
        import random
        
        # 生成模拟力/力矩数据
        raw_data = np.zeros(6)
        
        # 添加真实信号（如果有接触）
        contact_force = self._simulate_contact()
        
        # 添加传感器噪声
        noise = np.random.normal(0, self.params["noise_level"], 6)
        
        raw_data = contact_force + noise
        
        # 添加零偏
        raw_data += np.array(self.params["bias"])
        
        # 限制范围
        for i in range(6):
            min_val, max_val = self.params["range"][0], self.params["range"][1]
            raw_data[i] = np.clip(raw_data[i], min_val, max_val)
        
        return raw_data
    
    def read_calibrated(self) -> np.ndarray:
        """读取校准后的数据"""
        raw_data = self.read_raw()
        
        if self.is_calibrated:
            # 应用校准矩阵
            calibrated_data = self.calibration_matrix @ raw_data
        else:
            calibrated_data = raw_data
        
        # 更新当前读数
        self.force_torque = calibrated_data
        self.last_update_time = time.time()
        
        # 添加到缓冲区
        self.data_buffer.append(calibrated_data.copy())
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size:]
        
        return calibrated_data
    
    def _simulate_contact(self) -> np.ndarray:
        """模拟接触力"""
        # 简化的接触力模拟
        # 实际应用中应该根据机器人状态和环境模拟
        
        contact_force = np.zeros(6)
        
        # 模拟Z方向的轻微接触力
        contact_force[2] = np.random.uniform(-5, 0)  # 向上的力（负Z方向）
        
        # 偶尔模拟较大的接触力
        if np.random.random() < 0.05:  # 5%的概率
            contact_force[2] = np.random.uniform(-20, -10)
        
        return contact_force
    
    def get_filtered_reading(self, window_size: int = 10) -> np.ndarray:
        """获取滤波后的读数（移动平均）"""
        if len(self.data_buffer) < window_size:
            return self.force_torque
        
        # 计算移动平均
        recent_data = self.data_buffer[-window_size:]
        filtered = np.mean(recent_data, axis=0)
        
        return filtered
    
    def get_force_magnitude(self) -> float:
        """获取力大小"""
        force_vector = self.force_torque[:3]
        return np.linalg.norm(force_vector)
    
    def get_torque_magnitude(self) -> float:
        """获取力矩大小"""
        torque_vector = self.force_torque[3:]
        return np.linalg.norm(torque_vector)
    
    def reset(self) -> None:
        """重置传感器"""
        self.force_torque = np.zeros(6)
        self.data_buffer = []
        self.logger.info("力传感器已重置")


class ImpedanceController:
    """阻抗控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("ImpedanceController")
        
        # 阻抗参数：质量-阻尼-刚度 (M, B, K)
        self.impedance_params = {
            "mass": self.config.get("mass", np.eye(3)),      # 质量矩阵 (3x3)
            "damping": self.config.get("damping", np.eye(3) * 10),  # 阻尼矩阵 (3x3)
            "stiffness": self.config.get("stiffness", np.eye(3) * 100),  # 刚度矩阵 (3x3)
            "desired_force": self.config.get("desired_force", [0.0, 0.0, -5.0]),  # 期望力 (N)
            "desired_position": self.config.get("desired_position", [0.0, 0.0, 0.0]),  # 期望位置 (m)
        }
        
        # 状态变量
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_force = np.zeros(3)
        
        # 控制变量
        self.control_force = np.zeros(3)
        self.position_error = np.zeros(3)
        self.force_error = np.zeros(3)
        
        # 历史数据
        self.position_history: List[np.ndarray] = []
        self.force_history: List[np.ndarray] = []
        self.control_history: List[np.ndarray] = []
        
        self.logger.info("阻抗控制器初始化完成")
    
    def update(self, position: np.ndarray, velocity: np.ndarray, 
               measured_force: np.ndarray, dt: float) -> np.ndarray:
        """更新控制器
        
        Args:
            position: 当前位置 (3x1)
            velocity: 当前速度 (3x1)
            measured_force: 测量力 (3x1)
            dt: 时间步长 (s)
            
        Returns:
            控制力 (3x1)
        """
        # 更新状态
        self.current_position = position
        self.current_velocity = velocity
        self.current_force = measured_force
        
        # 计算误差
        desired_position = np.array(self.impedance_params["desired_position"])
        desired_force = np.array(self.impedance_params["desired_force"])
        
        self.position_error = desired_position - position
        self.force_error = desired_force - measured_force
        
        # 阻抗控制律: F_c = K * (x_d - x) + B * (v_d - v) + F_d
        # 其中 v_d = 0 (期望速度)
        M = self.impedance_params["mass"]
        B = self.impedance_params["damping"]
        K = self.impedance_params["stiffness"]
        
        # 计算控制力
        spring_force = K @ self.position_error
        damping_force = B @ (-velocity)  # v_d = 0
        
        self.control_force = spring_force + damping_force + desired_force
        
        # 记录历史
        self.position_history.append(position.copy())
        self.force_history.append(measured_force.copy())
        self.control_history.append(self.control_force.copy())
        
        # 限制历史长度
        max_history = 1000
        if len(self.position_history) > max_history:
            self.position_history = self.position_history[-max_history:]
            self.force_history = self.force_history[-max_history:]
            self.control_history = self.control_history[-max_history:]
        
        return self.control_force
    
    def set_impedance_params(self, mass: np.ndarray, damping: np.ndarray, 
                            stiffness: np.ndarray) -> None:
        """设置阻抗参数"""
        self.impedance_params["mass"] = mass
        self.impedance_params["damping"] = damping
        self.impedance_params["stiffness"] = stiffness
        
        self.logger.info(f"阻抗参数更新: M={mass[0,0]:.2f}, B={damping[0,0]:.2f}, K={stiffness[0,0]:.2f}")
    
    def set_desired_force(self, desired_force: List[float]) -> None:
        """设置期望力"""
        self.impedance_params["desired_force"] = desired_force
        self.logger.info(f"期望力更新: {desired_force}")
    
    def set_desired_position(self, desired_position: List[float]) -> None:
        """设置期望位置"""
        self.impedance_params["desired_position"] = desired_position
        self.logger.info(f"期望位置更新: {desired_position}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        return {
            "position_error": self.position_error.tolist(),
            "force_error": self.force_error.tolist(),
            "control_force": self.control_force.tolist(),
            "current_force": self.current_force.tolist(),
            "current_position": self.current_position.tolist()
        }


class AdmittanceController:
    """导纳控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("AdmittanceController")
        
        # 导纳参数：质量-阻尼-刚度 (M, B, K)
        self.admittance_params = {
            "mass": self.config.get("mass", np.eye(3)),      # 质量矩阵 (3x3)
            "damping": self.config.get("damping", np.eye(3) * 20),  # 阻尼矩阵 (3x3)
            "stiffness": self.config.get("stiffness", np.eye(3) * 200),  # 刚度矩阵 (3x3)
            "desired_force": self.config.get("desired_force", [0.0, 0.0, -5.0]),  # 期望力 (N)
            "desired_velocity": self.config.get("desired_velocity", [0.0, 0.0, 0.0]),  # 期望速度 (m/s)
        }
        
        # 状态变量
        self.current_velocity = np.zeros(3)
        self.current_force = np.zeros(3)
        self.integrated_position = np.zeros(3)
        
        # 控制变量
        self.control_velocity = np.zeros(3)
        self.force_error = np.zeros(3)
        
        # 历史数据
        self.velocity_history: List[np.ndarray] = []
        self.force_history: List[np.ndarray] = []
        
        self.logger.info("导纳控制器初始化完成")
    
    def update(self, measured_force: np.ndarray, dt: float) -> np.ndarray:
        """更新控制器
        
        Args:
            measured_force: 测量力 (3x1)
            dt: 时间步长 (s)
            
        Returns:
            控制速度 (3x1)
        """
        # 更新状态
        self.current_force = measured_force
        
        # 计算力误差
        desired_force = np.array(self.admittance_params["desired_force"])
        self.force_error = desired_force - measured_force
        
        # 导纳控制律: v_c = Admittance * (F_d - F)
        # 其中 Admittance = (M*s^2 + B*s + K)^-1
        # 离散化: v_c = (F_d - F) / B (简化)
        
        M = self.admittance_params["mass"]
        B = self.admittance_params["damping"]
        K = self.admittance_params["stiffness"]
        
        # 简化计算：v_c = B^-1 * (F_d - F)
        try:
            B_inv = np.linalg.inv(B)
            self.control_velocity = B_inv @ self.force_error
        except np.linalg.LinAlgError:
            # 如果B不可逆，使用简化的比例控制
            self.control_velocity = self.force_error / np.trace(B)
        
        # 添加期望速度
        desired_velocity = np.array(self.admittance_params["desired_velocity"])
        self.control_velocity += desired_velocity
        
        # 积分得到位置（可选）
        self.integrated_position += self.control_velocity * dt
        
        # 记录历史
        self.velocity_history.append(self.control_velocity.copy())
        self.force_history.append(measured_force.copy())
        
        # 限制历史长度
        max_history = 1000
        if len(self.velocity_history) > max_history:
            self.velocity_history = self.velocity_history[-max_history:]
            self.force_history = self.force_history[-max_history:]
        
        return self.control_velocity
    
    def set_admittance_params(self, mass: np.ndarray, damping: np.ndarray, 
                             stiffness: np.ndarray) -> None:
        """设置导纳参数"""
        self.admittance_params["mass"] = mass
        self.admittance_params["damping"] = damping
        self.admittance_params["stiffness"] = stiffness
        
        self.logger.info(f"导纳参数更新: M={mass[0,0]:.2f}, B={damping[0,0]:.2f}, K={stiffness[0,0]:.2f}")
    
    def set_desired_force(self, desired_force: List[float]) -> None:
        """设置期望力"""
        self.admittance_params["desired_force"] = desired_force
        self.logger.info(f"期望力更新: {desired_force}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        return {
            "force_error": self.force_error.tolist(),
            "control_velocity": self.control_velocity.tolist(),
            "current_force": self.current_force.tolist(),
            "integrated_position": self.integrated_position.tolist()
        }


class HybridForcePositionController:
    """力/位混合控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("HybridForcePositionController")
        
        # 选择矩阵：1表示力控制，0表示位置控制
        self.selection_matrix = self.config.get("selection_matrix", np.array([
            [0, 0, 0],  # X方向：位置控制
            [0, 0, 0],  # Y方向：位置控制
            [1, 1, 1],  # Z方向：力控制
        ]))
        
        # 位置控制器参数
        self.position_control_params = {
            "kp": self.config.get("position_kp", 100.0),  # 位置比例增益
            "kd": self.config.get("position_kd", 10.0),   # 位置微分增益
            "ki": self.config.get("position_ki", 0.0),    # 位置积分增益
        }
        
        # 力控制器参数
        self.force_control_params = {
            "kp": self.config.get("force_kp", 0.1),       # 力比例增益
            "kd": self.config.get("force_kd", 0.01),      # 力微分增益
            "ki": self.config.get("force_ki", 0.0),       # 力积分增益
        }
        
        # 期望值
        self.desired_position = np.array(self.config.get("desired_position", [0.0, 0.0, 0.0]))
        self.desired_force = np.array(self.config.get("desired_force", [0.0, 0.0, -5.0]))
        
        # 状态变量
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_force = np.zeros(3)
        
        # 积分项
        self.position_integral = np.zeros(3)
        self.force_integral = np.zeros(3)
        
        # 控制输出
        self.position_control = np.zeros(3)
        self.force_control = np.zeros(3)
        self.hybrid_control = np.zeros(3)
        
        # 历史数据
        self.control_history: List[np.ndarray] = []
        
        self.logger.info("力/位混合控制器初始化完成")
    
    def update(self, position: np.ndarray, velocity: np.ndarray, 
               measured_force: np.ndarray, dt: float) -> np.ndarray:
        """更新控制器
        
        Args:
            position: 当前位置 (3x1)
            velocity: 当前速度 (3x1)
            measured_force: 测量力 (3x1)
            dt: 时间步长 (s)
            
        Returns:
            混合控制输出 (3x1)
        """
        # 更新状态
        self.current_position = position
        self.current_velocity = velocity
        self.current_force = measured_force
        
        # 计算位置控制（PID）
        position_error = self.desired_position - position
        
        # 比例项
        position_p = self.position_control_params["kp"] * position_error
        
        # 微分项
        position_d = self.position_control_params["kd"] * (-velocity)  # 期望速度为0
        
        # 积分项
        self.position_integral += position_error * dt
        position_i = self.position_control_params["ki"] * self.position_integral
        
        self.position_control = position_p + position_d + position_i
        
        # 计算力控制（PID）
        force_error = self.desired_force - measured_force
        
        # 比例项
        force_p = self.force_control_params["kp"] * force_error
        
        # 微分项（力微分近似）
        force_d = self.force_control_params["kd"] * (-velocity)  # 简化
        
        # 积分项
        self.force_integral += force_error * dt
        force_i = self.force_control_params["ki"] * self.force_integral
        
        self.force_control = force_p + force_d + force_i
        
        # 混合控制：根据选择矩阵组合位置和力控制
        for i in range(3):
            if self.selection_matrix[i, 0] == 1:  # 力控制方向
                self.hybrid_control[i] = self.force_control[i]
            else:  # 位置控制方向
                self.hybrid_control[i] = self.position_control[i]
        
        # 记录历史
        self.control_history.append(self.hybrid_control.copy())
        
        # 限制历史长度
        max_history = 1000
        if len(self.control_history) > max_history:
            self.control_history = self.control_history[-max_history:]
        
        return self.hybrid_control
    
    def set_selection_matrix(self, matrix: np.ndarray) -> None:
        """设置选择矩阵"""
        self.selection_matrix = matrix
        self.logger.info(f"选择矩阵更新: \n{matrix}")
    
    def set_desired_position(self, position: List[float]) -> None:
        """设置期望位置"""
        self.desired_position = np.array(position)
        self.logger.info(f"期望位置更新: {position}")
    
    def set_desired_force(self, force: List[float]) -> None:
        """设置期望力"""
        self.desired_force = np.array(force)
        self.logger.info(f"期望力更新: {force}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        return {
            "position_control": self.position_control.tolist(),
            "force_control": self.force_control.tolist(),
            "hybrid_control": self.hybrid_control.tolist(),
            "selection_matrix": self.selection_matrix.tolist()
        }


class ForceControlSystem:
    """力控制系统（主控制器）"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("ForceControlSystem")
        
        # 控制类型
        self.control_type = ForceControlType(self.config.get("control_type", "impedance"))
        
        # 初始化传感器
        self.force_sensor = ForceSensor(self.config.get("force_sensor_config", {}))
        
        # 初始化控制器
        if self.control_type == ForceControlType.IMPEDANCE:
            self.controller = ImpedanceController(self.config.get("controller_config", {}))
        elif self.control_type == ForceControlType.ADMITTANCE:
            self.controller = AdmittanceController(self.config.get("controller_config", {}))
        elif self.control_type == ForceControlType.HYBRID:
            self.controller = HybridForcePositionController(self.config.get("controller_config", {}))
        else:
            self.controller = ImpedanceController(self.config.get("controller_config", {}))
        
        # 系统状态
        self.is_running = False
        self.control_frequency = self.config.get("control_frequency", 100)  # Hz
        self.control_period = 1.0 / self.control_frequency
        
        # 机器人状态（模拟）
        self.robot_position = np.zeros(3)
        self.robot_velocity = np.zeros(3)
        
        # 控制线程
        self._control_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 数据记录
        self.data_log: List[Dict[str, Any]] = []
        self.max_log_size = 10000
        
        # 回调函数
        self.control_update_callbacks: List[Callable[[np.ndarray], None]] = []
        self.status_update_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        self.logger.info(f"力控制系统初始化完成，控制类型: {self.control_type.value}")
    
    def start(self) -> bool:
        """启动力控制系统"""
        if self.is_running:
            self.logger.warning("力控制系统已经在运行")
            return False
        
        # 校准传感器
        if not self.force_sensor.calibrate():
            self.logger.error("传感器校准失败，无法启动系统")
            return False
        
        self.is_running = True
        self._stop_event.clear()
        
        # 启动控制线程
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        
        self.logger.info("力控制系统启动")
        return True
    
    def stop(self) -> bool:
        """停止力控制系统"""
        if not self.is_running:
            self.logger.warning("力控制系统没有在运行")
            return False
        
        self.is_running = False
        self._stop_event.set()
        
        # 等待控制线程结束
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0)
        
        self.logger.info("力控制系统停止")
        return True
    
    def _control_loop(self) -> None:
        """控制循环"""
        self.logger.info("力控制循环开始")
        
        last_time = time.time()
        
        while self.is_running and not self._stop_event.is_set():
            try:
                # 计算时间步长
                current_time = time.time()
                dt = current_time - last_time
                
                if dt < self.control_period:
                    # 等待下一个控制周期
                    time.sleep(self.control_period - dt)
                    continue
                
                last_time = current_time
                
                # 执行控制迭代
                self._control_iteration(dt)
                
            except Exception as e:
                self.logger.error(f"力控制循环异常: {e}")
                time.sleep(0.1)  # 防止异常导致循环过速
        
        self.logger.info("力控制循环结束")
    
    def _control_iteration(self, dt: float) -> None:
        """控制迭代"""
        try:
            # 读取传感器数据
            measured_force = self.force_sensor.read_calibrated()
            force_3d = measured_force[:3]  # 只使用力分量
            
            # 更新机器人状态（模拟）
            self._update_robot_state(dt)
            
            # 根据控制类型调用相应控制器
            if self.control_type == ForceControlType.IMPEDANCE:
                control_output = self.controller.update(
                    self.robot_position, self.robot_velocity, force_3d, dt
                )
                # 控制输出为力
                control_force = control_output
                
            elif self.control_type == ForceControlType.ADMITTANCE:
                control_output = self.controller.update(force_3d, dt)
                # 控制输出为速度，转换为位置变化
                control_velocity = control_output
                self.robot_velocity = control_velocity
                self.robot_position += control_velocity * dt
                control_force = np.zeros(3)  # 导纳控制不直接输出力
                
            elif self.control_type == ForceControlType.HYBRID:
                control_output = self.controller.update(
                    self.robot_position, self.robot_velocity, force_3d, dt
                )
                # 控制输出为混合控制信号（可能是力或位置指令）
                control_force = control_output
                
            else:
                # 默认：阻抗控制
                control_output = self.controller.update(
                    self.robot_position, self.robot_velocity, force_3d, dt
                )
                control_force = control_output
            
            # 记录数据
            log_entry = {
                "timestamp": time.time(),
                "measured_force": measured_force.tolist(),
                "robot_position": self.robot_position.tolist(),
                "robot_velocity": self.robot_velocity.tolist(),
                "control_output": control_output.tolist() if isinstance(control_output, np.ndarray) else control_output,
                "control_type": self.control_type.value
            }
            
            self.data_log.append(log_entry)
            if len(self.data_log) > self.max_log_size:
                self.data_log = self.data_log[-self.max_log_size:]
            
            # 通知控制更新
            for callback in self.control_update_callbacks:
                try:
                    callback(control_output)
                except Exception as e:
                    self.logger.error(f"控制更新回调执行失败: {e}")
            
            # 通知状态更新
            status = self.get_status()
            for callback in self.status_update_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    self.logger.error(f"状态更新回调执行失败: {e}")
            
            # 调试输出
            if len(self.data_log) % 100 == 0:
                self.logger.debug(
                    f"力控制迭代: F={np.linalg.norm(force_3d):.3f}N, "
                    f"Pos={self.robot_position}, Control={control_output}"
                )
                
        except Exception as e:
            self.logger.error(f"力控制迭代异常: {e}")
    
    def _update_robot_state(self, dt: float) -> None:
        """更新机器人状态（模拟）"""
        # 简化的机器人动力学模拟
        # 实际应用中应该从真实的机器人接口获取状态
        
        # 添加一些随机运动
        import random
        self.robot_velocity += np.random.uniform(-0.01, 0.01, 3)
        self.robot_position += self.robot_velocity * dt
        
        # 限制位置范围
        self.robot_position = np.clip(self.robot_position, -1.0, 1.0)
    
    def set_control_type(self, control_type: ForceControlType) -> None:
        """设置控制类型"""
        old_type = self.control_type
        self.control_type = control_type
        
        # 重新初始化控制器
        if self.control_type == ForceControlType.IMPEDANCE:
            self.controller = ImpedanceController(self.config.get("controller_config", {}))
        elif self.control_type == ForceControlType.ADMITTANCE:
            self.controller = AdmittanceController(self.config.get("controller_config", {}))
        elif self.control_type == ForceControlType.HYBRID:
            self.controller = HybridForcePositionController(self.config.get("controller_config", {}))
        else:
            self.controller = ImpedanceController(self.config.get("controller_config", {}))
        
        self.logger.info(f"控制类型变更: {old_type.value} -> {control_type.value}")
    
    def set_desired_force(self, desired_force: List[float]) -> None:
        """设置期望力"""
        if hasattr(self.controller, 'set_desired_force'):
            self.controller.set_desired_force(desired_force)
        else:
            self.logger.warning(f"当前控制器不支持设置期望力")
    
    def set_desired_position(self, desired_position: List[float]) -> None:
        """设置期望位置"""
        if hasattr(self.controller, 'set_desired_position'):
            self.controller.set_desired_position(desired_position)
        else:
            self.logger.warning(f"当前控制器不支持设置期望位置")
    
    def add_control_update_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """添加控制更新回调"""
        self.control_update_callbacks.append(callback)
    
    def add_status_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """添加状态更新回调"""
        self.status_update_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        sensor_status = {
            "force": self.force_sensor.force_torque.tolist(),
            "force_magnitude": self.force_sensor.get_force_magnitude(),
            "torque_magnitude": self.force_sensor.get_torque_magnitude(),
            "is_calibrated": self.force_sensor.is_calibrated
        }
        
        controller_status = self.controller.get_status() if hasattr(self.controller, 'get_status') else {}
        
        return {
            "is_running": self.is_running,
            "control_type": self.control_type.value,
            "control_frequency": self.control_frequency,
            "sensor": sensor_status,
            "controller": controller_status,
            "robot_position": self.robot_position.tolist(),
            "robot_velocity": self.robot_velocity.tolist(),
            "data_log_size": len(self.data_log)
        }
    
    def get_data_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取数据日志"""
        return self.data_log[-limit:] if self.data_log else []
    
    def reset(self) -> None:
        """重置系统"""
        self.stop()
        self.force_sensor.reset()
        self.data_log = []
        self.robot_position = np.zeros(3)
        self.robot_velocity = np.zeros(3)
        self.logger.info("力控制系统已重置")


# 全局力控制系统实例
_force_control_system: Optional[ForceControlSystem] = None


def get_force_control_system(config: Optional[Dict[str, Any]] = None) -> ForceControlSystem:
    """获取力控制系统单例实例"""
    global _force_control_system
    
    if _force_control_system is None:
        _force_control_system = ForceControlSystem(config)
    
    return _force_control_system