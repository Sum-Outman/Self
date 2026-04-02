"""
硬件训练管理器
提供标准化的硬件训练流程，包括设备初始化、训练执行、状态监控和错误处理

功能：
1. 硬件训练流程标准化
2. 设备状态管理和监控
3. 训练任务调度和执行
4. 错误处理和恢复机制
5. 训练数据记录和分析
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)


class TrainingDeviceType(Enum):
    """训练设备类型"""
    ROBOT_ARM = "robot_arm"          # 机器人手臂
    ROBOT_GRIPPER = "robot_gripper"  # 机器人抓手
    MOBILE_ROBOT = "mobile_robot"    # 移动机器人
    SENSOR_ARRAY = "sensor_array"    # 传感器阵列
    CAMERA_SYSTEM = "camera_system"  # 相机系统
    LIDAR_SYSTEM = "lidar_system"    # 激光雷达系统
    FORCE_SENSOR = "force_sensor"    # 力传感器
    JOINT_CONTROLLER = "joint_controller"  # 关节控制器


class TrainingPhase(Enum):
    """训练阶段"""
    INITIALIZATION = "initialization"    # 初始化
    CALIBRATION = "calibration"          # 校准
    WARMUP = "warmup"                    # 热身
    TRAINING = "training"                # 训练
    COOLDOWN = "cooldown"                # 冷却
    VALIDATION = "validation"            # 验证
    COMPLETION = "completion"            # 完成
    ERROR = "error"                      # 错误


class TrainingStatus(Enum):
    """训练状态"""
    PENDING = "pending"          # 等待
    INITIALIZING = "initializing"  # 初始化中
    RUNNING = "running"          # 运行中
    PAUSED = "paused"            # 暂停
    COMPLETED = "completed"      # 完成
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 取消


@dataclass
class DeviceConfig:
    """设备配置"""
    device_id: str
    device_type: TrainingDeviceType
    parameters: Dict[str, Any] = field(default_factory=dict)
    calibration_data: Optional[Dict[str, Any]] = None
    safety_limits: Dict[str, Any] = field(default_factory=dict)
    connection_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["device_type"] = self.device_type.value
        return result


@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
    name: str
    description: str
    device_configs: List[DeviceConfig]
    training_parameters: Dict[str, Any]
    phase_sequence: List[TrainingPhase]
    expected_duration: timedelta
    priority: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["device_configs"] = [cfg.to_dict() for cfg in self.device_configs]
        result["phase_sequence"] = [phase.value for phase in self.phase_sequence]
        result["expected_duration"] = str(self.expected_duration)
        return result


@dataclass
class TrainingProgress:
    """训练进度"""
    task_id: str
    current_phase: TrainingPhase
    phase_progress: float  # 0.0-1.0
    overall_progress: float  # 0.0-1.0
    start_time: datetime
    elapsed_time: timedelta
    estimated_remaining: timedelta
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["current_phase"] = self.current_phase.value
        result["elapsed_time"] = str(self.elapsed_time)
        result["estimated_remaining"] = str(self.estimated_remaining)
        return result


@dataclass
class TrainingResult:
    """训练结果"""
    task_id: str
    status: TrainingStatus
    start_time: datetime
    end_time: datetime
    duration: timedelta
    success: bool
    error_message: Optional[str] = None
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    validation_results: Optional[Dict[str, Any]] = None
    device_logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["status"] = self.status.value
        result["duration"] = str(self.duration)
        return result


class HardwareTrainingError(Exception):
    """硬件训练错误"""
    
    def __init__(self, message: str, error_code: str, device_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.device_id = device_id
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "message": self.message,
            "error_code": self.error_code,
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
        }


class DeviceManager:
    """设备管理器"""
    
    def __init__(self):
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.connection_pool = {}
        self.lock = threading.RLock()
        logger.info("设备管理器初始化完成")
    
    def connect_device(self, device_config: DeviceConfig) -> bool:
        """连接设备"""
        device_id = device_config.device_id
        
        with self.lock:
            if device_id in self.devices:
                logger.warning(f"设备已连接: {device_id}")
                return True
            
            try:
                # 模拟设备连接
                logger.info(f"连接设备: {device_id} ({device_config.device_type.value})")
                
                # 实际项目中这里会调用具体的硬件驱动
                device_info = {
                    "device_id": device_id,
                    "device_type": device_config.device_type.value,
                    "config": device_config.to_dict(),
                    "connected": True,
                    "connection_time": datetime.now(),
                    "status": "connected",
                    "last_heartbeat": datetime.now(),
                }
                
                self.devices[device_id] = device_info
                
                # 初始化设备
                self._initialize_device(device_config)
                
                logger.info(f"设备连接成功: {device_id}")
                return True
                
            except Exception as e:
                logger.error(f"设备连接失败: {device_id}, 错误: {e}")
                return False
    
    def _initialize_device(self, device_config: DeviceConfig):
        """初始化设备"""
        # 实际项目中这里会调用设备的初始化方法
        logger.debug(f"初始化设备: {device_config.device_id}")
        
        # 检查安全限制
        if device_config.safety_limits:
            self._validate_safety_limits(device_config)
    
    def _validate_safety_limits(self, device_config: DeviceConfig):
        """验证安全限制"""
        # 验证设备的安全限制
        logger.debug(f"验证安全限制: {device_config.device_id}")
    
    def disconnect_device(self, device_id: str) -> bool:
        """断开设备连接"""
        with self.lock:
            if device_id not in self.devices:
                logger.warning(f"设备未连接: {device_id}")
                return False
            
            try:
                # 模拟设备断开
                logger.info(f"断开设备连接: {device_id}")
                
                # 实际项目中这里会调用具体的断开方法
                del self.devices[device_id]
                
                logger.info(f"设备断开成功: {device_id}")
                return True
                
            except Exception as e:
                logger.error(f"设备断开失败: {device_id}, 错误: {e}")
                return False
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """获取设备状态"""
        with self.lock:
            if device_id in self.devices:
                device = self.devices[device_id].copy()
                device["current_time"] = datetime.now()
                return device
            return None  # 返回None
    
    def get_all_devices(self) -> List[Dict[str, Any]]:
        """获取所有设备"""
        with self.lock:
            return list(self.devices.values())
    
    def send_command(self, device_id: str, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """发送命令到设备"""
        with self.lock:
            if device_id not in self.devices:
                raise HardwareTrainingError(
                    f"设备未连接: {device_id}",
                    "DEVICE_NOT_CONNECTED",
                    device_id
                )
            
            try:
                # 模拟命令执行
                logger.info(f"发送命令到设备 {device_id}: {command}")
                
                # 实际项目中这里会调用具体的设备命令接口
                time.sleep(0.1)  # 模拟命令执行时间
                
                response = {
                    "success": True,
                    "device_id": device_id,
                    "command": command,
                    "response": f"命令 {command} 执行成功",
                    "timestamp": datetime.now().isoformat(),
                }
                
                # 更新设备心跳
                self.devices[device_id]["last_heartbeat"] = datetime.now()
                
                return response
                
            except Exception as e:
                logger.error(f"命令执行失败: {device_id}, 命令: {command}, 错误: {e}")
                raise HardwareTrainingError(
                    f"命令执行失败: {str(e)}",
                    "COMMAND_EXECUTION_FAILED",
                    device_id
                )
    
    def check_heartbeats(self) -> List[str]:
        """检查设备心跳，返回离线设备列表"""
        offline_devices = []
        current_time = datetime.now()
        
        with self.lock:
            for device_id, device in self.devices.items():
                last_heartbeat = device["last_heartbeat"]
                time_since_heartbeat = current_time - last_heartbeat
                
                if time_since_heartbeat > timedelta(seconds=30):
                    offline_devices.append(device_id)
                    logger.warning(f"设备心跳丢失: {device_id}, 最后心跳: {last_heartbeat}")
        
        return offline_devices


class TrainingPhaseExecutor:
    """训练阶段执行器"""
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.phase_handlers = {
            TrainingPhase.INITIALIZATION: self._execute_initialization,
            TrainingPhase.CALIBRATION: self._execute_calibration,
            TrainingPhase.WARMUP: self._execute_warmup,
            TrainingPhase.TRAINING: self._execute_training,
            TrainingPhase.COOLDOWN: self._execute_cooldown,
            TrainingPhase.VALIDATION: self._execute_validation,
            TrainingPhase.COMPLETION: self._execute_completion,
        }
        logger.info("训练阶段执行器初始化完成")
    
    def execute_phase(self, 
                     phase: TrainingPhase,
                     task: TrainingTask,
                     progress_callback: Optional[callable] = None) -> Tuple[bool, Dict[str, Any]]:
        """执行训练阶段"""
        logger.info(f"开始执行训练阶段: {phase.value}")
        
        if phase not in self.phase_handlers:
            logger.error(f"未知的训练阶段: {phase.value}")
            return False, {"error": f"未知的训练阶段: {phase.value}"}
        
        try:
            # 执行阶段处理函数
            handler = self.phase_handlers[phase]
            result = handler(task, progress_callback)
            
            logger.info(f"训练阶段完成: {phase.value}, 结果: {result}")
            return True, result
            
        except HardwareTrainingError as e:
            logger.error(f"训练阶段执行失败: {phase.value}, 错误: {e}")
            return False, {"error": e.to_dict()}
        except Exception as e:
            logger.error(f"训练阶段执行异常: {phase.value}, 错误: {e}")
            return False, {"error": str(e)}
    
    def _execute_initialization(self, task: TrainingTask, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """执行初始化阶段"""
        logger.info("执行初始化阶段")
        
        results = []
        
        # 连接所有设备
        for device_config in task.device_configs:
            success = self.device_manager.connect_device(device_config)
            results.append({
                "device_id": device_config.device_id,
                "success": success,
                "phase": "initialization",
            })
            
            if progress_callback:
                progress_callback("device_initialized", device_config.device_id, success)
        
        # 检查所有设备是否连接成功
        all_success = all(result["success"] for result in results)
        
        return {
            "phase": "initialization",
            "success": all_success,
            "device_results": results,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _execute_calibration(self, task: TrainingTask, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """执行校准阶段"""
        logger.info("执行校准阶段")
        
        results = []
        
        # 对每个设备进行校准
        for device_config in task.device_configs:
            try:
                # 发送校准命令
                response = self.device_manager.send_command(
                    device_config.device_id,
                    "calibrate",
                    device_config.calibration_data or {}
                )
                
                results.append({
                    "device_id": device_config.device_id,
                    "success": response.get("success", False),
                    "response": response,
                })
                
                if progress_callback:
                    progress_callback("device_calibrated", device_config.device_id, response.get("success", False))
                
            except Exception as e:
                logger.error(f"设备校准失败: {device_config.device_id}, 错误: {e}")
                results.append({
                    "device_id": device_config.device_id,
                    "success": False,
                    "error": str(e),
                })
        
        return {
            "phase": "calibration",
            "success": all(result["success"] for result in results),
            "device_results": results,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _execute_warmup(self, task: TrainingTask, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """执行热身阶段"""
        logger.info("执行热身阶段")
        
        # 模拟热身过程
        warmup_duration = task.training_parameters.get("warmup_duration", 10)  # 秒
        
        for i in range(warmup_duration):
            time.sleep(1)
            
            if progress_callback:
                progress = (i + 1) / warmup_duration
                progress_callback("warmup_progress", progress, i + 1)
        
        return {
            "phase": "warmup",
            "success": True,
            "duration": warmup_duration,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _execute_training(self, task: TrainingTask, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """执行训练阶段"""
        logger.info("执行训练阶段")
        
        training_duration = task.training_parameters.get("training_duration", 60)  # 秒
        training_iterations = task.training_parameters.get("iterations", 10)
        
        metrics = {
            "iterations_completed": 0,
            "total_iterations": training_iterations,
            "errors": [],
            "device_metrics": {},
        }
        
        # 执行训练迭代
        for iteration in range(training_iterations):
            logger.info(f"训练迭代 {iteration + 1}/{training_iterations}")
            
            try:
                # 模拟训练步骤
                for device_config in task.device_configs:
                    # 发送训练命令
                    response = self.device_manager.send_command(
                        device_config.device_id,
                        "train",
                        {"iteration": iteration, "parameters": task.training_parameters}
                    )
                    
                    # 记录设备指标
                    if device_config.device_id not in metrics["device_metrics"]:
                        metrics["device_metrics"][device_config.device_id] = []
                    
                    metrics["device_metrics"][device_config.device_id].append({
                        "iteration": iteration,
                        "response": response,
                    })
                
                metrics["iterations_completed"] += 1
                
                if progress_callback:
                    progress = (iteration + 1) / training_iterations
                    progress_callback("training_progress", progress, iteration + 1)
                
                # 模拟迭代间隔
                time.sleep(training_duration / training_iterations)
                
            except Exception as e:
                logger.error(f"训练迭代失败: {iteration}, 错误: {e}")
                metrics["errors"].append({
                    "iteration": iteration,
                    "error": str(e),
                })
                
                # 根据错误处理策略决定是否继续
                error_policy = task.training_parameters.get("error_policy", "continue")
                if error_policy == "stop":
                    break
        
        return {
            "phase": "training",
            "success": metrics["iterations_completed"] > 0,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _execute_cooldown(self, task: TrainingTask, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """执行冷却阶段"""
        logger.info("执行冷却阶段")
        
        # 模拟冷却过程
        cooldown_duration = task.training_parameters.get("cooldown_duration", 10)  # 秒
        
        for i in range(cooldown_duration):
            time.sleep(1)
            
            if progress_callback:
                progress = (i + 1) / cooldown_duration
                progress_callback("cooldown_progress", progress, i + 1)
        
        return {
            "phase": "cooldown",
            "success": True,
            "duration": cooldown_duration,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _execute_validation(self, task: TrainingTask, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """执行验证阶段"""
        logger.info("执行验证阶段")
        
        validation_results = {
            "device_validations": [],
            "overall_score": 0,
            "passed": False,
        }
        
        # 验证每个设备
        for device_config in task.device_configs:
            try:
                # 发送验证命令
                response = self.device_manager.send_command(
                    device_config.device_id,
                    "validate",
                    task.training_parameters.get("validation_params", {})
                )
                
                device_validation = {
                    "device_id": device_config.device_id,
                    "success": response.get("success", False),
                    "score": response.get("score", 0),
                    "details": response,
                }
                
                validation_results["device_validations"].append(device_validation)
                
                if progress_callback:
                    progress_callback("device_validated", device_config.device_id, response.get("success", False))
                
            except Exception as e:
                logger.error(f"设备验证失败: {device_config.device_id}, 错误: {e}")
                validation_results["device_validations"].append({
                    "device_id": device_config.device_id,
                    "success": False,
                    "error": str(e),
                })
        
        # 计算总体分数
        if validation_results["device_validations"]:
            scores = [v.get("score", 0) for v in validation_results["device_validations"] if v.get("score") is not None]
            if scores:
                validation_results["overall_score"] = sum(scores) / len(scores)
                validation_results["passed"] = validation_results["overall_score"] >= 0.8
        
        return {
            "phase": "validation",
            "success": validation_results["passed"],
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _execute_completion(self, task: TrainingTask, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """执行完成阶段"""
        logger.info("执行完成阶段")
        
        # 断开所有设备连接
        disconnect_results = []
        
        for device_config in task.device_configs:
            success = self.device_manager.disconnect_device(device_config.device_id)
            disconnect_results.append({
                "device_id": device_config.device_id,
                "success": success,
            })
            
            if progress_callback:
                progress_callback("device_disconnected", device_config.device_id, success)
        
        return {
            "phase": "completion",
            "success": all(result["success"] for result in disconnect_results),
            "disconnect_results": disconnect_results,
            "timestamp": datetime.now().isoformat(),
        }


class HardwareTrainingManager:
    """硬件训练管理器"""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.phase_executor = TrainingPhaseExecutor(self.device_manager)
        
        # 训练任务存储
        self.tasks: Dict[str, TrainingTask] = {}
        self.progress: Dict[str, TrainingProgress] = {}
        self.results: Dict[str, TrainingResult] = {}
        
        # 任务执行状态
        self.active_task_id: Optional[str] = None
        self.is_running = False
        self.execution_thread: Optional[threading.Thread] = None
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 回调函数
        self.callbacks = {
            "progress": [],
            "phase_complete": [],
            "task_complete": [],
            "error": [],
        }
        
        logger.info("硬件训练管理器初始化完成")
    
    def register_callback(self, event_type: str, callback: callable):
        """注册回调函数"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"未知的事件类型: {event_type}")
    
    def _trigger_callback(self, event_type: str, *args, **kwargs):
        """触发回调函数"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"回调函数执行失败: {event_type}, 错误: {e}")
    
    def create_task(self, task: TrainingTask) -> str:
        """创建训练任务"""
        if task.task_id in self.tasks:
            raise HardwareTrainingError(
                f"任务已存在: {task.task_id}",
                "TASK_ALREADY_EXISTS"
            )
        
        self.tasks[task.task_id] = task
        logger.info(f"创建训练任务: {task.task_id} ({task.name})")
        
        return task.task_id
    
    def start_task(self, task_id: str) -> bool:
        """开始训练任务"""
        if self.is_running:
            logger.warning("已有任务正在运行")
            return False
        
        if task_id not in self.tasks:
            logger.error(f"任务不存在: {task_id}")
            return False
        
        task = self.tasks[task_id]
        
        # 设置活动任务
        self.active_task_id = task_id
        self.is_running = True
        
        # 创建进度跟踪
        self.progress[task_id] = TrainingProgress(
            task_id=task_id,
            current_phase=task.phase_sequence[0],
            phase_progress=0.0,
            overall_progress=0.0,
            start_time=datetime.now(),
            elapsed_time=timedelta(0),
            estimated_remaining=task.expected_duration,
        )
        
        # 启动执行线程
        self.execution_thread = threading.Thread(
            target=self._execute_task,
            args=(task,),
            daemon=True
        )
        self.execution_thread.start()
        
        logger.info(f"开始训练任务: {task_id}")
        return True
    
    def _execute_task(self, task: TrainingTask):
        """执行训练任务"""
        try:
            task_id = task.task_id
            
            # 执行每个阶段
            total_phases = len(task.phase_sequence)
            
            for phase_index, phase in enumerate(task.phase_sequence):
                logger.info(f"执行阶段 {phase_index + 1}/{total_phases}: {phase.value}")
                
                # 更新当前阶段
                self.progress[task_id].current_phase = phase
                
                # 执行阶段
                success, phase_result = self.phase_executor.execute_phase(
                    phase,
                    task,
                    self._create_progress_callback(task_id, phase)
                )
                
                # 更新进度
                phase_progress = (phase_index + 1) / total_phases
                self.progress[task_id].phase_progress = 1.0
                self.progress[task_id].overall_progress = phase_progress
                self.progress[task_id].elapsed_time = datetime.now() - self.progress[task_id].start_time
                
                # 触发阶段完成回调
                self._trigger_callback("phase_complete", task_id, phase, success, phase_result)
                
                if not success:
                    # 阶段执行失败
                    raise HardwareTrainingError(
                        f"阶段执行失败: {phase.value}",
                        "PHASE_EXECUTION_FAILED",
                        phase_result.get("error", "未知错误")
                    )
                
                # 短暂暂停
                time.sleep(1)
            
            # 任务完成
            self._complete_task(task_id, True, None)
            
        except HardwareTrainingError as e:
            logger.error(f"训练任务执行失败: {task.task_id}, 错误: {e}")
            self._complete_task(task.task_id, False, e)
        except Exception as e:
            logger.error(f"训练任务执行异常: {task.task_id}, 错误: {e}")
            self._complete_task(task.task_id, False, e)
    
    def _create_progress_callback(self, task_id: str, phase: TrainingPhase):
        """创建进度回调函数"""
        def callback(event: str, *args):
            # 更新进度信息
            if event == "training_progress":
                progress = args[0] if args else 0.0
                self.progress[task_id].phase_progress = progress
                
                # 计算总体进度
                phase_index = list(TrainingPhase).index(phase)
                total_phases = len(self.tasks[task_id].phase_sequence)
                base_progress = phase_index / total_phases
                phase_weight = 1.0 / total_phases
                self.progress[task_id].overall_progress = base_progress + (progress * phase_weight)
            
            # 触发进度回调
            self._trigger_callback("progress", task_id, phase, event, args)
        
        return callback
    
    def _complete_task(self, task_id: str, success: bool, error: Optional[HardwareTrainingError]):
        """完成任务"""
        task = self.tasks.get(task_id)
        progress = self.progress.get(task_id)
        
        if not task or not progress:
            return
        
        # 创建结果
        end_time = datetime.now()
        duration = end_time - progress.start_time
        
        result = TrainingResult(
            task_id=task_id,
            status=TrainingStatus.COMPLETED if success else TrainingStatus.FAILED,
            start_time=progress.start_time,
            end_time=end_time,
            duration=duration,
            success=success,
            error_message=str(error) if error else None,
            metrics_summary={},
        )
        
        self.results[task_id] = result
        
        # 重置状态
        self.active_task_id = None
        self.is_running = False
        
        # 触发任务完成回调
        self._trigger_callback("task_complete", task_id, result)
        
        logger.info(f"训练任务完成: {task_id}, 成功: {success}")
    
    def pause_task(self, task_id: str) -> bool:
        """暂停训练任务"""
        # 实际项目中这里会暂停硬件操作
        logger.info(f"暂停训练任务: {task_id}")
        
        if self.progress.get(task_id):
            self.progress[task_id].current_phase = TrainingPhase.COOLDOWN
        
        return True
    
    def resume_task(self, task_id: str) -> bool:
        """恢复训练任务"""
        # 实际项目中这里会恢复硬件操作
        logger.info(f"恢复训练任务: {task_id}")
        return True
    
    def cancel_task(self, task_id: str) -> bool:
        """取消训练任务"""
        logger.info(f"取消训练任务: {task_id}")
        
        if self.is_running and self.active_task_id == task_id:
            self.is_running = False
            self.active_task_id = None
            
            # 创建取消结果
            if task_id in self.progress:
                progress = self.progress[task_id]
                result = TrainingResult(
                    task_id=task_id,
                    status=TrainingStatus.CANCELLED,
                    start_time=progress.start_time,
                    end_time=datetime.now(),
                    duration=datetime.now() - progress.start_time,
                    success=False,
                    error_message="任务被取消",
                )
                
                self.results[task_id] = result
                self._trigger_callback("task_complete", task_id, result)
        
        return True
    
    def get_task_progress(self, task_id: str) -> Optional[TrainingProgress]:
        """获取任务进度"""
        return self.progress.get(task_id)
    
    def get_task_result(self, task_id: str) -> Optional[TrainingResult]:
        """获取任务结果"""
        return self.results.get(task_id)
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """获取所有任务"""
        tasks = []
        
        for task_id, task in self.tasks.items():
            task_dict = task.to_dict()
            
            # 添加状态信息
            if task_id in self.progress:
                task_dict["progress"] = self.progress[task_id].to_dict()
            
            if task_id in self.results:
                task_dict["result"] = self.results[task_id].to_dict()
            
            tasks.append(task_dict)
        
        return tasks
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理硬件训练管理器资源")
        
        # 停止所有任务
        if self.is_running and self.active_task_id:
            self.cancel_task(self.active_task_id)
        
        # 断开所有设备
        for device_config in self.device_manager.get_all_devices():
            self.device_manager.disconnect_device(device_config["device_id"])
        
        # 关闭线程池
        self.executor.shutdown(wait=False)


# 全局实例
_hardware_training_manager_instance = None


def get_hardware_training_manager() -> HardwareTrainingManager:
    """获取硬件训练管理器单例"""
    global _hardware_training_manager_instance
    
    if _hardware_training_manager_instance is None:
        _hardware_training_manager_instance = HardwareTrainingManager()
    
    return _hardware_training_manager_instance


__all__ = [
    "HardwareTrainingManager",
    "get_hardware_training_manager",
    "TrainingDeviceType",
    "TrainingPhase",
    "TrainingStatus",
    "DeviceConfig",
    "TrainingTask",
    "TrainingProgress",
    "TrainingResult",
    "HardwareTrainingError",
]