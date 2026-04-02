#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能训练资源调度器

功能：
1. 动态资源分配（GPU/CPU/内存）
2. 基于任务重要性和紧急性的优先级调度
3. 实时资源监控和自适应调整
4. 任务队列管理和调度策略
5. 资源利用优化和负载平衡

工业级AGI系统要求：支持多模态训练任务的高效资源管理
"""

import time
import threading
import queue
import heapq
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import GPUtil  # type: ignore
    GPUUTIL_AVAILABLE = True
except ImportError:
    GPUtil = None
    GPUUTIL_AVAILABLE = False

import torch

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 5      # 关键任务：系统核心功能、紧急训练任务
    HIGH = 4          # 高优先级：重要训练任务、实时推理
    MEDIUM = 3        # 中优先级：常规训练任务、批量处理
    LOW = 2           # 低优先级：后台任务、非紧急计算
    BACKGROUND = 1    # 后台任务：数据预处理、日志分析等


class TaskType(Enum):
    """任务类型"""
    MODEL_TRAINING = "model_training"      # 模型训练
    INFERENCE = "inference"                # 推理任务
    DATA_PREPROCESSING = "data_preprocessing"  # 数据预处理
    EVALUATION = "evaluation"              # 模型评估
    HYPERPARAMETER_SEARCH = "hyperparameter_search"  # 超参数搜索
    VALIDATION = "validation"              # 验证任务
    EXPORT = "export"                      # 模型导出
    BACKUP = "backup"                      # 数据备份


@dataclass(order=True)
class ScheduledTask:
    """调度任务"""
    priority_score: float          # 优先级分数（基于紧急性和重要性）
    creation_time: float           # 创建时间
    task_id: str = field(compare=False)          # 任务ID
    task_type: TaskType = field(compare=False)   # 任务类型
    description: str = field(compare=False)      # 任务描述
    estimated_resources: Dict[str, float] = field(compare=False)  # 预估资源需求
    callback: Optional[Callable] = field(compare=False)  # 任务回调函数
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)  # 元数据
    start_time: Optional[float] = field(compare=False, default=None)  # 开始时间
    end_time: Optional[float] = field(compare=False, default=None)    # 结束时间
    status: str = field(compare=False, default='pending')  # 任务状态


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, update_interval: float = 2.0):
        """
        初始化资源监控器
        
        参数:
            update_interval: 更新间隔（秒）
        """
        self.update_interval = update_interval
        self.last_update = 0
        self.current_metrics = {}
        
        # 如果psutil不可用，使用基本监控
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil不可用，资源监控功能受限")
        
        # GPU监控
        self.gpu_available = False
        if GPUUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_available = True
                    logger.info(f"GPU监控已启用，检测到{len(gpus)}个GPU")
            except Exception:
                self.gpu_available = False
    
    def get_system_resources(self) -> Dict[str, Any]:
        """
        获取系统资源使用情况
        
        返回:
            资源使用情况字典
        """
        current_time = time.time()
        if current_time - self.last_update < self.update_interval and self.current_metrics:
            return self.current_metrics
        
        metrics = {
            'timestamp': current_time,
            'cpu': {},
            'memory': {},
            'gpu': [],
            'disk': {},
        }
        
        # CPU使用率
        if PSUTIL_AVAILABLE:
            metrics['cpu']['percent'] = psutil.cpu_percent(interval=0.1)
            metrics['cpu']['count'] = psutil.cpu_count(logical=True)
            metrics['cpu']['count_physical'] = psutil.cpu_count(logical=False)
            
            # CPU频率（如果可用）
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    metrics['cpu']['freq_current'] = cpu_freq.current
                    metrics['cpu']['freq_max'] = cpu_freq.max
            except Exception:
                pass  # 已实现
        
        # 内存使用情况
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            metrics['memory'] = {
                'total': mem.total,
                'available': mem.available,
                'percent': mem.percent,
                'used': mem.used,
                'free': mem.free,
            }
        
        # GPU使用情况
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    metrics['gpu'].append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load * 100,  # 转换为百分比
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed,
                        'memory_free': gpu.memoryFree,
                        'temperature': gpu.temperature,
                    })
            except Exception as e:
                logger.warning(f"获取GPU信息失败: {e}")
        
        # 磁盘使用情况
        if PSUTIL_AVAILABLE:
            try:
                disk = psutil.disk_usage('/')
                metrics['disk'] = {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent,
                }
            except Exception:
                pass  # 已实现
        
        # PyTorch GPU信息
        if torch.cuda.is_available():
            metrics['pytorch_gpu'] = []
            for i in range(torch.cuda.device_count()):
                metrics['pytorch_gpu'].append({
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_reserved': torch.cuda.memory_reserved(i),
                    'memory_cached': torch.cuda.memory_reserved(i),  # 旧版本兼容
                })
        
        self.current_metrics = metrics
        self.last_update = current_time
        
        return metrics
    
    def get_resource_availability(self) -> Dict[str, float]:
        """
        获取资源可用性
        
        返回:
            资源可用性字典（0-1之间的值）
        """
        resources = self.get_system_resources()
        
        availability = {
            'cpu': 1.0 - (resources.get('cpu', {}).get('percent', 0) / 100),
            'memory': 1.0 - (resources.get('memory', {}).get('percent', 0) / 100),
            'gpu': [],
        }
        
        # GPU可用性
        if resources.get('gpu'):
            for gpu in resources['gpu']:
                availability['gpu'].append({
                    'id': gpu['id'],
                    'load_availability': 1.0 - (gpu.get('load', 0) / 100),
                    'memory_availability': gpu.get('memory_free', 0) / max(gpu.get('memory_total', 1), 1),
                })
        
        # 磁盘可用性
        if resources.get('disk'):
            availability['disk'] = 1.0 - (resources['disk'].get('percent', 0) / 100)
        
        return availability
    
    def can_allocate_resources(self, required: Dict[str, float]) -> bool:
        """
        检查是否可以分配指定资源
        
        参数:
            required: 所需资源
            
        返回:
            是否可以分配
        """
        availability = self.get_resource_availability()
        
        # 检查CPU
        required_cpu = required.get('cpu', 0)
        if required_cpu > 0 and availability.get('cpu', 1.0) < (required_cpu / 100):
            logger.debug(f"CPU资源不足: 需要{required_cpu}%，可用{availability.get('cpu', 1.0)*100:.1f}%")
            return False
        
        # 检查内存
        required_memory = required.get('memory', 0)
        if required_memory > 0 and availability.get('memory', 1.0) < (required_memory / 100):
            logger.debug(f"内存资源不足: 需要{required_memory}%，可用{availability.get('memory', 1.0)*100:.1f}%")
            return False
        
        # 检查GPU
        required_gpu = required.get('gpu', 0)
        if required_gpu > 0:
            gpu_available = False
            if availability.get('gpu'):
                for gpu_info in availability['gpu']:
                    if gpu_info['load_availability'] >= (required_gpu / 100):
                        gpu_available = True
                        break
            if not gpu_available:
                logger.debug(f"GPU资源不足: 需要{required_gpu}%，无可用GPU")
                return False
        
        return True


class TaskScheduler:
    """任务调度器"""
    
    def __init__(
        self,
        max_concurrent_tasks: int = 3,
        resource_check_interval: float = 1.0,
        enable_adaptive_scheduling: bool = True,
    ):
        """
        初始化任务调度器
        
        参数:
            max_concurrent_tasks: 最大并发任务数
            resource_check_interval: 资源检查间隔（秒）
            enable_adaptive_scheduling: 是否启用自适应调度
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.resource_check_interval = resource_check_interval
        self.enable_adaptive_scheduling = enable_adaptive_scheduling
        
        # 任务队列（使用优先队列）
        self.task_queue = []  # 堆队列
        self.task_dict = {}   # 任务ID到任务的映射
        
        # 运行中任务
        self.running_tasks = {}
        
        # 历史任务
        self.completed_tasks = []
        self.failed_tasks = []
        
        # 资源监控器
        self.resource_monitor = ResourceMonitor()
        
        # 调度线程
        self.scheduler_thread = None
        self.scheduler_running = False
        
        # 统计信息
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0,
        }
        
        # 调度策略权重
        self.scheduling_weights = {
            'priority': 0.4,      # 任务优先级权重
            'urgency': 0.3,       # 任务紧急性权重
            'resource_efficiency': 0.2,  # 资源效率权重
            'fairness': 0.1,      # 公平性权重
        }
        
        logger.info(f"任务调度器初始化完成，最大并发任务数: {max_concurrent_tasks}")
    
    def submit_task(
        self,
        task_type: TaskType,
        description: str,
        estimated_resources: Dict[str, float],
        callback: Callable,
        priority: Union[TaskPriority, int] = TaskPriority.MEDIUM,
        urgency: float = 0.5,  # 0-1，1为最紧急
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        提交任务到调度器
        
        参数:
            task_type: 任务类型
            description: 任务描述
            estimated_resources: 预估资源需求
            callback: 任务回调函数
            priority: 任务优先级
            urgency: 紧急性（0-1）
            metadata: 任务元数据
            
        返回:
            任务ID
        """
        # 生成任务ID
        task_id = f"{task_type.value}_{int(time.time())}_{self.stats['tasks_submitted']}"
        
        # 计算优先级分数
        if isinstance(priority, TaskPriority):
            priority_value = priority.value
        else:
            priority_value = min(max(priority, 1), 5)  # 限制在1-5范围内
        
        # 计算总优先级分数
        priority_score = self._calculate_priority_score(
            priority_value=priority_value,
            urgency=urgency,
            task_type=task_type,
        )
        
        # 创建任务
        task = ScheduledTask(
            priority_score=priority_score,
            creation_time=time.time(),
            task_id=task_id,
            task_type=task_type,
            description=description,
            estimated_resources=estimated_resources,
            callback=callback,
            metadata=metadata or {},
        )
        
        # 添加到队列
        heapq.heappush(self.task_queue, task)
        self.task_dict[task_id] = task
        self.stats['tasks_submitted'] += 1
        
        logger.info(f"任务已提交: {task_id} - {description} (优先级: {priority_score:.2f})")
        
        # 如果调度器未运行，启动它
        if not self.scheduler_running:
            self.start()
        
        return task_id
    
    def _calculate_priority_score(
        self,
        priority_value: int,
        urgency: float,
        task_type: TaskType,
    ) -> float:
        """
        计算任务优先级分数
        
        参数:
            priority_value: 优先级值（1-5）
            urgency: 紧急性（0-1）
            task_type: 任务类型
            
        返回:
            优先级分数
        """
        # 基础分数
        base_score = priority_value * 10
        
        # 紧急性调整
        urgency_adjustment = urgency * 5
        
        # 任务类型调整（重要任务类型获得更高权重）
        type_multipliers = {
            TaskType.MODEL_TRAINING: 1.2,
            TaskType.INFERENCE: 1.1,
            TaskType.HYPERPARAMETER_SEARCH: 1.3,
            TaskType.EVALUATION: 1.0,
            TaskType.VALIDATION: 1.0,
            TaskType.DATA_PREPROCESSING: 0.8,
            TaskType.EXPORT: 0.9,
            TaskType.BACKUP: 0.7,
        }
        
        type_multiplier = type_multipliers.get(task_type, 1.0)
        
        # 计算总分
        total_score = (base_score + urgency_adjustment) * type_multiplier
        
        # 添加时间衰减因子（创建时间越早，分数越高）
        time_factor = 1.0  # 可以在调度时动态调整
        
        return total_score * time_factor
    
    def start(self):
        """启动调度器"""
        if self.scheduler_running:
            logger.warning("调度器已在运行")
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="TaskScheduler"
        )
        self.scheduler_thread.start()
        logger.info("任务调度器已启动")
    
    def stop(self):
        """停止调度器"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
            self.scheduler_thread = None
        logger.info("任务调度器已停止")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self.scheduler_running:
            try:
                self._schedule_tasks()
                time.sleep(self.resource_check_interval)
            except Exception as e:
                logger.error(f"调度器循环异常: {e}")
                time.sleep(self.resource_check_interval * 2)  # 发生错误时等待更久
    
    def _schedule_tasks(self):
        """调度任务"""
        # 检查运行中任务
        completed_tasks = []
        for task_id, task_thread in list(self.running_tasks.items()):
            if not task_thread.is_alive():
                completed_tasks.append(task_id)
        
        # 清理已完成任务
        for task_id in completed_tasks:
            task_thread = self.running_tasks.pop(task_id, None)
            if task_thread:
                task = self.task_dict.get(task_id)
                if task:
                    task.end_time = time.time()
                    execution_time = task.end_time - task.start_time
                    task.status = 'completed'
                    
                    # 更新统计信息
                    self.stats['tasks_completed'] += 1
                    self.stats['total_execution_time'] += execution_time
                    self.stats['avg_execution_time'] = self.stats['total_execution_time'] / self.stats['tasks_completed']
                    
                    # 添加到历史
                    self.completed_tasks.append(task)
                    
                    logger.info(f"任务完成: {task_id} - 执行时间: {execution_time:.2f}秒")
        
        # 调度新任务
        while (len(self.running_tasks) < self.max_concurrent_tasks and 
               self.task_queue and 
               self.scheduler_running):
            
            # 获取最高优先级任务
            task = heapq.heappop(self.task_queue)
            
            # 检查资源是否可用
            if self.resource_monitor.can_allocate_resources(task.estimated_resources):
                # 启动任务
                task.start_time = time.time()
                task.status = 'running'
                
                # 创建任务线程
                task_thread = threading.Thread(
                    target=self._execute_task,
                    args=(task,),
                    daemon=True,
                    name=f"Task-{task.task_id}"
                )
                
                self.running_tasks[task.task_id] = task_thread
                task_thread.start()
                
                logger.info(f"任务开始执行: {task.task_id} - {task.description}")
            else:
                # 资源不足，放回队列
                heapq.heappush(self.task_queue, task)
                break  # 资源不足，停止调度
    
    def _execute_task(self, task: ScheduledTask):
        """执行任务"""
        try:
            # 执行回调函数
            result = task.callback()
            
            # 记录结果
            task.metadata['result'] = result
            task.metadata['success'] = True
            
        except Exception as e:
            # 任务失败
            task.status = 'failed'
            task.metadata['error'] = str(e)
            task.metadata['success'] = False
            
            # 更新统计
            self.stats['tasks_failed'] += 1
            
            # 添加到失败任务列表
            self.failed_tasks.append(task)
            
            logger.error(f"任务执行失败: {task.task_id} - {e}")
            
            # 重新调度失败任务（根据策略）
            if self._should_retry_task(task):
                logger.info(f"重新调度失败任务: {task.task_id}")
                task.priority_score *= 1.1  # 提高优先级
                task.status = 'pending'
                heapq.heappush(self.task_queue, task)
    
    def _should_retry_task(self, task: ScheduledTask) -> bool:
        """判断是否应该重试任务"""
        # 检查重试次数
        retry_count = task.metadata.get('retry_count', 0)
        if retry_count >= 3:  # 最多重试3次
            return False
        
        # 更新重试次数
        task.metadata['retry_count'] = retry_count + 1
        
        # 根据错误类型决定是否重试
        error_msg = task.metadata.get('error', '')
        
        # 资源不足错误，稍后重试
        if 'resource' in error_msg.lower() or 'memory' in error_msg.lower():
            return True
        
        # 临时性错误，重试
        if 'timeout' in error_msg.lower() or 'connection' in error_msg.lower():
            return True
        
        # 其他错误，不重试
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self.task_dict.get(task_id)
        if not task:
            return None  # 返回None
        
        status_info = {
            'task_id': task.task_id,
            'task_type': task.task_type.value,
            'description': task.description,
            'status': task.status,
            'priority_score': task.priority_score,
            'creation_time': task.creation_time,
            'start_time': task.start_time,
            'end_time': task.end_time,
            'estimated_resources': task.estimated_resources,
            'metadata': task.metadata,
        }
        
        if task.status == 'running':
            if task.start_time:
                status_info['elapsed_time'] = time.time() - task.start_time
        
        return status_info
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        resources = self.resource_monitor.get_system_resources()
        availability = self.resource_monitor.get_resource_availability()
        
        return {
            'running': self.scheduler_running,
            'stats': self.stats.copy(),
            'queue_size': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'system_resources': resources,
            'resource_availability': availability,
            'max_concurrent_tasks': self.max_concurrent_tasks,
        }
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> bool:
        """等待任务完成"""
        start_time = time.time()
        
        while True:
            task = self.task_dict.get(task_id)
            if not task:
                return False
            
            if task.status in ['completed', 'failed']:
                return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            time.sleep(0.5)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.task_dict.get(task_id)
        if not task:
            return False
        
        # 如果在队列中，移除
        if task in self.task_queue:
            # 从堆中移除比较困难，所以我们标记它
            task.metadata['cancelled'] = True
            return True
        
        # 如果在运行中，无法直接取消
        if task.status == 'running':
            logger.warning(f"任务 {task_id} 正在运行，无法取消")
            return False
        
        return True


# 全局调度器实例
_default_scheduler = None

def get_default_scheduler() -> TaskScheduler:
    """获取默认调度器实例"""
    global _default_scheduler
    if _default_scheduler is None:
        _default_scheduler = TaskScheduler()
    return _default_scheduler

def submit_training_task(
    description: str,
    callback: Callable,
    estimated_resources: Optional[Dict[str, float]] = None,
    priority: Union[TaskPriority, int] = TaskPriority.HIGH,
    urgency: float = 0.7,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    快速提交训练任务
    
    参数:
        description: 任务描述
        callback: 回调函数
        estimated_resources: 预估资源需求
        priority: 优先级
        urgency: 紧急性
        metadata: 元数据
        
    返回:
        任务ID
    """
    if estimated_resources is None:
        estimated_resources = {
            'cpu': 60,    # 60% CPU
            'memory': 40, # 40% 内存
            'gpu': 70,    # 70% GPU（如果可用）
        }
    
    scheduler = get_default_scheduler()
    return scheduler.submit_task(
        task_type=TaskType.MODEL_TRAINING,
        description=description,
        estimated_resources=estimated_resources,
        callback=callback,
        priority=priority,
        urgency=urgency,
        metadata=metadata,
    )


# 测试函数
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("测试智能训练资源调度器...")
    
    # 创建调度器
    scheduler = TaskScheduler(max_concurrent_tasks=2)
    
    # 测试任务函数
    def sample_task(task_id: str, duration: float = 2.0):
        print(f"任务 {task_id} 开始执行，预计耗时 {duration} 秒")
        time.sleep(duration)
        print(f"任务 {task_id} 完成")
        return {"result": f"任务 {task_id} 完成"}
    
    # 提交测试任务
    task_ids = []
    
    # 高优先级任务
    task_id1 = scheduler.submit_task(
        task_type=TaskType.MODEL_TRAINING,
        description="高优先级模型训练",
        estimated_resources={'cpu': 30, 'memory': 20, 'gpu': 50},
        callback=lambda: sample_task("high_priority", 1.0),
        priority=TaskPriority.HIGH,
        urgency=0.9,
    )
    task_ids.append(task_id1)
    
    # 中优先级任务
    task_id2 = scheduler.submit_task(
        task_type=TaskType.DATA_PREPROCESSING,
        description="数据预处理",
        estimated_resources={'cpu': 40, 'memory': 30},
        callback=lambda: sample_task("medium_priority", 3.0),
        priority=TaskPriority.MEDIUM,
        urgency=0.5,
    )
    task_ids.append(task_id2)
    
    # 低优先级任务
    task_id3 = scheduler.submit_task(
        task_type=TaskType.EVALUATION,
        description="模型评估",
        estimated_resources={'cpu': 20, 'memory': 10},
        callback=lambda: sample_task("low_priority", 2.0),
        priority=TaskPriority.LOW,
        urgency=0.3,
    )
    task_ids.append(task_id3)
    
    # 启动调度器
    scheduler.start()
    
    # 等待任务完成
    print("\n等待任务完成...")
    for task_id in task_ids:
        if scheduler.wait_for_task(task_id, timeout=10.0):
            status = scheduler.get_task_status(task_id)
            print(f"任务 {task_id} 状态: {status['status']}")
        else:
            print(f"任务 {task_id} 超时")
    
    # 获取调度器状态
    status = scheduler.get_scheduler_status()
    print(f"\n调度器状态:")
    print(f"  提交任务数: {status['stats']['tasks_submitted']}")
    print(f"  完成任务数: {status['stats']['tasks_completed']}")
    print(f"  失败任务数: {status['stats']['tasks_failed']}")
    print(f"  平均执行时间: {status['stats']['avg_execution_time']:.2f}秒")
    
    # 停止调度器
    scheduler.stop()
    
    print("\n智能训练资源调度器测试完成！")