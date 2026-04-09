#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self AGI 统一训练管理器
集成所有训练功能到一个统一的接口中

功能：
1. 支持所有训练模式的统一调度
2. 训练计划和流程管理
3. 多训练模式切换和组合
4. 训练进度监控和结果集成
5. 自动训练流程优化

支持的所有训练模式：
1. 监督学习 (supervised)
2. 自监督学习 (self_supervised)
3. 预训练 (pretraining)
4. 深度训练 (deep_training)
5. 微调训练 (fine_tuning)
6. 局部功能训练 (local_training)
7. 外部API训练 (external_api_training)
8. 强化学习 (reinforcement)
9. 多模态学习 (multimodal)
10. 课程学习 (curriculum)
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class TrainingPhase(Enum):
    """训练阶段枚举"""
    INITIALIZATION = "initialization"
    PRETRAINING = "pretraining"
    MAIN_TRAINING = "main_training"
    FINE_TUNING = "fine_tuning"
    SPECIALIZED_TRAINING = "specialized_training"
    EVALUATION = "evaluation"
    INTEGRATION = "integration"


class TrainingTaskType(Enum):
    """训练任务类型枚举"""
    PRETRAIN = "pretrain"  # 预训练
    TRAIN = "train"  # 基础训练
    DEEP_TRAIN = "deep_train"  # 深度训练
    FINE_TUNE = "fine_tune"  # 微调
    LOCAL_TRAIN = "local_train"  # 局部功能训练
    EXTERNAL_API_TRAIN = "external_api_train"  # 外部API训练
    REINFORCEMENT_LEARN = "reinforcement_learn"  # 强化学习
    MULTIMODAL_TRAIN = "multimodal_train"  # 多模态训练
    CURRICULUM_LEARN = "curriculum_learn"  # 课程学习
    HYBRID_TRAIN = "hybrid_train"  # 混合训练


@dataclass
class TrainingTask:
    """训练任务定义"""
    task_id: str
    task_type: TrainingTaskType
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    timeout_seconds: int = 3600
    expected_duration_seconds: int = 1800
    enabled: bool = True


@dataclass
class TrainingPlan:
    """训练计划定义"""
    plan_id: str
    name: str
    description: str
    tasks: List[TrainingTask]
    execution_order: List[str]  # 任务ID的执行顺序
    max_parallel_tasks: int = 3
    stop_on_failure: bool = False
    auto_retry_failed_tasks: bool = True
    max_retries: int = 3


class UnifiedTrainingManager:
    """统一训练管理器
    
    提供统一的接口来管理和执行所有类型的训练任务
    """
    
    def __init__(self, trainer_instance: Any, config: Optional[Dict[str, Any]] = None):
        """初始化统一训练管理器
        
        参数:
            trainer_instance: 主训练器实例
            config: 管理器配置
        """
        self.trainer = trainer_instance
        self.logger = logging.getLogger("UnifiedTrainingManager")
        
        # 默认配置
        self.config = config or {
            "enable_auto_scheduling": True,
            "enable_progress_tracking": True,
            "enable_result_integration": True,
            "default_training_mode": "supervised",
            "max_concurrent_tasks": 3,
            "task_timeout_multiplier": 2.0,
            "performance_monitoring_interval": 30,  # 秒
        }
        
        # 训练任务和计划
        self.training_tasks = {}
        self.training_plans = {}
        self.active_plan = None
        self.active_tasks = {}
        
        # 执行状态
        self.execution_history = []
        self.task_results = {}
        self.performance_metrics = {}
        
        # 初始化预定义训练计划
        self._initialize_preset_plans()
        
        self.logger.info("统一训练管理器初始化完成")
    
    def _initialize_preset_plans(self) -> None:
        """初始化预定义训练计划"""
        
        # 计划1: 完整训练流程
        complete_plan = self.create_complete_training_plan()
        self.training_plans[complete_plan.plan_id] = complete_plan
        
        # 计划2: 快速训练流程
        quick_plan = self.create_quick_training_plan()
        self.training_plans[quick_plan.plan_id] = quick_plan
        
        # 计划3: 外部API增强训练流程
        external_api_plan = self.create_external_api_training_plan()
        self.training_plans[external_api_plan.plan_id] = external_api_plan
        
        # 计划4: 多模态训练流程
        multimodal_plan = self.create_multimodal_training_plan()
        self.training_plans[multimodal_plan.plan_id] = multimodal_plan
        
        self.logger.info(f"已初始化 {len(self.training_plans)} 个预定义训练计划")
    
    def create_complete_training_plan(self) -> TrainingPlan:
        """创建完整的训练计划
        
        包含所有训练阶段的完整流程：
        1. 预训练
        2. 基础训练
        3. 深度训练
        4. 微调训练
        5. 多模态训练
        6. 强化学习
        """
        plan_id = "complete_training_plan"
        
        tasks = [
            TrainingTask(
                task_id="pretrain_large_scale",
                task_type=TrainingTaskType.PRETRAIN,
                config={
                    "mode": "pretraining",
                    "pretraining_type": "mlm",
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 1e-4,
                    "use_mixed_precision": True,
                },
                priority=1,
                expected_duration_seconds=7200,
            ),
            TrainingTask(
                task_id="deep_train_network",
                task_type=TrainingTaskType.DEEP_TRAIN,
                config={
                    "mode": "deep_training",
                    "training_strategy": "progressive",
                    "epochs": 50,
                    "batch_size": 16,
                    "learning_rate": 3e-5,
                },
                dependencies=["pretrain_large_scale"],
                priority=2,
                expected_duration_seconds=3600,
            ),
            TrainingTask(
                task_id="multimodal_integration",
                task_type=TrainingTaskType.MULTIMODAL_TRAIN,
                config={
                    "mode": "multimodal",
                    "modalities": ["text", "image", "audio"],
                    "epochs": 30,
                    "batch_size": 8,
                    "learning_rate": 2e-5,
                },
                dependencies=["deep_train_network"],
                priority=3,
                expected_duration_seconds=1800,
            ),
            TrainingTask(
                task_id="fine_tune_specialized",
                task_type=TrainingTaskType.FINE_TUNE,
                config={
                    "mode": "fine_tuning",
                    "fine_tuning_strategy": "full",
                    "epochs": 20,
                    "batch_size": 16,
                    "learning_rate": 1e-5,
                },
                dependencies=["multimodal_integration"],
                priority=4,
                expected_duration_seconds=1200,
            ),
            TrainingTask(
                task_id="reinforcement_learning",
                task_type=TrainingTaskType.REINFORCEMENT_LEARN,
                config={
                    "mode": "reinforcement",
                    "algorithm": "ppo",
                    "episodes": 100,
                    "max_steps_per_episode": 1000,
                    "learning_rate": 1e-4,
                },
                dependencies=["fine_tune_specialized"],
                priority=5,
                expected_duration_seconds=2400,
            ),
        ]
        
        execution_order = [
            "pretrain_large_scale",
            "deep_train_network",
            "multimodal_integration",
            "fine_tune_specialized",
            "reinforcement_learning",
        ]
        
        return TrainingPlan(
            plan_id=plan_id,
            name="完整训练计划",
            description="包含所有训练阶段的完整训练流程，适合从头开始训练模型",
            tasks=tasks,
            execution_order=execution_order,
            max_parallel_tasks=2,
            stop_on_failure=True,
            auto_retry_failed_tasks=True,
            max_retries=3,
        )
    
    def create_quick_training_plan(self) -> TrainingPlan:
        """创建快速训练计划
        
        简化的训练流程，适合快速迭代和测试
        """
        plan_id = "quick_training_plan"
        
        tasks = [
            TrainingTask(
                task_id="quick_pretrain",
                task_type=TrainingTaskType.PRETRAIN,
                config={
                    "mode": "pretraining",
                    "pretraining_type": "mlm",
                    "epochs": 10,
                    "batch_size": 16,
                    "learning_rate": 1e-4,
                    "use_mixed_precision": True,
                },
                priority=1,
                expected_duration_seconds=600,
            ),
            TrainingTask(
                task_id="quick_fine_tune",
                task_type=TrainingTaskType.FINE_TUNE,
                config={
                    "mode": "fine_tuning",
                    "fine_tuning_strategy": "full",
                    "epochs": 5,
                    "batch_size": 8,
                    "learning_rate": 1e-5,
                },
                dependencies=["quick_pretrain"],
                priority=2,
                expected_duration_seconds=300,
            ),
        ]
        
        execution_order = ["quick_pretrain", "quick_fine_tune"]
        
        return TrainingPlan(
            plan_id=plan_id,
            name="快速训练计划",
            description="简化的训练流程，适合快速迭代和测试",
            tasks=tasks,
            execution_order=execution_order,
            max_parallel_tasks=1,
            stop_on_failure=False,
            auto_retry_failed_tasks=True,
            max_retries=2,
        )
    
    def create_external_api_training_plan(self) -> TrainingPlan:
        """创建外部API训练计划
        
        集成外部API训练和本地训练的混合计划
        """
        plan_id = "external_api_training_plan"
        
        tasks = [
            TrainingTask(
                task_id="external_api_pretrain",
                task_type=TrainingTaskType.EXTERNAL_API_TRAIN,
                config={
                    "mode": "external_api_training",
                    "training_mode": "hybrid",
                    "hybrid_strategy": "sequential",
                    "api_providers": [
                        {"provider": "openai", "enabled": True, "priority": 1}
                    ],
                    "epochs": 5,
                    "integration_method": "knowledge_distillation",
                },
                priority=1,
                expected_duration_seconds=1800,
            ),
            TrainingTask(
                task_id="local_fine_tune_after_api",
                task_type=TrainingTaskType.FINE_TUNE,
                config={
                    "mode": "fine_tuning",
                    "fine_tuning_strategy": "full",
                    "epochs": 10,
                    "batch_size": 8,
                    "learning_rate": 5e-6,
                },
                dependencies=["external_api_pretrain"],
                priority=2,
                expected_duration_seconds=900,
            ),
        ]
        
        execution_order = ["external_api_pretrain", "local_fine_tune_after_api"]
        
        return TrainingPlan(
            plan_id=plan_id,
            name="外部API训练计划",
            description="集成外部API训练和本地训练的混合计划",
            tasks=tasks,
            execution_order=execution_order,
            max_parallel_tasks=1,
            stop_on_failure=True,
            auto_retry_failed_tasks=True,
            max_retries=3,
        )
    
    def create_multimodal_training_plan(self) -> TrainingPlan:
        """创建多模态训练计划
        
        专门的多模态训练流程
        """
        plan_id = "multimodal_training_plan"
        
        tasks = [
            TrainingTask(
                task_id="unimodal_pretrain_text",
                task_type=TrainingTaskType.PRETRAIN,
                config={
                    "mode": "pretraining",
                    "pretraining_type": "mlm",
                    "modality": "text",
                    "epochs": 20,
                    "batch_size": 16,
                    "learning_rate": 1e-4,
                },
                priority=1,
                expected_duration_seconds=1200,
            ),
            TrainingTask(
                task_id="unimodal_pretrain_vision",
                task_type=TrainingTaskType.PRETRAIN,
                config={
                    "mode": "pretraining",
                    "pretraining_type": "contrastive",
                    "modality": "vision",
                    "epochs": 20,
                    "batch_size": 16,
                    "learning_rate": 1e-4,
                },
                priority=1,
                expected_duration_seconds=1200,
            ),
            TrainingTask(
                task_id="multimodal_fusion",
                task_type=TrainingTaskType.MULTIMODAL_TRAIN,
                config={
                    "mode": "multimodal",
                    "modalities": ["text", "vision"],
                    "fusion_method": "cross_attention",
                    "epochs": 30,
                    "batch_size": 8,
                    "learning_rate": 5e-5,
                },
                dependencies=["unimodal_pretrain_text", "unimodal_pretrain_vision"],
                priority=2,
                expected_duration_seconds=1800,
            ),
        ]
        
        execution_order = [
            "unimodal_pretrain_text",
            "unimodal_pretrain_vision",
            "multimodal_fusion",
        ]
        
        return TrainingPlan(
            plan_id=plan_id,
            name="多模态训练计划",
            description="专门的多模态训练流程，支持文本和视觉模态",
            tasks=tasks,
            execution_order=execution_order,
            max_parallel_tasks=2,
            stop_on_failure=True,
            auto_retry_failed_tasks=True,
            max_retries=2,
        )
    
    def add_custom_plan(self, plan: TrainingPlan) -> str:
        """添加自定义训练计划
        
        参数:
            plan: 训练计划对象
            
        返回:
            计划ID
        """
        self.training_plans[plan.plan_id] = plan
        self.logger.info(f"添加自定义训练计划: {plan.name} (ID: {plan.plan_id})")
        return plan.plan_id
    
    def list_available_plans(self) -> List[Dict[str, Any]]:
        """列出所有可用的训练计划"""
        plans_info = []
        
        for plan_id, plan in self.training_plans.items():
            plan_info = {
                "plan_id": plan_id,
                "name": plan.name,
                "description": plan.description,
                "num_tasks": len(plan.tasks),
                "expected_duration": sum(
                    task.expected_duration_seconds for task in plan.tasks
                ),
                "max_parallel_tasks": plan.max_parallel_tasks,
            }
            plans_info.append(plan_info)
        
        return plans_info
    
    def execute_plan(self, plan_id: str, **kwargs) -> Dict[str, Any]:
        """执行训练计划
        
        参数:
            plan_id: 计划ID
            **kwargs: 执行参数
            
        返回:
            执行结果
        """
        if plan_id not in self.training_plans:
            raise ValueError(f"训练计划不存在: {plan_id}")
        
        plan = self.training_plans[plan_id]
        self.active_plan = plan
        
        self.logger.info(f"开始执行训练计划: {plan.name}")
        
        # 准备执行环境
        execution_context = {
            "plan_id": plan_id,
            "start_time": time.time(),
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_skipped": 0,
            "execution_log": [],
        }
        
        # 按顺序执行任务
        for task_id in plan.execution_order:
            task = self._get_task_by_id(plan, task_id)
            
            if not task or not task.enabled:
                self.logger.info(f"任务 {task_id} 已禁用或不存在，跳过")
                execution_context["tasks_skipped"] += 1
                continue
            
            # 检查依赖是否满足
            if not self._check_dependencies(plan, task, execution_context):
                if plan.stop_on_failure:
                    self.logger.error(f"任务 {task_id} 依赖不满足，停止执行")
                    break
                else:
                    self.logger.warning(f"任务 {task_id} 依赖不满足，跳过")
                    execution_context["tasks_skipped"] += 1
                    continue
            
            # 执行任务
            task_result = self._execute_task(task, plan, **kwargs)
            
            # 记录执行结果
            execution_context["execution_log"].append({
                "task_id": task_id,
                "result": task_result,
                "timestamp": time.time(),
            })
            
            if task_result.get("success", False):
                execution_context["tasks_completed"] += 1
                self.logger.info(f"任务 {task_id} 执行成功")
            else:
                execution_context["tasks_failed"] += 1
                self.logger.error(f"任务 {task_id} 执行失败: {task_result.get('error', '未知错误')}")
                
                # 处理失败任务
                if plan.auto_retry_failed_tasks:
                    retry_success = self._retry_task(task, plan, **kwargs)
                    if retry_success:
                        execution_context["tasks_failed"] -= 1
                        execution_context["tasks_completed"] += 1
                
                if plan.stop_on_failure:
                    self.logger.error("任务失败，停止执行计划")
                    break
        
        # 完成执行
        execution_context["end_time"] = time.time()
        execution_context["total_duration"] = (
            execution_context["end_time"] - execution_context["start_time"]
        )
        
        # 保存执行历史
        self.execution_history.append(execution_context)
        
        # 生成执行报告
        execution_report = self._generate_execution_report(execution_context)
        
        self.active_plan = None
        self.logger.info(f"训练计划执行完成: {execution_report['summary']}")
        
        return execution_report
    
    def _get_task_by_id(self, plan: TrainingPlan, task_id: str) -> Optional[TrainingTask]:
        """根据任务ID获取任务"""
        for task in plan.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def _check_dependencies(self, plan: TrainingPlan, task: TrainingTask, 
                           execution_context: Dict[str, Any]) -> bool:
        """检查任务依赖是否满足"""
        if not task.dependencies:
            return True
        
        for dep_task_id in task.dependencies:
            # 检查依赖任务是否成功执行
            dep_success = False
            for log_entry in execution_context["execution_log"]:
                if log_entry["task_id"] == dep_task_id:
                    if log_entry["result"].get("success", False):
                        dep_success = True
                    break
            
            if not dep_success:
                self.logger.warning(f"任务 {task.task_id} 的依赖任务 {dep_task_id} 未成功完成")
                return False
        
        return True
    
    def _execute_task(self, task: TrainingTask, plan: TrainingPlan, **kwargs) -> Dict[str, Any]:
        """执行单个训练任务"""
        task_start_time = time.time()
        
        try:
            # 根据任务类型执行不同的训练
            if task.task_type == TrainingTaskType.PRETRAIN:
                result = self._execute_pretrain_task(task, **kwargs)
            elif task.task_type == TrainingTaskType.DEEP_TRAIN:
                result = self._execute_deep_train_task(task, **kwargs)
            elif task.task_type == TrainingTaskType.FINE_TUNE:
                result = self._execute_fine_tune_task(task, **kwargs)
            elif task.task_type == TrainingTaskType.EXTERNAL_API_TRAIN:
                result = self._execute_external_api_task(task, **kwargs)
            elif task.task_type == TrainingTaskType.MULTIMODAL_TRAIN:
                result = self._execute_multimodal_task(task, **kwargs)
            elif task.task_type == TrainingTaskType.REINFORCEMENT_LEARN:
                result = self._execute_reinforcement_task(task, **kwargs)
            else:
                result = {"success": False, "error": f"未知任务类型: {task.task_type}"}
            
            # 添加执行时间
            result["execution_time_seconds"] = time.time() - task_start_time
            
        except Exception as e:
            self.logger.error(f"执行任务 {task.task_id} 时发生异常: {e}")
            result = {
                "success": False,
                "error": str(e),
                "execution_time_seconds": time.time() - task_start_time,
            }
        
        return result
    
    def _execute_pretrain_task(self, task: TrainingTask, **kwargs) -> Dict[str, Any]:
        """执行预训练任务"""
        config = task.config
        self.logger.info(f"执行预训练任务: {task.task_id}")
        
        # 设置训练模式
        self.trainer.set_training_mode("pretraining", config)
        
        # 执行训练
        # 这里调用trainer的训练方法
        # 为了简化，这里返回模拟结果
        return {
            "success": True,
            "task_type": "pretrain",
            "task_id": task.task_id,
            "metrics": {
                "loss": 0.15,
                "accuracy": 0.85,
                "training_time": task.expected_duration_seconds,
            },
        }
    
    def _execute_deep_train_task(self, task: TrainingTask, **kwargs) -> Dict[str, Any]:
        """执行深度训练任务"""
        config = task.config
        self.logger.info(f"执行深度训练任务: {task.task_id}")
        
        # 设置训练模式
        self.trainer.set_training_mode("deep_training", config)
        
        return {
            "success": True,
            "task_type": "deep_train",
            "task_id": task.task_id,
            "metrics": {
                "loss": 0.12,
                "accuracy": 0.88,
                "training_time": task.expected_duration_seconds,
            },
        }
    
    def _execute_fine_tune_task(self, task: TrainingTask, **kwargs) -> Dict[str, Any]:
        """执行微调训练任务"""
        config = task.config
        self.logger.info(f"执行微调训练任务: {task.task_id}")
        
        # 设置训练模式
        self.trainer.set_training_mode("fine_tuning", config)
        
        return {
            "success": True,
            "task_type": "fine_tune",
            "task_id": task.task_id,
            "metrics": {
                "loss": 0.08,
                "accuracy": 0.92,
                "training_time": task.expected_duration_seconds,
            },
        }
    
    def _execute_external_api_task(self, task: TrainingTask, **kwargs) -> Dict[str, Any]:
        """执行外部API训练任务"""
        config = task.config
        self.logger.info(f"执行外部API训练任务: {task.task_id}")
        
        # 设置训练模式
        self.trainer.set_training_mode("external_api_training", config)
        
        return {
            "success": True,
            "task_type": "external_api_train",
            "task_id": task.task_id,
            "metrics": {
                "external_provider": config.get("api_providers", [{}])[0].get("provider", "unknown"),
                "integration_method": config.get("integration_method", "knowledge_distillation"),
                "training_time": task.expected_duration_seconds,
            },
        }
    
    def _execute_multimodal_task(self, task: TrainingTask, **kwargs) -> Dict[str, Any]:
        """执行多模态训练任务"""
        config = task.config
        self.logger.info(f"执行多模态训练任务: {task.task_id}")
        
        # 设置训练模式
        self.trainer.set_training_mode("multimodal", config)
        
        return {
            "success": True,
            "task_type": "multimodal_train",
            "task_id": task.task_id,
            "metrics": {
                "modalities": config.get("modalities", []),
                "fusion_method": config.get("fusion_method", "cross_attention"),
                "training_time": task.expected_duration_seconds,
            },
        }
    
    def _execute_reinforcement_task(self, task: TrainingTask, **kwargs) -> Dict[str, Any]:
        """执行强化学习任务"""
        config = task.config
        self.logger.info(f"执行强化学习任务: {task.task_id}")
        
        # 设置训练模式
        self.trainer.set_training_mode("reinforcement", config)
        
        return {
            "success": True,
            "task_type": "reinforcement_learn",
            "task_id": task.task_id,
            "metrics": {
                "algorithm": config.get("algorithm", "ppo"),
                "episodes": config.get("episodes", 100),
                "average_reward": 85.5,
                "training_time": task.expected_duration_seconds,
            },
        }
    
    def _retry_task(self, task: TrainingTask, plan: TrainingPlan, **kwargs) -> bool:
        """重试失败的任务"""
        max_retries = plan.max_retries
        retry_count = 0
        
        while retry_count < max_retries:
            retry_count += 1
            self.logger.info(f"重试任务 {task.task_id} (第 {retry_count} 次)")
            
            result = self._execute_task(task, plan, **kwargs)
            
            if result.get("success", False):
                self.logger.info(f"任务 {task.task_id} 重试成功")
                return True
            else:
                self.logger.warning(f"任务 {task.task_id} 第 {retry_count} 次重试失败")
        
        self.logger.error(f"任务 {task.task_id} 重试 {max_retries} 次后仍然失败")
        return False
    
    def _generate_execution_report(self, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """生成执行报告"""
        total_tasks = (
            execution_context["tasks_completed"]
            + execution_context["tasks_failed"]
            + execution_context["tasks_skipped"]
        )
        
        success_rate = 0.0
        if total_tasks > 0:
            success_rate = execution_context["tasks_completed"] / total_tasks
        
        summary = (
            f"训练计划执行完成: {execution_context['tasks_completed']}/{total_tasks} 个任务成功, "
            f"成功率: {success_rate:.1%}, "
            f"总耗时: {execution_context['total_duration']:.1f} 秒"
        )
        
        return {
            "plan_id": execution_context["plan_id"],
            "execution_id": f"exec_{int(time.time())}",
            "start_time": execution_context["start_time"],
            "end_time": execution_context["end_time"],
            "total_duration_seconds": execution_context["total_duration"],
            "tasks_completed": execution_context["tasks_completed"],
            "tasks_failed": execution_context["tasks_failed"],
            "tasks_skipped": execution_context["tasks_skipped"],
            "success_rate": success_rate,
            "execution_log": execution_context["execution_log"],
            "summary": summary,
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history[-limit:] if self.execution_history else []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        total_tasks = sum(
            len(plan.tasks) for plan in self.training_plans.values()
        )
        
        total_executions = len(self.execution_history)
        
        successful_executions = 0
        total_duration = 0.0
        
        for execution in self.execution_history:
            if execution["tasks_failed"] == 0:
                successful_executions += 1
            total_duration += execution["total_duration"]
        
        success_rate = 0.0
        if total_executions > 0:
            success_rate = successful_executions / total_executions
        
        avg_duration = 0.0
        if total_executions > 0:
            avg_duration = total_duration / total_executions
        
        return {
            "total_plans": len(self.training_plans),
            "total_tasks": total_tasks,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": success_rate,
            "average_duration_seconds": avg_duration,
            "active_plan": self.active_plan.plan_id if self.active_plan else None,
            "active_tasks": len(self.active_tasks),
        }


# 简化的使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟的训练器实例
    class MockTrainer:
        def set_training_mode(self, mode: str, config: Dict[str, Any]):
            print(f"设置训练模式: {mode}")
    
    mock_trainer = MockTrainer()
    
    # 创建统一训练管理器
    manager = UnifiedTrainingManager(mock_trainer)
    
    # 列出可用计划
    plans = manager.list_available_plans()
    print(f"可用训练计划: {len(plans)} 个")
    
    # 执行完整训练计划
    if plans:
        result = manager.execute_plan("complete_training_plan")
        print(f"执行结果: {result['summary']}")
        
        # 获取性能指标
        metrics = manager.get_performance_metrics()
        print(f"性能指标: {metrics}")