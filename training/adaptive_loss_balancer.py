#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应损失平衡器 - 多任务训练中的动态损失权重调整

实现自适应损失权重调整机制，基于：
1. 任务损失变化率 (GradNorm启发式)
2. 任务不确定性 (Uncertainty Weighting)
3. 训练进度动态调整
4. 任务优先级调度

参考论文：
- "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Multitask Networks" (Chen et al., 2018)
- "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (Kendall et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from collections import deque
import math

logger = logging.getLogger(__name__)

@dataclass
class TaskMetrics:
    """任务指标"""
    task_name: str
    loss_history: deque = field(default_factory=lambda: deque(maxlen=100))
    weight_history: deque = field(default_factory=lambda: deque(maxlen=100))
    current_weight: float = 1.0
    learning_speed: float = 0.0  # 学习速度（损失下降率）
    importance: float = 1.0  # 任务重要性
    uncertainty: float = 1.0  # 任务不确定性
    
    def update_loss(self, loss: float):
        """更新损失历史"""
        self.loss_history.append(loss)
        
        # 计算学习速度（最近10步的平均损失变化）
        if len(self.loss_history) >= 10:
            recent_losses = list(self.loss_history)[-10:]
            loss_change = recent_losses[0] - recent_losses[-1]
            self.learning_speed = loss_change / 10.0
        else:
            self.learning_speed = 0.0
    
    def update_weight(self, weight: float):
        """更新权重历史"""
        self.weight_history.append(weight)
        self.current_weight = weight


class AdaptiveLossBalancer:
    """自适应损失平衡器
    
    关键特性：
    1. 基于任务学习速度的权重调整（GradNorm启发式）
    2. 基于任务不确定性的权重调整（Uncertainty Weighting）
    3. 任务重要性优先级调度
    4. 训练进度感知的权重平滑
    
    算法：
    1. 计算每个任务的学习速度（损失下降率）
    2. 计算每个任务的相对学习速度
    3. 根据学习速度调整权重：学习速度慢的任务获得更高权重
    4. 结合任务不确定性和重要性
    5. 应用权重平滑以避免剧烈波动
    """
    
    def __init__(self, 
                 task_names: List[str],
                 initial_weights: Optional[Dict[str, float]] = None,
                 balancing_strategy: str = "gradnorm",  # gradnorm, uncertainty, hybrid
                 temperature: float = 1.0,
                 update_frequency: int = 10,
                 smoothing_factor: float = 0.1):
        """
        初始化自适应损失平衡器
        
        参数:
            task_names: 任务名称列表
            initial_weights: 初始权重字典，如果为None则使用均匀权重
            balancing_strategy: 平衡策略 ("gradnorm", "uncertainty", "hybrid")
            temperature: 权重调整温度参数，控制调整幅度
            update_frequency: 权重更新频率（训练步数）
            smoothing_factor: 权重平滑因子（0-1），越大越平滑
        """
        self.task_names = task_names
        self.balancing_strategy = balancing_strategy
        self.temperature = temperature
        self.update_frequency = update_frequency
        self.smoothing_factor = smoothing_factor
        self.training_step = 0
        
        # 初始化任务指标
        self.task_metrics: Dict[str, TaskMetrics] = {}
        for task_name in task_names:
            initial_weight = 1.0
            if initial_weights and task_name in initial_weights:
                initial_weight = initial_weights[task_name]
            
            self.task_metrics[task_name] = TaskMetrics(
                task_name=task_name,
                current_weight=initial_weight
            )
        
        # 策略特定参数
        if balancing_strategy in ["uncertainty", "hybrid"]:
            # 不确定性权重参数
            self.log_vars = nn.ParameterDict({
                task_name: nn.Parameter(torch.zeros(1))
                for task_name in task_names
            })
        
        logger.info(f"自适应损失平衡器初始化完成，任务数: {len(task_names)}，策略: {balancing_strategy}")
    
    def compute_task_weights(self, 
                           task_losses: Dict[str, torch.Tensor],
                           model: Optional[nn.Module] = None,
                           optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, float]:
        """
        计算任务权重
        
        参数:
            task_losses: 任务损失字典 {task_name: loss_tensor}
            model: 模型（用于GradNorm计算梯度）
            optimizer: 优化器（用于GradNorm计算梯度）
            
        返回:
            任务权重字典 {task_name: weight}
        """
        self.training_step += 1
        
        # 更新任务损失历史
        for task_name, loss_tensor in task_losses.items():
            if task_name in self.task_metrics:
                loss_value = loss_tensor.detach().item()
                self.task_metrics[task_name].update_loss(loss_value)
        
        # 按更新频率调整权重
        if self.training_step % self.update_frequency != 0:
            # 未到更新频率，返回当前权重
            return {task_name: metrics.current_weight 
                   for task_name, metrics in self.task_metrics.items()}
        
        # 根据策略计算新权重
        if self.balancing_strategy == "gradnorm":
            new_weights = self._compute_gradnorm_weights(task_losses, model, optimizer)
        elif self.balancing_strategy == "uncertainty":
            new_weights = self._compute_uncertainty_weights(task_losses)
        elif self.balancing_strategy == "hybrid":
            new_weights = self._compute_hybrid_weights(task_losses, model, optimizer)
        else:
            # 默认均匀权重
            new_weights = {task_name: 1.0 for task_name in self.task_names}
        
        # 应用权重平滑
        smoothed_weights = {}
        for task_name, new_weight in new_weights.items():
            old_weight = self.task_metrics[task_name].current_weight
            smoothed_weight = (self.smoothing_factor * old_weight + 
                              (1 - self.smoothing_factor) * new_weight)
            smoothed_weights[task_name] = smoothed_weight
            self.task_metrics[task_name].update_weight(smoothed_weight)
        
        return smoothed_weights
    
    def _compute_gradnorm_weights(self,
                                task_losses: Dict[str, torch.Tensor],
                                model: nn.Module,
                                optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """基于GradNorm启发式的权重计算"""
        if model is None or optimizer is None:
            logger.warning("GradNorm策略需要模型和优化器，返回均匀权重")
            return {task_name: 1.0 for task_name in self.task_names}
        
        # 计算每个任务的初始损失
        initial_losses = {}
        for task_name, loss_tensor in task_losses.items():
            initial_losses[task_name] = loss_tensor.detach().item()
        
        # 计算每个任务的相对反学习速度
        # 学习速度慢的任务获得更高权重
        learning_speeds = {}
        for task_name, metrics in self.task_metrics.items():
            learning_speeds[task_name] = metrics.learning_speed
        
        if len(learning_speeds) == 0:
            return {task_name: 1.0 for task_name in self.task_names}
        
        # 计算平均学习速度
        avg_learning_speed = np.mean(list(learning_speeds.values()))
        
        # 计算相对学习速度比率
        weight_ratios = {}
        for task_name, speed in learning_speeds.items():
            if avg_learning_speed != 0:
                # 学习速度越慢，权重越高
                ratio = max(0.5, min(2.0, avg_learning_speed / (speed + 1e-8)))
            else:
                ratio = 1.0
            weight_ratios[task_name] = ratio
        
        # 归一化权重
        total_ratio = sum(weight_ratios.values())
        if total_ratio > 0:
            normalized_weights = {
                task_name: ratio / total_ratio * len(self.task_names)
                for task_name, ratio in weight_ratios.items()
            }
        else:
            normalized_weights = {task_name: 1.0 for task_name in self.task_names}
        
        return normalized_weights
    
    def _compute_uncertainty_weights(self, 
                                   task_losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """基于不确定性的权重计算"""
        weights = {}
        
        for task_name, loss_tensor in task_losses.items():
            if task_name in self.log_vars:
                # 使用学习到的不确定性参数
                log_var = self.log_vars[task_name]
                weight = 1.0 / (2.0 * torch.exp(log_var))
                weights[task_name] = weight.item()
            else:
                # 基于损失方差的简单不确定性估计
                if task_name in self.task_metrics:
                    loss_history = list(self.task_metrics[task_name].loss_history)
                    if len(loss_history) >= 5:
                        variance = np.var(loss_history)
                        uncertainty = 1.0 / (1.0 + variance)
                        weights[task_name] = uncertainty
                    else:
                        weights[task_name] = 1.0
        
        # 归一化权重
        if weights:
            total_weight = sum(weights.values())
            if total_weight > 0:
                normalized_weights = {
                    task_name: weight / total_weight * len(weights)
                    for task_name, weight in weights.items()
                }
            else:
                normalized_weights = {task_name: 1.0 for task_name in weights.keys()}
        else:
            normalized_weights = {task_name: 1.0 for task_name in self.task_names}
        
        return normalized_weights
    
    def _compute_hybrid_weights(self,
                              task_losses: Dict[str, torch.Tensor],
                              model: nn.Module,
                              optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """混合策略权重计算（GradNorm + Uncertainty）"""
        # 计算GradNorm权重
        gradnorm_weights = self._compute_gradnorm_weights(task_losses, model, optimizer)
        
        # 计算不确定性权重
        uncertainty_weights = self._compute_uncertainty_weights(task_losses)
        
        # 合并权重（简单平均）
        hybrid_weights = {}
        for task_name in self.task_names:
            gradnorm_weight = gradnorm_weights.get(task_name, 1.0)
            uncertainty_weight = uncertainty_weights.get(task_name, 1.0)
            hybrid_weights[task_name] = (gradnorm_weight + uncertainty_weight) / 2.0
        
        return hybrid_weights
    
    def get_task_importance(self, task_name: str) -> float:
        """获取任务重要性分数（基于任务类型和训练进度）"""
        # 基础重要性（可根据任务类型定制）
        base_importance = {
            "text_understanding": 1.2,
            "image_understanding": 1.2,
            "cross_modal_alignment": 1.5,
            "retrieval": 1.0,
            "generation": 1.3,
            "planning": 1.4,
            "reasoning": 1.4,
        }.get(task_name, 1.0)
        
        # 基于训练进度调整重要性
        training_progress = min(1.0, self.training_step / 10000.0)
        
        # 早期阶段：基础任务更重要
        # 后期阶段：高级任务更重要
        if "cross_modal" in task_name or "generation" in task_name:
            # 高级任务，随训练进度增加重要性
            progress_factor = 0.5 + 0.5 * training_progress
        else:
            # 基础任务，随训练进度减少重要性
            progress_factor = 1.5 - 0.5 * training_progress
        
        return base_importance * progress_factor
    
    def adjust_weights_by_importance(self, weights: Dict[str, float]) -> Dict[str, float]:
        """根据任务重要性调整权重"""
        importance_adjusted = {}
        
        for task_name, weight in weights.items():
            importance = self.get_task_importance(task_name)
            importance_adjusted[task_name] = weight * importance
        
        # 归一化
        total = sum(importance_adjusted.values())
        if total > 0:
            normalized = {
                task_name: weight / total * len(importance_adjusted)
                for task_name, weight in importance_adjusted.items()
            }
        else:
            normalized = importance_adjusted
        
        return normalized
    
    def compute_weighted_loss(self, 
                            task_losses: Dict[str, torch.Tensor],
                            model: Optional[nn.Module] = None,
                            optimizer: Optional[torch.optim.Optimizer] = None) -> torch.Tensor:
        """
        计算加权总损失
        
        参数:
            task_losses: 任务损失字典
            model: 模型（用于某些策略）
            optimizer: 优化器（用于某些策略）
            
        返回:
            加权总损失
        """
        # 计算任务权重
        weights = self.compute_task_weights(task_losses, model, optimizer)
        
        # 根据任务重要性进一步调整权重
        weights = self.adjust_weights_by_importance(weights)
        
        # 计算加权总损失
        total_loss = torch.tensor(0.0, device=next(iter(task_losses.values())).device)
        for task_name, loss_tensor in task_losses.items():
            weight = weights.get(task_name, 1.0)
            total_loss = total_loss + weight * loss_tensor
        
        return total_loss
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """获取平衡器指标报告"""
        report = {
            "training_step": self.training_step,
            "balancing_strategy": self.balancing_strategy,
            "task_weights": {},
            "task_metrics": {}
        }
        
        for task_name, metrics in self.task_metrics.items():
            report["task_weights"][task_name] = metrics.current_weight
            report["task_metrics"][task_name] = {
                "learning_speed": metrics.learning_speed,
                "loss_history_length": len(metrics.loss_history),
                "importance": self.get_task_importance(task_name)
            }
        
        return report


def create_loss_balancer(config: Dict[str, Any]) -> AdaptiveLossBalancer:
    """
    从配置创建损失平衡器
    
    参数:
        config: 配置字典
        
    返回:
        AdaptiveLossBalancer实例
    """
    task_names = config.get("task_names", [])
    initial_weights = config.get("initial_weights", None)
    balancing_strategy = config.get("balancing_strategy", "hybrid")
    temperature = config.get("temperature", 1.0)
    update_frequency = config.get("update_frequency", 10)
    smoothing_factor = config.get("smoothing_factor", 0.1)
    
    return AdaptiveLossBalancer(
        task_names=task_names,
        initial_weights=initial_weights,
        balancing_strategy=balancing_strategy,
        temperature=temperature,
        update_frequency=update_frequency,
        smoothing_factor=smoothing_factor
    )