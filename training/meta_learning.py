#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self AGI 元学习模块
实现元学习算法，使模型能够学习如何学习

功能：
1. MAML (Model-Agnostic Meta-Learning)
2. Reptile 算法
3. 元训练和元测试
4. 任务分布采样
5. 快速适应新任务
"""

import os
import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import random
import copy

# 导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"PyTorch不可用: {e}")


class MetaLearningAlgorithm(Enum):
    """元学习算法枚举"""

    MAML = "maml"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"  # Reptile算法
    FOMAML = "fomaml"  # First-Order MAML
    META_SGD = "meta_sgd"  # Meta-SGD


@dataclass
class TaskDistribution:
    """任务分布"""

    task_family: str  # 任务族名称
    task_params: Dict[str, Any]  # 任务参数分布
    num_tasks: int = 100  # 任务总数
    support_set_size: int = 10  # 支持集大小（每个任务）
    query_set_size: int = 10  # 查询集大小（每个任务）

    @property
    def tasks(self) -> List[Dict[str, Any]]:
        """获取所有任务列表"""
        # 返回虚拟任务列表，实际任务在采样时生成
        tasks_list = []
        for i in range(self.num_tasks):
            tasks_list.append(
                {"task_id": i, "task_family": self.task_family, "params": {}}
            )
        return tasks_list

    def sample_task(self, task_id: Optional[int] = None) -> Dict[str, Any]:
        """从分布中采样一个任务"""
        if task_id is not None and task_id < self.num_tasks:
            # 确定性采样
            random.seed(task_id)
            np.random.seed(task_id)
            torch.manual_seed(task_id)

        task = {
            "task_id": (
                task_id
                if task_id is not None
                else random.randint(0, self.num_tasks - 1)
            ),
            "task_family": self.task_family,
            "params": {},
        }

        # 从分布中采样任务参数
        for param_name, param_dist in self.task_params.items():
            if isinstance(param_dist, dict) and "type" in param_dist:
                dist_type = param_dist["type"]

                if dist_type == "uniform":
                    low = param_dist.get("low", 0.0)
                    high = param_dist.get("high", 1.0)
                    task["params"][param_name] = random.uniform(low, high)

                elif dist_type == "normal":
                    mean = param_dist.get("mean", 0.0)
                    std = param_dist.get("std", 1.0)
                    task["params"][param_name] = random.gauss(mean, std)

                elif dist_type == "categorical":
                    values = param_dist.get("values", [])
                    probs = param_dist.get("probs", None)
                    if probs and len(probs) == len(values):
                        task["params"][param_name] = random.choices(
                            values, weights=probs
                        )[0]
                    else:
                        task["params"][param_name] = random.choice(values)

                elif dist_type == "fixed":
                    task["params"][param_name] = param_dist.get("value", None)

            else:
                # 简单值
                task["params"][param_name] = param_dist

        return task


class MetaDataset(Dataset):
    """元学习数据集"""

    def __init__(
        self,
        task_distribution: TaskDistribution,
        num_tasks: int = 100,
        support_size: int = 10,
        query_size: int = 10,
    ):

        self.task_distribution = task_distribution
        self.num_tasks = num_tasks
        self.support_size = support_size
        self.query_size = query_size

        # 生成任务
        self.tasks = []
        for task_id in range(num_tasks):
            task = task_distribution.sample_task(task_id)
            self.tasks.append(task)

        self.logger = logging.getLogger("MetaDataset")
        self.logger.info(f"元数据集创建: {num_tasks} 个任务")

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, idx):
        """获取一个任务的支持集和查询集"""
        task = self.tasks[idx]

        # 为任务生成支持集和查询集
        support_set = self._generate_data_for_task(task, self.support_size)
        query_set = self._generate_data_for_task(task, self.query_size)

        return {"task": task, "support_set": support_set, "query_set": query_set}

    def _generate_data_for_task(
        self, task: Dict[str, Any], num_samples: int
    ) -> Dict[str, torch.Tensor]:
        """为特定任务生成数据"""

        # 这里应该根据任务类型生成实际数据
        # 完整实现：生成真实数据

        task_family = task["task_family"]
        task_params = task["params"]

        # 生成输入数据（随机）
        inputs = torch.randn(num_samples, 20)  # 20维输入

        # 根据任务族生成目标
        if task_family == "regression":
            # 线性回归任务
            weights = torch.tensor(task_params.get("weights", np.random.randn(20, 1)))
            bias = task_params.get("bias", random.uniform(-1, 1))
            targets = inputs @ weights + bias
            targets = targets.squeeze()

        elif task_family == "classification":
            # 二分类任务
            weights = torch.tensor(task_params.get("weights", np.random.randn(20)))
            bias = task_params.get("bias", random.uniform(-1, 1))
            logits = inputs @ weights + bias
            targets = (logits > 0).float()

        elif task_family == "sinusoidal":
            # 正弦函数回归（标准元学习任务）
            amplitude = task_params.get("amplitude", random.uniform(0.1, 5.0))
            phase = task_params.get("phase", random.uniform(0, 2 * np.pi))

            # 输入是1维的x值
            x = torch.randn(num_samples, 1) * 2  # [-2, 2]范围
            inputs = x
            targets = amplitude * torch.sin(x + phase)

        else:
            # 默认：线性回归
            weights = torch.tensor(np.random.randn(20, 1))
            bias = random.uniform(-1, 1)
            targets = inputs @ weights + bias
            targets = targets.squeeze()

        return {"inputs": inputs, "targets": targets}


class MAML:
    """MAML (Model-Agnostic Meta-Learning) 算法"""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 1,
        adaptation_steps: Optional[int] = None,
        inner_algorithm: str = "sgd",
        first_order: bool = False,
    ):
        """
        初始化MAML

        参数:
            model: 基础模型（可选，如未提供则创建默认模型）
            inner_lr: 内部学习率（任务特定适应）
            meta_lr: 元学习率（跨任务学习）
            num_inner_steps: 内部更新步数
            adaptation_steps: 适应步数（num_inner_steps的别名，如提供则覆盖num_inner_steps）
            inner_algorithm: 内部优化算法 ("sgd", "adam")
            first_order: 是否使用一阶近似（FOMAML）
        """

        # 处理adaptation_steps别名
        if adaptation_steps is not None:
            num_inner_steps = adaptation_steps

        # 如果没有提供模型，创建默认的简单线性模型
        if model is None:
            model = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.inner_algorithm = inner_algorithm
        self.first_order = first_order

        # 元优化器
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)

        # 损失函数
        self.loss_fn = nn.MSELoss()  # 默认为回归任务

        self.logger = logging.getLogger("MAML")

        self.logger.info(
            f"MAML初始化: 内部学习率={inner_lr}, 元学习率={meta_lr}, "
            f"内部步数={num_inner_steps}, 一阶近似={first_order}"
        )

    def adapt(
        self, task_data: Dict[str, torch.Tensor], loss_fn: Optional[Callable] = None
    ) -> nn.Module:
        """
        快速适应新任务

        参数:
            task_data: 任务数据，包含"inputs"和"targets"
            loss_fn: 可选的自定义损失函数

        返回:
            adapted_model: 适应后的模型
        """

        if loss_fn is None:
            loss_fn = self.loss_fn

        # 克隆模型（创建任务特定副本）
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()

        # 内部优化器
        if self.inner_algorithm == "adam":
            inner_optimizer = optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        else:
            inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        # 内部循环（任务特定适应）
        inputs = task_data["inputs"]
        targets = task_data["targets"]

        for step in range(self.num_inner_steps):
            # 前向传播
            predictions = adapted_model(inputs)
            loss = loss_fn(predictions, targets)

            # 反向传播和更新
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return adapted_model

    def meta_update(
        self, meta_batch: List[Dict[str, Any]], loss_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        执行元更新（跨任务学习）

        参数:
            meta_batch: 元批次，每个元素包含"task", "support_set", "query_set"
            loss_fn: 可选的自定义损失函数

        返回:
            训练统计信息
        """

        if loss_fn is None:
            loss_fn = self.loss_fn

        self.model.train()
        self.meta_optimizer.zero_grad()

        total_meta_loss = 0.0
        task_losses = []

        # 对元批次中的每个任务
        for task_data in meta_batch:
            support_set = task_data["support_set"]
            query_set = task_data["query_set"]

            # 克隆模型参数（用于梯度计算）
            fast_weights = {}
            for name, param in self.model.named_parameters():
                fast_weights[name] = param.clone()

            # 内部循环（在支持集上适应）
            for step in range(self.num_inner_steps):
                # 使用fast_weights的前向传播
                predictions = self._forward_with_weights(
                    support_set["inputs"], fast_weights
                )
                support_loss = loss_fn(predictions, support_set["targets"])

                # 计算梯度并更新fast_weights
                grads = torch.autograd.grad(
                    support_loss,
                    fast_weights.values(),
                    create_graph=not self.first_order,  # 二阶梯度需要计算图
                )

                # 更新fast_weights
                for (name, weight), grad in zip(fast_weights.items(), grads):
                    if grad is not None:
                        fast_weights[name] = weight - self.inner_lr * grad

            # 在查询集上评估（元损失）
            query_predictions = self._forward_with_weights(
                query_set["inputs"], fast_weights
            )
            query_loss = loss_fn(query_predictions, query_set["targets"])

            # 累加元损失
            total_meta_loss += query_loss
            task_losses.append(query_loss.item())

        # 平均元损失
        avg_meta_loss = total_meta_loss / len(meta_batch)

        # 反向传播（元梯度）
        avg_meta_loss.backward()
        self.meta_optimizer.step()

        stats = {
            "meta_loss": avg_meta_loss.item(),
            "avg_task_loss": np.mean(task_losses),
            "std_task_loss": np.std(task_losses) if len(task_losses) > 1 else 0.0,
            "num_tasks": len(meta_batch),
        }

        return stats

    def _forward_with_weights(
        self, inputs: torch.Tensor, weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """使用指定的权重执行前向传播"""

        # 临时保存原始权重
        original_weights = {}
        for name, param in self.model.named_parameters():
            original_weights[name] = param.data.clone()
            param.data = weights[name]

        # 前向传播
        with torch.no_grad():
            outputs = self.model(inputs)

        # 恢复原始权重
        for name, param in self.model.named_parameters():
            param.data = original_weights[name]

        return outputs

    def evaluate(
        self,
        meta_dataset: MetaDataset,
        num_tasks: int = 10,
        adaptation_steps: int = 5,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        评估元学习性能

        参数:
            meta_dataset: 元数据集
            num_tasks: 评估任务数
            adaptation_steps: 适应步数
            loss_fn: 可选的自定义损失函数

        返回:
            评估结果
        """

        if loss_fn is None:
            loss_fn = self.loss_fn

        self.model.eval()

        task_results = []
        pre_adaptation_losses = []
        post_adaptation_losses = []

        # 随机选择评估任务
        eval_indices = random.sample(
            range(len(meta_dataset)), min(num_tasks, len(meta_dataset))
        )

        for idx in eval_indices:
            task_data = meta_dataset[idx]
            support_set = task_data["support_set"]
            query_set = task_data["query_set"]

            # 适应前的性能
            with torch.no_grad():
                pre_predictions = self.model(query_set["inputs"])
                pre_loss = loss_fn(pre_predictions, query_set["targets"]).item()

            # 快速适应
            adapted_model = self.adapt(support_set, loss_fn)

            # 适应后的性能
            with torch.no_grad():
                post_predictions = adapted_model(query_set["inputs"])
                post_loss = loss_fn(post_predictions, query_set["targets"]).item()

            # 记录结果
            task_results.append(
                {
                    "task_id": task_data["task"]["task_id"],
                    "pre_adaptation_loss": pre_loss,
                    "post_adaptation_loss": post_loss,
                    "improvement": pre_loss - post_loss,
                }
            )

            pre_adaptation_losses.append(pre_loss)
            post_adaptation_losses.append(post_loss)

        # 计算统计信息
        avg_pre_loss = np.mean(pre_adaptation_losses)
        avg_post_loss = np.mean(post_adaptation_losses)
        avg_improvement = avg_pre_loss - avg_post_loss

        results = {
            "avg_pre_adaptation_loss": avg_pre_loss,
            "avg_post_adaptation_loss": avg_post_loss,
            "avg_improvement": avg_improvement,
            "improvement_ratio": avg_improvement / max(avg_pre_loss, 1e-8),
            "num_tasks_evaluated": len(task_results),
            "task_results": task_results,
        }

        self.logger.info(
            f"元学习评估: 适应前损失={avg_pre_loss:.4f}, "
            f"适应后损失={avg_post_loss:.4f}, "
            f"改进={avg_improvement:.4f} ({results['improvement_ratio'] * 100:.1f}%)"
        )

        return results


class Reptile:
    """Reptile 元学习算法"""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 1,
        adaptation_steps: Optional[int] = None,
    ):
        """
        初始化Reptile

        参数:
            model: 基础模型（可选，如未提供则创建默认模型）
            inner_lr: 内部学习率
            meta_lr: 元学习率
            num_inner_steps: 内部更新步数
            adaptation_steps: 适应步数（num_inner_steps的别名，如提供则覆盖num_inner_steps）
        """

        # 处理adaptation_steps别名
        if adaptation_steps is not None:
            num_inner_steps = adaptation_steps

        # 如果没有提供模型，创建默认的简单线性模型
        if model is None:
            model = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps

        # 元优化器
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)

        # 损失函数
        self.loss_fn = nn.MSELoss()

        self.logger = logging.getLogger("Reptile")

        self.logger.info(
            f"Reptile初始化: 内部学习率={inner_lr}, 元学习率={meta_lr}, "
            f"内部步数={num_inner_steps}"
        )

    def meta_update(
        self, meta_batch: List[Dict[str, Any]], loss_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        执行Reptile元更新

        参数:
            meta_batch: 元批次
            loss_fn: 可选的自定义损失函数

        返回:
            训练统计信息
        """

        if loss_fn is None:
            loss_fn = self.loss_fn

        self.model.train()

        task_losses = []
        total_meta_grad = None

        # 对元批次中的每个任务
        for task_data in meta_batch:
            support_set = task_data["support_set"]

            # 克隆模型
            task_model = copy.deepcopy(self.model)
            task_model.train()

            # 任务优化器
            task_optimizer = optim.SGD(task_model.parameters(), lr=self.inner_lr)

            # 内部循环（任务特定训练）
            for step in range(self.num_inner_steps):
                predictions = task_model(support_set["inputs"])
                loss = loss_fn(predictions, support_set["targets"])

                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()

                task_losses.append(loss.item())

            # 计算模型参数差异（作为梯度）
            if total_meta_grad is None:
                total_meta_grad = []
                for p in task_model.parameters():
                    total_meta_grad.append(p.data.clone())
            else:
                for i, p in enumerate(task_model.parameters()):
                    total_meta_grad[i] += p.data

        # 计算平均参数差异
        if total_meta_grad:
            for i in range(len(total_meta_grad)):
                total_meta_grad[i] /= len(meta_batch)

        # 更新元模型：model ← model + meta_lr * (avg_task_model - model)
        # 等价于：model ← model + meta_lr * total_meta_grad - meta_lr * model
        # 完整实现：使用优化器更新

        self.meta_optimizer.zero_grad()

        # 手动设置梯度（基于参数差异）
        for model_param, avg_param in zip(self.model.parameters(), total_meta_grad):
            if model_param.grad is None:
                model_param.grad = torch.zeros_like(model_param.data)
            # 梯度 = (avg_task_param - model_param) / meta_lr
            model_param.grad.data = (avg_param - model_param.data) / self.meta_lr

        self.meta_optimizer.step()

        stats = {
            "avg_task_loss": np.mean(task_losses) if task_losses else 0.0,
            "num_tasks": len(meta_batch),
        }

        return stats


class MetaLearningManager:
    """元学习管理器"""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.MAML,
        config: Optional[Dict[str, Any]] = None,
    ):

        # 如果没有提供模型，创建默认的简单模型
        if model is None:
            model = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        self.model = model
        self.algorithm = algorithm
        self.config = config or {}

        # 创建元学习算法实例
        if algorithm == MetaLearningAlgorithm.MAML:
            self.meta_learner = MAML(
                model=model,
                inner_lr=self.config.get("inner_lr", 0.01),
                meta_lr=self.config.get("meta_lr", 0.001),
                num_inner_steps=self.config.get("num_inner_steps", 1),
                first_order=self.config.get("first_order", False),
            )

        elif algorithm == MetaLearningAlgorithm.REPTILE:
            self.meta_learner = Reptile(
                model=model,
                inner_lr=self.config.get("inner_lr", 0.01),
                meta_lr=self.config.get("meta_lr", 0.001),
                num_inner_steps=self.config.get("num_inner_steps", 1),
            )

        elif algorithm == MetaLearningAlgorithm.FOMAML:
            self.meta_learner = MAML(
                model=model,
                inner_lr=self.config.get("inner_lr", 0.01),
                meta_lr=self.config.get("meta_lr", 0.001),
                num_inner_steps=self.config.get("num_inner_steps", 1),
                first_order=True,  # 一阶近似
            )

        else:
            self.logger.warning(f"不支持的算法: {algorithm}, 使用MAML")
            self.meta_learner = MAML(model=model)

        self.logger = logging.getLogger("MetaLearningManager")
        self.logger.info(f"元学习管理器初始化: 算法={algorithm.value}")

    def train(
        self,
        meta_dataset: MetaDataset,
        num_epochs: int = 100,
        meta_batch_size: int = 4,
        eval_interval: int = 10,
    ) -> Dict[str, Any]:
        """
        训练元学习模型

        参数:
            meta_dataset: 元数据集
            num_epochs: 训练轮数
            meta_batch_size: 元批次大小
            eval_interval: 评估间隔

        返回:
            训练结果
        """

        self.logger.info(
            f"开始元学习训练: {num_epochs} 轮, "
            f"元批次大小={meta_batch_size}, 数据集大小={len(meta_dataset)}"
        )

        training_stats = []
        best_loss = float("inf")

        for epoch in range(num_epochs):
            # 创建数据加载器
            dataloader = DataLoader(
                meta_dataset, batch_size=meta_batch_size, shuffle=True
            )

            epoch_losses = []

            for meta_batch in dataloader:
                # 执行元更新
                stats = self.meta_learner.meta_update(meta_batch)
                epoch_losses.append(
                    stats.get("meta_loss", stats.get("avg_task_loss", 0.0))
                )

            # 计算平均损失
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0

            # 记录统计信息
            epoch_stats = {
                "epoch": epoch,
                "loss": avg_epoch_loss,
                "algorithm": self.algorithm.value,
            }

            training_stats.append(epoch_stats)

            # 定期评估
            if epoch % eval_interval == 0 or epoch == num_epochs - 1:
                eval_results = self.evaluate(meta_dataset, num_tasks=5)
                epoch_stats.update(eval_results)

                self.logger.info(
                    f"轮次 {epoch}/{num_epochs}: 损失={avg_epoch_loss:.6f}, "
                    f"适应后损失={eval_results.get('avg_post_adaptation_loss', 0.0):.6f}"
                )

                # 保存最佳模型
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    self._save_checkpoint(f"best_epoch_{epoch}")

            else:
                if epoch % 10 == 0:
                    self.logger.info(
                        f"轮次 {epoch}/{num_epochs}: 损失={avg_epoch_loss:.6f}"
                    )

        # 最终评估
        final_eval = self.evaluate(meta_dataset, num_tasks=20)

        result = {
            "success": True,
            "algorithm": self.algorithm.value,
            "num_epochs": num_epochs,
            "final_loss": training_stats[-1]["loss"] if training_stats else 0.0,
            "best_loss": best_loss,
            "training_stats": training_stats,
            "final_evaluation": final_eval,
        }

        self.logger.info(f"元学习训练完成: 最终损失={result['final_loss']:.6f}")

        return result

    def evaluate(
        self, meta_dataset: MetaDataset, num_tasks: int = 10, adaptation_steps: int = 5
    ) -> Dict[str, Any]:
        """评估元学习模型"""

        if hasattr(self.meta_learner, "evaluate"):
            return self.meta_learner.evaluate(
                meta_dataset=meta_dataset,
                num_tasks=num_tasks,
                adaptation_steps=adaptation_steps,
            )
        else:
            # 完整评估
            self.model.eval()

            task_losses = []

            # 随机选择任务
            eval_indices = random.sample(
                range(len(meta_dataset)), min(num_tasks, len(meta_dataset))
            )

            for idx in eval_indices:
                task_data = meta_dataset[idx]
                query_set = task_data["query_set"]

                with torch.no_grad():
                    predictions = self.model(query_set["inputs"])
                    loss = self.meta_learner.loss_fn(predictions, query_set["targets"])
                    task_losses.append(loss.item())

            avg_loss = np.mean(task_losses) if task_losses else 0.0

            return {"avg_loss": avg_loss, "num_tasks": len(task_losses)}

    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_dir = self.config.get("checkpoint_dir", "meta_learning_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pt")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "meta_learner_state_dict": (
                self.meta_learner.meta_optimizer.state_dict()
                if hasattr(self.meta_learner, "meta_optimizer")
                else None
            ),
            "algorithm": self.algorithm.value,
            "config": self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点保存到: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        try:
            checkpoint = torch.load(checkpoint_path)

            self.model.load_state_dict(checkpoint["model_state_dict"])

            if (
                hasattr(self.meta_learner, "meta_optimizer")
                and checkpoint["meta_learner_state_dict"] is not None
            ):
                self.meta_learner.meta_optimizer.load_state_dict(
                    checkpoint["meta_learner_state_dict"]
                )

            self.logger.info(f"检查点加载成功: {checkpoint_path}")
            return True

        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return False


def create_sinusoidal_task_distribution(
    num_tasks: int = 1000, support_set_size: int = 10, query_set_size: int = 10
) -> TaskDistribution:
    """创建正弦函数回归任务分布（标准元学习基准）"""

    return TaskDistribution(
        task_family="sinusoidal",
        task_params={
            "amplitude": {"type": "uniform", "low": 0.1, "high": 5.0},
            "phase": {"type": "uniform", "low": 0, "high": 2 * 3.14159},
        },
        num_tasks=num_tasks,
        support_set_size=support_set_size,
        query_set_size=query_set_size,
    )


def create_classification_task_distribution(num_classes: int = 5) -> TaskDistribution:
    """创建分类任务分布"""

    return TaskDistribution(
        task_family="classification",
        task_params={
            "weights": {"type": "normal", "mean": 0.0, "std": 1.0},
            "bias": {"type": "uniform", "low": -1.0, "high": 1.0},
        },
        num_tasks=1000,
        support_set_size=20,
        query_set_size=20,
    )


if __name__ == "__main__":
    # 测试元学习模块
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=== 测试元学习模块 ===")

    # 创建简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=1, output_dim=1, hidden_dim=64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.network(x)

    # 创建模型
    model = SimpleModel(input_dim=1, output_dim=1, hidden_dim=40)

    # 创建正弦函数任务分布
    task_distribution = create_sinusoidal_task_distribution()

    # 创建元数据集
    meta_dataset = MetaDataset(
        task_distribution=task_distribution,
        num_tasks=100,
        support_size=10,
        query_size=10,
    )

    # 创建元学习管理器
    manager = MetaLearningManager(
        model=model,
        algorithm=MetaLearningAlgorithm.MAML,
        config={
            "inner_lr": 0.01,
            "meta_lr": 0.001,
            "num_inner_steps": 1,
            "first_order": True,  # 使用FOMAML（更快）
        },
    )

    # 快速训练测试
    print("开始元学习训练测试...")

    try:
        result = manager.train(
            meta_dataset=meta_dataset,
            num_epochs=20,  # 测试时使用较少的轮次
            meta_batch_size=4,
            eval_interval=5,
        )

        print(f"训练结果: {json.dumps(result, indent=2, ensure_ascii=False)}")

        # 最终评估
        final_eval = manager.evaluate(meta_dataset, num_tasks=10)
        print(f"\n最终评估结果: {json.dumps(final_eval, indent=2, ensure_ascii=False)}")

        print("\n元学习模块测试完成!")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("这可能是因为PyTorch不可用或版本不兼容")
        print("请确保已安装正确版本的PyTorch")
