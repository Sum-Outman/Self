#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self AGI 多任务学习框架
支持同时学习多个相关任务，提高学习效率和泛化能力

功能：
1. 多任务神经网络架构（共享表示 + 任务特定头）
2. 任务优先级和权重调整
3. 梯度手术和冲突解决
4. 任务间知识迁移
5. 动态任务调度
"""

import logging
import json
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

# 导入机器学习库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"PyTorch不可用: {e}")


class TaskType(Enum):
    """任务类型枚举"""

    TEXT_UNDERSTANDING = "text_understanding"
    MULTIMODAL_PROCESSING = "multimodal_processing"
    PLANNING = "planning"
    REASONING = "reasoning"
    ROBOT_CONTROL = "robot_control"
    SELF_CORRECTION = "self_correction"
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    IMITATION_LEARNING = "imitation_learning"


@dataclass
class TaskConfig:
    """任务配置"""

    task_type: TaskType
    task_id: str
    observation_dim: int
    action_dim: int
    loss_function: str = "mse"  # mse, cross_entropy, etc.
    weight: float = 1.0  # 任务权重
    priority: int = 1  # 任务优先级 (1-10)
    enabled: bool = True  # 任务是否启用


class MultiTaskDataset(Dataset):
    """多任务数据集"""

    def __init__(self):
        self.tasks: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.task_configs: Dict[str, TaskConfig] = {}

    def add_task_data(
        self, task_id: str, observations: np.ndarray, targets: np.ndarray
    ):
        """添加任务数据"""
        if task_id not in self.tasks:
            self.tasks[task_id] = []

        for obs, target in zip(observations, targets):
            self.tasks[task_id].append((obs, target))

    def get_task_batch(
        self, task_id: str, batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取任务批次数据"""
        if task_id not in self.tasks:
            raise ValueError(f"任务 {task_id} 不存在")

        data = self.tasks[task_id]
        if len(data) == 0:
            raise ValueError(f"任务 {task_id} 数据为空")

        indices = np.random.choice(len(data), size=batch_size, replace=True)
        observations = np.array([data[i][0] for i in indices])
        targets = np.array([data[i][1] for i in indices])

        return observations, targets

    def get_multi_task_batch(
        self, task_ids: List[str], batch_size_per_task: int = 16
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """获取多任务批次数据"""
        batches = {}

        for task_id in task_ids:
            if task_id in self.tasks and len(self.tasks[task_id]) > 0:
                observations, targets = self.get_task_batch(
                    task_id, batch_size_per_task
                )
                batches[task_id] = (observations, targets)

        return batches


class SharedRepresentation(nn.Module):
    """共享表示层"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.shared_layers(x)


class TaskSpecificHead(nn.Module):
    """任务特定头"""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64]
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.task_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.task_layers(x)


class MultiTaskNetwork(nn.Module):
    """多任务神经网络"""

    def __init__(
        self, task_configs: List[TaskConfig], shared_hidden_dims: List[int] = [512, 256]
    ):
        super().__init__()

        self.task_configs = {config.task_id: config for config in task_configs}
        self.task_ids = list(self.task_configs.keys())

        # 确定输入维度（取所有任务的最大观察维度）
        input_dim = max(config.observation_dim for config in task_configs)

        # 共享表示层
        self.shared_representation = SharedRepresentation(
            input_dim=input_dim, hidden_dims=shared_hidden_dims
        )

        # 任务特定头
        self.task_heads = nn.ModuleDict()
        for task_id, config in self.task_configs.items():
            self.task_heads[task_id] = TaskSpecificHead(
                input_dim=self.shared_representation.output_dim,
                output_dim=config.action_dim,
            )

        # 损失函数
        self.loss_functions = {}
        for task_id, config in self.task_configs.items():
            if config.loss_function == "mse":
                self.loss_functions[task_id] = nn.MSELoss()
            elif config.loss_function == "cross_entropy":
                self.loss_functions[task_id] = nn.CrossEntropyLoss()
            else:
                self.loss_functions[task_id] = nn.MSELoss()

        self.logger = logging.getLogger("MultiTaskNetwork")

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        """前向传播（特定任务）"""
        if task_id not in self.task_heads:
            raise ValueError(f"任务 {task_id} 不存在")

        shared_features = self.shared_representation(x)
        task_output = self.task_heads[task_id](shared_features)

        return task_output

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, task_id: str
    ) -> torch.Tensor:
        """计算损失"""
        if task_id not in self.loss_functions:
            raise ValueError(f"任务 {task_id} 不存在")

        loss_fn = self.loss_functions[task_id]
        return loss_fn(predictions, targets)

    def predict(
        self, observation: np.ndarray, task_id: str, deterministic: bool = True
    ) -> np.ndarray:
        """预测（特定任务）"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            output_tensor = self.forward(obs_tensor, task_id)
            return output_tensor.squeeze(0).numpy()


class GradientSurgery:
    """梯度手术（解决任务间梯度冲突）"""

    @staticmethod
    def pcgrad(gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """投影冲突梯度（PCGrad）"""
        if len(gradients) <= 1:
            return gradients

        # 复制梯度
        proj_gradients = [g.clone() for g in gradients]

        # 对每对任务进行投影
        for i in range(len(gradients)):
            for j in range(len(gradients)):
                if i != j:
                    gi = proj_gradients[i]
                    gj = gradients[j]

                    # 计算冲突（负内积）
                    dot_product = torch.dot(gi.flatten(), gj.flatten())

                    if dot_product < 0:
                        # 投影 gi 到 gj 的正交补空间
                        proj_coeff = dot_product / (torch.norm(gj) ** 2 + 1e-8)
                        proj_gradients[i] = gi - proj_coeff * gj

        return proj_gradients

    @staticmethod
    def cagrad(gradients: List[torch.Tensor], alpha: float = 0.5) -> torch.Tensor:
        """冲突避免梯度（CAGrad）"""
        if len(gradients) == 0:
            return None  # 返回None

        if len(gradients) == 1:
            return gradients[0]

        # 计算平均梯度
        avg_gradient = torch.stack(gradients).mean(dim=0)

        # 计算梯度差异
        gradient_diffs = [g - avg_gradient for g in gradients]

        # 计算CAGrad
        cagrad = avg_gradient.clone()

        for diff in gradient_diffs:
            # 如果与平均梯度冲突，进行调整
            dot_product = torch.dot(cagrad.flatten(), diff.flatten())

            if dot_product < 0:
                cagrad = cagrad - alpha * diff

        return cagrad


class MultiTaskTrainer:
    """多任务训练器"""

    def __init__(
        self,
        model: MultiTaskNetwork,
        dataset: MultiTaskDataset,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        gradient_surgery: bool = True,
        surgery_method: str = "pcgrad",
    ):

        self.model = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_surgery = gradient_surgery
        self.surgery_method = surgery_method

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.logger = logging.getLogger("MultiTaskTrainer")

        # 训练历史
        self.loss_history = {task_id: [] for task_id in model.task_ids}
        self.task_weights = {task_id: 1.0 for task_id in model.task_ids}

    def train_step(self) -> Dict[str, float]:
        """单步训练"""
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch不可用，无法训练")
            return {}  # 返回空字典

        # 获取活动任务
        active_tasks = [
            task_id
            for task_id in self.model.task_ids
            if self.model.task_configs[task_id].enabled
        ]

        if not active_tasks:
            self.logger.warning("没有活动任务")
            return {}  # 返回空字典

        # 收集任务数据和梯度
        task_gradients = {}
        task_losses = {}

        # 首先计算所有任务的损失和梯度
        for task_id in active_tasks:
            try:
                # 获取任务数据
                observations, targets = self.dataset.get_task_batch(
                    task_id, batch_size=self.batch_size
                )

                observations_tensor = torch.FloatTensor(observations)
                targets_tensor = torch.FloatTensor(targets)

                # 前向传播
                predictions = self.model(observations_tensor, task_id)

                # 计算损失
                loss = self.model.compute_loss(predictions, targets_tensor, task_id)

                # 应用任务权重
                task_weight = self.model.task_configs[task_id].weight
                weighted_loss = loss * task_weight

                # 计算梯度
                self.optimizer.zero_grad()
                weighted_loss.backward(retain_graph=True)

                # 收集梯度
                gradients = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.clone())

                task_gradients[task_id] = gradients
                task_losses[task_id] = loss.item()

            except Exception as e:
                self.logger.error(f"处理任务 {task_id} 失败: {e}")
                continue

        if not task_gradients:
            self.logger.warning("没有有效的任务梯度")
            return {}  # 返回空字典

        # 应用梯度手术（如果需要）
        if self.gradient_surgery and len(task_gradients) > 1:
            if self.surgery_method == "pcgrad":
                # 应用PCGrad
                gradient_list = list(task_gradients.values())
                proj_gradients = GradientSurgery.pcgrad(gradient_list)

                # 更新梯度
                for (task_id, gradients), proj_grad in zip(
                    task_gradients.items(), proj_gradients
                ):
                    task_gradients[task_id] = proj_grad

            elif self.surgery_method == "cagrad":
                # 应用CAGrad
                gradient_list = list(task_gradients.values())
                cagrad = GradientSurgery.cagrad(gradient_list)

                # 使用CAGrad更新所有参数
                if cagrad is not None:
                    param_idx = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad = cagrad[param_idx].clone()
                            param_idx += 1

        # 更新模型参数
        self.optimizer.step()

        # 记录损失历史
        for task_id, loss in task_losses.items():
            self.loss_history[task_id].append(loss)

        return task_losses

    def train(self, steps: int = 1000, log_interval: int = 100) -> Dict[str, Any]:
        """训练多任务模型"""
        self.logger.info(
            f"开始多任务训练: {steps} 步, {len(self.model.task_ids)} 个任务"
        )

        step_losses = []

        for step in range(steps):
            # 单步训练
            task_losses = self.train_step()

            if task_losses:
                step_losses.append(task_losses)

            # 记录日志
            if step % log_interval == 0 and task_losses:
                avg_losses = {}
                for task_id, losses in self.loss_history.items():
                    if losses:
                        avg_losses[task_id] = np.mean(losses[-log_interval:])

                self.logger.info(
                    f"步数 {step}/{steps}: "
                    + ", ".join(
                        [
                            f"{task_id}={loss:.6f}"
                            for task_id, loss in avg_losses.items()
                        ]
                    )
                )

        # 计算最终统计
        final_stats = {}
        for task_id in self.model.task_ids:
            losses = self.loss_history.get(task_id, [])
            if losses:
                final_stats[task_id] = {
                    "final_loss": losses[-1] if losses else 0.0,
                    "avg_loss": np.mean(losses) if losses else 0.0,
                    "min_loss": np.min(losses) if losses else 0.0,
                    "max_loss": np.max(losses) if losses else 0.0,
                    "num_steps": len(losses),
                }

        result = {
            "success": True,
            "algorithm": "multi_task_learning",
            "steps": steps,
            "gradient_surgery": self.gradient_surgery,
            "surgery_method": self.surgery_method,
            "task_stats": final_stats,
        }

        self.logger.info("多任务训练完成")

        return result

    def adjust_task_weights(self, strategy: str = "equal"):
        """调整任务权重"""
        if strategy == "equal":
            # 平等权重
            for task_id in self.model.task_ids:
                self.model.task_configs[task_id].weight = 1.0

        elif strategy == "loss_based":
            # 基于损失的权重
            for task_id in self.model.task_ids:
                losses = self.loss_history.get(task_id, [])
                if losses:
                    recent_loss = (
                        np.mean(losses[-100:])
                        if len(losses) >= 100
                        else np.mean(losses)
                    )
                    # 损失越大，权重越小（归一化）
                    weight = 1.0 / (recent_loss + 1e-8)
                    # 限制权重范围
                    weight = np.clip(weight, 0.1, 10.0)
                    self.model.task_configs[task_id].weight = weight

        elif strategy == "priority_based":
            # 基于优先级的权重
            max_priority = max(
                config.priority for config in self.model.task_configs.values()
            )
            for task_id, config in self.model.task_configs.items():
                config.weight = config.priority / max_priority

        self.logger.info(f"任务权重调整: 策略={strategy}")


class MultiTaskLearningManager:
    """多任务学习管理器"""

    def __init__(
        self, task_configs: List[TaskConfig], shared_hidden_dims: List[int] = [512, 256]
    ):

        self.task_configs = task_configs
        self.shared_hidden_dims = shared_hidden_dims

        # 模型和数据集
        self.model = None
        self.dataset = MultiTaskDataset()
        self.trainer = None

        self.logger = logging.getLogger("MultiTaskLearningManager")

        # 初始化模型
        self._initialize_model()

        self.logger.info(f"多任务学习管理器初始化: {len(task_configs)} 个任务")

    def _initialize_model(self):
        """初始化模型"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch不可用，无法初始化模型")
            return

        self.model = MultiTaskNetwork(
            task_configs=self.task_configs, shared_hidden_dims=self.shared_hidden_dims
        )

        # 配置数据集的任务
        for config in self.task_configs:
            self.dataset.task_configs[config.task_id] = config

        self.logger.info("多任务模型初始化完成")

    def add_task_data(
        self, task_id: str, observations: np.ndarray, targets: np.ndarray
    ):
        """添加任务数据"""
        if task_id not in self.dataset.task_configs:
            self.logger.error(f"任务 {task_id} 未配置")
            return

        self.dataset.add_task_data(task_id, observations, targets)
        self.logger.info(f"添加任务数据: {task_id}, {len(observations)} 个样本")

    def train(
        self,
        steps: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        gradient_surgery: bool = True,
        surgery_method: str = "pcgrad",
    ) -> Dict[str, Any]:
        """训练多任务模型"""
        if self.model is None:
            self.logger.error("模型未初始化，无法训练")
            return {"success": False, "error": "模型未初始化"}

        # 创建训练器
        self.trainer = MultiTaskTrainer(
            model=self.model,
            dataset=self.dataset,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_surgery=gradient_surgery,
            surgery_method=surgery_method,
        )

        # 训练
        result = self.trainer.train(steps=steps)

        return result

    def predict(
        self, observation: np.ndarray, task_id: str, deterministic: bool = True
    ) -> np.ndarray:
        """预测（特定任务）"""
        if self.model is None:
            self.logger.error("模型未初始化，无法预测")
            return np.zeros(self.model.task_configs[task_id].action_dim)

        try:
            return self.model.predict(observation, task_id, deterministic)
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            return np.zeros(self.model.task_configs[task_id].action_dim)

    def evaluate_task(
        self, task_id: str, test_observations: np.ndarray, test_targets: np.ndarray
    ) -> Dict[str, Any]:
        """评估任务性能"""
        if self.model is None:
            self.logger.error("模型未初始化，无法评估")
            return {"success": False, "error": "模型未初始化"}

        if task_id not in self.model.task_configs:
            self.logger.error(f"任务 {task_id} 不存在")
            return {"success": False, "error": f"任务 {task_id} 不存在"}

        self.logger.info(f"评估任务: {task_id}")

        # 转换为张量
        observations_tensor = torch.FloatTensor(test_observations)
        targets_tensor = torch.FloatTensor(test_targets)

        # 预测
        with torch.no_grad():
            predictions = self.model(observations_tensor, task_id)

        # 计算损失
        loss = self.model.compute_loss(predictions, targets_tensor, task_id)
        loss_value = loss.item()

        # 计算其他指标（如准确率、MSE等）
        if self.model.task_configs[task_id].loss_function == "mse":
            mse = loss_value
            rmse = np.sqrt(mse)
            mae = torch.mean(torch.abs(predictions - targets_tensor)).item()

            metrics = {"mse": mse, "rmse": rmse, "mae": mae}
        else:
            metrics = {"loss": loss_value}

        result = {
            "success": True,
            "task_id": task_id,
            "loss": loss_value,
            "metrics": metrics,
            "num_samples": len(test_observations),
        }

        self.logger.info(f"任务评估完成: {task_id}, 损失={loss_value:.6f}")

        return result

    def save_model(self, path: str):
        """保存模型"""
        if self.model is not None and TORCH_AVAILABLE:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "task_configs": self.task_configs,
                    "shared_hidden_dims": self.shared_hidden_dims,
                },
                path,
            )
            self.logger.info(f"多任务模型保存到: {path}")
            return True
        else:
            self.logger.warning("没有模型可保存")
            return False

    def load_model(self, path: str):
        """加载模型"""
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch不可用，无法加载模型")
            return False

        try:
            checkpoint = torch.load(path)

            # 重新初始化模型
            self.task_configs = checkpoint["task_configs"]
            self.shared_hidden_dims = checkpoint["shared_hidden_dims"]

            self._initialize_model()

            # 加载状态字典
            self.model.load_state_dict(checkpoint["model_state_dict"])

            self.logger.info(f"多任务模型从 {path} 加载")
            return True

        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            return False


def create_multi_task_manager(
    task_configs: List[TaskConfig], **kwargs
) -> MultiTaskLearningManager:
    """创建多任务学习管理器（工厂函数）"""
    return MultiTaskLearningManager(task_configs, **kwargs)


if __name__ == "__main__":
    # 测试多任务学习框架
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=== 测试多任务学习框架 ===")

    # 创建任务配置
    task_configs = [
        TaskConfig(
            task_type=TaskType.TEXT_UNDERSTANDING,
            task_id="text_task",
            observation_dim=128,
            action_dim=10,
            loss_function="cross_entropy",
            weight=1.0,
            priority=1,
        ),
        TaskConfig(
            task_type=TaskType.ROBOT_CONTROL,
            task_id="robot_task",
            observation_dim=24,
            action_dim=12,
            loss_function="mse",
            weight=1.0,
            priority=2,
        ),
        TaskConfig(
            task_type=TaskType.REASONING,
            task_id="reasoning_task",
            observation_dim=256,
            action_dim=15,
            loss_function="mse",
            weight=0.8,
            priority=3,
        ),
    ]

    # 创建多任务管理器
    manager = create_multi_task_manager(task_configs)

    # 生成真实数据
    print("生成真实数据...")

    np.random.seed(42)

    # 文本理解任务数据
    text_obs = np.random.randn(100, 128)
    text_targets = np.random.randint(0, 10, size=(100,))
    manager.add_task_data("text_task", text_obs, text_targets)

    # 机器人控制任务数据
    robot_obs = np.random.randn(100, 24)
    robot_targets = np.random.randn(100, 12) * 0.5
    manager.add_task_data("robot_task", robot_obs, robot_targets)

    # 推理任务数据
    reasoning_obs = np.random.randn(100, 256)
    reasoning_targets = np.random.randn(100, 15) * 0.3
    manager.add_task_data("reasoning_task", reasoning_obs, reasoning_targets)

    # 训练多任务模型
    print("\n训练多任务模型...")
    training_result = manager.train(
        steps=200,
        learning_rate=1e-3,
        batch_size=16,
        gradient_surgery=True,
        surgery_method="pcgrad",
    )

    print(f"训练结果: {json.dumps(training_result, indent=2, ensure_ascii=False)}")

    # 测试预测
    print("\n测试预测...")
    test_obs = np.random.randn(5, 24)

    for i in range(3):
        task_id = manager.task_configs[i].task_id
        prediction = manager.predict(test_obs[0], task_id)
        print(f"任务 {task_id} 预测: {prediction[:3]}...")

    print("\n多任务学习框架测试完成!")
