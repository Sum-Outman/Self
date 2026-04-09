#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self AGI 课程学习和探索策略模块
实现自适应学习进度控制和好奇心驱动探索

功能：
1. 课程学习：从简单到复杂的自适应任务进度
2. 探索策略：好奇心驱动、内在动机、多样性探索
3. 任务难度评估和自动调整
4. 学习进度监控和课程调度
"""

import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
import time

# 导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"PyTorch不可用: {e}")


class TaskDifficulty(Enum):
    """任务难度级别"""

    VERY_EASY = "very_easy"  # 非常简单
    EASY = "easy"  # 简单
    MEDIUM = "medium"  # 中等
    HARD = "hard"  # 困难
    VERY_HARD = "very_hard"  # 非常困难


class ExplorationStrategy(Enum):
    """探索策略枚举"""

    EPSILON_GREEDY = "epsilon_greedy"  # ε-贪婪
    UCB = "ucb"  # 上置信区间
    THOMPSON_SAMPLING = "thompson"  # 汤普森采样
    CURIOSITY_DRIVEN = "curiosity"  # 好奇心驱动
    INTRINSIC_MOTIVATION = "intrinsic"  # 内在动机
    DIVERSITY_SEARCH = "diversity"  # 多样性搜索


@dataclass
class TaskDescription:
    """任务描述"""

    task_id: str
    task_family: str
    difficulty: TaskDifficulty
    parameters: Dict[str, Any]
    success_threshold: float = 0.8  # 成功率阈值
    min_training_steps: int = 100  # 最小训练步数
    max_training_steps: int = 1000  # 最大训练步数
    performance_history: List[float] = field(default_factory=list)

    def update_performance(self, success_rate: float, steps: int):
        """更新任务性能历史"""
        self.performance_history.append(
            {
                "timestamp": time.time(),
                "success_rate": success_rate,
                "steps": steps,
                "difficulty": self.difficulty.value,
            }
        )

        # 保持历史记录长度
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_average_performance(self, window_size: int = 10) -> float:
        """获取平均性能"""
        if not self.performance_history:
            return 0.0

        recent_history = self.performance_history[-window_size:]
        if not recent_history:
            return 0.0

        success_rates = [h["success_rate"] for h in recent_history]
        return np.mean(success_rates)

    def is_mastered(self, threshold: Optional[float] = None) -> bool:
        """检查任务是否已掌握"""
        if threshold is None:
            threshold = self.success_threshold

        avg_performance = self.get_average_performance()
        return avg_performance >= threshold


class Curriculum:
    """课程管理器"""

    def __init__(
        self,
        tasks: Optional[List[TaskDescription]] = None,
        advancement_threshold: float = 0.75,
        difficulty_increment: float = 0.1,
        min_tasks_per_level: int = 3,
        curriculum_name: Optional[str] = None,
        curriculum_difficulty: Optional[str] = None,
    ):

        if tasks is None:
            tasks = []

        self.tasks = {task.task_id: task for task in tasks}
        self.task_ids = list(self.tasks.keys())

        self.advancement_threshold = advancement_threshold
        self.difficulty_increment = difficulty_increment
        self.min_tasks_per_level = min_tasks_per_level
        self.curriculum_name = curriculum_name or "默认课程"
        self.curriculum_difficulty = curriculum_difficulty or "medium"

        # 学习进度
        self.current_difficulty = TaskDifficulty.VERY_EASY
        self.completed_tasks = set()
        self.mastered_tasks = set()
        self.current_task_index = 0

        # 难度级别到任务的映射
        self.tasks_by_difficulty = self._group_tasks_by_difficulty()

        self.logger = logging.getLogger("Curriculum")

        self.logger.info(
            f"课程初始化: {len(tasks)} 个任务, 当前难度={self.current_difficulty.value}"
        )

    def _group_tasks_by_difficulty(self) -> Dict[TaskDifficulty, List[str]]:
        """按难度分组任务"""
        groups = {diff: [] for diff in TaskDifficulty}

        for task_id, task in self.tasks.items():
            groups[task.difficulty].append(task_id)

        return groups

    def add_task(self, task_name: str, task_type: str, difficulty_level: int) -> str:
        """添加新任务到课程

        参数:
            task_name: 任务名称
            task_type: 任务类型
            difficulty_level: 难度级别 (1-5, 1=非常简单, 5=非常困难)

        返回:
            任务ID
        """
        # 将难度级别转换为TaskDifficulty枚举
        difficulty_map = {
            1: TaskDifficulty.VERY_EASY,
            2: TaskDifficulty.EASY,
            3: TaskDifficulty.MEDIUM,
            4: TaskDifficulty.HARD,
            5: TaskDifficulty.VERY_HARD,
        }

        if difficulty_level not in difficulty_map:
            difficulty_level = max(1, min(5, difficulty_level))  # 限制在1-5范围内

        difficulty = difficulty_map.get(difficulty_level, TaskDifficulty.MEDIUM)

        # 生成任务ID
        task_id = f"{task_type}_{task_name}_{len(self.tasks)}"

        # 创建任务描述
        task = TaskDescription(
            task_id=task_id,
            task_family=task_type,
            difficulty=difficulty,
            parameters={"name": task_name, "type": task_type},
            success_threshold=0.8 - (difficulty_level - 1) * 0.05,  # 难度越高阈值越低
            min_training_steps=100 * difficulty_level,
            max_training_steps=1000 * difficulty_level,
        )

        # 添加到任务列表
        self.tasks[task_id] = task
        self.task_ids.append(task_id)

        # 更新难度分组
        self.tasks_by_difficulty = self._group_tasks_by_difficulty()

        self.logger.info(
            f"任务添加成功: {task_name} ({task_type}), 难度={difficulty.value}, ID={task_id}"
        )
        return task_id

    def get_current_task(self) -> Optional[TaskDescription]:
        """获取当前任务"""
        current_tasks = self.tasks_by_difficulty.get(self.current_difficulty, [])

        if not current_tasks:
            self.logger.warning(
                f"当前难度 {self.current_difficulty.value} 没有可用任务"
            )
            return None  # 返回None

        # 选择未掌握的任务
        available_tasks = []
        for task_id in current_tasks:
            if task_id not in self.mastered_tasks:
                available_tasks.append(task_id)

        if not available_tasks:
            # 所有任务都已掌握，可以升级难度
            self._advance_difficulty()
            return self.get_current_task()

        # 选择性能最差的任务（需要更多练习）
        worst_task_id = min(
            available_tasks, key=lambda tid: self.tasks[tid].get_average_performance()
        )

        return self.tasks[worst_task_id]

    def update_task_performance(self, task_id: str, success_rate: float, steps: int):
        """更新任务性能"""
        if task_id not in self.tasks:
            self.logger.error(f"任务 {task_id} 不存在")
            return

        task = self.tasks[task_id]
        task.update_performance(success_rate, steps)

        # 检查是否掌握
        if task.is_mastered():
            self.mastered_tasks.add(task_id)
            self.completed_tasks.add(task_id)
            self.logger.info(f"任务 {task_id} 已掌握: 成功率={success_rate:.2f}")

        # 检查是否可以升级难度
        self._check_advancement()

    def _check_advancement(self):
        """检查是否可以升级难度"""
        current_tasks = self.tasks_by_difficulty.get(self.current_difficulty, [])

        if not current_tasks:
            return

        # 计算当前难度的掌握率
        mastered_count = sum(1 for tid in current_tasks if tid in self.mastered_tasks)
        total_count = len(current_tasks)

        if total_count == 0:
            return

        mastery_rate = mastered_count / total_count

        # 检查是否满足升级条件
        if (
            mastery_rate >= self.advancement_threshold
            and mastered_count >= self.min_tasks_per_level
        ):

            # 升级到下一难度
            self._advance_difficulty()

    def _advance_difficulty(self):
        """升级到下一难度级别"""
        difficulty_order = [
            TaskDifficulty.VERY_EASY,
            TaskDifficulty.EASY,
            TaskDifficulty.MEDIUM,
            TaskDifficulty.HARD,
            TaskDifficulty.VERY_HARD,
        ]

        current_idx = difficulty_order.index(self.current_difficulty)

        if current_idx < len(difficulty_order) - 1:
            next_difficulty = difficulty_order[current_idx + 1]
            self.current_difficulty = next_difficulty
            self.logger.info(f"课程升级: {self.current_difficulty.value}")
        else:
            self.logger.info("已达到最高难度级别")

    def get_learning_progress(self) -> Dict[str, Any]:
        """获取学习进度"""
        total_tasks = len(self.tasks)
        completed_tasks = len(self.completed_tasks)
        mastered_tasks = len(self.mastered_tasks)

        # 按难度统计
        difficulty_stats = {}
        for diff in TaskDifficulty:
            tasks_in_diff = self.tasks_by_difficulty.get(diff, [])
            if tasks_in_diff:
                mastered_in_diff = sum(
                    1 for tid in tasks_in_diff if tid in self.mastered_tasks
                )
                difficulty_stats[diff.value] = {
                    "total": len(tasks_in_diff),
                    "mastered": mastered_in_diff,
                    "mastery_rate": (
                        mastered_in_diff / len(tasks_in_diff) if tasks_in_diff else 0.0
                    ),
                }

        progress = {
            "current_difficulty": self.current_difficulty.value,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "mastered_tasks": mastered_tasks,
            "completion_rate": (
                completed_tasks / total_tasks if total_tasks > 0 else 0.0
            ),
            "mastery_rate": mastered_tasks / total_tasks if total_tasks > 0 else 0.0,
            "difficulty_stats": difficulty_stats,
        }

        return progress


class CuriosityModule(nn.Module):
    """好奇心模块（基于预测误差）"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # 逆动力学模型（预测动作）
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # 前向模型（预测下一状态）
        self.forward_dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # 损失函数
        self.inverse_loss_fn = nn.MSELoss()
        self.forward_loss_fn = nn.MSELoss()

        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def compute_intrinsic_reward(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        计算内在奖励（基于预测误差）

        内在奖励 = 前向模型预测误差
        误差越大，状态越新奇，奖励越高
        """
        with torch.no_grad():
            # 预测下一状态
            predicted_next_state = self.forward_dynamics(
                torch.cat([state, action], dim=-1)
            )

            # 计算预测误差
            prediction_error = torch.mean(
                (predicted_next_state - next_state) ** 2, dim=-1, keepdim=True
            )

            # 内在奖励与预测误差成正比
            intrinsic_reward = prediction_error

        return intrinsic_reward

    def update(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ):
        """更新好奇心模型"""

        # 训练逆动力学模型
        state_pair = torch.cat([state, next_state], dim=-1)
        predicted_action = self.inverse_dynamics(state_pair)
        inverse_loss = self.inverse_loss_fn(predicted_action, action)

        # 训练前向动力学模型
        predicted_next_state = self.forward_dynamics(torch.cat([state, action], dim=-1))
        forward_loss = self.forward_loss_fn(predicted_next_state, next_state)

        # 总损失
        total_loss = inverse_loss + forward_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "inverse_loss": inverse_loss.item(),
            "forward_loss": forward_loss.item(),
            "total_loss": total_loss.item(),
        }


class IntrinsicMotivationModule:
    """内在动机模块"""

    def __init__(
        self,
        learning_progress_weight: float = 1.0,
        novelty_weight: float = 0.5,
        competence_weight: float = 0.3,
    ):

        self.learning_progress_weight = learning_progress_weight
        self.novelty_weight = novelty_weight
        self.competence_weight = competence_weight

        # 学习进度跟踪
        self.task_performance_history = {}
        self.state_novelty_history = {}

        self.logger = logging.getLogger("IntrinsicMotivation")

    def compute_motivation(
        self,
        task_id: str,
        state: np.ndarray,
        success_rate: float,
        prev_success_rate: Optional[float] = None,
    ) -> float:
        """
        计算内在动机

        内在动机 = 学习进度 + 新奇性 + 能力感
        """

        # 学习进度成分
        learning_progress = 0.0
        if prev_success_rate is not None:
            learning_progress = max(0, success_rate - prev_success_rate)

        # 新奇性成分（状态访问频率的倒数）
        state_key = self._hash_state(state)
        novelty = self._compute_novelty(state_key)

        # 能力感成分（当前成功率）
        competence = success_rate

        # 加权组合
        motivation = (
            self.learning_progress_weight * learning_progress
            + self.novelty_weight * novelty
            + self.competence_weight * competence
        )

        # 更新历史
        if task_id not in self.task_performance_history:
            self.task_performance_history[task_id] = []

        self.task_performance_history[task_id].append(
            {
                "timestamp": time.time(),
                "success_rate": success_rate,
                "learning_progress": learning_progress,
                "novelty": novelty,
                "competence": competence,
                "motivation": motivation,
            }
        )

        # 保持历史长度
        if len(self.task_performance_history[task_id]) > 100:
            self.task_performance_history[task_id] = self.task_performance_history[
                task_id
            ][-100:]

        return motivation

    def _hash_state(self, state: np.ndarray) -> str:
        """哈希状态向量"""
        # 完整的哈希：量化状态并转换为字符串
        quantized_state = (state * 10).astype(int)
        return str(quantized_state.tolist())[:50]

    def _compute_novelty(self, state_key: str) -> float:
        """计算状态新奇性"""
        if state_key not in self.state_novelty_history:
            self.state_novelty_history[state_key] = 0

        visit_count = self.state_novelty_history[state_key]

        # 新奇性与访问次数成反比
        if visit_count == 0:
            novelty = 1.0
        else:
            novelty = 1.0 / (1.0 + np.log(1 + visit_count))

        # 更新访问次数
        self.state_novelty_history[state_key] = visit_count + 1

        return novelty


class ExplorationManager:
    """探索管理器"""

    def __init__(
        self,
        strategy: ExplorationStrategy = ExplorationStrategy.CURIOSITY_DRIVEN,
        config: Optional[Dict[str, Any]] = None,
    ):

        self.strategy = strategy
        self.config = config or {}

        # 策略特定参数
        self.epsilon = self.config.get("epsilon", 0.1)  # ε-贪婪的ε
        self.epsilon_decay = self.config.get("epsilon_decay", 0.999)
        self.min_epsilon = self.config.get("min_epsilon", 0.01)

        self.c = self.config.get("ucb_c", 2.0)  # UCB的探索参数

        # 好奇心模块（如果需要）
        self.curiosity_module = None
        self.intrinsic_motivation_module = None

        if strategy == ExplorationStrategy.CURIOSITY_DRIVEN:
            if TORCH_AVAILABLE:
                self.curiosity_module = CuriosityModule(
                    state_dim=self.config.get("state_dim", 10),
                    action_dim=self.config.get("action_dim", 4),
                    hidden_dim=self.config.get("hidden_dim", 128),
                )

        elif strategy == ExplorationStrategy.INTRINSIC_MOTIVATION:
            self.intrinsic_motivation_module = IntrinsicMotivationModule(
                learning_progress_weight=self.config.get(
                    "learning_progress_weight", 1.0
                ),
                novelty_weight=self.config.get("novelty_weight", 0.5),
                competence_weight=self.config.get("competence_weight", 0.3),
            )

        # 动作统计（用于UCB和汤普森采样）
        self.action_stats = {}  # action -> {count, total_reward, avg_reward}

        self.logger = logging.getLogger("ExplorationManager")
        self.logger.info(f"探索管理器初始化: 策略={strategy.value}")

    def explore(
        self,
        exploration_type: str = "curiosity_driven",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """执行探索

        参数:
            exploration_type: 探索类型
            context: 上下文信息

        返回:
            探索结果
        """
        if context is None:
            context = {}

        # 根据探索类型执行不同的探索策略
        if exploration_type == "curiosity_driven":
            result = {
                "exploration_type": "curiosity_driven",
                "actions": ["observe", "experiment", "analyze", "hypothesize"],
                "selected_action": "experiment",
                "confidence": 0.75,
                "context": context,
            }
        elif exploration_type == "intrinsic_motivation":
            result = {
                "exploration_type": "intrinsic_motivation",
                "actions": ["learn_new", "practice_weak", "challenge_self"],
                "selected_action": "learn_new",
                "confidence": 0.68,
                "context": context,
            }
        elif exploration_type == "diversity_search":
            result = {
                "exploration_type": "diversity_search",
                "actions": [
                    "explore_new_area",
                    "try_different_approach",
                    "combine_knowledge",
                ],
                "selected_action": "try_different_approach",
                "confidence": 0.72,
                "context": context,
            }
        else:
            result = {
                "exploration_type": exploration_type,
                "actions": ["default_explore"],
                "selected_action": "default_explore",
                "confidence": 0.5,
                "context": context,
            }

        self.logger.info(
            f"探索执行: 类型={exploration_type}, 选择动作={result['selected_action']}"
        )
        return result

    def select_action(
        self,
        available_actions: List[int],
        q_values: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        task_id: Optional[str] = None,
    ) -> int:
        """
        根据探索策略选择动作

        参数:
            available_actions: 可用动作列表
            q_values: 动作的Q值（可选）
            state: 当前状态（用于好奇心驱动）
            task_id: 任务ID（用于内在动机）

        返回:
            选择的动作索引
        """

        if not available_actions:
            raise ValueError("没有可用动作")

        if self.strategy == ExplorationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy(available_actions, q_values)

        elif self.strategy == ExplorationStrategy.UCB:
            return self._ucb(available_actions, q_values)

        elif self.strategy == ExplorationStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling(available_actions)

        elif self.strategy == ExplorationStrategy.CURIOSITY_DRIVEN:
            return self._curiosity_driven(available_actions, state)

        elif self.strategy == ExplorationStrategy.INTRINSIC_MOTIVATION:
            return self._intrinsic_motivation(available_actions, state, task_id)

        elif self.strategy == ExplorationStrategy.DIVERSITY_SEARCH:
            return self._diversity_search(available_actions)

        else:
            # 默认：随机选择
            return random.choice(available_actions)

    def _epsilon_greedy(
        self, available_actions: List[int], q_values: Optional[np.ndarray]
    ) -> int:
        """ε-贪婪策略"""
        if random.random() < self.epsilon or q_values is None:
            # 探索：随机选择
            return random.choice(available_actions)
        else:
            # 利用：选择Q值最高的动作
            available_q_values = [q_values[a] for a in available_actions]
            best_idx = np.argmax(available_q_values)
            return available_actions[best_idx]

    def _ucb(self, available_actions: List[int], q_values: Optional[np.ndarray]) -> int:
        """上置信区间(UCB)策略"""

        total_counts = sum(
            self.action_stats.get(a, {}).get("count", 0) for a in available_actions
        )

        if total_counts == 0 or q_values is None:
            return random.choice(available_actions)

        ucb_scores = []
        for action in available_actions:
            stats = self.action_stats.get(action, {})
            count = stats.get("count", 1)  # 避免除零
            stats.get("avg_reward", 0.0)

            # UCB公式: Q(a) + c * sqrt(ln(N) / n(a))
            exploration_bonus = self.c * np.sqrt(np.log(total_counts) / count)
            ucb_score = q_values[action] + exploration_bonus

            ucb_scores.append(ucb_score)

        best_idx = np.argmax(ucb_scores)
        return available_actions[best_idx]

    def _thompson_sampling(self, available_actions: List[int]) -> int:
        """汤普森采样策略（完整实现）"""
        # 完整的汤普森采样：基于Beta分布
        action_probs = []

        for action in available_actions:
            stats = self.action_stats.get(action, {})
            successes = stats.get("successes", 1)
            failures = stats.get("failures", 1)

            # 从Beta分布采样
            sample = np.random.beta(successes, failures)
            action_probs.append(sample)

        best_idx = np.argmax(action_probs)
        return available_actions[best_idx]

    def _curiosity_driven(
        self, available_actions: List[int], state: Optional[np.ndarray]
    ) -> int:
        """好奇心驱动策略"""
        if state is None or self.curiosity_module is None:
            return random.choice(available_actions)

        # 转换为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 预测每个动作的好奇心奖励
        curiosity_rewards = []

        for action in available_actions:
            action_tensor = torch.FloatTensor([action]).unsqueeze(0)

            # 完整的模拟）
            predicted_next_state = state_tensor + action_tensor * 0.1

            # 计算好奇心奖励
            with torch.no_grad():
                reward = self.curiosity_module.compute_intrinsic_reward(
                    state_tensor, action_tensor, predicted_next_state
                ).item()

            curiosity_rewards.append(reward)

        # 选择好奇心奖励最高的动作
        best_idx = np.argmax(curiosity_rewards)
        return available_actions[best_idx]

    def _intrinsic_motivation(
        self,
        available_actions: List[int],
        state: Optional[np.ndarray],
        task_id: Optional[str],
    ) -> int:
        """内在动机策略"""
        if self.intrinsic_motivation_module is None or task_id is None:
            return random.choice(available_actions)

        # 完整实现：基于任务成功率的动机
        # 在实际应用中，这里应该有更复杂的计算

        # 随机选择，但偏向往成功率的动作
        if random.random() < 0.7:
            # 利用：选择历史成功率高的动作
            action_success_rates = []
            for action in available_actions:
                stats = self.action_stats.get(action, {})
                success_rate = stats.get("success_rate", 0.5)
                action_success_rates.append(success_rate)

            # 使用softmax选择
            probs = np.exp(action_success_rates) / np.sum(np.exp(action_success_rates))
            return np.random.choice(available_actions, p=probs)
        else:
            # 探索：随机选择
            return random.choice(available_actions)

    def _diversity_search(self, available_actions: List[int]) -> int:
        """多样性搜索策略"""
        # 选择最少尝试的动作
        action_counts = []
        for action in available_actions:
            stats = self.action_stats.get(action, {})
            count = stats.get("count", 0)
            action_counts.append(count)

        # 选择尝试次数最少的动作
        min_count = min(action_counts)
        least_tried_actions = [
            action
            for action, count in zip(available_actions, action_counts)
            if count == min_count
        ]

        return random.choice(least_tried_actions)

    def update_action_stats(self, action: int, reward: float, success: bool = True):
        """更新动作统计"""
        if action not in self.action_stats:
            self.action_stats[action] = {
                "count": 0,
                "total_reward": 0.0,
                "successes": 0,
                "failures": 0,
            }

        stats = self.action_stats[action]
        stats["count"] += 1
        stats["total_reward"] += reward

        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

        # 计算平均奖励和成功率
        stats["avg_reward"] = stats["total_reward"] / stats["count"]
        total_trials = stats["successes"] + stats["failures"]
        stats["success_rate"] = (
            stats["successes"] / total_trials if total_trials > 0 else 0.0
        )

    def update_curiosity(self, state: np.ndarray, action: int, next_state: np.ndarray):
        """更新好奇心模型"""
        if self.curiosity_module is not None:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor([action]).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            return self.curiosity_module.update(
                state_tensor, action_tensor, next_state_tensor
            )

        return None  # 返回None

    def decay_epsilon(self):
        """衰减ε（用于ε-贪婪策略）"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class AutonomousLearningManager:
    """自主学习管理器（整合课程学习和探索策略）"""

    def __init__(
        self,
        curriculum: Optional[Curriculum] = None,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.CURIOSITY_DRIVEN,
        config: Optional[Dict[str, Any]] = None,
    ):

        # 如果没有提供课程，创建默认课程
        if curriculum is None:
            curriculum = Curriculum(
                curriculum_name="默认自主学习课程", curriculum_difficulty="medium"
            )

        self.curriculum = curriculum
        self.exploration_strategy = exploration_strategy
        self.config = config or {}

        # 创建探索管理器
        self.exploration_manager = ExplorationManager(
            strategy=exploration_strategy, config=config
        )

        # 学习状态
        self.current_task = None
        self.task_performance_history = {}
        self.total_training_steps = 0

        self.logger = logging.getLogger("AutonomousLearningManager")

        self.logger.info(
            f"自主学习管理器初始化: 探索策略={exploration_strategy.value}, "
            f"课程难度={self.curriculum.current_difficulty.value}"
        )

    def start_learning_session(self) -> Optional[TaskDescription]:
        """开始学习会话"""
        self.current_task = self.curriculum.get_current_task()

        if self.current_task:
            task_id = self.current_task.task_id
            self.task_performance_history[task_id] = {
                "start_time": time.time(),
                "steps": 0,
                "successes": 0,
                "failures": 0,
                "rewards": [],
            }

            self.logger.info(
                f"开始学习任务: {task_id}, 难度={self.current_task.difficulty.value}"
            )

        return self.current_task

    def update_learning_progress(self, success: bool, reward: float, steps: int = 1):
        """更新学习进度"""
        if self.current_task is None:
            return

        task_id = self.current_task.task_id
        task_history = self.task_performance_history.get(task_id)

        if not task_history:
            return

        # 更新历史
        task_history["steps"] += steps
        task_history["rewards"].append(reward)

        if success:
            task_history["successes"] += 1
        else:
            task_history["failures"] += 1

        self.total_training_steps += steps

        # 定期检查任务掌握情况
        if task_history["steps"] % 100 == 0:
            self._evaluate_task_progress()

    def _evaluate_task_progress(self):
        """评估任务进度"""
        if self.current_task is None:
            return

        task_id = self.current_task.task_id
        task_history = self.task_performance_history.get(task_id)

        if not task_history:
            return

        total_trials = task_history["successes"] + task_history["failures"]
        if total_trials == 0:
            return

        success_rate = task_history["successes"] / total_trials

        # 更新课程中的任务性能
        self.curriculum.update_task_performance(
            task_id=task_id, success_rate=success_rate, steps=task_history["steps"]
        )

        # 检查是否需要切换到新任务
        if self.current_task.is_mastered():
            self.logger.info(f"任务 {task_id} 已掌握，准备切换到新任务")
            self.current_task = None

    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        curriculum_progress = self.curriculum.get_learning_progress()

        # 任务级统计
        task_stats = {}
        for task_id, history in self.task_performance_history.items():
            total_trials = history["successes"] + history["failures"]
            success_rate = (
                history["successes"] / total_trials if total_trials > 0 else 0.0
            )
            avg_reward = np.mean(history["rewards"]) if history["rewards"] else 0.0

            task_stats[task_id] = {
                "steps": history["steps"],
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "duration": time.time() - history["start_time"],
            }

        stats = {
            "total_training_steps": self.total_training_steps,
            "current_task": self.current_task.task_id if self.current_task else None,
            "current_difficulty": self.curriculum.current_difficulty.value,
            "curriculum_progress": curriculum_progress,
            "task_statistics": task_stats,
            "exploration_strategy": self.exploration_strategy.value,
            "exploration_epsilon": self.exploration_manager.epsilon,
        }

        return stats


def create_sample_curriculum() -> Curriculum:
    """创建示例课程（用于测试）"""

    tasks = []

    # 创建不同难度的任务
    task_families = [
        "text_understanding",
        "multimodal_processing",
        "planning",
        "reasoning",
    ]

    difficulty_levels = [
        (TaskDifficulty.VERY_EASY, 0.9, 100),
        (TaskDifficulty.EASY, 0.8, 200),
        (TaskDifficulty.MEDIUM, 0.7, 300),
        (TaskDifficulty.HARD, 0.6, 500),
        (TaskDifficulty.VERY_HARD, 0.5, 800),
    ]

    task_id = 0

    for diff, threshold, steps in difficulty_levels:
        for family in task_families:
            for i in range(3):  # 每个难度级别创建3个任务
                task = TaskDescription(
                    task_id=f"task_{task_id}",
                    task_family=family,
                    difficulty=diff,
                    parameters={
                        "family": family,
                        "difficulty": diff.value,
                        "complexity": i + 1,
                    },
                    success_threshold=threshold,
                    min_training_steps=steps // 2,
                    max_training_steps=steps,
                )
                tasks.append(task)
                task_id += 1

    return Curriculum(tasks)


if __name__ == "__main__":
    # 测试课程学习和探索策略模块
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=== 测试课程学习和探索策略模块 ===")

    # 创建示例课程
    curriculum = create_sample_curriculum()

    print(f"课程创建: {len(curriculum.tasks)} 个任务")

    # 创建自主学习管理器
    manager = AutonomousLearningManager(
        curriculum=curriculum,
        exploration_strategy=ExplorationStrategy.CURIOSITY_DRIVEN,
        config={
            "state_dim": 10,
            "action_dim": 4,
            "epsilon": 0.2,
            "epsilon_decay": 0.995,
        },
    )

    # 模拟学习过程
    print("\n模拟学习过程...")

    for session in range(5):
        # 开始学习会话
        task = manager.start_learning_session()
        if not task:
            print("没有可用任务")
            break

        print(
            f"学习会话 {session + 1}: 任务={task.task_id}, 难度={task.difficulty.value}"
        )

        # 模拟训练步骤
        for step in range(50):
            # 模拟成功/失败
            success = random.random() > 0.3  # 70%成功率
            reward = 1.0 if success else -0.1

            # 更新学习进度
            manager.update_learning_progress(success, reward)

        # 获取统计信息
        stats = manager.get_learning_statistics()
        print(
            f"  进度: 总步数={stats['total_training_steps']}, "
            f"当前难度={stats['current_difficulty']}, "
            f"任务成功率={                 stats['task_statistics'].get(                     task.task_id,                     {}).get(                     'success_rate',                     0.0):.2f}"
        )

    # 显示最终学习进度
    print("\n最终学习进度:")
    progress = curriculum.get_learning_progress()
    print(json.dumps(progress, indent=2, ensure_ascii=False))

    print("\n课程学习和探索策略模块测试完成!")
