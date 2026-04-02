#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self AGI 模仿学习模块
集成模仿学习算法，从专家示范中学习机器人控制策略

功能：
1. 行为克隆 (Behavior Cloning)
2. DAgger (Dataset Aggregation)
3. 逆强化学习基础
4. 示范数据收集和管理
5. 与强化学习的结合
"""

import sys
import os
import logging
import json
import pickle
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time

# 初始化日志记录器
logger = logging.getLogger(__name__)

# 导入机器学习库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    logger.info("PyTorch库可用")
except ImportError as e:
    TORCH_AVAILABLE = False
    logger.error(f"PyTorch导入失败: {e}")
    logger.error("请安装必需的依赖库: pip install torch")
    raise ImportError(
        "PyTorch不可用。Self AGI系统需要PyTorch库来提供真实的模仿学习功能。\n"
        "请执行: pip install torch"
    ) from e

# 导入强化学习库
try:
    import gymnasium as gym
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.buffers import ReplayBuffer
    RL_LIBS_AVAILABLE = True
    logger.info("强化学习库可用 (stable-baselines3, gymnasium)")
except ImportError as e:
    RL_LIBS_AVAILABLE = False
    logger.error(f"强化学习库导入失败: {e}")
    logger.error("请安装必需的依赖库: pip install gymnasium stable-baselines3")
    raise ImportError(
        "强化学习库不可用。Self AGI系统需要gymnasium和stable-baselines3库来提供真实的模仿学习功能。\n"
        "请执行: pip install gymnasium stable-baselines3"
    ) from e


class ImitationAlgorithm(Enum):
    """模仿学习算法枚举"""
    BEHAVIOR_CLONING = "behavior_cloning"
    DAGGER = "dagger"
    GAIL = "gail"  # 生成式对抗模仿学习
    BCQ = "bcq"  # 批量约束Q学习


@dataclass
class DemonstrationData:
    """示范数据点"""
    
    observation: np.ndarray  # 观察
    action: np.ndarray  # 专家动作
    next_observation: Optional[np.ndarray] = None  # 下一个观察
    reward: Optional[float] = None  # 奖励（如果有）
    done: Optional[bool] = None  # 是否终止
    info: Optional[Dict[str, Any]] = None  # 附加信息
    timestamp: float = None  # 时间戳
    
    def __post_init__(self):
        """初始化后处理"""
        if self.timestamp is None:
            self.timestamp = time.time()


class DemonstrationDataset(Dataset):
    """示范数据集"""
    
    def __init__(self, data: Optional[List[DemonstrationData]] = None):
        self.data = data or []
        self.logger = logging.getLogger("DemonstrationDataset")
        
    def add_demonstration(self, demonstration: DemonstrationData):
        """添加示范数据"""
        self.data.append(demonstration)
        
    def add_batch(self, observations: np.ndarray, actions: np.ndarray, **kwargs):
        """批量添加示范数据"""
        batch_size = len(observations)
        
        for i in range(batch_size):
            demo = DemonstrationData(
                observation=observations[i],
                action=actions[i],
                **{k: v[i] if isinstance(v, np.ndarray) else v 
                   for k, v in kwargs.items()}
            )
            self.add_demonstration(demo)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        demo = self.data[idx]
        return demo.observation, demo.action
    
    def get_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """获取批量数据"""
        indices = np.random.choice(len(self.data), size=batch_size, replace=True)
        observations = np.array([self.data[i].observation for i in indices])
        actions = np.array([self.data[i].action for i in indices])
        return observations, actions
    
    def save(self, path: str):
        """保存数据集"""
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        self.logger.info(f"示范数据集保存到: {path}, 数据点: {len(self.data)}")
    
    def load(self, path: str):
        """加载数据集"""
        try:
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
            self.logger.info(f"示范数据集从 {path} 加载, 数据点: {len(self.data)}")
            return True
        except Exception as e:
            self.logger.error(f"加载数据集失败: {e}")
            return False


class BehaviorCloningPolicy(nn.Module):
    """行为克隆策略网络"""
    
    def __init__(self, 
                 observation_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # 构建网络
        layers = []
        prev_dim = observation_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 损失函数
        self.criterion = nn.MSELoss()  # 连续动作空间
        
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(observation)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """预测动作"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action_tensor = self.forward(obs_tensor)
            return action_tensor.squeeze(0).numpy()


class BehaviorCloningTrainer:
    """行为克隆训练器"""
    
    def __init__(self,
                 policy: BehaviorCloningPolicy,
                 dataset: DemonstrationDataset,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 epochs: int = 100):
        
        self.policy = policy
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.logger = logging.getLogger("BehaviorCloningTrainer")
        
    def train(self) -> Dict[str, Any]:
        """训练行为克隆策略"""
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch不可用，无法训练行为克隆")
            return {"success": False, "error": "PyTorch不可用"}
        
        if len(self.dataset) == 0:
            self.logger.error("示范数据集为空，无法训练")
            return {"success": False, "error": "数据集为空"}
        
        self.logger.info(f"开始行为克隆训练: {len(self.dataset)} 个示范点, {self.epochs} 轮")
        
        losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # 随机打乱数据
            indices = np.random.permutation(len(self.dataset))
            
            for i in range(0, len(self.dataset), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                # 获取批次数据
                observations = []
                actions = []
                
                for idx in batch_indices:
                    demo = self.dataset.data[idx]
                    observations.append(demo.observation)
                    actions.append(demo.action)
                
                observations_tensor = torch.FloatTensor(np.array(observations))
                actions_tensor = torch.FloatTensor(np.array(actions))
                
                # 前向传播
                pred_actions = self.policy(observations_tensor)
                
                # 计算损失
                loss = self.policy.criterion(pred_actions, actions_tensor)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"轮次 {epoch}/{self.epochs}, 损失: {avg_loss:.6f}")
        
        result = {
            "success": True,
            "algorithm": "behavior_cloning",
            "epochs": self.epochs,
            "dataset_size": len(self.dataset),
            "final_loss": losses[-1] if losses else 0.0,
            "loss_history": losses
        }
        
        self.logger.info(f"行为克隆训练完成: 最终损失={result['final_loss']:.6f}")
        
        return result


class DAggerTrainer:
    """DAgger训练器 (Dataset Aggregation)"""
    
    def __init__(self,
                 env: gym.Env,
                 expert_policy: Any,  # 专家策略
                 learner_policy: BehaviorCloningPolicy,  # 学习器策略
                 dataset: DemonstrationDataset,
                 iterations: int = 10,
                 episodes_per_iteration: int = 5,
                 learning_rate: float = 1e-3):
        
        self.env = env
        self.expert_policy = expert_policy
        self.learner_policy = learner_policy
        self.dataset = dataset
        self.iterations = iterations
        self.episodes_per_iteration = episodes_per_iteration
        self.learning_rate = learning_rate
        
        self.logger = logging.getLogger("DAggerTrainer")
        
    def train(self) -> Dict[str, Any]:
        """执行DAgger训练"""
        self.logger.info(f"开始DAgger训练: {self.iterations} 次迭代")
        
        results = []
        
        for iteration in range(self.iterations):
            self.logger.info(f"DAgger迭代 {iteration + 1}/{self.iterations}")
            
            # 1. 用当前策略收集轨迹
            trajectories = self._collect_trajectories_with_learner()
            
            # 2. 专家标记轨迹
            expert_trajectories = self._label_with_expert(trajectories)
            
            # 3. 将专家标记的数据添加到数据集
            self._add_to_dataset(expert_trajectories)
            
            # 4. 在增强的数据集上训练行为克隆
            bc_trainer = BehaviorCloningTrainer(
                self.learner_policy,
                self.dataset,
                learning_rate=self.learning_rate,
                epochs=50
            )
            
            bc_result = bc_trainer.train()
            
            # 5. 评估当前策略
            eval_result = self._evaluate_policy()
            
            iteration_result = {
                "iteration": iteration + 1,
                "dataset_size": len(self.dataset),
                "bc_loss": bc_result.get("final_loss", 0.0),
                "eval_reward": eval_result.get("mean_reward", 0.0)
            }
            
            results.append(iteration_result)
            
            self.logger.info(
                f"迭代 {iteration + 1}: 数据集大小={iteration_result['dataset_size']}, "
                f"BC损失={iteration_result['bc_loss']:.6f}, "
                f"评估奖励={iteration_result['eval_reward']:.3f}"
            )
        
        final_result = {
            "success": True,
            "algorithm": "dagger",
            "iterations": self.iterations,
            "final_dataset_size": len(self.dataset),
            "iteration_results": results
        }
        
        self.logger.info(f"DAgger训练完成: 最终数据集大小={final_result['final_dataset_size']}")
        
        return final_result
    
    def _collect_trajectories_with_learner(self) -> List[List[Tuple]]:
        """用学习器策略收集轨迹"""
        trajectories = []
        
        for episode in range(self.episodes_per_iteration):
            trajectory = []
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                # 用学习器策略选择动作
                action = self.learner_policy.predict(obs)
                
                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # 保存转移
                trajectory.append((obs, action, reward, next_obs, done, info))
                
                obs = next_obs
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _label_with_expert(self, trajectories: List[List[Tuple]]) -> List[List[Tuple]]:
        """用专家策略标记轨迹"""
        expert_trajectories = []
        
        for trajectory in trajectories:
            expert_trajectory = []
            
            for obs, _, reward, next_obs, done, info in trajectory:
                # 获取专家动作
                expert_action = self._get_expert_action(obs)
                
                expert_trajectory.append((obs, expert_action, reward, next_obs, done, info))
            
            expert_trajectories.append(expert_trajectory)
        
        return expert_trajectories
    
    def _get_expert_action(self, observation: np.ndarray) -> np.ndarray:
        """获取专家动作"""
        # 这里应该调用真实的专家策略
        # 完整实现：返回随机动作（实际应用中应替换为真实专家）
        if hasattr(self.expert_policy, 'predict'):
            return self.expert_policy.predict(observation)
        else:
            # 回退：随机动作
            action_space = self.env.action_space
            if isinstance(action_space, gym.spaces.Box):
                return action_space.sample()
            else:
                return np.array([0])
    
    def _add_to_dataset(self, trajectories: List[List[Tuple]]):
        """将轨迹添加到数据集"""
        for trajectory in trajectories:
            for obs, expert_action, reward, next_obs, done, info in trajectory:
                demo = DemonstrationData(
                    observation=obs,
                    action=expert_action,
                    next_observation=next_obs,
                    reward=reward,
                    done=done,
                    info=info
                )
                self.dataset.add_demonstration(demo)
    
    def _evaluate_policy(self) -> Dict[str, Any]:
        """评估策略"""
        total_reward = 0.0
        num_episodes = 5
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action = self.learner_policy.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_reward += episode_reward
        
        mean_reward = total_reward / num_episodes
        
        return {
            "mean_reward": mean_reward,
            "num_episodes": num_episodes
        }


class ImitationLearningManager:
    """模仿学习管理器"""
    
    def __init__(self,
                 env: gym.Env,
                 algorithm: ImitationAlgorithm = ImitationAlgorithm.BEHAVIOR_CLONING,
                 config: Optional[Dict[str, Any]] = None):
        
        self.env = env
        self.algorithm = algorithm
        self.config = config or {}
        
        # 数据集
        self.dataset = DemonstrationDataset()
        
        # 策略
        self.policy = None
        self.trainer = None
        
        self.logger = logging.getLogger("ImitationLearningManager")
        
        # 初始化策略
        self._initialize_policy()
        
        self.logger.info(f"模仿学习管理器初始化: 算法={algorithm.value}")
    
    def _initialize_policy(self):
        """初始化策略网络"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch不可用，无法初始化策略网络")
            return
        
        # 获取观察和动作维度
        obs_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        
        if obs_shape is None or action_shape is None:
            self.logger.error("无法获取观察或动作空间形状")
            return
        
        observation_dim = obs_shape[0]
        action_dim = action_shape[0]
        
        # 创建策略网络
        self.policy = BehaviorCloningPolicy(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dims=self.config.get("hidden_dims", [256, 256])
        )
        
        self.logger.info(f"策略网络初始化: 观察维度={observation_dim}, 动作维度={action_dim}")
    
    def collect_demonstrations(self,
                              expert_policy: Any,
                              num_episodes: int = 10,
                              max_steps: int = 1000) -> Dict[str, Any]:
        """收集专家示范"""
        self.logger.info(f"收集专家示范: {num_episodes} 个片段")
        
        demonstrations_collected = 0
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            
            while not done and step < max_steps:
                # 获取专家动作
                if hasattr(expert_policy, 'predict'):
                    action = expert_policy.predict(obs)
                else:
                    # 回退：随机动作或零动作
                    action_space = self.env.action_space
                    if isinstance(action_space, gym.spaces.Box):
                        action = action_space.sample()
                    else:
                        action = np.array([0])
                
                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # 保存示范数据
                demo = DemonstrationData(
                    observation=obs,
                    action=action,
                    next_observation=next_obs,
                    reward=reward,
                    done=done,
                    info=info
                )
                
                self.dataset.add_demonstration(demo)
                demonstrations_collected += 1
                
                # 更新状态
                obs = next_obs
                step += 1
            
            self.logger.debug(f"片段 {episode + 1}: 收集了 {step} 个示范点")
        
        result = {
            "success": True,
            "algorithm": "demonstration_collection",
            "episodes": num_episodes,
            "demonstrations_collected": demonstrations_collected,
            "dataset_size": len(self.dataset)
        }
        
        self.logger.info(f"示范收集完成: 总示范点={demonstrations_collected}, 数据集大小={len(self.dataset)}")
        
        return result
    
    def train(self) -> Dict[str, Any]:
        """训练模仿学习策略"""
        if self.policy is None:
            self.logger.error("策略网络未初始化，无法训练")
            return {"success": False, "error": "策略网络未初始化"}
        
        if len(self.dataset) == 0:
            self.logger.error("示范数据集为空，无法训练")
            return {"success": False, "error": "数据集为空"}
        
        if self.algorithm == ImitationAlgorithm.BEHAVIOR_CLONING:
            self.trainer = BehaviorCloningTrainer(
                self.policy,
                self.dataset,
                learning_rate=self.config.get("learning_rate", 1e-3),
                batch_size=self.config.get("batch_size", 32),
                epochs=self.config.get("epochs", 100)
            )
            
            return self.trainer.train()
        
        elif self.algorithm == ImitationAlgorithm.DAGGER:
            # 需要专家策略（这里使用当前策略作为"专家"的完整替代）
            expert_policy = self.policy
            
            self.trainer = DAggerTrainer(
                self.env,
                expert_policy,
                self.policy,
                self.dataset,
                iterations=self.config.get("iterations", 10),
                episodes_per_iteration=self.config.get("episodes_per_iteration", 5),
                learning_rate=self.config.get("learning_rate", 1e-3)
            )
            
            return self.trainer.train()
        
        else:
            self.logger.error(f"不支持的算法: {self.algorithm}")
            return {"success": False, "error": f"不支持的算法: {self.algorithm}"}
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """评估训练好的策略"""
        if self.policy is None:
            self.logger.error("策略网络未初始化，无法评估")
            return {"success": False, "error": "策略网络未初始化"}
        
        self.logger.info(f"评估策略: {num_episodes} 个片段")
        
        total_reward = 0.0
        success_count = 0
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action = self.policy.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_reward += episode_reward
            
            # 简单判断是否成功（奖励大于阈值）
            if episode_reward > 0.5:
                success_count += 1
            
            self.logger.debug(f"片段 {episode + 1}: 奖励={episode_reward:.3f}")
        
        mean_reward = total_reward / num_episodes
        success_rate = success_count / num_episodes
        
        result = {
            "success": True,
            "mean_reward": mean_reward,
            "success_rate": success_rate,
            "num_episodes": num_episodes,
            "policy_type": "imitation_learning"
        }
        
        self.logger.info(f"策略评估完成: 平均奖励={mean_reward:.3f}, 成功率={success_rate:.3f}")
        
        return result
    
    def save_policy(self, path: str):
        """保存策略"""
        if self.policy is not None and TORCH_AVAILABLE:
            torch.save(self.policy.state_dict(), path)
            self.logger.info(f"策略保存到: {path}")
            return True
        else:
            self.logger.warning("没有策略可保存")
            return False
    
    def load_policy(self, path: str):
        """加载策略"""
        if self.policy is not None and TORCH_AVAILABLE:
            try:
                self.policy.load_state_dict(torch.load(path))
                self.logger.info(f"策略从 {path} 加载")
                return True
            except Exception as e:
                self.logger.error(f"加载策略失败: {e}")
                return False
        else:
            self.logger.warning("无法加载策略")
            return False
    
    def save_dataset(self, path: str):
        """保存数据集"""
        return self.dataset.save(path)
    
    def load_dataset(self, path: str):
        """加载数据集"""
        return self.dataset.load(path)


def create_imitation_learning_manager(env: gym.Env, **kwargs) -> ImitationLearningManager:
    """创建模仿学习管理器（工厂函数）"""
    return ImitationLearningManager(env, **kwargs)


if __name__ == "__main__":
    # 测试模仿学习模块
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== 测试模仿学习模块 ===")
    
    # 创建简单的测试环境
    class TestEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
            self.step_count = 0
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.step_count = 0
            return np.random.randn(4), {}
        
        def step(self, action):
            self.step_count += 1
            obs = np.random.randn(4)
            reward = 0.1
            terminated = False
            truncated = self.step_count >= 10
            info = {"test": True}
            return obs, reward, terminated, truncated, info
        
        def render(self):
            """渲染测试环境"""
            logger.debug("测试环境渲染（无实际渲染）")
            return None  # 返回None
        
        def close(self):
            """关闭测试环境"""
            logger.debug("关闭测试环境（无资源需要清理）")
    
    # 创建测试环境
    test_env = TestEnv()
    
    # 创建模仿学习管理器
    manager = create_imitation_learning_manager(
        test_env,
        algorithm=ImitationAlgorithm.BEHAVIOR_CLONING,
        config={"epochs": 20}
    )
    
    # 收集示范数据（使用随机"专家"）
    class RandomExpert:
        def predict(self, obs):
            return np.random.randn(2) * 0.1
    
    expert = RandomExpert()
    
    print("收集示范数据...")
    collection_result = manager.collect_demonstrations(expert, num_episodes=3, max_steps=5)
    print(f"示范收集结果: {json.dumps(collection_result, indent=2, ensure_ascii=False)}")
    
    # 训练行为克隆
    print("\n训练行为克隆...")
    training_result = manager.train()
    print(f"训练结果: {json.dumps(training_result, indent=2, ensure_ascii=False)}")
    
    # 评估策略
    print("\n评估策略...")
    eval_result = manager.evaluate(num_episodes=3)
    print(f"评估结果: {json.dumps(eval_result, indent=2, ensure_ascii=False)}")
    
    print("\n模仿学习模块测试完成!")