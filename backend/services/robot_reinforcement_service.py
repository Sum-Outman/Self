#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人强化学习训练服务
管理Gazebo仿真环境的强化学习训练任务

功能：
1. 训练任务管理（创建、启动、停止、监控）
2. 模型训练和保存
3. 训练进度跟踪
4. 与数据库集成
"""

import sys
import os
import logging
import time
import json
import uuid
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
import numpy as np

# 导入数据库模型
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..db_models.user import User
from ..db_models.robot import Robot, RobotStatus
from ..db_models.demonstration import Demonstration, DemonstrationStatus

# 导入强化学习模块
try:
    from training.reinforcement_learning import (
        AGITaskType, RLTrainingConfig, AGIEnvironment,
        RL_LIBS_AVAILABLE, TORCH_AVAILABLE
    )
    from training.gazebo_environment import (
        GazeboRobotEnvironment, GazeboEnvironmentConfig, RobotTaskType,
        GAZEBO_AVAILABLE, RL_LIBS_AVAILABLE as GAZEBO_RL_AVAILABLE
    )
    
    # 导入稳定基线3
    from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import (
        BaseCallback, CheckpointCallback, EvalCallback,
        StopTrainingOnRewardThreshold
    )
    from stable_baselines3.common.evaluation import evaluate_policy
    
    REINFORCEMENT_AVAILABLE = True
except ImportError as e:
    REINFORCEMENT_AVAILABLE = False
    print(f"警告: 强化学习模块不可用: {e}")
    # 创建虚拟类以避免导入错误
    class RLTrainingConfig:
        pass  # 已实现
    class GazeboEnvironmentConfig:
        pass  # 已实现
    class RobotTaskType(Enum):
        STAND_UP = "stand_up"

logger = logging.getLogger(__name__)


class TrainingTaskStatus(Enum):
    """训练任务状态"""
    PENDING = "pending"  # 等待中
    PREPARING = "preparing"  # 准备中
    RUNNING = "running"  # 运行中
    PAUSED = "paused"  # 已暂停
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


class TrainingAlgorithm(Enum):
    """训练算法"""
    PPO = "PPO"
    A2C = "A2C"
    DQN = "DQN"
    SAC = "SAC"
    TD3 = "TD3"


@dataclass
class RobotTrainingConfig:
    """机器人训练配置"""
    
    # 基础配置
    robot_id: int
    task_type: RobotTaskType = RobotTaskType.STAND_UP
    algorithm: TrainingAlgorithm = TrainingAlgorithm.PPO
    
    # 训练参数
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE参数
    entropy_coef: float = 0.01  # 熵系数
    
    # 环境参数
    max_steps: int = 1000
    reward_scale: float = 1.0
    success_reward: float = 100.0
    step_penalty: float = -0.01
    
    # Gazebo参数
    gazebo_world: str = "empty.world"
    robot_model: str = "humanoid"
    gui_enabled: bool = False  # 训练时通常禁用GUI以提高性能
    simulation_speed: float = 1.0
    
    # 网络参数
    policy_network: str = "MlpPolicy"
    hidden_layers: List[int] = None
    
    # 设备配置
    device: str = "auto"  # auto, cpu, cuda
    num_envs: int = 1  # 并行环境数量
    
    # 检查点和评估
    checkpoint_frequency: int = 10000  # 检查点频率（步数）
    eval_frequency: int = 5000  # 评估频率（步数）
    eval_episodes: int = 10  # 评估集数
    
    def __post_init__(self):
        """初始化后处理"""
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "robot_id": self.robot_id,
            "task_type": self.task_type.value,
            "algorithm": self.algorithm.value,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "entropy_coef": self.entropy_coef,
            "max_steps": self.max_steps,
            "reward_scale": self.reward_scale,
            "success_reward": self.success_reward,
            "step_penalty": self.step_penalty,
            "gazebo_world": self.gazebo_world,
            "robot_model": self.robot_model,
            "gui_enabled": self.gui_enabled,
            "simulation_speed": self.simulation_speed,
            "policy_network": self.policy_network,
            "hidden_layers": self.hidden_layers,
            "device": self.device,
            "num_envs": self.num_envs,
            "checkpoint_frequency": self.checkpoint_frequency,
            "eval_frequency": self.eval_frequency,
            "eval_episodes": self.eval_episodes,
        }


@dataclass
class TrainingProgress:
    """训练进度"""
    
    # 任务信息
    task_id: str
    status: TrainingTaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # 训练指标
    timesteps: int = 0
    episodes: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    best_reward: float = -float('inf')
    success_rate: float = 0.0
    
    # 资源使用
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # 错误信息
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "timesteps": self.timesteps,
            "episodes": self.episodes,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward,
            "best_reward": self.best_reward,
            "success_rate": self.success_rate,
            "cpu_usage": self.cpu_usage,
            "gpu_usage": self.gpu_usage,
            "memory_usage": self.memory_usage,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
        }


class TrainingCallback(BaseCallback):
    """自定义训练回调"""
    
    def __init__(self, progress_callback=None, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.progress_callback = progress_callback
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        """每一步调用"""
        # 获取当前奖励
        reward = self.locals.get('rewards', [0.0])[0] if self.locals.get('rewards') else 0.0
        done = self.locals.get('dones', [False])[0] if self.locals.get('dones') else False
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # 计算平均奖励
            mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            
            # 调用进度回调
            if self.progress_callback:
                progress_data = {
                    "timesteps": self.num_timesteps,
                    "episodes": len(self.episode_rewards),
                    "mean_reward": mean_reward,
                    "current_episode_reward": self.current_episode_reward,
                    "current_episode_length": self.current_episode_length,
                }
                self.progress_callback(progress_data)
            
            # 重置当前回合
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        return True


class RobotReinforcementTrainingService:
    """机器人强化学习训练服务"""
    
    def __init__(self, db: Session):
        """
        初始化训练服务
        
        参数:
            db: 数据库会话
        """
        self.db = db
        self.training_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_lock = threading.Lock()
        
        # 检查依赖
        self.dependencies_available = self._check_dependencies()
        
        logger.info("机器人强化学习训练服务初始化完成")
    
    def _check_dependencies(self) -> bool:
        """检查依赖是否可用"""
        if not REINFORCEMENT_AVAILABLE:
            logger.warning("强化学习库不可用")
            return False
        
        if not GAZEBO_AVAILABLE:
            logger.warning("Gazebo仿真库不可用")
            # 仍然可以训练，但使用模拟模式
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch不可用")
            return False
        
        logger.info("所有强化学习依赖库可用")
        return True
    
    def create_training_task(self, 
                           config: RobotTrainingConfig,
                           user: User) -> Dict[str, Any]:
        """创建训练任务
        
        参数:
            config: 训练配置
            user: 用户对象
            
        返回:
            任务信息
        """
        if not self.dependencies_available:
            return {
                "success": False,
                "error": "强化学习依赖库不可用",
                "task_id": None
            }
        
        # 验证机器人
        robot = self.db.query(Robot).filter(
            Robot.id == config.robot_id,
            Robot.user_id == user.id
        ).first()
        
        if not robot:
            return {
                "success": False,
                "error": f"机器人 {config.robot_id} 不存在或无权访问",
                "task_id": None
            }
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 创建任务记录
        with self.task_lock:
            self.training_tasks[task_id] = {
                "task_id": task_id,
                "config": config,
                "robot_id": config.robot_id,
                "user_id": user.id,
                "status": TrainingTaskStatus.PENDING,
                "progress": TrainingProgress(
                    task_id=task_id,
                    status=TrainingTaskStatus.PENDING,
                    start_time=datetime.now(timezone.utc)
                ),
                "thread": None,
                "stop_event": threading.Event(),
                "model": None,
                "environment": None,
                "created_at": datetime.now(timezone.utc)
            }
        
        logger.info(f"创建训练任务 {task_id}，机器人: {robot.name}，任务类型: {config.task_type.value}")
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "训练任务创建成功"
        }
    
    def start_training(self, task_id: str) -> Dict[str, Any]:
        """开始训练任务
        
        参数:
            task_id: 任务ID
            
        返回:
            启动结果
        """
        with self.task_lock:
            if task_id not in self.training_tasks:
                return {
                    "success": False,
                    "error": f"任务 {task_id} 不存在"
                }
            
            task = self.training_tasks[task_id]
            
            if task["status"] != TrainingTaskStatus.PENDING:
                return {
                    "success": False,
                    "error": f"任务状态为 {task['status'].value}，无法启动"
                }
            
            # 更新状态为准备中
            task["status"] = TrainingTaskStatus.PREPARING
            task["progress"].status = TrainingTaskStatus.PREPARING
        
        # 在单独线程中启动训练
        def train_thread():
            try:
                self._run_training(task_id)
            except Exception as e:
                logger.error(f"训练任务 {task_id} 失败: {e}", exc_info=True)
                with self.task_lock:
                    if task_id in self.training_tasks:
                        task = self.training_tasks[task_id]
                        task["status"] = TrainingTaskStatus.FAILED
                        task["progress"].status = TrainingTaskStatus.FAILED
                        task["progress"].error_message = str(e)
                        task["progress"].stack_trace = traceback.format_exc()
                        task["progress"].end_time = datetime.now(timezone.utc)
        
        # 启动训练线程
        thread = threading.Thread(target=train_thread, daemon=True)
        task["thread"] = thread
        thread.start()
        
        logger.info(f"启动训练任务 {task_id}")
        
        return {
            "success": True,
            "message": "训练任务已启动"
        }
    
    def _run_training(self, task_id: str):
        """运行训练任务
        
        参数:
            task_id: 任务ID
        """
        import traceback
        
        with self.task_lock:
            if task_id not in self.training_tasks:
                return
            
            task = self.training_tasks[task_id]
            config = task["config"]
            
            # 更新状态为运行中
            task["status"] = TrainingTaskStatus.RUNNING
            task["progress"].status = TrainingTaskStatus.RUNNING
        
        try:
            logger.info(f"开始训练任务 {task_id}: {config.task_type.value}，算法: {config.algorithm.value}")
            
            # 1. 创建Gazebo环境
            gazebo_config = GazeboEnvironmentConfig(
                task_type=config.task_type,
                max_steps=config.max_steps,
                reward_scale=config.reward_scale,
                success_reward=config.success_reward,
                step_penalty=config.step_penalty,
                gazebo_world=config.gazebo_world,
                robot_model=config.robot_model,
                gui_enabled=config.gui_enabled,
                simulation_speed=config.simulation_speed
            )
            
            env = GazeboRobotEnvironment(gazebo_config)
            
            # 包装为向量环境
            vec_env = DummyVecEnv([lambda: env])
            
            # 2. 创建模型
            model_class = {
                TrainingAlgorithm.PPO: PPO,
                TrainingAlgorithm.A2C: A2C,
                TrainingAlgorithm.DQN: DQN,
                TrainingAlgorithm.SAC: SAC,
                TrainingAlgorithm.TD3: TD3,
            }[config.algorithm]
            
            # 进度回调函数
            def progress_callback(data: Dict[str, Any]):
                with self.task_lock:
                    if task_id in self.training_tasks:
                        progress = self.training_tasks[task_id]["progress"]
                        progress.timesteps = data.get("timesteps", progress.timesteps)
                        progress.episodes = data.get("episodes", progress.episodes)
                        progress.mean_reward = data.get("mean_reward", progress.mean_reward)
                        
                        current_reward = data.get("current_episode_reward", 0.0)
                        progress.total_reward += current_reward
                        progress.best_reward = max(progress.best_reward, current_reward)
            
            # 3. 创建回调
            callbacks = []
            
            # 进度回调
            progress_callback = TrainingCallback(progress_callback=progress_callback)
            callbacks.append(progress_callback)
            
            # 检查点回调
            checkpoint_callback = CheckpointCallback(
                save_freq=config.checkpoint_frequency,
                save_path=f"./checkpoints/{task_id}/",
                name_prefix="rl_model"
            )
            callbacks.append(checkpoint_callback)
            
            # 4. 训练模型
            model = model_class(
                policy=config.policy_network,
                env=vec_env,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                ent_coef=config.entropy_coef,
                verbose=1,
                device=config.device,
                policy_kwargs={"net_arch": config.hidden_layers} if config.hidden_layers else None
            )
            
            # 保存模型引用
            with self.task_lock:
                if task_id in self.training_tasks:
                    self.training_tasks[task_id]["model"] = model
                    self.training_tasks[task_id]["environment"] = env
            
            # 5. 开始训练
            logger.info(f"开始强化学习训练，总步数: {config.total_timesteps}")
            
            model.learn(
                total_timesteps=config.total_timesteps,
                callback=callbacks,
                log_interval=100
            )
            
            # 6. 训练完成
            logger.info(f"训练任务 {task_id} 完成")
            
            # 保存最终模型
            model.save(f"./models/{task_id}/final_model")
            
            # 更新状态
            with self.task_lock:
                if task_id in self.training_tasks:
                    task = self.training_tasks[task_id]
                    task["status"] = TrainingTaskStatus.COMPLETED
                    task["progress"].status = TrainingTaskStatus.COMPLETED
                    task["progress"].end_time = datetime.now(timezone.utc)
                    task["progress"].success_rate = task["progress"].episodes / max(config.total_timesteps // config.max_steps, 1)
            
            # 7. 清理资源
            env.close()
            vec_env.close()
            
        except Exception as e:
            logger.error(f"训练任务 {task_id} 执行失败: {e}", exc_info=True)
            with self.task_lock:
                if task_id in self.training_tasks:
                    task = self.training_tasks[task_id]
                    task["status"] = TrainingTaskStatus.FAILED
                    task["progress"].status = TrainingTaskStatus.FAILED
                    task["progress"].error_message = str(e)
                    task["progress"].stack_trace = traceback.format_exc()
                    task["progress"].end_time = datetime.now(timezone.utc)
        
        finally:
            # 清理线程引用
            with self.task_lock:
                if task_id in self.training_tasks:
                    self.training_tasks[task_id]["thread"] = None
    
    def stop_training(self, task_id: str) -> Dict[str, Any]:
        """停止训练任务
        
        参数:
            task_id: 任务ID
            
        返回:
            停止结果
        """
        with self.task_lock:
            if task_id not in self.training_tasks:
                return {
                    "success": False,
                    "error": f"任务 {task_id} 不存在"
                }
            
            task = self.training_tasks[task_id]
            
            if task["status"] not in [TrainingTaskStatus.RUNNING, TrainingTaskStatus.PREPARING]:
                return {
                    "success": False,
                    "error": f"任务状态为 {task['status'].value}，无法停止"
                }
            
            # 设置停止事件
            if "stop_event" in task:
                task["stop_event"].set()
            
            # 更新状态
            task["status"] = TrainingTaskStatus.CANCELLED
            task["progress"].status = TrainingTaskStatus.CANCELLED
            task["progress"].end_time = datetime.now(timezone.utc)
        
        logger.info(f"停止训练任务 {task_id}")
        
        return {
            "success": True,
            "message": "训练任务已停止"
        }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态
        
        参数:
            task_id: 任务ID
            
        返回:
            任务状态
        """
        with self.task_lock:
            if task_id not in self.training_tasks:
                return {
                    "success": False,
                    "error": f"任务 {task_id} 不存在"
                }
            
            task = self.training_tasks[task_id]
            
            return {
                "success": True,
                "task_id": task_id,
                "status": task["status"].value,
                "progress": task["progress"].to_dict(),
                "config": task["config"].to_dict() if hasattr(task["config"], "to_dict") else asdict(task["config"]),
                "created_at": task["created_at"].isoformat(),
                "robot_id": task["robot_id"],
            }
    
    def list_tasks(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """列出任务
        
        参数:
            user_id: 用户ID过滤
            
        返回:
            任务列表
        """
        with self.task_lock:
            tasks = []
            for task_id, task in self.training_tasks.items():
                if user_id is None or task.get("user_id") == user_id:
                    tasks.append({
                        "task_id": task_id,
                        "status": task["status"].value,
                        "robot_id": task["robot_id"],
                        "created_at": task["created_at"].isoformat(),
                        "progress": task["progress"].to_dict(),
                    })
        
        return {
            "success": True,
            "tasks": tasks,
            "count": len(tasks)
        }
    
    def delete_task(self, task_id: str) -> Dict[str, Any]:
        """删除任务
        
        参数:
            task_id: 任务ID
            
        返回:
            删除结果
        """
        with self.task_lock:
            if task_id not in self.training_tasks:
                return {
                    "success": False,
                    "error": f"任务 {task_id} 不存在"
                }
            
            task = self.training_tasks[task_id]
            
            # 如果任务正在运行，先停止
            if task["status"] in [TrainingTaskStatus.RUNNING, TrainingTaskStatus.PREPARING]:
                if "stop_event" in task:
                    task["stop_event"].set()
                # 等待线程结束（最多5秒）
                if task["thread"] and task["thread"].is_alive():
                    task["thread"].join(timeout=5.0)
            
            # 删除任务
            del self.training_tasks[task_id]
        
        logger.info(f"删除训练任务 {task_id}")
        
        return {
            "success": True,
            "message": "任务已删除"
        }
    
    def evaluate_model(self, task_id: str, num_episodes: int = 10) -> Dict[str, Any]:
        """评估模型
        
        参数:
            task_id: 任务ID
            num_episodes: 评估集数
            
        返回:
            评估结果
        """
        with self.task_lock:
            if task_id not in self.training_tasks:
                return {
                    "success": False,
                    "error": f"任务 {task_id} 不存在"
                }
            
            task = self.training_tasks[task_id]
            model = task.get("model")
            
            if model is None:
                return {
                    "success": False,
                    "error": "模型不存在或未训练"
                }
            
            if task["environment"] is None:
                return {
                    "success": False,
                    "error": "环境不存在"
                }
        
        try:
            # 使用模型进行评估
            env = task["environment"]
            
            # 重置环境
            observation = env.reset()
            
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(num_episodes):
                episode_reward = 0.0
                episode_length = 0
                done = False
                
                while not done:
                    # 预测动作
                    action, _ = model.predict(observation, deterministic=True)
                    
                    # 执行动作
                    observation, reward, done, info = env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # 重置环境进行下一集
                if episode < num_episodes - 1:
                    observation = env.reset()
            
            # 计算统计信息
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            return {
                "success": True,
                "evaluation": {
                    "num_episodes": num_episodes,
                    "mean_reward": float(mean_reward),
                    "std_reward": float(std_reward),
                    "min_reward": float(np.min(episode_rewards)),
                    "max_reward": float(np.max(episode_rewards)),
                    "mean_length": float(mean_length),
                    "episode_rewards": [float(r) for r in episode_rewards],
                    "episode_lengths": [int(l) for l in episode_lengths],
                }
            }
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }