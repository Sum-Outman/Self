"""
自主决策引擎
为Self AGI系统提供基于强化学习的自主决策能力

功能：
1. 基于强化学习的自主决策算法
2. 环境状态感知和动态适应
3. 风险评估和收益预测
4. 多目标优化和策略生成
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import random
from collections import deque
import time

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """决策类型枚举"""
    
    EXPLORATION = auto()        # 探索决策：收集环境信息
    EXPLOITATION = auto()       # 利用决策：基于已知知识行动
    SAFETY = auto()             # 安全决策：避免危险
    LEARNING = auto()           # 学习决策：优化自身能力
    COOPERATION = auto()        # 合作决策：与其他系统协作
    ADAPTATION = auto()         # 适应决策：应对环境变化


@dataclass
class EnvironmentState:
    """环境状态表示"""
    
    # 传感器数据
    sensor_readings: Dict[str, float] = field(default_factory=dict)
    
    # 系统状态
    system_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 任务状态
    task_progress: Dict[str, float] = field(default_factory=dict)
    
    # 外部环境
    external_factors: Dict[str, Any] = field(default_factory=dict)
    
    # 时间信息
    timestamp: float = field(default_factory=time.time)
    
    def to_feature_vector(self) -> np.ndarray:
        """转换为特征向量
        
        返回:
            np.ndarray: 特征向量
        """
        features = []
        
        # 传感器特征
        for key in sorted(self.sensor_readings.keys()):
            features.append(self.sensor_readings[key])
        
        # 系统指标特征
        for key in sorted(self.system_metrics.keys()):
            features.append(self.system_metrics[key])
        
        # 任务进度特征
        for key in sorted(self.task_progress.keys()):
            features.append(self.task_progress[key])
        
        return np.array(features, dtype=np.float32)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要
        
        返回:
            Dict[str, Any]: 状态摘要
        """
        return {
            "sensor_count": len(self.sensor_readings),
            "system_metrics_count": len(self.system_metrics),
            "task_count": len(self.task_progress),
            "timestamp": self.timestamp,
            "sensor_keys": list(self.sensor_readings.keys()),
            "system_metric_keys": list(self.system_metrics.keys()),
            "task_keys": list(self.task_progress.keys()),
        }


@dataclass
class Decision:
    """决策定义"""
    
    id: str                     # 决策ID
    type: DecisionType         # 决策类型
    action: str                # 具体动作
    parameters: Dict[str, Any] = field(default_factory=dict)  # 动作参数
    confidence: float = 0.0    # 决策置信度 0.0-1.0
    expected_reward: float = 0.0  # 预期奖励
    risk_level: float = 0.0    # 风险等级 0.0-1.0
    created_at: float = field(default_factory=time.time)  # 创建时间
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type.name,
            "action": self.action,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "expected_reward": self.expected_reward,
            "risk_level": self.risk_level,
            "created_at": self.created_at,
        }


class DecisionNetwork(nn.Module):
    """决策神经网络
    
    功能：
    - 状态特征提取
    - 动作价值预测
    - 风险评估
    - 策略生成
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        """初始化决策网络
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # 动作价值网络（Q网络）
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
        # 风险评估网络
        self.risk_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),  # 输出0-1的风险值
        )
        
        # 置信度网络
        self.confidence_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),  # 输出0-1的置信度
        )
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        
        参数:
            state: 状态张量 [batch_size, state_dim]
            
        返回:
            Dict[str, torch.Tensor]: 网络输出
        """
        # 编码状态
        state_features = self.state_encoder(state)
        
        # 计算动作价值
        q_values = self.q_network(state_features)
        
        # 计算风险
        risk = self.risk_network(state_features)
        
        # 计算置信度
        confidence = self.confidence_network(state_features)
        
        return {
            "q_values": q_values,
            "risk": risk,
            "confidence": confidence,
            "state_features": state_features,
        }
    
    def get_best_action(self, state: torch.Tensor) -> Tuple[int, Dict[str, float]]:
        """获取最佳动作
        
        参数:
            state: 状态张量 [1, state_dim]
            
        返回:
            Tuple[int, Dict[str, float]]: (动作索引, 动作信息)
        """
        with torch.no_grad():
            outputs = self.forward(state)
            q_values = outputs["q_values"]
            risk = outputs["risk"]
            confidence = outputs["confidence"]
            
            # 选择Q值最大的动作
            best_action_idx = torch.argmax(q_values).item()
            
            # 获取动作信息
            action_info = {
                "q_value": q_values[0, best_action_idx].item(),
                "risk": risk[0, 0].item(),
                "confidence": confidence[0, 0].item(),
            }
            
            return best_action_idx, action_info


class DecisionEngine:
    """自主决策引擎
    
    功能：
    - 基于强化学习的决策制定
    - 环境状态感知和适应
    - 风险评估和收益预测
    - 决策策略优化
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化决策引擎
        
        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(f"{__name__}.DecisionEngine")
        
        # 默认配置
        self.config = config or {
            "state_dim": 64,              # 状态维度
            "action_dim": 20,             # 动作维度
            "hidden_dim": 256,            # 网络隐藏层维度
            "learning_rate": 0.001,       # 学习率
            "gamma": 0.99,                # 折扣因子
            "epsilon_start": 1.0,         # 探索起始概率
            "epsilon_end": 0.01,          # 探索结束概率
            "epsilon_decay": 0.995,       # 探索衰减率
            "batch_size": 64,             # 批大小
            "memory_size": 10000,         # 经验回放内存大小
            "target_update_freq": 100,    # 目标网络更新频率
            "risk_threshold": 0.7,        # 风险阈值（超过此值不执行）
            "confidence_threshold": 0.3,  # 置信度阈值（低于此值不执行）
            "enable_learning": True,      # 是否启用学习
        }
        
        # 决策网络
        self.state_dim = self.config["state_dim"]
        self.action_dim = self.config["action_dim"]
        self.hidden_dim = self.config["hidden_dim"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"决策引擎使用设备: {self.device}")
        
        # 主网络和目标网络（用于稳定训练）
        self.policy_net = DecisionNetwork(
            self.state_dim, 
            self.action_dim, 
            self.hidden_dim
        ).to(self.device)
        
        self.target_net = DecisionNetwork(
            self.state_dim, 
            self.action_dim, 
            self.hidden_dim
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络设为评估模式
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        # 经验回放内存
        self.memory = deque(maxlen=self.config["memory_size"])
        
        # 探索参数
        self.epsilon = self.config["epsilon_start"]
        self.steps_done = 0
        
        # 决策历史
        self.decision_history: List[Decision] = []
        self.max_history_size = 1000
        
        # 动作映射（预定义动作集合）
        self.action_map = self._initialize_action_map()
        
        # 统计信息
        self.stats = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "avg_decision_time": 0.0,
            "avg_reward": 0.0,
            "avg_risk": 0.0,
            "avg_confidence": 0.0,
            "exploration_rate": self.epsilon,
        }
        
        self.logger.info("自主决策引擎初始化完成")
    
    def _initialize_action_map(self) -> Dict[int, Dict[str, Any]]:
        """初始化动作映射
        
        返回:
            Dict[int, Dict[str, Any]]: 动作ID到动作信息的映射
        """
        # 预定义动作集合
        actions = [
            # 探索类动作
            {"id": 0, "type": DecisionType.EXPLORATION, "name": "环境扫描", "description": "扫描周围环境，收集信息"},
            {"id": 1, "type": DecisionType.EXPLORATION, "name": "传感器校准", "description": "校准传感器，提高数据精度"},
            {"id": 2, "type": DecisionType.EXPLORATION, "name": "未知区域探索", "description": "探索未知区域，扩展认知"},
            
            # 利用类动作
            {"id": 3, "type": DecisionType.EXPLOITATION, "name": "执行已知任务", "description": "执行已掌握的任务"},
            {"id": 4, "type": DecisionType.EXPLOITATION, "name": "优化当前策略", "description": "基于现有知识优化策略"},
            {"id": 5, "type": DecisionType.EXPLOITATION, "name": "资源收集", "description": "收集所需资源"},
            
            # 安全类动作
            {"id": 6, "type": DecisionType.SAFETY, "name": "安全检查", "description": "进行系统安全检查"},
            {"id": 7, "type": DecisionType.SAFETY, "name": "风险规避", "description": "规避已知风险"},
            {"id": 8, "type": DecisionType.SAFETY, "name": "紧急停止", "description": "紧急停止当前操作"},
            
            # 学习类动作
            {"id": 9, "type": DecisionType.LEARNING, "name": "模型训练", "description": "训练决策模型"},
            {"id": 10, "type": DecisionType.LEARNING, "name": "经验总结", "description": "总结过往经验"},
            {"id": 11, "type": DecisionType.LEARNING, "name": "知识更新", "description": "更新知识库"},
            
            # 合作类动作
            {"id": 12, "type": DecisionType.COOPERATION, "name": "请求协助", "description": "向其他系统请求协助"},
            {"id": 13, "type": DecisionType.COOPERATION, "name": "资源共享", "description": "与其他系统共享资源"},
            {"id": 14, "type": DecisionType.COOPERATION, "name": "任务协调", "description": "协调多系统任务"},
            
            # 适应类动作
            {"id": 15, "type": DecisionType.ADAPTATION, "name": "环境适应", "description": "适应环境变化"},
            {"id": 16, "type": DecisionType.ADAPTATION, "name": "参数调整", "description": "调整系统参数"},
            {"id": 17, "type": DecisionType.ADAPTATION, "name": "策略切换", "description": "切换执行策略"},
            
            # 其他动作
            {"id": 18, "type": DecisionType.EXPLOITATION, "name": "等待观察", "description": "等待并观察环境变化"},
            {"id": 19, "type": DecisionType.EXPLORATION, "name": "测试新策略", "description": "测试新的决策策略"},
        ]
        
        # 转换为映射
        action_map = {}
        for action in actions:
            action_map[action["id"]] = action
        
        return action_map
    
    def make_decision(self, state: EnvironmentState) -> Optional[Decision]:
        """制定决策
        
        参数:
            state: 环境状态
            
        返回:
            Optional[Decision]: 决策对象，如果无法制定决策返回None
        """
        start_time = time.time()
        
        try:
            # 转换状态为特征向量
            state_features = state.to_feature_vector()
            
            # 如果特征维度不匹配，进行填充或截断
            if len(state_features) != self.state_dim:
                state_features = self._adjust_feature_dimension(state_features)
            
            # 转换为张量
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
            
            # ε-贪婪策略：以ε概率探索，否则利用
            if random.random() < self.epsilon:
                # 探索：随机选择动作
                action_idx = random.randint(0, self.action_dim - 1)
                decision_type = self._get_decision_type_for_action(action_idx)
                
                # 使用目标网络评估动作
                with torch.no_grad():
                    outputs = self.target_net(state_tensor)
                    q_value = outputs["q_values"][0, action_idx].item()
                    risk = outputs["risk"][0, 0].item()
                    confidence = outputs["confidence"][0, 0].item()
                
                exploration_flag = True
            else:
                # 利用：选择最佳动作
                action_idx, action_info = self.policy_net.get_best_action(state_tensor)
                decision_type = self._get_decision_type_for_action(action_idx)
                
                q_value = action_info["q_value"]
                risk = action_info["risk"]
                confidence = action_info["confidence"]
                
                exploration_flag = False
            
            # 检查风险和置信度阈值
            if (risk > self.config["risk_threshold"] or 
                confidence < self.config["confidence_threshold"]):
                self.logger.warning(f"决策被拒绝: 风险={risk:.3f}, 置信度={confidence:.3f}")
                return None  # 返回None
            
            # 获取动作信息
            action_info = self.action_map.get(action_idx, {
                "name": f"动作_{action_idx}",
                "description": "未知动作",
                "type": DecisionType.EXPLORATION,
            })
            
            # 计算预期奖励（基于Q值）
            expected_reward = q_value * (1 - risk) * confidence
            
            # 生成决策ID
            decision_id = f"decision_{int(time.time())}_{action_idx:03d}"
            
            # 创建决策对象
            decision = Decision(
                id=decision_id,
                type=decision_type,
                action=action_info["name"],
                parameters={
                    "action_idx": action_idx,
                    "exploration": exploration_flag,
                    "state_summary": state.get_state_summary(),
                },
                confidence=confidence,
                expected_reward=expected_reward,
                risk_level=risk,
            )
            
            # 更新统计信息
            decision_time = time.time() - start_time
            self.stats["total_decisions"] += 1
            self.stats["avg_decision_time"] = (
                self.stats["avg_decision_time"] * (self.stats["total_decisions"] - 1) + decision_time
            ) / self.stats["total_decisions"]
            self.stats["avg_risk"] = (
                self.stats["avg_risk"] * (self.stats["total_decisions"] - 1) + risk
            ) / self.stats["total_decisions"]
            self.stats["avg_confidence"] = (
                self.stats["avg_confidence"] * (self.stats["total_decisions"] - 1) + confidence
            ) / self.stats["total_decisions"]
            self.stats["exploration_rate"] = self.epsilon
            
            # 添加到历史
            self.decision_history.append(decision)
            if len(self.decision_history) > self.max_history_size:
                self.decision_history.pop(0)
            
            self.logger.info(
                f"决策制定: {decision.action} "
                f"(类型: {decision.type.name}, "
                f"置信度: {confidence:.3f}, "
                f"风险: {risk:.3f}, "
                f"预期奖励: {expected_reward:.3f})"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"决策制定失败: {e}")
            return None  # 返回None
    
    def update_decision_result(self, decision_id: str, 
                             success: bool, 
                             actual_reward: float,
                             new_state: Optional[EnvironmentState] = None) -> bool:
        """更新决策结果
        
        参数:
            decision_id: 决策ID
            success: 是否成功
            actual_reward: 实际获得的奖励
            new_state: 新状态（可选）
            
        返回:
            bool: 更新是否成功
        """
        # 查找决策
        decision = None
        for d in self.decision_history:
            if d.id == decision_id:
                decision = d
                break
        
        if not decision:
            self.logger.error(f"决策不存在: {decision_id}")
            return False
        
        # 更新决策结果信息
        decision.parameters["success"] = success
        decision.parameters["actual_reward"] = actual_reward
        decision.parameters["result_updated_at"] = time.time()
        
        # 更新统计信息
        if success:
            self.stats["successful_decisions"] += 1
        else:
            self.stats["failed_decisions"] += 1
        
        self.stats["avg_reward"] = (
            self.stats["avg_reward"] * (self.stats["successful_decisions"] + self.stats["failed_decisions"] - 1) + actual_reward
        ) / (self.stats["successful_decisions"] + self.stats["failed_decisions"])
        
        # 如果启用了学习，将经验存储到回放内存
        if self.config["enable_learning"] and new_state is not None:
            # 获取旧状态特征
            old_state_features = self._adjust_feature_dimension(
                decision.parameters["state_summary"]["sensor_keys"]  # 完整处理
            )
            
            # 获取新状态特征
            new_state_features = self._adjust_feature_dimension(
                new_state.to_feature_vector()
            )
            
            # 计算奖励（根据成功与否调整）
            reward = actual_reward if success else -abs(actual_reward)
            
            # 存储经验
            self._store_experience(
                old_state_features,
                decision.parameters.get("action_idx", 0),
                reward,
                new_state_features,
                success
            )
            
            # 学习
            self._learn_from_experience()
        
        self.logger.info(f"决策结果更新: {decision_id}, 成功: {success}, 奖励: {actual_reward}")
        return True
    
    def _adjust_feature_dimension(self, features: np.ndarray) -> np.ndarray:
        """调整特征维度
        
        参数:
            features: 原始特征
            
        返回:
            np.ndarray: 调整后的特征
        """
        if len(features) == self.state_dim:
            return features
        elif len(features) > self.state_dim:
            # 截断
            return features[:self.state_dim]
        else:
            # 填充零
            padded = np.zeros(self.state_dim, dtype=np.float32)
            padded[:len(features)] = features
            return padded
    
    def _get_decision_type_for_action(self, action_idx: int) -> DecisionType:
        """根据动作索引获取决策类型
        
        参数:
            action_idx: 动作索引
            
        返回:
            DecisionType: 决策类型
        """
        action_info = self.action_map.get(action_idx)
        if action_info and "type" in action_info:
            return action_info["type"]
        return DecisionType.EXPLORATION
    
    def _store_experience(self, 
                         state: np.ndarray, 
                         action: int, 
                         reward: float,
                         next_state: np.ndarray,
                         done: bool):
        """存储经验到回放内存
        
        参数:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一个状态
            done: 是否结束
        """
        experience = (
            torch.FloatTensor(state),
            torch.tensor([action], dtype=torch.long),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.tensor([done], dtype=torch.bool)
        )
        
        self.memory.append(experience)
    
    def _learn_from_experience(self):
        """从经验中学习"""
        if len(self.memory) < self.config["batch_size"]:
            return
        
        # 随机采样批次
        batch = random.sample(self.memory, self.config["batch_size"])
        
        # 解包批次
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states_tensor = torch.stack(states).to(self.device)
        actions_tensor = torch.cat(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.cat(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.stack(next_states).to(self.device)
        dones_tensor = torch.cat(dones).unsqueeze(1).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states_tensor)["q_values"]
        current_q = current_q_values.gather(1, actions_tensor)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor)["q_values"]
            max_next_q = next_q_values.max(1)[0].unsqueeze(1)
            
            # 如果回合结束，下一个状态的Q值为0
            target_q = rewards_tensor + self.config["gamma"] * max_next_q * (~dones_tensor)
        
        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        # 更新参数
        self.optimizer.step()
        
        # 更新探索率
        self.epsilon = max(
            self.config["epsilon_end"],
            self.epsilon * self.config["epsilon_decay"]
        )
        
        # 更新步数
        self.steps_done += 1
        
        # 定期更新目标网络
        if self.steps_done % self.config["target_update_freq"] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.logger.debug(f"学习步骤: {self.steps_done}, 损失: {loss.item():.4f}, 探索率: {self.epsilon:.4f}")
    
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取决策历史
        
        参数:
            limit: 返回的历史记录数量限制
            
        返回:
            List[Dict[str, Any]]: 决策历史
        """
        history = self.decision_history[-limit:] if limit > 0 else self.decision_history
        return [decision.to_dict() for decision in history]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        return {
            **self.stats,
            "memory_size": len(self.memory),
            "decision_history_size": len(self.decision_history),
            "action_space_size": self.action_dim,
            "device": str(self.device),
            "model_parameters": sum(p.numel() for p in self.policy_net.parameters()),
            "timestamp": time.time(),
        }
    
    def save_model(self, path: str):
        """保存模型
        
        参数:
            path: 保存路径
        """
        try:
            checkpoint = {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "stats": self.stats,
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
            }
            
            torch.save(checkpoint, path)
            self.logger.info(f"模型保存成功: {path}")
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
    
    def load_model(self, path: str):
        """加载模型
        
        参数:
            path: 加载路径
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            self.config = checkpoint.get("config", self.config)
            self.stats = checkpoint.get("stats", self.stats)
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.steps_done = checkpoint.get("steps_done", self.steps_done)
            
            self.logger.info(f"模型加载成功: {path}")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
    
    def reset(self):
        """重置决策引擎"""
        self.logger.info("重置决策引擎")
        
        # 重新初始化网络
        self.policy_net = DecisionNetwork(
            self.state_dim, 
            self.action_dim, 
            self.hidden_dim
        ).to(self.device)
        
        self.target_net = DecisionNetwork(
            self.state_dim, 
            self.action_dim, 
            self.hidden_dim
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 重新初始化优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        # 清空内存和历史
        self.memory.clear()
        self.decision_history.clear()
        
        # 重置参数
        self.epsilon = self.config["epsilon_start"]
        self.steps_done = 0
        
        # 重置统计信息
        self.stats = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "avg_decision_time": 0.0,
            "avg_reward": 0.0,
            "avg_risk": 0.0,
            "avg_confidence": 0.0,
            "exploration_rate": self.epsilon,
        }
        
        self.logger.info("决策引擎重置完成")


# 全局实例
_decision_engine_instance = None


def get_decision_engine(config: Optional[Dict[str, Any]] = None) -> DecisionEngine:
    """获取决策引擎单例
    
    参数:
        config: 配置字典
        
    返回:
        DecisionEngine: 决策引擎实例
    """
    global _decision_engine_instance
    
    if _decision_engine_instance is None:
        _decision_engine_instance = DecisionEngine(config)
    
    return _decision_engine_instance


__all__ = [
    "DecisionEngine",
    "get_decision_engine",
    "DecisionType",
    "EnvironmentState",
    "Decision",
]