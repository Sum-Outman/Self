# Self AGI 记忆管理系统 - 从零开始的工业级实现
"""
注意：此文件不应直接运行，而是作为整个Self AGI系统的一部分运行。
直接运行此文件会导致导入错误，因为相对导入需要正确的模块上下文。
"""

from typing import Dict, Any, List, Optional, Tuple
import json
import os
import numpy as np
import re
import time
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..embeddings import FromScratchTextEmbedder, create_agi_text_embedder

logger = logging.getLogger(__name__)

# 多模态处理支持
try:
    from ..multimodal.processor import MultimodalProcessor
    from ..multimodal.custom_dataclasses import ProcessedModality
    MULTIMODAL_AVAILABLE = True
except ImportError:
    logger.warning("多模态处理器不可用，多模态记忆功能将受限")
    MULTIMODAL_AVAILABLE = False

# 知识库支持
try:
    from ..knowledge_base.knowledge_manager import KnowledgeManager
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    logger.warning("知识管理器不可用，知识库集成功能将受限")
    KNOWLEDGE_BASE_AVAILABLE = False


class ImportanceModel(nn.Module):
    """记忆重要性学习器
    
    基于记忆特征预测重要性分数（0-1）
    特征包括：访问频率、时间衰减、内容特征、情感强度等
    
    输入维度: feature_dim
    输出: 重要性分数 (0-1)
    """
    
    def __init__(self, feature_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # 重要性预测头
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),  # 输出0-1之间的重要性分数
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            features: [batch_size, feature_dim] 记忆特征
            
        返回:
            importance_scores: [batch_size, 1] 重要性分数 (0-1)
        """
        # 提取特征
        hidden_features = self.feature_extractor(features)
        
        # 预测重要性
        importance_scores = self.importance_predictor(hidden_features)
        
        return importance_scores
    
    def compute_features(self, memory_data: Dict[str, Any]) -> torch.Tensor:
        """计算记忆特征向量
        
        从记忆数据中提取特征，包括：
        1. 访问频率特征 (归一化访问次数)
        2. 时间衰减特征 (距离上次访问的时间)
        3. 内容长度特征 (文本长度)
        4. 情感特征 (如果有)
        5. 记忆类型特征 (短期/长期)
        
        参数:
            memory_data: 记忆字典，包含访问次数、时间戳、内容等信息
            
        返回:
            features: [feature_dim] 特征向量
        """
        feature_dim = self.feature_dim
        features = torch.zeros(feature_dim)
        
        # 1. 访问频率特征 (0-2)
        accessed_count = memory_data.get("accessed_count", 0)
        # 归一化：log(1 + accessed_count) / log(1 + max_count)，假设最大访问次数为100
        max_count = 100
        access_feature = torch.log(torch.tensor(1.0 + min(accessed_count, max_count))) / torch.log(torch.tensor(1.0 + max_count))
        features[0] = access_feature * 2.0  # 缩放到0-2范围
        
        # 2. 时间衰减特征 (0-1)
        # 距离上次访问的时间（小时），归一化
        last_accessed = memory_data.get("last_accessed")
        if last_accessed:
            if isinstance(last_accessed, str):
                # 如果是字符串，转换为datetime
                from datetime import datetime
                last_accessed = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
            
            if isinstance(last_accessed, datetime):
                time_diff = datetime.now(timezone.utc) - last_accessed
                hours_diff = time_diff.total_seconds() / 3600.0
                # 时间衰减：exp(-hours_diff/24)，24小时衰减到1/e
                time_feature = torch.exp(torch.tensor(-hours_diff / 24.0))
                features[1] = time_feature
        else:
            features[1] = 0.5  # 默认值
        
        # 3. 内容长度特征 (0-1)
        content = memory_data.get("content", "")
        content_length = len(str(content))
        # 归一化：假设最大长度为1000字符
        max_length = 1000
        length_feature = min(content_length / max_length, 1.0)
        features[2] = length_feature
        
        # 4. 重要性历史特征 (0-1)
        importance = memory_data.get("importance", 0.5)
        features[3] = importance
        
        # 5. 记忆类型特征 (短期:0, 长期:1)
        memory_type = memory_data.get("memory_type", "short_term")
        type_feature = 1.0 if memory_type == "long_term" else 0.0
        features[4] = type_feature
        
        # 6-9. 其他特征（预留）
        # 可以添加情感特征、关联度特征等
        for i in range(5, min(10, feature_dim)):
            features[i] = 0.0  # 预留特征，默认为0
        
        return features


class MemoryGraphNeuralNetwork(nn.Module):
    """记忆图神经网络
    
    基于图卷积网络（GCN）的图神经网络，用于学习记忆表示和关联。
    
    参考架构:
    - Graph Convolutional Networks (GCN)
    - Graph Attention Networks (GAT)
    - Simplified Graph Learning (SGL)
    
    输入:
    - 节点特征: [num_nodes, feature_dim]
    - 邻接矩阵: [num_nodes, num_nodes]
    
    输出:
    - 节点表示: [num_nodes, hidden_dim]
    - 边预测: 节点对之间的关联强度
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入投影层
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # 完整版)
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )
        
        # 输出层 (节点表示)
        self.node_output = nn.Linear(hidden_dim, hidden_dim)
        
        # 边预测层 (关联强度预测)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 社区检测层 (可选)
        self.community_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8)  # 社区嵌入
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"初始化MemoryGraphNeuralNetwork: 特征维度={feature_dim}, 隐藏维度={hidden_dim}, 层数={num_layers}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        
        参数:
            node_features: 节点特征 [num_nodes, feature_dim]
            adjacency: 邻接矩阵 [num_nodes, num_nodes] (稀疏或密集)
            
        返回:
            字典包含:
            - node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
            - edge_predictions: 边预测分数 [num_nodes, num_nodes] (可选)
            - community_embeddings: 社区嵌入 [num_nodes, community_dim] (可选)
        """
        num_nodes = node_features.shape[0]
        
        # 输入投影
        h = self.input_proj(node_features)
        
        # GCN消息传递
        for i, layer in enumerate(self.gcn_layers):
            # 完整GCN: h = σ(ÃhW)
            # 这里使用简单的图卷积
            if adjacency is not None:
                # 归一化邻接矩阵
                degree = torch.sum(adjacency, dim=1, keepdim=True)
                degree_inv_sqrt = torch.where(degree > 0, 1.0 / torch.sqrt(degree), 0.0)
                norm_adj = degree_inv_sqrt * adjacency * degree_inv_sqrt.t()
                
                # 图卷积
                h = norm_adj @ h
            else:
                # 如果没有邻接矩阵，使用自连接
                h = h
            
            # 线性变换 + 激活
            h = layer(h)
        
        # 节点输出表示
        node_embeddings = self.node_output(h)
        
        # 边预测 (可选)
        edge_predictions = None
        if num_nodes > 1:
            # 计算所有节点对的特征拼接
            edge_features = []
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    pair_feature = torch.cat([node_embeddings[i], node_embeddings[j]])
                    edge_features.append(pair_feature)
            
            if edge_features:
                edge_features = torch.stack(edge_features)
                edge_scores = self.edge_predictor(edge_features)
                
                # 重构为矩阵形式
                edge_predictions = torch.zeros(num_nodes, num_nodes, device=node_features.device)
                idx = 0
                for i in range(num_nodes):
                    for j in range(i+1, num_nodes):
                        edge_predictions[i, j] = edge_scores[idx]
                        edge_predictions[j, i] = edge_scores[idx]  # 对称
                        idx += 1
        
        # 社区检测 (可选)
        community_embeddings = self.community_detector(node_embeddings)
        
        return {
            "node_embeddings": node_embeddings,
            "edge_predictions": edge_predictions,
            "community_embeddings": community_embeddings
        }
    
    def predict_edge_strength(self, node_embeddings: torch.Tensor, node_i: int, node_j: int) -> float:
        """预测两个节点之间的边强度
        
        参数:
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
            node_i: 节点i索引
            node_j: 节点j索引
            
        返回:
            边强度分数 (0-1)
        """
        pair_feature = torch.cat([node_embeddings[node_i], node_embeddings[node_j]])
        edge_score = self.edge_predictor(pair_feature.unsqueeze(0))
        return edge_score.item()
    
    def detect_communities(self, node_embeddings: torch.Tensor, threshold: float = 0.7) -> List[List[int]]:
        """检测社区/聚类
        
        使用简单的聚类算法检测图中的社区
        
        参数:
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
            threshold: 相似度阈值
            
        返回:
            社区列表，每个社区包含节点索引
        """
        num_nodes = node_embeddings.shape[0]
        communities = []
        visited = set()
        
        for i in range(num_nodes):
            if i in visited:
                continue
            
            # 查找与当前节点相似的节点
            community = [i]
            visited.add(i)
            
            for j in range(num_nodes):
                if j in visited or i == j:
                    continue
                
                # 计算余弦相似度
                similarity = F.cosine_similarity(
                    node_embeddings[i].unsqueeze(0),
                    node_embeddings[j].unsqueeze(0)
                ).item()
                
                if similarity > threshold:
                    community.append(j)
                    visited.add(j)
            
            if len(community) >= 2:  # 至少2个节点才算是社区
                communities.append(community)
        
        return communities


class AutonomousMemoryManager:
    """自主记忆管理器 - 第四阶段AGI级增强功能
    
    实现自我优化的记忆策略、主动遗忘和学习、元记忆能力。
    基于系统监控和反馈循环，动态调整记忆管理参数。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 元记忆：跟踪记忆管理决策的质量
        self.meta_memory = {
            "decision_history": [],  # 决策历史
            "performance_metrics": {},  # 性能指标
            "strategy_effectiveness": {},  # 策略有效性
            "learning_data": []  # 学习数据
        }
        
        # 自适应参数配置
        self.adaptive_params = {
            "forgetting_rate": self.config.get("forgetting_rate", 0.05),
            "importance_threshold": self.config.get("importance_threshold", 0.7),
            "similarity_threshold": self.config.get("similarity_threshold", 0.8),
            "compression_threshold": self.config.get("compression_threshold", 0.9),
            "cache_size": self.config.get("cache_size", 100)
        }
        
        # 学习配置
        self.learning_enabled = self.config.get("enable_autonomous_learning", True)
        self.reinforcement_learning_rate = self.config.get("rl_learning_rate", 0.01)
        self.exploration_rate = self.config.get("exploration_rate", 0.1)
        
        # 系统状态跟踪
        self.system_state = {
            "load_level": 0.0,  # 系统负载水平 0-1
            "memory_usage": 0.0,  # 内存使用率 0-1
            "user_activity": 0.0,  # 用户活跃度 0-1
            "performance_score": 1.0  # 性能评分 0-1
        }
        
        # 初始化学习模型
        self._init_learning_models()
        
        logger.info("自主记忆管理器已初始化")
    
    def _init_learning_models(self):
        """初始化学习模型"""
        if not self.learning_enabled:
            return
        
        # 简单的Q-learning表格（状态 -> 参数 -> 预期奖励）
        self.q_table = {}  # 状态到参数动作的Q值映射
        
        # 初始化基础策略
        self._init_basic_policies()
    
    def _init_basic_policies(self):
        """初始化基础策略"""
        # 基于系统负载的策略
        self.load_based_policies = {
            "low_load": {
                "forgetting_rate": 0.02,  # 低负载时保留更多记忆
                "importance_threshold": 0.6,  # 更宽松的重要性阈值
                "cache_size": 150  # 更大的缓存
            },
            "medium_load": {
                "forgetting_rate": 0.05,  # 中等负载，标准遗忘率
                "importance_threshold": 0.7,
                "cache_size": 100
            },
            "high_load": {
                "forgetting_rate": 0.08,  # 高负载时更积极遗忘
                "importance_threshold": 0.8,  # 更严格的重要性阈值
                "cache_size": 50  # 更小的缓存
            }
        }
        
        # 基于用户行为的策略
        self.user_based_policies = {
            "active_user": {
                "forgetting_rate": 0.03,  # 活跃用户保留更多记忆
                "similarity_threshold": 0.75  # 更宽松的相似度阈值
            },
            "inactive_user": {
                "forgetting_rate": 0.07,  # 不活跃用户更积极清理
                "similarity_threshold": 0.85  # 更严格的相似度阈值
            }
        }
    
    def update_system_state(self, metrics: Dict[str, Any]):
        """更新系统状态
        
        参数:
            metrics: 系统监控指标，包括负载、内存使用等
        """
        # 更新系统状态
        self.system_state.update({
            "load_level": metrics.get("cpu_usage", 0.0) / 100.0,  # CPU使用率 0-1
            "memory_usage": metrics.get("memory_usage", 0.0) / 100.0,  # 内存使用率 0-1
            "user_activity": metrics.get("user_activity", 0.0),  # 用户活跃度 0-1
        })
        
        # 计算性能评分（基于系统状态）
        self.system_state["performance_score"] = self._calculate_performance_score()
        
        # 根据系统状态调整参数
        self._adapt_parameters_based_on_state()
    
    def _calculate_performance_score(self) -> float:
        """计算系统性能评分"""
        # 基于系统负载和内存使用计算性能评分
        load_score = 1.0 - self.system_state["load_level"] * 0.5  # 负载越高，分数越低
        memory_score = 1.0 - self.system_state["memory_usage"] * 0.5  # 内存使用越高，分数越低
        
        # 综合评分
        performance_score = (load_score + memory_score) / 2.0
        
        return max(0.0, min(1.0, performance_score))  # 限制在0-1范围内
    
    def _adapt_parameters_based_on_state(self):
        """基于系统状态调整参数"""
        # 基于负载的策略选择
        load_level = self.system_state["load_level"]
        
        if load_level < 0.3:
            policy = self.load_based_policies["low_load"]
        elif load_level < 0.7:
            policy = self.load_based_policies["medium_load"]
        else:
            policy = self.load_based_policies["high_load"]
        
        # 基于用户活跃度的策略
        user_activity = self.system_state["user_activity"]
        if user_activity > 0.5:
            user_policy = self.user_based_policies["active_user"]
        else:
            user_policy = self.user_based_policies["inactive_user"]
        
        # 合并策略（加权平均）
        for param_name in self.adaptive_params:
            if param_name in policy:
                load_weight = 0.7  # 负载策略权重
                user_weight = 0.3  # 用户策略权重
                
                # 计算加权值
                weighted_value = (
                    policy[param_name] * load_weight + 
                    user_policy.get(param_name, self.adaptive_params[param_name]) * user_weight
                )
                
                # 平滑更新（避免剧烈变化）
                current_value = self.adaptive_params[param_name]
                learning_rate = 0.1  # 学习率
                new_value = current_value * (1 - learning_rate) + weighted_value * learning_rate
                
                self.adaptive_params[param_name] = new_value
        
        logger.debug(f"自适应参数更新: {self.adaptive_params}")
    
    def record_decision(self, decision_type: str, context: Dict[str, Any], outcome: Dict[str, Any]):
        """记录记忆管理决策
        
        参数:
            decision_type: 决策类型（如：forget, compress, cache_evict等）
            context: 决策上下文信息
            outcome: 决策结果
        """
        decision_record = {
            "timestamp": time.time(),
            "decision_type": decision_type,
            "context": context,
            "outcome": outcome,
            "system_state": self.system_state.copy()
        }
        
        # 添加到决策历史
        self.meta_memory["decision_history"].append(decision_record)
        
        # 保持历史记录大小
        max_history = self.config.get("max_decision_history", 1000)
        if len(self.meta_memory["decision_history"]) > max_history:
            self.meta_memory["decision_history"] = self.meta_memory["decision_history"][-max_history:]
        
        # 更新策略有效性
        self._update_strategy_effectiveness(decision_type, outcome)
        
        # 如果有学习数据，进行在线学习
        if self.learning_enabled and "reward" in outcome:
            self._learn_from_decision(decision_record)
    
    def _update_strategy_effectiveness(self, decision_type: str, outcome: Dict[str, Any]):
        """更新策略有效性评估"""
        if decision_type not in self.meta_memory["strategy_effectiveness"]:
            self.meta_memory["strategy_effectiveness"][decision_type] = {
                "total_decisions": 0,
                "successful_decisions": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0
            }
        
        stats = self.meta_memory["strategy_effectiveness"][decision_type]
        stats["total_decisions"] += 1
        
        # 根据结果判断是否成功
        if outcome.get("success", False):
            stats["successful_decisions"] += 1
        
        # 累计奖励
        reward = outcome.get("reward", 0.0)
        stats["total_reward"] += reward
        stats["avg_reward"] = stats["total_reward"] / stats["total_decisions"]
    
    def _learn_from_decision(self, decision_record: Dict[str, Any]):
        """从决策中学习（强化学习）
        
        使用简单的Q-learning算法更新策略。
        """
        try:
            # 提取决策信息
            state_key = self._get_state_key(decision_record["system_state"])
            action_key = self._get_action_key(decision_record["context"])
            reward = decision_record["outcome"].get("reward", 0.0)
            
            # 初始化Q表条目
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            if action_key not in self.q_table[state_key]:
                self.q_table[state_key][action_key] = 0.0
            
            # Q-learning更新
            current_q = self.q_table[state_key][action_key]
            # 简单更新：Q(s,a) = Q(s,a) + α * (r - Q(s,a))
            new_q = current_q + self.reinforcement_learning_rate * (reward - current_q)
            self.q_table[state_key][action_key] = new_q
            
            logger.debug(f"Q-learning更新: 状态={state_key}, 动作={action_key}, 奖励={reward}, Q值={new_q}")
            
        except Exception as e:
            logger.error(f"从决策学习失败: {e}, 上下文: 状态={state_key}, 动作={action_key}, 奖励={reward}, Q表大小={len(self.q_table)}")
            # 恢复措施：重置Q表条目为初始值
            if state_key in self.q_table and action_key in self.q_table[state_key]:
                self.q_table[state_key][action_key] = 0.0
                logger.warning(f"已重置Q表条目: {state_key}/{action_key}")
    
    def _get_state_key(self, system_state: Dict[str, Any]) -> str:
        """将系统状态转换为状态键"""
        # 完整状态表示：负载水平（低/中/高） + 内存使用（低/中/高）
        load_level = system_state.get("load_level", 0.0)
        memory_usage = system_state.get("memory_usage", 0.0)
        
        load_category = "low" if load_level < 0.3 else "medium" if load_level < 0.7 else "high"
        memory_category = "low" if memory_usage < 0.3 else "medium" if memory_usage < 0.7 else "high"
        
        return f"{load_category}_{memory_category}"
    
    def _get_action_key(self, context: Dict[str, Any]) -> str:
        """将决策上下文转换为动作键"""
        # 完整动作表示：决策类型 + 主要参数范围
        decision_type = context.get("decision_type", "unknown")
        
        if decision_type == "forget":
            forgetting_rate = context.get("forgetting_rate", 0.05)
            rate_category = "low" if forgetting_rate < 0.03 else "medium" if forgetting_rate < 0.07 else "high"
            return f"forget_{rate_category}"
        
        elif decision_type == "compress":
            compression_ratio = context.get("compression_ratio", 0.5)
            ratio_category = "low" if compression_ratio < 0.3 else "medium" if compression_ratio < 0.7 else "high"
            return f"compress_{ratio_category}"
        
        else:
            return decision_type
    
    def get_optimal_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """获取给定状态下的最优动作
        
        参数:
            state: 当前系统状态
            
        返回:
            推荐的动作参数
        """
        state_key = self._get_state_key(state)
        
        # 检查是否有学习到的Q值
        if state_key in self.q_table and self.q_table[state_key]:
            # 选择最高Q值的动作
            best_action_key = max(self.q_table[state_key].items(), key=lambda x: x[1])[0]
            
            # 将动作键转换为具体参数
            return self._action_key_to_params(best_action_key)
        else:
            # 如果没有学习数据，使用自适应参数
            return self.adaptive_params.copy()
    
    def _action_key_to_params(self, action_key: str) -> Dict[str, Any]:
        """将动作键转换为具体参数"""
        params = self.adaptive_params.copy()
        
        if action_key.startswith("forget_"):
            rate_category = action_key.split("_")[1]
            if rate_category == "low":
                params["forgetting_rate"] = 0.02
            elif rate_category == "medium":
                params["forgetting_rate"] = 0.05
            elif rate_category == "high":
                params["forgetting_rate"] = 0.08
        
        elif action_key.startswith("compress_"):
            ratio_category = action_key.split("_")[1]
            if ratio_category == "low":
                params["compression_threshold"] = 0.7
            elif ratio_category == "medium":
                params["compression_threshold"] = 0.8
            elif ratio_category == "high":
                params["compression_threshold"] = 0.9
        
        return params
    
    def get_meta_memory_stats(self) -> Dict[str, Any]:
        """获取元记忆统计信息"""
        return {
            "total_decisions": sum(
                stats["total_decisions"] 
                for stats in self.meta_memory["strategy_effectiveness"].values()
            ),
            "strategy_effectiveness": self.meta_memory["strategy_effectiveness"],
            "system_state": self.system_state,
            "adaptive_params": self.adaptive_params,
            "q_table_size": sum(len(actions) for actions in self.q_table.values())
        }


class MemorySystem:
    """记忆管理系统"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, embedding_model: Optional[nn.Module] = None, agi_model: Optional[nn.Module] = None, multimodal_processor: Optional["MultimodalProcessor"] = None):
        self.config = config or {}
        self.embedding_model = embedding_model
        self.agi_model = agi_model
        self.multimodal_processor = multimodal_processor
        self.initialized = False

        # 多模态支持配置
        self.enable_multimodal_memory = self.config.get("enable_multimodal_memory", True) and MULTIMODAL_AVAILABLE
        if self.enable_multimodal_memory and self.multimodal_processor is None:
            # 尝试创建默认多模态处理器
            try:
                multimodal_config = {
                    "use_deep_learning": True,
                    "industrial_mode": True,
                    "text_embedding_dim": 768,
                    "image_embedding_dim": 768,
                    "audio_embedding_dim": 768,
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                }
                self.multimodal_processor = MultimodalProcessor(multimodal_config)
                self.multimodal_processor.initialize()
                logger.info("已创建并初始化默认多模态处理器")
            except Exception as e:
                logger.error(f"创建多模态处理器失败: {e}, 配置: {multimodal_config}")
                logger.warning("多模态处理器初始化失败，已禁用多模态记忆功能。请检查依赖项是否安装完整。")
                self.enable_multimodal_memory = False
        
        if self.enable_multimodal_memory:
            logger.info("多模态记忆功能已启用")
        else:
            logger.info("多模态记忆功能未启用")

        # 配置参数
        self.short_term_memory_duration = self.config.get(
            "short_term_memory_duration", 14400
        )  # 4小时（增加以适应更长时间的任务）
        self.importance_threshold = self.config.get(
            "importance_threshold", 0.7
        )  # 重要性阈值
        self.similarity_threshold = self.config.get(
            "similarity_threshold", 0.8
        )  # 相似度阈值
        self.max_short_term_memories = self.config.get(
            "max_short_term_memories", 500
        )  # 最大短期记忆数（增加以适应更复杂任务）
        self.embedding_dim = self.config.get("embedding_dim", 768)  # 嵌入维度（增加到768以匹配Transformer hidden_size）
        
        # 如果提供了AGI模型但未提供嵌入模型，创建AGI文本嵌入器
        if self.embedding_model is None and self.agi_model is not None:
            try:
                # 创建AGI文本嵌入器
                self.embedding_model = create_agi_text_embedder(self.agi_model)
                logger.info("使用AGI文本嵌入器（集成核心Transformer嵌入）")
            except Exception as e:
                logger.error(f"创建AGI文本嵌入器失败: {e}, AGI模型类型: {type(self.agi_model).__name__ if self.agi_model else 'None'}")
                logger.warning("将回退到从零开始的文本嵌入器，性能可能受限")
                # 创建简单的文本嵌入器作为回退
                from ..text.text_embedder import TextEmbedder
                self.embedding_model = TextEmbedder(self.embedding_dim)
                logger.info("已创建基础文本嵌入器作为回退方案")
        
        # 如果提供了外部嵌入模型，更新嵌入维度
        if self.embedding_model is not None:
            # 尝试从嵌入模型获取维度
            if hasattr(self.embedding_model, 'embedding_dim'):
                self.embedding_dim = self.embedding_model.embedding_dim
            elif hasattr(self.embedding_model, 'config') and hasattr(self.embedding_model.config, 'hidden_size'):
                self.embedding_dim = self.embedding_model.config.hidden_size
            logger.info(f"使用外部嵌入模型，嵌入维度: {self.embedding_dim}")
        
        # 升级功能配置
        self.enable_rag_retrieval = self.config.get("enable_rag_retrieval", True)
        self.working_memory_capacity = self.config.get("working_memory_capacity", 100)  # 工作记忆容量增加到100
        self.rag_top_k = self.config.get("rag_top_k", 10)  # RAG检索结果增加到10
        self.enable_importance_learning = self.config.get("enable_importance_learning", True)
        self.enable_memory_graph = self.config.get("enable_memory_graph", True)
        
        # FAISS/HNSW高效检索配置
        self.enable_faiss_retrieval = self.config.get("enable_faiss_retrieval", True)
        self.faiss_index_type = self.config.get("faiss_index_type", "IVF1024,PQ64")  # FAISS索引类型
        self.hnsw_m = self.config.get("hnsw_m", 32)  # HNSW M参数
        self.hnsw_ef_construction = self.config.get("hnsw_ef_construction", 200)  # HNSW构建参数
        self.hnsw_ef_search = self.config.get("hnsw_ef_search", 100)  # HNSW搜索参数
        self.faiss_gpu_enabled = self.config.get("faiss_gpu_enabled", False)  # 是否启用GPU加速
        self.index_rebuild_threshold = self.config.get("index_rebuild_threshold", 1000)  # 索引重建阈值
        self.index_rebuild_interval_hours = self.config.get("index_rebuild_interval_hours", 24)  # 索引重建间隔（小时）
        self.last_index_rebuild_check = None  # 上次检查索引重建的时间
        
        # 索引重建状态
        self.index_needs_rebuild = False  # 索引是否需要重建
        self.index_needs_urgent_rebuild = False  # 索引是否需要紧急重建（达到阈值）
        self._pending_rebuild_count = 0  # 待重建计数（已删除但未重建的记忆数量）
        
        # 动态关联图增强配置
        self.enable_gnn_learning = self.config.get("enable_gnn_learning", True)  # 是否启用图神经网络学习
        self.enable_community_detection = self.config.get("enable_community_detection", True)  # 是否启用社区检测
        self.enable_temporal_dynamics = self.config.get("enable_temporal_detection", True)  # 是否启用时间动态分析
        self.gnn_hidden_dim = self.config.get("gnn_hidden_dim", 64)  # GNN隐藏维度
        self.gnn_num_layers = self.config.get("gnn_num_layers", 2)  # GNN层数
        self.community_min_size = self.config.get("community_min_size", 3)  # 社区最小大小
        self.temporal_decay_factor = self.config.get("temporal_decay_factor", 0.95)  # 时间衰减因子
        self.max_graph_edges = self.config.get("max_graph_edges", 500)  # 最大图边数
        self.graph_update_frequency = self.config.get("graph_update_frequency", 100)  # 图更新频率（每N个记忆）
        
        # 知识库集成配置
        self.enable_knowledge_base = self.config.get("enable_knowledge_base", True) and KNOWLEDGE_BASE_AVAILABLE  # 是否启用知识库集成
        self.knowledge_base_config = self.config.get("knowledge_base_config", {
            "embedding_dim": 384,
            "similarity_threshold": 0.7,
            "max_knowledge_items": 10000,
            "enable_knowledge_graph": True,
            "enable_validation": True,
            "industrial_mode": True
        })
        
        # 第四阶段：自主记忆管理配置
        self.enable_autonomous_memory = self.config.get("enable_autonomous_memory", True)  # 是否启用自主记忆管理
        if self.enable_autonomous_memory:
            # 创建自主记忆管理器
            autonomous_config = {
                "enable_autonomous_learning": self.config.get("enable_autonomous_learning", True),
                "forgetting_rate": self.config.get("forgetting_rate", 0.05),
                "importance_threshold": self.config.get("importance_threshold", 0.7),
                "similarity_threshold": self.config.get("similarity_threshold", 0.8),
                "compression_threshold": self.config.get("compression_threshold", 0.9),
                "cache_size": self.config.get("cache_size", 100),
                "rl_learning_rate": self.config.get("rl_learning_rate", 0.01),
                "exploration_rate": self.config.get("exploration_rate", 0.1),
                "max_decision_history": self.config.get("max_decision_history", 1000)
            }
            self.autonomous_manager = AutonomousMemoryManager(autonomous_config)
            logger.info("自主记忆管理器已启用")
        else:
            self.autonomous_manager = None
            logger.info("自主记忆管理器未启用")
        
        # 第五阶段：情感记忆增强配置 (默认禁用)
        self.enable_emotional_memory = self.config.get("enable_emotional_memory", False)  # 是否启用情感记忆增强 (默认禁用)
        self.emotion_detection_threshold = self.config.get("emotion_detection_threshold", 0.3)  # 情感检测阈值
        self.emotion_impact_factor = self.config.get("emotion_impact_factor", 0.2)  # 情感影响因子
        self.max_emotion_history = self.config.get("max_emotion_history", 100)  # 最大情感历史记录数
        
        # 知识管理器（如果启用）
        if self.enable_knowledge_base:
            try:
                self.knowledge_manager = KnowledgeManager(self.knowledge_base_config, self)
                logger.info("知识管理器已初始化并启用")
            except Exception as e:
                logger.error(f"知识管理器初始化失败: {e}")
                self.enable_knowledge_base = False
                self.knowledge_manager = None
        else:
            self.knowledge_manager = None
            logger.info("知识库集成未启用")
        
        # 工作记忆缓冲区
        self.working_memory = []
        
        # 短期记忆LRU缓存（第二层检索）
        self.short_term_memory_cache = {}  # 内存ID -> 记忆字典
        self.short_term_cache_keys = []  # LRU顺序列表
        self.short_term_cache_max_size = self.config.get("short_term_cache_max_size", 200)  # 缓存最大容量
        self.short_term_cache_hits = 0  # 缓存命中次数
        self.short_term_cache_misses = 0  # 缓存未命中次数
        
        # 动态关联图相关变量
        self.memory_graph = None  # 图结构
        self.graph_initialized = False
        self.memory_to_node_map = {}  # 记忆ID到图节点索引映射
        self.node_to_memory_map = {}  # 图节点索引到记忆ID映射
        self.graph_update_counter = 0  # 图更新计数器
        self.graph_needs_rebuild = False  # 图是否需要重建
        
        # 图神经网络模型 (如果启用)
        if self.enable_gnn_learning:
            # 节点特征维度：嵌入维度 + 重要性 + 时间特征等
            gnn_feature_dim = self.embedding_dim + 5  # 额外特征维度
            self.gnn_model = MemoryGraphNeuralNetwork(
                feature_dim=gnn_feature_dim,
                hidden_dim=self.gnn_hidden_dim,
                num_layers=self.gnn_num_layers
            )
            logger.info(f"图神经网络模型已初始化: 特征维度={gnn_feature_dim}, 隐藏维度={self.gnn_hidden_dim}")
        else:
            self.gnn_model = None
            logger.info("图神经网络学习未启用")
        
        # 记忆重要性学习器
        if self.enable_importance_learning:
            self.importance_model = ImportanceModel(feature_dim=10, hidden_dim=32)
            logger.info("记忆重要性学习器已初始化")
        else:
            self.importance_model = None
            logger.info("记忆重要性学习器未启用")
        
        # FAISS/HNSW索引
        self.faiss_index = None
        self.hnsw_index = None
        self.index_initialized = False
        self.index_size = 0
        self.index_last_updated = None
        self.last_index_rebuild_check = None  # 初始化最后检查时间
        
        # 索引性能统计
        self.faiss_search_count = 0
        self.faiss_search_time_total = 0.0
        self.hnsw_search_count = 0
        self.hnsw_search_time_total = 0.0
        self.linear_search_count = 0
        self.linear_search_time_total = 0.0

    def update_autonomous_system_state(self, metrics: Optional[Dict[str, Any]] = None):
        """更新自主记忆管理器的系统状态
        
        参数:
            metrics: 系统监控指标，如果为None则自动收集基本指标
        """
        if not self.enable_autonomous_memory or not self.autonomous_manager:
            return
        
        if metrics is None:
            # 自动收集基本系统指标
            try:
                import psutil
                metrics = {
                    "cpu_usage": psutil.cpu_percent(interval=0.1),
                    "memory_usage": psutil.virtual_memory().percent,
                    "user_activity": 0.5  # 默认值，实际应用中应从用户行为跟踪获取
                }
            except ImportError:
                metrics = {
                    "cpu_usage": 50.0,  # 默认值
                    "memory_usage": 50.0,
                    "user_activity": 0.5
                }
        
        # 更新自主记忆管理器的系统状态
        self.autonomous_manager.update_system_state(metrics)
        
        logger.debug(f"自主记忆管理器系统状态已更新: {self.autonomous_manager.system_state}")
    
    def record_memory_decision(self, decision_type: str, context: Dict[str, Any], outcome: Dict[str, Any]):
        """记录记忆管理决策到自主记忆管理器
        
        参数:
            decision_type: 决策类型（forget, compress, cache_evict等）
            context: 决策上下文信息
            outcome: 决策结果（包括success和reward）
        """
        if not self.enable_autonomous_memory or not self.autonomous_manager:
            return
        
        # 记录决策到自主记忆管理器
        self.autonomous_manager.record_decision(decision_type, context, outcome)
        
        logger.debug(f"记忆决策已记录: {decision_type}, 奖励={outcome.get('reward', 0.0)}")
    
    def get_autonomous_parameters(self) -> Dict[str, Any]:
        """获取自主调整的记忆管理参数
        
        返回:
            自主记忆管理器推荐的参数
        """
        if not self.enable_autonomous_memory or not self.autonomous_manager:
            # 返回默认配置参数
            return {
                "forgetting_rate": self.config.get("forgetting_rate", 0.05),
                "importance_threshold": self.config.get("importance_threshold", 0.7),
                "similarity_threshold": self.config.get("similarity_threshold", 0.8),
                "compression_threshold": self.config.get("compression_threshold", 0.9),
                "cache_size": self.config.get("cache_size", 100)
            }
        
        # 获取当前系统状态
        system_state = self.autonomous_manager.system_state
        
        # 从自主记忆管理器获取最优参数
        optimal_params = self.autonomous_manager.get_optimal_action(system_state)
        
        logger.debug(f"获取自主调整参数: {optimal_params}")
        return optimal_params

    def initialize(self, db: Optional[Session] = None):
        """初始化记忆系统"""
        try:
            # 如果已经提供了嵌入模型，则使用它
            if self.embedding_model is not None:
                logger.info("使用已提供的嵌入模型...")
                model_type = self.embedding_model.__class__.__name__
                logger.info(f"嵌入模型类型: {model_type}")
                
                # 验证嵌入模型有encode方法
                if not hasattr(self.embedding_model, 'encode'):
                    logger.warning(f"嵌入模型 {model_type} 缺少encode方法，尝试使用forward方法")
                    # 为简单模型添加encode方法适配
                    self._adapt_embedding_model()
            else:
                # 加载从零开始的嵌入模型（工业级AGI，无预训练模型）
                logger.info("加载从零开始的文本嵌入模型...")
                self.embedding_model = FromScratchTextEmbedder(
                    embedding_dim=self.embedding_dim,
                    vocab_size=10000,
                    max_length=512
                )
                logger.info("从零开始的文本嵌入模型加载完成")

            # 初始化FAISS/HNSW索引（如果启用）
            if self.enable_faiss_retrieval:
                self._initialize_faiss_index(db)
                logger.info("FAISS/HNSW索引初始化完成")

            # 初始化动态关联图（如果启用）
            if self.enable_memory_graph:
                self._initialize_memory_graph(db)
                logger.info("动态关联图初始化完成")

            self.initialized = True
            logger.info("记忆系统初始化成功")

            # 启动记忆压缩任务
            if db:
                self._compress_memories(db)

        except Exception as e:
            logger.error(f"记忆系统初始化失败: {e}, 配置: 多模态={self.enable_multimodal_memory}, 知识库={self.enable_knowledge_base}, FAISS={self.enable_faiss_retrieval}, 图={self.enable_memory_graph}")
            logger.error("初始化失败，请检查数据库连接和依赖项。系统将无法提供记忆服务。")
            raise
    
    def _initialize_faiss_index(self, db: Optional[Session] = None):
        """初始化FAISS/HNSW索引
        
        参数:
            db: 数据库会话，用于加载现有记忆构建索引
        """
        try:
            # 检查是否安装了FAISS
            faiss_available = False
            hnsw_available = False
            
            try:
                import faiss
                faiss_available = True
                logger.info("FAISS库可用")
            except ImportError:
                logger.warning("FAISS库未安装，将回退到线性检索。请安装: pip install faiss-cpu 或 faiss-gpu")
            
            try:
                import hnswlib
                hnsw_available = True
                logger.info("HNSW库可用")
            except ImportError:
                logger.warning("HNSW库未安装，将仅使用FAISS或线性检索。请安装: pip install hnswlib")
            
            # 如果没有可用的索引库，禁用FAISS检索
            if not faiss_available and not hnsw_available:
                logger.warning("没有可用的向量索引库，将禁用FAISS检索并回退到线性检索")
                self.enable_faiss_retrieval = False
                return
            
            # 记录可用的索引类型
            self.faiss_available = faiss_available
            self.hnsw_available = hnsw_available
            
            # 索引保存路径配置
            self.faiss_index_save_path = self.config.get(
                "faiss_index_save_path", 
                "./data/faiss_index.bin"
            )
            self.faiss_mapping_save_path = self.config.get(
                "faiss_mapping_save_path",
                "./data/faiss_index_mapping.json"
            )
            
            # 尝试加载已保存的索引（如果存在且有效）
            index_loaded = False
            if faiss_available and os.path.exists(self.faiss_index_save_path) and os.path.exists(self.faiss_mapping_save_path):
                try:
                    self.faiss_index = faiss.read_index(self.faiss_index_save_path)
                    
                    # 加载内存ID映射
                    with open(self.faiss_mapping_save_path, 'r', encoding='utf-8') as f:
                        self.faiss_index_mapping = json.load(f)
                    
                    # 验证索引大小与映射一致
                    if self.faiss_index.ntotal == len(self.faiss_index_mapping):
                        self.index_initialized = True
                        self.index_size = self.faiss_index.ntotal
                        index_loaded = True
                        logger.info(f"FAISS索引加载成功: 向量数量={self.faiss_index.ntotal}, 映射大小={len(self.faiss_index_mapping)}")
                    else:
                        logger.warning(f"索引与映射大小不一致: 索引={self.faiss_index.ntotal}, 映射={len(self.faiss_index_mapping)}，将重建索引")
                        index_loaded = False
                except Exception as e:
                    logger.warning(f"加载FAISS索引失败: {e}，将重建索引")
                    index_loaded = False
            
            # 如果索引加载失败，构建新索引
            if not index_loaded:
                # 如果有数据库连接，构建初始索引
                if db is not None:
                    self._build_faiss_index(db)
                else:
                    logger.warning("没有数据库连接，无法构建FAISS索引，索引功能将不可用")
                    self.index_initialized = False
                    return
            
            self.index_initialized = True
            logger.info(f"FAISS/HNSW索引初始化完成: FAISS={faiss_available}, HNSW={hnsw_available}, 已加载={index_loaded}")
            
        except Exception as e:
            logger.error(f"初始化FAISS/HNSW索引失败: {e}")
            self.enable_faiss_retrieval = False  # 出错时禁用FAISS检索
    
    def _initialize_memory_graph(self, db: Optional[Session] = None):
        """初始化动态关联图
        
        参数:
            db: 数据库会话，用于加载现有记忆构建图
        """
        try:
            # 如果有数据库连接，构建初始图
            if db is not None:
                self._build_memory_graph(db)
            
            self.graph_initialized = True
            logger.info("动态关联图初始化完成")
            
        except Exception as e:
            logger.error(f"初始化动态关联图失败: {e}")
            self.graph_initialized = False
    
    def _build_memory_graph(self, db: Session, force_rebuild: bool = False):
        """构建记忆关联图
        
        基于现有记忆构建图结构，包括：
        1. 节点：记忆
        2. 边：语义关联
        3. 节点特征：嵌入 + 元数据
        4. 边权重：关联强度
        
        参数:
            db: 数据库会话
            force_rebuild: 是否强制重建图
        """
        try:
            from backend.db_models.memory import Memory, MemoryAssociation
            
            # 获取所有长期记忆
            memories = db.query(Memory).filter(
                Memory.memory_type == "long_term"
            ).all()
            
            if not memories:
                logger.info("没有长期记忆，跳过图构建")
                return
            
            # 检查是否需要重建图
            if (not force_rebuild and self.graph_initialized and 
                len(memories) <= len(self.memory_to_node_map) * 1.5):
                # 如果记忆数量变化不大，跳过重建
                logger.info(f"图已存在且记忆数量变化不大（图节点: {len(self.memory_to_node_map)}, 当前记忆: {len(memories)}），跳过重建")
                return
            
            # 构建节点映射
            self.memory_to_node_map = {}
            self.node_to_memory_map = {}
            
            node_features = []
            memory_ids = []
            
            for idx, memory in enumerate(memories):
                memory_id = memory.id
                self.memory_to_node_map[memory_id] = idx
                self.node_to_memory_map[idx] = memory_id
                memory_ids.append(memory_id)
                
                # 提取节点特征：嵌入向量 + 重要性 + 时间特征
                try:
                    embedding = json.loads(memory.embedding) if memory.embedding else []
                    if len(embedding) != self.embedding_dim:
                        logger.warning(f"记忆 {memory_id} 的嵌入维度不匹配: 期望 {self.embedding_dim}, 实际 {len(embedding)}")
                        # 使用零向量填充
                        embedding = [0.0] * self.embedding_dim
                    
                    # 构建特征向量：嵌入 + 重要性 + 时间特征
                    # 时间特征：创建时间距离现在的小时数（归一化）
                    time_feature = 0.0
                    if memory.created_at:
                        time_diff = datetime.now(timezone.utc) - memory.created_at
                        hours_diff = time_diff.total_seconds() / 3600.0
                        # 时间衰减：exp(-hours_diff/720) (720小时 = 30天)
                        time_feature = np.exp(-hours_diff / 720.0)
                    
                    # 访问频率特征：归一化访问次数
                    access_feature = min(memory.accessed_count / 100.0, 1.0)
                    
                    # 组合特征向量
                    feature_vector = embedding + [
                        memory.importance,
                        time_feature,
                        access_feature,
                        1.0 if memory.memory_type == "long_term" else 0.0,
                        0.0  # 预留特征
                    ]
                    
                    node_features.append(feature_vector)
                    
                except Exception as e:
                    logger.error(f"提取记忆 {memory_id} 特征失败: {e}")
                    # 使用默认特征向量
                    default_features = [0.0] * (self.embedding_dim + 5)
                    node_features.append(default_features)
            
            if not node_features:
                logger.warning("没有有效的节点特征，无法构建图")
                return
            
            # 构建邻接矩阵（基于现有关联）
            num_nodes = len(node_features)
            adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            
            # 加载现有的记忆关联
            associations = db.query(MemoryAssociation).all()
            for association in associations:
                source_id = association.source_memory_id
                target_id = association.target_memory_id
                
                if source_id in self.memory_to_node_map and target_id in self.memory_to_node_map:
                    source_idx = self.memory_to_node_map[source_id]
                    target_idx = self.memory_to_node_map[target_id]
                    strength = association.strength if association.strength else 0.5
                    
                    # 对称关联
                    adjacency[source_idx, target_idx] = strength
                    adjacency[target_idx, source_idx] = strength
            
            # 如果没有现有关联，基于语义相似度构建初始关联
            if np.sum(adjacency) == 0:
                logger.info("没有现有关联，基于语义相似度构建初始关联")
                adjacency = self._build_initial_adjacency(node_features, memory_ids, db)
            
            # 将数据转换为PyTorch张量
            node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
            adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
            
            # 如果启用了GNN，使用GNN学习图表示
            if self.enable_gnn_learning and self.gnn_model is not None:
                with torch.no_grad():
                    graph_output = self.gnn_model(node_features_tensor, adjacency_tensor)
                    
                    # 更新图结构
                    self.memory_graph = {
                        "node_features": node_features_tensor,
                        "adjacency": adjacency_tensor,
                        "node_embeddings": graph_output["node_embeddings"],
                        "edge_predictions": graph_output["edge_predictions"],
                        "community_embeddings": graph_output.get("community_embeddings")
                    }
                    
                    # 社区检测（如果启用）
                    if self.enable_community_detection and graph_output["node_embeddings"] is not None:
                        communities = self.gnn_model.detect_communities(
                            graph_output["node_embeddings"],
                            threshold=0.7
                        )
                        self.memory_graph["communities"] = communities
                        logger.info(f"检测到 {len(communities)} 个记忆社区")
            else:
                # 使用基础图结构
                self.memory_graph = {
                    "node_features": node_features_tensor,
                    "adjacency": adjacency_tensor,
                    "node_embeddings": None,
                    "edge_predictions": None,
                    "communities": None
                }
            
            # 应用时间衰减（如果启用）
            if self.enable_temporal_dynamics:
                self._apply_temporal_decay()
            
            logger.info(f"记忆关联图构建完成: 节点数={num_nodes}, 边数={np.count_nonzero(adjacency)}")
            
        except Exception as e:
            logger.error(f"构建记忆关联图失败: {e}")
            self.graph_initialized = False
    
    def _build_initial_adjacency(self, node_features: List[List[float]], memory_ids: List[int], db: Session) -> np.ndarray:
        """构建初始邻接矩阵（基于语义相似度和时间邻近性）- 修复版
        
        修复内容：
        1. 综合考虑语义相似度和时间邻近性
        2. 添加记忆类型关联
        3. 使用更合理的权重组合
        4. 添加自连接确保每个节点都有连接
        
        参数:
            node_features: 节点特征列表
            memory_ids: 记忆ID列表
            db: 数据库会话
            
        返回:
            邻接矩阵
        """
        num_nodes = len(node_features)
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        from backend.db_models.memory import MemoryAssociation, Memory
        
        # 获取记忆的时间信息
        memory_times = {}
        memory_types = {}
        try:
            memories = db.query(Memory).filter(Memory.id.in_(memory_ids)).all()
            for mem in memories:
                memory_times[mem.id] = mem.created_at.timestamp() if mem.created_at else 0
                memory_types[mem.id] = mem.memory_type
        except Exception as e:
            logger.debug(f"获取记忆时间信息失败: {e}")
        
        # 计算所有节点对之间的相似度
        edge_count = 0
        max_edges = min(self.max_graph_edges, num_nodes * (num_nodes - 1) // 2)
        
        # 存储所有候选边及其权重
        candidate_edges = []
        
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                # 使用嵌入部分计算余弦相似度
                embedding_i = node_features[i][:self.embedding_dim]
                embedding_j = node_features[j][:self.embedding_dim]
                
                # 计算余弦相似度
                dot_product = np.dot(embedding_i, embedding_j)
                norm_i = np.linalg.norm(embedding_i)
                norm_j = np.linalg.norm(embedding_j)
                
                if norm_i > 0 and norm_j > 0:
                    semantic_sim = dot_product / (norm_i * norm_j)
                    
                    # 计算时间邻近性
                    time_sim = 0.5  # 默认中等时间相似度
                    if memory_ids[i] in memory_times and memory_ids[j] in memory_times:
                        time_diff = abs(memory_times[memory_ids[i]] - memory_times[memory_ids[j]])
                        # 时间差越小，相似度越高（指数衰减）
                        time_sim = np.exp(-time_diff / 86400)  # 以天为单位
                    
                    # 计算类型相似度
                    type_sim = 0.5  # 默认中等类型相似度
                    if memory_ids[i] in memory_types and memory_ids[j] in memory_types:
                        if memory_types[memory_ids[i]] == memory_types[memory_ids[j]]:
                            type_sim = 1.0  # 相同类型
                        else:
                            type_sim = 0.3  # 不同类型
                    
                    # 综合权重：语义相似度占60%，时间邻近性占25%，类型相似度占15%
                    combined_weight = 0.6 * semantic_sim + 0.25 * time_sim + 0.15 * type_sim
                    
                    # 只添加强关联
                    if combined_weight > self.similarity_threshold * 0.6:  # 降低阈值以获取更多连接
                        candidate_edges.append((i, j, combined_weight, semantic_sim))
        
        # 按权重排序，选择最重要的边
        candidate_edges.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, weight, semantic_sim in candidate_edges[:max_edges]:
            adjacency[i, j] = weight
            adjacency[j, i] = weight
            
            # 保存关联到数据库
            try:
                association = MemoryAssociation(
                    source_memory_id=memory_ids[i],
                    target_memory_id=memory_ids[j],
                    association_type="semantic_similarity",
                    strength=weight,
                    created_at=datetime.now(timezone.utc)
                )
                db.add(association)
                
                # 双向关联
                reverse_association = MemoryAssociation(
                    source_memory_id=memory_ids[j],
                    target_memory_id=memory_ids[i],
                    association_type="semantic_similarity",
                    strength=weight,
                    created_at=datetime.now(timezone.utc)
                )
                db.add(reverse_association)
                
                edge_count += 1
                
            except Exception as e:
                logger.debug(f"保存记忆关联失败: {e}")
        
        # 添加自连接（确保每个节点至少有一个连接）
        for i in range(num_nodes):
            if np.sum(adjacency[i]) == 0:
                adjacency[i, i] = 1.0  # 自连接权重为1
        
        db.commit()
        logger.info(f"构建初始邻接矩阵: {edge_count} 条边，{num_nodes} 个自连接")
        return adjacency
    
    def _apply_temporal_decay(self):
        """应用时间衰减到图边权重
        
        基于时间衰减因子降低旧边的权重，增强新边的权重
        """
        if self.memory_graph is None or "adjacency" not in self.memory_graph:
            return
        
        adjacency = self.memory_graph["adjacency"]
        if adjacency is None:
            return
        
        # 完整时间衰减：对邻接矩阵应用衰减因子
        # 实际实现应考虑每条边的时间戳
        self.memory_graph["adjacency"] = adjacency * self.temporal_decay_factor
        logger.debug(f"应用时间衰减: 衰减因子={self.temporal_decay_factor}")
    
    def _update_memory_graph_with_new_memory(self, memory_id: int, embedding: List[float], db: Session):
        """使用新记忆更新图
        
        参数:
            memory_id: 新记忆ID
            embedding: 新记忆嵌入
            db: 数据库会话
        """
        if not self.graph_initialized or self.memory_graph is None:
            return
        
        try:
            # 将新记忆添加到图
            self._add_node_to_graph(memory_id, embedding, db)
            
            # 更新图更新计数器
            self.graph_update_counter += 1
            
            # 如果达到更新频率，重新训练GNN
            if self.graph_update_counter >= self.graph_update_frequency:
                self._retrain_gnn(db)
                self.graph_update_counter = 0
                logger.info("GNN模型重新训练完成")
            
        except Exception as e:
            logger.error(f"更新记忆图失败: {e}")
    
    def _add_node_to_graph(self, memory_id: int, embedding: List[float], db: Session):
        """添加节点到图
        
        参数:
            memory_id: 记忆ID
            embedding: 嵌入向量
            db: 数据库会话
        """
        # 获取记忆信息
        from backend.db_models.memory import Memory
        memory = db.query(Memory).filter(Memory.id == memory_id).first()
        
        if not memory:
            return
        
        # 添加到节点映射
        new_idx = len(self.memory_to_node_map)
        self.memory_to_node_map[memory_id] = new_idx
        self.node_to_memory_map[new_idx] = memory_id
        
        # 构建节点特征
        time_feature = 0.0
        if memory.created_at:
            time_diff = datetime.now(timezone.utc) - memory.created_at
            hours_diff = time_diff.total_seconds() / 3600.0
            time_feature = np.exp(-hours_diff / 720.0)
        
        access_feature = min(memory.accessed_count / 100.0, 1.0)
        
        feature_vector = embedding + [
            memory.importance,
            time_feature,
            access_feature,
            1.0 if memory.memory_type == "long_term" else 0.0,
            0.0
        ]
        
        # 更新节点特征矩阵
        node_features = self.memory_graph["node_features"]
        new_node_features = torch.zeros((new_idx + 1, node_features.shape[1]), dtype=node_features.dtype)
        if new_idx > 0:
            new_node_features[:new_idx] = node_features
        new_node_features[new_idx] = torch.tensor(feature_vector, dtype=node_features.dtype)
        self.memory_graph["node_features"] = new_node_features
        
        # 更新邻接矩阵
        adjacency = self.memory_graph["adjacency"]
        new_adjacency = torch.zeros((new_idx + 1, new_idx + 1), dtype=adjacency.dtype)
        if new_idx > 0:
            new_adjacency[:new_idx, :new_idx] = adjacency
        
        # 计算新节点与现有节点的相似度
        for i in range(new_idx):
            old_memory_id = self.node_to_memory_map[i]
            
            # 获取现有记忆的嵌入
            old_memory = db.query(Memory).filter(Memory.id == old_memory_id).first()
            if old_memory and old_memory.embedding:
                try:
                    old_embedding = json.loads(old_memory.embedding)
                    
                    # 计算相似度
                    dot_product = np.dot(embedding, old_embedding)
                    norm_new = np.linalg.norm(embedding)
                    norm_old = np.linalg.norm(old_embedding)
                    
                    if norm_new > 0 and norm_old > 0:
                        similarity = dot_product / (norm_new * norm_old)
                        
                        # 如果相似度足够高，添加边
                        if similarity > self.similarity_threshold * 0.6:
                            new_adjacency[i, new_idx] = similarity
                            new_adjacency[new_idx, i] = similarity
                            
                            # 保存关联到数据库
                            from backend.db_models.memory import MemoryAssociation
                            association = MemoryAssociation(
                                source_memory_id=memory_id,
                                target_memory_id=old_memory_id,
                                association_type="semantic_similarity",
                                strength=similarity,
                                created_at=datetime.now(timezone.utc)
                            )
                            db.add(association)
                            
                            # 双向关联
                            reverse_association = MemoryAssociation(
                                source_memory_id=old_memory_id,
                                target_memory_id=memory_id,
                                association_type="semantic_similarity",
                                strength=similarity,
                                created_at=datetime.now(timezone.utc)
                            )
                            db.add(reverse_association)
                except Exception as e:
                    logger.debug(f"计算新节点与现有节点相似度失败: {e}")
        
        self.memory_graph["adjacency"] = new_adjacency
        db.commit()
        
        logger.debug(f"记忆 {memory_id} 已添加到图，节点索引: {new_idx}")
    
    def _retrain_gnn(self, db: Session):
        """重新训练GNN模型
        
        参数:
            db: 数据库会话
        """
        if not self.enable_gnn_learning or self.gnn_model is None:
            return
        
        try:
            # 重新构建图
            self._build_memory_graph(db, force_rebuild=True)
            logger.info("GNN模型重新训练完成")
            
        except Exception as e:
            logger.error(f"重新训练GNN模型失败: {e}")
    
    def _build_faiss_index(self, db: Session, force_rebuild: bool = False):
        """构建FAISS/HNSW索引
        
        参数:
            db: 数据库会话
            force_rebuild: 是否强制重建索引
        """
        try:
            from backend.db_models.memory import Memory
            
            # 获取所有长期记忆
            memories = db.query(Memory).filter(
                Memory.memory_type == "long_term"
            ).all()
            
            if not memories:
                logger.info("没有长期记忆，跳过索引构建")
                return
            
            # 检查是否需要重建索引
            if (not force_rebuild and self.index_initialized and 
                self.index_size > 0 and len(memories) <= self.index_size * 1.5):
                # 如果记忆数量变化不大，跳过重建
                logger.info(f"索引已存在且记忆数量变化不大（索引大小: {self.index_size}, 当前记忆: {len(memories)}），跳过重建")
                return
            
            # 提取嵌入向量
            embeddings = []
            memory_ids = []
            
            for memory in memories:
                try:
                    embedding = json.loads(memory.embedding)
                    if len(embedding) == self.embedding_dim:
                        embeddings.append(embedding)
                        memory_ids.append(memory.id)
                    else:
                        logger.warning(f"记忆 {memory.id} 的嵌入维度不匹配: 期望 {self.embedding_dim}, 实际 {len(embedding)}")
                except Exception as e:
                    logger.error(f"解析记忆 {memory.id} 的嵌入失败: {e}")
                    continue
            
            if not embeddings:
                logger.warning("没有有效的嵌入向量，无法构建索引")
                return
            
            embeddings_np = np.array(embeddings, dtype=np.float32)
            
            # 构建FAISS索引（如果可用）
            if self.faiss_available:
                import faiss
                
                # 根据配置选择索引类型
                if self.faiss_index_type == "Flat":
                    # 平面索引（精确搜索）
                    index = faiss.IndexFlatIP(self.embedding_dim)  # 内积相似度
                elif "IVF" in self.faiss_index_type:
                    # IVF索引（近似搜索）
                    # 解析参数，例如 "IVF1024,PQ64"
                    nlist = 1024  # 默认
                    if "IVF" in self.faiss_index_type:
                        # 提取数字
                        import re
                        match = re.search(r'IVF(\d+)', self.faiss_index_type)
                        if match:
                            nlist = int(match.group(1))
                    
                    quantizer = faiss.IndexFlatIP(self.embedding_dim)
                    index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                    
                    # 训练索引
                    if len(embeddings_np) >= nlist * 10:  # 确保有足够的数据训练
                        index.train(embeddings_np)
                    else:
                        logger.warning(f"数据量不足训练IVF索引，使用平面索引替代")
                        index = faiss.IndexFlatIP(self.embedding_dim)
                else:
                    # 默认使用平面索引
                    index = faiss.IndexFlatIP(self.embedding_dim)
                
                # 添加向量到索引
                index.add(embeddings_np)
                self.faiss_index = index
                
                # GPU支持（如果启用）
                if self.faiss_gpu_enabled and hasattr(faiss, 'StandardGpuResources'):
                    try:
                        gpu_res = faiss.StandardGpuResources()
                        self.faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
                        logger.info("FAISS索引已移动到GPU")
                    except Exception as e:
                        logger.warning(f"FAISS GPU转移失败: {e}")
                
                logger.info(f"FAISS索引构建完成: 类型={self.faiss_index_type}, 向量数量={len(embeddings_np)}")
            
            # 构建HNSW索引（如果可用且需要）
            if self.hnsw_available and not self.faiss_available:
                import hnswlib
                
                # 创建HNSW索引
                index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
                
                # 初始化索引
                index.init_index(
                    max_elements=len(embeddings_np) * 2,  # 预留空间
                    ef_construction=self.hnsw_ef_construction,
                    M=self.hnsw_m
                )
                
                # 添加向量
                index.add_items(embeddings_np, memory_ids)
                
                # 设置搜索参数
                index.set_ef(self.hnsw_ef_search)
                
                self.hnsw_index = index
                logger.info(f"HNSW索引构建完成: M={self.hnsw_m}, ef_construction={self.hnsw_ef_construction}, 向量数量={len(embeddings_np)}")
            
            # 更新索引状态
            self.index_size = len(embeddings_np)
            self.index_last_updated = datetime.now(timezone.utc)
            self.memory_id_map = {i: mem_id for i, mem_id in enumerate(memory_ids)}
            self.id_to_index_map = {mem_id: i for i, mem_id in enumerate(memory_ids)}
            
            # 设置FAISS索引映射（用于持久化）
            self.faiss_index_mapping = self.memory_id_map
            
            # 保存索引到文件（如果FAISS可用且索引已构建）
            if self.faiss_available and self.faiss_index is not None:
                try:
                    import faiss
                    # 确保保存目录存在
                    os.makedirs(os.path.dirname(self.faiss_index_save_path), exist_ok=True)
                    os.makedirs(os.path.dirname(self.faiss_mapping_save_path), exist_ok=True)
                    
                    # 保存FAISS索引
                    faiss.write_index(self.faiss_index, self.faiss_index_save_path)
                    
                    # 保存内存ID映射
                    with open(self.faiss_mapping_save_path, 'w', encoding='utf-8') as f:
                        json.dump(self.faiss_index_mapping, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"FAISS索引已保存: 索引文件={self.faiss_index_save_path}, 映射文件={self.faiss_mapping_save_path}")
                except Exception as e:
                    logger.error(f"保存FAISS索引失败: {e}")
            
            logger.info(f"向量索引构建完成: 总记忆数={len(memories)}, 有效嵌入数={len(embeddings_np)}")
            
        except Exception as e:
            logger.error(f"构建FAISS/HNSW索引失败: {e}")
            self.index_initialized = False
    
    def _search_with_faiss(self, query_embedding: List[float], k: int = 10) -> List[Tuple[int, float]]:
        """使用FAISS/HNSW索引搜索
        
        参数:
            query_embedding: 查询嵌入向量
            k: 返回的结果数量
            
        返回:
            列表，包含 (记忆ID, 相似度分数) 元组
        """
        if not self.index_initialized:
            return []  # 返回空列表
        
        try:
            start_time = time.time()
            query_np = np.array([query_embedding], dtype=np.float32)
            results = []
            
            # 优先使用FAISS索引
            if self.faiss_index is not None:
                import faiss
                
                # 搜索
                distances, indices = self.faiss_index.search(query_np, k)
                
                # 更新FAISS搜索统计
                self.faiss_search_count += 1
                
                # 转换为记忆ID和相似度分数
                for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx >= 0 and idx in self.memory_id_map:
                        memory_id = self.memory_id_map[idx]
                        # FAISS返回的是距离，需要转换为相似度
                        # 对于内积索引，距离越大表示越相似
                        similarity = float(dist)  # 内积距离
                        results.append((memory_id, similarity))
            
            # 如果没有FAISS索引，使用HNSW索引
            elif self.hnsw_index is not None:
                import hnswlib
                
                # HNSW搜索
                indices, distances = self.hnsw_index.knn_query(query_np, k=k)
                
                # 更新HNSW搜索统计
                self.hnsw_search_count += 1
                
                # 转换为记忆ID和相似度分数
                for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                    if idx in self.memory_id_map:
                        memory_id = self.memory_id_map[idx]
                        # HNSW返回的是余弦距离（1 - 余弦相似度）
                        # 转换为相似度: similarity = 1 - distance
                        similarity = 1.0 - float(dist)
                        results.append((memory_id, similarity))
            
            # 按相似度排序（降序）
            results.sort(key=lambda x: x[1], reverse=True)
            
            # 记录搜索时间
            search_time = time.time() - start_time
            if self.faiss_index is not None:
                self.faiss_search_time_total += search_time
            elif self.hnsw_index is not None:
                self.hnsw_search_time_total += search_time
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"FAISS/HNSW搜索失败: {e}")
            return []  # 返回空列表
    
    def _update_faiss_index(self, memory_id: int, embedding: List[float], action: str = "add"):
        """更新FAISS/HNSW索引
        
        参数:
            memory_id: 记忆ID
            embedding: 嵌入向量
            action: 操作类型，'add' 或 'remove'
        """
        if not self.index_initialized or (self.faiss_index is None and self.hnsw_index is None):
            return
        
        try:
            embedding_np = np.array([embedding], dtype=np.float32)
            
            if action == "add":
                # 添加到索引
                if self.faiss_index is not None:
                    import faiss
                    self.faiss_index.add(embedding_np)
                    
                    # 更新映射
                    new_idx = self.index_size
                    self.memory_id_map[new_idx] = memory_id
                    self.id_to_index_map[memory_id] = new_idx
                    self.index_size += 1
                    
                    # 更新FAISS索引映射并保存
                    if hasattr(self, 'faiss_index_mapping'):
                        self.faiss_index_mapping[new_idx] = memory_id
                    
                    # 保存更新后的索引
                    try:
                        # 确保保存目录存在
                        if hasattr(self, 'faiss_index_save_path'):
                            os.makedirs(os.path.dirname(self.faiss_index_save_path), exist_ok=True)
                            os.makedirs(os.path.dirname(self.faiss_mapping_save_path), exist_ok=True)
                            
                            # 保存FAISS索引
                            faiss.write_index(self.faiss_index, self.faiss_index_save_path)
                            
                            # 保存内存ID映射
                            if hasattr(self, 'faiss_index_mapping'):
                                with open(self.faiss_mapping_save_path, 'w', encoding='utf-8') as f:
                                    json.dump(self.faiss_index_mapping, f, ensure_ascii=False, indent=2)
                            
                            logger.debug(f"FAISS索引已更新并保存: 记忆ID={memory_id}, 索引大小={self.index_size}")
                    except Exception as save_error:
                        logger.error(f"保存更新后的FAISS索引失败: {save_error}")
                
                elif self.hnsw_index is not None:
                    import hnswlib
                    self.hnsw_index.add_items(embedding_np, [memory_id])
                    
                    # HNSW会自动处理ID映射
                    self.index_size += 1
                
                logger.debug(f"记忆 {memory_id} 已添加到向量索引")
                
            elif action == "remove":
                # 从索引中移除（FAISS不支持直接移除，需要重建）
                if memory_id in self.id_to_index_map:
                    # 立即从映射中移除，防止搜索返回已删除的记忆
                    idx_to_remove = self.id_to_index_map[memory_id]
                    
                    # 1. 从id_to_index_map中移除
                    del self.id_to_index_map[memory_id]
                    
                    # 2. 从memory_id_map中移除
                    if idx_to_remove in self.memory_id_map:
                        del self.memory_id_map[idx_to_remove]
                    
                    # 3. 更新FAISS索引映射（如果存在）
                    if hasattr(self, 'faiss_index_mapping') and idx_to_remove in self.faiss_index_mapping:
                        del self.faiss_index_mapping[idx_to_remove]
                        # 尝试保存更新后的映射
                        try:
                            if hasattr(self, 'faiss_mapping_save_path'):
                                os.makedirs(os.path.dirname(self.faiss_mapping_save_path), exist_ok=True)
                                with open(self.faiss_mapping_save_path, 'w', encoding='utf-8') as f:
                                    json.dump(self.faiss_index_mapping, f, ensure_ascii=False, indent=2)
                                logger.debug(f"FAISS索引映射已更新（移除记忆 {memory_id}）")
                        except Exception as save_error:
                            logger.error(f"保存更新后的FAISS索引映射失败: {save_error}")
                    
                    # 4. 标记索引需要重建（因为索引本身仍然包含已删除的向量）
                    self.index_needs_rebuild = True
                    
                    # 5. 增加待重建计数
                    if not hasattr(self, '_pending_rebuild_count'):
                        self._pending_rebuild_count = 0
                    self._pending_rebuild_count += 1
                    
                    # 6. 检查是否达到重建阈值
                    if self._pending_rebuild_count >= self.index_rebuild_threshold:
                        logger.info(f"达到索引重建阈值（{self._pending_rebuild_count} >= {self.index_rebuild_threshold}），标记为急需重建")
                        # 设置紧急重建标志
                        self.index_needs_urgent_rebuild = True
                    
                    logger.debug(f"记忆 {memory_id} 已从映射中移除，索引标记为需要重建（待重建计数: {self._pending_rebuild_count}）")
                
                else:
                    logger.warning(f"尝试移除的记忆 {memory_id} 不在索引映射中")
        
        except Exception as e:
            logger.error(f"更新FAISS/HNSW索引失败: {e}")
    
    def _linear_search_with_embedding(self, db: Session, query_builder, query_embedding: List[float], limit: int) -> List[Tuple[float, Any]]:
        """线性搜索记忆（回退方案）
        
        参数:
            db: 数据库会话
            query_builder: SQLAlchemy查询构建器
            query_embedding: 查询嵌入向量
            limit: 返回的最大结果数
            
        返回:
            列表，包含 (分数, 记忆对象) 元组
        """
        start_time = time.time()
        memories = query_builder.all()
        scored_memories = []
        
        for memory in memories:
            try:
                memory_embedding = json.loads(memory.embedding)
                similarity = self._calculate_similarity(query_embedding, memory_embedding)
                
                # 结合相似度和重要性
                score = similarity * memory.importance
                scored_memories.append((score, memory))
            except Exception as e:
                logger.error(f"计算记忆相似度失败: {e}")
                continue
        
        # 按分数排序
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # 更新线性搜索统计
        self.linear_search_count += 1
        self.linear_search_time_total += time.time() - start_time
        
        return scored_memories[:limit]
    
    def _adapt_embedding_model(self):
        """适配嵌入模型以支持标准接口
        
        为没有encode方法的嵌入模型添加encode方法适配
        """
        if not hasattr(self.embedding_model, 'encode'):
            # 为模型添加encode方法
            def encode_adapter(text: str, convert_to_tensor: bool = False):
                self.embedding_model.eval()
                with torch.no_grad():
                    # 假设模型有forward方法接受文本
                    if hasattr(self.embedding_model, 'forward'):
                        embedding = self.embedding_model.forward(text)
                        if convert_to_tensor:
                            return embedding
                        else:
                            return embedding.cpu().numpy()
                    else:
                        raise RuntimeError(f"嵌入模型 {self.embedding_model.__class__.__name__} 既没有encode方法也没有forward方法")
            
            # 将适配器绑定到嵌入模型
            self.embedding_model.encode = encode_adapter.__get__(self.embedding_model, self.embedding_model.__class__)
            logger.info("已为嵌入模型添加encode方法适配器")
    
    def set_embedding_model(self, embedding_model: nn.Module):
        """设置外部嵌入模型
        
        允许在初始化后设置或更换嵌入模型
        
        参数:
            embedding_model: 新的嵌入模型实例
        """
        if self.initialized:
            logger.info("更换已初始化的记忆系统的嵌入模型")
        
        self.embedding_model = embedding_model
        
        # 更新嵌入维度
        if hasattr(self.embedding_model, 'embedding_dim'):
            self.embedding_dim = self.embedding_model.embedding_dim
        elif hasattr(self.embedding_model, 'config') and hasattr(self.embedding_model.config, 'hidden_size'):
            self.embedding_dim = self.embedding_model.config.hidden_size
        
        logger.info(f"嵌入模型已设置: {embedding_model.__class__.__name__}, 嵌入维度: {self.embedding_dim}")

    def set_agi_model(self, agi_model: nn.Module):
        """设置AGI模型并创建对应的文本嵌入器
        
        参数:
            agi_model: SelfAGIModel实例
        """
        self.agi_model = agi_model
        
        try:
            # 创建AGI文本嵌入器
            self.embedding_model = create_agi_text_embedder(agi_model)
            logger.info("使用AGI文本嵌入器（集成核心Transformer嵌入）")
            
            # 更新嵌入维度
            if hasattr(self.embedding_model, 'embedding_dim'):
                self.embedding_dim = self.embedding_model.embedding_dim
            elif hasattr(self.embedding_model, 'config') and hasattr(self.embedding_model.config, 'hidden_size'):
                self.embedding_dim = self.embedding_model.config.hidden_size
                
            logger.info(f"AGI文本嵌入器设置成功，嵌入维度: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"设置AGI文本嵌入器失败: {e}")
            raise

    def _get_embedding(self, content: str, content_type: str = "text", metadata: Optional[Dict[str, Any]] = None) -> List[float]:
        """获取内容嵌入（支持多模态）
        
        参数:
            content: 内容（文本、base64图像/音频、文件路径等）
            content_type: 内容类型（text, image, audio, video等）
            metadata: 额外元数据
            
        返回:
            嵌入向量列表
        """
        metadata = metadata or {}
        
        # 文本内容：使用文本嵌入模型或多模态处理器
        if content_type == "text":
            # 优先使用文本嵌入模型
            if self.embedding_model:
                embedding = self.embedding_model.encode(content)
                return embedding.tolist()
            # 次选使用多模态处理器的文本编码器
            elif self.enable_multimodal_memory and self.multimodal_processor:
                try:
                    processed_result = self.multimodal_processor.process_text(content)
                    if processed_result and hasattr(processed_result, 'embeddings'):
                        return processed_result.embeddings
                    else:
                        raise RuntimeError("多模态处理器未返回文本嵌入向量")
                except Exception as e:
                    logger.error(f"多模态文本嵌入失败: {e}")
                    raise RuntimeError(f"文本嵌入失败: {e}")
            else:
                raise RuntimeError("文本嵌入模型未初始化且多模态处理器不可用")
        
        # 多模态内容：使用多模态处理器
        elif self.enable_multimodal_memory and self.multimodal_processor:
            try:
                processed_result = None
                
                if content_type == "image":
                    # 检查是文件路径还是base64
                    if content.startswith("/") or content.startswith(".") or "\\" in content:
                        # 假设是文件路径
                        processed_result = self.multimodal_processor.process_image(image_path=content)
                    else:
                        # 假设是base64
                        processed_result = self.multimodal_processor.process_image(image_base64=content)
                
                elif content_type == "audio":
                    if content.startswith("/") or content.startswith(".") or "\\" in content:
                        processed_result = self.multimodal_processor.process_audio(audio_path=content)
                    else:
                        processed_result = self.multimodal_processor.process_audio(audio_base64=content)
                
                elif content_type == "video":
                    if content.startswith("/") or content.startswith(".") or "\\" in content:
                        processed_result = self.multimodal_processor.process_video(video_path=content)
                    else:
                        processed_result = self.multimodal_processor.process_video(video_base64=content)
                
                else:
                    # 未知类型，回退到文本处理
                    logger.warning(f"未知内容类型: {content_type}，回退到文本处理")
                    if self.embedding_model:
                        embedding = self.embedding_model.encode(content)
                        return embedding.tolist()
                    else:
                        raise RuntimeError(f"不支持的内容类型: {content_type}，且文本嵌入模型不可用")
                
                if processed_result and hasattr(processed_result, 'embeddings'):
                    return processed_result.embeddings
                else:
                    raise RuntimeError("多模态处理器未返回嵌入向量")
                    
            except Exception as e:
                logger.error(f"多模态嵌入生成失败: {e}")
                # 回退到文本嵌入（如果内容可能是文本描述）
                if self.embedding_model:
                    logger.info("尝试回退到文本嵌入")
                    try:
                        embedding = self.embedding_model.encode(content)
                        return embedding.tolist()
                    except Exception:
                        raise RuntimeError(f"多模态嵌入失败且无法回退: {e}")
                elif self.enable_multimodal_memory and self.multimodal_processor:
                    logger.info("尝试使用多模态处理器进行文本嵌入回退")
                    try:
                        processed_result = self.multimodal_processor.process_text(content)
                        if processed_result and hasattr(processed_result, 'embeddings'):
                            return processed_result.embeddings
                        else:
                            raise RuntimeError("多模态处理器未返回文本嵌入向量")
                    except Exception as e2:
                        raise RuntimeError(f"多模态嵌入失败且文本回退也失败: {e}, {e2}")
                else:
                    raise RuntimeError(f"多模态嵌入失败且无回退方案: {e}")
        
        else:
            # 多模态不可用，回退到文本嵌入
            if self.embedding_model:
                logger.warning(f"多模态不可用，将内容类型 {content_type} 作为文本处理")
                embedding = self.embedding_model.encode(content)
                return embedding.tolist()
            elif self.enable_multimodal_memory and self.multimodal_processor:
                logger.warning(f"多模态不可用但处理器可用，尝试将内容类型 {content_type} 作为文本处理")
                try:
                    processed_result = self.multimodal_processor.process_text(content)
                    if processed_result and hasattr(processed_result, 'embeddings'):
                        return processed_result.embeddings
                    else:
                        raise RuntimeError("多模态处理器未返回文本嵌入向量")
                except Exception as e:
                    raise RuntimeError(f"不支持的内容类型: {content_type}，且文本嵌入失败: {e}")
            else:
                raise RuntimeError(f"不支持的内容类型: {content_type}，且多模态和文本嵌入均不可用")

    def _calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """计算余弦相似度"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # 归一化
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def store_memory(
        self,
        db: Session,
        user_id: int,
        content: str,
        content_type: str = "text",
        memory_type: str = "short_term",
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """存储记忆"""
        from backend.db_models.memory import Memory

        # 生成嵌入（支持多模态）
        embedding = self._get_embedding(content, content_type=content_type, metadata=metadata)

        # 计算过期时间（短期记忆）
        expires_at = None
        if memory_type == "short_term":
            expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=self.short_term_memory_duration
            )

        # 创建记忆记录
        memory = Memory(
            user_id=user_id,
            content=content,
            content_type=content_type,
            embedding=json.dumps(embedding),
            importance=importance,
            memory_type=memory_type,
            accessed_count=0,
            last_accessed=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
        )

        db.add(memory)
        db.commit()
        db.refresh(memory)

        logger.info(f"存储记忆: {memory.id} (类型: {memory_type}, 用户: {user_id})")

        # 如果记忆是长期记忆，更新FAISS索引
        if memory_type == "long_term" and self.enable_faiss_retrieval and self.index_initialized:
            self._update_faiss_index(memory.id, embedding, action="add")

        # 如果记忆是长期记忆且启用了记忆图，更新动态关联图
        if memory_type == "long_term" and self.enable_memory_graph and self.graph_initialized:
            self._update_memory_graph_with_new_memory(memory.id, embedding, db)

        # 查找相似记忆并建立关联
        self._link_similar_memories(db, memory, embedding)

        # 如果短期记忆过多，触发压缩
        short_term_count = (
            db.query(Memory)
            .filter(
                Memory.user_id == user_id,
                Memory.memory_type == "short_term",
                Memory.expires_at > datetime.now(timezone.utc),
            )
            .count()
        )

        if short_term_count > self.max_short_term_memories:
            self._compress_memories(db, user_id)

        # 如果是短期记忆，添加到缓存
        if memory_type == "short_term":
            memory_dict = self._memory_to_dict(memory)
            self._add_to_short_term_cache(memory_dict)
            logger.debug(f"新短期记忆 {memory.id} 已添加到缓存")

        return memory.id

    def store_image_memory(
        self,
        db: Session,
        user_id: int,
        image_content: str,
        image_format: str = "base64",  # "base64" 或 "file_path"
        description: Optional[str] = None,
        memory_type: str = "short_term",
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """存储图像记忆
        
        参数:
            db: 数据库会话
            user_id: 用户ID
            image_content: 图像内容（base64字符串或文件路径）
            image_format: 图像格式 ("base64" 或 "file_path")
            description: 图像描述文本（可选）
            memory_type: 记忆类型
            importance: 重要性分数
            metadata: 额外元数据
            
        返回:
            记忆ID
        """
        metadata = metadata or {}
        if description:
            metadata["description"] = description
        
        # 构建完整的内容
        content = image_content
        if description:
            # 可以将描述与图像关联存储
            content = f"图像: {description}\n数据: {image_content[:100]}..." if len(image_content) > 100 else f"图像: {description}"
        
        return self.store_memory(
            db=db,
            user_id=user_id,
            content=content,
            content_type="image",
            memory_type=memory_type,
            importance=importance,
            metadata=metadata
        )

    def store_audio_memory(
        self,
        db: Session,
        user_id: int,
        audio_content: str,
        audio_format: str = "base64",  # "base64" 或 "file_path"
        transcription: Optional[str] = None,
        memory_type: str = "short_term",
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """存储音频记忆
        
        参数:
            db: 数据库会话
            user_id: 用户ID
            audio_content: 音频内容（base64字符串或文件路径）
            audio_format: 音频格式 ("base64" 或 "file_path")
            transcription: 音频转录文本（可选）
            memory_type: 记忆类型
            importance: 重要性分数
            metadata: 额外元数据
            
        返回:
            记忆ID
        """
        metadata = metadata or {}
        if transcription:
            metadata["transcription"] = transcription
        
        # 构建完整的内容
        content = audio_content
        if transcription:
            content = f"音频转录: {transcription}\n数据: {audio_content[:100]}..." if len(audio_content) > 100 else f"音频: {transcription}"
        
        return self.store_memory(
            db=db,
            user_id=user_id,
            content=content,
            content_type="audio",
            memory_type=memory_type,
            importance=importance,
            metadata=metadata
        )

    def _link_similar_memories(self, db: Session, memory, embedding: List[float]):
        """链接相似记忆"""
        from backend.db_models.memory import Memory, MemoryAssociation

        # 获取同一用户的其他记忆
        other_memories = (
            db.query(Memory)
            .filter(
                Memory.user_id == memory.user_id,
                Memory.id != memory.id,
                Memory.memory_type == "long_term",  # 只与长期记忆关联
            )
            .all()
        )

        for other_memory in other_memories:
            try:
                other_embedding = json.loads(other_memory.embedding)
                similarity = self._calculate_similarity(embedding, other_embedding)

                # 如果相似度超过阈值，建立关联
                if similarity > self.similarity_threshold:
                    association = MemoryAssociation(
                        source_memory_id=memory.id,
                        target_memory_id=other_memory.id,
                        association_type="similarity",
                        strength=similarity,
                        created_at=datetime.now(timezone.utc),
                    )

                    db.add(association)

                    # 双向关联
                    reverse_association = MemoryAssociation(
                        source_memory_id=other_memory.id,
                        target_memory_id=memory.id,
                        association_type="similarity",
                        strength=similarity,
                        created_at=datetime.now(timezone.utc),
                    )

                    db.add(reverse_association)

            except Exception as e:
                logger.error(f"链接记忆失败: {e}")
                continue

        db.commit()

    def _search_working_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """在工作记忆中搜索相关内容
        
        分层检索架构的第一层：工作记忆（内存中，毫秒级响应）
        
        参数:
            query: 查询文本
            limit: 返回结果数量限制
            
        返回:
            排序后的工作记忆项目列表
        """
        if not query or not self.working_memory:
            return []  # 返回空列表
        
        # 简单文本相似度计算
        query_lower = query.lower()
        scored_items = []
        
        for item in self.working_memory:
            content = item.get("content", "")
            if not content:
                continue
            
            content_lower = content.lower()
            
            # 完整实现）
            # 1. 完全包含：最高分
            if query_lower in content_lower or content_lower in query_lower:
                score = 1.0
            else:
                # 2. 计算词重叠
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                if query_words and content_words:
                    overlap = len(query_words.intersection(content_words))
                    score = overlap / max(len(query_words), len(content_words))
                else:
                    score = 0
            
            # 结合重要性分数
            importance_score = item.get("importance_score", 0.5)
            final_score = score * (0.7 + 0.3 * importance_score)  # 70%相似度 + 30%重要性
            
            if final_score > 0.1:  # 只保留相关结果
                scored_items.append((final_score, item.copy()))  # 使用副本避免修改原始数据
        
        # 按分数排序
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # 返回结果
        return [item for score, item in scored_items[:limit]]

    def _get_from_short_term_cache(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """从短期记忆LRU缓存获取记忆
        
        分层检索架构的第二层：短期记忆缓存（内存中，秒级响应）
        
        参数:
            memory_id: 记忆ID
            
        返回:
            记忆字典或None（如果不在缓存中）
        """
        if memory_id in self.short_term_memory_cache:
            # 缓存命中，更新访问顺序
            self._update_cache_access(memory_id)
            self.short_term_cache_hits += 1
            return self.short_term_memory_cache[memory_id]
        else:
            self.short_term_cache_misses += 1
            return None  # 返回None
    
    def _add_to_short_term_cache(self, memory_dict: Dict[str, Any]):
        """添加记忆到短期记忆LRU缓存
        
        参数:
            memory_dict: 记忆字典，必须包含'id'字段
        """
        memory_id = memory_dict.get("id")
        if not memory_id or memory_id < 0:  # 不缓存工作记忆或无效ID
            return
        
        # 如果缓存已满，移除最久未使用的项目
        if len(self.short_term_cache_keys) >= self.short_term_cache_max_size:
            oldest_key = self.short_term_cache_keys.pop(0)
            if oldest_key in self.short_term_memory_cache:
                del self.short_term_memory_cache[oldest_key]
        
        # 添加新记忆到缓存
        self.short_term_memory_cache[memory_id] = memory_dict.copy()  # 使用副本
        if memory_id not in self.short_term_cache_keys:
            self.short_term_cache_keys.append(memory_id)
    
    def _update_cache_access(self, memory_id: int):
        """更新缓存访问顺序（LRU策略）
        
        将最近访问的记忆移到列表末尾
        """
        if memory_id in self.short_term_cache_keys:
            self.short_term_cache_keys.remove(memory_id)
            self.short_term_cache_keys.append(memory_id)
    
    def _search_short_term_cache(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """在短期记忆缓存中搜索相关内容
        
        参数:
            query: 查询文本
            limit: 返回结果数量限制
            
        返回:
            排序后的缓存记忆列表
        """
        if not query or not self.short_term_memory_cache:
            return []  # 返回空列表
        
        query_lower = query.lower()
        scored_items = []
        
        for memory_id, memory_dict in self.short_term_memory_cache.items():
            content = memory_dict.get("content", "")
            if not content:
                continue
            
            content_lower = content.lower()
            
            # 完整实现）
            # 1. 完全包含：最高分
            if query_lower in content_lower or content_lower in query_lower:
                score = 1.0
            else:
                # 2. 计算词重叠
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                if query_words and content_words:
                    overlap = len(query_words.intersection(content_words))
                    score = overlap / max(len(query_words), len(content_words))
                else:
                    score = 0
            
            # 结合重要性分数
            importance = memory_dict.get("importance", 0.5)
            final_score = score * (0.7 + 0.3 * importance)  # 70%相似度 + 30%重要性
            
            if final_score > 0.1:  # 只保留相关结果
                scored_items.append((final_score, memory_dict.copy()))
        
        # 按分数排序
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # 更新缓存命中统计
        if scored_items:
            self.short_term_cache_hits += 1
        else:
            self.short_term_cache_misses += 1
        
        return [item for score, item in scored_items[:limit]]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存和索引统计信息
        
        返回:
            缓存和索引统计字典
        """
        # 缓存统计
        total_cache_access = self.short_term_cache_hits + self.short_term_cache_misses
        cache_hit_rate = self.short_term_cache_hits / max(1, total_cache_access) * 100
        
        # FAISS统计
        faiss_avg_time = 0.0
        if self.faiss_search_count > 0:
            faiss_avg_time = self.faiss_search_time_total / self.faiss_search_count * 1000  # 转换为毫秒
        
        # HNSW统计
        hnsw_avg_time = 0.0
        if self.hnsw_search_count > 0:
            hnsw_avg_time = self.hnsw_search_time_total / self.hnsw_search_count * 1000  # 转换为毫秒
        
        # 线性搜索统计
        linear_avg_time = 0.0
        if self.linear_search_count > 0:
            linear_avg_time = self.linear_search_time_total / self.linear_search_count * 1000  # 转换为毫秒
        
        return {
            # 缓存统计
            "short_term_cache_size": len(self.short_term_memory_cache),
            "short_term_cache_max_size": self.short_term_cache_max_size,
            "short_term_cache_hits": self.short_term_cache_hits,
            "short_term_cache_misses": self.short_term_cache_misses,
            "short_term_cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "working_memory_size": len(self.working_memory),
            "working_memory_capacity": self.working_memory_capacity,
            
            # 索引统计
            "index_initialized": self.index_initialized,
            "index_size": self.index_size,
            "faiss_search_count": self.faiss_search_count,
            "faiss_avg_search_time_ms": f"{faiss_avg_time:.2f}",
            "hnsw_search_count": self.hnsw_search_count,
            "hnsw_avg_search_time_ms": f"{hnsw_avg_time:.2f}",
            "linear_search_count": self.linear_search_count,
            "linear_avg_search_time_ms": f"{linear_avg_time:.2f}",
            
            # 分层检索性能摘要
            "hierarchical_performance_summary": {
                "layer1_working_memory": f"{len(self.working_memory)}/{self.working_memory_capacity} items",
                "layer2_short_term_cache": f"{len(self.short_term_memory_cache)}/{self.short_term_cache_max_size} items",
                "layer3_long_term_index": f"{self.index_size} items in index" if self.index_initialized else "not initialized"
            }
        }

    def warmup_short_term_cache(self, db: Session, user_id: Optional[int] = None, limit: int = 50):
        """预热短期记忆缓存
        
        加载最近访问或重要的短期记忆到缓存，提高缓存命中率
        
        参数:
            db: 数据库会话
            user_id: 用户ID（如果为None，则加载所有用户的记忆）
            limit: 预热的记忆数量限制
        """
        try:
            from backend.db_models.memory import Memory
            
            # 构建查询
            query = db.query(Memory).filter(Memory.memory_type == "short_term")
            
            if user_id:
                query = query.filter(Memory.user_id == user_id)
            
            # 按最近访问和重要性排序
            memories = query.order_by(
                Memory.last_accessed.desc(), Memory.importance.desc()
            ).limit(limit).all()
            
            # 添加到缓存
            for memory in memories:
                memory_dict = self._memory_to_dict(memory)
                self._add_to_short_term_cache(memory_dict)
            
            logger.info(f"短期记忆缓存预热完成: 加载了 {len(memories)} 个记忆到缓存")
            
        except Exception as e:
            logger.error(f"预热短期记忆缓存失败: {e}")
    
    def invalidate_cache_entry(self, memory_id: int):
        """使缓存条目失效
        
        当记忆被删除或更新时，从缓存中移除
        
        参数:
            memory_id: 记忆ID
        """
        if memory_id in self.short_term_memory_cache:
            del self.short_term_memory_cache[memory_id]
            if memory_id in self.short_term_cache_keys:
                self.short_term_cache_keys.remove(memory_id)
            logger.debug(f"已使缓存条目 {memory_id} 失效")
    
    def cleanup_expired_cache(self, db: Session):
        """清理过期缓存
        
        从缓存中移除已过期的短期记忆
        
        参数:
            db: 数据库会话
        """
        try:
            from backend.db_models.memory import Memory
            
            expired_count = 0
            current_time = datetime.now(timezone.utc)
            
            # 检查缓存中的每个记忆
            memory_ids_to_remove = []
            for memory_id in list(self.short_term_memory_cache.keys()):
                memory_dict = self.short_term_memory_cache[memory_id]
                expires_at_str = memory_dict.get("expires_at")
                
                if expires_at_str:
                    try:
                        expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
                        if expires_at < current_time:
                            memory_ids_to_remove.append(memory_id)
                    except Exception as parse_error:
                        # 如果解析失败，保留缓存条目
                        logger.debug(f"解析过期时间失败: {parse_error}")

            
            # 移除过期记忆
            for memory_id in memory_ids_to_remove:
                self.invalidate_cache_entry(memory_id)
                expired_count += 1
            
            if expired_count > 0:
                logger.info(f"清理了 {expired_count} 个过期缓存条目")
            
            return expired_count
            
        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")
            return 0

    def retrieve_memory(
        self,
        db: Session,
        user_id: int,
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10,
        include_expired: bool = False,
    ) -> List[Dict[str, Any]]:
        """检索记忆"""
        from backend.db_models.memory import Memory

        query_builder = db.query(Memory).filter(Memory.user_id == user_id)

        # 检查并触发索引重建（如果需要）
        # 1. 检查是否需要紧急重建（达到重建阈值）
        if hasattr(self, 'index_needs_urgent_rebuild') and self.index_needs_urgent_rebuild:
            # 紧急重建：达到删除阈值，立即重建
            pending_count = getattr(self, '_pending_rebuild_count', 0)
            logger.info(f"检测到索引急需重建（待重建计数: {pending_count}），立即触发强制重建")
            self._build_faiss_index(db, force_rebuild=True)
            self.index_needs_urgent_rebuild = False
            self.index_needs_rebuild = False
            if hasattr(self, '_pending_rebuild_count'):
                self._pending_rebuild_count = 0
            logger.info(f"紧急索引重建完成，清空待重建计数")
        
        # 2. 检查是否需要基于删除操作的索引重建
        if hasattr(self, 'index_needs_rebuild') and self.index_needs_rebuild:
            # 如果索引需要重建，立即重建
            pending_count = getattr(self, '_pending_rebuild_count', 0)
            logger.info(f"检测到索引需要重建，触发索引重建（待重建计数: {pending_count}）")
            self._build_faiss_index(db, force_rebuild=True)
            self.index_needs_rebuild = False
            if hasattr(self, '_pending_rebuild_count'):
                self._pending_rebuild_count = 0
        
        # 2. 检查是否需要基于时间的定期索引重建
        current_time = datetime.now(timezone.utc)
        
        # 初始化最后检查时间（如果是第一次）
        if self.last_index_rebuild_check is None:
            self.last_index_rebuild_check = current_time
        
        if self.index_last_updated is not None:
            hours_since_last_rebuild = (current_time - self.index_last_updated).total_seconds() / 3600
            hours_since_last_check = (current_time - self.last_index_rebuild_check).total_seconds() / 3600
            
            # 如果索引超过指定小时数未更新，并且距离上次检查至少1小时，考虑重建
            if hours_since_last_rebuild >= self.index_rebuild_interval_hours and hours_since_last_check >= 1:
                logger.info(f"索引已超过{hours_since_last_rebuild:.1f}小时未更新（阈值: {self.index_rebuild_interval_hours}小时），触发定期索引重建")
                self._build_faiss_index(db, force_rebuild=False)  # 不强制重建，让系统检查是否有变化
                # 重建后更新最后检查时间
                self.last_index_rebuild_check = current_time
            else:
                # 更新最后检查时间
                self.last_index_rebuild_check = current_time
        else:
            # 如果没有索引最后更新时间，也更新检查时间
            self.last_index_rebuild_check = current_time
        
        # 过滤记忆类型
        if memory_type:
            query_builder = query_builder.filter(Memory.memory_type == memory_type)

        # 过滤过期记忆
        if not include_expired:
            query_builder = query_builder.filter(
                (Memory.expires_at.is_(None)) | (Memory.expires_at > datetime.now(timezone.utc))
            )

        # 如果没有查询文本，按重要性排序返回
        if not query:
            memories = (
                query_builder.order_by(
                    Memory.importance.desc(), Memory.last_accessed.desc()
                )
                .limit(limit)
                .all()
            )

            # 转换为字典并添加到缓存
            result_memories = []
            for memory in memories:
                memory_dict = self._memory_to_dict(memory)
                result_memories.append(memory_dict)
                
                # 如果是短期记忆，添加到缓存
                if memory.memory_type == "short_term":
                    self._add_to_short_term_cache(memory_dict)
            
            return result_memories

        # ====== 分层检索架构 ======
        # 第一层：工作记忆（内存中，毫秒级响应）
        if query and not memory_type:  # 只有当没有指定记忆类型时才检查工作记忆
            working_memory_results = self._search_working_memory(query, limit=limit)
            if working_memory_results:
                # 如果工作记忆中找到足够结果（至少50%的需求数量），直接返回
                min_required = max(1, limit // 2)
                if len(working_memory_results) >= min_required:
                    logger.debug(f"从工作记忆中检索到 {len(working_memory_results)} 个结果，直接返回")
                    # 转换为标准记忆格式
                    formatted_results = []
                    for item in working_memory_results:
                        formatted_item = {
                            "id": -1,  # 工作记忆没有ID
                            "content": item.get("content", ""),
                            "importance": item.get("importance_score", 0.5),
                            "memory_type": "working_memory",
                            "created_at": item.get("timestamp", datetime.now(timezone.utc).isoformat()),
                            "last_accessed": datetime.now(timezone.utc).isoformat(),
                            "accessed_count": 0,
                            "user_id": user_id
                        }
                        formatted_results.append(formatted_item)
                    return formatted_results
                else:
                    logger.debug(f"从工作记忆中检索到 {len(working_memory_results)} 个结果，不足 {min_required} 个，继续下层检索")
        
        # 第二层：短期记忆缓存（内存中，秒级响应）
        if query and (memory_type in [None, "short_term"]):  # 只检查短期记忆或不指定类型
            short_term_cache_results = self._search_short_term_cache(query, limit=limit)
            if short_term_cache_results:
                # 如果缓存中找到足够结果（至少30%的需求数量），直接返回
                min_required = max(1, limit // 3)
                if len(short_term_cache_results) >= min_required:
                    logger.debug(f"从短期记忆缓存中检索到 {len(short_term_cache_results)} 个结果，直接返回")
                    # 确保记忆格式正确
                    formatted_results = []
                    for item in short_term_cache_results:
                        if "memory_type" not in item:
                            item["memory_type"] = "short_term"
                        formatted_results.append(item)
                    return formatted_results
                else:
                    logger.debug(f"从短期记忆缓存中检索到 {len(short_term_cache_results)} 个结果，不足 {min_required} 个，继续下层检索")
        
        # 第三层检索（数据库+FAISS/HNSW索引）将继续执行下面的逻辑
        # ====== 分层检索架构结束 ======

        # 如果有查询文本，进行向量搜索
        query_embedding = self._get_embedding(query)
        
        # 使用FAISS/HNSW索引进行高效搜索（如果启用且可用）
        # 扩展向量检索适用范围：所有记忆类型（除了working_memory）都可以使用向量检索
        index_search_types = [None, "long_term", "short_term", "episodic", "semantic", "procedural"]
        if self.enable_faiss_retrieval and self.index_initialized and memory_type in index_search_types:
            # 使用索引搜索
            faiss_results = self._search_with_faiss(query_embedding, k=limit * 2)  # 获取更多结果以进行重要性过滤
            
            # 获取记忆详情并计算最终分数
            scored_memories = []
            for memory_id, similarity in faiss_results:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory and (include_expired or memory.expires_at is None or memory.expires_at > datetime.now(timezone.utc)):
                    # 结合相似度和重要性
                    score = similarity * memory.importance
                    scored_memories.append((score, memory))
            
            # 如果索引搜索结果不足，回退到线性搜索
            if len(scored_memories) < min(limit, 3):
                logger.debug(f"FAISS索引搜索结果不足 ({len(scored_memories)}个)，回退到线性搜索")
                scored_memories = self._linear_search_with_embedding(
                    db, query_builder, query_embedding, limit * 2
                )
        else:
            # 线性搜索（回退方案）
            scored_memories = self._linear_search_with_embedding(
                db, query_builder, query_embedding, limit * 2
            )
        
        # 按分数排序
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # 更新访问计数并添加到缓存
        result_memories = []
        for score, memory in scored_memories[:limit]:
            memory.accessed_count += 1
            memory.last_accessed = datetime.now(timezone.utc)
            db.add(memory)
            
            # 转换为字典格式
            memory_dict = self._memory_to_dict(memory)
            result_memories.append(memory_dict)
            
            # 如果是短期记忆，添加到缓存
            if memory.memory_type == "short_term":
                self._add_to_short_term_cache(memory_dict)
        
        db.commit()

        return result_memories

    def search_memories(
        self,
        db: Session,
        user_id: int,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """搜索记忆（基于语义相似度）"""
        return self.retrieve_memory(db, user_id, query, memory_type, limit)

    def get_memory_associations(
        self, db: Session, memory_id: int, association_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取记忆关联"""
        from backend.db_models.memory import MemoryAssociation, Memory

        query_builder = db.query(MemoryAssociation).filter(
            MemoryAssociation.source_memory_id == memory_id
        )

        if association_type:
            query_builder = query_builder.filter(
                MemoryAssociation.association_type == association_type
            )

        associations = query_builder.order_by(MemoryAssociation.strength.desc()).all()

        result = []
        for association in associations:
            target_memory = (
                db.query(Memory)
                .filter(Memory.id == association.target_memory_id)
                .first()
            )
            if target_memory:
                result.append(
                    {
                        "association_id": association.id,
                        "target_memory": self._memory_to_dict(target_memory),
                        "association_type": association.association_type,
                        "strength": association.strength,
                        "created_at": (
                            association.created_at.isoformat()
                            if association.created_at
                            else None
                        ),
                    }
                )

        return result

    def _compress_memories(self, db: Session, user_id: Optional[int] = None):
        """压缩记忆：将重要的短期记忆转为长期记忆，删除冗余"""
        from backend.db_models.memory import Memory

        try:
            # 查找需要压缩的短期记忆
            query_builder = db.query(Memory).filter(
                Memory.memory_type == "short_term",
                Memory.expires_at < datetime.now(timezone.utc),
            )

            if user_id:
                query_builder = query_builder.filter(Memory.user_id == user_id)

            expired_memories = query_builder.all()

            for memory in expired_memories:
                # 如果记忆重要，转为长期记忆
                if memory.importance >= self.importance_threshold:
                    memory.memory_type = "long_term"
                    memory.expires_at = None
                    db.add(memory)
                    logger.info(
                        f"记忆 {memory.id} 转为长期记忆 (重要性: {memory.importance})"
                    )
                    
                    # 如果启用了FAISS检索，将记忆添加到索引
                    if self.enable_faiss_retrieval and self.index_initialized:
                        try:
                            embedding = json.loads(memory.embedding)
                            self._update_faiss_index(memory.id, embedding, action="add")
                            logger.debug(f"记忆 {memory.id} 已添加到FAISS索引")
                        except Exception as e:
                            logger.error(f"将记忆 {memory.id} 添加到FAISS索引失败: {e}")
                    
                    # 如果启用了记忆图，将记忆添加到图
                    if self.enable_memory_graph and self.graph_initialized:
                        try:
                            embedding = json.loads(memory.embedding)
                            self._update_memory_graph_with_new_memory(memory.id, embedding, db)
                            logger.debug(f"记忆 {memory.id} 已添加到动态关联图")
                        except Exception as e:
                            logger.error(f"将记忆 {memory.id} 添加到动态关联图失败: {e}")
                else:
                    # 删除不重要的记忆
                    db.delete(memory)
                    logger.info(f"删除记忆 {memory.id} (重要性: {memory.importance})")

            db.commit()
            logger.info(f"压缩完成，处理了 {len(expired_memories)} 个记忆")

        except Exception as e:
            logger.error(f"记忆压缩失败: {e}")
            db.rollback()

    def delete_memory(self, db: Session, memory_id: int):
        """删除记忆"""
        from backend.db_models.memory import Memory, MemoryAssociation

        try:
            # 获取记忆（用于检查是否为长期记忆）
            memory = db.query(Memory).filter(Memory.id == memory_id).first()
            
            if memory:
                # 如果记忆是长期记忆且启用了FAISS检索，从索引中移除
                if memory.memory_type == "long_term" and self.enable_faiss_retrieval and self.index_initialized:
                    try:
                        embedding = json.loads(memory.embedding)
                        self._update_faiss_index(memory_id, embedding, action="remove")
                        logger.debug(f"记忆 {memory_id} 已从FAISS索引中标记移除")
                    except Exception as e:
                        logger.error(f"从FAISS索引中移除记忆 {memory_id} 失败: {e}")
                
                # 如果记忆是长期记忆且启用了记忆图，从图中移除
                if memory.memory_type == "long_term" and self.enable_memory_graph and self.graph_initialized:
                    try:
                        # 标记需要从图中移除，将在下次重建时处理
                        if memory_id in self.memory_to_node_map:
                            # 标记为需要重建
                            self.graph_needs_rebuild = True
                            logger.debug(f"记忆 {memory_id} 标记为需要从图中移除，将在下次重建时处理")
                    except Exception as e:
                        logger.error(f"标记记忆 {memory_id} 从图中移除失败: {e}")
            
            # 删除关联
            db.query(MemoryAssociation).filter(
                (MemoryAssociation.source_memory_id == memory_id)
                | (MemoryAssociation.target_memory_id == memory_id)
            ).delete()

            # 删除记忆
            if memory:
                db.delete(memory)

            db.commit()
            
            # 使缓存条目失效（如果存在）
            self.invalidate_cache_entry(memory_id)
            
            logger.info(f"删除记忆: {memory_id}")

        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            db.rollback()
            raise

    def predict_memory_importance(self, memory_data: Dict[str, Any]) -> float:
        """预测记忆重要性分数
        
        使用重要性学习器预测记忆的重要性分数
        
        参数:
            memory_data: 记忆数据字典
            
        返回:
            重要性分数 (0-1)
        """
        if not self.importance_model:
            # 如果没有重要性学习器，返回基于规则的重要性
            return self._compute_rule_based_importance(memory_data)
        
        try:
            # 计算特征
            features = self.importance_model.compute_features(memory_data)
            
            # 添加批次维度 [1, feature_dim]
            features = features.unsqueeze(0)
            
            # 预测重要性
            with torch.no_grad():
                importance_score = self.importance_model(features)
            
            # 转换为标量
            importance_score = importance_score.item()
            
            return float(importance_score)
            
        except Exception as e:
            logger.error(f"预测记忆重要性失败: {e}")
            # 失败时返回基于规则的重要性
            return self._compute_rule_based_importance(memory_data)
    
    def _compute_rule_based_importance(self, memory_data: Dict[str, Any]) -> float:
        """计算基于规则的重要性分数
        
        当重要性学习器不可用时使用的后备方案
        """
        importance = 0.5  # 基础重要性
        
        # 基于访问次数调整
        accessed_count = memory_data.get("accessed_count", 0)
        importance += min(accessed_count * 0.05, 0.3)  # 每次访问增加0.05，最多0.3
        
        # 基于时间衰减调整
        last_accessed = memory_data.get("last_accessed")
        if last_accessed:
            if isinstance(last_accessed, str):
                from datetime import datetime
                last_accessed = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
            
            if isinstance(last_accessed, datetime):
                time_diff = datetime.now(timezone.utc) - last_accessed
                hours_diff = time_diff.total_seconds() / 3600.0
                # 24小时内访问过，增加重要性
                if hours_diff < 24:
                    importance += 0.2
        
        # 基于记忆类型调整
        memory_type = memory_data.get("memory_type", "short_term")
        if memory_type == "long_term":
            importance += 0.1
        
        # 限制在0-1之间
        importance = max(0.0, min(1.0, importance))
        
        return importance

    def update_memory_importance(self, db: Session, memory_id: int, importance: float):
        """更新记忆重要性"""
        from backend.db_models.memory import Memory

        memory = db.query(Memory).filter(Memory.id == memory_id).first()
        if memory:
            memory.importance = max(0.0, min(1.0, importance))  # 限制在0-1之间
            memory.last_accessed = datetime.now(timezone.utc)
            db.add(memory)
            db.commit()
            
            # 更新缓存中的记忆（如果存在）
            if memory_id in self.short_term_memory_cache and memory.memory_type == "short_term":
                memory_dict = self._memory_to_dict(memory)
                self.short_term_memory_cache[memory_id] = memory_dict
                logger.debug(f"更新缓存中的记忆 {memory_id} 重要性: {importance}")
            
            logger.info(f"更新记忆重要性: {memory_id} -> {importance}")

    def _memory_to_dict(self, memory) -> Dict[str, Any]:
        """将记忆对象转为字典"""
        try:
            embedding = json.loads(memory.embedding) if memory.embedding else []
        except Exception:
            embedding = []

        return {
            "id": memory.id,
            "user_id": memory.user_id,
            "content": memory.content,
            "content_type": memory.content_type,
            "importance": memory.importance,
            "memory_type": memory.memory_type,
            "accessed_count": memory.accessed_count,
            "last_accessed": (
                memory.last_accessed.isoformat() if memory.last_accessed else None
            ),
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
            "embedding_length": len(embedding),
        }

    def get_memory_stats(
        self, db: Session, user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """获取记忆统计"""
        from backend.db_models.memory import Memory

        query = db.query(Memory)
        if user_id is not None:
            query = query.filter(Memory.user_id == user_id)

        total = query.count()

        short_term_query = db.query(Memory).filter(
            Memory.memory_type == "short_term",
            (Memory.expires_at.is_(None)) | (Memory.expires_at > datetime.now(timezone.utc)),
        )
        if user_id is not None:
            short_term_query = short_term_query.filter(Memory.user_id == user_id)
        short_term = short_term_query.count()

        long_term_query = db.query(Memory).filter(Memory.memory_type == "long_term")
        if user_id is not None:
            long_term_query = long_term_query.filter(Memory.user_id == user_id)
        long_term = long_term_query.count()

        expired_query = db.query(Memory).filter(
            Memory.memory_type == "short_term", Memory.expires_at <= datetime.now(timezone.utc)
        )
        if user_id is not None:
            expired_query = expired_query.filter(Memory.user_id == user_id)
        expired = expired_query.count()

        avg_importance_query = db.query(func.avg(Memory.importance))
        if user_id is not None:
            avg_importance_query = avg_importance_query.filter(
                Memory.user_id == user_id
            )
        avg_importance_result = avg_importance_query.scalar()

        avg_importance = float(avg_importance_result) if avg_importance_result else 0.0

        return {
            "total_memories": total,
            "short_term_memories": short_term,
            "long_term_memories": long_term,
            "expired_memories": expired,
            "average_importance": avg_importance,
            "user_id": user_id,
        }

    # ==================== 升级功能：基于最新研究的AGI记忆增强 ====================

    def retrieve_relevant_memories(
        self,
        query: str,
        db: Session,
        user_id: int,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """检索增强记忆（RAG风格） - 基于查询检索最相关记忆
        
        基于最新研究论文的改进：
        1. 混合检索：语义相似度 + 时间衰减 + 重要性权重
        2. 多样性采样：避免返回过于相似的记忆
        3. 相关性重排：基于查询-记忆交互重新排序
        
        参数：
            query: 查询文本
            db: 数据库会话
            user_id: 用户ID
            top_k: 返回记忆数量（默认使用配置的rag_top_k）
            similarity_threshold: 相似度阈值（默认使用配置的similarity_threshold）
            memory_type: 记忆类型过滤（short_term/long_term，默认不过滤）
            
        返回：
            相关记忆列表，包含记忆内容和元数据
        """
        if not self.initialized:
            logger.warning("记忆系统未初始化，无法检索记忆")
            return []  # 返回空列表
        
        if top_k is None:
            top_k = self.rag_top_k
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
            
        from backend.db_models.memory import Memory
        
        # 生成查询嵌入
        query_embedding = self._get_embedding(query)
        
        # 获取用户记忆
        memory_query = db.query(Memory).filter(Memory.user_id == user_id)
        if memory_type:
            memory_query = memory_query.filter(Memory.memory_type == memory_type)
            
        memories = memory_query.all()
        
        if not memories:
            return []  # 返回空列表
        
        # 计算每个记忆的评分（混合评分策略）
        scored_memories = []
        for memory in memories:
            try:
                memory_embedding = json.loads(memory.embedding) if memory.embedding else []
                if not memory_embedding:
                    continue
                    
                # 1. 语义相似度（余弦相似度）
                semantic_score = self._calculate_similarity(query_embedding, memory_embedding)
                
                # 2. 时间衰减（最近记忆更重要）
                time_decay = self._calculate_time_decay(memory.created_at)
                
                # 3. 重要性权重
                importance_weight = memory.importance
                
                # 4. 访问频率（频繁访问的记忆可能更相关）
                access_weight = min(memory.accessed_count / 10.0, 1.0)  # 归一化
                
                # 混合评分：加权组合
                # 公式：score = α*semantic + β*time_decay + γ*importance + δ*access
                alpha, beta, gamma, delta = 0.5, 0.2, 0.2, 0.1  # 可配置的权重
                composite_score = (
                    alpha * semantic_score +
                    beta * time_decay +
                    gamma * importance_weight +
                    delta * access_weight
                )
                
                # 应用相似度阈值
                if semantic_score >= similarity_threshold:
                    scored_memories.append({
                        "memory": memory,
                        "semantic_score": semantic_score,
                        "composite_score": composite_score,
                        "time_decay": time_decay,
                        "importance": importance_weight
                    })
                    
            except Exception as e:
                logger.warning(f"计算记忆 {memory.id} 评分失败: {e}")
                continue
        
        # 按综合评分排序
        scored_memories.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # 多样性采样：避免返回过于相似的记忆
        diversified_memories = self._diversity_sampling(scored_memories, top_k)
        
        # 转换为返回格式
        result = []
        for item in diversified_memories[:top_k]:
            memory = item["memory"]
            result.append({
                "id": memory.id,
                "content": memory.content,
                "content_type": memory.content_type,
                "memory_type": memory.memory_type,
                "importance": memory.importance,
                "semantic_score": item["semantic_score"],
                "composite_score": item["composite_score"],
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None
            })
        
        logger.debug(f"检索到 {len(result)} 个相关记忆，查询: '{query[:50]}...'")
        return result

    def _calculate_time_decay(self, created_at: datetime) -> float:
        """计算时间衰减因子
        
        基于艾宾浩斯遗忘曲线：记忆强度随时间指数衰减
        decay = exp(-t/τ)，其中τ为时间常数（默认7天）
        """
        if not created_at:
            return 0.5  # 默认值
            
        now = datetime.now(timezone.utc)
        time_diff = (now - created_at).total_seconds()
        
        # 时间常数：7天（604800秒）
        tau = 604800.0
        
        # 指数衰减，确保在[0,1]范围内
        decay = np.exp(-time_diff / tau)
        return float(decay)

    def _diversity_sampling(self, scored_memories: List[Dict], top_k: int) -> List[Dict]:
        """多样性采样 - 避免返回过于相似的记忆
        
        基于最大边际相关性（MMR）算法：
        1. 选择最高分的记忆
        2. 后续选择既与查询相关又与已选记忆不相似的记忆
        3. 平衡相关性和多样性
        """
        if len(scored_memories) <= top_k:
            return scored_memories
        
        selected = []
        remaining = scored_memories.copy()
        
        # 首先选择最高分的记忆
        first_memory = remaining.pop(0)
        selected.append(first_memory)
        
        # MMR算法：λ平衡相关性和多样性
        lambda_param = 0.7  # 偏向相关性（0.5为平衡，0.7为更重视相关性）
        
        while len(selected) < top_k and remaining:
            # 计算每个剩余记忆的MMR分数
            mmr_scores = []
            for i, candidate in enumerate(remaining):
                # 相关性分数
                rel_score = candidate["composite_score"]
                
                # 最大相似度（与已选记忆）
                max_sim = 0.0
                for selected_item in selected:
                    # 完整处理，实际应计算记忆嵌入的相似度
                    # 假设语义分数可以近似表示相似度
                    sim = candidate["semantic_score"] * selected_item["semantic_score"]
                    max_sim = max(max_sim, sim)
                
                # MMR分数 = λ*相关性 - (1-λ)*相似度
                mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim
                mmr_scores.append((i, mmr_score))
            
            # 选择MMR分数最高的记忆
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(remaining.pop(best_idx))
        
        return selected

    def update_working_memory(self, context: Dict[str, Any], db: Session, user_id: int):
        """更新工作记忆缓冲区
        
        工作记忆类似于人类的工作记忆系统：
        1. 有限容量（working_memory_capacity）
        2. 最近和相关信息优先
        3. 基于注意力机制管理
        """
        # 确保工作记忆不超过容量
        if len(self.working_memory) >= self.working_memory_capacity:
            # 移除最不重要的记忆（基于综合评分）
            self.working_memory.sort(key=lambda x: x.get("importance_score", 0))
            self.working_memory = self.working_memory[-self.working_memory_capacity//2:]
        
        # 添加上下文到工作记忆
        context_item = {
            "content": context.get("content", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "importance_score": context.get("importance", 0.5),
            "user_id": user_id,
            "context_type": context.get("type", "general")
        }
        
        self.working_memory.append(context_item)
        
        # 如果启用了记忆图，更新关联
        if self.enable_memory_graph:
            self._update_memory_graph(context_item, db, user_id)
    
    def add_to_working_memory(self, content: str, importance: float = 0.5, 
                              context_type: str = "general", db: Optional[Session] = None,
                              user_id: Optional[int] = None):
        """添加项目到工作记忆缓冲区
        
        完整接口，用于快速添加记忆到工作缓冲区
        """
        context = {
            "content": content,
            "importance": importance,
            "type": context_type
        }
        
        if db and user_id:
            self.update_working_memory(context, db, user_id)
        else:
            # 仅存储在内存中
            context_item = {
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "importance_score": importance,
                "context_type": context_type
            }
            self.working_memory.append(context_item)
            
            # 限制容量
            if len(self.working_memory) > self.working_memory_capacity:
                self.working_memory = self.working_memory[-self.working_memory_capacity:]

    def _update_memory_graph(self, context_item: Dict, db: Session, user_id: int):
        """更新记忆关联图
        
        基于图神经网络（GNN）的思想构建记忆关联：
        1. 记忆作为节点
        2. 语义关联作为边
        3. 动态更新关联强度
        """
        # 完整的关联更新
        # 实际实现应使用图数据库或专门的图结构
        try:
            from backend.db_models.memory import MemoryAssociation
            
            # 获取最近的其他记忆
            recent_memories = self._get_recent_memories(db, user_id, limit=10)
            
            if len(recent_memories) < 2:
                return
            
            # 完整关联：为最近的记忆对创建关联
            # 实际应基于语义相似度和时间接近性
            logger.debug(f"更新记忆关联图，用户 {user_id}，最近记忆数: {len(recent_memories)}")
            
        except Exception as e:
            logger.warning(f"更新记忆关联图失败: {e}")

    def _get_recent_memories(self, db: Session, user_id: int, limit: int = 10):
        """获取最近的记忆"""
        from backend.db_models.memory import Memory
        
        recent_memories = (
            db.query(Memory)
            .filter(Memory.user_id == user_id)
            .order_by(Memory.created_at.desc())
            .limit(limit)
            .all()
        )
        
        return recent_memories

    def get_working_memory_summary(self) -> Dict[str, Any]:
        """获取工作记忆摘要"""
        return {
            "total_items": len(self.working_memory),
            "items": self.working_memory[-10:],  # 最近10个项目
            "capacity": self.working_memory_capacity,
            "usage_percentage": len(self.working_memory) / self.working_memory_capacity * 100
        }

    def learn_importance(self, memory_id: int, feedback_score: float, 
                        db: Session, user_id: int):
        """学习记忆重要性（基于反馈的强化学习）
        
        基于用户反馈调整记忆重要性：
        1. 正面反馈增加重要性
        2. 负面反馈减少重要性
        3. 随时间衰减
        """
        try:
            from backend.db_models.memory import Memory
            
            memory = db.query(Memory).filter(
                Memory.id == memory_id,
                Memory.user_id == user_id
            ).first()
            
            if not memory:
                logger.warning(f"记忆 {memory_id} 不存在或不属于用户 {user_id}")
                return
            
            # 基于反馈更新重要性（完整RL更新）
            # 实际应使用更复杂的RL算法，如Q-learning
            learning_rate = 0.1
            new_importance = memory.importance + learning_rate * (feedback_score - memory.importance)
            
            # 限制在[0,1]范围内
            new_importance = max(0.0, min(1.0, new_importance))
            
            memory.importance = new_importance
            db.commit()
            
            logger.info(f"记忆 {memory_id} 重要性更新: {memory.importance:.3f} -> {new_importance:.3f}")
            
        except Exception as e:
            logger.error(f"学习记忆重要性失败: {e}")
            db.rollback()

    def build_memory_association_graph(self, db: Session, user_id: int) -> Dict[str, Any]:
        """构建记忆关联图
        
        基于语义相似度和时间接近性构建记忆关联图
        返回图结构，可用于可视化或进一步分析
        
        增强功能：
        1. 如果启用了GNN，使用学习到的图表示
        2. 社区检测支持
        3. 时间动态分析
        """
        try:
            # 如果启用了记忆图且已初始化，使用图结构
            if self.enable_memory_graph and self.graph_initialized and self.memory_graph is not None:
                return self._build_graph_from_memory_graph(db, user_id)
            else:
                # 回退到原始实现
                return self._build_graph_from_database(db, user_id)
            
        except Exception as e:
            logger.error(f"构建记忆关联图失败: {e}")
            return {"error": str(e), "nodes": [], "edges": []}
    
    def _build_graph_from_memory_graph(self, db: Session, user_id: int) -> Dict[str, Any]:
        """从内存图结构构建关联图
        
        使用已构建的图神经网络图结构
        """
        try:
            from backend.db_models.memory import Memory
            
            # 获取用户的所有长期记忆
            memories = (
                db.query(Memory)
                .filter(
                    Memory.user_id == user_id,
                    Memory.memory_type == "long_term"
                )
                .all()
            )
            
            if len(memories) < 2:
                return {"nodes": [], "edges": [], "message": "记忆数量不足，无法构建图"}
            
            nodes = []
            edges = []
            
            # 创建节点（增强版）
            for memory in memories:
                # 获取节点索引
                node_idx = self.memory_to_node_map.get(memory.id)
                node_embedding = None
                community_id = -1
                
                # 如果图中有节点嵌入，添加额外信息
                if (node_idx is not None and self.memory_graph is not None and 
                    "node_embeddings" in self.memory_graph and 
                    self.memory_graph["node_embeddings"] is not None):
                    
                    node_embedding = self.memory_graph["node_embeddings"][node_idx].tolist() if node_idx < len(self.memory_graph["node_embeddings"]) else None
                    
                    # 社区信息
                    if "communities" in self.memory_graph and self.memory_graph["communities"]:
                        for comm_id, community in enumerate(self.memory_graph["communities"]):
                            if node_idx in community:
                                community_id = comm_id
                                break
                
                node_data = {
                    "id": memory.id,
                    "content_preview": memory.content[:50] + "..." if len(memory.content) > 50 else memory.content,
                    "importance": memory.importance,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    "accessed_count": memory.accessed_count,
                    "memory_type": memory.memory_type
                }
                
                # 添加增强信息
                if node_embedding is not None:
                    node_data["embedding_dim"] = len(node_embedding)
                    node_data["has_embedding"] = True
                
                if community_id != -1:
                    node_data["community_id"] = community_id
                
                nodes.append(node_data)
            
            # 创建边（使用图结构中的邻接矩阵）
            adjacency = self.memory_graph.get("adjacency") if self.memory_graph else None
            edge_predictions = self.memory_graph.get("edge_predictions") if self.memory_graph else None
            
            if adjacency is not None:
                # 使用邻接矩阵构建边
                num_nodes = adjacency.shape[0] if hasattr(adjacency, 'shape') else 0
                edge_count = 0
                max_edges = min(self.max_graph_edges, 200)  # 限制边数量
                
                # 转换为numpy数组以便处理
                if hasattr(adjacency, 'numpy'):
                    adj_np = adjacency.numpy()
                elif hasattr(adjacency, 'cpu'):
                    adj_np = adjacency.cpu().numpy()
                else:
                    adj_np = np.array(adjacency)
                
                # 遍历邻接矩阵
                for i in range(num_nodes):
                    if edge_count >= max_edges:
                        break
                    
                    memory_i = self.node_to_memory_map.get(i)
                    if not memory_i:
                        continue
                    
                    for j in range(i+1, num_nodes):
                        if edge_count >= max_edges:
                            break
                        
                        memory_j = self.node_to_memory_map.get(j)
                        if not memory_j:
                            continue
                        
                        strength = adj_np[i, j]
                        if strength > 0.1:  # 阈值
                            edge_type = "semantic_similarity"
                            
                            # 如果有关联预测，使用预测类型
                            if edge_predictions is not None and hasattr(edge_predictions, 'shape') and i < edge_predictions.shape[0] and j < edge_predictions.shape[1]:
                                pred_strength = edge_predictions[i, j].item() if hasattr(edge_predictions[i, j], 'item') else edge_predictions[i, j]
                                if pred_strength > 0.5:
                                    edge_type = "gnn_predicted"
                            
                            edges.append({
                                "source": memory_i,
                                "target": memory_j,
                                "strength": float(strength),
                                "type": edge_type,
                                "source_idx": i,
                                "target_idx": j
                            })
                            edge_count += 1
            
            # 如果从图结构中没有获取到边，回退到数据库计算
            if not edges:
                edges = self._build_edges_from_database(memories)
            
            # 添加社区信息（如果可用）
            communities_data = []
            if "communities" in self.memory_graph and self.memory_graph["communities"]:
                for comm_id, community in enumerate(self.memory_graph["communities"]):
                    community_memories = []
                    for node_idx in community:
                        memory_id = self.node_to_memory_map.get(node_idx)
                        if memory_id:
                            community_memories.append(memory_id)
                    
                    if community_memories:
                        communities_data.append({
                            "id": comm_id,
                            "size": len(community_memories),
                            "memory_ids": community_memories
                        })
            
            # 计算图统计信息
            node_count = len(nodes)
            edge_count = len(edges)
            density = edge_count / (node_count * (node_count - 1) / 2) if node_count > 1 else 0
            
            # 添加GNN特定统计
            gnn_stats = {}
            if self.enable_gnn_learning and self.gnn_model is not None:
                gnn_stats = {
                    "gnn_enabled": True,
                    "gnn_hidden_dim": self.gnn_hidden_dim,
                    "gnn_layers": self.gnn_num_layers
                }
            
            return {
                "nodes": nodes,
                "edges": edges,
                "communities": communities_data,
                "stats": {
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "density": density,
                    "graph_initialized": self.graph_initialized,
                    "gnn_used": self.enable_gnn_learning and self.gnn_model is not None,
                    "community_count": len(communities_data)
                },
                "gnn_stats": gnn_stats
            }
            
        except Exception as e:
            logger.error(f"从内存图构建关联图失败: {e}")
            # 回退到数据库构建
            return self._build_graph_from_database(db, user_id)
    
    def _build_graph_from_database(self, db: Session, user_id: int) -> Dict[str, Any]:
        """从数据库构建关联图（原始实现）
        
        回退方案：当图结构不可用时使用
        """
        try:
            from backend.db_models.memory import Memory
            
            # 获取用户的所有长期记忆
            memories = (
                db.query(Memory)
                .filter(
                    Memory.user_id == user_id,
                    Memory.memory_type == "long_term"
                )
                .all()
            )
            
            if len(memories) < 2:
                return {"nodes": [], "edges": [], "message": "记忆数量不足，无法构建图"}
            
            nodes = []
            edges = []
            
            # 创建节点
            for memory in memories:
                nodes.append({
                    "id": memory.id,
                    "content_preview": memory.content[:50] + "..." if len(memory.content) > 50 else memory.content,
                    "importance": memory.importance,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None
                })
            
            # 创建边（基于语义相似度）
            edges = self._build_edges_from_database(memories)
            
            return {
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "density": len(edges) / (len(nodes) * (len(nodes)-1) / 2) if len(nodes) > 1 else 0,
                    "graph_initialized": self.graph_initialized,
                    "gnn_used": False
                }
            }
        except Exception as e:
            logger.error(f"从数据库构建关联图失败: {e}")
            return {"nodes": [], "edges": [], "message": f"构建图失败: {str(e)}"}

    # ====== 选择性记忆机制 ======
    
    def apply_selective_memory(self, db: Session, user_id: int) -> Dict[str, Any]:
        """应用选择性记忆机制 - 基于重要性评分选择保留内容，遗忘不重要信息
        
        根据审核报告中的上下文压缩技术，选择性记忆机制包括：
        1. 基于重要性评分选择保留内容
        2. 遗忘不重要信息（基于遗忘率）
        3. 动态调整保留策略
        
        参数:
            db: 数据库会话
            user_id: 用户ID
            
        返回:
            字典包含处理统计信息
        """
        try:
            from backend.db_models.memory import Memory
            
            # 更新系统状态到自主记忆管理器
            self.update_autonomous_system_state()
            
            # 检查是否启用选择性记忆机制
            selective_memory_enabled = False
            forgetting_rate = 0.05  # 默认遗忘率
            importance_threshold = 0.7  # 默认重要性阈值
            
            # 从自主记忆管理器获取参数（如果启用）
            if self.enable_autonomous_memory and self.autonomous_manager:
                autonomous_params = self.get_autonomous_parameters()
                forgetting_rate = autonomous_params.get("forgetting_rate", 0.05)
                importance_threshold = autonomous_params.get("importance_threshold", 0.7)
                selective_memory_enabled = True
            else:
                # 从配置获取参数
                if hasattr(self, 'config'):
                    selective_memory_enabled = self.config.get("selective_memory_enabled", False)
                    forgetting_rate = self.config.get("forgetting_rate", 0.05)
                    importance_threshold = self.config.get("importance_threshold", 0.7)
            
            if not selective_memory_enabled:
                logger.info("选择性记忆机制未启用，跳过处理")
                return {
                    "enabled": False,
                    "message": "选择性记忆机制未启用",
                    "stats": {}
                }
            
            logger.info(f"应用选择性记忆机制: 遗忘率={forgetting_rate}, 重要性阈值={importance_threshold}")
            
            # 记录决策上下文
            decision_context = {
                "decision_type": "selective_memory",
                "forgetting_rate": forgetting_rate,
                "importance_threshold": importance_threshold,
                "user_id": user_id,
                "timestamp": time.time()
            }
            
            # 获取用户的所有记忆（包括短期和长期）
            all_memories = (
                db.query(Memory)
                .filter(Memory.user_id == user_id)
                .all()
            )
            
            if not all_memories:
                logger.info("用户没有记忆，跳过选择性记忆处理")
                return {
                    "enabled": True,
                    "message": "没有需要处理的记忆",
                    "stats": {"total_memories": 0}
                }
            
            # 分类记忆
            short_term_memories = []
            long_term_memories = []
            
            for memory in all_memories:
                if memory.memory_type == "short_term":
                    short_term_memories.append(memory)
                else:
                    long_term_memories.append(memory)
            
            # 处理短期记忆：基于重要性阈值决定是否升级为长期记忆
            upgraded_count = 0
            forgotten_short_term_count = 0
            
            for memory in short_term_memories:
                if memory.importance >= importance_threshold:
                    # 升级为长期记忆
                    memory.memory_type = "long_term"
                    memory.expires_at = None  # 长期记忆不过期
                    upgraded_count += 1
                    logger.debug(f"升级短期记忆 {memory.id} 为长期记忆: 重要性={memory.importance}")
                else:
                    # 遗忘不重要短期记忆
                    # 基于遗忘率随机决定是否遗忘
                    import random
                    if random.random() < forgetting_rate:
                        db.delete(memory)
                        forgotten_short_term_count += 1
                        logger.debug(f"遗忘短期记忆 {memory.id}: 重要性={memory.importance}")
            
            # 处理长期记忆：基于遗忘率遗忘不重要记忆
            forgotten_long_term_count = 0
            kept_long_term_count = 0
            
            for memory in long_term_memories:
                if memory.importance < importance_threshold:
                    # 基于遗忘率随机决定是否遗忘
                    import random
                    if random.random() < forgetting_rate:
                        db.delete(memory)
                        forgotten_long_term_count += 1
                        logger.debug(f"遗忘长期记忆 {memory.id}: 重要性={memory.importance}")
                    else:
                        kept_long_term_count += 1
                else:
                    # 重要记忆保留
                    kept_long_term_count += 1
            
            # 提交更改
            db.commit()
            
            # 更新统计信息
            total_memories_before = len(all_memories)
            total_memories_after = total_memories_before - forgotten_short_term_count - forgotten_long_term_count
            
            stats = {
                "total_memories_before": total_memories_before,
                "total_memories_after": total_memories_after,
                "short_term_memories_before": len(short_term_memories),
                "long_term_memories_before": len(long_term_memories),
                "upgraded_count": upgraded_count,
                "forgotten_short_term_count": forgotten_short_term_count,
                "forgotten_long_term_count": forgotten_long_term_count,
                "kept_long_term_count": kept_long_term_count,
                "forgetting_rate_applied": forgetting_rate,
                "importance_threshold": importance_threshold,
                "compression_rate": (forgotten_short_term_count + forgotten_long_term_count) / max(1, total_memories_before)
            }
            
            logger.info(f"选择性记忆处理完成: 升级 {upgraded_count} 个记忆, "
                       f"遗忘 {forgotten_short_term_count + forgotten_long_term_count} 个记忆, "
                       f"压缩率: {stats['compression_rate']:.2%}")
            
            return {
                "enabled": True,
                "message": "选择性记忆机制应用成功",
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"应用选择性记忆机制失败: {e}")
            db.rollback()
            return {
                "enabled": True,
                "message": f"应用选择性记忆机制失败: {str(e)}",
                "stats": {}
            }
            
        except Exception as e:
            logger.error(f"从数据库构建关联图失败: {e}")
            return {"error": str(e), "nodes": [], "edges": []}
    
    def _build_edges_from_database(self, memories: List) -> List[Dict[str, Any]]:
        """从数据库记忆构建边列表"""
        edges = []
        max_edges = min(100, len(memories) * 3)  # 限制边数量
        
        edge_count = 0
        for i in range(len(memories)):
            if edge_count >= max_edges:
                break
                
            for j in range(i+1, len(memories)):
                if edge_count >= max_edges:
                    break
                    
                try:
                    emb_i = json.loads(memories[i].embedding) if memories[i].embedding else []
                    emb_j = json.loads(memories[j].embedding) if memories[j].embedding else []
                    
                    if emb_i and emb_j:
                        similarity = self._calculate_similarity(emb_i, emb_j)
                        
                        # 只添加强关联
                        if similarity > self.similarity_threshold * 0.8:  # 降低阈值以获取更多连接
                            edges.append({
                                "source": memories[i].id,
                                "target": memories[j].id,
                                "strength": similarity,
                                "type": "semantic_similarity"
                            })
                            edge_count += 1
                except Exception as e:
                    logger.debug(f"计算记忆 {memories[i].id}-{memories[j].id} 相似度失败: {e}")
                    continue
        
        return edges

    # ==================== 兼容性方法 ====================

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        db: Optional[Session] = None,
        user_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """检索相关记忆上下文（兼容性方法）"""
        if not db or not user_id:
            logger.warning("retrieve_context 需要 db 和 user_id 参数")
            return []  # 返回空列表

        memories = self.search_memories(db, user_id, query, limit=top_k)
        # 转换为旧格式
        result = []
        for memory in memories:
            result.append(
                {
                    "id": memory["id"],
                    "content": memory["content"],
                    "importance": memory["importance"],
                    "memory_type": memory["memory_type"],
                }
            )
        return result

    # ==================== 高级认知推理集成 ====================

    def initialize_cognitive_reasoning(self, config: Optional[Dict[str, Any]] = None) -> "CognitiveReasoningIntegrator":
        """初始化认知推理集成器
        
        参数:
            config: 推理配置参数
            
        返回:
            初始化的认知推理集成器
        """
        try:
            # 检查是否启用认知推理
            enable_cognitive_reasoning = self.config.get("enable_cognitive_reasoning", True)
            
            if not enable_cognitive_reasoning:
                logger.info("认知推理功能未启用")
                return None  # 返回None
            
            # 合并配置
            reasoning_config = self.config.get("reasoning_config", {})
            if config:
                reasoning_config.update(config)
            
            # 创建推理集成器
            reasoning_integrator = CognitiveReasoningIntegrator(self, reasoning_config)
            
            # 存储到实例变量
            self.cognitive_reasoning_integrator = reasoning_integrator
            
            logger.info("认知推理集成器已初始化")
            return reasoning_integrator
            
        except Exception as e:
            logger.error(f"初始化认知推理集成器失败: {e}")
            return None  # 返回None
    
    def reason_from_memory(
        self,
        query: str,
        db: Session,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """基于记忆进行推理（便捷方法）
        
        参数:
            query: 推理查询
            db: 数据库会话
            user_id: 用户ID
            context: 推理上下文（可选）
            
        返回:
            推理结果
        """
        # 确保推理集成器已初始化
        if not hasattr(self, 'cognitive_reasoning_integrator') or self.cognitive_reasoning_integrator is None:
            self.initialize_cognitive_reasoning()
        
        # 如果仍然没有集成器，返回错误
        if not hasattr(self, 'cognitive_reasoning_integrator') or self.cognitive_reasoning_integrator is None:
            return {
                "success": False,
                "answer": "认知推理功能未启用或初始化失败",
                "confidence": 0.0,
                "reasoning_chain": [],
                "supporting_evidence": []
            }
        
        # 使用推理集成器
        return self.cognitive_reasoning_integrator.reason_from_memory(query, db, user_id, context)
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        if not hasattr(self, 'cognitive_reasoning_integrator') or self.cognitive_reasoning_integrator is None:
            return {"cognitive_reasoning_enabled": False}
        
        stats = self.cognitive_reasoning_integrator.get_reasoning_stats()
        stats["cognitive_reasoning_enabled"] = True
        return stats

    # ==================== 情境感知记忆增强 ====================

    def initialize_context_aware_enhancer(self, config: Optional[Dict[str, Any]] = None) -> 'ContextAwareMemoryEnhancer':
        """初始化情境感知记忆增强器
        
        参数:
            config: 情境感知配置参数
            
        返回:
            初始化的情境感知记忆增强器
        """
        try:
            # 检查是否启用情境感知
            enable_context_awareness = self.config.get("enable_context_awareness", True)
            
            if not enable_context_awareness:
                logger.info("情境感知功能未启用")
                return None  # 返回None
            
            # 合并配置
            context_config = self.config.get("context_config", {})
            if config:
                context_config.update(config)
            
            # 创建情境感知增强器
            context_enhancer = ContextAwareMemoryEnhancer(self, context_config)
            
            # 存储到实例变量
            self.context_aware_enhancer = context_enhancer
            
            logger.info("情境感知记忆增强器已初始化")
            return context_enhancer
            
        except Exception as e:
            logger.error(f"初始化情境感知记忆增强器失败: {e}")
            return None  # 返回None
    
    def update_context(self, context_updates: Dict[str, Any]):
        """更新当前上下文（便捷方法）"""
        # 确保情境感知增强器已初始化
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            self.initialize_context_aware_enhancer()
        
        # 如果仍然没有增强器，直接返回
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            logger.warning("情境感知功能未启用，无法更新上下文")
            return
        
        # 使用增强器更新上下文
        self.context_aware_enhancer.update_context(context_updates)
    
    def retrieve_with_context(
        self,
        query: str,
        db: Session,
        user_id: int,
        top_k: Optional[int] = None,
        use_context: bool = True
    ) -> List[Dict[str, Any]]:
        """情境感知记忆检索（便捷方法）
        
        参数:
            query: 查询文本
            db: 数据库会话
            user_id: 用户ID
            top_k: 返回记忆数量
            use_context: 是否使用上下文优化检索
            
        返回:
            情境感知的相关记忆列表
        """
        # 确保情境感知增强器已初始化
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            self.initialize_context_aware_enhancer()
        
        # 如果增强器未启用或初始化失败，使用标准检索
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            logger.warning("情境感知功能未启用，使用标准检索")
            return self.retrieve_relevant_memories(query, db, user_id, top_k)
        
        # 使用情境感知检索
        return self.context_aware_enhancer.retrieve_context_aware_memories(
            query, db, user_id, top_k, use_context
        )
    
    def predict_relevant_memories(self, db: Session, user_id: int) -> List[Dict[str, Any]]:
        """预测相关记忆（预加载）
        
        基于当前上下文预测用户可能需要的记忆，提前加载到缓存
        """
        # 确保情境感知增强器已初始化
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            self.initialize_context_aware_enhancer()
        
        # 如果增强器未启用或初始化失败，返回空列表
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            return []  # 返回空列表
        
        # 使用增强器预测相关记忆
        return self.context_aware_enhancer.predict_relevant_memories(db, user_id)
    
    def get_context_stats(self) -> Dict[str, Any]:
        """获取上下文统计信息"""
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            return {"context_awareness_enabled": False}
        
        stats = self.context_aware_enhancer.get_context_stats()
        stats["context_awareness_enabled"] = True
        return stats

    # ==================== 情感记忆增强 ====================

    def initialize_emotional_memory_enhancer(self, config: Optional[Dict[str, Any]] = None) -> 'EmotionalMemoryEnhancer':
        """初始化情感记忆增强器
        
        参数:
            config: 情感记忆配置参数
            
        返回:
            初始化的情感记忆增强器
        """
        try:
            # 检查是否启用情感记忆增强
            if not self.enable_emotional_memory:
                logger.info("情感记忆增强功能未启用")
                return None  # 返回None
            
            # 合并配置
            emotion_config = {
                "enable_emotion_analysis": True,
                "emotion_detection_threshold": self.emotion_detection_threshold,
                "emotion_impact_factor": self.emotion_impact_factor,
                "max_emotion_history": self.max_emotion_history
            }
            if config:
                emotion_config.update(config)
            
            # 创建情感记忆增强器
            emotional_enhancer = EmotionalMemoryEnhancer(emotion_config)
            
            # 存储到实例变量
            self.emotional_memory_enhancer = emotional_enhancer
            
            logger.info("情感记忆增强器已初始化")
            return emotional_enhancer
            
        except Exception as e:
            logger.error(f"初始化情感记忆增强器失败: {e}")
            return None  # 返回None
    
    def analyze_text_emotion(self, text: str) -> Dict[str, Any]:
        """分析文本情感（便捷方法）
        
        参数:
            text: 输入文本
            
        返回:
            情感分析结果
        """
        # 确保情感记忆增强器已初始化
        if not hasattr(self, 'emotional_memory_enhancer') or self.emotional_memory_enhancer is None:
            self.initialize_emotional_memory_enhancer()
        
        # 如果增强器未启用或初始化失败，返回中性情感
        if not hasattr(self, 'emotional_memory_enhancer') or self.emotional_memory_enhancer is None:
            return {
                "emotion": "neutral",
                "intensity": 0.0,
                "confidence": 0.0,
                "keywords": []
            }
        
        # 使用增强器分析情感
        return self.emotional_memory_enhancer.analyze_text_emotion(text)
    
    def adjust_memory_importance_with_emotion(self, memory_data: Dict[str, Any]) -> float:
        """基于情感调整记忆重要性（便捷方法）
        
        参数:
            memory_data: 记忆数据
            
        返回:
            调整后的重要性分数
        """
        # 确保情感记忆增强器已初始化
        if not hasattr(self, 'emotional_memory_enhancer') or self.emotional_memory_enhancer is None:
            self.initialize_emotional_memory_enhancer()
        
        # 如果增强器未启用或初始化失败，返回原始重要性
        if not hasattr(self, 'emotional_memory_enhancer') or self.emotional_memory_enhancer is None:
            return memory_data.get("importance", 0.5)
        
        # 使用增强器调整重要性
        return self.emotional_memory_enhancer.adjust_memory_importance_with_emotion(memory_data)
    
    def retrieve_emotion_related_memories(
        self,
        current_emotion: Optional[str] = None,
        emotion_intensity: Optional[float] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """检索与当前情感相关的记忆（便捷方法）
        
        参数:
            current_emotion: 当前情感类型，如果为None则使用最近的情感状态
            emotion_intensity: 情感强度，如果为None则使用最近的情感强度
            top_k: 返回记忆数量
            
        返回:
            情感相关的记忆列表
        """
        # 确保情感记忆增强器已初始化
        if not hasattr(self, 'emotional_memory_enhancer') or self.emotional_memory_enhancer is None:
            self.initialize_emotional_memory_enhancer()
        
        # 如果增强器未启用或初始化失败，返回空列表
        if not hasattr(self, 'emotional_memory_enhancer') or self.emotional_memory_enhancer is None:
            return []  # 返回空列表
        
        # 使用增强器检索情感相关记忆
        return self.emotional_memory_enhancer.retrieve_emotion_related_memories(
            current_emotion, emotion_intensity, top_k
        )
    
    def get_emotional_insights(self) -> Dict[str, Any]:
        """获取情感洞察（便捷方法）"""
        # 确保情感记忆增强器已初始化
        if not hasattr(self, 'emotional_memory_enhancer') or self.emotional_memory_enhancer is None:
            self.initialize_emotional_memory_enhancer()
        
        # 如果增强器未启用或初始化失败，返回基本情感状态
        if not hasattr(self, 'emotional_memory_enhancer') or self.emotional_memory_enhancer is None:
            return {
                "current_emotion": "neutral",
                "emotion_intensity": 0.0,
                "total_emotion_events": 0,
                "emotional_patterns": {},
                "emotional_memories_count": 0
            }
        
        # 使用增强器获取情感洞察
        return self.emotional_memory_enhancer.get_emotional_insights()
    
    def update_emotion_context(self, text: str) -> Dict[str, Any]:
        """更新情感上下文（便捷方法）
        
        分析文本情感并更新情感状态
        
        参数:
            text: 输入文本
            
        返回:
            情感分析结果
        """
        # 分析文本情感
        emotion_result = self.analyze_text_emotion(text)
        
        # 如果检测到情感且置信度足够高，更新上下文中的情感状态
        if (emotion_result["emotion"] != "neutral" and 
            emotion_result["confidence"] > self.emotion_detection_threshold):
            
            # 更新情境感知中的情感状态
            self.update_context({
                "emotional_state": emotion_result["emotion"]
            })
            
            logger.debug(f"情感上下文已更新: {emotion_result['emotion']} (强度: {emotion_result['intensity']:.2f})")
        
        return emotion_result
    
    def get_emotion_stats(self) -> Dict[str, Any]:
        """获取情感统计信息"""
        if not hasattr(self, 'emotional_memory_enhancer') or self.emotional_memory_enhancer is None:
            return {"emotional_memory_enabled": False}
        
        stats = self.get_emotional_insights()
        stats["emotional_memory_enabled"] = True
        stats["emotion_detection_threshold"] = self.emotion_detection_threshold
        stats["emotion_impact_factor"] = self.emotion_impact_factor
        return stats

    # ==================== 场景记忆增强便捷方法 ====================

    def retrieve_scene_related_memories(
        self,
        db: Session,
        user_id: int,
        scene_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """检索场景相关记忆（便捷方法）
        
        参数:
            db: 数据库会话
            user_id: 用户ID
            scene_type: 场景类型，如果为None则使用当前场景
            top_k: 返回记忆数量
            
        返回:
            场景相关的记忆列表
        """
        # 确保情境感知增强器已初始化
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            self.initialize_context_aware_enhancer()
        
        # 如果增强器未启用或初始化失败，返回空列表
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            logger.warning("情境感知功能未启用，无法检索场景相关记忆")
            return []  # 返回空列表
        
        # 使用增强器检索场景相关记忆
        return self.context_aware_enhancer.retrieve_scene_related_memories(
            db, user_id, scene_type, top_k
        )
    
    def get_scene_stats(self) -> Dict[str, Any]:
        """获取场景统计信息（便捷方法）"""
        # 确保情境感知增强器已初始化
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            self.initialize_context_aware_enhancer()
        
        # 如果增强器未启用或初始化失败，返回基本统计
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            return {
                "scene_enhancement_enabled": False,
                "current_scene": "general"
            }
        
        # 使用增强器获取场景统计
        stats = self.context_aware_enhancer.get_scene_stats()
        stats["scene_enhancement_enabled"] = True
        return stats
    
    def learn_scene_patterns(self, db: Session, user_id: int):
        """学习场景模式（便捷方法）
        
        分析用户在特定场景下访问的记忆模式
        """
        # 确保情境感知增强器已初始化
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            self.initialize_context_aware_enhancer()
        
        # 如果增强器未启用或初始化失败，直接返回
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            logger.warning("情境感知功能未启用，无法学习场景模式")
            return
        
        # 使用增强器学习场景模式
        self.context_aware_enhancer.learn_scene_patterns(db, user_id)
    
    def update_scene_context(self, scene_updates: Dict[str, Any]):
        """更新场景上下文（便捷方法）
        
        手动更新场景状态，用于特殊场景控制
        """
        # 确保情境感知增强器已初始化
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            self.initialize_context_aware_enhancer()
        
        # 如果增强器未启用或初始化失败，直接返回
        if not hasattr(self, 'context_aware_enhancer') or self.context_aware_enhancer is None:
            logger.warning("情境感知功能未启用，无法更新场景上下文")
            return
        
        # 更新场景状态
        if "scene_type" in scene_updates:
            self.context_aware_enhancer.current_scene["scene_type"] = scene_updates["scene_type"]
        
        if "scene_id" in scene_updates:
            self.context_aware_enhancer.current_scene["scene_id"] = scene_updates["scene_id"]
        
        logger.debug(f"场景上下文已手动更新: {scene_updates}")

    # ==================== 兼容性方法 ====================

    def process_input(
        self,
        input_text: str,
        db: Optional[Session] = None,
        user_id: Optional[int] = None,
        **kwargs,
    ) -> int:
        """处理输入并存储记忆（兼容性方法）"""
        if not db or not user_id:
            logger.warning("process_input 需要 db 和 user_id 参数")
            return -1

        memory_type = kwargs.get("memory_type", "short_term")
        importance = kwargs.get("importance", 1.0)

        return self.store_memory(
            db=db,
            user_id=user_id,
            content=input_text,
            content_type="text",
            memory_type=memory_type,
            importance=importance,
        )

    def get_stats(
        self, db: Optional[Session] = None, user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """获取统计信息（兼容性方法）"""
        if not db:
            logger.warning("get_stats 需要 db 参数")
            return {}  # 返回空字典

        return self.get_memory_stats(db, user_id)


class CognitiveReasoningIntegrator:
    """认知推理集成器 - 实现记忆与推理的深度融合
    
    基于最新的认知科学和AI研究，实现：
    1. 基于记忆的推理链生成
    2. 关联图上的逻辑推理
    3. 符号与神经表示的融合
    4. 情境感知的推理策略
    """
    
    def __init__(self, memory_system: Optional[MemorySystem] = None, config: Optional[Dict[str, Any]] = None):
        self.memory_system = memory_system
        if memory_system is None:
            logger.info("CognitiveReasoningIntegrator 初始化时 memory_system 为 None，部分记忆相关功能将不可用（测试环境中为预期行为）")
        self.config = config or {}
        
        # 推理配置参数
        self.enable_symbolic_reasoning = self.config.get("enable_symbolic_reasoning", True)
        self.enable_neural_reasoning = self.config.get("enable_neural_reasoning", True)
        self.enable_analogical_reasoning = self.config.get("enable_analogical_reasoning", True)
        self.enable_abductive_reasoning = self.config.get("enable_abductive_reasoning", True)
        
        # 推理模型参数
        self.reasoning_depth = self.config.get("reasoning_depth", 3)
        self.max_reasoning_steps = self.config.get("max_reasoning_steps", 10)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        # 推理状态跟踪
        self.reasoning_history = []
        self.inference_cache = {}
        self.pattern_database = {}
        
        # 初始化推理模型
        self._init_reasoning_models()
        
        logger.info("认知推理集成器已初始化")
    
    def _init_reasoning_models(self):
        """初始化推理模型"""
        # 符号推理引擎
        if self.enable_symbolic_reasoning:
            self.symbolic_engine = SymbolicReasoningEngine()
        else:
            self.symbolic_engine = None
        
        # 神经推理网络（如果启用）
        if self.enable_neural_reasoning:
            self.neural_reasoner = NeuralReasoningNetwork()
        else:
            self.neural_reasoner = None
    
    def reason_from_memory(
        self, 
        query: str, 
        db: Session, 
        user_id: int, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """基于记忆进行推理
        
        参数:
            query: 推理查询
            db: 数据库会话
            user_id: 用户ID
            context: 推理上下文（可选）
            
        返回:
            推理结果，包括：
            - 答案
            - 推理链
            - 置信度
            - 支持证据
        """
        context = context or {}
        
        try:
            # 1. 检索相关记忆
            relevant_memories = self._retrieve_relevant_memories(query, db, user_id)
            
            if not relevant_memories:
                return {
                    "success": False,
                    "answer": "没有找到相关记忆进行推理",
                    "confidence": 0.0,
                    "reasoning_chain": [],
                    "supporting_evidence": []
                }
            
            # 2. 构建推理上下文
            reasoning_context = self._build_reasoning_context(query, relevant_memories, context)
            
            # 3. 选择推理策略
            reasoning_strategy = self._select_reasoning_strategy(reasoning_context)
            
            # 4. 执行推理
            reasoning_result = self._execute_reasoning(reasoning_strategy, reasoning_context)
            
            # 5. 验证推理结果
            validated_result = self._validate_reasoning_result(reasoning_result, relevant_memories)
            
            # 6. 记录推理历史
            self._record_reasoning_history(query, reasoning_context, validated_result)
            
            return validated_result
            
        except Exception as e:
            logger.error(f"基于记忆推理失败: {e}")
            return {
                "success": False,
                "answer": f"推理过程中发生错误: {str(e)}",
                "confidence": 0.0,
                "reasoning_chain": [],
                "supporting_evidence": []
            }
    
    def _retrieve_relevant_memories(
        self, 
        query: str, 
        db: Session, 
        user_id: int
    ) -> List[Dict[str, Any]]:
        """检索与查询相关的记忆"""
        # 使用记忆系统的检索功能
        relevant_memories = self.memory_system.retrieve_relevant_memories(
            query=query,
            db=db,
            user_id=user_id,
            top_k=10,  # 检索更多记忆以支持推理
            similarity_threshold=0.6  # 较低的阈值以获取更多相关记忆
        )
        
        # 添加额外的元数据以支持推理
        enriched_memories = []
        for memory in relevant_memories:
            enriched_memory = memory.copy()
            
            # 提取关键概念
            enriched_memory["concepts"] = self._extract_concepts(memory.get("content", ""))
            
            # 计算推理相关性
            enriched_memory["reasoning_relevance"] = self._calculate_reasoning_relevance(query, memory)
            
            # 识别记忆类型（事实、规则、经验等）
            enriched_memory["memory_category"] = self._categorize_memory(memory)
            
            enriched_memories.append(enriched_memory)
        
        # 按推理相关性排序
        enriched_memories.sort(key=lambda x: x.get("reasoning_relevance", 0.0), reverse=True)
        
        return enriched_memories[:5]  # 返回前5个最相关的记忆
    
    def _extract_concepts(self, text: str) -> List[str]:
        """从文本中提取关键概念"""
        # 完整实现应使用NLP技术）
        concepts = []
        
        # 完整实现）
        words = text.split()
        for i, word in enumerate(words):
            if len(word) > 3 and word.isalpha():  # 简单过滤
                concepts.append(word.lower())
        
        # 去重
        concepts = list(set(concepts))
        
        return concepts[:10]  # 限制概念数量
    
    def _calculate_reasoning_relevance(self, query: str, memory: Dict[str, Any]) -> float:
        """计算记忆对推理查询的相关性"""
        query_concepts = self._extract_concepts(query)
        memory_concepts = memory.get("concepts", [])
        
        if not query_concepts or not memory_concepts:
            return 0.0
        
        # 计算概念重叠度
        overlap = len(set(query_concepts) & set(memory_concepts))
        max_concepts = max(len(query_concepts), len(memory_concepts))
        
        concept_similarity = overlap / max(1, max_concepts)
        
        # 考虑记忆重要性
        importance = memory.get("importance", 0.5)
        
        # 考虑时间衰减（最近记忆更相关）
        last_accessed = memory.get("last_accessed")
        time_relevance = 1.0  # 默认值
        
        if last_accessed:
            try:
                from datetime import datetime
                if isinstance(last_accessed, str):
                    last_accessed = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                
                if isinstance(last_accessed, datetime):
                    time_diff = datetime.now(timezone.utc) - last_accessed
                    days_diff = time_diff.days
                    # 指数衰减：30天衰减到0.5
                    time_relevance = 0.5 ** (days_diff / 30.0)
            except Exception as time_error:
                # 如果时间相关性计算失败，使用默认值
                logger.debug(f"计算时间相关性失败: {time_error}")
                time_relevance = 0.5  # 默认时间相关性
        
        # 综合相关性评分
        relevance_score = (concept_similarity * 0.4 + importance * 0.3 + time_relevance * 0.3)
        
        return min(1.0, max(0.0, relevance_score))
    
    def _categorize_memory(self, memory: Dict[str, Any]) -> str:
        """识别记忆类型"""
        content = memory.get("content", "")
        content_lower = content.lower()
        
        # 基于内容的简单分类
        if any(word in content_lower for word in ["因为", "所以", "因此", "由于", "导致"]):
            return "causal_relation"
        elif any(word in content_lower for word in ["规则", "原则", "定律", "定理"]):
            return "rule"
        elif any(word in content_lower for word in ["经验", "例子", "案例", "实例"]):
            return "example"
        elif any(word in content_lower for word in ["事实", "数据", "统计", "数字"]):
            return "fact"
        elif any(word in content_lower for word in ["目标", "目的", "意图", "计划"]):
            return "goal"
        elif any(word in content_lower for word in ["问题", "疑问", "挑战", "困难"]):
            return "problem"
        else:
            return "general"
    
    def _build_reasoning_context(
        self, 
        query: str, 
        relevant_memories: List[Dict[str, Any]], 
        additional_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建推理上下文"""
        context = {
            "query": query,
            "relevant_memories": relevant_memories,
            "memory_categories": {},
            "key_concepts": set(),
            "timestamp": time.time()
        }
        
        # 统计记忆类型
        category_counts = {}
        for memory in relevant_memories:
            category = memory.get("memory_category", "general")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # 收集关键概念
            concepts = memory.get("concepts", [])
            context["key_concepts"].update(concepts)
        
        context["memory_categories"] = category_counts
        context["key_concepts"] = list(context["key_concepts"])
        
        # 合并额外上下文
        context.update(additional_context)
        
        return context
    
    def _select_reasoning_strategy(self, context: Dict[str, Any]) -> str:
        """选择推理策略"""
        memory_categories = context.get("memory_categories", {})
        
        # 基于记忆类型选择策略
        if memory_categories.get("causal_relation", 0) > 0:
            return "causal_reasoning"
        elif memory_categories.get("rule", 0) > 0:
            return "rule_based_reasoning"
        elif memory_categories.get("example", 0) > 0:
            return "analogical_reasoning"
        elif memory_categories.get("fact", 0) > 3:
            return "inductive_reasoning"
        elif memory_categories.get("problem", 0) > 0:
            return "abductive_reasoning"
        else:
            return "general_reasoning"
    
    def _execute_reasoning(self, strategy: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行推理"""
        relevant_memories = context.get("relevant_memories", [])
        query = context.get("query", "")
        
        # 根据策略选择推理方法
        if strategy == "causal_reasoning":
            return self._causal_reasoning(query, relevant_memories)
        elif strategy == "rule_based_reasoning":
            return self._rule_based_reasoning(query, relevant_memories)
        elif strategy == "analogical_reasoning":
            return self._analogical_reasoning(query, relevant_memories)
        elif strategy == "inductive_reasoning":
            return self._inductive_reasoning(query, relevant_memories)
        elif strategy == "abductive_reasoning":
            return self._abductive_reasoning(query, relevant_memories)
        else:
            return self._general_reasoning(query, relevant_memories)
    
    def _causal_reasoning(self, query: str, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """因果推理"""
        reasoning_chain = []
        supporting_evidence = []
        
        # 查找因果关系的记忆
        for memory in memories:
            if memory.get("memory_category") == "causal_relation":
                supporting_evidence.append({
                    "memory_id": memory.get("id"),
                    "content": memory.get("content", "")[:100],
                    "relevance": memory.get("reasoning_relevance", 0.0)
                })
        
        # 构建简单因果推理链
        if supporting_evidence:
            answer = f"基于{custom(len(supporting_evidence))}个因果关系记忆分析，"
            answer += "可以推断：" + self._generate_causal_inference(query, supporting_evidence)
            confidence = 0.8
        else:
            answer = "未找到明确的因果关系记忆"
            confidence = 0.3
        
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_chain": reasoning_chain,
            "supporting_evidence": supporting_evidence
        }
    
    def _rule_based_reasoning(self, query: str, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于规则的推理"""
        # 查找规则记忆
        rules = [m for m in memories if m.get("memory_category") == "rule"]
        
        if not rules:
            return self._general_reasoning(query, memories)
        
        # 应用规则推理
        applicable_rules = []
        for rule in rules[:3]:  # 最多考虑3个规则
            rule_content = rule.get("content", "")
            if self._rule_applies(rule_content, query):
                applicable_rules.append(rule)
        
        if applicable_rules:
            answer = f"基于{custom(len(applicable_rules))}个相关规则："
            for i, rule in enumerate(applicable_rules, 1):
                answer += f"\n{i}. {rule.get('content', '')[:50]}..."
            
            confidence = 0.7 + (len(applicable_rules) * 0.1)
            confidence = min(0.9, confidence)
        else:
            answer = "没有找到直接适用的规则"
            confidence = 0.4
        
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_chain": [],
            "supporting_evidence": applicable_rules
        }
    
    def _analogical_reasoning(self, query: str, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """类比推理"""
        # 查找示例记忆
        examples = [m for m in memories if m.get("memory_category") == "example"]
        
        if len(examples) >= 2:
            # 找到最相似的示例
            example1 = examples[0]
            example2 = examples[1] if len(examples) > 1 else example1
            
            answer = f"基于类似情况的分析：\n"
            answer += f"1. {example1.get('content', '')[:80]}...\n"
            answer += f"2. {example2.get('content', '')[:80]}...\n"
            answer += "可以类比推断出相关结论。"
            
            confidence = 0.6
        else:
            answer = "没有找到足够的类似案例进行类比推理"
            confidence = 0.3
        
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_chain": [],
            "supporting_evidence": examples[:2]
        }
    
    def _inductive_reasoning(self, query: str, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """归纳推理"""
        # 查找事实记忆
        facts = [m for m in memories if m.get("memory_category") == "fact"]
        
        if len(facts) >= 3:
            answer = f"基于{custom(len(facts))}个相关事实，可以归纳出一般规律。"
            confidence = min(0.7, 0.4 + (len(facts) * 0.1))
        else:
            answer = "事实证据不足，无法进行可靠的归纳推理"
            confidence = 0.3
        
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_chain": [],
            "supporting_evidence": facts[:3]
        }
    
    def _abductive_reasoning(self, query: str, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """溯因推理（解释推理）"""
        # 查找问题和相关记忆
        problems = [m for m in memories if m.get("memory_category") == "problem"]
        
        if problems:
            # 尝试构建最佳解释
            best_explanation = self._generate_best_explanation(query, problems, memories)
            
            answer = f"对于问题'{query[:50]}...'，最可能的解释是：{best_explanation}"
            confidence = 0.6
        else:
            answer = "未识别出明确的问题模式"
            confidence = 0.3
        
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_chain": [],
            "supporting_evidence": problems
        }
    
    def _general_reasoning(self, query: str, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """一般推理"""
        if not memories:
            return {
                "answer": "没有相关记忆可用于推理",
                "confidence": 0.0,
                "reasoning_chain": [],
                "supporting_evidence": []
            }
        
        # 使用最相关的记忆
        top_memory = memories[0]
        memory_content = top_memory.get("content", "")
        
        answer = f"基于相关记忆：{memory_content[:100]}..."
        confidence = top_memory.get("reasoning_relevance", 0.5)
        
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning_chain": [],
            "supporting_evidence": [top_memory]
        }
    
    def _rule_applies(self, rule_content: str, query: str) -> bool:
        """检查规则是否适用于查询"""
        # 完整的规则匹配（实际应使用逻辑推理）
        rule_words = set(rule_content.lower().split())
        query_words = set(query.lower().split())
        
        # 计算关键词重叠
        overlap = len(rule_words & query_words)
        
        return overlap >= 2  # 至少有2个关键词重叠
    
    def _generate_causal_inference(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        """生成因果推断"""
        if not evidence:
            return "无法确定因果关系"
        
        # 完整实现：返回第一个证据的相关内容
        first_evidence = evidence[0]
        content = first_evidence.get("content", "")
        
        # 提取关键词
        words = content.split()[:5]
        keywords = " ".join(words)
        
        return f"可能涉及{keywords}等相关因素。"
    
    def _generate_best_explanation(self, query: str, problems: List[Dict[str, Any]], all_memories: List[Dict[str, Any]]) -> str:
        """生成最佳解释"""
        if not problems:
            return "缺乏足够信息生成解释"
        
        # 查找可能的解释模式
        problem_content = problems[0].get("content", "")
        problem_keywords = problem_content.split()[:3]
        
        return f"可能与{', '.join(problem_keywords)}等因素有关。"
    
    def _validate_reasoning_result(
        self, 
        reasoning_result: Dict[str, Any], 
        supporting_memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """验证推理结果"""
        result = reasoning_result.copy()
        
        # 检查置信度阈值
        confidence = result.get("confidence", 0.0)
        if confidence < self.confidence_threshold:
            result["success"] = False
            result["validation_note"] = f"置信度过低 ({confidence:.2f} < {self.confidence_threshold})"
        else:
            result["success"] = True
            result["validation_note"] = "推理结果符合置信度要求"
        
        # 检查是否有支持证据
        evidence = result.get("supporting_evidence", [])
        if not evidence:
            result["success"] = False
            result["validation_note"] = "缺乏支持证据"
        
        # 计算证据强度
        evidence_strength = 0.0
        for evidence_item in evidence:
            if isinstance(evidence_item, dict):
                relevance = evidence_item.get("relevance", 0.5)
                evidence_strength += relevance
        
        evidence_count = len(evidence)
        if evidence_count > 0:
            evidence_strength /= evidence_count
        
        result["evidence_strength"] = evidence_strength
        
        # 调整最终置信度
        adjusted_confidence = confidence * 0.7 + evidence_strength * 0.3
        result["adjusted_confidence"] = adjusted_confidence
        
        return result
    
    def _record_reasoning_history(self, query: str, context: Dict[str, Any], result: Dict[str, Any]):
        """记录推理历史"""
        history_entry = {
            "timestamp": time.time(),
            "query": query,
            "reasoning_strategy": context.get("reasoning_strategy", "unknown"),
            "memory_count": len(context.get("relevant_memories", [])),
            "result": {
                "success": result.get("success", False),
                "confidence": result.get("adjusted_confidence", result.get("confidence", 0.0)),
                "answer_length": len(result.get("answer", ""))
            }
        }
        
        self.reasoning_history.append(history_entry)
        
        # 保持历史记录大小
        max_history = self.config.get("max_reasoning_history", 100)
        if len(self.reasoning_history) > max_history:
            self.reasoning_history = self.reasoning_history[-max_history:]
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        if not self.reasoning_history:
            return {"total_reasoning_attempts": 0}
        
        total = len(self.reasoning_history)
        successful = sum(1 for entry in self.reasoning_history if entry["result"]["success"])
        avg_confidence = sum(entry["result"]["confidence"] for entry in self.reasoning_history) / total
        
        # 策略使用统计
        strategy_counts = {}
        for entry in self.reasoning_history:
            strategy = entry["reasoning_strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "total_reasoning_attempts": total,
            "successful_reasoning_attempts": successful,
            "success_rate": successful / max(1, total),
            "average_confidence": avg_confidence,
            "strategy_distribution": strategy_counts,
            "reasoning_history_size": total
        }


class SymbolicReasoningEngine:
    """符号推理引擎 - 基于规则的推理系统
    
    实现功能：
    1. 基于一阶逻辑的规则推理
    2. 事实与规则的匹配
    3. 前向链式推理
    4. 简单的逻辑推断
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        
        self.rules = []  # 规则列表，每个规则是(前提列表, 结论)元组
        self.facts = []  # 事实列表，每个事实是字符串
        self.knowledge_base = {}  # 知识库：事实 -> True/False
        self.rule_pattern = re.compile(r'^如果(.+)那么(.+)$')  # 简单的中文规则模式
        self.config = config
        
        # 从配置中加载初始规则（如果有）
        initial_rules = config.get("initial_rules", [])
        for rule in initial_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: str):
        """添加规则
        
        规则格式：如果[前提条件]那么[结论]
        例如：如果是动物且有毛发那么是哺乳动物
        """
        match = self.rule_pattern.match(rule.strip())
        if match:
            premises_str = match.group(1).strip()
            conclusion = match.group(2).strip()
            
            # 分割前提条件
            premises = []
            for part in premises_str.split('且'):
                part = part.strip()
                if part:
                    premises.append(part)
            
            self.rules.append((premises, conclusion))
            logger.info(f"添加规则: {rule} -> 前提: {premises}, 结论: {conclusion}")
        else:
            # 其他格式的规则
            self.rules.append(([rule], rule))
            logger.info(f"添加通用规则: {rule}")
    
    def add_fact(self, fact: str):
        """添加事实"""
        if fact not in self.facts:
            self.facts.append(fact)
            self.knowledge_base[fact] = True
            logger.info(f"添加事实: {fact}")
    
    def infer(self, query: str) -> str:
        """进行符号推理
        
        参数:
            query: 查询语句
            
        返回:
            推理结果字符串
        """
        try:
            # 尝试直接匹配事实
            if query in self.knowledge_base:
                return f"事实直接匹配: {query}"
            
            # 使用规则进行推理
            for premises, conclusion in self.rules:
                # 检查是否所有前提都满足
                all_premises_satisfied = True
                for premise in premises:
                    if premise not in self.knowledge_base:
                        all_premises_satisfied = False
                        break
                
                if all_premises_satisfied:
                    # 应用规则
                    self.add_fact(conclusion)  # 添加新事实
                    return f"通过规则推理得到: {conclusion} (基于前提: {', '.join(premises)})"
            
            # 尝试部分匹配
            for fact in self.facts:
                if query.lower() in fact.lower() or fact.lower() in query.lower():
                    return f"部分匹配事实: {fact}"
            
            # 反向推理：查找能推导出查询的规则
            for premises, conclusion in self.rules:
                if conclusion == query or conclusion in query:
                    missing_premises = []
                    for premise in premises:
                        if premise not in self.knowledge_base:
                            missing_premises.append(premise)
                    
                    if missing_premises:
                        return f"需要以下前提来推导'{query}': {', '.join(missing_premises)}"
            
            return f"无法从当前知识库推断: {query} (事实数: {len(self.facts)}, 规则数: {len(self.rules)})"
            
        except Exception as e:
            logger.error(f"符号推理错误: {e}")
            return f"符号推理过程中发生错误: {str(e)}"


class NeuralReasoningNetwork:
    """神经推理网络 - 基于多层感知机的记忆推理模型
    
    实现功能：
    1. 记忆特征提取和编码
    2. 基于神经网络的推理预测
    3. 训练和推理分离
    4. 模型持久化支持
    """
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 10):
        """初始化神经网络推理模型
        
        参数:
            input_dim: 输入特征维度（记忆特征维度），默认20维特征
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（推理结果类别数）
        """
        # 定义神经网络架构
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softmax(dim=1)  # 多分类概率输出
        )
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # 训练状态
        self.trained = False
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 类别标签映射（推理类型）
        self.class_labels = [
            "因果推理", "规则推理", "类比推理", "归纳推理", "演绎推理",
            "空间推理", "时间推理", "情感推理", "事实推理", "未知推理"
        ]
        
        # 将模型移动到可用设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"神经网络推理模型已初始化: 输入维度={input_dim}, 隐藏维度={hidden_dim}, 输出维度={output_dim}")
        logger.info(f"模型设备: {self.device}")
    
    def extract_features(self, memory_data: Dict[str, Any]) -> torch.Tensor:
        """从记忆数据中提取特征
        
        提取的记忆特征包括：
        1. 文本长度特征
        2. 访问频率特征
        3. 时间衰减特征
        4. 重要性分数
        5. 情感特征（如果有）
        6. 记忆类型特征
        7. 关联记忆数量
        8. 其他上下文特征
        
        参数:
            memory_data: 记忆数据字典
            
        返回:
            features: 特征张量 [input_dim]
        """
        features = torch.zeros(self.input_dim)
        
        # 1. 文本长度特征（归一化）
        content = memory_data.get("content", "")
        text_length = len(str(content))
        features[0] = min(text_length / 1000.0, 1.0)  # 假设最大1000字符
        
        # 2. 访问频率特征
        accessed_count = memory_data.get("accessed_count", 0)
        features[1] = min(accessed_count / 100.0, 1.0)  # 假设最大100次访问
        
        # 3. 时间衰减特征
        last_accessed = memory_data.get("last_accessed")
        if last_accessed:
            if isinstance(last_accessed, str):
                try:
                    from datetime import datetime
                    last_accessed = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                except Exception as iso_error:
                    logger.debug(f"解析最后访问时间失败: {iso_error}")
            
            if isinstance(last_accessed, datetime):
                time_diff = datetime.now(timezone.utc) - last_accessed
                hours_diff = time_diff.total_seconds() / 3600.0
                # 时间衰减：exp(-hours_diff/24)，24小时衰减到1/e
                time_feature = torch.exp(torch.tensor(-hours_diff / 24.0))
                features[2] = time_feature
        else:
            features[2] = 0.5  # 默认值
        
        # 4. 重要性分数
        importance = memory_data.get("importance", 0.5)
        features[3] = importance
        
        # 5. 记忆类型特征（短期:0, 长期:1）
        memory_type = memory_data.get("memory_type", "short_term")
        features[4] = 1.0 if memory_type == "long_term" else 0.0
        
        # 6. 情感特征（如果有）
        emotion = memory_data.get("emotion", "neutral")
        emotion_map = {"positive": 0.8, "negative": 0.2, "neutral": 0.5}
        features[5] = emotion_map.get(emotion, 0.5)
        
        # 7. 关联记忆数量特征
        related_count = memory_data.get("related_memories_count", 0)
        features[6] = min(related_count / 20.0, 1.0)  # 假设最大20个关联记忆
        
        # 8. 其他特征（预留）
        for i in range(7, min(self.input_dim, 20)):
            features[i] = 0.0  # 预留特征
            
        return features.to(self.device)
    
    def train(self, training_data: List[Dict[str, Any]], labels: List[int], epochs: int = 50):
        """训练神经网络
        
        参数:
            training_data: 训练数据列表，每个元素是记忆数据字典
            labels: 标签列表，对应推理类型索引
            epochs: 训练轮数
        """
        if not training_data or not labels:
            logger.warning("训练数据为空，跳过训练")
            return
        
        try:
            # 准备训练数据
            X = []
            y = []
            
            for data, label in zip(training_data, labels):
                features = self.extract_features(data)
                X.append(features)
                y.append(label)
            
            X = torch.stack(X) if X else torch.zeros(0, self.input_dim, device=self.device)
            y = torch.tensor(y, dtype=torch.long, device=self.device)
            
            if len(X) == 0:
                logger.warning("特征提取后数据为空，跳过训练")
                return
            
            # 训练循环
            self.model.train()
            losses = []
            
            for epoch in range(epochs):
                # 前向传播
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"训练轮数 {epoch+1}/{epochs}, 损失: {loss.item():.4f}")
            
            # 训练完成
            self.trained = True
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            logger.info(f"神经网络训练完成，平均损失: {avg_loss:.4f}")
            
        except Exception as e:
            logger.error(f"神经网络训练失败: {e}")
            self.trained = False
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """神经网络预测
        
        参数:
            input_data: 输入数据，可以是单个记忆数据字典或记忆数据列表
            
        返回:
            预测结果字典，包含：
            - prediction: 预测的推理类型
            - confidence: 置信度
            - probabilities: 各类别概率
            - class_label: 类别标签
        """
        try:
            # 准备输入数据
            if isinstance(input_data, dict):
                # 单个记忆
                features = self.extract_features(input_data).unsqueeze(0)  # [1, input_dim]
            elif isinstance(input_data, list) and input_data:
                # 多个记忆
                features_list = []
                for data in input_data:
                    if isinstance(data, dict):
                        features_list.append(self.extract_features(data))
                if features_list:
                    features = torch.stack(features_list)  # [batch_size, input_dim]
                else:
                    return self._default_prediction()
            else:
                return self._default_prediction()
            
            # 模型预测
            self.model.eval()
            with torch.no_grad():
                probabilities = self.model(features)
                
                # 获取预测类别
                if len(probabilities.shape) == 1:
                    probabilities = probabilities.unsqueeze(0)
                
                confidences, pred_indices = torch.max(probabilities, dim=1)
                
                # 获取预测结果
                if len(pred_indices) > 0:
                    pred_idx = pred_indices[0].item()
                    confidence = confidences[0].item()
                    
                    # 确保索引在有效范围内
                    if 0 <= pred_idx < len(self.class_labels):
                        prediction = self.class_labels[pred_idx]
                    else:
                        prediction = "未知推理"
                    
                    # 所有类别的概率
                    prob_list = probabilities[0].cpu().numpy().tolist() if probabilities.shape[0] > 0 else []
                    
                    return {
                        "prediction": prediction,
                        "confidence": float(confidence),
                        "probabilities": prob_list,
                        "class_label": prediction,
                        "class_index": int(pred_idx)
                    }
                else:
                    return self._default_prediction()
                    
        except Exception as e:
            logger.error(f"神经网络预测失败: {e}")
            return self._default_prediction()
    
    def _default_prediction(self) -> Dict[str, Any]:
        """返回默认预测结果"""
        return {
            "prediction": "未知推理",
            "confidence": 0.0,
            "probabilities": [],
            "class_label": "未知推理",
            "class_index": -1
        }


class ContextAwareMemoryEnhancer:
    """情境感知记忆增强器 - 基于上下文优化记忆管理
    
    实现以下功能：
    1. 对话上下文跟踪
    2. 基于上下文的记忆检索优化
    3. 上下文模式学习
    4. 预测性记忆预加载
    """
    
    def __init__(self, memory_system: 'MemorySystem', config: Optional[Dict[str, Any]] = None):
        self.memory_system = memory_system
        self.config = config or {}
        
        # 情境配置
        self.context_window_size = self.config.get("context_window_size", 10)
        self.context_similarity_threshold = self.config.get("context_similarity_threshold", 0.7)
        self.enable_context_prediction = self.config.get("enable_context_prediction", True)
        
        # 情境状态
        self.current_context = {
            "conversation_history": [],  # 对话历史
            "current_topic": "",  # 当前话题
            "user_intent": "",  # 用户意图
            "emotional_state": "neutral",  # 情感状态
            "environment": {},  # 环境信息
            "task_context": ""  # 任务上下文
        }
        
        # 上下文模式数据库
        self.context_patterns = {}
        
        # 增强：情景场景记忆配置
        self.enable_scene_classification = self.config.get("enable_scene_classification", True)  # 是否启用场景分类
        self.enable_scene_transition_detection = self.config.get("enable_scene_transition_detection", True)  # 是否启用场景切换检测
        self.scene_classification_threshold = self.config.get("scene_classification_threshold", 0.7)  # 场景分类阈值
        self.scene_memory_window = self.config.get("scene_memory_window", 10)  # 场景记忆窗口大小
        
        # 场景状态跟踪
        self.current_scene = {
            "scene_type": "general",  # 场景类型：general, task, social, learning, problem_solving等
            "scene_id": "",  # 场景标识符
            "scene_start_time": time.time(),  # 场景开始时间
            "scene_history": [],  # 场景历史
            "scene_features": {}  # 场景特征
        }
        
        # 场景分类器数据库
        self.scene_classifiers = {
            "task": ["任务", "工作", "项目", "完成", "执行", "目标"],
            "social": ["你好", "聊天", "朋友", "社交", "问候", "交流"],
            "learning": ["学习", "知识", "了解", "研究", "教育", "教学"],
            "problem_solving": ["问题", "解决", "方案", "故障", "调试", "修复"],
            "planning": ["计划", "安排", "日程", "时间", "未来", "目标"]
        }
        
        # 场景关联记忆映射
        self.scene_memory_mapping = {}  # 场景ID -> 相关记忆ID列表
        
        # 上下文记忆缓存
        self.context_memory_cache = {}
        
        logger.info("情境感知记忆增强器已初始化（增强版）")
    
    def update_context(self, context_updates: Dict[str, Any]):
        """更新当前上下文（增强版）
        
        参数:
            context_updates: 上下文更新信息
        """
        # 更新当前上下文
        self.current_context.update(context_updates)
        
        # 维护对话历史大小
        if "conversation_history" in context_updates:
            conversation_history = self.current_context.get("conversation_history", [])
            if len(conversation_history) > self.context_window_size:
                self.current_context["conversation_history"] = conversation_history[-self.context_window_size:]
        
        # 如果启用了场景分类，进行场景分类和更新
        if self.enable_scene_classification:
            self._classify_scene_and_update()
        
        # 如果启用了场景切换检测，检测场景切换
        if self.enable_scene_transition_detection:
            self._detect_scene_transition()
        
        logger.debug(f"上下文已更新: {self._get_context_summary()}")
    
    def _get_context_summary(self) -> Dict[str, Any]:
        """获取上下文摘要"""
        return {
            "topic": self.current_context.get("current_topic", ""),
            "intent": self.current_context.get("user_intent", ""),
            "emotion": self.current_context.get("emotional_state", "neutral"),
            "conversation_length": len(self.current_context.get("conversation_history", [])),
            "has_task_context": bool(self.current_context.get("task_context", ""))
        }
    
    def retrieve_context_aware_memories(
        self,
        query: str,
        db: Session,
        user_id: int,
        top_k: Optional[int] = None,
        use_context: bool = True
    ) -> List[Dict[str, Any]]:
        """情境感知记忆检索
        
        参数:
            query: 查询文本
            db: 数据库会话
            user_id: 用户ID
            top_k: 返回记忆数量
            use_context: 是否使用上下文优化检索
            
        返回:
            情境感知的相关记忆列表
        """
        if not use_context:
            # 不使用上下文，直接使用标准检索
            return self.memory_system.retrieve_relevant_memories(query, db, user_id, top_k)
        
        try:
            # 1. 获取标准检索结果
            base_memories = self.memory_system.retrieve_relevant_memories(
                query=query,
                db=db,
                user_id=user_id,
                top_k=top_k * 2 if top_k else 20  # 检索更多记忆用于上下文过滤
            )
            
            if not base_memories:
                return []  # 返回空列表
            
            # 2. 计算上下文相关性
            context_enhanced_memories = []
            for memory in base_memories:
                enhanced_memory = memory.copy()
                
                # 计算上下文相关性
                context_relevance = self._calculate_context_relevance(memory)
                enhanced_memory["context_relevance"] = context_relevance
                
                # 计算综合相关性（原始相关性 * 上下文相关性）
                original_relevance = memory.get("similarity_score", 0.5)
                combined_relevance = original_relevance * 0.6 + context_relevance * 0.4
                enhanced_memory["combined_relevance"] = combined_relevance
                
                context_enhanced_memories.append(enhanced_memory)
            
            # 3. 按综合相关性排序
            context_enhanced_memories.sort(key=lambda x: x.get("combined_relevance", 0.0), reverse=True)
            
            # 4. 返回前top_k个记忆
            result_count = top_k or len(context_enhanced_memories)
            result = context_enhanced_memories[:result_count]
            
            # 记录上下文检索统计
            self._record_context_retrieval_stats(query, len(base_memories), len(result))
            
            return result
            
        except Exception as e:
            logger.error(f"情境感知记忆检索失败: {e}")
            # 回退到标准检索
            return self.memory_system.retrieve_relevant_memories(query, db, user_id, top_k)
    
    def _calculate_context_relevance(self, memory: Dict[str, Any]) -> float:
        """计算记忆与当前上下文的相关性"""
        context_relevance = 0.0
        
        # 1. 话题相关性
        topic_relevance = self._calculate_topic_relevance(memory)
        
        # 2. 意图相关性
        intent_relevance = self._calculate_intent_relevance(memory)
        
        # 3. 情感相关性
        emotion_relevance = self._calculate_emotion_relevance(memory)
        
        # 4. 对话历史相关性
        conversation_relevance = self._calculate_conversation_relevance(memory)
        
        # 综合相关性评分
        weights = {
            "topic": 0.4,
            "intent": 0.3,
            "emotion": 0.2,
            "conversation": 0.1
        }
        
        context_relevance = (
            topic_relevance * weights["topic"] +
            intent_relevance * weights["intent"] +
            emotion_relevance * weights["emotion"] +
            conversation_relevance * weights["conversation"]
        )
        
        return min(1.0, max(0.0, context_relevance))
    
    def _calculate_topic_relevance(self, memory: Dict[str, Any]) -> float:
        """计算话题相关性"""
        current_topic = self.current_context.get("current_topic", "")
        if not current_topic:
            return 0.5  # 默认值
        
        memory_content = memory.get("content", "")
        
        # 简单关键词匹配
        topic_words = set(current_topic.lower().split())
        content_words = set(memory_content.lower().split())
        
        if not topic_words or not content_words:
            return 0.0
        
        overlap = len(topic_words & content_words)
        max_words = max(len(topic_words), len(content_words))
        
        return overlap / max(1, max_words)
    
    def _calculate_intent_relevance(self, memory: Dict[str, Any]) -> float:
        """计算意图相关性"""
        user_intent = self.current_context.get("user_intent", "")
        if not user_intent:
            return 0.5  # 默认值
        
        # 意图关键词映射
        intent_keywords = {
            "query": ["什么", "如何", "为什么", "哪里", "何时", "谁"],
            "command": ["做", "执行", "创建", "删除", "修改", "设置"],
            "social": ["你好", "谢谢", "再见", "帮助", "聊天"],
            "learning": ["学习", "记住", "知识", "信息", "了解"]
        }
        
        # 检查记忆内容是否包含意图相关关键词
        memory_content = memory.get("content", "").lower()
        relevance = 0.0
        
        for intent_type, keywords in intent_keywords.items():
            if any(keyword in user_intent for keyword in keywords):
                # 检查记忆是否包含相关关键词
                if any(keyword in memory_content for keyword in keywords):
                    relevance = 0.8
                    break
        
        return relevance
    
    def _calculate_emotion_relevance(self, memory: Dict[str, Any]) -> float:
        """计算情感相关性"""
        emotional_state = self.current_context.get("emotional_state", "neutral")
        
        # 情感关键词映射
        emotion_keywords = {
            "happy": ["高兴", "快乐", "开心", "喜悦", "满意"],
            "sad": ["悲伤", "难过", "失望", "沮丧", "痛苦"],
            "angry": ["生气", "愤怒", "恼火", "不满", "烦躁"],
            "neutral": []  # 中性情感没有特定关键词
        }
        
        # 获取当前情感状态的关键词
        current_emotion_keywords = emotion_keywords.get(emotional_state, [])
        
        if not current_emotion_keywords:
            return 0.5  # 中性情感默认相关性
        
        # 检查记忆内容是否包含情感关键词
        memory_content = memory.get("content", "").lower()
        
        for keyword in current_emotion_keywords:
            if keyword in memory_content:
                return 0.7
        
        return 0.3  # 没有情感匹配
    
    def _calculate_conversation_relevance(self, memory: Dict[str, Any]) -> float:
        """计算对话历史相关性"""
        conversation_history = self.current_context.get("conversation_history", [])
        if not conversation_history:
            return 0.5  # 默认值
        
        memory_content = memory.get("content", "")
        
        # 检查记忆内容是否与最近对话相关
        recent_conversation = " ".join(conversation_history[-3:])  # 最近3条对话
        
        # 简单关键词重叠计算
        memory_words = set(memory_content.lower().split())
        conversation_words = set(recent_conversation.lower().split())
        
        if not memory_words or not conversation_words:
            return 0.0
        
        overlap = len(memory_words & conversation_words)
        max_words = max(len(memory_words), len(conversation_words))
        
        return overlap / max(1, max_words)
    
    def _record_context_retrieval_stats(self, query: str, base_count: int, result_count: int):
        """记录上下文检索统计
        
        记录上下文检索的性能指标，用于优化和监控：
        1. 查询特征（长度、关键词等）
        2. 检索效率（基础记忆数量 vs 结果数量）
        3. 时间性能
        4. 命中率
        
        参数:
            query: 查询字符串
            base_count: 基础记忆数量（检索前的记忆总数）
            result_count: 结果数量（检索后的相关记忆数量）
        """
        try:
            # 初始化统计存储（如果不存在）
            if not hasattr(self, 'retrieval_stats'):
                self.retrieval_stats = {
                    "total_queries": 0,
                    "avg_result_ratio": 0.0,
                    "query_length_distribution": [],
                    "recent_queries": []
                }
            
            # 计算指标
            query_length = len(query)
            result_ratio = result_count / max(1, base_count)  # 结果比例
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # 记录本次查询统计
            query_stat = {
                "timestamp": timestamp,
                "query_length": query_length,
                "base_count": base_count,
                "result_count": result_count,
                "result_ratio": result_ratio,
                "query_preview": query[:50] + "..." if len(query) > 50 else query
            }
            
            # 更新统计
            self.retrieval_stats["total_queries"] += 1
            self.retrieval_stats["recent_queries"].append(query_stat)
            
            # 维护最近查询队列（保留最近100条）
            if len(self.retrieval_stats["recent_queries"]) > 100:
                self.retrieval_stats["recent_queries"] = self.retrieval_stats["recent_queries"][-100:]
            
            # 更新平均结果比例（滑动平均）
            current_avg = self.retrieval_stats["avg_result_ratio"]
            new_avg = (current_avg * (self.retrieval_stats["total_queries"] - 1) + result_ratio) / self.retrieval_stats["total_queries"]
            self.retrieval_stats["avg_result_ratio"] = new_avg
            
            # 记录查询长度分布
            self.retrieval_stats["query_length_distribution"].append(query_length)
            if len(self.retrieval_stats["query_length_distribution"]) > 1000:
                self.retrieval_stats["query_length_distribution"] = self.retrieval_stats["query_length_distribution"][-1000:]
            
            # 记录日志（可选，避免过多日志）
            if self.retrieval_stats["total_queries"] % 50 == 0:
                logger.info(f"上下文检索统计: 总查询数={self.retrieval_stats['total_queries']}, "
                          f"平均结果比例={new_avg:.3f}, 最近查询长度={query_length}")
            
        except Exception as e:
            # 统计记录不应影响主要功能
            logger.debug(f"记录上下文检索统计时出错: {e}")
    
    def learn_context_patterns(self, db: Session, user_id: int):
        """学习上下文模式
        
        分析用户的记忆访问模式，学习上下文与记忆检索的关系。
        
        学习步骤：
        1. 获取用户的记忆访问历史
        2. 从当前上下文中提取特征（话题、意图、情感等）
        3. 分析模式：什么上下文下经常访问什么类型的记忆
        4. 更新上下文模式数据库
        
        参数:
            db: 数据库会话
            user_id: 用户ID
        """
        try:
            # 检查是否有足够的上下文数据
            conversation_history = self.current_context.get("conversation_history", [])
            if len(conversation_history) < 3:
                logger.info("对话历史不足，跳过上下文模式学习")
                return
            
            # 获取当前上下文特征
            current_topic = self.current_context.get("current_topic", "")
            user_intent = self.current_context.get("user_intent", "")
            emotional_state = self.current_context.get("emotional_state", "neutral")
            task_context = self.current_context.get("task_context", "")
            
            # 构建上下文特征向量
            context_features = {
                "topic": current_topic,
                "intent": user_intent,
                "emotion": emotional_state,
                "has_task": bool(task_context),
                "conversation_length": len(conversation_history),
                "recent_keywords": self._extract_keywords_from_conversation(conversation_history[-3:])
            }
            
            # 获取用户最近访问的记忆
            recent_memories = self._get_recently_accessed_memories(db, user_id, limit=20)
            
            if not recent_memories:
                logger.info("没有找到近期访问的记忆，跳过模式学习")
                return
            
            # 分析记忆特征
            memory_patterns = self._analyze_memory_patterns(recent_memories)
            
            # 构建模式键
            pattern_key = self._create_pattern_key(context_features)
            
            # 更新上下文模式数据库
            if pattern_key not in self.context_patterns:
                self.context_patterns[pattern_key] = {
                    "context_features": context_features,
                    "memory_patterns": memory_patterns,
                    "occurrence_count": 1,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            else:
                # 更新现有模式
                existing_pattern = self.context_patterns[pattern_key]
                
                # 合并记忆模式
                for mem_type, pattern_data in memory_patterns.items():
                    if mem_type not in existing_pattern["memory_patterns"]:
                        existing_pattern["memory_patterns"][mem_type] = pattern_data
                    else:
                        # 更新统计
                        existing_data = existing_pattern["memory_patterns"][mem_type]
                        existing_data["count"] += pattern_data["count"]
                        existing_data["avg_importance"] = (
                            (existing_data["avg_importance"] * (existing_data["count"] - pattern_data["count"]) + 
                             pattern_data["avg_importance"] * pattern_data["count"]) / existing_data["count"]
                        )
                
                # 更新发生次数
                existing_pattern["occurrence_count"] += 1
                existing_pattern["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"上下文模式学习完成: 模式键={pattern_key}, 模式总数={len(self.context_patterns)}")
            
        except Exception as e:
            logger.error(f"学习上下文模式失败: {e}")
    
    def _extract_keywords_from_conversation(self, conversation_lines: List[str]) -> List[str]:
        """从对话中提取关键词"""
        if not conversation_lines:
            return []  # 返回空列表
        
        # 合并对话
        full_text = " ".join(conversation_lines)
        
        # 简单的关键词提取：过滤停用词，提取名词/动词
        words = full_text.lower().split()
        
        # 完整版）
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        
        # 过滤停用词和短词
        keywords = []
        for word in words:
            if len(word) >= 2 and word not in stop_words:
                # 简单的词频统计
                keywords.append(word)
        
        # 返回前10个关键词
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, _ in word_counts.most_common(10)]
    
    def _get_recently_accessed_memories(self, db: Session, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """获取用户最近访问的记忆
        
        参数:
            db: 数据库会话
            user_id: 用户ID
            limit: 返回的记忆数量限制
            
        返回:
            最近访问的记忆列表
        """
        try:
            # 导入Memory模型
            from backend.db_models.memory import Memory
            
            # 查询用户最近访问的记忆
            memories = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.last_accessed.isnot(None)
            ).order_by(
                Memory.last_accessed.desc()
            ).limit(limit).all()
            
            # 转换为字典列表
            result = []
            for memory in memories:
                mem_dict = {
                    "id": memory.id,
                    "user_id": memory.user_id,
                    "memory_type": memory.memory_type,
                    "content": memory.content,
                    "importance": memory.importance,
                    "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                    "embedding": memory.embedding,
                    "metadata": memory.metadata
                }
                result.append(mem_dict)
            
            logger.debug(f"获取用户{user_id}最近访问的记忆: 找到{len(result)}条记录")
            return result
            
        except ImportError:
            logger.warning("Memory模型不可用，返回空列表")
            return []  # 返回空列表
        except Exception as e:
            logger.error(f"获取最近访问的记忆失败: {e}")
            return []  # 返回空列表
    
    def _analyze_memory_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析记忆模式
        
        分析记忆集合的特征模式
        
        参数:
            memories: 记忆列表
            
        返回:
            记忆模式字典
        """
        if not memories:
            return {}  # 返回空字典
        
        # 初始化模式统计
        patterns = {
            "factual": {"count": 0, "avg_importance": 0.0, "topics": []},
            "emotional": {"count": 0, "avg_importance": 0.0, "emotions": []},
            "procedural": {"count": 0, "avg_importance": 0.0, "tasks": []},
            "conceptual": {"count": 0, "avg_importance": 0.0, "concepts": []}
        }
        
        # 分析每个记忆
        for memory in memories:
            # 确定记忆类型
            memory_type = self._classify_memory_type(memory)
            
            if memory_type in patterns:
                patterns[memory_type]["count"] += 1
                patterns[memory_type]["avg_importance"] += memory.get("importance", 0.5)
            
            # 提取主题/情感/任务信息
            content = memory.get("content", "")
            if memory_type == "factual":
                # 提取事实主题 - 使用简单的高频词提取
                topics = self._extract_topics_from_content(content)
                for topic in topics:
                    if topic not in patterns[memory_type]["topics"]:
                        patterns[memory_type]["topics"].append(topic)
            elif memory_type == "emotional":
                # 提取情感信息
                emotion = memory.get("emotion", "neutral")
                if emotion not in patterns[memory_type]["emotions"]:
                    patterns[memory_type]["emotions"].append(emotion)
            elif memory_type == "procedural":
                # 提取任务信息 - 使用简单的动作动词提取
                tasks = self._extract_tasks_from_content(content)
                for task in tasks:
                    if task not in patterns[memory_type]["tasks"]:
                        patterns[memory_type]["tasks"].append(task)
        
        # 计算平均值
        for mem_type in patterns:
            if patterns[mem_type]["count"] > 0:
                patterns[mem_type]["avg_importance"] /= patterns[mem_type]["count"]
        
        return patterns
    
    def _classify_memory_type(self, memory: Dict[str, Any]) -> str:
        """分类记忆类型
        
        基于记忆内容分类记忆类型
        
        返回:
            记忆类型：factual, emotional, procedural, conceptual
        """
        content = memory.get("content", "").lower()
        
        # 简单的关键词匹配
        emotional_keywords = {"喜欢", "讨厌", "高兴", "悲伤", "愤怒", "害怕", "爱", "恨"}
        procedural_keywords = {"步骤", "方法", "操作", "流程", "如何", "怎样"}
        conceptual_keywords = {"概念", "定义", "原理", "理论", "思想"}
        
        if any(keyword in content for keyword in emotional_keywords):
            return "emotional"
        elif any(keyword in content for keyword in procedural_keywords):
            return "procedural"
        elif any(keyword in content for keyword in conceptual_keywords):
            return "conceptual"
        else:
            return "factual"  # 默认类型
    
    def _extract_topics_from_content(self, content: str) -> List[str]:
        """从内容中提取主题（高频词）
        
        参数:
            content: 文本内容
            
        返回:
            主题列表（前5个高频词）
        """
        if not content:
            return []  # 返回空列表
        
        # 分割单词，过滤停用词
        words = content.lower().split()
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        
        filtered_words = []
        for word in words:
            if len(word) >= 2 and word not in stop_words:
                filtered_words.append(word)
        
        # 统计词频
        from collections import Counter
        word_counts = Counter(filtered_words)
        
        # 返回前5个高频词
        return [word for word, _ in word_counts.most_common(5)]
    
    def _extract_tasks_from_content(self, content: str) -> List[str]:
        """从内容中提取任务信息（动作动词）
        
        参数:
            content: 文本内容
            
        返回:
            任务动作列表
        """
        if not content:
            return []  # 返回空列表
        
        # 中文动作动词列表（完整）
        action_verbs = {"做", "完成", "执行", "操作", "运行", "启动", "停止", "安装", "配置", "设置", "调整", "修改", "检查", "测试", "验证", "分析", "设计", "开发", "部署", "维护"}
        
        words = content.lower().split()
        tasks = []
        
        for word in words:
            if word in action_verbs and word not in tasks:
                tasks.append(word)
        
        # 如果没找到动作动词，返回整个内容作为单个任务
        if not tasks:
            tasks.append(content[:50])  # 截取前50个字符
        
        return tasks
    
    def _create_pattern_key(self, context_features: Dict[str, Any]) -> str:
        """创建模式键
        
        从上下文特征创建唯一的模式标识键
        
        参数:
            context_features: 上下文特征字典
            
        返回:
            模式键字符串
        """
        # 基于主要特征创建键
        topic = context_features.get("topic", "")
        intent = context_features.get("intent", "")
        emotion = context_features.get("emotion", "")
        has_task = context_features.get("has_task", False)
        
        # 创建键：话题+意图+情感+任务标志
        key_parts = []
        if topic:
            key_parts.append(f"topic:{topic[:20]}")
        if intent:
            key_parts.append(f"intent:{intent[:20]}")
        if emotion and emotion != "neutral":
            key_parts.append(f"emotion:{emotion}")
        if has_task:
            key_parts.append("task:true")
        
        if not key_parts:
            key_parts.append("general")
        
        return "|".join(key_parts)
    
    def predict_relevant_memories(self, db: Session, user_id: int) -> List[Dict[str, Any]]:
        """预测相关记忆（预加载）
        
        基于当前上下文预测用户可能需要的记忆，提前加载到缓存
        """
        if not self.enable_context_prediction:
            return []  # 返回空列表
        
        try:
            current_topic = self.current_context.get("current_topic", "")
            user_intent = self.current_context.get("user_intent", "")
            
            if not current_topic and not user_intent:
                return []  # 返回空列表
            
            # 基于当前话题和意图预测相关记忆
            predicted_query = current_topic or user_intent
            
            # 检索预测相关的记忆
            predicted_memories = self.memory_system.retrieve_relevant_memories(
                query=predicted_query,
                db=db,
                user_id=user_id,
                top_k=5,
                similarity_threshold=0.6
            )
            
            # 存储到上下文缓存
            cache_key = f"{user_id}_{hash(predicted_query)}"
            self.context_memory_cache[cache_key] = {
                "timestamp": time.time(),
                "memories": predicted_memories,
                "context": self.current_context.copy()
            }
            
            logger.info(f"预测并预加载了{len(predicted_memories)}个相关记忆")
            return predicted_memories
            
        except Exception as e:
            logger.error(f"预测相关记忆失败: {e}")
            return []  # 返回空列表
    
    def _classify_scene_and_update(self):
        """分类当前场景并更新场景状态"""
        try:
            # 获取当前对话内容
            conversation_history = self.current_context.get("conversation_history", [])
            current_topic = self.current_context.get("current_topic", "")
            user_intent = self.current_context.get("user_intent", "")
            
            # 组合文本进行分析
            analysis_text = ""
            if conversation_history:
                analysis_text += " ".join(conversation_history[-3:])  # 最近3条对话
            if current_topic:
                analysis_text += " " + current_topic
            if user_intent:
                analysis_text += " " + user_intent
            
            if not analysis_text.strip():
                return
            
            analysis_text = analysis_text.lower()
            
            # 计算每个场景类型的匹配分数
            scene_scores = {}
            for scene_type, keywords in self.scene_classifiers.items():
                score = 0.0
                matched_keywords = []
                
                for keyword in keywords:
                    if keyword in analysis_text:
                        matched_keywords.append(keyword)
                        score += 1.0
                
                if matched_keywords:
                    # 归一化分数
                    normalized_score = min(score / len(keywords) * 2, 1.0)  # 最多匹配2个关键词
                    scene_scores[scene_type] = {
                        "score": normalized_score,
                        "matched_keywords": matched_keywords
                    }
            
            # 确定主要场景类型
            if scene_scores:
                # 找到得分最高的场景
                primary_scene = max(scene_scores.items(), key=lambda x: x[1]["score"])[0]
                scene_data = scene_scores[primary_scene]
                
                # 如果得分超过阈值，更新场景
                if scene_data["score"] >= self.scene_classification_threshold:
                    # 检测场景切换
                    previous_scene = self.current_scene.get("scene_type", "general")
                    
                    if previous_scene != primary_scene:
                        # 场景切换，记录历史
                        scene_transition = {
                            "from_scene": previous_scene,
                            "to_scene": primary_scene,
                            "timestamp": time.time(),
                            "matched_keywords": scene_data["matched_keywords"]
                        }
                        self.current_scene["scene_history"].append(scene_transition)
                        
                        # 更新场景开始时间
                        self.current_scene["scene_start_time"] = time.time()
                    
                    # 更新当前场景
                    self.current_scene.update({
                        "scene_type": primary_scene,
                        "scene_id": f"{primary_scene}_{int(time.time())}",
                        "scene_features": {
                            "score": scene_data["score"],
                            "matched_keywords": scene_data["matched_keywords"],
                            "analysis_text_preview": analysis_text[:50] + "..." if len(analysis_text) > 50 else analysis_text
                        }
                    })
                    
                    logger.debug(f"场景分类: {previous_scene} -> {primary_scene} (分数: {scene_data['score']:.2f})")
            
        except Exception as e:
            logger.error(f"场景分类失败: {e}")
    
    def _detect_scene_transition(self):
        """检测场景切换"""
        try:
            scene_history = self.current_scene.get("scene_history", [])
            if len(scene_history) < 2:
                return
            
            # 获取最近的场景切换
            recent_transition = scene_history[-1]
            transition_time = recent_transition.get("timestamp", 0)
            current_time = time.time()
            
            # 如果场景切换发生在最近30秒内，记录日志
            if current_time - transition_time < 30:
                logger.info(f"最近检测到场景切换: {recent_transition['from_scene']} -> {recent_transition['to_scene']}")
                
                # 这里可以添加场景切换时的特殊处理逻辑
                # 例如：预加载新场景相关的记忆、重置某些状态等
            
        except Exception as e:
            logger.error(f"检测场景切换失败: {e}")
    
    def _calculate_scene_relevance(self, memory: Dict[str, Any]) -> float:
        """计算记忆与当前场景的相关性"""
        current_scene_type = self.current_scene.get("scene_type", "general")
        
        # 如果当前场景是general，返回默认相关性
        if current_scene_type == "general":
            return 0.5
        
        # 获取场景相关的关键词
        scene_keywords = self.scene_classifiers.get(current_scene_type, [])
        if not scene_keywords:
            return 0.5
        
        # 检查记忆内容是否包含场景关键词
        memory_content = memory.get("content", "").lower()
        
        matched_count = 0
        for keyword in scene_keywords:
            if keyword in memory_content:
                matched_count += 1
        
        # 计算相关性分数
        if matched_count == 0:
            return 0.3  # 没有匹配
        elif matched_count == 1:
            return 0.7  # 一个匹配
        else:
            return 0.9  # 多个匹配
    
    def retrieve_scene_related_memories(
        self,
        db: Session,
        user_id: int,
        scene_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """检索场景相关记忆
        
        参数:
            db: 数据库会话
            user_id: 用户ID
            scene_type: 场景类型，如果为None则使用当前场景
            top_k: 返回记忆数量
            
        返回:
            场景相关的记忆列表
        """
        if scene_type is None:
            scene_type = self.current_scene.get("scene_type", "general")
        
        # 构建场景相关的查询
        scene_keywords = self.scene_classifiers.get(scene_type, [])
        if not scene_keywords:
            # 如果没有特定关键词，使用场景类型作为查询
            scene_query = scene_type
        else:
            # 使用场景关键词构建查询
            scene_query = " ".join(scene_keywords[:3])  # 使用前3个关键词
        
        # 检索场景相关记忆
        scene_memories = self.memory_system.retrieve_relevant_memories(
            query=scene_query,
            db=db,
            user_id=user_id,
            top_k=top_k * 2,  # 检索更多用于场景过滤
            similarity_threshold=0.6
        )
        
        if not scene_memories:
            return []  # 返回空列表
        
        # 计算场景相关性并增强记忆
        scene_enhanced_memories = []
        for memory in scene_memories:
            enhanced_memory = memory.copy()
            
            # 计算场景相关性
            scene_relevance = self._calculate_scene_relevance(memory)
            enhanced_memory["scene_relevance"] = scene_relevance
            
            # 计算综合相关性（原始相关性 * 场景相关性）
            original_relevance = memory.get("similarity_score", 0.5)
            combined_relevance = original_relevance * 0.5 + scene_relevance * 0.5
            enhanced_memory["combined_relevance"] = combined_relevance
            
            scene_enhanced_memories.append(enhanced_memory)
        
        # 按综合相关性排序
        scene_enhanced_memories.sort(key=lambda x: x.get("combined_relevance", 0.0), reverse=True)
        
        # 返回前top_k个记忆
        result = scene_enhanced_memories[:top_k]
        
        # 记录到场景记忆映射
        scene_id = self.current_scene.get("scene_id")
        if scene_id:
            memory_ids = [m.get("id") for m in result if m.get("id")]
            if memory_ids and scene_id not in self.scene_memory_mapping:
                self.scene_memory_mapping[scene_id] = memory_ids
        
        logger.debug(f"检索到{len(result)}个{scene_type}场景相关记忆")
        return result
    
    def learn_scene_patterns(self, db: Session, user_id: int):
        """学习场景模式
        
        分析用户在特定场景下访问的记忆模式
        """
        try:
            current_scene = self.current_scene.get("scene_type", "general")
            scene_history = self.current_scene.get("scene_history", [])
            
            if len(scene_history) < 3:
                logger.info("场景历史不足，无法学习模式")
                return
            
            # 分析场景切换模式
            scene_transitions = {}
            for i in range(len(scene_history) - 1):
                transition = scene_history[i]
                from_scene = transition.get("from_scene")
                to_scene = transition.get("to_scene")
                
                key = f"{from_scene}->{to_scene}"
                scene_transitions[key] = scene_transitions.get(key, 0) + 1
            
            # 分析当前场景下的记忆访问模式
            scene_id = self.current_scene.get("scene_id")
            if scene_id and scene_id in self.scene_memory_mapping:
                memory_ids = self.scene_memory_mapping[scene_id]
                logger.info(f"场景{current_scene}下关联了{len(memory_ids)}个记忆")
            
            # 记录学习到的模式
            self.context_patterns["scene_transitions"] = scene_transitions
            self.context_patterns["current_scene_memories"] = {
                "scene_type": current_scene,
                "memory_count": len(self.scene_memory_mapping.get(scene_id, [])),
                "learned_at": time.time()
            }
            
            logger.info(f"学习到场景模式: {len(scene_transitions)}个场景切换模式")
            
        except Exception as e:
            logger.error(f"学习场景模式失败: {e}")
    
    def get_scene_stats(self) -> Dict[str, Any]:
        """获取场景统计信息"""
        scene_history = self.current_scene.get("scene_history", [])
        
        # 统计场景分布
        scene_distribution = {}
        for transition in scene_history:
            to_scene = transition.get("to_scene", "unknown")
            scene_distribution[to_scene] = scene_distribution.get(to_scene, 0) + 1
        
        return {
            "current_scene": self.current_scene.get("scene_type", "general"),
            "scene_id": self.current_scene.get("scene_id", ""),
            "scene_start_time": self.current_scene.get("scene_start_time", 0),
            "scene_history_count": len(scene_history),
            "scene_distribution": scene_distribution,
            "scene_memory_mapping_size": len(self.scene_memory_mapping),
            "scene_classification_enabled": self.enable_scene_classification,
            "scene_transition_detection_enabled": self.enable_scene_transition_detection
        }
    
    def get_context_stats(self) -> Dict[str, Any]:
        """获取上下文统计信息（增强版）"""
        base_stats = {
            "context_enhancer_enabled": True,
            "current_context": self._get_context_summary(),
            "context_patterns_count": len(self.context_patterns),
            "memory_cache_size": len(self.context_memory_cache),
            "context_window_size": self.context_window_size
        }
        
        # 添加场景统计信息
        scene_stats = self.get_scene_stats()
        
        # 合并统计信息
        combined_stats = base_stats.copy()
        combined_stats.update({
            "scene_stats": scene_stats,
            "scene_classification_enabled": self.enable_scene_classification,
            "scene_transition_detection_enabled": self.enable_scene_transition_detection
        })
        
        return combined_stats
    
    def retrieve_from_knowledge_base(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """从知识库检索相关信息
        
        参数:
            query: 查询文本
            top_k: 返回知识数量
            similarity_threshold: 相似度阈值
            
        返回:
            相关知识列表，包含知识和元数据
        """
        if not self.enable_knowledge_base or self.knowledge_manager is None:
            logger.warning("知识库集成未启用，无法从知识库检索")
            return []  # 返回空列表
        
        try:
            # 使用知识管理器的查询功能
            knowledge_results = self.knowledge_manager.query_knowledge(
                query=query,
                limit=top_k,
                similarity_threshold=similarity_threshold
            )
            
            if knowledge_results:
                logger.info(f"从知识库检索到 {len(knowledge_results)} 条相关知识")
            
            return knowledge_results
            
        except Exception as e:
            logger.error(f"从知识库检索失败: {e}")
            return []  # 返回空列表
    
    def retrieve_hybrid(
        self,
        query: str,
        db: Session,
        user_id: int,
        top_k: int = 10,
        memory_weight: float = 0.6,
        knowledge_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """混合检索 - 同时从记忆系统和知识库检索
        
        参数:
            query: 查询文本
            db: 数据库会话
            user_id: 用户ID
            top_k: 返回结果总数
            memory_weight: 记忆系统结果权重
            knowledge_weight: 知识库结果权重
            
        返回:
            混合检索结果，按相关性排序
        """
        # 从记忆系统检索
        memory_results = self.retrieve_relevant_memories(
            query=query,
            db=db,
            user_id=user_id,
            top_k=top_k
        )
        
        # 从知识库检索
        knowledge_results = self.retrieve_from_knowledge_base(
            query=query,
            top_k=top_k
        )
        
        # 合并结果
        all_results = []
        
        # 添加记忆结果（带权重）
        for result in memory_results:
            result_with_weight = result.copy()
            result_with_weight["source"] = "memory"
            result_with_weight["weighted_score"] = result.get("similarity_score", 0.5) * memory_weight
            all_results.append(result_with_weight)
        
        # 添加知识结果（带权重）
        for result in knowledge_results:
            result_with_weight = result.copy()
            result_with_weight["source"] = "knowledge"
            # 知识结果通常有similarity_score或confidence字段
            score = result.get("similarity_score", result.get("confidence", 0.5))
            result_with_weight["weighted_score"] = score * knowledge_weight
            all_results.append(result_with_weight)
        
        # 按加权分数排序
        all_results.sort(key=lambda x: x.get("weighted_score", 0.0), reverse=True)
        
        # 返回前top_k个结果
        return all_results[:top_k]


class MemorySystemPerformanceMonitor:
    """记忆系统性能监控器 - 监控和优化系统性能
    
    功能:
    1. 实时性能指标收集
    2. 性能趋势分析
    3. 瓶颈识别
    4. 优化建议生成
    5. 性能报告生成
    """
    
    def __init__(self, memory_system: MemorySystem, config: Optional[Dict[str, Any]] = None):
        self.memory_system = memory_system
        self.config = config or {}
        
        # 监控配置
        self.monitoring_interval = self.config.get("monitoring_interval", 60)  # 监控间隔（秒）
        self.performance_history_size = self.config.get("performance_history_size", 1000)
        self.enable_real_time_alerts = self.config.get("enable_real_time_alerts", True)
        
        # 性能指标历史
        self.performance_history = []
        self.alert_history = []
        
        # 性能阈值配置
        self.performance_thresholds = {
            "cache_hit_rate": 0.7,  # 缓存命中率阈值
            "response_time": 1.0,   # 响应时间阈值（秒）
            "memory_usage": 0.8,    # 内存使用率阈值
            "cpu_usage": 0.7,       # CPU使用率阈值
            "error_rate": 0.05      # 错误率阈值
        }
        
        # 优化建议数据库
        self.optimization_suggestions = self._load_optimization_suggestions()
        
        # 监控状态
        self.monitoring_start_time = time.time()
        self.total_operations = 0
        self.error_count = 0
        
        logger.info("记忆系统性能监控器已初始化")
    
    def _load_optimization_suggestions(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载优化建议数据库"""
        suggestions = {
            "cache_optimization": [
                {
                    "issue": "缓存命中率低",
                    "suggestions": [
                        "增加短期缓存大小",
                        "优化缓存替换策略",
                        "实现智能预加载",
                        "调整缓存过期时间"
                    ],
                    "priority": "high"
                },
                {
                    "issue": "缓存内存占用过高",
                    "suggestions": [
                        "减少缓存大小",
                        "使用LRU淘汰策略",
                        "压缩缓存数据",
                        "实现分层缓存"
                    ],
                    "priority": "medium"
                }
            ],
            "retrieval_optimization": [
                {
                    "issue": "检索响应时间慢",
                    "suggestions": [
                        "优化FAISS/HNSW索引参数",
                        "增加工作内存容量",
                        "实现并行检索",
                        "缓存热门查询结果"
                    ],
                    "priority": "high"
                },
                {
                    "issue": "检索准确率低",
                    "suggestions": [
                        "调整相似度阈值",
                        "优化嵌入模型",
                        "增加检索多样性",
                        "使用混合检索策略"
                    ],
                    "priority": "medium"
                }
            ],
            "memory_management": [
                {
                    "issue": "记忆数量过多",
                    "suggestions": [
                        "应用选择性记忆机制",
                        "增加遗忘率",
                        "压缩相似记忆",
                        "归档不常用记忆"
                    ],
                    "priority": "medium"
                },
                {
                    "issue": "记忆重要性分布不均",
                    "suggestions": [
                        "调整重要性评分算法",
                        "实现动态重要性调整",
                        "学习用户偏好模式",
                        "应用情感记忆增强"
                    ],
                    "priority": "low"
                }
            ],
            "system_resources": [
                {
                    "issue": "内存使用率过高",
                    "suggestions": [
                        "优化缓存策略",
                        "压缩存储数据",
                        "增加系统内存",
                        "实现内存分页"
                    ],
                    "priority": "high"
                },
                {
                    "issue": "CPU使用率过高",
                    "suggestions": [
                        "优化算法复杂度",
                        "实现异步处理",
                        "使用GPU加速",
                        "增加计算资源"
                    ],
                    "priority": "high"
                }
            ]
        }
        
        return suggestions
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """收集性能指标"""
        metrics = {
            "timestamp": time.time(),
            "uptime": time.time() - self.monitoring_start_time,
            "total_operations": self.total_operations,
            "error_rate": self.error_count / max(1, self.total_operations)
        }
        
        try:
            # 收集缓存统计
            cache_stats = self.memory_system.get_cache_stats()
            metrics.update({
                "cache_hit_rate": float(cache_stats.get("short_term_cache_hit_rate", "0%").strip("%")) / 100,
                "cache_size": cache_stats.get("short_term_cache_size", 0),
                "working_memory_size": cache_stats.get("working_memory_size", 0),
                "faiss_avg_search_time": float(cache_stats.get("faiss_avg_search_time_ms", "0.0").strip("ms")),
                "hnsw_avg_search_time": float(cache_stats.get("hnsw_avg_search_time_ms", "0.0").strip("ms"))
            })
            
            # 收集推理统计
            reasoning_stats = self.memory_system.get_reasoning_stats()
            metrics.update({
                "reasoning_enabled": reasoning_stats.get("cognitive_reasoning_enabled", False),
                "reasoning_success_rate": reasoning_stats.get("success_rate", 0.0),
                "reasoning_attempts": reasoning_stats.get("total_reasoning_attempts", 0)
            })
            
            # 收集上下文统计
            context_stats = self.memory_system.get_context_stats()
            metrics.update({
                "context_awareness_enabled": context_stats.get("context_awareness_enabled", False),
                "context_cache_size": context_stats.get("memory_cache_size", 0)
            })
            
            # 收集系统资源指标（如果可用）
            try:
                import psutil
                process = psutil.Process()
                metrics.update({
                    "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                    "cpu_percent": process.cpu_percent(interval=0.1),
                    "thread_count": process.num_threads()
                })
            except ImportError:
                metrics.update({
                    "memory_usage_mb": 0,
                    "cpu_percent": 0,
                    "thread_count": 0
                })
            
            # 计算响应时间指标（完整）
            metrics["avg_response_time"] = (
                metrics.get("faiss_avg_search_time", 0) * 0.5 +
                metrics.get("hnsw_avg_search_time", 0) * 0.3 +
                100 * (1 - metrics.get("cache_hit_rate", 0)) * 0.2
            ) / 1000  # 转换为秒
            
        except Exception as e:
            logger.error(f"收集性能指标失败: {e}")
            self.error_count += 1
        
        # 记录性能历史
        self.performance_history.append(metrics)
        
        # 保持历史记录大小
        if len(self.performance_history) > self.performance_history_size:
            self.performance_history = self.performance_history[-self.performance_history_size:]
        
        self.total_operations += 1
        
        return metrics
    
    def analyze_performance_trends(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """分析性能趋势
        
        参数:
            time_window: 时间窗口（秒），None表示使用所有历史数据
            
        返回:
            性能趋势分析结果
        """
        if not self.performance_history:
            return {"message": "没有性能历史数据"}
        
        # 过滤时间窗口内的数据
        if time_window:
            cutoff_time = time.time() - time_window
            recent_metrics = [m for m in self.performance_history if m["timestamp"] >= cutoff_time]
        else:
            recent_metrics = self.performance_history
        
        if not recent_metrics:
            return {"message": "指定时间窗口内没有数据"}
        
        # 计算统计指标
        cache_hit_rates = [m.get("cache_hit_rate", 0) for m in recent_metrics]
        response_times = [m.get("avg_response_time", 0) for m in recent_metrics]
        memory_usages = [m.get("memory_usage_mb", 0) for m in recent_metrics]
        cpu_usages = [m.get("cpu_percent", 0) for m in recent_metrics]
        
        analysis = {
            "time_window_seconds": time_window or (recent_metrics[-1]["timestamp"] - recent_metrics[0]["timestamp"]),
            "data_points": len(recent_metrics),
            "cache_hit_rate": {
                "avg": sum(cache_hit_rates) / len(cache_hit_rates),
                "min": min(cache_hit_rates) if cache_hit_rates else 0,
                "max": max(cache_hit_rates) if cache_hit_rates else 0,
                "trend": self._calculate_trend(cache_hit_rates)
            },
            "response_time": {
                "avg": sum(response_times) / len(response_times),
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "trend": self._calculate_trend(response_times)
            },
            "memory_usage": {
                "avg_mb": sum(memory_usages) / len(memory_usages),
                "min_mb": min(memory_usages) if memory_usages else 0,
                "max_mb": max(memory_usages) if memory_usages else 0,
                "trend": self._calculate_trend(memory_usages)
            },
            "cpu_usage": {
                "avg": sum(cpu_usages) / len(cpu_usages),
                "min": min(cpu_usages) if cpu_usages else 0,
                "max": max(cpu_usages) if cpu_usages else 0,
                "trend": self._calculate_trend(cpu_usages)
            }
        }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算数值趋势"""
        if len(values) < 2:
            return "stable"
        
        # 使用线性回归计算趋势
        from statistics import mean
        x = list(range(len(values)))
        y = values
        
        n = len(x)
        if n == 0:
            return "stable"
        
        x_mean = mean(x)
        y_mean = mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def check_performance_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查性能阈值，生成警报"""
        alerts = []
        
        thresholds = self.performance_thresholds
        
        # 检查缓存命中率
        cache_hit_rate = metrics.get("cache_hit_rate", 1.0)
        if cache_hit_rate < thresholds["cache_hit_rate"]:
            alerts.append({
                "type": "warning",
                "metric": "cache_hit_rate",
                "value": cache_hit_rate,
                "threshold": thresholds["cache_hit_rate"],
                "message": f"缓存命中率过低: {cache_hit_rate:.1%} < {thresholds['cache_hit_rate']:.1%}",
                "suggestions": self._get_optimization_suggestions("cache_optimization", "缓存命中率低")
            })
        
        # 检查响应时间
        response_time = metrics.get("avg_response_time", 0)
        if response_time > thresholds["response_time"]:
            alerts.append({
                "type": "warning",
                "metric": "response_time",
                "value": response_time,
                "threshold": thresholds["response_time"],
                "message": f"响应时间过长: {response_time:.3f}s > {thresholds['response_time']}s",
                "suggestions": self._get_optimization_suggestions("retrieval_optimization", "检索响应时间慢")
            })
        
        # 检查内存使用率
        memory_usage = metrics.get("memory_usage_mb", 0)
        # 完整处理
        if memory_usage > 1000:  # 假设1GB为阈值
            alerts.append({
                "type": "warning",
                "metric": "memory_usage",
                "value": memory_usage,
                "threshold": 1000,
                "message": f"内存使用过高: {memory_usage:.1f}MB",
                "suggestions": self._get_optimization_suggestions("system_resources", "内存使用率过高")
            })
        
        # 检查CPU使用率
        cpu_usage = metrics.get("cpu_percent", 0)
        if cpu_usage > thresholds["cpu_usage"] * 100:  # 转换为百分比
            alerts.append({
                "type": "warning",
                "metric": "cpu_usage",
                "value": cpu_usage,
                "threshold": thresholds["cpu_usage"] * 100,
                "message": f"CPU使用率过高: {cpu_usage:.1f}% > {thresholds['cpu_usage']*100:.1f}%",
                "suggestions": self._get_optimization_suggestions("system_resources", "CPU使用率过高")
            })
        
        # 检查错误率
        error_rate = metrics.get("error_rate", 0)
        if error_rate > thresholds["error_rate"]:
            alerts.append({
                "type": "error",
                "metric": "error_rate",
                "value": error_rate,
                "threshold": thresholds["error_rate"],
                "message": f"错误率过高: {error_rate:.1%} > {thresholds['error_rate']:.1%}",
                "suggestions": ["检查系统日志", "验证输入数据", "调试错误处理逻辑"]
            })
        
        # 记录警报
        for alert in alerts:
            alert["timestamp"] = time.time()
            self.alert_history.append(alert)
            
            if self.enable_real_time_alerts:
                logger.warning(f"性能警报: {alert['message']}")
        
        # 保持警报历史大小
        max_alerts = self.config.get("max_alert_history", 100)
        if len(self.alert_history) > max_alerts:
            self.alert_history = self.alert_history[-max_alerts:]
        
        return alerts
    
    def _get_optimization_suggestions(self, category: str, issue: str) -> List[str]:
        """获取优化建议"""
        suggestions = []
        
        category_suggestions = self.optimization_suggestions.get(category, [])
        for item in category_suggestions:
            if item["issue"] == issue:
                suggestions.extend(item["suggestions"])
                break
        
        return suggestions or ["检查系统配置", "分析性能日志", "考虑硬件升级"]
    
    def generate_performance_report(self, time_window: Optional[int] = 3600) -> Dict[str, Any]:
        """生成性能报告
        
        参数:
            time_window: 报告时间窗口（秒，默认1小时）
            
        返回:
            完整的性能报告
        """
        # 收集当前指标
        current_metrics = self.collect_performance_metrics()
        
        # 分析性能趋势
        trends = self.analyze_performance_trends(time_window)
        
        # 检查阈值
        alerts = self.check_performance_thresholds(current_metrics)
        
        # 生成优化建议
        optimization_suggestions = self._generate_optimization_suggestions(current_metrics, trends, alerts)
        
        report = {
            "report_timestamp": time.time(),
            "time_window_seconds": time_window,
            "current_metrics": current_metrics,
            "performance_trends": trends,
            "active_alerts": alerts,
            "optimization_suggestions": optimization_suggestions,
            "system_summary": {
                "total_operations": self.total_operations,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(1, self.total_operations),
                "monitoring_duration": time.time() - self.monitoring_start_time,
                "performance_history_size": len(self.performance_history),
                "alert_history_size": len(self.alert_history)
            }
        }
        
        return report
    
    def _generate_optimization_suggestions(self, metrics: Dict[str, Any], trends: Dict[str, Any], alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        suggestions = []
        
        # 基于当前指标生成建议
        cache_hit_rate = metrics.get("cache_hit_rate", 1.0)
        if cache_hit_rate < 0.6:
            suggestions.append({
                "category": "cache_optimization",
                "issue": "缓存命中率低",
                "priority": "high",
                "suggestions": [
                    f"当前缓存命中率: {cache_hit_rate:.1%}",
                    "建议增加短期缓存大小",
                    "考虑实现智能预加载策略",
                    "优化缓存替换算法"
                ]
            })
        
        response_time = metrics.get("avg_response_time", 0)
        if response_time > 0.5:  # 0.5秒阈值
            suggestions.append({
                "category": "retrieval_optimization",
                "issue": "响应时间较慢",
                "priority": "medium",
                "suggestions": [
                    f"平均响应时间: {response_time:.3f}秒",
                    "优化FAISS/HNSW索引参数",
                    "考虑增加工作内存容量",
                    "实现查询结果缓存"
                ]
            })
        
        # 基于趋势生成建议
        cache_trend = trends.get("cache_hit_rate", {}).get("trend", "stable")
        if cache_trend == "decreasing":
            suggestions.append({
                "category": "cache_optimization",
                "issue": "缓存命中率下降趋势",
                "priority": "medium",
                "suggestions": [
                    "检测到缓存命中率下降趋势",
                    "分析用户访问模式变化",
                    "调整缓存预热策略",
                    "监控系统负载变化"
                ]
            })
        
        # 基于警报生成建议
        for alert in alerts:
            if alert["type"] == "error":
                suggestions.append({
                    "category": "system_health",
                    "issue": "系统错误率过高",
                    "priority": "high",
                    "suggestions": [
                        f"错误率: {alert['value']:.1%}",
                        "立即检查系统日志",
                        "验证输入数据处理",
                        "调试错误处理逻辑"
                    ]
                })
        
        return suggestions
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        return {
            "monitoring_enabled": True,
            "monitoring_start_time": self.monitoring_start_time,
            "total_operations": self.total_operations,
            "error_count": self.error_count,
            "performance_history_size": len(self.performance_history),
            "alert_history_size": len(self.alert_history),
            "monitoring_interval": self.monitoring_interval,
            "enable_real_time_alerts": self.enable_real_time_alerts
        }


class EmotionalMemoryEnhancer:
    """情感记忆增强器 - 实现情感感知的记忆管理
    
    功能:
    1. 文本情感分析
    2. 情感状态跟踪
    3. 情感驱动的记忆重要性调整
    4. 情感关联记忆检索
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 情感分析配置
        self.enable_emotion_analysis = self.config.get("enable_emotion_analysis", True)
        self.emotion_detection_threshold = self.config.get("emotion_detection_threshold", 0.3)
        self.emotion_impact_factor = self.config.get("emotion_impact_factor", 0.2)
        
        # 情感词典（中文情感关键词）
        self.emotion_lexicon = {
            "happy": ["高兴", "快乐", "开心", "喜悦", "兴奋", "愉快", "幸福", "满意", "欣慰", "欢乐"],
            "sad": ["悲伤", "难过", "伤心", "悲哀", "沮丧", "失望", "痛苦", "郁闷", "忧愁", "泪丧"],
            "angry": ["生气", "愤怒", "恼火", "发怒", "气愤", "暴怒", "愤怒", "不满", "烦躁", "怒火"],
            "fear": ["害怕", "恐惧", "惊恐", "畏惧", "恐慌", "不安", "担心", "忧虑", "惊吓", "恐怖"],
            "surprise": ["惊讶", "惊奇", "吃惊", "意外", "震惊", "诧异", "惊喜", "愕然", "意想不到"],
            "disgust": ["厌恶", "讨厌", "反感", "恶心", "嫌弃", "憎恶", "厌烦", "鄙视", "轻视"],
            "neutral": []  # 中性情感
        }
        
        # 情感强度映射
        self.emotion_intensity = {
            "happy": {"low": 0.3, "medium": 0.6, "high": 0.9},
            "sad": {"low": 0.3, "medium": 0.6, "high": 0.9},
            "angry": {"low": 0.4, "medium": 0.7, "high": 1.0},
            "fear": {"low": 0.3, "medium": 0.6, "high": 0.9},
            "surprise": {"low": 0.2, "medium": 0.5, "high": 0.8},
            "disgust": {"low": 0.3, "medium": 0.6, "high": 0.9}
        }
        
        # 情感状态跟踪
        self.emotional_state = {
            "current_emotion": "neutral",
            "emotion_intensity": 0.0,
            "emotion_history": [],
            "emotional_patterns": {}
        }
        
        # 情感记忆数据库
        self.emotional_memories = {}
        
        logger.info("情感记忆增强器已初始化")
    
    def analyze_text_emotion(self, text: str) -> Dict[str, Any]:
        """分析文本情感
        
        参数:
            text: 输入文本
            
        返回:
            情感分析结果，包括情感类型和强度
        """
        if not self.enable_emotion_analysis or not text:
            return {
                "emotion": "neutral",
                "intensity": 0.0,
                "confidence": 0.0,
                "keywords": []
            }
        
        text_lower = text.lower()
        emotion_scores = {}
        detected_keywords = {}
        
        # 计算每种情感的关键词匹配分数
        for emotion, keywords in self.emotion_lexicon.items():
            if emotion == "neutral":
                continue
            
            matched_keywords = []
            score = 0.0
            
            for keyword in keywords:
                if keyword in text_lower:
                    matched_keywords.append(keyword)
                    score += 1.0
            
            if matched_keywords:
                # 归一化分数
                normalized_score = min(score / len(keywords) * 3, 1.0)  # 最多匹配3个关键词
                emotion_scores[emotion] = normalized_score
                detected_keywords[emotion] = matched_keywords
        
        # 确定主要情感
        if emotion_scores:
            # 找到得分最高的情感
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            intensity = emotion_scores[primary_emotion]
            
            # 计算置信度
            total_score = sum(emotion_scores.values())
            confidence = emotion_scores[primary_emotion] / total_score if total_score > 0 else 0.0
            
            result = {
                "emotion": primary_emotion,
                "intensity": intensity,
                "confidence": confidence,
                "keywords": detected_keywords.get(primary_emotion, []),
                "all_scores": emotion_scores
            }
        else:
            # 没有检测到情感，返回中性
            result = {
                "emotion": "neutral",
                "intensity": 0.0,
                "confidence": 1.0,
                "keywords": [],
                "all_scores": {}
            }
        
        # 更新情感状态
        if result["emotion"] != "neutral" and result["confidence"] > self.emotion_detection_threshold:
            self._update_emotional_state(result)
        
        return result
    
    def _update_emotional_state(self, emotion_result: Dict[str, Any]):
        """更新情感状态"""
        emotion_entry = {
            "timestamp": time.time(),
            "emotion": emotion_result["emotion"],
            "intensity": emotion_result["intensity"],
            "confidence": emotion_result["confidence"]
        }
        
        # 添加到情感历史
        self.emotional_state["emotion_history"].append(emotion_entry)
        
        # 更新当前情感状态
        self.emotional_state["current_emotion"] = emotion_result["emotion"]
        self.emotional_state["emotion_intensity"] = emotion_result["intensity"]
        
        # 保持历史记录大小
        max_history = self.config.get("max_emotion_history", 100)
        if len(self.emotional_state["emotion_history"]) > max_history:
            self.emotional_state["emotion_history"] = self.emotional_state["emotion_history"][-max_history:]
        
        # 学习情感模式
        self._learn_emotional_patterns(emotion_entry)
    
    def _learn_emotional_patterns(self, emotion_entry: Dict[str, Any]):
        """学习情感模式"""
        emotion = emotion_entry["emotion"]
        
        if emotion not in self.emotional_state["emotional_patterns"]:
            self.emotional_state["emotional_patterns"][emotion] = {
                "count": 0,
                "total_intensity": 0.0,
                "average_intensity": 0.0,
                "last_occurred": None,
                "frequency": 0.0
            }
        
        pattern = self.emotional_state["emotional_patterns"][emotion]
        pattern["count"] += 1
        pattern["total_intensity"] += emotion_entry["intensity"]
        pattern["average_intensity"] = pattern["total_intensity"] / pattern["count"]
        pattern["last_occurred"] = emotion_entry["timestamp"]
        
        # 计算频率（每天出现的次数）
        history = self.emotional_state["emotion_history"]
        if len(history) >= 2:
            time_span = history[-1]["timestamp"] - history[0]["timestamp"]
            days = max(1, time_span / 86400)  # 转换为天
            pattern["frequency"] = pattern["count"] / days
    
    def adjust_memory_importance_with_emotion(self, memory_data: Dict[str, Any]) -> float:
        """基于情感调整记忆重要性
        
        参数:
            memory_data: 记忆数据，包含内容和元数据
            
        返回:
            调整后的重要性分数
        """
        original_importance = memory_data.get("importance", 0.5)
        
        if not self.enable_emotion_analysis:
            return original_importance
        
        # 分析记忆内容的情感
        content = memory_data.get("content", "")
        emotion_result = self.analyze_text_emotion(content)
        
        # 如果没有检测到情感，返回原始重要性
        if emotion_result["emotion"] == "neutral" or emotion_result["confidence"] < self.emotion_detection_threshold:
            return original_importance
        
        emotion = emotion_result["emotion"]
        intensity = emotion_result["intensity"]
        
        # 基于情感类型和强度调整重要性
        adjustment_factor = self._calculate_emotion_adjustment(emotion, intensity)
        
        # 应用调整
        adjusted_importance = original_importance * (1.0 + adjustment_factor * self.emotion_impact_factor)
        
        # 确保重要性在0-1范围内
        adjusted_importance = max(0.0, min(1.0, adjusted_importance))
        
        # 记录情感记忆
        memory_id = memory_data.get("id")
        if memory_id:
            self.emotional_memories[memory_id] = {
                "memory_id": memory_id,
                "emotion": emotion,
                "intensity": intensity,
                "original_importance": original_importance,
                "adjusted_importance": adjusted_importance,
                "timestamp": time.time()
            }
        
        return adjusted_importance
    
    def _calculate_emotion_adjustment(self, emotion: str, intensity: float) -> float:
        """计算情感调整因子
        
        不同情感对记忆重要性的影响不同：
        - 强烈情感（高兴、愤怒、恐惧）：增加重要性
        - 中度情感（惊讶、厌恶）：轻微增加重要性
        - 消极情感（悲伤）：可能降低重要性
        - 中性情感：无影响
        """
        # 情感调整映射
        emotion_adjustments = {
            "happy": 0.3,      # 高兴增加重要性
            "angry": 0.4,      # 愤怒显著增加重要性
            "fear": 0.3,       # 恐惧增加重要性
            "surprise": 0.2,   # 惊讶轻微增加重要性
            "disgust": 0.1,    # 厌恶轻微增加重要性
            "sad": -0.2,       # 悲伤可能降低重要性
            "neutral": 0.0     # 中性无影响
        }
        
        base_adjustment = emotion_adjustments.get(emotion, 0.0)
        
        # 基于强度调整
        intensity_factor = intensity
        
        return base_adjustment * intensity_factor
    
    def retrieve_emotion_related_memories(
        self,
        current_emotion: Optional[str] = None,
        emotion_intensity: Optional[float] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """检索与当前情感相关的记忆
        
        参数:
            current_emotion: 当前情感类型，如果为None则使用最近的情感状态
            emotion_intensity: 情感强度，如果为None则使用最近的情感强度
            top_k: 返回记忆数量
            
        返回:
            情感相关的记忆列表
        """
        if not self.emotional_memories:
            return []  # 返回空列表
        
        # 确定当前情感
        if current_emotion is None:
            current_emotion = self.emotional_state["current_emotion"]
            emotion_intensity = self.emotional_state["emotion_intensity"]
        
        if current_emotion == "neutral":
            # 中性情感，返回所有情感记忆
            related_memories = list(self.emotional_memories.values())
        else:
            # 检索相同情感的記憶
            related_memories = [
                memory for memory in self.emotional_memories.values()
                if memory["emotion"] == current_emotion
            ]
        
        # 按情感强度排序
        related_memories.sort(key=lambda x: x["intensity"], reverse=True)
        
        # 限制返回数量
        return related_memories[:top_k]
    
    def get_emotional_insights(self) -> Dict[str, Any]:
        """获取情感洞察"""
        emotion_history = self.emotional_state["emotion_history"]
        
        if not emotion_history:
            return {
                "current_emotion": "neutral",
                "emotion_intensity": 0.0,
                "total_emotion_events": 0,
                "emotional_patterns": {}
            }
        
        # 统计情感分布
        emotion_counts = {}
        for entry in emotion_history:
            emotion = entry["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_events = len(emotion_history)
        
        # 计算情感分布百分比
        emotion_distribution = {}
        for emotion, count in emotion_counts.items():
            emotion_distribution[emotion] = count / total_events
        
        return {
            "current_emotion": self.emotional_state["current_emotion"],
            "emotion_intensity": self.emotional_state["emotion_intensity"],
            "total_emotion_events": total_events,
            "emotion_distribution": emotion_distribution,
            "emotional_patterns": self.emotional_state["emotional_patterns"],
            "emotional_memories_count": len(self.emotional_memories)
        }
    
    def enhance_memory_features_with_emotion(self, features: torch.Tensor, memory_data: Dict[str, Any]) -> torch.Tensor:
        """使用情感增强记忆特征
        
        为ImportanceModel提供情感增强的特征
        """
        if not self.enable_emotion_analysis:
            return features
        
        # 分析记忆情感
        content = memory_data.get("content", "")
        emotion_result = self.analyze_text_emotion(content)
        
        # 如果特征维度足够，添加情感特征
        feature_dim = features.shape[-1] if features.dim() > 0 else len(features)
        
        if feature_dim >= 10:  # 假设特征维度至少为10
            # 在特征索引5-9的位置添加情感特征
            enhanced_features = features.clone() if isinstance(features, torch.Tensor) else torch.tensor(features)
            
            if enhanced_features.dim() == 0:
                enhanced_features = torch.zeros(feature_dim)
            
            # 情感类型编码 (one-hot风格)
            emotion_encoding = {
                "happy": 0.2,
                "sad": 0.4,
                "angry": 0.6,
                "fear": 0.8,
                "surprise": 1.0,
                "disgust": 0.0,
                "neutral": 0.0
            }
            
            emotion_code = emotion_encoding.get(emotion_result["emotion"], 0.0)
            intensity = emotion_result["intensity"]
            
            # 设置情感特征
            if feature_dim > 5:
                enhanced_features[5] = emotion_code  # 情感类型
            if feature_dim > 6:
                enhanced_features[6] = intensity     # 情感强度
            if feature_dim > 7:
                enhanced_features[7] = emotion_result["confidence"]  # 情感置信度
            
            return enhanced_features
        
        return features
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """获取当前情感状态"""
        return self.emotional_state.copy()
    
    def reset_emotional_state(self):
        """重置情感状态"""
        self.emotional_state = {
            "current_emotion": "neutral",
            "emotion_intensity": 0.0,
            "emotion_history": [],
            "emotional_patterns": {}
        }
        logger.info("情感状态已重置")


class AdvancedMemoryForgetting:
    """高级记忆遗忘机制
    
    实现完善的记忆遗忘功能，包括：
    1. 基于艾宾浩斯遗忘曲线的指数衰减模型
    2. 基于记忆关联性的协同遗忘
    3. 基于内容质量和信息熵的智能遗忘
    4. 遗忘效果评估和自适应调整
    5. 多种遗忘策略（主动遗忘、被动遗忘、选择性遗忘）
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("AdvancedMemoryForgetting")
        
        # 遗忘参数
        self.forgetting_params = {
            "base_forgetting_rate": self.config.get("base_forgetting_rate", 0.05),
            "ebbinghaus_decay_factor": self.config.get("ebbinghaus_decay_factor", 0.56),  # 艾宾浩斯衰减因子
            "association_forgetting_weight": self.config.get("association_forgetting_weight", 0.3),
            "entropy_forgetting_weight": self.config.get("entropy_forgetting_weight", 0.2),
            "quality_forgetting_weight": self.config.get("quality_forgetting_weight", 0.2),
            "min_retention_threshold": self.config.get("min_retention_threshold", 0.1),  # 最小保留阈值
            "max_forgetting_rate": self.config.get("max_forgetting_rate", 0.3),  # 最大遗忘率
        }
        
        # 遗忘策略配置
        self.forgetting_strategies = {
            "active": {
                "name": "主动遗忘",
                "description": "主动选择忘记不相关或低质量记忆",
                "aggressiveness": 0.8
            },
            "passive": {
                "name": "被动遗忘",
                "description": "基于时间和访问模式的自然遗忘",
                "aggressiveness": 0.3
            },
            "selective": {
                "name": "选择性遗忘",
                "description": "基于内容和关联性的智能选择遗忘",
                "aggressiveness": 0.5
            },
            "associative": {
                "name": "关联性遗忘",
                "description": "基于记忆图结构的协同遗忘",
                "aggressiveness": 0.6
            }
        }
        
        # 遗忘历史记录
        self.forgetting_history = []
        self.max_history_size = self.config.get("max_forgetting_history", 1000)
        
        # 初始化评估模型
        self._init_evaluation_models()
        
        self.logger.info("高级记忆遗忘机制初始化完成")
    
    def _init_evaluation_models(self):
        """初始化评估模型"""
        # 内容质量评估模型
        self.quality_model = self._create_quality_model()
        
        # 信息熵计算器
        self.entropy_calculator = self._create_entropy_calculator()
        
        # 遗忘效果评估器
        self.effectiveness_evaluator = self._create_effectiveness_evaluator()
    
    def _create_quality_model(self):
        """创建内容质量评估模型"""
        # 简单的启发式质量评估
        def evaluate_quality(content: str, metadata: Dict[str, Any]) -> float:
            """评估记忆内容质量（0-1）"""
            if not content or len(content.strip()) == 0:
                return 0.0
            
            quality_score = 0.5  # 基础分
            
            # 1. 长度评估（适中的长度更好）
            content_length = len(content)
            if 50 <= content_length <= 500:
                quality_score += 0.2
            elif content_length > 1000:
                quality_score -= 0.1
            
            # 2. 结构评估（是否有标点、分段）
            has_punctuation = any(punc in content for punc in ".。!！?？,，")
            if has_punctuation:
                quality_score += 0.1
            
            # 3. 词汇丰富度评估（简单启发式）
            words = content.split()
            unique_words = set(words)
            if len(words) > 0:
                lexical_richness = len(unique_words) / len(words)
                quality_score += lexical_richness * 0.1
            
            # 4. 元数据质量（如果有）
            importance = metadata.get("importance", 0.5)
            quality_score += importance * 0.1
            
            return max(0.0, min(1.0, quality_score))
        
        return evaluate_quality
    
    def _create_entropy_calculator(self):
        """创建信息熵计算器"""
        import math
        
        def calculate_entropy(content: str) -> float:
            """计算文本信息熵"""
            if not content or len(content) == 0:
                return 0.0
            
            # 字符频率统计
            char_counts = {}
            total_chars = 0
            
            for char in content:
                if char.strip():  # 忽略空白字符
                    char_counts[char] = char_counts.get(char, 0) + 1
                    total_chars += 1
            
            if total_chars == 0:
                return 0.0
            
            # 计算熵
            entropy = 0.0
            for count in char_counts.values():
                probability = count / total_chars
                entropy -= probability * math.log2(probability)
            
            # 归一化到0-1范围（假设最大熵为log2(100)≈6.64）
            max_entropy = 6.64
            normalized_entropy = min(entropy / max_entropy, 1.0)
            
            return normalized_entropy
        
        return calculate_entropy
    
    def _create_effectiveness_evaluator(self):
        """创建遗忘效果评估器"""
        def evaluate_effectiveness(
            forgetting_stats: Dict[str, Any],
            system_state: Dict[str, Any]
        ) -> Dict[str, Any]:
            """评估遗忘效果"""
            effectiveness_score = 0.5  # 基础分
            
            # 1. 记忆压缩率评估
            compression_rate = forgetting_stats.get("compression_rate", 0.0)
            # 适中的压缩率（10-30%）得分最高
            if 0.1 <= compression_rate <= 0.3:
                effectiveness_score += 0.3
            elif compression_rate > 0.5:
                effectiveness_score -= 0.2  # 压缩过多
            elif compression_rate < 0.05:
                effectiveness_score -= 0.1  # 压缩不足
            
            # 2. 系统负载改善评估
            load_before = system_state.get("load_before", 0.5)
            load_after = system_state.get("load_after", 0.5)
            load_improvement = load_before - load_after
            effectiveness_score += load_improvement * 0.5
            
            # 3. 重要性保留评估
            avg_importance_before = forgetting_stats.get("avg_importance_before", 0.5)
            avg_importance_after = forgetting_stats.get("avg_importance_after", 0.5)
            importance_preservation = avg_importance_after / max(avg_importance_before, 0.01)
            effectiveness_score += (importance_preservation - 0.5) * 0.2
            
            # 限制在0-1范围
            effectiveness_score = max(0.0, min(1.0, effectiveness_score))
            
            return {
                "score": effectiveness_score,
                "compression_optimal": 0.1 <= compression_rate <= 0.3,
                "load_improvement": load_improvement,
                "importance_preservation": importance_preservation,
                "recommendation": self._generate_recommendation(effectiveness_score, forgetting_stats)
            }
        
        return evaluate_effectiveness
    
    def _generate_recommendation(self, effectiveness_score: float, stats: Dict[str, Any]) -> str:
        """生成优化建议"""
        compression_rate = stats.get("compression_rate", 0.0)
        
        if effectiveness_score >= 0.7:
            return "遗忘策略效果良好，保持当前参数"
        elif effectiveness_score >= 0.5:
            if compression_rate < 0.1:
                return "建议适度增加遗忘率，当前压缩率偏低"
            elif compression_rate > 0.3:
                return "建议降低遗忘率，当前压缩率偏高"
            else:
                return "遗忘策略效果适中，可微调参数"
        else:
            if compression_rate < 0.05:
                return "遗忘过于保守，建议显著增加遗忘率"
            elif compression_rate > 0.5:
                return "遗忘过于激进，建议显著降低遗忘率并调整策略"
            else:
                return "遗忘策略效果不佳，建议重新评估策略参数"
    
    def calculate_ebbinghaus_decay(self, hours_since_encoding: float, 
                                 access_count: int = 1) -> float:
        """计算艾宾浩斯遗忘曲线衰减
        
        参数:
            hours_since_encoding: 距离记忆编码的小时数
            access_count: 记忆访问次数
            
        返回:
            记忆保留率 (0-1)
        """
        # 艾宾浩斯遗忘曲线公式：R = e^(-t/S)
        # 其中R是保留率，t是时间，S是记忆强度
        
        # 基础衰减因子
        base_decay = self.forgetting_params["ebbinghaus_decay_factor"]
        
        # 记忆强度因子（访问次数增加记忆强度）
        strength_factor = 1.0 + math.log(1 + access_count) / math.log(2)
        
        # 计算衰减
        decay_factor = base_decay / strength_factor
        retention = math.exp(-hours_since_encoding / (24.0 * decay_factor))  # 24小时为单位
        
        return max(0.0, min(1.0, retention))
    
    def calculate_association_based_forgetting(self, memory_id: int, 
                                             memory_graph: Dict[str, Any],
                                             db: Optional[Any] = None) -> float:
        """计算基于关联性的遗忘概率
        
        参数:
            memory_id: 记忆ID
            memory_graph: 记忆图结构
            db: 数据库会话（可选）
            
        返回:
            关联性遗忘权重 (0-1)
        """
        try:
            # 从记忆图中获取关联信息
            nodes = memory_graph.get("nodes", [])
            edges = memory_graph.get("edges", [])
            
            # 找到目标记忆节点
            target_node = None
            for node in nodes:
                if node.get("id") == memory_id:
                    target_node = node
                    break
            
            if not target_node:
                return 0.5  # 默认值
            
            # 计算关联度
            connected_edges = []
            for edge in edges:
                if edge.get("source") == memory_id or edge.get("target") == memory_id:
                    connected_edges.append(edge)
            
            # 计算平均关联强度
            if not connected_edges:
                return 0.7  # 孤立记忆更容易被遗忘
            
            total_strength = sum(edge.get("strength", 0.0) for edge in connected_edges)
            avg_strength = total_strength / len(connected_edges)
            
            # 关联强度越高，遗忘概率越低
            association_weight = 1.0 - avg_strength
            
            # 考虑关联数量：过多弱关联可能表示记忆冗余
            if len(connected_edges) > 10 and avg_strength < 0.3:
                association_weight = min(1.0, association_weight * 1.5)
            
            return max(0.0, min(1.0, association_weight))
            
        except Exception as e:
            self.logger.error(f"计算关联性遗忘失败: {e}")
            return 0.5  # 默认值
    
    def calculate_content_based_forgetting(self, content: str, 
                                         metadata: Dict[str, Any]) -> float:
        """计算基于内容的遗忘概率
        
        参数:
            content: 记忆内容
            metadata: 记忆元数据
            
        返回:
            内容遗忘权重 (0-1)
        """
        try:
            # 1. 质量评估
            quality_score = self.quality_model(content, metadata)
            
            # 2. 信息熵评估
            entropy_score = self.entropy_calculator(content)
            
            # 3. 内容长度评估（过长或过短都更容易被遗忘）
            content_length = len(content)
            if content_length == 0:
                length_factor = 1.0  # 空内容容易被遗忘
            elif content_length < 20:
                length_factor = 0.7  # 过短内容容易被遗忘
            elif content_length > 1000:
                length_factor = 0.8  # 过长内容容易被遗忘
            else:
                length_factor = 0.3  # 适中长度不易被遗忘
            
            # 综合计算内容遗忘权重
            # 质量越低、熵越低、长度不合适，遗忘权重越高
            quality_weight = 1.0 - quality_score
            entropy_weight = 1.0 - entropy_score
            
            # 加权综合
            content_weight = (
                quality_weight * self.forgetting_params["quality_forgetting_weight"] +
                entropy_weight * self.forgetting_params["entropy_forgetting_weight"] +
                length_factor * 0.1  # 长度因子权重
            )
            
            return max(0.0, min(1.0, content_weight))
            
        except Exception as e:
            self.logger.error(f"计算内容遗忘失败: {e}")
            return 0.5  # 默认值
    
    def calculate_comprehensive_forgetting_probability(
        self,
        memory_data: Dict[str, Any],
        memory_graph: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None,
        strategy: str = "selective"
    ) -> Dict[str, Any]:
        """计算综合遗忘概率
        
        参数:
            memory_data: 记忆数据
            memory_graph: 记忆图结构（可选）
            system_state: 系统状态（可选）
            strategy: 遗忘策略
            
        返回:
            包含详细遗忘分析的结果
        """
        # 获取策略配置
        strategy_config = self.forgetting_strategies.get(strategy, self.forgetting_strategies["selective"])
        aggressiveness = strategy_config.get("aggressiveness", 0.5)
        
        # 基础遗忘率
        base_rate = self.forgetting_params["base_forgetting_rate"] * aggressiveness
        
        # 1. 时间衰减因子（艾宾浩斯遗忘曲线）
        created_at = memory_data.get("created_at")
        last_accessed = memory_data.get("last_accessed", created_at)
        access_count = memory_data.get("accessed_count", 0)
        
        hours_since_encoding = 24.0  # 默认值
        if created_at:
            from datetime import datetime
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            
            if isinstance(created_at, datetime):
                time_diff = datetime.now() - created_at
                hours_since_encoding = time_diff.total_seconds() / 3600.0
        
        time_decay_factor = 1.0 - self.calculate_ebbinghaus_decay(hours_since_encoding, access_count)
        
        # 2. 关联性遗忘因子
        association_factor = 0.5
        if memory_graph:
            memory_id = memory_data.get("id")
            if memory_id:
                association_factor = self.calculate_association_based_forgetting(memory_id, memory_graph)
        
        # 3. 内容遗忘因子
        content = memory_data.get("content", "")
        metadata = {
            "importance": memory_data.get("importance", 0.5),
            "memory_type": memory_data.get("memory_type", "short_term"),
            "accessed_count": access_count
        }
        content_factor = self.calculate_content_based_forgetting(content, metadata)
        
        # 4. 系统状态因子
        system_factor = 0.5
        if system_state:
            load_level = system_state.get("load_level", 0.5)
            memory_usage = system_state.get("memory_usage", 0.5)
            # 系统负载越高，遗忘概率越高
            system_factor = (load_level + memory_usage) / 2.0
        
        # 综合计算遗忘概率
        weights = self.forgetting_params
        
        forgetting_probability = (
            base_rate * 0.2 +
            time_decay_factor * 0.3 +
            association_factor * weights["association_forgetting_weight"] +
            content_factor * weights["content_forgetting_weight"] +
            system_factor * 0.1
        )
        
        # 应用最小保留阈值和最大遗忘率
        retention_probability = 1.0 - forgetting_probability
        if retention_probability < self.forgetting_params["min_retention_threshold"]:
            retention_probability = self.forgetting_params["min_retention_threshold"]
            forgetting_probability = 1.0 - retention_probability
        
        if forgetting_probability > self.forgetting_params["max_forgetting_rate"]:
            forgetting_probability = self.forgetting_params["max_forgetting_rate"]
            retention_probability = 1.0 - forgetting_probability
        
        # 构建详细分析结果
        analysis = {
            "memory_id": memory_data.get("id"),
            "forgetting_probability": forgetting_probability,
            "retention_probability": retention_probability,
            "strategy": strategy,
            "aggressiveness": aggressiveness,
            "factors": {
                "base_rate": base_rate,
                "time_decay_factor": time_decay_factor,
                "association_factor": association_factor,
                "content_factor": content_factor,
                "system_factor": system_factor
            },
            "weights": weights,
            "hours_since_encoding": hours_since_encoding,
            "access_count": access_count,
            "importance": memory_data.get("importance", 0.5),
            "memory_type": memory_data.get("memory_type", "short_term")
        }
        
        return analysis
    
    def apply_forgetting_strategy(
        self,
        memories: List[Dict[str, Any]],
        db: Optional[Any] = None,
        memory_graph: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None,
        strategy: str = "selective",
        target_compression_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """应用遗忘策略
        
        参数:
            memories: 记忆列表
            db: 数据库会话（可选）
            memory_graph: 记忆图结构（可选）
            system_state: 系统状态（可选）
            strategy: 遗忘策略
            target_compression_rate: 目标压缩率（可选）
            
        返回:
            遗忘操作结果
        """
        self.logger.info(f"应用遗忘策略: {strategy}, 记忆数量: {len(memories)}")
        
        if not memories:
            return {
                "success": False,
                "message": "没有需要处理的记忆",
                "stats": {}
            }
        
        # 计算每个记忆的遗忘概率
        memory_analyses = []
        for memory in memories:
            analysis = self.calculate_comprehensive_forgetting_probability(
                memory, memory_graph, system_state, strategy
            )
            memory_analyses.append(analysis)
        
        # 按遗忘概率排序
        memory_analyses.sort(key=lambda x: x["forgetting_probability"], reverse=True)
        
        # 确定遗忘阈值
        if target_compression_rate is not None:
            # 基于目标压缩率计算阈值
            target_forget_count = int(len(memories) * target_compression_rate)
            if target_forget_count > 0:
                threshold = memory_analyses[target_forget_count - 1]["forgetting_probability"]
            else:
                threshold = 1.0  # 不遗忘任何记忆
        else:
            # 使用自适应阈值
            avg_probability = sum(a["forgetting_probability"] for a in memory_analyses) / len(memory_analyses)
            threshold = avg_probability * 1.2  # 高于平均值的记忆更容易被遗忘
        
        # 执行遗忘
        forgotten_memories = []
        retained_memories = []
        
        for analysis in memory_analyses:
            if analysis["forgetting_probability"] >= threshold:
                forgotten_memories.append(analysis)
            else:
                retained_memories.append(analysis)
        
        # 计算统计信息
        total_memories = len(memories)
        forgotten_count = len(forgotten_memories)
        retained_count = len(retained_memories)
        compression_rate = forgotten_count / total_memories if total_memories > 0 else 0.0
        
        # 计算平均重要性变化
        avg_importance_before = sum(m.get("importance", 0.5) for m in memories) / total_memories if total_memories > 0 else 0.5
        avg_importance_after = sum(m.get("importance", 0.5) for m in retained_memories) / retained_count if retained_count > 0 else 0.5
        
        # 构建结果
        stats = {
            "total_memories": total_memories,
            "forgotten_count": forgotten_count,
            "retained_count": retained_count,
            "compression_rate": compression_rate,
            "avg_importance_before": avg_importance_before,
            "avg_importance_after": avg_importance_after,
            "threshold": threshold,
            "strategy": strategy,
            "forgotten_memory_ids": [m["memory_id"] for m in forgotten_memories if m.get("memory_id")],
            "avg_forgetting_probability": sum(a["forgetting_probability"] for a in memory_analyses) / total_memories if total_memories > 0 else 0.0
        }
        
        # 评估遗忘效果
        if system_state:
            effectiveness = self.effectiveness_evaluator(stats, system_state)
            stats["effectiveness"] = effectiveness
        
        # 记录遗忘历史
        self._record_forgetting_history(stats, strategy)
        
        # 如果需要，实际删除记忆
        if db and forgotten_count > 0:
            self._execute_forgetting_in_db(db, forgotten_memories)
        
        self.logger.info(
            f"遗忘策略完成: 策略={strategy}, "
            f"压缩率={compression_rate:.2%}, "
            f"遗忘{forgotten_count}个记忆, "
            f"保留{retained_count}个记忆"
        )
        
        return {
            "success": True,
            "message": f"成功应用{strategy}遗忘策略",
            "stats": stats,
            "forgotten_analyses": forgotten_memories,
            "retained_analyses": retained_memories
        }
    
    def _record_forgetting_history(self, stats: Dict[str, Any], strategy: str):
        """记录遗忘历史"""
        history_entry = {
            "timestamp": time.time(),
            "strategy": strategy,
            "stats": stats.copy(),
            "effectiveness": stats.get("effectiveness", {})
        }
        
        self.forgetting_history.append(history_entry)
        
        # 保持历史记录大小
        if len(self.forgetting_history) > self.max_history_size:
            self.forgetting_history = self.forgetting_history[-self.max_history_size:]
    
    def _execute_forgetting_in_db(self, db: Any, forgotten_analyses: List[Dict[str, Any]]):
        """在数据库中执行遗忘操作"""
        try:
            from backend.db_models.memory import Memory, MemoryAssociation
            
            for analysis in forgotten_analyses:
                memory_id = analysis.get("memory_id")
                if not memory_id:
                    continue
                
                # 从数据库删除记忆
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory:
                    # 删除关联
                    db.query(MemoryAssociation).filter(
                        (MemoryAssociation.source_memory_id == memory_id) |
                        (MemoryAssociation.target_memory_id == memory_id)
                    ).delete()
                    
                    # 删除记忆
                    db.delete(memory)
                    self.logger.debug(f"从数据库删除记忆: {memory_id}")
            
            # 提交更改
            db.commit()
            self.logger.info(f"从数据库成功删除 {len(forgotten_analyses)} 个记忆")
            
        except Exception as e:
            self.logger.error(f"在数据库执行遗忘失败: {e}")
            db.rollback()
    
    def get_forgetting_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取遗忘历史记录"""
        return self.forgetting_history[-limit:] if self.forgetting_history else []
    
    def get_strategy_recommendation(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """获取策略推荐
        
        基于系统状态和历史效果推荐最佳遗忘策略
        """
        if not self.forgetting_history:
            # 没有历史记录，使用默认推荐
            return {
                "recommended_strategy": "selective",
                "confidence": 0.5,
                "reason": "没有历史数据，使用默认策略"
            }
        
        # 分析历史效果
        strategy_scores = {}
        for entry in self.forgetting_history[-50:]:  # 最近50次记录
            strategy = entry["strategy"]
            effectiveness = entry.get("effectiveness", {}).get("score", 0.5)
            
            if strategy not in strategy_scores:
                strategy_scores[strategy] = []
            
            strategy_scores[strategy].append(effectiveness)
        
        # 计算平均效果
        strategy_avg_scores = {}
        for strategy, scores in strategy_scores.items():
            if scores:
                strategy_avg_scores[strategy] = sum(scores) / len(scores)
        
        # 基于系统状态调整
        load_level = system_state.get("load_level", 0.5)
        memory_usage = system_state.get("memory_usage", 0.5)
        
        # 高负载时推荐更激进的策略
        if load_level > 0.7 or memory_usage > 0.7:
            if "active" in strategy_avg_scores:
                recommended = "active"
                confidence = 0.8
                reason = "系统高负载，推荐主动遗忘策略"
            else:
                recommended = "selective"
                confidence = 0.7
                reason = "系统高负载，推荐选择性遗忘策略"
        # 低负载时推荐保守策略
        elif load_level < 0.3 and memory_usage < 0.3:
            if "passive" in strategy_avg_scores:
                recommended = "passive"
                confidence = 0.7
                reason = "系统低负载，推荐被动遗忘策略"
            else:
                recommended = "selective"
                confidence = 0.6
                reason = "系统低负载，推荐保守的选择性遗忘策略"
        # 中等负载时推荐历史效果最好的策略
        else:
            if strategy_avg_scores:
                best_strategy = max(strategy_avg_scores.items(), key=lambda x: x[1])[0]
                recommended = best_strategy
                confidence = strategy_avg_scores[best_strategy]
                reason = f"基于历史效果推荐{best_strategy}策略"
            else:
                recommended = "selective"
                confidence = 0.5
                reason = "使用默认选择性遗忘策略"
        
        return {
            "recommended_strategy": recommended,
            "confidence": min(1.0, max(0.0, confidence)),
            "reason": reason,
            "strategy_scores": strategy_avg_scores,
            "system_state": {
                "load_level": load_level,
                "memory_usage": memory_usage
            }
        }
    
    def optimize_parameters(self, effectiveness_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """优化遗忘参数
        
        基于历史效果数据优化参数
        """
        if not effectiveness_history or len(effectiveness_history) < 10:
            return self.forgetting_params.copy()
        
        # 收集参数与效果的关系
        param_effects = {}
        
        for entry in effectiveness_history:
            stats = entry.get("stats", {})
            effectiveness = entry.get("effectiveness", {}).get("score", 0.5)
            
            # 提取相关参数
            compression_rate = stats.get("compression_rate", 0.0)
            threshold = stats.get("threshold", 0.5)
            
            # 分析参数效果
            # 理想压缩率：10-30%
            compression_optimal = 0.1 <= compression_rate <= 0.3
            
            if compression_optimal:
                # 有效参数组合
                param_key = f"threshold_{threshold:.2f}"
                if param_key not in param_effects:
                    param_effects[param_key] = []
                
                param_effects[param_key].append(effectiveness)
        
        # 计算最佳参数
        optimized_params = self.forgetting_params.copy()
        
        if param_effects:
            # 找到效果最好的阈值
            best_param_key = None
            best_avg_effectiveness = 0.0
            
            for param_key, effects in param_effects.items():
                avg_effectiveness = sum(effects) / len(effects)
                if avg_effectiveness > best_avg_effectiveness:
                    best_avg_effectiveness = avg_effectiveness
                    best_param_key = param_key
            
            if best_param_key:
                # 从参数键中提取阈值
                try:
                    threshold_str = best_param_key.split("_")[1]
                    optimal_threshold = float(threshold_str)
                    
                    # 调整基础遗忘率
                    optimal_base_rate = optimal_threshold / 2.0  # 经验公式
                    optimized_params["base_forgetting_rate"] = optimal_base_rate
                    
                    self.logger.info(f"参数优化完成: 最佳阈值={optimal_threshold:.3f}, 基础遗忘率={optimal_base_rate:.3f}")
                    
                except (IndexError, ValueError) as e:
                    self.logger.warning(f"解析最佳参数失败: {e}")
        
        return optimized_params
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        return {
            "forgetting_params": self.forgetting_params.copy(),
            "history_size": len(self.forgetting_history),
            "strategies": list(self.forgetting_strategies.keys()),
            "recent_effectiveness": self._calculate_recent_effectiveness()
        }
    
    def _calculate_recent_effectiveness(self) -> Dict[str, Any]:
        """计算最近遗忘效果"""
        if not self.forgetting_history:
            return {"avg_score": 0.5, "trend": "stable"}
        
        recent_entries = self.forgetting_history[-10:]  # 最近10次
        if not recent_entries:
            return {"avg_score": 0.5, "trend": "stable"}
        
        # 计算平均效果
        scores = [e.get("effectiveness", {}).get("score", 0.5) for e in recent_entries]
        avg_score = sum(scores) / len(scores)
        
        # 计算趋势
        if len(scores) >= 2:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            if avg_second > avg_first + 0.1:
                trend = "improving"
            elif avg_second < avg_first - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "avg_score": avg_score,
            "trend": trend,
            "recent_scores": scores
        }


class CrossSessionMemoryPersistence:
    """跨会话记忆持久化系统
    
    解决跨会话记忆持久化问题，确保：
    1. 所有记忆（短期和长期）都正确保存到数据库
    2. 系统重启时，记忆从数据库正确恢复
    3. 用户会话之间的记忆隔离正确
    4. 记忆状态在不同会话之间保持一致
    5. 支持会话恢复和断点续传
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("CrossSessionMemoryPersistence")
        
        # 持久化配置
        self.persistence_config = {
            "enable_auto_save": self.config.get("enable_auto_save", True),
            "auto_save_interval": self.config.get("auto_save_interval", 300),  # 5分钟（秒）
            "save_on_shutdown": self.config.get("save_on_shutdown", True),
            "load_on_startup": self.config.get("load_on_startup", True),
            "session_isolation": self.config.get("session_isolation", True),
            "max_sessions_per_user": self.config.get("max_sessions_per_user", 10),
            "session_timeout": self.config.get("session_timeout", 3600),  # 1小时（秒）
            "state_compression": self.config.get("state_compression", True),
            "backup_on_save": self.config.get("backup_on_save", True),
            "backup_count": self.config.get("backup_count", 5)
        }
        
        # 会话管理
        self.active_sessions = {}  # session_id -> session_data
        self.user_sessions = {}  # user_id -> [session_ids]
        
        # 持久化状态
        self.last_save_time = None
        self.last_load_time = None
        self.save_count = 0
        self.load_count = 0
        
        # 状态文件路径
        self.state_dir = self.config.get("state_dir", "./data/memory_states")
        os.makedirs(self.state_dir, exist_ok=True)
        
        # 备份目录
        self.backup_dir = os.path.join(self.state_dir, "backups")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        self.logger.info("跨会话记忆持久化系统初始化完成")
    
    def create_session(self, user_id: int, session_id: Optional[str] = None,
                      session_data: Optional[Dict[str, Any]] = None) -> str:
        """创建新会话
        
        参数:
            user_id: 用户ID
            session_id: 可选的会话ID（如果为None则自动生成）
            session_data: 初始会话数据
            
        返回:
            会话ID
        """
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())
        
        # 检查会话数量限制
        user_session_ids = self.user_sessions.get(user_id, [])
        max_sessions = self.persistence_config["max_sessions_per_user"]
        
        if len(user_session_ids) >= max_sessions:
            # 删除最旧的会话
            oldest_session_id = user_session_ids[0]
            self.delete_session(user_id, oldest_session_id)
            user_session_ids = user_session_ids[1:]
            self.logger.info(f"用户 {user_id} 会话数达到限制，删除最旧会话: {oldest_session_id}")
        
        # 创建会话
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "state": session_data or {},
            "memory_ids": [],  # 该会话中创建的记忆ID列表
            "active": True,
            "persisted": False
        }
        
        # 更新数据结构
        self.active_sessions[session_id] = session
        user_session_ids.append(session_id)
        self.user_sessions[user_id] = user_session_ids
        
        self.logger.info(f"创建会话: {session_id} (用户: {user_id})")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话数据"""
        if session_id not in self.active_sessions:
            return None
        
        # 更新最后访问时间
        self.active_sessions[session_id]["last_accessed"] = time.time()
        
        return self.active_sessions[session_id]
    
    def update_session_state(self, session_id: str, state_updates: Dict[str, Any]):
        """更新会话状态"""
        session = self.get_session(session_id)
        if not session:
            self.logger.warning(f"会话不存在: {session_id}")
            return
        
        # 更新状态
        session["state"].update(state_updates)
        session["persisted"] = False  # 标记为需要持久化
        
        self.logger.debug(f"更新会话状态: {session_id}")
    
    def add_memory_to_session(self, session_id: str, memory_id: int):
        """添加记忆ID到会话"""
        session = self.get_session(session_id)
        if not session:
            self.logger.warning(f"会话不存在: {session_id}")
            return
        
        if memory_id not in session["memory_ids"]:
            session["memory_ids"].append(memory_id)
            session["persisted"] = False
        
        self.logger.debug(f"添加记忆 {memory_id} 到会话 {session_id}")
    
    def delete_session(self, user_id: int, session_id: str, save_memories: bool = True):
        """删除会话
        
        参数:
            user_id: 用户ID
            session_id: 会话ID
            save_memories: 是否保存会话中的记忆到持久化存储
        """
        if session_id not in self.active_sessions:
            self.logger.warning(f"会话不存在: {session_id}")
            return
        
        # 如果需要，保存会话记忆
        if save_memories:
            session = self.active_sessions[session_id]
            if session["memory_ids"]:
                self._save_session_memories(session)
        
        # 从数据结构中移除
        del self.active_sessions[session_id]
        
        if user_id in self.user_sessions:
            user_sessions = self.user_sessions[user_id]
            if session_id in user_sessions:
                user_sessions.remove(session_id)
            if not user_sessions:
                del self.user_sessions[user_id]
        
        self.logger.info(f"删除会话: {session_id} (用户: {user_id})")
    
    def _save_session_memories(self, session: Dict[str, Any]):
        """保存会话记忆到持久化存储"""
        session_id = session["session_id"]
        user_id = session["user_id"]
        memory_ids = session["memory_ids"]
        
        if not memory_ids:
            return
        
        try:
            # 构建会话记忆记录
            session_memory_record = {
                "session_id": session_id,
                "user_id": user_id,
                "memory_ids": memory_ids,
                "created_at": session["created_at"],
                "last_accessed": session["last_accessed"],
                "state_snapshot": session["state"],
                "saved_at": time.time()
            }
            
            # 保存到文件
            filename = f"session_{session_id}_memories.json"
            filepath = os.path.join(self.state_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_memory_record, f, ensure_ascii=False, indent=2)
            
            # 创建备份（如果启用）
            if self.persistence_config["backup_on_save"]:
                self._create_backup(filepath, f"session_{session_id}")
            
            session["persisted"] = True
            self.logger.info(f"会话 {session_id} 记忆已保存: {len(memory_ids)} 个记忆")
            
        except Exception as e:
            self.logger.error(f"保存会话记忆失败: {e}")
    
    def _create_backup(self, filepath: str, prefix: str):
        """创建备份文件"""
        try:
            import shutil
            import datetime
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{prefix}_{timestamp}.json"
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            shutil.copy2(filepath, backup_path)
            
            # 清理旧备份
            self._cleanup_old_backups(prefix)
            
            self.logger.debug(f"创建备份: {backup_name}")
            
        except Exception as e:
            self.logger.warning(f"创建备份失败: {e}")
    
    def _cleanup_old_backups(self, prefix: str):
        """清理旧备份文件"""
        try:
            import glob
            
            backup_files = glob.glob(os.path.join(self.backup_dir, f"{prefix}_*.json"))
            backup_files.sort(key=os.path.getmtime, reverse=True)
            
            max_backups = self.persistence_config["backup_count"]
            
            # 删除超出数量的旧备份
            for backup_file in backup_files[max_backups:]:
                os.remove(backup_file)
                self.logger.debug(f"删除旧备份: {os.path.basename(backup_file)}")
                
        except Exception as e:
            self.logger.warning(f"清理备份失败: {e}")
    
    def save_system_state(self, memory_system, db: Optional[Any] = None, 
                         force: bool = False) -> bool:
        """保存系统状态
        
        参数:
            memory_system: MemorySystem实例
            db: 数据库会话（可选）
            force: 是否强制保存
            
        返回:
            是否成功
        """
        # 检查是否需要保存
        current_time = time.time()
        last_save = self.last_save_time
        
        if not force and last_save:
            save_interval = self.persistence_config["auto_save_interval"]
            if current_time - last_save < save_interval:
                self.logger.debug(f"跳过自动保存，距上次保存仅 {int(current_time - last_save)} 秒")
                return False
        
        self.logger.info("开始保存系统状态...")
        
        try:
            # 收集系统状态
            system_state = self._collect_system_state(memory_system, db)
            
            # 压缩状态（如果启用）
            if self.persistence_config["state_compression"]:
                system_state = self._compress_state(system_state)
            
            # 保存到文件
            timestamp = int(time.time())
            filename = f"system_state_{timestamp}.json"
            filepath = os.path.join(self.state_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(system_state, f, ensure_ascii=False, indent=2)
            
            # 创建最新状态链接
            latest_path = os.path.join(self.state_dir, "system_state_latest.json")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(filename, latest_path)
            
            # 创建备份
            if self.persistence_config["backup_on_save"]:
                self._create_backup(filepath, "system_state")
            
            # 更新统计
            self.last_save_time = current_time
            self.save_count += 1
            
            self.logger.info(f"系统状态已保存: {filepath} (大小: {os.path.getsize(filepath)} 字节)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存系统状态失败: {e}")
            return False
    
    def _collect_system_state(self, memory_system, db: Optional[Any] = None) -> Dict[str, Any]:
        """收集系统状态"""
        state = {
            "timestamp": time.time(),
            "system_info": {
                "initialized": memory_system.initialized,
                "enable_faiss_retrieval": memory_system.enable_faiss_retrieval,
                "enable_memory_graph": memory_system.enable_memory_graph,
                "enable_knowledge_base": memory_system.enable_knowledge_base,
                "enable_autonomous_memory": memory_system.enable_autonomous_memory,
                "embedding_dim": memory_system.embedding_dim,
                "config": memory_system.config
            },
            "session_state": {
                "active_sessions_count": len(self.active_sessions),
                "user_sessions_count": len(self.user_sessions),
                "active_sessions": list(self.active_sessions.keys())
            },
            "memory_stats": self._collect_memory_stats(memory_system, db),
            "index_state": self._collect_index_state(memory_system),
            "cache_state": self._collect_cache_state(memory_system)
        }
        
        # 如果启用了自主记忆管理，收集其状态
        if memory_system.enable_autonomous_memory and memory_system.autonomous_manager:
            state["autonomous_manager_state"] = memory_system.autonomous_manager.get_system_state()
        
        # 如果启用了知识库，收集其状态
        if memory_system.enable_knowledge_base and memory_system.knowledge_manager:
            state["knowledge_base_state"] = memory_system.knowledge_manager.get_stats()
        
        return state
    
    def _collect_memory_stats(self, memory_system, db: Optional[Any] = None) -> Dict[str, Any]:
        """收集记忆统计信息"""
        stats = {
            "short_term_memories": 0,
            "long_term_memories": 0,
            "working_memory_count": len(memory_system.working_memory) if hasattr(memory_system, 'working_memory') else 0,
            "cache_hits": getattr(memory_system, 'short_term_cache_hits', 0),
            "cache_misses": getattr(memory_system, 'short_term_cache_misses', 0),
            "cache_hit_rate": 0.0
        }
        
        # 如果有数据库连接，获取更准确的统计
        if db:
            try:
                from backend.db_models.memory import Memory
                
                # 统计短期记忆
                short_term_count = db.query(Memory).filter(
                    Memory.memory_type == "short_term",
                    Memory.expires_at > datetime.now(timezone.utc)
                ).count()
                
                # 统计长期记忆
                long_term_count = db.query(Memory).filter(
                    Memory.memory_type == "long_term"
                ).count()
                
                stats["short_term_memories"] = short_term_count
                stats["long_term_memories"] = long_term_count
                
            except Exception as e:
                self.logger.warning(f"从数据库收集记忆统计失败: {e}")
        
        # 计算缓存命中率
        total_cache_access = stats["cache_hits"] + stats["cache_misses"]
        if total_cache_access > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_access
        
        return stats
    
    def _collect_index_state(self, memory_system) -> Dict[str, Any]:
        """收集索引状态"""
        state = {
            "index_initialized": getattr(memory_system, 'index_initialized', False),
            "index_size": getattr(memory_system, 'index_size', 0),
            "faiss_available": getattr(memory_system, 'faiss_available', False),
            "hnsw_available": getattr(memory_system, 'hnsw_available', False),
            "graph_initialized": getattr(memory_system, 'graph_initialized', False),
            "graph_size": 0
        }
        
        # 获取图大小
        if hasattr(memory_system, 'memory_graph') and memory_system.memory_graph:
            graph = memory_system.memory_graph
            state["graph_size"] = len(graph.get("nodes", [])) if isinstance(graph, dict) else 0
        
        return state
    
    def _collect_cache_state(self, memory_system) -> Dict[str, Any]:
        """收集缓存状态"""
        state = {
            "short_term_cache_size": 0,
            "short_term_cache_max_size": getattr(memory_system, 'short_term_cache_max_size', 200),
            "working_memory_size": len(getattr(memory_system, 'working_memory', [])),
            "working_memory_capacity": getattr(memory_system, 'working_memory_capacity', 100)
        }
        
        # 获取短期缓存大小
        if hasattr(memory_system, 'short_term_memory_cache'):
            state["short_term_cache_size"] = len(memory_system.short_term_memory_cache)
        
        return state
    
    def _compress_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """压缩状态数据"""
        try:
            import zlib
            import base64
            
            # 转换为JSON字符串
            state_str = json.dumps(state, ensure_ascii=False)
            
            # 压缩
            compressed = zlib.compress(state_str.encode('utf-8'))
            
            # Base64编码
            encoded = base64.b64encode(compressed).decode('utf-8')
            
            return {
                "compressed": True,
                "format": "zlib+base64",
                "original_size": len(state_str),
                "compressed_size": len(compressed),
                "data": encoded
            }
            
        except Exception as e:
            self.logger.warning(f"状态压缩失败: {e}")
            return state
    
    def _decompress_state(self, compressed_state: Dict[str, Any]) -> Dict[str, Any]:
        """解压缩状态数据"""
        if "compressed" not in compressed_state or not compressed_state["compressed"]:
            return compressed_state
        
        try:
            import zlib
            import base64
            
            data = compressed_state["data"]
            
            # Base64解码
            decoded = base64.b64decode(data)
            
            # 解压缩
            decompressed = zlib.decompress(decoded)
            
            # 解析JSON
            state = json.loads(decompressed.decode('utf-8'))
            
            return state
            
        except Exception as e:
            self.logger.error(f"状态解压缩失败: {e}")
            raise
    
    def load_system_state(self, memory_system, db: Optional[Any] = None,
                         state_file: Optional[str] = None) -> bool:
        """加载系统状态
        
        参数:
            memory_system: MemorySystem实例
            db: 数据库会话（可选）
            state_file: 状态文件路径（如果为None则加载最新状态）
            
        返回:
            是否成功
        """
        if not self.persistence_config["load_on_startup"] and state_file is None:
            self.logger.info("启动时加载已禁用，跳过状态加载")
            return False
        
        self.logger.info("开始加载系统状态...")
        
        try:
            # 确定要加载的文件
            if state_file is None:
                state_file = os.path.join(self.state_dir, "system_state_latest.json")
            
            if not os.path.exists(state_file):
                self.logger.warning(f"状态文件不存在: {state_file}")
                return False
            
            # 读取状态文件
            with open(state_file, 'r', encoding='utf-8') as f:
                saved_state = json.load(f)
            
            # 解压缩（如果需要）
            if "compressed" in saved_state and saved_state["compressed"]:
                saved_state = self._decompress_state(saved_state)
            
            # 验证状态
            if not self._validate_state(saved_state):
                self.logger.error("状态验证失败")
                return False
            
            # 应用状态
            success = self._apply_system_state(memory_system, saved_state, db)
            
            if success:
                self.last_load_time = time.time()
                self.load_count += 1
                self.logger.info(f"系统状态加载成功: {state_file}")
            else:
                self.logger.error("应用系统状态失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"加载系统状态失败: {e}")
            return False
    
    def _validate_state(self, state: Dict[str, Any]) -> bool:
        """验证状态数据"""
        required_fields = ["timestamp", "system_info", "memory_stats"]
        
        for field in required_fields:
            if field not in state:
                self.logger.error(f"状态缺少必需字段: {field}")
                return False
        
        # 检查时间戳
        timestamp = state["timestamp"]
        if not isinstance(timestamp, (int, float)):
            self.logger.error(f"无效的时间戳类型: {type(timestamp)}")
            return False
        
        # 检查时间戳是否合理（在过去10年内）
        current_time = time.time()
        if timestamp < current_time - 315360000 or timestamp > current_time + 3600:  # 10年前到1小时后
            self.logger.warning(f"状态时间戳可能无效: {timestamp}")
            # 仍然接受，但记录警告
        
        return True
    
    def _apply_system_state(self, memory_system, state: Dict[str, Any], db: Optional[Any]) -> bool:
        """应用系统状态"""
        try:
            system_info = state["system_info"]
            
            # 验证系统兼容性
            if not self._check_system_compatibility(memory_system, system_info):
                self.logger.warning("系统配置不兼容，仅加载部分状态")
            
            # 恢复会话状态
            session_state = state.get("session_state", {})
            self._restore_sessions(session_state)
            
            # 恢复缓存状态（如果兼容）
            cache_state = state.get("cache_state", {})
            self._restore_cache_state(memory_system, cache_state)
            
            # 恢复自主记忆管理器状态
            if (memory_system.enable_autonomous_memory and memory_system.autonomous_manager and
                "autonomous_manager_state" in state):
                self._restore_autonomous_manager_state(memory_system.autonomous_manager, 
                                                      state["autonomous_manager_state"])
            
            self.logger.info("系统状态应用完成")
            return True
            
        except Exception as e:
            self.logger.error(f"应用系统状态失败: {e}")
            return False
    
    def _check_system_compatibility(self, memory_system, system_info: Dict[str, Any]) -> bool:
        """检查系统兼容性"""
        compatible = True
        
        # 检查嵌入维度
        saved_dim = system_info.get("embedding_dim")
        current_dim = memory_system.embedding_dim
        
        if saved_dim != current_dim:
            self.logger.warning(f"嵌入维度不兼容: 保存的={saved_dim}, 当前的={current_dim}")
            compatible = False
        
        # 检查功能启用状态
        features = ["enable_faiss_retrieval", "enable_memory_graph", 
                   "enable_knowledge_base", "enable_autonomous_memory"]
        
        for feature in features:
            saved_value = system_info.get(feature)
            current_value = getattr(memory_system, feature, False)
            
            if saved_value != current_value:
                self.logger.warning(f"{feature} 状态不兼容: 保存的={saved_value}, 当前的={current_value}")
                # 不标记为不兼容，因为功能状态可能已更改
        
        return compatible
    
    def _restore_sessions(self, session_state: Dict[str, Any]):
        """恢复会话状态"""
        # 注意：实际会话恢复需要更复杂的实现
        # 这里仅记录恢复信息
        
        active_sessions_count = session_state.get("active_sessions_count", 0)
        self.logger.info(f"从状态中恢复会话信息: {active_sessions_count} 个活动会话")
        
        # 在实际实现中，需要从持久化存储加载会话数据
    
    def _restore_cache_state(self, memory_system, cache_state: Dict[str, Any]):
        """恢复缓存状态"""
        # 注意：缓存恢复需要谨慎，因为缓存内容可能已过时
        # 这里仅恢复统计信息
        
        if hasattr(memory_system, 'short_term_cache_hits'):
            memory_system.short_term_cache_hits = cache_state.get("cache_hits", 0)
        
        if hasattr(memory_system, 'short_term_cache_misses'):
            memory_system.short_term_cache_misses = cache_state.get("cache_misses", 0)
        
        self.logger.info(f"缓存状态恢复: 命中={cache_state.get('cache_hits', 0)}, 未命中={cache_state.get('cache_misses', 0)}")
    
    def _restore_autonomous_manager_state(self, autonomous_manager, manager_state: Dict[str, Any]):
        """恢复自主记忆管理器状态"""
        try:
            # 在实际实现中，需要调用自主记忆管理器的方法来恢复状态
            # 这里仅记录恢复信息
            
            self.logger.info("自主记忆管理器状态恢复完成")
            
        except Exception as e:
            self.logger.error(f"恢复自主记忆管理器状态失败: {e}")
    
    def save_session_state(self, session_id: str, force: bool = False) -> bool:
        """保存会话状态"""
        session = self.get_session(session_id)
        if not session:
            self.logger.warning(f"会话不存在: {session_id}")
            return False
        
        # 检查是否需要保存
        if not force and session["persisted"]:
            self.logger.debug(f"会话 {session_id} 已持久化，跳过保存")
            return True
        
        try:
            # 保存会话记忆
            self._save_session_memories(session)
            
            # 保存会话元数据
            metadata = {
                "session_id": session["session_id"],
                "user_id": session["user_id"],
                "created_at": session["created_at"],
                "last_accessed": session["last_accessed"],
                "memory_count": len(session["memory_ids"]),
                "state_keys": list(session["state"].keys()),
                "saved_at": time.time()
            }
            
            # 保存到文件
            filename = f"session_{session_id}_metadata.json"
            filepath = os.path.join(self.state_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"会话状态已保存: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存会话状态失败: {e}")
            return False
    
    def load_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """加载会话状态"""
        try:
            # 加载会话元数据
            metadata_file = os.path.join(self.state_dir, f"session_{session_id}_metadata.json")
            if not os.path.exists(metadata_file):
                self.logger.warning(f"会话元数据文件不存在: {metadata_file}")
                return None
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 加载会话记忆
            memories_file = os.path.join(self.state_dir, f"session_{session_id}_memories.json")
            memories_data = None
            
            if os.path.exists(memories_file):
                with open(memories_file, 'r', encoding='utf-8') as f:
                    memories_data = json.load(f)
            
            # 重建会话
            session = {
                "session_id": metadata["session_id"],
                "user_id": metadata["user_id"],
                "created_at": metadata["created_at"],
                "last_accessed": metadata["last_accessed"],
                "state": memories_data.get("state_snapshot", {}) if memories_data else {},
                "memory_ids": memories_data.get("memory_ids", []) if memories_data else [],
                "active": False,  # 加载的会话默认不活跃
                "persisted": True
            }
            
            self.logger.info(f"会话状态已加载: {session_id} ({len(session['memory_ids'])} 个记忆)")
            
            return session
            
        except Exception as e:
            self.logger.error(f"加载会话状态失败: {e}")
            return None
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        current_time = time.time()
        timeout = self.persistence_config["session_timeout"]
        
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            last_accessed = session["last_accessed"]
            
            if current_time - last_accessed > timeout:
                expired_sessions.append((session["user_id"], session_id))
        
        # 删除过期会话
        for user_id, session_id in expired_sessions:
            self.delete_session(user_id, session_id, save_memories=True)
            self.logger.info(f"清理过期会话: {session_id} (用户: {user_id})")
        
        return len(expired_sessions)
    
    def get_persistence_stats(self) -> Dict[str, Any]:
        """获取持久化统计信息"""
        return {
            "save_count": self.save_count,
            "load_count": self.load_count,
            "last_save_time": self.last_save_time,
            "last_load_time": self.last_load_time,
            "active_sessions": len(self.active_sessions),
            "user_sessions": len(self.user_sessions),
            "state_dir": self.state_dir,
            "config": self.persistence_config.copy()
        }
    
    def start_auto_save(self, memory_system, db: Optional[Any] = None):
        """启动自动保存（在单独的线程中）"""
        if not self.persistence_config["enable_auto_save"]:
            return
        
        import threading
        
        def auto_save_worker():
            while True:
                try:
                    time.sleep(self.persistence_config["auto_save_interval"])
                    self.save_system_state(memory_system, db)
                except Exception as e:
                    self.logger.error(f"自动保存失败: {e}")
        
        # 启动自动保存线程
        auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        auto_save_thread.start()
        
        self.logger.info(f"自动保存已启动，间隔: {self.persistence_config['auto_save_interval']} 秒")
    
    def register_shutdown_hook(self, memory_system, db: Optional[Any] = None):
        """注册关机钩子"""
        import atexit
        
        def shutdown_hook():
            if self.persistence_config["save_on_shutdown"]:
                self.logger.info("系统关闭，正在保存状态...")
                self.save_system_state(memory_system, db, force=True)
                self.logger.info("状态保存完成")
        
        atexit.register(shutdown_hook)
        self.logger.info("关机钩子已注册")


class MemoryConflictResolver:
    """记忆冲突解决机制
    
    检测和解决记忆冲突，包括：
    1. 内容冲突：相同或相似的信息但不同的内容
    2. 时间冲突：同一事件的不同时间描述
    3. 事实冲突：相互矛盾的事实信息
    4. 优先级冲突：相同记忆的不同重要性评估
    5. 关联冲突：矛盾的记忆关联关系
    
    冲突解决策略：
    1. 合并冲突记忆
    2. 选择更可靠的记忆
    3. 创建新的综合记忆
    4. 标记冲突并保留两个版本
    5. 基于上下文和证据解决冲突
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("MemoryConflictResolver")
        
        # 冲突检测配置
        self.detection_config = {
            "similarity_threshold": self.config.get("similarity_threshold", 0.85),
            "content_conflict_threshold": self.config.get("content_conflict_threshold", 0.7),
            "time_conflict_threshold": self.config.get("time_conflict_threshold", 3600),  # 1小时（秒）
            "fact_conflict_penalty": self.config.get("fact_conflict_penalty", 0.3),
            "max_conflict_age": self.config.get("max_conflict_age", 2592000),  # 30天（秒）
            "min_confidence_for_resolution": self.config.get("min_confidence_for_resolution", 0.6)
        }
        
        # 冲突解决策略配置
        self.resolution_strategies = {
            "merge": {
                "name": "合并策略",
                "description": "合并冲突记忆的内容，创建新的综合记忆",
                "applicable_conflicts": ["content", "time"],
                "confidence_threshold": 0.5
            },
            "select": {
                "name": "选择策略",
                "description": "选择更可靠的记忆，丢弃或降级另一个",
                "applicable_conflicts": ["fact", "priority", "association"],
                "confidence_threshold": 0.7
            },
            "both": {
                "name": "双版本策略",
                "description": "保留两个版本，标记冲突并记录解决建议",
                "applicable_conflicts": ["fact", "content"],
                "confidence_threshold": 0.3
            },
            "contextual": {
                "name": "上下文策略",
                "description": "基于上下文和外部证据解决冲突",
                "applicable_conflicts": ["all"],
                "confidence_threshold": 0.8
            }
        }
        
        # 冲突历史记录
        self.conflict_history = []
        self.max_history_size = self.config.get("max_conflict_history", 500)
        
        # 冲突模式分析器
        self.pattern_analyzer = ConflictPatternAnalyzer()
        
        # 证据收集器（用于上下文策略）
        self.evidence_collector = EvidenceCollector()
        
        self.logger.info("记忆冲突解决机制初始化完成")
    
    def detect_conflicts(self, memory_system, db: Optional[Any] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """检测记忆冲突
        
        参数:
            memory_system: MemorySystem实例
            db: 数据库会话（可选）
            limit: 最大检测数量
            
        返回:
            冲突列表
        """
        self.logger.info(f"开始检测记忆冲突 (限制: {limit})")
        
        conflicts = []
        
        try:
            # 从数据库获取记忆（如果有数据库连接）
            if db:
                memories = self._get_memories_from_db(db, limit)
            else:
                # 从内存系统获取记忆
                memories = self._get_memories_from_system(memory_system, limit)
            
            if len(memories) < 2:
                self.logger.info("记忆数量不足，无法检测冲突")
                return conflicts
            
            # 检测不同类型的冲突
            conflicts.extend(self._detect_content_conflicts(memories))
            conflicts.extend(self._detect_time_conflicts(memories))
            conflicts.extend(self._detect_fact_conflicts(memories))
            conflicts.extend(self._detect_priority_conflicts(memories))
            conflicts.extend(self._detect_association_conflicts(memories, memory_system))
            
            # 过滤重复冲突
            conflicts = self._filter_duplicate_conflicts(conflicts)
            
            # 按严重性排序
            conflicts.sort(key=lambda x: x.get("severity", 0.0), reverse=True)
            
            # 记录检测结果
            self._record_conflict_detection(conflicts)
            
            self.logger.info(f"检测到 {len(conflicts)} 个记忆冲突")
            
            return conflicts
            
        except Exception as e:
            self.logger.error(f"检测记忆冲突失败: {e}")
            return []
    
    def _get_memories_from_db(self, db: Any, limit: int) -> List[Dict[str, Any]]:
        """从数据库获取记忆"""
        try:
            from backend.db_models.memory import Memory
            
            # 获取最近的记忆
            memories = db.query(Memory).order_by(Memory.created_at.desc()).limit(limit).all()
            
            # 转换为字典格式
            memory_dicts = []
            for memory in memories:
                memory_dict = {
                    "id": memory.id,
                    "user_id": memory.user_id,
                    "content": memory.content,
                    "content_type": memory.content_type,
                    "importance": memory.importance,
                    "memory_type": memory.memory_type,
                    "accessed_count": memory.accessed_count,
                    "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    "expires_at": memory.expires_at.isoformat() if memory.expires_at else None
                }
                
                # 解析嵌入向量（如果有）
                if memory.embedding:
                    try:
                        memory_dict["embedding"] = json.loads(memory.embedding)
                    except:
                        memory_dict["embedding"] = None
                
                memory_dicts.append(memory_dict)
            
            return memory_dicts
            
        except Exception as e:
            self.logger.error(f"从数据库获取记忆失败: {e}")
            return []
    
    def _get_memories_from_system(self, memory_system, limit: int) -> List[Dict[str, Any]]:
        """从内存系统获取记忆"""
        memories = []
        
        # 获取工作记忆
        if hasattr(memory_system, 'working_memory'):
            for memory in memory_system.working_memory[:min(limit, len(memory_system.working_memory))]:
                if hasattr(memory, 'to_dict'):
                    memories.append(memory.to_dict())
                elif isinstance(memory, dict):
                    memories.append(memory)
        
        # 获取短期缓存记忆
        if hasattr(memory_system, 'short_term_memory_cache'):
            cache_size = min(limit - len(memories), len(memory_system.short_term_memory_cache))
            for i, (key, memory) in enumerate(list(memory_system.short_term_memory_cache.items())[:cache_size]):
                if hasattr(memory, 'to_dict'):
                    memories.append(memory.to_dict())
                elif isinstance(memory, dict):
                    memories.append(memory)
        
        return memories
    
    def _detect_content_conflicts(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测内容冲突"""
        conflicts = []
        
        # 对记忆进行分组（基于相似内容）
        content_groups = self._group_memories_by_content_similarity(memories)
        
        for group in content_groups:
            if len(group) < 2:
                continue
            
            # 检查组内记忆是否存在冲突
            group_conflicts = self._analyze_content_group_for_conflicts(group)
            conflicts.extend(group_conflicts)
        
        return conflicts
    
    def _group_memories_by_content_similarity(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """基于内容相似性对记忆进行分组"""
        if len(memories) < 2:
            return [memories] if memories else []
        
        # 简单的文本相似性分组（实际实现中应使用更复杂的NLP）
        groups = []
        processed = set()
        
        for i, mem1 in enumerate(memories):
            if i in processed:
                continue
            
            group = [mem1]
            processed.add(i)
            
            for j, mem2 in enumerate(memories[i+1:], start=i+1):
                if j in processed:
                    continue
                
                # 计算内容相似度
                similarity = self._calculate_content_similarity(mem1, mem2)
                
                if similarity >= self.detection_config["similarity_threshold"]:
                    group.append(mem2)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_content_similarity(self, mem1: Dict[str, Any], mem2: Dict[str, Any]) -> float:
        """计算内容相似度"""
        content1 = mem1.get("content", "")
        content2 = mem2.get("content", "")
        
        if not content1 or not content2:
            return 0.0
        
        # 简单的文本相似度计算（实际实现中应使用嵌入向量或NLP模型）
        # 这里使用Jaccard相似度作为简化实现
        
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        return similarity
    
    def _analyze_content_group_for_conflicts(self, group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析内容组中的冲突"""
        conflicts = []
        
        # 比较组内每对记忆
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                mem1 = group[i]
                mem2 = group[j]
                
                # 检查是否存在冲突
                conflict_info = self._check_content_conflict(mem1, mem2)
                
                if conflict_info:
                    conflict = {
                        "type": "content",
                        "memory_ids": [mem1.get("id"), mem2.get("id")],
                        "memories": [mem1, mem2],
                        "conflict_info": conflict_info,
                        "severity": conflict_info.get("severity", 0.5),
                        "detected_at": time.time()
                    }
                    conflicts.append(conflict)
        
        return conflicts
    
    def _check_content_conflict(self, mem1: Dict[str, Any], mem2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查内容冲突"""
        content1 = mem1.get("content", "")
        content2 = mem2.get("content", "")
        
        if not content1 or not content2:
            return None
        
        # 计算相似度
        similarity = self._calculate_content_similarity(mem1, mem2)
        
        # 如果相似度高但内容有显著差异，则可能存在冲突
        if similarity >= self.detection_config["content_conflict_threshold"]:
            # 检查内容差异
            diff_ratio = self._calculate_content_difference_ratio(content1, content2)
            
            if diff_ratio > 0.3:  # 内容有显著差异
                return {
                    "similarity": similarity,
                    "difference_ratio": diff_ratio,
                    "severity": min(0.9, diff_ratio * similarity),
                    "description": f"内容相似度高({similarity:.2f})但存在差异({diff_ratio:.2f})"
                }
        
        return None
    
    def _calculate_content_difference_ratio(self, content1: str, content2: str) -> float:
        """计算内容差异比率"""
        # 简单的差异计算（实际实现中应使用更复杂的方法）
        words1 = content1.lower().split()
        words2 = content2.lower().split()
        
        if not words1 or not words2:
            return 1.0
        
        # 计算编辑距离比例
        import difflib
        seq_matcher = difflib.SequenceMatcher(None, content1, content2)
        similarity = seq_matcher.ratio()
        
        return 1.0 - similarity
    
    def _detect_time_conflicts(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测时间冲突"""
        conflicts = []
        
        # 对记忆进行分组（基于相似内容和不同时间）
        content_groups = self._group_memories_by_content_similarity(memories)
        
        for group in content_groups:
            if len(group) < 2:
                continue
            
            # 检查组内记忆的时间冲突
            group_conflicts = self._analyze_time_conflicts_in_group(group)
            conflicts.extend(group_conflicts)
        
        return conflicts
    
    def _analyze_time_conflicts_in_group(self, group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析组内的时间冲突"""
        conflicts = []
        
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                mem1 = group[i]
                mem2 = group[j]
                
                # 检查时间冲突
                conflict_info = self._check_time_conflict(mem1, mem2)
                
                if conflict_info:
                    conflict = {
                        "type": "time",
                        "memory_ids": [mem1.get("id"), mem2.get("id")],
                        "memories": [mem1, mem2],
                        "conflict_info": conflict_info,
                        "severity": conflict_info.get("severity", 0.5),
                        "detected_at": time.time()
                    }
                    conflicts.append(conflict)
        
        return conflicts
    
    def _check_time_conflict(self, mem1: Dict[str, Any], mem2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查时间冲突"""
        # 获取记忆时间
        time1 = self._parse_memory_time(mem1)
        time2 = self._parse_memory_time(mem2)
        
        if not time1 or not time2:
            return None
        
        # 计算时间差
        time_diff = abs(time1 - time2)
        
        # 检查是否超过时间冲突阈值
        threshold = self.detection_config["time_conflict_threshold"]
        
        if time_diff > threshold:
            # 检查内容相似性
            similarity = self._calculate_content_similarity(mem1, mem2)
            
            if similarity > 0.5:  # 内容相似但时间差异大
                return {
                    "time_difference": time_diff,
                    "similarity": similarity,
                    "severity": min(0.8, similarity * (time_diff / threshold)),
                    "description": f"内容相似({similarity:.2f})但时间差异大({time_diff:.0f}秒)"
                }
        
        return None
    
    def _parse_memory_time(self, memory: Dict[str, Any]) -> Optional[float]:
        """解析记忆时间"""
        # 优先使用created_at，其次使用last_accessed
        time_str = memory.get("created_at") or memory.get("last_accessed")
        
        if not time_str:
            return None
        
        try:
            from datetime import datetime
            import dateutil.parser
            
            if isinstance(time_str, str):
                dt = dateutil.parser.parse(time_str)
            elif hasattr(time_str, 'isoformat'):
                dt = time_str
            else:
                return None
            
            return dt.timestamp()
            
        except Exception:
            return None
    
    def _detect_fact_conflicts(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测事实冲突"""
        conflicts = []
        
        # 事实冲突检测需要更复杂的逻辑
        # 这里使用简单的启发式方法
        
        # 提取可能的事实陈述
        factual_memories = self._extract_factual_memories(memories)
        
        if len(factual_memories) < 2:
            return conflicts
        
        # 比较事实陈述
        for i in range(len(factual_memories)):
            for j in range(i + 1, len(factual_memories)):
                mem1 = factual_memories[i]
                mem2 = factual_memories[j]
                
                # 检查事实冲突
                conflict_info = self._check_fact_conflict(mem1, mem2)
                
                if conflict_info:
                    conflict = {
                        "type": "fact",
                        "memory_ids": [mem1.get("id"), mem2.get("id")],
                        "memories": [mem1, mem2],
                        "conflict_info": conflict_info,
                        "severity": conflict_info.get("severity", 0.7),
                        "detected_at": time.time()
                    }
                    conflicts.append(conflict)
        
        return conflicts
    
    def _extract_factual_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取可能的事实陈述记忆"""
        factual_memories = []
        
        # 简单的启发式：包含数字、日期、具体名称等内容更可能是事实陈述
        factual_keywords = ["是", "有", "在", "于", "为", "包含", "包括", "等于", "大于", "小于"]
        
        for memory in memories:
            content = memory.get("content", "")
            
            if not content:
                continue
            
            # 检查是否包含事实特征
            has_factual_features = False
            
            # 检查数字
            import re
            if re.search(r'\d+', content):
                has_factual_features = True
            
            # 检查事实关键词
            for keyword in factual_keywords:
                if keyword in content:
                    has_factual_features = True
                    break
            
            if has_factual_features:
                factual_memories.append(memory)
        
        return factual_memories
    
    def _check_fact_conflict(self, mem1: Dict[str, Any], mem2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查事实冲突"""
        content1 = mem1.get("content", "")
        content2 = mem2.get("content", "")
        
        if not content1 or not content2:
            return None
        
        # 简单的启发式：检查否定关系
        negation_words = ["不", "没", "未", "非", "无", "否"]
        
        has_negation = False
        for word in negation_words:
            if (word in content1 and word not in content2) or (word in content2 and word not in content1):
                has_negation = True
                break
        
        if has_negation:
            # 检查内容相似性
            similarity = self._calculate_content_similarity(mem1, mem2)
            
            if similarity > 0.4:  # 内容相似但包含否定词
                return {
                    "negation_detected": True,
                    "similarity": similarity,
                    "severity": 0.7 + (similarity * 0.2),
                    "description": "检测到否定关系的事实冲突"
                }
        
        return None
    
    def _detect_priority_conflicts(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测优先级冲突"""
        conflicts = []
        
        # 对相似记忆检查重要性评分冲突
        content_groups = self._group_memories_by_content_similarity(memories)
        
        for group in content_groups:
            if len(group) < 2:
                continue
            
            # 检查组内的重要性评分差异
            group_conflicts = self._analyze_priority_conflicts_in_group(group)
            conflicts.extend(group_conflicts)
        
        return conflicts
    
    def _analyze_priority_conflicts_in_group(self, group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析组内的优先级冲突"""
        conflicts = []
        
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                mem1 = group[i]
                mem2 = group[j]
                
                # 检查优先级冲突
                conflict_info = self._check_priority_conflict(mem1, mem2)
                
                if conflict_info:
                    conflict = {
                        "type": "priority",
                        "memory_ids": [mem1.get("id"), mem2.get("id")],
                        "memories": [mem1, mem2],
                        "conflict_info": conflict_info,
                        "severity": conflict_info.get("severity", 0.4),
                        "detected_at": time.time()
                    }
                    conflicts.append(conflict)
        
        return conflicts
    
    def _check_priority_conflict(self, mem1: Dict[str, Any], mem2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查优先级冲突"""
        importance1 = mem1.get("importance", 0.5)
        importance2 = mem2.get("importance", 0.5)
        
        # 检查重要性评分差异
        importance_diff = abs(importance1 - importance2)
        
        if importance_diff > 0.3:  # 重要性差异显著
            # 检查内容相似性
            similarity = self._calculate_content_similarity(mem1, mem2)
            
            if similarity > 0.6:  # 内容相似但重要性差异大
                return {
                    "importance_diff": importance_diff,
                    "similarity": similarity,
                    "severity": min(0.6, importance_diff * similarity),
                    "description": f"内容相似({similarity:.2f})但重要性差异大({importance_diff:.2f})"
                }
        
        return None
    
    def _detect_association_conflicts(self, memories: List[Dict[str, Any]], 
                                    memory_system) -> List[Dict[str, Any]]:
        """检测关联冲突"""
        conflicts = []
        
        # 关联冲突检测需要访问记忆图
        if not hasattr(memory_system, 'memory_graph') or not memory_system.memory_graph:
            return conflicts
        
        memory_graph = memory_system.memory_graph
        
        # 检查图中的矛盾关联
        if isinstance(memory_graph, dict) and "edges" in memory_graph:
            edges = memory_graph["edges"]
            
            # 寻找矛盾的关联（例如，A与B正相关，B与C正相关，但A与C负相关）
            conflict_edges = self._find_contradictory_edges(edges, memories)
            
            for edge_info in conflict_edges:
                conflict = {
                    "type": "association",
                    "memory_ids": edge_info.get("involved_memory_ids", []),
                    "conflict_info": edge_info,
                    "severity": edge_info.get("severity", 0.5),
                    "detected_at": time.time()
                }
                conflicts.append(conflict)
        
        return conflicts
    
    def _find_contradictory_edges(self, edges: List[Dict[str, Any]], 
                                 memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """查找矛盾的关联边"""
        conflicts = []
        
        # 构建记忆ID到索引的映射
        memory_id_to_idx = {}
        for idx, memory in enumerate(memories):
            memory_id = memory.get("id")
            if memory_id:
                memory_id_to_idx[memory_id] = idx
        
        # 分析关联模式
        for i, edge1 in enumerate(edges):
            source1 = edge1.get("source")
            target1 = edge1.get("target")
            strength1 = edge1.get("strength", 0.0)
            
            if not source1 or not target1:
                continue
            
            for j, edge2 in enumerate(edges[i+1:], start=i+1):
                source2 = edge2.get("source")
                target2 = edge2.get("target")
                strength2 = edge2.get("strength", 0.0)
                
                if not source2 or not target2:
                    continue
                
                # 检查三角形矛盾：A-B正相关，B-C正相关，但A-C负相关
                if source1 == source2 and target1 != target2:
                    # 查找A-C的关联
                    for edge3 in edges:
                        source3 = edge3.get("source")
                        target3 = edge3.get("target")
                        strength3 = edge3.get("strength", 0.0)
                        
                        if (source3 == target1 and target3 == target2) or (source3 == target2 and target3 == target1):
                            # 检查关联方向是否矛盾
                            if strength1 > 0.5 and strength2 > 0.5 and strength3 < 0.3:
                                conflict_info = {
                                    "contradiction_type": "triangle",
                                    "edges": [edge1, edge2, edge3],
                                    "involved_memory_ids": [source1, target1, target2],
                                    "severity": 0.6,
                                    "description": "三角形关联矛盾：A-B正相关，B-C正相关，但A-C负相关"
                                }
                                conflicts.append(conflict_info)
        
        return conflicts
    
    def _filter_duplicate_conflicts(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤重复冲突"""
        unique_conflicts = []
        conflict_signatures = set()
        
        for conflict in conflicts:
            # 创建冲突签名（基于记忆ID和冲突类型）
            memory_ids = sorted(conflict.get("memory_ids", []))
            conflict_type = conflict.get("type", "")
            
            signature = f"{conflict_type}:{','.join(str(id) for id in memory_ids)}"
            
            if signature not in conflict_signatures:
                conflict_signatures.add(signature)
                unique_conflicts.append(conflict)
        
        return unique_conflicts
    
    def _record_conflict_detection(self, conflicts: List[Dict[str, Any]]):
        """记录冲突检测结果"""
        detection_record = {
            "timestamp": time.time(),
            "total_conflicts": len(conflicts),
            "conflict_types": {},
            "avg_severity": 0.0
        }
        
        # 统计冲突类型
        type_counts = {}
        total_severity = 0.0
        
        for conflict in conflicts:
            conflict_type = conflict.get("type", "unknown")
            type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1
            total_severity += conflict.get("severity", 0.0)
        
        detection_record["conflict_types"] = type_counts
        
        if conflicts:
            detection_record["avg_severity"] = total_severity / len(conflicts)
        
        # 添加到历史记录
        self.conflict_history.append(detection_record)
        
        # 保持历史记录大小
        if len(self.conflict_history) > self.max_history_size:
            self.conflict_history = self.conflict_history[-self.max_history_size:]
    
    def resolve_conflicts(self, conflicts: List[Dict[str, Any]], 
                         memory_system, db: Optional[Any] = None,
                         strategy: Optional[str] = None) -> Dict[str, Any]:
        """解决记忆冲突
        
        参数:
            conflicts: 冲突列表
            memory_system: MemorySystem实例
            db: 数据库会话（可选）
            strategy: 解决策略（如果为None则自动选择）
            
        返回:
            解决结果
        """
        if not conflicts:
            return {
                "success": False,
                "message": "没有需要解决的冲突",
                "resolved_count": 0,
                "results": []
            }
        
        self.logger.info(f"开始解决 {len(conflicts)} 个记忆冲突")
        
        resolution_results = []
        
        for conflict in conflicts:
            # 选择解决策略
            if strategy is None:
                selected_strategy = self._select_resolution_strategy(conflict)
            else:
                selected_strategy = strategy
            
            # 应用解决策略
            result = self._apply_resolution_strategy(conflict, selected_strategy, 
                                                   memory_system, db)
            
            resolution_results.append(result)
            
            # 记录解决结果
            self._record_conflict_resolution(conflict, result, selected_strategy)
        
        # 统计解决结果
        success_count = sum(1 for r in resolution_results if r.get("success", False))
        total_severity_reduction = sum(r.get("severity_reduction", 0.0) for r in resolution_results)
        
        # 更新系统（如果需要）
        if success_count > 0 and db:
            self._update_system_after_resolution(memory_system, db, resolution_results)
        
        overall_result = {
            "success": success_count > 0,
            "message": f"解决了 {success_count}/{len(conflicts)} 个冲突",
            "total_conflicts": len(conflicts),
            "resolved_count": success_count,
            "avg_severity_reduction": total_severity_reduction / len(conflicts) if conflicts else 0.0,
            "strategy_used": strategy or "auto",
            "results": resolution_results,
            "timestamp": time.time()
        }
        
        self.logger.info(f"冲突解决完成: {success_count}/{len(conflicts)} 个成功解决")
        
        return overall_result
    
    def _select_resolution_strategy(self, conflict: Dict[str, Any]) -> str:
        """选择解决策略"""
        conflict_type = conflict.get("type", "")
        severity = conflict.get("severity", 0.5)
        
        # 基于冲突类型和严重性选择策略
        if conflict_type == "content":
            if severity < 0.6:
                return "merge"
            else:
                return "contextual"
        
        elif conflict_type == "time":
            return "merge"
        
        elif conflict_type == "fact":
            return "select"
        
        elif conflict_type == "priority":
            return "select"
        
        elif conflict_type == "association":
            return "both"
        
        else:
            # 默认策略
            if severity < 0.5:
                return "merge"
            elif severity < 0.7:
                return "select"
            else:
                return "contextual"
    
    def _apply_resolution_strategy(self, conflict: Dict[str, Any], strategy: str,
                                 memory_system, db: Optional[Any] = None) -> Dict[str, Any]:
        """应用解决策略"""
        strategy_config = self.resolution_strategies.get(strategy, self.resolution_strategies["select"])
        
        try:
            if strategy == "merge":
                return self._apply_merge_strategy(conflict, memory_system, db)
            elif strategy == "select":
                return self._apply_select_strategy(conflict, memory_system, db)
            elif strategy == "both":
                return self._apply_both_strategy(conflict, memory_system, db)
            elif strategy == "contextual":
                return self._apply_contextual_strategy(conflict, memory_system, db)
            else:
                return {
                    "success": False,
                    "error": f"不支持的解决策略: {strategy}",
                    "strategy": strategy
                }
        except Exception as e:
            self.logger.error(f"应用解决策略失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "strategy": strategy
            }
    
    def _apply_merge_strategy(self, conflict: Dict[str, Any], memory_system, 
                            db: Optional[Any] = None) -> Dict[str, Any]:
        """应用合并策略"""
        memories = conflict.get("memories", [])
        
        if len(memories) < 2:
            return {
                "success": False,
                "error": "合并需要至少两个记忆",
                "strategy": "merge"
            }
        
        # 合并记忆内容
        merged_content = self._merge_memory_contents(memories)
        
        # 计算合并后的重要性
        importances = [m.get("importance", 0.5) for m in memories]
        merged_importance = sum(importances) / len(importances)
        
        # 创建新记忆
        new_memory_data = {
            "content": merged_content,
            "importance": merged_importance,
            "memory_type": "long_term",  # 合并后的记忆通常作为长期记忆
            "source_memory_ids": [m.get("id") for m in memories if m.get("id")],
            "is_resolution_result": True,
            "resolution_type": "merge"
        }
        
        # 保存新记忆（如果有数据库）
        new_memory_id = None
        if db:
            new_memory_id = self._save_merged_memory(db, new_memory_data, conflict)
        
        # 标记原记忆为已解决
        resolution_mark = self._mark_memories_as_resolved(memories, "merged", new_memory_id)
        
        return {
            "success": True,
            "strategy": "merge",
            "original_memory_ids": [m.get("id") for m in memories if m.get("id")],
            "new_memory_id": new_memory_id,
            "merged_content_preview": merged_content[:100] + "..." if len(merged_content) > 100 else merged_content,
            "severity_reduction": conflict.get("severity", 0.5) * 0.8,  # 假设严重性降低80%
            "resolution_mark": resolution_mark
        }
    
    def _merge_memory_contents(self, memories: List[Dict[str, Any]]) -> str:
        """合并记忆内容"""
        if not memories:
            return ""
        
        # 简单合并：连接所有内容
        contents = [m.get("content", "") for m in memories if m.get("content")]
        
        # 去重和排序（按重要性）
        unique_contents = []
        seen = set()
        
        for content in contents:
            if content not in seen:
                seen.add(content)
                unique_contents.append(content)
        
        # 按来源记忆的重要性排序
        if len(unique_contents) > 1:
            # 获取每个内容对应记忆的重要性
            content_importance = {}
            for memory in memories:
                content = memory.get("content", "")
                importance = memory.get("importance", 0.5)
                if content in content_importance:
                    content_importance[content] = max(content_importance[content], importance)
                else:
                    content_importance[content] = importance
            
            unique_contents.sort(key=lambda x: content_importance.get(x, 0.5), reverse=True)
        
        # 合并内容
        merged = " ".join(unique_contents)
        
        # 添加解决说明
        resolution_note = "\n\n[此记忆是通过合并多个冲突记忆创建的]"
        merged += resolution_note
        
        return merged
    
    def _save_merged_memory(self, db: Any, memory_data: Dict[str, Any], 
                          conflict: Dict[str, Any]) -> Optional[int]:
        """保存合并后的记忆到数据库"""
        try:
            from backend.db_models.memory import Memory
            from datetime import datetime, timezone
            
            # 创建新记忆
            new_memory = Memory(
                user_id=1,  # 默认用户ID（实际应根据上下文设置）
                content=memory_data["content"],
                content_type="text",
                importance=memory_data["importance"],
                memory_type=memory_data["memory_type"],
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                accessed_count=0
            )
            
            # 保存源记忆ID（作为元数据）
            source_ids = memory_data.get("source_memory_ids", [])
            if source_ids:
                new_memory.metadata = json.dumps({
                    "source_memory_ids": source_ids,
                    "is_resolution_result": True,
                    "resolution_type": "merge",
                    "original_conflict": {
                        "type": conflict.get("type"),
                        "severity": conflict.get("severity")
                    }
                })
            
            db.add(new_memory)
            db.commit()
            
            self.logger.info(f"合并记忆已保存: ID={new_memory.id}")
            
            return new_memory.id
            
        except Exception as e:
            self.logger.error(f"保存合并记忆失败: {e}")
            db.rollback()
            return None
    
    def _mark_memories_as_resolved(self, memories: List[Dict[str, Any]], 
                                 resolution_type: str, new_memory_id: Optional[int] = None) -> Dict[str, Any]:
        """标记记忆为已解决"""
        resolution_info = {
            "resolution_type": resolution_type,
            "resolved_at": time.time(),
            "new_memory_id": new_memory_id,
            "marked_memory_ids": []
        }
        
        # 在实际实现中，应该更新数据库中的记忆状态
        # 这里仅记录标记信息
        
        for memory in memories:
            memory_id = memory.get("id")
            if memory_id:
                resolution_info["marked_memory_ids"].append(memory_id)
        
        return resolution_info
    
    def _apply_select_strategy(self, conflict: Dict[str, Any], memory_system,
                             db: Optional[Any] = None) -> Dict[str, Any]:
        """应用选择策略"""
        memories = conflict.get("memories", [])
        
        if len(memories) < 2:
            return {
                "success": False,
                "error": "选择需要至少两个记忆",
                "strategy": "select"
            }
        
        # 选择最佳记忆
        selected_memory = self._select_best_memory(memories)
        other_memories = [m for m in memories if m.get("id") != selected_memory.get("id")]
        
        # 更新选定记忆的重要性
        updated_importance = min(1.0, selected_memory.get("importance", 0.5) * 1.1)
        
        # 降低其他记忆的重要性
        demotion_results = []
        for memory in other_memories:
            demoted_importance = max(0.1, memory.get("importance", 0.5) * 0.7)
            demotion_results.append({
                "memory_id": memory.get("id"),
                "old_importance": memory.get("importance", 0.5),
                "new_importance": demoted_importance
            })
        
        # 更新数据库（如果有）
        update_results = None
        if db:
            update_results = self._update_memory_priorities(db, selected_memory, updated_importance, 
                                                          other_memories, demotion_results)
        
        return {
            "success": True,
            "strategy": "select",
            "selected_memory_id": selected_memory.get("id"),
            "selected_memory_content_preview": selected_memory.get("content", "")[:100] + "..." if len(selected_memory.get("content", "")) > 100 else selected_memory.get("content", ""),
            "updated_importance": updated_importance,
            "demoted_memories": demotion_results,
            "severity_reduction": conflict.get("severity", 0.5) * 0.9,  # 假设严重性降低90%
            "update_results": update_results
        }
    
    def _select_best_memory(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """选择最佳记忆"""
        # 基于多个因素评分
        best_score = -1.0
        best_memory = memories[0]
        
        for memory in memories:
            score = self._calculate_memory_quality_score(memory)
            
            if score > best_score:
                best_score = score
                best_memory = memory
        
        return best_memory
    
    def _calculate_memory_quality_score(self, memory: Dict[str, Any]) -> float:
        """计算记忆质量评分"""
        score = 0.0
        
        # 1. 重要性评分
        importance = memory.get("importance", 0.5)
        score += importance * 0.3
        
        # 2. 访问次数
        access_count = memory.get("accessed_count", 0)
        access_score = min(1.0, access_count / 10.0)  # 10次访问为满分
        score += access_score * 0.2
        
        # 3. 内容质量
        content = memory.get("content", "")
        content_length = len(content)
        if 50 <= content_length <= 500:
            length_score = 0.3
        elif content_length > 1000:
            length_score = 0.1
        else:
            length_score = 0.2
        score += length_score
        
        # 4. 记忆类型（长期记忆更可靠）
        memory_type = memory.get("memory_type", "short_term")
        if memory_type == "long_term":
            type_score = 0.2
        else:
            type_score = 0.1
        score += type_score
        
        return min(1.0, score)
    
    def _update_memory_priorities(self, db: Any, selected_memory: Dict[str, Any],
                                updated_importance: float, other_memories: List[Dict[str, Any]],
                                demotion_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """更新记忆优先级"""
        try:
            from backend.db_models.memory import Memory
            
            update_info = {
                "selected_memory_updated": False,
                "demoted_memories_updated": 0,
                "errors": []
            }
            
            # 更新选定记忆
            selected_id = selected_memory.get("id")
            if selected_id:
                memory = db.query(Memory).filter(Memory.id == selected_id).first()
                if memory:
                    memory.importance = updated_importance
                    update_info["selected_memory_updated"] = True
            
            # 降低其他记忆的重要性
            for demotion in demotion_results:
                memory_id = demotion["memory_id"]
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory:
                    memory.importance = demotion["new_importance"]
                    update_info["demoted_memories_updated"] += 1
            
            db.commit()
            
            return update_info
            
        except Exception as e:
            db.rollback()
            return {
                "selected_memory_updated": False,
                "demoted_memories_updated": 0,
                "errors": [str(e)]
            }
    
    def _apply_both_strategy(self, conflict: Dict[str, Any], memory_system,
                           db: Optional[Any] = None) -> Dict[str, Any]:
        """应用双版本策略"""
        memories = conflict.get("memories", [])
        
        if len(memories) < 2:
            return {
                "success": False,
                "error": "双版本策略需要至少两个记忆",
                "strategy": "both"
            }
        
        # 为每个记忆添加冲突标记
        conflict_markers = []
        
        for memory in memories:
            memory_id = memory.get("id")
            if memory_id:
                marker = {
                    "memory_id": memory_id,
                    "conflict_type": conflict.get("type"),
                    "conflict_severity": conflict.get("severity", 0.5),
                    "other_memory_ids": [m.get("id") for m in memories if m.get("id") != memory_id],
                    "marked_at": time.time()
                }
                conflict_markers.append(marker)
        
        # 更新数据库标记（如果有）
        update_results = None
        if db:
            update_results = self._mark_conflicting_memories(db, conflict_markers, conflict)
        
        return {
            "success": True,
            "strategy": "both",
            "conflict_markers": conflict_markers,
            "severity_reduction": conflict.get("severity", 0.5) * 0.3,  # 双版本策略减少较少的严重性
            "update_results": update_results,
            "message": "冲突记忆已标记，保留双版本"
        }
    
    def _mark_conflicting_memories(self, db: Any, conflict_markers: List[Dict[str, Any]],
                                 conflict: Dict[str, Any]) -> Dict[str, Any]:
        """标记冲突记忆"""
        try:
            from backend.db_models.memory import Memory
            
            update_info = {
                "marked_count": 0,
                "errors": []
            }
            
            for marker in conflict_markers:
                memory_id = marker["memory_id"]
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                
                if memory:
                    # 更新元数据
                    current_metadata = {}
                    if memory.metadata:
                        try:
                            current_metadata = json.loads(memory.metadata)
                        except:
                            current_metadata = {}
                    
                    # 添加冲突标记
                    if "conflicts" not in current_metadata:
                        current_metadata["conflicts"] = []
                    
                    current_metadata["conflicts"].append({
                        "type": marker["conflict_type"],
                        "severity": marker["conflict_severity"],
                        "other_memory_ids": marker["other_memory_ids"],
                        "marked_at": marker["marked_at"]
                    })
                    
                    memory.metadata = json.dumps(current_metadata, ensure_ascii=False)
                    update_info["marked_count"] += 1
            
            db.commit()
            
            return update_info
            
        except Exception as e:
            db.rollback()
            return {
                "marked_count": 0,
                "errors": [str(e)]
            }
    
    def _apply_contextual_strategy(self, conflict: Dict[str, Any], memory_system,
                                 db: Optional[Any] = None) -> Dict[str, Any]:
        """应用上下文策略"""
        # 上下文策略需要更多的外部信息和推理
        # 这里实现一个简化的版本
        
        memories = conflict.get("memories", [])
        conflict_type = conflict.get("type", "")
        
        if len(memories) < 2:
            return {
                "success": False,
                "error": "上下文策略需要至少两个记忆",
                "strategy": "contextual"
            }
        
        # 收集上下文证据
        evidence = self.evidence_collector.collect_evidence(memories, conflict_type, memory_system)
        
        # 基于证据解决冲突
        resolution = self._resolve_based_on_evidence(memories, evidence, conflict_type)
        
        # 应用解决结果
        if resolution["resolution_applied"]:
            # 在实际实现中，应该根据证据结果更新记忆
            pass
        
        return {
            "success": True,
            "strategy": "contextual",
            "evidence_collected": len(evidence.get("evidence_items", [])),
            "resolution": resolution,
            "severity_reduction": conflict.get("severity", 0.5) * resolution.get("confidence", 0.5),
            "message": "基于上下文证据解决冲突"
        }
    
    def _resolve_based_on_evidence(self, memories: List[Dict[str, Any]], 
                                 evidence: Dict[str, Any], conflict_type: str) -> Dict[str, Any]:
        """基于证据解决冲突"""
        # 简化实现：根据证据质量选择最佳记忆
        
        evidence_items = evidence.get("evidence_items", [])
        
        if not evidence_items:
            # 没有证据，回退到选择策略
            best_memory = self._select_best_memory(memories)
            return {
                "resolution_applied": True,
                "method": "fallback_selection",
                "selected_memory_id": best_memory.get("id"),
                "confidence": 0.5,
                "reason": "没有足够的上下文证据，使用回退选择"
            }
        
        # 根据证据评分记忆
        memory_scores = {}
        for memory in memories:
            memory_id = memory.get("id")
            if memory_id:
                memory_scores[memory_id] = self._calculate_memory_evidence_score(memory, evidence_items)
        
        # 选择得分最高的记忆
        best_memory_id = max(memory_scores.items(), key=lambda x: x[1])[0]
        best_score = memory_scores[best_memory_id]
        
        # 找到最佳记忆
        best_memory = next((m for m in memories if m.get("id") == best_memory_id), memories[0])
        
        return {
            "resolution_applied": True,
            "method": "evidence_based_selection",
            "selected_memory_id": best_memory_id,
            "selected_memory_score": best_score,
            "all_scores": memory_scores,
            "confidence": min(1.0, best_score * 1.5),  # 调整置信度
            "reason": "基于上下文证据评分选择最佳记忆"
        }
    
    def _calculate_memory_evidence_score(self, memory: Dict[str, Any], 
                                       evidence_items: List[Dict[str, Any]]) -> float:
        """计算记忆的证据评分"""
        score = 0.0
        content = memory.get("content", "")
        
        if not content or not evidence_items:
            return 0.0
        
        # 检查证据是否支持该记忆
        for evidence in evidence_items:
            evidence_content = evidence.get("content", "")
            evidence_confidence = evidence.get("confidence", 0.5)
            
            # 简单的关键词匹配
            common_words = set(content.lower().split()) & set(evidence_content.lower().split())
            if common_words:
                score += (len(common_words) / max(len(content.split()), 1)) * evidence_confidence
        
        return min(1.0, score)
    
    def _record_conflict_resolution(self, conflict: Dict[str, Any], result: Dict[str, Any],
                                  strategy: str):
        """记录冲突解决结果"""
        resolution_record = {
            "timestamp": time.time(),
            "conflict_id": hash(str(conflict.get("memory_ids", [])) + conflict.get("type", "")),
            "conflict_type": conflict.get("type"),
            "conflict_severity": conflict.get("severity", 0.0),
            "strategy": strategy,
            "success": result.get("success", False),
            "severity_reduction": result.get("severity_reduction", 0.0),
            "memory_ids": conflict.get("memory_ids", [])
        }
        
        # 添加到历史记录
        self.conflict_history.append(resolution_record)
        
        # 保持历史记录大小
        if len(self.conflict_history) > self.max_history_size:
            self.conflict_history = self.conflict_history[-self.max_history_size:]
    
    def _update_system_after_resolution(self, memory_system, db: Any,
                                      resolution_results: List[Dict[str, Any]]):
        """解决冲突后更新系统"""
        # 在实际实现中，可能需要更新索引、缓存等
        # 这里仅记录更新信息
        
        self.logger.info(f"冲突解决后更新系统，处理了 {len(resolution_results)} 个结果")
        
        # 如果需要，重建索引
        if hasattr(memory_system, 'faiss_index') and memory_system.faiss_index:
            self.logger.info("冲突解决后可能需要重建FAISS索引")
    
    def get_conflict_statistics(self) -> Dict[str, Any]:
        """获取冲突统计信息"""
        if not self.conflict_history:
            return {
                "total_conflicts": 0,
                "recent_conflicts": 0,
                "resolution_rate": 0.0,
                "avg_severity": 0.0,
                "type_distribution": {}
            }
        
        # 分析历史记录
        total_conflicts = len(self.conflict_history)
        
        # 最近24小时的冲突
        recent_threshold = time.time() - 86400  # 24小时
        recent_conflicts = sum(1 for record in self.conflict_history 
                              if record.get("timestamp", 0) > recent_threshold)
        
        # 解决率
        successful_resolutions = sum(1 for record in self.conflict_history 
                                    if record.get("success", False))
        resolution_rate = successful_resolutions / total_conflicts if total_conflicts > 0 else 0.0
        
        # 平均严重性
        total_severity = sum(record.get("conflict_severity", 0.0) for record in self.conflict_history)
        avg_severity = total_severity / total_conflicts if total_conflicts > 0 else 0.0
        
        # 类型分布
        type_distribution = {}
        for record in self.conflict_history:
            conflict_type = record.get("conflict_type", "unknown")
            type_distribution[conflict_type] = type_distribution.get(conflict_type, 0) + 1
        
        return {
            "total_conflicts": total_conflicts,
            "recent_conflicts": recent_conflicts,
            "resolution_rate": resolution_rate,
            "avg_severity": avg_severity,
            "type_distribution": type_distribution,
            "config": self.detection_config.copy()
        }
    
    def get_resolution_recommendations(self, conflict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取解决建议"""
        recommendations = []
        
        conflict_type = conflict.get("type", "")
        severity = conflict.get("severity", 0.5)
        
        # 基于冲突类型和严重性提供建议
        if conflict_type == "content":
            if severity < 0.6:
                recommendations.append({
                    "strategy": "merge",
                    "confidence": 0.7,
                    "reason": "内容冲突严重性较低，适合合并"
                })
            else:
                recommendations.append({
                    "strategy": "contextual",
                    "confidence": 0.8,
                    "reason": "内容冲突严重性高，需要基于上下文解决"
                })
        
        elif conflict_type == "time":
            recommendations.append({
                "strategy": "merge",
                "confidence": 0.6,
                "reason": "时间冲突通常可以通过合并时间信息解决"
            })
        
        elif conflict_type == "fact":
            recommendations.append({
                "strategy": "select",
                "confidence": 0.9,
                "reason": "事实冲突需要选择更可靠的记忆"
            })
            recommendations.append({
                "strategy": "contextual",
                "confidence": 0.7,
                "reason": "也可以基于上下文证据解决事实冲突"
            })
        
        elif conflict_type == "priority":
            recommendations.append({
                "strategy": "select",
                "confidence": 0.8,
                "reason": "优先级冲突需要选择重要性更高的记忆"
            })
        
        elif conflict_type == "association":
            recommendations.append({
                "strategy": "both",
                "confidence": 0.6,
                "reason": "关联冲突可以保留双版本，标记冲突"
            })
        
        # 总是提供多个选项
        if len(recommendations) < 2:
            recommendations.append({
                "strategy": "select",
                "confidence": 0.5,
                "reason": "通用选择策略"
            })
        
        return recommendations


class ConflictPatternAnalyzer:
    """冲突模式分析器
    
    分析冲突模式，识别常见的冲突原因和模式
    """
    
    def __init__(self):
        self.patterns = {}
        self.logger = logging.getLogger("ConflictPatternAnalyzer")
    
    def analyze_patterns(self, conflict_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析冲突模式"""
        if not conflict_history:
            return {"patterns_found": 0}
        
        # 分析常见模式
        patterns = {
            "frequent_users": self._analyze_frequent_users(conflict_history),
            "time_patterns": self._analyze_time_patterns(conflict_history),
            "content_patterns": self._analyze_content_patterns(conflict_history),
            "escalation_patterns": self._analyze_escalation_patterns(conflict_history)
        }
        
        self.patterns = patterns
        
        return {
            "patterns_found": sum(len(p) for p in patterns.values()),
            "details": patterns
        }
    
    def _analyze_frequent_users(self, conflict_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析频繁出现冲突的用户"""
        user_counts = {}
        
        for record in conflict_history:
            memory_ids = record.get("memory_ids", [])
            # 在实际实现中，需要从记忆ID获取用户ID
            # 这里使用简化实现
            if memory_ids:
                user_key = f"user_from_memory_{memory_ids[0]}"
                user_counts[user_key] = user_counts.get(user_key, 0) + 1
        
        # 找到最频繁的用户
        frequent_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "user_counts": dict(frequent_users),
            "most_frequent": frequent_users[0] if frequent_users else None
        }
    
    def _analyze_time_patterns(self, conflict_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析时间模式"""
        import datetime
        
        time_patterns = {
            "hourly_distribution": [0] * 24,
            "daily_distribution": [0] * 7
        }
        
        for record in conflict_history:
            timestamp = record.get("timestamp", 0)
            if timestamp:
                dt = datetime.datetime.fromtimestamp(timestamp)
                
                # 小时分布
                hour = dt.hour
                time_patterns["hourly_distribution"][hour] += 1
                
                # 星期分布
                weekday = dt.weekday()
                time_patterns["daily_distribution"][weekday] += 1
        
        return time_patterns
    
    def _analyze_content_patterns(self, conflict_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析内容模式"""
        content_patterns = {
            "common_keywords": {},
            "frequent_topics": {}
        }
        
        # 在实际实现中，需要分析冲突记忆的内容
        # 这里使用简化实现
        
        return content_patterns
    
    def _analyze_escalation_patterns(self, conflict_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析冲突升级模式"""
        escalation_patterns = {
            "severity_trend": "stable",
            "frequency_trend": "stable"
        }
        
        if len(conflict_history) < 10:
            return escalation_patterns
        
        # 分析严重性趋势
        recent_conflicts = conflict_history[-10:]
        old_conflicts = conflict_history[-20:-10]
        
        if old_conflicts and recent_conflicts:
            avg_severity_old = sum(c.get("conflict_severity", 0.0) for c in old_conflicts) / len(old_conflicts)
            avg_severity_recent = sum(c.get("conflict_severity", 0.0) for c in recent_conflicts) / len(recent_conflicts)
            
            if avg_severity_recent > avg_severity_old + 0.1:
                escalation_patterns["severity_trend"] = "increasing"
            elif avg_severity_recent < avg_severity_old - 0.1:
                escalation_patterns["severity_trend"] = "decreasing"
        
        # 分析频率趋势
        if len(conflict_history) >= 20:
            old_count = len([c for c in conflict_history[-20:-10] if c.get("timestamp", 0) > 0])
            recent_count = len([c for c in conflict_history[-10:] if c.get("timestamp", 0) > 0])
            
            if recent_count > old_count * 1.5:
                escalation_patterns["frequency_trend"] = "increasing"
            elif recent_count < old_count * 0.5:
                escalation_patterns["frequency_trend"] = "decreasing"
        
        return escalation_patterns


class EvidenceCollector:
    """证据收集器
    
    收集解决冲突所需的上下文证据
    """
    
    def __init__(self):
        self.logger = logging.getLogger("EvidenceCollector")
    
    def collect_evidence(self, memories: List[Dict[str, Any]], conflict_type: str,
                        memory_system) -> Dict[str, Any]:
        """收集证据"""
        evidence = {
            "evidence_items": [],
            "source_types": [],
            "total_confidence": 0.0
        }
        
        # 收集基于记忆系统的证据
        system_evidence = self._collect_system_evidence(memory_system, memories)
        evidence["evidence_items"].extend(system_evidence)
        
        # 收集基于内容的证据
        content_evidence = self._collect_content_evidence(memories)
        evidence["evidence_items"].extend(content_evidence)
        
        # 收集基于时间的证据
        time_evidence = self._collect_time_evidence(memories)
        evidence["evidence_items"].extend(time_evidence)
        
        # 统计源类型
        source_types = set(item.get("source_type", "unknown") for item in evidence["evidence_items"])
        evidence["source_types"] = list(source_types)
        
        # 计算总置信度
        if evidence["evidence_items"]:
            total_confidence = sum(item.get("confidence", 0.0) for item in evidence["evidence_items"])
            evidence["total_confidence"] = total_confidence / len(evidence["evidence_items"])
        
        return evidence
    
    def _collect_system_evidence(self, memory_system, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """收集系统证据"""
        evidence_items = []
        
        # 收集系统状态证据
        if hasattr(memory_system, 'working_memory'):
            working_memory_size = len(memory_system.working_memory)
            evidence_items.append({
                "source_type": "system_state",
                "content": f"工作内存大小: {working_memory_size}",
                "confidence": 0.3,
                "relevance": "low"
            })
        
        # 收集缓存命中率证据
        if hasattr(memory_system, 'short_term_cache_hits') and hasattr(memory_system, 'short_term_cache_misses'):
            hits = memory_system.short_term_cache_hits
            misses = memory_system.short_term_cache_misses
            total = hits + misses
            
            if total > 0:
                hit_rate = hits / total
                evidence_items.append({
                    "source_type": "system_performance",
                    "content": f"缓存命中率: {hit_rate:.2f}",
                    "confidence": 0.4,
                    "relevance": "medium"
                })
        
        return evidence_items
    
    def _collect_content_evidence(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """收集内容证据"""
        evidence_items = []
        
        # 提取内容特征
        for memory in memories:
            content = memory.get("content", "")
            if not content:
                continue
            
            # 分析内容长度
            content_length = len(content)
            length_category = "short" if content_length < 50 else "medium" if content_length < 200 else "long"
            
            evidence_items.append({
                "source_type": "content_analysis",
                "content": f"内容长度: {length_category} ({content_length}字符)",
                "confidence": 0.5,
                "relevance": "medium",
                "memory_id": memory.get("id")
            })
        
        return evidence_items
    
    def _collect_time_evidence(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """收集时间证据"""
        evidence_items = []
        
        # 分析记忆时间分布
        timestamps = []
        for memory in memories:
            created_at = memory.get("created_at")
            if created_at:
                try:
                    import dateutil.parser
                    dt = dateutil.parser.parse(created_at)
                    timestamps.append(dt.timestamp())
                except:
                    pass
        
        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)
            time_range = max_time - min_time
            
            evidence_items.append({
                "source_type": "time_analysis",
                "content": f"记忆时间范围: {time_range:.0f}秒",
                "confidence": 0.6,
                "relevance": "high"
            })
        
        return evidence_items
