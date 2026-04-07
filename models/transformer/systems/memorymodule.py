# MemoryModule - 从self_agi_model.py拆分
"""Memory模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

class MemoryModule(nn.Module):
    """记忆管理模块 - 实现长期和短期记忆功能

    功能：
    - 长期记忆存储：知识库、经验库的持久化存储
    - 短期记忆缓存：工作记忆、上下文记忆的临时存储
    - 记忆检索和关联：基于内容的记忆检索和关联机制
    - 记忆压缩和整理：自动清理和组织记忆

    基于神经记忆网络实现，支持动态记忆管理
    """

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 记忆编码器 - 将输入编码为记忆表示
        self.memory_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 记忆键网络 - 生成记忆键用于检索
        self.key_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 记忆值网络 - 生成记忆值用于存储
        self.value_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 记忆查询网络 - 生成查询向量用于检索
        self.query_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 记忆门控 - 控制记忆读写
        self.memory_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid(),
        )

        # 记忆矩阵 - 可学习的记忆存储
        self.memory_slots = 100  # 记忆槽数量
        self.memory_matrix = nn.Parameter(
            torch.randn(self.memory_slots, config.hidden_size) * 0.01
        )

        # 记忆重要性网络 - 学习记忆的重要性
        self.importance_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # 关联网络 - 建立记忆之间的关联
        self.association_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2, eps=config.layer_norm_eps),
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 遗忘机制网络 - 基于记忆重要性和时间衰减的遗忘机制
        self.forget_gate_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid(),
        )

        # 时间衰减网络 - 学习时间对记忆的影响
        self.temporal_decay_network = nn.Sequential(
            nn.Linear(1, config.hidden_size // 2),  # 输入：时间间隔
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # 记忆槽使用情况跟踪
        self.register_buffer('memory_access_count', torch.zeros(self.memory_slots))
        self.register_buffer('memory_last_accessed', torch.zeros(self.memory_slots))
        self.register_buffer('memory_creation_time', torch.zeros(self.memory_slots))
        
        # 遗忘阈值 - 低于此阈值的记忆会被遗忘
        self.forget_threshold = 0.1

    def apply_forgetting(self, current_step: int = 0) -> Dict[str, Any]:
        """应用遗忘机制
        
        参数:
            current_step: 当前训练步骤（用于时间衰减计算）
            
        返回:
            遗忘统计信息
        """
        # 计算每个记忆槽的遗忘概率
        # 基于：1) 记忆重要性 2) 访问频率 3) 时间衰减
        
        with torch.no_grad():
            # 计算记忆重要性（使用重要性网络）
            memory_importance = self.importance_network(self.memory_matrix)
            
            # 计算时间衰减（距离上次访问的时间）
            time_since_last_access = current_step - self.memory_last_accessed
            # 归一化时间间隔（假设最大时间间隔为10000步）
            normalized_time = time_since_last_access.float() / 10000.0
            normalized_time = normalized_time.clamp(0, 1).unsqueeze(1)  # [memory_slots, 1]
            
            # 计算时间衰减因子
            temporal_decay = self.temporal_decay_network(normalized_time)  # [memory_slots, 1]
            
            # 计算访问频率衰减（不常访问的记忆更容易被遗忘）
            access_count_normalized = self.memory_access_count / (self.memory_access_count.max() + 1e-8)
            access_frequency_factor = 1.0 - access_count_normalized.unsqueeze(1)  # 访问越少，越容易被遗忘
            
            # 计算综合遗忘概率
            # 遗忘概率 = (1 - 重要性) * 时间衰减 * 访问频率因子
            forget_prob = (1.0 - memory_importance) * temporal_decay * access_frequency_factor
            
            # 确定哪些记忆槽需要被遗忘（遗忘概率超过阈值）
            forget_mask = forget_prob > self.forget_threshold
            
            # 统计信息
            num_to_forget = forget_mask.sum().item()
            total_slots = self.memory_slots
            
            # 如果需要遗忘的记忆超过一定比例，重置记忆槽
            if num_to_forget > total_slots * 0.3:  # 超过30%需要遗忘
                # 重置最不重要的记忆槽
                sorted_indices = torch.argsort(memory_importance.squeeze(), descending=False)
                num_to_reset = int(total_slots * 0.2)  # 重置20%最不重要的记忆
                reset_indices = sorted_indices[:num_to_reset]
                
                # 重置记忆槽为随机值
                self.memory_matrix.data[reset_indices] = torch.randn(
                    num_to_reset, self.config.hidden_size, device=self.memory_matrix.device
                ) * 0.01
                
                # 重置统计信息
                self.memory_access_count[reset_indices] = 0
                self.memory_last_accessed[reset_indices] = current_step
                self.memory_creation_time[reset_indices] = current_step
                
                return {
                    "forget_applied": True,
                    "num_reset": num_to_reset,
                    "forget_prob_mean": forget_prob.mean().item(),
                    "forget_prob_std": forget_prob.std().item(),
                    "memory_importance_mean": memory_importance.mean().item(),
                    "memory_importance_std": memory_importance.std().item(),
                    "forget_type": "批量重置"
                }
            elif num_to_forget > 0:
                # 部分遗忘：减弱被遗忘记忆槽的强度
                forget_strength = 0.5  # 遗忘强度（0-1）
                self.memory_matrix.data[forget_mask.squeeze()] *= (1.0 - forget_strength)
                
                return {
                    "forget_applied": True,
                    "num_forgotten": num_to_forget,
                    "forget_prob_mean": forget_prob.mean().item(),
                    "forget_prob_std": forget_prob.std().item(),
                    "memory_importance_mean": memory_importance.mean().item(),
                    "memory_importance_std": memory_importance.std().item(),
                    "forget_type": "部分衰减"
                }
            else:
                return {
                    "forget_applied": False,
                    "num_forgotten": 0,
                    "forget_prob_mean": forget_prob.mean().item(),
                    "forget_prob_std": forget_prob.std().item(),
                    "memory_importance_mean": memory_importance.mean().item(),
                    "memory_importance_std": memory_importance.std().item(),
                    "forget_type": "无需遗忘"
                }

    def update_memory_statistics(self, top_indices: torch.Tensor, current_step: int):
        """更新记忆统计信息
        
        参数:
            top_indices: 最近访问的记忆槽索引 [batch_size]
            current_step: 当前训练步骤
        """
        # 更新访问计数
        for idx in top_indices.cpu().numpy():
            self.memory_access_count[idx] += 1
            self.memory_last_accessed[idx] = current_step
            
            # 如果是新创建的记忆槽，记录创建时间
            if self.memory_creation_time[idx] == 0:
                self.memory_creation_time[idx] = current_step

    def forward(
        self, hidden_states: torch.Tensor, query: Optional[torch.Tensor] = None,
        current_step: Optional[int] = None, apply_forgetting: bool = False
    ) -> Dict[str, Any]:
        """前向传播 - 包含遗忘机制的完整记忆系统

        参数:
            hidden_states: 输入特征 [batch_size, seq_len, hidden_size]
            query: 查询向量 [batch_size, query_dim] (可选)
            current_step: 当前训练步骤（用于时间衰减和遗忘机制）
            apply_forgetting: 是否应用遗忘机制

        返回:
            记忆输出字典（包含遗忘统计信息）
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 默认当前步骤
        if current_step is None:
            current_step = 0

        # 编码记忆
        encoded_memory = self.memory_encoder(hidden_states)

        # 生成记忆键和值
        memory_keys = self.key_network(
            encoded_memory.mean(dim=1)
        )  # [batch_size, key_dim]
        memory_values = self.value_network(
            encoded_memory
        )  # [batch_size, seq_len, hidden_size]

        # 生成查询向量（如果未提供，使用记忆键）
        if query is not None:
            memory_queries = self.query_network(query)
        else:
            memory_queries = memory_keys

        # 记忆检索：计算查询与记忆矩阵的相似度
        # 扩展记忆查询以匹配记忆矩阵
        queries_expanded = memory_queries.unsqueeze(1)  # [batch_size, 1, key_dim]

        # 计算相似度（完整：使用点积）
        similarities = torch.matmul(
            queries_expanded, self.memory_matrix.transpose(0, 1)
        )
        similarities = similarities.squeeze(1)  # [batch_size, memory_slots]

        # 应用softmax获取注意力权重
        attention_weights = F.softmax(similarities, dim=-1)

        # 检索记忆：加权求和记忆矩阵
        retrieved_memory = torch.matmul(
            attention_weights, self.memory_matrix
        )  # [batch_size, hidden_size]

        # 记忆门控：控制记忆写入
        gate_input = torch.cat([encoded_memory.mean(dim=1), retrieved_memory], dim=-1)
        write_gate = self.memory_gate(gate_input)

        # 更新记忆矩阵（完整：只更新最相关的记忆槽）
        # 找到每个批次最相关的记忆槽
        top_indices = torch.argmax(similarities, dim=-1)  # [batch_size]

        # 计算新记忆值
        new_memory_values = memory_values.mean(dim=1)  # [batch_size, hidden_size]

        # 更新记忆矩阵（在训练中，这应该通过梯度下降学习）
        # 这里我们只是计算更新，实际更新在训练过程中通过优化器完成
        memory_updates = write_gate.unsqueeze(1) * new_memory_values.unsqueeze(1)

        # 计算记忆重要性
        memory_importance = self.importance_network(retrieved_memory)

        # 关联记忆（如果存在多个记忆片段）
        if batch_size > 1:
            # 计算记忆之间的关联
            associations = []
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    pair = torch.cat([retrieved_memory[i], retrieved_memory[j]], dim=-1)
                    association = self.association_network(pair)
                    associations.append(association)

            if associations:
                association_features = torch.stack(associations, dim=0)
            else:
                association_features = torch.zeros(
                    1, hidden_dim // 2, device=hidden_states.device
                )
        else:
            association_features = torch.zeros(
                1, hidden_dim // 2, device=hidden_states.device
            )

        # === 遗忘机制处理 ===
        forget_stats = None
        if apply_forgetting and current_step > 0:
            # 应用遗忘机制
            forget_stats = self.apply_forgetting(current_step)
        
        # 更新记忆统计信息
        if current_step > 0:
            self.update_memory_statistics(top_indices, current_step)

        # 构建返回结果
        result = {
            "encoded_memory": encoded_memory,
            "memory_keys": memory_keys,
            "memory_values": memory_values,
            "retrieved_memory": retrieved_memory,
            "attention_weights": attention_weights,
            "write_gate": write_gate,
            "memory_updates": memory_updates,
            "memory_importance": memory_importance,
            "association_features": association_features,
            "top_memory_indices": top_indices,
            "current_step": current_step,
        }
        
        # 添加遗忘统计信息
        if forget_stats is not None:
            result["forget_stats"] = forget_stats
        
        # 添加记忆系统状态信息
        result["memory_system_stats"] = {
            "memory_slots_used": (self.memory_access_count > 0).sum().item(),
            "memory_access_mean": self.memory_access_count.mean().item(),
            "memory_access_std": self.memory_access_count.std().item(),
            "memory_age_mean": (current_step - self.memory_creation_time).mean().item() if current_step > 0 else 0,
            "memory_age_std": (current_step - self.memory_creation_time).std().item() if current_step > 0 else 0,
        }
        
        return result



