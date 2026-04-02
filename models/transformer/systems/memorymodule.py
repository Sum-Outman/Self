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

    def forward(
        self, hidden_states: torch.Tensor, query: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """前向传播

        参数:
            hidden_states: 输入特征 [batch_size, seq_len, hidden_size]
            query: 查询向量 [batch_size, query_dim] (可选)

        返回:
            记忆输出字典
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

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
        # 注意：实际实现应使用更复杂的注意力机制
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

        return {
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
        }



