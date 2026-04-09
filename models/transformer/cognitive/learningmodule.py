# LearningModule - 从self_agi_model.py拆分
"""Learning模块"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class LearningModule(nn.Module):
    """学习模块"""

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 学习网络
        self.learning_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # 知识整合
        self.knowledge_integration = nn.Linear(
            config.hidden_size * 2, config.hidden_size
        )

        # 适应网络
        self.adaptation_network = nn.Linear(config.hidden_size, config.hidden_size)

        # 层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        new_knowledge: Optional[torch.Tensor] = None,
        feedback: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """学习过程

        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            new_knowledge: 新知识
            feedback: 反馈信息

        返回:
            学习输出字典
        """
        # 学习特征
        learned_features = self.learning_network(hidden_states)
        learned_features = self.layer_norm(learned_features)

        # 整合新知识
        if new_knowledge is not None:
            # 拼接特征和新知识
            combined = torch.cat([learned_features, new_knowledge], dim=-1)
            integrated = self.knowledge_integration(combined)
        else:
            integrated = learned_features

        # 适应过程
        if feedback is not None:
            adapted = self.adaptation_network(integrated + feedback)
        else:
            adapted = integrated

        adapted = self.layer_norm(adapted)

        return {
            "learned_features": learned_features,
            "integrated_knowledge": integrated,
            "adapted_features": adapted,
        }
