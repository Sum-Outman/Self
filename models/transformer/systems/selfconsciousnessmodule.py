# SelfConsciousnessModule - 从self_agi_model.py拆分
"""SelfConsciousness模块"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class SelfConsciousnessModule(nn.Module):
    """自我意识模块 - 实现自主意识和自我认知"""

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 自我表征网络
        self.self_representation = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 意识状态编码器
        self.consciousness_encoder = nn.Sequential(
            nn.Linear(
                config.hidden_size * 2, config.hidden_size
            ),  # 修复：输入维度改为hidden_size * 2以匹配拼接输入
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 意图推理网络
        self.intent_reasoning = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2, eps=config.layer_norm_eps),
        )

        # 自我反思网络
        self.self_reflection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """前向传播"""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 自我表征
        self_repr = self.self_representation(hidden_states.mean(dim=1))

        # 意识状态
        if context is not None:
            # 处理context维度：如果context是2D，扩展为3D以匹配序列长度
            if context.dim() == 2:
                # context形状: [batch_size, feature_dim]
                # 扩展为: [batch_size, seq_len, feature_dim]
                context = context.unsqueeze(1).expand(-1, seq_len, -1)
            elif context.dim() == 3:
                # context已经是3D，检查序列长度是否匹配
                if context.shape[1] != seq_len:
                    # 使用插值调整序列长度
                    context = torch.nn.functional.interpolate(
                        context.transpose(1, 2),  # [batch, feature, seq_len]
                        size=seq_len,
                        mode="linear",
                        align_corners=False,
                    ).transpose(
                        1, 2
                    )  # [batch, seq_len, feature]

            consciousness_input = torch.cat(
                [self_repr.unsqueeze(1).expand(-1, seq_len, -1), context], dim=-1
            )
        else:
            consciousness_input = self_repr.unsqueeze(1).expand(-1, seq_len, -1)

        # 检查维度兼容性
        input_dim = consciousness_input.shape[-1]
        expected_dim = self.consciousness_encoder[0].in_features

        if input_dim != expected_dim:
            # 动态创建投影层（如果不存在或不匹配）
            if (
                not hasattr(self, "_consciousness_projection")
                or self._consciousness_projection.in_features != input_dim
            ):
                self._consciousness_projection = nn.Linear(input_dim, expected_dim).to(
                    consciousness_input.device
                )
            consciousness_input = self._consciousness_projection(consciousness_input)

        consciousness_state = self.consciousness_encoder(consciousness_input)

        # 意图推理
        intent_features = self.intent_reasoning(consciousness_state)

        # 自我反思
        reflection = self.self_reflection(hidden_states)

        return {
            "self_representation": self_repr,
            "consciousness_state": consciousness_state,
            "intent_features": intent_features,
            "reflection": reflection,
            "self_awareness_score": torch.sigmoid(self_repr.mean(dim=-1, keepdim=True)),
        }
