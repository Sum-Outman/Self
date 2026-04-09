# StripedHyenaBlock - 从self_agi_model.py拆分
"""StripedHyenaBlock模块"""

import torch
import torch.nn as nn
from typing import Optional


class StripedHyenaBlock(nn.Module):
    """StripedHyena混合块 - 交替使用Hyena和注意力

    参考论文: "StripedHyena: Moving Beyond Transformers with Hybrid Signal Processing Models" (Poli et al., 2024)

    关键特性:
    1. 交替架构: Hyena层和注意力层交替
    2. 信号处理混合: 结合卷积和注意力优势
    3. 长程依赖: 处理超长序列
    """

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # Hyena层
        self.hyena_layer = HyenaConv(
            dim=config.hidden_size,
            order=config.hyena_order,
            l_max=config.hyena_max_length,
        )

        # 注意力层
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

        logger.info(
            f"初始化StripedHyenaBlock: 隐藏大小={config.hidden_size}, "
            f"Hyena阶数={config.hyena_order}, 注意力头数={config.num_attention_heads}"
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """StripedHyena前向传播"""
        residual = x

        # Hyena路径
        x_hyena = x.transpose(1, 2)  # [batch, hidden, seq]
        x_hyena = self.hyena_layer(x_hyena)
        x_hyena = x_hyena.transpose(1, 2)  # [batch, seq, hidden]
        x_hyena = self.norm1(residual + x_hyena)

        # 注意力路径
        x_attn, _ = self.attention_layer(
            x_hyena, x_hyena, x_hyena, key_padding_mask=attention_mask
        )
        x_attn = self.norm2(x_hyena + x_attn)

        # 前馈网络
        x_out = self.ffn(x_attn)

        return x_out


# ============================================================================
# Switch Transformers 实现
# ============================================================================
