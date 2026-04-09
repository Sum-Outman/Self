# TransformerBlock - 从self_agi_model.py拆分
"""TransformerBlock模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, config: "AGIModelConfig"):
        super().__init__()
        self.config = config

        # 自注意力 - 确保头数能整除嵌入维度
        num_heads = config.num_attention_heads
        if config.hidden_size % num_heads != 0:
            # 调整为最大可整除的头数
            num_heads = max(1, config.hidden_size // 64)
            if config.hidden_size % num_heads != 0:
                num_heads = 1  # 最终回退

        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=num_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        # 前馈网络
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)

        # 层归一化
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.output_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 激活函数
        self.activation = self._get_activation_fn(config.hidden_act)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        # 数值稳定性检查
        if torch.isnan(hidden_states).any():
            logger.warning(f"TransformerBlock输入包含NaN，形状: {hidden_states.shape}")
            # 用零替换NaN
            hidden_states = torch.where(
                torch.isnan(hidden_states),
                torch.zeros_like(hidden_states),
                hidden_states,
            )

        # 输入数值裁剪（防止过大值导致注意力分数溢出）
        if self.config.enable_input_clipping:
            clip_value = 10.0  # 裁剪到[-10, 10]范围
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)
            # 调试：记录裁剪情况
            if torch.abs(hidden_states).max() > clip_value:
                logger.debug(
                    f"TransformerBlock输入被裁剪，最大绝对值: {                         torch.abs(hidden_states).max().item():.2f}"
                )

        # 自注意力
        attention_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states, key_padding_mask=attention_mask
        )

        # 检查注意力输出NaN
        if torch.isnan(attention_output).any():
            logger.warning(
                f"TransformerBlock注意力输出包含NaN，形状: {attention_output.shape}"
            )
            attention_output = torch.where(
                torch.isnan(attention_output),
                torch.zeros_like(attention_output),
                attention_output,
            )

        # 残差连接和层归一化
        attention_output = self.dropout(attention_output)
        hidden_states = self.attention_layer_norm(hidden_states + attention_output)

        # 检查层归一化后NaN
        if torch.isnan(hidden_states).any():
            logger.warning(
                f"TransformerBlock层归一化后包含NaN，形状: {hidden_states.shape}"
            )
            hidden_states = torch.where(
                torch.isnan(hidden_states),
                torch.zeros_like(hidden_states),
                hidden_states,
            )

        # 前馈网络
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.activation(intermediate_output)

        # 检查激活输出NaN
        if torch.isnan(intermediate_output).any():
            logger.warning(
                f"TransformerBlock激活输出包含NaN，形状: {intermediate_output.shape}"
            )
            intermediate_output = torch.where(
                torch.isnan(intermediate_output),
                torch.zeros_like(intermediate_output),
                intermediate_output,
            )

        ff_output = self.output(intermediate_output)
        ff_output = self.dropout(ff_output)

        # 检查前馈输出NaN
        if torch.isnan(ff_output).any():
            logger.warning(f"TransformerBlock前馈输出包含NaN，形状: {ff_output.shape}")
            ff_output = torch.where(
                torch.isnan(ff_output), torch.zeros_like(ff_output), ff_output
            )

        # 残差连接和层归一化
        hidden_states = self.output_layer_norm(hidden_states + ff_output)

        # 最终检查
        if torch.isnan(hidden_states).any():
            logger.warning(
                f"TransformerBlock最终输出包含NaN，形状: {hidden_states.shape}"
            )
            hidden_states = torch.where(
                torch.isnan(hidden_states),
                torch.zeros_like(hidden_states),
                hidden_states,
            )

        return hidden_states

    def _get_activation_fn(
        self, activation: str
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """获取激活函数"""
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "tanh":
            return torch.tanh
        else:
            raise ValueError(f"未知激活函数: {activation}")
