# AttnResAttentionBlock - 从self_agi_model.py拆分
"""AttnResAttentionBlock模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging

class AttnResAttentionBlock(nn.Module):
    """Attention Residuals块 - 基于Kimi 2026年3月16日论文《Attention Residuals: Dynamic Depthwise Aggregation for Large Language Models》

    核心创新：将残差连接从固定累加改为动态注意力聚合
    传统：x_{l+1} = x_l + f(x_l)
    AttnRes：x_{l+1} = α(x_l) * x_l + β(x_l) * f(x_l)

    关键特性：
    1. 动态深度聚合：α和β是输入的函数，通过注意力机制计算
    2. 输入依赖权重：权重根据输入内容动态调整
    3. 深度方向聚合：每个特征维度有不同的聚合权重
    4. 训练效率提升：论文报告+25%训练效率
    5. 推理延迟降低：<2%额外延迟

    参考论文：Kimi 2026年3月16日发布的最新核心论文《Attention Residuals: Dynamic Depthwise Aggregation for Large Language Models》
    """

    def __init__(self, config: 'AGIModelConfig'):
        super().__init__()
        self.config = config

        # 自注意力（使用高效注意力块或标准注意力）
        if config.efficient_attention_enabled:
            self.attention = EfficientAttentionBlock(config)
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
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

        # ============ AttnRes 动态深度聚合层 ============
        # 论文核心：动态计算α和β权重
        self.hidden_size = config.hidden_size

        # 动态聚合权重网络（输入依赖的权重计算）
        # 输出两个权重向量：α和β，每个都是[hidden_size]维度
        self.dynamic_aggregation = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2, eps=config.layer_norm_eps),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.Sigmoid(),  # 输出在[0,1]范围内，确保稳定性
        )

        # 深度方向注意力（可选，论文中的深度聚合机制）
        self.depthwise_attention_enabled = getattr(
            config, "depthwise_attention_enabled", True
        )
        if self.depthwise_attention_enabled:
            # 深度注意力：计算每个特征维度的聚合权重
            self.depth_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=min(4, config.hidden_size // 64),  # 减少头数，专注于深度
                dropout=config.attention_probs_dropout_prob,
                batch_first=True,
                kdim=config.hidden_size,
                vdim=config.hidden_size,
            )

        # 残差缩放因子（学习全局缩放）
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.attention_scale = nn.Parameter(torch.ones(1))

        logger.info(
            f"初始化AttnRes注意力块: 隐藏大小={config.hidden_size}, "
            f"深度注意力启用={self.depthwise_attention_enabled}"
        )

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

    def _compute_dynamic_weights(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算动态聚合权重α和β

        根据输入x动态计算残差权重和注意力权重。
        论文核心：权重是输入的函数，实现动态深度聚合。

        参数:
            x: 输入张量 [batch_size, seq_len, hidden_size]

        返回:
            alpha: 残差权重 [batch_size, seq_len, hidden_size]
            beta: 注意力权重 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape

        # 方法1：简单的动态聚合（线性层+激活）
        # 计算每个位置的聚合权重
        aggregated = self.dynamic_aggregation(x)  # [batch_size, seq_len, hidden_size*2]

        # 分割为alpha和beta
        alpha, beta = torch.split(aggregated, self.hidden_size, dim=-1)

        # 确保权重和为1（softmax风格，但保持深度方向独立性）
        # 使用sigmoid确保在[0,1]范围内，但不强制和为1
        # 论文中可能使用softmax，但深度聚合允许每个维度独立

        # 可选：应用深度注意力增强
        if self.depthwise_attention_enabled and hasattr(self, "depth_attention"):
            # 使用深度注意力进一步细化权重
            # 将x视为查询，计算深度方向的注意力
            depth_weights, _ = self.depth_attention(x, x, x)
            # 混合原始权重和注意力权重
            alpha = 0.7 * alpha + 0.3 * depth_weights
            beta = 0.7 * beta + 0.3 * (1.0 - depth_weights)  # 互补

        # 应用缩放因子
        alpha = alpha * self.residual_scale
        beta = beta * self.attention_scale

        return alpha, beta

    def _apply_dynamic_aggregation(
        self, residual: torch.Tensor, attention_out: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """应用动态深度聚合

        实现AttnRes核心公式：x_new = α(x) * x + β(x) * f(x)

        参数:
            residual: 残差连接输入 [batch_size, seq_len, hidden_size]
            attention_out: 注意力输出 [batch_size, seq_len, hidden_size]
            x: 原始输入（用于计算权重）[batch_size, seq_len, hidden_size]

        返回:
            聚合后的张量 [batch_size, seq_len, hidden_size]
        """
        # 计算动态权重
        alpha, beta = self._compute_dynamic_weights(x)

        # 应用动态聚合
        # 论文公式：x_new = α(x) * residual + β(x) * attention_out
        aggregated = alpha * residual + beta * attention_out

        return aggregated

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播 - AttnRes增强版

        实现动态深度聚合的Transformer块。

        参数:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len] 可选

        返回:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        # 保存原始输入用于动态聚合
        residual_input = hidden_states

        # 数值稳定性检查
        if torch.isnan(hidden_states).any():
            logger.warning(
                f"AttnResAttentionBlock输入包含NaN，形状: {hidden_states.shape}"
            )
            hidden_states = torch.where(
                torch.isnan(hidden_states),
                torch.zeros_like(hidden_states),
                hidden_states,
            )

        # 输入数值裁剪
        if self.config.enable_input_clipping:
            clip_value = 10.0
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)

        # ============ 自注意力部分 ============
        # 计算注意力输出
        if isinstance(self.attention, nn.MultiheadAttention):
            attention_output, _ = self.attention(
                hidden_states,
                hidden_states,
                hidden_states,
                key_padding_mask=attention_mask,
            )
        else:
            # 使用高效注意力块
            attention_output = self.attention(hidden_states, attention_mask)

        # 检查注意力输出NaN
        if torch.isnan(attention_output).any():
            logger.warning(
                f"AttnResAttentionBlock注意力输出包含NaN，形状: {attention_output.shape}"
            )
            attention_output = torch.where(
                torch.isnan(attention_output),
                torch.zeros_like(attention_output),
                attention_output,
            )

        # ============ AttnRes动态聚合 ============
        # 应用动态深度聚合代替传统的残差连接
        # 传统：hidden_states = hidden_states + attention_output
        # AttnRes：hidden_states = α(hidden_states) * hidden_states + β(hidden_states) * attention_output
        aggregated_attention = self._apply_dynamic_aggregation(
            residual=hidden_states, attention_out=attention_output, x=hidden_states
        )

        # Dropout和层归一化
        aggregated_attention = self.dropout(aggregated_attention)
        hidden_states = self.attention_layer_norm(aggregated_attention)

        # 检查层归一化后NaN
        if torch.isnan(hidden_states).any():
            logger.warning(
                f"AttnResAttentionBlock层归一化后包含NaN，形状: {hidden_states.shape}"
            )
            hidden_states = torch.where(
                torch.isnan(hidden_states),
                torch.zeros_like(hidden_states),
                hidden_states,
            )

        # ============ 前馈网络部分 ============
        # 保存前馈网络前的状态
        ff_residual = hidden_states

        # 前馈网络计算
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.activation(intermediate_output)

        # 检查激活输出NaN
        if torch.isnan(intermediate_output).any():
            logger.warning(
                f"AttnResAttentionBlock激活输出包含NaN，形状: {intermediate_output.shape}"
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
            logger.warning(
                f"AttnResAttentionBlock前馈输出包含NaN，形状: {ff_output.shape}"
            )
            ff_output = torch.where(
                torch.isnan(ff_output), torch.zeros_like(ff_output), ff_output
            )

        # ============ AttnRes动态聚合（前馈网络部分） ============
        # 对前馈网络也应用动态聚合
        aggregated_ff = self._apply_dynamic_aggregation(
            residual=ff_residual, attention_out=ff_output, x=ff_residual
        )

        # 检查聚合输出NaN
        if torch.isnan(aggregated_ff).any():
            logger.warning(
                f"AttnResAttentionBlock聚合输出包含NaN，形状: {aggregated_ff.shape}"
            )
            aggregated_ff = torch.where(
                torch.isnan(aggregated_ff),
                torch.zeros_like(aggregated_ff),
                aggregated_ff,
            )

        # 层归一化和最终输出
        hidden_states = self.output_layer_norm(aggregated_ff)

        # 最终检查
        if torch.isnan(hidden_states).any():
            logger.warning(
                f"AttnResAttentionBlock最终输出包含NaN，形状: {hidden_states.shape}"
            )
            hidden_states = torch.where(
                torch.isnan(hidden_states),
                torch.zeros_like(hidden_states),
                hidden_states,
            )

        return hidden_states



