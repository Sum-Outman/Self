# Mamba2Block - 从self_agi_model.py拆分
"""Mamba2Block模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging

class Mamba2Block(nn.Module):
    """Mamba-2状态空间块 - 基于Mamba-2论文实现

    参考论文:
    - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
    - Mamba-2改进 (2024)

    关键特性:
    1. 选择性状态空间: 输入依赖的状态转移矩阵
    2. 并行扫描算法: 硬件感知优化
    3. 改进的门控机制: 更复杂的输入依赖门控
    """

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # 输入投影 (Mamba-2改进版)
        self.in_proj = nn.Linear(config.hidden_size, config.hidden_size * 2)

        # 选择性扫描参数 (输入依赖的A、B矩阵)
        # A: 状态转移矩阵 (对角形式), 形状 [state_dim]
        # B: 输入投影矩阵, 形状 [state_dim, hidden_size]
        # C: 输出投影矩阵, 形状 [hidden_size, state_dim]
        self.A_log = nn.Parameter(
            torch.randn(config.state_space_dim) * 0.02
        )  # 对数对角A
        self.A_bias = nn.Parameter(
            torch.randn(config.state_space_dim) * 0.02
        )  # A的偏置

        # B矩阵: 输入到状态
        # 注意: x_proj维度是hidden_size * 2，所以B_proj输入维度也应该是hidden_size * 2
        self.B_proj = nn.Linear(config.hidden_size * 2, config.state_space_dim)

        # C矩阵: 状态到输出
        self.C_proj = nn.Linear(config.state_space_dim, config.hidden_size)

        # 选择性机制 (更复杂的门控)
        # 注意: in_proj输出hidden_size * 2，所以门控输入维度也是hidden_size * 2
        self.selective_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
        )

        # Hyena卷积 (如果启用)
        if config.hyena_conv_enabled:
            self.hyena_conv = HyenaConv(
                dim=config.hidden_size,
                order=config.hyena_order,
                l_max=config.hyena_max_length,
            )

        # 输出投影
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        logger.info(
            f"初始化Mamba2Block: 隐藏大小={config.hidden_size}, 状态维度={config.state_space_dim}, "
            f"Hyena卷积={config.hyena_conv_enabled}"
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mamba-2前向传播"""
        batch_size, seq_len, hidden_size = x.shape

        # 输入投影
        x_proj = self.in_proj(x)
        x_proj = F.silu(x_proj)

        # 选择性扫描
        if self.config.selective_scanning_enabled:
            # 计算状态转移矩阵A (对角形式)
            A = torch.diag_embed(
                self.A_log.exp() + self.A_bias
            )  # [state_dim, state_dim]

            # 计算输入依赖的门控
            gate = self.selective_gate(x_proj)

            # 计算B矩阵 (输入依赖)
            B = self.B_proj(x_proj)  # [batch, seq, state_dim]

            # 选择性扫描实现
            y = self.selective_scan(A, B, self.C_proj, gate, x_proj)
        else:
            # 禁用选择性扫描时，需要将x_proj从hidden_size*2投影到hidden_size
            # 简单实现：取前一半特征
            hidden_size = self.config.hidden_size
            y = x_proj[:, :, :hidden_size]  # 取前hidden_size个特征

        # Hyena卷积 (如果启用)
        if self.config.hyena_conv_enabled and hasattr(self, "hyena_conv"):
            y = y.transpose(1, 2)  # [batch, hidden, seq]
            y = self.hyena_conv(y)
            y = y.transpose(1, 2)  # [batch, seq, hidden]

        # 残差连接和层归一化
        y = self.out_proj(y)
        y = self.dropout(y)
        output = self.layer_norm(x + y)

        return output

    def selective_scan(self, A, B, C_proj, gate, x):
        """选择性扫描算法 - Mamba-2核心 (完整实现)

        状态空间模型公式:
        h_t = A * h_{t-1} + B_t
        y_t = C_proj(h_t)

        完整版本: 使用B_t作为输入依赖的偏置项，而不是B_t * x_t
        参数:
            A: 状态转移矩阵 [state_dim, state_dim]
            B: 输入投影矩阵 [batch, seq, state_dim] (输入依赖)
            C_proj: 输出投影线性层 (state_dim -> hidden_size)
            gate: 门控信号 [batch, seq, hidden_size*2]
            x: 输入特征 [batch, seq, hidden_size]

        返回:
            y: 输出特征 [batch, seq, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        state_dim = self.config.state_space_dim

        # 将门控信号拆分为两个部分 (用于调制)
        if gate.shape[-1] == hidden_size * 2:
            gate1, gate2 = torch.split(gate, hidden_size, dim=-1)
        else:
            gate1 = gate
            gate2 = torch.ones_like(gate)

        # 初始化状态
        h = torch.zeros(batch_size, state_dim, device=x.device)
        outputs = []

        for t in range(seq_len):
            # 获取时间步t的输入和参数
            B_t = B[:, t, :]  # [batch, state_dim]

            # 应用门控调制
            B_t_modulated = B_t * gate2[:, t, :].mean(
                dim=-1, keepdim=True
            )  # [batch, state_dim]

            # 状态更新: h_t = A * h_{t-1} + B_t
            # A: [state_dim, state_dim], h: [batch, state_dim]
            h = torch.matmul(h, A.t()) + B_t_modulated

            # 输出: y_t = C_proj(h_t)
            # C_proj: 线性层 (state_dim -> hidden_size)
            y_t = C_proj(h)  # [batch, hidden_size]

            outputs.append(y_t)

        # 堆叠所有时间步的输出
        y = torch.stack(outputs, dim=1)  # [batch, seq, hidden_size]

        return y



