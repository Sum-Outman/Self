# HyenaConv - 从self_agi_model.py拆分
"""HyenaConv模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class HyenaConv(nn.Module):
    """Hyena卷积层 - 长卷积核实现

    参考论文: "Hyena Hierarchy: Towards Larger Convolutional Language Models" (Poli et al., 2023)

    关键特性:
    1. 长卷积核: 支持超长序列建模
    2. FFT加速: 频域卷积实现
    3. 可学习参数: 自适应的卷积核
    """

    def __init__(self, dim: int, order: int = 4, l_max: int = 2048):
        super().__init__()
        self.dim = dim
        self.order = order
        self.l_max = l_max

        # 长卷积核参数
        self.kernel = nn.Parameter(torch.randn(order, dim, l_max) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))

        logger.info(f"初始化HyenaConv: 维度={dim}, 阶数={order}, 最大长度={l_max}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hyena卷积前向传播"""
        # 实现快速卷积算法 (FFT-based)
        batch_size, dim, seq_len = x.shape

        # 确保序列长度不超过最大长度
        if seq_len > self.l_max:
            raise ValueError(f"序列长度{seq_len}超过最大长度{self.l_max}")

        # FFT变换
        x_fft = torch.fft.rfft(x, n=seq_len * 2, dim=-1)

        # 频域卷积
        kernel_fft = torch.fft.rfft(self.kernel, n=seq_len * 2, dim=-1)

        # 逐元素乘法 (卷积定理)
        y_fft = x_fft * kernel_fft

        # 逆FFT
        y = torch.fft.irfft(y_fft, n=seq_len * 2, dim=-1)[:, :, :seq_len]

        # 残差连接 + 偏置
        y = y + x + self.bias.unsqueeze(-1)

        return y



