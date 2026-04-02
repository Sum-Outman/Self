# DoRALinear - 从self_agi_model.py拆分
"""DoRALinear模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class DoRALinear(nn.Module):
    """DoRA线性层 - 权重分解的低秩适应

    参考论文: "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al., 2024)

    核心思想:
    1. 将权重矩阵分解为幅度(magnitude)和方向(direction)分量
    2. 方向分量通过低秩分解进行参数化
    3. 幅度分量作为可学习的缩放因子

    公式:
    W' = m * V / ||V||
    V = W + ΔW = W + BA (低秩分解)

    其中:
    - W: 预训练权重
    - B: 低秩矩阵 (r × out_features)
    - A: 低秩矩阵 (in_features × r)
    - m: 可学习的幅度参数
    - V: 方向矩阵
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha

        # 获取基础层参数
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.weight = base_layer.weight  # [out_features, in_features]
        self.bias = base_layer.bias  # [out_features] 或 None

        # 冻结基础权重
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # DoRA低秩适配器 (ΔW = BA)
        self.lora_A = nn.Parameter(torch.randn(self.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))

        # 幅度参数 (每个输出特征一个缩放因子)
        self.magnitude = nn.Parameter(torch.ones(self.out_features))

        # 缩放因子 (alpha / rank)
        self.scaling = alpha / rank

        logger.info(
            f"初始化DoRA线性层: 输入特征={self.in_features}, "
            f"输出特征={self.out_features}, 秩={rank}, alpha={alpha}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DoRA前向传播"""
        # 计算基础权重输出
        base_output = F.linear(x, self.weight, self.bias)

        # 计算DoRA适配器输出 (ΔW = BA)
        # ΔW = (A^T B^T)^T = BA^T
        delta_W = torch.matmul(
            self.lora_B.T, self.lora_A.T
        )  # [out_features, in_features]

        # 计算方向矩阵 V = W + scaling * ΔW
        V = self.weight + self.scaling * delta_W

        # 计算方向矩阵的范数 (列范数)
        # 论文中计算V的Frobenius范数，但实际实现通常计算每行的L2范数
        V_norm = torch.norm(V, p=2, dim=1, keepdim=True)  # [out_features, 1]

        # 避免除零
        V_norm = torch.where(V_norm == 0, torch.ones_like(V_norm), V_norm)

        # 归一化方向矩阵
        V_normalized = V / V_norm

        # 应用幅度缩放
        # W' = magnitude * V_normalized
        weight_decomposed = self.magnitude.unsqueeze(1) * V_normalized

        # 计算DoRA输出
        dora_output = F.linear(x, weight_decomposed, None)  # 不使用基础偏置

        # 组合输出: 基础输出 + DoRA输出
        # 论文中DoRA完全替换权重，但我们可以使用残差连接
        output = base_output + dora_output

        return output

    def merge_weights(self):
        """合并DoRA权重到基础层中

        训练完成后，将DoRA适配器合并到基础权重中，
        以便推理时不需要额外的计算。
        """
        with torch.no_grad():
            # 计算ΔW
            delta_W = torch.matmul(self.lora_B.T, self.lora_A.T)

            # 计算方向矩阵 V = W + scaling * ΔW
            V = self.weight + self.scaling * delta_W

            # 计算方向矩阵范数
            V_norm = torch.norm(V, p=2, dim=1, keepdim=True)
            V_norm = torch.where(V_norm == 0, torch.ones_like(V_norm), V_norm)

            # 归一化方向矩阵
            V_normalized = V / V_norm

            # 应用幅度缩放
            weight_merged = self.magnitude.unsqueeze(1) * V_normalized

            # 更新基础权重
            self.weight.data.copy_(weight_merged)

            # 删除DoRA参数以释放内存
            del self.lora_A
            del self.lora_B
            del self.magnitude

            logger.info("DoRA权重已合并到基础层中")



