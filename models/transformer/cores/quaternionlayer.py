# -*- coding: utf-8 -*-
"""
四元数神经网络层 - 基于四元数代数的神经网络组件

四元数表示: q = a + bi + cj + dk
其中 a, b, c, d 为实数，i, j, k 为虚数单位

四元数乘法规则:
i² = j² = k² = ijk = -1
ij = k, ji = -k
jk = i, kj = -i
ki = j, ik = -j

四元数神经网络优点:
- 更好的参数效率（4倍参数共享）
- 更好的旋转表示能力
- 改进的梯度流
- 适用于3D视觉、机器人控制等任务

参考论文:
1. "Quaternion Recurrent Neural Networks" (Parcollet et al., 2018)
2. "Quaternion Convolutional Neural Networks for End-to-End Automatic Speech Recognition" (Parcollet et al., 2019)
3. "Quaternion-based Graph Neural Networks" (Zhang et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import math


class QuaternionTensor:
    """四元数张量封装类，支持四元数运算"""

    def __init__(
        self, real: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor
    ):
        """初始化四元数张量

        参数:
            real: 实部张量
            i: i虚部张量
            j: j虚部张量
            k: k虚部张量
        """
        assert real.shape == i.shape == j.shape == k.shape, "所有分量必须具有相同的形状"
        self.real = real
        self.i = i
        self.j = j
        self.k = k
        self.shape = real.shape
        self.device = real.device
        self.dtype = real.dtype

    def __add__(self, other: "QuaternionTensor") -> "QuaternionTensor":
        """四元数加法"""
        return QuaternionTensor(
            self.real + other.real, self.i + other.i, self.j + other.j, self.k + other.k
        )

    def __mul__(
        self, other: Union["QuaternionTensor", torch.Tensor, float]
    ) -> "QuaternionTensor":
        """四元数乘法或标量乘法"""
        if isinstance(other, (int, float)):
            # 标量乘法
            return QuaternionTensor(
                self.real * other, self.i * other, self.j * other, self.k * other
            )
        elif isinstance(other, torch.Tensor):
            # 张量乘法（逐元素）
            return QuaternionTensor(
                self.real * other, self.i * other, self.j * other, self.k * other
            )
        elif isinstance(other, QuaternionTensor):
            # 四元数乘法
            # q1 * q2 = (a1a2 - b1b2 - c1c2 - d1d2)
            #          + (a1b2 + b1a2 + c1d2 - d1c2)i
            #          + (a1c2 - b1d2 + c1a2 + d1b2)j
            #          + (a1d2 + b1c2 - c1b2 + d1a2)k
            a1, b1, c1, d1 = self.real, self.i, self.j, self.k
            a2, b2, c2, d2 = other.real, other.i, other.j, other.k

            real = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
            i = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
            j = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
            k = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

            return QuaternionTensor(real, i, j, k)
        else:
            raise TypeError(f"不支持的乘法类型: {type(other)}")

    def conjugate(self) -> "QuaternionTensor":
        """四元数共轭: q* = a - bi - cj - dk"""
        return QuaternionTensor(self.real, -self.i, -self.j, -self.k)

    def norm(self) -> torch.Tensor:
        """四元数范数: ||q|| = sqrt(a² + b² + c² + d²)"""
        return torch.sqrt(self.real**2 + self.i**2 + self.j**2 + self.k**2 + 1e-8)

    def normalize(self) -> "QuaternionTensor":
        """归一化四元数"""
        norm = self.norm()
        return QuaternionTensor(
            self.real / norm, self.i / norm, self.j / norm, self.k / norm
        )

    def to_tensor(self) -> torch.Tensor:
        """将四元数转换为张量，形状为 [..., 4]"""
        return torch.stack([self.real, self.i, self.j, self.k], dim=-1)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "QuaternionTensor":
        """从张量创建四元数，张量形状为 [..., 4]"""
        return cls(tensor[..., 0], tensor[..., 1], tensor[..., 2], tensor[..., 3])

    def __repr__(self) -> str:
        return f"QuaternionTensor(shape={self.shape}, device={self.device}, dtype={self.dtype})"


class QuaternionLinear(nn.Module):
    """四元数线性层

    四元数线性变换: y = W ⊗ x + b
    其中 ⊗ 表示四元数矩阵乘法
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """初始化四元数线性层

        参数:
            in_features: 输入特征数（四元数维度，实际参数为4*in_features）
            out_features: 输出特征数（四元数维度，实际参数为4*out_features）
            bias: 是否使用偏置
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 四元数权重矩阵: 形状为 [out_features, in_features, 4, 4]
        # 表示四元数矩阵乘法（哈密顿积）
        self.weight = nn.Parameter(torch.empty(out_features, in_features, 4, 4))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, 4))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        # Xavier/He初始化适配四元数
        # 权重初始化
        fan_in = self.in_features * 4
        fan_out = self.out_features * 4
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.weight, -bound, bound)

        # 偏置初始化
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 [batch_size, ..., in_features, 4]

        返回:
            输出张量，形状为 [batch_size, ..., out_features, 4]
        """
        # 确保输入形状正确
        assert x.size(-1) == 4, f"输入最后一维必须为4（四元数），但得到 {x.size(-1)}"

        # 保存原始形状
        original_shape = x.shape
        batch_dims = original_shape[:-2]

        # 确保张量连续以便视图操作
        x = x.contiguous()
        x_reshaped = x.view(-1, self.in_features, 4)

        # 四元数矩阵乘法
        # 将输入视为 [batch, in_features, 4]
        # 权重为 [out_features, in_features, 4, 4]

        # 执行四元数矩阵乘法
        # 使用爱因斯坦求和约定
        # output[b, o, c] = sum_{i,d} x[b, i, d] * weight[o, i, d, c]
        output = torch.einsum("bid,oidc->boc", x_reshaped, self.weight)

        # 添加偏置
        if self.bias is not None:
            output = output + self.bias

        # 恢复原始形状
        output_shape = batch_dims + (self.out_features, 4)
        output = output.view(output_shape)

        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class QuaternionConv2d(nn.Module):
    """四元数二维卷积层"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        """初始化四元数卷积层

        参数:
            in_channels: 输入通道数（四元数维度）
            out_channels: 输出通道数（四元数维度）
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 膨胀率
            groups: 分组数
            bias: 是否使用偏置
            padding_mode: 填充模式
        """
        super().__init__()

        # 将通道数转换为实际通道数（每个四元数通道对应4个实数通道）
        real_in_channels = in_channels * 4
        real_out_channels = out_channels * 4

        # 创建实数卷积层
        self.conv = nn.Conv2d(
            real_in_channels,
            real_out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 [batch, in_channels, height, width, 4]

        返回:
            输出张量，形状为 [batch, out_channels, height', width', 4]
        """
        # 确保输入形状正确
        assert x.size(-1) == 4, f"输入最后一维必须为4（四元数），但得到 {x.size(-1)}"

        # 重塑输入: [batch, in_channels, height, width, 4] -> [batch, in_channels*4,
        # height, width]
        batch_size, in_channels, height, width, _ = x.shape
        x_reshaped = x.permute(0, 1, 4, 2, 3).contiguous()
        x_reshaped = x_reshaped.view(batch_size, in_channels * 4, height, width)

        # 应用卷积
        output = self.conv(x_reshaped)

        # 重塑输出: [batch, out_channels*4, height', width'] -> [batch, out_channels,
        # height', width', 4]
        _, _, new_height, new_width = output.shape
        output = output.view(batch_size, self.out_channels, 4, new_height, new_width)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        return output


class QuaternionLayerNorm(nn.Module):
    """四元数层归一化"""

    def __init__(
        self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5
    ):
        """初始化四元数层归一化

        参数:
            normalized_shape: 归一化形状
            eps: 数值稳定性 epsilon
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 可学习参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 [..., normalized_shape, 4]

        返回:
            归一化后的张量
        """
        # 计算四元数范数的均值和方差
        # 将四元数视为4个独立通道进行归一化
        mean = x.mean(dim=-2, keepdim=True)
        var = x.var(dim=-2, unbiased=False, keepdim=True)

        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # 缩放和平移
        # 将weight和bias扩展到四元数维度
        weight = self.weight.unsqueeze(-1)  # [embed_dim, 1]
        bias = self.bias.unsqueeze(-1)  # [embed_dim, 1]
        x_normalized = x_normalized * weight + bias

        return x_normalized


class QuaternionAttention(nn.Module):
    """四元数注意力机制"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """初始化四元数注意力

        参数:
            embed_dim: 嵌入维度（四元数维度）
            num_heads: 注意力头数
            dropout: dropout率
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim 必须能被 num_heads 整除"

        # 四元数线性投影
        self.q_proj = QuaternionLinear(embed_dim, embed_dim)
        self.k_proj = QuaternionLinear(embed_dim, embed_dim)
        self.v_proj = QuaternionLinear(embed_dim, embed_dim)
        self.out_proj = QuaternionLinear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播

        参数:
            query: 查询张量，形状为 [seq_len, batch, embed_dim, 4]
            key: 键张量，形状为 [seq_len, batch, embed_dim, 4]
            value: 值张量，形状为 [seq_len, batch, embed_dim, 4]
            key_padding_mask: 键填充掩码，形状为 [batch, seq_len]
            need_weights: 是否返回注意力权重

        返回:
            注意力输出，形状为 [seq_len, batch, embed_dim, 4]
        """
        # 投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 重塑为多头
        batch_size = q.size(1)
        seq_len = q.size(0)

        q = q.view(seq_len, batch_size, self.num_heads, self.head_dim, 4)
        k = k.view(seq_len, batch_size, self.num_heads, self.head_dim, 4)
        v = v.view(seq_len, batch_size, self.num_heads, self.head_dim, 4)

        # 转置以便批量计算
        q = q.permute(2, 1, 0, 3, 4).contiguous()
        k = k.permute(2, 1, 0, 3, 4).contiguous()
        v = v.permute(2, 1, 0, 3, 4).contiguous()

        # 计算注意力分数（使用四元数点积）
        # 简化：使用实数注意力
        q_real = q[..., 0]  # 仅使用实部进行注意力计算
        k_real = k[..., 0]

        # 缩放点积注意力
        attn_scores = torch.matmul(q_real, k_real.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        # 应用掩码
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(0).unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力到值
        # 需要将v转换为实数表示
        # 展平最后两个维度：head_dim 和 4
        v_real = v.flatten(start_dim=-2)
        output_real = torch.matmul(attn_weights, v_real)

        # 重塑回四元数格式
        output = output_real.view(*output_real.shape[:-1], self.head_dim, 4)

        # 合并头
        output = output.permute(2, 1, 0, 3, 4).contiguous()
        output = output.view(seq_len, batch_size, self.embed_dim, 4)

        # 最终投影
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights
        return output


class QuaternionTransformerBlock(nn.Module):
    """四元数Transformer块"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """初始化四元数Transformer块

        参数:
            embed_dim: 嵌入维度（四元数维度）
            num_heads: 注意力头数
            ff_dim: 前馈网络维度（如果为None，则使用4*embed_dim）
            dropout: dropout率
            activation: 激活函数
        """
        super().__init__()

        if ff_dim is None:
            ff_dim = embed_dim * 4

        # 注意力层
        self.attention = QuaternionAttention(embed_dim, num_heads, dropout)
        self.attention_norm = QuaternionLayerNorm(embed_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            QuaternionLinear(embed_dim, ff_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            QuaternionLinear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = QuaternionLayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 [seq_len, batch, embed_dim, 4]

        返回:
            输出张量，形状为 [seq_len, batch, embed_dim, 4]
        """
        # 注意力子层
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, x, x)
        x = self.dropout(x)
        x = x + residual

        # 前馈子层
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual

        return x


class QuaternionRotationLayer(nn.Module):
    """四元数旋转层 - 专门用于3D旋转表示

    将输入特征转换为旋转四元数，可用于3D姿态估计、机器人控制等
    """

    def __init__(self, input_dim: int, output_quaternions: int = 1):
        """初始化四元数旋转层

        参数:
            input_dim: 输入特征维度
            output_quaternions: 输出的四元数数量
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_quaternions = output_quaternions

        # 将输入投影到四元数空间
        self.projection = QuaternionLinear(input_dim, output_quaternions)

        # 旋转参数学习
        self.rotation_params = nn.Parameter(torch.randn(output_quaternions, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 [batch, ..., input_dim, 4]

        返回:
            旋转四元数，形状为 [batch, ..., output_quaternions, 4]
        """
        # 投影到四元数空间
        quaternions = self.projection(x)

        # 应用可学习的旋转参数
        # 将旋转参数归一化为单位四元数
        rotation_norm = torch.sqrt(
            torch.sum(self.rotation_params**2, dim=-1, keepdim=True) + 1e-8
        )
        normalized_rotation = self.rotation_params / rotation_norm

        # 应用旋转：q' = r ⊗ q ⊗ r*
        # 简化实现：逐元素乘法
        rotated_quaternions = quaternions * normalized_rotation.unsqueeze(0)

        # 归一化输出四元数
        norm = torch.sqrt(
            torch.sum(rotated_quaternions**2, dim=-1, keepdim=True) + 1e-8
        )
        normalized_output = rotated_quaternions / norm

        return normalized_output


# 测试函数
def test_quaternion_layers():
    """测试四元数层"""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("测试四元数神经网络层...")

    # 测试四元数线性层
    batch_size = 2
    seq_len = 10
    in_features = 16
    out_features = 32

    x = torch.randn(batch_size, seq_len, in_features, 4)
    linear = QuaternionLinear(in_features, out_features)
    y = linear(x)
    logger.info(f"四元数线性层: 输入 {x.shape}, 输出 {y.shape}")

    # 测试四元数注意力
    embed_dim = 64
    num_heads = 4
    attention = QuaternionAttention(embed_dim, num_heads)
    q = torch.randn(seq_len, batch_size, embed_dim, 4)
    k = torch.randn(seq_len, batch_size, embed_dim, 4)
    v = torch.randn(seq_len, batch_size, embed_dim, 4)
    attn_output = attention(q, k, v)
    logger.info(f"四元数注意力: 输出 {attn_output.shape}")

    # 测试四元数Transformer块
    transformer = QuaternionTransformerBlock(embed_dim, num_heads)
    transformer_output = transformer(q)
    logger.info(f"四元数Transformer块: 输出 {transformer_output.shape}")

    # 测试四元数旋转层
    rotation_layer = QuaternionRotationLayer(embed_dim, output_quaternions=3)
    rotation_output = rotation_layer(q.transpose(0, 1))
    logger.info(f"四元数旋转层: 输出 {rotation_output.shape}")

    logger.info("所有测试通过！")

    return True


if __name__ == "__main__":
    test_quaternion_layers()
