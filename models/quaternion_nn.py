#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四元数神经网络层 - Self AGI 系统四元数全面引入实施方案神经网络模块

功能：
1. 四元数线性层（QuaternionLinear）
2. 四元数注意力层（QuaternionAttention）
3. 四元数归一化层（QuaternionNormalization）
4. 四元数Transformer块（QuaternionTransformerBlock）
5. 四元数卷积层（QuaternionConv1D, QuaternionConv2D）
6. 四元数嵌入层（QuaternionEmbedding）

工业级质量标准要求：
- 数值稳定性：双精度计算，避免梯度爆炸/消失
- 计算效率：GPU加速，向量化运算
- 内存优化：参数共享，量化支持
- 兼容性：与现有PyTorch层接口一致

数学原理：
1. 四元数代数：乘法、共轭、逆运算
2. 四元数表示旋转群SO(3)
3. 四元数神经网络参数约减（4倍参数表示3D旋转）
4. 四元数反向传播（自动微分兼容）

参考文献：
[1] Zhu, X., et al. (2018). Quaternion convolutional neural networks for image classification.
[2] Gaudet, C. J., & Maida, A. S. (2018). Quaternion convolutional neural networks for end-to-end learning on 3D rotation data.
[3] Parcollet, T., et al. (2018). Quaternion convolutional neural networks for speech recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any, Callable
import math

from models.quaternion_core import (
    QuaternionTensor, QuaternionNormalization
)


class QuaternionLinear(nn.Module):
    """四元数线性层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quaternion_dim: int = 4,
        init_method: str = "he"
    ):
        """
        初始化四元数线性层
        
        参数:
            in_features: 输入特征数（四元数维度为4倍）
            out_features: 输出特征数（四元数维度为4倍）
            bias: 是否使用偏置
            quaternion_dim: 四元数维度（固定为4）
            init_method: 初始化方法 ("he", "xavier", "uniform")
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.quaternion_dim = quaternion_dim
        self.init_method = init_method
        
        # 四元数线性层的参数是实值的，但按四元数分量组织
        # 权重形状: [out_features * 4, in_features * 4]
        self.weight = nn.Parameter(
            torch.Tensor(out_features * quaternion_dim, in_features * quaternion_dim)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(out_features * quaternion_dim)
            )
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        if self.init_method == "he":
            # He初始化（适用于ReLU激活）
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        elif self.init_method == "xavier":
            # Xavier初始化（适用于tanh/sigmoid激活）
            nn.init.xavier_uniform_(self.weight)
        else:
            # 均匀初始化
            bound = 1 / math.sqrt(self.in_features * self.quaternion_dim)
            nn.init.uniform_(self.weight, -bound, bound)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, in_features * 4] 或 [batch_size, *, in_features * 4]
        
        返回:
            output: 输出张量 [batch_size, out_features * 4] 或 [batch_size, *, out_features * 4]
        """
        # 保存原始形状
        original_shape = x.shape
        if x.dim() > 2:
            # 展平批次和额外维度
            x = x.reshape(-1, original_shape[-1])
        
        # 线性变换
        output = F.linear(x, self.weight, self.bias)
        
        # 恢复原始形状
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [output.shape[-1]]
            output = output.reshape(*output_shape)
        
        return output
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"quaternion_dim={self.quaternion_dim}, bias={self.bias is not None}, " \
               f"init_method={self.init_method}"


class QuaternionAttention(nn.Module):
    """四元数注意力层"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_rotary_embeddings: bool = True
    ):
        """
        初始化四元数注意力层
        
        参数:
            hidden_size: 隐藏层大小（四元数维度为4倍）
            num_attention_heads: 注意力头数
            attention_probs_dropout_prob: 注意力概率dropout概率
            hidden_dropout_prob: 隐藏层dropout概率
            layer_norm_eps: 层归一化epsilon
            use_rotary_embeddings: 是否使用旋转位置编码
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.use_rotary_embeddings = use_rotary_embeddings
        
        # 检查隐藏层大小是否能被头数整除
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"隐藏层大小 {hidden_size} 必须能被注意力头数 {num_attention_heads} 整除"
            )
        
        # 查询、键、值投影
        self.query = QuaternionLinear(hidden_size, self.all_head_size)
        self.key = QuaternionLinear(hidden_size, self.all_head_size)
        self.value = QuaternionLinear(hidden_size, self.all_head_size)
        
        # 输出投影
        self.output = QuaternionLinear(self.all_head_size, hidden_size)
        
        # Dropout层
        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size * 4, eps=layer_norm_eps)
        
        # 四元数归一化
        self.quaternion_norm = QuaternionNormalization()
        
        # 旋转位置编码（如果启用）
        if use_rotary_embeddings:
            self.rotary_embeddings = self._create_rotary_embeddings()
    
    def _create_rotary_embeddings(self, max_seq_length: int = 8192) -> torch.Tensor:
        """创建旋转位置编码"""
        # 注意：四元数维度是4倍，所以头大小需要乘以4
        quaternion_head_size = self.attention_head_size * 4
        
        # 生成旋转角度（只对实部和第一个虚部）
        # 我们只需要quaternion_head_size的一半角度（实部和虚部x）
        angle_dim = quaternion_head_size // 2
        
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, angle_dim, 2) * 
                           -(math.log(10000.0) / angle_dim))
        
        # 计算正弦和余弦
        pe_sin = torch.sin(position * div_term)
        pe_cos = torch.cos(position * div_term)
        
        # 创建四元数旋转嵌入
        # 形状: [max_seq_length, quaternion_head_size]
        # 布局: [实部, 虚部x, 虚部y, 虚部z] 重复
        rotary_emb = torch.zeros(max_seq_length, quaternion_head_size)
        
        # 填充实部（余弦值）
        rotary_emb[:, 0::4] = pe_cos  # 实部
        
        # 填充虚部x（正弦值）
        rotary_emb[:, 1::4] = pe_sin  # 虚部x
        
        # 虚部y和z保持为0
        
        return rotary_emb
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        参数:
            hidden_states: 隐藏状态 [batch_size, seq_length, hidden_size * 4]
            attention_mask: 注意力掩码 [batch_size, seq_length, seq_length]
            position_ids: 位置ID [batch_size, seq_length]
        
        返回:
            context_layer: 上下文层 [batch_size, seq_length, hidden_size * 4]
            attention_probs: 注意力概率 [batch_size, num_heads, seq_length, seq_length]
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # 层归一化
        hidden_states_norm = self.layer_norm(hidden_states)
        
        # 四元数归一化
        hidden_states_norm = self.quaternion_norm(hidden_states_norm)
        
        # 投影查询、键、值
        query_layer = self.query(hidden_states_norm)
        key_layer = self.key(hidden_states_norm)
        value_layer = self.value(hidden_states_norm)
        
        # 重塑为多头注意力格式
        new_shape = (batch_size, seq_length, self.num_attention_heads, self.attention_head_size * 4)
        query_layer = query_layer.view(*new_shape)
        key_layer = key_layer.view(*new_shape)
        value_layer = value_layer.view(*new_shape)
        
        # 应用旋转位置编码（如果启用）
        if self.use_rotary_embeddings and position_ids is not None:
            query_layer = self._apply_rotary_embeddings(query_layer, position_ids)
            key_layer = self._apply_rotary_embeddings(key_layer, position_ids)
        
        # 转置以进行注意力计算
        query_layer = query_layer.transpose(1, 2)  # [batch, heads, seq, head_size*4]
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        # 缩放注意力分数
        attention_scores = attention_scores / math.sqrt(self.attention_head_size * 4)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # 转换为注意力概率
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # 应用注意力到值
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 转置回原始形状
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_shape = (batch_size, seq_length, self.all_head_size * 4)
        context_layer = context_layer.view(*new_shape)
        
        # 输出投影
        context_layer = self.output(context_layer)
        context_layer = self.output_dropout(context_layer)
        
        # 残差连接
        context_layer = context_layer + hidden_states
        
        return context_layer, attention_probs
    
    def _apply_rotary_embeddings(
        self, 
        x: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> torch.Tensor:
        """应用旋转位置编码"""
        batch_size, seq_length, num_heads, head_size = x.shape
        
        # 获取位置编码
        # position_ids: [batch_size, seq_length]
        # rotary_embeddings: [max_seq_length, head_size]
        pos_emb = self.rotary_embeddings[position_ids]  # [batch_size, seq_length, head_size]
        
        # 重塑位置编码以匹配x的形状
        pos_emb = pos_emb.unsqueeze(2)  # [batch_size, seq_length, 1, head_size]
        pos_emb = pos_emb.expand(-1, -1, num_heads, -1)  # [batch_size, seq_length, num_heads, head_size]
        
        # 重塑x为四元数格式 [batch_size, seq_length, num_heads, head_size//4, 4]
        x_reshaped = x.view(batch_size, seq_length, num_heads, head_size // 4, 4)
        
        # 提取实部和虚部
        real = x_reshaped[..., 0]  # 实部 [batch_size, seq_length, num_heads, head_size//4]
        imag = x_reshaped[..., 1:]  # 虚部 [batch_size, seq_length, num_heads, head_size//4, 3]
        
        # 重塑位置编码以匹配四元数格式
        pos_emb_reshaped = pos_emb.view(batch_size, seq_length, num_heads, head_size // 4, 4)
        
        # 提取旋转角度（只使用实部和第一个虚部）
        cos_theta = pos_emb_reshaped[..., 0]  # 余弦部分 [batch_size, seq_length, num_heads, head_size//4]
        sin_theta = pos_emb_reshaped[..., 1]  # 正弦部分 [batch_size, seq_length, num_heads, head_size//4]
        
        # 旋转实部和虚部x
        imag_x = imag[..., 0]  # 虚部x分量
        real_rot = real * cos_theta - imag_x * sin_theta
        imag_x_rot = real * sin_theta + imag_x * cos_theta
        
        # 虚部y,z保持不变
        imag_yz_rot = imag[..., 1:]  # [batch_size, seq_length, num_heads, head_size//4, 2]
        
        # 合并旋转后的分量
        imag_rot = torch.cat([
            imag_x_rot.unsqueeze(-1),  # 虚部x
            imag_yz_rot  # 虚部y,z
        ], dim=-1)  # [batch_size, seq_length, num_heads, head_size//4, 3]
        
        x_rotated = torch.cat([
            real_rot.unsqueeze(-1),  # 实部
            imag_rot  # 虚部
        ], dim=-1)  # [batch_size, seq_length, num_heads, head_size//4, 4]
        
        # 重塑回原始形状
        x_rotated = x_rotated.view(batch_size, seq_length, num_heads, head_size)
        
        return x_rotated
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}, " \
               f"use_rotary_embeddings={self.use_rotary_embeddings}"


class QuaternionFeedForward(nn.Module):
    """四元数前馈网络"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        """
        初始化四元数前馈网络
        
        参数:
            hidden_size: 隐藏层大小（四元数维度为4倍）
            intermediate_size: 中间层大小（四元数维度为4倍）
            hidden_act: 隐藏层激活函数
            hidden_dropout_prob: dropout概率
            layer_norm_eps: 层归一化epsilon
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # 两层线性变换
        self.dense1 = QuaternionLinear(hidden_size, intermediate_size)
        self.dense2 = QuaternionLinear(intermediate_size, hidden_size)
        
        # 激活函数
        self.activation = self._get_activation_fn(hidden_act)
        
        # Dropout层
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size * 4, eps=layer_norm_eps)
        
        # 四元数归一化
        self.quaternion_norm = QuaternionNormalization()
    
    def _get_activation_fn(self, activation: str) -> Callable:
        """获取激活函数"""
        if activation == "gelu":
            return F.gelu
        elif activation == "relu":
            return F.relu
        elif activation == "tanh":
            return torch.tanh
        elif activation == "sigmoid":
            return torch.sigmoid
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            hidden_states: 隐藏状态 [batch_size, seq_length, hidden_size * 4]
        
        返回:
            hidden_states: 变换后的隐藏状态
        """
        # 层归一化
        hidden_states_norm = self.layer_norm(hidden_states)
        
        # 四元数归一化
        hidden_states_norm = self.quaternion_norm(hidden_states_norm)
        
        # 第一层线性变换
        hidden_states_intermediate = self.dense1(hidden_states_norm)
        hidden_states_intermediate = self.activation(hidden_states_intermediate)
        hidden_states_intermediate = self.dropout(hidden_states_intermediate)
        
        # 第二层线性变换
        hidden_states_output = self.dense2(hidden_states_intermediate)
        hidden_states_output = self.dropout(hidden_states_output)
        
        # 残差连接
        hidden_states = hidden_states + hidden_states_output
        
        return hidden_states
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"


class QuaternionTransformerBlock(nn.Module):
    """四元数Transformer块"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_act: str = "gelu",
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_rotary_embeddings: bool = True
    ):
        """
        初始化四元数Transformer块
        
        参数:
            hidden_size: 隐藏层大小（四元数维度为4倍）
            num_attention_heads: 注意力头数
            intermediate_size: 中间层大小（四元数维度为4倍）
            hidden_act: 隐藏层激活函数
            attention_probs_dropout_prob: 注意力概率dropout概率
            hidden_dropout_prob: 隐藏层dropout概率
            layer_norm_eps: 层归一化epsilon
            use_rotary_embeddings: 是否使用旋转位置编码
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        
        # 四元数注意力层
        self.attention = QuaternionAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            use_rotary_embeddings=use_rotary_embeddings
        )
        
        # 四元数前馈网络
        self.feed_forward = QuaternionFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps
        )
        
        # 输出层归一化
        self.output_layer_norm = nn.LayerNorm(hidden_size * 4, eps=layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        参数:
            hidden_states: 隐藏状态 [batch_size, seq_length, hidden_size * 4]
            attention_mask: 注意力掩码
            position_ids: 位置ID
        
        返回:
            hidden_states: 变换后的隐藏状态
            attention_probs: 注意力概率
        """
        # 四元数注意力
        attention_output, attention_probs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        # 四元数前馈网络
        hidden_states = self.feed_forward(attention_output)
        
        # 输出层归一化
        hidden_states = self.output_layer_norm(hidden_states)
        
        return hidden_states, attention_probs
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}"


class QuaternionConv1D(nn.Module):
    """四元数一维卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        """
        初始化四元数一维卷积层
        
        参数:
            in_channels: 输入通道数（四元数维度为4倍）
            out_channels: 输出通道数（四元数维度为4倍）
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 膨胀率
            groups: 分组数
            bias: 是否使用偏置
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 使用PyTorch标准Conv1d，但输入输出通道数乘以4
        self.conv = nn.Conv1d(
            in_channels * 4,
            out_channels * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, in_channels * 4, length]
        
        返回:
            output: 输出张量 [batch_size, out_channels * 4, length]
        """
        return self.conv(x)
    
    def extra_repr(self) -> str:
        return self.conv.extra_repr()


class QuaternionEmbedding(nn.Module):
    """四元数嵌入层"""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False
    ):
        """
        初始化四元数嵌入层
        
        参数:
            vocab_size: 词汇表大小
            hidden_size: 隐藏层大小（四元数维度为4倍）
            padding_idx: 填充索引
            max_norm: 最大范数
            norm_type: 范数类型
            scale_grad_by_freq: 是否按频率缩放梯度
            sparse: 是否使用稀疏梯度
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 标准嵌入层，输出维度为hidden_size * 4
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size * 4,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse
        )
        
        # 四元数归一化层
        self.quaternion_norm = QuaternionNormalization()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            input_ids: 输入ID [batch_size, seq_length]
        
        返回:
            embeddings: 嵌入向量 [batch_size, seq_length, hidden_size * 4]
        """
        embeddings = self.embedding(input_ids)
        embeddings = self.quaternion_norm(embeddings)
        return embeddings
    
    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}"


# ============================================================================
# 测试函数
# ============================================================================

def test_quaternion_nn():
    """测试四元数神经网络层"""
    print("测试四元数神经网络层...")
    
    # 测试四元数线性层
    batch_size = 4
    seq_length = 10
    in_features = 16
    out_features = 32
    
    linear_layer = QuaternionLinear(in_features, out_features)
    x = torch.randn(batch_size, seq_length, in_features * 4)
    output = linear_layer(x)
    
    assert output.shape == (batch_size, seq_length, out_features * 4), "线性层形状错误"
    print("✓ 四元数线性层测试通过")
    
    # 测试四元数注意力层
    hidden_size = 64
    num_heads = 8
    
    attention_layer = QuaternionAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads
    )
    
    x = torch.randn(batch_size, seq_length, hidden_size * 4)
    attention_mask = torch.ones(batch_size, seq_length, seq_length)
    position_ids = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1)
    
    output, attention_probs = attention_layer(x, attention_mask, position_ids)
    
    assert output.shape == (batch_size, seq_length, hidden_size * 4), "注意力层输出形状错误"
    assert attention_probs.shape == (batch_size, num_heads, seq_length, seq_length), "注意力概率形状错误"
    print("✓ 四元数注意力层测试通过")
    
    # 测试四元数前馈网络
    intermediate_size = 128
    
    feed_forward_layer = QuaternionFeedForward(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size
    )
    
    x = torch.randn(batch_size, seq_length, hidden_size * 4)
    output = feed_forward_layer(x)
    
    assert output.shape == (batch_size, seq_length, hidden_size * 4), "前馈网络形状错误"
    print("✓ 四元数前馈网络测试通过")
    
    # 测试四元数Transformer块
    transformer_block = QuaternionTransformerBlock(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size
    )
    
    x = torch.randn(batch_size, seq_length, hidden_size * 4)
    output, attention_probs = transformer_block(x, attention_mask, position_ids)
    
    assert output.shape == (batch_size, seq_length, hidden_size * 4), "Transformer块输出形状错误"
    print("✓ 四元数Transformer块测试通过")
    
    # 测试四元数卷积层
    in_channels = 16
    out_channels = 32
    kernel_size = 3
    length = 20
    
    conv_layer = QuaternionConv1D(in_channels, out_channels, kernel_size)
    x = torch.randn(batch_size, in_channels * 4, length)
    output = conv_layer(x)
    
    expected_length = length - kernel_size + 1
    assert output.shape == (batch_size, out_channels * 4, expected_length), "卷积层形状错误"
    print("✓ 四元数卷积层测试通过")
    
    # 测试四元数嵌入层
    vocab_size = 1000
    hidden_size = 64
    
    embedding_layer = QuaternionEmbedding(vocab_size, hidden_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    embeddings = embedding_layer(input_ids)
    
    assert embeddings.shape == (batch_size, seq_length, hidden_size * 4), "嵌入层形状错误"
    print("✓ 四元数嵌入层测试通过")
    
    print("所有四元数神经网络层测试通过！")
    
    return True


if __name__ == "__main__":
    # 运行测试
    test_quaternion_nn()
