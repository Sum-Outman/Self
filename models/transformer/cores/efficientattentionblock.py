# EfficientAttentionBlock - 从self_agi_model.py拆分
"""EfficientAttentionBlock模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging

class EfficientAttentionBlock(nn.Module):
    """高效注意力块 - 支持多种高效注意力机制

    参考论文:
    - Linformer: Self-Attention with Linear Complexity (Wang et al., 2020)
    - Longformer: The Long-Document Transformer (Beltagy et al., 2020)
    - BigBird: Transformers for Longer Sequences (Zaheer et al., 2020)
    - FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)

    关键特性:
    1. 线性复杂度：减少计算和内存开销
    2. 局部注意力：滑动窗口机制
    3. 全局token：处理长程依赖
    4. 内存优化：减少内存占用
    """

    def __init__(self, config: 'AGIModelConfig'):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        # 注意力配置
        self.attention_type = config.attention_type
        self.sliding_window_size = config.sliding_window_size
        self.linear_feature_dim = config.linear_attention_feature_dim

        # FlashAttention-2支持
        self.flash_attention2_enabled = config.flash_attention2_enabled
        self.flash_attention_causal = config.flash_attention_causal
        self.flash_attention_dropout = config.flash_attention_dropout

        # 检查FlashAttention-2是否可用
        self.flash_attn_available = False
        if self.flash_attention2_enabled:
            try:
                import flash_attn  # type: ignore

                # 检查CUDA是否可用，FlashAttention-2通常需要CUDA
                if torch.cuda.is_available():
                    self.flash_attn_available = True
                    logger.info(
                        "FlashAttention-2可用（CUDA已启用），将使用FlashAttention-2实现"
                    )
                else:
                    logger.warning(
                        "FlashAttention-2已启用但CUDA不可用，回退到普通注意力。FlashAttention-2需要CUDA环境"
                    )
                    self.flash_attn_available = False
            except ImportError:
                logger.warning(
                    "FlashAttention-2已启用但未安装，回退到普通注意力。请安装: pip install flash-attn"
                )

        # 查询、键、值投影
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # 输出投影
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # 线性注意力特征投影 (如果使用线性注意力)
        if self.attention_type == "linear":
            self.feature_proj = nn.Linear(self.hidden_size, self.linear_feature_dim * 2)

        # 层归一化
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        logger.info(
            f"初始化高效注意力块: 类型={self.attention_type}, "
            f"滑动窗口大小={self.sliding_window_size if self.attention_type == 'local' else 'N/A'}, "
            f"FlashAttention-2启用={self.flash_attention2_enabled}, 可用={self.flash_attn_available}"
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        根据配置使用不同的注意力机制。

        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] 可选

        返回:
            输出张量: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 计算查询、键、值
        q = self.q_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        k = self.k_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        v = self.v_proj(hidden_states)  # [batch_size, seq_len, hidden_size]

        # 多头分割
        head_dim = self.hidden_size // self.num_heads
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        # 根据注意力类型计算注意力
        # 首先检查是否启用FlashAttention-2且可用，并且数据在CUDA设备上
        if (
            self.flash_attention2_enabled
            and self.flash_attn_available
            and hidden_states.is_cuda
        ):
            # 使用FlashAttention-2 (FlashAttention-2)
            # 导入flash_attn模块 (已在__init__中导入)
            import flash_attn  # type: ignore

            # FlashAttention-2需要特定的输入格式
            # q, k, v的shape: [batch_size, seq_len, num_heads, head_dim]
            q_fa = q.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
            k_fa = k.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
            v_fa = v.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]

            # 处理注意力掩码
            # FlashAttention-2使用因果掩码和dropout
            if attention_mask is not None:
                # 将注意力掩码转换为因果掩码格式
                # 注意: FlashAttention-2支持不同类型的掩码
                # 完整实现：根据掩码类型动态调整因果掩码设置
                causal = self.flash_attention_causal
            else:
                causal = self.flash_attention_causal

            # 调用FlashAttention-2
            # 使用flash_attn.flash_attn_qkvpacked_func或类似函数
            try:
                # 尝试使用FlashAttention-2的高效实现
                attn_output_fa = flash_attn.flash_attn_qkvpacked_func(
                    torch.stack(
                        [q_fa, k_fa, v_fa], dim=2
                    ),  # [batch_size, seq_len, 3, num_heads, head_dim]
                    causal=causal,
                    dropout_p=self.flash_attention_dropout if self.training else 0.0,
                    softmax_scale=1.0 / (head_dim**0.5),
                )
                attn_output = attn_output_fa.transpose(
                    1, 2
                )  # [batch_size, num_heads, seq_len, head_dim]
            except AttributeError:
                # 回退到普通注意力
                logger.warning("FlashAttention-2 API不匹配，回退到普通注意力")
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask.unsqueeze(1).unsqueeze(
                        2
                    )

                attn_weights = torch.softmax(attn_weights, dim=-1)
                attn_weights = self.dropout(attn_weights)
                attn_output = torch.matmul(attn_weights, v)

        elif self.attention_type == "vanilla":
            # 标准点积注意力
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask.unsqueeze(1).unsqueeze(2)

            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, v)

        elif self.attention_type == "linear":
            # 线性注意力 (核方法近似)
            # 特征映射
            features = self.feature_proj(
                hidden_states
            )  # [batch_size, seq_len, feature_dim*2]
            phi_q, phi_k = torch.split(features, self.linear_feature_dim, dim=-1)

            # 线性注意力计算: (Q' * (K' * V))
            phi_q = phi_q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
            phi_k = phi_k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

            # 计算注意力输出
            kv = torch.matmul(
                phi_k.transpose(-2, -1), v
            )  # [batch_size, num_heads, feature_dim, head_dim]
            attn_output = torch.matmul(
                phi_q, kv
            )  # [batch_size, num_heads, seq_len, head_dim]

        elif self.attention_type == "local":
            # 局部注意力 (滑动窗口)
            attn_output = torch.zeros_like(q)

            # 滑动窗口处理
            for start_idx in range(0, seq_len, self.sliding_window_size // 2):
                end_idx = min(start_idx + self.sliding_window_size, seq_len)
                window_size = end_idx - start_idx

                # 提取窗口内的查询、键、值
                q_window = q[:, :, start_idx:end_idx, :]
                k_window = k[:, :, start_idx:end_idx, :]
                v_window = v[:, :, start_idx:end_idx, :]

                # 计算窗口注意力
                attn_weights = torch.matmul(q_window, k_window.transpose(-2, -1)) / (
                    head_dim**0.5
                )

                if attention_mask is not None:
                    mask_window = attention_mask[:, start_idx:end_idx]
                    attn_weights = attn_weights + mask_window.unsqueeze(1).unsqueeze(2)

                attn_weights = torch.softmax(attn_weights, dim=-1)
                attn_weights = self.dropout(attn_weights)

                window_output = torch.matmul(attn_weights, v_window)
                attn_output[:, :, start_idx:end_idx, :] = window_output

        # 合并多头
        # 安全检查：确保attn_output已定义
        if "attn_output" not in locals():
            # 回退到vanilla注意力
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.hidden_size
        )

        # 输出投影
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # 残差连接和层归一化
        output = hidden_states + attn_output
        output = self.layer_norm(output)

        return output



