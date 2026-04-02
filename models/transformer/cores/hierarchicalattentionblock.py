# HierarchicalAttentionBlock - 从self_agi_model.py拆分
"""HierarchicalAttentionBlock模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging

class HierarchicalAttentionBlock(nn.Module):
    """层次化注意力块 - 实现文档级、段落级、句子级多层次注意力

    参考审核报告中的上下文压缩技术:
    - 层次化注意力: 文档级 → 段落级 → 句子级
    - 自适应压缩率: 基于重要性动态调整注意力粒度
    - 减少计算复杂度: 仅在必要时进行细粒度计算

    关键特性:
    1. 三级层次结构: 文档、段落、句子
    2. 自适应粒度: 基于内容重要性动态选择注意力级别
    3. 跨层次交互: 不同层次间的信息流动
    4. 选择性计算: 仅对重要内容进行细粒度处理
    """

    def __init__(self, config: 'AGIModelConfig'):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        # 层次化配置
        self.hierarchical_levels = config.hierarchical_levels
        self.importance_threshold = config.importance_threshold

        # 不同层次的注意力模块
        self.document_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        self.paragraph_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        self.sentence_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        # 重要性预测器 - 预测每个token的重要性得分
        self.importance_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid(),  # 输出[0, 1]的重要性得分
        )

        # 层次选择器 - 基于重要性选择适当的注意力层次
        self.level_selector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.hierarchical_levels),
            nn.Softmax(dim=-1),
        )

        # 层次融合门 - 控制不同层次输出的融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, 3),  # 三个层次的融合权重
            nn.Softmax(dim=-1),
        )

        # 输出投影和归一化
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        logger.info(
            f"初始化层次化注意力块: 层次数={self.hierarchical_levels}, "
            f"重要性阈值={self.importance_threshold}"
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播 - 多层次注意力计算

        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] 可选

        返回:
            输出张量: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. 计算重要性得分
        importance_scores = self.importance_predictor(
            hidden_states
        )  # [batch_size, seq_len, 1]
        importance_scores = importance_scores.squeeze(-1)  # [batch_size, seq_len]

        # 2. 基于重要性确定注意力层次
        # 高重要性token使用句子级细粒度注意力
        # 中等重要性token使用段落级注意力
        # 低重要性token使用文档级粗粒度注意力
        high_importance_mask = importance_scores > self.importance_threshold
        medium_importance_mask = (
            importance_scores > self.importance_threshold * 0.5
        ) & ~high_importance_mask
        low_importance_mask = ~(high_importance_mask | medium_importance_mask)

        # 3. 准备不同层次的输入
        # 文档级输入 (对所有token)
        doc_input = hidden_states

        # 段落级输入 (根据序列位置分组)
        # 假设每16个token为一个段落
        paragraph_size = 16
        num_paragraphs = (seq_len + paragraph_size - 1) // paragraph_size

        # 创建段落表示 (平均池化)
        paragraph_states = []
        paragraph_masks = []

        for i in range(num_paragraphs):
            start = i * paragraph_size
            end = min(start + paragraph_size, seq_len)
            paragraph = hidden_states[:, start:end, :]
            paragraph_avg = paragraph.mean(
                dim=1, keepdim=True
            )  # [batch_size, 1, hidden_size]
            paragraph_states.append(paragraph_avg)

            # 创建段落掩码
            if attention_mask is not None:
                para_mask = attention_mask[:, start:end].any(
                    dim=1, keepdim=True
                )  # [batch_size, 1]
                paragraph_masks.append(para_mask)

        paragraph_states = torch.cat(
            paragraph_states, dim=1
        )  # [batch_size, num_paragraphs, hidden_size]

        if attention_mask is not None and paragraph_masks:
            paragraph_mask = torch.cat(
                paragraph_masks, dim=1
            )  # [batch_size, num_paragraphs]
        else:
            paragraph_mask = None

        # 句子级输入 (原始token级)
        sentence_states = hidden_states

        # 4. 计算不同层次的注意力
        # 文档级注意力 (粗粒度)
        doc_output, _ = self.document_attention(
            doc_input, doc_input, doc_input, key_padding_mask=attention_mask
        )

        # 段落级注意力 (中粒度)
        para_output, _ = self.paragraph_attention(
            paragraph_states,
            paragraph_states,
            paragraph_states,
            key_padding_mask=paragraph_mask,
        )

        # 将段落输出扩展回token级
        para_output_expanded = []
        for i in range(num_paragraphs):
            start = i * paragraph_size
            end = min(start + paragraph_size, seq_len)
            para_len = end - start
            # 重复段落表示到每个token
            para_token = para_output[:, i : i + 1, :].expand(
                batch_size, para_len, self.hidden_size
            )
            para_output_expanded.append(para_token)

        para_output_token = torch.cat(para_output_expanded, dim=1)

        # 句子级注意力 (细粒度，仅对高重要性token)
        # 创建句子级注意力掩码 (仅对高重要性token应用)
        sentence_mask = attention_mask.clone() if attention_mask is not None else None
        if sentence_mask is not None:
            # 对低重要性token添加极大负值，使其在softmax中被忽略
            sentence_mask = sentence_mask.float()
            sentence_mask[~high_importance_mask] = -1e9

        sentence_output, _ = self.sentence_attention(
            sentence_states,
            sentence_states,
            sentence_states,
            key_padding_mask=sentence_mask,
        )

        # 5. 层次融合
        # 基于重要性动态融合不同层次的输出
        fusion_input = torch.cat(
            [doc_output, para_output_token, sentence_output], dim=-1
        )
        fusion_weights = self.fusion_gate(fusion_input)  # [batch_size, seq_len, 3]

        # 应用融合权重
        fused_output = (
            fusion_weights[:, :, 0:1] * doc_output
            + fusion_weights[:, :, 1:2] * para_output_token
            + fusion_weights[:, :, 2:3] * sentence_output
        )

        # 6. 输出处理
        output = self.output_proj(fused_output)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = hidden_states + output
        output = self.layer_norm(output)

        # 返回结果和重要性得分（用于调试）
        if self.training:
            return output, {"importance_scores": importance_scores}
        else:
            return output



