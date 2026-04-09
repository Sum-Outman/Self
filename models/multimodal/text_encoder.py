#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业级文本编码器模块

包含：
1. IndustrialTextEncoder - 从零开始的文本Transformer编码器

工业级AGI系统要求：从零开始训练，不使用预训练模型依赖
"""

import torch
import torch.nn as nn


class IndustrialTextEncoder(nn.Module):
    """工业级从零开始的文本编码器

    特征：
    - 完整的Transformer架构，支持2048序列长度
    - 词嵌入 + 位置嵌入 + 段落嵌入
    - 12层Transformer编码器
    - 支持注意力掩码和token类型
    """

    def __init__(
        self,
        vocab_size=100000,
        embedding_dim=768,
        num_layers=12,
        max_position_embeddings=2048,
    ):
        super().__init__()
        # 词嵌入 + 位置嵌入 + 段落嵌入
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_dim)
        self.segment_embeddings = nn.Embedding(2, embedding_dim)  # 支持2种段落类型

        # Transformer编码器层
        # 动态计算注意力头数，确保embedding_dim能被nhead整除
        # 标准Transformer设置：每个注意力头维度为64
        base_head_dim = 64
        if embedding_dim >= base_head_dim:
            nhead = embedding_dim // base_head_dim
        else:
            nhead = 1

        # 确保nhead能整除embedding_dim，否则调整nhead
        while embedding_dim % nhead != 0 and nhead > 1:
            nhead -= 1

        # 确保nhead至少为1
        nhead = max(1, nhead)

        # 计算每个头的维度
        embedding_dim // nhead

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=3072,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

        # 层归一化
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重 - 从零开始"""
        # 使用Xavier初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _prepare_attention_mask(self, attention_mask):
        """准备注意力掩码"""
        # 将0转换为True（需要mask的位置），1转换为False
        return attention_mask == 0

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        前向传播

        参数:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1表示有效token，0表示padding)
            token_type_ids: [batch_size, seq_len] (段落类型，0或1)

        返回:
            [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = input_ids.shape

        # 词嵌入
        token_embeds = self.token_embeddings(input_ids)  # [batch, seq, dim]

        # 添加段落嵌入
        if token_type_ids is not None:
            token_embeds = token_embeds + self.segment_embeddings(token_type_ids)

        # 添加位置信息
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = token_embeds + position_embeds

        # 准备注意力掩码
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = self._prepare_attention_mask(attention_mask)

        # Transformer编码
        encoded = self.transformer(embeddings, src_key_padding_mask=key_padding_mask)

        # 层归一化
        encoded = self.layer_norm(encoded)

        return encoded  # [batch_size, seq_len, embedding_dim]

    def get_pooled_output(self, sequence_output, attention_mask=None):
        """获取池化输出（使用[CLS] token或平均池化）"""
        if attention_mask is not None:
            # 使用平均池化（排除padding token）
            mask = attention_mask.unsqueeze(-1).float()
            masked_output = sequence_output * mask
            sum_output = masked_output.sum(dim=1)
            mask_sum = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_output / mask_sum
        else:
            # 使用[CLS] token（第一个token）
            pooled = sequence_output[:, 0, :]

        return pooled
