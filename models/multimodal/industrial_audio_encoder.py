#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业级音频编码器模块

包含：
1. IndustrialAudioEncoder - 从零开始的工业级音频编码器

工业级AGI系统要求：从零开始训练，不使用预训练模型依赖
"""

import torch
import torch.nn as nn


class IndustrialAudioEncoder(nn.Module):
    """工业级从零开始的音频编码器 - Spectrogram Transformer
    
    特征：
    - Spectrogram Transformer架构
    - 频谱图块嵌入（16x16 patches）
    - 位置嵌入 + CLS token
    - 12层Transformer编码器
    """
    def __init__(self, spectrogram_size=128, patch_size=16, embedding_dim=768, num_layers=12):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (spectrogram_size // patch_size) ** 2
        
        # 频谱图块嵌入（单通道输入）
        self.patch_embeddings = nn.Conv2d(
            1, embedding_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # 位置嵌入
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        # Transformer编码器
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
        head_dim = embedding_dim // nhead
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重 - 从零开始"""
        # 初始化卷积层
        nn.init.xavier_uniform_(self.patch_embeddings.weight)
        if self.patch_embeddings.bias is not None:
            nn.init.zeros_(self.patch_embeddings.bias)
            
        # 初始化位置嵌入
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        
        # 初始化Transformer层
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.patch_embeddings:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, spectrograms):
        """
        前向传播
        
        参数:
            spectrograms: [batch_size, 1, spectrogram_size, spectrogram_size]
            
        返回:
            [batch_size, num_patches+1, embedding_dim]
        """
        batch_size = spectrograms.size(0)
        
        # 频谱图块嵌入 [batch, 1, 128, 128] -> [batch, embedding_dim, 8, 8]
        x = self.patch_embeddings(spectrograms)
        
        # 展平为序列 [batch, embedding_dim, 8*8] -> [batch, 64, embedding_dim]
        x = x.flatten(2).transpose(1, 2)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch, 65, embedding_dim]
        
        # 添加位置嵌入
        x = x + self.position_embeddings
        
        # Transformer编码
        encoded = self.transformer(x)
        
        # 层归一化
        encoded = self.layer_norm(encoded)
        
        return encoded  # [batch_size, 65, embedding_dim]
        
    def get_pooled_output(self, sequence_output):
        """获取池化输出（使用CLS token）"""
        # 使用CLS token（第一个token）
        pooled = sequence_output[:, 0, :]
        return pooled