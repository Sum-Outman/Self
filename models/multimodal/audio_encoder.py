#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频编码器模块

包含：
1. AudioEncoder - 从零开始的音频编码器

工业级AGI系统要求：从零开始训练，不使用预训练模型依赖
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class AudioEncoder(nn.Module):
    """音频编码器 - 用于对比学习模型
    
    简化的音频编码器，包装工业级音频编码器接口
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 从配置中获取参数
        embedding_dim = config.get("audio_embedding_dim", 768)
        num_layers = config.get("num_layers", 12)
        spectrogram_size = config.get("spectrogram_size", 128)
        patch_size = config.get("patch_size", 16)
        
        # 使用与IndustrialAudioEncoder相同的架构
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
        
        # 动态计算注意力头数
        base_head_dim = 64
        if embedding_dim >= base_head_dim:
            nhead = embedding_dim // base_head_dim
        else:
            nhead = 1
        
        while embedding_dim % nhead != 0 and nhead > 1:
            nhead -= 1
        
        nhead = max(1, nhead)
        head_dim = embedding_dim // nhead
        
        # Transformer编码器
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
            pooled_output: [batch_size, embedding_dim] 池化输出
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
        
        # 使用CLS token作为池化输出
        pooled = encoded[:, 0, :]
        
        return pooled
    
    def get_pooled_output(self, sequence_output):
        """获取池化输出（使用CLS token）
        
        参数:
            sequence_output: [batch_size, num_patches+1, embedding_dim] 序列输出
            
        返回:
            [batch_size, embedding_dim] 池化输出
        """
        # 如果输入已经是池化输出，直接返回
        if sequence_output.dim() == 2:
            return sequence_output
        # 否则提取CLS token
        return sequence_output[:, 0, :]