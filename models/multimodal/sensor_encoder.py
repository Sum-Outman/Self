#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器编码器模块

包含：
1. IndustrialSensorEncoder - 从零开始的传感器编码器

工业级AGI系统要求：从零开始训练，不使用预训练模型依赖
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class IndustrialSensorEncoder(nn.Module):
    """工业级传感器编码器 - 基于1D Transformer架构
    
    支持多传感器时间序列输入：
    - IMU数据（加速度计、陀螺仪、磁力计）
    - 关节位置传感器
    - 力传感器
    - 温度传感器
    - 其他物理传感器
    
    输入形状: [batch_size, num_channels, sequence_length]
    输出形状: [batch_size, embedding_dim]
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 从配置中获取参数
        embedding_dim = config.get("sensor_embedding_dim", 256)
        num_layers = config.get("num_layers", 6)
        sequence_length = config.get("sensor_sequence_length", 100)
        patch_size = config.get("sensor_patch_size", 10)
        num_channels = config.get("sensor_num_channels", 9)  # 默认9通道: 3轴加速度+3轴陀螺+3轴磁力
        
        # 传感器参数
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.num_channels = num_channels
        
        # 计算块数量
        self.num_patches = (sequence_length // patch_size)
        
        # 1D卷积进行时间块嵌入
        self.patch_embeddings = nn.Conv1d(
            num_channels, embedding_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 位置嵌入
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        # Transformer编码器
        # 动态计算注意力头数，确保embedding_dim能被nhead整除
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
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # 初始化参数
        self._init_weights()
        
        logger.info(f"初始化IndustrialSensorEncoder: 嵌入维度={embedding_dim}, 层数={num_layers}")
        logger.info(f"输入: {num_channels}通道, {sequence_length}时间步, {self.num_patches}个时间块")
    
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
    
    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            sensor_data: 传感器数据张量 [batch_size, num_channels, sequence_length]
            
        返回:
            pooled_output: [batch_size, embedding_dim] 池化输出
        """
        batch_size = sensor_data.size(0)
        
        # 时间块嵌入 [batch, channels, seq_len] -> [batch, embedding_dim, num_patches]
        x = self.patch_embeddings(sensor_data)
        
        # 转置为序列 [batch, embedding_dim, num_patches] -> [batch, num_patches, embedding_dim]
        x = x.transpose(1, 2)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch, num_patches+1, embedding_dim]
        
        # 添加位置嵌入
        x = x + self.position_embeddings
        
        # Transformer编码
        encoded = self.transformer(x)
        
        # 层归一化
        encoded = self.layer_norm(encoded)
        
        # 使用CLS token作为池化输出
        pooled = encoded[:, 0, :]
        
        return pooled
    
    def get_sequence_output(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """获取完整的序列输出（包括所有时间块）"""
        batch_size = sensor_data.size(0)
        
        # 时间块嵌入
        x = self.patch_embeddings(sensor_data)
        x = x.transpose(1, 2)
        
        # 添加CLS token和位置嵌入
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        
        # Transformer编码
        encoded = self.transformer(x)
        encoded = self.layer_norm(encoded)
        
        return encoded  # [batch_size, num_patches+1, embedding_dim]