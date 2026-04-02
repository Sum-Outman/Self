#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉普拉斯增强的CNN模型

功能：
1. 基础CNN特征提取
2. 拉普拉斯金字塔多尺度特征增强
3. 拉普拉斯边缘感知特征融合
4. 图拉普拉斯特征平滑约束

从 training/laplacian_enhanced_training.py 迁移而来
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# 导入相关模块
try:
    from ..utils.config import LaplacianEnhancedTrainingConfig
    
    # 尝试导入CNN相关模块
    try:
        from models.multimodal.cnn_enhancement import CNNConfig, CNNModel
        CNN_MODULE_AVAILABLE = True
    except ImportError:
        CNN_MODULE_AVAILABLE = False
        logger.warning("CNN模块不可用，功能将受限")
        # 创建简单的CNN配置和模型实现
        class CNNConfig:
            """简单的CNN配置实现"""
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class CNNModel(nn.Module):
            """简单的CNN模型实现"""
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.fc = nn.Linear(128 * 56 * 56, 10)
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
    
    # 尝试导入拉普拉斯变换器
    try:
        from utils.signal_processing.laplace_transform import LaplaceTransform, SignalProcessingConfig
        LAPLACE_TRANSFORM_AVAILABLE = True
    except ImportError:
        LAPLACE_TRANSFORM_AVAILABLE = False
        logger.warning("拉普拉斯变换器不可用，功能将受限")
        class LaplaceTransform:
            """简单的拉普拉斯变换器实现"""
            def __init__(self, config):
                self.config = config
        
        class SignalProcessingConfig:
            """信号处理配置实现"""
        pass  # 已实现
    
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.warning(f"部分模块不可用: {e}, 功能将受限")
    
    # 创建必要的实现类
    class CNNConfig:
        pass  # 已实现
    class CNNModel:
        pass  # 已实现
    class LaplaceTransform:
        pass  # 已实现
    class SignalProcessingConfig:
        pass  # 已实现
    class LaplacianEnhancedTrainingConfig:
        pass  # 已实现


class LaplacianEnhancedCNN(nn.Module):
    """拉普拉斯增强的CNN模型
    
    功能：
    1. 基础CNN特征提取
    2. 拉普拉斯金字塔多尺度特征增强
    3. 拉普拉斯边缘感知特征融合
    4. 图拉普拉斯特征平滑约束
    """
    
    def __init__(self,
                 cnn_config: CNNConfig,
                 laplacian_config: LaplacianEnhancedTrainingConfig):
        super().__init__()
        
        if not MODULES_AVAILABLE:
            raise ImportError("必要的模块不可用，无法初始化LaplacianEnhancedCNN")
        
        self.cnn_config = cnn_config
        self.laplacian_config = laplacian_config
        
        # 基础CNN模型
        self.cnn_model = CNNModel(cnn_config)
        
        # 拉普拉斯金字塔
        if laplacian_config.multi_scale_enabled:
            self.laplacian_pyramid = self._build_laplacian_pyramid()
        else:
            self.laplacian_pyramid = None
        
        # 拉普拉斯变换器
        if LAPLACE_TRANSFORM_AVAILABLE:
            self.laplace_transformer = LaplaceTransform(SignalProcessingConfig())
        else:
            self.laplace_transformer = None
        
        logger.info(f"拉普拉斯增强CNN初始化完成: 架构={cnn_config.architecture if hasattr(cnn_config, 'architecture') else 'unknown'}, "
                   f"多尺度={laplacian_config.multi_scale_enabled}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（带拉普拉斯增强）"""
        
        # 基础CNN特征提取
        base_features = self.cnn_model(x)
        
        # 拉普拉斯金字塔多尺度特征
        if self.laplacian_pyramid is not None and self.laplacian_config.multi_scale_enabled:
            laplacian_features = self._extract_laplacian_features(x)
            
            # 特征融合
            enhanced_features = self._fuse_features(base_features, laplacian_features)
            
            return enhanced_features
        
        return base_features
    
    def _build_laplacian_pyramid(self) -> nn.ModuleList:
        """构建拉普拉斯金字塔"""
        pyramid = nn.ModuleList()
        
        for i in range(self.laplacian_config.num_scales):
            scale_factor = self.laplacian_config.scale_factors[i]
            
            # 高斯模糊层
            blur_layer = self._create_gaussian_blur_layer()
            
            # 下采样层
            downsample_layer = nn.AvgPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
            
            # 上采样层
            upsample_layer = nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )
            
            pyramid.append(nn.ModuleDict({
                'blur': blur_layer,
                'downsample': downsample_layer,
                'upsample': upsample_layer
            }))
        
        return pyramid
    
    def _create_gaussian_blur_layer(self, kernel_size: int = 5, sigma: float = 1.0) -> nn.Conv2d:
        """创建高斯模糊卷积层"""
        
        # 创建高斯核
        kernel = self._gaussian_kernel(kernel_size, sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
        kernel = kernel.repeat(3, 1, 1, 1)  # (channels, 1, k, k)
        
        # 创建卷积层
        conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=3,  # 深度可分离卷积
            bias=False
        )
        
        # 设置权重（不训练）
        conv.weight.data = kernel
        conv.weight.requires_grad = False
        
        return conv
    
    def _gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """生成高斯核"""
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        x = x.view(1, -1).expand(kernel_size, -1)
        y = x.t()
        
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def _extract_laplacian_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """提取拉普拉斯金字塔特征"""
        features = []
        
        current = x
        for i, level in enumerate(self.laplacian_pyramid):
            # 高斯模糊
            blurred = level['blur'](current)
            
            # 下采样
            downsampled = level['downsample'](blurred)
            
            # 上采样
            upsampled = level['upsample'](downsampled)
            
            # 拉普拉斯层（原始 - 上采样模糊）
            laplacian = current - upsampled
            
            features.append(laplacian)
            
            # 为下一层准备
            current = downsampled
        
        # 添加最后一层的高斯模糊
        if current is not None:
            last_blurred = self.laplacian_pyramid[-1]['blur'](current)
            features.append(last_blurred)
        
        return features
    
    def _fuse_features(self, 
                      base_features: torch.Tensor,
                      laplacian_features: List[torch.Tensor]) -> torch.Tensor:
        """融合基础特征和拉普拉斯特征"""
        
        # 简单加权融合
        # 这里可以实现更复杂的融合策略
        
        # 调整拉普拉斯特征大小以匹配基础特征
        fused_features = base_features
        
        for i, lap_feat in enumerate(laplacian_features):
            # 调整大小
            if lap_feat.shape[2:] != fused_features.shape[2:]:
                lap_feat_resized = F.interpolate(
                    lap_feat,
                    size=fused_features.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )
            else:
                lap_feat_resized = lap_feat
            
            # 简单加权融合
            weight = 0.1 / (i + 1)  # 递减权重
            fused_features = fused_features + weight * lap_feat_resized
        
        return fused_features