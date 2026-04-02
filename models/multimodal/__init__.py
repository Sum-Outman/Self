#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态处理器模块 - 工业级AGI系统多模态融合

模块化架构：
1. dataclasses - 基础数据结构
2. text_encoder - 文本编码器
3. vision_encoder - 视觉编码器
4. audio_encoder - 音频编码器
5. sensor_encoder - 传感器编码器
6. fusion_networks - 融合网络
7. contrastive_learning - 对比学习模型
8. temporal_processor - 时序处理器
9. tokenizer - 文本分词器
10. processor - 主处理器
11. attention_analyzer - 注意力分析器

工业级AGI系统要求：从零开始训练，不使用预训练模型依赖
"""

# 导出所有模块的类
from .custom_dataclasses import MultimodalInput, ProcessedModality
from .text_encoder import IndustrialTextEncoder
from .vision_encoder import IndustrialVisionEncoder
from .cnn_enhancement import CNNConfig, ResNetEncoder, HybridVisionEncoder, CNNArchitectureSearch, \
    ConvBlock, ResidualBlock, CBAM, SEBlock, ECABlock, FeaturePyramidNetwork, EnhancedVisionEncoder
from .audio_encoder import AudioEncoder
from .industrial_audio_encoder import IndustrialAudioEncoder
from .sensor_encoder import IndustrialSensorEncoder
from .fusion_networks import CrossModalAttention, ProjectionLayerManager, HierarchicalFusionNetwork
from .contrastive_learning import ContrastiveAlignmentModel
from .temporal_processor import TemporalMultimodalProcessor
from .tokenizer import IndustrialTokenizer
from .processor import MultimodalProcessor
from .attention_analyzer import AttentionAnalyzer, AttentionAnalysisResult

# 别名：为了向后兼容，提供MultimodalFusionNetwork作为HierarchicalFusionNetwork的别名
MultimodalFusionNetwork = HierarchicalFusionNetwork

# 主处理器仍然保留
__all__ = [
    # 基础数据结构
    "MultimodalInput",
    "ProcessedModality",
    
    # 编码器
    "IndustrialTextEncoder",
    "IndustrialVisionEncoder",
    "AudioEncoder",
    "IndustrialAudioEncoder",
    "IndustrialSensorEncoder",
    
    # CNN增强
    "CNNConfig",
    "ResNetEncoder",
    "HybridVisionEncoder",
    "CNNArchitectureSearch",
    "ConvBlock",
    "ResidualBlock",
    "CBAM",
    "SEBlock",
    "ECABlock",
    "FeaturePyramidNetwork",
    "EnhancedVisionEncoder",
    
    # 融合网络
    "CrossModalAttention",
    "ProjectionLayerManager",
    "HierarchicalFusionNetwork",
    "MultimodalFusionNetwork",  # 添加别名
    
    # 对比学习
    "ContrastiveAlignmentModel",
    
    # 时序处理
    "TemporalMultimodalProcessor",
    
    # 分词器
    "IndustrialTokenizer",
    
    # 主处理器
    "MultimodalProcessor",
    
    # 注意力分析器
    "AttentionAnalyzer",
    "AttentionAnalysisResult",
]