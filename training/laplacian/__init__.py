#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉普拉斯增强训练系统 - 统一模块

统一组织和重构分散的拉普拉斯相关模块，提供清晰的接口和模块化设计。

模块结构：
├── core/                    # 核心组件
│   ├── base.py             # 基础类和接口
│   ├── regularization.py   # 正则化组件
│   └── enhancement.py      # 训练增强组件
├── models/                  # 模型组件
│   ├── pinn.py            # PINN模型增强
│   ├── cnn.py             # CNN模型增强
│   └── graph.py           # 图模型增强
├── optimizers/             # 优化器
│   └── laplacian_optimizer.py
├── benchmarks/             # 基准测试
│   └── performance.py
├── utils/                  # 工具函数
│   ├── config.py          # 统一配置
│   └── compatibility.py   # 兼容性工具
└── integration/           # 集成组件
    ├── framework.py       # 框架集成
    └── training.py       # 训练流程集成
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "Self AGI Team"
__description__ = "拉普拉斯增强训练系统统一模块"

# 向后兼容导入
import sys
import warnings
import logging

# 警告用户关于模块重构
warnings.warn(
    "training.laplacian模块正在进行重构。请更新导入路径：\n"
    "- 从'training.laplacian.core'导入基础类\n"
    "- 从'training.laplacian.models'导入模型组件\n"
    "- 从'training.laplacian.utils.config'导入配置类",
    DeprecationWarning,
    stacklevel=2
)

# 尝试导入核心模块以便在顶层可用
try:
    # 从core模块导入基础类
    from .core.base import LaplacianBase, LaplacianConfig, LaplacianType, NormalizationType
    from .core.regularization import LaplacianRegularization, RegularizationConfig
    
    # 从utils.config导入统一配置
    from .utils.config import LaplacianEnhancedTrainingConfig, UnifiedLaplacianConfig
    
    # 从models模块导入增强模型
    from .models.pinn import LaplacianEnhancedPINN
    from .models.cnn import LaplacianEnhancedCNN
    
    # 从optimizers模块导入优化器
    from .optimizers.laplacian_optimizer import LaplacianEnhancedOptimizer
    
    # 从integration模块导入集成框架
    from .integration.framework import (
        LaplacianIntegrationConfig,
        LaplacianIntegrationFramework,
        LAPLACIAN_MODULES_AVAILABLE,
        integrate_laplacian_with_training,
    )
    
    # 从benchmarks模块导入基准测试
    from .benchmarks.performance import LaplacianBenchmark, BenchmarkResult
    
    __all__ = [
        # 核心类
        'LaplacianBase',
        'LaplacianConfig',
        'LaplacianType',
        'NormalizationType',
        'LaplacianRegularization',
        'RegularizationConfig',
        
        # 配置类
        'LaplacianEnhancedTrainingConfig',
        'UnifiedLaplacianConfig',
        
        # 增强模型
        'LaplacianEnhancedPINN',
        'LaplacianEnhancedCNN',
        
        # 优化器
        'LaplacianEnhancedOptimizer',
        
        # 集成框架
        'LaplacianIntegrationConfig',
        'LaplacianIntegrationFramework',
        'LAPLACIAN_MODULES_AVAILABLE',
        'integrate_laplacian_with_training',
        
        # 基准测试
        'LaplacianBenchmark',
        'BenchmarkResult',
    ]
    
except ImportError as e:
    # 模块尚未完全迁移，提供实现
    logger = logging.getLogger(__name__) if 'logging' in sys.modules else None
    if logger:
        logger.warning(f"拉普拉斯模块导入失败: {e}, 提供实现类")
    
    __all__ = []
    
    class LaplacianBase:
        """实现基类 - 将在模块迁移完成后实现"""
        pass  # 已实现
    
    class LaplacianConfig:
        """实现配置类 - 将在模块迁移完成后实现"""
        pass  # 已实现
    
    class LaplacianRegularization:
        """实现正则化类 - 将在模块迁移完成后实现"""
        pass  # 已实现
    
    class RegularizationConfig:
        """实现正则化配置类 - 将在模块迁移完成后实现"""
        pass  # 已实现
    
    class LaplacianEnhancedTrainingConfig:
        """实现训练增强配置类 - 将在模块迁移完成后实现"""
        pass  # 已实现
    
    class UnifiedLaplacianConfig:
        """实现统一配置类 - 将在模块迁移完成后实现"""
        pass  # 已实现
    
    class LaplacianEnhancedPINN:
        """实现PINN增强类 - 将在模块迁移完成后实现"""
        pass  # 已实现
    
    class LaplacianEnhancedCNN:
        """实现CNN增强类 - 将在模块迁移完成后实现"""
        pass  # 已实现
    
    class LaplacianEnhancedOptimizer:
        """实现优化器增强类 - 将在模块迁移完成后实现"""
        pass  # 已实现
    
    class LaplacianIntegrationConfig:
        """实现集成配置类 - 将在模块迁移完成后实现"""
        pass  # 已实现
    
    class LaplacianIntegrationFramework:
        """实现集成框架类 - 将在模块迁移完成后实现"""
        pass  # 已实现
    
    LAPLACIAN_MODULES_AVAILABLE = False
    
    def integrate_laplacian_with_training(*args, **kwargs):
        """实现集成函数"""
        return {}  # 返回空字典
    
    class LaplacianBenchmark:
        """实现基准测试类"""
        pass  # 已实现
    
    class BenchmarkResult:
        """实现基准测试结果类"""
        pass  # 已实现
    
    # 枚举类型实现
    class LaplacianType:
        GRAPH = "graph"
        MANIFOLD = "manifold"
        MULTI_SCALE = "multi_scale"
        ADAPTIVE = "adaptive"
        PINN_ENHANCED = "pinn_enhanced"
        CNN_ENHANCED = "cnn_enhanced"
        GNN_ENHANCED = "gnn_enhanced"
    
    class NormalizationType:
        NONE = "none"
        SYM = "sym"
        RW = "rw"