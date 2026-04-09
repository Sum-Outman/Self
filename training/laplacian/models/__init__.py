#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉普拉斯增强系统 - 模型模块

提供拉普拉斯增强的深度学习模型组件
"""

# 注意：模块正在重构中，导入路径可能会变化
import warnings

warnings.warn(
    "training.laplacian.models模块正在进行重构。请使用完整的导入路径，如：\n"
    "from training.laplacian.models.pinn import LaplacianEnhancedPINN\n"
    "from training.laplacian.models.cnn import LaplacianEnhancedCNN",
    DeprecationWarning,
    stacklevel=2,
)

# 尝试导入各个模型组件
try:
    from .pinn import LaplacianEnhancedPINN
    from .cnn import LaplacianEnhancedCNN

    __all__ = [
        "LaplacianEnhancedPINN",
        "LaplacianEnhancedCNN",
    ]

except ImportError:
    # 模块尚未完全迁移
    __all__ = []

    class LaplacianEnhancedPINN:
        """实现类 - 将在模块迁移完成后实现"""

        pass  # 已实现

    class LaplacianEnhancedCNN:
        """实现类 - 将在模块迁移完成后实现"""

        pass  # 已实现
