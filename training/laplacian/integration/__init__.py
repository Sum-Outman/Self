#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉普拉斯增强系统 - 集成模块

提供与主Self AGI框架的集成接口和向后兼容性
"""

from .framework import (
    LaplacianIntegrationConfig,
    LaplacianIntegrationFramework,
    LAPLACIAN_MODULES_AVAILABLE,
    integrate_laplacian_with_training,
)

__all__ = [
    "LaplacianIntegrationConfig",
    "LaplacianIntegrationFramework",
    "LAPLACIAN_MODULES_AVAILABLE",
    "integrate_laplacian_with_training",
]
