#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉普拉斯增强系统 - 基准测试模块

提供性能基准测试和评估工具
"""

from .performance import LaplacianBenchmark, BenchmarkResult

__all__ = [
    "LaplacianBenchmark",
    "BenchmarkResult",
]
