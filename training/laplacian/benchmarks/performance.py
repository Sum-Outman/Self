#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉普拉斯增强系统性能基准测试（完整版本）

从 training/laplacian_benchmark.py 迁移而来，更新了导入路径
"""

import torch
import time
import gc
import psutil
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# 尝试导入拉普拉斯模块
try:
    from ..core.regularization import LaplacianRegularization, RegularizationConfig
    from ..models.pinn import LaplacianEnhancedPINN
    from ..models.cnn import LaplacianEnhancedCNN
    from ..utils.config import LaplacianEnhancedTrainingConfig

    # 尝试导入图模块
    try:
        from models.graph.laplacian_matrix import GraphLaplacian, GraphStructure

        GRAPH_MODULE_AVAILABLE = True
    except ImportError:
        GRAPH_MODULE_AVAILABLE = False
        logger.warning("图模块不可用，基准测试功能将受限")

    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.warning(f"拉普拉斯模块导入失败: {e}, 基准测试功能将受限")


# 内存测量工具
def get_memory_usage() -> Tuple[float, float]:
    """获取当前内存使用情况（MB）"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024, memory_info.vms / 1024 / 1024
    except Exception:
        return 0.0, 0.0


def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class BenchmarkResult:
    """基准测试结果"""

    module_name: str
    test_name: str
    avg_time_ms: float
    std_time_ms: float
    memory_usage_mb: float
    throughput: float  # 样本/秒
    success: bool
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module_name,
            "test": self.test_name,
            "avg_time_ms": self.avg_time_ms,
            "std_time_ms": self.std_time_ms,
            "memory_mb": self.memory_usage_mb,
            "throughput": self.throughput,
            "success": self.success,
            "error": self.error_message,
        }


class LaplacianBenchmark:
    """拉普拉斯增强系统性能基准测试（完整版本）"""

    def __init__(self, device: str = "cpu", num_iterations: int = 10):
        self.device = device
        self.num_iterations = num_iterations
        self.results: List[BenchmarkResult] = []

        # 设置设备
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA不可用，使用CPU")
            self.device = "cpu"

        self.device_torch = torch.device(self.device)
        logger.info(
            f"基准测试初始化: device={self.device}, iterations={num_iterations}"
        )

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """运行所有基准测试"""

        logger.info("开始运行拉普拉斯增强系统性能基准测试")

        try:
            # 1. 基础性能测试
            self._benchmark_basic_operations()

            # 2. 拉普拉斯矩阵计算测试（如果可用）
            if GRAPH_MODULE_AVAILABLE:
                self._benchmark_laplacian_matrix()

            # 3. 拉普拉斯正则化测试
            if MODULES_AVAILABLE:
                self._benchmark_laplacian_regularization()

            # 4. 拉普拉斯增强模型测试（如果可用）
            if MODULES_AVAILABLE:
                self._benchmark_laplacian_models()

        except Exception as e:
            logger.error(f"基准测试运行失败: {e}")

        logger.info(f"基准测试完成，共 {len(self.results)} 个测试结果")
        return self.results

    def _benchmark_basic_operations(self):
        """基础操作性能测试"""

        logger.info("运行基础操作性能测试")

        # 测试矩阵乘法
        sizes = [100, 500, 1000]
        for size in sizes:
            self._run_single_benchmark(
                module_name="basic_ops",
                test_name=f"matmul_{size}x{size}",
                test_func=self._test_matmul,
                test_args=(size, size),
            )

    def _benchmark_laplacian_matrix(self):
        """拉普拉斯矩阵计算性能测试"""

        logger.info("运行拉普拉斯矩阵计算性能测试")

        if not GRAPH_MODULE_AVAILABLE:
            logger.warning("图模块不可用，跳过拉普拉斯矩阵测试")
            return

        # 测试不同大小的图
        graph_sizes = [100, 500, 1000]
        for size in graph_sizes:
            self._run_single_benchmark(
                module_name="laplacian_matrix",
                test_name=f"graph_{size}_nodes",
                test_func=self._test_graph_laplacian,
                test_args=(size,),
            )

    def _benchmark_laplacian_regularization(self):
        """拉普拉斯正则化性能测试"""

        logger.info("运行拉普拉斯正则化性能测试")

        if not MODULES_AVAILABLE:
            logger.warning("拉普拉斯模块不可用，跳过正则化测试")
            return

        # 测试不同大小的特征矩阵
        feature_sizes = [(100, 50), (500, 100), (1000, 200)]
        for n_samples, n_features in feature_sizes:
            self._run_single_benchmark(
                module_name="laplacian_regularization",
                test_name=f"features_{n_samples}x{n_features}",
                test_func=self._test_laplacian_regularization,
                test_args=(n_samples, n_features),
            )

    def _benchmark_laplacian_models(self):
        """拉普拉斯增强模型性能测试"""

        logger.info("运行拉普拉斯增强模型性能测试")

        # 测试PINN模型（如果可用）
        try:
            self._run_single_benchmark(
                module_name="laplacian_models",
                test_name="enhanced_pinn",
                test_func=self._test_enhanced_pinn,
                test_args=(),
            )
        except Exception as e:
            logger.warning(f"拉普拉斯增强PINN测试失败: {e}")

        # 测试CNN模型（如果可用）
        try:
            self._run_single_benchmark(
                module_name="laplacian_models",
                test_name="enhanced_cnn",
                test_func=self._test_enhanced_cnn,
                test_args=(),
            )
        except Exception as e:
            logger.warning(f"拉普拉斯增强CNN测试失败: {e}")

    def _run_single_benchmark(
        self, module_name: str, test_name: str, test_func: callable, test_args: tuple
    ) -> None:
        """运行单个基准测试"""

        logger.debug(f"运行基准测试: {module_name}.{test_name}")

        try:
            # 预热
            test_func(*test_args)

            # 测量内存使用前
            mem_before, _ = get_memory_usage()

            # 多次运行取平均值
            times = []
            for i in range(self.num_iterations):
                clear_memory()

                start_time = time.perf_counter()
                result = test_func(*test_args)
                end_time = time.perf_counter()

                elapsed_ms = (end_time - start_time) * 1000
                times.append(elapsed_ms)

            # 测量内存使用后
            mem_after, _ = get_memory_usage()
            memory_used = max(0, mem_after - mem_before)

            # 计算统计
            avg_time = np.mean(times)
            std_time = np.std(times)

            # 计算吞吐量（样本/秒）
            throughput = 0
            if avg_time > 0:
                # 尝试从测试函数获取样本数
                throughput = 1000 / avg_time  # 默认假设处理一个样本

            # 创建结果
            result = BenchmarkResult(
                module_name=module_name,
                test_name=test_name,
                avg_time_ms=avg_time,
                std_time_ms=std_time,
                memory_usage_mb=memory_used,
                throughput=throughput,
                success=True,
            )

            self.results.append(result)
            logger.info(
                f"测试完成: {test_name}, 平均时间: {avg_time:.2f}ms, 内存: {memory_used:.1f}MB"
            )

        except Exception as e:
            logger.error(f"测试失败: {test_name}, 错误: {e}")

            result = BenchmarkResult(
                module_name=module_name,
                test_name=test_name,
                avg_time_ms=0,
                std_time_ms=0,
                memory_usage_mb=0,
                throughput=0,
                success=False,
                error_message=str(e),
            )

            self.results.append(result)

    # 测试函数实现
    def _test_matmul(self, m: int, n: int) -> torch.Tensor:
        """测试矩阵乘法"""
        A = torch.randn(m, n, device=self.device_torch)
        B = torch.randn(n, m, device=self.device_torch)
        return A @ B

    def _test_graph_laplacian(self, n_nodes: int) -> Optional[Any]:
        """测试图拉普拉斯计算"""
        if not GRAPH_MODULE_AVAILABLE:
            return None  # 返回None

        # 创建随机邻接矩阵
        adjacency = torch.rand(n_nodes, n_nodes, device=self.device_torch)
        adjacency = (adjacency > 0.5).float()

        # 创建图结构
        graph = GraphStructure(adjacency_matrix=adjacency)

        # 创建拉普拉斯计算器
        laplacian_calculator = GraphLaplacian(
            normalization="sym", device=self.device_torch
        )

        # 计算拉普拉斯矩阵
        result = laplacian_calculator.compute_laplacian(graph)

        return result

    def _test_laplacian_regularization(
        self, n_samples: int, n_features: int
    ) -> Optional[torch.Tensor]:
        """测试拉普拉斯正则化"""
        if not MODULES_AVAILABLE:
            return None  # 返回None

        # 创建随机特征
        features = torch.randn(n_samples, n_features, device=self.device_torch)

        # 创建随机邻接矩阵
        adjacency = torch.rand(n_samples, n_samples, device=self.device_torch)
        adjacency = (adjacency > 0.5).float()

        # 创建正则化器
        config = RegularizationConfig(
            regularization_type="graph_laplacian", lambda_reg=0.01, normalization="sym"
        )

        regularizer = LaplacianRegularization(config, feature_dim=n_features)

        # 计算正则化损失
        reg_loss = regularizer(features, adjacency_matrix=adjacency)

        return reg_loss

    def _test_enhanced_pinn(self) -> Optional[Any]:
        """测试拉普拉斯增强PINN"""
        if not MODULES_AVAILABLE:
            return None  # 返回None

        # 完整的PINN配置
        class SimplePINNConfig:
            input_dim = 2
            output_dim = 1
            hidden_dim = 32
            num_layers = 3

        # 创建拉普拉斯配置
        laplacian_config = LaplacianEnhancedTrainingConfig(
            laplacian_reg_enabled=True, laplacian_reg_lambda=0.01, adaptive_lambda=False
        )

        # 创建模型
        model = LaplacianEnhancedPINN(SimplePINNConfig(), laplacian_config)
        model.to(self.device_torch)

        # 测试前向传播
        test_input = torch.randn(100, 2, device=self.device_torch)
        output = model(test_input)

        return output

    def _test_enhanced_cnn(self) -> Optional[Any]:
        """测试拉普拉斯增强CNN"""
        if not MODULES_AVAILABLE:
            return None  # 返回None

        # 完整的CNN配置
        class SimpleCNNConfig:
            architecture = "simple"
            input_channels = 3

        # 创建拉普拉斯配置
        laplacian_config = LaplacianEnhancedTrainingConfig(
            multi_scale_enabled=True, num_scales=2
        )

        # 创建模型
        model = LaplacianEnhancedCNN(SimpleCNNConfig(), laplacian_config)
        model.to(self.device_torch)

        # 测试前向传播
        test_input = torch.randn(4, 3, 64, 64, device=self.device_torch)
        output = model(test_input)

        return output

    def generate_report(self) -> Dict[str, Any]:
        """生成性能报告"""

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device,
            "num_iterations": self.num_iterations,
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.success),
            "failed_tests": sum(1 for r in self.results if not r.success),
            "results": [r.to_dict() for r in self.results],
            "summary": {},
        }

        # 计算总体统计
        if self.results:
            avg_times = [r.avg_time_ms for r in self.results if r.success]
            report["summary"]["avg_time_all_tests_ms"] = (
                np.mean(avg_times) if avg_times else 0
            )
            report["summary"]["total_memory_used_mb"] = sum(
                r.memory_usage_mb for r in self.results
            )

        return report
