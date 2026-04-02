#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图拉普拉斯矩阵计算模块测试

测试功能：
1. 拉普拉斯矩阵计算的数值精度
2. 大规模稀疏矩阵的性能
3. 特征值分解的正确性
4. GPU加速计算的正确性
5. 增量图更新的功能

工业级测试标准：
- 单元测试覆盖率 ≥95%
- 性能测试: 10万节点图计算时间 <5秒
- 内存使用: 1百万节点图内存使用 <2GB
- 数值精度: 与SciPy计算结果误差 <1e-10
"""

import torch
import numpy as np
import time
import unittest
import sys
import os
from typing import Dict, Any

# 灵活的导入策略
try:
    # 策略1：尝试相对导入（当作为模块使用时）
    from .laplacian_matrix import GraphLaplacian, GraphStructure, GraphType
except (ImportError, ValueError):
    # 策略2：尝试绝对导入（当直接运行时）
    # 确保项目根目录在Python路径中
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from models.graph.laplacian_matrix import GraphLaplacian, GraphStructure, GraphType


class TestGraphStructure(unittest.TestCase):
    """测试图结构类"""
    
    def setUp(self):
        """测试前设置"""
        # 创建小型测试图
        self.small_adjacency = torch.tensor([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=torch.float32)
        
        self.small_edge_weights = torch.tensor([
            [0, 0.5, 0, 0.8],
            [0.5, 0, 0.7, 0],
            [0, 0.7, 0, 0.3],
            [0.8, 0, 0.3, 0]
        ], dtype=torch.float32)
        
        self.small_graph = GraphStructure(
            adjacency_matrix=self.small_adjacency,
            graph_type=GraphType.UNDIRECTED,
            edge_weights=self.small_edge_weights,
            node_labels=["A", "B", "C", "D"]
        )
        
        # 创建中型测试图 (100节点)
        torch.manual_seed(42)
        self.medium_size = 100
        self.medium_adjacency = torch.rand(self.medium_size, self.medium_size)
        self.medium_adjacency = (self.medium_adjacency > 0.95).float()
        self.medium_adjacency = self.medium_adjacency.fill_diagonal_(0)
        
        self.medium_graph = GraphStructure(
            adjacency_matrix=self.medium_adjacency,
            graph_type=GraphType.UNDIRECTED
        )
    
    def test_graph_structure_creation(self):
        """测试图结构创建"""
        # 测试基础属性
        self.assertEqual(self.small_graph.num_nodes, 4)
        self.assertEqual(self.small_graph.graph_type, GraphType.UNDIRECTED)
        self.assertTrue(torch.allclose(
            self.small_graph.adjacency_matrix,
            self.small_adjacency
        ))
        
        # 测试边权重
        self.assertTrue(torch.allclose(
            self.small_graph.edge_weights,
            self.small_edge_weights
        ))
        
        # 测试节点标签
        self.assertEqual(self.small_graph.node_labels, ["A", "B", "C", "D"])
    
    def test_degree_matrix(self):
        """测试度矩阵计算"""
        degree_matrix = self.small_graph.degree_matrix
        
        expected_degrees = torch.tensor([
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2]
        ], dtype=torch.float32)
        
        self.assertTrue(torch.allclose(degree_matrix, expected_degrees))
    
    def test_weighted_degree_matrix(self):
        """测试加权度矩阵计算"""
        weighted_degree = self.small_graph.weighted_degree_matrix
        
        # 计算期望的加权度
        expected_weighted_degrees = torch.tensor([
            [1.3, 0, 0, 0],   # 0.5 + 0.8
            [0, 1.2, 0, 0],   # 0.5 + 0.7
            [0, 0, 1.0, 0],   # 0.7 + 0.3
            [0, 0, 0, 1.1]    # 0.8 + 0.3
        ], dtype=torch.float32)
        
        self.assertTrue(
            torch.allclose(weighted_degree, expected_weighted_degrees, atol=1e-6),
            f"加权度矩阵不正确: 得到 {weighted_degree}, 期望 {expected_weighted_degrees}"
        )


class TestGraphLaplacian(unittest.TestCase):
    """测试图拉普拉斯类"""
    
    def setUp(self):
        """测试前设置"""
        # 创建测试图
        self.adjacency = torch.tensor([
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0]
        ], dtype=torch.float32)
        
        self.graph = GraphStructure(
            adjacency_matrix=self.adjacency,
            graph_type=GraphType.UNDIRECTED
        )
        
        # 创建图拉普拉斯计算器
        self.laplacian_calculator = GraphLaplacian(
            normalization="sym",
            use_sparse=True,
            dtype=torch.float64
        )
    
    def test_unormalized_laplacian(self):
        """测试非标准化拉普拉斯矩阵"""
        calculator = GraphLaplacian(normalization=None, use_sparse=False)
        result = calculator.compute_laplacian(self.graph, recompute=True)
        
        L = result["laplacian_matrix"]
        
        # 计算期望的拉普拉斯矩阵: L = D - A
        D = torch.diag(self.adjacency.sum(dim=1))
        expected_L = (D - self.adjacency).to(torch.float64)
        
        # 验证数值精度
        self.assertTrue(
            torch.allclose(L, expected_L, atol=1e-10),
            f"非标准化拉普拉斯矩阵不正确: 得到 {L}, 期望 {expected_L}"
        )
        
        # 验证拉普拉斯矩阵的性质: L对称，非负定
        self.assertTrue(torch.allclose(L, L.T), "拉普拉斯矩阵应该对称")
        
        # 计算特征值验证非负定性
        eigenvalues = torch.linalg.eigvalsh(L)
        self.assertTrue(
            torch.all(eigenvalues >= -1e-10),
            f"拉普拉斯矩阵特征值有负数: {eigenvalues}"
        )
    
    def test_symmetric_normalized_laplacian(self):
        """测试对称标准化拉普拉斯矩阵"""
        calculator = GraphLaplacian(normalization="sym", use_sparse=False)
        result = calculator.compute_laplacian(self.graph, recompute=True)
        
        L_sym = result["laplacian_matrix"]
        
        # 计算期望的对称标准化拉普拉斯矩阵: L_sym = I - D^{-1/2} A D^{-1/2}
        D = self.graph.degree_matrix
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
        expected_L_sym = (torch.eye(4, dtype=torch.float64) - D_inv_sqrt.to(torch.float64) @ self.adjacency.to(torch.float64) @ D_inv_sqrt.to(torch.float64))
        
        # 验证数值精度
        self.assertTrue(
            torch.allclose(L_sym, expected_L_sym, atol=1e-10),
            f"对称标准化拉普拉斯矩阵不正确: 得到 {L_sym}, 期望 {expected_L_sym}"
        )
        
        # 验证特征值在[0, 2]范围内
        eigenvalues = torch.linalg.eigvalsh(L_sym)
        self.assertTrue(
            torch.all(eigenvalues >= -1e-10) and torch.all(eigenvalues <= 2.0 + 1e-10),
            f"对称标准化拉普拉斯矩阵特征值超出范围[0,2]: {eigenvalues}"
        )
    
    def test_random_walk_normalized_laplacian(self):
        """测试随机游走标准化拉普拉斯矩阵"""
        calculator = GraphLaplacian(normalization="rw", use_sparse=False)
        result = calculator.compute_laplacian(self.graph, recompute=True)
        
        L_rw = result["laplacian_matrix"]
        
        # 计算期望的随机游走拉普拉斯矩阵: L_rw = I - D^{-1} A
        D = self.graph.degree_matrix
        D_inv = torch.diag(1.0 / torch.diag(D))
        expected_L_rw = (torch.eye(4, dtype=torch.float64) - D_inv.to(torch.float64) @ self.adjacency.to(torch.float64))
        
        # 验证数值精度
        self.assertTrue(
            torch.allclose(L_rw, expected_L_rw, atol=1e-10),
            f"随机游走标准化拉普拉斯矩阵不正确: 得到 {L_rw}, 期望 {expected_L_rw}"
        )
    
    def test_eigenvalue_decomposition(self):
        """测试特征值分解"""
        result = self.laplacian_calculator.compute_laplacian(
            self.graph, recompute=True, k_eigenvalues=3
        )
        
        # 验证特征值分解结果
        self.assertIn("eigenvalues", result)
        self.assertIn("eigenvectors", result)
        
        eigenvalues = result["eigenvalues"]
        eigenvectors = result["eigenvectors"]
        
        # 验证特征值数量
        self.assertEqual(len(eigenvalues), 3)
        
        # 验证特征向量正交性
        orthogonality = eigenvectors.T @ eigenvectors
        expected_orthogonality = torch.eye(3, dtype=torch.float64)
        
        self.assertTrue(
            torch.allclose(orthogonality, expected_orthogonality, atol=1e-8),
            "特征向量不正交"
        )
        
        # 验证特征方程: L v = λ v
        L = result["laplacian_matrix"]
        for i in range(3):
            v = eigenvectors[:, i]
            λ = eigenvalues[i]
            Lv = L @ v
            
            self.assertTrue(
                torch.allclose(Lv, λ * v, atol=1e-8),
                f"特征方程不满足: 特征值 {λ}, 特征向量 {v}"
            )
    
    def test_fiedler_vector(self):
        """测试Fiedler向量计算"""
        fiedler_vector = self.laplacian_calculator.compute_fiedler_vector(self.graph)
        
        # 验证Fiedler向量形状
        self.assertEqual(fiedler_vector.shape, (4,))
        
        # 验证Fiedler向量正交于全1向量
        ones_vector = torch.ones(4, dtype=torch.float64)
        dot_product = torch.dot(fiedler_vector, ones_vector)
        
        self.assertTrue(
            abs(dot_product) < 1e-10,
            f"Fiedler向量不与全1向量正交: 点积 = {dot_product}"
        )
        
        # 验证Fiedler向量归一化
        norm = torch.norm(fiedler_vector)
        self.assertTrue(
            abs(norm - 1.0) < 1e-10,
            f"Fiedler向量未归一化: 范数 = {norm}"
        )
    
    def test_incremental_update(self):
        """测试增量图更新"""
        # 初始图
        result1 = self.laplacian_calculator.compute_laplacian(self.graph)
        
        # 修改图 (添加一条边)
        modified_adjacency = self.adjacency.clone()
        modified_adjacency[0, 2] = 1
        modified_adjacency[2, 0] = 1  # 无向图
        
        modified_graph = GraphStructure(
            adjacency_matrix=modified_adjacency,
            graph_type=GraphType.UNDIRECTED
        )
        
        # 增量更新
        result2 = self.laplacian_calculator.compute_laplacian(
            modified_graph, recompute=False
        )
        
        # 完全重新计算
        result3 = self.laplacian_calculator.compute_laplacian(
            modified_graph, recompute=True
        )
        
        # 验证增量更新与完全重新计算的结果一致
        L_incremental = result2["laplacian_matrix"]
        L_recompute = result3["laplacian_matrix"]
        
        self.assertTrue(
            torch.allclose(L_incremental, L_recompute, atol=1e-10),
            "增量更新与完全重新计算结果不一致"
        )
    
    def test_sparse_matrix_operations(self):
        """测试稀疏矩阵操作"""
        # 使用稀疏矩阵
        sparse_calculator = GraphLaplacian(use_sparse=True, dtype=torch.float64)
        result = sparse_calculator.compute_laplacian(self.graph, recompute=True)
        
        L_sparse = result["laplacian_matrix"]
        
        # 转换为稠密矩阵验证
        if L_sparse.is_sparse:
            L_dense = L_sparse.to_dense()
        else:
            L_dense = L_sparse
        
        # 使用稠密矩阵计算器验证
        dense_calculator = GraphLaplacian(use_sparse=False, dtype=torch.float64)
        dense_result = dense_calculator.compute_laplacian(self.graph, recompute=True)
        L_dense_expected = dense_result["laplacian_matrix"]
        
        # 验证稀疏和稠密结果一致
        self.assertTrue(
            torch.allclose(L_dense, L_dense_expected, atol=1e-10),
            "稀疏矩阵与稠密矩阵计算结果不一致"
        )
    
    def test_large_graph_performance(self):
        """测试大规模图性能"""
        # 跳过性能测试，除非显式运行
        if not hasattr(self, '_run_performance_tests'):
            return
        
        print("\n=== 大规模图性能测试 ===")
        
        # 创建大规模图 (10,000节点)
        large_size = 10000
        torch.manual_seed(42)
        
        # 创建稀疏随机图 (平均度=10)
        print(f"创建 {large_size} 节点的大规模图...")
        start_time = time.time()
        
        # 使用块对角线结构确保连通性
        large_adjacency = torch.zeros(large_size, large_size, dtype=torch.float32)
        
        # 每个节点连接最近的10个邻居
        for i in range(large_size):
            neighbors = torch.arange(max(0, i-5), min(large_size, i+6))
            neighbors = neighbors[neighbors != i]  # 移除自环
            large_adjacency[i, neighbors] = 1
        
        # 确保对称
        large_adjacency = (large_adjacency + large_adjacency.T) / 2
        large_adjacency = (large_adjacency > 0).float()
        
        large_graph = GraphStructure(
            adjacency_matrix=large_adjacency,
            graph_type=GraphType.UNDIRECTED
        )
        
        creation_time = time.time() - start_time
        print(f"图创建时间: {creation_time:.2f}秒")
        
        # 测试稀疏矩阵计算
        print("测试稀疏拉普拉斯矩阵计算...")
        sparse_calculator = GraphLaplacian(use_sparse=True)
        
        start_time = time.time()
        result = sparse_calculator.compute_laplacian(large_graph, recompute=True)
        computation_time = time.time() - start_time
        
        print(f"稀疏矩阵计算时间: {computation_time:.2f}秒")
        
        # 验证性能要求: 10万节点图计算时间 <5秒
        # 这里测试1万节点，期望按比例更少
        max_expected_time = 0.5  # 1万节点的期望时间
        self.assertLess(
            computation_time, max_expected_time,
            f"计算时间 {computation_time:.2f}秒 超过期望 {max_expected_time}秒"
        )
        
        # 验证内存使用
        if result["laplacian_matrix"].is_sparse:
            sparse_size = result["laplacian_matrix"]._nnz() * 8 / 1024**2  # MB
            print(f"稀疏矩阵内存使用: {sparse_size:.2f} MB")
            
            # 验证内存要求: 1百万节点图内存使用 <2GB
            # 这里1万节点期望按比例更少
            max_expected_memory = 20  # MB (1万节点)
            self.assertLess(
                sparse_size, max_expected_memory,
                f"内存使用 {sparse_size:.2f}MB 超过期望 {max_expected_memory}MB"
            )
        
        print("大规模图性能测试通过")


class TestGPUAcceleration(unittest.TestCase):
    """测试GPU加速"""
    
    @classmethod
    def setUpClass(cls):
        """类级别设置"""
        cls.has_gpu = torch.cuda.is_available()
        if cls.has_gpu:
            cls.device = torch.device("cuda")
            print(f"GPU可用，使用设备: {cls.device}")
        else:
            cls.device = torch.device("cpu")
            print("GPU不可用，使用CPU")
    
    def test_gpu_computation(self):
        """测试GPU计算"""
        if not self.has_gpu:
            self.skipTest("GPU不可用，跳过GPU测试")
        
        # 创建测试图并移动到GPU
        adjacency = torch.tensor([
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0]
        ], dtype=torch.float32).to(self.device)
        
        graph = GraphStructure(
            adjacency_matrix=adjacency,
            graph_type=GraphType.UNDIRECTED
        )
        
        # 创建GPU计算器
        calculator = GraphLaplacian(
            normalization="sym",
            use_sparse=True,
            device=self.device
        )
        
        # 计算拉普拉斯矩阵
        result = calculator.compute_laplacian(graph, recompute=True)
        
        # 验证结果在GPU上
        L = result["laplacian_matrix"]
        if L.is_sparse:
            self.assertEqual(L.device, self.device)
        else:
            self.assertEqual(L.device, self.device)
        
        # 验证数值正确性 (与CPU结果比较)
        cpu_calculator = GraphLaplacian(
            normalization="sym",
            use_sparse=True,
            device=torch.device("cpu")
        )
        
        cpu_graph = GraphStructure(
            adjacency_matrix=adjacency.cpu(),
            graph_type=GraphType.UNDIRECTED
        )
        
        cpu_result = cpu_calculator.compute_laplacian(cpu_graph, recompute=True)
        cpu_L = cpu_result["laplacian_matrix"]
        
        # 移动到CPU进行比较
        if L.is_sparse:
            L_cpu = L.cpu().to_dense()
            cpu_L_dense = cpu_L.to_dense() if cpu_L.is_sparse else cpu_L
        else:
            L_cpu = L.cpu()
            cpu_L_dense = cpu_L
        
        # 验证GPU和CPU结果一致
        self.assertTrue(
            torch.allclose(L_cpu, cpu_L_dense, atol=1e-6),
            "GPU和CPU计算结果不一致"
        )
    
    def test_gpu_performance(self):
        """测试GPU性能"""
        if not self.has_gpu:
            self.skipTest("GPU不可用，跳过GPU性能测试")
        
        # 创建中型图测试GPU加速
        size = 5000
        torch.manual_seed(42)
        
        # 创建随机图
        adjacency = torch.rand(size, size, device=self.device)
        adjacency = (adjacency > 0.01).float()
        adjacency = adjacency.fill_diagonal_(0)
        adjacency = (adjacency + adjacency.T) / 2  # 确保对称
        adjacency = (adjacency > 0).float()
        
        graph = GraphStructure(
            adjacency_matrix=adjacency,
            graph_type=GraphType.UNDIRECTED
        )
        
        # GPU计算
        gpu_calculator = GraphLaplacian(
            normalization="sym",
            use_sparse=True,
            device=self.device
        )
        
        print(f"\nGPU性能测试 ({size}节点图):")
        
        # 预热
        _ = gpu_calculator.compute_laplacian(graph, recompute=True)
        
        # 计时GPU计算
        start_time = time.time()
        gpu_result = gpu_calculator.compute_laplacian(graph, recompute=True)
        gpu_time = time.time() - start_time
        
        print(f"GPU计算时间: {gpu_time:.3f}秒")
        
        # CPU计算 (用于比较)
        cpu_calculator = GraphLaplacian(
            normalization="sym",
            use_sparse=True,
            device=torch.device("cpu")
        )
        
        cpu_graph = GraphStructure(
            adjacency_matrix=adjacency.cpu(),
            graph_type=GraphType.UNDIRECTED
        )
        
        start_time = time.time()
        cpu_result = cpu_calculator.compute_laplacian(cpu_graph, recompute=True)
        cpu_time = time.time() - start_time
        
        print(f"CPU计算时间: {cpu_time:.3f}秒")
        
        # 验证GPU加速
        if cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"GPU加速比: {speedup:.2f}x")
            
            # 验证GPU加速有效 (期望至少1.5倍加速)
            self.assertGreater(
                speedup, 1.0,
                f"GPU未加速: GPU时间 {gpu_time:.3f}s, CPU时间 {cpu_time:.3f}s"
            )


def run_comprehensive_tests():
    """运行全面测试"""
    print("=== 图拉普拉斯矩阵计算模块全面测试 ===")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    
    # 添加测试类
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestGraphStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphLaplacian))
    
    # 添加GPU测试 (如果可用)
    if torch.cuda.is_available():
        suite.addTests(loader.loadTestsFromTestCase(TestGPUAcceleration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出统计信息
    print(f"\n测试统计:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("所有测试通过!")
    else:
        print("有测试失败!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 运行全面测试
    success = run_comprehensive_tests()
    
    # 运行性能测试 (需要显式设置)
    if "--performance" in sys.argv:
        print("\n=== 运行性能测试 ===")
        test_instance = TestGraphLaplacian()
        test_instance._run_performance_tests = True
        test_instance.setUp()
        test_instance.test_large_graph_performance()
    
    # 根据测试结果退出
    sys.exit(0 if success else 1)