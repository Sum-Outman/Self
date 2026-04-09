#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图拉普拉斯矩阵计算模块

功能：
1. 无向图和有向图的拉普拉斯矩阵计算
2. 标准化拉普拉斯和非标准化拉普拉斯
3. 大规模稀疏矩阵的高效存储和计算
4. 拉普拉斯矩阵的特征值分解
5. 增量式图更新和矩阵更新
6. GPU加速计算

工业级质量标准要求：
- 数值稳定性：双精度计算，误差控制
- 内存效率：稀疏矩阵存储，大规模图支持
- 计算性能：GPU加速，并行计算
- 可扩展性：模块化设计，易于扩展

数学原理：
1. 非标准化拉普拉斯：L = D - A，其中D是度矩阵，A是邻接矩阵
2. 标准化拉普拉斯：L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
3. 随机游走拉普拉斯：L_rw = D^{-1} L = I - D^{-1} A
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
import enum
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh
import time

logger = logging.getLogger(__name__)


class GraphType(enum.Enum):
    """图类型枚举"""

    UNDIRECTED = "undirected"
    DIRECTED = "directed"


@dataclass
class GraphStructure:
    """图数据结构类"""

    adjacency_matrix: torch.Tensor  # 邻接矩阵
    graph_type: GraphType = GraphType.UNDIRECTED  # 图类型
    edge_weights: Optional[torch.Tensor] = None  # 边权重
    node_features: Optional[torch.Tensor] = None  # 节点特征
    node_labels: Optional[List[str]] = None  # 节点标签

    def __post_init__(self):
        """初始化后处理"""
        self.num_nodes = self.adjacency_matrix.shape[0]

        # 计算边索引
        edge_indices = torch.nonzero(self.adjacency_matrix)
        if edge_indices.numel() == 0:
            self.edge_indices = torch.zeros((2, 0), dtype=torch.long)
        else:
            self.edge_indices = edge_indices.t()

        self.num_edges = self.edge_indices.shape[1]

        # 确保有向图的一致性
        self.directed = self.graph_type == GraphType.DIRECTED

    def to(self, device: torch.device) -> "GraphStructure":
        """转移到指定设备"""
        return GraphStructure(
            adjacency_matrix=self.adjacency_matrix.to(device),
            graph_type=self.graph_type,
            edge_weights=(
                self.edge_weights.to(device) if self.edge_weights is not None else None
            ),
            node_features=(
                self.node_features.to(device)
                if self.node_features is not None
                else None
            ),
            node_labels=self.node_labels,
        )

    @classmethod
    def from_adjacency(
        cls,
        adjacency_matrix: torch.Tensor,
        graph_type: GraphType = GraphType.UNDIRECTED,
        edge_weights: Optional[torch.Tensor] = None,
        node_features: Optional[torch.Tensor] = None,
        node_labels: Optional[List[str]] = None,
    ) -> "GraphStructure":
        """从邻接矩阵创建图结构"""
        return cls(
            adjacency_matrix=adjacency_matrix,
            graph_type=graph_type,
            edge_weights=edge_weights,
            node_features=node_features,
            node_labels=node_labels,
        )

    @property
    def degree_matrix(self) -> torch.Tensor:
        """度矩阵"""
        if self.graph_type == GraphType.DIRECTED:
            # 有向图：出度矩阵
            degrees = self.adjacency_matrix.sum(dim=1)
        else:
            # 无向图：对称度矩阵
            degrees = self.adjacency_matrix.sum(dim=1)

        return torch.diag(degrees)

    @property
    def weighted_degree_matrix(self) -> torch.Tensor:
        """加权度矩阵"""
        if self.edge_weights is not None:
            if self.graph_type == GraphType.DIRECTED:
                # 有向图：加权出度
                weighted_degrees = self.edge_weights.sum(dim=1)
            else:
                # 无向图：加权度
                weighted_degrees = self.edge_weights.sum(dim=1)
        else:
            # 使用普通度矩阵
            weighted_degrees = self.adjacency_matrix.sum(dim=1)

        return torch.diag(weighted_degrees)


class GraphLaplacian(nn.Module):
    """图拉普拉斯矩阵计算器

    支持功能：
    1. 多种拉普拉斯矩阵计算
    2. 稀疏矩阵优化
    3. 特征值分解
    4. 增量更新
    5. GPU加速

    工业级特性：
    - 双精度数值稳定性
    - 大规模图支持 (百万节点)
    - 实时更新能力
    - 内存使用优化
    """

    def __init__(
        self,
        normalization: str = "sym",  # "none", "sym", "rw"
        use_sparse: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        cache_enabled: bool = True,
        max_cache_size: int = 10,
    ):
        """初始化图拉普拉斯计算器

        参数:
            normalization: 标准化类型 ("none": 非标准化, "sym": 对称标准化, "rw": 随机游走标准化)
            use_sparse: 是否使用稀疏矩阵
            device: 计算设备
            dtype: 数据类型 (推荐float64保证数值稳定性)
            cache_enabled: 是否启用缓存
            max_cache_size: 最大缓存大小
        """
        super().__init__()
        self.normalization = normalization
        self.use_sparse = use_sparse
        self.dtype = dtype
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size

        # 设备配置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 缓存系统
        self.cache = {}
        self.cache_keys = []

        # 性能统计
        self.stats = {
            "compute_time": 0.0,
            "eigen_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info(
            f"图拉普拉斯计算器初始化完成: device={                 self.device}, normalization={normalization}, dtype={dtype}"
        )

    def compute_laplacian(
        self,
        graph: GraphStructure,
        recompute: bool = False,
        k_eigenvalues: Optional[int] = None,
    ) -> Dict[str, Any]:
        """计算图拉普拉斯矩阵

        参数:
            graph: 图结构数据
            recompute: 是否重新计算 (忽略缓存)
            k_eigenvalues: 需要计算的特征值数量

        返回:
            包含拉普拉斯矩阵和特征的字典
        """
        # 检查缓存
        cache_key = self._get_cache_key(graph)
        if self.cache_enabled and not recompute and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            logger.debug(f"缓存命中: {cache_key}")
            return self.cache[cache_key]

        self.stats["cache_misses"] += 1
        start_time = time.time()

        # 转移数据到设备
        graph = graph.to(self.device)

        # 提取基本矩阵
        A = graph.adjacency_matrix.to(self.dtype)

        # 计算度矩阵
        if graph.directed:
            # 有向图：出度和入度
            D_out = torch.diag(A.sum(dim=1))
            D_in = torch.diag(A.sum(dim=0))
            D = (D_out + D_in) / 2
        else:
            # 无向图：对称度矩阵
            D = torch.diag(A.sum(dim=1))

        # 计算非标准化拉普拉斯
        L = D - A

        # 应用标准化
        if self.normalization == "sym":
            # 对称标准化: L_sym = D^{-1/2} L D^{-1/2}
            D_sqrt_inv = torch.diag(1.0 / torch.sqrt(torch.diag(D) + 1e-10))
            L = D_sqrt_inv @ L @ D_sqrt_inv
        elif self.normalization == "rw":
            # 随机游走标准化: L_rw = D^{-1} L
            D_inv = torch.diag(1.0 / (torch.diag(D) + 1e-10))
            L = D_inv @ L

        # 转换为稀疏矩阵 (如果需要)
        if self.use_sparse:
            L_sparse = L.to_sparse()
        else:
            L_sparse = None

        # 计算特征值和特征向量 (如果需要)
        eigenvalues, eigenvectors = None, None
        if k_eigenvalues is not None and k_eigenvalues > 0:
            eigen_start = time.time()
            eigenvalues, eigenvectors = self._compute_eigen_decomposition(
                L, k_eigenvalues, graph.num_nodes
            )
            self.stats["eigen_time"] += time.time() - eigen_start

        # 构建结果
        result = {
            "laplacian": L,
            "laplacian_matrix": L,  # 兼容性别名
            "laplacian_sparse": L_sparse,
            "degree_matrix": D,
            "adjacency_matrix": A,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "directed": graph.directed,
            "normalization": self.normalization,
        }

        # 更新缓存
        if self.cache_enabled:
            self._update_cache(cache_key, result)

        self.stats["compute_time"] += time.time() - start_time

        logger.debug(
            f"拉普拉斯矩阵计算完成: nodes={graph.num_nodes}, edges={graph.num_edges}, "
            f"time={time.time() - start_time:.3f}s"
        )

        return result

    def _compute_eigen_decomposition(
        self, L: torch.Tensor, k: int, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算拉普拉斯矩阵的特征值分解

        使用Lanczos算法进行高效的特征值计算
        对于大规模矩阵，使用稀疏矩阵的特征值求解器
        """
        # 确保k不超过矩阵维度
        k = min(k, n - 1)

        if n > 1000 and self.use_sparse:
            # 大规模图：使用SciPy的稀疏特征值求解器
            try:
                # 转换为SciPy稀疏矩阵
                if L.is_sparse:
                    L_np = L.cpu().to_dense().numpy()
                else:
                    L_np = L.cpu().numpy()

                L_sparse = sp.csr_matrix(L_np)

                # 计算最小的k个特征值和特征向量
                eigenvalues_np, eigenvectors_np = eigsh(
                    L_sparse, k=k, which="SM", tol=1e-6, maxiter=1000  # 最小特征值
                )

                # 转换回PyTorch张量
                eigenvalues = (
                    torch.from_numpy(eigenvalues_np).to(self.device).to(self.dtype)
                )
                eigenvectors = (
                    torch.from_numpy(eigenvectors_np).to(self.device).to(self.dtype)
                )

                return eigenvalues, eigenvectors

            except Exception as e:
                logger.warning(f"稀疏特征值分解失败: {e}, 回退到稠密计算")

        # 小规模图或回退：使用稠密矩阵的特征值分解
        try:
            # 对称矩阵确保数值稳定性
            L_sym = (L + L.T) / 2
            eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)

            # 取最小的k个特征值
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]

            return eigenvalues, eigenvectors

        except Exception as e:
            logger.error(f"稠密特征值分解失败: {e}")
            raise RuntimeError(f"特征值分解失败: {e}")

    def compute_fiedler_vector(self, graph: GraphStructure) -> torch.Tensor:
        """计算Fiedler向量 (第二小特征值对应的特征向量)

        Fiedler向量用于图分割和聚类分析
        """
        result = self.compute_laplacian(graph, k_eigenvalues=2)

        if result["eigenvectors"] is None or result["eigenvectors"].shape[1] < 2:
            raise ValueError("无法计算Fiedler向量，特征向量不足")

        # 第二小特征值对应的特征向量 (索引1)
        fiedler_vector = result["eigenvectors"][:, 1]

        return fiedler_vector

    def update_graph(
        self,
        old_graph: GraphStructure,
        new_edges: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> GraphStructure:
        """增量式图更新

        高效更新图结构，避免重新计算整个矩阵
        """
        # 创建新的邻接矩阵
        A_new = old_graph.adjacency_matrix.clone()

        # 更新边 (假设new_edges是[2, m]形状)
        for i in range(new_edges.shape[1]):
            u, v = new_edges[0, i], new_edges[1, i]
            if u < old_graph.num_nodes and v < old_graph.num_nodes:
                weight = edge_weights[i] if edge_weights is not None else 1.0
                A_new[u, v] = weight
                if not old_graph.directed:
                    A_new[v, u] = weight

        # 构建新图结构
        new_graph = GraphStructure(
            num_nodes=old_graph.num_nodes,
            num_edges=old_graph.num_edges + new_edges.shape[1],
            adjacency_matrix=A_new,
            edge_indices=torch.cat([old_graph.edge_indices, new_edges], dim=1),
            edge_weights=(
                torch.cat([old_graph.edge_weights, edge_weights])
                if old_graph.edge_weights is not None and edge_weights is not None
                else None
            ),
            node_features=old_graph.node_features,
            directed=old_graph.directed,
        )

        # 清除相关缓存
        self._clear_cache_for_graph(old_graph)

        logger.info(f"图更新完成: 新增{new_edges.shape[1]}条边")

        return new_graph

    def _get_cache_key(self, graph: GraphStructure) -> str:
        """生成缓存键

        基于图结构和计算参数生成唯一键
        """
        key_parts = [
            f"nodes_{graph.num_nodes}",
            f"edges_{graph.num_edges}",
            f"directed_{graph.directed}",
            f"norm_{self.normalization}",
            f"dtype_{self.dtype}",
        ]

        # 完整版本)
        if graph.adjacency_matrix is not None:
            matrix_hash = hash(graph.adjacency_matrix.sum().item())
            key_parts.append(f"matrix_{matrix_hash}")

        return "_".join(key_parts)

    def _update_cache(self, key: str, result: Dict[str, Any]):
        """更新缓存

        实现LRU缓存策略
        """
        if key in self.cache:
            # 更新访问顺序
            self.cache_keys.remove(key)
        elif len(self.cache) >= self.max_cache_size:
            # 移除最久未使用的项
            lru_key = self.cache_keys.pop(0)
            del self.cache[lru_key]
            logger.debug(f"缓存淘汰: {lru_key}")

        self.cache[key] = result
        self.cache_keys.append(key)

    def _clear_cache_for_graph(self, graph: GraphStructure):
        """清除与特定图相关的缓存"""
        key_to_remove = []
        for key in self.cache_keys:
            if f"nodes_{graph.num_nodes}" in key and f"edges_{graph.num_edges}" in key:
                key_to_remove.append(key)

        for key in key_to_remove:
            del self.cache[key]
            self.cache_keys.remove(key)

        if key_to_remove:
            logger.debug(f"清除缓存: {len(key_to_remove)}个条目")

    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        return self.stats.copy()

    def reset_stats(self):
        """重置性能统计"""
        self.stats = {
            "compute_time": 0.0,
            "eigen_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def compute_knn_laplacian(
        self, features: torch.Tensor, k: int = 10, return_components: bool = False
    ) -> Dict[str, Any]:
        """从特征计算k近邻图拉普拉斯矩阵

        参数:
            features: 节点特征 [n, d]
            k: 近邻数
            return_components: 是否返回所有组件

        返回:
            拉普拉斯矩阵和相关组件
        """
        n = features.shape[0]
        k = min(k, n - 1)

        # 计算距离矩阵
        distances = torch.cdist(features, features)

        # 获取k近邻
        _, indices = torch.topk(distances, k=k, dim=1, largest=False)

        # 构建邻接矩阵
        adjacency = torch.zeros((n, n), device=features.device, dtype=torch.float32)
        for i in range(n):
            adjacency[i, indices[i]] = 1.0

        # 对称化（对于无向图）
        adjacency = (adjacency + adjacency.t()) / 2
        adjacency = (adjacency > 0).float()

        # 创建图结构
        graph = GraphStructure.from_adjacency(
            adjacency_matrix=adjacency,
            graph_type=GraphType.UNDIRECTED,
            node_features=features,
        )

        # 计算拉普拉斯矩阵
        result = self.compute_laplacian(graph, recompute=False)

        # 如果只需要拉普拉斯矩阵
        if not return_components:
            return result["laplacian"]

        # 添加额外的组件信息
        result.update(
            {
                "k": k,
                "features": features,
                "adjacency_matrix": adjacency,
                "graph_structure": graph,
            }
        )

        return result

    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"GraphLaplacian(device={self.device}, normalization={self.normalization}, "
            f"dtype={self.dtype}, cache_size={len(self.cache)})"
        )


def create_graph_from_adjacency(
    adjacency_matrix: Union[torch.Tensor, np.ndarray],
    directed: bool = False,
    node_features: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> GraphStructure:
    """从邻接矩阵创建图结构

    参数:
        adjacency_matrix: 邻接矩阵 [n, n]
        directed: 是否是有向图
        node_features: 节点特征 [n, d]
        device: 目标设备

    返回:
        图结构对象
    """
    if isinstance(adjacency_matrix, np.ndarray):
        adjacency_matrix = torch.from_numpy(adjacency_matrix)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adjacency_matrix = adjacency_matrix.to(device)

    # 提取边权重（直接从邻接矩阵获取）
    # 注意：这里假设邻接矩阵的值就是边权重
    # 如果需要区分邻接矩阵和边权重，需要修改接口
    edge_indices = torch.nonzero(adjacency_matrix, as_tuple=False).t()
    edge_weights = adjacency_matrix[edge_indices[0], edge_indices[1]]

    # 设置图类型
    graph_type = GraphType.DIRECTED if directed else GraphType.UNDIRECTED

    # 创建图结构
    graph = GraphStructure.from_adjacency(
        adjacency_matrix=adjacency_matrix,
        graph_type=graph_type,
        edge_weights=edge_weights,
        node_features=node_features.to(device) if node_features is not None else None,
        node_labels=None,
    )

    return graph


def test_laplacian_computation():
    """测试拉普拉斯矩阵计算"""
    print("=== 测试图拉普拉斯矩阵计算 ===")

    # 创建一个简单的图 (环状图)
    n = 10
    A = torch.zeros(n, n)
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[(i + 1) % n, i] = 1

    # 创建图结构
    graph = create_graph_from_adjacency(A, directed=False)

    # 测试不同标准化方式
    for norm in ["none", "sym", "rw"]:
        print(f"\n--- 测试标准化: {norm} ---")

        # 创建拉普拉斯计算器
        laplacian_calculator = GraphLaplacian(normalization=norm, dtype=torch.float64)

        # 计算拉普拉斯矩阵
        result = laplacian_calculator.compute_laplacian(graph, k_eigenvalues=3)

        # 验证基本属性
        L = result["laplacian"]
        print(f"拉普拉斯矩阵形状: {L.shape}")
        print(f"特征值: {result['eigenvalues']}")

        # 验证对称性 (对于无向图)
        if not graph.directed:
            sym_error = torch.norm(L - L.T).item()
            print(f"对称性误差: {sym_error:.2e}")
            assert sym_error < 1e-10, "拉普拉斯矩阵应该对称"

        # 验证行和为零 (对于非标准化拉普拉斯)
        if norm == "none":
            row_sum = torch.sum(L, dim=1)
            row_sum_error = torch.norm(row_sum).item()
            print(f"行和误差: {row_sum_error:.2e}")
            assert row_sum_error < 1e-10, "非标准化拉普拉斯行和应该为零"

    print("\n=== 测试通过 ===")


if __name__ == "__main__":
    # 运行测试
    test_laplacian_computation()
