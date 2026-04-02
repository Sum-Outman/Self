#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉普拉斯正则化组件

提供多种拉普拉斯正则化方法，包括图拉普拉斯正则化、流形正则化、
多尺度正则化和自适应正则化。

基于拉普拉斯增强系统的新架构重新实现，继承自LaplacianBase基类。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import warnings
import math
from collections import deque
import time

from .base import LaplacianBase, LaplacianConfig, LaplacianType, NormalizationType

logger = logging.getLogger(__name__)

# 尝试导入图模块
try:
    from models.graph.laplacian_matrix import GraphLaplacian, GraphStructure, create_graph_from_adjacency
    GRAPH_MODULE_AVAILABLE = True
except ImportError as e:
    GRAPH_MODULE_AVAILABLE = False
    logger.warning(f"图模块不可用: {e}, 部分正则化功能将受限")


@dataclass
class RegularizationConfig(LaplacianConfig):
    """正则化配置类，继承自LaplacianConfig"""
    
    # 正则化特定配置
    regularization_type: str = "graph_laplacian"  # 正则化类型
    
    # 图配置
    graph_sparsity: float = 0.1  # 图稀疏度
    k_neighbors: int = 10  # k近邻图中的邻居数
    
    # 流形正则化配置
    use_labeled_data: bool = True  # 是否使用标记数据
    use_unlabeled_data: bool = True  # 是否使用未标记数据
    manifold_dimension: int = 2  # 流形维度估计
    
    # 多尺度配置
    num_scales: int = 3  # 多尺度数量
    scale_factors: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])  # 尺度因子
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保laplacian_type正确设置
        if self.laplacian_type == LaplacianType.GRAPH:
            # 根据regularization_type设置合适的laplacian_type
            if self.regularization_type == "manifold":
                self.laplacian_type = LaplacianType.MANIFOLD
            elif self.regularization_type == "multi_scale":
                self.laplacian_type = LaplacianType.MULTI_SCALE
            elif self.regularization_type == "adaptive":
                self.laplacian_type = LaplacianType.ADAPTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        base_dict = super().to_dict()
        base_dict.update({
            "regularization_type": self.regularization_type,
            "graph_sparsity": self.graph_sparsity,
            "k_neighbors": self.k_neighbors,
            "use_labeled_data": self.use_labeled_data,
            "use_unlabeled_data": self.use_unlabeled_data,
            "manifold_dimension": self.manifold_dimension,
            "num_scales": self.num_scales,
            "scale_factors": self.scale_factors,
        })
        return base_dict


class LaplacianRegularization(LaplacianBase):
    """拉普拉斯正则化组件
    
    继承自LaplacianBase，提供多种正则化方法：
    1. 图拉普拉斯正则化
    2. 流形正则化（半监督学习）
    3. 多尺度正则化
    4. 自适应正则化
    
    支持稀疏矩阵优化、GPU加速和缓存机制。
    """
    
    def __init__(
        self,
        config: RegularizationConfig,
        feature_dim: Optional[int] = None,
        num_samples: Optional[int] = None,
        name: Optional[str] = None
    ):
        """初始化拉普拉斯正则化
        
        参数:
            config: 正则化配置
            feature_dim: 特征维度 (可选)
            num_samples: 样本数量 (可选)
            name: 组件名称
        """
        # 调用父类初始化
        super().__init__(config, feature_dim, num_samples, name)
        
        # 初始化图拉普拉斯计算器（如果可用）
        self.laplacian_calculator = None
        if GRAPH_MODULE_AVAILABLE:
            try:
                self.laplacian_calculator = GraphLaplacian(
                    normalization=config.normalization.value if isinstance(config.normalization, NormalizationType) else config.normalization,
                    use_sparse=config.use_sparse,
                    device=self.device,
                    dtype=torch.float64,  # 双精度保证数值稳定性
                    cache_enabled=config.cache_enabled
                )
            except Exception as e:
                logger.warning(f"创建图拉普拉斯计算器失败: {e}")
                self.laplacian_calculator = None
        
        # 额外的正则化特定统计
        self._regularization_stats = {
            "graph_construction_time": 0.0,
            "laplacian_computation_time": 0.0,
            "reg_loss_computation_time": 0.0,
        }
        
        logger.info(f"拉普拉斯正则化初始化: type={config.regularization_type}, lambda={config.lambda_reg}")
    
    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        adjacency_matrix: Optional[torch.Tensor] = None,
        graph_structure: Optional[GraphStructure] = None,
        **kwargs
    ) -> torch.Tensor:
        """前向传播：计算正则化损失
        
        参数:
            features: 特征矩阵 [batch_size, feature_dim] 或 [num_samples, feature_dim]
            labels: 标签向量 [batch_size] 或 [num_samples] (用于半监督学习)
            adjacency_matrix: 邻接矩阵 [num_samples, num_samples] (可选)
            graph_structure: 图结构对象 (可选)
            **kwargs: 附加参数
            
        返回:
            正则化损失标量
        """
        start_time = time.time()
        
        # 验证输入特征
        features = self._validate_features(features)
        
        # 根据配置选择正则化方法
        reg_config = self.config  # 类型: RegularizationConfig
        
        if reg_config.regularization_type == "graph_laplacian":
            reg_loss = self._graph_laplacian_regularization(
                features, adjacency_matrix, graph_structure
            )
        elif reg_config.regularization_type == "manifold":
            reg_loss = self._manifold_regularization(
                features, labels, adjacency_matrix, graph_structure
            )
        elif reg_config.regularization_type == "multi_scale":
            reg_loss = self._multi_scale_regularization(
                features, adjacency_matrix, graph_structure
            )
        elif reg_config.regularization_type == "adaptive":
            reg_loss = self._adaptive_regularization(
                features, labels, adjacency_matrix, graph_structure
            )
        else:
            raise ValueError(f"未知的正则化类型: {reg_config.regularization_type}")
        
        # 应用自适应调整
        if reg_config.adaptive_enabled:
            reg_loss = self._apply_adaptive_adjustment(reg_loss, features)
        
        # 更新统计信息
        compute_time = time.time() - start_time
        self._update_stats(compute_time, reg_loss.item())
        
        # 更新正则化特定统计
        self._regularization_stats["reg_loss_computation_time"] += compute_time
        
        return reg_loss
    
    def compute_regularization(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算正则化损失 - 实现基类接口
        
        参数:
            features: 特征张量
            **kwargs: 附加参数，如labels, adjacency_matrix等
            
        返回:
            正则化损失标量
        """
        return self.forward(features, **kwargs)
    
    def _graph_laplacian_regularization(
        self,
        features: torch.Tensor,
        adjacency_matrix: Optional[torch.Tensor],
        graph_structure: Optional[GraphStructure]
    ) -> torch.Tensor:
        """图拉普拉斯正则化
        
        R(f) = ∑_{i,j} A_{ij} ‖f_i - f_j‖² = f^T L f
        
        其中L是图拉普拉斯矩阵，A是邻接矩阵
        """
        n = features.shape[0]
        
        # 获取或计算图结构
        graph_start_time = time.time()
        if graph_structure is not None:
            graph = graph_structure
        elif adjacency_matrix is not None:
            if GRAPH_MODULE_AVAILABLE:
                graph = create_graph_from_adjacency(adjacency_matrix, device=self.device)
            else:
                raise RuntimeError("图模块不可用，无法从邻接矩阵创建图")
        else:
            # 自动构建k近邻图
            graph = self._build_knn_graph(features)
        self._regularization_stats["graph_construction_time"] += time.time() - graph_start_time
        
        if self.laplacian_calculator is None:
            raise RuntimeError("图拉普拉斯计算器不可用")
        
        # 计算拉普拉斯矩阵
        laplacian_start_time = time.time()
        result = self.laplacian_calculator.compute_laplacian(graph)
        L = result["laplacian"]
        self._regularization_stats["laplacian_computation_time"] += time.time() - laplacian_start_time
        
        # 计算正则化损失: f^T L f
        # 使用双精度计算保证数值稳定性
        features_double = features.to(torch.float64)
        L_double = L.to(torch.float64)
        
        # 高效计算: f^T L f = ∑_i ∑_j L_{ij} f_i·f_j
        # 使用矩阵乘法优化
        reg_loss = torch.trace(features_double.T @ L_double @ features_double)
        
        # 归一化到每个样本
        reg_loss = reg_loss / n if n > 0 else reg_loss
        
        # 应用正则化强度
        reg_loss = self.current_lambda * reg_loss
        
        return reg_loss.to(features.dtype)
    
    def _manifold_regularization(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor],
        adjacency_matrix: Optional[torch.Tensor],
        graph_structure: Optional[GraphStructure]
    ) -> torch.Tensor:
        """流形正则化
        
        结合标记数据和未标记数据的图结构正则化
        R(f) = λ_l ‖f_l - y_l‖² + λ_u f^T L f
        
        其中f_l是标记数据的预测，y_l是真实标签
        """
        n = features.shape[0]
        
        if labels is None:
            logger.warning("流形正则化需要标签信息，回退到图拉普拉斯正则化")
            return self._graph_laplacian_regularization(
                features, adjacency_matrix, graph_structure
            )
        
        # 分离标记和未标记数据
        labeled_mask = ~torch.isnan(labels)
        unlabeled_mask = torch.isnan(labels)
        
        n_labeled = labeled_mask.sum().item()
        n_unlabeled = unlabeled_mask.sum().item()
        
        if n_labeled == 0:
            logger.warning("没有标记数据，回退到图拉普拉斯正则化")
            return self._graph_laplacian_regularization(
                features, adjacency_matrix, graph_structure
            )
        
        # 标记数据部分: 监督损失
        supervised_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_labeled_data and n_labeled > 0:
            labeled_features = features[labeled_mask]
            labeled_labels = labels[labeled_mask]
            
            # 计算监督损失 (假设回归问题)
            supervised_loss = F.mse_loss(labeled_features, labeled_labels.unsqueeze(1))
        
        # 未标记数据部分: 图拉普拉斯正则化
        graph_reg_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_unlabeled_data and n_unlabeled > 0:
            unlabeled_features = features[unlabeled_mask]
            
            # 构建子图 (只包含未标记数据)
            if graph_structure is not None:
                # 从完整图中提取未标记数据的子图
                subgraph = self._extract_subgraph(graph_structure, unlabeled_mask)
                graph_reg_loss = self._graph_laplacian_regularization(
                    unlabeled_features, None, subgraph
                )
            elif adjacency_matrix is not None:
                # 提取未标记数据的邻接子矩阵
                sub_adj = adjacency_matrix[unlabeled_mask][:, unlabeled_mask]
                graph_reg_loss = self._graph_laplacian_regularization(
                    unlabeled_features, sub_adj, None
                )
            else:
                # 为未标记数据构建新图
                graph_reg_loss = self._graph_laplacian_regularization(
                    unlabeled_features, None, None
                )
        
        # 组合损失
        # 平衡监督损失和图正则化损失
        lambda_supervised = 1.0
        lambda_graph = self.current_lambda
        
        total_loss = lambda_supervised * supervised_loss + lambda_graph * graph_reg_loss
        
        return total_loss
    
    def _multi_scale_regularization(
        self,
        features: torch.Tensor,
        adjacency_matrix: Optional[torch.Tensor],
        graph_structure: Optional[GraphStructure]
    ) -> torch.Tensor:
        """多尺度拉普拉斯正则化
        
        在不同图尺度下应用正则化，捕获多尺度结构信息
        R(f) = ∑_{s=1}^S λ_s f^T L_s f
        
        其中L_s是第s个尺度的拉普拉斯矩阵
        """
        n = features.shape[0]
        num_scales = min(self.config.num_scales, len(self.config.scale_factors))
        
        if num_scales == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        for scale_idx in range(num_scales):
            scale_factor = self.config.scale_factors[scale_idx]
            
            # 构建当前尺度的图
            if graph_structure is not None:
                # 对图进行尺度变换
                scaled_graph = self._scale_graph(graph_structure, scale_factor)
            elif adjacency_matrix is not None:
                # 对邻接矩阵进行尺度变换
                scaled_adj = self._scale_adjacency(adjacency_matrix, scale_factor)
                scaled_graph = create_graph_from_adjacency(scaled_adj, device=self.device)
            else:
                # 从特征构建当前尺度的图
                scaled_features = self._scale_features(features, scale_factor)
                scaled_graph = self._build_knn_graph(scaled_features)
            
            # 计算当前尺度的正则化损失
            scale_loss = self._graph_laplacian_regularization(
                features, None, scaled_graph
            )
            
            # 尺度特定的正则化强度
            scale_lambda = self.current_lambda * (1.0 / (scale_idx + 1))
            
            total_loss = total_loss + scale_lambda * scale_loss
        
        # 平均到尺度数量
        total_loss = total_loss / num_scales
        
        return total_loss
    
    def _adaptive_regularization(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor],
        adjacency_matrix: Optional[torch.Tensor],
        graph_structure: Optional[GraphStructure]
    ) -> torch.Tensor:
        """自适应正则化
        
        根据训练状态动态调整正则化强度
        """
        # 计算基础正则化损失
        if self.config.use_labeled_data and labels is not None:
            base_loss = self._manifold_regularization(
                features, labels, adjacency_matrix, graph_structure
            )
        else:
            base_loss = self._graph_laplacian_regularization(
                features, adjacency_matrix, graph_structure
            )
        
        # 基于特征复杂度自适应调整
        feature_complexity = self._compute_feature_complexity(features)
        
        # 调整正则化强度: 特征越复杂，正则化越强
        adaptive_lambda = self.current_lambda * (1.0 + feature_complexity)
        
        # 限制在合理范围内
        adaptive_lambda = torch.clamp(
            adaptive_lambda,
            self.config.min_lambda,
            self.config.max_lambda
        )
        
        # 应用调整后的正则化强度
        adaptive_loss = (adaptive_lambda / self.current_lambda) * base_loss
        
        # 记录调整历史
        self._adaptation_history.append({
            "feature_complexity": feature_complexity.item(),
            "adaptive_lambda": adaptive_lambda.item(),
            "loss": base_loss.item()
        })
        
        return adaptive_loss
    
    def _apply_adaptive_adjustment(
        self,
        reg_loss: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """应用自适应调整
        
        根据训练进度和特征特性动态调整正则化
        """
        # 计算梯度统计信息
        if features.requires_grad:
            # 估计梯度大小
            grad_norm = self._estimate_gradient_norm(features)
            
            # 根据梯度大小调整正则化
            if grad_norm > 1.0:
                # 梯度爆炸，增加正则化
                adjustment_factor = 1.0 + self.config.adaptation_rate * grad_norm
                self.current_lambda *= adjustment_factor
            elif grad_norm < 0.1:
                # 梯度消失，减少正则化
                adjustment_factor = 1.0 - self.config.adaptation_rate
                self.current_lambda *= adjustment_factor
        
        # 限制正则化强度在合理范围内
        self.current_lambda = max(
            self.config.min_lambda,
            min(self.config.max_lambda, self.current_lambda)
        )
        
        self._adaptation_step += 1
        
        return reg_loss
    
    def _build_knn_graph(self, features: torch.Tensor) -> GraphStructure:
        """构建k近邻图
        
        基于特征相似度自动构建图结构
        """
        n = features.shape[0]
        k = min(self.config.k_neighbors, n - 1)
        
        # 计算特征相似度 (余弦相似度)
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = features_norm @ features_norm.T
        
        # 构建k近邻邻接矩阵
        _, indices = torch.topk(similarity, k=k+1, dim=1)  # +1 包含自身
        
        # 创建邻接矩阵
        adjacency_matrix = torch.zeros(n, n, device=self.device)
        for i in range(n):
            neighbors = indices[i, 1:]  # 排除自身
            adjacency_matrix[i, neighbors] = 1
        
        # 确保对称性 (无向图)
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
        adjacency_matrix = (adjacency_matrix > 0).float()
        
        return create_graph_from_adjacency(adjacency_matrix, device=self.device)
    
    def _extract_subgraph(
        self,
        graph: GraphStructure,
        mask: torch.Tensor
    ) -> GraphStructure:
        """提取子图
        
        从完整图中提取指定节点的子图
        """
        mask_indices = torch.where(mask)[0]
        n_sub = len(mask_indices)
        
        if n_sub == 0:
            raise ValueError("子图节点数为零")
        
        # 创建索引映射
        idx_map = torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device)
        idx_map[mask_indices] = torch.arange(n_sub, device=self.device)
        
        # 提取子图的边
        edge_indices = graph.edge_indices
        edge_mask = mask[edge_indices[0]] & mask[edge_indices[1]]
        
        sub_edge_indices = edge_indices[:, edge_mask]
        sub_edge_indices = idx_map[sub_edge_indices]
        
        # 提取子图的边权重
        if graph.edge_weights is not None:
            sub_edge_weights = graph.edge_weights[edge_mask]
        else:
            sub_edge_weights = None
        
        # 创建子图的邻接矩阵
        sub_adjacency = torch.zeros(n_sub, n_sub, device=self.device)
        sub_adjacency[sub_edge_indices[0], sub_edge_indices[1]] = 1
        
        # 提取子图的节点特征
        if graph.node_features is not None:
            sub_node_features = graph.node_features[mask]
        else:
            sub_node_features = None
        
        return GraphStructure(
            num_nodes=n_sub,
            num_edges=sub_edge_indices.shape[1],
            adjacency_matrix=sub_adjacency,
            edge_indices=sub_edge_indices,
            edge_weights=sub_edge_weights,
            node_features=sub_node_features,
            directed=graph.directed
        )
    
    def _scale_graph(
        self,
        graph: GraphStructure,
        scale_factor: float
    ) -> GraphStructure:
        """图尺度变换
        
        通过调整边权重实现图的多尺度表示
        """
        if scale_factor >= 1.0:
            # 放大尺度: 增强强连接
            scaled_adjacency = torch.pow(graph.adjacency_matrix, scale_factor)
        else:
            # 缩小尺度: 弱化弱连接
            scaled_adjacency = torch.pow(graph.adjacency_matrix, 1.0/scale_factor)
        
        # 重新归一化
        scaled_adjacency = scaled_adjacency / scaled_adjacency.max()
        
        return GraphStructure(
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,  # 边数量不变，权重变化
            adjacency_matrix=scaled_adjacency,
            edge_indices=graph.edge_indices,
            edge_weights=scaled_adjacency[graph.edge_indices[0], graph.edge_indices[1]],
            node_features=graph.node_features,
            directed=graph.directed
        )
    
    def _scale_adjacency(
        self,
        adjacency: torch.Tensor,
        scale_factor: float
    ) -> torch.Tensor:
        """邻接矩阵尺度变换"""
        if scale_factor >= 1.0:
            scaled_adj = torch.pow(adjacency, scale_factor)
        else:
            scaled_adj = torch.pow(adjacency, 1.0/scale_factor)
        
        return scaled_adj / scaled_adj.max()
    
    def _scale_features(
        self,
        features: torch.Tensor,
        scale_factor: float
    ) -> torch.Tensor:
        """特征尺度变换"""
        # 应用高斯平滑
        if scale_factor < 1.0:
            # 下采样: 应用低通滤波
            kernel_size = int(1.0 / scale_factor)
            if kernel_size > 1:
                # 简单平均滤波
                weights = torch.ones(1, 1, kernel_size, device=self.device) / kernel_size
                features_2d = features.unsqueeze(0).unsqueeze(0)  # [1, 1, n, d]
                smoothed = F.conv2d(features_2d, weights, padding=kernel_size//2)
                scaled_features = smoothed.squeeze()
            else:
                scaled_features = features
        else:
            # 上采样: 添加噪声增强细节
            noise_scale = scale_factor - 1.0
            scaled_features = features + noise_scale * torch.randn_like(features)
        
        return scaled_features
    
    def _compute_feature_complexity(self, features: torch.Tensor) -> torch.Tensor:
        """计算特征复杂度
        
        使用奇异值分解估计特征空间的复杂度
        """
        # 中心化特征
        features_centered = features - features.mean(dim=0, keepdim=True)
        
        # 计算协方差矩阵的奇异值
        try:
            U, S, V = torch.svd_lowrank(features_centered, q=min(10, features.shape[1]))
            
            # 特征复杂度 = 奇异值的熵
            S_norm = S / (S.sum() + 1e-10)
            entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
            
            # 归一化到[0, 1]
            max_entropy = math.log(min(features.shape))
            complexity = entropy / max_entropy
            
        except Exception as e:
            logger.warning(f"特征复杂度计算失败: {e}, 使用默认值")
            complexity = torch.tensor(0.5, device=self.device)
        
        return complexity
    
    def _estimate_gradient_norm(self, features: torch.Tensor) -> torch.Tensor:
        """估计梯度范数
        
        通过有限差分估计特征空间的梯度大小
        """
        if not features.requires_grad:
            return torch.tensor(0.0, device=self.device)
        
        # 计算当前梯度
        if features.grad is not None:
            grad_norm = torch.norm(features.grad)
        else:
            # 如果没有梯度，使用随机估计
            grad_norm = torch.tensor(0.5, device=self.device)
        
        return grad_norm
    
    def get_regularization_stats(self) -> Dict[str, Any]:
        """获取正则化特定统计信息"""
        stats = self._regularization_stats.copy()
        stats.update({
            "graph_module_available": GRAPH_MODULE_AVAILABLE,
            "laplacian_calculator_available": self.laplacian_calculator is not None,
        })
        return stats
    
    def get_full_stats(self) -> Dict[str, Any]:
        """获取完整统计信息（包括基类统计）"""
        base_stats = self.get_stats()
        reg_stats = self.get_regularization_stats()
        base_stats.update(reg_stats)
        return base_stats
    
    def reset_regularization_stats(self) -> None:
        """重置正则化特定统计"""
        for key in self._regularization_stats:
            self._regularization_stats[key] = 0.0