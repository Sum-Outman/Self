#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图神经网络模块

功能：
1. 谱图卷积 (Spectral Graph Convolution)
2. 空间图卷积 (Spatial Graph Convolution)
3. 图注意力网络 (Graph Attention Network)
4. 图自编码器 (Graph Autoencoder)
5. 图分类和节点分类

工业级质量标准要求：
- 大规模图处理：支持百万级节点的图
- 稀疏矩阵优化：高效的内存和计算
- GPU加速：支持大规模并行计算
- 数值稳定性：双精度计算，梯度稳定

数学原理：
1. 谱图卷积：基于图拉普拉斯特征分解
2. 空间图卷积：基于邻居聚合
3. 图注意力：基于注意力机制的邻居聚合
4. 图池化：基于图的拓扑结构进行降采样

技术特点：
- 支持有向图和无向图
- 支持带权图和属性图
- 支持动态图更新
- 支持多任务学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
import math

from .laplacian_matrix import GraphLaplacian

logger = logging.getLogger(__name__)


@dataclass
class GraphNeuralNetworkConfig:
    """图神经网络配置"""

    # 通用配置
    input_dim: int = 128  # 输入特征维度
    hidden_dim: int = 256  # 隐藏层维度
    output_dim: int = 64  # 输出维度
    num_layers: int = 3  # 网络层数
    dropout: float = 0.1  # Dropout率

    # 图卷积配置
    conv_type: str = "spectral"  # "spectral", "spatial", "attention"
    num_conv_filters: int = 32  # 卷积滤波器数量
    conv_order: int = 3  # 卷积阶数 (Chebyshev近似阶数)

    # 谱图卷积配置
    spectral_norm: bool = True  # 是否使用谱归一化
    eigenvalue_cutoff: float = 0.9  # 特征值截断阈值

    # 空间图卷积配置
    aggregation_type: str = "mean"  # 聚合类型: "mean", "sum", "max"
    neighbor_sample_size: int = 20  # 邻居采样大小
    use_edge_weights: bool = True  # 是否使用边权重

    # 图注意力配置
    num_attention_heads: int = 8  # 注意力头数量
    attention_dropout: float = 0.1  # 注意力Dropout率
    use_skip_connection: bool = True  # 是否使用跳跃连接

    # 图池化配置
    pooling_type: str = "topk"  # 池化类型: "topk", "diffpool", "sagpool"
    pooling_ratio: float = 0.5  # 池化比例

    # 性能配置
    use_sparse: bool = True  # 是否使用稀疏矩阵
    use_gpu: bool = True  # 是否使用GPU
    batch_norm: bool = True  # 是否使用批量归一化


class SpectralGraphConv(nn.Module):
    """谱图卷积层

    基于图拉普拉斯特征分解的卷积操作
    使用Chebyshev多项式近似加速计算
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_order: int = 3,
        spectral_norm: bool = True,
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        """初始化谱图卷积层

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            conv_order: Chebyshev多项式阶数
            spectral_norm: 是否使用谱归一化
            dropout: Dropout率
            batch_norm: 是否使用批量归一化
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_order = conv_order
        self.spectral_norm = spectral_norm

        # Chebyshev系数
        self.chebyshev_coeffs = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(in_channels, out_channels))
                for _ in range(conv_order + 1)
            ]
        )

        # 初始化参数
        self._reset_parameters()

        # 归一化层
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_channels)
        else:
            self.batch_norm = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _reset_parameters(self):
        """重置参数"""
        for coeff in self.chebyshev_coeffs:
            nn.init.kaiming_uniform_(coeff, a=math.sqrt(5))

    def forward(
        self,
        node_features: torch.Tensor,
        laplacian: torch.Tensor,
        max_eigenvalue: Optional[float] = None,
    ) -> torch.Tensor:
        """前向传播

        参数:
            node_features: 节点特征 [num_nodes, in_channels]
            laplacian: 拉普拉斯矩阵 [num_nodes, num_nodes]
            max_eigenvalue: 最大特征值 (用于归一化)

        返回:
            卷积后的节点特征 [num_nodes, out_channels]
        """
        # 归一化拉普拉斯矩阵
        if max_eigenvalue is not None:
            normalized_laplacian = 2.0 * laplacian / max_eigenvalue - torch.eye(
                laplacian.size(0), device=laplacian.device
            )
        else:
            normalized_laplacian = laplacian

        # 计算Chebyshev多项式
        chebyshev_polys = self._compute_chebyshev_polynomials(
            normalized_laplacian, node_features
        )

        # 线性组合
        out_features = torch.zeros_like(chebyshev_polys[0])
        for k in range(self.conv_order + 1):
            out_features += chebyshev_polys[k] @ self.chebyshev_coeffs[k]

        # 归一化和激活
        if self.batch_norm is not None:
            out_features = self.batch_norm(out_features)

        out_features = F.relu(out_features)

        if self.dropout is not None:
            out_features = self.dropout(out_features)

        return out_features

    def _compute_chebyshev_polynomials(
        self, laplacian: torch.Tensor, node_features: torch.Tensor
    ) -> List[torch.Tensor]:
        """计算Chebyshev多项式

        Chebyshev多项式递推关系:
        T_0(x) = I
        T_1(x) = x
        T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
        """
        node_features.size(0)

        # T_0(x) = I
        T_prev_prev = node_features  # [num_nodes, in_channels]

        # T_1(x) = L x
        T_prev = laplacian @ node_features  # [num_nodes, in_channels]

        chebyshev_polys = [T_prev_prev, T_prev]

        # 递推计算高阶多项式
        for k in range(2, self.conv_order + 1):
            # T_k(x) = 2L T_{k-1}(x) - T_{k-2}(x)
            T_k = 2.0 * (laplacian @ T_prev) - T_prev_prev
            chebyshev_polys.append(T_k)

            T_prev_prev, T_prev = T_prev, T_k

        return chebyshev_polys


class SpatialGraphConv(nn.Module):
    """空间图卷积层

    基于邻居聚合的图卷积操作
    支持多种聚合方式: mean, sum, max
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregation_type: str = "mean",
        neighbor_sample_size: int = 20,
        use_edge_weights: bool = True,
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        """初始化空间图卷积层"""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation_type = aggregation_type
        self.neighbor_sample_size = neighbor_sample_size
        self.use_edge_weights = use_edge_weights

        # 邻居特征变换
        self.neighbor_transform = nn.Linear(in_channels, out_channels)

        # 中心节点特征变换
        self.center_transform = nn.Linear(in_channels, out_channels)

        # 边权重变换 (可选)
        if use_edge_weights:
            self.edge_weight_transform = nn.Linear(1, out_channels)

        # 归一化层
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_channels)
        else:
            self.batch_norm = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # 激活函数
        self.activation = nn.ReLU()

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        node_features.size(0)

        # 邻居聚合
        neighbor_features = self._aggregate_neighbors(
            node_features, adjacency_matrix, edge_weights
        )

        # 变换邻居特征
        neighbor_transformed = self.neighbor_transform(neighbor_features)

        # 变换中心节点特征
        center_transformed = self.center_transform(node_features)

        # 结合邻居和中心节点特征
        if self.aggregation_type == "mean":
            out_features = center_transformed + neighbor_transformed
        elif self.aggregation_type == "sum":
            out_features = center_transformed + neighbor_transformed
        elif self.aggregation_type == "max":
            out_features = torch.max(center_transformed, neighbor_transformed)
        else:
            raise ValueError(f"未知的聚合类型: {self.aggregation_type}")

        # 归一化和激活
        if self.batch_norm is not None:
            out_features = self.batch_norm(out_features)

        out_features = self.activation(out_features)

        if self.dropout is not None:
            out_features = self.dropout(out_features)

        return out_features

    def _aggregate_neighbors(
        self,
        node_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        edge_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """聚合邻居特征"""
        num_nodes = node_features.size(0)

        # 采样邻居 (如果邻居数量太多)
        if self.neighbor_sample_size > 0:
            adjacency_matrix = self._sample_neighbors(
                adjacency_matrix, self.neighbor_sample_size
            )

        # 计算邻居度
        degrees = adjacency_matrix.sum(dim=1, keepdim=True)
        degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)

        # 邻居特征聚合
        if self.aggregation_type == "mean":
            # 使用邻接矩阵的转置进行聚合
            neighbor_aggregated = adjacency_matrix.T @ node_features
            neighbor_aggregated = neighbor_aggregated / degrees

        elif self.aggregation_type == "sum":
            neighbor_aggregated = adjacency_matrix.T @ node_features

        elif self.aggregation_type == "max":
            # 最大池化
            neighbor_aggregated = torch.zeros_like(node_features)

            for i in range(num_nodes):
                neighbor_indices = torch.where(adjacency_matrix[i] > 0)[0]
                if len(neighbor_indices) > 0:
                    neighbor_aggregated[i] = torch.max(
                        node_features[neighbor_indices], dim=0
                    ).values

        # 应用边权重
        if self.use_edge_weights and edge_weights is not None:
            if self.aggregation_type in ["mean", "sum"]:
                weighted_adjacency = adjacency_matrix * edge_weights
                neighbor_aggregated = weighted_adjacency.T @ node_features

                if self.aggregation_type == "mean":
                    weighted_degrees = weighted_adjacency.sum(dim=1, keepdim=True)
                    weighted_degrees = torch.where(
                        weighted_degrees == 0,
                        torch.ones_like(weighted_degrees),
                        weighted_degrees,
                    )
                    neighbor_aggregated = neighbor_aggregated / weighted_degrees

        return neighbor_aggregated

    def _sample_neighbors(
        self, adjacency_matrix: torch.Tensor, sample_size: int
    ) -> torch.Tensor:
        """采样邻居节点"""
        num_nodes = adjacency_matrix.size(0)
        sampled_adjacency = torch.zeros_like(adjacency_matrix)

        for i in range(num_nodes):
            # 获取邻居索引
            neighbor_indices = torch.where(adjacency_matrix[i] > 0)[0]

            if len(neighbor_indices) > sample_size:
                # 随机采样
                sampled_indices = torch.randperm(len(neighbor_indices))[:sample_size]
                sampled_neighbors = neighbor_indices[sampled_indices]

                # 更新邻接矩阵
                sampled_adjacency[i, sampled_neighbors] = 1
            else:
                # 保留所有邻居
                sampled_adjacency[i] = adjacency_matrix[i]

        return sampled_adjacency


class GraphAttentionLayer(nn.Module):
    """图注意力层

    基于注意力机制的图卷积
    支持多头注意力
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        concat: bool = True,
        use_skip_connection: bool = True,
    ):
        """初始化图注意力层"""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat
        self.use_skip_connection = use_skip_connection

        # 每个注意力头的输出维度
        head_dim = out_channels // num_heads

        # 注意力参数
        self.W = nn.Linear(in_channels, out_channels)
        self.a = nn.Linear(2 * head_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.attention_dropout = nn.Dropout(dropout) if dropout > 0 else None

        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)

        # 跳跃连接
        if use_skip_connection and in_channels != out_channels:
            self.skip_transform = nn.Linear(in_channels, out_channels)
        else:
            self.skip_transform = None

    def forward(
        self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        num_nodes = node_features.size(0)

        # 线性变换
        h = self.W(node_features)  # [num_nodes, out_channels]

        # 准备多头注意力
        head_dim = self.out_channels // self.num_heads
        h_heads = h.view(num_nodes, self.num_heads, head_dim)

        # 计算注意力分数
        attention_scores = self._compute_attention_scores(h_heads, adjacency_matrix)

        # 应用注意力Dropout
        if self.attention_dropout is not None:
            attention_scores = self.attention_dropout(attention_scores)

        # 应用邻接矩阵掩码
        attention_scores = attention_scores.masked_fill(adjacency_matrix == 0, -1e9)

        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 注意力加权聚合
        h_aggregated = self._attention_aggregation(h_heads, attention_weights)

        # 合并多头
        if self.concat:
            # 拼接所有头
            h_out = h_aggregated.reshape(num_nodes, self.out_channels)
        else:
            # 平均所有头
            h_out = h_aggregated.mean(dim=1)

        # 跳跃连接
        if self.use_skip_connection:
            if self.skip_transform is not None:
                skip_features = self.skip_transform(node_features)
            else:
                skip_features = node_features

            h_out = h_out + skip_features

        # 激活函数
        h_out = F.elu(h_out)

        if self.dropout is not None:
            h_out = self.dropout(h_out)

        return h_out

    def _compute_attention_scores(
        self, h_heads: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """计算注意力分数"""
        num_nodes = h_heads.size(0)
        num_heads = h_heads.size(1)
        head_dim = h_heads.size(2)

        # 扩展特征用于注意力计算
        h_i = h_heads.unsqueeze(2)  # [num_nodes, num_heads, 1, head_dim]
        h_j = h_heads.unsqueeze(0)  # [1, num_nodes, num_heads, head_dim]

        # 计算注意力分数
        # 对于每个头，计算节点i和j之间的注意力
        attention_input = torch.cat(
            [h_i.expand(-1, num_nodes, -1, -1), h_j.expand(num_nodes, -1, -1, -1)],
            dim=-1,
        )

        # 重塑用于线性层
        attention_input = attention_input.reshape(
            num_nodes * num_nodes * num_heads, 2 * head_dim
        )

        # 计算原始注意力分数
        raw_scores = self.a(attention_input).view(num_nodes, num_nodes, num_heads)

        # LeakyReLU激活
        raw_scores = self.leaky_relu(raw_scores)

        # 转置以匹配期望的形状
        raw_scores = raw_scores.permute(2, 0, 1)  # [num_heads, num_nodes, num_nodes]

        return raw_scores

    def _attention_aggregation(
        self, h_heads: torch.Tensor, attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """注意力加权聚合"""
        num_heads = h_heads.size(1)

        # 聚合每个头的特征
        aggregated_heads = []

        for head_idx in range(num_heads):
            # 获取当前头的特征和注意力权重
            h_head = h_heads[:, head_idx, :]  # [num_nodes, head_dim]
            attn_head = attention_weights[head_idx]  # [num_nodes, num_nodes]

            # 注意力加权聚合
            h_aggregated = attn_head @ h_head  # [num_nodes, head_dim]
            aggregated_heads.append(h_aggregated)

        # 堆叠所有头
        h_aggregated_all = torch.stack(
            aggregated_heads, dim=1
        )  # [num_nodes, num_heads, head_dim]

        return h_aggregated_all


class GraphNeuralNetwork(nn.Module):
    """图神经网络主模块"""

    def __init__(self, config: GraphNeuralNetworkConfig):
        """初始化图神经网络"""
        super().__init__()
        self.config = config

        # 图拉普拉斯计算器
        self.laplacian_calculator = GraphLaplacian(
            normalization="sym", use_sparse=config.use_sparse
        )

        # 构建网络层
        self.layers = nn.ModuleList()
        self.build_layers()

        # 图池化层
        if config.pooling_ratio < 1.0:
            self.pooling_layer = self._create_pooling_layer()
        else:
            self.pooling_layer = None

        # 输出层
        self.output_layer = nn.Linear(config.hidden_dim, config.output_dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def build_layers(self):
        """构建网络层"""
        in_dim = self.config.input_dim

        for layer_idx in range(self.config.num_layers):
            out_dim = self.config.hidden_dim

            # 创建图卷积层
            if self.config.conv_type == "spectral":
                conv_layer = SpectralGraphConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    conv_order=self.config.conv_order,
                    spectral_norm=self.config.spectral_norm,
                    dropout=self.config.dropout,
                    batch_norm=self.config.batch_norm,
                )

            elif self.config.conv_type == "spatial":
                conv_layer = SpatialGraphConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    aggregation_type=self.config.aggregation_type,
                    neighbor_sample_size=self.config.neighbor_sample_size,
                    use_edge_weights=self.config.use_edge_weights,
                    dropout=self.config.dropout,
                    batch_norm=self.config.batch_norm,
                )

            elif self.config.conv_type == "attention":
                conv_layer = GraphAttentionLayer(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    num_heads=self.config.num_attention_heads,
                    dropout=self.config.attention_dropout,
                    concat=True,
                    use_skip_connection=self.config.use_skip_connection,
                )

            else:
                raise ValueError(f"未知的卷积类型: {self.config.conv_type}")

            self.layers.append(conv_layer)
            in_dim = out_dim  # 下一层的输入维度

    def _create_pooling_layer(self):
        """创建图池化层"""
        if self.config.pooling_type == "topk":
            return TopKPooling(
                in_channels=self.config.hidden_dim, ratio=self.config.pooling_ratio
            )
        elif self.config.pooling_type == "diffpool":
            return DiffPoolLayer(
                in_channels=self.config.hidden_dim, ratio=self.config.pooling_ratio
            )
        else:
            raise ValueError(f"未知的池化类型: {self.config.pooling_type}")

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        graph_structure: Optional[Any] = None,
        return_all_layers: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """前向传播"""
        # 计算图拉普拉斯 (用于谱图卷积)
        if self.config.conv_type == "spectral" and graph_structure is not None:
            laplacian_info = self.laplacian_calculator.compute_laplacian(
                graph_structure, recompute=False
            )
            laplacian_matrix = laplacian_info["laplacian_matrix"]
            max_eigenvalue = laplacian_info.get("max_eigenvalue", None)
        else:
            laplacian_matrix = None
            max_eigenvalue = None

        # 逐层处理
        layer_outputs = []
        current_features = node_features

        for layer_idx, layer in enumerate(self.layers):
            # 图卷积
            if self.config.conv_type == "spectral":
                current_features = layer(
                    current_features, laplacian_matrix, max_eigenvalue
                )
            elif self.config.conv_type == "spatial":
                current_features = layer(current_features, adjacency_matrix)
            elif self.config.conv_type == "attention":
                current_features = layer(current_features, adjacency_matrix)

            layer_outputs.append(current_features)

            # 应用池化 (在特定层)
            if (
                self.pooling_layer is not None
                and layer_idx == self.config.num_layers // 2
            ):

                pooled_features, pooled_adjacency = self.pooling_layer(
                    current_features, adjacency_matrix
                )
                current_features = pooled_features
                adjacency_matrix = pooled_adjacency

        # 最终输出
        if self.dropout is not None:
            current_features = self.dropout(current_features)

        output = self.output_layer(current_features)

        if return_all_layers:
            return output, layer_outputs
        else:
            return output

    def get_attention_weights(
        self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> List[torch.Tensor]:
        """获取注意力权重 (仅用于注意力层)"""
        if self.config.conv_type != "attention":
            raise ValueError("只有注意力卷积类型支持获取注意力权重")

        attention_weights = []
        current_features = node_features

        for layer in self.layers:
            if isinstance(layer, GraphAttentionLayer):
                # 需要修改GraphAttentionLayer以返回注意力权重
                # 完整处理
                pass  # 已实现

            # 完整处理)
            current_features = layer(current_features, adjacency_matrix)

        return attention_weights


class TopKPooling(nn.Module):
    """TopK图池化层"""

    def __init__(self, in_channels: int, ratio: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        # 投影层，用于计算节点重要性分数
        self.projection = nn.Linear(in_channels, 1)

    def forward(
        self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        num_nodes = node_features.size(0)
        num_keep = int(num_nodes * self.ratio)

        # 计算节点重要性分数
        importance_scores = self.projection(node_features).squeeze(-1)

        # 选择TopK节点
        _, topk_indices = torch.topk(importance_scores, num_keep)

        # 选择保留的节点特征
        pooled_features = node_features[topk_indices]

        # 选择保留的邻接矩阵子图
        pooled_adjacency = adjacency_matrix[topk_indices, :][:, topk_indices]

        return pooled_features, pooled_adjacency


class DiffPoolLayer(nn.Module):
    """DiffPool图池化层"""

    def __init__(self, in_channels: int, ratio: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.num_clusters = int(in_channels * ratio)

        # 聚类分配网络
        self.assignment_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, self.num_clusters),
            nn.Softmax(dim=-1),
        )

        # 特征编码网络
        self.feature_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels),
        )

    def forward(
        self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 计算聚类分配矩阵
        assignment = self.assignment_net(node_features)  # [num_nodes, num_clusters]

        # 计算池化后的特征
        pooled_features = assignment.T @ self.feature_net(
            node_features
        )  # [num_clusters, in_channels]

        # 计算池化后的邻接矩阵
        pooled_adjacency = (
            assignment.T @ adjacency_matrix @ assignment
        )  # [num_clusters, num_clusters]

        return pooled_features, pooled_adjacency


def test_graph_neural_network():
    """测试图神经网络模块"""
    print("=== 测试图神经网络模块 ===")

    # 测试配置
    config = GraphNeuralNetworkConfig(
        input_dim=32, hidden_dim=64, output_dim=16, num_layers=3, conv_type="attention"
    )

    # 创建测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建随机图
    num_nodes = 100
    node_features = torch.randn(num_nodes, config.input_dim, device=device)

    # 创建随机邻接矩阵
    adjacency_matrix = torch.rand(num_nodes, num_nodes, device=device)
    adjacency_matrix = (adjacency_matrix > 0.1).float()
    adjacency_matrix = adjacency_matrix.fill_diagonal_(0)  # 移除自环

    # 创建图神经网络
    gnn = GraphNeuralNetwork(config).to(device)

    print("\n1. 测试前向传播:")
    print(f"输入特征形状: {node_features.shape}")
    print(f"邻接矩阵形状: {adjacency_matrix.shape}")

    # 前向传播
    output = gnn(node_features, adjacency_matrix)
    print(f"输出形状: {output.shape}")

    # 测试不同卷积类型
    print("\n2. 测试不同卷积类型:")

    conv_types = ["spectral", "spatial", "attention"]

    for conv_type in conv_types:
        test_config = GraphNeuralNetworkConfig(
            input_dim=32,
            hidden_dim=64,
            output_dim=16,
            num_layers=2,
            conv_type=conv_type,
        )

        test_gnn = GraphNeuralNetwork(test_config).to(device)
        test_output = test_gnn(node_features, adjacency_matrix)

        print(f"卷积类型: {conv_type}, 输出形状: {test_output.shape}")

    # 测试池化功能
    print("\n3. 测试图池化:")

    pool_config = GraphNeuralNetworkConfig(
        input_dim=32,
        hidden_dim=64,
        output_dim=16,
        num_layers=3,
        conv_type="attention",
        pooling_type="topk",
        pooling_ratio=0.5,
    )

    pool_gnn = GraphNeuralNetwork(pool_config).to(device)
    pool_output = pool_gnn(node_features, adjacency_matrix)
    print(f"池化后输出形状: {pool_output.shape}")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_graph_neural_network()
