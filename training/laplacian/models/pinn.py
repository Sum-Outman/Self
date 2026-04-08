#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉普拉斯增强的PINN模型

功能：
1. 基础PINN物理建模
2. 拉普拉斯正则化物理约束平滑性
3. 多尺度物理场分析
4. 自适应正则化强度调整

从 training/laplacian_enhanced_training.py 迁移而来
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)

# 导入相关模块
try:
    from models.graph.laplacian_matrix import GraphLaplacian, GraphStructure, GraphType
    from ..core.regularization import LaplacianRegularization, RegularizationConfig
    from ..utils.config import LaplacianEnhancedTrainingConfig
    
    # 尝试导入PINN相关模块
    try:
        from models.physics.pinn_framework import PINNConfig, PINNModel
        PINN_MODULE_AVAILABLE = True
    except ImportError:
        PINN_MODULE_AVAILABLE = False
        logger.warning("PINN模块不可用，功能将受限")
        # PINN模块不可用，不提供虚拟实现
        PINN_MODULE_AVAILABLE = False
        logger.error("PINN模块不可用。根据项目要求'禁止使用虚拟数据'，不能提供虚拟实现。")
        
        # 创建抛出异常的占位符类
        class PINNConfig:
            """PINN配置占位符 - 模块不可用时抛出异常"""
            def __init__(self, **kwargs):
                raise RuntimeError(
                    "PINNConfig不可用。PINN模块导入失败，根据项目要求'禁止使用虚拟数据'，不能提供虚拟实现。\n"
                    "请确保models.physics.pinn_framework模块已正确实现。"
                )
        
        class PINNModel(nn.Module):
            """PINN模型占位符 - 模块不可用时抛出异常"""
            def __init__(self, config):
                super().__init__()
                raise RuntimeError(
                    "PINNModel不可用。PINN模块导入失败，根据项目要求'禁止使用虚拟数据'，不能提供虚拟实现。\n"
                    "请确保models.physics.pinn_framework模块已正确实现。"
                )
    
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.warning(f"部分模块不可用: {e}, 功能将受限")
    
    # 根据项目要求"禁止使用虚拟数据"，模块导入失败时不提供虚拟实现
    # 所有类都定义为抛出异常的占位符
    
    class PINNConfig:
        """PINN配置占位符 - 模块不可用时抛出异常"""
        def __init__(self, **kwargs):
            raise RuntimeError(
                "PINNConfig不可用。必要的模块导入失败，根据项目要求'禁止使用虚拟数据'，不能提供虚拟实现。\n"
                "请确保models.graph.laplacian_matrix模块已正确实现。"
            )
    
    class PINNModel:
        """PINN模型占位符 - 模块不可用时抛出异常"""
        def __init__(self, config):
            super().__init__()
            raise RuntimeError(
                "PINNModel不可用。必要的模块导入失败，根据项目要求'禁止使用虚拟数据'，不能提供虚拟实现。\n"
                "请确保models.graph.laplacian_matrix模块已正确实现。"
            )
    
    class GraphLaplacian:
        """图拉普拉斯占位符 - 模块不可用时抛出异常"""
        def __init__(self, **kwargs):
            raise RuntimeError(
                "GraphLaplacian不可用。必要的模块导入失败，根据项目要求'禁止使用虚拟数据'，不能提供虚拟实现。\n"
                "请确保models.graph.laplacian_matrix模块已正确实现。"
            )
    
    class GraphStructure:
        """图结构占位符 - 模块不可用时抛出异常"""
        def __init__(self, **kwargs):
            raise RuntimeError(
                "GraphStructure不可用。必要的模块导入失败，根据项目要求'禁止使用虚拟数据'，不能提供虚拟实现。\n"
                "请确保models.graph.laplacian_matrix模块已正确实现。"
            )
    
    class GraphType:
        """图类型占位符 - 模块不可用时抛出异常"""
        def __init__(self, **kwargs):
            raise RuntimeError(
                "GraphType不可用。必要的模块导入失败，根据项目要求'禁止使用虚拟数据'，不能提供虚拟实现。\n"
                "请确保models.graph.laplacian_matrix模块已正确实现。"
            )
        UNDIRECTED = "undirected"
        DIRECTED = "directed"
    
    class LaplacianRegularization:
        """拉普拉斯正则化占位符 - 模块不可用时抛出异常"""
        def __init__(self, **kwargs):
            raise RuntimeError(
                "LaplacianRegularization不可用。必要的模块导入失败，根据项目要求'禁止使用虚拟数据'，不能提供虚拟实现。\n"
                "请确保training.laplacian.core.regularization模块已正确实现。"
            )
    
    class RegularizationConfig:
        """正则化配置占位符 - 模块不可用时抛出异常"""
        def __init__(self, **kwargs):
            raise RuntimeError(
                "RegularizationConfig不可用。必要的模块导入失败，根据项目要求'禁止使用虚拟数据'，不能提供虚拟实现。\n"
                "请确保training.laplacian.core.regularization模块已正确实现。"
            )
    
    class LaplacianEnhancedTrainingConfig:
        """拉普拉斯增强训练配置占位符 - 模块不可用时抛出异常"""
        def __init__(self, **kwargs):
            raise RuntimeError(
                "LaplacianEnhancedTrainingConfig不可用。必要的模块导入失败，根据项目要求'禁止使用虚拟数据'，不能提供虚拟实现。\n"
                "请确保training.laplacian.utils.config模块已正确实现。"
            )


class LaplacianEnhancedPINN(nn.Module):
    """拉普拉斯增强的PINN模型
    
    功能：
    1. 基础PINN物理建模
    2. 拉普拉斯正则化物理约束平滑性
    3. 多尺度物理场分析
    4. 自适应正则化强度调整
    """
    
    def __init__(self, 
                 pinn_config: PINNConfig,
                 laplacian_config: LaplacianEnhancedTrainingConfig):
        super().__init__()
        
        if not MODULES_AVAILABLE:
            raise ImportError("必要的模块不可用，无法初始化LaplacianEnhancedPINN")
        
        self.pinn_config = pinn_config
        self.laplacian_config = laplacian_config
        
        # 基础PINN模型
        self.pinn_model = PINNModel(pinn_config)
        
        # 拉普拉斯正则化器
        if laplacian_config.laplacian_reg_enabled:
            self.laplacian_regularizer = LaplacianRegularization(
                config=RegularizationConfig(
                    regularization_type="graph_laplacian",
                    lambda_reg=laplacian_config.laplacian_reg_lambda,
                    normalization=laplacian_config.laplacian_normalization
                )
            )
        else:
            self.laplacian_regularizer = None
        
        # 图拉普拉斯计算器
        self.graph_laplacian = GraphLaplacian(
            normalization=laplacian_config.laplacian_normalization,
            use_sparse=laplacian_config.use_sparse,
            dtype=torch.float64,
            cache_enabled=laplacian_config.cache_enabled,
            max_cache_size=laplacian_config.max_cache_size
        )
        
        # 自适应正则化参数
        if laplacian_config.adaptive_lambda:
            self.lambda_history = deque(maxlen=100)
            self.current_lambda = laplacian_config.laplacian_reg_lambda
        else:
            self.current_lambda = laplacian_config.laplacian_reg_lambda
        
        # 训练统计
        self.training_stats = {
            "total_iterations": 0,
            "laplacian_loss_history": [],
            "pde_loss_history": [],
            "total_loss_history": [],
            "lambda_history": []
        }
        
        logger.info(f"拉普拉斯增强PINN初始化完成: 模式={laplacian_config.training_mode}, "
                   f"正则化强度={laplacian_config.laplacian_reg_lambda}")
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.pinn_model(inputs)
    
    def compute_total_loss(self, 
                          inputs: torch.Tensor,
                          targets: Optional[torch.Tensor] = None,
                          iteration: int = 0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """计算总损失（包括拉普拉斯正则化）"""
        
        # 基础PINN损失
        if targets is not None:
            pinn_loss, loss_dict = self.pinn_model.compute_total_loss(inputs, targets, iteration)
        else:
            # 仅物理约束
            outputs = self.pinn_model(inputs)
            pinn_loss, loss_dict = self.pinn_model.compute_total_loss(inputs, outputs, iteration)
        
        total_loss = pinn_loss
        loss_dict["pinn_loss"] = pinn_loss.item()
        
        # 拉普拉斯正则化损失
        if self.laplacian_regularizer is not None and self.laplacian_config.laplacian_reg_enabled:
            # 构建特征图（使用模型输出作为节点特征）
            outputs = self.pinn_model(inputs)
            
            # 构建图结构（基于输入空间位置）
            graph = self._construct_graph_from_inputs(inputs)
            
            # 计算拉普拉斯正则化损失
            laplacian_loss = self.laplacian_regularizer(
                features=outputs,
                graph_structure=graph
            )
            
            # 应用自适应lambda
            if self.laplacian_config.adaptive_lambda:
                laplacian_loss = self.current_lambda * laplacian_loss
                self._update_adaptive_lambda(laplacian_loss, pinn_loss, iteration)
            else:
                laplacian_loss = self.laplacian_config.laplacian_reg_lambda * laplacian_loss
            
            total_loss = total_loss + laplacian_loss
            
            loss_dict["laplacian_loss"] = laplacian_loss.item()
            loss_dict["laplacian_lambda"] = self.current_lambda if self.laplacian_config.adaptive_lambda else self.laplacian_config.laplacian_reg_lambda
        
        loss_dict["total_loss"] = total_loss.item()
        
        # 更新统计
        self.training_stats["total_iterations"] += 1
        self.training_stats["total_loss_history"].append(total_loss.item())
        if "laplacian_loss" in loss_dict:
            self.training_stats["laplacian_loss_history"].append(loss_dict["laplacian_loss"])
        self.training_stats["pde_loss_history"].append(loss_dict.get("pde_residual", 0.0))
        
        if self.laplacian_config.adaptive_lambda:
            self.training_stats["lambda_history"].append(self.current_lambda)
        
        return total_loss, loss_dict
    
    def _construct_graph_from_inputs(self, inputs: torch.Tensor) -> GraphStructure:
        """从输入构建图结构"""
        
        method = self.laplacian_config.graph_construction_method
        
        if method == "knn":
            # k近邻图
            return self._construct_knn_graph(inputs)
        elif method == "radius":
            # 半径图
            return self._construct_radius_graph(inputs)
        elif method == "precomputed":
            # 预计算图（简单全连接）
            return self._construct_precomputed_graph(inputs)
        else:
            logger.warning(f"未知的图构建方法: {method}, 使用预计算图")
            return self._construct_precomputed_graph(inputs)
    
    def _construct_knn_graph(self, inputs: torch.Tensor) -> GraphStructure:
        """构建k近邻图"""
        n = inputs.shape[0]
        k = min(self.laplacian_config.k_neighbors, n - 1)
        
        # 计算距离矩阵
        distances = torch.cdist(inputs, inputs)
        
        # 获取k近邻
        _, indices = torch.topk(distances, k=k, dim=1, largest=False)
        
        # 构建邻接矩阵
        adjacency = torch.zeros((n, n), device=inputs.device, dtype=torch.float32)
        for i in range(n):
            adjacency[i, indices[i]] = 1.0
        
        # 对称化（对于无向图）
        adjacency = (adjacency + adjacency.t()) / 2
        adjacency = (adjacency > 0).float()
        
        return GraphStructure(
            adjacency_matrix=adjacency,
            graph_type=GraphType.UNDIRECTED
        )
    
    def _construct_radius_graph(self, inputs: torch.Tensor) -> GraphStructure:
        """构建半径图"""
        n = inputs.shape[0]
        radius = self.laplacian_config.radius
        
        # 计算距离矩阵
        distances = torch.cdist(inputs, inputs)
        
        # 构建邻接矩阵
        adjacency = (distances < radius).float()
        
        # 移除自环
        adjacency = adjacency - torch.eye(n, device=inputs.device)
        
        return GraphStructure(
            adjacency_matrix=adjacency,
            graph_type=GraphType.UNDIRECTED
        )
    
    def _construct_precomputed_graph(self, inputs: torch.Tensor) -> GraphStructure:
        """构建预计算图（简单实现）"""
        n = inputs.shape[0]
        
        # 简单全连接图（移除自环）
        adjacency = torch.ones((n, n), device=inputs.device, dtype=torch.float32)
        adjacency = adjacency - torch.eye(n, device=inputs.device)
        
        return GraphStructure(
            adjacency_matrix=adjacency,
            graph_type=GraphType.UNDIRECTED
        )
    
    def _update_adaptive_lambda(self, 
                               laplacian_loss: torch.Tensor,
                               pinn_loss: torch.Tensor,
                               iteration: int):
        """更新自适应正则化强度"""
        
        # 基于损失比例调整lambda
        loss_ratio = laplacian_loss.item() / max(pinn_loss.item(), 1e-8)
        
        # 目标比例：拉普拉斯损失占总损失的10-20%
        target_ratio = 0.15
        
        if loss_ratio < 0.05:  # 拉普拉斯损失太小
            self.current_lambda *= 1.1
        elif loss_ratio > 0.3:  # 拉普拉斯损失太大
            self.current_lambda *= 0.9
        # 否则保持当前lambda
        
        # 限制lambda范围
        self.current_lambda = max(1e-6, min(1.0, self.current_lambda))
        
        # 记录历史
        self.lambda_history.append(self.current_lambda)
        
        if iteration % self.laplacian_config.logging_frequency == 0:
            logger.info(f"自适应lambda更新: iteration={iteration}, "
                       f"lambda={self.current_lambda:.6f}, "
                       f"损失比例={loss_ratio:.4f}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        stats = self.training_stats.copy()
        
        # 计算平均损失
        if stats["total_loss_history"]:
            stats["avg_total_loss"] = np.mean(stats["total_loss_history"][-100:])
        if stats["laplacian_loss_history"]:
            stats["avg_laplacian_loss"] = np.mean(stats["laplacian_loss_history"][-100:])
        if stats["pde_loss_history"]:
            stats["avg_pde_loss"] = np.mean(stats["pde_loss_history"][-100:])
        
        # 添加当前lambda
        if self.laplacian_config.adaptive_lambda:
            stats["current_lambda"] = self.current_lambda
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self.pinn_model, 'cleanup'):
            self.pinn_model.cleanup()
        
        logger.info("拉普拉斯增强PINN资源已清理")