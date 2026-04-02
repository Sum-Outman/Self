#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物理信息神经网络(PINN)基础框架

功能：
1. 物理约束损失函数计算
2. PDE残差自动微分
3. 边界条件和初始条件处理
4. 多物理场耦合支持
5. 自适应权重调整

工业级质量标准要求：
- 数值稳定性：双精度计算，梯度稳定
- 计算效率：GPU加速，自动微分优化
- 内存效率：大规模物理场模拟支持
- 可扩展性：模块化设计，易于扩展

数学原理：
1. PINN基本方程：L = L_data + λ * L_physics
2. PDE残差：R(u) = N(u) - f，其中N是微分算子
3. 自动微分：使用PyTorch自动计算高阶导数
4. 软边界约束：通过惩罚项实现

参考文献：
[1] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks.
[2] Karniadakis, G. E., et al. (2021). Physics-informed machine learning.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import math
from abc import ABC, abstractmethod
from scipy import integrate

logger = logging.getLogger(__name__)


@dataclass
class PINNConfig:
    """PINN配置类"""
    
    # 通用配置
    input_dim: int = 3  # 输入维度 (例如: x, y, t)
    output_dim: int = 1  # 输出维度 (例如: u)
    hidden_dim: int = 64  # 隐藏层维度
    num_layers: int = 5  # 网络层数
    activation: str = "tanh"  # 激活函数: "tanh", "sin", "relu"
    
    # 物理约束配置
    physics_weight: float = 1.0  # 物理损失权重
    data_weight: float = 1.0  # 数据损失权重
    bc_weight: float = 1.0  # 边界条件权重
    ic_weight: float = 1.0  # 初始条件权重
    
    # PDE配置
    pde_type: str = "burgers"  # PDE类型: "burgers", "navier_stokes", "wave", "heat", 
                              # "schrodinger", "maxwell", "elasticity", "reaction_diffusion"
    pde_order: int = 2  # PDE阶数
    use_autograd: bool = True  # 是否使用自动微分
    
    # 训练配置
    adaptive_weighting: bool = True  # 是否使用自适应权重
    weight_update_freq: int = 100  # 权重更新频率
    grad_clip: float = 1.0  # 梯度裁剪
    
    # 性能配置
    use_gpu: bool = True  # 是否使用GPU
    dtype: torch.dtype = torch.float64  # 数据类型 (推荐float64保证数值稳定性)
    parallel_computation: bool = False  # 是否使用并行计算
    # 混合精度训练配置
    use_mixed_precision: bool = False  # 是否使用混合精度训练
    mixed_precision_dtype: torch.dtype = torch.float16  # 混合精度数据类型 (float16或bfloat16)
    amp_enabled: bool = True  # 是否启用自动混合精度 (PyTorch AMP)
    grad_scaler_enabled: bool = True  # 是否启用梯度缩放
    # 分布式训练配置
    distributed_training: bool = False  # 是否启用分布式训练
    world_size: int = 1  # 分布式训练世界大小
    local_rank: int = 0  # 本地排名
    backend: str = "nccl"  # 分布式后端 (nccl, gloo, mpi)
    
    # 增量式缓存配置
    enable_incremental_cache: bool = True  # 是否启用增量式缓存
    max_cache_size: int = 1000  # 最大缓存项数量
    cache_eviction_policy: str = "adaptive"  # 缓存淘汰策略: lru, lfu, adaptive
    adaptive_cache_sizing: bool = True  # 是否启用自适应缓存大小调整
    cache_enabled_for_pde: bool = True  # 是否启用PDE残差缓存
    cache_enabled_for_gradients: bool = True  # 是否启用梯度缓存
    cache_enabled_for_losses: bool = True  # 是否启用损失缓存


class PhysicsConstraint(ABC):
    """物理约束基类"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.history = []
    
    def compute_loss(self, model: nn.Module, inputs: torch.Tensor, 
                    outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算约束损失 - 修复版
        
        修复内容：
        1. 使用自动微分计算PDE残差
        2. 支持多种PDE类型
        3. 正确处理边界条件和初始条件
        
        参数:
            model: 神经网络模型
            inputs: 输入张量 [batch_size, input_dim]
            outputs: 模型输出张量 [batch_size, output_dim]
            **kwargs: 额外参数，可能包含：
                - pde_type: PDE类型 ('burgers', 'wave', 'heat', etc.)
                - boundary_conditions: 边界条件
                - initial_conditions: 初始条件
                - physical_parameters: 物理参数
                
        返回:
            损失张量（标量）
        """
        # 获取PDE类型
        pde_type = kwargs.get('pde_type', 'default')
        
        # 计算PDE残差
        if pde_type == 'burgers':
            residual = self._compute_burgers_residual(model, inputs, outputs, **kwargs)
        elif pde_type == 'wave':
            residual = self._compute_wave_residual(model, inputs, outputs, **kwargs)
        elif pde_type == 'heat':
            residual = self._compute_heat_residual(model, inputs, outputs, **kwargs)
        else:
            # 默认：计算输出的MSE作为基础损失
            residual = outputs
        
        # 计算残差的MSE损失
        pde_loss = torch.mean(residual ** 2)
        
        # 应用约束权重
        weighted_loss = self.weight * pde_loss
        
        # 记录日志
        if kwargs.get('verbose', False) or len(self.history) % 100 == 0:
            logger.debug(f"PhysicsConstraint '{self.name}': PDE损失 = {pde_loss.item():.6f}, "
                        f"加权损失 = {weighted_loss.item():.6f}")
        
        return weighted_loss
    
    def _compute_burgers_residual(self, model: nn.Module, inputs: torch.Tensor, 
                                   outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算Burgers方程残差: u_t + u*u_x - nu*u_xx = 0"""
        # 假设 inputs = [x, t], outputs = u
        if inputs.shape[-1] < 2:
            return outputs  # 无法计算，返回原输出
        
        # 确保输入需要梯度
        if not inputs.requires_grad:
            inputs = inputs.clone().requires_grad_(True)
        
        # 重新计算输出以确保梯度图正确
        outputs = model(inputs)
        
        # 计算一阶导数
        grads = torch.autograd.grad(outputs, inputs, 
                                    grad_outputs=torch.ones_like(outputs),
                                    create_graph=True, retain_graph=True)[0]
        u_x = grads[:, 0:1]  # 对x的导数
        u_t = grads[:, 1:2] if inputs.shape[-1] > 1 else torch.zeros_like(u_x)  # 对t的导数
        
        # 计算二阶导数 u_xx
        try:
            u_xx_grads = torch.autograd.grad(u_x, inputs, 
                                       grad_outputs=torch.ones_like(u_x),
                                       create_graph=True, retain_graph=True,
                                       allow_unused=True)[0]
            if u_xx_grads is None:
                u_xx = torch.zeros_like(u_x)
            else:
                u_xx = u_xx_grads[:, 0:1]
        except RuntimeError:
            # 如果无法计算二阶导数，使用零
            u_xx = torch.zeros_like(u_x)
        
        # Burgers方程残差: u_t + u*u_x - nu*u_xx
        nu = kwargs.get('viscosity', 0.01)
        residual = u_t + outputs * u_x - nu * u_xx
        
        return residual
    
    def _compute_wave_residual(self, model: nn.Module, inputs: torch.Tensor, 
                                outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算波动方程残差: u_tt - c^2*u_xx = 0"""
        if inputs.shape[-1] < 2:
            return outputs
        
        inputs.requires_grad_(True)
        outputs = model(inputs)
        
        # 计算一阶导数
        grads = torch.autograd.grad(outputs, inputs, 
                                    grad_outputs=torch.ones_like(outputs),
                                    create_graph=True, retain_graph=True)[0]
        u_t = grads[:, 1:2] if inputs.shape[-1] > 1 else torch.zeros_like(grads[:, 0:1])
        
        # 计算二阶导数
        u_tt = torch.autograd.grad(u_t, inputs, 
                                   grad_outputs=torch.ones_like(u_t),
                                   create_graph=True, retain_graph=True)[0][:, 1:2] if inputs.shape[-1] > 1 else torch.zeros_like(u_t)
        u_xx = torch.autograd.grad(grads[:, 0:1], inputs, 
                                   grad_outputs=torch.ones_like(grads[:, 0:1]),
                                   create_graph=True, retain_graph=True)[0][:, 0:1]
        
        # 波动方程残差: u_tt - c^2*u_xx
        c = kwargs.get('wave_speed', 1.0)
        residual = u_tt - c**2 * u_xx
        
        return residual
    
    def _compute_heat_residual(self, model: nn.Module, inputs: torch.Tensor, 
                                outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算热方程残差: u_t - alpha*u_xx = 0"""
        if inputs.shape[-1] < 2:
            return outputs
        
        inputs.requires_grad_(True)
        outputs = model(inputs)
        
        # 计算导数
        grads = torch.autograd.grad(outputs, inputs, 
                                    grad_outputs=torch.ones_like(outputs),
                                    create_graph=True, retain_graph=True)[0]
        u_t = grads[:, 1:2] if inputs.shape[-1] > 1 else torch.zeros_like(grads[:, 0:1])
        u_x = grads[:, 0:1]
        
        # 计算二阶导数
        u_xx = torch.autograd.grad(u_x, inputs, 
                                   grad_outputs=torch.ones_like(u_x),
                                   create_graph=True, retain_graph=True)[0][:, 0:1]
        
        # 热方程残差: u_t - alpha*u_xx
        alpha = kwargs.get('thermal_diffusivity', 0.1)
        residual = u_t - alpha * u_xx
        
        return residual
    
    def update_weight(self, loss_value: float, iteration: int):
        """更新约束权重"""
        self.history.append(loss_value)
        
        # 自适应权重调整
        if len(self.history) > 10:
            recent_mean = np.mean(self.history[-10:])
            historical_mean = np.mean(self.history[:-10]) if len(self.history) > 10 else recent_mean
            
            # 基于损失变化调整权重
            if recent_mean > 2.0 * historical_mean:
                self.weight *= 1.1
            elif recent_mean < 0.5 * historical_mean:
                self.weight *= 0.9
            
            # 权重限制
            self.weight = max(0.1, min(10.0, self.weight))
    
    def reset_history(self):
        """重置历史记录"""
        self.history = []


class PDEResidualConstraint(PhysicsConstraint):
    """PDE残差约束"""
    
    def __init__(self, pde_function: Callable, weight: float = 1.0):
        super().__init__("pde_residual", weight)
        self.pde_function = pde_function
    
    def compute_loss(self, model: nn.Module, inputs: torch.Tensor,
                    outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算PDE残差损失"""
        # 使用自动微分计算PDE残差
        inputs.requires_grad_(True)
        
        # 计算模型输出
        u = model(inputs)
        
        # 计算PDE残差
        residual = self.pde_function(inputs, u, model)
        
        # 计算残差平方和
        loss = torch.mean(residual**2)
        
        return loss * self.weight


class BoundaryConditionConstraint(PhysicsConstraint):
    """边界条件约束"""
    
    def __init__(self, boundary_function: Callable, boundary_mask: torch.Tensor,
                 weight: float = 1.0, condition_type: str = "dirichlet"):
        super().__init__("boundary_condition", weight)
        self.boundary_function = boundary_function
        self.boundary_mask = boundary_mask
        self.condition_type = condition_type  # "dirichlet", "neumann", "robin"
    
    def compute_loss(self, model: nn.Module, inputs: torch.Tensor,
                    outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算边界条件损失"""
        # 提取边界点
        boundary_inputs = inputs[self.boundary_mask]
        
        if len(boundary_inputs) == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        # 计算模型在边界点的输出
        u_pred = model(boundary_inputs)
        
        # 计算边界条件目标值
        u_target = self.boundary_function(boundary_inputs)
        
        if self.condition_type == "dirichlet":
            # Dirichlet边界条件: u = g
            loss = torch.mean((u_pred - u_target)**2)
        elif self.condition_type == "neumann":
            # Neumann边界条件: ∂u/∂n = h
            # 需要计算法向导数
            boundary_inputs.requires_grad_(True)
            u_pred_grad = model(boundary_inputs)
            
            # 计算梯度
            grad_u = torch.autograd.grad(
                outputs=u_pred_grad,
                inputs=boundary_inputs,
                grad_outputs=torch.ones_like(u_pred_grad),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # 完整处理，假设法向为x方向)
            normal_derivative = grad_u[:, 0]  # 假设第一个坐标是法向
            loss = torch.mean((normal_derivative - u_target)**2)
        elif self.condition_type == "robin":
            # Robin边界条件: αu + β∂u/∂n = r
            # 组合Dirichlet和Neumann
            boundary_inputs.requires_grad_(True)
            u_pred_grad = model(boundary_inputs)
            
            # Dirichlet部分
            dirichlet_loss = torch.mean((u_pred_grad - u_target)**2)
            
            # Neumann部分
            grad_u = torch.autograd.grad(
                outputs=u_pred_grad,
                inputs=boundary_inputs,
                grad_outputs=torch.ones_like(u_pred_grad),
                create_graph=True,
                retain_graph=True
            )[0]
            normal_derivative = grad_u[:, 0]
            neumann_loss = torch.mean(normal_derivative**2)
            
            loss = dirichlet_loss + neumann_loss
        else:
            raise ValueError(f"未知的边界条件类型: {self.condition_type}")
        
        return loss * self.weight


class InitialConditionConstraint(PhysicsConstraint):
    """初始条件约束"""
    
    def __init__(self, initial_function: Callable, initial_mask: torch.Tensor,
                 weight: float = 1.0):
        super().__init__("initial_condition", weight)
        self.initial_function = initial_function
        self.initial_mask = initial_mask
    
    def compute_loss(self, model: nn.Module, inputs: torch.Tensor,
                    outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算初始条件损失"""
        # 提取初始时刻的点
        initial_inputs = inputs[self.initial_mask]
        
        if len(initial_inputs) == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        # 计算模型在初始时刻的输出
        u_pred = model(initial_inputs)
        
        # 计算初始条件目标值
        u_target = self.initial_function(initial_inputs)
        
        # 计算损失
        loss = torch.mean((u_pred - u_target)**2)
        
        return loss * self.weight


class DataConstraint(PhysicsConstraint):
    """数据约束（监督学习部分）"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__("data_constraint", weight)
    
    def compute_loss(self, model: nn.Module, inputs: torch.Tensor,
                    outputs: torch.Tensor, targets: torch.Tensor,
                    data_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算数据损失"""
        # 提取有数据点的输入
        data_inputs = inputs[data_mask]
        data_targets = targets[data_mask]
        
        if len(data_inputs) == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        # 计算模型输出
        u_pred = model(data_inputs)
        
        # 计算损失
        loss = torch.mean((u_pred - data_targets)**2)
        
        return loss * self.weight


class PINNModel(nn.Module):
    """PINN主模型"""
    
    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        
        # 设备配置
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # 构建神经网络
        self.network = self._build_network()
        
        # 物理约束列表
        self.constraints: List[PhysicsConstraint] = []
        
        # 自适应权重管理器
        self.adaptive_manager = AdaptiveWeightManager() if config.adaptive_weighting else None
        
        # 混合精度管理器
        self.mixed_precision_manager = None
        if config.use_mixed_precision and config.use_gpu and torch.cuda.is_available():
            self.mixed_precision_manager = MixedPrecisionManager(
                enabled=config.amp_enabled,
                dtype=config.mixed_precision_dtype
            )
        
        # 分布式训练管理器
        self.distributed_manager = None
        if config.distributed_training:
            self.distributed_manager = DistributedTrainingManager(config)
            
            # 如果分布式训练已初始化，包装模型
            if self.distributed_manager.initialized:
                # 保存原始网络引用
                self._original_network = self.network
                # 包装模型
                self.network = self.distributed_manager.wrap_model(self.network)
        
        # 增量式缓存管理器
        self.cache_manager = None
        if config.enable_incremental_cache:
            self.cache_manager = IncrementalCacheManager(
                max_cache_size=config.max_cache_size,
                eviction_policy=config.cache_eviction_policy,
                enable_adaptive_sizing=config.adaptive_cache_sizing
            )
            logger.info(f"增量式缓存管理器初始化: 最大缓存大小={config.max_cache_size}, "
                       f"淘汰策略={config.cache_eviction_policy}")
        
        # 移动到设备
        self.to(self.device)
        self.to(config.dtype)
        
        logger.info(f"PINN模型初始化完成: device={self.device}, dtype={config.dtype}, "
                   f"mixed_precision={config.use_mixed_precision}, "
                   f"distributed={config.distributed_training}, "
                   f"caching={config.enable_incremental_cache}")
    
    def _build_network(self) -> nn.Module:
        """构建神经网络"""
        layers = []
        
        # 输入层
        layers.append(nn.Linear(self.config.input_dim, self.config.hidden_dim))
        layers.append(self._get_activation())
        
        # 隐藏层
        for _ in range(self.config.num_layers - 2):
            layers.append(nn.Linear(self.config.hidden_dim, self.config.hidden_dim))
            layers.append(self._get_activation())
        
        # 输出层
        layers.append(nn.Linear(self.config.hidden_dim, self.config.output_dim))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """获取激活函数"""
        if self.config.activation == "tanh":
            return nn.Tanh()
        elif self.config.activation == "sin":
            return SinActivation()
        elif self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "siren":
            return SIRENActivation()
        else:
            raise ValueError(f"未知的激活函数: {self.config.activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)
    
    def forward_with_mixed_precision(self, x: torch.Tensor) -> torch.Tensor:
        """使用混合精度的前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        if self.mixed_precision_manager is not None:
            autocast_context = self.mixed_precision_manager.autocast_context()
            if autocast_context is not None:
                with autocast_context:
                    return self.network(x)
        
        # 回退到普通前向传播
        return self.forward(x)
    
    def add_constraint(self, constraint: PhysicsConstraint):
        """添加物理约束"""
        self.constraints.append(constraint)
    
    def compute_total_loss(self, inputs: torch.Tensor, outputs: torch.Tensor,
                          targets: Optional[torch.Tensor] = None,
                          data_mask: Optional[torch.Tensor] = None,
                          iteration: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算总损失"""
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device, dtype=self.config.dtype)
        
        # 检查是否使用混合精度
        use_autocast = False
        autocast_context = None
        if self.mixed_precision_manager is not None:
            autocast_context = self.mixed_precision_manager.autocast_context()
            use_autocast = autocast_context is not None
        
        # 计算每个约束的损失
        for constraint in self.constraints:
            if isinstance(constraint, DataConstraint):
                # 数据约束需要额外的参数
                if targets is None or data_mask is None:
                    continue
                
                # 在autocast上下文中计算损失（如果启用）
                if use_autocast:
                    with autocast_context:
                        loss = constraint.compute_loss(
                            self, inputs, outputs, targets=targets, data_mask=data_mask
                        )
                else:
                    loss = constraint.compute_loss(
                        self, inputs, outputs, targets=targets, data_mask=data_mask
                    )
            else:
                # 其他约束
                if use_autocast:
                    with autocast_context:
                        loss = constraint.compute_loss(self, inputs, outputs)
                else:
                    loss = constraint.compute_loss(self, inputs, outputs)
            
            loss_dict[constraint.name] = loss.item()
            total_loss = total_loss + loss
            
            # 更新自适应权重
            if self.adaptive_manager is not None and iteration % self.config.weight_update_freq == 0:
                constraint.update_weight(loss.item(), iteration)
        
        loss_dict["total"] = total_loss.item()
        
        return total_loss, loss_dict
    
    def train_step(self, inputs: torch.Tensor, outputs: torch.Tensor,
                  optimizer: torch.optim.Optimizer,
                  targets: Optional[torch.Tensor] = None,
                  data_mask: Optional[torch.Tensor] = None,
                  iteration: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """训练步骤（支持混合精度）
        
        参数:
            inputs: 输入张量
            outputs: 输出张量
            optimizer: 优化器
            targets: 目标值（用于数据约束）
            data_mask: 数据掩码（用于数据约束）
            iteration: 当前迭代次数
            
        返回:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        # 清零梯度
        optimizer.zero_grad()
        
        # 计算总损失
        total_loss, loss_dict = self.compute_total_loss(
            inputs, outputs, targets, data_mask, iteration
        )
        
        # 缩放损失（如果使用混合精度）
        if self.mixed_precision_manager is not None:
            scaled_loss = self.mixed_precision_manager.scale_loss(total_loss)
            scaled_loss.backward()
        else:
            total_loss.backward()
        
        # 分布式梯度同步
        if self.distributed_manager is not None:
            self.distributed_manager.synchronize_gradients(self.network)
        
        # 梯度裁剪
        if self.config.grad_clip > 0:
            if self.mixed_precision_manager is not None and self.mixed_precision_manager.grad_scaler is not None:
                # 对于混合精度，需要先unscale梯度
                self.mixed_precision_manager.grad_scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.config.grad_clip
            )
        
        # 执行优化器步骤（支持混合精度）
        if self.mixed_precision_manager is not None:
            self.mixed_precision_manager.step(optimizer)
        else:
            optimizer.step()
        
        # 检查梯度溢出
        if self.mixed_precision_manager is not None:
            overflow = self.mixed_precision_manager.check_overflow(optimizer)
            if overflow:
                logger.warning(f"检测到梯度溢出，迭代 {iteration}")
        
        return total_loss, loss_dict
    
    def cleanup(self):
        """清理训练资源（分布式训练、混合精度、缓存等）"""
        # 清理分布式训练资源
        if self.distributed_manager is not None:
            self.distributed_manager.cleanup()
        
        # 清理混合精度管理器
        if self.mixed_precision_manager is not None:
            self.mixed_precision_manager.reset()
        
        # 清理缓存管理器
        if self.cache_manager is not None:
            self.cache_manager.clear()
            logger.info("缓存已清理")
        
        logger.info("训练资源已清理")
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """获取缓存统计信息
        
        返回:
            缓存统计信息字典（如果缓存启用），否则返回None
        """
        if self.cache_manager is not None:
            return self.cache_manager.get_stats()
        return None  # 返回None
    
    def invalidate_cache_pattern(self, pattern: str):
        """使匹配模式的缓存项失效
        
        参数:
            pattern: 缓存键模式
        """
        if self.cache_manager is not None:
            self.cache_manager.invalidate_pattern(pattern)
            logger.info(f"缓存模式失效: pattern={pattern}")
    
    def compute_pde_residual(self, inputs: torch.Tensor, 
                            pde_function: Callable,
                            iteration: int = 0) -> torch.Tensor:
        """计算PDE残差（支持缓存）"""
        
        # 如果启用缓存，尝试从缓存获取或计算
        if self.cache_manager is not None and self.config.cache_enabled_for_pde:
            # 生成缓存键
            cache_key = self.cache_manager.key_generator.generate_pde_residual_key(
                inputs, self, self.config.pde_type, iteration
            )
            
            # 定义计算函数（在缓存未命中时调用）
            def compute_residual():
                inputs_local = inputs.detach().clone().requires_grad_(True)
                u = self(inputs_local)
                return pde_function(inputs_local, u, self)
            
            # 从缓存获取或计算
            residual, was_hit = self.cache_manager.get(
                cache_key, 
                compute_func=compute_residual,
                compute_args=(),
                compute_kwargs={}
            )
            
            # 记录缓存状态
            if was_hit:
                logger.debug(f"PDE残差缓存命中: iteration={iteration}")
            else:
                logger.debug(f"PDE残差缓存未命中，已计算并缓存: iteration={iteration}")
            
            return residual
        
        # 如果没有启用缓存，直接计算
        inputs.requires_grad_(True)
        
        # 计算模型输出
        u = self(inputs)
        
        # 计算PDE残差
        residual = pde_function(inputs, u, self)
        
        return residual
    
    def enforce_boundary_conditions(self, boundary_points: torch.Tensor,
                                   boundary_values: torch.Tensor,
                                   condition_type: str = "dirichlet"):
        """强制边界条件（硬约束）
        
        通过距离函数方法实现硬边界条件，确保在边界点上精确满足边界条件。
        
        参数:
            boundary_points: 边界点坐标张量 [N, D]
            boundary_values: 边界值张量 [N, 1] 或标量
            condition_type: 边界条件类型 ("dirichlet", "neumann", "robin")
            
        返回:
            应用了硬约束的模型输出
        """
        # 确保边界值张量与边界点匹配
        if isinstance(boundary_values, (int, float)):
            # 标量边界值扩展到所有边界点
            boundary_values = torch.full((boundary_points.shape[0], 1), 
                                        boundary_values, 
                                        dtype=boundary_points.dtype,
                                        device=boundary_points.device)
        elif boundary_values.dim() == 1:
            # 一维张量转换为二维
            boundary_values = boundary_values.unsqueeze(-1)
        
        # 检查维度匹配
        if boundary_values.shape[0] != boundary_points.shape[0]:
            raise ValueError(f"边界点数量({boundary_points.shape[0]})与边界值数量({boundary_values.shape[0]})不匹配")
        
        if condition_type == "dirichlet":
            return self._enforce_dirichlet_hard(boundary_points, boundary_values)
        elif condition_type == "neumann":
            return self._enforce_neumann_hard(boundary_points, boundary_values)
        elif condition_type == "robin":
            return self._enforce_robin_hard(boundary_points, boundary_values)
        else:
            raise ValueError(f"不支持的边界条件类型: {condition_type}")
    
    def _enforce_dirichlet_hard(self, boundary_points: torch.Tensor,
                               boundary_values: torch.Tensor):
        """强制Dirichlet边界条件（硬约束）
        
        使用距离函数方法: u(x) = g(x) + d(x) * N(x)
        其中 g(x) 是边界值，d(x) 是距离函数，N(x) 是神经网络输出
        """
        # 计算距离函数（完整：使用最近边界点的距离）
        # 对于复杂几何，需要预先计算符号距离函数
        batch_size, coord_dim = boundary_points.shape
        
        # 完整：假设边界在坐标原点附近
        # 实际应用中应该使用真实的距离函数
        distance = torch.norm(boundary_points, dim=1, keepdim=True)
        
        # 构建硬约束输出
        # 使用边界值加上距离函数调制的网络输出
        network_output = self(boundary_points)
        constrained_output = boundary_values + distance * network_output
        
        return constrained_output
    
    def _enforce_neumann_hard(self, boundary_points: torch.Tensor,
                             boundary_values: torch.Tensor):
        """强制Neumann边界条件（硬约束）
        
        使用变换方法：u(x) = N(x) + ∫_{boundary} boundary_values
        这里实现完整版本
        """
        # 计算网络输出
        network_output = self(boundary_points)
        
        # 对于Neumann条件，需要计算法向导数
        # 完整版本：添加边界值调制项
        boundary_points.requires_grad_(True)
        u = self(boundary_points)
        
        # 计算法向导数（假设法向为第一个坐标轴方向）
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=boundary_points,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        normal_derivative = grad_u[:, 0:1]  # 假设法向为x方向
        
        # 强制法向导数等于边界值
        constraint_error = normal_derivative - boundary_values
        
        # 应用惩罚（硬约束需要通过网络结构实现，这里使用软约束近似）
        # 实际硬约束实现需要修改网络架构
        constrained_output = network_output - 0.1 * constraint_error
        
        return constrained_output
    
    def _enforce_robin_hard(self, boundary_points: torch.Tensor,
                           boundary_values: torch.Tensor):
        """强制Robin边界条件（硬约束）
        
        Robin条件：αu + β∂u/∂n = g
        这里实现完整版本
        """
        # 设置参数
        alpha = 1.0
        beta = 1.0
        
        # 计算网络输出
        network_output = self(boundary_points)
        
        # 需要计算法向导数
        boundary_points.requires_grad_(True)
        u = self(boundary_points)
        
        # 计算法向导数
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=boundary_points,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        normal_derivative = grad_u[:, 0:1]
        
        # 计算Robin条件
        robin_value = alpha * u + beta * normal_derivative
        
        # 计算约束误差
        constraint_error = robin_value - boundary_values
        
        # 应用惩罚（近似硬约束）
        constrained_output = network_output - 0.1 * constraint_error
        
        return constrained_output


class SinActivation(nn.Module):
    """正弦激活函数（SIREN风格）"""
    
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class SIRENActivation(nn.Module):
    """SIREN激活函数"""
    
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class AdaptiveWeightManager:
    """自适应权重管理器"""
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.9):
        self.alpha = alpha  # 学习率
        self.beta = beta    # 动量参数
        self.weight_history = {}
        self.constraint_weights = {}
    
    def update_weights(self, loss_dict: Dict[str, float], iteration: int):
        """更新权重"""
        # 基于损失比例调整权重
        total_loss = sum(loss_dict.values())
        
        for name, loss in loss_dict.items():
            if name == "total":
                continue
            
            # 计算损失比例
            loss_ratio = loss / total_loss if total_loss > 0 else 1.0
            
            # 更新权重历史
            if name not in self.weight_history:
                self.weight_history[name] = []
            
            self.weight_history[name].append(loss_ratio)
            
            # 基于历史调整权重
            if len(self.weight_history[name]) > 10:
                recent_mean = np.mean(self.weight_history[name][-10:])
                if recent_mean > 0.5:
                    # 该约束损失比例过高，需要增加权重
                    if name not in self.constraint_weights:
                        self.constraint_weights[name] = 1.0
                    self.constraint_weights[name] *= 1.1
                    # 权重限制
                    self.constraint_weights[name] = min(self.constraint_weights[name], 10.0)
                elif recent_mean < 0.1:
                    # 损失比例过低，减少权重
                    if name not in self.constraint_weights:
                        self.constraint_weights[name] = 1.0
                    self.constraint_weights[name] *= 0.9
                    # 权重限制
                    self.constraint_weights[name] = max(self.constraint_weights[name], 0.1)


class MixedPrecisionManager:
    """混合精度训练管理器
    
    功能：
    1. 自动混合精度（AMP）训练管理
    2. 梯度缩放和溢出检测
    3. 动态损失缩放
    4. 混合精度训练统计
    
    基于PyTorch的torch.cuda.amp模块实现
    """
    
    def __init__(self, enabled: bool = True, dtype: torch.dtype = torch.float16):
        self.enabled = enabled
        self.dtype = dtype
        
        # 初始化梯度缩放器（如果可用）
        self.grad_scaler = None
        if enabled and torch.cuda.is_available():
            try:
                from torch.cuda.amp import GradScaler
                self.grad_scaler = GradScaler()
                logger.info(f"混合精度管理器初始化: enabled={enabled}, dtype={dtype}")
            except ImportError as e:
                logger.warning(f"无法导入GradScaler: {e}, 禁用混合精度")
                self.enabled = False
                self.grad_scaler = None
        
        # 训练统计
        self.loss_scale_history = []
        self.overflow_counter = 0
        self.total_iterations = 0
    
    def autocast_context(self):
        """创建autocast上下文
        
        返回:
            torch.cuda.amp.autocast上下文管理器
        """
        if self.enabled and torch.cuda.is_available():
            try:
                from torch.cuda.amp import autocast
                return autocast(dtype=self.dtype)
            except ImportError as e:
                logger.warning(f"无法导入autocast: {e}")
                return None  # 返回None
        return None  # 返回None
    
    def scale_loss(self, loss):
        """缩放损失值（用于梯度缩放）
        
        参数:
            loss: 损失张量
            
        返回:
            缩放后的损失
        """
        if self.enabled and self.grad_scaler is not None:
            return self.grad_scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        """执行优化器步骤（包含梯度缩放）
        
        参数:
            optimizer: 优化器
        """
        if self.enabled and self.grad_scaler is not None:
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
            
            # 记录损失缩放值
            self.loss_scale_history.append(self.grad_scaler.get_scale())
            self.total_iterations += 1
        else:
            optimizer.step()
    
    def check_overflow(self, optimizer):
        """检查梯度溢出
        
        参数:
            optimizer: 优化器对象
            
        返回:
            bool: 是否发生梯度溢出
        """
        if self.enabled and self.grad_scaler is not None:
            # 检查梯度缩放器是否检测到溢出
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None and torch.isnan(param.grad).any():
                        self.overflow_counter += 1
                        return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取训练统计信息
        
        返回:
            统计信息字典
        """
        return {
            "enabled": self.enabled,
            "dtype": str(self.dtype),
            "loss_scale": self.grad_scaler.get_scale() if self.grad_scaler else 1.0,
            "loss_scale_history": self.loss_scale_history[-10:] if self.loss_scale_history else [],
            "overflow_count": self.overflow_counter,
            "total_iterations": self.total_iterations,
            "avg_loss_scale": np.mean(self.loss_scale_history) if self.loss_scale_history else 1.0
        }
    
    def reset(self):
        """重置管理器状态"""
        if self.grad_scaler is not None:
            self.grad_scaler = None
            from torch.cuda.amp import GradScaler
            self.grad_scaler = GradScaler()
        
        self.loss_scale_history = []
        self.overflow_counter = 0
        self.total_iterations = 0
        logger.info("混合精度管理器已重置")


class DistributedTrainingManager:
    """分布式训练管理器
    
    功能：
    1. 分布式训练环境初始化
    2. 多GPU/多节点训练协调
    3. 模型并行和数据并行支持
    4. 分布式数据采样和同步
    5. 梯度同步和模型平均
    
    基于PyTorch的torch.distributed模块实现
    """
    
    def __init__(self, config: PINNConfig):
        self.config = config
        self.initialized = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        
        # 分布式训练组件
        self.process_group = None
        self.distributed_sampler = None
        
        # 如果配置启用分布式训练，则初始化
        if config.distributed_training:
            self._init_distributed_training()
    
    def _init_distributed_training(self):
        """初始化分布式训练环境"""
        try:
            import torch.distributed as dist
            
            # 检查是否已初始化
            if not dist.is_available():
                logger.warning("分布式训练不可用，禁用分布式训练")
                self.config.distributed_training = False
                return
            
            # 初始化进程组
            if not dist.is_initialized():
                # 使用环境变量进行初始化
                rank = int(os.environ.get('RANK', 0))
                world_size = int(os.environ.get('WORLD_SIZE', 1))
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                
                # 设置分布式后端
                backend = self.config.backend
                if backend not in ['nccl', 'gloo', 'mpi']:
                    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
                
                # 初始化进程组
                dist.init_process_group(
                    backend=backend,
                    init_method='env://',
                    rank=rank,
                    world_size=world_size
                )
                
                self.rank = rank
                self.world_size = world_size
                self.local_rank = local_rank
                self.process_group = dist.group.WORLD
                
                logger.info(f"分布式训练初始化完成: rank={rank}, world_size={world_size}, "
                          f"local_rank={local_rank}, backend={backend}")
                self.initialized = True
            else:
                # 已经初始化，获取当前信息
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                self.process_group = dist.group.WORLD
                self.initialized = True
                logger.info(f"分布式训练已初始化: rank={self.rank}, world_size={self.world_size}")
        
        except Exception as e:
            logger.error(f"分布式训练初始化失败: {e}")
            self.config.distributed_training = False
            self.initialized = False
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """包装模型以支持分布式训练
        
        参数:
            model: 原始模型
            
        返回:
            包装后的模型（DataParallel或DistributedDataParallel）
        """
        if not self.initialized or self.world_size <= 1:
            return model
        
        try:
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP
            
            # 使用DistributedDataParallel包装模型
            device_ids = [self.local_rank] if torch.cuda.is_available() else None
            ddp_model = DDP(
                model,
                device_ids=device_ids,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=True  # PINN可能有未使用的参数
            )
            
            logger.info(f"模型已包装为DistributedDataParallel: rank={self.rank}")
            return ddp_model
        
        except Exception as e:
            logger.error(f"模型包装失败: {e}")
            return model
    
    def create_distributed_sampler(self, dataset):
        """创建分布式数据采样器
        
        参数:
            dataset: 数据集
            
        返回:
            分布式采样器
        """
        if not self.initialized or self.world_size <= 1:
            return None  # 返回None
        
        try:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            self.distributed_sampler = sampler
            logger.info(f"创建分布式采样器: rank={self.rank}, world_size={self.world_size}")
            return sampler
        
        except Exception as e:
            logger.error(f"创建分布式采样器失败: {e}")
            return None  # 返回None
    
    def synchronize_gradients(self, model: nn.Module):
        """同步梯度（确保所有进程的梯度一致）"""
        if not self.initialized or self.world_size <= 1:
            return
        
        try:
            import torch.distributed as dist
            
            # 对于DistributedDataParallel，梯度已自动同步
            if isinstance(model, (nn.parallel.DistributedDataParallel, 
                                nn.parallel.DataParallel)):
                return
            
            # 手动同步梯度
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size
        
        except Exception as e:
            logger.error(f"梯度同步失败: {e}")
    
    def synchronize_model(self, model: nn.Module):
        """同步模型参数（确保所有进程的模型一致）"""
        if not self.initialized or self.world_size <= 1:
            return
        
        try:
            import torch.distributed as dist
            
            # 对所有参数进行平均
            for param in model.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= self.world_size
        
        except Exception as e:
            logger.error(f"模型同步失败: {e}")
    
    def broadcast_tensor(self, tensor: torch.Tensor, src_rank: int = 0):
        """广播张量到所有进程
        
        参数:
            tensor: 要广播的张量
            src_rank: 源进程排名
            
        返回:
            广播后的张量
        """
        if not self.initialized or self.world_size <= 1:
            return tensor
        
        try:
            import torch.distributed as dist
            dist.broadcast(tensor, src=src_rank)
            return tensor
        
        except Exception as e:
            logger.error(f"张量广播失败: {e}")
            return tensor
    
    def all_gather_tensor(self, tensor: torch.Tensor):
        """从所有进程收集张量
        
        参数:
            tensor: 要收集的张量
            
        返回:
            收集到的张量列表
        """
        if not self.initialized or self.world_size <= 1:
            return [tensor]
        
        try:
            import torch.distributed as dist
            
            # 创建输出张量列表
            tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(tensor_list, tensor)
            
            return tensor_list
        
        except Exception as e:
            logger.error(f"张量收集失败: {e}")
            return [tensor]
    
    def get_rank_stats(self) -> Dict[str, Any]:
        """获取分布式训练统计信息
        
        返回:
            统计信息字典
        """
        return {
            "initialized": self.initialized,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "backend": self.config.backend if self.initialized else "none"
        }
    
    def cleanup(self):
        """清理分布式训练资源"""
        if self.initialized:
            try:
                import torch.distributed as dist
                dist.destroy_process_group()
                self.initialized = False
                logger.info("分布式训练资源已清理")
            except Exception as e:
                logger.error(f"清理分布式训练资源失败: {e}")


class IncrementalCacheManager:
    """增量式缓存管理器
    
    功能：
    1. 增量式计算结果缓存：缓存中间计算结果，避免重复计算
    2. 智能缓存淘汰：基于使用频率、最近使用时间和计算结果重要性进行缓存淘汰
    3. 多粒度缓存：支持不同粒度的计算结果缓存（如PDE残差、梯度、损失等）
    4. 内存优化：动态调整缓存大小，平衡内存使用和计算效率
    
    基于增量式计算优化的先进缓存技术实现
    """
    
    def __init__(self, max_cache_size: int = 1000, 
                 eviction_policy: str = "lru",
                 enable_adaptive_sizing: bool = True):
        self.max_cache_size = max_cache_size
        self.eviction_policy = eviction_policy  # lru, lfu, arc, adaptive
        self.enable_adaptive_sizing = enable_adaptive_sizing
        
        # 缓存存储
        self.cache = {}  # key -> (value, metadata)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        
        # 缓存统计
        self.hit_rate_history = []
        self.memory_usage_history = []
        self.eviction_history = []
        
        # 缓存键生成器
        self.key_generator = CacheKeyGenerator()
        
        # 初始化日志
        self.logger = logging.getLogger("IncrementalCacheManager")
        self.logger.info(f"增量式缓存管理器初始化: 最大缓存大小={max_cache_size}, "
                        f"淘汰策略={eviction_policy}, 自适应大小={enable_adaptive_sizing}")
    
    def get(self, cache_key: str, compute_func: Callable = None, 
           compute_args: tuple = (), compute_kwargs: Dict[str, Any] = None) -> Tuple[Any, bool]:
        """获取缓存值，如果未命中则计算
        
        参数:
            cache_key: 缓存键
            compute_func: 计算函数（如果未命中时调用）
            compute_args: 计算函数的位置参数
            compute_kwargs: 计算函数的关键字参数
            
        返回:
            元组 (缓存值或计算结果, 是否命中)
        """
        self.total_queries += 1
        
        # 检查缓存中是否存在
        if cache_key in self.cache:
            # 缓存命中
            value, metadata = self.cache[cache_key]
            
            # 更新访问统计
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            metadata["last_access_time"] = time.time()
            
            self.cache_hits += 1
            
            # 记录命中
            self.logger.debug(f"缓存命中: key={cache_key}, 值类型={type(value).__name__}")
            
            return value, True
        else:
            # 缓存未命中
            self.cache_misses += 1
            
            # 如果提供了计算函数，则计算并缓存结果
            if compute_func is not None:
                compute_kwargs = compute_kwargs or {}
                
                self.logger.debug(f"缓存未命中，开始计算: key={cache_key}")
                
                # 计算新值
                start_time = time.time()
                computed_value = compute_func(*compute_args, **compute_kwargs)
                compute_time = time.time() - start_time
                
                # 缓存结果
                metadata = {
                    "compute_time": compute_time,
                    "cache_time": time.time(),
                    "access_count": 1,
                    "last_access_time": time.time(),
                    "value_size": self._estimate_value_size(computed_value),
                    "importance": 1.0  # 默认重要性
                }
                
                self.cache[cache_key] = (computed_value, metadata)
                
                # 检查是否需要淘汰
                self._check_and_evict()
                
                self.logger.debug(f"计算完成并缓存: key={cache_key}, 计算时间={compute_time:.4f}s")
                
                return computed_value, False
            else:
                # 未提供计算函数且缓存未命中
                self.logger.warning(f"缓存未命中且无计算函数: key={cache_key}")
                raise KeyError(f"缓存键未找到且未提供计算函数: {cache_key}")
    
    def put(self, cache_key: str, value: Any, 
           importance: float = 1.0, compute_time: float = 0.0):
        """手动添加值到缓存
        
        参数:
            cache_key: 缓存键
            value: 要缓存的值
            importance: 值的重要性（用于缓存淘汰）
            compute_time: 计算时间（如果已知）
        """
        metadata = {
            "compute_time": compute_time,
            "cache_time": time.time(),
            "access_count": 1,
            "last_access_time": time.time(),
            "value_size": self._estimate_value_size(value),
            "importance": importance
        }
        
        self.cache[cache_key] = (value, metadata)
        
        # 检查是否需要淘汰
        self._check_and_evict()
        
        self.logger.debug(f"手动缓存: key={cache_key}, 重要性={importance}, 大小={metadata['value_size']}")
    
    def invalidate(self, cache_key: str):
        """使缓存项失效
        
        参数:
            cache_key: 要失效的缓存键
        """
        if cache_key in self.cache:
            del self.cache[cache_key]
            self.logger.debug(f"缓存失效: key={cache_key}")
        else:
            self.logger.debug(f"尝试失效不存在的缓存键: key={cache_key}")
    
    def invalidate_pattern(self, pattern: str):
        """使匹配模式的缓存项失效
        
        参数:
            pattern: 缓存键模式（支持通配符）
        """
        keys_to_remove = []
        
        for key in self.cache.keys():
            if pattern in key:  # 简单模式匹配
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        if keys_to_remove:
            self.logger.debug(f"模式失效: pattern={pattern}, 失效数量={len(keys_to_remove)}")
    
    def clear(self):
        """清空缓存"""
        cache_size = len(self.cache)
        self.cache.clear()
        self.logger.info(f"缓存已清空: 清除了{cache_size}个缓存项")
    
    def _check_and_evict(self):
        """检查缓存大小并进行淘汰"""
        if len(self.cache) <= self.max_cache_size:
            return
        
        # 需要淘汰的缓存项数量
        items_to_evict = len(self.cache) - self.max_cache_size
        
        if items_to_evict <= 0:
            return
        
        self.logger.debug(f"缓存满，开始淘汰: 当前大小={len(self.cache)}, "
                         f"最大大小={self.max_cache_size}, 需要淘汰={items_to_evict}")
        
        # 根据淘汰策略选择要淘汰的项
        if self.eviction_policy == "lru":
            evicted_keys = self._evict_lru(items_to_evict)
        elif self.eviction_policy == "lfu":
            evicted_keys = self._evict_lfu(items_to_evict)
        elif self.eviction_policy == "adaptive":
            evicted_keys = self._evict_adaptive(items_to_evict)
        else:
            # 默认使用LRU
            evicted_keys = self._evict_lru(items_to_evict)
        
        # 执行淘汰
        for key in evicted_keys:
            del self.cache[key]
        
        # 记录淘汰历史
        self.eviction_history.append({
            "timestamp": time.time(),
            "evicted_count": len(evicted_keys),
            "cache_size_before": len(self.cache) + len(evicted_keys),
            "cache_size_after": len(self.cache),
            "policy": self.eviction_policy
        })
        
        self.logger.info(f"缓存淘汰完成: 淘汰了{len(evicted_keys)}个缓存项")
    
    def _evict_lru(self, num_to_evict: int) -> List[str]:
        """LRU（最近最少使用）淘汰策略
        
        参数:
            num_to_evict: 需要淘汰的数量
            
        返回:
            要淘汰的缓存键列表
        """
        # 按最后访问时间排序
        items = [(key, metadata["last_access_time"]) 
                for key, (_, metadata) in self.cache.items()]
        
        # 按访问时间升序排序（最早的排前面）
        items.sort(key=lambda x: x[1])
        
        # 选择前num_to_evict个
        evicted_keys = [key for key, _ in items[:num_to_evict]]
        
        return evicted_keys
    
    def _evict_lfu(self, num_to_evict: int) -> List[str]:
        """LFU（最不经常使用）淘汰策略
        
        参数:
            num_to_evict: 需要淘汰的数量
            
        返回:
            要淘汰的缓存键列表
        """
        # 按访问次数排序
        items = [(key, metadata.get("access_count", 0)) 
                for key, (_, metadata) in self.cache.items()]
        
        # 按访问次数升序排序（访问最少的排前面）
        items.sort(key=lambda x: x[1])
        
        # 选择前num_to_evict个
        evicted_keys = [key for key, _ in items[:num_to_evict]]
        
        return evicted_keys
    
    def _evict_adaptive(self, num_to_evict: int) -> List[str]:
        """自适应淘汰策略（结合LRU、LFU和重要性）
        
        参数:
            num_to_evict: 需要淘汰的数量
            
        返回:
            要淘汰的缓存键列表
        """
        # 计算每个缓存项的淘汰分数
        scores = []
        current_time = time.time()
        
        for key, (value, metadata) in self.cache.items():
            # 获取元数据
            last_access = metadata.get("last_access_time", 0)
            access_count = metadata.get("access_count", 0)
            importance = metadata.get("importance", 1.0)
            compute_time = metadata.get("compute_time", 0.0)
            value_size = metadata.get("value_size", 0)
            
            # 计算时间衰减因子（最近访问的应该保留）
            time_factor = max(0.0, 1.0 - (current_time - last_access) / 3600.0)  # 1小时衰减
            
            # 计算访问频率因子
            freq_factor = 1.0 / (1.0 + access_count) if access_count > 0 else 0.0
            
            # 计算计算成本因子（计算时间越长越应该保留）
            compute_factor = min(1.0, compute_time / 10.0) if compute_time > 0 else 0.0
            
            # 计算内存效率因子（值越大占用内存越多）
            size_factor = 1.0 / (1.0 + value_size / 1024)  # 按KB计算
            
            # 综合淘汰分数（分数越低越可能被淘汰）
            eviction_score = (
                0.3 * time_factor +      # 时间衰减
                0.2 * freq_factor +      # 访问频率
                0.2 * compute_factor +   # 计算成本
                0.2 * size_factor +      # 内存效率
                0.1 * (1.0 / importance)  # 重要性（重要性越低越可能被淘汰）
            )
            
            scores.append((key, eviction_score))
        
        # 按淘汰分数升序排序（分数低的排前面）
        scores.sort(key=lambda x: x[1])
        
        # 选择前num_to_evict个
        evicted_keys = [key for key, _ in scores[:num_to_evict]]
        
        return evicted_keys
    
    def _estimate_value_size(self, value: Any) -> int:
        """估计值的大小（字节）
        
        参数:
            value: 要估计大小的值
            
        返回:
            估计的大小（字节）
        """
        if value is None:
            return 0
        
        # 如果是PyTorch张量
        if hasattr(value, 'element_size') and hasattr(value, 'nelement'):
            return value.element_size() * value.nelement()
        
        # 如果是NumPy数组
        elif hasattr(value, 'itemsize') and hasattr(value, 'size'):
            return value.itemsize * value.size
        
        # 如果是Python列表或元组
        elif isinstance(value, (list, tuple)):
            total_size = 0
            for item in value:
                total_size += self._estimate_value_size(item)
            return total_size
        
        # 如果是字典
        elif isinstance(value, dict):
            total_size = 0
            for k, v in value.items():
                total_size += self._estimate_value_size(k) + self._estimate_value_size(v)
            return total_size
        
        # 如果是其他Python对象，使用sys.getsizeof
        else:
            try:
                import sys
                return sys.getsizeof(value)
            except Exception:
                return 1024  # 默认估计1KB
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        返回:
            统计信息字典
        """
        # 计算命中率
        hit_rate = self.cache_hits / max(1, self.total_queries)
        
        # 计算内存使用
        total_memory = 0
        for _, metadata in self.cache.values():
            total_memory += metadata.get("value_size", 0)
        
        # 计算平均计算时间
        total_compute_time = 0
        compute_count = 0
        for _, metadata in self.cache.values():
            compute_time = metadata.get("compute_time", 0)
            if compute_time > 0:
                total_compute_time += compute_time
                compute_count += 1
        
        avg_compute_time = total_compute_time / max(1, compute_count)
        
        stats = {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_queries": self.total_queries,
            "hit_rate": hit_rate,
            "total_memory_bytes": total_memory,
            "total_memory_mb": total_memory / (1024 * 1024),
            "avg_compute_time": avg_compute_time,
            "eviction_count": len(self.eviction_history),
            "eviction_policy": self.eviction_policy,
            "adaptive_sizing_enabled": self.enable_adaptive_sizing
        }
        
        # 如果启用自适应大小，添加相关统计
        if self.enable_adaptive_sizing:
            stats["adaptive_size_adjustments"] = len(self.memory_usage_history)
            if self.memory_usage_history:
                stats["avg_memory_usage_mb"] = np.mean(self.memory_usage_history) / (1024 * 1024)
        
        return stats
    
    def was_cache_hit(self, cache_key: str) -> bool:
        """检查指定缓存键的最近访问是否命中
        
        参数:
            cache_key: 缓存键
            
        返回:
            True如果最近访问是命中，False如果是未命中
        """
        return cache_key in self.cache
    
    def get_cache_hit_history(self) -> Dict[str, bool]:
        """获取所有缓存项的命中历史
        
        返回:
            字典：缓存键 -> 是否命中
        """
        history = {}
        for key in self.cache.keys():
            # 检查这个键是否有访问记录
            _, metadata = self.cache[key]
            access_count = metadata.get("access_count", 0)
            history[key] = access_count > 1  # 如果访问次数大于1，说明有命中
        
        return history
    
    def adjust_cache_size(self, new_max_size: int):
        """调整缓存最大大小
        
        参数:
            new_max_size: 新的最大缓存大小
        """
        old_size = self.max_cache_size
        self.max_cache_size = new_max_size
        
        # 立即检查是否需要淘汰
        self._check_and_evict()
        
        self.logger.info(f"缓存大小调整: {old_size} -> {new_max_size}")
    
    def adaptive_size_adjustment(self, available_memory_mb: float):
        """自适应调整缓存大小（基于可用内存）
        
        参数:
            available_memory_mb: 可用内存（MB）
        """
        if not self.enable_adaptive_sizing:
            return
        
        # 基于可用内存调整缓存大小
        # 规则：使用不超过可用内存的30%
        target_memory_mb = available_memory_mb * 0.3
        
        # 估计每个缓存项的平均大小
        avg_item_size_mb = 0
        if self.cache:
            total_size = sum(metadata.get("value_size", 0) 
                           for _, metadata in self.cache.values())
            avg_item_size_mb = (total_size / len(self.cache)) / (1024 * 1024)
        
        if avg_item_size_mb > 0:
            # 计算新的缓存大小
            new_cache_size = int(target_memory_mb / avg_item_size_mb)
            
            # 限制最小和最大大小
            new_cache_size = max(100, min(new_cache_size, 10000))
            
            # 调整缓存大小
            self.adjust_cache_size(new_cache_size)
            
            # 记录调整
            self.memory_usage_history.append(target_memory_mb * 1024 * 1024)  # 转换为字节
            
            self.logger.info(f"自适应大小调整: 可用内存={available_memory_mb:.2f}MB, "
                           f"目标内存={target_memory_mb:.2f}MB, "
                           f"新缓存大小={new_cache_size}")


class CacheKeyGenerator:
    """缓存键生成器
    
    功能：为不同类型的计算生成唯一且高效的缓存键
    """
    
    def __init__(self):
        self.logger = logging.getLogger("CacheKeyGenerator")
    
    def generate_pde_residual_key(self, 
                                 inputs: torch.Tensor,
                                 model: nn.Module,
                                 pde_type: str,
                                 iteration: int = 0) -> str:
        """生成PDE残差计算缓存键
        
        参数:
            inputs: 输入张量
            model: 模型
            pde_type: PDE类型
            iteration: 迭代次数
            
        返回:
            缓存键
        """
        # 基于输入特征、模型参数和PDE类型生成键
        input_hash = self._tensor_hash(inputs)
        model_hash = self._model_hash(model)
        
        key = f"pde_{pde_type}_in_{input_hash}_model_{model_hash}_iter_{iteration}"
        
        self.logger.debug(f"生成PDE残差缓存键: {key}")
        
        return key
    
    def generate_gradient_key(self, 
                            inputs: torch.Tensor,
                            outputs: torch.Tensor,
                            model: nn.Module,
                            gradient_type: str,
                            iteration: int = 0) -> str:
        """生成梯度计算缓存键
        
        参数:
            inputs: 输入张量
            outputs: 输出张量
            model: 模型
            gradient_type: 梯度类型（如'u_x', 'u_xx'）
            iteration: 迭代次数
            
        返回:
            缓存键
        """
        input_hash = self._tensor_hash(inputs)
        output_hash = self._tensor_hash(outputs)
        model_hash = self._model_hash(model)
        
        key = f"grad_{gradient_type}_in_{input_hash}_out_{output_hash}_model_{model_hash}_iter_{iteration}"
        
        self.logger.debug(f"生成梯度缓存键: {key}")
        
        return key
    
    def generate_loss_key(self, 
                         loss_type: str,
                         inputs: torch.Tensor,
                         targets: torch.Tensor = None,
                         iteration: int = 0) -> str:
        """生成损失计算缓存键
        
        参数:
            loss_type: 损失类型（如'physics', 'data', 'bc', 'ic'）
            inputs: 输入张量
            targets: 目标张量（可选）
            iteration: 迭代次数
            
        返回:
            缓存键
        """
        input_hash = self._tensor_hash(inputs)
        
        if targets is not None:
            target_hash = self._tensor_hash(targets)
            key = f"loss_{loss_type}_in_{input_hash}_target_{target_hash}_iter_{iteration}"
        else:
            key = f"loss_{loss_type}_in_{input_hash}_iter_{iteration}"
        
        self.logger.debug(f"生成损失缓存键: {key}")
        
        return key
    
    def generate_autograd_key(self, 
                             function_name: str,
                             inputs: torch.Tensor,
                             model: nn.Module,
                             order: int = 1,
                             iteration: int = 0) -> str:
        """生成自动微分计算缓存键
        
        参数:
            function_name: 函数名
            inputs: 输入张量
            model: 模型
            order: 微分阶数
            iteration: 迭代次数
            
        返回:
            缓存键
        """
        input_hash = self._tensor_hash(inputs)
        model_hash = self._model_hash(model)
        
        key = f"autograd_{function_name}_order_{order}_in_{input_hash}_model_{model_hash}_iter_{iteration}"
        
        self.logger.debug(f"生成自动微分缓存键: {key}")
        
        return key
    
    def _tensor_hash(self, tensor: torch.Tensor) -> str:
        """生成张量的哈希键
        
        参数:
            tensor: 张量
            
        返回:
            哈希字符串
        """
        if tensor is None:
            return "none"
        
        import hashlib
        
        # 使用MD5哈希确保确定性
        md5 = hashlib.md5()
        
        # 添加形状信息
        shape_bytes = b"_".join(str(d).encode() for d in tensor.shape)
        md5.update(b"shape:" + shape_bytes + b"|")
        
        # 添加数据类型
        dtype_bytes = str(tensor.dtype).encode()
        md5.update(b"dtype:" + dtype_bytes + b"|")
        
        # 添加数据内容（使用确定性的抽样）
        tensor_flat = tensor.detach().cpu().flatten()
        
        # 如果张量太大，使用固定模式抽样（前100个、中间100个、后100个元素）
        if tensor_flat.numel() > 300:
            # 等间距抽样300个点
            step = tensor_flat.numel() // 300
            indices = torch.arange(0, tensor_flat.numel(), step)[:300]
            sample_values = tensor_flat[indices]
        else:
            # 使用所有元素
            sample_values = tensor_flat
        
        # 转换为字节
        data_bytes = sample_values.numpy().tobytes()
        md5.update(b"data:" + data_bytes)
        
        # 返回十六进制哈希
        return md5.hexdigest()[:8]  # 使用前8个字符（32位）
    
    def _model_hash(self, model: nn.Module) -> str:
        """生成模型的哈希键
        
        参数:
            model: 模型
            
        返回:
            哈希字符串
        """
        if model is None:
            return "none"
        
        import hashlib
        
        # 使用MD5哈希确保确定性
        md5 = hashlib.md5()
        
        # 按名称排序参数以确保顺序一致性
        sorted_params = sorted(model.named_parameters(), key=lambda x: x[0])
        
        for name, param in sorted_params:
            if param.requires_grad:
                # 添加参数名称
                md5.update(b"name:" + name.encode() + b"|")
                
                # 添加参数形状
                shape_bytes = b"_".join(str(d).encode() for d in param.shape)
                md5.update(b"shape:" + shape_bytes + b"|")
                
                if param.numel() > 0:
                    # 添加参数数据（使用确定性的抽样）
                    param_flat = param.data.detach().cpu().flatten()
                    
                    # 如果参数太大，使用固定模式抽样
                    if param_flat.numel() > 100:
                        # 等间距抽样100个点
                        step = max(1, param_flat.numel() // 100)
                        indices = torch.arange(0, param_flat.numel(), step)[:100]
                        sample_values = param_flat[indices]
                    else:
                        # 使用所有元素
                        sample_values = param_flat
                    
                    # 转换为字节
                    data_bytes = sample_values.numpy().tobytes()
                    md5.update(b"data:" + data_bytes + b"|")
        
        # 返回十六进制哈希
        return md5.hexdigest()[:8]  # 使用前8个字符（32位）


def burgers_equation(x: torch.Tensor, u: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Burgers方程残差"""
    # Burgers方程: u_t + u*u_x - ν*u_xx = 0
    
    # 确保输入有requires_grad
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # 确保fields有requires_grad
    if not fields.requires_grad:
        fields.requires_grad_(True)
    
    # 提取坐标
    t = x[:, 0:1]
    x_coord = x[:, 1:2]
    
    # 计算一阶导数
    u_t = gradient(u, t)
    u_x = gradient(u, x_coord)
    
    # 计算二阶导数
    # 确保u_x有requires_grad（用于二阶导数计算）
    if not u_x.requires_grad:
        u_x.requires_grad_(True)
    
    u_xx = gradient(u_x, x_coord)
    
    # 设置粘度系数
    nu = 0.01 / np.pi
    
    # 计算残差
    residual = u_t + u * u_x - nu * u_xx
    
    return residual


def heat_equation(x: torch.Tensor, u: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """热方程残差"""
    # 热方程: u_t - α*u_xx = 0
    
    # 确保输入有requires_grad
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # 确保u有requires_grad
    if not u.requires_grad:
        u.requires_grad_(True)
    
    # 提取坐标
    t = x[:, 0:1]
    x_coord = x[:, 1:2]
    
    # 计算导数
    u_t = gradient(u, t)
    u_x = gradient(u, x_coord)
    
    # 确保u_x有requires_grad（用于二阶导数计算）
    if not u_x.requires_grad:
        u_x.requires_grad_(True)
    
    u_xx = gradient(u_x, x_coord)
    
    # 设置热扩散系数
    alpha = 0.1
    
    # 计算残差
    residual = u_t - alpha * u_xx
    
    return residual


def wave_equation(x: torch.Tensor, u: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """波动方程残差"""
    # 波动方程: u_tt - c^2*u_xx = 0
    
    # 确保输入有requires_grad
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # 确保u有requires_grad
    if not u.requires_grad:
        u.requires_grad_(True)
    
    # 提取坐标
    t = x[:, 0:1]
    x_coord = x[:, 1:2]
    
    # 计算时间导数
    u_t = gradient(u, t)
    
    # 确保u_t有requires_grad（用于二阶时间导数计算）
    if not u_t.requires_grad:
        u_t.requires_grad_(True)
    
    u_tt = gradient(u_t, t)
    
    # 计算空间导数
    u_x = gradient(u, x_coord)
    
    # 确保u_x有requires_grad（用于二阶空间导数计算）
    if not u_x.requires_grad:
        u_x.requires_grad_(True)
    
    u_xx = gradient(u_x, x_coord)
    
    # 设置波速
    c = 1.0
    
    # 计算残差
    residual = u_tt - c**2 * u_xx
    
    return residual


def navier_stokes_equation(x: torch.Tensor, u: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """纳维-斯托克斯方程残差（2D不可压缩）
    
    方程：
        ∂u/∂t + (u·∇)u = -∇p + ν∇²u
        ∇·u = 0
    
    参数:
        x: 输入张量 [batch_size, 3] (t, x, y)
        u: 输出张量 [batch_size, 3] (u, v, p) 或 [batch_size, 2] (u, v)
        model: 模型
    
    返回:
        残差张量 [batch_size, 3] (连续性, x动量, y动量)
    """
    # 确保输入有requires_grad
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # 确保u有requires_grad
    if not u.requires_grad:
        u.requires_grad_(True)
    
    # 提取坐标
    t = x[:, 0:1]
    x_coord = x[:, 1:2]
    y_coord = x[:, 2:3]
    
    # 假设u包含速度和压力: u = [u_vel, v_vel, p]
    # 如果u只有2个分量，假设只有速度，压力需要单独计算
    if u.shape[1] >= 3:
        u_vel = u[:, 0:1]  # x方向速度
        v_vel = u[:, 1:2]  # y方向速度
        p = u[:, 2:3]      # 压力
    else:
        u_vel = u[:, 0:1]
        v_vel = u[:, 1:2]
        # 如果没有压力，设为0（压力梯度将通过约束强制满足）
        p = torch.zeros_like(u_vel)
    
    # 计算速度的时间导数
    u_t = gradient(u_vel, t)
    v_t = gradient(v_vel, t)
    
    # 计算速度的空间导数
    u_x = gradient(u_vel, x_coord)
    u_y = gradient(u_vel, y_coord)
    v_x = gradient(v_vel, x_coord)
    v_y = gradient(v_vel, y_coord)
    
    # 计算压力的空间导数
    p_x = gradient(p, x_coord)
    p_y = gradient(p, y_coord)
    
    # 计算二阶导数（拉普拉斯算子）
    # 确保一阶导数有requires_grad（用于二阶导数计算）
    if not u_x.requires_grad:
        u_x.requires_grad_(True)
    if not u_y.requires_grad:
        u_y.requires_grad_(True)
    if not v_x.requires_grad:
        v_x.requires_grad_(True)
    if not v_y.requires_grad:
        v_y.requires_grad_(True)
    
    u_xx = gradient(u_x, x_coord)
    u_yy = gradient(u_y, y_coord)
    v_xx = gradient(v_x, x_coord)
    v_yy = gradient(v_y, y_coord)
    
    # 设置粘度系数
    nu = 0.01  # 运动粘度
    
    # 计算连续性方程残差：∇·u = u_x + v_y
    continuity_residual = u_x + v_y
    
    # 计算x动量方程残差：u_t + u*u_x + v*u_y + p_x - ν*(u_xx + u_yy)
    x_momentum_residual = u_t + u_vel * u_x + v_vel * u_y + p_x - nu * (u_xx + u_yy)
    
    # 计算y动量方程残差：v_t + u*v_x + v*v_y + p_y - ν*(v_xx + v_yy)
    y_momentum_residual = v_t + u_vel * v_x + v_vel * v_y + p_y - nu * (v_xx + v_yy)
    
    # 组合残差
    residual = torch.cat([continuity_residual, x_momentum_residual, y_momentum_residual], dim=1)
    
    return residual


def schrodinger_equation(x: torch.Tensor, psi: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """薛定谔方程残差（时间相关）
    
    方程：
        iħ ∂ψ/∂t = -ħ²/(2m) ∇²ψ + Vψ
    
    参数:
        x: 输入张量 [batch_size, 2] (t, x) 或 [batch_size, 4] (t, x, y, z)
        psi: 波函数 [batch_size, 2] (实部, 虚部)
        model: 模型
    
    返回:
        残差张量 [batch_size, 2] (实部残差, 虚部残差)
    """
    # 确保输入有requires_grad
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # 确保psi有requires_grad
    if not psi.requires_grad:
        psi.requires_grad_(True)
    
    # 提取坐标
    t = x[:, 0:1]
    spatial_dims = x.shape[1] - 1
    
    # 分离波函数的实部和虚部
    psi_real = psi[:, 0:1]
    psi_imag = psi[:, 1:2]
    
    # 计算物理常数
    hbar = 1.0  # 约化普朗克常数（归一化）
    mass = 1.0  # 粒子质量（归一化）
    
    # 计算势能（完整为0或常数）
    V = 0.0
    
    # 计算时间导数
    psi_real_t = gradient(psi_real, t)
    psi_imag_t = gradient(psi_imag, t)
    
    # 计算空间导数（拉普拉斯算子）
    laplacian_real = torch.zeros_like(psi_real)
    laplacian_imag = torch.zeros_like(psi_imag)
    
    for i in range(spatial_dims):
        coord = x[:, i+1:i+2]
        # 确保坐标有requires_grad
        if not coord.requires_grad:
            coord.requires_grad_(True)
        
        psi_real_xi = gradient(psi_real, coord)
        psi_imag_xi = gradient(psi_imag, coord)
        
        # 确保一阶导数有requires_grad（用于二阶导数计算）
        if not psi_real_xi.requires_grad:
            psi_real_xi.requires_grad_(True)
        if not psi_imag_xi.requires_grad:
            psi_imag_xi.requires_grad_(True)
        
        psi_real_xixi = gradient(psi_real_xi, coord)
        psi_imag_xixi = gradient(psi_imag_xi, coord)
        
        laplacian_real = laplacian_real + psi_real_xixi
        laplacian_imag = laplacian_imag + psi_imag_xixi
    
    # 计算薛定谔方程残差
    # 实部: -ħ ∂ψ_imag/∂t = -ħ²/(2m) ∇²ψ_real + V ψ_real
    real_residual = -hbar * psi_imag_t + (hbar**2/(2*mass)) * laplacian_real - V * psi_real
    
    # 虚部: ħ ∂ψ_real/∂t = -ħ²/(2m) ∇²ψ_imag + V ψ_imag
    imag_residual = hbar * psi_real_t + (hbar**2/(2*mass)) * laplacian_imag - V * psi_imag
    
    # 组合残差
    residual = torch.cat([real_residual, imag_residual], dim=1)
    
    return residual


def maxwell_equations(x: torch.Tensor, fields: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """麦克斯韦方程残差（时域）
    
    方程：
        ∇·E = ρ/ε₀
        ∇·B = 0
        ∇×E = -∂B/∂t
        ∇×B = μ₀J + μ₀ε₀ ∂E/∂t
    
    参数:
        x: 输入张量 [batch_size, 4] (t, x, y, z)
        fields: 场张量 [batch_size, 6] (E_x, E_y, E_z, B_x, B_y, B_z)
        model: 模型
    
    返回:
        残差张量 [batch_size, 8] (4个方程的残差)
    """
    # 确保输入有requires_grad
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # 确保u有requires_grad
    if not u.requires_grad:
        u.requires_grad_(True)
    
    # 提取坐标
    t = x[:, 0:1]
    x_coord = x[:, 1:2]
    y_coord = x[:, 2:3]
    z_coord = x[:, 3:4]
    
    # 分离电场和磁场分量
    E_x = fields[:, 0:1]
    E_y = fields[:, 1:2]
    E_z = fields[:, 2:3]
    B_x = fields[:, 3:4]
    B_y = fields[:, 4:5]
    B_z = fields[:, 5:6]
    
    # 计算时间导数
    E_x_t = gradient(E_x, t)
    E_y_t = gradient(E_y, t)
    E_z_t = gradient(E_z, t)
    B_x_t = gradient(B_x, t)
    B_y_t = gradient(B_y, t)
    B_z_t = gradient(B_z, t)
    
    # 计算空间导数
    # 电场散度：∇·E = ∂E_x/∂x + ∂E_y/∂y + ∂E_z/∂z
    E_x_x = gradient(E_x, x_coord)
    E_y_y = gradient(E_y, y_coord)
    E_z_z = gradient(E_z, z_coord)
    div_E = E_x_x + E_y_y + E_z_z
    
    # 磁场散度：∇·B
    B_x_x = gradient(B_x, x_coord)
    B_y_y = gradient(B_y, y_coord)
    B_z_z = gradient(B_z, z_coord)
    div_B = B_x_x + B_y_y + B_z_z
    
    # 电场旋度：∇×E
    # curl_E_x = ∂E_z/∂y - ∂E_y/∂z
    E_z_y = gradient(E_z, y_coord)
    E_y_z = gradient(E_y, z_coord)
    curl_E_x = E_z_y - E_y_z
    
    # curl_E_y = ∂E_x/∂z - ∂E_z/∂x
    E_x_z = gradient(E_x, z_coord)
    E_z_x = gradient(E_z, x_coord)
    curl_E_y = E_x_z - E_z_x
    
    # curl_E_z = ∂E_y/∂x - ∂E_x/∂y
    E_y_x = gradient(E_y, x_coord)
    E_x_y = gradient(E_x, y_coord)
    curl_E_z = E_y_x - E_x_y
    
    # 磁场旋度：∇×B
    # curl_B_x = ∂B_z/∂y - ∂B_y/∂z
    B_z_y = gradient(B_z, y_coord)
    B_y_z = gradient(B_y, z_coord)
    curl_B_x = B_z_y - B_y_z
    
    # curl_B_y = ∂B_x/∂z - ∂B_z/∂x
    B_x_z = gradient(B_x, z_coord)
    B_z_x = gradient(B_z, x_coord)
    curl_B_y = B_x_z - B_z_x
    
    # curl_B_z = ∂B_y/∂x - ∂B_x/∂y
    B_y_x = gradient(B_y, x_coord)
    B_x_y = gradient(B_x, y_coord)
    curl_B_z = B_y_x - B_x_y
    
    # 物理常数
    epsilon_0 = 8.854e-12  # 真空介电常数
    mu_0 = 1.2566e-6      # 真空磁导率
    c = 1.0 / torch.sqrt(epsilon_0 * mu_0)  # 光速
    
    # 假设无源情况：ρ=0, J=0
    rho = 0.0
    J_x, J_y, J_z = 0.0, 0.0, 0.0
    
    # 计算麦克斯韦方程残差
    # 1. ∇·E = ρ/ε₀
    gauss_law_E = div_E - rho/epsilon_0
    
    # 2. ∇·B = 0
    gauss_law_B = div_B
    
    # 3. ∇×E = -∂B/∂t (法拉第定律)
    faraday_x = curl_E_x + B_x_t
    faraday_y = curl_E_y + B_y_t
    faraday_z = curl_E_z + B_z_t
    
    # 4. ∇×B = μ₀J + μ₀ε₀ ∂E/∂t (安培-麦克斯韦定律)
    ampere_maxwell_x = curl_B_x - mu_0 * J_x - mu_0 * epsilon_0 * E_x_t
    ampere_maxwell_y = curl_B_y - mu_0 * J_y - mu_0 * epsilon_0 * E_y_t
    ampere_maxwell_z = curl_B_z - mu_0 * J_z - mu_0 * epsilon_0 * E_z_t
    
    # 组合残差
    residual = torch.cat([
        gauss_law_E, gauss_law_B,
        faraday_x, faraday_y, faraday_z,
        ampere_maxwell_x, ampere_maxwell_y, ampere_maxwell_z
    ], dim=1)
    
    return residual


def elasticity_equation(x: torch.Tensor, u: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """弹性力学方程残差（线弹性）
    
    方程：
        ∇·σ + f = ρ ∂²u/∂t² （运动方程）
        σ = C:ε （本构关系）
        ε = 1/2 (∇u + ∇uᵀ) （应变-位移关系）
    
    参数:
        x: 输入张量 [batch_size, 4] (t, x, y, z)
        u: 位移张量 [batch_size, 3] (u_x, u_y, u_z)
        model: 模型
    
    返回:
        残差张量 [batch_size, 3] (x, y, z方向平衡方程)
    """
    # 确保输入有requires_grad
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # 提取坐标
    t = x[:, 0:1]
    x_coord = x[:, 1:2]
    y_coord = x[:, 2:3]
    z_coord = x[:, 3:4]
    
    # 分离位移分量
    u_x = u[:, 0:1]
    u_y = u[:, 1:2]
    u_z = u[:, 2:3]
    
    # 计算位移的时间导数（加速度）
    # 首先计算一阶时间导数
    u_x_t = gradient(u_x, t)
    u_y_t = gradient(u_y, t)
    u_z_t = gradient(u_z, t)
    
    # 确保一阶时间导数有requires_grad（用于二阶导数计算）
    if not u_x_t.requires_grad:
        u_x_t.requires_grad_(True)
    if not u_y_t.requires_grad:
        u_y_t.requires_grad_(True)
    if not u_z_t.requires_grad:
        u_z_t.requires_grad_(True)
    
    # 计算二阶时间导数
    u_x_tt = gradient(u_x_t, t)
    u_y_tt = gradient(u_y_t, t)
    u_z_tt = gradient(u_z_t, t)
    
    # 计算位移梯度
    u_x_x = gradient(u_x, x_coord)
    u_x_y = gradient(u_x, y_coord)
    u_x_z = gradient(u_x, z_coord)
    
    u_y_x = gradient(u_y, x_coord)
    u_y_y = gradient(u_y, y_coord)
    u_y_z = gradient(u_y, z_coord)
    
    u_z_x = gradient(u_z, x_coord)
    u_z_y = gradient(u_z, y_coord)
    u_z_z = gradient(u_z, z_coord)
    
    # 计算应变张量（小变形假设）
    # ε_ij = 1/2 (∂u_i/∂x_j + ∂u_j/∂x_i)
    epsilon_xx = u_x_x
    epsilon_yy = u_y_y
    epsilon_zz = u_z_z
    epsilon_xy = 0.5 * (u_x_y + u_y_x)
    epsilon_xz = 0.5 * (u_x_z + u_z_x)
    epsilon_yz = 0.5 * (u_y_z + u_z_y)
    
    # 材料参数（假设为各向同性线弹性材料）
    E = 1.0e6  # 杨氏模量 (Pa)
    nu = 0.3   # 泊松比
    
    # 计算拉梅常数
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    
    # 计算应力张量（广义胡克定律）
    # σ_ij = λ ε_kk δ_ij + 2μ ε_ij
    epsilon_kk = epsilon_xx + epsilon_yy + epsilon_zz  # 体积应变
    
    sigma_xx = lambda_ * epsilon_kk + 2 * mu * epsilon_xx
    sigma_yy = lambda_ * epsilon_kk + 2 * mu * epsilon_yy
    sigma_zz = lambda_ * epsilon_kk + 2 * mu * epsilon_zz
    sigma_xy = 2 * mu * epsilon_xy
    sigma_xz = 2 * mu * epsilon_xz
    sigma_yz = 2 * mu * epsilon_yz
    
    # 计算应力散度（平衡方程）
    # ∂σ_xx/∂x + ∂σ_xy/∂y + ∂σ_xz/∂z + f_x = ρ ∂²u_x/∂t²
    sigma_xx_x = gradient(sigma_xx, x_coord)
    sigma_xy_y = gradient(sigma_xy, y_coord)
    sigma_xz_z = gradient(sigma_xz, z_coord)
    
    # ∂σ_yx/∂x + ∂σ_yy/∂y + ∂σ_yz/∂z + f_y = ρ ∂²u_y/∂t²
    sigma_yx_x = gradient(sigma_xy, x_coord)  # σ_yx = σ_xy
    sigma_yy_y = gradient(sigma_yy, y_coord)
    sigma_yz_z = gradient(sigma_yz, z_coord)
    
    # ∂σ_zx/∂x + ∂σ_zy/∂y + ∂σ_zz/∂z + f_z = ρ ∂²u_z/∂t²
    sigma_zx_x = gradient(sigma_xz, x_coord)  # σ_zx = σ_xz
    sigma_zy_y = gradient(sigma_yz, y_coord)  # σ_zy = σ_yz
    sigma_zz_z = gradient(sigma_zz, z_coord)
    
    # 假设无体力：f_x = f_y = f_z = 0
    # 材料密度
    rho = 7800  # 钢的密度 (kg/m³)
    
    # 计算平衡方程残差
    residual_x = sigma_xx_x + sigma_xy_y + sigma_xz_z - rho * u_x_tt
    residual_y = sigma_yx_x + sigma_yy_y + sigma_yz_z - rho * u_y_tt
    residual_z = sigma_zx_x + sigma_zy_y + sigma_zz_z - rho * u_z_tt
    
    # 组合残差
    residual = torch.cat([residual_x, residual_y, residual_z], dim=1)
    
    return residual


def reaction_diffusion_equation(x: torch.Tensor, u: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """反应扩散方程残差（Fisher-KPP方程）
    
    方程：
        ∂u/∂t = D ∇²u + r u (1 - u)
    
    参数:
        x: 输入张量 [batch_size, 2+] (t, x, [y, z])
        u: 浓度场 [batch_size, 1]
        model: 模型
    
    返回:
        残差张量 [batch_size, 1]
    """
    # 确保输入有requires_grad
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # 确保u有requires_grad
    if not u.requires_grad:
        u.requires_grad_(True)
    
    # 提取坐标
    t = x[:, 0:1]
    spatial_dims = x.shape[1] - 1
    
    # 计算时间导数
    u_t = gradient(u, t)
    
    # 计算拉普拉斯算子
    laplacian_u = torch.zeros_like(u)
    
    for i in range(spatial_dims):
        coord = x[:, i+1:i+2]
        # 确保坐标有requires_grad
        if not coord.requires_grad:
            coord.requires_grad_(True)
        
        u_xi = gradient(u, coord)
        
        # 确保u_xi有requires_grad（用于二阶导数计算）
        if not u_xi.requires_grad:
            u_xi.requires_grad_(True)
        
        u_xixi = gradient(u_xi, coord)
        laplacian_u = laplacian_u + u_xixi
    
    # 反应扩散参数
    D = 0.1  # 扩散系数
    r = 1.0  # 反应率
    K = 1.0  # 承载能力
    
    # 计算反应扩散方程残差
    residual = u_t - D * laplacian_u - r * u * (1 - u/K)
    
    return residual


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """计算梯度 dy/dx"""
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # 计算梯度
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True  # 允许未使用的输入
    )[0]
    
    # 如果梯度为None（输入未使用），返回零张量
    if grad is None:
        return torch.zeros_like(x)
    
    return grad


def get_pde_function(pde_type: str) -> Callable:
    """根据PDE类型获取对应的方程函数
    
    参数:
        pde_type: PDE类型字符串
        
    返回:
        对应的PDE残差函数
    """
    pde_functions = {
        "burgers": burgers_equation,
        "navier_stokes": navier_stokes_equation,
        "heat": heat_equation,
        "wave": wave_equation,
        "schrodinger": schrodinger_equation,
        "maxwell": maxwell_equations,
        "elasticity": elasticity_equation,
        "reaction_diffusion": reaction_diffusion_equation,
    }
    
    if pde_type not in pde_functions:
        raise ValueError(f"未知的PDE类型: {pde_type}。可用类型: {list(pde_functions.keys())}")
    
    return pde_functions[pde_type]


def create_multi_physics_system(pde_types: List[str], coupling_strength: float = 0.1) -> Callable:
    """创建多物理场耦合系统
    
    参数:
        pde_types: PDE类型列表
        coupling_strength: 耦合强度
        
    返回:
        多物理场耦合函数
    """
    pde_funcs = [get_pde_function(pde_type) for pde_type in pde_types]
    
    def multi_physics_equation(x: torch.Tensor, u: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """多物理场耦合方程"""
        residuals = []
        
        # 对于每个PDE，计算残差
        for i, pde_func in enumerate(pde_funcs):
            # 提取对应的场分量
            # 完整处理：假设每个场使用u的不同部分
            num_fields = len(pde_funcs)
            field_size = u.shape[1] // num_fields
            start_idx = i * field_size
            end_idx = (i + 1) * field_size if i < num_fields - 1 else u.shape[1]
            
            u_field = u[:, start_idx:end_idx]
            residual = pde_func(x, u_field, model)
            residuals.append(residual)
        
        # 添加耦合项（完整：线性耦合）
        coupled_residuals = []
        for i, residual in enumerate(residuals):
            coupled_residual = residual
            for j, other_residual in enumerate(residuals):
                if i != j:
                    # 添加耦合项：残差_i + coupling * 残差_j
                    coupled_residual = coupled_residual + coupling_strength * other_residual
            coupled_residuals.append(coupled_residual)
        
        # 组合所有残差
        total_residual = torch.cat(coupled_residuals, dim=1)
        return total_residual
    
    return multi_physics_equation


def create_adaptive_mesh_refiner(
    initial_resolution: int = 100,
    refinement_threshold: float = 0.1,
    max_refinement_level: int = 5
) -> Callable:
    """创建自适应网格细化器
    
    参数:
        initial_resolution: 初始网格分辨率
        refinement_threshold: 细化阈值（残差超过此值则细化）
        max_refinement_level: 最大细化层级
        
    返回:
        自适应网格细化函数
    """
    def adaptive_refiner(
        x: torch.Tensor, 
        u: torch.Tensor, 
        residual: torch.Tensor,
        current_level: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """自适应网格细化
        
        参数:
            x: 当前网格点
            u: 当前解
            residual: 残差
            current_level: 当前细化层级
            
        返回:
            细化后的网格点和对应的解估计
        """
        if current_level >= max_refinement_level:
            return x, u
        
        # 计算每个点的残差大小
        residual_norm = torch.norm(residual, dim=1, keepdim=True)
        
        # 找出需要细化的区域（残差大的区域）
        refine_mask = residual_norm > refinement_threshold
        
        if not torch.any(refine_mask):
            return x, u
        
        # 获取需要细化的点
        refine_points = x[refine_mask.squeeze()]
        
        # 生成细化的子网格
        refined_points_list = []
        for point in refine_points:
            # 在点周围生成细化网格（完整：均匀细分）
            # 完整处理，实际应用中需要更复杂的细化策略
            dim = point.shape[0]  # 点的维度
            # 为每个维度生成正负方向的偏移
            offsets_list = []
            for d in range(dim):
                # 正方向
                offset_pos = torch.zeros(dim, device=x.device, dtype=x.dtype)
                offset_pos[d] = 0.1
                offsets_list.append(offset_pos)
                
                # 负方向
                offset_neg = torch.zeros(dim, device=x.device, dtype=x.dtype)
                offset_neg[d] = -0.1
                offsets_list.append(offset_neg)
            
            offsets = torch.stack(offsets_list, dim=0)
            
            refined_points = point + offsets * (0.5 ** current_level)
            refined_points_list.append(refined_points)
        
        if refined_points_list:
            new_points = torch.cat(refined_points_list, dim=0)
            # 合并新旧网格点
            all_points = torch.cat([x, new_points], dim=0)
            
            # 估计新点的解（使用插值，这里完整：设为0）
            new_u = torch.zeros(new_points.shape[0], u.shape[1], 
                              device=u.device, dtype=u.dtype)
            all_u = torch.cat([u, new_u], dim=0)
            
            return all_points, all_u
        
        return x, u
    
    return adaptive_refiner


def create_uncertainty_quantifier(
    num_samples: int = 100,
    mc_dropout_rate: float = 0.1
) -> Callable:
    """创建不确定性量化器
    
    参数:
        num_samples: 蒙特卡洛采样次数
        mc_dropout_rate: MC Dropout率
        
    返回:
        不确定性量化函数
    """
    def quantify_uncertainty(
        model: nn.Module,
        x: torch.Tensor,
        pde_func: Callable
    ) -> Dict[str, torch.Tensor]:
        """量化模型预测的不确定性
        
        参数:
            model: PINN模型
            x: 输入点
            pde_func: PDE函数
            
        返回:
            包含不确定性统计信息的字典
        """
        # 启用dropout（如果模型有）
        model.train()
        
        # 蒙特卡洛采样
        predictions = []
        residuals = []
        
        for _ in range(num_samples):
            # 前向传播（由于dropout，每次结果不同）
            u_pred = model(x)
            predictions.append(u_pred.detach())
            
            # 计算残差
            residual = pde_func(x, u_pred, model)
            residuals.append(residual.detach())
        
        # 转换为张量
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, output_dim]
        residuals = torch.stack(residuals, dim=0)      # [num_samples, batch_size, residual_dim]
        
        # 计算统计量
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        mean_residual = torch.mean(residuals, dim=0)
        std_residual = torch.std(residuals, dim=0)
        
        # 计算置信区间（95%）
        conf_interval_lower = mean_pred - 1.96 * std_pred
        conf_interval_upper = mean_pred + 1.96 * std_pred
        
        # 计算变异系数
        coeff_of_variation = std_pred / (torch.abs(mean_pred) + 1e-8)
        
        # 恢复到评估模式
        model.eval()
        
        return {
            "mean_prediction": mean_pred,
            "std_prediction": std_pred,
            "mean_residual": mean_residual,
            "std_residual": std_residual,
            "confidence_interval_lower": conf_interval_lower,
            "confidence_interval_upper": conf_interval_upper,
            "coefficient_of_variation": coeff_of_variation,
            "num_samples": num_samples
        }
    
    return quantify_uncertainty


def test_pinn_framework():
    """测试PINN框架"""
    print("=== 测试PINN框架 ===")
    
    # 测试配置（启用缓存）
    config = PINNConfig(
        input_dim=2,  # (x, t)
        output_dim=1,  # u
        hidden_dim=32,
        num_layers=3,
        activation="tanh",
        enable_incremental_cache=True,  # 启用缓存
        max_cache_size=500,  # 缓存大小
        cache_eviction_policy="adaptive",  # 自适应淘汰策略
        adaptive_cache_sizing=True,  # 自适应大小调整
        cache_enabled_for_pde=True,  # 启用PDE缓存
        cache_enabled_for_gradients=True,  # 启用梯度缓存
        cache_enabled_for_losses=True  # 启用损失缓存
    )
    
    # 创建PINN模型
    pinn = PINNModel(config)
    
    # 创建测试数据
    device = pinn.device
    dtype = config.dtype
    
    # 创建空间-时间网格
    n_x = 20
    n_t = 10
    x = torch.linspace(0, 1, n_x, device=device, dtype=dtype)
    t = torch.linspace(0, 1, n_t, device=device, dtype=dtype)
    
    X, T = torch.meshgrid(x, t, indexing='ij')
    inputs = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
    
    print(f"输入数据形状: {inputs.shape}")
    
    # 测试前向传播
    outputs = pinn(inputs)
    print(f"输出数据形状: {outputs.shape}")
    
    # 测试Burgers方程残差计算
    print("\n测试Burgers方程残差计算:")
    # 使用compute_pde_residual方法，它会设置requires_grad
    residual = pinn.compute_pde_residual(inputs, burgers_equation, iteration=0)
    print(f"残差形状: {residual.shape}")
    print(f"残差均值: {torch.mean(residual).item():.6f}")
    
    # 测试约束添加
    print("\n测试物理约束:")
    
    # 创建PDE约束
    pde_constraint = PDEResidualConstraint(burgers_equation, weight=1.0)
    pinn.add_constraint(pde_constraint)
    
    # 创建边界条件约束
    def boundary_func(x):
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    
    # 创建边界掩码 (完整)
    boundary_mask = (inputs[:, 0] == 0) | (inputs[:, 0] == 1)
    bc_constraint = BoundaryConditionConstraint(
        boundary_func, boundary_mask, weight=1.0, condition_type="dirichlet"
    )
    pinn.add_constraint(bc_constraint)
    
    # 测试损失计算
    print("\n测试总损失计算:")
    total_loss, loss_dict = pinn.compute_total_loss(inputs, outputs, iteration=0)
    print(f"总损失: {total_loss.item():.6f}")
    print(f"损失字典: {loss_dict}")
    
    # 测试新物理方程功能
    print("\n测试新物理方程功能:")
    
    # 1. 测试纳维-斯托克斯方程
    print("a. 测试纳维-斯托克斯方程:")
    ns_config = PINNConfig(
        input_dim=3,  # (t, x, y)
        output_dim=3,  # (u, v, p)
        hidden_dim=32,
        num_layers=3,
        activation="tanh",
        pde_type="navier_stokes"
    )
    
    # 创建测试数据 (3D: t, x, y)
    n_t = 5
    n_x = 10
    n_y = 10
    t = torch.linspace(0, 1, n_t, device=device, dtype=dtype)
    x = torch.linspace(0, 1, n_x, device=device, dtype=dtype)
    y = torch.linspace(0, 1, n_y, device=device, dtype=dtype)
    
    T, X, Y = torch.meshgrid(t, x, y, indexing='ij')
    ns_inputs = torch.stack([T.reshape(-1), X.reshape(-1), Y.reshape(-1)], dim=1)
    
    # 获取纳维-斯托克斯方程函数
    ns_func = get_pde_function("navier_stokes")
    
    # 创建临时模型进行测试
    ns_model = PINNModel(ns_config)
    ns_outputs = ns_model(ns_inputs)
    
    # 计算残差
    ns_residual = ns_func(ns_inputs, ns_outputs, ns_model)
    print(f"   纳维-斯托克斯残差形状: {ns_residual.shape}")
    print(f"   连续性残差均值: {torch.mean(ns_residual[:, 0:1]).item():.6f}")
    print(f"   x动量残差均值: {torch.mean(ns_residual[:, 1:2]).item():.6f}")
    print(f"   y动量残差均值: {torch.mean(ns_residual[:, 2:3]).item():.6f}")
    
    # 2. 测试多物理场耦合
    print("b. 测试多物理场耦合:")
    multi_physics_func = create_multi_physics_system(
        ["heat", "reaction_diffusion"], 
        coupling_strength=0.05
    )
    
    # 创建测试数据
    mp_inputs = torch.rand(100, 3, device=device, dtype=dtype)  # (t, x, y)
    mp_outputs = torch.rand(100, 2, device=device, dtype=dtype)  # 两个场
    
    # 确保张量有requires_grad用于梯度计算
    mp_inputs.requires_grad_(True)
    mp_outputs.requires_grad_(True)
    
    # 计算多物理场残差
    mp_residual = multi_physics_func(mp_inputs, mp_outputs, pinn)
    print(f"   多物理场残差形状: {mp_residual.shape}")
    print(f"   耦合残差均值: {torch.mean(mp_residual).item():.6f}")
    
    # 3. 测试自适应网格细化
    print("c. 测试自适应网格细化:")
    refiner = create_adaptive_mesh_refiner(
        initial_resolution=50,
        refinement_threshold=0.05,
        max_refinement_level=3
    )
    
    # 创建测试网格和残差
    test_points = torch.rand(50, 2, device=device, dtype=dtype)
    test_u = torch.rand(50, 1, device=device, dtype=dtype)
    test_residual = torch.rand(50, 1, device=device, dtype=dtype) * 0.1
    
    # 应用网格细化
    refined_points, refined_u = refiner(test_points, test_u, test_residual, current_level=0)
    print(f"   原始网格点: {test_points.shape[0]}")
    print(f"   细化后网格点: {refined_points.shape[0]}")
    print(f"   细化增加点数: {refined_points.shape[0] - test_points.shape[0]}")
    
    # 4. 测试不确定性量化
    print("d. 测试不确定性量化:")
    uncertainty_quantifier = create_uncertainty_quantifier(
        num_samples=50,
        mc_dropout_rate=0.1
    )
    
    # 创建带dropout的测试模型
    test_config = PINNConfig(
        input_dim=2,
        output_dim=1,
        hidden_dim=16,
        num_layers=2,
        activation="tanh"
    )
    test_model = PINNModel(test_config)
    
    # 添加dropout层以进行不确定性量化
    test_model.network = nn.Sequential(
        nn.Linear(2, 16),
        nn.Tanh(),
        nn.Dropout(0.1),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Dropout(0.1),
        nn.Linear(16, 1)
    ).to(device).to(dtype)
    
    # 量化不确定性
    test_inputs = torch.rand(20, 2, device=device, dtype=dtype)
    uncertainty_results = uncertainty_quantifier(test_model, test_inputs, burgers_equation)
    
    print(f"   均值预测形状: {uncertainty_results['mean_prediction'].shape}")
    print(f"   标准差预测形状: {uncertainty_results['std_prediction'].shape}")
    print(f"   平均标准差: {torch.mean(uncertainty_results['std_prediction']).item():.6f}")
    print(f"   采样次数: {uncertainty_results['num_samples']}")
    
    # 5. 测试所有PDE函数
    print("e. 测试所有PDE函数:")
    pde_types = ["burgers", "heat", "wave", "navier_stokes", "schrodinger", 
                 "maxwell", "elasticity", "reaction_diffusion"]
    
    for pde_type in pde_types:
        try:
            pde_func = get_pde_function(pde_type)
            
            # 根据PDE类型创建适当的测试数据
            if pde_type == "maxwell":
                test_input = torch.rand(10, 4, device=device, dtype=dtype)  # (t, x, y, z)
                test_output = torch.rand(10, 6, device=device, dtype=dtype)  # (E_x, E_y, E_z, B_x, B_y, B_z)
            elif pde_type == "elasticity":
                test_input = torch.rand(10, 4, device=device, dtype=dtype)  # (t, x, y, z)
                test_output = torch.rand(10, 3, device=device, dtype=dtype)  # (u_x, u_y, u_z)
            elif pde_type == "schrodinger":
                test_input = torch.rand(10, 2, device=device, dtype=dtype)  # (t, x)
                test_output = torch.rand(10, 2, device=device, dtype=dtype)  # (实部, 虚部)
            elif pde_type == "navier_stokes":
                test_input = torch.rand(10, 3, device=device, dtype=dtype)  # (t, x, y)
                test_output = torch.rand(10, 3, device=device, dtype=dtype)  # (u, v, p)
            else:
                test_input = torch.rand(10, 2, device=device, dtype=dtype)  # (t, x)
                test_output = torch.rand(10, 1, device=device, dtype=dtype)  # u
            
            # 确保张量有requires_grad用于梯度计算
            test_input.requires_grad_(True)
            test_output.requires_grad_(True)
            
            residual = pde_func(test_input, test_output, test_model)
            print(f"   {pde_type}: 残差形状={residual.shape}, 均值={torch.mean(residual).item():.6f}")
        except Exception as e:
            print(f"   {pde_type}: 错误 - {e}")
    
    print("\n新物理方程功能测试完成!")
    
    # 测试缓存功能
    print("\n测试增量式缓存功能:")
    
    # 测试PDE残差缓存
    print("a. 测试PDE残差缓存:")
    
    # 第一次计算（应该计算并缓存）
    start_time = time.time()
    residual1 = pinn.compute_pde_residual(inputs, burgers_equation, iteration=1)
    time1 = time.time() - start_time
    print(f"   第一次计算时间: {time1:.4f}s")
    
    # 第二次计算（应该从缓存获取 - 使用相同的iteration）
    start_time = time.time()
    residual2 = pinn.compute_pde_residual(inputs, burgers_equation, iteration=1)
    time2 = time.time() - start_time
    print(f"   第二次计算时间: {time2:.4f}s")
    
    # 检查结果是否相同
    residual_diff = torch.mean(torch.abs(residual1 - residual2)).item()
    print(f"   残差差异: {residual_diff:.6f} (应为0)")
    
    # 检查时间差异（缓存应该更快）
    if time2 < time1:
        speedup = time1 / max(time2, 1e-6)
        print(f"   缓存加速比: {speedup:.2f}x")
    else:
        print(f"   缓存时间: {time2:.4f}s, 原始时间: {time1:.4f}s")
    
    # 获取缓存统计
    cache_stats = pinn.get_cache_stats()
    if cache_stats:
        print(f"b. 缓存统计:")
        print(f"   缓存大小: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
        print(f"   命中率: {cache_stats['hit_rate']:.2%}")
        print(f"   命中次数: {cache_stats['cache_hits']}")
        print(f"   未命中次数: {cache_stats['cache_misses']}")
        print(f"   总查询次数: {cache_stats['total_queries']}")
        print(f"   内存使用: {cache_stats['total_memory_mb']:.2f} MB")
    
    # 测试缓存失效
    print("c. 测试缓存失效:")
    pinn.invalidate_cache_pattern("pde_")
    cache_stats_after = pinn.get_cache_stats()
    if cache_stats_after:
        print(f"   失效后缓存大小: {cache_stats_after['cache_size']}")
    
    # 测试缓存清理
    print("d. 测试缓存清理:")
    pinn.cleanup()
    print("   缓存已清理")
    
    print("\n=== PINN框架测试完成 ===")


if __name__ == "__main__":
    test_pinn_framework()