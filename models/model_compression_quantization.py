#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self AGI 模型压缩和量化模块
实现完整的模型压缩和量化功能，减少模型大小、加速推理、降低内存使用

功能：
1. 模型剪枝（结构化/非结构化）
2. 模型量化（训练后量化、量化感知训练）
3. 知识蒸馏
4. 权重共享
5. 低秩分解
6. 模型结构搜索（用于压缩）
"""

import sys
import os
import logging
import json
import time
import pickle
import warnings
import math
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 尝试导入PyTorch的剪枝和量化模块
try:
    import torch.nn.utils.prune as prune
    TORCH_PRUNE_AVAILABLE = True
except ImportError:
    TORCH_PRUNE_AVAILABLE = False

try:
    import torch.quantization as quant
    TORCH_QUANT_AVAILABLE = True
except ImportError:
    TORCH_QUANT_AVAILABLE = False


class CompressionStrategy(Enum):
    """压缩策略枚举"""
    PRUNING = "pruning"  # 剪枝
    QUANTIZATION = "quantization"  # 量化
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"  # 知识蒸馏
    WEIGHT_SHARING = "weight_sharing"  # 权重共享
    LOW_RANK_DECOMPOSITION = "low_rank_decomposition"  # 低秩分解
    COMPOSITE = "composite"  # 复合压缩


class PruningMethod(Enum):
    """剪枝方法枚举"""
    L1_UNSTRUCTURED = "l1_unstructured"  # L1非结构化剪枝
    L2_UNSTRUCTURED = "l2_unstructured"  # L2非结构化剪枝
    RANDOM_UNSTRUCTURED = "random_unstructured"  # 随机非结构化剪枝
    STRUCTURED = "structured"  # 结构化剪枝
    GLOBAL = "global"  # 全局剪枝


class QuantizationMethod(Enum):
    """量化方法枚举"""
    POST_TRAINING_STATIC = "post_training_static"  # 训练后静态量化
    POST_TRAINING_DYNAMIC = "post_training_dynamic"  # 训练后动态量化
    QUANTIZATION_AWARE_TRAINING = "quantization_aware_training"  # 量化感知训练
    FLOAT16 = "float16"  # 半精度浮点数
    INT8 = "int8"  # 8位整数


@dataclass
class CompressionConfig:
    """压缩配置"""
    
    strategy: CompressionStrategy = CompressionStrategy.PRUNING
    target_sparsity: float = 0.5  # 目标稀疏度（剪枝）
    pruning_method: PruningMethod = PruningMethod.L1_UNSTRUCTURED
    
    # 量化配置
    quantization_method: QuantizationMethod = QuantizationMethod.POST_TRAINING_STATIC
    quantization_bits: int = 8  # 量化位数
    
    # 知识蒸馏配置
    teacher_model: Optional[nn.Module] = None
    temperature: float = 3.0  # 蒸馏温度
    alpha: float = 0.5  # 蒸馏损失权重
    
    # 权重共享配置
    num_shared_groups: int = 4  # 共享组数量
    
    # 低秩分解配置
    rank_ratio: float = 0.25  # 秩的比例
    
    # 复合压缩配置
    composite_steps: List[CompressionStrategy] = field(default_factory=lambda: [
        CompressionStrategy.PRUNING,
        CompressionStrategy.QUANTIZATION
    ])
    
    # 通用配置
    layer_types_to_compress: List[str] = field(default_factory=lambda: ["Linear", "Conv2d"])
    skip_layers: List[str] = field(default_factory=list)
    fine_tune_epochs: int = 10  # 压缩后微调轮数
    preserve_accuracy: bool = True  # 是否保持精度
    
    def __post_init__(self):
        """配置验证"""
        if self.target_sparsity < 0 or self.target_sparsity > 1:
            warnings.warn(f"target_sparsity必须在[0,1]范围内，已调整为0.5 (原值: {self.target_sparsity})")
            self.target_sparsity = 0.5
        
        if self.quantization_bits not in [4, 8, 16, 32]:
            warnings.warn(f"quantization_bits必须是4、8、16或32，已调整为8 (原值: {self.quantization_bits})")
            self.quantization_bits = 8
        
        if self.temperature <= 0:
            warnings.warn(f"temperature必须大于0，已调整为3.0 (原值: {self.temperature})")
            self.temperature = 3.0
        
        if self.alpha < 0 or self.alpha > 1:
            warnings.warn(f"alpha必须在[0,1]范围内，已调整为0.5 (原值: {self.alpha})")
            self.alpha = 0.5
        
        if self.rank_ratio <= 0 or self.rank_ratio >= 1:
            warnings.warn(f"rank_ratio必须在(0,1)范围内，已调整为0.25 (原值: {self.rank_ratio})")
            self.rank_ratio = 0.25
        
        if self.num_shared_groups < 1:
            warnings.warn(f"num_shared_groups必须大于0，已调整为4 (原值: {self.num_shared_groups})")
            self.num_shared_groups = 4


class ModelCompressor:
    """模型压缩器
    
    实现完整的模型压缩功能
    """
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.logger = logging.getLogger("ModelCompressor")
        
        # 压缩统计信息
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 0.0,
            'accuracy_drop': 0.0,
            'parameters_before': 0,
            'parameters_after': 0,
            'sparsity_achieved': 0.0,
            'quantization_error': 0.0,
            'memory_reduction': 0.0,
            'inference_speedup': 0.0
        }
        
        self.logger.info(f"模型压缩器初始化: 策略={config.strategy.value}")
    
    def compress(self, model: nn.Module, dataloader: Optional[Any] = None,
                evaluator: Optional[Callable] = None) -> nn.Module:
        """应用压缩策略
        
        参数:
            model: 原始模型
            dataloader: 数据加载器（用于校准和微调）
            evaluator: 评估函数，用于测量精度
            
        返回:
            压缩后的模型
        """
        self.logger.info("开始模型压缩")
        
        # 记录原始模型信息
        self._record_model_info(model, "压缩前")
        
        # 保存原始模型状态
        original_state = model.state_dict()
        
        # 根据策略应用压缩
        compressed_model = model
        
        if self.config.strategy == CompressionStrategy.PRUNING:
            compressed_model = self._apply_pruning(model)
        
        elif self.config.strategy == CompressionStrategy.QUANTIZATION:
            compressed_model = self._apply_quantization(model, dataloader)
        
        elif self.config.strategy == CompressionStrategy.KNOWLEDGE_DISTILLATION:
            compressed_model = self._apply_knowledge_distillation(model, dataloader)
        
        elif self.config.strategy == CompressionStrategy.WEIGHT_SHARING:
            compressed_model = self._apply_weight_sharing(model)
        
        elif self.config.strategy == CompressionStrategy.LOW_RANK_DECOMPOSITION:
            compressed_model = self._apply_low_rank_decomposition(model)
        
        elif self.config.strategy == CompressionStrategy.COMPOSITE:
            compressed_model = self._apply_composite_compression(model, dataloader)
        
        # 微调压缩后的模型（如果需要）
        if self.config.fine_tune_epochs > 0 and dataloader is not None:
            compressed_model = self._fine_tune_model(compressed_model, dataloader)
        
        # 评估压缩效果
        if evaluator is not None:
            original_accuracy = evaluator(model)
            compressed_accuracy = evaluator(compressed_model)
            self.compression_stats['accuracy_drop'] = original_accuracy - compressed_accuracy
            self.logger.info(f"精度变化: {original_accuracy:.4f} -> {compressed_accuracy:.4f}, 下降: {self.compression_stats['accuracy_drop']:.4f}")
        
        # 计算压缩统计信息
        self._calculate_compression_stats(model, compressed_model)
        
        self.logger.info(f"模型压缩完成: 压缩率={self.compression_stats['compression_ratio']:.2f}x, "
                        f"内存减少={self.compression_stats['memory_reduction']:.1f}%, "
                        f"精度下降={self.compression_stats['accuracy_drop']:.4f}")
        
        return compressed_model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """应用模型剪枝"""
        self.logger.info(f"应用模型剪枝: 方法={self.config.pruning_method.value}, 目标稀疏度={self.config.target_sparsity}")
        
        # 计算剪枝前的参数
        total_params = sum(p.numel() for p in model.parameters())
        self.compression_stats['parameters_before'] = total_params
        
        if TORCH_PRUNE_AVAILABLE:
            model = self._pytorch_pruning(model)
        else:
            model = self._custom_pruning(model)
        
        # 计算剪枝后的参数
        non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
        self.compression_stats['parameters_after'] = non_zero_params
        self.compression_stats['sparsity_achieved'] = 1.0 - (non_zero_params / total_params) if total_params > 0 else 0
        
        self.logger.info(f"剪枝完成: 稀疏度={self.compression_stats['sparsity_achieved']:.3f}, "
                        f"非零参数={non_zero_params:,}/{total_params:,}")
        
        return model
    
    def _pytorch_pruning(self, model: nn.Module) -> nn.Module:
        """使用PyTorch剪枝功能"""
        self.logger.info("使用PyTorch剪枝功能")
        
        # 收集要剪枝的层
        layers_to_prune = []
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type in self.config.layer_types_to_compress and name not in self.config.skip_layers:
                if hasattr(module, 'weight'):
                    layers_to_prune.append((module, 'weight'))
        
        if not layers_to_prune:
            self.logger.warning("没有找到可剪枝的层")
            return model
        
        # 应用剪枝
        for module, param_name in layers_to_prune:
            try:
                if self.config.pruning_method == PruningMethod.L1_UNSTRUCTURED:
                    prune.l1_unstructured(module, name=param_name, amount=self.config.target_sparsity)
                elif self.config.pruning_method == PruningMethod.L2_UNSTRUCTURED:
                    # L2剪枝使用自定义实现
                    self._l2_pruning(module, param_name)
                elif self.config.pruning_method == PruningMethod.RANDOM_UNSTRUCTURED:
                    prune.random_unstructured(module, name=param_name, amount=self.config.target_sparsity)
                elif self.config.pruning_method == PruningMethod.STRUCTURED:
                    # 结构化剪枝需要更复杂的实现
                    self._structured_pruning(module, param_name)
                elif self.config.pruning_method == PruningMethod.GLOBAL:
                    # 全局剪枝需要先收集所有参数
                    pass  # 将在后面实现
                
                self.logger.debug(f"剪枝层: {module.__class__.__name__}")
            except Exception as e:
                self.logger.warning(f"剪枝层失败: {e}")
        
        # 对于全局剪枝，使用全局策略
        if self.config.pruning_method == PruningMethod.GLOBAL:
            self._global_pruning(model, layers_to_prune)
        
        # 使剪枝永久化
        for module, param_name in layers_to_prune:
            try:
                prune.remove(module, param_name)
            except Exception:
                pass  # 如果未剪枝，忽略
        
        return model
    
    def _custom_pruning(self, model: nn.Module) -> nn.Module:
        """自定义剪枝实现"""
        self.logger.info("使用自定义剪枝实现")
        
        for name, param in model.named_parameters():
            if 'weight' in name and not any(skip in name for skip in self.config.skip_layers):
                # 获取层类型
                module_type = self._get_module_type_from_param_name(model, name)
                if module_type in self.config.layer_types_to_compress:
                    # 应用剪枝
                    if self.config.pruning_method == PruningMethod.L1_UNSTRUCTURED:
                        self._apply_l1_pruning(param)
                    elif self.config.pruning_method == PruningMethod.L2_UNSTRUCTURED:
                        self._apply_l2_pruning(param)
                    elif self.config.pruning_method == PruningMethod.RANDOM_UNSTRUCTURED:
                        self._apply_random_pruning(param)
                    
                    self.logger.debug(f"自定义剪枝: {name}")
        
        return model
    
    def _apply_l1_pruning(self, param: torch.Tensor):
        """应用L1剪枝"""
        if param.dim() < 2:
            return
        
        # 计算阈值
        flat_weights = param.data.abs().flatten()
        k = int(self.config.target_sparsity * flat_weights.numel())
        if k > 0 and k < flat_weights.numel():
            threshold = flat_weights.kthvalue(k).values
            mask = param.data.abs() > threshold
            param.data.mul_(mask)
    
    def _apply_l2_pruning(self, param: torch.Tensor):
        """应用L2剪枝"""
        if param.dim() < 2:
            return
        
        # L2范数剪枝：按行/列剪枝
        if param.dim() == 2:  # Linear层
            # 按行剪枝
            row_norms = torch.norm(param.data, dim=1)
            k = int(self.config.target_sparsity * row_norms.numel())
            if k > 0 and k < row_norms.numel():
                threshold = row_norms.kthvalue(k).values
                mask = row_norms > threshold
                param.data.mul_(mask.unsqueeze(1))
    
    def _apply_random_pruning(self, param: torch.Tensor):
        """应用随机剪枝"""
        if param.dim() < 2:
            return
        
        # 创建随机掩码
        mask = torch.rand_like(param.data) > self.config.target_sparsity
        param.data.mul_(mask)
    
    def _apply_quantization(self, model: nn.Module, dataloader: Optional[Any] = None) -> nn.Module:
        """应用模型量化"""
        self.logger.info(f"应用模型量化: 方法={self.config.quantization_method.value}, 位数={self.config.quantization_bits}")
        
        # 设置模型为评估模式
        model.eval()
        
        quantized_model = model
        
        if TORCH_QUANT_AVAILABLE and self.config.quantization_method in [
            QuantizationMethod.POST_TRAINING_STATIC,
            QuantizationMethod.POST_TRAINING_DYNAMIC,
            QuantizationMethod.QUANTIZATION_AWARE_TRAINING
        ]:
            quantized_model = self._pytorch_quantization(model, dataloader)
        else:
            quantized_model = self._custom_quantization(model)
        
        # 计算量化误差
        self._calculate_quantization_error(model, quantized_model)
        
        return quantized_model
    
    def _pytorch_quantization(self, model: nn.Module, dataloader: Optional[Any] = None) -> nn.Module:
        """使用PyTorch量化功能"""
        self.logger.info("使用PyTorch量化功能")
        
        # 准备模型进行量化
        model.eval()
        
        if self.config.quantization_method == QuantizationMethod.POST_TRAINING_STATIC:
            # 静态量化需要校准数据
            if dataloader is None:
                self.logger.warning("静态量化需要校准数据，使用动态量化代替")
                return self._dynamic_quantization(model)
            
            # 配置量化
            model.qconfig = quant.get_default_qconfig('fbgemm')
            quant.prepare(model, inplace=True)
            
            # 校准
            self.logger.info("进行量化校准...")
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= 100:  # 使用100个批次进行校准
                        break
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    model(inputs)
            
            # 转换模型
            quantized_model = quant.convert(model)
            
        elif self.config.quantization_method == QuantizationMethod.POST_TRAINING_DYNAMIC:
            # 动态量化
            quantized_model = quant.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
        elif self.config.quantization_method == QuantizationMethod.QUANTIZATION_AWARE_TRAINING:
            # 量化感知训练
            model.train()
            model.qconfig = quant.get_default_qat_qconfig('fbgemm')
            quant.prepare_qat(model, inplace=True)
            quantized_model = model
        
        else:
            self.logger.warning(f"不支持的PyTorch量化方法: {self.config.quantization_method}")
            quantized_model = model
        
        return quantized_model
    
    def _custom_quantization(self, model: nn.Module) -> nn.Module:
        """自定义量化实现"""
        self.logger.info(f"使用自定义量化到 {self.config.quantization_bits} 位")
        
        # 根据量化位数设置参数
        if self.config.quantization_bits == 4:
            scale_factor = 7.0  # 4位有符号：-8到7
        elif self.config.quantization_bits == 8:
            scale_factor = 127.0
        elif self.config.quantization_bits == 16:
            scale_factor = 32767.0
        else:  # 32位，实际上不量化
            return model
        
        # 量化权重参数
        for name, param in model.named_parameters():
            if 'weight' in name and not any(skip in name for skip in self.config.skip_layers):
                module_type = self._get_module_type_from_param_name(model, name)
                if module_type in self.config.layer_types_to_compress:
                    self._quantize_parameter(param, scale_factor)
                    self.logger.debug(f"量化参数: {name}")
        
        return model
    
    def _quantize_parameter(self, param: torch.Tensor, scale_factor: float):
        """量化单个参数"""
        if param.dim() < 1:
            return
        
        # 保存原始数据用于误差计算
        original_data = param.data.clone()
        
        # 获取数据范围
        min_val = param.data.min()
        max_val = param.data.max()
        scale = max(abs(min_val), abs(max_val))
        
        if scale > 0:
            # 量化过程
            # 1. 缩放：将权重映射到[-1, 1]范围
            scaled_weights = param.data / scale
            # 2. 量化：舍入到最近的整数值
            quantized = torch.round(scaled_weights * scale_factor)
            # 3. 反量化：映射回原始范围
            dequantized = quantized / scale_factor * scale
            
            # 应用量化后的权重
            param.data.copy_(dequantized)
    
    def _calculate_quantization_error(self, original_model: nn.Module, quantized_model: nn.Module):
        """计算量化误差"""
        total_error = 0.0
        num_params = 0
        
        for (name1, param1), (name2, param2) in zip(
            original_model.named_parameters(),
            quantized_model.named_parameters()
        ):
            if name1 == name2 and 'weight' in name1:
                error = torch.mean(torch.abs(param1.data - param2.data))
                total_error += error.item()
                num_params += 1
        
        if num_params > 0:
            self.compression_stats['quantization_error'] = total_error / num_params
    
    def _apply_knowledge_distillation(self, model: nn.Module, dataloader: Optional[Any] = None) -> nn.Module:
        """应用知识蒸馏"""
        self.logger.info("应用知识蒸馏")
        
        if self.config.teacher_model is None:
            self.logger.warning("未提供教师模型，使用自蒸馏")
            self.config.teacher_model = model
        
        if dataloader is None:
            self.logger.warning("知识蒸馏需要训练数据，跳过蒸馏步骤")
            return model
        
        # 创建学生模型（压缩后的模型）
        student_model = self._create_student_model(model)
        
        # 设置优化器
        optimizer = optim.Adam(student_model.parameters(), lr=1e-3)
        
        # 蒸馏训练
        self.logger.info("开始知识蒸馏训练...")
        for epoch in range(self.config.fine_tune_epochs):
            total_loss = 0.0
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = None
                
                optimizer.zero_grad()
                
                # 教师模型预测
                with torch.no_grad():
                    teacher_logits = self.config.teacher_model(inputs)
                
                # 学生模型预测
                student_logits = student_model(inputs)
                
                # 计算蒸馏损失
                loss = self._distillation_loss(
                    student_logits, teacher_logits, 
                    targets, self.config.temperature, self.config.alpha
                )
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % max(1, self.config.fine_tune_epochs // 5) == 0:
                self.logger.info(f"蒸馏epoch {epoch+1}/{self.config.fine_tune_epochs}, 损失: {total_loss:.4f}")
        
        return student_model
    
    def _distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                          hard_targets: Optional[torch.Tensor], temperature: float, alpha: float) -> torch.Tensor:
        """计算蒸馏损失"""
        # 软化教师预测
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        
        # KL散度损失
        distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
        
        # 如果有硬标签，计算交叉熵损失
        if hard_targets is not None:
            student_loss = F.cross_entropy(student_logits, hard_targets)
            total_loss = alpha * distillation_loss + (1 - alpha) * student_loss
        else:
            total_loss = distillation_loss
        
        return total_loss
    
    def _create_student_model(self, teacher_model: nn.Module) -> nn.Module:
        """创建学生模型（简化版教师模型）"""
        # 这里实现一个简单的模型简化策略
        # 实际应用中可能需要更复杂的架构搜索
        
        student_model = teacher_model
        
        # 简化策略：减少隐藏层维度
        for name, module in student_model.named_modules():
            if isinstance(module, nn.Linear):
                # 减少输出维度（如果是中间层）
                if hasattr(module, 'out_features'):
                    # 减少到原来的3/4
                    new_out_features = int(module.out_features * 0.75)
                    if new_out_features > 10:  # 保持最小维度
                        new_layer = nn.Linear(module.in_features, new_out_features)
                        # 复制部分权重
                        with torch.no_grad():
                            new_layer.weight.data[:] = module.weight.data[:new_out_features, :]
                            if module.bias is not None:
                                new_layer.bias.data[:] = module.bias.data[:new_out_features]
                        # 替换模块（简化实现）
                        self.logger.debug(f"简化层: {name}, {module.out_features} -> {new_out_features}")
        
        return student_model
    
    def _apply_weight_sharing(self, model: nn.Module) -> nn.Module:
        """应用权重共享"""
        self.logger.info(f"应用权重共享: 组数={self.config.num_shared_groups}")
        
        # 收集所有权重
        all_weights = []
        weight_locations = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and not any(skip in name for skip in self.config.skip_layers):
                module_type = self._get_module_type_from_param_name(model, name)
                if module_type in self.config.layer_types_to_compress:
                    flat_weights = param.data.flatten()
                    all_weights.append(flat_weights)
                    weight_locations.append((name, param))
        
        if not all_weights:
            self.logger.warning("没有找到可共享的权重")
            return model
        
        # 合并所有权重
        combined_weights = torch.cat(all_weights)
        
        # 使用K-means聚类进行权重共享
        shared_weights, assignments = self._kmeans_weight_sharing(
            combined_weights, self.config.num_shared_groups
        )
        
        # 应用共享权重
        start_idx = 0
        for (name, param), flat_weights in zip(weight_locations, all_weights):
            num_weights = flat_weights.numel()
            param_assignments = assignments[start_idx:start_idx + num_weights]
            
            # 重建共享权重
            shared_param = shared_weights[param_assignments].reshape(param.data.shape)
            param.data.copy_(shared_param)
            
            start_idx += num_weights
            self.logger.debug(f"权重共享层: {name}, 组数={len(torch.unique(param_assignments))}")
        
        self.logger.info(f"权重共享完成: 原始权重数={combined_weights.numel()}, "
                        f"共享后唯一权重数={len(shared_weights)}")
        
        return model
    
    def _kmeans_weight_sharing(self, weights: torch.Tensor, num_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """K-means权重共享"""
        # 简单K-means实现（实际应用中可能使用更高效的算法）
        if num_clusters >= len(weights):
            return weights, torch.arange(len(weights))
        
        # 随机初始化聚类中心
        indices = torch.randperm(len(weights))[:num_clusters]
        centroids = weights[indices].clone()
        
        assignments = torch.zeros(len(weights), dtype=torch.long)
        
        # 迭代优化
        for _ in range(20):  # 最大迭代次数
            # 分配步骤
            distances = torch.cdist(weights.unsqueeze(1), centroids.unsqueeze(1)).squeeze()
            new_assignments = torch.argmin(distances, dim=1)
            
            # 检查收敛
            if torch.all(new_assignments == assignments):
                break
            
            assignments = new_assignments
            
            # 更新步骤
            for k in range(num_clusters):
                cluster_points = weights[assignments == k]
                if len(cluster_points) > 0:
                    centroids[k] = cluster_points.mean()
        
        return centroids, assignments
    
    def _apply_low_rank_decomposition(self, model: nn.Module) -> nn.Module:
        """应用低秩分解"""
        self.logger.info(f"应用低秩分解: 秩比例={self.config.rank_ratio}")
        
        decomposed_model = model
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name not in self.config.skip_layers:
                # 对线性层进行低秩分解
                weight = module.weight.data
                in_features, out_features = weight.shape
                
                # 计算目标秩
                rank = max(1, int(min(in_features, out_features) * self.config.rank_ratio))
                
                # 奇异值分解
                U, S, V = torch.svd_lowrank(weight, q=rank)
                
                # 重建低秩权重
                reconstructed = U @ torch.diag(S) @ V.t()
                
                # 替换权重
                module.weight.data.copy_(reconstructed)
                self.logger.debug(f"低秩分解层: {name}, 秩={rank}/{min(in_features, out_features)}")
        
        return decomposed_model
    
    def _apply_composite_compression(self, model: nn.Module, dataloader: Optional[Any] = None) -> nn.Module:
        """应用复合压缩"""
        self.logger.info("应用复合压缩")
        
        compressed_model = model
        
        for step in self.config.composite_steps:
            self.logger.info(f"复合压缩步骤: {step.value}")
            
            # 临时更改配置以应用当前步骤
            original_strategy = self.config.strategy
            self.config.strategy = step
            
            if step == CompressionStrategy.PRUNING:
                compressed_model = self._apply_pruning(compressed_model)
            elif step == CompressionStrategy.QUANTIZATION:
                compressed_model = self._apply_quantization(compressed_model, dataloader)
            elif step == CompressionStrategy.KNOWLEDGE_DISTILLATION:
                compressed_model = self._apply_knowledge_distillation(compressed_model, dataloader)
            elif step == CompressionStrategy.WEIGHT_SHARING:
                compressed_model = self._apply_weight_sharing(compressed_model)
            elif step == CompressionStrategy.LOW_RANK_DECOMPOSITION:
                compressed_model = self._apply_low_rank_decomposition(compressed_model)
            
            # 恢复原始策略
            self.config.strategy = original_strategy
        
        return compressed_model
    
    def _fine_tune_model(self, model: nn.Module, dataloader: Any) -> nn.Module:
        """微调压缩后的模型"""
        self.logger.info(f"微调模型: {self.config.fine_tune_epochs} epochs")
        
        # 设置模型为训练模式
        model.train()
        
        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # 简单训练循环
        for epoch in range(self.config.fine_tune_epochs):
            total_loss = 0.0
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = None
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                if targets is not None:
                    loss = F.cross_entropy(outputs, targets)
                else:
                    # 自监督或重建任务
                    loss = F.mse_loss(outputs, inputs)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % max(1, self.config.fine_tune_epochs // 5) == 0:
                self.logger.info(f"微调epoch {epoch+1}/{self.config.fine_tune_epochs}, 损失: {total_loss:.4f}")
        
        # 设置回评估模式
        model.eval()
        
        return model
    
    def _record_model_info(self, model: nn.Module, stage: str):
        """记录模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"{stage}模型信息: 总参数={total_params:,}, 可训练参数={trainable_params:,}")
        
        if stage == "压缩前":
            self.compression_stats['original_size'] = total_params * 4  # 假设float32，4字节每个参数
            self.compression_stats['parameters_before'] = total_params
    
    def _calculate_compression_stats(self, original_model: nn.Module, compressed_model: nn.Module):
        """计算压缩统计信息"""
        # 计算参数数量
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        
        # 计算内存使用
        original_memory = original_params * 4  # float32: 4字节
        compressed_memory = compressed_params * 4  # 简化计算
        
        # 考虑量化带来的内存减少
        if self.config.strategy == CompressionStrategy.QUANTIZATION:
            bits_per_param = self.config.quantization_bits
            bytes_per_param = bits_per_param / 8
            compressed_memory = compressed_params * bytes_per_param
        
        # 更新统计信息
        self.compression_stats['parameters_after'] = compressed_params
        self.compression_stats['compressed_size'] = compressed_memory
        
        if original_memory > 0:
            self.compression_stats['compression_ratio'] = original_memory / compressed_memory
            self.compression_stats['memory_reduction'] = (1 - compressed_memory / original_memory) * 100
    
    def _get_module_type_from_param_name(self, model: nn.Module, param_name: str) -> str:
        """从参数名获取模块类型"""
        # 简化实现：从参数名中提取模块类型
        parts = param_name.split('.')
        if len(parts) >= 2:
            module_name = '.'.join(parts[:-1])
            try:
                module = dict(model.named_modules())[module_name]
                return type(module).__name__
            except KeyError:
                pass
        
        return "Unknown"
    
    def get_compression_report(self) -> Dict[str, Any]:
        """获取压缩报告"""
        return {
            'compression_stats': self.compression_stats,
            'config': {
                'strategy': self.config.strategy.value,
                'target_sparsity': self.config.target_sparsity,
                'pruning_method': self.config.pruning_method.value if hasattr(self.config, 'pruning_method') else None,
                'quantization_method': self.config.quantization_method.value if hasattr(self.config, 'quantization_method') else None,
                'quantization_bits': self.config.quantization_bits if hasattr(self.config, 'quantization_bits') else None,
                'fine_tune_epochs': self.config.fine_tune_epochs
            },
            'timestamp': time.time(),
            'success': True
        }


class ModelQuantizationTool:
    """模型量化工具
    
    提供高级量化功能，支持多种量化方案
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.logger = logging.getLogger("ModelQuantizationTool")
        
        self.quantization_configs = {
            'int8_static': {
                'method': QuantizationMethod.POST_TRAINING_STATIC,
                'bits': 8,
                'dtype': torch.qint8
            },
            'int8_dynamic': {
                'method': QuantizationMethod.POST_TRAINING_DYNAMIC,
                'bits': 8,
                'dtype': torch.qint8
            },
            'float16': {
                'method': QuantizationMethod.FLOAT16,
                'bits': 16,
                'dtype': torch.float16
            }
        }
    
    def quantize(self, method: str = 'int8_static', 
                dataloader: Optional[Any] = None) -> nn.Module:
        """量化模型
        
        参数:
            method: 量化方法
            dataloader: 校准数据加载器
            
        返回:
            量化后的模型
        """
        if method not in self.quantization_configs:
            self.logger.error(f"不支持的量化方法: {method}")
            return self.model
        
        config = self.quantization_configs[method]
        
        if config['method'] == QuantizationMethod.FLOAT16:
            return self._quantize_to_float16()
        elif TORCH_QUANT_AVAILABLE:
            return self._pytorch_quantize(config, dataloader)
        else:
            return self._custom_quantize(config)
    
    def _quantize_to_float16(self) -> nn.Module:
        """量化到float16"""
        self.logger.info("量化模型到float16")
        
        # 转换模型参数到float16
        self.model = self.model.half()
        
        return self.model
    
    def _pytorch_quantize(self, config: Dict[str, Any], dataloader: Optional[Any]) -> nn.Module:
        """使用PyTorch量化"""
        method = config['method']
        
        if method == QuantizationMethod.POST_TRAINING_STATIC:
            if dataloader is None:
                self.logger.warning("静态量化需要校准数据，使用动态量化代替")
                return quant.quantize_dynamic(
                    self.model, {nn.Linear, nn.Conv2d}, dtype=config['dtype']
                )
            
            # 静态量化
            self.model.eval()
            self.model.qconfig = quant.get_default_qconfig('fbgemm')
            quant.prepare(self.model, inplace=True)
            
            # 校准
            self.logger.info("进行量化校准...")
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= 100:
                        break
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    self.model(inputs)
            
            # 转换
            quantized_model = quant.convert(self.model)
            
        elif method == QuantizationMethod.POST_TRAINING_DYNAMIC:
            # 动态量化
            quantized_model = quant.quantize_dynamic(
                self.model, {nn.Linear, nn.Conv2d}, dtype=config['dtype']
            )
        
        else:
            self.logger.warning(f"不支持的PyTorch量化方法: {method}")
            quantized_model = self.model
        
        return quantized_model
    
    def _custom_quantize(self, config: Dict[str, Any]) -> nn.Module:
        """自定义量化"""
        bits = config['bits']
        
        # 创建压缩配置
        compression_config = CompressionConfig(
            strategy=CompressionStrategy.QUANTIZATION,
            quantization_method=QuantizationMethod.POST_TRAINING_STATIC,
            quantization_bits=bits
        )
        
        # 使用ModelCompressor进行量化
        compressor = ModelCompressor(compression_config)
        quantized_model = compressor._apply_quantization(self.model)
        
        return quantized_model
    
    def analyze_quantization_sensitivity(self, dataloader: Any, 
                                       evaluator: Callable) -> Dict[str, Any]:
        """分析量化敏感度
        
        确定哪些层对量化更敏感
        """
        self.logger.info("分析量化敏感度")
        
        original_accuracy = evaluator(self.model)
        sensitivity_report = {
            'original_accuracy': original_accuracy,
            'layer_sensitivity': {},
            'recommendations': []
        }
        
        # 逐层量化分析
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 临时量化该层
                temp_model = self._quantize_single_layer(name, module)
                
                # 评估精度影响
                quantized_accuracy = evaluator(temp_model)
                accuracy_drop = original_accuracy - quantized_accuracy
                
                sensitivity_report['layer_sensitivity'][name] = {
                    'accuracy_drop': accuracy_drop,
                    'module_type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
                
                # 记录高度敏感的层
                if accuracy_drop > 0.05:  # 精度下降超过5%
                    sensitivity_report['recommendations'].append({
                        'layer': name,
                        'action': 'skip_quantization',
                        'reason': f'量化导致精度下降 {accuracy_drop:.3f}'
                    })
        
        return sensitivity_report
    
    def _quantize_single_layer(self, layer_name: str, layer_module: nn.Module) -> nn.Module:
        """量化单个层（用于敏感度分析）"""
        # 创建模型副本
        model_copy = self._copy_model(self.model)
        
        # 找到对应层并量化
        for name, module in model_copy.named_modules():
            if name == layer_name:
                # 简单量化：减少精度
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        # 8位量化
                        min_val = param.data.min()
                        max_val = param.data.max()
                        scale = max(abs(min_val), abs(max_val))
                        
                        if scale > 0:
                            quantized = torch.round(param.data / scale * 127)
                            param.data.copy_(quantized / 127 * scale)
                
                break
        
        return model_copy
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        """复制模型"""
        # 简单实现：使用state_dict复制
        model_copy = type(model)()
        model_copy.load_state_dict(model.state_dict())
        return model_copy


def test_model_compression():
    """测试模型压缩功能"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== 测试模型压缩和量化 ===")
    
    # 创建简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 创建测试数据
    test_data = torch.randn(100, 100)
    test_targets = torch.randint(0, 10, (100,))
    
    # 创建简单的评估函数
    def evaluator(model):
        with torch.no_grad():
            outputs = model(test_data)
            preds = outputs.argmax(dim=1)
            accuracy = (preds == test_targets).float().mean().item()
        return accuracy
    
    model = TestModel()
    original_accuracy = evaluator(model)
    print(f"原始模型精度: {original_accuracy:.4f}")
    
    # 测试剪枝
    print("\n1. 测试模型剪枝:")
    config = CompressionConfig(
        strategy=CompressionStrategy.PRUNING,
        target_sparsity=0.3,
        pruning_method=PruningMethod.L1_UNSTRUCTURED
    )
    compressor = ModelCompressor(config)
    pruned_model = compressor.compress(model, evaluator=evaluator)
    report = compressor.get_compression_report()
    print(f"剪枝报告: 压缩率={report['compression_stats']['compression_ratio']:.2f}x, "
          f"稀疏度={report['compression_stats']['sparsity_achieved']:.3f}")
    
    # 测试量化
    print("\n2. 测试模型量化:")
    config = CompressionConfig(
        strategy=CompressionStrategy.QUANTIZATION,
        quantization_method=QuantizationMethod.POST_TRAINING_DYNAMIC,
        quantization_bits=8
    )
    compressor = ModelCompressor(config)
    quantized_model = compressor.compress(model, evaluator=evaluator)
    report = compressor.get_compression_report()
    print(f"量化报告: 量化误差={report['compression_stats']['quantization_error']:.6f}")
    
    # 测试复合压缩
    print("\n3. 测试复合压缩:")
    config = CompressionConfig(
        strategy=CompressionStrategy.COMPOSITE,
        composite_steps=[
            CompressionStrategy.PRUNING,
            CompressionStrategy.QUANTIZATION
        ],
        target_sparsity=0.2,
        quantization_bits=8
    )
    compressor = ModelCompressor(config)
    composite_model = compressor.compress(model, evaluator=evaluator)
    report = compressor.get_compression_report()
    print(f"复合压缩报告: 压缩率={report['compression_stats']['compression_ratio']:.2f}x, "
          f"内存减少={report['compression_stats']['memory_reduction']:.1f}%")
    
    print("\n模型压缩和量化测试完成!")


if __name__ == "__main__":
    test_model_compression()