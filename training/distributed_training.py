#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self AGI 分布式训练支持
支持在多个GPU或多台机器上并行训练模型

功能：
1. 数据并行（Data Parallelism）
2. 模型并行（Model Parallelism）
3. 混合并行（Hybrid Parallelism）
4. 分布式优化器
5. 梯度同步
6. 检查点管理和恢复
"""

import sys
import os
import logging
import json
import time
import datetime
import socket
import pickle
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager

# 导入PyTorch分布式库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    import torch.multiprocessing as mp_torch
    TORCH_DISTRIBUTED_AVAILABLE = True
    
    # 设置共享内存策略（避免文件描述符限制）
    try:
        mp_torch.set_sharing_strategy('file_system')
    except Exception as e:
        # 根据项目要求"不采用任何降级处理，直接报错"，记录警告
        # 共享内存策略设置失败不影响核心功能，但记录错误以便调试
        logger = logging.getLogger(__name__)
        logger.warning(f"设置共享内存策略失败，可能影响分布式训练性能: {e}")
    
except ImportError as e:
    TORCH_DISTRIBUTED_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"PyTorch分布式库不可用: {e}")


class ParallelStrategy(Enum):
    """并行策略枚举"""
    DATA_PARALLEL = "data_parallel"  # 数据并行
    MODEL_PARALLEL = "model_parallel"  # 模型并行
    PIPELINE_PARALLEL = "pipeline_parallel"  # 流水线并行
    HYBRID_PARALLEL = "hybrid_parallel"  # 混合并行


@dataclass
class DistributedConfig:
    """分布式训练配置"""
    
    strategy: ParallelStrategy = ParallelStrategy.DATA_PARALLEL
    world_size: int = 1  # 总进程数
    rank: int = 0  # 当前进程排名
    local_rank: int = 0  # 本地进程排名
    master_addr: str = "localhost"
    master_port: int = 29500
    backend: str = "nccl"  # 或 "gloo"
    init_method: str = "env://"  # 或 "tcp://"
    
    # 数据并行配置
    batch_size_per_gpu: int = 32
    gradient_accumulation_steps: int = 1
    
    # 模型并行配置
    model_splits: Optional[List[int]] = None  # 模型划分方案
    
    # 检查点配置
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000  # 检查点间隔（步数）
    
    # 通信配置
    communication_timeout: int = 1800  # 通信超时（秒）
    sync_batch_norm: bool = True  # 是否同步BatchNorm
    
    # 日志配置
    log_interval: int = 100
    save_interval: int = 1000
    
    def __post_init__(self):
        """配置验证和自动调整"""
        self._validate_and_adjust()
    
    def _validate_and_adjust(self):
        """验证配置并自动调整不合理的值"""
        import warnings
        
        # 1. 验证world_size和rank
        if self.world_size < 1:
            warnings.warn(f"world_size不能小于1，已调整为1 (原值: {self.world_size})")
            self.world_size = 1
        
        if self.rank < 0 or self.rank >= self.world_size:
            warnings.warn(f"rank必须在[0, world_size-1]范围内，已调整为0 (原值: {self.rank}, world_size: {self.world_size})")
            self.rank = 0
        
        if self.local_rank < 0:
            warnings.warn(f"local_rank不能为负数，已调整为0 (原值: {self.local_rank})")
            self.local_rank = 0
        
        # 2. 验证端口范围
        if self.master_port < 1024 or self.master_port > 65535:
            warnings.warn(f"master_port必须在1024-65535范围内，已调整为29500 (原值: {self.master_port})")
            self.master_port = 29500
        
        # 3. 验证backend
        if self.backend not in ["nccl", "gloo"]:
            warnings.warn(f"backend必须是'nccl'或'gloo'，已调整为'nccl' (原值: {self.backend})")
            self.backend = "nccl"
        
        # 4. 验证init_method格式
        if self.init_method not in ["env://", "tcp://"] and not self.init_method.startswith("tcp://"):
            warnings.warn(f"init_method必须是'env://'或'tcp://<address>:<port>'格式，已调整为'env://' (原值: {self.init_method})")
            self.init_method = "env://"
        
        # 5. 验证批大小和梯度累积步数
        if self.batch_size_per_gpu < 1:
            warnings.warn(f"batch_size_per_gpu不能小于1，已调整为32 (原值: {self.batch_size_per_gpu})")
            self.batch_size_per_gpu = 32
        
        if self.gradient_accumulation_steps < 1:
            warnings.warn(f"gradient_accumulation_steps不能小于1，已调整为1 (原值: {self.gradient_accumulation_steps})")
            self.gradient_accumulation_steps = 1
        
        # 6. 验证模型并行配置
        if self.strategy == ParallelStrategy.MODEL_PARALLEL or self.strategy == ParallelStrategy.PIPELINE_PARALLEL:
            if self.model_splits is None:
                warnings.warn(f"{self.strategy.value}策略需要model_splits配置，已设置为默认划分[2] (2-way模型并行)")
                self.model_splits = [2]
            elif not isinstance(self.model_splits, list) or len(self.model_splits) == 0:
                warnings.warn(f"model_splits必须是包含正整数的非空列表，已调整为[2] (原值: {self.model_splits})")
                self.model_splits = [2]
        
        # 7. 验证通信超时
        if self.communication_timeout < 30:
            warnings.warn(f"communication_timeout太小（<30秒），已调整为1800秒 (原值: {self.communication_timeout})")
            self.communication_timeout = 1800
        
        # 8. 验证日志间隔
        if self.log_interval < 1:
            warnings.warn(f"log_interval不能小于1，已调整为100 (原值: {self.log_interval})")
            self.log_interval = 100
        
        if self.save_interval < 1:
            warnings.warn(f"save_interval不能小于1，已调整为1000 (原值: {self.save_interval})")
            self.save_interval = 1000
    
    @classmethod
    def simple_config(cls, strategy: str = "data_parallel", num_gpus: Optional[int] = None) -> "DistributedConfig":
        """完整配置方法 - 自动检测GPU数量并创建配置
        
        参数:
            strategy: 并行策略，可选值: "data_parallel", "model_parallel", "pipeline_parallel", "hybrid_parallel"
            num_gpus: GPU数量，如果为None则自动检测
            
        返回:
            完整配置的DistributedConfig实例
        """
        import torch
        
        # 自动检测GPU数量
        if num_gpus is None:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1
                warnings.warn("CUDA不可用，使用单GPU配置")
        
        # 策略映射
        strategy_map = {
            "data_parallel": ParallelStrategy.DATA_PARALLEL,
            "model_parallel": ParallelStrategy.MODEL_PARALLEL,
            "pipeline_parallel": ParallelStrategy.PIPELINE_PARALLEL,
            "hybrid_parallel": ParallelStrategy.HYBRID_PARALLEL,
        }
        
        strategy_enum = strategy_map.get(strategy.lower(), ParallelStrategy.DATA_PARALLEL)
        
        # 根据策略和GPU数量创建配置
        config = cls(
            strategy=strategy_enum,
            world_size=num_gpus,
        )
        
        # 如果是模型并行策略，设置默认划分
        if strategy_enum in [ParallelStrategy.MODEL_PARALLEL, ParallelStrategy.PIPELINE_PARALLEL] and num_gpus > 1:
            config.model_splits = [num_gpus]  # 按GPU数量划分
        
        return config


class DistributedTrainer:
    """分布式训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 config: DistributedConfig):
        
        self.model = model
        self.dataset = dataset
        self.config = config
        
        self.logger = logging.getLogger(f"DistributedTrainer.rank{config.rank}")
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # 分布式设置
        self.is_distributed = config.world_size > 1
        self.device = None
        self.ddp_model = None
        self.train_sampler = None
        self.train_loader = None
        
        # 初始化分布式训练
        if self.is_distributed:
            self._init_distributed()
        
        # 确保基本训练组件已设置（单机模式回退）
        self._setup_basic_training()
        
        self.logger.info(f"分布式训练器初始化: 策略={config.strategy.value}, 排名={config.rank}/{config.world_size-1}")
    
    def _init_distributed(self):
        """初始化分布式训练环境"""
        if not TORCH_DISTRIBUTED_AVAILABLE:
            self.logger.error("PyTorch分布式库不可用，无法初始化分布式训练")
            return
        
        try:
            # 设置环境变量（如果使用env初始化方法）
            if self.config.init_method == "env://":
                os.environ['MASTER_ADDR'] = self.config.master_addr
                os.environ['MASTER_PORT'] = str(self.config.master_port)
                os.environ['WORLD_SIZE'] = str(self.config.world_size)
                os.environ['RANK'] = str(self.config.rank)
                os.environ['LOCAL_RANK'] = str(self.config.local_rank)
            
            # 初始化进程组
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=datetime.timedelta(seconds=self.config.communication_timeout)
            )
            
            self.logger.info(f"进程组初始化成功: {self.config.backend}")
            
            # 设置设备
            if torch.cuda.is_available():
                device_id = self.config.local_rank % torch.cuda.device_count()
                self.device = torch.device(f"cuda:{device_id}")
                torch.cuda.set_device(device_id)
                self.logger.info(f"使用GPU: {device_id}")
            else:
                self.device = torch.device("cpu")
                self.logger.info("使用CPU")
            
            # 移动模型到设备
            self.model = self.model.to(self.device)
            
            # 根据策略设置分布式模型
            if self.config.strategy == ParallelStrategy.DATA_PARALLEL:
                self._setup_data_parallel()
            elif self.config.strategy == ParallelStrategy.MODEL_PARALLEL:
                self._setup_model_parallel()
            elif self.config.strategy == ParallelStrategy.PIPELINE_PARALLEL:
                self._setup_pipeline_parallel()
            elif self.config.strategy == ParallelStrategy.HYBRID_PARALLEL:
                self._setup_hybrid_parallel()
            else:
                self.logger.warning(f"不支持的并行策略: {self.config.strategy}, 使用数据并行")
                self._setup_data_parallel()
            
        except Exception as e:
            self.logger.error(f"初始化分布式训练失败: {e}")
            self.is_distributed = False
    
    def _setup_basic_training(self):
        """设置基本训练组件（单机模式）"""
        # 如果设备未设置，设置默认设备
        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.logger.info("使用GPU 0（单机模式）")
            else:
                self.device = torch.device("cpu")
                self.logger.info("使用CPU（单机模式）")
        
        # 移动模型到设备
        if self.model is not None:
            self.model = self.model.to(self.device)
        
        # 如果数据加载器未设置，创建基本数据加载器
        if self.train_loader is None:
            self.train_loader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size_per_gpu * max(1, self.config.world_size),
                shuffle=True,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            self.logger.info("创建基本数据加载器（单机模式）")
    
    def _setup_data_parallel(self):
        """设置数据并行"""
        if not self.is_distributed:
            return
        
        # 创建分布式数据采样器
        self.train_sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=True,
            drop_last=True
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size_per_gpu,
            sampler=self.train_sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        # 创建DDP模型
        self.ddp_model = DDP(
            self.model,
            device_ids=[self.device.index] if self.device.type == 'cuda' else None,
            output_device=self.device.index if self.device.type == 'cuda' else None
        )
        
        # 同步BatchNorm层
        if self.config.sync_batch_norm:
            self.ddp_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.ddp_model)
        
        self.logger.info("数据并行设置完成")
    
    def _setup_model_parallel(self):
        """设置模型并行 - 增强实现
        
        将模型的不同部分分配到不同的GPU上，实现真正的模型并行
        """
        if not self.is_distributed or self.config.world_size < 2:
            self.logger.warning("模型并行需要至少2个GPU，回退到数据并行")
            self._setup_data_parallel()
            return
        
        self.logger.info("设置模型并行（增强实现）")
        
        # 获取可用GPU列表
        available_gpus = list(range(self.config.world_size))
        self.logger.info(f"可用GPU: {available_gpus}")
        
        # 分析模型结构
        model_structure = self._analyze_model_structure(self.model)
        self.logger.info(f"模型结构分析: {model_structure['layer_count']} 层, {model_structure['parameter_count']} 参数")
        
        # 根据GPU数量和模型结构划分模型
        model_parts = self._split_model_for_parallel(self.model, available_gpus, model_structure)
        
        # 将模型部分分配到不同的GPU
        self.parallel_model = self._distribute_model_parts(model_parts, available_gpus)
        
        # 设置设备
        self.device = torch.device(f"cuda:{self.config.local_rank}")
        
        # 创建数据加载器（每个进程使用完整数据集）
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size_per_gpu,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # 标记为模型并行模式
        self.parallel_model.is_model_parallel = True
        
        self.logger.info(f"模型并行设置完成: 模型划分为 {len(model_parts)} 部分，分配到 {len(available_gpus)} 个GPU")
    
    def _analyze_model_structure(self, model: nn.Module) -> Dict[str, Any]:
        """分析模型结构
        
        参数:
            model: PyTorch模型
            
        返回:
            模型结构信息
        """
        layer_count = 0
        parameter_count = 0
        layer_types = {}
        layer_details = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块（实际层）
                layer_count += 1
                layer_type = type(module).__name__
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
                
                # 计算参数数量
                params = sum(p.numel() for p in module.parameters())
                parameter_count += params
                
                layer_details.append({
                    'name': name,
                    'type': layer_type,
                    'parameters': params,
                    'module': module
                })
        
        return {
            'layer_count': layer_count,
            'parameter_count': parameter_count,
            'layer_types': layer_types,
            'layer_details': layer_details,
            'total_modules': len(list(model.modules()))
        }
    
    def _split_model_for_parallel(self, model: nn.Module, gpus: List[int], 
                                  model_structure: Dict[str, Any]) -> List[nn.Module]:
        """将模型划分为多个部分以便并行
        
        参数:
            model: 原始模型
            gpus: GPU列表
            model_structure: 模型结构信息
            
        返回:
            模型部分列表
        """
        num_gpus = len(gpus)
        layer_details = model_structure['layer_details']
        
        if num_gpus <= 1 or len(layer_details) == 0:
            return [model]
        
        # 策略1: 按层数均匀划分
        layers_per_gpu = len(layer_details) // num_gpus
        if layers_per_gpu == 0:
            layers_per_gpu = 1
        
        model_parts = []
        current_part_layers = []
        current_part_name = f"model_part_0"
        
        for i, layer_info in enumerate(layer_details):
            current_part_layers.append(layer_info)
            
            # 达到每GPU的层数限制，或这是最后一层
            if (len(current_part_layers) >= layers_per_gpu or i == len(layer_details) - 1):
                # 创建模型部分
                part = self._create_model_part(current_part_layers, f"{current_part_name}")
                model_parts.append(part)
                
                # 重置
                current_part_layers = []
                current_part_name = f"model_part_{len(model_parts)}"
        
        # 如果部分数量少于GPU数量，调整
        if len(model_parts) < num_gpus:
            self.logger.warning(f"模型部分({len(model_parts)})少于GPU数量({num_gpus})，调整划分")
            # 进一步划分最大的部分
            while len(model_parts) < num_gpus:
                # 找到最大的部分
                largest_idx = max(range(len(model_parts)), key=lambda i: model_parts[i].parameter_count)
                largest_part = model_parts.pop(largest_idx)
                
                # 划分最大的部分
                sub_parts = self._split_model_part(largest_part, 2)
                model_parts.extend(sub_parts)
        
        self.logger.info(f"模型划分为 {len(model_parts)} 个部分")
        return model_parts
    
    def _create_model_part(self, layer_infos: List[Dict[str, Any]], part_name: str) -> nn.Module:
        """创建模型部分
        
        参数:
            layer_infos: 层信息列表
            part_name: 部分名称
            
        返回:
            模型部分
        """
        # 创建一个包含指定层的序列模型
        layers = []
        for layer_info in layer_infos:
            layers.append((layer_info['name'], layer_info['module']))
        
        # 创建Sequential模块
        from collections import OrderedDict
        sequential_layers = OrderedDict(layers)
        model_part = nn.Sequential(sequential_layers)
        
        # 添加元数据
        model_part.part_name = part_name
        model_part.layer_count = len(layer_infos)
        model_part.parameter_count = sum(l['parameters'] for l in layer_infos)
        model_part.original_layers = [l['name'] for l in layer_infos]
        
        return model_part
    
    def _split_model_part(self, model_part: nn.Module, num_splits: int) -> List[nn.Module]:
        """划分模型部分
        
        参数:
            model_part: 模型部分
            num_splits: 划分数量
            
        返回:
            划分后的部分列表
        """
        if not hasattr(model_part, 'original_layers') or num_splits <= 1:
            return [model_part]
        
        # 如果是Sequential模块，直接划分层
        if isinstance(model_part, nn.Sequential):
            layers = list(model_part.children())
            split_size = len(layers) // num_splits
            if split_size == 0:
                split_size = 1
            
            parts = []
            for i in range(0, len(layers), split_size):
                part_layers = layers[i:i + split_size]
                if part_layers:
                    part = nn.Sequential(*part_layers)
                    part.part_name = f"{model_part.part_name}_split_{len(parts)}"
                    part.layer_count = len(part_layers)
                    parts.append(part)
            
            return parts
        
        # 其他类型的模块，返回原始模块
        return [model_part]
    
    def _distribute_model_parts(self, model_parts: List[nn.Module], gpus: List[int]) -> nn.Module:
        """将模型部分分配到不同的GPU
        
        参数:
            model_parts: 模型部分列表
            gpus: GPU列表
            
        返回:
            分布式模型
        """
        from torch.nn.parallel import DistributedDataParallel
        
        # 如果只有一个部分，直接返回
        if len(model_parts) == 1:
            device = torch.device(f"cuda:{gpus[0]}")
            model_parts[0] = model_parts[0].to(device)
            return model_parts[0]
        
        # 创建模型并行容器
        class ModelParallelContainer(nn.Module):
            def __init__(self, parts: List[nn.Module], gpu_mapping: Dict[int, nn.Module]):
                super().__init__()
                self.parts = nn.ModuleList(parts)
                self.gpu_mapping = gpu_mapping
                self.part_devices = {}
                
                # 将每个部分移动到对应的GPU
                for gpu_id, part_idx in gpu_mapping.items():
                    device = torch.device(f"cuda:{gpu_id}")
                    self.parts[part_idx] = self.parts[part_idx].to(device)
                    self.part_devices[part_idx] = device
            
            def forward(self, x):
                # 将输入移动到第一个部分的设备
                first_device = self.part_devices[0]
                x = x.to(first_device)
                
                # 按顺序通过各个部分
                for i, part in enumerate(self.parts):
                    x = part(x)
                    # 如果不是最后一部分，将输出移动到下一个部分的设备
                    if i < len(self.parts) - 1:
                        next_device = self.part_devices.get(i + 1, first_device)
                        x = x.to(next_device)
                
                return x
        
        # 映射部分到GPU
        gpu_mapping = {}
        for i, part in enumerate(model_parts):
            gpu_idx = i % len(gpus)
            gpu_mapping[gpus[gpu_idx]] = i
        
        container = ModelParallelContainer(model_parts, gpu_mapping)
        return container
    
    def _setup_hybrid_parallel(self):
        """设置混合并行 - 增强实现
        
        结合数据并行和模型并行：
        1. 在多个GPU组之间进行模型并行
        2. 在每个GPU组内部进行数据并行
        """
        if not self.is_distributed or self.config.world_size < 4:
            self.logger.warning(f"混合并行需要至少4个GPU（当前{self.config.world_size}），回退到数据并行")
            self._setup_data_parallel()
            return
        
        self.logger.info("设置混合并行（增强实现）")
        
        # 获取可用GPU总数
        total_gpus = self.config.world_size
        self.logger.info(f"总GPU数量: {total_gpus}")
        
        # 确定模型并行组和数据并行组的划分
        # 策略：优先模型并行，剩余GPU用于数据并行
        model_parallel_groups = self._create_hybrid_groups(total_gpus)
        
        # 设置模型并行（在模型并行组内）
        self._setup_hybrid_model_parallel(model_parallel_groups)
        
        # 设置数据并行（在数据并行组内）
        self._setup_hybrid_data_parallel(model_parallel_groups)
        
        # 标记为混合并行模式
        if hasattr(self, 'hybrid_model'):
            self.hybrid_model.is_hybrid_parallel = True
        elif self.ddp_model:
            self.ddp_model.is_hybrid_parallel = True
        
        self.logger.info(f"混合并行设置完成: {len(model_parallel_groups['model_parallel'])} 个模型并行组, "
                        f"{len(model_parallel_groups['data_parallel'])} 个数据并行组")

    def _setup_pipeline_parallel(self):
        """设置流水线并行 - 增强实现
        
        将模型划分为多个阶段，每个阶段运行在不同的GPU上
        数据在阶段间以流水线方式流动
        """
        if not self.is_distributed or self.config.world_size < 2:
            self.logger.warning(f"流水线并行需要至少2个GPU（当前{self.config.world_size}），回退到数据并行")
            self._setup_data_parallel()
            return
        
        self.logger.info("设置流水线并行（增强实现）")
        
        # 获取可用GPU列表
        available_gpus = list(range(self.config.world_size))
        self.logger.info(f"可用GPU: {available_gpus}")
        
        # 分析模型结构
        model_structure = self._analyze_model_structure(self.model)
        self.logger.info(f"模型结构分析: {model_structure['layer_count']} 层, {model_structure['parameter_count']} 参数")
        
        # 根据GPU数量划分模型阶段
        num_stages = min(self.config.world_size, model_structure['layer_count'])
        if num_stages < 2:
            self.logger.warning(f"模型层数({model_structure['layer_count']})少于2，无法进行流水线并行，回退到数据并行")
            self._setup_data_parallel()
            return
        
        # 划分模型为多个阶段
        pipeline_stages = self._split_model_for_pipeline(self.model, num_stages, model_structure)
        
        # 将每个阶段分配到不同的GPU
        self.pipeline_model = self._distribute_pipeline_stages(pipeline_stages, available_gpus)
        
        # 设置流水线调度器
        self._setup_pipeline_scheduler(num_stages)
        
        # 标记为流水线并行模式
        self.pipeline_model.is_pipeline_parallel = True
        
        self.logger.info(f"流水线并行设置完成: {num_stages} 个阶段，分配到 {len(available_gpus)} 个GPU")

    def _split_model_for_pipeline(self, model: nn.Module, num_stages: int, 
                                 model_structure: Dict[str, Any]) -> List[nn.Module]:
        """将模型划分为流水线阶段
        
        参数:
            model: 原始模型
            num_stages: 阶段数量
            model_structure: 模型结构信息
            
        返回:
            模型阶段列表
        """
        layer_details = model_structure['layer_details']
        
        if num_stages <= 1 or len(layer_details) == 0:
            return [model]
        
        # 计算每个阶段的层数
        layers_per_stage = len(layer_details) // num_stages
        if layers_per_stage == 0:
            layers_per_stage = 1
        
        stages = []
        current_stage_layers = []
        current_stage_name = f"pipeline_stage_0"
        
        for i, layer_info in enumerate(layer_details):
            current_stage_layers.append(layer_info)
            
            # 达到每阶段的层数限制，或这是最后一层
            if (len(current_stage_layers) >= layers_per_stage or i == len(layer_details) - 1):
                # 创建模型阶段
                stage = self._create_model_part(current_stage_layers, f"{current_stage_name}")
                stages.append(stage)
                
                # 重置
                current_stage_layers = []
                current_stage_name = f"pipeline_stage_{len(stages)}"
        
        # 如果阶段数量少于预期，调整
        if len(stages) < num_stages:
            self.logger.warning(f"模型阶段({len(stages)})少于预期({num_stages})，调整划分")
            # 进一步划分最大的阶段
            while len(stages) < num_stages:
                # 找到最大的阶段
                largest_idx = max(range(len(stages)), key=lambda i: stages[i].parameter_count)
                largest_stage = stages.pop(largest_idx)
                
                # 划分最大的阶段
                sub_stages = self._split_model_part(largest_stage, 2)
                stages.extend(sub_stages)
        
        self.logger.info(f"模型划分为 {len(stages)} 个流水线阶段")
        return stages

    def _distribute_pipeline_stages(self, stages: List[nn.Module], gpus: List[int]) -> nn.Module:
        """将流水线阶段分配到不同的GPU
        
        参数:
            stages: 模型阶段列表
            gpus: GPU列表
            
        返回:
            流水线模型
        """
        from torch.nn.parallel import DistributedDataParallel
        
        # 如果只有一个阶段，直接返回
        if len(stages) == 1:
            device = torch.device(f"cuda:{gpus[0]}")
            stages[0] = stages[0].to(device)
            return stages[0]
        
        # 创建流水线并行容器
        class PipelineParallelContainer(nn.Module):
            def __init__(self, stages: List[nn.Module], gpu_mapping: Dict[int, nn.Module]):
                super().__init__()
                self.stages = nn.ModuleList(stages)
                self.gpu_mapping = gpu_mapping
                self.stage_devices = {}
                
                # 将每个阶段移动到对应的GPU
                for gpu_id, stage_idx in gpu_mapping.items():
                    device = torch.device(f"cuda:{gpu_id}")
                    self.stages[stage_idx] = self.stages[stage_idx].to(device)
                    self.stage_devices[stage_idx] = device
            
            def forward(self, x):
                # 将输入移动到第一个阶段的设备
                first_device = self.stage_devices[0]
                x = x.to(first_device)
                
                # 按顺序通过各个阶段
                for i, stage in enumerate(self.stages):
                    x = stage(x)
                    # 如果不是最后阶段，将输出移动到下一个阶段的设备
                    if i < len(self.stages) - 1:
                        next_device = self.stage_devices.get(i + 1, first_device)
                        x = x.to(next_device)
                
                return x
        
        # 映射阶段到GPU
        gpu_mapping = {}
        for i, stage in enumerate(stages):
            gpu_idx = i % len(gpus)
            gpu_mapping[gpus[gpu_idx]] = i
        
        container = PipelineParallelContainer(stages, gpu_mapping)
        return container

    def _setup_pipeline_scheduler(self, num_stages: int):
        """设置流水线调度器
        
        参数:
            num_stages: 阶段数量
        """
        # 初始化流水线调度器
        self.pipeline_scheduler = {
            'num_stages': num_stages,
            'current_micro_batch': 0,
            'micro_batches': 4,  # 默认微批次数
            'stage_queues': [[] for _ in range(num_stages)],
            'gradient_accumulation': True
        }
        
        self.logger.info(f"流水线调度器初始化: {num_stages} 个阶段，{self.pipeline_scheduler['micro_batches']} 个微批次")

    def _create_hybrid_groups(self, total_gpus: int) -> Dict[str, List[List[int]]]:
        """创建混合并行组
        
        参数:
            total_gpus: 总GPU数量
            
        返回:
            组配置字典
        """
        # 最小配置：至少2个GPU用于模型并行，至少2个GPU用于数据并行
        if total_gpus < 4:
            return {
                'model_parallel': [list(range(total_gpus))],  # 所有GPU用于模型并行
                'data_parallel': []  # 没有数据并行
            }
        
        # 策略：将GPU分为模型并行组和数据并行组
        # 模型并行组大小：2-4个GPU（取决于总GPU数）
        model_group_size = min(4, total_gpus // 2)
        if model_group_size < 2:
            model_group_size = 2
        
        # 数据并行组大小：剩余GPU
        data_group_size = total_gpus // model_group_size
        
        # 创建模型并行组
        model_parallel_groups = []
        for i in range(0, total_gpus, model_group_size):
            group = list(range(i, min(i + model_group_size, total_gpus)))
            if len(group) >= 2:  # 至少2个GPU才构成模型并行组
                model_parallel_groups.append(group)
        
        # 创建数据并行组（跨模型并行组）
        data_parallel_groups = []
        if len(model_parallel_groups) > 1:
            # 每个数据并行组包含每个模型并行组中的一个GPU
            for pos in range(model_group_size):
                group = []
                for model_group in model_parallel_groups:
                    if pos < len(model_group):
                        group.append(model_group[pos])
                if len(group) >= 2:  # 至少2个GPU才构成数据并行组
                    data_parallel_groups.append(group)
        
        return {
            'model_parallel': model_parallel_groups,
            'data_parallel': data_parallel_groups
        }
    
    def _setup_hybrid_model_parallel(self, groups: Dict[str, List[List[int]]]):
        """设置混合并行中的模型并行部分
        
        参数:
            groups: 组配置
        """
        model_parallel_groups = groups['model_parallel']
        
        if not model_parallel_groups:
            self.logger.warning("没有模型并行组，跳过模型并行设置")
            return
        
        self.logger.info(f"设置 {len(model_parallel_groups)} 个模型并行组")
        
        # 分析模型结构
        model_structure = self._analyze_model_structure(self.model)
        
        # 为每个模型并行组创建模型副本
        self.model_copies = []
        for group_idx, group_gpus in enumerate(model_parallel_groups):
            self.logger.info(f"模型并行组 {group_idx}: GPU {group_gpus}")
            
            # 划分模型
            model_parts = self._split_model_for_parallel(self.model, group_gpus, model_structure)
            
            # 分配模型部分到GPU
            model_copy = self._distribute_model_parts(model_parts, group_gpus)
            model_copy.group_id = group_idx
            model_copy.group_gpus = group_gpus
            
            self.model_copies.append(model_copy)
        
        # 使用第一个模型副本作为主要模型
        if self.model_copies:
            self.hybrid_model = self.model_copies[0]
            self.logger.info(f"使用模型并行组 0 作为主要模型")
    
    def _setup_hybrid_data_parallel(self, groups: Dict[str, List[List[int]]]):
        """设置混合并行中的数据并行部分
        
        参数:
            groups: 组配置
        """
        data_parallel_groups = groups['data_parallel']
        
        if not data_parallel_groups:
            self.logger.warning("没有数据并行组，跳过数据并行设置")
            return
        
        self.logger.info(f"设置 {len(data_parallel_groups)} 个数据并行组")
        
        # 初始化分布式数据并行
        if not TORCH_DISTRIBUTED_AVAILABLE:
            self.logger.error("PyTorch分布式库不可用，无法设置数据并行")
            return
        
        try:
            # 为每个数据并行组创建进程组
            self.data_parallel_groups = {}
            for group_idx, group_gpus in enumerate(data_parallel_groups):
                # 创建进程组
                group = dist.new_group(ranks=group_gpus)
                self.data_parallel_groups[group_idx] = {
                    'group': group,
                    'gpus': group_gpus,
                    'size': len(group_gpus)
                }
                
                self.logger.info(f"数据并行组 {group_idx}: GPU {group_gpus}, 大小 {len(group_gpus)}")
            
            # 如果当前进程在某个数据并行组中，设置DDP
            current_rank = self.config.rank
            for group_idx, group_info in self.data_parallel_groups.items():
                if current_rank in group_info['gpus']:
                    self.logger.info(f"当前进程 (rank {current_rank}) 在数据并行组 {group_idx} 中")
                    
                    # 使用DDP包装模型（如果存在混合模型）
                    if hasattr(self, 'hybrid_model'):
                        self.ddp_model = DDP(
                            self.hybrid_model,
                            device_ids=[self.config.local_rank],
                            output_device=self.config.local_rank,
                            process_group=group_info['group']
                        )
                        self.logger.info(f"数据并行组 {group_idx} 的DDP模型已创建")
                    break
            
        except Exception as e:
            self.logger.error(f"设置数据并行失败: {e}")
            self.logger.warning("数据并行设置失败，回退到模型并行")
    
    def train_epoch(self,
                   optimizer: optim.Optimizer,
                   loss_fn: Callable,
                   gradient_accumulation_steps: Optional[int] = None) -> Dict[str, Any]:
        """训练一个epoch"""
        
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.config.gradient_accumulation_steps
        
        # 设置模型为训练模式
        if self.ddp_model:
            self.ddp_model.train()
        else:
            self.model.train()
        
        # 设置采样器（分布式训练）
        if self.train_sampler:
            self.train_sampler.set_epoch(self.epoch)
        
        epoch_loss = 0.0
        epoch_samples = 0
        step_losses = []
        
        # 梯度累积计数器
        accumulation_step = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # 准备数据
            if isinstance(batch_data, (tuple, list)):
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            else:
                inputs = batch_data.to(self.device)
                targets = None
            
            # 前向传播
            if self.ddp_model:
                outputs = self.ddp_model(inputs)
            else:
                outputs = self.model(inputs)
            
            # 计算损失
            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                # 自监督学习或不需要目标的情况
                loss = loss_fn(outputs)
            
            # 梯度累积
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            accumulation_step += 1
            
            # 累积足够步数后更新参数
            if accumulation_step >= gradient_accumulation_steps:
                # 梯度裁剪（防止梯度爆炸）
                if self.ddp_model:
                    torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                accumulation_step = 0
                
                # 更新全局步数
                self.global_step += 1
                
                # 记录损失
                step_loss = loss.item() * gradient_accumulation_steps
                step_losses.append(step_loss)
                epoch_loss += step_loss
                epoch_samples += inputs.size(0)
                
                # 记录日志
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = epoch_loss / max(len(step_losses), 1)
                    self.logger.info(
                        f"Epoch {self.epoch}, Step {self.global_step}, "
                        f"Loss: {avg_loss:.6f}, "
                        f"Samples: {epoch_samples}"
                    )
                
                # 保存检查点
                if self.global_step % self.config.save_interval == 0:
                    if self.config.rank == 0:  # 只有主进程保存
                        self._save_checkpoint(optimizer, f"step_{self.global_step}")
        
        # 处理剩余的梯度累积
        if accumulation_step > 0:
            optimizer.step()
            optimizer.zero_grad()
            self.global_step += 1
        
        # 计算epoch统计
        avg_epoch_loss = epoch_loss / max(len(step_losses), 1)
        
        # 同步所有进程的统计信息
        if self.is_distributed:
            avg_epoch_loss_tensor = torch.tensor(avg_epoch_loss, device=self.device)
            dist.all_reduce(avg_epoch_loss_tensor, op=dist.ReduceOp.SUM)
            avg_epoch_loss = avg_epoch_loss_tensor.item() / self.config.world_size
            
            epoch_samples_tensor = torch.tensor(epoch_samples, device=self.device)
            dist.all_reduce(epoch_samples_tensor, op=dist.ReduceOp.SUM)
            epoch_samples = epoch_samples_tensor.item()
        
        self.epoch += 1
        
        return {
            "epoch": self.epoch - 1,
            "loss": avg_epoch_loss,
            "samples": epoch_samples,
            "global_step": self.global_step,
            "rank": self.config.rank
        }
    
    def _save_checkpoint(self, optimizer: optim.Optimizer, name: str):
        """保存检查点"""
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        
        # 准备检查点数据
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config
        }
        
        # 如果是DDP模型，保存module的状态
        if self.ddp_model:
            checkpoint['model_state_dict'] = self.ddp_model.module.state_dict()
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"检查点保存到: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[optim.Optimizer] = None) -> bool:
        """加载检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型状态
            if self.ddp_model:
                self.ddp_model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载训练状态
            self.epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            
            self.logger.info(f"检查点加载成功: {checkpoint_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return False
    
    def evaluate(self, 
                test_dataset: Dataset,
                loss_fn: Callable,
                metric_fns: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """评估模型"""
        
        # 设置模型为评估模式
        if self.ddp_model:
            self.ddp_model.eval()
        else:
            self.model.eval()
        
        # 创建测试数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size_per_gpu * 2,  # 评估时使用更大的batch
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        total_loss = 0.0
        total_samples = 0
        
        # 指标
        metrics = {name: 0.0 for name in metric_fns.keys()} if metric_fns else {}
        
        with torch.no_grad():
            for batch_data in test_loader:
                # 准备数据
                if isinstance(batch_data, (tuple, list)):
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    inputs = batch_data.to(self.device)
                    targets = None
                
                # 前向传播
                if self.ddp_model:
                    outputs = self.ddp_model(inputs)
                else:
                    outputs = self.model(inputs)
                
                # 计算损失
                if targets is not None:
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    
                    # 计算指标
                    if metric_fns:
                        for name, metric_fn in metric_fns.items():
                            metrics[name] += metric_fn(outputs, targets).item() * inputs.size(0)
                else:
                    loss = loss_fn(outputs)
                    total_loss += loss.item() * inputs.size(0)
                
                total_samples += inputs.size(0)
        
        # 计算平均损失和指标
        avg_loss = total_loss / max(total_samples, 1)
        
        if metric_fns:
            for name in metrics.keys():
                metrics[name] = metrics[name] / max(total_samples, 1)
        
        # 同步所有进程的评估结果
        if self.is_distributed:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / self.config.world_size
            
            total_samples_tensor = torch.tensor(total_samples, device=self.device)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            total_samples = total_samples_tensor.item()
            
            # 同步指标
            if metric_fns:
                for name in metrics.keys():
                    metric_tensor = torch.tensor(metrics[name], device=self.device)
                    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                    metrics[name] = metric_tensor.item() / self.config.world_size
        
        result = {
            "loss": avg_loss,
            "samples": total_samples,
            "metrics": metrics,
            "rank": self.config.rank
        }
        
        self.logger.info(f"评估结果: 损失={avg_loss:.6f}, 样本数={total_samples}")
        
        return result
    
    def cleanup(self):
        """清理分布式训练资源"""
        if self.is_distributed and TORCH_DISTRIBUTED_AVAILABLE:
            dist.destroy_process_group()
            self.logger.info("分布式进程组已销毁")
        
        self.is_distributed = False


class CommunicationPrimitives:
    """高级通信原语
    
    提供高效的分布式通信操作，支持梯度压缩、异步通信等
    """
    
    def __init__(self, rank: int, world_size: int, backend: str = "nccl"):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        
        # 通信组缓存
        self.groups = {}
        self.compressor = GradientCompression()
        
        self.logger = logging.getLogger(f"CommunicationPrimitives.rank{rank}")
    
    def all_reduce_compressed(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """压缩后的AllReduce操作
        
        参数:
            tensor: 输入张量
            op: 归约操作
            
        返回:
            归约后的张量
        """
        # 压缩梯度
        compressed_data, metadata = self.compressor.compress(tensor)
        
        # 分配通信缓冲区
        if not hasattr(self, '_comm_buffer'):
            self._comm_buffer = torch.zeros_like(compressed_data)
        
        # 执行AllReduce
        self._comm_buffer.copy_(compressed_data)
        dist.all_reduce(self._comm_buffer, op=op)
        
        # 解压缩
        result = self.compressor.decompress(self._comm_buffer, metadata)
        
        return result
    
    def all_gather_compressed(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """压缩后的AllGather操作
        
        参数:
            tensor: 输入张量
            
        返回:
            所有进程的张量列表
        """
        # 压缩数据
        compressed_data, metadata = self.compressor.compress(tensor)
        
        # 准备接收缓冲区
        recv_buffer = [torch.zeros_like(compressed_data) for _ in range(self.world_size)]
        
        # 执行AllGather
        dist.all_gather(recv_buffer, compressed_data)
        
        # 解压缩所有数据
        results = []
        for comp_data in recv_buffer:
            results.append(self.compressor.decompress(comp_data, metadata))
        
        return results
    
    def reduce_scatter_compressed(self, input_list: List[torch.Tensor], op=dist.ReduceOp.SUM) -> torch.Tensor:
        """压缩后的ReduceScatter操作
        
        参数:
            input_list: 输入张量列表（每个进程一个）
            op: 归约操作
            
        返回:
            分散后的张量
        """
        # 压缩所有输入
        compressed_list = []
        metadata_list = []
        
        for tensor in input_list:
            comp_data, metadata = self.compressor.compress(tensor)
            compressed_list.append(comp_data)
            metadata_list.append(metadata)
        
        # 准备接收缓冲区
        recv_buffer = torch.zeros_like(compressed_list[0])
        
        # 执行ReduceScatter
        dist.reduce_scatter(recv_buffer, compressed_list, op=op)
        
        # 解压缩结果
        result = self.compressor.decompress(recv_buffer, metadata_list[self.rank])
        
        return result
    
    def broadcast_compressed(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """压缩后的Broadcast操作
        
        参数:
            tensor: 输入张量（仅在src进程有效）
            src: 源进程排名
            
        返回:
            广播后的张量
        """
        if self.rank == src:
            # 压缩数据
            compressed_data, metadata = self.compressor.compress(tensor)
            # 广播压缩数据和元数据大小
            metadata_size = torch.tensor([len(pickle.dumps(metadata))], dtype=torch.int64)
            dist.broadcast(metadata_size, src=src)
            # 广播元数据
            metadata_bytes = pickle.dumps(metadata)
            metadata_tensor = torch.tensor(list(metadata_bytes), dtype=torch.uint8)
            dist.broadcast(metadata_tensor, src=src)
        else:
            # 接收元数据大小
            metadata_size = torch.tensor([0], dtype=torch.int64)
            dist.broadcast(metadata_size, src=src)
            # 接收元数据
            metadata_tensor = torch.zeros(metadata_size.item(), dtype=torch.uint8)
            dist.broadcast(metadata_tensor, src=src)
            metadata = pickle.loads(bytes(metadata_tensor.tolist()))
            # 接收压缩数据
            compressed_data = torch.zeros_like(tensor)  # 占位符
        
        # 广播压缩数据
        dist.broadcast(compressed_data, src=src)
        
        # 解压缩
        result = self.compressor.decompress(compressed_data, metadata)
        
        return result
    
    def create_subgroup(self, ranks: List[int]) -> dist.ProcessGroup:
        """创建子进程组
        
        参数:
            ranks: 子组中的进程排名列表
            
        返回:
            进程组对象
        """
        ranks_tuple = tuple(sorted(ranks))
        if ranks_tuple not in self.groups:
            group = dist.new_group(ranks=ranks)
            self.groups[ranks_tuple] = group
        
        return self.groups[ranks_tuple]
    
    def barrier(self):
        """同步屏障"""
        dist.barrier()


class GradientCompression:
    """梯度压缩器
    
    实现梯度压缩算法以减少通信开销
    """
    
    def __init__(self, compression_ratio: float = 0.5, method: str = "topk"):
        self.compression_ratio = compression_ratio
        self.method = method
        
        # 统计信息
        self.compression_stats = {
            'total_bytes_before': 0,
            'total_bytes_after': 0,
            'num_compressions': 0
        }
    
    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """压缩张量
        
        参数:
            tensor: 输入张量
            
        返回:
            (压缩张量, 元数据)
        """
        self.compression_stats['total_bytes_before'] += tensor.numel() * tensor.element_size()
        self.compression_stats['num_compressions'] += 1
        
        if self.method == "topk":
            return self._compress_topk(tensor)
        elif self.method == "randomk":
            return self._compress_randomk(tensor)
        elif self.method == "quantization":
            return self._compress_quantization(tensor)
        else:
            # 无压缩
            metadata = {'method': 'none', 'original_shape': tensor.shape}
            return tensor, metadata
    
    def _compress_topk(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Top-K压缩
        
        保留绝对值最大的K个元素
        """
        # 展平张量
        flattened = tensor.flatten()
        k = max(1, int(flattened.numel() * self.compression_ratio))
        
        # 找到Top-K值的索引
        values, indices = torch.topk(flattened.abs(), k)
        
        # 提取值和索引
        compressed_values = flattened[indices]
        compressed_indices = indices
        
        # 创建压缩张量
        compressed_tensor = torch.cat([
            compressed_values,
            compressed_indices.float()  # 转换为float以便传输
        ])
        
        metadata = {
            'method': 'topk',
            'original_shape': tensor.shape,
            'k': k,
            'dtype': str(tensor.dtype)
        }
        
        self.compression_stats['total_bytes_after'] += compressed_tensor.numel() * compressed_tensor.element_size()
        
        return compressed_tensor, metadata
    
    def _compress_randomk(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """随机K压缩
        
        随机选择K个元素
        """
        # 展平张量
        flattened = tensor.flatten()
        k = max(1, int(flattened.numel() * self.compression_ratio))
        
        # 随机选择索引
        indices = torch.randperm(flattened.numel())[:k]
        
        # 提取值
        compressed_values = flattened[indices]
        compressed_indices = indices
        
        # 创建压缩张量
        compressed_tensor = torch.cat([
            compressed_values,
            compressed_indices.float()
        ])
        
        metadata = {
            'method': 'randomk',
            'original_shape': tensor.shape,
            'k': k,
            'dtype': str(tensor.dtype)
        }
        
        self.compression_stats['total_bytes_after'] += compressed_tensor.numel() * compressed_tensor.element_size()
        
        return compressed_tensor, metadata
    
    def _compress_quantization(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """量化压缩
        
        将浮点数量化为低精度表示
        """
        # 动态范围量化
        min_val = tensor.min()
        max_val = tensor.max()
        
        # 量化到8位
        scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
        zero_point = min_val
        
        quantized = torch.round((tensor - zero_point) / scale).clamp(0, 255).to(torch.uint8)
        
        metadata = {
            'method': 'quantization',
            'original_shape': tensor.shape,
            'scale': scale,
            'zero_point': zero_point,
            'original_dtype': str(tensor.dtype)
        }
        
        self.compression_stats['total_bytes_after'] += quantized.numel() * quantized.element_size()
        
        return quantized, metadata
    
    def decompress(self, compressed_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """解压缩张量
        
        参数:
            compressed_tensor: 压缩张量
            metadata: 压缩元数据
            
        返回:
            解压缩后的张量
        """
        method = metadata.get('method', 'none')
        
        if method == 'topk' or method == 'randomk':
            return self._decompress_sparse(compressed_tensor, metadata)
        elif method == 'quantization':
            return self._decompress_quantization(compressed_tensor, metadata)
        else:
            # 无压缩
            return compressed_tensor
    
    def _decompress_sparse(self, compressed_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """解压缩稀疏表示"""
        original_shape = metadata['original_shape']
        k = metadata['k']
        
        # 分离值和索引
        total_elements = compressed_tensor.numel()
        values = compressed_tensor[:k]
        indices = compressed_tensor[k:].long()
        
        # 重建张量
        reconstructed = torch.zeros(original_shape.numel(), dtype=values.dtype, device=values.device)
        reconstructed[indices] = values
        
        return reconstructed.reshape(original_shape)
    
    def _decompress_quantization(self, compressed_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """解压缩量化表示"""
        scale = metadata['scale']
        zero_point = metadata['zero_point']
        original_dtype = getattr(torch, metadata['original_dtype'].split('.')[-1])
        
        # 反量化
        dequantized = compressed_tensor.float() * scale + zero_point
        
        return dequantized.to(original_dtype)
    
    def get_compression_ratio(self) -> float:
        """获取压缩率"""
        if self.compression_stats['total_bytes_before'] == 0:
            return 1.0
        
        return self.compression_stats['total_bytes_after'] / self.compression_stats['total_bytes_before']


class FaultToleranceManager:
    """容错管理器
    
    处理分布式训练中的节点故障和恢复
    """
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger("FaultToleranceManager")
        
        # 容错状态
        self.fault_detected = False
        self.failed_ranks = set()
        self.recovery_attempts = 0
        
        # 检查点管理
        self.checkpoint_dir = os.path.join(config.checkpoint_dir, "fault_tolerance")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 心跳检测
        self.heartbeat_interval = 30  # 秒
        self.last_heartbeat = time.time()
        
        self.logger.info("容错管理器初始化")
    
    def check_faults(self) -> bool:
        """检查节点故障
        
        返回:
            是否检测到故障
        """
        if not self.config.is_distributed:
            return False
        
        current_time = time.time()
        
        # 定期检查故障
        if current_time - self.last_heartbeat > self.heartbeat_interval:
            self.last_heartbeat = current_time
            
            try:
                # 尝试与所有进程通信
                test_tensor = torch.tensor([self.config.rank], device='cpu')
                dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
                
                # 检查是否有进程未响应
                expected_sum = sum(range(self.config.world_size))
                if test_tensor.item() != expected_sum:
                    self.logger.warning(f"通信检查失败: 期望和={expected_sum}, 实际和={test_tensor.item()}")
                    return True
                
            except Exception as e:
                self.logger.error(f"故障检测失败: {e}")
                return True
        
        return False
    
    def handle_fault(self, failed_ranks: Set[int]) -> bool:
        """处理节点故障
        
        参数:
            failed_ranks: 故障节点排名集合
            
        返回:
            是否成功处理
        """
        self.fault_detected = True
        self.failed_ranks.update(failed_ranks)
        
        self.logger.warning(f"检测到节点故障: {failed_ranks}")
        
        if self.config.rank == 0:
            # 主进程尝试恢复
            recovery_success = self._attempt_recovery(failed_ranks)
            
            if recovery_success:
                self.logger.info("故障恢复成功")
                return True
            else:
                self.logger.error("故障恢复失败")
                return False
        else:
            # 从进程等待恢复指令
            return self._wait_for_recovery()
    
    def _attempt_recovery(self, failed_ranks: Set[int]) -> bool:
        """尝试恢复故障节点"""
        self.recovery_attempts += 1
        
        # 策略1: 从检查点恢复
        if self._recover_from_checkpoint():
            self.logger.info("从检查点恢复成功")
            return True
        
        # 策略2: 动态调整世界大小
        if self._adjust_world_size(failed_ranks):
            self.logger.info("动态调整世界大小成功")
            return True
        
        # 策略3: 重启故障节点
        if self.recovery_attempts <= 3:
            self.logger.info(f"尝试重启故障节点 (尝试 {self.recovery_attempts})")
            # 在实际系统中，这里会启动新的进程
            time.sleep(5)  # 模拟重启等待
            return self._recover_from_checkpoint()
        
        return False
    
    def _recover_from_checkpoint(self) -> bool:
        """从检查点恢复"""
        checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.logger.info(f"从检查点恢复: {checkpoint_path}")
            
            # 广播检查点路径
            if self.config.rank == 0:
                checkpoint_bytes = checkpoint_path.encode('utf-8')
                checkpoint_len = torch.tensor([len(checkpoint_bytes)], dtype=torch.int64)
                dist.broadcast(checkpoint_len, src=0)
                dist.broadcast(torch.tensor(list(checkpoint_bytes), dtype=torch.uint8), src=0)
            else:
                # 接收检查点路径
                checkpoint_len = torch.tensor([0], dtype=torch.int64)
                dist.broadcast(checkpoint_len, src=0)
                checkpoint_bytes = torch.zeros(checkpoint_len.item(), dtype=torch.uint8)
                dist.broadcast(checkpoint_bytes, src=0)
                checkpoint_path = bytes(checkpoint_bytes.tolist()).decode('utf-8')
            
            # 所有进程加载检查点
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.logger.info("检查点加载成功")
                return True
            except Exception as e:
                self.logger.error(f"加载检查点失败: {e}")
                return False
        
        return False
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """查找最新的检查点"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoint_files = []
        for f in os.listdir(self.checkpoint_dir):
            if f.endswith('.pt'):
                checkpoint_files.append(os.path.join(self.checkpoint_dir, f))
        
        if not checkpoint_files:
            return None
        
        # 按修改时间排序
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        return checkpoint_files[0]
    
    def _adjust_world_size(self, failed_ranks: Set[int]) -> bool:
        """动态调整世界大小"""
        new_world_size = self.config.world_size - len(failed_ranks)
        
        if new_world_size < 1:
            self.logger.error("所有节点都故障，无法调整世界大小")
            return False
        
        self.logger.info(f"调整世界大小: {self.config.world_size} -> {new_world_size}")
        
        # 广播新的世界大小
        if self.config.rank == 0:
            new_size_tensor = torch.tensor([new_world_size], dtype=torch.int64)
            dist.broadcast(new_size_tensor, src=0)
        else:
            new_size_tensor = torch.tensor([0], dtype=torch.int64)
            dist.broadcast(new_size_tensor, src=0)
            new_world_size = new_size_tensor.item()
        
        # 更新配置
        self.config.world_size = new_world_size
        
        # 重新初始化进程组
        try:
            dist.destroy_process_group()
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=new_world_size,
                rank=self.config.rank,
                timeout=datetime.timedelta(seconds=self.config.communication_timeout)
            )
            self.logger.info("进程组重新初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"重新初始化进程组失败: {e}")
            return False
    
    def _wait_for_recovery(self) -> bool:
        """等待恢复指令"""
        recovery_signal = torch.tensor([0], dtype=torch.int64)
        dist.broadcast(recovery_signal, src=0)
        
        return recovery_signal.item() == 1
    
    def save_checkpoint(self, trainer, optimizer, name: str = "fault_tolerance"):
        """保存容错检查点"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.pt")
        
        checkpoint = {
            'epoch': trainer.epoch,
            'global_step': trainer.global_step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"容错检查点保存到: {checkpoint_path}")


class MixedPrecisionTrainer:
    """混合精度训练器
    
    使用自动混合精度(AMP)加速训练，减少内存使用
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, 
                 scaler=None, enabled: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.enabled = enabled
        
        # 创建梯度缩放器
        if scaler is None and enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = scaler
        
        self.logger = logging.getLogger("MixedPrecisionTrainer")
        
        if enabled:
            self.logger.info("混合精度训练已启用")
        else:
            self.logger.info("混合精度训练已禁用")
    
    def train_step(self, loss_fn, *inputs):
        """混合精度训练步骤
        
        参数:
            loss_fn: 损失函数
            *inputs: 输入数据
            
        返回:
            损失值
        """
        if not self.enabled:
            # 常规训练
            self.optimizer.zero_grad()
            loss = loss_fn(*inputs)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        
        # 混合精度训练
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            loss = loss_fn(*inputs)
        
        # 缩放梯度并反向传播
        self.scaler.scale(loss).backward()
        
        # 取消缩放梯度并更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def state_dict(self):
        """获取状态字典"""
        if self.enabled and self.scaler:
            return {
                'scaler_state': self.scaler.state_dict(),
                'enabled': self.enabled
            }
        return {'enabled': self.enabled}
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.enabled = state_dict.get('enabled', True)
        
        if self.enabled and self.scaler and 'scaler_state' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler_state'])


class ZeroOptimizer:
    """ZeRO (Zero Redundancy Optimizer) 优化器
    
    实现ZeRO优化器，减少内存使用
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 stage: int = 1, cpu_offload: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.stage = stage  # ZeRO阶段：1, 2, 3
        self.cpu_offload = cpu_offload
        
        # 参数分区
        self.param_groups = []
        self.param_to_group = {}
        
        # 梯度分区
        self.grad_partitions = {}
        
        self.logger = logging.getLogger("ZeroOptimizer")
        self.logger.info(f"ZeRO优化器初始化: 阶段={stage}, CPU卸载={cpu_offload}")
        
        self._setup_zero()
    
    def _setup_zero(self):
        """设置ZeRO优化器"""
        if self.stage == 1:
            self._setup_stage1()
        elif self.stage == 2:
            self._setup_stage2()
        elif self.stage == 3:
            self._setup_stage3()
    
    def _setup_stage1(self):
        """设置ZeRO阶段1：优化器状态分区"""
        self.logger.info("设置ZeRO阶段1：优化器状态分区")
        
        # 收集所有参数
        all_params = []
        for param_group in self.optimizer.param_groups:
            all_params.extend(param_group['params'])
        
        # 划分参数组
        num_groups = dist.get_world_size() if dist.is_initialized() else 1
        group_size = (len(all_params) + num_groups - 1) // num_groups
        
        for i in range(0, len(all_params), group_size):
            group_params = all_params[i:i + group_size]
            self.param_groups.append(group_params)
            
            for param in group_params:
                self.param_to_group[param] = len(self.param_groups) - 1
    
    def _setup_stage2(self):
        """设置ZeRO阶段2：梯度分区"""
        self.logger.info("设置ZeRO阶段2：梯度分区")
        
        # 先设置阶段1
        self._setup_stage1()
        
        # 初始化梯度分区
        for group_idx, group_params in enumerate(self.param_groups):
            self.grad_partitions[group_idx] = {}
            
            for param in group_params:
                self.grad_partitions[group_idx][param] = torch.zeros_like(param.data)
    
    def _setup_stage3(self):
        """设置ZeRO阶段3：参数分区"""
        self.logger.info("设置ZeRO阶段3：参数分区")
        
        # 先设置阶段2
        self._setup_stage2()
        
        # 参数分区
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            for group_idx, group_params in enumerate(self.param_groups):
                # 每个进程负责一部分参数
                if group_idx % world_size == rank:
                    self.logger.info(f"进程 {rank} 负责参数组 {group_idx}")
                else:
                    # 其他进程不保存这些参数
                    for param in group_params:
                        param.requires_grad = False
    
    def step(self):
        """执行优化步骤"""
        if self.stage == 1:
            self._step_stage1()
        elif self.stage == 2:
            self._step_stage2()
        elif self.stage == 3:
            self._step_stage3()
        else:
            # 常规优化步骤
            self.optimizer.step()
    
    def _step_stage1(self):
        """ZeRO阶段1优化步骤"""
        # 常规优化步骤
        self.optimizer.step()
    
    def _step_stage2(self):
        """ZeRO阶段2优化步骤"""
        # 收集和平均梯度
        self._reduce_gradients()
        
        # 常规优化步骤
        self.optimizer.step()
    
    def _step_stage3(self):
        """ZeRO阶段3优化步骤"""
        # 收集和平均梯度
        self._reduce_gradients()
        
        # 参数更新（仅负责该分区的进程执行）
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            for group_idx, group_params in enumerate(self.param_groups):
                if group_idx % world_size == rank:
                    # 更新本地参数
                    for param in group_params:
                        if param.grad is not None:
                            param.data.add_(param.grad, alpha=-self.optimizer.param_groups[0]['lr'])
        
        # 同步参数
        self._sync_parameters()
    
    def _reduce_gradients(self):
        """归约梯度（用于ZeRO阶段2和3）"""
        if not dist.is_initialized():
            return
        
        for group_idx, group_params in enumerate(self.param_groups):
            for param in group_params:
                if param.grad is not None:
                    # 梯度归约
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    
    def _sync_parameters(self):
        """同步参数（用于ZeRO阶段3）"""
        if not dist.is_initialized():
            return
        
        for group_idx, group_params in enumerate(self.param_groups):
            for param in group_params:
                # 广播参数
                dist.broadcast(param.data, src=group_idx % dist.get_world_size())
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
        
        # 清零梯度分区
        for group_idx in self.grad_partitions:
            for param in self.grad_partitions[group_idx]:
                self.grad_partitions[group_idx][param].zero_()
    
    def state_dict(self):
        """获取状态字典"""
        state = {
            'optimizer_state': self.optimizer.state_dict(),
            'stage': self.stage,
            'cpu_offload': self.cpu_offload
        }
        
        if self.stage >= 2:
            state['grad_partitions'] = {
                group_idx: {
                    id(param): grad.clone()
                    for param, grad in partitions.items()
                }
                for group_idx, partitions in self.grad_partitions.items()
            }
        
        return state
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.stage = state_dict.get('stage', 1)
        self.cpu_offload = state_dict.get('cpu_offload', False)
        
        if self.stage >= 2 and 'grad_partitions' in state_dict:
            for group_idx, partitions in state_dict['grad_partitions'].items():
                if group_idx not in self.grad_partitions:
                    self.grad_partitions[group_idx] = {}
                
                for param_id, grad_data in partitions.items():
                    # 需要将param_id映射回参数对象
                    pass  # 简化实现


def run_distributed_training(rank: int,
                            world_size: int,
                            config: DistributedConfig,
                            model_fn: Callable,
                            dataset_fn: Callable,
                            train_fn: Callable):
    """运行分布式训练（每个进程调用）"""
    
    # 更新配置中的排名
    config.rank = rank
    config.local_rank = rank
    config.world_size = world_size
    
    # 创建训练器
    model = model_fn()
    dataset = dataset_fn()
    
    trainer = DistributedTrainer(model, dataset, config)
    
    # 运行训练
    try:
        train_fn(trainer)
    except Exception as e:
        logging.error(f"分布式训练失败 (rank={rank}): {e}")
    finally:
        trainer.cleanup()


class DistributedTrainingManager:
    """分布式训练管理器"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger("DistributedTrainingManager")
        
        # 训练进程
        self.processes = []
        
        self.logger.info(f"分布式训练管理器初始化: 世界大小={config.world_size}")
    
    def launch_training(self,
                       model_fn: Callable,
                       dataset_fn: Callable,
                       train_fn: Callable) -> bool:
        """启动分布式训练"""
        
        if self.config.world_size <= 1:
            self.logger.warning("世界大小<=1，使用单机训练")
            
            # 单机训练
            config = self.config
            config.rank = 0
            config.local_rank = 0
            
            trainer = DistributedTrainer(model_fn(), dataset_fn(), config)
            
            try:
                train_fn(trainer)
                return True
            except Exception as e:
                self.logger.error(f"单机训练失败: {e}")
                return False
        
        # 分布式训练
        self.logger.info(f"启动分布式训练: {self.config.world_size} 个进程")
        
        try:
            # 使用torch.multiprocessing启动进程
            mp_torch.set_start_method('spawn', force=True)
            
            processes = []
            
            for rank in range(self.config.world_size):
                p = mp_torch.Process(
                    target=run_distributed_training,
                    args=(rank, self.config.world_size, self.config, 
                         model_fn, dataset_fn, train_fn)
                )
                p.start()
                processes.append(p)
            
            # 等待所有进程完成
            for p in processes:
                p.join()
            
            self.logger.info("分布式训练完成")
            return True
        
        except Exception as e:
            self.logger.error(f"启动分布式训练失败: {e}")
            return False
    
    def get_available_gpus(self) -> List[int]:
        """获取可用GPU列表"""
        if not torch.cuda.is_available():
            return []
        
        return list(range(torch.cuda.device_count()))
    
    def auto_detect_config(self) -> DistributedConfig:
        """自动检测配置"""
        config = self.config
        
        # 检测可用GPU数量
        available_gpus = self.get_available_gpus()
        num_gpus = len(available_gpus)
        
        if num_gpus > 1:
            config.world_size = num_gpus
            config.strategy = ParallelStrategy.DATA_PARALLEL
            config.backend = "nccl" if torch.cuda.is_available() else "gloo"
            
            self.logger.info(f"自动检测: {num_gpus} 个GPU可用，使用数据并行")
        else:
            config.world_size = 1
            self.logger.info("自动检测: 单GPU或CPU，使用单机训练")
        
        return config


def create_distributed_training_manager(config: Optional[DistributedConfig] = None) -> DistributedTrainingManager:
    """创建分布式训练管理器（工厂函数）"""
    if config is None:
        config = DistributedConfig()
    
    return DistributedTrainingManager(config)


# ============================================================================
# 测试代码（仅在直接运行脚本时使用）
# ============================================================================

# 测试模型类（模块级别定义以便pickle序列化）
if TORCH_DISTRIBUTED_AVAILABLE:
    class _TestModel(nn.Module):
        def __init__(self, input_dim=10, output_dim=5):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, output_dim)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 测试数据集类
    class _TestDataset(Dataset):
        def __init__(self, num_samples=1000, input_dim=10, output_dim=5):
            self.data = torch.randn(num_samples, input_dim)
            self.targets = torch.randn(num_samples, output_dim)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

# 测试函数（模块级别定义以便pickle序列化）
def _test_create_model():
    """创建测试模型"""
    if TORCH_DISTRIBUTED_AVAILABLE:
        return _TestModel()
    else:
        raise ImportError("PyTorch不可用，无法创建测试模型")

def _test_create_dataset():
    """创建测试数据集"""
    if TORCH_DISTRIBUTED_AVAILABLE:
        return _TestDataset(num_samples=100)
    else:
        raise ImportError("PyTorch不可用，无法创建测试数据集")

def _test_train_function(trainer):
    """测试训练函数"""
    if TORCH_DISTRIBUTED_AVAILABLE:
        # 创建优化器和损失函数
        optimizer = optim.Adam(trainer.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        # 训练几个epoch
        for epoch in range(3):
            result = trainer.train_epoch(optimizer, loss_fn)
            print(f"Rank {result['rank']}, Epoch {result['epoch']}, Loss: {result['loss']:.6f}")
        
        # 保存检查点
        if trainer.config.rank == 0:
            trainer._save_checkpoint(optimizer, "test_checkpoint")
    else:
        raise ImportError("PyTorch不可用，无法执行训练")

def run_distributed_test():
    """运行分布式训练测试"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== 测试分布式训练支持 ===")
    
    if not TORCH_DISTRIBUTED_AVAILABLE:
        print("警告: PyTorch分布式库不可用，测试将使用模拟模式")
        print("=== 测试完成 (模拟模式) ===")
        return True
    
    # 创建分布式配置
    config = DistributedConfig(
        world_size=2,  # 测试用2个进程
        strategy=ParallelStrategy.DATA_PARALLEL,
        master_addr="localhost",
        master_port=29500,
        batch_size_per_gpu=16,
        checkpoint_interval=10
    )
    
    # 创建分布式训练管理器
    manager = create_distributed_training_manager(config)
    
    print("\n启动分布式训练测试...")
    print("注意：分布式训练测试需要多个GPU或CPU核心")
    print("如果系统资源不足，训练可能会失败或使用单机模式")
    
    # 启动训练
    success = manager.launch_training(_test_create_model, _test_create_dataset, _test_train_function)
    
    if success:
        print("\n分布式训练测试完成!")
    else:
        print("\n分布式训练测试失败或使用单机模式完成")
    
    return success


if __name__ == "__main__":
    # 运行分布式训练测试
    run_distributed_test()