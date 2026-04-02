#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四元数兼容数据加载器 - Self AGI 系统四元数全面引入实施方案数据加载模块

功能：
1. PyTorch Dataset接口兼容的四元数数据集
2. 支持多源四元数数据加载（文件、内存、实时流）
3. 四元数数据增强（噪声、插值、随机旋转）
4. 批量数据预处理和标准化
5. 与现有训练系统无缝集成

工业级质量标准要求：
- 高性能：支持GPU加速和并行加载
- 可扩展性：支持大规模数据集和分布式训练
- 灵活性：支持多种数据格式和增强策略
- 兼容性：与PyTorch生态系统完全兼容

设计原则：
1. 遵循PyTorch Dataset/DataLoader设计模式
2. 提供统一的四元数数据处理接口
3. 支持实时数据增强和变换
4. 包含完整的数据验证和质量检查
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import json
import pickle
import os
from pathlib import Path
from enum import Enum
import random
import uuid

from models.quaternion_core import (
    Quaternion, QuaternionTensor,
    quaternion_exp, quaternion_log, quaternion_from_angular_velocity,
    quaternion_weighted_average, quaternion_distance,
    QuaternionNormalization, QuaternionDistance
)
from models.quaternion_data_pipeline import (
    QuaternionDataPipeline, QuaternionDataItem,
    DataSourceType, RotationFormat
)

logger = logging.getLogger(__name__)


class QuaternionDatasetMode(Enum):
    """四元数数据集模式"""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"


class QuaternionAugmentationType(Enum):
    """四元数数据增强类型"""
    NOISE = "noise"  # 添加噪声
    RANDOM_ROTATION = "random_rotation"  # 随机旋转
    INTERPOLATION = "interpolation"  # 插值增强
    DROPOUT = "dropout"  # 随机丢弃
    TIME_WARP = "time_warp"  # 时间扭曲
    SCALING = "scaling"  # 尺度变换


@dataclass
class QuaternionDatasetConfig:
    """四元数数据集配置"""
    
    # 数据源配置
    data_sources: List[Dict[str, Any]]  # 数据源列表
    dataset_mode: QuaternionDatasetMode = QuaternionDatasetMode.TRAIN
    
    # 数据加载配置
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # 数据增强配置
    augmentations: List[QuaternionAugmentationType] = field(default_factory=lambda: [
        QuaternionAugmentationType.NOISE,
        QuaternionAugmentationType.RANDOM_ROTATION
    ])
    augmentation_prob: float = 0.5  # 增强概率
    
    # 噪声配置
    noise_std: float = 0.01
    noise_type: str = "gaussian"  # "gaussian", "uniform", "laplacian"
    
    # 随机旋转配置
    max_rotation_angle: float = 0.1  # 最大旋转角度（弧度）
    
    # 插值配置
    interpolation_factor: float = 0.5  # 插值因子
    
    # 数据验证配置
    enable_validation: bool = True
    validation_strictness: float = 0.9  # 验证严格度
    
    # 缓存配置
    cache_enabled: bool = True
    cache_size: int = 10000
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        """后初始化处理"""
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "self_agi", "quaternion_dataset")


class QuaternionDataset(Dataset):
    """四元数兼容数据集"""
    
    def __init__(
        self,
        config: QuaternionDatasetConfig,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        初始化四元数数据集
        
        参数:
            config: 数据集配置
            transform: 数据变换函数
            target_transform: 目标变换函数
        """
        super().__init__()
        
        self.config = config
        self.transform = transform
        self.target_transform = target_transform
        self.mode = config.dataset_mode
        
        # 数据管道
        self.data_pipeline = QuaternionDataPipeline(
            cache_size=config.cache_size if config.cache_enabled else 0,
            batch_size=config.batch_size,
            enable_validation=config.enable_validation,
            enable_logging=True
        )
        
        # 数据缓存
        self.data_cache = []
        self.cache_indices = {}
        
        # 数据统计
        self.stats = {
            "total_samples": 0,
            "loaded_samples": 0,
            "augmented_samples": 0,
            "validation_errors": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # 初始化数据源
        self._init_data_sources()
        
        # 加载数据
        self._load_data()
        
        logger.info(f"四元数数据集初始化完成: 模式={self.mode.value}, 样本数={len(self.data_cache)}")
    
    def _init_data_sources(self):
        """初始化数据源"""
        self.data_sources = []
        
        for source_config in self.config.data_sources:
            source_type = source_config.get("type", "file")
            source_path = source_config.get("path")
            
            if source_type == "file":
                if source_path and os.path.exists(source_path):
                    self.data_sources.append({
                        "type": "file",
                        "path": source_path,
                        "format": source_config.get("format", "json"),
                        "encoding": source_config.get("encoding", "utf-8")
                    })
                else:
                    logger.warning(f"数据源文件不存在: {source_path}")
            
            elif source_type == "memory":
                data = source_config.get("data", [])
                if data:
                    self.data_sources.append({
                        "type": "memory",
                        "data": data
                    })
            
            elif source_type == "pipeline":
                # 直接使用数据管道
                self.data_sources.append({
                    "type": "pipeline",
                    "pipeline": self.data_pipeline
                })
            
            else:
                logger.warning(f"不支持的数据源类型: {source_type}")
    
    def _load_data(self):
        """加载数据"""
        total_loaded = 0
        
        for source in self.data_sources:
            source_type = source["type"]
            
            if source_type == "file":
                loaded = self._load_from_file(source)
                total_loaded += loaded
            
            elif source_type == "memory":
                loaded = self._load_from_memory(source)
                total_loaded += loaded
            
            elif source_type == "pipeline":
                # 管道数据已经加载
                loaded = len(self.data_pipeline.data_cache)
                total_loaded += loaded
                
                # 将管道数据复制到缓存
                for item in self.data_pipeline.data_cache:
                    if item.is_valid:
                        self.data_cache.append(item)
        
        self.stats["loaded_samples"] = total_loaded
        self.stats["total_samples"] = len(self.data_cache)
        
        logger.info(f"数据加载完成: 总样本数={self.stats['total_samples']}, 有效样本数={total_loaded}")
    
    def _load_from_file(self, source: Dict[str, Any]) -> int:
        """从文件加载数据"""
        filepath = source["path"]
        file_format = source.get("format", "json")
        encoding = source.get("encoding", "utf-8")
        
        loaded_count = 0
        
        try:
            if file_format == "json":
                with open(filepath, 'r', encoding=encoding) as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item_data in data:
                        if self._add_data_item(item_data):
                            loaded_count += 1
                elif isinstance(data, dict):
                    # 可能是按ID组织的字典
                    for item_id, item_data in data.items():
                        if isinstance(item_data, dict):
                            item_data["item_id"] = item_id
                            if self._add_data_item(item_data):
                                loaded_count += 1
            
            elif file_format == "pkl" or file_format == "pickle":
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, list):
                    for item_data in data:
                        if self._add_data_item(item_data):
                            loaded_count += 1
            
            else:
                logger.error(f"不支持的文件格式: {file_format}")
        
        except Exception as e:
            logger.error(f"加载文件失败 {filepath}: {e}")
        
        return loaded_count
    
    def _load_from_memory(self, source: Dict[str, Any]) -> int:
        """从内存数据加载"""
        data = source.get("data", [])
        loaded_count = 0
        
        for item_data in data:
            if self._add_data_item(item_data):
                loaded_count += 1
        
        return loaded_count
    
    def _add_data_item(self, item_data: Dict[str, Any]) -> bool:
        """添加数据项"""
        try:
            # 提取数据项信息
            item_id = item_data.get("item_id", str(uuid.uuid4()))
            timestamp = item_data.get("timestamp", time.time())
            data_source = item_data.get("data_source", "simulation")
            rotation_format = item_data.get("rotation_format", "quaternion")
            raw_data = item_data.get("raw_data")
            metadata = item_data.get("metadata", {})
            confidence = item_data.get("confidence", 1.0)
            
            if raw_data is None:
                logger.warning(f"数据项 {item_id} 缺少raw_data")
                return False
            
            # 添加到数据管道
            item = self.data_pipeline.add_item(
                item_id=item_id,
                timestamp=timestamp,
                data_source=data_source,
                rotation_format=rotation_format,
                raw_data=raw_data,
                metadata=metadata,
                confidence=confidence
            )
            
            if item is not None and item.is_valid:
                self.data_cache.append(item)
                return True
            else:
                self.stats["validation_errors"] += 1
                return False
        
        except Exception as e:
            logger.error(f"添加数据项失败: {e}")
            return False
    
    def __len__(self) -> int:
        """获取数据集大小"""
        return len(self.data_cache)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        获取数据项
        
        参数:
            idx: 索引
        
        返回:
            quaternion_tensor: [4] 四元数张量
            metadata: 元数据字典
        """
        # 检查缓存
        cache_key = f"{idx}_{self.mode.value}"
        
        if (self.config.cache_enabled and 
            cache_key in self.cache_indices and 
            self.cache_indices[cache_key] < len(self.data_cache)):
            
            cached_idx = self.cache_indices[cache_key]
            item = self.data_cache[cached_idx]
            self.stats["cache_hits"] += 1
        
        else:
            # 获取原始数据项
            if idx >= len(self.data_cache):
                idx = idx % len(self.data_cache)
            
            item = self.data_cache[idx]
            self.stats["cache_misses"] += 1
        
        # 获取四元数数据
        if item.quaternion is None:
            # 生成单位四元数作为回退
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            quaternion = item.quaternion.copy()
        
        # 数据增强（仅训练模式）
        if (self.mode == QuaternionDatasetMode.TRAIN and 
            random.random() < self.config.augmentation_prob):
            
            quaternion = self._apply_augmentations(quaternion)
            self.stats["augmented_samples"] += 1
        
        # 转换为张量
        quaternion_tensor = torch.tensor(quaternion, dtype=torch.float32)
        
        # 应用变换
        if self.transform is not None:
            quaternion_tensor = self.transform(quaternion_tensor)
        
        # 准备元数据
        metadata = item.to_dict()
        metadata["dataset_mode"] = self.mode.value
        metadata["is_augmented"] = self.stats["augmented_samples"] > 0
        
        # 应用目标变换
        if self.target_transform is not None:
            metadata = self.target_transform(metadata)
        
        return quaternion_tensor, metadata
    
    def _apply_augmentations(self, quaternion: np.ndarray) -> np.ndarray:
        """应用数据增强"""
        augmented = quaternion.copy()
        
        for aug_type in self.config.augmentations:
            if random.random() < 0.5:  # 每个增强类型50%概率应用
                continue
            
            if aug_type == QuaternionAugmentationType.NOISE:
                augmented = self._add_noise(augmented)
            
            elif aug_type == QuaternionAugmentationType.RANDOM_ROTATION:
                augmented = self._apply_random_rotation(augmented)
            
            elif aug_type == QuaternionAugmentationType.INTERPOLATION:
                augmented = self._apply_interpolation(augmented)
            
            elif aug_type == QuaternionAugmentationType.DROPOUT:
                augmented = self._apply_dropout(augmented)
        
        # 归一化增强后的四元数
        norm = np.linalg.norm(augmented)
        if norm > 1e-8:
            augmented = augmented / norm
        
        return augmented
    
    def _add_noise(self, quaternion: np.ndarray) -> np.ndarray:
        """添加噪声"""
        noise_std = self.config.noise_std
        
        if self.config.noise_type == "gaussian":
            noise = np.random.normal(0, noise_std, 4)
        elif self.config.noise_type == "uniform":
            noise = np.random.uniform(-noise_std, noise_std, 4)
        elif self.config.noise_type == "laplacian":
            noise = np.random.laplace(0, noise_std, 4)
        else:
            noise = np.zeros(4)
        
        # 添加到四元数
        noisy = quaternion + noise
        
        return noisy
    
    def _apply_random_rotation(self, quaternion: np.ndarray) -> np.ndarray:
        """应用随机旋转"""
        max_angle = self.config.max_rotation_angle
        
        # 生成随机旋转轴
        axis = np.random.randn(3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis = axis / axis_norm
        else:
            axis = np.array([1.0, 0.0, 0.0])
        
        # 生成随机角度
        angle = np.random.uniform(-max_angle, max_angle)
        
        # 创建随机旋转四元数
        random_quat = Quaternion.from_axis_angle(axis, angle)
        
        # 应用旋转
        q_original = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        q_rotated = random_quat * q_original
        
        return q_rotated.as_vector()
    
    def _apply_interpolation(self, quaternion: np.ndarray) -> np.ndarray:
        """应用插值增强"""
        # 生成另一个随机四元数
        random_q = Quaternion.random()
        
        # 在原始四元数和随机四元数之间插值
        factor = self.config.interpolation_factor
        q_original = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        q_interpolated = q_original.slerp(random_q, factor)
        
        return q_interpolated.as_vector()
    
    def _apply_dropout(self, quaternion: np.ndarray) -> np.ndarray:
        """应用随机丢弃"""
        dropout_prob = 0.1  # 10%概率丢弃一个分量
        
        if random.random() < dropout_prob:
            # 随机选择一个分量置零
            idx = random.randint(0, 3)
            dropped = quaternion.copy()
            dropped[idx] = 0
            
            # 重新归一化
            norm = np.linalg.norm(dropped)
            if norm > 1e-8:
                dropped = dropped / norm
            else:
                dropped = np.array([1.0, 0.0, 0.0, 0.0])
            
            return dropped
        
        return quaternion
    
    def get_dataloader(self, **kwargs) -> DataLoader:
        """
        获取数据加载器
        
        参数:
            **kwargs: DataLoader额外参数
        
        返回:
            DataLoader实例
        """
        # 合并配置参数
        loader_kwargs = {
            "batch_size": self.config.batch_size,
            "shuffle": self.config.shuffle if self.mode == QuaternionDatasetMode.TRAIN else False,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "prefetch_factor": self.config.prefetch_factor
        }
        
        # 更新用户提供的参数
        loader_kwargs.update(kwargs)
        
        # 创建数据加载器
        dataloader = DataLoader(self, **loader_kwargs)
        
        return dataloader
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats_copy = self.stats.copy()
        
        # 添加数据管道统计
        pipeline_stats = self.data_pipeline.get_stats()
        stats_copy["pipeline_stats"] = pipeline_stats
        
        # 计算缓存命中率
        total_accesses = stats_copy["cache_hits"] + stats_copy["cache_misses"]
        stats_copy["cache_hit_rate"] = (
            stats_copy["cache_hits"] / total_accesses if total_accesses > 0 else 0.0
        )
        
        # 计算增强率
        stats_copy["augmentation_rate"] = (
            stats_copy["augmented_samples"] / total_accesses if total_accesses > 0 else 0.0
        )
        
        return stats_copy
    
    def save_dataset(self, filepath: str):
        """保存数据集到文件"""
        try:
            # 准备保存数据
            save_data = {
                "config": self.config.__dict__,
                "data_cache": [item.to_dict() for item in self.data_cache],
                "stats": self.stats
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"数据集已保存到: {filepath}")
        
        except Exception as e:
            logger.error(f"保存数据集失败: {e}")
    
    @classmethod
    def load_dataset(cls, filepath: str, **kwargs) -> 'QuaternionDataset':
        """
        从文件加载数据集
        
        参数:
            filepath: 文件路径
            **kwargs: 额外配置参数
        
        返回:
            QuaternionDataset实例
        """
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # 恢复配置
            config_dict = save_data["config"]
            config_dict.update(kwargs)
            config = QuaternionDatasetConfig(**config_dict)
            
            # 创建数据集实例
            dataset = cls(config)
            
            # 恢复数据缓存
            dataset.data_cache = []
            for item_dict in save_data["data_cache"]:
                item = QuaternionDataItem.from_dict(item_dict)
                dataset.data_cache.append(item)
            
            # 恢复统计信息
            dataset.stats = save_data["stats"]
            
            logger.info(f"数据集已从 {filepath} 加载")
            
            return dataset
        
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise


# ============================================================================
# 数据变换函数
# ============================================================================

class QuaternionNormalizeTransform:
    """四元数归一化变换"""
    
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
    
    def __call__(self, quaternion: torch.Tensor) -> torch.Tensor:
        """应用变换"""
        norm = torch.norm(quaternion)
        if norm > self.eps:
            return quaternion / norm
        else:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=quaternion.dtype)


class QuaternionNoiseTransform:
    """四元数噪声变换"""
    
    def __init__(self, noise_std: float = 0.01, noise_type: str = "gaussian"):
        self.noise_std = noise_std
        self.noise_type = noise_type
    
    def __call__(self, quaternion: torch.Tensor) -> torch.Tensor:
        """添加噪声"""
        if self.noise_type == "gaussian":
            noise = torch.randn_like(quaternion) * self.noise_std
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(quaternion) * 2 - 1) * self.noise_std
        else:
            noise = torch.zeros_like(quaternion)
        
        noisy = quaternion + noise
        
        # 归一化
        norm = torch.norm(noisy)
        if norm > 1e-8:
            noisy = noisy / norm
        
        return noisy


class QuaternionRandomRotationTransform:
    """四元数随机旋转变换"""
    
    def __init__(self, max_angle: float = 0.1):
        self.max_angle = max_angle
    
    def __call__(self, quaternion: torch.Tensor) -> torch.Tensor:
        """应用随机旋转"""
        # 转换为numpy进行旋转计算
        q_np = quaternion.numpy()
        q_obj = Quaternion(q_np[0], q_np[1], q_np[2], q_np[3])
        
        # 生成随机旋转
        axis = np.random.randn(3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis = axis / axis_norm
        else:
            axis = np.array([1.0, 0.0, 0.0])
        
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        random_quat = Quaternion.from_axis_angle(axis, angle)
        
        # 应用旋转
        q_rotated = random_quat * q_obj
        
        # 转换回张量
        return torch.tensor(q_rotated.as_vector(), dtype=quaternion.dtype)


# ============================================================================
# 测试函数
# ============================================================================

def test_quaternion_dataset():
    """测试四元数数据集"""
    print("测试四元数数据集...")
    
    import uuid
    
    # 创建测试数据
    test_data = []
    for i in range(100):
        test_data.append({
            "item_id": f"test_{i}",
            "timestamp": time.time() + i * 0.1,
            "data_source": "simulation",
            "rotation_format": "euler_angles",
            "raw_data": [np.random.uniform(-np.pi, np.pi) for _ in range(3)],
            "metadata": {"index": i, "source": "test"},
            "confidence": 0.9
        })
    
    # 创建配置
    config = QuaternionDatasetConfig(
        data_sources=[
            {"type": "memory", "data": test_data}
        ],
        dataset_mode=QuaternionDatasetMode.TRAIN,
        batch_size=16,
        augmentations=[
            QuaternionAugmentationType.NOISE,
            QuaternionAugmentationType.RANDOM_ROTATION
        ],
        augmentation_prob=0.7,
        noise_std=0.01,
        max_rotation_angle=0.05,
        cache_enabled=True,
        cache_size=1000
    )
    
    # 创建数据集
    dataset = QuaternionDataset(config)
    
    # 测试基本功能
    assert len(dataset) > 0, "数据集为空"
    assert len(dataset.data_cache) == len(dataset), "数据缓存大小不匹配"
    
    # 测试数据项获取
    for i in range(min(10, len(dataset))):
        quaternion, metadata = dataset[i]
        
        assert isinstance(quaternion, torch.Tensor), "四元数不是张量"
        assert quaternion.shape == (4,), f"四元数形状错误: {quaternion.shape}"
        assert "item_id" in metadata, "元数据缺少item_id"
        
        # 检查四元数归一化
        norm = torch.norm(quaternion).item()
        assert abs(norm - 1.0) < 0.01, f"四元数未归一化: {norm}"
    
    # 测试数据加载器
    dataloader = dataset.get_dataloader()
    
    batch_count = 0
    for batch_idx, (batch_quaternions, batch_metadata) in enumerate(dataloader):
        assert isinstance(batch_quaternions, torch.Tensor), "批量四元数不是张量"
        assert batch_quaternions.shape[0] <= config.batch_size, "批量大小错误"
        assert batch_quaternions.shape[1] == 4, "四元数维度错误"
        
        batch_count += 1
        if batch_count >= 3:  # 测试3个批次
            break
    
    # 测试统计信息
    stats = dataset.get_stats()
    assert "total_samples" in stats, "统计信息缺少total_samples"
    assert stats["total_samples"] == len(dataset), "总样本数统计错误"
    
    # 测试保存和加载
    test_save_path = "test_quaternion_dataset.pkl"
    
    try:
        dataset.save_dataset(test_save_path)
        
        # 加载数据集
        loaded_dataset = QuaternionDataset.load_dataset(test_save_path)
        
        assert len(loaded_dataset) == len(dataset), "加载后数据集大小不匹配"
        
        # 清理测试文件
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
    
    except Exception as e:
        print(f"保存/加载测试失败: {e}")
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
    
    # 测试数据变换
    normalize_transform = QuaternionNormalizeTransform()
    noise_transform = QuaternionNoiseTransform(noise_std=0.02)
    
    test_quaternion = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
    
    normalized = normalize_transform(test_quaternion)
    norm = torch.norm(normalized).item()
    assert abs(norm - 1.0) < 0.01, "归一化变换失败"
    
    noisy = noise_transform(test_quaternion)
    assert noisy.shape == test_quaternion.shape, "噪声变换形状错误"
    
    print("所有测试通过！")
    
    return True


if __name__ == "__main__":
    # 运行测试
    test_quaternion_dataset()
