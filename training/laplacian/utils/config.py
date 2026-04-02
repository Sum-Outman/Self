#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉普拉斯增强系统配置管理

提供统一的配置类，用于管理拉普拉斯增强训练的各个方面
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LaplacianEnhancedTrainingConfig:
    """拉普拉斯增强训练配置
    
    从 training/laplacian_enhanced_training.py 迁移而来
    """
    
    # 基础配置
    enabled: bool = True  # 是否启用拉普拉斯增强
    training_mode: str = "pinn"  # "pinn", "cnn", "gnn", "multimodal"
    
    # 拉普拉斯正则化配置
    laplacian_reg_enabled: bool = True
    laplacian_reg_lambda: float = 0.01
    laplacian_normalization: str = "sym"  # "none", "sym", "rw"
    adaptive_lambda: bool = True  # 自适应正则化强度
    
    # 多尺度拉普拉斯配置
    multi_scale_enabled: bool = True
    num_scales: int = 3  # 多尺度数量
    scale_factors: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    
    # 图结构配置
    graph_construction_method: str = "knn"  # "knn", "radius", "precomputed"
    k_neighbors: int = 10  # k近邻数
    radius: float = 0.1  # 半径阈值
    
    # 性能配置
    use_sparse: bool = True  # 使用稀疏矩阵
    cache_enabled: bool = True  # 启用缓存
    max_cache_size: int = 100
    
    # 优化配置
    optimizer_integration: bool = True  # 优化器集成
    gradient_clipping: bool = True  # 梯度裁剪
    clip_value: float = 1.0
    
    # 监控配置
    logging_frequency: int = 100  # 日志频率
    metrics_tracking: bool = True  # 指标跟踪
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "enabled": self.enabled,
            "training_mode": self.training_mode,
            "laplacian_reg_enabled": self.laplacian_reg_enabled,
            "laplacian_reg_lambda": self.laplacian_reg_lambda,
            "laplacian_normalization": self.laplacian_normalization,
            "adaptive_lambda": self.adaptive_lambda,
            "multi_scale_enabled": self.multi_scale_enabled,
            "num_scales": self.num_scales,
            "scale_factors": self.scale_factors,
            "graph_construction_method": self.graph_construction_method,
            "k_neighbors": self.k_neighbors,
            "radius": self.radius,
            "use_sparse": self.use_sparse,
            "cache_enabled": self.cache_enabled,
            "max_cache_size": self.max_cache_size,
            "optimizer_integration": self.optimizer_integration,
            "gradient_clipping": self.gradient_clipping,
            "clip_value": self.clip_value,
            "logging_frequency": self.logging_frequency,
            "metrics_tracking": self.metrics_tracking,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LaplacianEnhancedTrainingConfig":
        """从字典创建配置"""
        # 过滤掉不在类字段中的键
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)


@dataclass
class UnifiedLaplacianConfig:
    """统一拉普拉斯配置
    
    集成所有拉普拉斯相关模块的配置，提供统一的配置接口
    """
    
    # 基础配置
    enabled: bool = True
    config_name: str = "default"
    
    # 拉普拉斯正则化配置
    regularization: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "lambda_reg": 0.01,
        "normalization": "sym",
        "adaptive_enabled": True,
        "min_lambda": 1e-6,
        "max_lambda": 1.0,
        "adaptation_rate": 0.01,
    })
    
    # 训练增强配置
    training_enhancement: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "training_mode": "pinn",  # "pinn", "cnn", "gnn", "multimodal"
        "multi_scale_enabled": True,
        "num_scales": 3,
        "scale_factors": [1.0, 0.5, 0.25],
        "graph_construction_method": "knn",
        "k_neighbors": 10,
    })
    
    # 优化器配置
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "gradient_clipping": True,
        "clip_value": 1.0,
        "laplacian_gradient_weight": 0.1,
    })
    
    # 性能配置
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "use_sparse": True,
        "cache_enabled": True,
        "max_cache_size": 100,
        "dtype": "float64",  # "float32", "float64"
        "device": "auto",  # "cpu", "cuda", "auto"
    })
    
    # 监控配置
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        "logging_enabled": True,
        "logging_frequency": 100,
        "metrics_tracking": True,
        "statistics_interval": 10,
    })
    
    def update(self, updates: Dict[str, Any]) -> None:
        """更新配置
        
        参数:
            updates: 配置更新字典，支持嵌套路径如 "regularization.lambda_reg"
        """
        for key, value in updates.items():
            if '.' in key:
                # 嵌套路径
                parts = key.split('.')
                current = self
                for part in parts[:-1]:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    elif isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        raise AttributeError(f"配置路径不存在: {key}")
                
                if isinstance(current, dict) and parts[-1] in current:
                    current[parts[-1]] = value
                else:
                    raise AttributeError(f"配置路径不存在: {key}")
            else:
                # 直接属性
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise AttributeError(f"配置属性不存在: {key}")
    
    def get_nested(self, key: str, default: Any = None) -> Any:
        """获取嵌套配置值
        
        参数:
            key: 配置键，支持嵌套路径如 "regularization.lambda_reg"
            default: 默认值
            
        返回:
            配置值
        """
        if '.' not in key:
            return getattr(self, key, default)
        
        parts = key.split('.')
        current = self
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证正则化配置
            reg_config = self.regularization
            if reg_config["enabled"]:
                lambda_reg = reg_config["lambda_reg"]
                if not (0 <= lambda_reg <= 1):
                    logger.warning(f"正则化强度lambda_reg超出建议范围: {lambda_reg}")
            
            # 验证训练增强配置
            train_config = self.training_enhancement
            if train_config["enabled"]:
                num_scales = train_config["num_scales"]
                if num_scales < 1 or num_scales > 10:
                    logger.warning(f"多尺度数量超出建议范围: {num_scales}")
            
            # 验证性能配置
            perf_config = self.performance
            if perf_config["dtype"] not in ["float32", "float64"]:
                logger.warning(f"不支持的数据类型: {perf_config['dtype']}")
            
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False