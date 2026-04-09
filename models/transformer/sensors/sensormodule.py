# SensorModule - 从self_agi_model.py拆分
"""Sensor模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional

# 尝试导入传感器接口
try:
    from models.system_control.sensor_interface import SensorInterface

    SENSOR_INTERFACE_AVAILABLE = True
except ImportError:
    SENSOR_INTERFACE_AVAILABLE = False
    SensorInterface = None

# 导入配置
try:
    from models.transformer.config import AGIModelConfig
except ImportError:
    # 定义备用配置类
    class AGIModelConfig:
        def __init__(self):
            self.hidden_size = 768
            self.sensor_embedding_dim = 128
            self.layer_norm_eps = 1e-12
            self.hidden_dropout_prob = 0.1
            self.sensor_integration_enabled = False


class SensorModule(nn.Module):
    """传感器模块 - 处理传感器数据接入和融合

    功能：
    - 多传感器数据采集和管理
    - 传感器数据预处理和滤波
    - 多传感器数据融合
    - 传感器状态监控和校准
    """

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 传感器编码器 - 将传感器数据编码为模型可处理的特征
        self.sensor_encoder = nn.Sequential(
            nn.Linear(config.sensor_embedding_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
        )

        # 多传感器融合层 - 现在处理可变数量的传感器
        self.sensor_fusion = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),  # 输入维度为hidden_size
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 传感器接口（如果可用）
        self.sensor_interface = None
        if SENSOR_INTERFACE_AVAILABLE and config.sensor_integration_enabled:
            try:
                self.sensor_interface = SensorInterface()
                logger.info("传感器接口初始化成功")
            except Exception as e:
                logger.warning(f"传感器接口初始化失败: {e}")

    def forward(
        self, sensor_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """前向传播

        参数:
            sensor_data: 传感器数据字典，键为传感器类型，值为传感器数据张量

        返回:
            处理后的传感器特征和元数据
        """
        # 处理输入类型：如果sensor_data是张量，转换为字典
        if torch.is_tensor(sensor_data):
            # 将张量转换为字典，使用默认传感器类型
            sensor_data = {"default_sensor": sensor_data}

        if sensor_data is None or not sensor_data:
            # 返回空特征
            return {
                "sensor_features": None,
                "sensor_available": False,
                "sensor_types": [],
                "confidence": 0.0,
                "num_sensors": 0,
            }

        encoded_features = []
        sensor_types = []

        # 编码每个传感器数据
        for sensor_type, data in sensor_data.items():
            if data is not None:
                # 确保数据维度正确 [batch_size, seq_len, sensor_dim] 或 [batch_size, sensor_dim]
                if data.dim() == 2:
                    data = data.unsqueeze(1)  # 添加序列维度

                # 编码传感器数据
                encoded = self.sensor_encoder(data)
                encoded_features.append(encoded)
                sensor_types.append(sensor_type)

        if not encoded_features:
            return {
                "sensor_features": None,
                "sensor_available": False,
                "sensor_types": [],
                "confidence": 0.0,
                "num_sensors": 0,
            }

        # 多传感器融合 - 使用高级融合算法
        if len(encoded_features) > 1:
            # 使用自适应加权融合而非简单平均
            fused = self._adaptive_sensor_fusion(encoded_features, sensor_types)
        else:
            fused = encoded_features[0]

        # 通过融合层
        sensor_features = self.sensor_fusion(fused)

        # 计算置信度（基于传感器质量、数量和融合效果）
        confidence = self._calculate_fusion_confidence(encoded_features, sensor_types)

        return {
            "sensor_features": sensor_features,
            "sensor_available": True,
            "sensor_types": sensor_types,
            "confidence": confidence,
            "num_sensors": len(sensor_types),
        }

    def _adaptive_sensor_fusion(
        self, encoded_features: List[torch.Tensor], sensor_types: List[str]
    ) -> torch.Tensor:
        """自适应传感器融合算法

        实现多种融合策略：
        1. 基于特征质量的加权融合
        2. 基于传感器类型的优先级融合
        3. 基于特征一致性的可靠性融合

        参数:
            encoded_features: 编码后的传感器特征列表
            sensor_types: 传感器类型列表

        返回:
            融合后的特征张量
        """
        if len(encoded_features) == 1:
            return encoded_features[0]

        # 计算每个特征的权重
        weights = []

        for i, feature in enumerate(encoded_features):
            # 1. 基于特征幅度的权重（特征值越大，通常信息量越大）
            feature_magnitude = torch.norm(feature).item()

            # 2. 基于特征方差的权重（方差越大，可能信息越丰富）
            feature_variance = torch.var(feature).item()

            # 3. 基于传感器类型的权重（某些传感器类型更可靠）
            sensor_type = sensor_types[i] if i < len(sensor_types) else "unknown"
            type_weight = self._get_sensor_type_weight(sensor_type)

            # 综合权重
            weight = (
                feature_magnitude * 0.3 + feature_variance * 0.3 + type_weight * 0.4
            )

            # 避免零权重
            weight = max(weight, 0.1)
            weights.append(weight)

        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # 如果所有权重都为零，使用均匀权重
            weights = [1.0 / len(weights)] * len(weights)

        # 加权融合
        fused = torch.zeros_like(encoded_features[0])
        for feature, weight in zip(encoded_features, weights):
            fused += feature * weight

        return fused

    def _get_sensor_type_weight(self, sensor_type: str) -> float:
        """获取传感器类型权重

        不同类型传感器具有不同的可靠性和重要性
        """
        type_weights = {
            # 高可靠性传感器
            "imu": 0.9,
            "accelerometer": 0.9,
            "gyroscope": 0.9,
            "magnetometer": 0.8,
            "gps": 0.8,
            # 中可靠性传感器
            "camera": 0.7,
            "lidar": 0.7,
            "radar": 0.7,
            "force": 0.6,
            "torque": 0.6,
            # 环境传感器
            "temperature": 0.5,
            "humidity": 0.5,
            "pressure": 0.5,
            "light": 0.4,
            # 其他传感器
            "proximity": 0.3,
            "ultrasonic": 0.3,
            "infrared": 0.3,
            "touch": 0.2,
            # 默认权重
            "unknown": 0.5,
            "default_sensor": 0.5,
        }

        return type_weights.get(sensor_type.lower(), 0.5)

    def _calculate_fusion_confidence(
        self, encoded_features: List[torch.Tensor], sensor_types: List[str]
    ) -> float:
        """计算融合置信度

        基于：
        1. 传感器数量
        2. 传感器类型多样性
        3. 特征一致性
        4. 特征质量
        """
        num_sensors = len(encoded_features)

        if num_sensors == 0:
            return 0.0

        # 1. 基于传感器数量的基础置信度
        base_confidence = min(0.3 + 0.1 * num_sensors, 0.9)

        # 2. 基于传感器类型多样性的加成
        unique_types = len(set(sensor_types))
        diversity_bonus = min(0.2, 0.05 * unique_types)

        # 3. 基于特征一致性的加成
        if num_sensors > 1:
            consistency_score = self._calculate_feature_consistency(encoded_features)
            consistency_bonus = consistency_score * 0.2
        else:
            consistency_bonus = 0.0

        # 4. 基于特征质量的加成
        quality_score = self._calculate_feature_quality(encoded_features)
        quality_bonus = quality_score * 0.2

        # 综合置信度
        confidence = (
            base_confidence + diversity_bonus + consistency_bonus + quality_bonus
        )

        # 限制在合理范围内
        return max(0.0, min(1.0, confidence))

    def _calculate_feature_consistency(self, features: List[torch.Tensor]) -> float:
        """计算特征一致性

        多个传感器特征之间的一致性越高，融合效果越好
        """
        if len(features) < 2:
            return 0.0

        try:
            # 计算所有特征对之间的余弦相似度
            similarities = []

            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    # 展平特征以计算相似度
                    feat_i = features[i].flatten()
                    feat_j = features[j].flatten()

                    # 计算余弦相似度
                    cos_sim = F.cosine_similarity(
                        feat_i.unsqueeze(0), feat_j.unsqueeze(0)
                    )
                    similarities.append(cos_sim.item())

            if similarities:
                # 平均相似度作为一致性分数
                return float(np.mean(similarities))
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"特征一致性计算失败: {e}")
            return 0.0

    def _calculate_feature_quality(self, features: List[torch.Tensor]) -> float:
        """计算特征质量

        基于特征的幅度、方差和稀疏性评估质量
        """
        quality_scores = []

        for feature in features:
            try:
                # 1. 特征幅度（非零）
                magnitude = torch.norm(feature).item()

                # 2. 特征方差（信息丰富度）
                variance = torch.var(feature).item()

                # 3. 特征稀疏性（非零元素比例）
                sparsity = (feature != 0).float().mean().item()

                # 综合质量分数
                # 幅度越大越好，但需要避免过大异常值
                mag_score = min(1.0, magnitude / 10.0)

                # 适中的方差最好，太小表示信息少，太大可能是噪声
                var_score = min(1.0, abs(np.log10(variance + 1e-10)))

                # 适中的稀疏性，完全密集或完全稀疏都不好
                sparse_score = 1.0 - abs(sparsity - 0.5) * 2.0

                # 综合分数
                quality = mag_score * 0.4 + var_score * 0.4 + sparse_score * 0.2
                quality_scores.append(quality)

            except Exception as e:
                logger.warning(f"单个特征质量计算失败: {e}")
                quality_scores.append(0.5)

        if quality_scores:
            return float(np.mean(quality_scores))
        else:
            return 0.5


# 味觉传感器模块
