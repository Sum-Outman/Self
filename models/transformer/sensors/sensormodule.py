# SensorModule - 从self_agi_model.py拆分
"""Sensor模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging

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

        # 多传感器融合 - 计算平均值
        if len(encoded_features) > 1:
            # 计算所有传感器特征的平均值
            fused = torch.stack(encoded_features, dim=0).mean(dim=0)
        else:
            fused = encoded_features[0]

        # 通过融合层
        sensor_features = self.sensor_fusion(fused)

        # 计算置信度（基于传感器数量和类型）
        confidence = min(0.3 + 0.1 * len(sensor_types), 0.9)

        return {
            "sensor_features": sensor_features,
            "sensor_available": True,
            "sensor_types": sensor_types,
            "confidence": confidence,
            "num_sensors": len(sensor_types),
        }


# 味觉传感器模块

