# TasteSensorModule - 从self_agi_model.py拆分
"""TasteSensor模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

class TasteSensorModule(nn.Module):
    """味觉传感器模块 - 专门处理味觉传感器数据

    功能：
    - 味觉传感器数据采集和处理
    - 味觉特征提取（酸、甜、苦、咸、鲜等）
    - 味觉模式识别和分类
    - 多模态味觉-视觉融合

    基于真实味觉传感器原理实现，支持复杂味觉感知
    """

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 味觉特征编码器 - 处理味觉传感器原始数据
        self.taste_encoder = nn.Sequential(
            nn.Linear(config.sensor_embedding_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
        )

        # 味觉分类器 - 识别基本味觉类别
        self.taste_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 5),  # 5种基本味觉：酸、甜、苦、咸、鲜
            nn.Softmax(dim=-1),
        )

        # 味觉强度回归器 - 估计每种味觉的强度
        self.taste_intensity_regressor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 5),  # 5种味觉的强度
            nn.Sigmoid(),  # 归一化到0-1范围
        )

        # 味觉质量评估网络 - 评估味觉质量（好/坏）
        self.taste_quality_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 2),  # 好/坏
            nn.Softmax(dim=-1),
        )

        # 味觉-视觉融合网络 - 整合味觉和视觉信息
        self.taste_vision_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),  # 味觉+视觉特征
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 味觉记忆网络 - 学习和记忆味觉模式
        self.taste_memory_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        taste_sensor_data: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """前向传播

        参数:
            taste_sensor_data: 味觉传感器数据 [batch_size, sensor_dim] 或 [batch_size, seq_len, sensor_dim]
            visual_features: 视觉特征 [batch_size, hidden_dim] (可选，用于味觉-视觉融合)

        返回:
            包含味觉特征、分类、强度、质量等的字典
        """
        results = {
            "taste_features": None,
            "taste_classification": None,
            "taste_intensity": None,
            "taste_quality": None,
            "taste_vision_fused": None,
            "taste_memory_state": None,
            "taste_available": False,
        }

        if taste_sensor_data is None:
            return results

        results["taste_available"] = True

        # 处理输入维度
        if taste_sensor_data.dim() == 2:
            # [batch_size, sensor_dim] -> 添加序列维度
            taste_sensor_data = taste_sensor_data.unsqueeze(1)

        batch_size, seq_len, sensor_dim = taste_sensor_data.shape

        # 1. 味觉特征编码
        taste_encoded = self.taste_encoder(
            taste_sensor_data
        )  # [batch_size, seq_len, hidden_dim]
        taste_features = taste_encoded.mean(dim=1)  # [batch_size, hidden_dim]
        results["taste_features"] = taste_features

        # 2. 味觉分类
        taste_class_probs = self.taste_classifier(taste_features)  # [batch_size, 5]
        results["taste_classification"] = taste_class_probs

        # 3. 味觉强度估计
        taste_intensity = self.taste_intensity_regressor(
            taste_features
        )  # [batch_size, 5]
        results["taste_intensity"] = taste_intensity

        # 4. 味觉质量评估
        taste_quality = self.taste_quality_net(taste_features)  # [batch_size, 2]
        results["taste_quality"] = taste_quality

        # 5. 味觉记忆更新
        taste_memory_output, taste_memory_state = self.taste_memory_gru(taste_encoded)
        results["taste_memory_state"] = taste_memory_state

        # 6. 味觉-视觉融合（如果提供视觉特征）
        if visual_features is not None:
            # 视觉特征形状: [batch_size, hidden_dim]
            taste_vision_combined = torch.cat([taste_features, visual_features], dim=-1)
            taste_vision_fused = self.taste_vision_fusion(taste_vision_combined)
            results["taste_vision_fused"] = taste_vision_fused

        return results


# 数量认知模块

