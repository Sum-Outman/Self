# VisionModule - 从self_agi_model.py拆分
"""Vision模块"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class VisionModule(nn.Module):
    """视觉模块 - 处理图像识别和生成，支持红外线图像识别和温度识别"""

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 视觉编码器（图像到特征）
        self.vision_encoder = nn.Sequential(
            nn.Linear(config.image_embedding_dim, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 红外线图像编码器（可选，与视觉编码器共享权重）
        self.infrared_encoder = self.vision_encoder

        # 温度回归头（从红外图像特征回归温度值）
        self.temperature_regressor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),  # 输出单个温度值
        )

        # 红外线图像检测器（判断是否为红外图像）
        self.infrared_detector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 2),  # 二分类：红外或非红外
            nn.Softmax(dim=-1),
        )

        # 视觉解码器（特征到图像）
        self.vision_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.image_embedding_dim),
            nn.LayerNorm(config.image_embedding_dim, eps=1e-12),
        )

        # 视觉注意力
        self.vision_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads // 2,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_inputs: Optional[torch.Tensor] = None,
        is_infrared: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """前向传播"""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 图像识别（图像到特征）
        image_features = None
        temperature = None
        infrared_prob = None

        if image_inputs is not None:
            encoded_image = self.vision_encoder(image_inputs)
            image_features, _ = self.vision_attention(
                encoded_image, encoded_image, encoded_image
            )
            image_features = self.dropout(image_features)

            # 红外线图像检测
            infrared_prob = self.infrared_detector(image_features.mean(dim=1))

            # 如果提供is_infrared标签或检测为红外图像，则计算温度
            if is_infrared is not None:
                infrared_flag = is_infrared
            else:
                # 使用检测器的输出概率（类别1为红外）
                infrared_flag = infrared_prob[:, 1] > 0.5

            # 如果是红外图像，计算温度
            if infrared_flag.any():
                # 使用红外图像编码器（与视觉编码器相同）
                infrared_features = self.infrared_encoder(image_inputs)
                temperature = self.temperature_regressor(infrared_features.mean(dim=1))

        # 图像生成（特征到图像）
        features_to_image = self.vision_decoder(hidden_states)

        return {
            "image_features": image_features,
            "features_to_image": features_to_image,
            "image_embeddings": image_inputs,
            "temperature": temperature,
            "infrared_probability": infrared_prob,
            "is_infrared": infrared_flag if image_inputs is not None else None,
        }
