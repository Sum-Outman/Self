#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比学习跨模态对齐模型模块

包含：
1. ContrastiveAlignmentModel - 对比学习跨模态对齐模型

工业级AGI系统要求：从零开始训练，不使用预训练模型依赖
"""

from .sensor_encoder import IndustrialSensorEncoder
from .industrial_audio_encoder import IndustrialAudioEncoder
from .vision_encoder import IndustrialVisionEncoder
from .text_encoder import IndustrialTextEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# 导入其他模块的类


class ContrastiveAlignmentModel(nn.Module):
    """对比学习跨模态对齐模型 - 基于CLIP和ALIGN架构

    参考论文:
    - "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, Radford et al., 2021)
    - "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision" (ALIGN, Jia et al., 2021)

    关键特性:
    1. 对比损失: InfoNCE损失函数，对齐不同模态表示
    2. 共享嵌入空间: 所有模态映射到同一空间
    3. 温度参数: 可学习的对比学习温度
    4. 硬负样本挖掘: 困难负样本增强训练
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # 模态编码器 - 从配置中提取参数
        text_embedding_dim = config.get("text_embedding_dim", 768)
        vocab_size = config.get("vocab_size", 100000)
        num_layers = config.get("num_layers", 12)
        max_position_embeddings = config.get("max_position_embeddings", 2048)
        self.text_encoder = IndustrialTextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            num_layers=num_layers,
            max_position_embeddings=max_position_embeddings,
        )
        self.text_embedding_dim = text_embedding_dim

        image_embedding_dim = config.get("image_embedding_dim", 768)
        image_size = config.get("image_size", 224)
        patch_size = config.get("patch_size", 16)
        self.image_encoder = IndustrialVisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=image_embedding_dim,
            num_layers=num_layers,
        )
        self.image_embedding_dim = image_embedding_dim

        self.audio_encoder = IndustrialAudioEncoder(config, return_pooled_output=True)
        self.audio_embedding_dim = config.get("audio_embedding_dim", 256)

        # 传感器编码器
        sensor_embedding_dim = config.get("sensor_embedding_dim", 256)
        config.get("sensor_sequence_length", 100)
        config.get("sensor_patch_size", 10)
        config.get("sensor_num_channels", 9)
        self.sensor_encoder = IndustrialSensorEncoder(config)
        self.sensor_embedding_dim = sensor_embedding_dim

        # 投影网络 (映射到共享空间)
        shared_dim = config.get("shared_dim", 512)
        self.text_projection = nn.Sequential(
            nn.Linear(config.get("text_embedding_dim", 768), shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )

        self.image_projection = nn.Sequential(
            nn.Linear(config.get("image_embedding_dim", 768), shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(config.get("audio_embedding_dim", 256), shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )

        self.sensor_projection = nn.Sequential(
            nn.Linear(config.get("sensor_embedding_dim", 256), shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )

        # 温度参数 (对比学习)
        self.temperature = nn.Parameter(
            torch.ones([]) * config.get("initial_temperature", 0.07)
        )

        # 负样本队列 (用于动量对比)
        self.queue_size = config.get("queue_size", 65536)
        self.momentum = config.get("momentum", 0.999)

        logger.info(
            f"初始化ContrastiveAlignmentModel: 共享维度={shared_dim}, 温度={                 self.temperature.item()}"
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """前向传播 - 计算对比损失和特征"""
        # 处理文本输入 - 检查是否是特征张量（2D）还是原始输入（3D或token IDs）
        text_input = batch.get("text_input", None)
        if text_input is not None:
            if text_input.dim() == 2 and text_input.size(1) == self.text_embedding_dim:
                # 输入已经是特征张量，跳过编码器
                text_features = text_input
            else:
                # 输入是原始数据，通过编码器处理
                text_features = self.text_encoder(text_input)
        else:
            text_features = None

        # 处理图像输入
        image_input = batch.get("image_input", None)
        if image_input is not None:
            if (
                image_input.dim() == 2
                and image_input.size(1) == self.image_embedding_dim
            ):
                # 输入已经是特征张量，跳过编码器
                image_features = image_input
            else:
                # 输入是原始数据，通过编码器处理
                image_features = self.image_encoder(image_input)
        else:
            image_features = None

        # 处理音频输入
        audio_input = batch.get("audio_input", None)
        if audio_input is not None:
            if (
                audio_input.dim() == 2
                and audio_input.size(1) == self.audio_embedding_dim
            ):
                # 输入已经是特征张量，跳过编码器
                audio_features = audio_input
            else:
                # 输入是原始数据，通过编码器处理
                audio_features = self.audio_encoder(audio_input)
        else:
            audio_features = None

        # 处理传感器输入
        sensor_input = batch.get("sensor_input", None)
        if sensor_input is not None:
            if (
                sensor_input.dim() == 2
                and sensor_input.size(1) == self.sensor_embedding_dim
            ):
                # 输入已经是特征张量，跳过编码器
                sensor_features = sensor_input
            else:
                # 输入是原始数据，通过编码器处理
                sensor_features = self.sensor_encoder(sensor_input)
        else:
            sensor_features = None

        # 投影到共享空间
        text_projected = (
            self.text_projection(text_features) if text_features is not None else None
        )
        image_projected = (
            self.image_projection(image_features)
            if image_features is not None
            else None
        )
        audio_projected = (
            self.audio_projection(audio_features)
            if audio_features is not None
            else None
        )
        sensor_projected = (
            self.sensor_projection(sensor_features)
            if sensor_features is not None
            else None
        )

        # 归一化 (对比学习关键步骤)
        if text_projected is not None:
            text_projected = F.normalize(text_projected, dim=-1)
        if image_projected is not None:
            image_projected = F.normalize(image_projected, dim=-1)
        if audio_projected is not None:
            audio_projected = F.normalize(audio_projected, dim=-1)
        if sensor_projected is not None:
            sensor_projected = F.normalize(sensor_projected, dim=-1)

        # 计算对比损失
        losses = {}

        # 文本-图像对比损失
        if text_projected is not None and image_projected is not None:
            losses["text_image_loss"] = self._compute_contrastive_loss(
                text_projected, image_projected, "text-image"
            )

        # 文本-音频对比损失
        if text_projected is not None and audio_projected is not None:
            losses["text_audio_loss"] = self._compute_contrastive_loss(
                text_projected, audio_projected, "text-audio"
            )

        # 图像-音频对比损失
        if image_projected is not None and audio_projected is not None:
            losses["image_audio_loss"] = self._compute_contrastive_loss(
                image_projected, audio_projected, "image-audio"
            )

        # 传感器相关对比损失
        if sensor_projected is not None:
            # 文本-传感器对比损失
            if text_projected is not None:
                losses["text_sensor_loss"] = self._compute_contrastive_loss(
                    text_projected, sensor_projected, "text-sensor"
                )

            # 图像-传感器对比损失
            if image_projected is not None:
                losses["image_sensor_loss"] = self._compute_contrastive_loss(
                    image_projected, sensor_projected, "image-sensor"
                )

            # 音频-传感器对比损失
            if audio_projected is not None:
                losses["audio_sensor_loss"] = self._compute_contrastive_loss(
                    audio_projected, sensor_projected, "audio-sensor"
                )

        return {
            "text_features": text_projected,
            "image_features": image_projected,
            "audio_features": audio_projected,
            "sensor_features": sensor_projected,
            "losses": losses,
            "temperature": self.temperature,
        }

    def _compute_contrastive_loss(
        self, features_a: torch.Tensor, features_b: torch.Tensor, modality_pair: str
    ) -> torch.Tensor:
        """计算对比损失 (InfoNCE)"""
        batch_size = features_a.size(0)

        # 相似度矩阵
        logits = torch.matmul(features_a, features_b.t()) / self.temperature

        # 标签：对角线为正样本
        labels = torch.arange(batch_size, device=features_a.device)

        # 对称损失
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)

        loss = (loss_a + loss_b) / 2

        # 记录统计信息
        if self.training:
            with torch.no_grad():
                accuracy = (logits.argmax(dim=1) == labels).float().mean()
                logger.debug(
                    f"对比损失 [{modality_pair}]: loss={                         loss.item():.4f}, 准确率={                         accuracy.item():.4f}"
                )

        return loss

    def compute_alignment_accuracy(
        self, features_a: torch.Tensor, features_b: torch.Tensor
    ) -> Dict[str, float]:
        """计算对齐准确率"""
        batch_size = features_a.size(0)

        # 相似度矩阵
        similarity = torch.matmul(features_a, features_b.t())

        # 预测和标签
        predictions = similarity.argmax(dim=1)
        labels = torch.arange(batch_size, device=features_a.device)

        # 计算准确率
        accuracy = (predictions == labels).float().mean().item()

        # 检索指标
        recall_at_1 = accuracy
        recall_at_5 = (
            (similarity.topk(5, dim=1).indices == labels.unsqueeze(1))
            .any(dim=1)
            .float()
            .mean()
            .item()
        )

        return {
            "accuracy": accuracy,
            "recall@1": recall_at_1,
            "recall@5": recall_at_5,
            "similarity_mean": similarity.diag().mean().item(),
        }
