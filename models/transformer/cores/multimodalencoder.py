# MultiModalEncoder - 从self_agi_model.py拆分
"""MultiModalEncoder模块"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union


class MultiModalEncoder(nn.Module):
    """多模态编码器 - 增强版，支持跨模态对齐和概念统一

    功能：
    - 多模态特征编码（文本、图像、音频、视频、传感器）
    - 跨模态注意力对齐
    - 概念统一表示学习
    - 自适应模态融合

    基于最新多模态学习研究，支持苹果例子中的多模态认知
    """

    def __init__(self, config: "AGIModelConfig"):
        super().__init__()
        self.config = config

        # 文本编码器
        self.text_encoder = nn.Linear(config.hidden_size, config.hidden_size)

        # 图像编码器
        if config.multimodal_enabled:
            self.image_encoder = nn.Linear(
                config.image_embedding_dim, config.hidden_size
            )
            self.audio_encoder = nn.Linear(
                config.audio_embedding_dim, config.hidden_size
            )
            self.video_encoder = nn.Linear(
                config.video_embedding_dim, config.hidden_size
            )
            self.sensor_encoder = nn.Linear(
                config.sensor_embedding_dim, config.hidden_size
            )

            # 跨模态注意力对齐网络
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads // 2,
                dropout=config.attention_probs_dropout_prob,
                batch_first=True,
            )

            # 概念对齐投影层 - 将不同模态映射到统一概念空间
            self.concept_alignment_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 2),
                nn.GELU(),
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            )

            # 模态自适应权重网络 - 学习每个模态的重要性权重
            self.modality_weight_net = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, 1),  # 输出模态权重
                nn.Sigmoid(),
            )

            # 概念统一融合层
            self.concept_fusion_layer = nn.Sequential(
                nn.Linear(config.hidden_size * 5, config.hidden_size * 2),
                nn.GELU(),
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            )

            # 模态不变性编码器 - 提取跨模态不变特征
            self.modality_invariant_encoder = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            )

            # 保持向后兼容性：原始融合层
            self.fusion_layer = nn.Linear(config.hidden_size * 5, config.hidden_size)

        # 模态编码器
        self.modality_embeddings = nn.Embedding(
            6, config.hidden_size
        )  # 文本, 图像, 音频, 视频, 传感器, 融合

        # 层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        text_embeddings: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        video_embeddings: Optional[torch.Tensor] = None,
        sensor_embeddings: Optional[torch.Tensor] = None,
        modality_types: Optional[List[int]] = None,
        return_alignment_info: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """前向传播 - 增强版，支持跨模态对齐

        参数:
            text_embeddings: [batch_size, seq_len, text_embedding_dim]
            image_embeddings: [batch_size, seq_len, image_embedding_dim]
            audio_embeddings: [batch_size, seq_len, audio_embedding_dim]
            video_embeddings: [batch_size, seq_len, video_embedding_dim]
            sensor_embeddings: [batch_size, seq_len, sensor_embedding_dim]
            modality_types: 模态类型列表
            return_alignment_info: 是否返回对齐信息

        返回:
            如果return_alignment_info为False: encoded_embeddings [batch_size, seq_len, hidden_size]
            如果return_alignment_info为True: 包含编码嵌入和对齐信息的字典
        """
        encoded_features = []
        modality_list = []
        alignment_info = {}

        # 文本编码
        if text_embeddings is not None:
            text_encoded = self.text_encoder(text_embeddings)
            encoded_features.append(text_encoded)
            modality_list.append("text")

        # 多模态编码
        if self.config.multimodal_enabled:
            # 图像编码
            if image_embeddings is not None:
                image_encoded = self.image_encoder(image_embeddings)
                encoded_features.append(image_encoded)
                modality_list.append("image")
                alignment_info["image_features"] = image_encoded

            # 音频编码
            if audio_embeddings is not None:
                audio_encoded = self.audio_encoder(audio_embeddings)
                encoded_features.append(audio_encoded)
                modality_list.append("audio")
                alignment_info["audio_features"] = audio_encoded

            # 视频编码
            if video_embeddings is not None:
                video_encoded = self.video_encoder(video_embeddings)
                encoded_features.append(video_encoded)
                modality_list.append("video")
                alignment_info["video_features"] = video_encoded

            # 传感器编码
            if sensor_embeddings is not None:
                sensor_encoded = self.sensor_encoder(sensor_embeddings)
                encoded_features.append(sensor_encoded)
                modality_list.append("sensor")
                alignment_info["sensor_features"] = sensor_encoded

        # 如果没有特征，返回空张量
        if not encoded_features:
            batch_size = text_embeddings.shape[0] if text_embeddings is not None else 1
            seq_len = text_embeddings.shape[1] if text_embeddings is not None else 1
            empty_result = torch.zeros(batch_size, seq_len, self.config.hidden_size).to(
                text_embeddings.device
                if text_embeddings is not None
                else torch.device("cpu")
            )

            if return_alignment_info:
                return {
                    "encoded_embeddings": empty_result,
                    "alignment_info": {},
                    "modality_weights": {},
                    "concept_features": empty_result,
                }
            else:
                return empty_result

        # 跨模态对齐（如果有多模态）
        if self.config.multimodal_enabled and len(encoded_features) > 1:
            # 1. 跨模态注意力对齐
            all_features = torch.cat(encoded_features, dim=1)  # 拼接所有模态特征

            # 创建注意力掩码（假设所有位置都有效）
            batch_size, seq_len, _ = all_features.shape
            # 创建key_padding_mask：False表示有效位置，True表示需要mask的位置
            key_padding_mask = torch.zeros(
                batch_size, seq_len, dtype=torch.bool, device=all_features.device
            )

            aligned_features, attention_weights = self.cross_modal_attention(
                all_features,
                all_features,
                all_features,
                key_padding_mask=key_padding_mask,
            )
            alignment_info["cross_modal_attention_weights"] = attention_weights

            # 2. 概念对齐投影
            concept_aligned_features = []
            for i, feature in enumerate(encoded_features):
                aligned = self.concept_alignment_projector(feature)
                concept_aligned_features.append(aligned)

            alignment_info["concept_aligned_features"] = concept_aligned_features

            # 3. 模态自适应权重学习
            modality_weights = []
            for feature in encoded_features:
                # 计算平均特征作为模态表示
                modality_rep = feature.mean(
                    dim=1, keepdim=True
                )  # [batch_size, 1, hidden_size]
                weight = self.modality_weight_net(modality_rep)  # [batch_size, 1, 1]
                modality_weights.append(weight.squeeze(-1).squeeze(-1))  # [batch_size]

            alignment_info["modality_weights"] = modality_weights

            # 4. 加权融合
            weighted_features = []
            for i, (feature, weight) in enumerate(
                zip(encoded_features, modality_weights)
            ):
                # 扩展权重以匹配特征维度
                weight_expanded = weight.view(-1, 1, 1).expand_as(feature)
                weighted_feature = feature * weight_expanded
                weighted_features.append(weighted_feature)

            # 5. 概念统一融合
            if len(weighted_features) > 1:
                # 拼接所有加权特征
                concatenated = torch.cat(weighted_features, dim=-1)
                fused_concept = self.concept_fusion_layer(concatenated)
                alignment_info["fused_concept_features"] = fused_concept

                # 6. 模态不变性编码
                modality_invariant = self.modality_invariant_encoder(fused_concept)
                alignment_info["modality_invariant_features"] = modality_invariant

                # 使用融合特征作为主要特征
                encoded_features = [fused_concept]
            else:
                # 单个模态，直接使用
                alignment_info["fused_concept_features"] = encoded_features[0]
                alignment_info["modality_invariant_features"] = encoded_features[0]
        else:
            # 单模态情况
            alignment_info["modality_weights"] = [1.0] * len(encoded_features)
            alignment_info["fused_concept_features"] = (
                encoded_features[0] if encoded_features else None
            )
            alignment_info["modality_invariant_features"] = (
                encoded_features[0] if encoded_features else None
            )

        # 加权平均（原始融合方式，保持向后兼容）
        if (
            self.config.multimodal_enabled
            and self.config.multimodal_fusion_enabled
            and len(encoded_features) > 1
        ):
            # 拼接所有特征
            fused = torch.cat(encoded_features, dim=-1)
            # 注意：这里需要检查fusion_layer是否存在
            if hasattr(self, "fusion_layer"):
                fused = self.fusion_layer(fused)
            encoded_features = [fused]

        # 加权平均（最终融合）
        combined = torch.stack(encoded_features, dim=0).mean(dim=0)

        # 添加模态嵌入
        if modality_types is not None:
            modality_embeds = self.modality_embeddings(
                torch.tensor(modality_types).to(combined.device)
            )
            # 获取batch_size和seq_len
            batch_size, seq_len, _ = combined.shape
            # 扩展模态嵌入以匹配batch维度
            if modality_embeds.dim() == 2:  # [seq_len, hidden_size]
                # 扩展到 [batch_size, seq_len, hidden_size]
                modality_embeds = modality_embeds.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
            combined = combined + modality_embeds

        # 层归一化和dropout
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)

        alignment_info["encoded_embeddings"] = combined
        alignment_info["modality_list"] = modality_list

        if return_alignment_info:
            return alignment_info
        else:
            return combined
