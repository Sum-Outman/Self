# SpeechModule - 从self_agi_model.py拆分
"""Speech模块"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class SpeechModule(nn.Module):
    """语音模块 - 处理语音识别和合成"""

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 语音编码器（音频到文本）
        self.speech_encoder = nn.Sequential(
            nn.Linear(config.audio_embedding_dim, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 语音解码器（文本到音频）
        self.speech_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.audio_embedding_dim),
            nn.LayerNorm(config.audio_embedding_dim, eps=1e-12),
        )

        # 语音识别注意力
        self.speech_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads // 2,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, audio_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """前向传播"""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 语音识别（音频到文本）
        speech_text_features = None
        if audio_inputs is not None:
            encoded_audio = self.speech_encoder(audio_inputs)
            speech_text_features, _ = self.speech_attention(
                encoded_audio, encoded_audio, encoded_audio
            )
            speech_text_features = self.dropout(speech_text_features)

        # 语音合成（文本到音频）
        text_to_audio = self.speech_decoder(hidden_states)

        return {
            "speech_text_features": speech_text_features,
            "text_to_audio": text_to_audio,
            "audio_embeddings": audio_inputs,
        }
