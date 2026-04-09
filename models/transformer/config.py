# Self AGI 模型配置
from typing import Dict, Any
import json
from .self_agi_model import AGIModelConfig


class ModelConfig(AGIModelConfig):
    """模型配置

    兼容性配置类，与SelfAGIModel的AGIModelConfig保持兼容
    """

    def __init__(self, **kwargs: Any) -> None:
        # AGIModelConfig 字段列表
        agi_fields = {
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "hidden_act",
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "max_position_embeddings",
            "initializer_range",
            "layer_norm_eps",
            "multimodal_enabled",
            "text_embedding_dim",
            "image_embedding_dim",
            "audio_embedding_dim",
            "video_embedding_dim",
            "sensor_embedding_dim",
            "fused_embedding_dim",
            "planning_enabled",
            "reasoning_enabled",
            "execution_control_enabled",
            "self_cognition_enabled",
            "learning_enabled",
            "self_correction_enabled",
            "multimodal_fusion_enabled",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "max_grad_norm",
        }

        # 提取父类字段
        parent_kwargs = {}
        extra_kwargs = {}

        for key, value in kwargs.items():
            if key in agi_fields:
                parent_kwargs[key] = value
            else:
                extra_kwargs[key] = value

        # 调用父类初始化
        super().__init__(**parent_kwargs)

        # 设置额外属性（不在AGIModelConfig中的字段）
        self.model_type = extra_kwargs.get("model_type", "transformer")

        # 设置其他额外属性
        for key, value in extra_kwargs.items():
            if key != "model_type":
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """从字典创建配置"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def __str__(self) -> str:
        """字符串表示"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        """表示"""
        return f"ModelConfig({json.dumps(self.to_dict(), ensure_ascii=False)})"
