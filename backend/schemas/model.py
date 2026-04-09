"""
模型相关数据模型
包含模型创建、部署、评估、推理等请求和响应模型
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import json


class ModelCreate(BaseModel):
    """模型创建请求"""

    name: str = Field(..., min_length=1, max_length=100, description="模型名称")
    description: str = Field("", max_length=1000, description="模型描述")
    model_type: str = Field("transformer", description="模型类型")
    version: str = Field(
        "1.0.0", pattern=r"^\d+\.\d+\.\d+$", description="语义化版本号"
    )
    config: Dict[str, Any] = Field(default_factory=dict, description="模型配置")

    @validator("config", pre=True)
    def validate_config(cls, v):
        """验证配置字段"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("配置必须是有效的JSON格式")
        elif not isinstance(v, dict):
            raise ValueError("配置必须是字典或JSON字符串")
        return v

    @validator("model_type")
    def validate_model_type(cls, v):
        """验证模型类型"""
        valid_types = ["transformer", "cnn", "rnn", "lstm", "multimodal", "custom"]
        if v not in valid_types:
            raise ValueError(f"模型类型必须是以下之一: {', '.join(valid_types)}")
        return v


class ModelDeploy(BaseModel):
    """模型部署请求"""

    resources: str = Field("default", description="资源分配配置")
    replicas: int = Field(1, ge=1, le=10, description="副本数量")
    gpu_count: int = Field(0, ge=0, le=8, description="GPU数量")
    memory_limit_mb: int = Field(4096, ge=1024, le=65536, description="内存限制(MB)")

    @validator("resources")
    def validate_resources(cls, v):
        """验证资源分配配置"""
        valid_resources = ["default", "low", "medium", "high", "custom"]
        if v not in valid_resources:
            raise ValueError(f"资源分配必须是以下之一: {', '.join(valid_resources)}")
        return v


class ModelEvaluate(BaseModel):
    """模型评估请求"""

    dataset_path: Optional[str] = Field(None, max_length=500, description="数据集路径")
    metrics: List[str] = Field(
        ["accuracy", "precision", "recall", "f1"], description="评估指标"
    )
    sample_limit: int = Field(1000, ge=1, le=1000000, description="样本限制")

    @validator("metrics")
    def validate_metrics(cls, v):
        """验证评估指标"""
        valid_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "loss",
            "auc",
            "mse",
            "mae",
        ]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(
                    f"无效的评估指标: {metric}。有效的指标包括: {', '.join(valid_metrics)}"
                )
        return v


class ModelInfer(BaseModel):
    """模型推理请求"""

    input_data: Dict[str, Any] = Field(..., description="输入数据")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="推理参数")
    session_id: str = Field("default", description="会话ID")

    @validator("input_data")
    def validate_input_data(cls, v):
        """验证输入数据"""
        if not isinstance(v, dict):
            raise ValueError("输入数据必须是字典")
        if not v:
            raise ValueError("输入数据不能为空")
        return v

    @validator("session_id")
    def validate_session_id(cls, v):
        """验证会话ID"""
        if not v or not v.strip():
            raise ValueError("会话ID不能为空")
        return v.strip()


class ModelUpdate(BaseModel):
    """模型更新请求"""

    description: Optional[str] = Field(None, max_length=1000, description="模型描述")
    is_active: Optional[bool] = Field(None, description="是否激活")
    config: Optional[Dict[str, Any]] = Field(None, description="模型配置")


class TrainingConfig(BaseModel):
    """训练配置请求"""

    dataset_path: str = Field(..., max_length=500, description="数据集路径")
    epochs: int = Field(10, ge=1, le=1000, description="训练轮数")
    batch_size: int = Field(32, ge=1, le=1024, description="批次大小")
    learning_rate: float = Field(0.001, ge=1e-6, le=1.0, description="学习率")
    validation_split: float = Field(0.2, ge=0.0, le=0.5, description="验证集比例")
    save_checkpoints: bool = Field(True, description="是否保存检查点")


class ModelResponse(BaseModel):
    """模型响应基础模型"""

    model_id: int
    model_name: str
    status: str
    message: str


class ModelDeployResponse(ModelResponse):
    """模型部署响应"""

    endpoint: str
    replicas: int
    resource_allocation: str


class ModelEvaluateResponse(ModelResponse):
    """模型评估响应"""

    metrics: Dict[str, float]
    dataset_info: Dict[str, Any]


class ModelInferResponse(ModelResponse):
    """模型推理响应"""

    output: Dict[str, Any]
    processing_time_ms: float
    tokens_used: int


__all__ = [
    "ModelCreate",
    "ModelDeploy",
    "ModelEvaluate",
    "ModelInfer",
    "ModelUpdate",
    "TrainingConfig",
    "ModelResponse",
    "ModelDeployResponse",
    "ModelEvaluateResponse",
    "ModelInferResponse",
]
