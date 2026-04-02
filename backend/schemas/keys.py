"""
API密钥相关数据模型
包含API密钥创建、列表等请求和响应模型
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class APIKeyCreate(BaseModel):
    """API密钥创建请求"""

    name: str = Field(..., min_length=1, max_length=50)
    rate_limit: Optional[int] = Field(100, ge=1, le=1000)


class APIKeyResponse(BaseModel):
    """API密钥响应"""

    id: int
    name: str
    key: Optional[str]  # 可能只显示部分
    is_active: bool
    rate_limit: int
    created_at: str
    expires_at: Optional[str]


class APIKeyListResponse(BaseModel):
    """API密钥列表响应"""

    keys: list[APIKeyResponse]


class APIKeyCreateResponse(BaseModel):
    """API密钥创建响应"""

    message: str
    api_key: str
    name: str
    rate_limit: int
    created_at: str


__all__ = [
    "APIKeyCreate",
    "APIKeyResponse",
    "APIKeyListResponse",
    "APIKeyCreateResponse",
]