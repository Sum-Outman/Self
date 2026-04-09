"""
认证相关数据模型
包含用户注册、登录、更新等请求和响应模型
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any


class UserCreate(BaseModel):
    """用户创建请求"""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """用户登录请求"""

    username: str
    password: str


class UserUpdate(BaseModel):
    """用户更新请求"""

    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None


class AdminUserUpdate(BaseModel):
    """管理员用户更新请求"""

    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None


class LoginResponse(BaseModel):
    """登录响应"""

    access_token: str
    token_type: str
    session_token: str
    user: Dict[str, Any]


class RegisterResponse(BaseModel):
    """注册响应"""

    message: str
    user_id: int
    api_key: str


__all__ = [
    "UserCreate",
    "UserLogin",
    "UserUpdate",
    "AdminUserUpdate",
    "LoginResponse",
    "RegisterResponse",
]
