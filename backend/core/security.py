"""
安全模块
包含密码哈希、JWT令牌生成、API密钥生成等安全功能
"""

import uuid
from jose import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional
from passlib.context import CryptContext

from .config import Config

# 密码加密上下文 - 使用pbkdf2_sha256算法
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码

    使用标准bcrypt算法进行密码验证。
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """获取密码哈希

    使用标准bcrypt算法生成密码哈希。
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, Config.SECRET_KEY, algorithm=Config.ALGORITHM)
    return encoded_jwt


def generate_api_key() -> str:
    """生成API密钥"""
    return f"{Config.API_KEY_PREFIX}{uuid.uuid4().hex}"


def verify_token(token: str) -> Optional[dict]:
    """验证JWT令牌

    验证JWT令牌的有效性，返回解码后的payload。
    如果令牌无效或过期，返回None。
    """
    try:
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        return payload
    except jwt.JWTError:
        return None
    except Exception:
        return None


__all__ = [
    "pwd_context",
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "generate_api_key",
    "verify_token",
]
