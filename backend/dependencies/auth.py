"""
认证依赖项
包含获取当前用户和管理员的依赖项函数
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import jwt

from .database import get_db
from ..core.config import Config
from ..core.redis import redis_client
from ..db_models.user import User

# 安全
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    """获取当前用户"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的认证凭证"
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌已过期"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的认证凭证"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在"
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="用户已被禁用"
        )

    return user


def get_current_admin(user: User = Depends(get_current_user)):
    """获取当前管理员用户"""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限"
        )
    return user


def rate_limit(api_key: str):
    """速率限制"""
    key = f"rate_limit:{api_key}"
    current = redis_client.get(key)

    if current is None:
        redis_client.setex(key, 60, 1)  # 1分钟过期
        return True
    elif int(current) < Config.API_RATE_LIMIT:
        redis_client.incr(key)
        return True
    else:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后再试",
        )


__all__ = [
    "security",
    "get_current_user",
    "get_current_admin",
    "rate_limit",
]
