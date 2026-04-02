"""
依赖注入模块
包含FastAPI依赖项定义
"""

from .database import get_db
from .auth import security, get_current_user, get_current_admin, rate_limit

__all__ = [
    "get_db",
    "security",
    "get_current_user",
    "get_current_admin",
    "rate_limit",
]