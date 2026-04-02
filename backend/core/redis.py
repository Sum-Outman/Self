"""
Redis缓存模块
包含Redis客户端和缓存相关功能
"""

import redis

from .config import Config

# Redis客户端
redis_client = redis.Redis.from_url(Config.REDIS_URL)


def get_redis_client():
    """获取Redis客户端"""
    return redis_client


__all__ = ["redis_client", "get_redis_client"]