"""
API响应缓存模块
为FastAPI路由提供响应缓存功能

功能：
1. 基于请求参数的响应缓存
2. 支持TTL（生存时间）和缓存失效
3. 支持多种缓存级别（内存、Redis）
4. 缓存统计和监控
5. 缓存键生成和命名空间管理
"""

import asyncio
import hashlib
import json
import logging
from typing import Any, Callable, Dict, Optional
from functools import wraps
from datetime import datetime

from .cache import MultiLevelCache, CacheLevel

# 配置日志
logger = logging.getLogger(__name__)

# 获取缓存管理器实例
cache_manager = MultiLevelCache()


class ResponseCache:
    """API响应缓存类"""

    def __init__(
        self,
        ttl: int = 300,
        cache_level: CacheLevel = CacheLevel.MEMORY,
        key_prefix: str = "api_response",
    ):
        """
        初始化响应缓存

        Args:
            ttl: 缓存生存时间（秒），默认5分钟
            cache_level: 缓存级别，默认内存缓存
            key_prefix: 缓存键前缀
        """
        self.ttl = ttl
        self.cache_level = cache_level
        self.key_prefix = key_prefix

    def generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """
        生成缓存键

        Args:
            func_name: 函数名称
            args: 函数位置参数
            kwargs: 函数关键字参数

        Returns:
            缓存键字符串
        """
        # 过滤掉不相关的参数（如request对象）
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith("_") and not callable(v)
        }

        # 创建键的字符串表示
        key_data = {
            "func": func_name,
            "args": str(args),
            "kwargs": str(filtered_kwargs),
        }

        # 使用SHA256生成哈希
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()

        return f"{self.key_prefix}:{func_name}:{key_hash}"

    def __call__(self, func: Callable):
        """
        缓存装饰器

        Args:
            func: 被装饰的函数

        Returns:
            包装后的函数
        """

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = self.generate_cache_key(func.__name__, args, kwargs)

            # 尝试从缓存获取
            cached_data = cache_manager.get(cache_key)
            if cached_data is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_data

            # 缓存未命中，执行函数
            logger.debug(f"缓存未命中: {cache_key}")
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            # 将结果存入缓存
            cache_manager.set(cache_key, result, ttl=self.ttl)

            return result

        return wrapper


# 导入asyncio用于协程检测


# 预定义的缓存配置
def cache_response(ttl: int = 300, cache_level: CacheLevel = CacheLevel.MEMORY):
    """
    快速创建响应缓存装饰器

    Args:
        ttl: 缓存生存时间（秒）
        cache_level: 缓存级别

    Returns:
        响应缓存装饰器
    """
    cache = ResponseCache(ttl=ttl, cache_level=cache_level)
    return cache


# 缓存统计和管理功能
class CacheMonitor:
    """缓存监控器"""

    def __init__(self):
        self.stats: Dict[str, Dict[str, int]] = {}

    def record_hit(self, cache_key: str):
        """记录缓存命中"""
        if cache_key not in self.stats:
            self.stats[cache_key] = {"hits": 0, "misses": 0}
        self.stats[cache_key]["hits"] += 1

    def record_miss(self, cache_key: str):
        """记录缓存未命中"""
        if cache_key not in self.stats:
            self.stats[cache_key] = {"hits": 0, "misses": 0}
        self.stats[cache_key]["misses"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_hits = sum(stat["hits"] for stat in self.stats.values())
        total_misses = sum(stat["misses"] for stat in self.stats.values())
        total_requests = total_hits + total_misses
        hit_rate = total_hits / total_requests if total_requests > 0 else 0

        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "detailed_stats": self.stats,
        }

    def clear_stats(self):
        """清除统计信息"""
        self.stats.clear()


# 全局缓存监控器实例
cache_monitor = CacheMonitor()


# 带监控的缓存装饰器
def monitored_cache_response(
    ttl: int = 300, cache_level: CacheLevel = CacheLevel.MEMORY
):
    """
    带监控的响应缓存装饰器

    Args:
        ttl: 缓存生存时间（秒）
        cache_level: 缓存级别

    Returns:
        带监控的缓存装饰器
    """

    def decorator(func):
        cache = ResponseCache(ttl=ttl, cache_level=cache_level)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = cache.generate_cache_key(func.__name__, args, kwargs)

            # 尝试从缓存获取
            cached_data = cache_manager.get(cache_key, level=cache_level)
            if cached_data is not None:
                cache_monitor.record_hit(cache_key)
                logger.debug(f"缓存命中（监控）: {cache_key}")
                return cached_data

            # 缓存未命中
            cache_monitor.record_miss(cache_key)
            logger.debug(f"缓存未命中（监控）: {cache_key}")

            # 执行函数
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            # 存入缓存
            cache_manager.set(cache_key, result, ttl=ttl, level=cache_level)

            return result

        return wrapper

    return decorator


# 缓存管理端点
def get_cache_stats() -> Dict[str, Any]:
    """获取缓存统计信息"""
    return cache_monitor.get_stats()


def clear_cache(prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    清除缓存

    Args:
        prefix: 缓存键前缀，如果提供则只清除指定前缀的缓存

    Returns:
        清除结果
    """
    cleared_count = cache_manager.clear(prefix)

    return {
        "success": True,
        "cleared_count": cleared_count,
        "prefix": prefix,
        "timestamp": datetime.now().isoformat(),
    }


def get_cache_info() -> Dict[str, Any]:
    """获取缓存系统信息"""
    return {
        "cache_manager": cache_manager.get_info(),
        "monitor_stats": cache_monitor.get_stats(),
        "config": {
            "default_ttl": 300,
            "available_levels": [level.value for level in CacheLevel],
            "monitoring_enabled": True,
        },
    }
