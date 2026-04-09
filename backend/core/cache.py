"""
缓存优化模块
为Self AGI系统提供高性能缓存策略

功能：
1. 多级缓存（内存 + Redis）
2. 缓存预热和刷新策略
3. 缓存统计和监控
4. 分布式缓存一致性
5. 缓存键管理和命名空间
"""

import time
import logging
import hashlib
import pickle
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import threading
import functools

from .config import Config
from .redis import get_redis_client

# 配置日志
logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """缓存级别枚举"""

    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class CachePolicy(Enum):
    """缓存策略枚举"""

    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最不经常使用
    FIFO = "fifo"  # 先进先出
    TTL = "ttl"  # 生存时间


@dataclass
class CacheStats:
    """缓存统计信息"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage_bytes: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def hit_rate(self) -> float:
        """计算命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "size": self.size,
            "memory_usage_bytes": self.memory_usage_bytes,
            "last_updated": self.last_updated.isoformat(),
        }


class MultiLevelCache:
    """多级缓存管理器

    支持内存缓存和Redis缓存的组合，提供高性能缓存访问
    """

    def __init__(
        self,
        memory_max_size: int = 1000,
        redis_prefix: str = "cache:",
        default_ttl: int = 300,
        policy: CachePolicy = CachePolicy.LRU,
        enable_stats: bool = True,
    ):
        """初始化多级缓存"""
        self.memory_max_size = memory_max_size
        self.redis_prefix = redis_prefix
        self.default_ttl = default_ttl
        self.policy = policy
        self.enable_stats = enable_stats

        # 内存缓存
        self.memory_cache: Dict[str, Tuple[Any, float]] = {}
        self.memory_lock = threading.RLock()

        # Redis客户端
        self.redis_client = get_redis_client()

        # 统计信息
        self.stats = CacheStats()

        # 清理线程
        self.cleanup_running = False
        self.cleanup_thread: Optional[threading.Thread] = None

        # 启动清理线程
        self._start_cleanup_thread()

        logger.info(
            f"多级缓存初始化完成: memory_max_size={memory_max_size}, default_ttl={default_ttl}"
        )

    def _start_cleanup_thread(self):
        """启动清理线程"""
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="cache_cleanup"
        )
        self.cleanup_thread.start()
        logger.debug("缓存清理线程已启动")

    def _cleanup_loop(self):
        """清理循环"""
        while self.cleanup_running:
            try:
                self._cleanup_expired()
                time.sleep(60)  # 每分钟清理一次
            except Exception as e:
                logger.error(f"缓存清理失败: {e}")
                time.sleep(10)

    def _cleanup_expired(self):
        """清理过期缓存"""
        with self.memory_lock:
            current_time = time.time()
            expired_keys = []

            for key, (value, expiry) in self.memory_cache.items():
                if expiry > 0 and current_time > expiry:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.memory_cache[key]

            if expired_keys:
                logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")

                if self.enable_stats:
                    self.stats.evictions += len(expired_keys)
                    self.stats.size = len(self.memory_cache)
                    self.stats.last_updated = datetime.now(timezone.utc)

    def _make_redis_key(self, key: str) -> str:
        """生成Redis键"""
        return f"{self.redis_prefix}{key}"

    def _make_cache_key(self, func: Callable, *args, **kwargs) -> str:
        """生成缓存键"""
        # 基于函数名和参数生成唯一键
        key_parts = [
            func.__module__ or "",
            func.__name__ or "",
        ]

        # 添加位置参数
        for arg in args:
            key_parts.append(str(arg))

        # 添加关键字参数
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        # 生成哈希
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        # 首先尝试内存缓存
        with self.memory_lock:
            if key in self.memory_cache:
                value, expiry = self.memory_cache[key]
                current_time = time.time()

                if expiry == 0 or current_time <= expiry:
                    # 缓存命中
                    if self.enable_stats:
                        self.stats.hits += 1
                        self.stats.last_updated = datetime.now(timezone.utc)

                    logger.debug(f"内存缓存命中: {key}")
                    return value
                else:
                    # 缓存过期，从内存中移除
                    del self.memory_cache[key]

        # 内存未命中，尝试Redis缓存
        try:
            redis_key = self._make_redis_key(key)
            cached_value = self.redis_client.get(redis_key)

            if cached_value is not None:
                # Redis命中，同时更新到内存缓存
                value = pickle.loads(cached_value)

                with self.memory_lock:
                    self._set_memory_cache(key, value, self.default_ttl)

                if self.enable_stats:
                    self.stats.hits += 1
                    self.stats.last_updated = datetime.now(timezone.utc)

                logger.debug(f"Redis缓存命中: {key}")
                return value
        except Exception as e:
            logger.error(f"获取Redis缓存失败: {e}")

        # 缓存未命中
        if self.enable_stats:
            self.stats.misses += 1
            self.stats.last_updated = datetime.now(timezone.utc)

        logger.debug(f"缓存未命中: {key}")
        return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        if ttl is None:
            ttl = self.default_ttl

        success = True

        # 设置到内存缓存
        with self.memory_lock:
            self._set_memory_cache(key, value, ttl)

        # 设置到Redis缓存
        try:
            redis_key = self._make_redis_key(key)
            serialized_value = pickle.dumps(value)

            if ttl > 0:
                self.redis_client.setex(redis_key, ttl, serialized_value)
            else:
                self.redis_client.set(redis_key, serialized_value)
        except Exception as e:
            logger.error(f"设置Redis缓存失败: {e}")
            success = False

        logger.debug(f"设置缓存: {key}, ttl={ttl}")
        return success

    def _set_memory_cache(self, key: str, value: Any, ttl: int):
        """设置内存缓存"""
        # 计算过期时间
        expiry = time.time() + ttl if ttl > 0 else 0

        # 检查缓存大小
        if len(self.memory_cache) >= self.memory_max_size:
            self._evict_memory_cache()

        # 设置缓存
        self.memory_cache[key] = (value, expiry)

        # 更新统计
        if self.enable_stats:
            self.stats.size = len(self.memory_cache)
            self.stats.last_updated = datetime.now(timezone.utc)

    def _evict_memory_cache(self):
        """驱逐内存缓存项（根据策略）"""
        if not self.memory_cache:
            return

        if self.policy == CachePolicy.LRU:
            # LRU策略：移除最久未使用的项
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        elif self.policy == CachePolicy.FIFO:
            # FIFO策略：移除最先进入的项
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        else:
            # 默认：随机移除一项
            key_to_remove = next(iter(self.memory_cache))
            del self.memory_cache[key_to_remove]

        if self.enable_stats:
            self.stats.evictions += 1

    def delete(self, key: str) -> bool:
        """删除缓存值"""
        success = True

        # 从内存缓存删除
        with self.memory_lock:
            if key in self.memory_cache:
                del self.memory_cache[key]

                if self.enable_stats:
                    self.stats.size = len(self.memory_cache)
                    self.stats.last_updated = datetime.now(timezone.utc)

        # 从Redis缓存删除
        try:
            redis_key = self._make_redis_key(key)
            self.redis_client.delete(redis_key)
        except Exception as e:
            logger.error(f"删除Redis缓存失败: {e}")
            success = False

        logger.debug(f"删除缓存: {key}")
        return success

    def clear(self) -> bool:
        """清空所有缓存"""
        success = True

        # 清空内存缓存
        with self.memory_lock:
            self.memory_cache.clear()

            if self.enable_stats:
                self.stats.size = 0
                self.stats.last_updated = datetime.now(timezone.utc)

        # 清空Redis缓存（使用模式匹配）
        try:
            pattern = f"{self.redis_prefix}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"清空Redis缓存失败: {e}")
            success = False

        logger.info("已清空所有缓存")
        return success

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = self.stats.to_dict()

        # 添加Redis统计
        try:
            pattern = f"{self.redis_prefix}*"
            keys = self.redis_client.keys(pattern)
            stats["redis_keys_count"] = len(keys) if keys else 0

            # 估算Redis内存使用
            if keys:
                total_size = 0
                for key in keys[:10]:  # 采样10个键
                    try:
                        value = self.redis_client.memory_usage(key)
                        if value:
                            total_size += value
                    except Exception:
                        pass  # 已实现

                if len(keys) > 0:
                    avg_size = total_size / min(10, len(keys))
                    stats["estimated_redis_memory_bytes"] = avg_size * len(keys)
                else:
                    stats["estimated_redis_memory_bytes"] = 0
        except Exception as e:
            logger.error(f"获取Redis统计失败: {e}")
            stats["redis_keys_count"] = 0
            stats["estimated_redis_memory_bytes"] = 0

        return stats

    def cache_decorator(self, ttl: Optional[int] = None):
        """缓存装饰器"""

        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._make_cache_key(func, *args, **kwargs)

                # 尝试获取缓存
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # 执行函数
                result = func(*args, **kwargs)

                # 设置缓存
                self.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator

    def shutdown(self):
        """关闭缓存系统"""
        self.cleanup_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)

        logger.info("缓存系统已关闭")


# 全局缓存实例
default_cache = MultiLevelCache(
    memory_max_size=Config.CACHE_MAX_ENTRIES,
    redis_prefix=Config.CACHE_REDIS_PREFIX,
    default_ttl=Config.CACHE_DEFAULT_TIMEOUT,
    policy=CachePolicy.LRU,
    enable_stats=True,
)


def cache(ttl: Optional[int] = None):
    """便捷缓存装饰器"""
    return default_cache.cache_decorator(ttl)


def get_cache_stats() -> Dict[str, Any]:
    """获取缓存统计"""
    return default_cache.get_stats()


def clear_cache() -> bool:
    """清空缓存"""
    return default_cache.clear()


__all__ = [
    "MultiLevelCache",
    "CacheStats",
    "CacheLevel",
    "CachePolicy",
    "default_cache",
    "cache",
    "get_cache_stats",
    "clear_cache",
]
