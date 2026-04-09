"""
API速率限制模块
基于Redis和slowapi实现分布式速率限制
"""

import logging
from typing import Dict, Any, Optional
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse

from .config import Config
from .redis import get_redis_client

# 配置日志
logger = logging.getLogger(__name__)


class RateLimiter:
    """速率限制器"""

    def __init__(self):
        """初始化速率限制器"""
        try:
            self.redis_client = get_redis_client()
            # 测试Redis连接
            self.redis_client.ping()
            self.redis_available = True
        except Exception as e:
            # Redis是可选的，使用info级别记录，避免过多警告
            logger.info(f"Redis连接失败，速率限制将以降级模式运行（可选服务）: {e}")
            self.redis_available = False
            # 创建虚拟Redis客户端
            self.redis_client = None

        self.default_limits = Config.RATE_LIMITS

        try:
            self.limiter = Limiter(
                key_func=get_remote_address,
                default_limits=self.default_limits,
                storage_uri=Config.REDIS_URL if self.redis_available else "memory://",
                headers_enabled=True,
            )
        except Exception as e:
            logger.error(f"初始化速率限制器失败: {e}")
            # 创建降级限制器
            self.limiter = Limiter(
                key_func=get_remote_address,
                default_limits=["1000 per hour"],  # 宽松的限制
                storage_uri="memory://",
                headers_enabled=False,
            )

        logger.info(f"速率限制器初始化完成，Redis可用: {self.redis_available}")

    def get_limiter(self) -> Limiter:
        """获取速率限制器实例"""
        return self.limiter

    def get_rate_limit_key(self, request: Request, endpoint: str) -> str:
        """获取速率限制键"""
        # 使用客户端IP地址和端点作为键
        client_ip = get_remote_address(request)
        key = f"rate_limit:{endpoint}:{client_ip}"
        return key

    def get_rate_limit_info(self, request: Request, endpoint: str) -> Dict[str, Any]:
        """获取当前速率限制信息"""
        try:
            # 如果Redis不可用，返回降级信息
            if not self.redis_available or self.redis_client is None:
                return {
                    "endpoint": endpoint,
                    "client_ip": get_remote_address(request),
                    "limits": self._get_endpoint_limits(endpoint),
                    "remaining": None,
                    "reset_time": None,
                    "limited": False,
                    "degraded": True,
                }

            key = self.get_rate_limit_key(request, endpoint)

            # 获取Redis中的限制信息
            remaining_key = f"{key}:remaining"
            reset_key = f"{key}:reset"

            remaining = self.redis_client.get(remaining_key)
            reset_time = self.redis_client.get(reset_key)

            # 获取限制配置
            limits = self._get_endpoint_limits(endpoint)

            return {
                "endpoint": endpoint,
                "client_ip": get_remote_address(request),
                "limits": limits,
                "remaining": int(remaining) if remaining else None,
                "reset_time": int(reset_time) if reset_time else None,
                "limited": remaining is not None and int(remaining) <= 0,
                "degraded": False,
            }
        except Exception as e:
            logger.error(f"获取速率限制信息失败: {e}")
            return {
                "endpoint": endpoint,
                "client_ip": get_remote_address(request),
                "limits": self._get_endpoint_limits(endpoint),
                "remaining": None,
                "reset_time": None,
                "limited": False,
                "degraded": True,
                "error": str(e),
            }

    def _get_endpoint_limits(self, endpoint: str) -> list:
        """根据端点获取限制配置"""
        # 默认限制
        default_limits = self.default_limits

        # 特殊端点的限制配置
        endpoint_limits = Config.ENDPOINT_RATE_LIMITS.get(endpoint, [])

        # 返回端点特定限制或默认限制
        return endpoint_limits if endpoint_limits else default_limits

    def check_rate_limit(self, request: Request, endpoint: str) -> bool:
        """检查速率限制（手动检查）"""
        try:
            # 获取限制信息
            info = self.get_rate_limit_info(request, endpoint)

            # 如果处于降级模式，允许所有请求
            if info.get("degraded", False):
                logger.debug(f"速率限制降级模式: {endpoint} - 允许请求")
                return True

            if info.get("limited", False):
                logger.warning(f"速率限制触发: {endpoint} - {info['client_ip']}")
                return False

            return True
        except Exception as e:
            logger.error(f"检查速率限制失败: {e}")
            # 出错时允许通过
            return True

    def get_rate_limit_headers(self, request: Request, endpoint: str) -> Dict[str, str]:
        """获取速率限制相关HTTP头"""
        try:
            info = self.get_rate_limit_info(request, endpoint)

            # 完整的头信息
            if info.get("degraded", False):
                return {
                    "X-RateLimit-Limit": "unlimited",
                    "X-RateLimit-Remaining": "unlimited",
                    "X-RateLimit-Reset": "N/A",
                    "X-RateLimit-Policy": "degraded",
                    "X-RateLimit-Mode": "degraded",
                }

            headers = {
                "X-RateLimit-Limit": str(len(info.get("limits", []))),
                "X-RateLimit-Remaining": str(info.get("remaining", "unknown")),
                "X-RateLimit-Reset": str(info.get("reset_time", "unknown")),
                "X-RateLimit-Policy": "; ".join(info.get("limits", [])),
            }

            return headers
        except Exception as e:
            logger.error(f"获取速率限制头信息失败: {e}")
            return {
                "X-RateLimit-Limit": "unlimited",
                "X-RateLimit-Remaining": "unlimited",
                "X-RateLimit-Reset": "N/A",
                "X-RateLimit-Policy": "error",
                "X-RateLimit-Mode": "error",
            }


def custom_rate_limit_exceeded_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """自定义速率限制异常处理器

    处理RateLimitExceeded异常，兼容各种异常类型
    """
    try:
        # 尝试获取异常详情
        if hasattr(exc, "detail"):
            detail = exc.detail
        elif hasattr(exc, "message"):
            detail = exc.message
        else:
            detail = str(exc) if str(exc) else "请求过于频繁，请稍后再试"

        logger.warning(f"速率限制触发: {detail}")

        return JSONResponse(
            status_code=429,
            content={
                "error": f"速率限制: {detail}",
                "message": "请求过于频繁，请稍后再试",
                "retry_after": 60,  # 默认60秒后重试
            },
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "60",
            },
        )
    except Exception as e:
        # 如果处理异常时出错，返回通用错误
        logger.error(f"处理速率限制异常时出错: {e}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "速率限制",
                "message": "请求过于频繁，请稍后再试",
                "retry_after": 60,
            },
            headers={"Retry-After": "60"},
        )


def setup_rate_limiting(app):
    """设置速率限制中间件"""
    try:
        rate_limiter = RateLimiter()

        # 设置速率限制器
        app.state.limiter = rate_limiter.get_limiter()

        # 添加自定义速率限制异常处理器
        app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)

        # 添加中间件
        app.add_middleware(SlowAPIMiddleware)

        logger.info("速率限制中间件设置完成")

        return rate_limiter
    except Exception as e:
        logger.error(f"设置速率限制失败，将使用降级模式: {e}")
        # 即使失败也返回降级速率限制器
        try:
            # 创建降级速率限制器
            rate_limiter = RateLimiter()
            return rate_limiter
        except Exception as inner_e:
            logger.error(f"创建降级速率限制器也失败: {inner_e}")
            # 返回None，让应用继续运行
            return None  # 返回None


# 全局速率限制器实例
rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> Optional[RateLimiter]:
    """获取全局速率限制器实例"""
    return rate_limiter


def init_rate_limiter(app):
    """初始化速率限制器"""
    global rate_limiter
    rate_limiter = setup_rate_limiting(app)
    return rate_limiter
