#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一错误处理模块
减少重复的错误处理代码，提供一致的错误处理模式
"""

import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar
from functools import wraps
from contextlib import contextmanager

from fastapi import HTTPException
from fastapi.responses import JSONResponse

# 类型变量
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# 创建日志记录器
logger = logging.getLogger(__name__)


class ErrorHandler:
    """统一错误处理器"""

    def __init__(self, default_logger: Optional[logging.Logger] = None):
        """初始化错误处理器

        参数:
            default_logger: 默认日志记录器
        """
        self.default_logger = default_logger or logger
        self.error_handlers = {}

        # 注册默认错误处理器
        self._register_default_handlers()

    def _register_default_handlers(self):
        """注册默认错误处理器"""
        self.register_handler(Exception, self._handle_general_exception)
        self.register_handler(HTTPException, self._handle_http_exception)
        self.register_handler(ValueError, self._handle_value_error)
        self.register_handler(TypeError, self._handle_type_error)
        self.register_handler(KeyError, self._handle_key_error)
        self.register_handler(AttributeError, self._handle_attribute_error)

    def register_handler(self, exception_type: Type[Exception], handler: Callable):
        """注册错误处理器

        参数:
            exception_type: 异常类型
            handler: 处理函数
        """
        self.error_handlers[exception_type] = handler

    def handle(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """处理异常

        参数:
            exception: 异常对象
            context: 上下文信息

        返回:
            处理结果
        """
        context = context or {}

        # 查找匹配的处理器
        handler = None
        for exc_type in self.error_handlers:
            if isinstance(exception, exc_type):
                handler = self.error_handlers[exc_type]
                break

        # 使用通用处理器作为后备
        if not handler:
            handler = self._handle_general_exception

        try:
            return handler(exception, context)
        except Exception as e:
            # 处理器本身出错时使用最小化处理
            self.default_logger.error(f"错误处理器失败: {e}")
            return self._minimal_error_response(exception, context)

    def _handle_general_exception(
        self, exception: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理一般异常"""
        error_id = context.get("error_id", "unknown")
        operation = context.get("operation", "unknown_operation")

        # 记录详细错误信息
        self.default_logger.error(
            f"操作 '{operation}' 失败 (错误ID: {error_id}): {exception}",
            exc_info=True,
            extra={
                "error_id": error_id,
                "operation": operation,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
            },
        )

        return {
            "success": False,
            "error": {
                "id": error_id,
                "type": "internal_error",
                "message": "内部服务器错误",
                "details": (
                    str(exception) if self._should_show_details(context) else None
                ),
                "operation": operation,
            },
            "timestamp": context.get("timestamp"),
        }

    def _handle_http_exception(
        self, exception: HTTPException, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理HTTP异常"""
        error_id = context.get("error_id", "http_error")
        operation = context.get("operation", "http_operation")

        self.default_logger.warning(
            f"HTTP错误 (错误ID: {error_id}, 状态码: {exception.status_code}): {exception.detail}",
            extra={
                "error_id": error_id,
                "operation": operation,
                "status_code": exception.status_code,
                "detail": exception.detail,
            },
        )

        return {
            "success": False,
            "error": {
                "id": error_id,
                "type": "http_error",
                "message": exception.detail,
                "status_code": exception.status_code,
                "operation": operation,
            },
            "timestamp": context.get("timestamp"),
        }

    def _handle_value_error(
        self, exception: ValueError, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理值错误"""
        error_id = context.get("error_id", "validation_error")
        operation = context.get("operation", "validation_operation")

        self.default_logger.warning(
            f"验证错误 (错误ID: {error_id}): {exception}",
            extra={
                "error_id": error_id,
                "operation": operation,
                "exception_message": str(exception),
            },
        )

        return {
            "success": False,
            "error": {
                "id": error_id,
                "type": "validation_error",
                "message": f"数据验证失败: {exception}",
                "operation": operation,
            },
            "timestamp": context.get("timestamp"),
        }

    def _handle_type_error(
        self, exception: TypeError, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理类型错误"""
        error_id = context.get("error_id", "type_error")
        operation = context.get("operation", "type_operation")

        self.default_logger.warning(
            f"类型错误 (错误ID: {error_id}): {exception}",
            extra={
                "error_id": error_id,
                "operation": operation,
                "exception_message": str(exception),
            },
        )

        return {
            "success": False,
            "error": {
                "id": error_id,
                "type": "type_error",
                "message": f"类型错误: {exception}",
                "operation": operation,
            },
            "timestamp": context.get("timestamp"),
        }

    def _handle_key_error(
        self, exception: KeyError, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理键错误"""
        error_id = context.get("error_id", "key_error")
        operation = context.get("operation", "key_operation")

        self.default_logger.warning(
            f"键错误 (错误ID: {error_id}): 键 '{exception}' 不存在",
            extra={
                "error_id": error_id,
                "operation": operation,
                "missing_key": str(exception),
            },
        )

        return {
            "success": False,
            "error": {
                "id": error_id,
                "type": "key_error",
                "message": f"键 '{exception}' 不存在",
                "operation": operation,
            },
            "timestamp": context.get("timestamp"),
        }

    def _handle_attribute_error(
        self, exception: AttributeError, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理属性错误"""
        error_id = context.get("error_id", "attribute_error")
        operation = context.get("operation", "attribute_operation")

        self.default_logger.warning(
            f"属性错误 (错误ID: {error_id}): {exception}",
            extra={
                "error_id": error_id,
                "operation": operation,
                "exception_message": str(exception),
            },
        )

        return {
            "success": False,
            "error": {
                "id": error_id,
                "type": "attribute_error",
                "message": f"属性错误: {exception}",
                "operation": operation,
            },
            "timestamp": context.get("timestamp"),
        }

    def _minimal_error_response(
        self, exception: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """最小化错误响应（处理器失败时的后备）"""
        self.default_logger.critical(
            f"严重: 错误处理器失败: {exception}",
            exc_info=True,
        )

        return {
            "success": False,
            "error": {
                "id": "critical_error",
                "type": "critical",
                "message": "系统内部错误",
                "operation": context.get("operation", "unknown"),
            },
        }

    def _should_show_details(self, context: Dict[str, Any]) -> bool:
        """是否显示错误详情"""
        return context.get("show_details", False) or context.get("debug_mode", False)


# 创建全局错误处理器实例
global_error_handler = ErrorHandler()


def handle_errors(
    error_id: str = "unknown",
    operation: str = "unknown_operation",
    show_details: bool = False,
    rethrow: bool = False,
    default_return: Any = None,
):
    """错误处理装饰器

    参数:
        error_id: 错误ID
        operation: 操作名称
        show_details: 是否显示错误详情
        rethrow: 是否重新抛出异常
        default_return: 发生错误时的默认返回值

    返回:
        装饰器函数
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                import datetime

                context = {
                    "error_id": error_id,
                    "operation": operation,
                    "show_details": show_details,
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "function_name": func.__name__,
                    "module_name": func.__module__,
                }

                # 处理异常
                error_response = global_error_handler.handle(e, context)

                # 记录错误
                logger.error(
                    f"函数 {func.__name__} 执行失败 (错误ID: {error_id}): {e}",
                    extra=context,
                )

                if rethrow:
                    raise e

                return default_return if default_return is not None else error_response

        return wrapper

    return decorator


@contextmanager
def error_context(
    error_id: str = "unknown",
    operation: str = "unknown_operation",
    show_details: bool = False,
    suppress_exception: bool = False,
    default_return: Any = None,
):
    """错误处理上下文管理器

    参数:
        error_id: 错误ID
        operation: 操作名称
        show_details: 是否显示错误详情
        suppress_exception: 是否抑制异常
        default_return: 发生错误时的默认返回值

    返回:
        上下文管理器
    """
    import datetime

    context = {
        "error_id": error_id,
        "operation": operation,
        "show_details": show_details,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    try:
        yield context
    except Exception as e:
        # 处理异常
        error_response = global_error_handler.handle(e, context)

        # 记录错误
        logger.error(
            f"操作 '{operation}' 失败 (错误ID: {error_id}): {e}",
            extra=context,
        )

        if not suppress_exception:
            raise e

        if default_return is not None:
            return default_return

        return error_response


def api_error_handler(func=None, *, status_code=500, error_id="api_error"):
    """API错误处理装饰器

    专门用于FastAPI路由函数的错误处理装饰器

    参数:
        func: 被装饰的函数（当作为无参数装饰器使用时为None）
        status_code: HTTP状态码，默认500
        error_id: 错误ID，默认'api_error'

    使用方式:
        @api_error_handler
        async def func1(): ...

        @api_error_handler(status_code=503, error_id='service_unavailable')
        async def func2(): ...
    """
    if func is None:
        # 带参数调用，返回装饰器
        def decorator(f: F) -> F:
            @wraps(f)
            async def wrapper(*args, **kwargs):
                try:
                    return (
                        await f(*args, **kwargs) if callable(f) else f(*args, **kwargs)
                    )
                except HTTPException:
                    # 重新抛出HTTP异常（让FastAPI处理）
                    raise
                except Exception as e:
                    import datetime

                    context = {
                        "error_id": error_id,
                        "operation": f.__name__,
                        "timestamp": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                        "function_name": f.__name__,
                        "module_name": f.__module__,
                        "status_code": status_code,
                    }

                    # 处理异常
                    error_response = global_error_handler.handle(e, context)

                    # 返回JSON响应
                    return JSONResponse(
                        status_code=status_code,
                        content=error_response,
                    )

            return wrapper

        return decorator

    # 无参数调用，func是函数
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return (
                await func(*args, **kwargs) if callable(func) else func(*args, **kwargs)
            )
        except HTTPException:
            # 重新抛出HTTP异常（让FastAPI处理）
            raise
        except Exception as e:
            import datetime

            context = {
                "error_id": error_id,
                "operation": func.__name__,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "function_name": func.__name__,
                "module_name": func.__module__,
                "status_code": status_code,
            }

            # 处理异常
            error_response = global_error_handler.handle(e, context)

            # 返回JSON响应
            return JSONResponse(
                status_code=status_code,
                content=error_response,
            )

    return wrapper


def db_error_handler(func: F) -> F:
    """数据库错误处理装饰器

    专门用于数据库操作的错误处理
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import datetime

            context = {
                "error_id": "db_error",
                "operation": "database_operation",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "function_name": func.__name__,
                "module_name": func.__module__,
                "db_operation": func.__name__,
            }

            # 处理数据库相关异常
            error_response = global_error_handler.handle(e, context)

            # 记录数据库错误
            logger.error(
                f"数据库操作 '{func.__name__}' 失败: {e}",
                extra=context,
            )

            # 返回错误响应
            return error_response

    return wrapper


# 导出常用函数和类
__all__ = [
    "ErrorHandler",
    "global_error_handler",
    "handle_errors",
    "error_context",
    "api_error_handler",
    "db_error_handler",
]
