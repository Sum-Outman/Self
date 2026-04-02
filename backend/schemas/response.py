"""
API响应数据模型
统一前后端API响应格式，确保错误处理和类型安全

设计原则：
1. 所有API响应遵循相同结构
2. 支持成功响应和错误响应
3. 提供类型安全的TypeScript定义
4. 兼容现有前端ApiResponse接口
"""

from pydantic import BaseModel, Field
from typing import Optional, Any, TypeVar, Generic
from enum import Enum

T = TypeVar('T')


class ResponseStatus(str, Enum):
    """响应状态枚举"""
    SUCCESS = "success"
    ERROR = "error"


class ApiResponse(BaseModel, Generic[T]):
    """通用API响应模型
    
    对应前端TypeScript接口：
    interface ApiResponse<T = any> {
      success: boolean;
      data?: T;
      message?: string;
      error?: string;
      code?: number;
    }
    """
    
    # 状态标识 (兼容前端success字段)
    success: bool = Field(..., description="请求是否成功")
    
    # 数据负载 (成功时返回)
    data: Optional[T] = Field(None, description="响应数据")
    
    # 信息字段
    message: Optional[str] = Field(None, description="成功或错误信息")
    error: Optional[str] = Field(None, description="错误详情")
    
    # 状态码
    code: Optional[int] = Field(None, description="HTTP状态码或自定义错误码")
    
    # 时间戳
    timestamp: Optional[str] = Field(None, description="响应时间戳")
    
    # 请求追踪
    request_id: Optional[str] = Field(None, description="请求ID，用于追踪")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"id": 1, "name": "示例数据"},
                "message": "操作成功",
                "code": 200,
                "timestamp": "2026-03-14T04:15:30Z",
                "request_id": "req_123456789"
            }
        }


class ErrorResponse(ApiResponse[Any]):
    """错误响应模型
    
    用于统一错误响应格式，确保前端可以一致地处理错误
    """
    
    @classmethod
    def create(
        cls,
        success: bool = False,
        message: str = "操作失败",
        error: str = "未知错误",
        code: int = 500,
        data: Any = None,
        request_id: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> 'ErrorResponse':
        """创建错误响应
        
        参数:
            success: 是否成功 (始终为False)
            message: 用户友好的错误信息
            error: 详细的错误信息 (用于调试)
            code: HTTP状态码或自定义错误码
            data: 可选的额外错误数据
            request_id: 请求ID
            timestamp: 时间戳（ISO格式），如未提供则使用当前时间
            
        返回:
            ErrorResponse实例
        """
        if timestamp is None:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).isoformat()
        
        return cls(
            success=success,
            data=data,
            message=message,
            error=error,
            code=code,
            request_id=request_id,
            timestamp=timestamp
        )
    
    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        message: Optional[str] = None,
        code: int = 500,
        request_id: Optional[str] = None
    ) -> 'ErrorResponse':
        """从异常创建错误响应
        
        参数:
            exc: 异常对象
            message: 用户友好的错误信息 (如果为空，使用异常消息)
            code: HTTP状态码
            request_id: 请求ID
            
        返回:
            ErrorResponse实例
        """
        error_detail = str(exc) if str(exc) else "未知错误"
        user_message = message or error_detail
        
        return cls.create(
            success=False,
            message=user_message,
            error=error_detail,
            code=code,
            request_id=request_id
        )


class SuccessResponse(ApiResponse[T]):
    """成功响应模型"""
    
    @classmethod
    def create(
        cls,
        data: T,
        message: str = "操作成功",
        code: int = 200,
        request_id: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> 'SuccessResponse[T]':
        """创建成功响应
        
        参数:
            data: 响应数据
            message: 成功信息
            code: HTTP状态码
            request_id: 请求ID
            timestamp: 时间戳（ISO格式），如未提供则使用当前时间
            
        返回:
            SuccessResponse实例
        """
        if timestamp is None:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).isoformat()
        
        return cls(
            success=True,
            data=data,
            message=message,
            error=None,
            code=code,
            request_id=request_id,
            timestamp=timestamp
        )


class PaginatedResponse(ApiResponse[dict]):
    """分页响应模型
    
    对应前端TypeScript接口：
    interface PaginatedResponse<T> {
      items: T[];
      total: number;
      page: number;
      size: number;
      pages: number;
    }
    """
    
    @classmethod
    def create(
        cls,
        items: list,
        total: int,
        page: int,
        size: int,
        message: str = "查询成功",
        code: int = 200,
        request_id: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> 'PaginatedResponse':
        """创建分页响应
        
        参数:
            items: 当前页数据列表
            total: 总数据量
            page: 当前页码
            size: 每页大小
            message: 成功信息
            code: HTTP状态码
            request_id: 请求ID
            timestamp: 时间戳（ISO格式），如未提供则使用当前时间
            
        返回:
            PaginatedResponse实例
        """
        if timestamp is None:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).isoformat()
        
        data = {
            "items": items,
            "total": total,
            "page": page,
            "size": size,
            "pages": (total + size - 1) // size  # 计算总页数
        }
        
        return cls(
            success=True,
            data=data,
            message=message,
            error=None,
            code=code,
            request_id=request_id,
            timestamp=timestamp
        )


# 导出常用函数
def success_response(data: Any = None, message: str = "操作成功", code: int = 200) -> dict:
    """快速创建成功响应 (兼容现有代码)
    
    注意: 此函数返回字典，而不是ApiResponse实例，以保持与现有代码的兼容性
    """
    return {
        "success": True,
        "data": data,
        "message": message,
        "code": code
    }


def error_response(message: str = "操作失败", error: str = "未知错误", code: int = 500) -> dict:
    """快速创建错误响应 (兼容现有代码)"""
    return {
        "success": False,
        "message": message,
        "error": error,
        "code": code
    }

# 别名定义，用于向后兼容
StandardResponse = ApiResponse


__all__ = [
    "ApiResponse",
    "ErrorResponse",
    "SuccessResponse",
    "PaginatedResponse",
    "success_response",
    "error_response",
    "ResponseStatus",
]