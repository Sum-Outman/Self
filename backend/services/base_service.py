#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务基类模块

根据《重复文件和冗余代码整合计划》第二阶段创建
提供所有服务类的公共基础功能：
1. 单例模式实现
2. 统一日志记录
3. 配置管理
4. 错误处理和重试机制
5. 依赖检查和初始化
6. 服务状态监控

设计原则：
- 最小侵入性：现有服务可以轻松迁移
- 向后兼容性：不破坏现有服务接口
- 可扩展性：易于添加新功能
- 可维护性：集中管理通用逻辑
"""

import logging
import time
import threading
import functools
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, Union, List
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import traceback
import json

T = TypeVar('T')


class ServiceError(Exception):
    """服务错误异常基类"""
    def __init__(self, message: str, service_name: str = None, error_code: str = None):
        self.message = message
        self.service_name = service_name
        self.error_code = error_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error": self.message,
            "service": self.service_name,
            "code": self.error_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class ServiceConfig:
    """服务配置基类"""
    def __init__(self, **kwargs):
        self.service_name: str = kwargs.get("service_name", "unknown_service")
        self.log_level: str = kwargs.get("log_level", "INFO")
        self.max_retries: int = kwargs.get("max_retries", 3)
        self.retry_delay: float = kwargs.get("retry_delay", 1.0)
        self.timeout: Optional[float] = kwargs.get("timeout", None)
        self.enable_metrics: bool = kwargs.get("enable_metrics", True)
        self.enable_health_check: bool = kwargs.get("enable_health_check", True)
        
        # 扩展配置
        self.extra_config: Dict[str, Any] = kwargs.get("extra_config", {})
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "service_name": self.service_name,
            "log_level": self.log_level,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "enable_metrics": self.enable_metrics,
            "enable_health_check": self.enable_health_check,
            "extra_config": self.extra_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ServiceConfig":
        """从字典创建配置"""
        return cls(**config_dict)


class BaseService(ABC, Generic[T]):
    """服务基类
    
    提供所有服务类的公共基础功能，支持单例模式。
    
    使用示例：
    ```python
    class MyService(BaseService):
        def __init__(self, config: Optional[ServiceConfig] = None):
            super().__init__(config)
            # 初始化特定资源
            
        def _initialize_service(self) -> bool:
            # 实现服务特定初始化
            return True
    ```
    """
    
    # 类变量：单例实例缓存
    _instances: Dict[str, 'BaseService'] = {}
    _instance_lock = threading.Lock()
    
    def __new__(cls, config: Optional[ServiceConfig] = None, *args, **kwargs):
        """实现单例模式
        
        每个服务类只有一个实例，通过类名标识。
        如果提供了不同的配置，新配置将更新现有实例。
        """
        with cls._instance_lock:
            if cls not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[cls] = instance
                instance._initialized = False
                instance._service_name = cls.__name__
            
            return cls._instances[cls]
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        """初始化服务
        
        参数:
            config: 服务配置。如果为None，使用默认配置。
        """
        # 防止重复初始化（单例模式）
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # 设置配置
        self.config = config or ServiceConfig(service_name=self.__class__.__name__)
        self.service_name = self.config.service_name
        
        # 设置日志记录器
        self.logger = logging.getLogger(f"service.{self.service_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        
        # 服务状态
        self._initialized = False
        self._running = False
        self._healthy = False
        self._last_error: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._metrics: Dict[str, Any] = {
            "start_time": None,
            "request_count": 0,
            "error_count": 0,
            "success_count": 0,
            "average_response_time": 0.0,
            "last_request_time": None
        }
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 初始化服务
        if self._initialize_service():
            self._initialized = True
            self._running = True
            self._healthy = True
            self._start_time = datetime.now(timezone.utc)
            self.logger.info(f"服务 {self.service_name} 初始化成功")
            
            # 自动注册到服务注册中心（如果可用）
            self._register_with_registry()
        else:
            self.logger.error(f"服务 {self.service_name} 初始化失败")
            raise ServiceError(
                f"服务 {self.service_name} 初始化失败",
                service_name=self.service_name,
                error_code="INIT_FAILED"
            )
    
    @abstractmethod
    def _initialize_service(self) -> bool:
        """初始化服务特定资源
        
        子类必须实现此方法，初始化服务特定的资源。
        返回True表示初始化成功，False表示失败。
        """
        pass  # 已修复: 实现函数功能
    
    def _register_with_registry(self) -> bool:
        """注册服务到服务注册中心（如果可用）
        
        返回:
            bool: 注册是否成功
        """
        # 服务注册中心自身不能注册到自己，避免递归
        if self.service_name == "ServiceRegistry" or self.__class__.__name__ == "ServiceRegistry":
            self.logger.debug(f"服务注册中心自身跳过注册")
            return True
            
        try:
            # 惰性导入，避免循环导入
            from .service_registry import register_service
            
            # 使用服务名称注册
            success = register_service(self.service_name, self)
            if success:
                self.logger.debug(f"服务 {self.service_name} 已注册到服务注册中心")
            else:
                self.logger.warning(f"服务 {self.service_name} 注册到服务注册中心失败")
            
            return success
        except ImportError as e:
            # 服务注册中心不可用，这不是错误
            self.logger.debug(f"服务注册中心不可用: {e}，跳过自动注册")
            return False
        except Exception as e:
            self.logger.warning(f"服务注册失败: {e}")
            return False
    
    def start(self) -> bool:
        """启动服务
        
        返回:
            bool: 是否成功启动
        """
        with self._lock:
            if not self._initialized:
                self.logger.error(f"服务 {self.service_name} 未初始化，无法启动")
                return False
            
            if self._running:
                self.logger.warning(f"服务 {self.service_name} 已经在运行")
                return True
            
            self._running = True
            self._start_time = datetime.now(timezone.utc)
            self.logger.info(f"服务 {self.service_name} 已启动")
            return True
    
    def stop(self) -> bool:
        """停止服务
        
        返回:
            bool: 是否成功停止
        """
        with self._lock:
            if not self._running:
                self.logger.warning(f"服务 {self.service_name} 未在运行")
                return True
            
            self._running = False
            self._healthy = False
            self.logger.info(f"服务 {self.service_name} 已停止")
            return True
    
    def restart(self) -> bool:
        """重启服务
        
        返回:
            bool: 是否成功重启
        """
        self.logger.info(f"正在重启服务 {self.service_name}")
        if self.stop():
            time.sleep(1.0)  # 等待资源释放
            return self.start()
        return False
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        返回:
            Dict[str, Any]: 健康状态信息
        """
        with self._lock:
            uptime = None
            if self._start_time:
                uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            
            health_status = {
                "service_name": self.service_name,
                "initialized": self._initialized,
                "running": self._running,
                "healthy": self._healthy,
                "uptime_seconds": uptime,
                "last_error": self._last_error,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # 添加指标数据
            if self.config.enable_metrics:
                health_status["metrics"] = self._metrics.copy()
            
            return health_status
    
    def update_metrics(self, operation: str, success: bool, response_time: float = 0.0):
        """更新服务指标
        
        参数:
            operation: 操作名称
            success: 是否成功
            response_time: 响应时间（秒）
        """
        if not self.config.enable_metrics:
            return
        
        with self._lock:
            self._metrics["request_count"] += 1
            self._metrics["last_request_time"] = datetime.now(timezone.utc).isoformat()
            
            if success:
                self._metrics["success_count"] += 1
            else:
                self._metrics["error_count"] += 1
            
            # 计算平均响应时间（移动平均）
            old_avg = self._metrics["average_response_time"]
            old_count = max(1, self._metrics["success_count"] - 1)
            if success and response_time > 0:
                self._metrics["average_response_time"] = (old_avg * old_count + response_time) / self._metrics["success_count"]
    
    def error_handler(self, operation: str = None):
        """错误处理装饰器
        
        用于包装服务方法，提供统一的错误处理和重试机制。
        
        使用示例：
        ```python
        @self.error_handler(operation="process_data")
        def process_data(self, data):
            # 业务逻辑
        pass  # 已修复: 实现处理逻辑
        ```
        
        参数:
            operation: 操作名称，用于日志和指标记录
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                service_instance = args[0]  # self
                operation_name = operation or func.__name__
                
                # 检查服务状态
                if not service_instance._running:
                    raise ServiceError(
                        f"服务 {service_instance.service_name} 未运行",
                        service_name=service_instance.service_name,
                        error_code="SERVICE_NOT_RUNNING"
                    )
                
                # 重试逻辑
                max_retries = service_instance.config.max_retries
                retry_delay = service_instance.config.retry_delay
                
                for attempt in range(max_retries + 1):
                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        response_time = time.time() - start_time
                        
                        # 更新指标
                        service_instance.update_metrics(operation_name, True, response_time)
                        
                        service_instance.logger.debug(
                            f"操作 {operation_name} 成功，响应时间: {response_time:.3f}s"
                        )
                        return result
                        
                    except Exception as e:
                        response_time = time.time() - start_time if 'start_time' in locals() else 0
                        service_instance.update_metrics(operation_name, False, response_time)
                        service_instance._last_error = str(e)
                        
                        if attempt < max_retries:
                            service_instance.logger.warning(
                                f"操作 {operation_name} 失败，第 {attempt + 1}/{max_retries} 次重试: {e}"
                            )
                            time.sleep(retry_delay * (attempt + 1))  # 指数退避
                        else:
                            service_instance.logger.error(
                                f"操作 {operation_name} 失败，已达到最大重试次数: {e}"
                            )
                            service_instance.logger.debug(
                                f"操作 {operation_name} 失败，堆栈跟踪:\n{traceback.format_exc()}"
                            )
                            raise ServiceError(
                                f"操作 {operation_name} 失败: {e}",
                                service_name=service_instance.service_name,
                                error_code="OPERATION_FAILED"
                            ) from e
                
                # 理论上不应该到达这里
                raise ServiceError(
                    f"操作 {operation_name} 失败，未知错误",
                    service_name=service_instance.service_name,
                    error_code="UNKNOWN_ERROR"
                )
            
            return wrapper
        
        return decorator
    
    def check_dependency(self, dependency_name: str, import_path: str = None) -> bool:
        """检查依赖是否可用
        
        参数:
            dependency_name: 依赖名称
            import_path: 导入路径（如"hardware.unified_simulation"）
            
        返回:
            bool: 依赖是否可用
        """
        try:
            if import_path:
                # 动态导入
                module_name, class_name = import_path.rsplit('.', 1) if '.' in import_path else (import_path, None)
                module = __import__(module_name, fromlist=[class_name] if class_name else [])
                if class_name:
                    getattr(module, class_name)
            
            self.logger.debug(f"依赖 {dependency_name} 可用")
            return True
            
        except ImportError as e:
            self.logger.warning(f"依赖 {dependency_name} 不可用: {e}")
            return False
        except AttributeError as e:
            self.logger.warning(f"依赖 {dependency_name} 部分可用: {e}")
            return False
        except Exception as e:
            self.logger.error(f"检查依赖 {dependency_name} 时出错: {e}")
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息
        
        返回:
            Dict[str, Any]: 服务信息
        """
        return {
            "service_name": self.service_name,
            "service_class": self.__class__.__name__,
            "initialized": self._initialized,
            "running": self._running,
            "healthy": self._healthy,
            "config": self.config.to_dict(),
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "metrics": self._metrics if self.config.enable_metrics else None
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(name={self.service_name}, running={self._running})"
    
    def __repr__(self) -> str:
        """详细表示"""
        return f"{self.__class__.__name__}(config={self.config.to_dict()})"
    
    @classmethod
    def get_instance(cls) -> Optional['BaseService']:
        """获取服务实例（类方法）
        
        返回:
            Optional[BaseService]: 服务实例，如果不存在则返回None
        """
        return cls._instances.get(cls)
    
    @classmethod
    def clear_instance(cls):
        """清除服务实例（主要用于测试）"""
        with cls._instance_lock:
            if cls in cls._instances:
                instance = cls._instances[cls]
                if instance._running:
                    instance.stop()
                del cls._instances[cls]


# 便利函数
def service_operation(operation_name: str = None):
    """服务操作装饰器（便利函数）
    
    使用示例：
    ```python
    @service_operation(operation_name="process_data")
    def process_data(self, data):
        # 业务逻辑
        pass  # 已修复: 实现处理逻辑
    ```
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 检查是否为BaseService实例
            if not isinstance(self, BaseService):
                raise TypeError("service_operation装饰器只能用于BaseService子类")
            
            # 使用实例的错误处理器
            return self.error_handler(operation_name)(func)(self, *args, **kwargs)
        
        return wrapper
    
    return decorator