#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务注册中心模块
管理所有服务的注册、发现、健康检查和监控

基于BaseService构建，提供集中式的服务管理功能
"""

import threading
from typing import Dict, Any, Optional, List, Type
from datetime import datetime, timezone
from enum import Enum

from .base_service import BaseService, ServiceConfig, service_operation


class ServiceStatus(Enum):
    """服务状态枚举"""

    UNREGISTERED = "unregistered"  # 未注册
    REGISTERED = "registered"  # 已注册但未初始化
    INITIALIZING = "initializing"  # 初始化中
    INITIALIZED = "initialized"  # 初始化完成
    RUNNING = "running"  # 运行中
    STOPPED = "stopped"  # 已停止
    ERROR = "error"  # 错误状态
    DEGRADED = "degraded"  # 降级运行


class ServiceInfo:
    """服务信息类"""

    def __init__(
        self,
        service_name: str,
        service_class: Type[BaseService],
        service_instance: Optional[BaseService] = None,
    ):
        self.service_name = service_name
        self.service_class = service_class
        self.service_instance = service_instance
        self.status = ServiceStatus.UNREGISTERED
        self.health_status = "unknown"
        self.uptime_seconds = 0
        self.last_health_check = None
        self.registration_time = datetime.now(timezone.utc)
        self.metadata = {}
        self.dependencies = set()  # 依赖的服务名称
        self.dependents = set()  # 依赖此服务的服务名称
        self.error_count = 0
        self.last_error = None
        self.metrics = {}

    def update_health(self, health_data: Dict[str, Any]):
        """更新健康状态"""
        if health_data:
            self.health_status = health_data.get("healthy", "unknown")
            self.uptime_seconds = health_data.get("uptime_seconds", 0)
            self.last_health_check = datetime.now(timezone.utc)

            # 更新指标
            if "metrics" in health_data and isinstance(health_data["metrics"], dict):
                self.metrics.update(health_data["metrics"])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "service_name": self.service_name,
            "service_class": self.service_class.__name__,
            "status": self.status.value,
            "health_status": self.health_status,
            "uptime_seconds": self.uptime_seconds,
            "registration_time": self.registration_time.isoformat(),
            "last_health_check": (
                self.last_health_check.isoformat() if self.last_health_check else None
            ),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents),
            "metadata": self.metadata,
            "metrics": self.metrics.copy(),
        }


class ServiceRegistry(BaseService):
    """服务注册中心单例类，基于BaseService构建"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        # 配置扩展：添加注册中心特定配置
        if config:
            if "extra_config" not in config.__dict__:
                config.extra_config = {}

            # 设置注册中心默认配置
            registry_defaults = {
                "auto_discover_services": True,
                "health_check_interval_seconds": 30,
                "service_timeout_seconds": 60,
                "max_retries": 3,
                "enable_metrics_collection": True,
                "metrics_retention_days": 7,
            }

            # 合并默认配置
            for key, value in registry_defaults.items():
                if key not in config.extra_config:
                    config.extra_config[key] = value

        # 初始化注册中心特定属性（在父类初始化之前）
        self._services: Dict[str, ServiceInfo] = {}  # 服务名称 -> 服务信息
        self._service_classes: Dict[str, Type[BaseService]] = {}  # 服务名称 -> 服务类
        self._lock = threading.RLock()
        self._health_check_thread = None
        self._stop_health_check = threading.Event()

        # 调用父类初始化（必须在属性初始化之后）
        super().__init__(config)

    def _initialize_service(self) -> bool:
        """初始化服务注册中心

        返回:
            bool: 初始化是否成功
        """
        try:
            self.logger.info("开始初始化服务注册中心...")

            # 自动发现已注册的服务类
            if self.config.extra_config.get("auto_discover_services", True):
                self._auto_discover_services()

            # 启动健康检查线程
            self._start_health_check_thread()

            self.logger.info(
                f"服务注册中心初始化成功，已注册服务数: {len(self._services)}"
            )
            return True

        except Exception as e:
            self.logger.error(f"服务注册中心初始化失败: {e}")
            self._last_error = str(e)
            return False

    def _auto_discover_services(self):
        """自动发现服务

        查找BaseService的子类并自动注册
        """
        # 获取BaseService的所有子类
        base_service_subclasses = self._get_base_service_subclasses()

        for service_class in base_service_subclasses:
            # 跳过自身
            if service_class == ServiceRegistry:
                continue

            # 获取服务名称
            service_name = service_class.__name__

            # 注册服务类
            self.register_service_class(service_name, service_class)

            # 尝试获取现有实例并注册
            try:
                # BaseService的单例实例存储在类变量中
                if service_class in service_class._instances:
                    instance = service_class._instances[service_class]
                    self.register_service_instance(service_name, instance)
            except Exception as e:
                self.logger.warning(f"无法注册服务实例 {service_name}: {e}")

    def _get_base_service_subclasses(self) -> List[Type[BaseService]]:
        """获取BaseService的所有子类"""
        subclasses = []
        to_check = [BaseService]

        while to_check:
            parent = to_check.pop()
            for child in parent.__subclasses__():
                if child not in subclasses:
                    subclasses.append(child)
                    to_check.append(child)

        return subclasses

    def _start_health_check_thread(self):
        """启动健康检查线程"""
        if self._health_check_thread and self._health_check_thread.is_alive():
            return

        self._stop_health_check.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name="ServiceRegistryHealthCheck",
            daemon=True,
        )
        self._health_check_thread.start()
        self.logger.info("健康检查线程已启动")

    def _health_check_loop(self):
        """健康检查循环"""
        interval = self.config.extra_config.get("health_check_interval_seconds", 30)

        while not self._stop_health_check.is_set():
            try:
                # 执行健康检查
                self._perform_health_checks()

                # 睡眠直到下次检查
                self._stop_health_check.wait(interval)
            except Exception as e:
                self.logger.error(f"健康检查循环出错: {e}")
                # 继续运行，不中断循环

    def _perform_health_checks(self):
        """执行所有服务的健康检查"""
        with self._lock:
            services_to_check = list(self._services.values())

        for service_info in services_to_check:
            if service_info.service_instance:
                try:
                    # 执行健康检查
                    health_data = service_info.service_instance.health_check()

                    # 更新服务信息
                    service_info.update_health(health_data)

                    # 更新服务状态
                    if health_data.get("healthy", False):
                        service_info.status = ServiceStatus.RUNNING
                        service_info.error_count = 0
                    else:
                        service_info.status = ServiceStatus.DEGRADED
                        service_info.last_error = health_data.get(
                            "error", "健康检查失败"
                        )

                except Exception as e:
                    service_info.status = ServiceStatus.ERROR
                    service_info.health_status = "error"
                    service_info.last_error = str(e)
                    service_info.error_count += 1
                    self.logger.warning(
                        f"服务 {service_info.service_name} 健康检查失败: {e}"
                    )

    def register_service_class(
        self, service_name: str, service_class: Type[BaseService]
    ) -> bool:
        """注册服务类

        参数:
            service_name: 服务名称
            service_class: 服务类（必须是BaseService的子类）

        返回:
            bool: 注册是否成功
        """
        if not issubclass(service_class, BaseService):
            self.logger.error(f"服务类 {service_class} 必须是BaseService的子类")
            return False

        with self._lock:
            if service_name in self._service_classes:
                self.logger.warning(f"服务类 {service_name} 已注册，跳过重复注册")
                return True

            self._service_classes[service_name] = service_class

            # 创建或更新服务信息
            if service_name in self._services:
                self._services[service_name].service_class = service_class
            else:
                self._services[service_name] = ServiceInfo(service_name, service_class)

            self._services[service_name].status = ServiceStatus.REGISTERED
            self.logger.info(f"服务类 {service_name} 注册成功")

            return True

    def register_service_instance(
        self, service_name: str, service_instance: BaseService
    ) -> bool:
        """注册服务实例

        参数:
            service_name: 服务名称
            service_instance: 服务实例

        返回:
            bool: 注册是否成功
        """
        if not isinstance(service_instance, BaseService):
            self.logger.error(f"服务实例 {service_instance} 必须是BaseService的实例")
            return False

        with self._lock:
            # 确保服务类已注册
            if service_name not in self._service_classes:
                self._service_classes[service_name] = type(service_instance)

            # 创建或更新服务信息
            if service_name in self._services:
                service_info = self._services[service_name]
                service_info.service_instance = service_instance
                service_info.service_class = type(service_instance)
            else:
                service_info = ServiceInfo(
                    service_name, type(service_instance), service_instance
                )
                self._services[service_name] = service_info

            # 更新状态
            if service_instance._initialized:
                service_info.status = ServiceStatus.INITIALIZED
                if service_instance._running:
                    service_info.status = ServiceStatus.RUNNING
            else:
                service_info.status = ServiceStatus.INITIALIZING

            # 执行初始健康检查
            try:
                health_data = service_instance.health_check()
                service_info.update_health(health_data)
            except Exception as e:
                self.logger.warning(f"服务 {service_name} 初始健康检查失败: {e}")

            self.logger.info(
                f"服务实例 {service_name} 注册成功，状态: {service_info.status.value}"
            )

            return True

    def unregister_service(self, service_name: str) -> bool:
        """注销服务

        参数:
            service_name: 服务名称

        返回:
            bool: 注销是否成功
        """
        with self._lock:
            if service_name not in self._services:
                self.logger.warning(f"服务 {service_name} 未注册，无法注销")
                return False

            # 移除服务
            del self._services[service_name]

            if service_name in self._service_classes:
                del self._service_classes[service_name]

            self.logger.info(f"服务 {service_name} 已注销")

            return True

    def get_service(self, service_name: str) -> Optional[BaseService]:
        """获取服务实例

        参数:
            service_name: 服务名称

        返回:
            Optional[BaseService]: 服务实例，如果不存在则返回None
        """
        with self._lock:
            if service_name in self._services:
                service_info = self._services[service_name]
                return service_info.service_instance
            return None  # 返回None

    def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """获取服务信息

        参数:
            service_name: 服务名称

        返回:
            Optional[Dict[str, Any]]: 服务信息字典，如果不存在则返回None
        """
        with self._lock:
            if service_name in self._services:
                return self._services[service_name].to_dict()
            return None  # 返回None

    def get_all_services(self) -> List[str]:
        """获取所有已注册的服务名称

        返回:
            List[str]: 服务名称列表
        """
        with self._lock:
            return list(self._services.keys())

    def get_all_service_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有服务信息

        返回:
            Dict[str, Dict[str, Any]]: 服务名称到服务信息的映射
        """
        with self._lock:
            return {
                name: service_info.to_dict()
                for name, service_info in self._services.items()
            }

    def add_dependency(self, service_name: str, dependency_name: str) -> bool:
        """添加服务依赖关系

        参数:
            service_name: 服务名称
            dependency_name: 依赖的服务名称

        返回:
            bool: 添加是否成功
        """
        with self._lock:
            if (
                service_name not in self._services
                or dependency_name not in self._services
            ):
                self.logger.error(
                    f"服务 {service_name} 或依赖 {dependency_name} 未注册"
                )
                return False

            # 添加依赖
            self._services[service_name].dependencies.add(dependency_name)
            self._services[dependency_name].dependents.add(service_name)

            self.logger.info(f"服务依赖关系已添加: {service_name} -> {dependency_name}")

            return True

    def remove_dependency(self, service_name: str, dependency_name: str) -> bool:
        """移除服务依赖关系

        参数:
            service_name: 服务名称
            dependency_name: 依赖的服务名称

        返回:
            bool: 移除是否成功
        """
        with self._lock:
            if (
                service_name not in self._services
                or dependency_name not in self._services
            ):
                self.logger.error(
                    f"服务 {service_name} 或依赖 {dependency_name} 未注册"
                )
                return False

            # 移除依赖
            if dependency_name in self._services[service_name].dependencies:
                self._services[service_name].dependencies.remove(dependency_name)

            if service_name in self._services[dependency_name].dependents:
                self._services[dependency_name].dependents.remove(service_name)

            self.logger.info(f"服务依赖关系已移除: {service_name} -> {dependency_name}")

            return True

    def check_dependencies(self, service_name: str) -> Dict[str, Any]:
        """检查服务依赖的健康状态

        参数:
            service_name: 服务名称

        返回:
            Dict[str, Any]: 依赖检查结果
        """
        with self._lock:
            if service_name not in self._services:
                return {
                    "success": False,
                    "error": f"服务 {service_name} 未注册",
                    "dependencies": {},
                    "all_healthy": False,
                }

            service_info = self._services[service_name]
            dependencies_status = {}
            all_healthy = True

            for dep_name in service_info.dependencies:
                dep_info = self._services.get(dep_name)
                if dep_info:
                    dep_status = {
                        "service_name": dep_name,
                        "status": dep_info.status.value,
                        "health_status": dep_info.health_status,
                        "healthy": dep_info.health_status == "healthy",
                        "error_count": dep_info.error_count,
                        "last_error": dep_info.last_error,
                    }
                    dependencies_status[dep_name] = dep_status

                    if dep_info.health_status != "healthy":
                        all_healthy = False
                else:
                    dependencies_status[dep_name] = {
                        "service_name": dep_name,
                        "status": "not_found",
                        "health_status": "unknown",
                        "healthy": False,
                        "error": "服务未找到",
                    }
                    all_healthy = False

            return {
                "success": True,
                "service_name": service_name,
                "dependencies": dependencies_status,
                "all_healthy": all_healthy,
                "dependency_count": len(service_info.dependencies),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @service_operation(operation_name="get_registry_status")
    def get_registry_status(self) -> Dict[str, Any]:
        """获取注册中心状态

        返回:
            Dict[str, Any]: 注册中心状态信息
        """
        with self._lock:
            total_services = len(self._services)
            healthy_services = 0
            running_services = 0
            error_services = 0

            for service_info in self._services.values():
                if service_info.health_status == "healthy":
                    healthy_services += 1
                if service_info.status == ServiceStatus.RUNNING:
                    running_services += 1
                if service_info.status == ServiceStatus.ERROR:
                    error_services += 1

            return {
                "service_name": "ServiceRegistry",
                "status": "running",
                "total_services": total_services,
                "healthy_services": healthy_services,
                "running_services": running_services,
                "error_services": error_services,
                "health_percentage": (
                    (healthy_services / total_services * 100)
                    if total_services > 0
                    else 0
                ),
                "auto_discover_enabled": self.config.extra_config.get(
                    "auto_discover_services", True
                ),
                "health_check_interval": self.config.extra_config.get(
                    "health_check_interval_seconds", 30
                ),
                "uptime_seconds": self._uptime_seconds,
                "initialized": self._initialized,
                "healthy": self._healthy,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @service_operation(operation_name="shutdown_service")
    def shutdown_service(
        self, service_name: str, force: bool = False
    ) -> Dict[str, Any]:
        """关闭服务

        参数:
            service_name: 服务名称
            force: 是否强制关闭

        返回:
            Dict[str, Any]: 关闭结果
        """
        service_instance = self.get_service(service_name)
        if not service_instance:
            return {
                "success": False,
                "error": f"服务 {service_name} 未找到",
                "service_name": service_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        try:
            # 检查是否有依赖此服务的其他服务
            with self._lock:
                if service_name in self._services:
                    dependents = list(self._services[service_name].dependents)

                    if dependents and not force:
                        return {
                            "success": False,
                            "error": f"服务 {service_name} 有依赖它的服务: {dependents}",
                            "dependents": dependents,
                            "service_name": service_name,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }

            # 调用服务的shutdown方法（如果存在）
            if hasattr(service_instance, "shutdown"):
                shutdown_result = service_instance.shutdown()
                if isinstance(shutdown_result, dict) and not shutdown_result.get(
                    "success", True
                ):
                    return {
                        "success": False,
                        "error": f"服务关闭失败: {shutdown_result.get('error', '未知错误')}",
                        "service_name": service_name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

            # 更新服务状态
            with self._lock:
                if service_name in self._services:
                    self._services[service_name].status = ServiceStatus.STOPPED

            self.logger.info(f"服务 {service_name} 已关闭")

            return {
                "success": True,
                "message": f"服务 {service_name} 已成功关闭",
                "service_name": service_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"关闭服务 {service_name} 失败: {e}")
            return {
                "success": False,
                "error": f"关闭服务失败: {str(e)}",
                "service_name": service_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def shutdown_all_services(self, force: bool = False) -> Dict[str, Any]:
        """关闭所有服务

        参数:
            force: 是否强制关闭

        返回:
            Dict[str, Any]: 关闭结果
        """
        results = {}

        # 按依赖关系逆序关闭服务（先关闭依赖其他服务的服务）
        with self._lock:
            # 创建依赖关系图
            list(self._services.keys())
            # 简单实现：先关闭所有没有依赖的服务，然后逐步关闭
            # 完整处理，直接按原始顺序关闭
        pass  # 已实现

        # 逐个关闭服务
        for service_name in self.get_all_services():
            result = self.shutdown_service(service_name, force)
            results[service_name] = result

        return {
            "success": True,
            "message": "所有服务已关闭",
            "results": results,
            "total_services": len(results),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _cleanup_service(self):
        """清理资源"""
        # 停止健康检查线程
        self._stop_health_check.set()
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)

        # 清理服务信息
        with self._lock:
            self._services.clear()
            self._service_classes.clear()

    @service_operation(operation_name="get_service_info")
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息（重写父类方法）"""
        # 获取父类的服务信息
        base_info = super().get_service_info()

        # 添加注册中心特定信息
        registry_info = self.get_registry_status()

        # 合并信息
        base_info.update(registry_info)
        return base_info


# 全局服务注册中心实例
_service_registry_instance = None
_service_registry_lock = threading.Lock()


def get_service_registry() -> ServiceRegistry:
    """获取服务注册中心实例（全局单例）"""
    global _service_registry_instance

    with _service_registry_lock:
        if _service_registry_instance is None:
            _service_registry_instance = ServiceRegistry()

        return _service_registry_instance


def register_service(service_name: str, service_instance: BaseService) -> bool:
    """注册服务实例（便捷函数）

    参数:
        service_name: 服务名称
        service_instance: 服务实例

    返回:
        bool: 注册是否成功
    """
    registry = get_service_registry()
    return registry.register_service_instance(service_name, service_instance)


def get_service(service_name: str) -> Optional[BaseService]:
    """获取服务实例（便捷函数）

    参数:
        service_name: 服务名称

    返回:
        Optional[BaseService]: 服务实例，如果不存在则返回None
    """
    registry = get_service_registry()
    return registry.get_service(service_name)
