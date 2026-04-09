#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实硬件接口基类

定义真实硬件接口的基本结构和行为
核心原则：禁止模拟，强制真实（根据项目要求"禁止使用虚拟数据"）
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """硬件类型枚举"""

    MOTOR = "motor"
    SENSOR = "sensor"
    ROBOT = "robot"
    ACTUATOR = "actuator"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    SPEAKER = "speaker"
    DISPLAY = "display"
    INPUT_DEVICE = "input_device"
    NETWORK = "network"
    STORAGE = "storage"
    UNKNOWN = "unknown"


class ConnectionStatus(Enum):
    """连接状态枚举"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class HardwareError(Exception):
    """硬件错误异常"""

    def __init__(self, message: str = "HardwareError发生错误"):
        super().__init__(message)


class ConnectionError(HardwareError):
    """连接错误异常"""

    def __init__(self, message: str = "ConnectionError发生错误"):
        super().__init__(message)


class OperationError(HardwareError):
    """操作错误异常"""

    def __init__(self, message: str = "OperationError发生错误"):
        super().__init__(message)


class RealHardwareInterface(ABC):
    """真实硬件接口基类

    所有真实硬件接口的基类，定义标准接口和行为
    """

    def __init__(self, hardware_type: HardwareType, interface_name: str):
        """
        初始化真实硬件接口

        参数:
            hardware_type: 硬件类型
            interface_name: 接口名称
        """
        self.hardware_type = hardware_type
        self.interface_name = interface_name
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.last_error = None
        self.connection_timeout = 10.0  # 连接超时时间（秒）
        self.operation_timeout = 5.0  # 操作超时时间（秒）
        self.auto_reconnect = True  # 自动重连
        self.reconnect_interval = 2.0  # 重连间隔（秒）
        self.max_reconnect_attempts = 5  # 最大重连尝试次数

        # 性能监控
        self.operation_count = 0
        self.error_count = 0
        self.total_operation_time = 0.0
        self.last_operation_time = None

        # 连接锁
        self._connection_lock = threading.RLock()
        self._reconnect_thread = None
        self._stop_event = threading.Event()

        logger.info(f"初始化真实硬件接口: {interface_name} ({hardware_type.value})")

    @abstractmethod
    def connect(self) -> bool:
        """
        连接到真实硬件

        返回:
            连接是否成功

        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            f"真实硬件接口'{self.interface_name}'必须实现connect()方法。"
            "根据项目要求'禁止使用虚拟数据'，不能提供默认模拟实现。"
        )

    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开与真实硬件的连接

        返回:
            断开是否成功

        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            f"真实硬件接口'{self.interface_name}'必须实现disconnect()方法。"
            "根据项目要求'禁止使用虚拟数据'，不能提供默认模拟实现。"
        )

    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查是否已连接到真实硬件

        返回:
            是否已连接

        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            f"真实硬件接口'{self.interface_name}'必须实现is_connected()方法。"
            "根据项目要求'禁止使用虚拟数据'，不能提供默认模拟实现。"
        )

    @abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        获取硬件信息

        返回:
            硬件信息字典

        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            f"真实硬件接口'{self.interface_name}'必须实现get_hardware_info()方法。"
            "根据项目要求'禁止使用虚拟数据'，不能提供默认模拟实现。"
        )

    def execute_operation(self, operation: str, **kwargs) -> Any:
        """
        执行硬件操作

        根据用户要求"禁止使用虚拟数据"和"不采用任何降级处理，直接报错"，
        当没有具体硬件实现时，抛出HardwareError。

        参数:
            operation: 操作名称
            **kwargs: 操作参数

        返回:
            操作结果

        抛出:
            HardwareError: 硬件操作不可用
        """
        raise HardwareError(
            f"无法执行硬件操作（接口: {self.interface_name}, "
            f"硬件类型: {self.hardware_type.value}, 操作: {operation}）。"
            "根据项目要求'禁止使用虚拟数据'，硬件接口必须实现具体硬件操作。"
        )

    def safe_execute(self, operation: str, **kwargs) -> Any:
        """
        安全执行硬件操作（带错误处理和重试）

        参数:
            operation: 操作名称
            **kwargs: 操作参数

        返回:
            操作结果

        抛出:
            HardwareError: 硬件操作失败（包括重试后）
        """
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                result = self.execute_operation(operation, **kwargs)
                end_time = time.time()

                # 更新性能统计
                self.operation_count += 1
                self.total_operation_time += end_time - start_time
                self.last_operation_time = end_time

                return result

            except (HardwareError, ConnectionError) as e:
                self.error_count += 1
                self.last_error = str(e)

                if attempt < max_retries - 1:
                    logger.warning(
                        f"硬件操作失败，正在重试 ({attempt + 1}/{max_retries}): "
                        f"{operation}, 错误: {e}"
                    )
                    time.sleep(retry_delay)

                    # 如果是连接错误，尝试重新连接
                    if isinstance(e, ConnectionError) and self.auto_reconnect:
                        self._attempt_reconnect()
                else:
                    logger.error(
                        f"硬件操作失败，已达到最大重试次数: {operation}, " f"错误: {e}"
                    )
                    raise

    def _attempt_reconnect(self):
        """尝试重新连接"""
        if not self.auto_reconnect:
            return

        with self._connection_lock:
            if self._reconnect_thread and self._reconnect_thread.is_alive():
                return

            self._reconnect_thread = threading.Thread(
                target=self._reconnect_loop, daemon=True
            )
            self._reconnect_thread.start()

    def _reconnect_loop(self):
        """重连循环"""
        logger.info(f"开始重连循环: {self.interface_name}")

        attempts = 0
        while attempts < self.max_reconnect_attempts and not self._stop_event.is_set():

            try:
                logger.info(f"重连尝试 {attempts + 1}/{self.max_reconnect_attempts}")

                # 先断开（如果已连接）
                if self.is_connected():
                    self.disconnect()

                # 尝试连接
                if self.connect():
                    logger.info(f"重连成功: {self.interface_name}")
                    return

            except Exception as e:
                logger.warning(f"重连尝试失败: {e}")

            attempts += 1
            if attempts < self.max_reconnect_attempts:
                time.sleep(self.reconnect_interval)

        logger.error(f"重连失败，已达到最大尝试次数: {self.interface_name}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        avg_time = (
            self.total_operation_time / self.operation_count
            if self.operation_count > 0
            else 0.0
        )

        error_rate = (
            self.error_count / self.operation_count if self.operation_count > 0 else 0.0
        )

        return {
            "interface_name": self.interface_name,
            "hardware_type": self.hardware_type.value,
            "connection_status": self.connection_status.value,
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "average_operation_time": avg_time,
            "last_error": self.last_error,
            "last_operation_time": self.last_operation_time,
        }

    def health_check(self) -> Dict[str, Any]:
        """
        硬件健康检查

        返回:
            健康状态信息
        """
        try:
            is_connected = self.is_connected()
            info = self.get_hardware_info()
            stats = self.get_performance_stats()

            return {
                "healthy": is_connected and self.error_count < 10,
                "connected": is_connected,
                "hardware_info": info,
                "performance_stats": stats,
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "healthy": False,
                "connected": False,
                "error": str(e),
                "timestamp": time.time(),
            }

    def cleanup(self):
        """清理资源"""
        self._stop_event.set()

        if self._reconnect_thread and self._reconnect_thread.is_alive():
            self._reconnect_thread.join(timeout=2.0)

        try:
            if self.is_connected():
                self.disconnect()
        except Exception as e:
            logger.warning(f"断开连接时出错: {e}")

        logger.info(f"硬件接口清理完成: {self.interface_name}")

    def __enter__(self):
        """上下文管理器入口"""
        if not self.is_connected():
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()

    def __del__(self):
        """析构函数"""
        self.cleanup()
