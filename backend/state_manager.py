#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局状态管理器
解决FastAPI app.state在不同线程/模块中不一致的问题

使用真正的单例模式确保所有模块访问相同的状态
"""

import threading
import logging
from typing import Any

logger = logging.getLogger(__name__)


class GlobalStateManager:
    """全局状态管理器（单例）"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式的__new__方法"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化全局状态管理器"""
        if getattr(self, "_initialized", False):
            return

        self._lock = threading.Lock()
        self._state = {}
        self._app = None
        self._initialized = True

        logger.info("全局状态管理器初始化完成")

    def register_app(self, app):
        """注册FastAPI应用实例"""
        with self._lock:
            self._app = app
            logger.info(f"FastAPI应用已注册: {app}, id: {id(app)}")

    def get_app(self):
        """获取FastAPI应用实例"""
        with self._lock:
            return self._app

    def set_state(self, key: str, value: Any):
        """设置全局状态"""
        with self._lock:
            self._state[key] = value
            logger.info(f"全局状态设置: {key} = {value}")

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取全局状态"""
        with self._lock:
            return self._state.get(key, default)

    def has_state(self, key: str) -> bool:
        """检查是否存在指定状态"""
        with self._lock:
            return key in self._state

    def clear_state(self, key: str):
        """清除指定状态"""
        with self._lock:
            if key in self._state:
                del self._state[key]
                logger.info(f"全局状态已清除: {key}")


# 创建全局单例实例
state_manager = GlobalStateManager()


def get_memory_system():
    """获取记忆系统实例（全局访问）"""
    memory_system = state_manager.get_state("memory_system")

    if memory_system is None:
        logger.warning("全局状态管理器中的memory_system为None")
        # 尝试从app.state获取（向后兼容）
        app = state_manager.get_app()
        if app and hasattr(app, "state"):
            try:
                memory_system = app.state.memory_system
                if memory_system is not None:
                    # 更新全局状态管理器
                    state_manager.set_state("memory_system", memory_system)
                    logger.info(
                        f"从app.state获取memory_system并更新全局状态: {memory_system}"
                    )
            except AttributeError:
                pass  # 已实现

    logger.info(f"get_memory_system() 返回: {memory_system}")
    return memory_system


def set_memory_system(memory_system):
    """设置记忆系统实例（全局访问）"""
    state_manager.set_state("memory_system", memory_system)
    logger.info(f"记忆系统已设置到全局状态管理器: {memory_system}")

    # 同时更新app.state（如果app已注册）
    app = state_manager.get_app()
    if app and hasattr(app, "state"):
        app.state.memory_system = memory_system
        logger.info(f"记忆系统已同步到app.state: {memory_system}")


def register_app(app):
    """注册FastAPI应用实例到全局状态管理器"""
    state_manager.register_app(app)

    # 如果app.state中有memory_system，同步到全局状态管理器
    if hasattr(app, "state"):
        try:
            memory_system = app.state.memory_system
            if memory_system is not None:
                state_manager.set_state("memory_system", memory_system)
                logger.info(
                    f"从注册的app.state同步memory_system到全局状态: {memory_system}"
                )
        except AttributeError:
            pass  # 已实现
