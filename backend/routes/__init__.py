"""
Self AGI 路由模块
包含所有API路由定义
"""

from . import auth_routes
from . import keys_routes
from . import knowledge_routes
from . import chat_routes
from . import training_routes
from . import hardware_routes
from . import programming_routes
from . import database_routes
from . import robot_routes
from . import autonomous_routes

# 以下模块将在后续创建
# from . import system_routes

__all__ = [
    "auth_routes",
    "keys_routes",
    "knowledge_routes",
    "chat_routes",
    "training_routes",
    "hardware_routes",
    "programming_routes",
    "database_routes",
    "robot_routes",
    "autonomous_routes",
    # "system_routes",
]
