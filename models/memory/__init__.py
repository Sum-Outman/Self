"""
记忆管理系统兼容性导入
解决MemoryManager vs MemorySystem命名问题
"""

from .memory_manager import MemorySystem as MemoryManager
from .memory_manager import MemorySystem

__all__ = ["MemoryManager", "MemorySystem"]