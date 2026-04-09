"""
训练系统兼容性导入
解决Trainer vs AGITrainer命名问题
使用存根函数和延迟导入来避免循环导入问题
"""

from .trainer import TrainingConfig
from .trainer import AGITrainer
from .trainer import AGITrainer as Trainer
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 基础导入 - 这些应该不会导致循环导入

# 模块可用性标志
_REINFORCEMENT_LEARNING_AVAILABLE = False
_META_LEARNING_AVAILABLE = False

# 缓存实际导入的函数和类
_import_cache = {}

__all__ = [
    "Trainer",
    "AGITrainer",
    "TrainingConfig",
    "get_global_rl_manager",
    "get_reinforcement_learning_trainer",
    "get_meta_learning_trainer",
    "RLTrainingConfig",
    "RLTrainingManager",
]


# 存根函数 - 这些函数在导入时就已经存在，但会延迟加载实际实现
def get_reinforcement_learning_trainer(config: Optional[Dict[str, Any]] = None):
    """强化学习训练器（延迟导入实现）"""
    global _REINFORCEMENT_LEARNING_AVAILABLE

    if "get_reinforcement_learning_trainer" not in _import_cache:
        try:
            from .reinforcement_learning import (
                get_reinforcement_learning_trainer as func,
            )

            _import_cache["get_reinforcement_learning_trainer"] = func
            _REINFORCEMENT_LEARNING_AVAILABLE = True
        except ImportError as e:
            logger.warning(f"强化学习模块导入失败: {e}")
            raise

    func = _import_cache["get_reinforcement_learning_trainer"]
    return func(config)


def get_meta_learning_trainer(config: Optional[Dict[str, Any]] = None):
    """元学习训练器（延迟导入实现）"""
    global _META_LEARNING_AVAILABLE

    if "get_meta_learning_trainer" not in _import_cache:
        try:
            from .meta_learning import MetaLearningManager

            # 创建一个包装函数来匹配预期的接口
            def wrapper(cfg: Optional[Dict[str, Any]] = None):
                return MetaLearningManager(config=cfg)

            _import_cache["get_meta_learning_trainer"] = wrapper
            _META_LEARNING_AVAILABLE = True
        except ImportError as e:
            logger.warning(f"元学习模块导入失败: {e}")
            raise

    func = _import_cache["get_meta_learning_trainer"]
    return func(config)


def get_global_rl_manager():
    """全局强化学习管理器（延迟导入实现）"""
    global _REINFORCEMENT_LEARNING_AVAILABLE

    if "get_global_rl_manager" not in _import_cache:
        try:
            from .reinforcement_learning import get_global_rl_manager as func

            _import_cache["get_global_rl_manager"] = func
            _REINFORCEMENT_LEARNING_AVAILABLE = True
        except ImportError as e:
            logger.warning(f"强化学习模块导入失败: {e}")
            raise

    func = _import_cache["get_global_rl_manager"]
    return func()


# 类和配置的存根 - 在首次访问时延迟导入
class _RLTrainingConfigStub:
    """RLTrainingConfig存根类（延迟导入实际类）"""

    def __init__(self):
        if "RLTrainingConfig" not in _import_cache:
            try:
                from .reinforcement_learning import RLTrainingConfig as cls

                _import_cache["RLTrainingConfig"] = cls
            except ImportError as e:
                logger.warning(f"强化学习模块导入失败: {e}")
                raise
        self._cls = _import_cache["RLTrainingConfig"]

    def __call__(self, *args, **kwargs):
        # 允许像类一样调用
        return self._cls(*args, **kwargs)


class _RLTrainingManagerStub:
    """RLTrainingManager存根类（延迟导入实际类）"""

    def __init__(self):
        if "RLTrainingManager" not in _import_cache:
            try:
                from .reinforcement_learning import RLTrainingManager as cls

                _import_cache["RLTrainingManager"] = cls
            except ImportError as e:
                logger.warning(f"强化学习模块导入失败: {e}")
                raise
        self._cls = _import_cache["RLTrainingManager"]

    def __call__(self, *args, **kwargs):
        # 允许像类一样调用
        return self._cls(*args, **kwargs)


# 创建存根实例
RLTrainingConfig = _RLTrainingConfigStub()
RLTrainingManager = _RLTrainingManagerStub()
