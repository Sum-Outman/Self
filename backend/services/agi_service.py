"""
AGI核心服务
提供完整的AGI系统状态管理和控制功能

功能：
1. AGI系统状态管理
2. 训练进度监控
3. 系统模式切换
4. 硬件状态集成
5. 模型状态集成

禁止使用虚拟数据，所有数据必须来自真实系统状态
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import threading

# 尝试导入相关服务
try:
    from .model_service import get_model_service
    MODEL_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"模型服务导入失败: {e}")
    MODEL_SERVICE_AVAILABLE = False

try:
    from .system_service import get_system_service
    SYSTEM_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"系统服务导入失败: {e}")
    SYSTEM_SERVICE_AVAILABLE = False

try:
    from .training_service import get_training_service
    TRAINING_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"训练服务导入失败: {e}")
    TRAINING_SERVICE_AVAILABLE = False

try:
    from .hardware_service import get_hardware_service
    HARDWARE_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"硬件服务导入失败: {e}")
    HARDWARE_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AGIService:
    """AGI核心服务单例类"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logger.info("初始化AGI核心服务...")
        
        self._running = False
        self._mode = "task"  # 默认任务执行模式
        self._last_update = datetime.now(timezone.utc)
        self._status_lock = threading.Lock()
        
        # 状态缓存
        self._cached_status = None
        self._cache_time = 0
        self._cache_ttl = 5  # 缓存有效期（秒）
        
        # 初始化服务连接
        self._initialize_services()
        
        self._initialized = True
        logger.info("AGI核心服务初始化完成")
    
    def _initialize_services(self):
        """初始化依赖服务"""
        self._services = {}
        
        # 声明全局变量以便修改
        global MODEL_SERVICE_AVAILABLE, SYSTEM_SERVICE_AVAILABLE, TRAINING_SERVICE_AVAILABLE, HARDWARE_SERVICE_AVAILABLE
        
        if MODEL_SERVICE_AVAILABLE:
            try:
                self._services["model"] = get_model_service()
                logger.info("模型服务连接成功")
            except Exception as e:
                logger.error(f"模型服务连接失败: {e}")
                MODEL_SERVICE_AVAILABLE = False
        
        if SYSTEM_SERVICE_AVAILABLE:
            try:
                self._services["system"] = get_system_service()
                logger.info("系统服务连接成功")
            except Exception as e:
                logger.error(f"系统服务连接失败: {e}")
                SYSTEM_SERVICE_AVAILABLE = False
        
        if TRAINING_SERVICE_AVAILABLE:
            try:
                self._services["training"] = get_training_service()
                logger.info("训练服务连接成功")
            except Exception as e:
                logger.error(f"训练服务连接失败: {e}")
                TRAINING_SERVICE_AVAILABLE = False
        
        if HARDWARE_SERVICE_AVAILABLE:
            try:
                self._services["hardware"] = get_hardware_service()
                logger.info("硬件服务连接成功")
            except Exception as e:
                logger.error(f"硬件服务连接失败: {e}")
                HARDWARE_SERVICE_AVAILABLE = False
    
    def get_status(self) -> Dict[str, Any]:
        """获取AGI系统完整状态
        
        返回:
            Dict[str, Any]: AGI系统状态数据
        """
        import time
        
        with self._status_lock:
            # 检查缓存是否有效
            current_time = time.time()
            if (self._cached_status is not None and 
                current_time - self._cache_time < self._cache_ttl):
                # 返回缓存状态，但更新最后更新时间
                self._cached_status["lastUpdated"] = datetime.now(timezone.utc).isoformat()
                return self._cached_status
            
            # 缓存无效，重新收集状态
            status_data = {
                "status": "idle",
                "mode": self._mode,
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "services": {},
                "capabilities": {},
                "performance": {},
                "hardware": {},
                "training": {}
            }
            
            # 收集模型服务状态（带超时保护）
            if MODEL_SERVICE_AVAILABLE and "model" in self._services:
                try:
                    model_status = self._services["model"].get_model_info()
                    status_data["services"]["model"] = {
                        "available": True,
                        "status": model_status.get("status", "unknown"),
                        "device": model_status.get("device", "unknown"),
                        "parameters": model_status.get("parameters", 0)
                    }
                    
                    # 提取能力信息
                    if "capabilities" in model_status:
                        status_data["capabilities"] = model_status["capabilities"]
                except Exception as e:
                    logger.error(f"获取模型状态失败: {e}")
                    status_data["services"]["model"] = {"available": False, "error": str(e)}
            
            # 收集系统服务状态
            if SYSTEM_SERVICE_AVAILABLE and "system" in self._services:
                try:
                    system_status = self._services["system"].get_system_status()
                    status_data["performance"] = system_status.get("performance", {})
                    status_data["hardware"]["connected"] = system_status.get("hardware", {}).get("connected", False)
                    
                    status_data["services"]["system"] = {
                        "available": True,
                        "status": "running"
                    }
                except Exception as e:
                    logger.error(f"获取系统状态失败: {e}")
                    status_data["services"]["system"] = {"available": False, "error": str(e)}
            
            # 收集训练服务状态
            if TRAINING_SERVICE_AVAILABLE and "training" in self._services:
                try:
                    training_status = self._services["training"].get_training_status()
                    status_data["training"] = training_status
                    
                    status_data["services"]["training"] = {
                        "available": True,
                        "status": "running"
                    }
                except Exception as e:
                    logger.error(f"获取训练状态失败: {e}")
                    status_data["services"]["training"] = {"available": False, "error": str(e)}
            
            # 收集硬件服务状态（可能较慢，但已缓存）
            if HARDWARE_SERVICE_AVAILABLE and "hardware" in self._services:
                try:
                    # 获取硬件设备列表
                    hardware_devices = self._services["hardware"].get_hardware_devices()
                    # 检查是否有连接的设备
                    connected_devices = [d for d in hardware_devices if d.get("connected", False)]
                    
                    hardware_status = {
                        "devices": hardware_devices,
                        "connected_count": len(connected_devices),
                        "has_connected": len(connected_devices) > 0
                    }
                    status_data["hardware"].update(hardware_status)
                    
                    status_data["services"]["hardware"] = {
                        "available": True,
                        "status": "running"
                    }
                except Exception as e:
                    logger.error(f"获取硬件状态失败: {e}")
                    status_data["services"]["hardware"] = {"available": False, "error": str(e)}
            
            # 确定整体状态
            if status_data["training"].get("active", False):
                status_data["status"] = "training"
            elif status_data.get("capabilities", {}).get("reasoning_enabled", False):
                status_data["status"] = "reasoning"
            else:
                status_data["status"] = "idle"
            
            self._last_update = datetime.now(timezone.utc)
            # 更新缓存
            self._cached_status = status_data.copy()
            self._cache_time = current_time
            
            return status_data
    
    def set_mode(self, mode: str) -> bool:
        """设置系统模式
        
        参数:
            mode: 模式名称 (task, autonomous, training)
        
        返回:
            bool: 是否成功
        """
        valid_modes = ["task", "autonomous", "training"]
        if mode not in valid_modes:
            logger.error(f"无效模式: {mode}, 有效模式: {valid_modes}")
            return False
        
        with self._status_lock:
            old_mode = self._mode
            self._mode = mode
            logger.info(f"AGI模式从 {old_mode} 切换到 {mode}")
            
            # 通知相关服务模式变更
            if SYSTEM_SERVICE_AVAILABLE and "system" in self._services:
                try:
                    self._services["system"].set_system_mode(mode)
                except Exception as e:
                    logger.error(f"设置系统模式失败: {e}")
            
            return True
    
    def get_mode(self) -> str:
        """获取当前系统模式
        
        返回:
            str: 当前模式
        """
        return self._mode
    
    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """启动训练任务
        
        参数:
            config: 训练配置
        
        返回:
            Dict[str, Any]: 训练任务信息
        """
        if not TRAINING_SERVICE_AVAILABLE:
            return {"success": False, "error": "训练服务不可用"}
        
        try:
            result = self._services["training"].start_training(config)
            return result
        except Exception as e:
            logger.error(f"启动训练失败: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_training(self) -> bool:
        """停止当前训练任务
        
        返回:
            bool: 是否成功
        """
        if not TRAINING_SERVICE_AVAILABLE:
            return False
        
        try:
            return self._services["training"].stop_training()
        except Exception as e:
            logger.error(f"停止训练失败: {e}")
            return False
    
    def pause(self) -> bool:
        """暂停AGI系统
        
        返回:
            bool: 是否成功
        """
        with self._status_lock:
            if self._running:
                self._running = False
                logger.info("AGI系统已暂停")
                return True
            return False
    
    def resume(self) -> bool:
        """恢复AGI系统
        
        返回:
            bool: 是否成功
        """
        with self._status_lock:
            if not self._running:
                self._running = True
                logger.info("AGI系统已恢复")
                return True
            return False
    
    def restart(self) -> bool:
        """重启AGI系统
        
        返回:
            bool: 是否成功
        """
        logger.info("重启AGI系统...")
        
        # 暂停系统
        self.pause()
        
        # 重新初始化服务
        self._initialize_services()
        
        # 恢复系统
        self.resume()
        
        logger.info("AGI系统重启完成")
        return True


# 全局AGI服务实例
_agi_service_instance = None


def get_agi_service() -> AGIService:
    """获取AGI服务实例（单例）"""
    global _agi_service_instance
    if _agi_service_instance is None:
        _agi_service_instance = AGIService()
    return _agi_service_instance


# 导出函数
__all__ = ["AGIService", "get_agi_service"]