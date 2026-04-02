"""
系统状态服务
提供真实的系统状态信息，替换真实数据

功能：
1. 系统模式管理（获取/设置）
2. 系统状态监控（性能指标、硬件状态、运行状态）
3. 系统健康检查
4. 系统能力报告

集成SystemMonitor获取真实系统性能数据
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import threading

# 尝试导入SystemMonitor
try:
    from models.system_control.system_monitor import SystemMonitor, SystemStatus, AlertLevel
    SYSTEM_MONITOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SystemMonitor导入失败: {e}")
    SYSTEM_MONITOR_AVAILABLE = False

# 尝试导入psutil进行系统监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError as e:
    logging.info(f"psutil不可用，部分系统监控功能受限: {e}")
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class SystemService:
    """系统状态服务单例类"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._current_mode = "assist"  # 默认助手模式
            self._mode_lock = threading.Lock()
            self._mode_history = []
            self._system_monitor = None
            
            # 系统能力定义
            self._capabilities = {
                "assist": [
                    "对话交流",
                    "知识查询", 
                    "任务执行",
                    "文件处理",
                    "硬件控制",
                    "多模态交互"
                ],
                "autonomous": [
                    "自主决策",
                    "任务规划",
                    "系统控制",
                    "数据收集与分析",
                    "自我优化",
                    "环境交互"
                ],
                "training": [
                    "模型训练",
                    "数据收集",
                    "超参数优化",
                    "性能评估",
                    "知识库更新"
                ],
                "maintenance": [
                    "系统诊断",
                    "硬件检测",
                    "数据备份",
                    "安全扫描",
                    "性能调优"
                ]
            }
            
            # 系统限制定义
            self._restrictions = {
                "assist": [
                    "需要用户确认敏感操作",
                    "不执行潜在危险操作",
                    "遵循用户指令",
                    "保护用户隐私"
                ],
                "autonomous": [
                    "遵循安全协议",
                    "不违反伦理规范",
                    "定期报告状态",
                    "接受人工干预"
                ],
                "training": [
                    "不干扰正常运行",
                    "资源使用限制",
                    "数据隐私保护",
                    "训练过程监控"
                ],
                "maintenance": [
                    "只允许授权访问",
                    "维护操作记录",
                    "系统状态备份",
                    "恢复计划准备"
                ]
            }
            
            # 模式描述
            self._mode_descriptions = {
                "assist": "助手模式 - AI协助用户完成任务",
                "autonomous": "自主模式 - AI自主决策和执行任务",
                "training": "训练模式 - 系统正在训练模型",
                "maintenance": "维护模式 - 系统维护中"
            }
            
            # 初始化SystemMonitor
            self._init_system_monitor()
            
            logger.info("系统状态服务初始化完成")
    
    def _init_system_monitor(self):
        """初始化系统监控器"""
        if SYSTEM_MONITOR_AVAILABLE:
            try:
                self._system_monitor = SystemMonitor()
                logger.info("SystemMonitor初始化成功")
            except Exception as e:
                logger.error(f"SystemMonitor初始化失败: {e}")
                self._system_monitor = None
        else:
            logger.warning("SystemMonitor不可用，使用基础监控功能")
            self._system_monitor = None
    
    def get_system_mode(self) -> Dict[str, Any]:
        """获取系统模式信息"""
        with self._mode_lock:
            mode = self._current_mode
            
            # 获取真实系统性能指标
            performance_metrics = self._get_real_performance_metrics()
            
            # 构建系统模式响应
            system_mode = {
                "mode": mode,
                "description": self._mode_descriptions.get(mode, "未知模式"),
                "capabilities": self._capabilities.get(mode, []),
                "restrictions": self._restrictions.get(mode, []),
                "performance_metrics": performance_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "monitor_available": self._system_monitor is not None
            }
            
            return system_mode
    
    def set_system_mode(self, mode: str) -> Dict[str, Any]:
        """设置系统模式"""
        valid_modes = ["assist", "autonomous", "training", "maintenance"]
        
        if mode not in valid_modes:
            raise ValueError(f"无效的系统模式。有效模式: {', '.join(valid_modes)}")
        
        with self._mode_lock:
            old_mode = self._current_mode
            self._current_mode = mode
            
            # 记录模式切换历史
            mode_change = {
                "from": old_mode,
                "to": mode,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user": "system"  # 实际应用中应该记录真实用户
            }
            self._mode_history.append(mode_change)
            
            # 限制历史记录大小
            if len(self._mode_history) > 100:
                self._mode_history = self._mode_history[-100:]
            
            logger.info(f"系统模式从 {old_mode} 切换到 {mode}")
            
            # 返回新模式信息
            return self.get_system_mode()
    
    def _get_real_performance_metrics(self) -> Dict[str, float]:
        """获取真实系统性能指标"""
        metrics = {
            "response_time": 0.0,  # 默认值0.0表示未知，会被真实数据覆盖
            "accuracy": 0.0,
            "availability": 0.0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_latency": 0.0,
            "active_connections": 0,
            "system_uptime": 0
        }
        
        # 如果SystemMonitor可用，获取真实指标
        if self._system_monitor is not None:
            try:
                # 获取系统指标
                system_metrics = self._system_monitor.get_system_metrics()
                
                # 提取关键指标
                for metric in system_metrics:
                    metric_id = metric.get("metric_id", "")
                    value = metric.get("value", 0.0)
                    
                    if metric_id == "cpu_percent":
                        metrics["cpu_usage"] = value / 100.0  # 转换为比例
                    elif metric_id == "memory_percent":
                        metrics["memory_usage"] = value / 100.0  # 转换为比例
                    elif metric_id == "disk_percent":
                        metrics["disk_usage"] = value / 100.0  # 转换为比例
                    elif metric_id == "network_latency_ms":
                        metrics["network_latency"] = value
                    elif metric_id == "system_uptime_seconds":
                        metrics["system_uptime"] = value
                
                # 获取活动连接数
                metrics["active_connections"] = self._get_active_connections_count()
                
                # 计算响应时间（基于最近的历史数据）
                metrics["response_time"] = self._calculate_response_time()
                
                # 计算准确率（基于最近的评估结果）
                metrics["accuracy"] = self._calculate_accuracy()
                
                # 计算可用性
                metrics["availability"] = self._calculate_availability()
                
            except Exception as e:
                logger.error(f"获取真实性能指标失败: {e}")
                # 使用默认值
        
        return metrics
    
    def _get_active_connections_count(self) -> int:
        """获取活动连接数"""
        try:
            # 如果psutil可用，获取真实的连接数
            if PSUTIL_AVAILABLE:
                connections = psutil.net_connections()
                return len([conn for conn in connections if conn.status == 'ESTABLISHED'])
            else:
                # psutil不可用，返回0而不是模拟值
                logger.debug("psutil不可用，无法获取真实连接数")
                return 0
        except Exception as e:
            logger.warning(f"获取活动连接数失败: {e}")
            return 0
    
    def _calculate_response_time(self) -> float:
        """计算平均响应时间"""
        try:
            # 如果SystemMonitor可用，尝试获取真实响应时间
            if self._system_monitor is not None:
                # 从SystemMonitor获取最近响应时间
                recent_metrics = self._system_monitor.get_recent_metrics(metric_type="response_time", limit=10)
                if recent_metrics:
                    total_time = sum(metric.get("value", 0.0) for metric in recent_metrics)
                    return total_time / len(recent_metrics)
            
            # SystemMonitor不可用或没有数据，返回0.0表示未知
            logger.debug("SystemMonitor不可用，无法计算真实响应时间")
            return 0.0
        except Exception as e:
            logger.warning(f"计算响应时间失败: {e}")
            return 0.0
    
    def _calculate_accuracy(self) -> float:
        """计算系统准确率"""
        try:
            # 如果SystemMonitor可用，尝试获取真实准确率
            if self._system_monitor is not None:
                # 从SystemMonitor获取最近的准确率评估
                accuracy_metrics = self._system_monitor.get_recent_metrics(metric_type="accuracy", limit=5)
                if accuracy_metrics:
                    total_accuracy = sum(metric.get("value", 0.0) for metric in accuracy_metrics)
                    return total_accuracy / len(accuracy_metrics)
            
            # SystemMonitor不可用或没有数据，返回0.0表示未知
            logger.debug("SystemMonitor不可用，无法计算真实准确率")
            return 0.0
        except Exception as e:
            logger.warning(f"计算准确率失败: {e}")
            return 0.0
    
    def _calculate_availability(self) -> float:
        """计算系统可用性"""
        try:
            # 如果SystemMonitor可用，尝试获取真实可用性
            if self._system_monitor is not None:
                # 从SystemMonitor获取系统运行时间和故障时间
                uptime_metric = self._system_monitor.get_metric(metric_id="system_uptime_seconds")
                downtime_metric = self._system_monitor.get_metric(metric_id="system_downtime_seconds")
                
                if uptime_metric and downtime_metric:
                    uptime = uptime_metric.get("value", 0.0)
                    downtime = downtime_metric.get("value", 0.0)
                    
                    if uptime + downtime > 0:
                        return uptime / (uptime + downtime)
            
            # SystemMonitor不可用或没有数据，返回0.0表示未知
            logger.debug("SystemMonitor不可用，无法计算真实可用性")
            return 0.0
        except Exception as e:
            logger.warning(f"计算可用性失败: {e}")
            return 0.0
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        health_status = {
            "overall_health": "healthy",
            "components": {},
            "alerts": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if self._system_monitor is not None:
            try:
                # 获取系统健康状态
                health = self._system_monitor.get_system_health()
                if health:
                    health_status["overall_health"] = health.get("overall_health", "unknown")
                    health_status["components"] = health.get("components", {})
                
                # 获取活跃警报
                alerts = self._system_monitor.get_active_alerts()
                if alerts:
                    health_status["alerts"] = [alert.to_dict() for alert in alerts[:10]]  # 限制数量
                    
            except Exception as e:
                logger.error(f"获取系统健康状态失败: {e}")
                health_status["overall_health"] = "unknown"
        
        return health_status
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取完整的系统状态信息"""
        return {
            "mode": self.get_system_mode(),
            "health": self.get_system_health(),
            "monitor_available": self._system_monitor is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_mode_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取模式切换历史"""
        with self._mode_lock:
            return self._mode_history[-limit:] if limit > 0 else self._mode_history.copy()


# 全局系统服务实例
_system_service_instance = None


def get_system_service() -> SystemService:
    """获取系统服务实例（单例）"""
    global _system_service_instance
    if _system_service_instance is None:
        _system_service_instance = SystemService()
    return _system_service_instance