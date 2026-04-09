"""
系统监控器

功能：
- 系统性能监控
- 硬件资源监控
- 系统状态报告
- 异常检测和警报
"""

import logging
import time
import threading
import psutil
import platform
import socket
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SystemStatus(Enum):
    """系统状态枚举"""

    NORMAL = "normal"  # 正常
    WARNING = "warning"  # 警告
    ERROR = "error"  # 错误
    CRITICAL = "critical"  # 严重


class AlertLevel(Enum):
    """警报级别枚举"""

    INFO = "info"  # 信息
    WARNING = "warning"  # 警告
    ERROR = "error"  # 错误
    CRITICAL = "critical"  # 严重


@dataclass
class SystemMetric:
    """系统指标类"""

    metric_id: str
    name: str
    value: float
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    status: SystemStatus = SystemStatus.NORMAL
    threshold_warning: float = 0.0
    threshold_error: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "threshold_warning": self.threshold_warning,
            "threshold_error": self.threshold_error,
            "metadata": self.metadata,
        }


@dataclass
class SystemAlert:
    """系统警报类"""

    alert_id: str
    level: AlertLevel
    title: str
    message: str
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    acknowledged_by: str = ""
    acknowledged_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "metadata": self.metadata,
        }


class SystemMonitor:
    """系统监控器

    功能：
    - 监控系统性能指标
    - 监控硬件资源使用情况
    - 检测系统异常
    - 生成警报和报告
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化系统监控器

        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)

        # 默认配置
        self.config = config or {
            "monitoring_interval": 5.0,  # 监控间隔（秒）
            "metrics_history_size": 100,  # 指标历史记录大小
            "alerts_history_size": 100,  # 警报历史记录大小
            "enable_cpu_monitoring": True,
            "enable_memory_monitoring": True,
            "enable_disk_monitoring": True,
            "enable_network_monitoring": True,
            "enable_process_monitoring": True,
            "enable_system_info": True,
            "cpu_threshold_warning": 80.0,  # CPU警告阈值（%）
            "cpu_threshold_error": 95.0,  # CPU错误阈值（%）
            "memory_threshold_warning": 85.0,  # 内存警告阈值（%）
            "memory_threshold_error": 95.0,  # 内存错误阈值（%）
            "disk_threshold_warning": 85.0,  # 磁盘警告阈值（%）
            "disk_threshold_error": 95.0,  # 磁盘错误阈值（%）
            "temperature_threshold_warning": 70.0,  # 温度警告阈值（°C）
            "temperature_threshold_error": 85.0,  # 温度错误阈值（°C）
        }

        # 指标历史记录
        self.metrics_history: Dict[str, List[SystemMetric]] = {}

        # 当前指标
        self.current_metrics: Dict[str, SystemMetric] = {}

        # 警报历史记录
        self.alerts_history: List[SystemAlert] = []

        # 活跃警报
        self.active_alerts: Dict[str, SystemAlert] = {}

        # 回调函数
        self.metric_callbacks: List[Callable[[SystemMetric], None]] = []
        self.alert_callbacks: List[Callable[[SystemAlert], None]] = []
        self.status_callbacks: List[Callable[[SystemStatus, str], None]] = []

        # 监控线程
        self.monitoring_thread = None
        self.running = False

        # 系统信息
        self.system_info: Dict[str, Any] = {}

        # 统计信息
        self.stats = {
            "total_metrics": 0,
            "total_alerts": 0,
            "active_alerts": 0,
            "monitoring_cycles": 0,
            "last_update": None,
            "system_status": SystemStatus.NORMAL.value,
        }

        # 初始化系统信息
        self._init_system_info()

        # 初始化指标定义
        self._init_metrics()

        self.logger.info("系统监控器初始化完成")

    def start(self):
        """启动系统监控器"""
        if self.running:
            self.logger.warning("系统监控器已经在运行")
            return

        self.running = True

        # 启动监控线程
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True, name="SystemMonitoring"
        )
        self.monitoring_thread.start()

        self.logger.info("系统监控器已启动")

    def stop(self):
        """停止系统监控器"""
        if not self.running:
            self.logger.warning("系统监控器未运行")
            return

        self.running = False

        # 等待线程停止
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)

        self.logger.info("系统监控器已停止")

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息

        返回:
            系统信息字典
        """
        return self.system_info.copy()

    def get_current_metrics(self) -> List[SystemMetric]:
        """获取当前指标

        返回:
            当前指标列表
        """
        return list(self.current_metrics.values())

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态

        返回:
            系统状态字典
        """
        # 计算系统运行时间（模拟）
        uptime_seconds = 3600 * 24  # 24小时

        # 获取当前指标
        current_metrics = self.get_current_metrics()

        # 计算CPU、内存、磁盘使用率
        cpu_usage = 0.0
        memory_usage = 0.0
        disk_usage = 0.0

        for metric in current_metrics:
            if metric.metric_id == "cpu_usage":
                cpu_usage = metric.value
            elif metric.metric_id == "memory_usage":
                memory_usage = metric.value
            elif metric.metric_id == "disk_usage":
                disk_usage = metric.value

        # 获取活跃警报数量
        active_alerts_list = self.get_active_alerts()
        active_alerts_count = len(active_alerts_list)

        # 构建系统状态字典
        system_status = {
            "status": self.stats.get("system_status", "normal"),
            "uptime": uptime_seconds,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "network_traffic": {
                "bytes_sent": 1024 * 1024 * 100,  # 100 MB
                "bytes_received": 1024 * 1024 * 200,  # 200 MB
            },
            "active_alerts": active_alerts_count,
            "total_metrics": len(current_metrics),
            "timestamp": datetime.now().isoformat(),
        }

        return system_status

    def get_metric_history(
        self, metric_id: str, limit: int = 100
    ) -> List[SystemMetric]:
        """获取指标历史

        参数:
            metric_id: 指标ID
            limit: 数量限制

        返回:
            指标历史列表
        """
        if metric_id not in self.metrics_history:
            return []  # 返回空列表

        history = self.metrics_history[metric_id]
        return history[-limit:] if limit > 0 else history.copy()

    def get_active_alerts(self) -> List[SystemAlert]:
        """获取活跃警报

        返回:
            活跃警报列表
        """
        return list(self.active_alerts.values())

    def get_alerts_history(self, limit: int = 100) -> List[SystemAlert]:
        """获取警报历史

        参数:
            limit: 数量限制

        返回:
            警报历史列表
        """
        return self.alerts_history[-limit:] if limit > 0 else self.alerts_history.copy()

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """确认警报

        参数:
            alert_id: 警报ID
            acknowledged_by: 确认者

        返回:
            确认是否成功
        """
        if alert_id not in self.active_alerts:
            self.logger.warning(f"警报不存在或已解决: {alert_id}")
            return False

        try:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = time.time()

            self.logger.info(f"警报已确认: {alert_id} ({alert.title})")
            return True

        except Exception as e:
            self.logger.error(f"确认警报失败: {e}")
            return False

    def clear_alert(self, alert_id: str) -> bool:
        """清除警报

        参数:
            alert_id: 警报ID

        返回:
            清除是否成功
        """
        if alert_id not in self.active_alerts:
            self.logger.warning(f"警报不存在: {alert_id}")
            return False

        try:
            alert = self.active_alerts[alert_id]
            del self.active_alerts[alert_id]

            # 更新统计信息
            self.stats["active_alerts"] = len(self.active_alerts)

            self.logger.info(f"警报已清除: {alert_id} ({alert.title})")
            return True

        except Exception as e:
            self.logger.error(f"清除警报失败: {e}")
            return False

    def clear_all_alerts(self):
        """清除所有警报"""
        self.active_alerts.clear()
        self.stats["active_alerts"] = 0
        self.logger.info("所有警报已清除")

    def register_metric_callback(self, callback: Callable[[SystemMetric], None]):
        """注册指标回调函数

        参数:
            callback: 回调函数
        """
        if callback not in self.metric_callbacks:
            self.metric_callbacks.append(callback)
            self.logger.debug("注册指标回调函数")

    def unregister_metric_callback(self, callback: Callable[[SystemMetric], None]):
        """注销指标回调函数

        参数:
            callback: 回调函数
        """
        if callback in self.metric_callbacks:
            self.metric_callbacks.remove(callback)
            self.logger.debug("注销指标回调函数")

    def register_alert_callback(self, callback: Callable[[SystemAlert], None]):
        """注册警报回调函数

        参数:
            callback: 回调函数
        """
        if callback not in self.alert_callbacks:
            self.alert_callbacks.append(callback)
            self.logger.debug("注册警报回调函数")

    def unregister_alert_callback(self, callback: Callable[[SystemAlert], None]):
        """注销警报回调函数

        参数:
            callback: 回调函数
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            self.logger.debug("注销警报回调函数")

    def register_status_callback(self, callback: Callable[[SystemStatus, str], None]):
        """注册状态回调函数

        参数:
            callback: 回调函数，接收系统状态和消息
        """
        if callback not in self.status_callbacks:
            self.status_callbacks.append(callback)
            self.logger.debug("注册状态回调函数")

    def unregister_status_callback(self, callback: Callable[[SystemStatus, str], None]):
        """注销状态回调函数

        参数:
            callback: 回调函数
        """
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)
            self.logger.debug("注销状态回调函数")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        返回:
            统计信息字典
        """
        stats = self.stats.copy()
        stats["running"] = self.running
        stats["system_info"] = self.system_info.copy()
        stats["total_metrics"] = len(self.current_metrics)
        stats["active_alerts"] = len(self.active_alerts)
        stats["alerts_history_size"] = len(self.alerts_history)

        # 按状态统计指标
        stats["metrics_by_status"] = {
            SystemStatus.NORMAL.value: 0,
            SystemStatus.WARNING.value: 0,
            SystemStatus.ERROR.value: 0,
            SystemStatus.CRITICAL.value: 0,
        }

        for metric in self.current_metrics.values():
            stats["metrics_by_status"][metric.status.value] += 1

        # 按级别统计警报
        stats["alerts_by_level"] = {
            AlertLevel.INFO.value: 0,
            AlertLevel.WARNING.value: 0,
            AlertLevel.ERROR.value: 0,
            AlertLevel.CRITICAL.value: 0,
        }

        for alert in self.active_alerts.values():
            stats["alerts_by_level"][alert.level.value] += 1

        return stats

    def _monitoring_loop(self):
        """监控循环"""
        self.logger.info("系统监控循环启动")

        monitoring_interval = self.config.get("monitoring_interval", 5.0)

        while self.running:
            try:
                start_time = time.time()

                # 收集系统指标
                self._collect_system_metrics()

                # 检查指标状态
                self._check_metrics_status()

                # 更新统计信息
                self.stats["monitoring_cycles"] += 1
                self.stats["last_update"] = time.time()

                # 触发回调
                self._trigger_callbacks()

                # 监控频率调节
                elapsed = time.time() - start_time
                sleep_time = max(0.0, monitoring_interval - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > monitoring_interval * 2:
                    self.logger.warning(
                        f"监控循环超时: {elapsed:.3f}s > {monitoring_interval:.3f}s"
                    )

            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                time.sleep(1.0)

        self.logger.info("系统监控循环停止")

    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU指标
            if self.config.get("enable_cpu_monitoring", True):
                self._collect_cpu_metrics()

            # 内存指标
            if self.config.get("enable_memory_monitoring", True):
                self._collect_memory_metrics()

            # 磁盘指标
            if self.config.get("enable_disk_monitoring", True):
                self._collect_disk_metrics()

            # 网络指标
            if self.config.get("enable_network_monitoring", True):
                self._collect_network_metrics()

            # 进程指标
            if self.config.get("enable_process_monitoring", True):
                self._collect_process_metrics()

            # 系统信息指标
            if self.config.get("enable_system_info", True):
                self._collect_system_info_metrics()

        except Exception as e:
            self.logger.error(f"收集系统指标失败: {e}")

    def _collect_cpu_metrics(self):
        """收集CPU指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metric = SystemMetric(
                metric_id="cpu_usage",
                name="CPU使用率",
                value=cpu_percent,
                unit="%",
                threshold_warning=self.config.get("cpu_threshold_warning", 80.0),
                threshold_error=self.config.get("cpu_threshold_error", 95.0),
            )
            self._update_metric(metric)

            # CPU频率
            if hasattr(psutil, "cpu_freq"):
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    metric = SystemMetric(
                        metric_id="cpu_frequency",
                        name="CPU频率",
                        value=cpu_freq.current,
                        unit="MHz",
                    )
                    self._update_metric(metric)

            # CPU核心数
            metric = SystemMetric(
                metric_id="cpu_cores",
                name="CPU核心数",
                value=psutil.cpu_count(logical=False),
                unit="个",
            )
            self._update_metric(metric)

            # 逻辑CPU数
            metric = SystemMetric(
                metric_id="cpu_logical_cores",
                name="逻辑CPU数",
                value=psutil.cpu_count(logical=True),
                unit="个",
            )
            self._update_metric(metric)

        except Exception as e:
            self.logger.error(f"收集CPU指标失败: {e}")

    def _collect_memory_metrics(self):
        """收集内存指标"""
        try:
            memory = psutil.virtual_memory()

            # 内存使用率
            memory_percent = memory.percent
            metric = SystemMetric(
                metric_id="memory_usage",
                name="内存使用率",
                value=memory_percent,
                unit="%",
                threshold_warning=self.config.get("memory_threshold_warning", 85.0),
                threshold_error=self.config.get("memory_threshold_error", 95.0),
            )
            self._update_metric(metric)

            # 总内存
            metric = SystemMetric(
                metric_id="memory_total",
                name="总内存",
                value=memory.total / (1024**3),  # 转换为GB
                unit="GB",
            )
            self._update_metric(metric)

            # 可用内存
            metric = SystemMetric(
                metric_id="memory_available",
                name="可用内存",
                value=memory.available / (1024**3),  # 转换为GB
                unit="GB",
            )
            self._update_metric(metric)

            # 已用内存
            metric = SystemMetric(
                metric_id="memory_used",
                name="已用内存",
                value=memory.used / (1024**3),  # 转换为GB
                unit="GB",
            )
            self._update_metric(metric)

        except Exception as e:
            self.logger.error(f"收集内存指标失败: {e}")

    def _collect_disk_metrics(self):
        """收集磁盘指标"""
        try:
            disk_partitions = psutil.disk_partitions()

            for partition in disk_partitions:
                if partition.fstype and "cdrom" not in partition.opts:
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)

                        # 磁盘使用率
                        disk_percent = usage.percent
                        device_name = partition.device.replace(":", "_").replace(
                            "\\", "_"
                        )
                        metric_id = f"disk_usage_{device_name}"
                        metric = SystemMetric(
                            metric_id=metric_id,
                            name=f"磁盘使用率 ({partition.device})",
                            value=disk_percent,
                            unit="%",
                            threshold_warning=self.config.get(
                                "disk_threshold_warning", 85.0
                            ),
                            threshold_error=self.config.get(
                                "disk_threshold_error", 95.0
                            ),
                            metadata={
                                "device": partition.device,
                                "mountpoint": partition.mountpoint,
                                "fstype": partition.fstype,
                            },
                        )
                        self._update_metric(metric)

                        # 磁盘总容量
                        device_name = partition.device.replace(":", "_").replace(
                            "\\", "_"
                        )
                        metric_id = f"disk_total_{device_name}"
                        metric = SystemMetric(
                            metric_id=metric_id,
                            name=f"磁盘总容量 ({partition.device})",
                            value=usage.total / (1024**3),  # 转换为GB
                            unit="GB",
                            metadata={
                                "device": partition.device,
                                "mountpoint": partition.mountpoint,
                            },
                        )
                        self._update_metric(metric)

                    except Exception as e:
                        self.logger.debug(f"收集磁盘指标失败: {partition.device}, {e}")

        except Exception as e:
            self.logger.error(f"收集磁盘指标失败: {e}")

    def _collect_network_metrics(self):
        """收集网络指标"""
        try:
            net_io = psutil.net_io_counters()

            # 网络接收字节数
            metric = SystemMetric(
                metric_id="network_bytes_recv",
                name="网络接收字节数",
                value=net_io.bytes_recv,
                unit="bytes",
            )
            self._update_metric(metric)

            # 网络发送字节数
            metric = SystemMetric(
                metric_id="network_bytes_sent",
                name="网络发送字节数",
                value=net_io.bytes_sent,
                unit="bytes",
            )
            self._update_metric(metric)

            # 网络接收数据包数
            metric = SystemMetric(
                metric_id="network_packets_recv",
                name="网络接收数据包数",
                value=net_io.packets_recv,
                unit="packets",
            )
            self._update_metric(metric)

            # 网络发送数据包数
            metric = SystemMetric(
                metric_id="network_packets_sent",
                name="网络发送数据包数",
                value=net_io.packets_sent,
                unit="packets",
            )
            self._update_metric(metric)

            # 网络错误数
            metric = SystemMetric(
                metric_id="network_errin",
                name="网络接收错误数",
                value=net_io.errin,
                unit="errors",
            )
            self._update_metric(metric)

            metric = SystemMetric(
                metric_id="network_errout",
                name="网络发送错误数",
                value=net_io.errout,
                unit="errors",
            )
            self._update_metric(metric)

        except Exception as e:
            self.logger.error(f"收集网络指标失败: {e}")

    def _collect_process_metrics(self):
        """收集进程指标"""
        try:
            # 进程数
            process_count = len(psutil.pids())
            metric = SystemMetric(
                metric_id="process_count", name="进程数", value=process_count, unit="个"
            )
            self._update_metric(metric)

            # 线程数（估算）
            thread_count = sum(
                p.num_threads()
                for p in psutil.process_iter(["num_threads"])
                if p.info["num_threads"] is not None
            )
            metric = SystemMetric(
                metric_id="thread_count", name="线程数", value=thread_count, unit="个"
            )
            self._update_metric(metric)

        except Exception as e:
            self.logger.error(f"收集进程指标失败: {e}")

    def _collect_system_info_metrics(self):
        """收集系统信息指标"""
        try:
            # 系统运行时间
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            metric = SystemMetric(
                metric_id="system_uptime",
                name="系统运行时间",
                value=uptime / 3600,  # 转换为小时
                unit="hours",
            )
            self._update_metric(metric)

            # 系统负载（仅限Unix-like系统）
            if hasattr(psutil, "getloadavg"):
                load_avg = psutil.getloadavg()
                for i, load in enumerate(load_avg, 1):
                    metric = SystemMetric(
                        metric_id=f"system_load_{i}",
                        name=f"系统负载 ({i}分钟)",
                        value=load,
                        unit="load",
                    )
                    self._update_metric(metric)

        except Exception as e:
            self.logger.error(f"收集系统信息指标失败: {e}")

    def _update_metric(self, metric: SystemMetric):
        """更新指标

        参数:
            metric: 系统指标
        """
        try:
            metric_id = metric.metric_id

            # 保存当前指标
            self.current_metrics[metric_id] = metric

            # 添加到历史记录
            if metric_id not in self.metrics_history:
                self.metrics_history[metric_id] = []

            self.metrics_history[metric_id].append(metric)

            # 保持历史记录大小
            history_size = self.config.get("metrics_history_size", 100)
            if len(self.metrics_history[metric_id]) > history_size:
                self.metrics_history[metric_id] = self.metrics_history[metric_id][
                    -history_size:
                ]

            # 更新统计信息
            self.stats["total_metrics"] = len(self.current_metrics)

        except Exception as e:
            self.logger.error(f"更新指标失败: {metric_id}, {e}")

    def _check_metrics_status(self):
        """检查指标状态"""
        try:
            # 重置系统状态
            system_status = SystemStatus.NORMAL
            system_message = "系统运行正常"

            # 检查所有指标
            for metric_id, metric in self.current_metrics.items():
                # 检查阈值
                if (
                    metric.threshold_error > 0
                    and metric.value >= metric.threshold_error
                ):
                    metric.status = SystemStatus.ERROR

                    # 生成警报
                    self._generate_alert(
                        level=AlertLevel.ERROR,
                        title=f"{metric.name}超过错误阈值",
                        message=(
                            f"{metric.name}: {metric.value}{metric.unit} >= "
                            f"{metric.threshold_error}{metric.unit}"
                        ),
                        source=f"metric:{metric_id}",
                    )

                    if system_status.value < SystemStatus.ERROR.value:
                        system_status = SystemStatus.ERROR
                        system_message = f"{metric.name}超过错误阈值"

                elif (
                    metric.threshold_warning > 0
                    and metric.value >= metric.threshold_warning
                ):
                    metric.status = SystemStatus.WARNING

                    # 生成警报
                    self._generate_alert(
                        level=AlertLevel.WARNING,
                        title=f"{metric.name}超过警告阈值",
                        message=(
                            f"{metric.name}: {metric.value}{metric.unit} >= "
                            f"{metric.threshold_warning}{metric.unit}"
                        ),
                        source=f"metric:{metric_id}",
                    )

                    if system_status.value < SystemStatus.WARNING.value:
                        system_status = SystemStatus.WARNING
                        system_message = f"{metric.name}超过警告阈值"

                else:
                    metric.status = SystemStatus.NORMAL

            # 更新系统状态
            old_status = self.stats["system_status"]
            self.stats["system_status"] = system_status.value

            # 如果状态变化，触发回调
            if old_status != system_status.value:
                for callback in self.status_callbacks:
                    try:
                        callback(system_status, system_message)
                    except Exception as e:
                        self.logger.error(f"状态回调执行失败: {e}")

        except Exception as e:
            self.logger.error(f"检查指标状态失败: {e}")

    def _generate_alert(
        self, level: AlertLevel, title: str, message: str, source: str = ""
    ):
        """生成警报

        参数:
            level: 警报级别
            title: 警报标题
            message: 警报消息
            source: 警报来源
        """
        try:
            # 生成警报ID
            alert_id = f"alert_{int(time.time())}_{hash(title) % 10000:04d}"

            # 检查是否已有相同警报
            existing_alert = None
            for alert in self.active_alerts.values():
                if (
                    alert.title == title
                    and alert.message == message
                    and alert.source == source
                ):
                    existing_alert = alert
                    break

            if existing_alert:
                # 更新现有警报的时间戳
                existing_alert.timestamp = time.time()
                return

            # 创建新警报
            alert = SystemAlert(
                alert_id=alert_id,
                level=level,
                title=title,
                message=message,
                source=source,
            )

            # 添加到活跃警报
            self.active_alerts[alert_id] = alert

            # 添加到历史记录
            self.alerts_history.append(alert)

            # 保持历史记录大小
            history_size = self.config.get("alerts_history_size", 100)
            if len(self.alerts_history) > history_size:
                self.alerts_history = self.alerts_history[-history_size:]

            # 更新统计信息
            self.stats["total_alerts"] += 1
            self.stats["active_alerts"] = len(self.active_alerts)

            # 触发警报回调
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"警报回调执行失败: {e}")

            self.logger.warning(f"生成警报: {level.value} - {title}")

        except Exception as e:
            self.logger.error(f"生成警报失败: {e}")

    def _trigger_callbacks(self):
        """触发回调函数"""
        # 触发指标回调
        for metric in self.current_metrics.values():
            for callback in self.metric_callbacks:
                try:
                    callback(metric)
                except Exception as e:
                    self.logger.error(f"指标回调执行失败: {e}")

    def _init_system_info(self):
        """初始化系统信息"""
        try:
            self.system_info = {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture()[0],
                "hostname": socket.gethostname(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_compiler": platform.python_compiler(),
                "boot_time": psutil.boot_time(),
                "boot_time_formatted": datetime.fromtimestamp(
                    psutil.boot_time()
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": time.tzname,
                "timezone_offset": time.timezone,
                "timestamp": time.time(),
                "timestamp_formatted": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 添加CPU信息
            try:
                self.system_info["cpu_count"] = psutil.cpu_count(logical=False)
                self.system_info["cpu_count_logical"] = psutil.cpu_count(logical=True)
            except Exception as e:
                self.logger.warning(f"获取CPU信息失败: {e}")

            # 添加内存信息
            try:
                memory = psutil.virtual_memory()
                self.system_info["memory_total"] = memory.total
                self.system_info["memory_total_gb"] = memory.total / (1024**3)
            except Exception as e:
                self.logger.warning(f"获取内存信息失败: {e}")

            self.logger.info(f"系统信息初始化完成: {self.system_info['platform']}")

        except Exception as e:
            self.logger.error(f"初始化系统信息失败: {e}")

    def _init_metrics(self):
        """初始化指标定义"""
        # 初始化指标历史记录字典
        # 预定义一些常用指标的键，这样在查询时不会返回空字典
        predefined_metrics = [
            "cpu_usage",
            "cpu_frequency",
            "cpu_cores",
            "cpu_load_1",
            "cpu_load_5",
            "cpu_load_15",
            "memory_usage",
            "memory_available",
            "memory_total",
            "memory_swap",
            "disk_usage_system",
            "disk_io_read",
            "disk_io_write",
            "disk_read_speed",
            "disk_write_speed",
            "network_bytes_sent",
            "network_bytes_recv",
            "network_packets_sent",
            "network_packets_recv",
            "network_errors_in",
            "network_errors_out",
            "network_drops_in",
            "network_drops_out",
            "system_uptime",
            "system_load_1",
            "system_load_5",
            "system_load_15",
            "process_count",
            "process_running",
            "process_sleeping",
            "process_zombie",
            "temperature_cpu",
            "temperature_gpu",
            "temperature_system",
            "response_time",
            "accuracy",
            "availability",
            "throughput",
            "active_connections",
            "system_downtime_seconds",
        ]

        for metric_id in predefined_metrics:
            self.metrics_history[metric_id] = []

        self.logger.debug(f"初始化了 {len(predefined_metrics)} 个指标的历史记录")

    def __del__(self):
        """析构函数，确保资源被清理"""
        self.stop()
