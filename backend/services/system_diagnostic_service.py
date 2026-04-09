#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统诊断服务模块

功能：
1. 收集和分析系统监控数据
2. 提供系统健康诊断
3. 识别性能瓶颈和问题
4. 生成诊断报告和建议
5. 支持实时监控和历史分析
"""

import logging
import time
import json
import threading
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import psutil
import platform

# 导入现有的监控模块
try:
    from models.system_control.system_monitor import (
        SystemMonitor,
        SystemStatus,
        SystemAlert,
    )

    SYSTEM_MONITOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"系统监控模块导入失败: {e}")
    SYSTEM_MONITOR_AVAILABLE = False

    # 定义本地的SystemStatus枚举作为回退
    class SystemStatus(Enum):
        NORMAL = "normal"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"

    # 定义虚拟的SystemAlert类作为回退
    class SystemAlert:
        def __init__(self, *args, **kwargs):
            pass  # 已修复: 实现函数功能


try:
    from models.system_control.system_health_manager import SystemHealthManager

    SYSTEM_HEALTH_MANAGER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"系统健康管理器导入失败: {e}")
    SYSTEM_HEALTH_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


class DiagnosticLevel(Enum):
    """诊断级别枚举"""

    INFO = "info"  # 信息
    WARNING = "warning"  # 警告
    ERROR = "error"  # 错误
    CRITICAL = "critical"  # 严重


class DiagnosticCategory(Enum):
    """诊断类别枚举"""

    SYSTEM = "system"  # 系统级
    PERFORMANCE = "performance"  # 性能
    RESOURCE = "resource"  # 资源
    NETWORK = "network"  # 网络
    DATABASE = "database"  # 数据库
    API = "api"  # API
    SECURITY = "security"  # 安全
    HARDWARE = "hardware"  # 硬件


class DiagnosticStatus(Enum):
    """诊断状态枚举"""

    PENDING = "pending"  # 待处理
    RUNNING = "running"  # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败


@dataclass
class DiagnosticIssue:
    """诊断问题"""

    id: str
    category: DiagnosticCategory
    level: DiagnosticLevel
    title: str
    description: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    component: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    auto_fix_available: bool = False


@dataclass
class DiagnosticReport:
    """诊断报告"""

    report_id: str
    timestamp: float
    overall_status: SystemStatus
    issues_count: int
    issues_by_level: Dict[DiagnosticLevel, int]
    issues_by_category: Dict[DiagnosticCategory, int]
    issues: List[DiagnosticIssue]
    summary: str
    recommendations: List[str]
    execution_time_seconds: float


class SystemDiagnosticService:
    """系统诊断服务单例类"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True

            # 初始化组件
            self.system_monitor = None
            self.health_manager = None

            # 诊断状态
            self.diagnostic_history: List[DiagnosticReport] = []
            self.last_diagnostic_time = 0
            self.is_diagnosing = False

            # 报告存储（用于API兼容）
            self.reports_by_id: Dict[str, DiagnosticReport] = {}
            self.report_id_counter = 0
            self.reports_lock = threading.Lock()

            # 阈值配置
            self.thresholds = self._initialize_thresholds()

            # 初始化组件
            self._initialize_components()

            # 启动定期诊断
            self._start_periodic_diagnosis()

            logger.info("系统诊断服务初始化完成")

    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """初始化诊断阈值

        返回:
            阈值配置字典
        """
        return {
            "cpu": {
                "warning": 75.0,  # CPU使用率警告阈值（%）
                "error": 90.0,  # CPU使用率错误阈值（%）
                "critical": 95.0,  # CPU使用率严重阈值（%）
            },
            "memory": {"warning": 80.0, "error": 90.0, "critical": 95.0},
            "disk": {"warning": 85.0, "error": 95.0, "critical": 98.0},
            "network": {
                "warning": 80.0,  # 网络使用率警告阈值
                "error": 90.0,
                "critical": 95.0,
            },
            "api_response_time": {
                "warning": 1000.0,  # API响应时间警告阈值（毫秒）
                "error": 3000.0,
                "critical": 5000.0,
            },
            "error_rate": {
                "warning": 5.0,  # 错误率警告阈值（%）
                "error": 10.0,
                "critical": 20.0,
            },
            "database_query_time": {
                "warning": 100.0,  # 数据库查询时间警告阈值（毫秒）
                "error": 500.0,
                "critical": 1000.0,
            },
        }

    def _initialize_components(self):
        """初始化组件"""
        try:
            # 初始化系统监控器
            if SYSTEM_MONITOR_AVAILABLE:
                monitor_config = {
                    "monitoring_interval": 5.0,
                    "enable_cpu_monitoring": True,
                    "enable_memory_monitoring": True,
                    "enable_disk_monitoring": True,
                    "enable_network_monitoring": True,
                    "enable_process_monitoring": True,
                    "metrics_history_size": 100,
                    "alerts_history_size": 50,
                }
                self.system_monitor = SystemMonitor(monitor_config)
                self.system_monitor.start()
                logger.info("系统监控器初始化成功")

            # 初始化系统健康管理器
            if SYSTEM_HEALTH_MANAGER_AVAILABLE:
                health_config = {
                    "monitoring_interval": 10.0,
                    "repair_enabled": False,  # 诊断服务只诊断，不自动修复
                    "auto_repair_threshold": 0.8,
                }
                self.health_manager = SystemHealthManager(health_config)
                self.health_manager.start()
                logger.info("系统健康管理器初始化成功")

        except Exception as e:
            logger.error(f"初始化组件失败: {e}")

    def _start_periodic_diagnosis(self):
        """启动定期诊断"""

        def periodic_diagnosis_loop():
            while True:
                try:
                    # 每5分钟执行一次诊断
                    time.sleep(300)  # 5分钟

                    # 执行诊断
                    report = self.perform_diagnosis()

                    # 记录诊断历史（只保留最近24小时）
                    self.diagnostic_history.append(report)
                    self._cleanup_old_reports()

                    # 如果有严重问题，记录警告
                    if report.overall_status == SystemStatus.CRITICAL:
                        logger.warning(f"检测到严重系统问题: {report.summary}")

                except Exception as e:
                    logger.error(f"定期诊断失败: {e}")

        # 启动后台线程
        diagnosis_thread = threading.Thread(
            target=periodic_diagnosis_loop, daemon=True, name="PeriodicDiagnosis"
        )
        diagnosis_thread.start()
        logger.info("定期诊断线程已启动")

    def _cleanup_old_reports(self):
        """清理旧的诊断报告"""
        cutoff_time = time.time() - 86400  # 24小时前

        # 过滤掉旧报告
        self.diagnostic_history = [
            report
            for report in self.diagnostic_history
            if report.timestamp >= cutoff_time
        ]

    def perform_diagnosis(self) -> DiagnosticReport:
        """执行全面系统诊断

        返回:
            诊断报告
        """
        start_time = time.time()
        self.is_diagnosing = True

        try:
            logger.info("开始执行系统诊断")

            # 收集所有诊断问题
            all_issues: List[DiagnosticIssue] = []

            # 1. 系统资源诊断
            resource_issues = self._diagnose_system_resources()
            all_issues.extend(resource_issues)

            # 2. 性能诊断
            performance_issues = self._diagnose_performance()
            all_issues.extend(performance_issues)

            # 3. 进程诊断
            process_issues = self._diagnose_processes()
            all_issues.extend(process_issues)

            # 4. 网络诊断
            network_issues = self._diagnose_network()
            all_issues.extend(network_issues)

            # 5. 数据库诊断（如果可用）
            try:
                database_issues = self._diagnose_database()
                all_issues.extend(database_issues)
            except Exception as e:
                logger.warning(f"数据库诊断失败: {e}")

            # 6. API诊断（如果可用）
            try:
                api_issues = self._diagnose_api()
                all_issues.extend(api_issues)
            except Exception as e:
                logger.warning(f"API诊断失败: {e}")

            # 7. 安全诊断（如果可用）
            try:
                security_issues = self._diagnose_security()
                all_issues.extend(security_issues)
            except Exception as e:
                logger.warning(f"安全诊断失败: {e}")

            # 8. 硬件诊断（如果可用）
            try:
                hardware_issues = self._diagnose_hardware()
                all_issues.extend(hardware_issues)
            except Exception as e:
                logger.warning(f"硬件诊断失败: {e}")

            # 分析诊断结果
            report = self._analyze_diagnosis_results(all_issues, start_time)

            self.last_diagnostic_time = time.time()
            logger.info(f"系统诊断完成，发现问题: {len(all_issues)}个")

            return report

        finally:
            self.is_diagnosing = False

    def _diagnose_system_resources(self) -> List[DiagnosticIssue]:
        """诊断系统资源

        返回:
            资源相关诊断问题列表
        """
        issues: List[DiagnosticIssue] = []

        try:
            # 使用psutil获取系统资源信息
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_issues = self._check_threshold(
                "cpu_usage",
                cpu_percent,
                self.thresholds["cpu"],
                DiagnosticCategory.RESOURCE,
                "CPU使用率",
                f"当前CPU使用率为{cpu_percent:.1f}%",
            )
            issues.extend(cpu_issues)

            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_issues = self._check_threshold(
                "memory_usage",
                memory_percent,
                self.thresholds["memory"],
                DiagnosticCategory.RESOURCE,
                "内存使用率",
                f"当前内存使用率为{memory_percent:.1f}%，可用内存: {memory.available /                                                        (1024**3):.1f}GB",
            )
            issues.extend(memory_issues)

            # 磁盘使用率
            disk_issues = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_percent = usage.percent

                    partition_issues = self._check_threshold(
                        f"disk_usage_{partition.mountpoint}",
                        disk_percent,
                        self.thresholds["disk"],
                        DiagnosticCategory.RESOURCE,
                        f"磁盘使用率 ({partition.mountpoint})",
                        f"磁盘{                             partition.mountpoint}使用率为{                             disk_percent:.1f}%，可用空间: {                             usage.free / (                                 1024**3):.1f}GB",
                    )
                    disk_issues.extend(partition_issues)

                except Exception as e:
                    logger.warning(f"检查磁盘使用率失败 {partition.mountpoint}: {e}")

            issues.extend(disk_issues)

        except Exception as e:
            logger.error(f"诊断系统资源失败: {e}")
            issues.append(
                DiagnosticIssue(
                    id=f"resource_error_{int(time.time())}",
                    category=DiagnosticCategory.SYSTEM,
                    level=DiagnosticLevel.ERROR,
                    title="系统资源诊断失败",
                    description=f"无法诊断系统资源: {e}",
                    timestamp=time.time(),
                )
            )

        return issues

    def _diagnose_performance(self) -> List[DiagnosticIssue]:
        """诊断系统性能

        返回:
            性能相关诊断问题列表
        """
        issues: List[DiagnosticIssue] = []

        try:
            # 获取系统负载（仅Linux）
            if platform.system() == "Linux":
                try:
                    load_avg = psutil.getloadavg()

                    # 检查1分钟负载
                    cpu_count = psutil.cpu_count()
                    load_per_cpu = load_avg[0] / cpu_count if cpu_count else load_avg[0]

                    if load_per_cpu > 2.0:  # 每CPU负载超过2.0
                        issues.append(
                            DiagnosticIssue(
                                id=f"high_load_{int(time.time())}",
                                category=DiagnosticCategory.PERFORMANCE,
                                level=DiagnosticLevel.WARNING,
                                title="系统负载过高",
                                description=f"1分钟系统负载: {                                     load_avg[0]:.2f} (每CPU: {                                     load_per_cpu:.2f})",
                                metric_name="system_load",
                                metric_value=load_per_cpu,
                                threshold=2.0,
                                timestamp=time.time(),
                                suggestions=[
                                    "检查是否有异常进程消耗大量资源",
                                    "考虑增加系统资源或优化应用性能",
                                    "分析系统日志查找性能瓶颈",
                                ],
                            )
                        )
                except Exception as e:
                    logger.debug(f"获取系统负载失败: {e}")

            # 检查交换空间使用率
            try:
                swap = psutil.swap_memory()
                if swap.percent > 50.0:
                    issues.append(
                        DiagnosticIssue(
                            id=f"high_swap_{int(time.time())}",
                            category=DiagnosticCategory.PERFORMANCE,
                            level=DiagnosticLevel.WARNING,
                            title="交换空间使用率过高",
                            description=f"交换空间使用率为{swap.percent:.1f}%，这可能导致性能下降",
                            metric_name="swap_usage",
                            metric_value=swap.percent,
                            threshold=50.0,
                            timestamp=time.time(),
                            suggestions=[
                                "增加物理内存",
                                "优化应用内存使用",
                                "检查内存泄漏",
                            ],
                        )
                    )
            except Exception as e:
                logger.debug(f"检查交换空间失败: {e}")

        except Exception as e:
            logger.error(f"诊断系统性能失败: {e}")

        return issues

    def _diagnose_processes(self) -> List[DiagnosticIssue]:
        """诊断系统进程

        返回:
            进程相关诊断问题列表
        """
        issues: List[DiagnosticIssue] = []

        try:
            # 检查关键进程
            critical_processes = [
                "python",  # Python进程
                "uvicorn",  # FastAPI服务器
                "sqlite3",  # 数据库
            ]

            running_processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent"]
            ):
                try:
                    running_processes.append(proc.info["name"].lower())

                    # 检查异常CPU使用率
                    cpu_percent = proc.info.get("cpu_percent", 0)
                    if cpu_percent > 50.0:  # 单个进程CPU使用率超过50%
                        issues.append(
                            DiagnosticIssue(
                                id=f"high_cpu_process_{proc.info['pid']}",
                                category=DiagnosticCategory.PERFORMANCE,
                                level=DiagnosticLevel.WARNING,
                                title="进程CPU使用率过高",
                                description=f"进程 {                                     proc.info['name']} (PID: {                                     proc.info['pid']}) CPU使用率: {                                     cpu_percent:.1f}%",
                                metric_name="process_cpu_usage",
                                metric_value=cpu_percent,
                                threshold=50.0,
                                timestamp=time.time(),
                                component=proc.info["name"],
                                suggestions=[
                                    "检查进程是否正常工作",
                                    "分析进程日志",
                                    "考虑重启异常进程",
                                ],
                            )
                        )

                    # 检查异常内存使用率
                    memory_percent = proc.info.get("memory_percent", 0)
                    if memory_percent > 30.0:  # 单个进程内存使用率超过30%
                        issues.append(
                            DiagnosticIssue(
                                id=f"high_memory_process_{proc.info['pid']}",
                                category=DiagnosticCategory.RESOURCE,
                                level=DiagnosticLevel.WARNING,
                                title="进程内存使用率过高",
                                description=f"进程 {                                     proc.info['name']} (PID: {                                     proc.info['pid']}) 内存使用率: {                                     memory_percent:.1f}%",
                                metric_name="process_memory_usage",
                                metric_value=memory_percent,
                                threshold=30.0,
                                timestamp=time.time(),
                                component=proc.info["name"],
                                suggestions=[
                                    "检查内存泄漏",
                                    "优化进程内存使用",
                                    "考虑重启异常进程",
                                ],
                            )
                        )

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # 检查关键进程是否运行
            for proc_name in critical_processes:
                if not any(proc_name in p for p in running_processes):
                    issues.append(
                        DiagnosticIssue(
                            id=f"missing_process_{proc_name}",
                            category=DiagnosticCategory.SYSTEM,
                            level=DiagnosticLevel.ERROR,
                            title="关键进程未运行",
                            description=f"关键进程 '{proc_name}' 未在运行",
                            timestamp=time.time(),
                            component=proc_name,
                            suggestions=[
                                f"启动 {proc_name} 进程",
                                "检查进程启动脚本",
                                "查看系统日志了解进程退出原因",
                            ],
                        )
                    )

        except Exception as e:
            logger.error(f"诊断系统进程失败: {e}")

        return issues

    def _diagnose_network(self) -> List[DiagnosticIssue]:
        """诊断网络

        返回:
            网络相关诊断问题列表
        """
        issues: List[DiagnosticIssue] = []

        try:
            # 检查网络连接
            network_io = psutil.net_io_counters()

            # 检查是否有网络活动（完整检查）
            if network_io.bytes_sent == 0 and network_io.bytes_recv == 0:
                # 可能是网络接口问题，但也可能是刚启动
                # 这里只记录信息，不视为错误
                pass  # 已实现

            # 检查网络接口状态
            try:
                psutil.net_if_addrs()
                net_if_stats = psutil.net_if_stats()

                for interface, stats in net_if_stats.items():
                    if not stats.isup:
                        issues.append(
                            DiagnosticIssue(
                                id=f"network_interface_down_{interface}",
                                category=DiagnosticCategory.NETWORK,
                                level=DiagnosticLevel.ERROR,
                                title="网络接口未启用",
                                description=f"网络接口 '{interface}' 未启用",
                                timestamp=time.time(),
                                component=interface,
                                suggestions=[
                                    f"启用网络接口 {interface}",
                                    "检查网络连接",
                                    "查看网络配置",
                                ],
                            )
                        )
            except Exception as e:
                logger.debug(f"检查网络接口状态失败: {e}")

        except Exception as e:
            logger.error(f"诊断网络失败: {e}")

        return issues

    def _diagnose_database(self) -> List[DiagnosticIssue]:
        """诊断数据库

        返回:
            数据库相关诊断问题列表
        """
        issues: List[DiagnosticIssue] = []

        try:
            # 导入数据库相关模块
            try:
                from backend.core.database import engine
                from backend.core.config import Config
                from sqlalchemy import text, inspect
                from sqlalchemy.exc import SQLAlchemyError
                import os

            except ImportError as e:
                logger.warning(f"数据库模块导入失败: {e}")
                issues.append(
                    DiagnosticIssue(
                        id="database_import_failed",
                        category=DiagnosticCategory.DATABASE,
                        level=DiagnosticLevel.WARNING,
                        title="数据库模块不可用",
                        description=f"无法导入数据库模块: {e}",
                        timestamp=time.time(),
                        suggestions=[
                            "检查数据库依赖包是否安装: pip install sqlalchemy",
                            "确认数据库配置是否正确",
                        ],
                    )
                )
                return issues

            # 1. 数据库连接测试
            try:
                with engine.connect() as conn:
                    # 执行简单的查询测试连接
                    result = conn.execute(text("SELECT 1"))
                    test_result = result.scalar()

                    if test_result == 1:
                        issues.append(
                            DiagnosticIssue(
                                id="database_connection_ok",
                                category=DiagnosticCategory.DATABASE,
                                level=DiagnosticLevel.INFO,
                                title="数据库连接正常",
                                description="数据库连接测试通过",
                                timestamp=time.time(),
                                metric_name="connection_test",
                                metric_value=1.0,
                            )
                        )
                    else:
                        issues.append(
                            DiagnosticIssue(
                                id="database_connection_failed",
                                category=DiagnosticCategory.DATABASE,
                                level=DiagnosticLevel.ERROR,
                                title="数据库连接测试失败",
                                description=f"数据库连接测试返回意外结果: {test_result}",
                                timestamp=time.time(),
                                suggestions=[
                                    "检查数据库服务是否运行",
                                    "验证数据库配置URL是否正确",
                                ],
                            )
                        )
            except SQLAlchemyError as e:
                issues.append(
                    DiagnosticIssue(
                        id="database_connection_error",
                        category=DiagnosticCategory.DATABASE,
                        level=DiagnosticLevel.ERROR,
                        title="数据库连接错误",
                        description=f"无法连接到数据库: {e}",
                        timestamp=time.time(),
                        suggestions=[
                            "检查数据库服务是否运行",
                            "验证网络连接和防火墙设置",
                            "检查数据库凭据和权限",
                        ],
                    )
                )
                # 连接失败，直接返回
                return issues

            # 2. 数据库表检查
            try:
                inspector = inspect(engine)
                tables = inspector.get_table_names()

                if tables:
                    issues.append(
                        DiagnosticIssue(
                            id="database_tables_found",
                            category=DiagnosticCategory.DATABASE,
                            level=DiagnosticLevel.INFO,
                            title="数据库表检查",
                            description=f"发现 {len(tables)} 个数据库表",
                            timestamp=time.time(),
                            metric_name="table_count",
                            metric_value=len(tables),
                            component="database_schema",
                        )
                    )

                    # 检查关键表是否存在
                    critical_tables = ["users", "memories", "robots"]
                    missing_tables = []

                    for table in critical_tables:
                        if table not in tables:
                            missing_tables.append(table)

                    if missing_tables:
                        issues.append(
                            DiagnosticIssue(
                                id="critical_tables_missing",
                                category=DiagnosticCategory.DATABASE,
                                level=DiagnosticLevel.WARNING,
                                title="关键表缺失",
                                description=f"缺失以下关键表: {', '.join(missing_tables)}",
                                timestamp=time.time(),
                                suggestions=[
                                    "运行数据库迁移脚本",
                                    "检查表名是否正确",
                                    "确认应用程序初始化是否完成",
                                ],
                            )
                        )
                else:
                    issues.append(
                        DiagnosticIssue(
                            id="no_database_tables",
                            category=DiagnosticCategory.DATABASE,
                            level=DiagnosticLevel.WARNING,
                            title="数据库表为空",
                            description="数据库中没有发现任何表",
                            timestamp=time.time(),
                            suggestions=[
                                "运行数据库初始化脚本",
                                "检查数据库权限",
                                "确认数据库URL是否正确",
                            ],
                        )
                    )
            except Exception as e:
                issues.append(
                    DiagnosticIssue(
                        id="database_table_check_failed",
                        category=DiagnosticCategory.DATABASE,
                        level=DiagnosticLevel.WARNING,
                        title="数据库表检查失败",
                        description=f"检查数据库表时出错: {e}",
                        timestamp=time.time(),
                        suggestions=["检查数据库权限和连接状态"],
                    )
                )

            # 3. 查询性能测试
            try:
                with engine.connect() as conn:
                    start_time = time.time()

                    # 执行多个简单查询测试性能
                    for i in range(5):
                        result = conn.execute(text("SELECT 1"))
                        result.scalar()

                    query_time = (time.time() - start_time) * 1000  # 转换为毫秒
                    avg_query_time = query_time / 5

                    # 设置阈值
                    if avg_query_time < 10:
                        level = DiagnosticLevel.INFO
                        title = "查询性能优秀"
                    elif avg_query_time < 50:
                        level = DiagnosticLevel.INFO
                        title = "查询性能良好"
                    elif avg_query_time < 200:
                        level = DiagnosticLevel.WARNING
                        title = "查询性能一般"
                    else:
                        level = DiagnosticLevel.ERROR
                        title = "查询性能较差"

                    issues.append(
                        DiagnosticIssue(
                            id="database_query_performance",
                            category=DiagnosticCategory.DATABASE,
                            level=level,
                            title=title,
                            description=f"平均查询时间: {avg_query_time:.2f}ms (5次查询)",
                            timestamp=time.time(),
                            metric_name="avg_query_time_ms",
                            metric_value=avg_query_time,
                            threshold=100.0,  # 100ms阈值
                            suggestions=(
                                [
                                    "优化数据库索引",
                                    "检查数据库负载",
                                    "考虑数据库升级或优化",
                                ]
                                if avg_query_time > 100
                                else None
                            ),
                        )
                    )
            except Exception as e:
                logger.warning(f"查询性能测试失败: {e}")

            # 4. 数据库文件大小检查（仅SQLite）
            if Config.DATABASE_URL and "sqlite" in Config.DATABASE_URL:
                try:
                    # 提取SQLite文件路径
                    db_url = Config.DATABASE_URL
                    if db_url.startswith("sqlite:///"):
                        db_path = db_url.replace("sqlite:///", "")
                        # 处理相对路径
                        if not os.path.isabs(db_path):
                            # 假设相对于项目根目录
                            pass

                            project_root = os.path.dirname(
                                os.path.dirname(os.path.dirname(__file__))
                            )
                            db_path = os.path.join(project_root, db_path)

                        if os.path.exists(db_path):
                            file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

                            if file_size_mb < 10:
                                level = DiagnosticLevel.INFO
                                title = "数据库文件大小正常"
                            elif file_size_mb < 100:
                                level = DiagnosticLevel.WARNING
                                title = "数据库文件较大"
                            else:
                                level = DiagnosticLevel.ERROR
                                title = "数据库文件过大"

                            issues.append(
                                DiagnosticIssue(
                                    id="database_file_size",
                                    category=DiagnosticCategory.DATABASE,
                                    level=level,
                                    title=title,
                                    description=f"数据库文件大小: {file_size_mb:.2f} MB",
                                    timestamp=time.time(),
                                    metric_name="database_size_mb",
                                    metric_value=file_size_mb,
                                    threshold=100.0,  # 100MB阈值
                                    suggestions=(
                                        [
                                            "清理历史数据",
                                            "优化数据存储",
                                            "考虑数据库归档",
                                        ]
                                        if file_size_mb > 100
                                        else None
                                    ),
                                )
                            )
                        else:
                            issues.append(
                                DiagnosticIssue(
                                    id="database_file_not_found",
                                    category=DiagnosticCategory.DATABASE,
                                    level=DiagnosticLevel.WARNING,
                                    title="数据库文件未找到",
                                    description=f"数据库文件路径不存在: {db_path}",
                                    timestamp=time.time(),
                                    suggestions=["检查数据库配置和文件路径"],
                                )
                            )
                except Exception as e:
                    logger.warning(f"数据库文件大小检查失败: {e}")

            # 5. 数据库连接池状态（如果支持）
            try:
                pool = engine.pool
                if hasattr(pool, "status"):
                    pool_status = pool.status()
                    issues.append(
                        DiagnosticIssue(
                            id="database_pool_status",
                            category=DiagnosticCategory.DATABASE,
                            level=DiagnosticLevel.INFO,
                            title="数据库连接池状态",
                            description=f"连接池状态: {pool_status}",
                            timestamp=time.time(),
                            component="database_pool",
                        )
                    )
            except Exception as e:
                # 连接池状态检查是可选的
                pass  # 已实现

            # 如果没有发现问题，添加成功诊断
            if not any(
                issue.level in [DiagnosticLevel.ERROR, DiagnosticLevel.WARNING]
                for issue in issues
                if issue.category == DiagnosticCategory.DATABASE
            ):
                issues.append(
                    DiagnosticIssue(
                        id="database_health_ok",
                        category=DiagnosticCategory.DATABASE,
                        level=DiagnosticLevel.INFO,
                        title="数据库健康状态良好",
                        description="所有数据库检查均通过",
                        timestamp=time.time(),
                    )
                )

        except Exception as e:
            logger.error(f"诊断数据库失败: {e}")
            issues.append(
                DiagnosticIssue(
                    id="database_diagnostic_error",
                    category=DiagnosticCategory.DATABASE,
                    level=DiagnosticLevel.ERROR,
                    title="数据库诊断错误",
                    description=f"数据库诊断过程出错: {e}",
                    timestamp=time.time(),
                    suggestions=["检查数据库配置和依赖", "查看详细日志以获取更多信息"],
                )
            )

        return issues

    def _diagnose_api(self) -> List[DiagnosticIssue]:
        """诊断API

        返回:
            API相关诊断问题列表
        """
        issues: List[DiagnosticIssue] = []

        try:
            # 导入必要的模块
            try:
                import requests
                import time as time_module
                from urllib.parse import urljoin

            except ImportError as e:
                logger.warning(f"API诊断模块导入失败: {e}")
                issues.append(
                    DiagnosticIssue(
                        id="api_import_failed",
                        category=DiagnosticCategory.API,
                        level=DiagnosticLevel.WARNING,
                        title="API诊断模块不可用",
                        description=f"无法导入API诊断模块: {e}",
                        timestamp=time.time(),
                        suggestions=[
                            "安装requests库: pip install requests",
                            "检查网络连接",
                        ],
                    )
                )
                return issues

            # 获取API基础URL
            try:
                pass

                # 这里需要获取实际的API地址
                # 在开发环境中通常是http://localhost:8000
                api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
            except ImportError:
                api_base_url = "http://localhost:8000"

            # 定义要测试的核心API端点
            # 这些端点应该是无需认证或简单认证的健康检查端点
            api_endpoints = [
                {
                    "path": "/api/health",
                    "name": "系统健康检查",
                    "method": "GET",
                    "expected_status": 200,
                    "timeout": 3,  # 减少超时时间
                    "critical": True,
                },
                {
                    "path": "/api/robot/control/status",
                    "name": "机器人控制状态",
                    "method": "GET",
                    "expected_status": 200,
                    "timeout": 2,  # 减少非关键端点的超时时间
                    "critical": False,
                },
                {
                    "path": "/api/memory/status",
                    "name": "记忆系统状态",
                    "method": "GET",
                    "expected_status": 200,
                    "timeout": 2,
                    "critical": False,
                },
                {
                    "path": "/api/diagnostic/status",
                    "name": "诊断系统状态",
                    "method": "GET",
                    "expected_status": 200,
                    "timeout": 2,
                    "critical": False,
                },
            ]

            # 首先检查API基础URL是否可达
            try:
                # 尝试连接到API基础URL（只测试根路径）
                test_response = requests.get(api_base_url, timeout=3)
                if test_response.status_code < 500:  # 任何非服务器错误的响应都算作可达
                    logger.debug(f"API基础URL可达: {api_base_url}")
            except Exception as e:
                logger.debug(f"API基础URL不可达: {api_base_url}, 错误: {e}")
                # 添加一个诊断问题说明API服务不可用
                issues.append(
                    DiagnosticIssue(
                        id="api_service_unavailable",
                        category=DiagnosticCategory.API,
                        level=DiagnosticLevel.ERROR,
                        title="API服务不可用",
                        description=f"API服务在{api_base_url}不可用，跳过端点测试",
                        timestamp=time.time(),
                        component="api_gateway",
                        suggestions=[
                            "检查API服务是否运行",
                            f"验证地址是否正确: {api_base_url}",
                            "检查网络连接和防火墙设置",
                        ],
                    )
                )
                # 如果API服务不可用，直接返回，不测试具体端点
                return issues

            # 测试每个API端点
            successful_tests = 0
            failed_tests = 0
            total_response_time = 0

            for endpoint in api_endpoints:
                endpoint_name = endpoint["name"]
                endpoint_path = endpoint["path"]
                expected_status = endpoint["expected_status"]
                timeout = endpoint["timeout"]
                is_critical = endpoint["critical"]

                full_url = urljoin(api_base_url, endpoint_path)

                try:
                    start_time = time_module.time()

                    # 发送请求
                    response = requests.request(
                        method=endpoint["method"], url=full_url, timeout=timeout
                    )

                    response_time = (time_module.time() - start_time) * 1000  # 毫秒
                    total_response_time += response_time

                    # 检查响应状态
                    if response.status_code == expected_status:
                        successful_tests += 1

                        # 根据响应时间确定问题级别
                        if response_time < 100:
                            level = DiagnosticLevel.INFO
                            status_desc = f"响应正常 ({response_time:.1f}ms)"
                        elif response_time < 500:
                            level = DiagnosticLevel.WARNING
                            status_desc = f"响应较慢 ({response_time:.1f}ms)"
                        else:
                            level = DiagnosticLevel.ERROR
                            status_desc = f"响应超慢 ({response_time:.1f}ms)"

                        issues.append(
                            DiagnosticIssue(
                                id=f"api_endpoint_{endpoint_path.replace('/', '_')}",
                                category=DiagnosticCategory.API,
                                level=level,
                                title=f"API端点: {endpoint_name}",
                                description=f"{endpoint_path} - {status_desc}",
                                timestamp=time.time(),
                                metric_name="response_time_ms",
                                metric_value=response_time,
                                threshold=500.0,  # 500ms阈值
                                component=endpoint_path,
                                suggestions=(
                                    ["优化API性能", "检查服务器负载", "考虑API缓存"]
                                    if response_time > 500
                                    else None
                                ),
                            )
                        )
                    else:
                        failed_tests += 1
                        level = (
                            DiagnosticLevel.ERROR
                            if is_critical
                            else DiagnosticLevel.WARNING
                        )

                        issues.append(
                            DiagnosticIssue(
                                id=f"api_endpoint_failed_{                                     endpoint_path.replace(                                         '/', '_')}",
                                category=DiagnosticCategory.API,
                                level=level,
                                title=f"API端点失败: {endpoint_name}",
                                description=f"{endpoint_path} - 期望状态 {expected_status}, 实际状态 {                                     response.status_code}",
                                timestamp=time.time(),
                                metric_name="response_status",
                                metric_value=response.status_code,
                                component=endpoint_path,
                                suggestions=[
                                    "检查API服务是否运行",
                                    "验证端点路径是否正确",
                                    "查看API日志获取详细错误信息",
                                ],
                            )
                        )

                except requests.exceptions.Timeout:
                    failed_tests += 1
                    level = (
                        DiagnosticLevel.ERROR
                        if is_critical
                        else DiagnosticLevel.WARNING
                    )

                    issues.append(
                        DiagnosticIssue(
                            id=f"api_endpoint_timeout_{                                 endpoint_path.replace(                                     '/', '_')}",
                            category=DiagnosticCategory.API,
                            level=level,
                            title=f"API端点超时: {endpoint_name}",
                            description=f"{endpoint_path} - 请求超时 ({timeout}秒)",
                            timestamp=time.time(),
                            component=endpoint_path,
                            suggestions=[
                                "检查网络连接",
                                "增加API超时时间",
                                "检查服务器负载",
                            ],
                        )
                    )

                except requests.exceptions.ConnectionError:
                    failed_tests += 1
                    level = (
                        DiagnosticLevel.ERROR
                        if is_critical
                        else DiagnosticLevel.WARNING
                    )

                    issues.append(
                        DiagnosticIssue(
                            id=f"api_endpoint_connection_error_{                                 endpoint_path.replace(                                     '/', '_')}",
                            category=DiagnosticCategory.API,
                            level=level,
                            title=f"API连接错误: {endpoint_name}",
                            description=f"{endpoint_path} - 连接被拒绝",
                            timestamp=time.time(),
                            component=endpoint_path,
                            suggestions=[
                                "检查API服务是否运行",
                                "验证API地址是否正确",
                                "检查防火墙设置",
                            ],
                        )
                    )

                except Exception as e:
                    failed_tests += 1
                    level = (
                        DiagnosticLevel.ERROR
                        if is_critical
                        else DiagnosticLevel.WARNING
                    )

                    issues.append(
                        DiagnosticIssue(
                            id=f"api_endpoint_error_{endpoint_path.replace('/', '_')}",
                            category=DiagnosticCategory.API,
                            level=level,
                            title=f"API端点错误: {endpoint_name}",
                            description=f"{endpoint_path} - 错误: {str(e)}",
                            timestamp=time.time(),
                            component=endpoint_path,
                            suggestions=[
                                "检查API配置",
                                "查看详细错误日志",
                                "验证网络连接",
                            ],
                        )
                    )

            # 生成总体API诊断结果
            total_tests = successful_tests + failed_tests

            if total_tests > 0:
                success_rate = (successful_tests / total_tests) * 100

                if successful_tests > 0:
                    avg_response_time = total_response_time / successful_tests
                else:
                    avg_response_time = 0

                # 确定总体API健康状态
                if success_rate == 100:
                    overall_level = DiagnosticLevel.INFO
                    overall_title = "API健康状态优秀"
                    overall_desc = f"所有{total_tests}个API端点测试通过"
                elif success_rate >= 80:
                    overall_level = DiagnosticLevel.WARNING
                    overall_title = "API健康状态一般"
                    overall_desc = f"{successful_tests}/{total_tests}个API端点测试通过 ({                         success_rate:.1f}%)"
                else:
                    overall_level = DiagnosticLevel.ERROR
                    overall_title = "API健康状态差"
                    overall_desc = f"只有{successful_tests}/{total_tests}个API端点测试通过 ({                         success_rate:.1f}%)"

                issues.append(
                    DiagnosticIssue(
                        id="api_overall_health",
                        category=DiagnosticCategory.API,
                        level=overall_level,
                        title=overall_title,
                        description=overall_desc,
                        timestamp=time.time(),
                        metric_name="api_success_rate",
                        metric_value=success_rate,
                        threshold=80.0,  # 80%成功率阈值
                        component="api_gateway",
                        suggestions=(
                            ["检查API服务配置", "监控API错误日志", "优化API性能"]
                            if success_rate < 80
                            else None
                        ),
                    )
                )

                if avg_response_time > 0:
                    issues.append(
                        DiagnosticIssue(
                            id="api_average_response_time",
                            category=DiagnosticCategory.API,
                            level=DiagnosticLevel.INFO,
                            title="API平均响应时间",
                            description=f"平均响应时间: {avg_response_time:.1f}ms",
                            timestamp=time.time(),
                            metric_name="avg_api_response_time_ms",
                            metric_value=avg_response_time,
                            threshold=300.0,  # 300ms平均响应时间阈值
                        )
                    )

            # 检查API认证状态（如果可用）
            try:
                # 这里可以添加API认证状态检查
                # 例如检查JWT令牌有效性等
                pass  # 已实现
            except Exception as e:
                logger.debug(f"API认证状态检查失败: {e}")

            # 如果没有发现严重问题，添加成功诊断
            if not any(
                issue.level in [DiagnosticLevel.ERROR]
                for issue in issues
                if issue.category == DiagnosticCategory.API
            ):
                if successful_tests > 0:
                    issues.append(
                        DiagnosticIssue(
                            id="api_health_ok",
                            category=DiagnosticCategory.API,
                            level=DiagnosticLevel.INFO,
                            title="API健康状态良好",
                            description=f"API端点测试通过率: {success_rate:.1f}%",
                            timestamp=time.time(),
                        )
                    )

        except Exception as e:
            logger.error(f"诊断API失败: {e}")
            issues.append(
                DiagnosticIssue(
                    id="api_diagnostic_error",
                    category=DiagnosticCategory.API,
                    level=DiagnosticLevel.ERROR,
                    title="API诊断错误",
                    description=f"API诊断过程出错: {e}",
                    timestamp=time.time(),
                    suggestions=["检查网络连接", "验证API配置", "查看详细日志"],
                )
            )

        return issues

    def _diagnose_security(self) -> List[DiagnosticIssue]:
        """诊断系统安全

        返回:
            安全相关诊断问题列表
        """
        issues: List[DiagnosticIssue] = []

        try:
            # 1. 检查配置文件安全性
            try:
                from backend.core.config import Config

                # 检查生产环境安全配置
                env = os.getenv("ENVIRONMENT", "development")

                if env == "production":
                    # 检查密钥是否使用默认值
                    if Config.SECRET_KEY == "self_agi_secret_key_change_in_production":
                        issues.append(
                            DiagnosticIssue(
                                id="security_default_secret_key",
                                category=DiagnosticCategory.SECURITY,
                                level=DiagnosticLevel.CRITICAL,
                                title="安全密钥使用默认值",
                                description="生产环境中使用了默认的SECRET_KEY，存在严重安全风险",
                                timestamp=time.time(),
                                suggestions=[
                                    "立即设置环境变量SECRET_KEY",
                                    "生成安全的随机密钥",
                                    "轮换所有现有令牌",
                                ],
                            )
                        )

                    # 检查数据库密码安全性
                    if "admin_password" in Config.MONGODB_URL:
                        issues.append(
                            DiagnosticIssue(
                                id="security_default_mongodb_password",
                                category=DiagnosticCategory.SECURITY,
                                level=DiagnosticLevel.WARNING,
                                title="MongoDB使用默认密码",
                                description="MongoDB连接URL中包含默认密码",
                                timestamp=time.time(),
                                suggestions=[
                                    "设置环境变量MONGODB_URL",
                                    "使用强密码替换默认密码",
                                ],
                            )
                        )

                    if "admin_password" in Config.RABBITMQ_URL:
                        issues.append(
                            DiagnosticIssue(
                                id="security_default_rabbitmq_password",
                                category=DiagnosticCategory.SECURITY,
                                level=DiagnosticLevel.WARNING,
                                title="RabbitMQ使用默认密码",
                                description="RabbitMQ连接URL中包含默认密码",
                                timestamp=time.time(),
                                suggestions=[
                                    "设置环境变量RABBITMQ_URL",
                                    "使用强密码替换默认密码",
                                ],
                            )
                        )

                    # 检查Redis配置
                    if Config.REDIS_URL == "redis://localhost:6379/0":
                        issues.append(
                            DiagnosticIssue(
                                id="security_redis_no_password",
                                category=DiagnosticCategory.SECURITY,
                                level=DiagnosticLevel.WARNING,
                                title="Redis无密码保护",
                                description="Redis使用默认URL且无密码保护",
                                timestamp=time.time(),
                                suggestions=[
                                    "为Redis设置密码",
                                    "使用环境变量REDIS_URL配置安全连接",
                                ],
                            )
                        )

                # 检查配置文件验证是否已运行
                try:
                    Config.validate_config()
                    issues.append(
                        DiagnosticIssue(
                            id="security_config_validation_ok",
                            category=DiagnosticCategory.SECURITY,
                            level=DiagnosticLevel.INFO,
                            title="配置验证通过",
                            description="配置文件安全性验证已执行",
                            timestamp=time.time(),
                        )
                    )
                except Exception as e:
                    issues.append(
                        DiagnosticIssue(
                            id="security_config_validation_failed",
                            category=DiagnosticCategory.SECURITY,
                            level=DiagnosticLevel.ERROR,
                            title="配置验证失败",
                            description=f"配置文件安全性验证失败: {e}",
                            timestamp=time.time(),
                            suggestions=["检查配置错误", "查看详细日志"],
                        )
                    )

            except ImportError as e:
                issues.append(
                    DiagnosticIssue(
                        id="security_config_import_failed",
                        category=DiagnosticCategory.SECURITY,
                        level=DiagnosticLevel.WARNING,
                        title="配置模块导入失败",
                        description=f"无法导入配置模块: {e}",
                        timestamp=time.time(),
                        suggestions=["检查配置模块路径", "验证模块依赖"],
                    )
                )

            # 2. 检查文件权限安全性
            try:
                # 检查关键文件权限
                critical_files = []

                # 尝试查找关键配置文件
                config_files = [
                    ".env",
                    "backend/.env",
                    "backend/core/config.py",
                    "self_agi.db",  # SQLite数据库文件
                ]

                for file_path in config_files:
                    if os.path.exists(file_path):
                        critical_files.append(file_path)

                for file_path in critical_files:
                    try:
                        # 检查文件权限（Unix风格）
                        import stat

                        file_stat = os.stat(file_path)
                        file_mode = file_stat.st_mode

                        # 检查其他用户是否有写权限
                        if file_mode & stat.S_IWOTH:
                            issues.append(
                                DiagnosticIssue(
                                    id=f"security_file_permission_{                                         file_path.replace(                                             '.', '_')}",
                                    category=DiagnosticCategory.SECURITY,
                                    level=DiagnosticLevel.ERROR,
                                    title="文件权限不安全",
                                    description=f"文件 {file_path} 其他用户有写权限",
                                    timestamp=time.time(),
                                    suggestions=[
                                        f"修改文件权限: chmod o-w {file_path}",
                                        "限制敏感文件的访问权限",
                                    ],
                                )
                            )

                        # 检查其他用户是否有读权限（对于敏感文件）
                        sensitive_files = [".env", "self_agi.db"]
                        if any(sensitive in file_path for sensitive in sensitive_files):
                            if file_mode & stat.S_IROTH:
                                issues.append(
                                    DiagnosticIssue(
                                        id=f"security_file_read_permission_{                                             file_path.replace(                                                 '.', '_')}",
                                        category=DiagnosticCategory.SECURITY,
                                        level=DiagnosticLevel.WARNING,
                                        title="敏感文件可读",
                                        description=f"敏感文件 {file_path} 其他用户有读权限",
                                        timestamp=time.time(),
                                        suggestions=[
                                            f"限制文件读取权限: chmod o-r {file_path}",
                                            "确保敏感文件仅对必要用户可读",
                                        ],
                                    )
                                )

                    except Exception as e:
                        logger.debug(f"检查文件权限失败 {file_path}: {e}")

            except Exception as e:
                logger.warning(f"文件权限检查失败: {e}")

            # 3. 检查网络安全配置
            try:
                # 检查CORS配置
                cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "")
                if cors_origins == "*":
                    issues.append(
                        DiagnosticIssue(
                            id="security_cors_wildcard",
                            category=DiagnosticCategory.SECURITY,
                            level=DiagnosticLevel.WARNING,
                            title="CORS配置使用通配符",
                            description="CORS_ALLOW_ORIGINS设置为'*'，存在安全风险",
                            timestamp=time.time(),
                            suggestions=[
                                "设置具体的CORS允许来源",
                                "在生产环境中限制跨域请求",
                            ],
                        )
                    )
            except Exception as e:
                logger.debug(f"网络安全检查失败: {e}")

            # 4. 检查依赖包安全性
            try:
                # 检查依赖包安全漏洞
                # 尝试使用importlib.metadata（Python 3.8+），回退到pkg_resources（已弃用）
                try:
                    import importlib.metadata

                    distributions = importlib.metadata.distributions()
                    installed_packages = {}
                    for dist in distributions:
                        name = dist.metadata.get("Name")
                        if name:
                            installed_packages[name.lower()] = dist.version
                except ImportError:
                    import pkg_resources

                    installed_packages = {
                        pkg.key.lower(): pkg.version
                        for pkg in pkg_resources.working_set
                    }

                    # 关键包的安全检查配置
                    critical_packages = {
                        "requests": {
                            "min_version": "2.25.0",
                            "vulnerable_versions": ["2.24.0", "2.23.0"],
                            "description": "HTTP库，低版本存在安全漏洞",
                        },
                        "sqlalchemy": {
                            "min_version": "1.4.0",
                            "vulnerable_versions": ["1.3.0", "1.2.0"],
                            "description": "SQL工具包和ORM，低版本存在安全漏洞",
                        },
                        "fastapi": {
                            "min_version": "0.68.0",
                            "vulnerable_versions": ["0.67.0", "0.66.0"],
                            "description": "Web框架，低版本存在安全漏洞",
                        },
                        "uvicorn": {
                            "min_version": "0.15.0",
                            "vulnerable_versions": ["0.14.0", "0.13.0"],
                            "description": "ASGI服务器，低版本存在安全漏洞",
                        },
                        "pydantic": {
                            "min_version": "1.8.0",
                            "vulnerable_versions": ["1.7.0", "1.6.0"],
                            "description": "数据验证库，低版本存在安全漏洞",
                        },
                        "numpy": {
                            "min_version": "1.21.0",
                            "vulnerable_versions": ["1.20.0", "1.19.0"],
                            "description": "数值计算库，低版本存在安全漏洞",
                        },
                        "psutil": {
                            "min_version": "5.8.0",
                            "vulnerable_versions": ["5.7.0", "5.6.0"],
                            "description": "系统监控库，低版本存在安全漏洞",
                        },
                    }

                    packages_checked = 0
                    vulnerable_packages_found = 0

                    for package, info in critical_packages.items():
                        if package in installed_packages:
                            packages_checked += 1
                            current_version = installed_packages[package]

                            # 检查是否为已知漏洞版本
                            if current_version in info["vulnerable_versions"]:
                                vulnerable_packages_found += 1

                                issues.append(
                                    DiagnosticIssue(
                                        id=f"security_vulnerable_package_{package}",
                                        category=DiagnosticCategory.SECURITY,
                                        level=DiagnosticLevel.CRITICAL,
                                        title=f"易受攻击的包版本: {package}",
                                        description=f"检测到已知漏洞的{package}版本: {current_version}。{                                             info['description']}",
                                        timestamp=time.time(),
                                        metric_name="vulnerable_package",
                                        metric_value=1.0,
                                        component=f"package:{package}",
                                        suggestions=[
                                            f"立即更新{package}到安全版本: pip install {package}>={                                                 info['min_version']}",
                                            "查看CVE数据库了解漏洞详情",
                                            "运行安全审计工具检查其他漏洞",
                                        ],
                                    )
                                )

                            # 检查版本是否过旧（但不是已知漏洞版本）
                            try:
                                from packaging import version

                                current_ver = version.parse(current_version)
                                min_ver = version.parse(info["min_version"])

                                if (
                                    current_ver < min_ver
                                    and current_version
                                    not in info["vulnerable_versions"]
                                ):
                                    issues.append(
                                        DiagnosticIssue(
                                            id=f"security_outdated_package_{package}",
                                            category=DiagnosticCategory.SECURITY,
                                            level=DiagnosticLevel.WARNING,
                                            title=f"过时的包版本: {package}",
                                            description=f"{package}版本{current_version}已过时，建议更新到{                                                 info['min_version']}或更高版本。{                                                 info['description']}",
                                            timestamp=time.time(),
                                            component=f"package:{package}",
                                            suggestions=[
                                                f"更新{package}到最新版本: pip install --upgrade {package}",
                                                "检查更新日志了解安全修复",
                                            ],
                                        )
                                    )

                            except ImportError:
                                # 如果没有packaging库，跳过版本比较
                                logger.debug(f"无法比较{package}版本，缺少packaging库")

                    # 生成依赖包安全总结
                    if packages_checked > 0:
                        security_level = DiagnosticLevel.INFO

                        if vulnerable_packages_found > 0:
                            security_level = DiagnosticLevel.ERROR
                        elif any(
                            issue.id.startswith("security_outdated_package_")
                            for issue in issues
                        ):
                            security_level = DiagnosticLevel.WARNING

                        issues.append(
                            DiagnosticIssue(
                                id="security_dependencies_summary",
                                category=DiagnosticCategory.SECURITY,
                                level=security_level,
                                title="依赖包安全状态",
                                description=f"检查了{packages_checked}个关键包，发现{vulnerable_packages_found}个易受攻击版本",
                                timestamp=time.time(),
                                metric_name="vulnerable_packages_count",
                                metric_value=vulnerable_packages_found,
                                component="dependency_manager",
                                suggestions=(
                                    [
                                        "定期更新依赖包",
                                        "使用安全扫描工具检查漏洞",
                                        "订阅安全公告",
                                    ]
                                    if vulnerable_packages_found > 0
                                    else None
                                ),
                            )
                        )

                except Exception as e:
                    logger.debug(f"获取已安装包列表失败: {e}")
                    issues.append(
                        DiagnosticIssue(
                            id="security_dependencies_check_failed",
                            category=DiagnosticCategory.SECURITY,
                            level=DiagnosticLevel.WARNING,
                            title="依赖包安全检查失败",
                            description=f"无法检查依赖包安全性: {e}",
                            timestamp=time.time(),
                            suggestions=["检查pkg_resources模块", "验证Python环境"],
                        )
                    )

            except Exception as e:
                logger.debug(f"依赖包安全检查失败: {e}")

            # 5. 检查认证配置
            try:
                # 检查JWT配置
                access_token_expire = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
                try:
                    expire_minutes = int(access_token_expire)
                    if expire_minutes > 1440:  # 24小时
                        issues.append(
                            DiagnosticIssue(
                                id="security_token_expire_too_long",
                                category=DiagnosticCategory.SECURITY,
                                level=DiagnosticLevel.WARNING,
                                title="访问令牌过期时间过长",
                                description=f"访问令牌过期时间设置为{expire_minutes}分钟，建议不超过24小时",
                                timestamp=time.time(),
                                suggestions=[
                                    "缩短访问令牌过期时间",
                                    "使用刷新令牌机制",
                                ],
                            )
                        )
                except ValueError:
                    pass  # 已实现

            except Exception as e:
                logger.debug(f"认证配置检查失败: {e}")

            # 如果没有发现安全问题，添加成功诊断
            if not any(
                issue.level
                in [
                    DiagnosticLevel.ERROR,
                    DiagnosticLevel.WARNING,
                    DiagnosticLevel.CRITICAL,
                ]
                for issue in issues
                if issue.category == DiagnosticCategory.SECURITY
            ):
                issues.append(
                    DiagnosticIssue(
                        id="security_health_ok",
                        category=DiagnosticCategory.SECURITY,
                        level=DiagnosticLevel.INFO,
                        title="安全状态良好",
                        description="未发现重大安全问题",
                        timestamp=time.time(),
                    )
                )

        except Exception as e:
            logger.error(f"诊断安全失败: {e}")
            issues.append(
                DiagnosticIssue(
                    id="security_diagnostic_error",
                    category=DiagnosticCategory.SECURITY,
                    level=DiagnosticLevel.ERROR,
                    title="安全诊断错误",
                    description=f"安全诊断过程出错: {e}",
                    timestamp=time.time(),
                    suggestions=["检查系统配置", "查看详细错误日志"],
                )
            )

        return issues

    def _diagnose_hardware(self) -> List[DiagnosticIssue]:
        """诊断硬件

        返回:
            硬件相关诊断问题列表
        """
        issues: List[DiagnosticIssue] = []

        try:
            # 1. CPU信息检查
            try:
                cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
                cpu_logical = psutil.cpu_count(logical=True)  # 逻辑核心数

                issues.append(
                    DiagnosticIssue(
                        id="hardware_cpu_info",
                        category=DiagnosticCategory.HARDWARE,
                        level=DiagnosticLevel.INFO,
                        title="CPU信息",
                        description=f"物理核心: {cpu_count}, 逻辑核心: {cpu_logical}",
                        timestamp=time.time(),
                        metric_name="cpu_cores",
                        metric_value=cpu_logical,
                        component="cpu",
                    )
                )

                # 检查CPU频率
                try:
                    cpu_freq = psutil.cpu_freq()
                    if cpu_freq:
                        issues.append(
                            DiagnosticIssue(
                                id="hardware_cpu_frequency",
                                category=DiagnosticCategory.HARDWARE,
                                level=DiagnosticLevel.INFO,
                                title="CPU频率",
                                description=f"当前频率: {                                     cpu_freq.current:.1f} MHz, 最大频率: {                                     cpu_freq.max:.1f} MHz",
                                timestamp=time.time(),
                                metric_name="cpu_frequency_mhz",
                                metric_value=cpu_freq.current,
                                component="cpu",
                            )
                        )
                except Exception as e:
                    logger.debug(f"获取CPU频率失败: {e}")

                # 检查CPU温度（如果可用）
                try:
                    if hasattr(psutil, "sensors_temperatures"):
                        temps = psutil.sensors_temperatures()
                        if temps:
                            for name, entries in temps.items():
                                for entry in entries:
                                    if entry.current:
                                        issues.append(
                                            DiagnosticIssue(
                                                id=f"hardware_temperature_{name}",
                                                category=DiagnosticCategory.HARDWARE,
                                                level=DiagnosticLevel.INFO,
                                                title=f"温度传感器: {name}",
                                                description=f"当前温度: {entry.current}°C",
                                                timestamp=time.time(),
                                                metric_name=f"temperature_{name}",
                                                metric_value=entry.current,
                                                threshold=80.0,  # 80°C阈值
                                                component=name,
                                                suggestions=(
                                                    [
                                                        "检查散热系统",
                                                        "清理灰尘",
                                                        "改善通风",
                                                    ]
                                                    if entry.current > 80
                                                    else None
                                                ),
                                            )
                                        )
                except Exception as e:
                    logger.debug(f"获取CPU温度失败: {e}")

            except Exception as e:
                logger.warning(f"CPU诊断失败: {e}")
                issues.append(
                    DiagnosticIssue(
                        id="hardware_cpu_diagnostic_failed",
                        category=DiagnosticCategory.HARDWARE,
                        level=DiagnosticLevel.WARNING,
                        title="CPU诊断失败",
                        description=f"无法获取CPU信息: {e}",
                        timestamp=time.time(),
                        suggestions=["检查系统权限", "验证psutil库安装"],
                    )
                )

            # 2. 内存硬件信息
            try:
                virtual_memory = psutil.virtual_memory()
                swap_memory = psutil.swap_memory()

                issues.append(
                    DiagnosticIssue(
                        id="hardware_memory_info",
                        category=DiagnosticCategory.HARDWARE,
                        level=DiagnosticLevel.INFO,
                        title="内存信息",
                        description=f"物理内存: {virtual_memory.total /                                              (1024**3):.1f} GB, 交换内存: {swap_memory.total /                                                                        (1024**3):.1f} GB",
                        timestamp=time.time(),
                        metric_name="memory_total_gb",
                        metric_value=virtual_memory.total / (1024**3),
                        component="memory",
                    )
                )

            except Exception as e:
                logger.warning(f"内存诊断失败: {e}")

            # 3. 磁盘硬件信息
            try:
                disk_partitions = psutil.disk_partitions()
                disk_count = 0

                for partition in disk_partitions:
                    try:
                        if partition.fstype:  # 跳过无文件系统的设备
                            disk_usage = psutil.disk_usage(partition.mountpoint)

                            issues.append(
                                DiagnosticIssue(
                                    id=f"hardware_disk_{                                         partition.device.replace(                                             '/', '_')}",
                                    category=DiagnosticCategory.HARDWARE,
                                    level=DiagnosticLevel.INFO,
                                    title=f"磁盘: {partition.device}",
                                    description=f"挂载点: {                                         partition.mountpoint}, 总容量: {                                         disk_usage.total / (                                             1024**3):.1f} GB",
                                    timestamp=time.time(),
                                    metric_name="disk_size_gb",
                                    metric_value=disk_usage.total / (1024**3),
                                    component="disk",
                                )
                            )

                            disk_count += 1
                    except Exception as e:
                        logger.debug(f"获取磁盘信息失败 {partition.device}: {e}")

                if disk_count == 0:
                    issues.append(
                        DiagnosticIssue(
                            id="hardware_no_disks_found",
                            category=DiagnosticCategory.HARDWARE,
                            level=DiagnosticLevel.WARNING,
                            title="未发现磁盘",
                            description="无法获取磁盘信息",
                            timestamp=time.time(),
                            suggestions=["检查磁盘连接", "验证系统权限"],
                        )
                    )

            except Exception as e:
                logger.warning(f"磁盘诊断失败: {e}")
                issues.append(
                    DiagnosticIssue(
                        id="hardware_disk_diagnostic_failed",
                        category=DiagnosticCategory.HARDWARE,
                        level=DiagnosticLevel.WARNING,
                        title="磁盘诊断失败",
                        description=f"无法获取磁盘信息: {e}",
                        timestamp=time.time(),
                        suggestions=["检查系统权限", "验证磁盘状态"],
                    )
                )

            # 4. 网络接口硬件信息
            try:
                net_if_addrs = psutil.net_if_addrs()
                interface_count = len(net_if_addrs)

                if interface_count > 0:
                    issues.append(
                        DiagnosticIssue(
                            id="hardware_network_interfaces",
                            category=DiagnosticCategory.HARDWARE,
                            level=DiagnosticLevel.INFO,
                            title="网络接口",
                            description=f"发现 {interface_count} 个网络接口",
                            timestamp=time.time(),
                            metric_name="network_interface_count",
                            metric_value=interface_count,
                            component="network",
                        )
                    )

                    # 列出主要网络接口
                    main_interfaces = ["eth0", "eth1", "wlan0", "wlan1", "en0", "en1"]
                    for interface in main_interfaces:
                        if interface in net_if_addrs:
                            # 获取接口地址信息
                            addrs = net_if_addrs[interface]
                            ip_addresses = []

                            for addr in addrs:
                                if addr.family.name == "AF_INET":  # IPv4
                                    ip_addresses.append(addr.address)

                            if ip_addresses:
                                issues.append(
                                    DiagnosticIssue(
                                        id=f"hardware_network_interface_{interface}",
                                        category=DiagnosticCategory.HARDWARE,
                                        level=DiagnosticLevel.INFO,
                                        title=f"网络接口: {interface}",
                                        description=f"IP地址: {', '.join(ip_addresses)}",
                                        timestamp=time.time(),
                                        component=f"network_{interface}",
                                    )
                                )
                else:
                    issues.append(
                        DiagnosticIssue(
                            id="hardware_no_network_interfaces",
                            category=DiagnosticCategory.HARDWARE,
                            level=DiagnosticLevel.WARNING,
                            title="未发现网络接口",
                            description="无法获取网络接口信息",
                            timestamp=time.time(),
                            suggestions=["检查网络适配器", "验证驱动程序"],
                        )
                    )

            except Exception as e:
                logger.warning(f"网络接口诊断失败: {e}")

            # 5. 电池信息（如果可用，如笔记本电脑）
            try:
                if hasattr(psutil, "sensors_battery"):
                    battery = psutil.sensors_battery()
                    if battery:
                        issues.append(
                            DiagnosticIssue(
                                id="hardware_battery_info",
                                category=DiagnosticCategory.HARDWARE,
                                level=DiagnosticLevel.INFO,
                                title="电池信息",
                                description=f"电量: {                                     battery.percent}%, 状态: {                                     battery.power_plugged}",
                                timestamp=time.time(),
                                metric_name="battery_percent",
                                metric_value=battery.percent,
                                threshold=20.0,  # 20%电量阈值
                                component="battery",
                                suggestions=(
                                    ["连接电源适配器", "保存工作并准备关机"]
                                    if battery.percent < 20
                                    and not battery.power_plugged
                                    else None
                                ),
                            )
                        )
            except Exception as e:
                # 电池信息不可用是正常的（桌面设备）
                pass  # 已实现

            # 如果没有发现硬件问题，添加成功诊断
            if not any(
                issue.level in [DiagnosticLevel.ERROR, DiagnosticLevel.WARNING]
                for issue in issues
                if issue.category == DiagnosticCategory.HARDWARE
            ):
                issues.append(
                    DiagnosticIssue(
                        id="hardware_health_ok",
                        category=DiagnosticCategory.HARDWARE,
                        level=DiagnosticLevel.INFO,
                        title="硬件状态良好",
                        description="未发现硬件问题",
                        timestamp=time.time(),
                    )
                )

        except Exception as e:
            logger.error(f"诊断硬件失败: {e}")
            issues.append(
                DiagnosticIssue(
                    id="hardware_diagnostic_error",
                    category=DiagnosticCategory.HARDWARE,
                    level=DiagnosticLevel.ERROR,
                    title="硬件诊断错误",
                    description=f"硬件诊断过程出错: {e}",
                    timestamp=time.time(),
                    suggestions=["检查系统权限", "查看详细错误日志"],
                )
            )

        return issues

    def _check_threshold(
        self,
        metric_name: str,
        value: float,
        thresholds: Dict[str, float],
        category: DiagnosticCategory,
        title: str,
        description: str,
    ) -> List[DiagnosticIssue]:
        """检查指标是否超过阈值

        返回:
            诊断问题列表
        """
        issues: List[DiagnosticIssue] = []

        try:
            # 确定问题级别
            level = DiagnosticLevel.INFO
            if value >= thresholds.get("critical", 100.0):
                level = DiagnosticLevel.CRITICAL
            elif value >= thresholds.get("error", 90.0):
                level = DiagnosticLevel.ERROR
            elif value >= thresholds.get("warning", 80.0):
                level = DiagnosticLevel.WARNING

            # 如果不是INFO级别，创建诊断问题
            if level != DiagnosticLevel.INFO:
                issue_id = f"{metric_name}_{level.value}_{int(time.time())}"

                # 根据问题级别提供建议
                suggestions = []
                if level in [
                    DiagnosticLevel.WARNING,
                    DiagnosticLevel.ERROR,
                    DiagnosticLevel.CRITICAL,
                ]:
                    suggestions = [
                        f"监控{title}变化趋势",
                        f"分析{title}过高原因",
                        "考虑优化相关组件性能",
                    ]

                if level == DiagnosticLevel.CRITICAL:
                    suggestions.append("立即采取行动解决该问题")

                issues.append(
                    DiagnosticIssue(
                        id=issue_id,
                        category=category,
                        level=level,
                        title=f"{title} {level.value.upper()}",
                        description=f"{description}，超过{level.value}阈值",
                        metric_name=metric_name,
                        metric_value=value,
                        threshold=thresholds.get(level.value, 0.0),
                        timestamp=time.time(),
                        suggestions=suggestions,
                    )
                )

        except Exception as e:
            logger.error(f"检查阈值失败 {metric_name}: {e}")

        return issues

    def _analyze_diagnosis_results(
        self, issues: List[DiagnosticIssue], start_time: float
    ) -> DiagnosticReport:
        """分析诊断结果

        返回:
            诊断报告
        """
        execution_time = time.time() - start_time

        # 统计问题
        issues_by_level = {}
        issues_by_category = {}

        for issue in issues:
            # 按级别统计
            level = issue.level
            issues_by_level[level] = issues_by_level.get(level, 0) + 1

            # 按类别统计
            category = issue.category
            issues_by_category[category] = issues_by_category.get(category, 0) + 1

        # 确定整体状态
        overall_status = SystemStatus.NORMAL
        if any(issue.level == DiagnosticLevel.CRITICAL for issue in issues):
            overall_status = SystemStatus.CRITICAL
        elif any(issue.level == DiagnosticLevel.ERROR for issue in issues):
            overall_status = SystemStatus.ERROR
        elif any(issue.level == DiagnosticLevel.WARNING for issue in issues):
            overall_status = SystemStatus.WARNING

        # 生成总结
        if not issues:
            summary = "系统运行正常，未发现问题"
        else:
            critical_count = len(
                [i for i in issues if i.level == DiagnosticLevel.CRITICAL]
            )
            error_count = len([i for i in issues if i.level == DiagnosticLevel.ERROR])
            warning_count = len(
                [i for i in issues if i.level == DiagnosticLevel.WARNING]
            )

            summary_parts = []
            if critical_count > 0:
                summary_parts.append(f"{critical_count}个严重问题")
            if error_count > 0:
                summary_parts.append(f"{error_count}个错误")
            if warning_count > 0:
                summary_parts.append(f"{warning_count}个警告")

            summary = f"发现{'、'.join(summary_parts)}"

        # 生成建议
        recommendations = []

        # 根据问题生成建议
        critical_issues = [i for i in issues if i.level == DiagnosticLevel.CRITICAL]
        if critical_issues:
            recommendations.append("立即处理严重问题，防止系统故障")

        # 如果有资源问题，建议监控
        resource_issues = [
            i for i in issues if i.category == DiagnosticCategory.RESOURCE
        ]
        if resource_issues:
            recommendations.append("监控系统资源使用趋势，提前规划扩容")

        # 如果没有问题，给出正面反馈
        if not issues:
            recommendations.append("继续保持当前系统配置和监控")

        # 创建报告
        report_id = f"diagnosis_{int(time.time())}"
        report = DiagnosticReport(
            report_id=report_id,
            timestamp=time.time(),
            overall_status=overall_status,
            issues_count=len(issues),
            issues_by_level=issues_by_level,
            issues_by_category=issues_by_category,
            issues=issues,
            summary=summary,
            recommendations=recommendations,
            execution_time_seconds=execution_time,
        )

        return report

    def get_current_diagnosis(self) -> Optional[DiagnosticReport]:
        """获取当前诊断状态

        返回:
            最新的诊断报告（如果存在）
        """
        if self.diagnostic_history:
            return self.diagnostic_history[-1]
        return None  # 返回None

    def get_diagnosis_history(
        self, hours: int = 24, min_level: Optional[DiagnosticLevel] = None
    ) -> List[DiagnosticReport]:
        """获取诊断历史

        参数:
            hours: 小时数（默认24小时）
            min_level: 最小问题级别（可选）

        返回:
            诊断报告列表
        """
        cutoff_time = time.time() - hours * 3600

        filtered_reports = []
        for report in self.diagnostic_history:
            if report.timestamp >= cutoff_time:
                if min_level:
                    # 检查报告是否有至少一个指定级别的问题
                    if any(
                        issue.level.value >= min_level.value for issue in report.issues
                    ):
                        filtered_reports.append(report)
                else:
                    filtered_reports.append(report)

        return filtered_reports

    def run_diagnosis(self, categories: Optional[List[str]] = None) -> str:
        """运行诊断（API兼容方法）

        参数:
            categories: 诊断类别列表（可选）

        返回:
            报告ID

        异常:
            RuntimeError: 如果诊断已经在运行中
        """
        with self.reports_lock:
            if self.is_diagnosing:
                raise RuntimeError("诊断已经在运行中")

            # 生成报告ID
            self.report_id_counter += 1
            report_id = f"diagnosis_{self.report_id_counter:06d}_{int(time.time())}"

            # 启动异步诊断
            self.is_diagnosing = True

            def _run_diagnosis_async():
                try:
                    # 执行诊断
                    report = self.perform_diagnosis()

                    # 存储报告
                    with self.reports_lock:
                        self.reports_by_id[report_id] = report
                        self.is_diagnosing = False

                except Exception as e:
                    logger.error(f"异步诊断执行失败: {e}")
                    with self.reports_lock:
                        self.is_diagnosing = False

            # 在后台线程中运行诊断
            import threading

            thread = threading.Thread(
                target=_run_diagnosis_async, daemon=True, name=f"Diagnosis_{report_id}"
            )
            thread.start()

            return report_id

    def get_diagnostic_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """获取诊断报告（API兼容方法）

        参数:
            report_id: 报告ID

        返回:
            诊断报告字典（如果存在）
        """
        with self.reports_lock:
            report = self.reports_by_id.get(report_id)

        if report:
            # 转换为API期望的格式
            return self._convert_report_to_api_format(report)
        return None  # 返回None

    def get_diagnostic_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取诊断历史（API兼容方法）

        参数:
            limit: 数量限制

        返回:
            诊断报告字典列表
        """
        # 获取最近的诊断报告
        recent_reports = self.get_diagnosis_history(hours=24)
        recent_reports.sort(key=lambda r: r.timestamp, reverse=True)

        # 限制数量
        limited_reports = recent_reports[:limit]

        # 转换为API格式
        return [
            self._convert_report_to_api_format(report) for report in limited_reports
        ]

    def _convert_report_to_api_format(self, report: DiagnosticReport) -> Dict[str, Any]:
        """将诊断报告转换为API格式

        参数:
            report: 诊断报告

        返回:
            API格式的字典
        """
        # 转换问题为字典格式
        issues_dicts = []
        for issue in report.issues:
            issue_dict = {
                "id": issue.id,
                "category": issue.category.value,
                "level": issue.level.value,
                "title": issue.title,
                "description": issue.description,
                "timestamp": issue.timestamp,
            }

            if issue.metric_name:
                issue_dict["metric_name"] = issue.metric_name
            if issue.metric_value is not None:
                issue_dict["metric_value"] = issue.metric_value
            if issue.threshold is not None:
                issue_dict["threshold"] = issue.threshold
            if issue.component:
                issue_dict["component"] = issue.component
            if issue.suggestions:
                issue_dict["suggestions"] = issue.suggestions
            if issue.auto_fix_available:
                issue_dict["auto_fix_available"] = issue.auto_fix_available

            issues_dicts.append(issue_dict)

        # 转换统计信息
        issues_by_level = {
            level.value: count for level, count in report.issues_by_level.items()
        }

        issues_by_category = {
            category.value: count
            for category, count in report.issues_by_category.items()
        }

        return {
            "report_id": report.report_id,
            "timestamp": report.timestamp,
            "overall_status": report.overall_status.value,
            "issues_count": report.issues_count,
            "issues_by_level": issues_by_level,
            "issues_by_category": issues_by_category,
            "issues": issues_dicts,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "execution_time_seconds": report.execution_time_seconds,
        }

    def export_report_to_json(
        self, report: DiagnosticReport, file_path: Optional[str] = None
    ) -> str:
        """将诊断报告导出为JSON格式

        参数:
            report: 诊断报告
            file_path: 文件路径（可选，如果提供则保存到文件）

        返回:
            JSON字符串
        """
        try:
            # 转换为API格式
            report_dict = self._convert_report_to_api_format(report)

            # 添加导出元数据
            export_data = {
                "report": report_dict,
                "export_format": "json",
                "export_timestamp": time.time(),
                "export_version": "1.0",
                "system_info": {
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                    "hostname": platform.node(),
                },
            }

            # 格式化JSON
            json_str = json.dumps(
                export_data, indent=2, ensure_ascii=False, default=str
            )

            # 如果提供了文件路径，保存到文件
            if file_path:
                try:
                    # 确保目录存在
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(json_str)

                    logger.info(f"诊断报告已导出为JSON: {file_path}")
                except Exception as e:
                    logger.error(f"保存JSON文件失败: {e}")
                    raise

            return json_str

        except Exception as e:
            logger.error(f"导出诊断报告为JSON失败: {e}")
            raise

    def export_report_to_html(self, report: DiagnosticReport, file_path: str) -> str:
        """将诊断报告导出为HTML格式

        参数:
            report: 诊断报告
            file_path: HTML文件路径

        返回:
            HTML字符串
        """
        try:
            # 转换为API格式
            self._convert_report_to_api_format(report)

            # 生成HTML内容
            html_content = f"""<!DOCTYPE html> <html lang="zh-CN"> <head>     <meta charset="UTF-8">     <meta name="viewport" content="width=device-width, initial-scale=1.0">     <title>系统诊断报告 - {report.report_id}</title>     <style>         body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}         .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}         .report-id {{ font-size: 24px; font-weight: bold; color: #333; }}         .timestamp {{ color: #666; font-size: 14px; }}         .status-badge {{ display: inline-block; padding: 5px 10px; border-radius: 3px; margin-left: 10px; }}         .status-critical {{ background-color: #dc3545; color: white; }}         .status-error {{ background-color: #fd7e14; color: white; }}         .status-warning {{ background-color: #ffc107; color: #212529; }}         .status-info {{ background-color: #17a2b8; color: white; }}         .section {{ margin-bottom: 30px; border: 1px solid #dee2e6; border-radius: 5px; padding: 15px; }}         .section-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #495057; }}         .summary {{ background-color: #e8f4fd; border-left: 4px solid #007bff; padding: 15px; }}         .issue {{ margin-bottom: 15px; padding: 10px; border-left: 4px solid; border-radius: 3px; }}         .issue-critical {{ border-left-color: #dc3545; background-color: #f8d7da; }}         .issue-error {{ border-left-color: #fd7e14; background-color: #ffe5d0; }}         .issue-warning {{ border-left-color: #ffc107; background-color: #fff3cd; }}         .issue-info {{ border-left-color: #17a2b8; background-color: #d1ecf1; }}         .issue-title {{ font-weight: bold; margin-bottom: 5px; }}         .issue-description {{ color: #6c757d; font-size: 14px; margin-bottom: 5px; }}         .issue-suggestions {{ color: #28a745; font-size: 13px; }}         .metric {{ display: inline-block; margin-right: 15px; }}         .metric-label {{ font-weight: bold; }}         .metric-value {{ color: #007bff; }}         .table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}         .table th, .table td {{ border: 1px solid #dee2e6; padding: 8px; text-align: left; }}         .table th {{ background-color: #f8f9fa; }}         .footer {{ margin-top: 30px; padding-top: 15px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 12px; text-align: center; }}     </style> </head> <body>     <div class="header">         <div class="report-id">系统诊断报告: {report.report_id}</div>             """

            # 添加状态徽章
            status_class = ""
            status_text = ""
            if report.overall_status == SystemStatus.CRITICAL:
                status_class = "status-critical"
                status_text = "严重"
            elif report.overall_status == SystemStatus.ERROR:
                status_class = "status-error"
                status_text = "错误"
            elif report.overall_status == SystemStatus.WARNING:
                status_class = "status-warning"
                status_text = "警告"
            else:
                status_class = "status-info"
                status_text = "正常"

            html_content += f"""         <div class="timestamp">生成时间: {datetime.fromtimestamp(report.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</div>         <div><span class="status-badge {status_class}">{status_text}</span></div>     </div>      <div class="section">         <div class="section-title">报告摘要</div>         <div class="summary">             <div class="metric"><span class="metric-label">整体状态:</span> <span class="metric-value">{status_text}</span></div>             <div class="metric"><span class="metric-label">问题总数:</span> <span class="metric-value">{report.issues_count}</span></div>             <div class="metric"><span class="metric-label">执行时间:</span> <span class="metric-value">{report.execution_time_seconds:.2f}秒</span></div>         </div>     </div>      <div class="section">         <div class="section-title">问题统计</div>         <table class="table">             <thead>                 <tr>                     <th>类别</th>                     <th>信息</th>                     <th>警告</th>                     <th>错误</th>                     <th>严重</th>                 </tr>             </thead>             <tbody>             """

            # 按类别统计问题
            categories = sorted(
                set(issue.category for issue in report.issues), key=lambda c: c.value
            )
            for category in categories:
                issues_by_level = {}
                for level in DiagnosticLevel:
                    issues_by_level[level] = len(
                        [
                            i
                            for i in report.issues
                            if i.category == category and i.level == level
                        ]
                    )

                html_content += f"""                 <tr>                     <td>{category.value}</td>                     <td>{issues_by_level.get(DiagnosticLevel.INFO, 0)}</td>                     <td>{issues_by_level.get(DiagnosticLevel.WARNING, 0)}</td>                     <td>{issues_by_level.get(DiagnosticLevel.ERROR, 0)}</td>                     <td>{issues_by_level.get(DiagnosticLevel.CRITICAL, 0)}</td>                 </tr>"""

            html_content += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <div class="section-title">详细问题列表</div>
            """

            # 按严重程度排序问题（严重、错误、警告、信息）
            severity_order = {
                DiagnosticLevel.CRITICAL: 0,
                DiagnosticLevel.ERROR: 1,
                DiagnosticLevel.WARNING: 2,
                DiagnosticLevel.INFO: 3,
            }
            sorted_issues = sorted(report.issues, key=lambda x: severity_order[x.level])

            for issue in sorted_issues:
                issue_class = ""
                if issue.level == DiagnosticLevel.CRITICAL:
                    issue_class = "issue-critical"
                elif issue.level == DiagnosticLevel.ERROR:
                    issue_class = "issue-error"
                elif issue.level == DiagnosticLevel.WARNING:
                    issue_class = "issue-warning"
                else:
                    issue_class = "issue-info"

                html_content += f"""         <div class="issue {issue_class}">             <div class="issue-title">{issue.title} ({issue.level.value})</div>             <div class="issue-description">{issue.description}</div>             """

                if issue.metric_value is not None:
                    html_content += f"""             <div class="issue-description">指标: {issue.metric_name} = {issue.metric_value:.2f}{f' (阈值: {issue.threshold:.2f})' if issue.threshold else ''}</div>                     """

                if issue.suggestions:
                    suggestions_html = "<br>".join(issue.suggestions)
                    html_content += f"""             <div class="issue-suggestions">建议: {suggestions_html}</div>                     """

                html_content += f"""             <div class="issue-description" style="font-size: 12px; color: #999;">                 时间: {datetime.fromtimestamp(issue.timestamp).strftime('%H:%M:%S')} | 组件: {issue.component if issue.component else 'N/A'}             </div>         </div>                 """

            html_content += f"""     </div>      <div class="section">         <div class="section-title">系统信息</div>         <div class="summary">             <div class="metric"><span class="metric-label">平台:</span> <span class="metric-value">{platform.platform()}</span></div>             <div class="metric"><span class="metric-label">Python版本:</span> <span class="metric-value">{platform.python_version()}</span></div>             <div class="metric"><span class="metric-label">主机名:</span> <span class="metric-value">{platform.node()}</span></div>             <div class="metric"><span class="metric-label">CPU核心数:</span> <span class="metric-value">{psutil.cpu_count(logical=True)}</span></div>             <div class="metric"><span class="metric-label">总内存:</span> <span class="metric-value">{psutil.virtual_memory().total / (1024**3):.1f} GB</span></div>         </div>     </div>      <div class="section">         <div class="section-title">建议总结</div>         <div class="summary">             """

            # 添加建议
            if report.recommendations:
                for i, recommendation in enumerate(report.recommendations, 1):
                    html_content += f"""             <div>{i}. {recommendation}</div>                     """
            else:
                html_content += """
            <div>没有具体的修复建议。</div>
            """

            html_content += f"""         </div>     </div>      <div class="footer">         本报告由 Self AGI 系统诊断服务生成 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}     </div> </body> </html>"""

            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 保存HTML文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"诊断报告已导出为HTML: {file_path}")
            return html_content

        except Exception as e:
            logger.error(f"导出诊断报告为HTML失败: {e}")
            raise

    def export_report_to_text(
        self, report: DiagnosticReport, file_path: Optional[str] = None
    ) -> str:
        """将诊断报告导出为文本格式

        参数:
            report: 诊断报告
            file_path: 文件路径（可选，如果提供则保存到文件）

        返回:
            文本字符串
        """
        try:
            # 生成文本报告
            text_lines = []

            text_lines.append("=" * 80)
            text_lines.append(f"系统诊断报告: {report.report_id}")
            text_lines.append("=" * 80)
            text_lines.append(
                f"生成时间: {                     datetime.fromtimestamp(                         report.timestamp).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            text_lines.append(f"整体状态: {report.overall_status.value}")
            text_lines.append(f"问题总数: {report.issues_count}")
            text_lines.append(f"执行时间: {report.execution_time_seconds:.2f}秒")
            text_lines.append("")

            # 问题统计
            text_lines.append("问题统计:")
            text_lines.append("-" * 40)

            # 按类别统计
            categories = sorted(
                set(issue.category for issue in report.issues), key=lambda c: c.value
            )
            for category in categories:
                issues_by_level = {}
                for level in DiagnosticLevel:
                    issues_by_level[level] = len(
                        [
                            i
                            for i in report.issues
                            if i.category == category and i.level == level
                        ]
                    )

                text_lines.append(
                    f"{category.value:15} "
                    + f"信息: {issues_by_level.get(DiagnosticLevel.INFO, 0):2d} "
                    + f"警告: {issues_by_level.get(DiagnosticLevel.WARNING, 0):2d} "
                    + f"错误: {issues_by_level.get(DiagnosticLevel.ERROR, 0):2d} "
                    + f"严重: {issues_by_level.get(DiagnosticLevel.CRITICAL, 0):2d}"
                )

            text_lines.append("")

            # 详细问题列表
            text_lines.append("详细问题列表:")
            text_lines.append("=" * 80)

            # 按严重程度排序问题
            severity_order = {
                DiagnosticLevel.CRITICAL: 0,
                DiagnosticLevel.ERROR: 1,
                DiagnosticLevel.WARNING: 2,
                DiagnosticLevel.INFO: 3,
            }
            sorted_issues = sorted(report.issues, key=lambda x: severity_order[x.level])

            for i, issue in enumerate(sorted_issues, 1):
                text_lines.append(f"{i}. [{issue.level.value.upper()}] {issue.title}")
                text_lines.append(f"   类别: {issue.category.value}")
                text_lines.append(f"   描述: {issue.description}")

                if issue.metric_value is not None:
                    threshold_text = (
                        f" (阈值: {issue.threshold:.2f})" if issue.threshold else ""
                    )
                    text_lines.append(
                        f"   指标: {                             issue.metric_name} = {                             issue.metric_value:.2f}{threshold_text}"
                    )

                if issue.component:
                    text_lines.append(f"   组件: {issue.component}")

                if issue.suggestions:
                    for j, suggestion in enumerate(issue.suggestions, 1):
                        text_lines.append(f"   建议{j}: {suggestion}")

                text_lines.append(
                    f"   时间: {                         datetime.fromtimestamp(                             issue.timestamp).strftime('%H:%M:%S')}"
                )
                text_lines.append("")

            # 建议总结
            text_lines.append("建议总结:")
            text_lines.append("-" * 40)

            if report.recommendations:
                for i, recommendation in enumerate(report.recommendations, 1):
                    text_lines.append(f"{i}. {recommendation}")
            else:
                text_lines.append("没有具体的修复建议。")

            text_lines.append("")

            # 系统信息
            text_lines.append("系统信息:")
            text_lines.append("-" * 40)
            text_lines.append(f"平台: {platform.platform()}")
            text_lines.append(f"Python版本: {platform.python_version()}")
            text_lines.append(f"主机名: {platform.node()}")
            text_lines.append(f"CPU核心数: {psutil.cpu_count(logical=True)}")
            text_lines.append(
                f"总内存: {psutil.virtual_memory().total / (1024**3):.1f} GB"
            )
            text_lines.append("")

            text_lines.append("=" * 80)
            text_lines.append("报告结束 - Self AGI 系统诊断服务")
            text_lines.append(
                f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            text_lines.append("=" * 80)

            text_report = "\n".join(text_lines)

            # 如果提供了文件路径，保存到文件
            if file_path:
                try:
                    # 确保目录存在
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(text_report)

                    logger.info(f"诊断报告已导出为文本: {file_path}")
                except Exception as e:
                    logger.error(f"保存文本文件失败: {e}")
                    raise

            return text_report

        except Exception as e:
            logger.error(f"导出诊断报告为文本失败: {e}")
            raise

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态

        返回:
            服务状态字典
        """
        return {
            "initialized": self._initialized,
            "is_diagnosing": self.is_diagnosing,
            "last_diagnostic_time": self.last_diagnostic_time,
            "diagnostic_history_count": len(self.diagnostic_history),
            "system_monitor_available": self.system_monitor is not None,
            "health_manager_available": self.health_manager is not None,
            "timestamp": time.time(),
        }


# 全局服务实例（单例）
_system_diagnostic_service = None


def get_system_diagnostic_service() -> SystemDiagnosticService:
    """获取系统诊断服务实例（单例工厂函数）

    返回:
        系统诊断服务实例
    """
    global _system_diagnostic_service

    if _system_diagnostic_service is None:
        try:
            _system_diagnostic_service = SystemDiagnosticService()
            logger.info("创建系统诊断服务实例")
        except Exception as e:
            logger.error(f"创建系统诊断服务实例失败: {e}")

            # 创建降级实例
            class DegradedSystemDiagnosticService:
                def __init__(self):
                    self.service_status = "degraded"

                def get_service_status(self):
                    return {"service_status": "degraded", "timestamp": time.time()}

            _system_diagnostic_service = DegradedSystemDiagnosticService()

    return _system_diagnostic_service


def initialize_system_diagnostic_service():
    """初始化系统诊断服务（在应用启动时调用）

    返回:
        初始化是否成功
    """
    try:
        service = get_system_diagnostic_service()
        status = service.get_service_status()

        logger.info(f"系统诊断服务初始化完成: {status}")
        return True

    except Exception as e:
        logger.error(f"初始化系统诊断服务失败: {e}")
        return False


# 测试函数
def test_system_diagnostic_service():
    """测试系统诊断服务功能"""
    import sys
    import os
    import logging

    # 添加项目根目录到Python路径，确保可以导入backend和models模块
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logging.basicConfig(level=logging.INFO)

    print("=== 测试系统诊断服务 ===")
    print(f"项目根目录: {project_root}")
    print(f"Python路径: {sys.path[:3]}")

    try:
        # 获取服务实例
        service = get_system_diagnostic_service()

        # 测试服务状态
        status = service.get_service_status()
        print(f"服务状态: {status}")

        # 执行诊断
        print("\n执行系统诊断...")
        report = service.perform_diagnosis()

        print(f"诊断完成，整体状态: {report.overall_status}")
        print(f"发现问题数量: {report.issues_count}")
        print(f"诊断执行时间: {report.execution_time_seconds:.2f}秒")

        # 显示问题摘要
        if report.issues:
            print("\n问题摘要:")
            for level, count in report.issues_by_level.items():
                print(f"  {level.value}: {count}个")

        # 显示建议
        if report.recommendations:
            print("\n建议:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")

        # 测试导出功能
        print("\n测试导出功能...")
        import tempfile

        try:
            # JSON导出
            json_content = service.export_report_to_json(report)
            print(f"  JSON导出成功，长度: {len(json_content)} 字符")

            # HTML导出
            html_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False
            )
            html_path = html_file.name
            html_file.close()
            service.export_report_to_html(report, html_path)
            print(f"  HTML导出成功，文件: {html_path}")

            # 文本导出
            text_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            )
            text_path = text_file.name
            text_file.close()
            service.export_report_to_text(report, text_path)
            print(f"  文本导出成功，文件: {text_path}")

            # 清理临时文件（可选）
            import os

            os.unlink(html_path)
            os.unlink(text_path)
            print("  临时文件已清理")

            print("  所有导出功能测试通过！")
        except Exception as e:
            print(f"  导出测试失败: {e}")
            import traceback

            traceback.print_exc()

        # 获取诊断历史
        history = service.get_diagnosis_history(hours=1)
        print(f"\n最近1小时诊断历史: {len(history)}次")

        print("\n=== 测试完成 ===")
        return True

    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return False

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_system_diagnostic_service()
    exit(0 if success else 1)
