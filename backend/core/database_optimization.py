"""
数据库查询优化模块
包含数据库性能监控、查询优化、索引管理等功能
"""

import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timezone
from collections import defaultdict
import threading
from functools import wraps

from sqlalchemy.orm import Session
from sqlalchemy import event, text, inspect
import sqlparse

from .config import Config
from .database import engine

logger = logging.getLogger("DatabaseOptimization")


class QueryPerformanceMetrics:
    """查询性能指标"""

    def __init__(self):
        self.query_times: Dict[str, List[float]] = defaultdict(list)
        self.query_counts: Dict[str, int] = defaultdict(int)
        self.slow_queries: List[Dict[str, Any]] = []
        self.optimized_queries: List[Dict[str, Any]] = []
        self.start_time = datetime.now(timezone.utc)

        # 阈值配置（毫秒）
        self.slow_query_threshold = Config.SLOW_QUERY_THRESHOLD  # 100ms
        self.max_query_history = Config.MAX_QUERY_HISTORY

        logger.info("查询性能监控器初始化完成")

    def record_query(
        self, query: str, execution_time_ms: float, params: Optional[Dict] = None
    ):
        """记录查询执行时间"""
        # 生成查询指纹（移除参数差异）
        query_fingerprint = self._generate_query_fingerprint(query)

        self.query_times[query_fingerprint].append(execution_time_ms)
        self.query_counts[query_fingerprint] += 1

        # 限制历史记录大小
        if len(self.query_times[query_fingerprint]) > self.max_query_history:
            self.query_times[query_fingerprint].pop(0)

        # 检测慢查询
        if execution_time_ms > self.slow_query_threshold:
            slow_query_info = {
                "query": query[:200] + ("..." if len(query) > 200 else ""),
                "fingerprint": query_fingerprint,
                "execution_time_ms": execution_time_ms,
                "params": params,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "threshold": self.slow_query_threshold,
            }
            self.slow_queries.append(slow_query_info)

            # 限制慢查询记录数量
            if len(self.slow_queries) > 100:
                self.slow_queries.pop(0)

            logger.warning(
                f"慢查询检测: {execution_time_ms:.2f}ms > {self.slow_query_threshold}ms"
            )
            logger.debug(f"查询: {query[:100]}...")

    def record_optimization(
        self,
        original_query: str,
        optimized_query: str,
        improvement_percent: float,
        original_time_ms: float,
        optimized_time_ms: float,
    ):
        """记录查询优化结果"""
        optimization_info = {
            "original_query": original_query[:200]
            + ("..." if len(original_query) > 200 else ""),
            "optimized_query": optimized_query[:200]
            + ("..." if len(optimized_query) > 200 else ""),
            "improvement_percent": improvement_percent,
            "original_time_ms": original_time_ms,
            "optimized_time_ms": optimized_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.optimized_queries.append(optimization_info)

        # 限制优化记录数量
        if len(self.optimized_queries) > 50:
            self.optimized_queries.pop(0)

        logger.info(
            f"查询优化完成: 性能提升 {improvement_percent:.1f}% "
            f"({original_time_ms:.2f}ms -> {optimized_time_ms:.2f}ms)"
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        avg_times = {}
        for query_fingerprint, times in self.query_times.items():
            if times:
                avg_times[query_fingerprint] = sum(times) / len(times)

        # 按平均执行时间排序
        sorted_queries = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)

        total_queries = sum(self.query_counts.values())
        total_slow_queries = len(self.slow_queries)
        total_optimized = len(self.optimized_queries)

        uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime_seconds,
            "total_queries": total_queries,
            "queries_per_second": (
                total_queries / uptime_seconds if uptime_seconds > 0 else 0
            ),
            "slow_query_threshold_ms": self.slow_query_threshold,
            "total_slow_queries": total_slow_queries,
            "total_optimized_queries": total_optimized,
            "top_slow_queries": [
                {
                    "fingerprint": fingerprint,
                    "average_time_ms": avg_time,
                    "count": self.query_counts.get(fingerprint, 0),
                }
                for fingerprint, avg_time in sorted_queries[:10]
            ],
            "recent_slow_queries": self.slow_queries[-5:] if self.slow_queries else [],
            "recent_optimizations": (
                self.optimized_queries[-5:] if self.optimized_queries else []
            ),
            "query_statistics": {
                "unique_queries": len(self.query_counts),
                "most_frequent_query": (
                    max(self.query_counts.items(), key=lambda x: x[1])
                    if self.query_counts
                    else None
                ),
                "average_execution_time_ms": (
                    sum(avg_times.values()) / len(avg_times) if avg_times else 0
                ),
            },
        }

    def _generate_query_fingerprint(self, query: str) -> str:
        """生成查询指纹（标准化查询字符串）"""
        # 移除空格、换行和制表符
        normalized = " ".join(query.split())
        # 生成MD5哈希
        return hashlib.md5(normalized.encode()).hexdigest()[:16]


class QueryOptimizer:
    """查询优化器"""

    def __init__(self, db: Session):
        self.db = db
        self.metrics = QueryPerformanceMetrics()
        self.optimization_rules = self._initialize_optimization_rules()

        # 设置SQLAlchemy事件监听
        self._setup_sqlalchemy_events()

        logger.info("查询优化器初始化完成")

    def _initialize_optimization_rules(self) -> List[Dict[str, Any]]:
        """初始化优化规则"""
        return [
            {
                "name": "添加索引",
                "description": "为频繁查询的列添加索引",
                "condition": self._check_missing_indexes,
                "action": self._add_indexes,
                "priority": 1,
            },
            {
                "name": "优化JOIN操作",
                "description": "优化关联查询，确保使用正确的JOIN类型",
                "condition": self._check_join_optimization,
                "action": self._optimize_joins,
                "priority": 2,
            },
            {
                "name": "优化WHERE条件",
                "description": "优化WHERE子句中的条件顺序和索引使用",
                "condition": self._check_where_optimization,
                "action": self._optimize_where_clauses,
                "priority": 3,
            },
            {
                "name": "查询缓存",
                "description": "为频繁执行的查询添加缓存",
                "condition": self._check_query_caching,
                "action": self._add_query_cache,
                "priority": 4,
            },
            {
                "name": "分页优化",
                "description": "优化分页查询，避免深度分页性能问题",
                "condition": self._check_pagination_optimization,
                "action": self._optimize_pagination,
                "priority": 5,
            },
        ]

    def _setup_sqlalchemy_events(self):
        """设置SQLAlchemy事件监听"""

        @event.listens_for(engine, "before_execute")
        def before_execute(conn, clauseelement, multiparams, params):
            """查询执行前事件"""
            self._current_query_start = time.time()
            self._current_query = str(clauseelement)

        @event.listens_for(engine, "after_execute")
        def after_execute(conn, clauseelement, multiparams, params, result):
            """查询执行后事件"""
            if hasattr(self, "_current_query_start"):
                execution_time = (
                    time.time() - self._current_query_start
                ) * 1000  # 转换为毫秒
                query_str = (
                    self._current_query
                    if hasattr(self, "_current_query")
                    else str(clauseelement)
                )

                # 记录查询性能
                self.metrics.record_query(query_str, execution_time, params)

    def optimize_query(self, original_query: str) -> str:
        """优化SQL查询"""
        try:
            # 解析查询
            parsed = sqlparse.parse(original_query)
            if not parsed:
                return original_query

            statement = parsed[0]

            # 应用优化规则
            optimized_query = original_query
            optimizations_applied = []

            for rule in sorted(self.optimization_rules, key=lambda x: x["priority"]):
                try:
                    condition_result = rule["condition"](statement)
                    if condition_result.get("needs_optimization", False):
                        optimized = rule["action"](statement, condition_result)
                        if optimized and optimized != optimized_query:
                            optimized_query = optimized
                            optimizations_applied.append(
                                {
                                    "rule": rule["name"],
                                    "description": rule["description"],
                                    "details": condition_result.get("details", {}),
                                }
                            )
                except Exception as e:
                    logger.error(f"应用优化规则失败 {rule['name']}: {e}")

            if optimizations_applied:
                logger.info(f"查询优化完成，应用了 {len(optimizations_applied)} 条规则")
                for opt in optimizations_applied:
                    logger.debug(f"  - {opt['rule']}: {opt['description']}")

            return optimized_query

        except Exception as e:
            logger.error(f"查询优化失败: {e}")
            return original_query

    def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """分析表结构，提供优化建议"""
        try:
            inspector = inspect(engine)

            # 获取表信息
            columns = inspector.get_columns(table_name)
            indexes = inspector.get_indexes(table_name)
            primary_keys = inspector.get_pk_constraint(table_name)

            # 分析列使用情况
            column_analysis = []
            for column in columns:
                column_info = {
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column.get("nullable", True),
                    "default": column.get("default"),
                    "is_primary_key": column["name"]
                    in primary_keys.get("constrained_columns", []),
                    "has_index": any(
                        idx["column_names"] == [column["name"]] for idx in indexes
                    ),
                    "optimization_suggestions": [],
                }

                # 生成优化建议
                suggestions = self._analyze_column(column_info, table_name)
                column_info["optimization_suggestions"] = suggestions

                column_analysis.append(column_info)

            # 分析索引
            index_analysis = []
            for idx in indexes:
                index_info = {
                    "name": idx["name"],
                    "columns": idx["column_names"],
                    "unique": idx.get("unique", False),
                    "optimization_suggestions": [],
                }

                # 生成索引优化建议
                suggestions = self._analyze_index(idx, table_name)
                index_info["optimization_suggestions"] = suggestions

                index_analysis.append(index_info)

            return {
                "table_name": table_name,
                "column_count": len(columns),
                "index_count": len(indexes),
                "primary_keys": primary_keys.get("constrained_columns", []),
                "columns": column_analysis,
                "indexes": index_analysis,
                "overall_suggestions": self._generate_table_suggestions(
                    column_analysis, index_analysis
                ),
            }

        except Exception as e:
            logger.error(f"分析表结构失败 {table_name}: {e}")
            return {"error": str(e), "table_name": table_name}

    def _analyze_column(
        self, column_info: Dict[str, Any], table_name: str
    ) -> List[str]:
        """分析列并提供优化建议"""
        suggestions = []

        # 检查是否应该添加索引
        if not column_info["has_index"] and not column_info["is_primary_key"]:
            # 常见需要索引的列：外键、搜索条件、排序字段
            if column_info["name"].endswith("_id"):
                suggestions.append(f"考虑为外键列 {column_info['name']} 添加索引")
            elif column_info["name"] in ["created_at", "updated_at", "last_login"]:
                suggestions.append(
                    f"考虑为时间戳列 {column_info['name']} 添加索引以优化时间范围查询"
                )

        # 检查数据类型
        column_type = str(column_info["type"]).lower()
        if "text" in column_type and column_info["nullable"]:
            suggestions.append(
                f"Text类型列 {column_info['name']} 可为空，考虑添加非空约束或使用默认值"
            )

        return suggestions

    def _analyze_index(self, index_info: Dict[str, Any], table_name: str) -> List[str]:
        """分析索引并提供优化建议"""
        suggestions = []

        # 检查复合索引的顺序
        if len(index_info["columns"]) > 1:
            suggestions.append(
                f"复合索引 {                     index_info['name']} 包含 {                     len(                         index_info['columns'])} 列，确保最频繁查询的列在前"
            )

        # 检查索引是否被过度使用
        if len(index_info["columns"]) > 3:
            suggestions.append(
                f"索引 {index_info['name']} 包含过多列（{len(index_info['columns'])}），考虑拆分为多个索引"
            )

        return suggestions

    def _generate_table_suggestions(
        self, columns: List[Dict[str, Any]], indexes: List[Dict[str, Any]]
    ) -> List[str]:
        """生成表的整体优化建议"""
        suggestions = []

        # 统计需要索引的列
        columns_needing_index = sum(
            1
            for col in columns
            if not col["has_index"]
            and not col["is_primary_key"]
            and col["name"].endswith("_id")
        )

        if columns_needing_index > 0:
            suggestions.append(
                f"有 {columns_needing_index} 个外键列缺少索引，建议添加索引以优化关联查询"
            )

        # 检查是否有过多索引
        if len(indexes) > len(columns) * 0.5:  # 索引数超过列数的一半
            suggestions.append("索引数量可能过多，考虑合并或删除不常用的索引")

        return suggestions

    def _check_missing_indexes(self, statement) -> Dict[str, Any]:
        """检查缺失索引"""
        # 这里可以集成更复杂的索引分析逻辑
        return {"needs_optimization": False, "details": {}}

    def _add_indexes(self, statement, condition_result) -> str:
        """添加索引"""
        # 这里可以实际执行添加索引的SQL
        return str(statement)

    def _check_join_optimization(self, statement) -> Dict[str, Any]:
        """检查JOIN优化"""
        return {"needs_optimization": False, "details": {}}

    def _optimize_joins(self, statement, condition_result) -> str:
        """优化JOIN操作"""
        return str(statement)

    def _check_where_optimization(self, statement) -> Dict[str, Any]:
        """检查WHERE条件优化"""
        return {"needs_optimization": False, "details": {}}

    def _optimize_where_clauses(self, statement, condition_result) -> str:
        """优化WHERE子句"""
        return str(statement)

    def _check_query_caching(self, statement) -> Dict[str, Any]:
        """检查查询缓存"""
        return {"needs_optimization": False, "details": {}}

    def _add_query_cache(self, statement, condition_result) -> str:
        """添加查询缓存"""
        return str(statement)

    def _check_pagination_optimization(self, statement) -> Dict[str, Any]:
        """检查分页优化"""
        return {"needs_optimization": False, "details": {}}

    def _optimize_pagination(self, statement, condition_result) -> str:
        """优化分页查询"""
        return str(statement)

    def execute_with_optimization(self, query_func: Callable, *args, **kwargs) -> Any:
        """执行查询函数并进行优化"""
        start_time = time.time()

        try:
            # 执行原始查询
            result = query_func(*args, **kwargs)
            (time.time() - start_time) * 1000

            # 这里可以添加优化后的查询执行和比较逻辑
            # 目前先记录原始查询性能

            return result

        except Exception as e:
            logger.error(f"执行优化查询失败: {e}")
            raise


class DatabaseMonitor:
    """数据库监控器"""

    def __init__(self, update_interval: float = 60.0):
        self.update_interval = update_interval
        self.metrics_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(
            list
        )
        self.alerts: List[Dict[str, Any]] = []
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None

        logger.info(f"数据库监控器初始化完成: 更新间隔={update_interval}s")

    def start_monitoring(self):
        """开始监控"""
        if self.running:
            logger.warning("数据库监控器已在运行")
            return

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()

        logger.info("数据库监控器已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("数据库监控器已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                self._collect_metrics()
                self._check_alerts()

                # 等待下次收集
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"数据库监控循环错误: {e}")
                time.sleep(self.update_interval)

    def _collect_metrics(self):
        """收集数据库指标"""
        try:
            with engine.connect() as conn:
                # 收集连接信息
                result = conn.execute(text("""
                    SELECT
                        COUNT(*) as connection_count,
                        SUM(CASE WHEN state = 'active' THEN 1 ELSE 0 END) as active_connections,
                        SUM(CASE WHEN state = 'idle' THEN 1 ELSE 0 END) as idle_connections
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                """))
                connection_stats = result.fetchone()

                # 收集表大小信息
                result = conn.execute(text("""
                    SELECT
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
                        pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) as table_size,
                        pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename) - pg_relation_size(schemaname || '.' || tablename)) as index_size
                    FROM pg_tables
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC
                    LIMIT 10
                """))
                table_sizes = result.fetchall()

                # 收集索引使用统计
                result = conn.execute(text("""
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan as index_scans,
                        idx_tup_read as tuples_read,
                        idx_tup_fetch as tuples_fetched
                    FROM pg_stat_user_indexes
                    ORDER BY idx_scan DESC
                    LIMIT 10
                """))
                index_stats = result.fetchall()

                timestamp = datetime.now(timezone.utc)

                # 存储指标历史
                if connection_stats:
                    self.metrics_history["connection_count"].append(
                        (timestamp, connection_stats[0])
                    )
                    self.metrics_history["active_connections"].append(
                        (timestamp, connection_stats[1])
                    )
                    self.metrics_history["idle_connections"].append(
                        (timestamp, connection_stats[2])
                    )

                # 限制历史记录大小
                for key in self.metrics_history:
                    if len(self.metrics_history[key]) > 1000:
                        self.metrics_history[key] = self.metrics_history[key][-1000:]

                logger.debug(
                    f"数据库指标收集完成: {len(table_sizes)}个表, {len(index_stats)}个索引"
                )

        except Exception as e:
            logger.error(f"收集数据库指标失败: {e}")

    def _check_alerts(self):
        """检查警报条件"""
        try:
            # 检查连接数异常
            if (
                "connection_count" in self.metrics_history
                and self.metrics_history["connection_count"]
            ):
                recent_connections = [
                    val for _, val in self.metrics_history["connection_count"][-5:]
                ]
                avg_connections = sum(recent_connections) / len(recent_connections)

                if avg_connections > Config.get("MAX_DATABASE_CONNECTIONS", 100) * 0.8:
                    self._add_alert(
                        "high_connection_count",
                        f"数据库连接数过高: {avg_connections:.1f}",
                        "warning",
                    )

            # 这里可以添加更多警报检查逻辑

        except Exception as e:
            logger.error(f"检查数据库警报失败: {e}")

    def _add_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """添加警报"""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.alerts.append(alert)

        # 限制警报数量
        if len(self.alerts) > 100:
            self.alerts.pop(0)

        if severity == "critical":
            logger.critical(f"数据库警报: {message}")
        elif severity == "warning":
            logger.warning(f"数据库警报: {message}")
        else:
            logger.info(f"数据库警报: {message}")

    def get_monitoring_report(self) -> Dict[str, Any]:
        """获取监控报告"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "running": self.running,
            "update_interval": self.update_interval,
            "metrics_history_summary": {
                metric: len(values) for metric, values in self.metrics_history.items()
            },
            "active_alerts": [
                alert
                for alert in self.alerts
                if alert.get("acknowledged", False) == False
            ],
            "total_alerts": len(self.alerts),
            "recent_alerts": self.alerts[-10:] if self.alerts else [],
        }


# 全局实例
_query_optimizer = None
_database_monitor = None


def get_query_optimizer(db: Session) -> QueryOptimizer:
    """获取查询优化器实例"""
    global _query_optimizer

    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer(db)

    return _query_optimizer


def get_database_monitor() -> DatabaseMonitor:
    """获取数据库监控器实例"""
    global _database_monitor

    if _database_monitor is None:
        _database_monitor = DatabaseMonitor()

    return _database_monitor


def query_performance_decorator(func):
    """查询性能监控装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000

            # 获取优化器实例并记录性能
            db = None
            for arg in args:
                if isinstance(arg, Session):
                    db = arg
                    break

            if db:
                optimizer = get_query_optimizer(db)
                optimizer.metrics.record_query(
                    query=func.__name__, execution_time_ms=execution_time, params=kwargs
                )

            return result

        except Exception as e:
            logger.error(f"查询执行失败 {func.__name__}: {e}")
            raise

    return wrapper
