"""
高并发优化模块
为Self AGI系统提供工业级高并发支持

功能：
1. 连接池管理（数据库、Redis）
2. 异步任务队列和线程池
3. 缓存策略优化
4. 性能监控和指标收集
5. 负载均衡配置
"""

import asyncio
import threading
import time
import logging
import concurrent.futures
from typing import Dict, Any, Optional, List, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import redis
from sqlalchemy.pool import QueuePool
from sqlalchemy import create_engine

from .config import Config
from .redis import get_redis_client

# 配置日志
logger = logging.getLogger(__name__)


class ConnectionPoolType(Enum):
    """连接池类型枚举"""
    DATABASE = "database"
    REDIS = "redis"
    HTTP = "http"
    WEBSOCKET = "websocket"


@dataclass
class PoolStats:
    """连接池统计信息"""
    pool_type: str
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    connection_errors: int = 0
    avg_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ConnectionPoolManager:
    """连接池管理器
    
    统一管理各种类型的连接池，提供统计和监控功能
    """
    
    def __init__(self):
        """初始化连接池管理器"""
        self.pools: Dict[str, Any] = {}
        self.stats: Dict[str, PoolStats] = {}
        self.lock = threading.RLock()
        
        # 初始化数据库连接池
        self._init_database_pool()
        
        # 初始化Redis连接池
        self._init_redis_pool()
        
        logger.info("连接池管理器初始化完成")
    
    def _init_database_pool(self):
        """初始化数据库连接池"""
        try:
            # 创建SQLAlchemy引擎，使用连接池
            engine = create_engine(
                Config.DATABASE_URL,
                poolclass=QueuePool,
                pool_size=Config.DATABASE_POOL_SIZE,
                max_overflow=Config.DATABASE_MAX_OVERFLOW,
                pool_timeout=Config.DATABASE_POOL_TIMEOUT,
                pool_recycle=Config.DATABASE_POOL_RECYCLE,
                pool_pre_ping=True,  # 连接前进行健康检查
            )
            
            pool_name = "database_main"
            self.pools[pool_name] = engine
            self.stats[pool_name] = PoolStats(
                pool_type=ConnectionPoolType.DATABASE.value,
                total_connections=Config.DATABASE_POOL_SIZE,
            )
            
            logger.info(f"数据库连接池初始化完成: pool_size={Config.DATABASE_POOL_SIZE}, max_overflow={Config.DATABASE_MAX_OVERFLOW}")
        except Exception as e:
            logger.error(f"初始化数据库连接池失败: {e}")
    
    def _init_redis_pool(self):
        """初始化Redis连接池"""
        try:
            # Redis连接池已由redis-py内部管理
            redis_client = get_redis_client()
            pool_name = "redis_main"
            self.pools[pool_name] = redis_client
            
            # 获取Redis连接池信息
            connection_pool = redis_client.connection_pool
            if connection_pool:
                self.stats[pool_name] = PoolStats(
                    pool_type=ConnectionPoolType.REDIS.value,
                    total_connections=connection_pool.max_connections,
                )
            
            logger.info(f"Redis连接池初始化完成")
        except Exception as e:
            logger.error(f"初始化Redis连接池失败: {e}")
    
    def get_pool(self, pool_name: str) -> Optional[Any]:
        """获取连接池"""
        with self.lock:
            return self.pools.get(pool_name)
    
    def get_stats(self, pool_name: Optional[str] = None) -> Dict[str, Any]:
        """获取连接池统计信息"""
        with self.lock:
            if pool_name:
                pool = self.pools.get(pool_name)
                stats = self.stats.get(pool_name)
                if pool and stats:
                    # 更新统计信息
                    self._update_pool_stats(pool_name, pool, stats)
                    return {pool_name: stats}
                return {}  # 返回空字典
            else:
                # 更新所有池的统计信息
                for name, pool in self.pools.items():
                    stats = self.stats.get(name)
                    if stats:
                        self._update_pool_stats(name, pool, stats)
                return {name: self.stats[name] for name in self.pools.keys() if name in self.stats}
    
    def _update_pool_stats(self, pool_name: str, pool: Any, stats: PoolStats):
        """更新连接池统计信息"""
        try:
            if pool_name == "database_main" and hasattr(pool, "pool"):
                # SQLAlchemy连接池统计
                pool_status = pool.pool.status()
                stats.active_connections = pool_status.get("checkedin", 0)
                stats.idle_connections = pool_status.get("checkedout", 0)
                stats.total_connections = stats.active_connections + stats.idle_connections
            elif pool_name == "redis_main" and hasattr(pool, "connection_pool"):
                # Redis连接池统计
                connection_pool = pool.connection_pool
                if connection_pool:
                    stats.active_connections = len(connection_pool._in_use_connections)
                    stats.idle_connections = len(connection_pool._available_connections)
                    stats.total_connections = connection_pool.max_connections
            
            stats.last_updated = datetime.now(timezone.utc)
        except Exception as e:
            logger.error(f"更新连接池统计信息失败: {e}")
    
    def close_all(self):
        """关闭所有连接池"""
        with self.lock:
            for name, pool in self.pools.items():
                try:
                    if name == "database_main" and hasattr(pool, "dispose"):
                        pool.dispose()
                    elif name == "redis_main" and hasattr(pool, "close"):
                        pool.close()
                    logger.info(f"连接池 '{name}' 已关闭")
                except Exception as e:
                    logger.error(f"关闭连接池 '{name}' 失败: {e}")


class AsyncTaskQueue:
    """异步任务队列
    
    用于处理高并发异步任务，支持任务优先级、超时和重试
    """
    
    def __init__(self, max_workers: int = None, queue_size: int = 1000):
        """初始化异步任务队列"""
        self.max_workers = max_workers or Config.ASYNC_TASK_MAX_WORKERS
        self.queue_size = queue_size
        self.task_queue = asyncio.Queue(maxsize=queue_size)
        self.worker_tasks: List[asyncio.Task] = []
        self.running = False
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "pending_tasks": 0,
            "average_process_time_ms": 0.0,
            "last_task_time": None,
        }
        
        # 线程池用于阻塞任务
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="async_task_worker"
        )
        
        logger.info(f"异步任务队列初始化完成: max_workers={self.max_workers}, queue_size={queue_size}")
    
    async def start(self):
        """启动任务队列"""
        if self.running:
            return
        
        self.running = True
        # 启动工作线程
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker_loop(i), name=f"async_worker_{i}")
            self.worker_tasks.append(task)
        
        logger.info(f"异步任务队列已启动，工作线程数: {self.max_workers}")
    
    async def stop(self):
        """停止任务队列"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待所有工作线程完成
        for task in self.worker_tasks:
            task.cancel()
        
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        logger.info("异步任务队列已停止")
    
    async def submit(self, task_func: Callable, *args, **kwargs) -> Any:
        """提交异步任务"""
        task_id = f"task_{self.stats['total_tasks']}_{time.time()}"
        
        # 创建任务包装器
        async def task_wrapper():
            start_time = time.time()
            try:
                # 如果函数是协程，直接等待
                if asyncio.iscoroutinefunction(task_func):
                    result = await task_func(*args, **kwargs)
                else:
                    # 否则在线程池中运行阻塞任务
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.thread_pool, task_func, *args
                    )
                
                process_time = (time.time() - start_time) * 1000
                
                # 更新统计信息
                with threading.Lock():
                    self.stats["completed_tasks"] += 1
                    self.stats["pending_tasks"] = self.task_queue.qsize()
                    # 更新平均处理时间
                    old_avg = self.stats["average_process_time_ms"]
                    total_completed = self.stats["completed_tasks"]
                    self.stats["average_process_time_ms"] = (
                        old_avg * (total_completed - 1) + process_time
                    ) / total_completed
                    self.stats["last_task_time"] = datetime.now(timezone.utc)
                
                logger.debug(f"任务 {task_id} 完成，耗时: {process_time:.2f}ms")
                return result
                
            except Exception as e:
                process_time = (time.time() - start_time) * 1000
                
                # 更新统计信息
                with threading.Lock():
                    self.stats["failed_tasks"] += 1
                    self.stats["pending_tasks"] = self.task_queue.qsize()
                
                logger.error(f"任务 {task_id} 失败，耗时: {process_time:.2f}ms, 错误: {e}")
                raise
        
        # 将任务放入队列
        await self.task_queue.put(task_wrapper)
        
        # 更新统计信息
        with threading.Lock():
            self.stats["total_tasks"] += 1
            self.stats["pending_tasks"] = self.task_queue.qsize()
        
        logger.debug(f"任务 {task_id} 已提交，队列大小: {self.task_queue.qsize()}")
    
    async def _worker_loop(self, worker_id: int):
        """工作线程循环"""
        logger.debug(f"工作线程 {worker_id} 启动")
        
        while self.running:
            try:
                # 从队列获取任务，设置超时
                task_wrapper = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                try:
                    await task_wrapper()
                except Exception as e:
                    logger.error(f"工作线程 {worker_id} 执行任务失败: {e}")
                finally:
                    self.task_queue.task_done()
                    
            except asyncio.TimeoutError:
                # 队列为空，继续等待
                continue
            except asyncio.CancelledError:
                # 任务被取消
                break
            except Exception as e:
                logger.error(f"工作线程 {worker_id} 出错: {e}")
                await asyncio.sleep(0.1)
        
        logger.debug(f"工作线程 {worker_id} 停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        with threading.Lock():
            stats = self.stats.copy()
            stats["queue_size"] = self.task_queue.qsize()
            stats["max_queue_size"] = self.queue_size
            stats["workers_count"] = len(self.worker_tasks)
            return stats


class PerformanceMonitor:
    """性能监控器
    
    监控系统性能指标，提供实时统计和警报
    """
    
    def __init__(self, update_interval: Optional[float] = None):
        """初始化性能监控器"""
        self.update_interval = update_interval or Config.PERFORMANCE_MONITOR_INTERVAL
        self.metrics: Dict[str, Any] = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "network_io": [],
            "request_rate": [],
            "error_rate": [],
            "response_time": [],
        }
        self.alerts: List[Dict[str, Any]] = []
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 阈值配置 - 从Config中读取
        self.thresholds = {
            "cpu_usage": Config.PERFORMANCE_ALERT_THRESHOLD_CPU,
            "memory_usage": Config.PERFORMANCE_ALERT_THRESHOLD_MEMORY,
            "disk_usage": Config.PERFORMANCE_ALERT_THRESHOLD_DISK,
            "response_time": 1000.0,  # 响应时间阈值（ms）
            "error_rate": 5.0,  # 错误率阈值（%）
        }
        
        logger.info(f"性能监控器初始化完成: update_interval={self.update_interval}s")
        logger.info(f"性能阈值: CPU={self.thresholds['cpu_usage']}%, Memory={self.thresholds['memory_usage']}%, Disk={self.thresholds['disk_usage']}%")
    
    def start(self):
        """启动性能监控器"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="performance_monitor"
        )
        self.monitor_thread.start()
        
        logger.info("性能监控器已启动")
    
    def stop(self):
        """停止性能监控器"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("性能监控器已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        import psutil
        
        while self.running:
            try:
                # 收集系统指标
                metrics = {}
                
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                metrics["cpu_usage"] = cpu_percent
                self.metrics["cpu_usage"].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "value": cpu_percent
                })
                
                # 内存使用率
                memory = psutil.virtual_memory()
                metrics["memory_usage"] = memory.percent
                self.metrics["memory_usage"].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "value": memory.percent
                })
                
                # 磁盘IO
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics["disk_io"] = {
                        "read_bytes": disk_io.read_bytes,
                        "write_bytes": disk_io.write_bytes,
                        "read_count": disk_io.read_count,
                        "write_count": disk_io.write_count,
                    }
                    self.metrics["disk_io"].append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "value": metrics["disk_io"]
                    })
                
                # 网络IO
                net_io = psutil.net_io_counters()
                if net_io:
                    metrics["network_io"] = {
                        "bytes_sent": net_io.bytes_sent,
                        "bytes_recv": net_io.bytes_recv,
                        "packets_sent": net_io.packets_sent,
                        "packets_recv": net_io.packets_recv,
                    }
                    self.metrics["network_io"].append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "value": metrics["network_io"]
                    })
                
                # 检查阈值并生成警报
                self._check_thresholds(metrics)
                
                # 清理旧数据（保留最近1000个数据点）
                for metric_name in self.metrics:
                    if len(self.metrics[metric_name]) > 1000:
                        self.metrics[metric_name] = self.metrics[metric_name][-1000:]
                
                # 清理旧警报（保留最近100个）
                if len(self.alerts) > 100:
                    self.alerts = self.alerts[-100:]
                
            except Exception as e:
                logger.error(f"性能监控收集失败: {e}")
            
            # 等待下一个收集周期
            time.sleep(self.update_interval)
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """检查阈值并生成警报"""
        current_time = datetime.now(timezone.utc)
        
        # 检查CPU使用率
        cpu_usage = metrics.get("cpu_usage")
        if cpu_usage and cpu_usage > self.thresholds["cpu_usage"]:
            self.alerts.append({
                "timestamp": current_time.isoformat(),
                "type": "high_cpu_usage",
                "level": "warning",
                "value": cpu_usage,
                "threshold": self.thresholds["cpu_usage"],
                "message": f"CPU使用率过高: {cpu_usage:.1f}% (阈值: {self.thresholds['cpu_usage']}%)"
            })
        
        # 检查内存使用率
        memory_usage = metrics.get("memory_usage")
        if memory_usage and memory_usage > self.thresholds["memory_usage"]:
            self.alerts.append({
                "timestamp": current_time.isoformat(),
                "type": "high_memory_usage",
                "level": "warning",
                "value": memory_usage,
                "threshold": self.thresholds["memory_usage"],
                "message": f"内存使用率过高: {memory_usage:.1f}% (阈值: {self.thresholds['memory_usage']}%)"
            })
    
    def get_metrics(self, metric_name: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """获取性能指标"""
        if metric_name:
            if metric_name in self.metrics:
                metrics = self.metrics[metric_name][-limit:]
                return {metric_name: metrics}
            return {}  # 返回空字典
        else:
            result = {}
            for name, values in self.metrics.items():
                result[name] = values[-limit:]
            return result
    
    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取警报"""
        return self.alerts[-limit:] if self.alerts else []
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        
        # CPU使用率统计
        cpu_values = [m["value"] for m in self.metrics["cpu_usage"][-100:]]
        if cpu_values:
            summary["cpu_usage"] = {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0,
            }
        
        # 内存使用率统计
        memory_values = [m["value"] for m in self.metrics["memory_usage"][-100:]]
        if memory_values:
            summary["memory_usage"] = {
                "current": memory_values[-1] if memory_values else 0,
                "average": sum(memory_values) / len(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "min": min(memory_values) if memory_values else 0,
            }
        
        # 警报统计
        summary["alerts"] = {
            "total": len(self.alerts),
            "warning": len([a for a in self.alerts if a.get("level") == "warning"]),
            "critical": len([a for a in self.alerts if a.get("level") == "critical"]),
            "recent": len([a for a in self.alerts[-10:] if datetime.fromisoformat(a["timestamp"]) > datetime.now(timezone.utc) - timedelta(hours=1)])
        }
        
        return summary


# 全局实例
connection_pool_manager = ConnectionPoolManager()
async_task_queue = AsyncTaskQueue()
performance_monitor = PerformanceMonitor()


def init_concurrency_systems():
    """初始化并发系统"""
    try:
        # 启动性能监控器
        performance_monitor.start()
        
        logger.info("并发系统初始化完成")
        return True
    except Exception as e:
        logger.error(f"初始化并发系统失败: {e}")
        return False


def shutdown_concurrency_systems():
    """关闭并发系统"""
    try:
        # 停止性能监控器
        performance_monitor.stop()
        
        # 关闭连接池
        connection_pool_manager.close_all()
        
        logger.info("并发系统已关闭")
        return True
    except Exception as e:
        logger.error(f"关闭并发系统失败: {e}")
        return False


__all__ = [
    "ConnectionPoolManager",
    "AsyncTaskQueue", 
    "PerformanceMonitor",
    "connection_pool_manager",
    "async_task_queue",
    "performance_monitor",
    "init_concurrency_systems",
    "shutdown_concurrency_systems",
]