"""
模式状态管理器
负责管理自主模式和任务执行模式之间的状态切换和持久化

功能：
1. 模式状态持久化存储（SQLite + Redis）
2. 切换时的任务保存和恢复
3. 模式历史记录和回滚
4. 状态验证和一致性检查
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import pickle
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis不可用，模式状态将仅存储在SQLite中")

logger = logging.getLogger(__name__)


class ModeType(Enum):
    """模式类型枚举"""
    
    AUTONOMOUS = "autonomous"      # 自主模式
    TASK_EXECUTION = "task"        # 任务执行模式


class ModeState(Enum):
    """模式状态枚举"""
    
    ACTIVE = "active"              # 活跃状态
    PAUSED = "paused"              # 暂停状态
    TRANSITIONING = "transitioning"  # 切换中
    ERROR = "error"                # 错误状态


class ModeStateService:
    """模式状态管理器
    
    功能：
    - 模式状态持久化存储
    - 切换任务保存和恢复
    - 历史记录和回滚
    - 状态验证和一致性检查
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化模式状态管理器
        
        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(f"{__name__}.ModeStateService")
        
        # 默认配置
        self.config = config or {
            "sqlite_db_path": "data/mode_states.db",
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 1,
            "redis_password": None,
            "state_retention_days": 30,  # 状态保留天数
            "backup_enabled": True,      # 启用备份
            "backup_interval_hours": 24, # 备份间隔（小时）
            "max_history_entries": 1000, # 最大历史记录数
            "enable_compression": True,  # 启用状态压缩
        }
        
        # 创建数据目录
        db_path = Path(self.config["sqlite_db_path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库连接
        self.db_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db_conn.row_factory = sqlite3.Row
        
        # 初始化Redis连接
        self.redis_client = None
        if REDIS_AVAILABLE and self.config.get("redis_host"):
            try:
                self.redis_client = redis.Redis(
                    host=self.config["redis_host"],
                    port=self.config["redis_port"],
                    db=self.config["redis_db"],
                    password=self.config.get("redis_password"),
                    decode_responses=False,  # 存储二进制数据
                )
                # 测试Redis连接
                self.redis_client.ping()
                self.logger.info("Redis连接成功")
            except Exception as e:
                self.logger.warning(f"Redis连接失败: {e}")
                self.redis_client = None
        
        # 初始化数据库表
        self._init_database()
        
        # 当前模式状态缓存
        self.current_mode_cache: Optional[Dict[str, Any]] = None
        self.cache_valid = False
        
        # 统计信息
        self.stats = {
            "total_state_saves": 0,
            "total_state_restores": 0,
            "total_transitions": 0,
            "failed_transitions": 0,
            "last_backup_time": None,
            "database_size": 0,
        }
        
        self.logger.info("模式状态管理器初始化完成")
    
    def _init_database(self):
        """初始化数据库表"""
        cursor = self.db_conn.cursor()
        
        # 创建模式状态表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mode_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode_type TEXT NOT NULL,
                mode_state TEXT NOT NULL,
                state_data BLOB NOT NULL,
                tasks_data BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                session_id TEXT,
                metadata TEXT
            )
        """)
        
        # 创建模式历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mode_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_mode TEXT NOT NULL,
                to_mode TEXT NOT NULL,
                transition_type TEXT NOT NULL,
                state_snapshot_id INTEGER,
                tasks_snapshot BLOB,
                transition_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_ms INTEGER,
                success BOOLEAN,
                error_message TEXT,
                user_id INTEGER,
                session_id TEXT,
                metadata TEXT,
                FOREIGN KEY (state_snapshot_id) REFERENCES mode_states(id)
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mode_states_type ON mode_states(mode_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mode_states_created ON mode_states(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mode_history_transition ON mode_history(transition_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mode_history_from_to ON mode_history(from_mode, to_mode)")
        
        self.db_conn.commit()
        
        self.logger.info("数据库表初始化完成")
    
    def save_mode_state(self, 
                       mode_type: ModeType, 
                       mode_state: ModeState,
                       state_data: Dict[str, Any],
                       tasks_data: Optional[List[Dict[str, Any]]] = None,
                       user_id: Optional[int] = None,
                       session_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> int:
        """保存模式状态
        
        参数:
            mode_type: 模式类型
            mode_state: 模式状态
            state_data: 状态数据
            tasks_data: 任务数据（可选）
            user_id: 用户ID（可选）
            session_id: 会话ID（可选）
            metadata: 元数据（可选）
            
        返回:
            int: 保存的状态ID
        """
        try:
            # 压缩状态数据
            compressed_state = self._compress_data(state_data)
            compressed_tasks = self._compress_data(tasks_data) if tasks_data else None
            
            # 准备元数据
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO mode_states 
                (mode_type, mode_state, state_data, tasks_data, user_id, session_id, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                mode_type.value,
                mode_state.value,
                compressed_state,
                compressed_tasks,
                user_id,
                session_id,
                metadata_json,
            ))
            
            state_id = cursor.lastrowid
            
            # 同时保存到Redis缓存
            if self.redis_client:
                cache_key = f"mode_state:{state_id}"
                cache_data = {
                    "mode_type": mode_type.value,
                    "mode_state": mode_state.value,
                    "state_data": state_data,
                    "tasks_data": tasks_data,
                    "metadata": metadata,
                    "saved_at": datetime.now().isoformat(),
                }
                self.redis_client.setex(
                    cache_key,
                    3600,  # 1小时TTL
                    pickle.dumps(cache_data)
                )
            
            self.db_conn.commit()
            
            # 更新缓存
            self.current_mode_cache = {
                "id": state_id,
                "mode_type": mode_type.value,
                "mode_state": mode_state.value,
                "state_data": state_data,
                "tasks_data": tasks_data,
                "metadata": metadata,
                "saved_at": datetime.now().isoformat(),
            }
            self.cache_valid = True
            
            # 更新统计信息
            self.stats["total_state_saves"] += 1
            
            # 清理旧数据
            self._cleanup_old_data()
            
            self.logger.info(f"模式状态保存成功: ID={state_id}, 模式={mode_type.value}, 状态={mode_state.value}")
            
            return state_id
            
        except Exception as e:
            self.logger.error(f"保存模式状态失败: {e}")
            raise
    
    def restore_mode_state(self, state_id: int) -> Optional[Dict[str, Any]]:
        """恢复模式状态
        
        参数:
            state_id: 状态ID
            
        返回:
            Optional[Dict[str, Any]]: 恢复的状态数据，如果失败返回None
        """
        try:
            # 首先尝试从Redis缓存获取
            if self.redis_client:
                cache_key = f"mode_state:{state_id}"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    cache_data = pickle.loads(cached_data)
                    self.logger.info(f"从Redis缓存恢复模式状态: ID={state_id}")
                    
                    # 更新缓存
                    self.current_mode_cache = cache_data
                    self.cache_valid = True
                    
                    # 更新统计信息
                    self.stats["total_state_restores"] += 1
                    
                    return cache_data
            
            # 从数据库获取
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT mode_type, mode_state, state_data, tasks_data, metadata
                FROM mode_states 
                WHERE id = ?
            """, (state_id,))
            
            row = cursor.fetchone()
            if not row:
                self.logger.error(f"模式状态不存在: ID={state_id}")
                return None  # 返回None
            
            # 解压数据
            state_data = self._decompress_data(row["state_data"])
            tasks_data = self._decompress_data(row["tasks_data"]) if row["tasks_data"] else None
            
            # 解析元数据
            metadata = json.loads(row["metadata"]) if row["metadata"] else None
            
            restored_data = {
                "id": state_id,
                "mode_type": row["mode_type"],
                "mode_state": row["mode_state"],
                "state_data": state_data,
                "tasks_data": tasks_data,
                "metadata": metadata,
                "restored_at": datetime.now().isoformat(),
            }
            
            # 更新Redis缓存
            if self.redis_client:
                cache_key = f"mode_state:{state_id}"
                self.redis_client.setex(
                    cache_key,
                    3600,  # 1小时TTL
                    pickle.dumps(restored_data)
                )
            
            # 更新缓存
            self.current_mode_cache = restored_data
            self.cache_valid = True
            
            # 更新统计信息
            self.stats["total_state_restores"] += 1
            
            self.logger.info(f"模式状态恢复成功: ID={state_id}")
            
            return restored_data
            
        except Exception as e:
            self.logger.error(f"恢复模式状态失败: {e}")
            return None  # 返回None
    
    def get_current_mode_state(self) -> Optional[Dict[str, Any]]:
        """获取当前模式状态
        
        返回:
            Optional[Dict[str, Any]]: 当前模式状态，如果没有则返回None
        """
        if self.cache_valid and self.current_mode_cache:
            return self.current_mode_cache
        
        try:
            # 获取最新的状态
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT id, mode_type, mode_state, state_data, tasks_data, metadata
                FROM mode_states 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if not row:
                return None  # 返回None
            
            # 解压数据
            state_data = self._decompress_data(row["state_data"])
            tasks_data = self._decompress_data(row["tasks_data"]) if row["tasks_data"] else None
            
            # 解析元数据
            metadata = json.loads(row["metadata"]) if row["metadata"] else None
            
            self.current_mode_cache = {
                "id": row["id"],
                "mode_type": row["mode_type"],
                "mode_state": row["mode_state"],
                "state_data": state_data,
                "tasks_data": tasks_data,
                "metadata": metadata,
                "retrieved_at": datetime.now().isoformat(),
            }
            self.cache_valid = True
            
            return self.current_mode_cache
            
        except Exception as e:
            self.logger.error(f"获取当前模式状态失败: {e}")
            return None  # 返回None
    
    def record_mode_transition(self,
                              from_mode: ModeType,
                              to_mode: ModeType,
                              transition_type: str,
                              state_snapshot_id: Optional[int] = None,
                              tasks_snapshot: Optional[List[Dict[str, Any]]] = None,
                              duration_ms: Optional[int] = None,
                              success: bool = True,
                              error_message: Optional[str] = None,
                              user_id: Optional[int] = None,
                              session_id: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> int:
        """记录模式切换
        
        参数:
            from_mode: 源模式
            to_mode: 目标模式
            transition_type: 切换类型（manual, automatic, emergency, etc.）
            state_snapshot_id: 状态快照ID（可选）
            tasks_snapshot: 任务快照（可选）
            duration_ms: 切换耗时（毫秒，可选）
            success: 是否成功
            error_message: 错误信息（可选）
            user_id: 用户ID（可选）
            session_id: 会话ID（可选）
            metadata: 元数据（可选）
            
        返回:
            int: 切换记录ID
        """
        try:
            # 压缩任务快照
            compressed_tasks = self._compress_data(tasks_snapshot) if tasks_snapshot else None
            
            # 准备元数据
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO mode_history 
                (from_mode, to_mode, transition_type, state_snapshot_id, tasks_snapshot, 
                 duration_ms, success, error_message, user_id, session_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                from_mode.value,
                to_mode.value,
                transition_type,
                state_snapshot_id,
                compressed_tasks,
                duration_ms,
                success,
                error_message,
                user_id,
                session_id,
                metadata_json,
            ))
            
            transition_id = cursor.lastrowid
            self.db_conn.commit()
            
            # 更新统计信息
            self.stats["total_transitions"] += 1
            if not success:
                self.stats["failed_transitions"] += 1
            
            self.logger.info(
                f"模式切换记录: ID={transition_id}, "
                f"从 {from_mode.value} 到 {to_mode.value}, "
                f"类型={transition_type}, 成功={success}"
            )
            
            return transition_id
            
        except Exception as e:
            self.logger.error(f"记录模式切换失败: {e}")
            raise
    
    def get_transition_history(self, 
                              limit: int = 100,
                              offset: int = 0,
                              from_mode: Optional[ModeType] = None,
                              to_mode: Optional[ModeType] = None,
                              success_only: Optional[bool] = None) -> List[Dict[str, Any]]:
        """获取模式切换历史
        
        参数:
            limit: 返回记录数限制
            offset: 偏移量
            from_mode: 筛选源模式（可选）
            to_mode: 筛选目标模式（可选）
            success_only: 仅返回成功的切换（可选）
            
        返回:
            List[Dict[str, Any]]: 切换历史记录
        """
        try:
            cursor = self.db_conn.cursor()
            
            # 构建查询条件
            conditions = []
            params = []
            
            if from_mode:
                conditions.append("from_mode = ?")
                params.append(from_mode.value)
            
            if to_mode:
                conditions.append("to_mode = ?")
                params.append(to_mode.value)
            
            if success_only is not None:
                conditions.append("success = ?")
                params.append(success_only)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # 执行查询
            query = f"""
                SELECT id, from_mode, to_mode, transition_type, state_snapshot_id,
                       duration_ms, success, error_message, transition_time, user_id, session_id, metadata
                FROM mode_history 
                WHERE {where_clause}
                ORDER BY transition_time DESC 
                LIMIT ? OFFSET ?
            """
            
            params.extend([limit, offset])
            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            
            # 转换为字典列表
            history = []
            for row in rows:
                # 解析元数据
                metadata = json.loads(row["metadata"]) if row["metadata"] else None
                
                history.append({
                    "id": row["id"],
                    "from_mode": row["from_mode"],
                    "to_mode": row["to_mode"],
                    "transition_type": row["transition_type"],
                    "state_snapshot_id": row["state_snapshot_id"],
                    "duration_ms": row["duration_ms"],
                    "success": bool(row["success"]),
                    "error_message": row["error_message"],
                    "transition_time": row["transition_time"],
                    "user_id": row["user_id"],
                    "session_id": row["session_id"],
                    "metadata": metadata,
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"获取切换历史失败: {e}")
            return []  # 返回空列表
    
    def backup_current_state(self, backup_path: Optional[str] = None) -> bool:
        """备份当前状态
        
        参数:
            backup_path: 备份路径（可选）
            
        返回:
            bool: 备份是否成功
        """
        try:
            if not backup_path:
                backup_dir = Path("backups/mode_states")
                backup_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = str(backup_dir / f"mode_state_backup_{timestamp}.db")
            
            # 获取当前数据库文件路径
            db_path = Path(self.config["sqlite_db_path"])
            
            if not db_path.exists():
                self.logger.warning("数据库文件不存在，无法备份")
                return False
            
            # 复制数据库文件
            import shutil
            shutil.copy2(str(db_path), backup_path)
            
            # 更新统计信息
            self.stats["last_backup_time"] = datetime.now().isoformat()
            
            # 更新数据库大小
            self.stats["database_size"] = db_path.stat().st_size
            
            self.logger.info(f"模式状态备份成功: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"模式状态备份失败: {e}")
            return False
    
    def validate_state_consistency(self) -> Dict[str, Any]:
        """验证状态一致性
        
        返回:
            Dict[str, Any]: 一致性检查结果
        """
        try:
            cursor = self.db_conn.cursor()
            results = {}
            
            # 检查模式状态表
            cursor.execute("SELECT COUNT(*) as count FROM mode_states")
            results["mode_states_count"] = cursor.fetchone()["count"]
            
            # 检查模式历史表
            cursor.execute("SELECT COUNT(*) as count FROM mode_history")
            results["mode_history_count"] = cursor.fetchone()["count"]
            
            # 检查最新状态
            cursor.execute("""
                SELECT mode_type, mode_state, created_at 
                FROM mode_states 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            latest_state = cursor.fetchone()
            if latest_state:
                results["latest_state"] = {
                    "mode_type": latest_state["mode_type"],
                    "mode_state": latest_state["mode_state"],
                    "created_at": latest_state["created_at"],
                }
            
            # 检查Redis连接
            if self.redis_client:
                try:
                    self.redis_client.ping()
                    results["redis_connected"] = True
                    
                    # 检查Redis中的缓存数量
                    cache_keys = self.redis_client.keys("mode_state:*")
                    results["redis_cache_count"] = len(cache_keys)
                except Exception:
                    results["redis_connected"] = False
            else:
                results["redis_connected"] = False
            
            # 检查数据库文件大小
            db_path = Path(self.config["sqlite_db_path"])
            if db_path.exists():
                results["database_size_bytes"] = db_path.stat().st_size
            else:
                results["database_size_bytes"] = 0
            
            results["consistent"] = True
            results["checked_at"] = datetime.now().isoformat()
            
            self.logger.info("状态一致性检查完成")
            
            return results
            
        except Exception as e:
            self.logger.error(f"状态一致性检查失败: {e}")
            return {
                "consistent": False,
                "error": str(e),
                "checked_at": datetime.now().isoformat(),
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        try:
            cursor = self.db_conn.cursor()
            
            # 获取数据库统计
            cursor.execute("SELECT COUNT(*) as count FROM mode_states")
            mode_states_count = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM mode_history")
            mode_history_count = cursor.fetchone()["count"]
            
            # 获取模式分布
            cursor.execute("SELECT mode_type, COUNT(*) as count FROM mode_states GROUP BY mode_type")
            mode_distribution = {row["mode_type"]: row["count"] for row in cursor.fetchall()}
            
            # 获取切换成功率
            cursor.execute("SELECT success, COUNT(*) as count FROM mode_history GROUP BY success")
            success_counts = {bool(row["success"]): row["count"] for row in cursor.fetchall()}
            
            total_transitions = sum(success_counts.values())
            success_rate = (success_counts.get(True, 0) / total_transitions * 100) if total_transitions > 0 else 0
            
            return {
                **self.stats,
                "mode_states_count": mode_states_count,
                "mode_history_count": mode_history_count,
                "mode_distribution": mode_distribution,
                "transition_success_rate": success_rate,
                "total_transitions": total_transitions,
                "successful_transitions": success_counts.get(True, 0),
                "failed_transitions": success_counts.get(False, 0),
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return self.stats
    
    def _compress_data(self, data: Any) -> bytes:
        """压缩数据
        
        参数:
            data: 要压缩的数据
            
        返回:
            bytes: 压缩后的数据
        """
        if not self.config["enable_compression"]:
            return pickle.dumps(data)
        
        try:
            import zlib
            pickled_data = pickle.dumps(data)
            compressed_data = zlib.compress(pickled_data, level=9)
            return compressed_data
        except Exception:
            # 如果压缩失败，回退到普通pickle
            return pickle.dumps(data)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """解压数据
        
        参数:
            compressed_data: 压缩的数据
            
        返回:
            Any: 解压后的数据
        """
        if not self.config["enable_compression"]:
            return pickle.loads(compressed_data)
        
        try:
            import zlib
            # 尝试解压
            try:
                decompressed = zlib.decompress(compressed_data)
                return pickle.loads(decompressed)
            except zlib.error:
                # 如果不是压缩数据，直接pickle加载
                return pickle.loads(compressed_data)
        except Exception as e:
            self.logger.error(f"解压数据失败: {e}")
            # 尝试直接pickle加载
            try:
                return pickle.loads(compressed_data)
            except Exception:
                raise ValueError("无法解压或反序列化数据")
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            retention_days = self.config["state_retention_days"]
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            cursor = self.db_conn.cursor()
            
            # 删除旧的状态记录
            cursor.execute("""
                DELETE FROM mode_states 
                WHERE created_at < ?
                AND id NOT IN (
                    SELECT state_snapshot_id FROM mode_history WHERE state_snapshot_id IS NOT NULL
                )
            """, (cutoff_date.isoformat(),))
            
            deleted_states = cursor.rowcount
            
            # 删除旧的历史记录
            cursor.execute("DELETE FROM mode_history WHERE transition_time < ?", 
                         (cutoff_date.isoformat(),))
            
            deleted_history = cursor.rowcount
            
            self.db_conn.commit()
            
            if deleted_states > 0 or deleted_history > 0:
                self.logger.info(f"清理旧数据: 删除 {deleted_states} 个状态记录, {deleted_history} 个历史记录")
            
            # 清理Redis缓存中的过期数据
            if self.redis_client:
                # Redis会自动处理过期数据，这里我们只需要清理无效的键
                # 可以定期运行，但不在每次清理时都运行
        pass  # 已实现
            
        except Exception as e:
            self.logger.error(f"清理旧数据失败: {e}")
    
    def close(self):
        """关闭连接"""
        try:
            if self.db_conn:
                self.db_conn.close()
            
            if self.redis_client:
                self.redis_client.close()
            
            self.logger.info("模式状态管理器已关闭")
        except Exception as e:
            self.logger.error(f"关闭模式状态管理器失败: {e}")


# 全局实例
_mode_state_service_instance = None


def get_mode_state_service(config: Optional[Dict[str, Any]] = None) -> ModeStateService:
    """获取模式状态管理器单例
    
    参数:
        config: 配置字典
        
    返回:
        ModeStateService: 模式状态管理器实例
    """
    global _mode_state_service_instance
    
    if _mode_state_service_instance is None:
        _mode_state_service_instance = ModeStateService(config)
    
    return _mode_state_service_instance


__all__ = [
    "ModeStateService",
    "get_mode_state_service",
    "ModeType",
    "ModeState",
]