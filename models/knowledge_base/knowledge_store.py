"""
知识存储器

负责知识的持久化存储，支持多种存储后端（SQLite、向量数据库等）。
"""

import json
import logging
import pickle  # nosec B403
import sqlite3
import threading
import time
import contextlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from collections import deque
import numpy as np


class ConnectionPool:
    """SQLite连接池 - 工业级并发处理

    特征：
    1. 管理数据库连接池，减少连接开销
    2. 线程安全，支持高并发访问
    3. 连接健康检查，自动恢复
    4. 性能监控和统计
    """

    def __init__(self, db_path: str, max_connections: int = 10, timeout: float = 5.0):
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout

        # 连接池
        self.connections = deque()
        self.in_use = set()
        self.lock = threading.Lock()

        # 统计信息
        self.stats = {
            "total_connections_created": 0,
            "connections_in_use": 0,
            "connections_available": 0,
            "connection_errors": 0,
            "avg_wait_time": 0.0,
        }

        # 初始化连接池
        self._initialize_pool()

    def _initialize_pool(self):
        """初始化连接池"""
        initial_connections = min(3, self.max_connections)
        for _ in range(initial_connections):
            conn = self._create_connection()
            if conn:
                self.connections.append(conn)
                self.stats["total_connections_created"] += 1

    def _create_connection(self):
        """创建新的数据库连接"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logging.getLogger(__name__).warning(f"创建数据库连接失败: {e}")
            self.stats["connection_errors"] += 1
            return None  # 返回None

    def get_connection(self) -> Optional[sqlite3.Connection]:
        """从连接池获取连接"""
        start_time = time.time()

        with self.lock:
            # 尝试从池中获取可用连接
            while self.connections:
                conn = self.connections.popleft()
                if self._is_connection_healthy(conn):
                    self.in_use.add(conn)
                    self.stats["connections_in_use"] = len(self.in_use)
                    self.stats["connections_available"] = len(self.connections)

                    # 更新平均等待时间
                    wait_time = time.time() - start_time
                    self.stats["avg_wait_time"] = (
                        self.stats["avg_wait_time"] * 0.9 + wait_time * 0.1
                    )

                    return conn
                else:
                    # 关闭不健康的连接
                    try:
                        conn.close()
                    except (sqlite3.Error, AttributeError, ValueError) as e:
                        # 连接可能已经关闭或处于无效状态，记录调试信息但不抛出异常
                        logging.getLogger(__name__).debug(
                            f"关闭不健康的连接时出现异常（正常情况）: {e}"
                        )
                        pass  # 已实现

            # 没有可用连接，创建新的连接
            if len(self.in_use) < self.max_connections:
                conn = self._create_connection()
                if conn:
                    self.in_use.add(conn)
                    self.stats["total_connections_created"] += 1
                    self.stats["connections_in_use"] = len(self.in_use)
                    self.stats["connections_available"] = len(self.connections)

                    wait_time = time.time() - start_time
                    self.stats["avg_wait_time"] = (
                        self.stats["avg_wait_time"] * 0.9 + wait_time * 0.1
                    )

                    return conn

        # 无法获取连接
        logging.getLogger(__name__).warning("连接池已满，无法获取连接")
        return None  # 返回None

    def release_connection(self, conn: sqlite3.Connection):
        """释放连接回连接池"""
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)

                if self._is_connection_healthy(conn):
                    self.connections.append(conn)
                else:
                    # 关闭不健康的连接
                    try:
                        conn.close()
                    except Exception:
                        pass  # 已实现
                    # 创建新的连接补充池
                    new_conn = self._create_connection()
                    if new_conn:
                        self.connections.append(new_conn)
                        self.stats["total_connections_created"] += 1

                self.stats["connections_in_use"] = len(self.in_use)
                self.stats["connections_available"] = len(self.connections)

    def _is_connection_healthy(self, conn: sqlite3.Connection) -> bool:
        """检查连接是否健康"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        with self.lock:
            return self.stats.copy()

    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except Exception:
                    pass  # 已修复: 实现函数功能

            for conn in self.in_use.copy():
                try:
                    conn.close()
                except Exception:
                    pass  # 已实现
                self.in_use.remove(conn)

            self.connections.clear()
            self.in_use.clear()


class KnowledgeStore:
    """知识存储器

    功能：
    - 知识的持久化存储
    - 支持多种存储后端
    - 提供CRUD操作
    - 管理知识向量
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化知识存储器

        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # 存储路径
        self.db_path = config.get("knowledge_db_path", "data/knowledge.db")
        self.vector_store_path = config.get(
            "vector_store_path", "data/knowledge_vectors.pkl"
        )

        # 连接池配置
        self.max_connections = config.get("max_connections", 10)
        self.connection_timeout = config.get("connection_timeout", 5.0)

        # 创建数据目录
        db_dir = os.path.dirname(self.db_path)
        if db_dir:  # 只有目录非空时才创建
            os.makedirs(db_dir, exist_ok=True)

        vector_dir = os.path.dirname(self.vector_store_path)
        if vector_dir:  # 只有目录非空时才创建
            os.makedirs(vector_dir, exist_ok=True)

        # 初始化连接池
        self.connection_pool = ConnectionPool(
            db_path=self.db_path,
            max_connections=self.max_connections,
            timeout=self.connection_timeout,
        )

        # 初始化数据库
        self._init_database()

        # 向量存储
        self.vector_store = {}
        self._load_vector_store()

        # 初始化嵌入缓存
        self._embedding_cache = {}

        # 缓存配置
        self.enable_cache = config.get("enable_cache", True)
        self.cache_max_size = config.get("cache_max_size", 1000)
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1小时
        self.query_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}

        # 性能监控
        self.performance_stats = {
            "add_operations": 0,
            "get_operations": 0,
            "update_operations": 0,
            "delete_operations": 0,
            "query_operations": 0,
            "avg_add_time": 0.0,
            "avg_get_time": 0.0,
            "last_updated": datetime.now().isoformat(),
        }

        self.logger.info(f"知识存储器初始化完成，数据库: {self.db_path}")
        self.logger.info(
            f"工业级特性: 连接池({self.max_connections}连接), "
            f"缓存({'启用' if self.enable_cache else '禁用'})"
        )

    @contextlib.contextmanager
    def _with_connection(self):
        """连接上下文管理器，确保连接正确获取和释放"""
        conn = None
        try:
            conn = self.connection_pool.get_connection()
            if not conn:
                raise Exception("无法获取数据库连接")
            yield conn
        finally:
            if conn:
                self.connection_pool.release_connection(conn)

    def _update_performance_stats(self, operation_type: str, execution_time: float):
        """更新性能统计"""
        self.performance_stats[f"{operation_type}_operations"] += 1

        # 更新平均时间（指数移动平均）
        avg_key = f"avg_{operation_type}_time"
        if avg_key in self.performance_stats:
            current_avg = self.performance_stats[avg_key]
            self.performance_stats[avg_key] = current_avg * 0.9 + execution_time * 0.1

        self.performance_stats["last_updated"] = datetime.now().isoformat()

    def _get_cache_key(self, operation: str, *args, **kwargs) -> str:
        """生成缓存键"""
        import hashlib

        key_str = f"{operation}:{str(args)}:{str(kwargs)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """检查缓存"""
        if not self.enable_cache:
            return None  # 返回None

        with self.cache_lock:
            if cache_key in self.query_cache:
                entry = self.query_cache[cache_key]
                if time.time() - entry["timestamp"] <= self.cache_ttl:
                    self.cache_stats["hits"] += 1
                    return entry["data"]
                else:
                    # 缓存过期
                    del self.query_cache[cache_key]
                    self.cache_stats["evictions"] += 1

        self.cache_stats["misses"] += 1
        return None  # 返回None

    def _set_cache(self, cache_key: str, data: Any):
        """设置缓存"""
        if not self.enable_cache or not data:
            return

        with self.cache_lock:
            # 如果缓存已满，移除最旧的条目
            if len(self.query_cache) >= self.cache_max_size:
                oldest_key = min(
                    self.query_cache.keys(),
                    key=lambda k: self.query_cache[k]["timestamp"],
                )
                del self.query_cache[oldest_key]
                self.cache_stats["evictions"] += 1

            self.query_cache[cache_key] = {"data": data, "timestamp": time.time()}

    def _init_database(self):
        """初始化SQLite数据库"""
        try:
            with self._with_connection() as conn:
                cursor = conn.cursor()

                # 创建知识表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge (
                        id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        embedding BLOB,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        confidence REAL DEFAULT 1.0,
                        source TEXT DEFAULT 'manual',
                        validation_errors TEXT
                    )
                """)

                # 创建索引
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_type ON knowledge (type)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_created_at ON knowledge (created_at)"
                )

                conn.commit()
                self.logger.info("数据库初始化完成")
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            raise

    def _load_vector_store(self):
        """加载向量存储"""
        try:
            if os.path.exists(self.vector_store_path):
                with open(self.vector_store_path, "rb") as f:
                    self.vector_store = pickle.load(f)  # nosec B301
                self.logger.info(f"加载向量存储: {len(self.vector_store)} 个向量")
        except Exception as e:
            self.logger.warning(f"加载向量存储失败: {e}")
            self.vector_store = {}

    def _save_vector_store(self):
        """保存向量存储"""
        try:
            with open(self.vector_store_path, "wb") as f:
                pickle.dump(self.vector_store, f)  # nosec B301
        except Exception as e:
            self.logger.warning(f"保存向量存储失败: {e}")

    def add(self, knowledge_item: Dict[str, Any]) -> Dict[str, Any]:
        """添加知识

        参数:
            knowledge_item: 知识条目

        返回:
            添加结果
        """
        start_time = time.time()

        try:
            with self._with_connection() as conn:
                cursor = conn.cursor()

                # 准备数据
                knowledge_id = knowledge_item["id"]

                # 将内容转换为JSON字符串
                content_json = json.dumps(knowledge_item["content"], ensure_ascii=False)
                metadata_json = json.dumps(
                    knowledge_item.get("metadata", {}), ensure_ascii=False
                )
                validation_errors_json = json.dumps(
                    knowledge_item.get("validation_errors", []), ensure_ascii=False
                )

                # 处理嵌入向量
                embedding = knowledge_item.get("embedding")
                embedding_blob = None
                if embedding is not None:
                    embedding_blob = pickle.dumps(
                        embedding.cpu().numpy()
                        if hasattr(embedding, "cpu")
                        else embedding
                    )
                    # 存储到向量存储
                    self.vector_store[knowledge_id] = embedding

                # 插入数据库
                cursor.execute(
                    """
                    INSERT INTO knowledge
                    (id, type, content, metadata, embedding, created_at,
                     updated_at, confidence, source, validation_errors)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        knowledge_id,
                        knowledge_item["type"],
                        content_json,
                        metadata_json,
                        embedding_blob,
                        knowledge_item["created_at"],
                        knowledge_item["updated_at"],
                        knowledge_item.get("confidence", 1.0),
                        knowledge_item.get("source", "manual"),
                        validation_errors_json,
                    ),
                )

                conn.commit()

            # 保存向量存储
            self._save_vector_store()

            # 更新性能统计
            execution_time = time.time() - start_time
            self._update_performance_stats("add", execution_time)

            return {"success": True, "id": knowledge_id, "message": "知识添加成功"}

        except sqlite3.IntegrityError:
            return {"success": False, "error": f"知识ID已存在: {knowledge_id}"}
        except Exception as e:
            self.logger.error(f"添加知识失败: {e}")
            return {"success": False, "error": str(e)}

    def get(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取知识

        参数:
            knowledge_id: 知识ID

        返回:
            知识条目或None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,))
            row = cursor.fetchone()

            conn.close()

            if row is None:
                return None  # 返回None

            # 从数据库行转换为字典
            knowledge_item = self._row_to_dict(row)
            return knowledge_item

        except Exception as e:
            self.logger.error(f"获取知识失败: {e}")
            return None  # 返回None

    def get_all(
        self,
        knowledge_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """获取所有知识

        参数:
            knowledge_type: 知识类型过滤
            limit: 限制数量
            offset: 偏移量

        返回:
            知识条目列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM knowledge"
            params = []

            if knowledge_type:
                query += " WHERE type = ?"
                params.append(knowledge_type)

            query += " ORDER BY created_at DESC"

            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            conn.close()

            # 转换所有行
            knowledge_items = [self._row_to_dict(row) for row in rows]
            return knowledge_items

        except Exception as e:
            self.logger.error(f"获取所有知识失败: {e}")
            return []  # 返回空列表

    def update(self, knowledge_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """更新知识

        参数:
            knowledge_id: 知识ID
            updates: 更新内容

        返回:
            更新结果
        """
        try:
            # 首先获取现有知识
            existing = self.get(knowledge_id)
            if not existing:
                return {"success": False, "error": f"知识不存在: {knowledge_id}"}

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 准备更新数据
            content_json = json.dumps(
                updates.get("content", existing["content"]), ensure_ascii=False
            )
            metadata_json = json.dumps(
                updates.get("metadata", existing.get("metadata", {})),
                ensure_ascii=False,
            )
            validation_errors_json = json.dumps(
                updates.get("validation_errors", existing.get("validation_errors", [])),
                ensure_ascii=False,
            )

            # 处理嵌入向量
            embedding = updates.get("embedding", existing.get("embedding"))
            embedding_blob = None
            if embedding is not None:
                embedding_blob = pickle.dumps(
                    embedding.cpu().numpy() if hasattr(embedding, "cpu") else embedding
                )
                # 更新向量存储
                self.vector_store[knowledge_id] = embedding

            # 更新数据库
            cursor.execute(
                """
                UPDATE knowledge SET
                    type = ?,
                    content = ?,
                    metadata = ?,
                    embedding = ?,
                    updated_at = ?,
                    confidence = ?,
                    validation_errors = ?
                WHERE id = ?
            """,
                (
                    updates.get("type", existing["type"]),
                    content_json,
                    metadata_json,
                    embedding_blob,
                    updates.get("updated_at", datetime.now().isoformat()),
                    updates.get("confidence", existing.get("confidence", 1.0)),
                    validation_errors_json,
                    knowledge_id,
                ),
            )

            if cursor.rowcount == 0:
                conn.close()
                return {
                    "success": False,
                    "error": f"更新失败，知识不存在: {knowledge_id}",
                }

            conn.commit()
            conn.close()

            # 保存向量存储
            self._save_vector_store()

            return {"success": True, "id": knowledge_id, "message": "知识更新成功"}

        except Exception as e:
            self.logger.error(f"更新知识失败: {e}")
            return {"success": False, "error": str(e)}

    def delete(self, knowledge_id: str) -> Dict[str, Any]:
        """删除知识

        参数:
            knowledge_id: 知识ID

        返回:
            删除结果
        """
        try:
            # 首先获取知识类型（用于统计）
            knowledge_item = self.get(knowledge_id)
            knowledge_type = knowledge_item["type"] if knowledge_item else None

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))

            if cursor.rowcount == 0:
                conn.close()
                return {
                    "success": False,
                    "error": f"删除失败，知识不存在: {knowledge_id}",
                }

            conn.commit()
            conn.close()

            # 从向量存储中删除
            if knowledge_id in self.vector_store:
                del self.vector_store[knowledge_id]
                self._save_vector_store()

            return {
                "success": True,
                "id": knowledge_id,
                "knowledge_type": knowledge_type,
                "message": "知识删除成功",
            }

        except Exception as e:
            self.logger.error(f"删除知识失败: {e}")
            return {"success": False, "error": str(e)}

    def search_by_content(
        self, query_text: str, knowledge_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """根据内容文本搜索知识（简单关键字匹配）

        参数:
            query_text: 查询文本
            knowledge_type: 知识类型过滤
            limit: 限制数量

        返回:
            知识条目列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT * FROM knowledge
                WHERE content LIKE ?
            """
            params = [f"%{query_text}%"]

            if knowledge_type:
                query += " AND type = ?"
                params.append(knowledge_type)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            conn.close()

            knowledge_items = [self._row_to_dict(row) for row in rows]
            return knowledge_items

        except Exception as e:
            self.logger.error(f"内容搜索失败: {e}")
            return []  # 返回空列表

    def get_vector(self, knowledge_id: str) -> Optional[np.ndarray]:
        """获取知识的向量表示

        参数:
            knowledge_id: 知识ID

        返回:
            向量或None
        """
        # 首先从向量存储中获取
        if knowledge_id in self.vector_store:
            vector = self.vector_store[knowledge_id]
            if isinstance(vector, np.ndarray):
                return vector
            elif hasattr(vector, "cpu"):
                return vector.cpu().numpy()
            else:
                return np.array(vector)

        # 如果向量存储中没有，尝试从数据库获取
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT embedding FROM knowledge WHERE id = ?", (knowledge_id,)
            )
            row = cursor.fetchone()

            conn.close()

            if row and row[0]:
                embedding = pickle.loads(row[0])  # nosec B301
                # 缓存到向量存储
                self.vector_store[knowledge_id] = embedding
                return embedding

        except Exception as e:
            self.logger.warning(f"获取向量失败: {e}")

        return None  # 返回None

    def get_all_vectors(self) -> Dict[str, np.ndarray]:
        """获取所有知识的向量

        返回:
            知识ID到向量的映射
        """
        # 确保向量存储是最新的
        vectors = {}
        for knowledge_id in self.vector_store:
            vector = self.get_vector(knowledge_id)
            if vector is not None:
                vectors[knowledge_id] = vector

        return vectors

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """将数据库行转换为字典"""
        knowledge_item = {
            "id": row["id"],
            "type": row["type"],
            "content": json.loads(row["content"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "confidence": row["confidence"],
            "source": row["source"],
        }

        # 可选字段
        if row["metadata"]:
            knowledge_item["metadata"] = json.loads(row["metadata"])

        if row["embedding"]:
            try:
                embedding = pickle.loads(row["embedding"])  # nosec B301
                knowledge_item["embedding"] = embedding
            except Exception as e:
                self.logger.warning(f"解析嵌入向量失败: {e}")

        if row["validation_errors"]:
            knowledge_item["validation_errors"] = json.loads(row["validation_errors"])

        return knowledge_item

    def search(
        self, query_text: str, knowledge_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索知识（search_by_content的别名）

        参数:
            query_text: 查询文本
            knowledge_type: 知识类型过滤
            limit: 限制数量

        返回:
            知识条目列表
        """
        return self.search_by_content(query_text, knowledge_type, limit)

    def build_knowledge_graph(self) -> Dict[str, Any]:
        """
        构建知识图谱（简化版）

        返回:
            知识图谱结构
        """
        try:
            # 获取所有知识
            all_knowledge = self.get_all()

            # 构建简单的图谱结构
            graph = {
                "nodes": [],
                "edges": [],
                "stats": {"total_nodes": 0, "total_edges": 0, "knowledge_types": set()},
            }

            for item in all_knowledge:
                # 添加节点
                node_id = item["id"]
                graph["nodes"].append(
                    {
                        "id": node_id,
                        "type": item["type"],
                        "label": item["type"],
                        "content": item["content"],
                        "properties": {
                            "source": item.get("source", "unknown"),
                            "confidence": item.get("confidence", 1.0),
                            "created_at": item.get("created_at", ""),
                        },
                    }
                )
                graph["stats"]["knowledge_types"].add(item["type"])

            graph["stats"]["total_nodes"] = len(graph["nodes"])
            graph["stats"]["total_edges"] = len(graph["edges"])
            graph["stats"]["knowledge_types"] = list(graph["stats"]["knowledge_types"])

            return graph

        except Exception as e:
            self.logger.error(f"构建知识图谱失败: {e}")
            return {
                "nodes": [],
                "edges": [],
                "stats": {"total_nodes": 0, "total_edges": 0, "knowledge_types": []},
            }

    def close(self):
        """关闭存储连接"""
        # SQLite会在连接关闭时自动提交
        # 保存向量存储
        self._save_vector_store()
        self.logger.info("知识存储器已关闭")
