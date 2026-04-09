#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库工具模块
提供通用的数据库操作函数，减少重复的查询代码

设计原则：
1. 统一性：所有数据库操作使用相同模式
2. 安全性：自动处理会话管理和事务
3. 可读性：提供清晰的函数名和参数
4. 性能：优化查询，避免N+1问题
5. 错误处理：统一的错误处理机制
"""

import logging
from typing import Type, TypeVar, List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session, Query

from .error_handler import db_error_handler

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar("T")  # 模型类型
ModelType = Type[T]


@db_error_handler
def get_by_id(db: Session, model: ModelType, id: Any) -> Optional[T]:
    """根据ID获取单个记录

    参数:
        db: 数据库会话
        model: SQLAlchemy模型类
        id: 主键值

    返回:
        模型实例或None
    """
    return db.query(model).get(id)


@db_error_handler
def get_all(
    db: Session,
    model: ModelType,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> List[T]:
    """获取所有记录

    参数:
        db: 数据库会话
        model: SQLAlchemy模型类
        limit: 限制返回数量
        offset: 偏移量

    返回:
        模型实例列表
    """
    query = db.query(model)
    if offset is not None:
        query = query.offset(offset)
    if limit is not None:
        query = query.limit(limit)
    return query.all()


@db_error_handler
def filter_by(db: Session, model: ModelType, **filters) -> List[T]:
    """根据条件过滤记录

    参数:
        db: 数据库会话
        model: SQLAlchemy模型类
        **filters: 过滤条件

    返回:
        模型实例列表
    """
    return db.query(model).filter_by(**filters).all()


@db_error_handler
def filter_by_one(db: Session, model: ModelType, **filters) -> Optional[T]:
    """根据条件获取单个记录

    参数:
        db: 数据库会话
        model: SQLAlchemy模型类
        **filters: 过滤条件

    返回:
        模型实例或None
    """
    return db.query(model).filter_by(**filters).first()


@db_error_handler
def create(db: Session, model: ModelType, **data) -> T:
    """创建新记录

    参数:
        db: 数据库会话
        model: SQLAlchemy模型类
        **data: 记录数据

    返回:
        创建的模型实例
    """
    instance = model(**data)
    db.add(instance)
    db.commit()
    db.refresh(instance)
    return instance


@db_error_handler
def update(db: Session, instance: T, **data) -> T:
    """更新现有记录

    参数:
        db: 数据库会话
        instance: 要更新的模型实例
        **data: 更新的数据

    返回:
        更新后的模型实例
    """
    for key, value in data.items():
        if hasattr(instance, key):
            setattr(instance, key, value)

    db.commit()
    db.refresh(instance)
    return instance


@db_error_handler
def delete(db: Session, instance: T) -> bool:
    """删除记录

    参数:
        db: 数据库会话
        instance: 要删除的模型实例

    返回:
        是否成功删除
    """
    db.delete(instance)
    db.commit()
    return True


@db_error_handler
def delete_by_id(db: Session, model: ModelType, id: Any) -> bool:
    """根据ID删除记录

    参数:
        db: 数据库会话
        model: SQLAlchemy模型类
        id: 主键值

    返回:
        是否成功删除
    """
    instance = get_by_id(db, model, id)
    if instance:
        return delete(db, instance)
    return False


@db_error_handler
def count(db: Session, model: ModelType, **filters) -> int:
    """统计记录数量

    参数:
        db: 数据库会话
        model: SQLAlchemy模型类
        **filters: 过滤条件

    返回:
        记录数量
    """
    query = db.query(model)
    if filters:
        query = query.filter_by(**filters)
    return query.count()


@db_error_handler
def exists(db: Session, model: ModelType, **filters) -> bool:
    """检查记录是否存在

    参数:
        db: 数据库会话
        model: SQLAlchemy模型类
        **filters: 过滤条件

    返回:
        是否存在
    """
    return db.query(model).filter_by(**filters).first() is not None


@db_error_handler
def bulk_create(
    db: Session, model: ModelType, data_list: List[Dict[str, Any]]
) -> List[T]:
    """批量创建记录

    参数:
        db: 数据库会话
        model: SQLAlchemy模型类
        data_list: 数据字典列表

    返回:
        创建的模型实例列表
    """
    instances = [model(**data) for data in data_list]
    db.add_all(instances)
    db.commit()
    for instance in instances:
        db.refresh(instance)
    return instances


@db_error_handler
def bulk_update(
    db: Session, model: ModelType, updates: List[Tuple[Any, Dict[str, Any]]]
) -> int:
    """批量更新记录

    参数:
        db: 数据库会话
        model: SQLAlchemy模型类
        updates: (主键值, 更新数据)列表

    返回:
        更新的记录数量
    """
    updated_count = 0
    for id_value, data in updates:
        instance = get_by_id(db, model, id_value)
        if instance:
            update(db, instance, **data)
            updated_count += 1
    return updated_count


@db_error_handler
def execute_raw_sql(
    db: Session, sql: str, params: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """执行原始SQL查询

    参数:
        db: 数据库会话
        sql: SQL语句
        params: 参数

    返回:
        结果字典列表
    """
    if params is None:
        params = {}

    result = db.execute(sql, params)

    # 将结果转换为字典列表
    columns = result.keys()
    return [dict(zip(columns, row)) for row in result]


class DatabaseManager:
    """数据库管理器

    提供更高级的数据库操作，支持事务和复杂查询
    """

    def __init__(self, db: Session):
        self.db = db

    def transaction(self):
        """事务上下文管理器"""

        class TransactionContext:
            def __init__(self, db: Session):
                self.db = db
                self.original_autocommit = db.autocommit
                self.original_autoflush = db.autoflush

            def __enter__(self):
                self.db.begin()
                return self.db

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    self.db.rollback()
                else:
                    self.db.commit()

        return TransactionContext(self.db)

    def paginate(
        self, query: Query, page: int = 1, per_page: int = 20
    ) -> Dict[str, Any]:
        """分页查询

        参数:
            query: SQLAlchemy查询对象
            page: 页码（从1开始）
            per_page: 每页数量

        返回:
            分页结果字典
        """
        if page < 1:
            page = 1

        total = query.count()
        items = query.offset((page - 1) * per_page).limit(per_page).all()

        return {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page if per_page > 0 else 0,
            "has_prev": page > 1,
            "has_next": page * per_page < total,
        }


# 导出常用函数
__all__ = [
    "get_by_id",
    "get_all",
    "filter_by",
    "filter_by_one",
    "create",
    "update",
    "delete",
    "delete_by_id",
    "count",
    "exists",
    "bulk_create",
    "bulk_update",
    "execute_raw_sql",
    "DatabaseManager",
]
