"""
记忆相关数据库模型
包含记忆存储和记忆关联模型
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Float,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from datetime import datetime

from ..core.database import Base


class Memory(Base):
    """记忆存储"""

    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)  # 记忆内容
    content_type = Column(String(50))  # text, image, audio, video
    embedding = Column(Text)  # JSON数组存储向量嵌入
    importance = Column(Float, default=1.0)  # 重要性评分
    memory_type = Column(String(20), default="short_term")  # short_term, long_term
    accessed_count = Column(Integer, default=0)  # 访问次数
    last_accessed = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # 过期时间（短期记忆）

    # 关系
    user = relationship("User", back_populates="memories")
    associations = relationship("MemoryAssociation", back_populates="source_memory", foreign_keys="[MemoryAssociation.source_memory_id]")


class MemoryAssociation(Base):
    """记忆关联"""

    __tablename__ = "memory_associations"

    id = Column(Integer, primary_key=True, index=True)
    source_memory_id = Column(Integer, ForeignKey("memories.id"), nullable=False)
    target_memory_id = Column(Integer, ForeignKey("memories.id"), nullable=False)
    association_type = Column(String(50))  # similarity, temporal, causal, etc.
    strength = Column(Float, default=1.0)  # 关联强度
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    source_memory = relationship(
        "Memory", foreign_keys=[source_memory_id], back_populates="associations"
    )
    target_memory = relationship("Memory", foreign_keys=[target_memory_id])


__all__ = ["Memory", "MemoryAssociation"]