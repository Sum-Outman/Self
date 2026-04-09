"""
知识库相关数据库模型
包含知识库项目和搜索历史模型
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from datetime import datetime

from ..core.database import Base


class KnowledgeItem(Base):
    """知识库项目"""

    __tablename__ = "knowledge_items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    type = Column(
        String(50), default="document", index=True
    )  # text, image, video, audio, document, code, dataset
    content = Column(Text)  # 文本内容或文件路径
    file_path = Column(String(500))  # 实际文件存储路径
    size = Column(Integer, default=0)  # 文件大小（字节）
    upload_date = Column(DateTime, default=datetime.utcnow, index=True)
    tags = Column(Text)  # JSON字符串存储标签列表
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    access_count = Column(Integer, default=0, index=True)  # 访问次数
    last_accessed = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(Text)  # JSON字符串存储元数据
    embedding = Column(Text)  # JSON数组存储向量嵌入
    is_public = Column(Boolean, default=True)
    checksum = Column(String(64))  # 文件校验和

    # 关系
    user = relationship("User")


class KnowledgeSearchHistory(Base):
    """知识搜索历史"""

    __tablename__ = "knowledge_search_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    query = Column(Text, nullable=False)
    filters = Column(Text)  # JSON字符串存储过滤条件
    results_count = Column(Integer, default=0)
    search_time = Column(DateTime, default=datetime.utcnow)

    # 关系
    user = relationship("User")


__all__ = ["KnowledgeItem", "KnowledgeSearchHistory"]
