"""
聊天相关数据库模型
包含聊天会话和聊天消息模型
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


class ChatSession(Base):
    """聊天会话"""
    
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), default="新会话")  # 会话标题
    model_name = Column(String(100))  # 使用的模型名称
    is_active = Column(Boolean, default=True)  # 是否活跃
    message_count = Column(Integer, default=0)  # 消息数量
    total_tokens = Column(Integer, default=0)  # 总token数
    last_message_at = Column(DateTime)  # 最后消息时间
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """聊天消息"""
    
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)  # 消息内容
    tokens = Column(Integer, default=0)  # token数量
    message_metadata = Column(Text)  # JSON格式的元数据
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    session = relationship("ChatSession", back_populates="messages")


# 在User模型中添加反向关系（需要在user.py中更新）
# 这里只是声明，实际关系在user.py中定义
__all__ = ["ChatSession", "ChatMessage"]