"""
系统日志数据库模型
记录系统运行时日志，支持日志查询和分析
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Index
from datetime import datetime

from ..core.database import Base


class SystemLog(Base):
    """系统日志模型"""
    
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(20), nullable=False, index=True)  # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    source = Column(String(100), nullable=False, index=True)  # 日志来源模块，例如：backend.main, training.trainer
    message = Column(Text, nullable=False)  # 日志消息内容
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)  # 日志时间戳
    log_metadata = Column(Text, nullable=True)  # 附加元数据，JSON格式（避免使用SQLAlchemy保留字metadata）
    
    # 创建索引
    __table_args__ = (
        Index('ix_system_logs_level_timestamp', 'level', 'timestamp'),
        Index('ix_system_logs_source_timestamp', 'source', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, level='{self.level}', source='{self.source}', timestamp={self.timestamp})>"
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "level": self.level,
            "source": self.source,
            "message": self.message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.log_metadata  # 注意：字段名已改为log_metadata，但API输出仍使用metadata
        }