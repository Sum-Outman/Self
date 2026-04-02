"""
AGI相关数据库模型
包含AGI模型、训练任务等模型
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


class AGIModel(Base):
    """AGI模型信息"""

    __tablename__ = "agi_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    model_type = Column(String(50))  # transformer, multimodal, etc.
    model_path = Column(String(200))
    config = Column(Text)  # JSON配置
    version = Column(String(20))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    training_results = relationship("TrainingResult", back_populates="model", cascade="all, delete-orphan")


class TrainingJob(Base):
    """训练任务"""

    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("agi_models.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(
        String(20), default="pending"
    )  # pending, running, completed, failed
    progress = Column(Float, default=0.0)
    config = Column(Text)  # JSON训练配置
    result = Column(Text)  # JSON训练结果
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class ProfessionalCapability(Base):
    """专业领域能力"""

    __tablename__ = "professional_capabilities"

    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    icon = Column(String(50), default="brain")
    enabled = Column(Boolean, default=False)
    status = Column(String(20), default="inactive")  # active, inactive, testing, error
    performance = Column(Float, default=0.0)  # 0-100 性能评分
    last_tested = Column(DateTime)
    tests_passed = Column(Integer, default=0)
    tests_failed = Column(Integer, default=0)
    tests_total = Column(Integer, default=0)
    capabilities_list = Column(Text)  # JSON列表存储能力项
    type = Column(String(50), default="general")  # programming, mathematics, physics, medical, financial, chemistry, general
    level = Column(Integer, default=1)  # 能力级别
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


__all__ = ["AGIModel", "TrainingJob", "ProfessionalCapability"]