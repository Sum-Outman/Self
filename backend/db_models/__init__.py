"""
数据库模型模块
包含所有SQLAlchemy数据库模型
"""

from .user import (
    User,
    APIKey,
    UserSession,
    PasswordResetToken,
    EmailVerificationToken,
    LoginAttempt,
    TwoFactorTempSession,
    EmailTwoFactorCode,
)
from .agi import AGIModel, TrainingJob, ProfessionalCapability
from .memory import Memory, MemoryAssociation
from .knowledge import KnowledgeItem, KnowledgeSearchHistory
from .demonstration import (
    Demonstration,
    DemonstrationFrame,
    CameraFrame,
    DemonstrationTask,
    TrainingResult,
    DemonstrationType,
    DemonstrationStatus,
    DemonstrationFormat,
)
from .robot import Robot, RobotJoint, RobotSensor
from .robot_market import (
    RobotMarketListing,
    RobotMarketRating,
    RobotMarketComment,
    RobotMarketDownload,
    RobotMarketFavorite,
    RobotMarketStatus,
    RobotMarketCategory,
)
from .chat import ChatSession, ChatMessage
from .system_log import SystemLog

__all__ = [
    "User",
    "APIKey",
    "UserSession",
    "PasswordResetToken",
    "EmailVerificationToken",
    "LoginAttempt",
    "TwoFactorTempSession",
    "EmailTwoFactorCode",
    "AGIModel",
    "TrainingJob",
    "ProfessionalCapability",
    "Memory",
    "MemoryAssociation",
    "KnowledgeItem",
    "KnowledgeSearchHistory",
    "Demonstration",
    "DemonstrationFrame",
    "CameraFrame",
    "DemonstrationTask",
    "TrainingResult",
    "DemonstrationType",
    "DemonstrationStatus",
    "DemonstrationFormat",
    "Robot",
    "RobotJoint",
    "RobotSensor",
    "RobotMarketListing",
    "RobotMarketRating",
    "RobotMarketComment",
    "RobotMarketDownload",
    "RobotMarketFavorite",
    "RobotMarketStatus",
    "RobotMarketCategory",
    "ChatSession",
    "ChatMessage",
    "SystemLog",
]
