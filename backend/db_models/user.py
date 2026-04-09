"""
用户相关数据库模型
包含用户、API密钥、会话等模型
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
from ..core.config import Config


class User(Base):
    """用户模型"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_method = Column(String(20), default="email")  # email or totp
    totp_secret = Column(String(32), nullable=True)
    backup_codes = Column(Text, nullable=True)  # JSON数组存储备份代码
    role = Column(String(20), default="user")  # 用户角色: admin, user, manager, viewer

    # 关系
    api_keys = relationship("APIKey", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")
    memories = relationship("Memory", back_populates="user")
    robots = relationship("Robot", back_populates="user")
    demonstrations = relationship(
        "Demonstration", back_populates="user", cascade="all, delete-orphan"
    )
    demonstration_tasks = relationship(
        "DemonstrationTask", back_populates="user", cascade="all, delete-orphan"
    )

    # 机器人市场关系
    market_listings_owned = relationship(
        "RobotMarketListing",
        foreign_keys="[RobotMarketListing.owner_id]",
        back_populates="owner",
        cascade="all, delete-orphan",
    )
    market_listings_reviewed = relationship(
        "RobotMarketListing",
        foreign_keys="[RobotMarketListing.reviewer_id]",
        back_populates="reviewer",
        cascade="all, delete-orphan",
    )
    market_ratings = relationship(
        "RobotMarketRating", back_populates="user", cascade="all, delete-orphan"
    )
    market_comments = relationship(
        "RobotMarketComment", back_populates="user", cascade="all, delete-orphan"
    )
    market_downloads = relationship(
        "RobotMarketDownload", back_populates="user", cascade="all, delete-orphan"
    )
    market_favorites = relationship(
        "RobotMarketFavorite", back_populates="user", cascade="all, delete-orphan"
    )
    chat_sessions = relationship(
        "ChatSession", back_populates="user", cascade="all, delete-orphan"
    )


class APIKey(Base):
    """API密钥模型"""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, index=True, nullable=False)
    prefix = Column(
        String(10), nullable=False, default=Config.API_KEY_PREFIX
    )  # 密钥前缀，使用可配置值
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(50))
    is_active = Column(Boolean, default=True)
    rate_limit = Column(Integer, default=100)  # 每分钟请求数
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)

    # 关系
    user = relationship("User", back_populates="api_keys")


class UserSession(Base):
    """会话模型"""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(200), unique=True, nullable=False)
    device_info = Column(Text)
    ip_address = Column(String(50))
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    user = relationship("User", back_populates="sessions")


class PasswordResetToken(Base):
    """密码重置令牌模型"""

    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(100), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    user = relationship("User")


class EmailVerificationToken(Base):
    """邮箱验证令牌模型"""

    __tablename__ = "email_verification_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(100), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    verified_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    user = relationship("User")


class LoginAttempt(Base):
    """登录尝试模型，记录失败登录尝试"""

    __tablename__ = "login_attempts"

    id = Column(Integer, primary_key=True, index=True)
    username_or_email = Column(String(100), nullable=False, index=True)
    attempt_count = Column(Integer, default=1, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    last_attempt_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TwoFactorTempSession(Base):
    """双因素认证临时会话模型"""

    __tablename__ = "twofa_temp_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    username_or_email = Column(String(100), nullable=False)
    temp_token = Column(String(100), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    user = relationship("User")


class EmailTwoFactorCode(Base):
    """邮箱双因素认证验证码模型"""

    __tablename__ = "email_2fa_codes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    code = Column(String(10), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    user = relationship("User")


class Payment(Base):
    """支付模型"""

    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount = Column(Float, nullable=False, default=0.0)
    currency = Column(String(10), default="CNY")
    status = Column(
        String(20), default="pending"
    )  # pending, completed, failed, refunded
    payment_method = Column(String(20))  # wechat, alipay, bank, credit_card
    transaction_id = Column(String(100), unique=True, index=True, nullable=True)
    description = Column(Text, nullable=True)
    payment_metadata = Column(Text, nullable=True)  # JSON格式的额外数据
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # 关系
    user = relationship("User")


__all__ = [
    "User",
    "APIKey",
    "Payment",
    "UserSession",
    "PasswordResetToken",
    "EmailVerificationToken",
    "LoginAttempt",
    "TwoFactorTempSession",
    "EmailTwoFactorCode",
]
