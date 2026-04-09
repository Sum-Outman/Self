#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人市场数据库模型
包含机器人配置共享、评分、评论等功能
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
    JSON,
    Enum as SQLEnum,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum as PyEnum

from ..core.database import Base


class RobotMarketStatus(PyEnum):
    """机器人市场状态枚举"""

    PENDING = "pending"  # 待审核
    APPROVED = "approved"  # 已审核通过
    REJECTED = "rejected"  # 审核拒绝
    ARCHIVED = "archived"  # 已归档
    DELETED = "deleted"  # 已删除


class RobotMarketCategory(PyEnum):
    """机器人市场分类枚举"""

    HUMANOID = "humanoid"  # 人形机器人
    MOBILE = "mobile"  # 移动机器人
    MANIPULATOR = "manipulator"  # 机械臂
    AERIAL = "aerial"  # 空中机器人
    UNDERWATER = "underwater"  # 水下机器人
    EDUCATIONAL = "educational"  # 教育机器人
    INDUSTRIAL = "industrial"  # 工业机器人
    MEDICAL = "medical"  # 医疗机器人
    SERVICE = "service"  # 服务机器人
    RESEARCH = "research"  # 研究机器人
    CUSTOM = "custom"  # 自定义机器人


class RobotMarketListing(Base):
    """机器人市场列表"""

    __tablename__ = "robot_market_listings"

    id = Column(Integer, primary_key=True, index=True)

    # 机器人信息
    robot_id = Column(
        Integer, ForeignKey("robots.id"), nullable=False, comment="机器人ID"
    )
    title = Column(String(255), nullable=False, comment="列表标题")
    description = Column(Text, comment="详细描述")
    category = Column(
        SQLEnum(RobotMarketCategory),
        nullable=False,
        default=RobotMarketCategory.CUSTOM,
        comment="分类",
    )
    tags = Column(JSON, default=list, comment="标签列表")

    # 版本信息
    version = Column(String(50), default="1.0.0", comment="版本号")
    changelog = Column(Text, comment="更新日志")
    compatible_versions = Column(JSON, default=list, comment="兼容的软件版本")

    # 文件信息
    config_file_path = Column(String(500), comment="配置文件路径")
    urdf_file_path = Column(String(500), comment="URDF文件路径")
    preview_image_path = Column(String(500), comment="预览图路径")
    documentation_path = Column(String(500), comment="文档路径")
    license_type = Column(String(100), default="MIT", comment="许可证类型")
    license_text = Column(Text, comment="许可证文本")

    # 统计信息
    download_count = Column(Integer, default=0, comment="下载次数")
    view_count = Column(Integer, default=0, comment="查看次数")
    rating_count = Column(Integer, default=0, comment="评分次数")
    average_rating = Column(Float, default=0.0, comment="平均评分")

    # 状态信息
    status = Column(
        SQLEnum(RobotMarketStatus),
        nullable=False,
        default=RobotMarketStatus.PENDING,
        comment="审核状态",
    )
    is_featured = Column(Boolean, default=False, comment="是否推荐")
    is_verified = Column(Boolean, default=False, comment="是否已验证")

    # 权限和归属
    owner_id = Column(
        Integer, ForeignKey("users.id"), nullable=False, comment="所有者ID"
    )
    reviewer_id = Column(Integer, ForeignKey("users.id"), comment="审核员ID")
    review_notes = Column(Text, comment="审核备注")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    reviewed_at = Column(DateTime, comment="审核时间")
    published_at = Column(DateTime, comment="发布时间")

    # 关系
    robot = relationship("Robot", back_populates="market_listings")
    owner = relationship(
        "User", foreign_keys=[owner_id], back_populates="market_listings_owned"
    )
    reviewer = relationship(
        "User", foreign_keys=[reviewer_id], back_populates="market_listings_reviewed"
    )
    ratings = relationship(
        "RobotMarketRating", back_populates="listing", cascade="all, delete-orphan"
    )
    comments = relationship(
        "RobotMarketComment", back_populates="listing", cascade="all, delete-orphan"
    )
    downloads = relationship(
        "RobotMarketDownload", back_populates="listing", cascade="all, delete-orphan"
    )

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "robot_id": self.robot_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value if self.category else None,
            "tags": self.tags,
            "version": self.version,
            "changelog": self.changelog,
            "compatible_versions": self.compatible_versions,
            "config_file_path": self.config_file_path,
            "urdf_file_path": self.urdf_file_path,
            "preview_image_path": self.preview_image_path,
            "documentation_path": self.documentation_path,
            "license_type": self.license_type,
            "license_text": self.license_text,
            "download_count": self.download_count,
            "view_count": self.view_count,
            "rating_count": self.rating_count,
            "average_rating": self.average_rating,
            "status": self.status.value if self.status else None,
            "is_featured": self.is_featured,
            "is_verified": self.is_verified,
            "owner_id": self.owner_id,
            "reviewer_id": self.reviewer_id,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "published_at": (
                self.published_at.isoformat() if self.published_at else None
            ),
            "robot_info": self.robot.to_dict() if self.robot else None,
        }


class RobotMarketRating(Base):
    """机器人市场评分"""

    __tablename__ = "robot_market_ratings"

    # 唯一约束：每个用户对每个列表只能评分一次
    __table_args__ = (
        UniqueConstraint("listing_id", "user_id", name="unique_rating_per_user"),
    )

    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(
        Integer,
        ForeignKey("robot_market_listings.id"),
        nullable=False,
        comment="列表ID",
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="用户ID")

    # 评分信息
    rating = Column(Integer, nullable=False, comment="评分 (1-5)")
    comment = Column(Text, comment="评分评论")

    # 评分维度（可选）
    ease_of_use = Column(Integer, comment="易用性评分 (1-5)")
    documentation_quality = Column(Integer, comment="文档质量评分 (1-5)")
    performance = Column(Integer, comment="性能评分 (1-5)")
    reliability = Column(Integer, comment="可靠性评分 (1-5)")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # 关系
    listing = relationship("RobotMarketListing", back_populates="ratings")
    user = relationship("User", back_populates="market_ratings")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "listing_id": self.listing_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "comment": self.comment,
            "ease_of_use": self.ease_of_use,
            "documentation_quality": self.documentation_quality,
            "performance": self.performance,
            "reliability": self.reliability,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class RobotMarketComment(Base):
    """机器人市场评论"""

    __tablename__ = "robot_market_comments"

    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(
        Integer,
        ForeignKey("robot_market_listings.id"),
        nullable=False,
        comment="列表ID",
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="用户ID")
    parent_comment_id = Column(
        Integer, ForeignKey("robot_market_comments.id"), comment="父评论ID"
    )

    # 评论内容
    content = Column(Text, nullable=False, comment="评论内容")
    is_question = Column(Boolean, default=False, comment="是否为问题")
    is_answer = Column(Boolean, default=False, comment="是否为答案")
    is_helpful = Column(Boolean, default=False, comment="是否有帮助")

    # 状态
    is_approved = Column(Boolean, default=True, comment="是否已审核通过")
    is_edited = Column(Boolean, default=False, comment="是否已编辑")

    # 统计
    helpful_count = Column(Integer, default=0, comment="有帮助计数")
    reply_count = Column(Integer, default=0, comment="回复计数")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    edited_at = Column(DateTime, comment="编辑时间")

    # 关系
    listing = relationship("RobotMarketListing", back_populates="comments")
    user = relationship("User", back_populates="market_comments")
    parent = relationship(
        "RobotMarketComment", remote_side=[id], back_populates="replies"
    )
    replies = relationship(
        "RobotMarketComment", back_populates="parent", cascade="all, delete-orphan"
    )

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "listing_id": self.listing_id,
            "user_id": self.user_id,
            "parent_comment_id": self.parent_comment_id,
            "content": self.content,
            "is_question": self.is_question,
            "is_answer": self.is_answer,
            "is_helpful": self.is_helpful,
            "is_approved": self.is_approved,
            "is_edited": self.is_edited,
            "helpful_count": self.helpful_count,
            "reply_count": self.reply_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "edited_at": self.edited_at.isoformat() if self.edited_at else None,
            "user_info": self.user.username if self.user else None,
        }


class RobotMarketDownload(Base):
    """机器人市场下载记录"""

    __tablename__ = "robot_market_downloads"

    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(
        Integer,
        ForeignKey("robot_market_listings.id"),
        nullable=False,
        comment="列表ID",
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="用户ID")

    # 下载信息
    download_type = Column(
        String(50), default="config", comment="下载类型: config, urdf, full"
    )
    file_size = Column(Integer, comment="文件大小 (字节)")
    download_success = Column(Boolean, default=True, comment="下载是否成功")

    # 客户端信息
    user_agent = Column(String(500), comment="用户代理")
    ip_address = Column(String(45), comment="IP地址")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 索引
    __table_args__ = (
        Index("idx_downloads_listing_user", "listing_id", "user_id"),
        Index("idx_downloads_created_at", "created_at"),
    )

    # 关系
    listing = relationship("RobotMarketListing", back_populates="downloads")
    user = relationship("User", back_populates="market_downloads")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "listing_id": self.listing_id,
            "user_id": self.user_id,
            "download_type": self.download_type,
            "file_size": self.file_size,
            "download_success": self.download_success,
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
            "created_at": self.created_at.isoformat(),
        }


class RobotMarketFavorite(Base):
    """机器人市场收藏"""

    __tablename__ = "robot_market_favorites"

    # 唯一约束：每个用户对每个列表只能收藏一次
    __table_args__ = (
        UniqueConstraint("listing_id", "user_id", name="unique_favorite_per_user"),
    )

    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(
        Integer,
        ForeignKey("robot_market_listings.id"),
        nullable=False,
        comment="列表ID",
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="用户ID")

    # 收藏信息
    folder = Column(String(100), default="default", comment="收藏夹")
    notes = Column(Text, comment="收藏备注")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # 关系
    listing = relationship("RobotMarketListing")
    user = relationship("User", back_populates="market_favorites")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "listing_id": self.listing_id,
            "user_id": self.user_id,
            "folder": self.folder,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# 需要在User模型中添加关系
# 在User类中添加以下属性：
# market_listings_owned = relationship("RobotMarketListing", foreign_keys=[RobotMarketListing.owner_id], back_populates="owner")
# market_listings_reviewed = relationship("RobotMarketListing", foreign_keys=[RobotMarketListing.reviewer_id], back_populates="reviewer")
# market_ratings = relationship("RobotMarketRating", back_populates="user", cascade="all, delete-orphan")
# market_comments = relationship("RobotMarketComment", back_populates="user", cascade="all, delete-orphan")
# market_downloads = relationship("RobotMarketDownload", back_populates="user", cascade="all, delete-orphan")
# market_favorites = relationship("RobotMarketFavorite", back_populates="user", cascade="all, delete-orphan")

# 需要在Robot模型中添加关系
# 在Robot类中添加以下属性：
# market_listings = relationship("RobotMarketListing", back_populates="robot", cascade="all, delete-orphan")
