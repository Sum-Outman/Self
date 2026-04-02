#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人市场服务
提供机器人配置共享、下载、评分、评论等功能
"""

import logging
import json
import os
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from ..db_models.robot_market import (
    RobotMarketListing,
    RobotMarketRating,
    RobotMarketComment,
    RobotMarketDownload,
    RobotMarketFavorite,
    RobotMarketStatus,
    RobotMarketCategory
)
from ..db_models.robot import Robot
from ..db_models.user import User

logger = logging.getLogger(__name__)


class RobotMarketService:
    """机器人市场服务"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_listing(self,
                      robot_id: int,
                      title: str,
                      description: str,
                      owner_id: int,
                      category: RobotMarketCategory = RobotMarketCategory.CUSTOM,
                      tags: Optional[List[str]] = None,
                      version: str = "1.0.0",
                      changelog: str = "",
                      license_type: str = "MIT") -> Tuple[bool, str, Optional[RobotMarketListing]]:
        """
        创建机器人市场列表
        
        参数:
            robot_id: 机器人ID
            title: 列表标题
            description: 描述
            owner_id: 所有者ID
            category: 分类
            tags: 标签列表
            version: 版本号
            changelog: 更新日志
            license_type: 许可证类型
            
        返回:
            (成功状态, 消息, 列表对象)
        """
        try:
            # 验证机器人是否存在且属于该用户
            robot = self.db.query(Robot).filter(
                Robot.id == robot_id,
                Robot.user_id == owner_id
            ).first()
            
            if not robot:
                return False, "机器人不存在或无权访问", None
            
            # 检查是否已有相同标题的列表
            existing_listing = self.db.query(RobotMarketListing).filter(
                RobotMarketListing.robot_id == robot_id,
                RobotMarketListing.title == title,
                RobotMarketListing.owner_id == owner_id
            ).first()
            
            if existing_listing:
                return False, f"机器人 '{robot.name}' 已有标题为 '{title}' 的列表", None
            
            # 创建列表
            listing = RobotMarketListing(
                robot_id=robot_id,
                title=title,
                description=description,
                owner_id=owner_id,
                category=category,
                tags=tags or [],
                version=version,
                changelog=changelog,
                license_type=license_type,
                status=RobotMarketStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            self.db.add(listing)
            self.db.commit()
            
            logger.info(f"创建机器人市场列表: {title} (机器人ID: {robot_id}, 所有者: {owner_id})")
            return True, "列表创建成功，等待审核", listing
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建列表失败: {e}")
            return False, f"创建列表失败: {str(e)}", None
    
    def update_listing(self,
                      listing_id: int,
                      owner_id: int,
                      title: Optional[str] = None,
                      description: Optional[str] = None,
                      category: Optional[RobotMarketCategory] = None,
                      tags: Optional[List[str]] = None,
                      version: Optional[str] = None,
                      changelog: Optional[str] = None,
                      license_type: Optional[str] = None) -> Tuple[bool, str, Optional[RobotMarketListing]]:
        """
        更新机器人市场列表
        
        参数:
            listing_id: 列表ID
            owner_id: 所有者ID（用于权限验证）
            title: 新标题（可选）
            description: 新描述（可选）
            category: 新分类（可选）
            tags: 新标签（可选）
            version: 新版本号（可选）
            changelog: 新更新日志（可选）
            license_type: 新许可证类型（可选）
            
        返回:
            (成功状态, 消息, 更新后的列表对象)
        """
        try:
            # 获取列表
            listing = self.db.query(RobotMarketListing).filter(
                RobotMarketListing.id == listing_id,
                RobotMarketListing.owner_id == owner_id
            ).first()
            
            if not listing:
                return False, "列表不存在或无权访问", None
            
            # 更新字段
            if title is not None:
                listing.title = title
            if description is not None:
                listing.description = description
            if category is not None:
                listing.category = category
            if tags is not None:
                listing.tags = tags
            if version is not None:
                listing.version = version
            if changelog is not None:
                listing.changelog = changelog
            if license_type is not None:
                listing.license_type = license_type
            
            listing.updated_at = datetime.now(timezone.utc)
            
            # 如果列表已发布，更新后需要重新审核
            if listing.status == RobotMarketStatus.APPROVED:
                listing.status = RobotMarketStatus.PENDING
                listing.reviewer_id = None
                listing.review_notes = None
                listing.reviewed_at = None
            
            self.db.commit()
            
            logger.info(f"更新机器人市场列表: {listing.title} (ID: {listing_id})")
            return True, "列表更新成功", listing
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新列表失败: {e}")
            return False, f"更新列表失败: {str(e)}", None
    
    def review_listing(self,
                      listing_id: int,
                      reviewer_id: int,
                      status: RobotMarketStatus,
                      notes: str = "") -> Tuple[bool, str, Optional[RobotMarketListing]]:
        """
        审核机器人市场列表（管理员功能）
        
        参数:
            listing_id: 列表ID
            reviewer_id: 审核员ID
            status: 审核状态
            notes: 审核备注
            
        返回:
            (成功状态, 消息, 列表对象)
        """
        try:
            # 获取列表
            listing = self.db.query(RobotMarketListing).filter(
                RobotMarketListing.id == listing_id
            ).first()
            
            if not listing:
                return False, "列表不存在", None
            
            # 更新审核信息
            listing.status = status
            listing.reviewer_id = reviewer_id
            listing.review_notes = notes
            listing.reviewed_at = datetime.now(timezone.utc)
            
            # 如果审核通过，设置发布时间
            if status == RobotMarketStatus.APPROVED and not listing.published_at:
                listing.published_at = datetime.now(timezone.utc)
            
            self.db.commit()
            
            status_text = "通过" if status == RobotMarketStatus.APPROVED else "拒绝"
            logger.info(f"审核机器人市场列表: {listing.title} - {status_text} (审核员: {reviewer_id})")
            return True, f"列表审核{status_text}成功", listing
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"审核列表失败: {e}")
            return False, f"审核列表失败: {str(e)}", None
    
    def get_listing(self, listing_id: int, increment_view: bool = True) -> Optional[RobotMarketListing]:
        """
        获取列表详情
        
        参数:
            listing_id: 列表ID
            increment_view: 是否增加查看次数
            
        返回:
            列表对象或None
        """
        try:
            listing = self.db.query(RobotMarketListing).filter(
                RobotMarketListing.id == listing_id,
                RobotMarketListing.status == RobotMarketStatus.APPROVED  # 只返回已审核通过的
            ).first()
            
            if listing and increment_view:
                listing.view_count += 1
                self.db.commit()
            
            return listing
            
        except Exception as e:
            logger.error(f"获取列表失败: {e}")
            return None  # 返回None
    
    def search_listings(self,
                       query: str = "",
                       category: Optional[RobotMarketCategory] = None,
                       tags: Optional[List[str]] = None,
                       min_rating: float = 0.0,
                       sort_by: str = "relevance",
                       sort_order: str = "desc",
                       page: int = 1,
                       page_size: int = 20) -> Tuple[List[RobotMarketListing], int]:
        """
        搜索机器人市场列表
        
        参数:
            query: 搜索关键词
            category: 分类过滤
            tags: 标签过滤
            min_rating: 最小评分
            sort_by: 排序字段 (relevance, rating, downloads, views, date)
            sort_order: 排序方向 (asc, desc)
            page: 页码
            page_size: 每页大小
            
        返回:
            (列表列表, 总数量)
        """
        try:
            # 基础查询：只返回已审核通过的列表
            db_query = self.db.query(RobotMarketListing).filter(
                RobotMarketListing.status == RobotMarketStatus.APPROVED
            )
            
            # 应用文本搜索
            if query:
                db_query = db_query.filter(
                    or_(
                        RobotMarketListing.title.contains(query),
                        RobotMarketListing.description.contains(query),
                        RobotMarketListing.tags.contains([query])
                    )
                )
            
            # 应用分类过滤
            if category:
                db_query = db_query.filter(RobotMarketListing.category == category)
            
            # 应用标签过滤
            if tags and len(tags) > 0:
                for tag in tags:
                    db_query = db_query.filter(RobotMarketListing.tags.contains([tag]))
            
            # 应用评分过滤
            if min_rating > 0:
                db_query = db_query.filter(RobotMarketListing.average_rating >= min_rating)
            
            # 计算总数量
            total_count = db_query.count()
            
            # 应用排序
            if sort_by == "rating":
                order_field = RobotMarketListing.average_rating
            elif sort_by == "downloads":
                order_field = RobotMarketListing.download_count
            elif sort_by == "views":
                order_field = RobotMarketListing.view_count
            elif sort_by == "date":
                order_field = RobotMarketListing.published_at
            else:  # relevance
                order_field = RobotMarketListing.view_count  # 默认按热度排序
            
            if sort_order == "asc":
                db_query = db_query.order_by(order_field.asc())
            else:
                db_query = db_query.order_by(order_field.desc())
            
            # 应用分页
            offset = (page - 1) * page_size
            listings = db_query.offset(offset).limit(page_size).all()
            
            return listings, total_count
            
        except Exception as e:
            logger.error(f"搜索列表失败: {e}")
            return []  # 返回空列表, 0
    
    def add_rating(self,
                  listing_id: int,
                  user_id: int,
                  rating: int,
                  comment: str = "",
                  ease_of_use: Optional[int] = None,
                  documentation_quality: Optional[int] = None,
                  performance: Optional[int] = None,
                  reliability: Optional[int] = None) -> Tuple[bool, str]:
        """
        添加评分
        
        参数:
            listing_id: 列表ID
            user_id: 用户ID
            rating: 评分 (1-5)
            comment: 评论
            ease_of_use: 易用性评分
            documentation_quality: 文档质量评分
            performance: 性能评分
            reliability: 可靠性评分
            
        返回:
            (成功状态, 消息)
        """
        try:
            # 验证评分范围
            if rating < 1 or rating > 5:
                return False, "评分必须在1-5之间"
            
            # 检查列表是否存在且已审核通过
            listing = self.db.query(RobotMarketListing).filter(
                RobotMarketListing.id == listing_id,
                RobotMarketListing.status == RobotMarketStatus.APPROVED
            ).first()
            
            if not listing:
                return False, "列表不存在或未审核通过"
            
            # 检查用户是否已评分
            existing_rating = self.db.query(RobotMarketRating).filter(
                RobotMarketRating.listing_id == listing_id,
                RobotMarketRating.user_id == user_id
            ).first()
            
            if existing_rating:
                # 更新现有评分
                existing_rating.rating = rating
                existing_rating.comment = comment
                existing_rating.ease_of_use = ease_of_use
                existing_rating.documentation_quality = documentation_quality
                existing_rating.performance = performance
                existing_rating.reliability = reliability
                existing_rating.updated_at = datetime.now(timezone.utc)
            else:
                # 创建新评分
                new_rating = RobotMarketRating(
                    listing_id=listing_id,
                    user_id=user_id,
                    rating=rating,
                    comment=comment,
                    ease_of_use=ease_of_use,
                    documentation_quality=documentation_quality,
                    performance=performance,
                    reliability=reliability,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                self.db.add(new_rating)
            
            # 重新计算平均评分
            self._update_listing_rating(listing_id)
            
            self.db.commit()
            logger.info(f"添加评分: 列表ID={listing_id}, 用户ID={user_id}, 评分={rating}")
            return True, "评分提交成功"
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"添加评分失败: {e}")
            return False, f"添加评分失败: {str(e)}"
    
    def _update_listing_rating(self, listing_id: int):
        """更新列表的评分统计"""
        try:
            # 计算平均评分
            result = self.db.query(
                func.count(RobotMarketRating.id).label("count"),
                func.avg(RobotMarketRating.rating).label("average")
            ).filter(
                RobotMarketRating.listing_id == listing_id
            ).first()
            
            if result and result.count > 0:
                listing = self.db.query(RobotMarketListing).filter(
                    RobotMarketListing.id == listing_id
                ).first()
                
                if listing:
                    listing.rating_count = result.count
                    listing.average_rating = float(result.average) if result.average else 0.0
                    
        except Exception as e:
            logger.error(f"更新列表评分失败: {e}")
    
    def add_comment(self,
                   listing_id: int,
                   user_id: int,
                   content: str,
                   parent_comment_id: Optional[int] = None,
                   is_question: bool = False,
                   is_answer: bool = False) -> Tuple[bool, str, Optional[RobotMarketComment]]:
        """
        添加评论
        
        参数:
            listing_id: 列表ID
            user_id: 用户ID
            content: 评论内容
            parent_comment_id: 父评论ID（用于回复）
            is_question: 是否为问题
            is_answer: 是否为答案
            
        返回:
            (成功状态, 消息, 评论对象)
        """
        try:
            # 检查列表是否存在且已审核通过
            listing = self.db.query(RobotMarketListing).filter(
                RobotMarketListing.id == listing_id,
                RobotMarketListing.status == RobotMarketStatus.APPROVED
            ).first()
            
            if not listing:
                return False, "列表不存在或未审核通过", None
            
            # 创建评论
            comment = RobotMarketComment(
                listing_id=listing_id,
                user_id=user_id,
                parent_comment_id=parent_comment_id,
                content=content,
                is_question=is_question,
                is_answer=is_answer,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            self.db.add(comment)
            
            # 如果是回复，更新父评论的回复计数
            if parent_comment_id:
                parent_comment = self.db.query(RobotMarketComment).filter(
                    RobotMarketComment.id == parent_comment_id
                ).first()
                
                if parent_comment:
                    parent_comment.reply_count += 1
                    parent_comment.updated_at = datetime.now(timezone.utc)
            
            self.db.commit()
            
            logger.info(f"添加评论: 列表ID={listing_id}, 用户ID={user_id}, 内容长度={len(content)}")
            return True, "评论提交成功", comment
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"添加评论失败: {e}")
            return False, f"添加评论失败: {str(e)}", None
    
    def record_download(self,
                       listing_id: int,
                       user_id: int,
                       download_type: str = "config",
                       file_size: Optional[int] = None,
                       user_agent: str = "",
                       ip_address: str = "") -> Tuple[bool, str]:
        """
        记录下载
        
        参数:
            listing_id: 列表ID
            user_id: 用户ID
            download_type: 下载类型
            file_size: 文件大小
            user_agent: 用户代理
            ip_address: IP地址
            
        返回:
            (成功状态, 消息)
        """
        try:
            # 检查列表是否存在且已审核通过
            listing = self.db.query(RobotMarketListing).filter(
                RobotMarketListing.id == listing_id,
                RobotMarketListing.status == RobotMarketStatus.APPROVED
            ).first()
            
            if not listing:
                return False, "列表不存在或未审核通过"
            
            # 创建下载记录
            download = RobotMarketDownload(
                listing_id=listing_id,
                user_id=user_id,
                download_type=download_type,
                file_size=file_size,
                user_agent=user_agent,
                ip_address=ip_address,
                created_at=datetime.now(timezone.utc)
            )
            
            self.db.add(download)
            
            # 更新列表的下载计数
            listing.download_count += 1
            listing.updated_at = datetime.now(timezone.utc)
            
            self.db.commit()
            
            logger.info(f"记录下载: 列表ID={listing_id}, 用户ID={user_id}, 类型={download_type}")
            return True, "下载记录成功"
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"记录下载失败: {e}")
            return False, f"记录下载失败: {str(e)}"
    
    def toggle_favorite(self,
                       listing_id: int,
                       user_id: int,
                       folder: str = "default",
                       notes: str = "") -> Tuple[bool, str, bool]:
        """
        切换收藏状态
        
        参数:
            listing_id: 列表ID
            user_id: 用户ID
            folder: 收藏夹
            notes: 备注
            
        返回:
            (成功状态, 消息, 是否已收藏)
        """
        try:
            # 检查列表是否存在且已审核通过
            listing = self.db.query(RobotMarketListing).filter(
                RobotMarketListing.id == listing_id,
                RobotMarketListing.status == RobotMarketStatus.APPROVED
            ).first()
            
            if not listing:
                return False, "列表不存在或未审核通过", False
            
            # 检查是否已收藏
            favorite = self.db.query(RobotMarketFavorite).filter(
                RobotMarketFavorite.listing_id == listing_id,
                RobotMarketFavorite.user_id == user_id
            ).first()
            
            if favorite:
                # 取消收藏
                self.db.delete(favorite)
                is_favorited = False
                message = "已取消收藏"
                logger.info(f"取消收藏: 列表ID={listing_id}, 用户ID={user_id}")
            else:
                # 添加收藏
                new_favorite = RobotMarketFavorite(
                    listing_id=listing_id,
                    user_id=user_id,
                    folder=folder,
                    notes=notes,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                self.db.add(new_favorite)
                is_favorited = True
                message = "已添加到收藏"
                logger.info(f"添加收藏: 列表ID={listing_id}, 用户ID={user_id}, 收藏夹={folder}")
            
            self.db.commit()
            return True, message, is_favorited
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"切换收藏状态失败: {e}")
            return False, f"操作失败: {str(e)}", False
    
    def get_user_favorites(self,
                          user_id: int,
                          folder: Optional[str] = None,
                          page: int = 1,
                          page_size: int = 20) -> Tuple[List[RobotMarketListing], int]:
        """
        获取用户收藏的列表
        
        参数:
            user_id: 用户ID
            folder: 收藏夹（可选）
            page: 页码
            page_size: 每页大小
            
        返回:
            (列表列表, 总数量)
        """
        try:
            # 基础查询
            db_query = self.db.query(RobotMarketListing).join(
                RobotMarketFavorite,
                RobotMarketFavorite.listing_id == RobotMarketListing.id
            ).filter(
                RobotMarketFavorite.user_id == user_id,
                RobotMarketListing.status == RobotMarketStatus.APPROVED
            )
            
            # 应用收藏夹过滤
            if folder:
                db_query = db_query.filter(RobotMarketFavorite.folder == folder)
            
            # 按收藏时间排序
            db_query = db_query.order_by(RobotMarketFavorite.created_at.desc())
            
            # 计算总数量
            total_count = db_query.count()
            
            # 应用分页
            offset = (page - 1) * page_size
            listings = db_query.offset(offset).limit(page_size).all()
            
            return listings, total_count
            
        except Exception as e:
            logger.error(f"获取用户收藏失败: {e}")
            return []  # 返回空列表, 0
    
    def get_popular_listings(self,
                            limit: int = 10,
                            category: Optional[RobotMarketCategory] = None) -> List[RobotMarketListing]:
        """
        获取热门列表
        
        参数:
            limit: 返回数量
            category: 分类过滤
            
        返回:
            列表列表
        """
        try:
            db_query = self.db.query(RobotMarketListing).filter(
                RobotMarketListing.status == RobotMarketStatus.APPROVED,
                RobotMarketListing.is_featured == True
            )
            
            if category:
                db_query = db_query.filter(RobotMarketListing.category == category)
            
            listings = db_query.order_by(
                desc(RobotMarketListing.download_count),
                desc(RobotMarketListing.view_count),
                desc(RobotMarketListing.average_rating)
            ).limit(limit).all()
            
            return listings
            
        except Exception as e:
            logger.error(f"获取热门列表失败: {e}")
            return []  # 返回空列表
    
    def get_listing_statistics(self, listing_id: int) -> Dict[str, Any]:
        """
        获取列表统计信息
        
        参数:
            listing_id: 列表ID
            
        返回:
            统计信息字典
        """
        try:
            listing = self.db.query(RobotMarketListing).filter(
                RobotMarketListing.id == listing_id
            ).first()
            
            if not listing:
                return {}  # 返回空字典
            
            # 获取评分分布
            rating_distribution = self.db.query(
                RobotMarketRating.rating,
                func.count(RobotMarketRating.id).label("count")
            ).filter(
                RobotMarketRating.listing_id == listing_id
            ).group_by(RobotMarketRating.rating).all()
            
            # 转换为字典
            rating_dist_dict = {rating: count for rating, count in rating_distribution}
            
            # 获取下载趋势（最近30天）
            thirty_days_ago = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            
            downloads_by_day = self.db.query(
                func.date(RobotMarketDownload.created_at).label("date"),
                func.count(RobotMarketDownload.id).label("count")
            ).filter(
                RobotMarketDownload.listing_id == listing_id,
                RobotMarketDownload.created_at >= thirty_days_ago
            ).group_by(func.date(RobotMarketDownload.created_at)).all()
            
            downloads_trend = {str(date): count for date, count in downloads_by_day}
            
            return {
                "listing_id": listing_id,
                "title": listing.title,
                "total_views": listing.view_count,
                "total_downloads": listing.download_count,
                "total_ratings": listing.rating_count,
                "average_rating": listing.average_rating,
                "rating_distribution": rating_dist_dict,
                "downloads_trend": downloads_trend,
                "created_at": listing.created_at.isoformat(),
                "published_at": listing.published_at.isoformat() if listing.published_at else None
            }
            
        except Exception as e:
            logger.error(f"获取列表统计信息失败: {e}")
            return {}  # 返回空字典