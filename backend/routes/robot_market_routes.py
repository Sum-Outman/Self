#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人市场API路由
提供机器人配置共享、下载、评分、评论等功能
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Body,
    Query,
)
from sqlalchemy.orm import Session

from backend.dependencies import get_db
from backend.dependencies.auth import get_current_user, get_current_admin
from backend.db_models.user import User
from backend.db_models.robot_market import RobotMarketStatus, RobotMarketCategory
from backend.services.robot_market_service import RobotMarketService

router = APIRouter(prefix="/api/market", tags=["机器人市场"])
logger = logging.getLogger(__name__)


def get_market_service(db: Session = Depends(get_db)) -> RobotMarketService:
    """获取机器人市场服务"""
    return RobotMarketService(db)


@router.post("/listings/create", response_model=Dict[str, Any])
async def create_market_listing(
    robot_id: int = Body(..., description="机器人ID"),
    title: str = Body(..., description="列表标题"),
    description: str = Body(..., description="详细描述"),
    category: RobotMarketCategory = Body(
        RobotMarketCategory.CUSTOM, description="分类"
    ),
    tags: Optional[List[str]] = Body(None, description="标签列表"),
    version: str = Body("1.0.0", description="版本号"),
    changelog: str = Body("", description="更新日志"),
    license_type: str = Body("MIT", description="许可证类型"),
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """创建机器人市场列表"""
    try:
        success, message, listing = service.create_listing(
            robot_id=robot_id,
            title=title,
            description=description,
            owner_id=user.id,
            category=category,
            tags=tags,
            version=version,
            changelog=changelog,
            license_type=license_type,
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        return {
            "success": True,
            "message": message,
            "listing_id": listing.id,
            "listing_title": listing.title,
            "status": listing.status.value,
            "created_at": listing.created_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建市场列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建市场列表失败: {str(e)}",
        )


@router.put("/listings/{listing_id}", response_model=Dict[str, Any])
async def update_market_listing(
    listing_id: int,
    title: Optional[str] = Body(None, description="新标题"),
    description: Optional[str] = Body(None, description="新描述"),
    category: Optional[RobotMarketCategory] = Body(None, description="新分类"),
    tags: Optional[List[str]] = Body(None, description="新标签"),
    version: Optional[str] = Body(None, description="新版本号"),
    changelog: Optional[str] = Body(None, description="新更新日志"),
    license_type: Optional[str] = Body(None, description="新许可证类型"),
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """更新机器人市场列表"""
    try:
        success, message, listing = service.update_listing(
            listing_id=listing_id,
            owner_id=user.id,
            title=title,
            description=description,
            category=category,
            tags=tags,
            version=version,
            changelog=changelog,
            license_type=license_type,
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        return {
            "success": True,
            "message": message,
            "listing_id": listing.id,
            "listing_title": listing.title,
            "status": listing.status.value,
            "updated_at": listing.updated_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新市场列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新市场列表失败: {str(e)}",
        )


@router.post("/listings/{listing_id}/review", response_model=Dict[str, Any])
async def review_market_listing(
    listing_id: int,
    status: RobotMarketStatus = Body(..., description="审核状态"),
    notes: str = Body("", description="审核备注"),
    service: RobotMarketService = Depends(get_market_service),
    admin_user: User = Depends(get_current_admin),
):
    """审核机器人市场列表（管理员功能）"""
    try:
        success, message, listing = service.review_listing(
            listing_id=listing_id, reviewer_id=admin_user.id, status=status, notes=notes
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        return {
            "success": True,
            "message": message,
            "listing_id": listing.id,
            "listing_title": listing.title,
            "status": listing.status.value,
            "reviewed_at": (
                listing.reviewed_at.isoformat() if listing.reviewed_at else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"审核市场列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"审核市场列表失败: {str(e)}",
        )


@router.get("/listings/search", response_model=Dict[str, Any])
async def search_market_listings(
    query: str = Query("", description="搜索关键词"),
    category: Optional[RobotMarketCategory] = Query(None, description="分类过滤"),
    tags: Optional[List[str]] = Query(None, description="标签过滤"),
    min_rating: float = Query(0.0, description="最小评分"),
    sort_by: str = Query(
        "relevance", description="排序字段 (relevance, rating, downloads, views, date)"
    ),
    sort_order: str = Query("desc", description="排序方向 (asc, desc)"),
    page: int = Query(1, description="页码"),
    page_size: int = Query(20, description="每页大小"),
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """搜索机器人市场列表"""
    try:
        listings, total_count = service.search_listings(
            query=query,
            category=category,
            tags=tags,
            min_rating=min_rating,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
        )

        listings_data = []
        for listing in listings:
            listing_dict = listing.to_dict()
            listings_data.append(listing_dict)

        return {
            "success": True,
            "listings": listings_data,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size,
        }

    except Exception as e:
        logger.error(f"搜索市场列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"搜索市场列表失败: {str(e)}",
        )


@router.get("/listings/{listing_id}", response_model=Dict[str, Any])
async def get_market_listing(
    listing_id: int,
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """获取机器人市场列表详情"""
    try:
        listing = service.get_listing(listing_id)

        if not listing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="列表不存在或未审核通过"
            )

        return {"success": True, "listing": listing.to_dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取市场列表详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取市场列表详情失败: {str(e)}",
        )


@router.post("/listings/{listing_id}/rate", response_model=Dict[str, Any])
async def rate_market_listing(
    listing_id: int,
    rating: int = Body(..., description="评分 (1-5)"),
    comment: str = Body("", description="评分评论"),
    ease_of_use: Optional[int] = Body(None, description="易用性评分 (1-5)"),
    documentation_quality: Optional[int] = Body(None, description="文档质量评分 (1-5)"),
    performance: Optional[int] = Body(None, description="性能评分 (1-5)"),
    reliability: Optional[int] = Body(None, description="可靠性评分 (1-5)"),
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """为机器人市场列表评分"""
    try:
        success, message = service.add_rating(
            listing_id=listing_id,
            user_id=user.id,
            rating=rating,
            comment=comment,
            ease_of_use=ease_of_use,
            documentation_quality=documentation_quality,
            performance=performance,
            reliability=reliability,
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        return {
            "success": True,
            "message": message,
            "listing_id": listing_id,
            "rating": rating,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"评分失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"评分失败: {str(e)}",
        )


@router.post("/listings/{listing_id}/comment", response_model=Dict[str, Any])
async def comment_market_listing(
    listing_id: int,
    content: str = Body(..., description="评论内容"),
    parent_comment_id: Optional[int] = Body(None, description="父评论ID"),
    is_question: bool = Body(False, description="是否为问题"),
    is_answer: bool = Body(False, description="是否为答案"),
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """评论机器人市场列表"""
    try:
        success, message, comment = service.add_comment(
            listing_id=listing_id,
            user_id=user.id,
            content=content,
            parent_comment_id=parent_comment_id,
            is_question=is_question,
            is_answer=is_answer,
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        return {
            "success": True,
            "message": message,
            "comment_id": comment.id if comment else None,
            "listing_id": listing_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"评论失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"评论失败: {str(e)}",
        )


@router.post("/listings/{listing_id}/download", response_model=Dict[str, Any])
async def download_market_listing(
    listing_id: int,
    download_type: str = Body("config", description="下载类型"),
    user_agent: str = Body("", description="用户代理"),
    ip_address: str = Body("", description="IP地址"),
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """下载机器人市场列表配置"""
    try:
        # 首先获取列表信息
        listing = service.get_listing(listing_id, increment_view=False)

        if not listing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="列表不存在或未审核通过"
            )

        # 记录下载
        success, message = service.record_download(
            listing_id=listing_id,
            user_id=user.id,
            download_type=download_type,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        # 返回下载信息
        # 注意：实际的文件下载需要另外实现
        return {
            "success": True,
            "message": "下载记录成功",
            "listing_id": listing_id,
            "listing_title": listing.title,
            "download_type": download_type,
            "file_info": {
                "config_file": listing.config_file_path,
                "urdf_file": listing.urdf_file_path,
                "documentation": listing.documentation_path,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载记录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"下载记录失败: {str(e)}",
        )


@router.post("/listings/{listing_id}/favorite", response_model=Dict[str, Any])
async def toggle_favorite_listing(
    listing_id: int,
    folder: str = Body("default", description="收藏夹"),
    notes: str = Body("", description="备注"),
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """切换机器人市场列表收藏状态"""
    try:
        success, message, is_favorited = service.toggle_favorite(
            listing_id=listing_id, user_id=user.id, folder=folder, notes=notes
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        return {
            "success": True,
            "message": message,
            "listing_id": listing_id,
            "is_favorited": is_favorited,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"切换收藏状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"切换收藏状态失败: {str(e)}",
        )


@router.get("/favorites", response_model=Dict[str, Any])
async def get_user_favorites(
    folder: Optional[str] = Query(None, description="收藏夹过滤"),
    page: int = Query(1, description="页码"),
    page_size: int = Query(20, description="每页大小"),
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """获取用户收藏的机器人市场列表"""
    try:
        listings, total_count = service.get_user_favorites(
            user_id=user.id, folder=folder, page=page, page_size=page_size
        )

        listings_data = []
        for listing in listings:
            listings_data.append(listing.to_dict())

        return {
            "success": True,
            "favorites": listings_data,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size,
        }

    except Exception as e:
        logger.error(f"获取用户收藏失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取用户收藏失败: {str(e)}",
        )


@router.get("/popular", response_model=Dict[str, Any])
async def get_popular_listings(
    category: Optional[RobotMarketCategory] = Query(None, description="分类过滤"),
    limit: int = Query(10, description="返回数量"),
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """获取热门机器人市场列表"""
    try:
        listings = service.get_popular_listings(limit=limit, category=category)

        listings_data = []
        for listing in listings:
            listings_data.append(listing.to_dict())

        return {
            "success": True,
            "popular_listings": listings_data,
            "total_count": len(listings_data),
        }

    except Exception as e:
        logger.error(f"获取热门列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取热门列表失败: {str(e)}",
        )


@router.get("/categories", response_model=Dict[str, Any])
async def get_market_categories():
    """获取机器人市场分类列表"""
    try:
        categories = [
            {
                "id": RobotMarketCategory.HUMANOID.value,
                "name": "人形机器人",
                "description": "人形机器人配置，包括NAO、Pepper、Atlas等",
                "icon": "🤖",
                "listing_count": 0,  # 需要从数据库统计
            },
            {
                "id": RobotMarketCategory.MOBILE.value,
                "name": "移动机器人",
                "description": "移动机器人配置，包括轮式、履带式等",
                "icon": "🚗",
                "listing_count": 0,
            },
            {
                "id": RobotMarketCategory.MANIPULATOR.value,
                "name": "机械臂",
                "description": "机械臂配置，包括工业机械臂、协作机器人等",
                "icon": "🦾",
                "listing_count": 0,
            },
            {
                "id": RobotMarketCategory.AERIAL.value,
                "name": "空中机器人",
                "description": "无人机和空中机器人配置",
                "icon": "🚁",
                "listing_count": 0,
            },
            {
                "id": RobotMarketCategory.UNDERWATER.value,
                "name": "水下机器人",
                "description": "水下机器人和ROV配置",
                "icon": "🌊",
                "listing_count": 0,
            },
            {
                "id": RobotMarketCategory.EDUCATIONAL.value,
                "name": "教育机器人",
                "description": "教育用途的机器人配置",
                "icon": "🎓",
                "listing_count": 0,
            },
            {
                "id": RobotMarketCategory.INDUSTRIAL.value,
                "name": "工业机器人",
                "description": "工业用途的机器人配置",
                "icon": "🏭",
                "listing_count": 0,
            },
            {
                "id": RobotMarketCategory.MEDICAL.value,
                "name": "医疗机器人",
                "description": "医疗和手术机器人配置",
                "icon": "🏥",
                "listing_count": 0,
            },
            {
                "id": RobotMarketCategory.SERVICE.value,
                "name": "服务机器人",
                "description": "服务型机器人配置",
                "icon": "🤝",
                "listing_count": 0,
            },
            {
                "id": RobotMarketCategory.RESEARCH.value,
                "name": "研究机器人",
                "description": "研究用途的机器人配置",
                "icon": "🔬",
                "listing_count": 0,
            },
            {
                "id": RobotMarketCategory.CUSTOM.value,
                "name": "自定义机器人",
                "description": "自定义和混合型机器人配置",
                "icon": "🔧",
                "listing_count": 0,
            },
        ]

        return {
            "success": True,
            "categories": categories,
            "total_count": len(categories),
        }

    except Exception as e:
        logger.error(f"获取分类列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取分类列表失败: {str(e)}",
        )


@router.get("/listings/{listing_id}/statistics", response_model=Dict[str, Any])
async def get_listing_statistics(
    listing_id: int,
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """获取机器人市场列表统计信息"""
    try:
        statistics = service.get_listing_statistics(listing_id)

        if not statistics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="列表不存在"
            )

        return {"success": True, "statistics": statistics, "listing_id": listing_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取列表统计信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取列表统计信息失败: {str(e)}",
        )


@router.get("/user/listings", response_model=Dict[str, Any])
async def get_user_listings(
    status: Optional[RobotMarketStatus] = Query(None, description="状态过滤"),
    page: int = Query(1, description="页码"),
    page_size: int = Query(20, description="每页大小"),
    service: RobotMarketService = Depends(get_market_service),
    user: User = Depends(get_current_user),
):
    """获取用户自己的机器人市场列表"""
    try:
        # 注意：这个功能需要在服务中添加
        # 完整实现，直接查询数据库
        pass

        db = service.db
        db_query = db.query(RobotMarketListing).filter(
            RobotMarketListing.owner_id == user.id
        )

        if status:
            db_query = db_query.filter(RobotMarketListing.status == status)

        total_count = db_query.count()

        # 按更新时间排序
        db_query = db_query.order_by(RobotMarketListing.updated_at.desc())

        offset = (page - 1) * page_size
        listings = db_query.offset(offset).limit(page_size).all()

        listings_data = []
        for listing in listings:
            listings_data.append(listing.to_dict())

        return {
            "success": True,
            "listings": listings_data,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size,
        }

    except Exception as e:
        logger.error(f"获取用户列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取用户列表失败: {str(e)}",
        )
