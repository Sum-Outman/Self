#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
碰撞检测API路由
提供3D碰撞检测功能，支持形状注册、碰撞检查和射线检测
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Body

from backend.dependencies.auth import get_current_user
from backend.db_models.user import User
from backend.services.collision_detection_service import (
    get_collision_detection_service,
    CollisionDetectionService,
    CollisionShape,
    CollisionShapeType,
)

router = APIRouter(prefix="/api/collision", tags=["碰撞检测"])
logger = logging.getLogger(__name__)


def get_collision_service() -> CollisionDetectionService:
    """获取碰撞检测服务"""
    return get_collision_detection_service()


@router.post("/register", response_model=Dict[str, Any])
async def register_collision_shape(
    shape_type: CollisionShapeType = Body(..., description="形状类型"),
    position: List[float] = Body([0.0, 0.0, 0.0], description="位置 [x, y, z]"),
    orientation: List[float] = Body(
        [0.0, 0.0, 0.0, 1.0], description="方向四元数 [x, y, z, w]"
    ),
    dimensions: Optional[List[float]] = Body(
        None, description="尺寸（对于BOX）[长, 宽, 高]"
    ),
    radius: float = Body(0.5, description="半径（对于SPHERE和CYLINDER）"),
    height: float = Body(1.0, description="高度（对于CYLINDER）"),
    name: str = Body("", description="形状名称"),
    object_id: Optional[str] = Body(None, description="对象ID（可选）"),
    service: CollisionDetectionService = Depends(get_collision_service),
    user: User = Depends(get_current_user),
):
    """注册碰撞形状"""
    try:
        # 根据形状类型设置默认尺寸
        if shape_type == CollisionShapeType.BOX:
            if dimensions is None:
                dimensions = [1.0, 1.0, 1.0]
        else:
            dimensions = dimensions or [1.0, 1.0, 1.0]

        # 创建碰撞形状
        shape = CollisionShape(
            shape_type=shape_type,
            position=position,
            orientation=orientation,
            dimensions=dimensions,
            radius=radius,
            height=height,
            name=name,
            object_id=object_id,
        )

        # 注册形状
        success = service.register_shape(shape)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="形状注册失败"
            )

        return {
            "success": True,
            "message": "形状注册成功",
            "object_id": shape.object_id,
            "shape_info": shape.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注册碰撞形状失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"注册碰撞形状失败: {str(e)}",
        )


@router.post("/update", response_model=Dict[str, Any])
async def update_collision_shape(
    object_id: str = Body(..., description="对象ID"),
    position: List[float] = Body(..., description="位置 [x, y, z]"),
    orientation: Optional[List[float]] = Body(
        None, description="方向四元数 [x, y, z, w]"
    ),
    service: CollisionDetectionService = Depends(get_collision_service),
    user: User = Depends(get_current_user),
):
    """更新碰撞形状位置"""
    try:
        success = service.update_shape_position(object_id, position, orientation)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"对象 {object_id} 不存在"
            )

        return {
            "success": True,
            "message": "形状位置更新成功",
            "object_id": object_id,
            "position": position,
            "orientation": orientation,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新形状位置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新形状位置失败: {str(e)}",
        )


@router.post("/check", response_model=Dict[str, Any])
async def check_collision(
    object_id1: str = Body(..., description="对象1 ID"),
    object_id2: str = Body(..., description="对象2 ID"),
    service: CollisionDetectionService = Depends(get_collision_service),
    user: User = Depends(get_current_user),
):
    """检查两个形状之间的碰撞"""
    try:
        collision, depth, normal = service.check_collision(object_id1, object_id2)

        return {
            "success": True,
            "collision": collision,
            "depth": depth,
            "normal": normal,
            "object_id1": object_id1,
            "object_id2": object_id2,
        }

    except Exception as e:
        logger.error(f"检查碰撞失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检查碰撞失败: {str(e)}",
        )


@router.get("/check-all", response_model=Dict[str, Any])
async def check_all_collisions(
    service: CollisionDetectionService = Depends(get_collision_service),
    user: User = Depends(get_current_user),
):
    """检查所有注册形状之间的碰撞"""
    try:
        collisions = service.check_all_collisions()

        return {
            "success": True,
            "collisions": collisions,
            "collision_count": len(collisions),
        }

    except Exception as e:
        logger.error(f"检查所有碰撞失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检查所有碰撞失败: {str(e)}",
        )


@router.post("/raycast", response_model=Dict[str, Any])
async def ray_cast(
    origin: List[float] = Body([0.0, 0.0, 0.0], description="射线起点 [x, y, z]"),
    direction: List[float] = Body([0.0, 0.0, 1.0], description="射线方向 [x, y, z]"),
    max_distance: float = Body(100.0, description="最大检测距离"),
    service: CollisionDetectionService = Depends(get_collision_service),
    user: User = Depends(get_current_user),
):
    """执行射线检测"""
    try:
        hits = service.ray_cast(origin, direction, max_distance)

        return {
            "success": True,
            "hits": hits,
            "hit_count": len(hits),
            "origin": origin,
            "direction": direction,
            "max_distance": max_distance,
        }

    except Exception as e:
        logger.error(f"射线检测失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"射线检测失败: {str(e)}",
        )


@router.delete("/{object_id}", response_model=Dict[str, Any])
async def unregister_collision_shape(
    object_id: str,
    service: CollisionDetectionService = Depends(get_collision_service),
    user: User = Depends(get_current_user),
):
    """注销碰撞形状"""
    try:
        success = service.unregister_shape(object_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"对象 {object_id} 不存在"
            )

        return {"success": True, "message": "形状注销成功", "object_id": object_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注销形状失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"注销形状失败: {str(e)}",
        )


@router.get("/shapes", response_model=Dict[str, Any])
async def get_all_shapes(
    service: CollisionDetectionService = Depends(get_collision_service),
    user: User = Depends(get_current_user),
):
    """获取所有注册的碰撞形状"""
    try:
        # 注意：service.shapes是字典，需要转换为列表
        shapes_dict = service.shapes
        shapes_list = []

        for object_id, shape in shapes_dict.items():
            shape_info = shape.to_dict()
            shape_info["object_id"] = object_id
            shapes_list.append(shape_info)

        return {"success": True, "shapes": shapes_list, "shape_count": len(shapes_list)}

    except Exception as e:
        logger.error(f"获取形状列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取形状列表失败: {str(e)}",
        )
