#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多机器人协同控制API路由
支持多机器人协同任务创建、管理和监控
"""

import logging
import time
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session

from backend.dependencies import get_db
from backend.dependencies.auth import get_current_user
from backend.db_models.user import User
from backend.services.multi_robot_coordinator import (
    get_multi_robot_coordinator,
    MultiRobotCoordinator,
    CoordinationMode,
    RobotRole,
    FormationType,
    TaskStatus
)

router = APIRouter(prefix="/api/coordination", tags=["多机器人协同"])
logger = logging.getLogger(__name__)


def get_coordinator() -> MultiRobotCoordinator:
    """获取多机器人协调器"""
    return get_multi_robot_coordinator()


@router.post("/tasks/create", response_model=Dict[str, Any])
async def create_cooperative_task(
    name: str = Body(..., description="任务名称"),
    description: str = Body(..., description="任务描述"),
    coordination_mode: CoordinationMode = Body(CoordinationMode.CENTRALIZED, description="协同模式"),
    formation_type: Optional[str] = Body(None, description="队形类型"),
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """创建协同任务"""
    try:
        success, message, task = coordinator.create_task(
            name=name,
            description=description,
            created_by=user.id,
            coordination_mode=coordination_mode,
            formation_type=formation_type
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        return {
            "success": True,
            "message": message,
            "task_id": task.task_id,
            "task_name": task.name,
            "created_at": task.start_time if task.start_time else None,
            "coordination_mode": task.coordination_mode.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建协同任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建协同任务失败: {str(e)}"
        )


@router.post("/tasks/{task_id}/add-robot", response_model=Dict[str, Any])
async def add_robot_to_task(
    task_id: str,
    robot_id: int = Body(..., description="机器人ID"),
    robot_name: str = Body(..., description="机器人名称"),
    role: RobotRole = Body(RobotRole.FOLLOWER, description="机器人角色"),
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """添加机器人到协同任务"""
    try:
        success, message = coordinator.add_robot_to_task(
            task_id=task_id,
            robot_id=robot_id,
            robot_name=robot_name,
            role=role
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        return {
            "success": True,
            "message": message,
            "task_id": task_id,
            "robot_id": robot_id,
            "role": role.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加机器人到任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加机器人到任务失败: {str(e)}"
        )


@router.post("/tasks/{task_id}/start", response_model=Dict[str, Any])
async def start_task(
    task_id: str,
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """开始执行协同任务"""
    try:
        success, message = coordinator.start_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        return {
            "success": True,
            "message": message,
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"开始任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"开始任务失败: {str(e)}"
        )


@router.post("/tasks/{task_id}/pause", response_model=Dict[str, Any])
async def pause_task(
    task_id: str,
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """暂停协同任务"""
    try:
        success, message = coordinator.pause_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        return {
            "success": True,
            "message": message,
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"暂停任务失败: {str(e)}"
        )


@router.post("/tasks/{task_id}/resume", response_model=Dict[str, Any])
async def resume_task(
    task_id: str,
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """恢复协同任务"""
    try:
        success, message = coordinator.resume_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        return {
            "success": True,
            "message": message,
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"恢复任务失败: {str(e)}"
        )


@router.post("/tasks/{task_id}/cancel", response_model=Dict[str, Any])
async def cancel_task(
    task_id: str,
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """取消协同任务"""
    try:
        success, message = coordinator.cancel_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        return {
            "success": True,
            "message": message,
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取消任务失败: {str(e)}"
        )


@router.post("/tasks/{task_id}/assign-formation", response_model=Dict[str, Any])
async def assign_formation(
    task_id: str,
    formation_type: str = Body(..., description="队形类型"),
    spacing: float = Body(1.0, description="间距（米）"),
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """分配队形给协同任务"""
    try:
        success, message = coordinator.assign_formation(
            task_id=task_id,
            formation_type=formation_type,
            spacing=spacing
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        return {
            "success": True,
            "message": message,
            "task_id": task_id,
            "formation_type": formation_type,
            "spacing": spacing
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分配队形失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分配队形失败: {str(e)}"
        )


@router.post("/tasks/{task_id}/assign-role", response_model=Dict[str, Any])
async def assign_robot_role(
    task_id: str,
    robot_id: int = Body(..., description="机器人ID"),
    role: RobotRole = Body(..., description="机器人角色"),
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """分配角色给机器人"""
    try:
        success, message = coordinator.assign_robot_role(
            task_id=task_id,
            robot_id=robot_id,
            role=role
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        return {
            "success": True,
            "message": message,
            "task_id": task_id,
            "robot_id": robot_id,
            "role": role.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分配角色失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分配角色失败: {str(e)}"
        )


@router.get("/tasks", response_model=Dict[str, Any])
async def get_all_tasks(
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """获取所有协同任务"""
    try:
        tasks = coordinator.get_all_tasks()
        
        task_list = []
        for task_id, task in tasks.items():
            task_list.append({
                "task_id": task_id,
                "name": task.name,
                "description": task.description,
                "status": task.status.value,
                "progress": task.progress,
                "robot_count": len(task.robot_ids),
                "created_by": task.created_by,
                "start_time": task.start_time,
                "end_time": task.end_time
            })
        
        return {
            "success": True,
            "tasks": task_list,
            "total_count": len(task_list)
        }
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务列表失败: {str(e)}"
        )


@router.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task_details(
    task_id: str,
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """获取协同任务详情"""
    try:
        task = coordinator.get_task_info(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务 {task_id} 不存在"
            )
        
        # 获取机器人状态
        robot_states = []
        for robot_id in task.robot_ids:
            state = coordinator.get_robot_state(robot_id)
            if state:
                robot_states.append({
                    "robot_id": robot_id,
                    "robot_name": state.robot_name,
                    "position": state.position,
                    "orientation": state.orientation,
                    "velocity": state.velocity,
                    "battery_level": state.battery_level,
                    "status": state.status,
                    "role": task.robot_roles.get(robot_id, RobotRole.FOLLOWER).value
                })
        
        return {
            "success": True,
            "task": task.to_dict(),
            "robot_states": robot_states,
            "formation_type": task.formation.formation_type.value if task.formation else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务详情失败: {str(e)}"
        )


@router.get("/robots/states", response_model=Dict[str, Any])
async def get_all_robot_states(
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """获取所有机器人状态"""
    try:
        states = coordinator.get_all_robot_states()
        
        state_list = []
        for robot_id, state in states.items():
            state_list.append(state.to_dict())
        
        return {
            "success": True,
            "robot_states": state_list,
            "total_count": len(state_list)
        }
        
    except Exception as e:
        logger.error(f"获取机器人状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取机器人状态失败: {str(e)}"
        )


@router.post("/robots/{robot_id}/update-state", response_model=Dict[str, Any])
async def update_robot_state(
    robot_id: int,
    position: List[float] = Body([0.0, 0.0, 0.0], description="位置 [x, y, z]"),
    orientation: List[float] = Body([0.0, 0.0, 0.0, 1.0], description="方向四元数 [x, y, z, w]"),
    velocity: List[float] = Body([0.0, 0.0, 0.0], description="速度 [vx, vy, vz]"),
    battery_level: float = Body(100.0, description="电池电量 (%)"),
    status: str = Body("idle", description="状态"),
    coordinator: MultiRobotCoordinator = Depends(get_coordinator),
    user: User = Depends(get_current_user),
):
    """更新机器人状态"""
    try:
        success, message = coordinator.update_robot_state(
            robot_id=robot_id,
            position=position,
            orientation=orientation,
            velocity=velocity,
            battery_level=battery_level,
            status=status
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        return {
            "success": True,
            "message": message,
            "robot_id": robot_id,
            "timestamp": time.time() if 'time' in globals() else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新机器人状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新机器人状态失败: {str(e)}"
        )


@router.get("/formations", response_model=Dict[str, Any])
async def get_available_formations():
    """获取可用队形列表"""
    try:
        formations = [
            {"type": "line", "name": "直线队形", "description": "机器人排列在一条直线上"},
            {"type": "column", "name": "纵队队形", "description": "机器人排列在一列上"},
            {"type": "v_shape", "name": "V字形队形", "description": "机器人排列成V字形"},
            {"type": "circle", "name": "圆形队形", "description": "机器人排列成圆形"},
            {"type": "square", "name": "方形队形", "description": "机器人排列成方形"},
            {"type": "triangle", "name": "三角形队形", "description": "机器人排列成三角形"}
        ]
        
        return {
            "success": True,
            "formations": formations,
            "total_count": len(formations)
        }
        
    except Exception as e:
        logger.error(f"获取队形列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取队形列表失败: {str(e)}"
        )