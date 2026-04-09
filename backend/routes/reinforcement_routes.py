#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习路由模块
处理Gazebo仿真环境的强化学习训练API请求

功能：
1. 训练任务管理（创建、启动、停止、删除）
2. 训练进度监控
3. 模型评估和部署
4. 与机器人管理集成
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.robot_reinforcement_service import (
    RobotReinforcementTrainingService,
    RobotTrainingConfig,
    TrainingTaskStatus,
    TrainingAlgorithm,
    RobotTaskType,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reinforcement", tags=["强化学习"])

# 全局训练服务实例
_training_service = None


def get_training_service(db: Session = Depends(get_db)):
    """获取训练服务实例"""
    global _training_service
    if _training_service is None:
        _training_service = RobotReinforcementTrainingService(db)
    return _training_service


@router.get("/dependencies", response_model=Dict[str, Any])
async def check_dependencies(
    service: RobotReinforcementTrainingService = Depends(get_training_service),
):
    """检查强化学习依赖库是否可用"""
    try:
        dependencies_available = service.dependencies_available
        return {
            "success": True,
            "dependencies_available": dependencies_available,
            "message": "强化学习依赖库已检查",
        }
    except Exception as e:
        logger.error(f"检查依赖库失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检查依赖库失败: {str(e)}",
        )


@router.post("/training/create", response_model=Dict[str, Any])
async def create_training_task(
    robot_id: int = Body(..., description="机器人ID"),
    task_type: RobotTaskType = Body(RobotTaskType.STAND_UP, description="任务类型"),
    algorithm: TrainingAlgorithm = Body(TrainingAlgorithm.PPO, description="训练算法"),
    total_timesteps: int = Body(100000, description="总训练步数"),
    learning_rate: float = Body(3e-4, description="学习率"),
    max_steps: int = Body(1000, description="每回合最大步数"),
    gazebo_world: str = Body("empty.world", description="Gazebo世界"),
    robot_model: str = Body("humanoid", description="机器人模型"),
    gui_enabled: bool = Body(False, description="启用GUI"),
    service: RobotReinforcementTrainingService = Depends(get_training_service),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """创建强化学习训练任务"""
    try:
        # 创建训练配置
        config = RobotTrainingConfig(
            robot_id=robot_id,
            task_type=task_type,
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            learning_rate=learning_rate,
            max_steps=max_steps,
            gazebo_world=gazebo_world,
            robot_model=robot_model,
            gui_enabled=gui_enabled,
        )

        # 创建训练任务
        result = service.create_training_task(config, user)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
            )

        return {
            "success": True,
            "task_id": result["task_id"],
            "message": result["message"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建训练任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建训练任务失败: {str(e)}",
        )


@router.post("/training/{task_id}/start", response_model=Dict[str, Any])
async def start_training_task(
    task_id: str,
    service: RobotReinforcementTrainingService = Depends(get_training_service),
    user: User = Depends(get_current_user),
):
    """开始训练任务"""
    try:
        # 验证任务所有权
        # 注意：服务内部会验证，这里也可以添加额外验证

        # 启动训练
        result = service.start_training(task_id)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
            )

        return {
            "success": True,
            "message": result["message"],
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动训练任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动训练任务失败: {str(e)}",
        )


@router.post("/training/{task_id}/stop", response_model=Dict[str, Any])
async def stop_training_task(
    task_id: str,
    service: RobotReinforcementTrainingService = Depends(get_training_service),
    user: User = Depends(get_current_user),
):
    """停止训练任务"""
    try:
        # 停止训练
        result = service.stop_training(task_id)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
            )

        return {
            "success": True,
            "message": result["message"],
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止训练任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止训练任务失败: {str(e)}",
        )


@router.get("/training/{task_id}/status", response_model=Dict[str, Any])
async def get_training_status(
    task_id: str,
    service: RobotReinforcementTrainingService = Depends(get_training_service),
    user: User = Depends(get_current_user),
):
    """获取训练任务状态"""
    try:
        # 获取任务状态
        result = service.get_task_status(task_id)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=result["error"]
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取训练状态失败: {str(e)}",
        )


@router.get("/training/list", response_model=Dict[str, Any])
async def list_training_tasks(
    robot_id: Optional[int] = Query(None, description="机器人ID过滤"),
    status_filter: Optional[TrainingTaskStatus] = Query(None, description="状态过滤"),
    skip: int = Query(0, ge=0, description="跳过记录数"),
    limit: int = Query(100, ge=1, le=1000, description="返回记录数"),
    service: RobotReinforcementTrainingService = Depends(get_training_service),
    user: User = Depends(get_current_user),
):
    """列出训练任务"""
    try:
        # 获取任务列表（按用户过滤）
        result = service.list_tasks(user.id)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "获取任务列表失败"),
            )

        tasks = result["tasks"]

        # 应用过滤器
        filtered_tasks = []
        for task in tasks:
            # 机器人ID过滤
            if robot_id is not None and task["robot_id"] != robot_id:
                continue

            # 状态过滤
            if status_filter is not None and task["status"] != status_filter.value:
                continue

            filtered_tasks.append(task)

        # 分页
        total_count = len(filtered_tasks)
        paginated_tasks = filtered_tasks[skip: skip + limit]

        return {
            "success": True,
            "tasks": paginated_tasks,
            "pagination": {
                "total": total_count,
                "skip": skip,
                "limit": limit,
                "has_more": skip + limit < total_count,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"列出训练任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"列出训练任务失败: {str(e)}",
        )


@router.post("/training/{task_id}/evaluate", response_model=Dict[str, Any])
async def evaluate_training_model(
    task_id: str,
    num_episodes: int = Body(10, ge=1, le=100, description="评估集数"),
    service: RobotReinforcementTrainingService = Depends(get_training_service),
    user: User = Depends(get_current_user),
):
    """评估训练模型"""
    try:
        # 评估模型
        result = service.evaluate_model(task_id, num_episodes)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
            )

        return {
            "success": True,
            "evaluation": result["evaluation"],
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"评估模型失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"评估模型失败: {str(e)}",
        )


@router.delete("/training/{task_id}", response_model=Dict[str, Any])
async def delete_training_task(
    task_id: str,
    service: RobotReinforcementTrainingService = Depends(get_training_service),
    user: User = Depends(get_current_user),
):
    """删除训练任务"""
    try:
        # 删除任务
        result = service.delete_task(task_id)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
            )

        return {
            "success": True,
            "message": result["message"],
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除训练任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除训练任务失败: {str(e)}",
        )


@router.get("/environments", response_model=Dict[str, Any])
async def list_available_environments():
    """获取可用的强化学习环境"""
    try:
        environments = [
            {
                "id": "Gazebo-HumanoidStand-v0",
                "name": "人形机器人站立",
                "description": "训练人形机器人站立和保持平衡",
                "task_type": RobotTaskType.STAND_UP.value,
                "observation_space": "关节位置 + 速度 + IMU",
                "action_space": "25维关节控制",
                "reward_function": "高度奖励 + 稳定性奖励 + 能量惩罚",
            },
            {
                "id": "Gazebo-HumanoidWalk-v0",
                "name": "人形机器人行走",
                "description": "训练人形机器人行走",
                "task_type": RobotTaskType.WALK.value,
                "observation_space": "关节位置 + 速度 + IMU",
                "action_space": "25维关节控制",
                "reward_function": "前进距离奖励 + 稳定性奖励 + 能量惩罚",
            },
            {
                "id": "Gazebo-HumanoidBalance-v0",
                "name": "人形机器人平衡",
                "description": "训练人形机器人在扰动下保持平衡",
                "task_type": RobotTaskType.BALANCE.value,
                "observation_space": "关节位置 + 速度 + IMU",
                "action_space": "25维关节控制",
                "reward_function": "稳定性奖励 + 姿态惩罚",
            },
            {
                "id": "Gazebo-HumanoidReach-v0",
                "name": "人形机器人到达目标",
                "description": "训练人形机器人移动到目标位置",
                "task_type": RobotTaskType.REACH_TARGET.value,
                "observation_space": "关节位置 + 速度 + IMU + 目标位置",
                "action_space": "25维关节控制",
                "reward_function": "接近目标奖励 + 能量惩罚",
            },
        ]

        return {
            "success": True,
            "environments": environments,
            "count": len(environments),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"获取环境列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取环境列表失败: {str(e)}",
        )


@router.get("/algorithms", response_model=Dict[str, Any])
async def list_available_algorithms():
    """获取可用的强化学习算法"""
    try:
        algorithms = [
            {
                "id": TrainingAlgorithm.PPO.value,
                "name": "近端策略优化 (PPO)",
                "description": "稳定高效的策略梯度算法，适合连续控制任务",
                "suitable_for": ["连续控制", "机器人控制", "高维动作空间"],
                "advantages": ["稳定", "样本效率高", "易于调参"],
                "disadvantages": ["计算量较大", "需要并行环境"],
            },
            {
                "id": TrainingAlgorithm.A2C.value,
                "name": "优势演员评论家 (A2C)",
                "description": "同步版本的A3C算法，适合并行训练",
                "suitable_for": ["连续/离散控制", "并行训练"],
                "advantages": ["简单", "易于实现", "可并行"],
                "disadvantages": ["样本效率较低", "需要调参"],
            },
            {
                "id": TrainingAlgorithm.DQN.value,
                "name": "深度Q网络 (DQN)",
                "description": "基于值函数的深度强化学习算法",
                "suitable_for": ["离散动作空间", "游戏", "决策任务"],
                "advantages": ["稳定", "理论成熟", "易于理解"],
                "disadvantages": ["仅限离散动作", "样本效率低"],
            },
            {
                "id": TrainingAlgorithm.SAC.value,
                "name": "软演员评论家 (SAC)",
                "description": "基于最大熵的强化学习算法，探索能力强",
                "suitable_for": ["连续控制", "需要探索的任务", "机器人控制"],
                "advantages": ["探索能力强", "稳定", "自动调参"],
                "disadvantages": ["计算复杂", "需要调参"],
            },
            {
                "id": TrainingAlgorithm.TD3.value,
                "name": "双延迟深度确定性策略梯度 (TD3)",
                "description": "DDPG的改进版本，更稳定",
                "suitable_for": ["连续控制", "需要精确控制的任务"],
                "advantages": ["稳定", "避免过估计", "高效"],
                "disadvantages": ["调参复杂", "需要大量样本"],
            },
        ]

        return {
            "success": True,
            "algorithms": algorithms,
            "count": len(algorithms),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"获取算法列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取算法列表失败: {str(e)}",
        )
