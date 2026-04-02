#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径规划API路由
提供机器人路径规划和可视化功能
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session

from backend.dependencies import get_db
from backend.dependencies.auth import get_current_user
from backend.db_models.user import User
from backend.services.path_planning_service import (
    get_path_planning_service,
    PathPlanningService,
    PathPlanningAlgorithm,
    EnvironmentType,
    PlanningResult
)

router = APIRouter(prefix="/api/path-planning", tags=["路径规划"])
logger = logging.getLogger(__name__)


def get_planning_service() -> PathPlanningService:
    """获取路径规划服务"""
    return get_path_planning_service()


@router.post("/environment/create", response_model=Dict[str, Any])
async def create_environment(
    env_id: str = Body(..., description="环境ID"),
    width: int = Body(10, description="宽度（单元格数）"),
    height: int = Body(10, description="高度（单元格数）"),
    depth: int = Body(1, description="深度（单元格数，3D环境使用）"),
    cell_size: float = Body(1.0, description="单元格大小"),
    service: PathPlanningService = Depends(get_planning_service),
    user: User = Depends(get_current_user),
):
    """创建网格环境"""
    try:
        success = service.create_grid_environment(
            env_id=env_id,
            width=width,
            height=height,
            depth=depth,
            cell_size=cell_size
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="环境创建失败"
            )
        
        return {
            "success": True,
            "message": "环境创建成功",
            "env_id": env_id,
            "dimensions": {
                "width": width,
                "height": height,
                "depth": depth
            },
            "cell_size": cell_size,
            "grid_size": width * height * depth
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建环境失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建环境失败: {str(e)}"
        )


@router.post("/environment/{env_id}/obstacle", response_model=Dict[str, Any])
async def add_obstacle(
    env_id: str,
    position: List[float] = Body(..., description="位置 [x, y, z]"),
    dimensions: List[float] = Body([1.0, 1.0, 1.0], description="尺寸 [长, 宽, 高]"),
    shape_type: str = Body("box", description="形状类型 (box, sphere, cylinder)"),
    name: str = Body("", description="障碍物名称"),
    service: PathPlanningService = Depends(get_planning_service),
    user: User = Depends(get_current_user),
):
    """添加障碍物到环境"""
    try:
        success = service.add_obstacle(
            env_id=env_id,
            position=position,
            dimensions=dimensions,
            shape_type=shape_type,
            name=name
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="障碍物添加失败"
            )
        
        return {
            "success": True,
            "message": "障碍物添加成功",
            "env_id": env_id,
            "position": position,
            "dimensions": dimensions,
            "shape_type": shape_type,
            "name": name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加障碍物失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加障碍物失败: {str(e)}"
        )


@router.post("/plan", response_model=Dict[str, Any])
async def plan_path(
    env_id: str = Body(..., description="环境ID"),
    start: List[float] = Body(..., description="起点 [x, y, z]"),
    goal: List[float] = Body(..., description="终点 [x, y, z]"),
    algorithm: PathPlanningAlgorithm = Body(PathPlanningAlgorithm.ASTAR, description="规划算法"),
    max_iterations: int = Body(10000, description="最大迭代次数"),
    step_size: float = Body(1.0, description="步长（连续环境使用）"),
    service: PathPlanningService = Depends(get_planning_service),
    user: User = Depends(get_current_user),
):
    """规划路径"""
    try:
        result = service.plan_path(
            env_id=env_id,
            start=start,
            goal=goal,
            algorithm=algorithm,
            max_iterations=max_iterations,
            step_size=step_size
        )
        
        return {
            "success": result.success,
            "planning_result": result.to_dict(),
            "message": result.message
        }
        
    except Exception as e:
        logger.error(f"路径规划失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"路径规划失败: {str(e)}"
        )


@router.get("/environment/{env_id}", response_model=Dict[str, Any])
async def get_environment_info(
    env_id: str,
    service: PathPlanningService = Depends(get_planning_service),
    user: User = Depends(get_current_user),
):
    """获取环境信息"""
    try:
        info = service.get_environment_info(env_id)
        
        if not info["exists"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"环境 {env_id} 不存在"
            )
        
        return {
            "success": True,
            "environment_info": info,
            "env_id": env_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取环境信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取环境信息失败: {str(e)}"
        )


@router.get("/environment/{env_id}/visualization", response_model=Dict[str, Any])
async def get_visualization_data(
    env_id: str,
    service: PathPlanningService = Depends(get_planning_service),
    user: User = Depends(get_current_user),
):
    """获取可视化数据"""
    try:
        data = service.get_visualization_data(env_id)
        
        if not data["exists"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"环境 {env_id} 不存在"
            )
        
        return {
            "success": True,
            "visualization_data": data,
            "env_id": env_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取可视化数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取可视化数据失败: {str(e)}"
        )


@router.get("/algorithms", response_model=Dict[str, Any])
async def get_available_algorithms():
    """获取可用的路径规划算法"""
    try:
        algorithms = [
            {
                "id": PathPlanningAlgorithm.ASTAR.value,
                "name": "A*算法",
                "description": "启发式搜索算法，结合了Dijkstra和最佳优先搜索的优点",
                "suitable_for": ["网格环境", "已知地图", "最短路径"],
                "time_complexity": "O(b^d)，其中b是分支因子，d是深度",
                "space_complexity": "O(b^d)",
                "advantages": ["保证找到最短路径", "高效"],
                "disadvantages": ["需要启发式函数", "内存消耗大"]
            },
            {
                "id": PathPlanningAlgorithm.DIJKSTRA.value,
                "name": "Dijkstra算法",
                "description": "单源最短路径算法，适用于带权图",
                "suitable_for": ["带权图", "网格环境", "已知地图"],
                "time_complexity": "O((V+E) log V)",
                "space_complexity": "O(V)",
                "advantages": ["保证找到最短路径", "不需要启发式函数"],
                "disadvantages": ["计算量大", "不适合大型地图"]
            },
            {
                "id": PathPlanningAlgorithm.RRT.value,
                "name": "快速随机扩展树 (RRT)",
                "description": "基于采样的路径规划算法，适用于高维空间和动态环境",
                "suitable_for": ["连续空间", "高维空间", "动态环境"],
                "time_complexity": "O(n log n)",
                "space_complexity": "O(n)",
                "advantages": ["高维空间有效", "动态环境适用"],
                "disadvantages": ["不保证最优", "可能找到次优路径"]
            },
            {
                "id": PathPlanningAlgorithm.RRT_STAR.value,
                "name": "RRT*算法",
                "description": "RRT的渐进最优版本，能够收敛到最优路径",
                "suitable_for": ["连续空间", "需要最优解", "机器人运动规划"],
                "time_complexity": "O(n log n)",
                "space_complexity": "O(n)",
                "advantages": ["渐进最优", "高维空间有效"],
                "disadvantages": ["计算量大", "收敛慢"]
            },
            {
                "id": PathPlanningAlgorithm.PRM.value,
                "name": "概率路线图 (PRM)",
                "description": "基于采样的路线图构建算法，适用于复杂环境",
                "suitable_for": ["复杂环境", "多查询场景", "高维空间"],
                "time_complexity": "O(n log n)",
                "space_complexity": "O(n^2)",
                "advantages": ["支持多查询", "离线预处理"],
                "disadvantages": ["内存消耗大", "构建时间较长"]
            }
        ]
        
        return {
            "success": True,
            "algorithms": algorithms,
            "algorithm_count": len(algorithms)
        }
        
    except Exception as e:
        logger.error(f"获取算法列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取算法列表失败: {str(e)}"
        )


@router.get("/environment-types", response_model=Dict[str, Any])
async def get_environment_types():
    """获取可用的环境类型"""
    try:
        environment_types = [
            {
                "id": EnvironmentType.GRID_2D.value,
                "name": "2D网格环境",
                "description": "二维网格环境，适用于平面导航和移动机器人",
                "dimensions": 2,
                "typical_use_cases": ["移动机器人", "平面导航", "2D游戏"],
                "advantages": ["计算简单", "易于实现"],
                "disadvantages": ["分辨率限制", "不能表示3D地形"]
            },
            {
                "id": EnvironmentType.GRID_3D.value,
                "name": "3D网格环境",
                "description": "三维网格环境，适用于空中机器人、水下机器人等",
                "dimensions": 3,
                "typical_use_cases": ["无人机", "水下机器人", "3D导航"],
                "advantages": ["真实3D表示", "支持复杂地形"],
                "disadvantages": ["计算量大", "内存消耗高"]
            },
            {
                "id": EnvironmentType.CONTINUOUS_2D.value,
                "name": "2D连续环境",
                "description": "二维连续环境，适用于精确运动规划",
                "dimensions": 2,
                "typical_use_cases": ["机器人手臂", "精确运动控制", "连续空间规划"],
                "advantages": ["高精度", "连续位置"],
                "disadvantages": ["算法复杂", "计算量大"]
            },
            {
                "id": EnvironmentType.CONTINUOUS_3D.value,
                "name": "3D连续环境",
                "description": "三维连续环境，适用于复杂机械系统和机器人",
                "dimensions": 3,
                "typical_use_cases": ["人形机器人", "复杂机械系统", "3D连续控制"],
                "advantages": ["最高精度", "真实连续空间"],
                "disadvantages": ["非常复杂", "计算量巨大"]
            },
            {
                "id": EnvironmentType.VOXEL.value,
                "name": "体素环境",
                "description": "基于体素的环境表示，适用于复杂3D场景和障碍物",
                "dimensions": 3,
                "typical_use_cases": ["复杂场景", "动态障碍物", "3D重建环境"],
                "advantages": ["精确体积表示", "支持复杂几何"],
                "disadvantages": ["内存消耗高", "计算量大"]
            }
        ]
        
        return {
            "success": True,
            "environment_types": environment_types,
            "type_count": len(environment_types)
        }
        
    except Exception as e:
        logger.error(f"获取环境类型列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取环境类型列表失败: {str(e)}"
        )


@router.delete("/environment/{env_id}", response_model=Dict[str, Any])
async def delete_environment(
    env_id: str,
    service: PathPlanningService = Depends(get_planning_service),
    user: User = Depends(get_current_user),
):
    """删除环境"""
    try:
        # 注意：实际的服务可能需要删除环境的实现
        # 完整处理，假设服务有删除功能
        if hasattr(service, 'grids') and env_id in service.grids:
            del service.grids[env_id]
            
        if hasattr(service, 'obstacles') and env_id in service.obstacles:
            del service.obstacles[env_id]
        
        return {
            "success": True,
            "message": "环境删除成功",
            "env_id": env_id
        }
        
    except Exception as e:
        logger.error(f"删除环境失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除环境失败: {str(e)}"
        )