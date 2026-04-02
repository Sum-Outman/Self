#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运动控制API路由
提供机器人运动控制、路径规划和执行功能
"""

import sys
import os
import logging
from typing import Dict, Any, List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session

from backend.dependencies import get_db
from backend.dependencies.auth import get_current_user
from backend.db_models.user import User

# 导入仿真管理器 - 多种导入方式尝试
try:
    # 方式1：绝对导入（从项目根目录）
    from hardware.simulation import get_global_simulation_manager
    from hardware.gazebo_simulation import GazeboSimulation
except ImportError:
    try:
        # 方式2：相对导入（假设从backend目录运行）
        from backend.hardware.simulation import get_global_simulation_manager
        from backend.hardware.gazebo_simulation import GazeboSimulation
    except ImportError:
        try:
            # 方式3：添加项目根目录到路径
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from hardware.simulation import get_global_simulation_manager
            from hardware.gazebo_simulation import GazeboSimulation
        except ImportError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"导入仿真模块失败: {e}")
            # 创建模拟函数以防导入失败
            def get_global_simulation_manager():
                class MockSimulationManager:
                    def get_simulation(self, name):
                        return None  # 返回None
                    def create_simulation(self, name, **kwargs):
                        return None  # 返回None
                return MockSimulationManager()
            
            class GazeboSimulation:
                def __init__(self, **kwargs):
                    pass  # 已修复: 实现函数功能
                def connect(self):
                    return False
                def is_connected(self):
                    return False
                def plan_path(self, **kwargs):
                    return {"success": False, "message": "Gazebo仿真不可用"}
                def execute_path(self, **kwargs):
                    return False
                def move_to_position(self, **kwargs):
                    return False
                def get_interface_info(self):
                    return {}  # 返回空字典

router = APIRouter(prefix="/api/motion-control", tags=["运动控制"])
logger = logging.getLogger(__name__)


def get_simulation_manager():
    """获取全局仿真管理器"""
    return get_global_simulation_manager()


@router.post("/plan-path", response_model=Dict[str, Any])
async def plan_path(
    start_position: List[float] = Body(..., description="起点位置 [x, y, z]"),
    goal_position: List[float] = Body(..., description="目标位置 [x, y, z]"),
    algorithm: str = Body("astar", description="规划算法 (astar, rrt, rrt_star)"),
    grid_size: float = Body(0.1, description="网格大小"),
    max_iterations: int = Body(1000, description="最大迭代次数"),
    simulation_type: str = Body("pybullet", description="仿真类型 (pybullet, gazebo)"),

):
    """规划机器人从起点到目标的路径"""
    try:
        logger.info(f"运动控制API: 规划路径 {start_position} -> {goal_position}, 算法: {algorithm}")
        
        # 获取仿真管理器
        sim_manager = get_simulation_manager()
        logger.info(f"仿真管理器: {sim_manager}")
        
        # 获取仿真实例
        simulation = None
        if simulation_type == "pybullet":
            simulation = sim_manager.get_simulation("default")
            if not simulation:
                # 创建默认仿真
                simulation = sim_manager.create_simulation("default", gui_enabled=False)
                if simulation:
                    simulation.connect()
        elif simulation_type == "gazebo":
            # 创建Gazebo仿真实例
            simulation = GazeboSimulation(gui_enabled=False)
            simulation.connect()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的仿真类型: {simulation_type}"
            )
        
        if not simulation:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="仿真环境不可用"
            )
        
        # 调用仿真路径规划
        result = simulation.plan_path(
            start_position=start_position,
            goal_position=goal_position,
            algorithm=algorithm,
            grid_size=grid_size,
            max_iterations=max_iterations
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "路径规划失败")
            )
        
        return {
            "success": True,
            "message": result.get("message", "路径规划成功"),
            "data": {
                "path": result.get("path", []),
                "path_length": result.get("path_length", 0.0),
                "computation_time": result.get("computation_time", 0.0),
                "nodes_explored": result.get("nodes_explored", 0),
                "algorithm": result.get("algorithm", algorithm),
                "simulation_type": simulation_type
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"路径规划API错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"路径规划失败: {str(e)}"
        )


@router.post("/execute-path", response_model=Dict[str, Any])
async def execute_path(
    path: List[List[float]] = Body(..., description="路径点列表 [[x1, y1, z1], [x2, y2, z2], ...]"),
    speed: float = Body(0.1, description="移动速度（米/秒）"),
    simulation_type: str = Body("pybullet", description="仿真类型 (pybullet, gazebo)"),

):
    """执行规划好的路径"""
    try:
        logger.info(f"运动控制API: 执行路径, {len(path)} 个点, 速度: {speed} m/s")
        
        if len(path) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="路径点至少需要2个"
            )
        
        # 获取仿真管理器
        sim_manager = get_simulation_manager()
        
        # 获取仿真实例
        simulation = None
        if simulation_type == "pybullet":
            simulation = sim_manager.get_simulation("default")
            if not simulation:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="PyBullet仿真环境未初始化"
                )
        elif simulation_type == "gazebo":
            # 创建Gazebo仿真实例
            simulation = GazeboSimulation(gui_enabled=False)
            simulation.connect()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的仿真类型: {simulation_type}"
            )
        
        if not simulation:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="仿真环境不可用"
            )
        
        # 执行路径
        success = simulation.execute_path(path, speed)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="路径执行失败"
            )
        
        return {
            "success": True,
            "message": "路径执行成功",
            "data": {
                "points_executed": len(path),
                "speed": speed,
                "simulation_type": simulation_type
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"路径执行API错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"路径执行失败: {str(e)}"
        )


@router.post("/move-to-position", response_model=Dict[str, Any])
async def move_to_position(
    target_position: List[float] = Body(..., description="目标位置 [x, y, z]"),
    speed: float = Body(0.1, description="移动速度（米/秒）"),
    simulation_type: str = Body("pybullet", description="仿真类型 (pybullet, gazebo)"),

):
    """移动机器人到指定位置"""
    try:
        logger.info(f"运动控制API: 移动到位置 {target_position}, 速度: {speed} m/s")
        
        # 获取仿真管理器
        sim_manager = get_simulation_manager()
        
        # 获取仿真实例
        simulation = None
        if simulation_type == "pybullet":
            simulation = sim_manager.get_simulation("default")
            if not simulation:
                # 创建默认仿真
                simulation = sim_manager.create_simulation("default", gui_enabled=False)
                if simulation:
                    simulation.connect()
        elif simulation_type == "gazebo":
            # 创建Gazebo仿真实例
            simulation = GazeboSimulation(gui_enabled=False)
            simulation.connect()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的仿真类型: {simulation_type}"
            )
        
        if not simulation:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="仿真环境不可用"
            )
        
        # 移动到位置
        success = simulation.move_to_position(target_position, speed)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="移动机器人失败"
            )
        
        return {
            "success": True,
            "message": "机器人移动成功",
            "data": {
                "target_position": target_position,
                "speed": speed,
                "simulation_type": simulation_type
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"移动机器人API错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"移动机器人失败: {str(e)}"
        )


@router.get("/simulation-info", response_model=Dict[str, Any])
async def get_simulation_info(
    simulation_type: str = "pybullet",

):
    """获取仿真环境信息"""
    try:
        logger.info(f"运动控制API: 获取仿真信息, 类型: {simulation_type}")
        
        # 获取仿真管理器
        sim_manager = get_simulation_manager()
        
        # 获取仿真实例
        simulation = None
        if simulation_type == "pybullet":
            simulation = sim_manager.get_simulation("default")
        elif simulation_type == "gazebo":
            # 创建Gazebo仿真实例
            simulation = GazeboSimulation(gui_enabled=False)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的仿真类型: {simulation_type}"
            )
        
        if not simulation:
            return {
                "success": True,
                "message": "仿真环境未初始化",
                "data": {
                    "initialized": False,
                    "simulation_type": simulation_type
                }
            }
        
        # 获取仿真信息
        if simulation_type == "pybullet":
            info = simulation.get_simulation_info()
        else:  # gazebo
            info = simulation.get_interface_info()
        
        return {
            "success": True,
            "message": "获取仿真信息成功",
            "data": {
                "initialized": True,
                "connected": simulation.is_connected(),
                "simulation_type": simulation_type,
                "info": info
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取仿真信息API错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取仿真信息失败: {str(e)}"
        )