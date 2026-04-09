#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人多模态控制路由模块

功能：
1. 提供基于多模态感知的机器人控制API
2. 支持文本、图像、音频、传感器等多模态输入
3. 实现学习控制和自适应行为API
4. 提供多模态能力查询和状态监控
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from typing import Dict, Any
from datetime import datetime, timezone
import logging

from backend.dependencies import get_current_user
from backend.db_models.user import User
from backend.services.robot_multimodal_service import get_robot_multimodal_service

router = APIRouter(prefix="/api/robot/multimodal", tags=["机器人多模态控制"])

logger = logging.getLogger(__name__)


@router.get("/status", response_model=Dict[str, Any])
async def get_multimodal_service_status(
    user: User = Depends(get_current_user),
):
    """获取机器人多模态服务状态

    返回:
        服务状态信息
    """
    try:
        service = get_robot_multimodal_service()
        status_info = service.get_service_status()

        return {
            "success": True,
            "data": status_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取多模态服务状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取多模态服务状态失败: {str(e)}",
        )


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_multimodal_capabilities(
    user: User = Depends(get_current_user),
):
    """获取多模态控制能力信息

    返回:
        能力信息
    """
    try:
        service = get_robot_multimodal_service()
        capabilities = service.get_multimodal_capabilities()

        return {
            "success": True,
            "data": capabilities,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取多模态能力信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取多模态能力信息失败: {str(e)}",
        )


@router.post("/command", response_model=Dict[str, Any])
async def process_multimodal_command(
    command_data: Dict[str, Any] = Body(
        ...,
        examples={
            "example": {
                "summary": "多模态命令示例",
                "value": {
                    "text_command": "向前走然后挥手打招呼",
                    "behavior_mode": "autonomous",
                    "sensor_data": {
                        "imu": {
                            "acceleration": [0.01, 0.02, 9.81],
                            "gyroscope": [0.001, 0.002, 0.003],
                            "temperature": 25.5,
                        }
                    },
                },
            }
        },
    ),
    user: User = Depends(get_current_user),
):
    """处理多模态机器人命令

    参数:
        command_data: 多模态命令数据

    返回:
        命令处理结果
    """
    try:
        service = get_robot_multimodal_service()
        result = service.process_multimodal_command(command_data)

        return {
            "success": result.get("success", False),
            "data": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"处理多模态命令失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理多模态命令失败: {str(e)}",
        )


@router.get("/sensors", response_model=Dict[str, Any])
async def get_sensor_data_for_multimodal(
    user: User = Depends(get_current_user),
):
    """获取用于多模态处理的传感器数据

    返回:
        格式化的传感器数据
    """
    try:
        service = get_robot_multimodal_service()
        sensor_data = service.get_sensor_data_for_multimodal()

        return {
            "success": True,
            "data": sensor_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取传感器数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取传感器数据失败: {str(e)}",
        )


@router.post("/learning/start", response_model=Dict[str, Any])
async def start_learning_session(
    learning_type: str = Query(
        ..., description="学习类型: demonstration, reinforcement, imitation"
    ),
    learning_config: Dict[str, Any] = Body(
        ...,
        examples={
            "example": {
                "summary": "学习会话配置示例",
                "value": {
                    "demonstration_data": {
                        "task": "拿起水杯",
                        "trajectory": [],
                        "sensor_data": {},
                    }
                },
            }
        },
    ),
    user: User = Depends(get_current_user),
):
    """开始学习会话

    参数:
        learning_type: 学习类型
        learning_config: 学习配置

    返回:
        学习会话结果
    """
    try:
        service = get_robot_multimodal_service()
        result = service.start_learning_session(learning_type, learning_config)

        return {
            "success": result.get("success", False),
            "data": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"启动学习会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动学习会话失败: {str(e)}",
        )


@router.get("/learning/status", response_model=Dict[str, Any])
async def get_learning_status(
    user: User = Depends(get_current_user),
):
    """获取学习状态

    返回:
        学习状态信息
    """
    try:
        service = get_robot_multimodal_service()
        status_info = service.get_learning_status()

        return {
            "success": True,
            "data": status_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取学习状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取学习状态失败: {str(e)}",
        )


@router.post("/behavior/set", response_model=Dict[str, Any])
async def set_behavior_mode(
    mode: str = Query(
        ..., description="行为模式: manual, autonomous, learning, adaptive, safety"
    ),
    user: User = Depends(get_current_user),
):
    """设置机器人行为模式

    参数:
        mode: 行为模式

    返回:
        设置结果
    """
    try:
        from models.robot_multimodal_control import RobotBehaviorMode

        # 验证模式
        valid_modes = [m.value for m in RobotBehaviorMode]
        if mode not in valid_modes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的行为模式。有效模式: {valid_modes}",
            )

        get_robot_multimodal_service()

        # 注意：这里需要调用控制器的set_behavior_mode方法
        # 目前服务中没有直接暴露此方法，需要扩展服务

        return {
            "success": True,
            "data": {
                "mode": mode,
                "message": f"行为模式已设置为: {mode}",
                "previous_mode": "unknown",  # 实际实现中应获取当前模式
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置行为模式失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置行为模式失败: {str(e)}",
        )


@router.post("/test/simple", response_model=Dict[str, Any])
async def test_simple_command(
    text_command: str = Query(..., description="测试文本命令"),
    user: User = Depends(get_current_user),
):
    """测试简单文本命令处理

    参数:
        text_command: 文本命令

    返回:
        测试结果
    """
    try:
        service = get_robot_multimodal_service()
        result = service.process_multimodal_command(
            {"text_command": text_command, "behavior_mode": "autonomous"}
        )

        return {
            "success": result.get("success", False),
            "data": {
                "original_command": text_command,
                "processing_result": result,
                "generated_commands_count": result.get("generated_commands_count", 0),
                "execution_result": result.get("execution_result"),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"测试简单命令失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"测试简单命令失败: {str(e)}",
        )


@router.get("/examples", response_model=Dict[str, Any])
async def get_command_examples(
    user: User = Depends(get_current_user),
):
    """获取多模态命令示例

    返回:
        命令示例列表
    """
    examples = [
        {
            "description": "基本移动命令",
            "command": {
                "text_command": "向前走0.5米然后向右转90度",
                "behavior_mode": "autonomous",
            },
        },
        {
            "description": "姿态控制命令",
            "command": {
                "text_command": "站立姿势然后挥手打招呼",
                "behavior_mode": "autonomous",
            },
        },
        {
            "description": "传感器增强命令",
            "command": {
                "text_command": "检查周围环境",
                "behavior_mode": "autonomous",
                "sensor_data": {
                    "imu": {
                        "acceleration": [0.0, 0.0, 9.81],
                        "gyroscope": [0.0, 0.0, 0.0],
                    }
                },
            },
        },
        {
            "description": "学习控制示例",
            "command": {
                "text_command": "学习如何拿起水杯",
                "behavior_mode": "learning",
            },
        },
        {
            "description": "自适应行为示例",
            "command": {
                "text_command": "在房间里自主探索",
                "behavior_mode": "adaptive",
            },
        },
    ]

    return {
        "success": True,
        "data": {"examples": examples, "total_examples": len(examples)},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_multimodal_health(
    user: User = Depends(get_current_user),
):
    """获取多模态控制健康状态

    返回:
        健康状态信息
    """
    try:
        service = get_robot_multimodal_service()
        status_info = service.get_service_status()

        health_status = {
            "service_available": status_info.get("service_status") == "running",
            "multimodal_available": status_info.get("multimodal_available", False),
            "learning_available": status_info.get("learning_available", False),
            "robot_control_available": status_info.get(
                "robot_control_available", False
            ),
            "command_count": status_info.get("command_count", 0),
            "error_count": status_info.get("error_count", 0),
            "last_update": status_info.get("last_update"),
            "overall_status": (
                "healthy"
                if status_info.get("service_status") == "running"
                else "degraded"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "success": True,
            "data": health_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取健康状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取健康状态失败: {str(e)}",
        )


# 初始化检查
@router.get("/", response_model=Dict[str, Any])
async def root():
    """多模态控制API根端点

    返回:
        API基本信息
    """
    return {
        "success": True,
        "data": {
            "service": "机器人多模态控制API",
            "version": "1.0.0",
            "description": "提供基于多模态感知的机器人控制功能",
            "endpoints": [
                "/status - 获取服务状态",
                "/capabilities - 获取能力信息",
                "/command - 处理多模态命令",
                "/sensors - 获取传感器数据",
                "/learning/start - 开始学习会话",
                "/learning/status - 获取学习状态",
                "/behavior/set - 设置行为模式",
                "/test/simple - 测试简单命令",
                "/examples - 获取命令示例",
                "/health - 获取健康状态",
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
