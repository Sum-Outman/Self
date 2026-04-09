"""
自主模式API路由
提供全自主模式和任务执行模式的切换和控制接口

功能：
1. 模式切换：全自主模式 ↔ 任务执行模式
2. 状态查询：获取当前模式状态和统计信息
3. 参数配置：配置自主模式参数
4. 目标管理：管理自主目标
5. 决策管理：查看决策历史和统计
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

# 导入自主模式管理器
try:
    from models.system_control.autonomous_mode_manager import (
        get_autonomous_mode_manager,
        AutonomousState,
        GoalPriority,
    )

    AUTONOMOUS_MODE_AVAILABLE = True
except ImportError as e:
    AUTONOMOUS_MODE_AVAILABLE = False
    logging.warning(f"自主模式管理器导入失败: {e}")

    # 创建虚拟函数和枚举以避免导入错误
    def get_autonomous_mode_manager():
        raise ImportError("自主模式管理器不可用")

    from enum import Enum

    class AutonomousState(Enum):
        DISABLED = "disabled"
        IDLE = "idle"
        ACTIVE = "active"
        ERROR = "error"

    class GoalPriority(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"


# 导入决策引擎
try:
    from models.autonomous.decision_engine import (
        get_decision_engine,
        DecisionType,
        EnvironmentState as DecisionEnvironmentState,
    )

    DECISION_ENGINE_AVAILABLE = True
except ImportError as e:
    DECISION_ENGINE_AVAILABLE = False
    logging.warning(f"决策引擎导入失败: {e}")

    # 创建虚拟函数和枚举以避免导入错误
    def get_decision_engine():
        raise ImportError("决策引擎不可用")

    from enum import Enum

    class DecisionType(Enum):
        ACTION = "action"
        NAVIGATION = "navigation"
        MANIPULATION = "manipulation"
        COMMUNICATION = "communication"
        LEARNING = "learning"

    class DecisionEnvironmentState:
        def __init__(self):
            self.state = {}
            self.observations = []
            self.rewards = 0.0


logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/api/system/mode",
    tags=["autonomous_mode"],
    responses={404: {"description": "未找到"}},
)


# Pydantic模型定义
class ModeSwitchRequest(BaseModel):
    """模式切换请求"""

    target_mode: str = Field(
        description="目标模式: 'autonomous'（自主模式）, 'task'（任务执行模式）",
        example="autonomous",
    )
    graceful: bool = Field(
        default=True,
        description="是否优雅切换（True: 完成当前任务后切换, False: 立即切换）",
        example=True,
    )
    reason: Optional[str] = Field(
        default=None, description="切换原因", example="用户请求切换到自主模式"
    )


class ModeConfigRequest(BaseModel):
    """模式配置请求"""

    enable_autonomous_mode: Optional[bool] = Field(
        default=None, description="是否启用自主模式", example=True
    )
    max_concurrent_goals: Optional[int] = Field(
        default=None, description="最大并发目标数", example=3, ge=1, le=10
    )
    goal_timeout_seconds: Optional[int] = Field(
        default=None, description="目标超时时间（秒）", example=3600, ge=60, le=86400
    )
    safety_check_enabled: Optional[bool] = Field(
        default=None, description="是否启用安全检查", example=True
    )
    ethical_check_enabled: Optional[bool] = Field(
        default=None, description="是否启用伦理检查", example=True
    )
    learning_enabled: Optional[bool] = Field(
        default=None, description="是否启用学习功能", example=True
    )
    exploration_enabled: Optional[bool] = Field(
        default=None, description="是否启用探索功能", example=True
    )


class GoalCreateRequest(BaseModel):
    """目标创建请求"""

    description: str = Field(description="目标描述", example="学习如何识别苹果")
    priority: str = Field(
        default="MEDIUM",
        description="目标优先级: CRITICAL, HIGH, MEDIUM, LOW, BACKGROUND",
        example="MEDIUM",
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="目标参数",
        example={"learning_topic": "apple_recognition", "duration_minutes": 30},
    )
    deadline_minutes: Optional[int] = Field(
        default=None,
        description="截止时间（分钟，从当前时间开始计算）",
        example=60,
        ge=1,
        le=10080,  # 7天
    )


class DecisionRequest(BaseModel):
    """决策请求"""

    sensor_data: Optional[Dict[str, float]] = Field(
        default=None,
        description="传感器数据",
        example={"temperature": 25.5, "humidity": 60.0},
    )
    system_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="系统指标",
        example={"cpu_usage": 45.0, "memory_usage": 60.0},
    )
    task_progress: Optional[Dict[str, float]] = Field(
        default=None,
        description="任务进度",
        example={"current_task": 0.75, "overall_progress": 0.5},
    )


class ModeStatusResponse(BaseModel):
    """模式状态响应"""

    autonomous_mode_enabled: bool
    autonomous_mode_running: bool
    current_state: str
    previous_state: str
    active_goals_count: int
    pending_goals_count: int
    total_goals: int
    completed_goals: int
    failed_goals: int
    avg_goal_duration: float
    safety_violations: int
    ethical_violations: int
    last_state_change: Optional[str]
    timestamp: str


class GoalResponse(BaseModel):
    """目标响应"""

    id: str
    description: str
    priority: str
    status: str
    progress: float
    created_at: str
    deadline: Optional[str]
    estimated_duration: Optional[float]
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]]


class DecisionResponse(BaseModel):
    """决策响应"""

    id: str
    type: str
    action: str
    confidence: float
    expected_reward: float
    risk_level: float
    created_at: float
    parameters: Dict[str, Any]


class DecisionStatisticsResponse(BaseModel):
    """决策统计响应"""

    total_decisions: int
    successful_decisions: int
    failed_decisions: int
    avg_decision_time: float
    avg_reward: float
    avg_risk: float
    avg_confidence: float
    exploration_rate: float
    memory_size: int
    decision_history_size: int
    action_space_size: int


# 依赖项：获取自主模式管理器
def get_autonomous_manager():
    """获取自主模式管理器"""
    if not AUTONOMOUS_MODE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="自主模式管理器不可用"
        )

    try:
        return get_autonomous_mode_manager()
    except Exception as e:
        logger.error(f"获取自主模式管理器失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取自主模式管理器失败: {str(e)}",
        )


# 依赖项：获取决策引擎
def get_decision_engine_dependency():
    """获取决策引擎"""
    if not DECISION_ENGINE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="决策引擎不可用"
        )

    try:
        return get_decision_engine()
    except Exception as e:
        logger.error(f"获取决策引擎失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取决策引擎失败: {str(e)}",
        )


# API端点
@router.post("/switch", summary="切换运行模式")
async def switch_mode(
    request: ModeSwitchRequest, manager: Any = Depends(get_autonomous_manager)
):
    """切换系统运行模式

    支持在自主模式和任务执行模式之间切换
    """
    try:
        if request.target_mode.lower() == "autonomous":
            # 切换到自主模式
            if manager.running:
                logger.info("自主模式已在运行中")
                return {
                    "success": True,
                    "message": "自主模式已在运行中",
                    "current_mode": "autonomous",
                }

            success = manager.start_autonomous_mode()
            if success:
                logger.info(f"切换到自主模式成功，原因: {request.reason}")
                return {
                    "success": True,
                    "message": "切换到自主模式成功",
                    "current_mode": "autonomous",
                    "reason": request.reason,
                    "graceful": request.graceful,
                }
            else:
                logger.error("切换到自主模式失败")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="切换到自主模式失败",
                )

        elif request.target_mode.lower() == "task":
            # 切换到任务执行模式
            if not manager.running:
                logger.info("系统未在自主模式中运行")
                return {
                    "success": True,
                    "message": "系统未在自主模式中运行",
                    "current_mode": "task",
                }

            if request.graceful:
                # 优雅切换：等待当前任务完成
                success = manager.switch_to_task_execution_mode()
            else:
                # 立即切换：停止自主模式
                success = manager.stop_autonomous_mode()

            if success:
                logger.info(f"切换到任务执行模式成功，原因: {request.reason}")
                return {
                    "success": True,
                    "message": "切换到任务执行模式成功",
                    "current_mode": "task",
                    "reason": request.reason,
                    "graceful": request.graceful,
                }
            else:
                logger.error("切换到任务执行模式失败")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="切换到任务执行模式失败",
                )

        else:
            logger.error(f"不支持的目标模式: {request.target_mode}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的目标模式: {request.target_mode}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模式切换失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模式切换失败: {str(e)}",
        )


@router.get("/status", response_model=ModeStatusResponse, summary="获取模式状态")
async def get_mode_status(manager: Any = Depends(get_autonomous_manager)):
    """获取当前模式状态和统计信息"""
    try:
        status_info = manager.get_current_status()
        return status_info
    except Exception as e:
        logger.error(f"获取模式状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模式状态失败: {str(e)}",
        )


@router.post("/config", summary="配置模式参数")
async def configure_mode(
    request: ModeConfigRequest, manager: Any = Depends(get_autonomous_manager)
):
    """配置自主模式参数"""
    try:
        config_updates = {}

        # 更新配置参数
        if request.enable_autonomous_mode is not None:
            manager.config["enable_autonomous_mode"] = request.enable_autonomous_mode
            config_updates["enable_autonomous_mode"] = request.enable_autonomous_mode

        if request.max_concurrent_goals is not None:
            manager.config["max_concurrent_goals"] = request.max_concurrent_goals
            config_updates["max_concurrent_goals"] = request.max_concurrent_goals

        if request.goal_timeout_seconds is not None:
            manager.config["goal_timeout_seconds"] = request.goal_timeout_seconds
            config_updates["goal_timeout_seconds"] = request.goal_timeout_seconds

        if request.safety_check_enabled is not None:
            manager.config["safety_check_enabled"] = request.safety_check_enabled
            config_updates["safety_check_enabled"] = request.safety_check_enabled

        if request.ethical_check_enabled is not None:
            manager.config["ethical_check_enabled"] = request.ethical_check_enabled
            config_updates["ethical_check_enabled"] = request.ethical_check_enabled

        if request.learning_enabled is not None:
            manager.config["learning_enabled"] = request.learning_enabled
            config_updates["learning_enabled"] = request.learning_enabled

        if request.exploration_enabled is not None:
            manager.config["exploration_enabled"] = request.exploration_enabled
            config_updates["exploration_enabled"] = request.exploration_enabled

        logger.info(f"模式配置更新: {config_updates}")

        return {
            "success": True,
            "message": "模式配置更新成功",
            "config_updates": config_updates,
            "current_config": manager.config,
        }

    except Exception as e:
        logger.error(f"模式配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模式配置失败: {str(e)}",
        )


@router.post("/goals", summary="创建自主目标")
async def create_goal(
    request: GoalCreateRequest, manager: Any = Depends(get_autonomous_manager)
):
    """创建自主目标"""
    try:
        # 转换优先级字符串为枚举
        priority_map = {
            "CRITICAL": GoalPriority.CRITICAL,
            "HIGH": GoalPriority.HIGH,
            "MEDIUM": GoalPriority.MEDIUM,
            "LOW": GoalPriority.LOW,
            "BACKGROUND": GoalPriority.BACKGROUND,
        }

        priority = priority_map.get(request.priority.upper())
        if not priority:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的优先级: {request.priority}",
            )

        # 计算截止时间
        deadline = None
        if request.deadline_minutes:
            deadline = datetime.now() + timedelta(minutes=request.deadline_minutes)

        # 创建目标
        goal_id = manager.add_goal(
            goal_description=request.description,
            priority=priority,
            parameters=request.parameters,
            deadline=deadline,
        )

        if not goal_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="目标创建失败（可能由于安全检查或伦理检查失败）",
            )

        # 获取创建的目标信息
        goal = manager.goals.get(goal_id)
        if not goal:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="目标创建后无法获取目标信息",
            )

        logger.info(f"自主目标创建成功: {goal_id} - {request.description}")

        return {
            "success": True,
            "message": "目标创建成功",
            "goal_id": goal_id,
            "goal": goal.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"目标创建失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"目标创建失败: {str(e)}",
        )


@router.get("/goals", summary="获取目标列表")
async def get_goals(
    manager: Any = Depends(get_autonomous_manager),
    status_filter: Optional[str] = None,
    priority_filter: Optional[str] = None,
    limit: int = 50,
):
    """获取自主目标列表

    参数:
        status_filter: 状态过滤（active, pending, completed, failed, all）
        priority_filter: 优先级过滤（CRITICAL, HIGH, MEDIUM, LOW, BACKGROUND）
        limit: 返回数量限制
    """
    try:
        # 获取所有目标
        goals = list(manager.goals.values())

        # 应用状态过滤
        if status_filter and status_filter.lower() != "all":
            goals = [g for g in goals if g.status.lower() == status_filter.lower()]

        # 应用优先级过滤
        if priority_filter:
            goals = [g for g in goals if g.priority.name == priority_filter.upper()]

        # 按创建时间排序（最新的在前）
        goals.sort(key=lambda g: g.created_at, reverse=True)

        # 限制数量
        goals = goals[:limit]

        # 转换为响应格式
        goals_response = [goal.to_dict() for goal in goals]

        return {
            "success": True,
            "message": "获取目标列表成功",
            "count": len(goals_response),
            "total": len(manager.goals),
            "goals": goals_response,
            "filters": {
                "status": status_filter,
                "priority": priority_filter,
                "limit": limit,
            },
        }

    except Exception as e:
        logger.error(f"获取目标列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取目标列表失败: {str(e)}",
        )


@router.post("/goals/{goal_id}/activate", summary="激活目标")
async def activate_goal(goal_id: str, manager: Any = Depends(get_autonomous_manager)):
    """激活目标"""
    try:
        success = manager.activate_goal(goal_id)

        if success:
            logger.info(f"目标激活成功: {goal_id}")
            return {
                "success": True,
                "message": "目标激活成功",
                "goal_id": goal_id,
            }
        else:
            logger.error(f"目标激活失败: {goal_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="目标激活失败"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"目标激活失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"目标激活失败: {str(e)}",
        )


@router.post("/goals/{goal_id}/complete", summary="完成目标")
async def complete_goal(
    goal_id: str,
    result: Optional[Dict[str, Any]] = None,
    manager: Any = Depends(get_autonomous_manager),
):
    """完成目标"""
    try:
        success = manager.complete_goal(goal_id, result)

        if success:
            logger.info(f"目标完成成功: {goal_id}")
            return {
                "success": True,
                "message": "目标完成成功",
                "goal_id": goal_id,
            }
        else:
            logger.error(f"目标完成失败: {goal_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="目标完成失败"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"目标完成失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"目标完成失败: {str(e)}",
        )


@router.post("/decision", response_model=DecisionResponse, summary="制定决策")
async def make_decision(
    request: DecisionRequest,
    decision_engine: Any = Depends(get_decision_engine_dependency),
):
    """制定自主决策"""
    try:
        # 创建环境状态
        env_state = DecisionEnvironmentState(
            sensor_readings=request.sensor_data or {},
            system_metrics=request.system_metrics or {},
            task_progress=request.task_progress or {},
        )

        # 制定决策
        decision = decision_engine.make_decision(env_state)

        if not decision:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="决策制定失败（可能由于风险或置信度阈值）",
            )

        logger.info(f"决策制定成功: {decision.id} - {decision.action}")

        return decision.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"决策制定失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"决策制定失败: {str(e)}",
        )


@router.post("/decision/{decision_id}/result", summary="更新决策结果")
async def update_decision_result(
    decision_id: str,
    success: bool,
    actual_reward: float,
    decision_engine: Any = Depends(get_decision_engine_dependency),
):
    """更新决策结果"""
    try:
        # 更新决策结果
        result_success = decision_engine.update_decision_result(
            decision_id=decision_id,
            success=success,
            actual_reward=actual_reward,
        )

        if result_success:
            logger.info(f"决策结果更新成功: {decision_id}")
            return {
                "success": True,
                "message": "决策结果更新成功",
                "decision_id": decision_id,
            }
        else:
            logger.error(f"决策结果更新失败: {decision_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="决策结果更新失败",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"决策结果更新失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"决策结果更新失败: {str(e)}",
        )


@router.get("/decision/history", summary="获取决策历史")
async def get_decision_history(
    decision_engine: Any = Depends(get_decision_engine_dependency),
    limit: int = 50,
):
    """获取决策历史"""
    try:
        history = decision_engine.get_decision_history(limit)

        return {
            "success": True,
            "message": "获取决策历史成功",
            "count": len(history),
            "limit": limit,
            "history": history,
        }

    except Exception as e:
        logger.error(f"获取决策历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取决策历史失败: {str(e)}",
        )


@router.get(
    "/decision/statistics",
    response_model=DecisionStatisticsResponse,
    summary="获取决策统计",
)
async def get_decision_statistics(
    decision_engine: Any = Depends(get_decision_engine_dependency),
):
    """获取决策统计信息"""
    try:
        stats = decision_engine.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"获取决策统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取决策统计失败: {str(e)}",
        )


@router.post("/decision/reset", summary="重置决策引擎")
async def reset_decision_engine(
    decision_engine: Any = Depends(get_decision_engine_dependency),
):
    """重置决策引擎"""
    try:
        decision_engine.reset()

        logger.info("决策引擎重置成功")
        return {
            "success": True,
            "message": "决策引擎重置成功",
        }

    except Exception as e:
        logger.error(f"决策引擎重置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"决策引擎重置失败: {str(e)}",
        )


# 健康检查端点
@router.get("/health", summary="模式系统健康检查")
async def mode_health_check(
    manager: Any = Depends(get_autonomous_manager),
    decision_engine: Any = Depends(get_decision_engine_dependency),
):
    """模式系统健康检查"""
    try:
        # 检查自主模式管理器
        manager_status = {
            "autonomous_manager_available": AUTONOMOUS_MODE_AVAILABLE,
            "manager_running": (
                manager.running if hasattr(manager, "running") else False
            ),
            "current_state": (
                manager.current_state.name
                if hasattr(manager, "current_state")
                else "unknown"
            ),
        }

        # 检查决策引擎
        decision_engine_status = {
            "decision_engine_available": DECISION_ENGINE_AVAILABLE,
            "memory_size": (
                len(decision_engine.memory) if hasattr(decision_engine, "memory") else 0
            ),
            "decision_history_size": (
                len(decision_engine.decision_history)
                if hasattr(decision_engine, "decision_history")
                else 0
            ),
        }

        # 综合健康状态
        healthy = (
            AUTONOMOUS_MODE_AVAILABLE
            and DECISION_ENGINE_AVAILABLE
            and (not manager.running or manager.current_state != AutonomousState.ERROR)
        )

        return {
            "success": True,
            "healthy": healthy,
            "timestamp": datetime.now().isoformat(),
            "autonomous_manager": manager_status,
            "decision_engine": decision_engine_status,
        }

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "success": False,
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# 完整实现（兼容前端）
@router.post("/enable", summary="启用自主模式")
async def enable_autonomous_mode(manager: Any = Depends(get_autonomous_manager)):
    """启用自主模式"""
    try:
        success = manager.start_autonomous_mode()
        if success:
            logger.info("自主模式启用成功")
            return {
                "success": True,
                "message": "自主模式启用成功",
                "current_mode": "autonomous",
            }
        else:
            logger.error("自主模式启用失败")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="自主模式启用失败",
            )
    except Exception as e:
        logger.error(f"启用自主模式失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启用自主模式失败: {str(e)}",
        )


@router.post("/disable", summary="禁用自主模式")
async def disable_autonomous_mode(manager: Any = Depends(get_autonomous_manager)):
    """禁用自主模式"""
    try:
        success = manager.stop_autonomous_mode()
        if success:
            logger.info("自主模式禁用成功")
            return {
                "success": True,
                "message": "自主模式禁用成功",
                "current_mode": "task",
            }
        else:
            logger.error("自主模式禁用失败")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="自主模式禁用失败",
            )
    except Exception as e:
        logger.error(f"禁用自主模式失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"禁用自主模式失败: {str(e)}",
        )


@router.post("/start", summary="启动自主模式运行")
async def start_autonomous_running(manager: Any = Depends(get_autonomous_manager)):
    """启动自主模式运行"""
    try:
        success = manager.start_autonomous_mode()
        if success:
            logger.info("自主模式启动成功")
            return {
                "success": True,
                "message": "自主模式启动成功",
                "current_mode": "autonomous",
            }
        else:
            logger.error("自主模式启动失败")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="自主模式启动失败",
            )
    except Exception as e:
        logger.error(f"启动自主模式失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动自主模式失败: {str(e)}",
        )


@router.post("/stop", summary="停止自主模式运行")
async def stop_autonomous_running(manager: Any = Depends(get_autonomous_manager)):
    """停止自主模式运行"""
    try:
        success = manager.stop_autonomous_mode()
        if success:
            logger.info("自主模式停止成功")
            return {
                "success": True,
                "message": "自主模式停止成功",
                "current_mode": "task",
            }
        else:
            logger.error("自主模式停止失败")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="自主模式停止失败",
            )
    except Exception as e:
        logger.error(f"停止自主模式失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止自主模式失败: {str(e)}",
        )


@router.get("/export", summary="导出自主模式数据")
async def export_autonomous_data(manager: Any = Depends(get_autonomous_manager)):
    """导出自主模式数据"""
    try:
        # 获取所有数据
        status_info = manager.get_current_status()
        goals = list(manager.goals.values())

        # 准备导出数据
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "status": status_info,
            "goals": [goal.to_dict() for goal in goals],
            "config": manager.config,
            "statistics": {
                "total_goals": len(goals),
                "active_goals": len([g for g in goals if g.status == "active"]),
                "completed_goals": len([g for g in goals if g.status == "completed"]),
                "failed_goals": len([g for g in goals if g.status == "failed"]),
            },
        }

        logger.info("自主模式数据导出成功")
        return {
            "success": True,
            "message": "自主模式数据导出成功",
            "data": export_data,
            "export_timestamp": export_data["export_timestamp"],
        }
    except Exception as e:
        logger.error(f"导出自主模式数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出自主模式数据失败: {str(e)}",
        )


# 启动时初始化
@router.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("自主模式路由模块启动初始化")

    # 初始化自主模式管理器
    if AUTONOMOUS_MODE_AVAILABLE:
        try:
            get_autonomous_mode_manager()
            logger.info("自主模式管理器初始化完成")
        except Exception as e:
            logger.error(f"自主模式管理器初始化失败: {e}")

    # 初始化决策引擎
    if DECISION_ENGINE_AVAILABLE:
        try:
            get_decision_engine()
            logger.info("决策引擎初始化完成")
        except Exception as e:
            logger.error(f"决策引擎初始化失败: {e}")


__all__ = ["router"]
