"""
硬件训练路由模块
提供硬件训练的标准API接口，包括任务管理、状态监控和流程控制
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime, timedelta, timezone

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.hardware_training_manager import (
    get_hardware_training_manager,
    TrainingDeviceType,
    TrainingPhase,
    DeviceConfig,
    TrainingTask,
    TrainingResult,
    HardwareTrainingError,
)

router = APIRouter(prefix="/api/hardware-training", tags=["硬件训练"])


@router.get("/tasks", response_model=Dict[str, Any])
async def get_training_tasks(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取所有训练任务"""
    try:
        manager = get_hardware_training_manager()
        tasks = manager.get_all_tasks()

        return {
            "success": True,
            "data": tasks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取训练任务失败: {str(e)}",
        )


@router.post("/tasks", response_model=Dict[str, Any])
async def create_training_task(
    task_data: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """创建训练任务"""
    try:
        # 验证任务数据
        if "task_id" not in task_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="任务ID不能为空"
            )

        # 创建设备配置
        device_configs = []
        for device_data in task_data.get("device_configs", []):
            device_config = DeviceConfig(
                device_id=device_data["device_id"],
                device_type=TrainingDeviceType(device_data["device_type"]),
                parameters=device_data.get("parameters", {}),
                calibration_data=device_data.get("calibration_data"),
                safety_limits=device_data.get("safety_limits", {}),
                connection_info=device_data.get("connection_info", {}),
            )
            device_configs.append(device_config)

        # 创建训练任务
        task = TrainingTask(
            task_id=task_data["task_id"],
            name=task_data.get("name", "未命名任务"),
            description=task_data.get("description", ""),
            device_configs=device_configs,
            training_parameters=task_data.get("training_parameters", {}),
            phase_sequence=[
                TrainingPhase(phase)
                for phase in task_data.get(
                    "phase_sequence",
                    [
                        "initialization",
                        "calibration",
                        "warmup",
                        "training",
                        "cooldown",
                        "validation",
                        "completion",
                    ],
                )
            ],
            expected_duration=timedelta(
                seconds=task_data.get("expected_duration", 3600)
            ),
            priority=task_data.get("priority", 100),
        )

        # 添加到管理器
        manager = get_hardware_training_manager()
        task_id = manager.create_task(task)

        return {
            "success": True,
            "task_id": task_id,
            "message": "训练任务创建成功",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"参数验证失败: {str(e)}"
        )
    except HardwareTrainingError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"训练任务创建失败: {e.message}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建训练任务失败: {str(e)}",
        )


@router.post("/tasks/{task_id}/start", response_model=Dict[str, Any])
async def start_training_task(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """开始训练任务"""
    try:
        manager = get_hardware_training_manager()
        success = manager.start_task(task_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无法开始训练任务: {task_id}",
            )

        return {
            "success": True,
            "task_id": task_id,
            "message": "训练任务开始执行",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HardwareTrainingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"开始训练任务失败: {e.message}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"开始训练任务失败: {str(e)}",
        )


@router.post("/tasks/{task_id}/pause", response_model=Dict[str, Any])
async def pause_training_task(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """暂停训练任务"""
    try:
        manager = get_hardware_training_manager()
        success = manager.pause_task(task_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无法暂停训练任务: {task_id}",
            )

        return {
            "success": True,
            "task_id": task_id,
            "message": "训练任务已暂停",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"暂停训练任务失败: {str(e)}",
        )


@router.post("/tasks/{task_id}/resume", response_model=Dict[str, Any])
async def resume_training_task(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """恢复训练任务"""
    try:
        manager = get_hardware_training_manager()
        success = manager.resume_task(task_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无法恢复训练任务: {task_id}",
            )

        return {
            "success": True,
            "task_id": task_id,
            "message": "训练任务已恢复",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"恢复训练任务失败: {str(e)}",
        )


@router.post("/tasks/{task_id}/cancel", response_model=Dict[str, Any])
async def cancel_training_task(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """取消训练任务"""
    try:
        manager = get_hardware_training_manager()
        success = manager.cancel_task(task_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无法取消训练任务: {task_id}",
            )

        return {
            "success": True,
            "task_id": task_id,
            "message": "训练任务已取消",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取消训练任务失败: {str(e)}",
        )


@router.get("/tasks/{task_id}/progress", response_model=Dict[str, Any])
async def get_training_progress(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取训练任务进度"""
    try:
        manager = get_hardware_training_manager()
        progress = manager.get_task_progress(task_id)

        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务进度未找到: {task_id}",
            )

        return {
            "success": True,
            "data": progress.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务进度失败: {str(e)}",
        )


@router.get("/tasks/{task_id}/result", response_model=Dict[str, Any])
async def get_training_result(
    task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取训练任务结果"""
    try:
        manager = get_hardware_training_manager()
        result = manager.get_task_result(task_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务结果未找到: {task_id}",
            )

        return {
            "success": True,
            "data": result.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务结果失败: {str(e)}",
        )


@router.get("/device-types", response_model=Dict[str, Any])
async def get_device_types(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取支持的设备类型"""
    try:
        device_types = [
            {
                "value": device_type.value,
                "label": device_type.name.replace("_", " ").title(),
                "description": self._get_device_type_description(device_type),
            }
            for device_type in TrainingDeviceType
        ]

        return {
            "success": True,
            "data": device_types,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取设备类型失败: {str(e)}",
        )


def _get_device_type_description(device_type: TrainingDeviceType) -> str:
    """获取设备类型描述"""
    descriptions = {
        TrainingDeviceType.ROBOT_ARM: "机器人手臂，用于精确的运动控制和操作",
        TrainingDeviceType.ROBOT_GRIPPER: "机器人抓手，用于物体的抓取和释放",
        TrainingDeviceType.MOBILE_ROBOT: "移动机器人，用于自主导航和环境探索",
        TrainingDeviceType.SENSOR_ARRAY: "传感器阵列，用于环境感知和数据采集",
        TrainingDeviceType.CAMERA_SYSTEM: "相机系统，用于视觉感知和图像采集",
        TrainingDeviceType.LIDAR_SYSTEM: "激光雷达系统，用于三维环境建模和测距",
        TrainingDeviceType.FORCE_SENSOR: "力传感器，用于力觉感知和力控制",
        TrainingDeviceType.JOINT_CONTROLLER: "关节控制器，用于机器人关节的精确控制",
    }

    return descriptions.get(device_type, "未知设备类型")


@router.get("/training-phases", response_model=Dict[str, Any])
async def get_training_phases(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取训练阶段"""
    try:
        training_phases = [
            {
                "value": phase.value,
                "label": phase.name.replace("_", " ").title(),
                "description": self._get_phase_description(phase),
                "order": index,
            }
            for index, phase in enumerate(TrainingPhase)
        ]

        return {
            "success": True,
            "data": training_phases,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取训练阶段失败: {str(e)}",
        )


def _get_phase_description(phase: TrainingPhase) -> str:
    """获取训练阶段描述"""
    descriptions = {
        TrainingPhase.INITIALIZATION: "设备初始化和连接阶段",
        TrainingPhase.CALIBRATION: "设备校准和参数调整阶段",
        TrainingPhase.WARMUP: "设备热身和准备工作阶段",
        TrainingPhase.TRAINING: "主要训练执行阶段",
        TrainingPhase.COOLDOWN: "设备冷却和清理阶段",
        TrainingPhase.VALIDATION: "训练结果验证和评估阶段",
        TrainingPhase.COMPLETION: "任务完成和设备断开阶段",
        TrainingPhase.ERROR: "错误处理和恢复阶段",
    }

    return descriptions.get(phase, "未知阶段")


@router.websocket("/ws/{task_id}/progress")
async def websocket_progress(
    websocket: WebSocket,
    task_id: str,
    db: Session = Depends(get_db),
):
    """WebSocket连接，实时获取训练进度"""
    await websocket.accept()

    try:
        manager = get_hardware_training_manager()

        # 注册进度回调
        async def progress_callback(
            task_id_cb: str, phase: TrainingPhase, event: str, args: tuple
        ):
            if task_id_cb == task_id:
                await websocket.send_json(
                    {
                        "type": "progress",
                        "task_id": task_id_cb,
                        "phase": phase.value,
                        "event": event,
                        "args": args,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        # 注册阶段完成回调
        async def phase_complete_callback(
            task_id_cb: str, phase: TrainingPhase, success: bool, result: Dict[str, Any]
        ):
            if task_id_cb == task_id:
                await websocket.send_json(
                    {
                        "type": "phase_complete",
                        "task_id": task_id_cb,
                        "phase": phase.value,
                        "success": success,
                        "result": result,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        # 注册任务完成回调
        async def task_complete_callback(task_id_cb: str, result: TrainingResult):
            if task_id_cb == task_id:
                await websocket.send_json(
                    {
                        "type": "task_complete",
                        "task_id": task_id_cb,
                        "result": result.to_dict(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        # 完整处理
        # 实际项目中需要适配async回调

        # 定期发送进度更新
        while True:
            try:
                # 获取当前进度
                progress = manager.get_task_progress(task_id)

                if progress:
                    await websocket.send_json(
                        {
                            "type": "status_update",
                            "task_id": task_id,
                            "progress": progress.to_dict(),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                # 等待1秒
                await asyncio.sleep(1)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket错误: {e}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                break

    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}")
    finally:
        await websocket.close()


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """健康检查"""
    try:
        manager = get_hardware_training_manager()

        return {
            "success": True,
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {
                "active_task": manager.active_task_id,
                "is_running": manager.is_running,
                "total_tasks": len(manager.tasks),
                "device_count": len(manager.device_manager.get_all_devices()),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"健康检查失败: {str(e)}",
        )


# 添加全局异常处理
@router.exception_handler(HardwareTrainingError)
async def hardware_training_error_handler(request, exc: HardwareTrainingError):
    """硬件训练错误处理"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error": exc.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


logger = logging.getLogger(__name__)
