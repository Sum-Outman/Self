"""
AGI系统路由模块
处理AGI状态、训练控制、模式切换等API请求
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import logging
import asyncio

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.schemas.response import SuccessResponse

# 尝试导入AGI服务
try:
    from backend.services.agi_service import get_agi_service

    AGI_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AGI服务导入失败: {e}")
    AGI_SERVICE_AVAILABLE = False

# 尝试导入系统服务
try:
    from backend.services.system_service import get_system_service

    SYSTEM_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"系统服务导入失败: {e}")
    SYSTEM_SERVICE_AVAILABLE = False

# 尝试导入训练服务
try:
    from backend.services.training_service import get_training_service

    TRAINING_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"训练服务导入失败: {e}")
    TRAINING_SERVICE_AVAILABLE = False

router = APIRouter(prefix="/api/agi", tags=["AGI系统"])

logger = logging.getLogger(__name__)


@router.get("/status", response_model=SuccessResponse)
async def get_agi_status(
    db: Session = Depends(get_db),
):
    """获取AGI系统状态"""
    try:
        # 尝试使用AGI服务
        if AGI_SERVICE_AVAILABLE:
            agi_service = get_agi_service()
            try:
                # 在线程池中执行同步方法，避免阻塞事件循环，设置5秒超时
                status_data = await asyncio.wait_for(
                    asyncio.to_thread(agi_service.get_status), timeout=5.0
                )
                return SuccessResponse.create(
                    data=status_data, message="获取AGI状态成功"
                )
            except asyncio.TimeoutError:
                logger.warning("获取AGI状态超时，返回基本状态")
                # 超时时返回基本状态数据（使用下方定义的备用状态）

        # 如果AGI服务不可用，使用系统服务和训练服务组合数据
        status_data = {
            "status": "idle",
            "mode": "task",
            "trainingProgress": 0.0,
            "reasoningDepth": 0,
            "memoryUsage": 0.0,
            "hardwareConnected": False,
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "activeModels": [],
            "trainingEpoch": 0,
            "totalEpochs": 0,
            "learningRate": 0.0,
            "batchSize": 0,
            "datasetSize": 0,
            "gpuUsage": 0.0,
            "cpuUsage": 0.0,
            "systemMemory": 0.0,
            "networkStatus": "online",
        }

        # 如果系统服务可用，获取真实数据
        if SYSTEM_SERVICE_AVAILABLE:
            system_service = get_system_service()
            system_status = system_service.get_system_status()

            if system_status and "hardware" in system_status:
                status_data["hardwareConnected"] = system_status["hardware"].get(
                    "connected", False
                )

            if system_status and "performance" in system_status:
                perf = system_status["performance"]
                status_data["gpuUsage"] = perf.get("gpu_usage", 0.0)
                status_data["cpuUsage"] = perf.get("cpu_usage", 0.0)
                status_data["systemMemory"] = perf.get("memory_used", 0.0)

        # 如果训练服务可用，获取训练进度
        if TRAINING_SERVICE_AVAILABLE:
            try:
                training_service = get_training_service()
                jobs = training_service.get_training_jobs("running")
                if jobs:
                    # 假设第一个运行中的任务是AGI训练
                    job = jobs[0]
                    status_data["status"] = "training"
                    status_data["trainingProgress"] = job.get("progress", 0.0)
                    status_data["trainingEpoch"] = job.get("current_epoch", 0)
                    status_data["totalEpochs"] = job.get("epochs", 0)
                    status_data["batchSize"] = job.get("batch_size", 0)

                    # 从配置中提取学习率
                    config = job.get("config", {})
                    if isinstance(config, dict):
                        status_data["learningRate"] = config.get("learning_rate", 0.001)
                        status_data["datasetSize"] = config.get("dataset_size", 0)
            except Exception as e:
                logger.warning(f"获取训练进度失败: {e}")

        return SuccessResponse.create(data=status_data, message="获取AGI状态成功")
    except Exception as e:
        logger.error(f"获取AGI状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取AGI状态失败: {str(e)}",
        )


@router.post("/training/start", response_model=SuccessResponse)
async def start_agi_training(
    config: Optional[Dict[str, Any]] = Body(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """启动AGI训练"""
    try:
        # 如果训练服务可用，启动训练任务
        if TRAINING_SERVICE_AVAILABLE:
            training_service = get_training_service()

            # 构建训练配置
            training_config = config or {
                "model_type": "transformer",
                "dataset_path": "default",
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "use_gpu": True,
                "save_checkpoints": True,
                "early_stopping": True,
                "validation_split": 0.2,
            }

            # 创建训练任务
            job_id = training_service.create_training_job(
                name=f"AGI训练-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                description="AGI系统训练任务",
                config=training_config,
                user_id=user.id,
            )

            return SuccessResponse.create(
                data={"jobId": job_id}, message="AGI训练启动成功"
            )
        else:
            # 训练服务不可用，抛出明确错误
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="训练服务不可用，无法启动AGI训练",
            )
    except Exception as e:
        logger.error(f"启动AGI训练失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动AGI训练失败: {str(e)}",
        )


@router.post("/training/stop", response_model=SuccessResponse)
async def stop_agi_training(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """停止AGI训练"""
    try:
        # 如果训练服务可用，停止训练任务
        if TRAINING_SERVICE_AVAILABLE:
            training_service = get_training_service()

            # 获取运行中的AGI训练任务
            jobs = training_service.get_training_jobs("running")
            for job in jobs:
                # 检查是否是AGI训练任务（根据名称或配置判断）
                job_name = job.get("name", "")
                if "AGI" in job_name or "agi" in job_name:
                    training_service.stop_training_job(job["id"])

            return SuccessResponse.create(message="AGI训练停止成功")
        else:
            # 训练服务不可用，返回模拟响应
            return SuccessResponse.create(message="AGI训练停止成功（模拟模式）")
    except Exception as e:
        logger.error(f"停止AGI训练失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止AGI训练失败: {str(e)}",
        )


@router.get("/training/progress", response_model=SuccessResponse)
async def get_agi_training_progress(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取AGI训练进度"""
    try:
        # 如果训练服务可用，获取训练进度
        if TRAINING_SERVICE_AVAILABLE:
            training_service = get_training_service()

            # 获取运行中的训练任务
            jobs = training_service.get_training_jobs("running")
            if jobs:
                # 假设第一个运行中的任务是AGI训练
                job = jobs[0]
                progress_data = {
                    "epoch": job.get("current_epoch", 0),
                    "totalEpochs": job.get("epochs", 100),
                    "progress": job.get("progress", 0.0),
                    "loss": job.get("current_loss", 0.0),
                    "accuracy": job.get("current_accuracy", 0.0),
                    "learningRate": job.get("learning_rate", 0.001),
                    "batchSize": job.get("batch_size", 32),
                    "samplesProcessed": job.get("samples_processed", 0),
                    "totalSamples": job.get("total_samples", 10000),
                    "timeElapsed": job.get("time_elapsed", 0),
                    "estimatedTimeRemaining": job.get("estimated_time_remaining", 0),
                    "currentPhase": "forward",
                    "gpuMemoryUsage": job.get("gpu_memory_usage", 0.0),
                    "cpuMemoryUsage": job.get("cpu_memory_usage", 0.0),
                }

                # 根据进度确定当前阶段
                progress = job.get("progress", 0.0)
                if progress < 30:
                    progress_data["currentPhase"] = "forward"
                elif progress < 60:
                    progress_data["currentPhase"] = "backward"
                elif progress < 90:
                    progress_data["currentPhase"] = "optimization"
                elif progress < 95:
                    progress_data["currentPhase"] = "validation"
                else:
                    progress_data["currentPhase"] = "checkpointing"

                return SuccessResponse.create(
                    data=progress_data, message="获取AGI训练进度成功"
                )

        # 如果没有运行中的训练任务或服务不可用，返回默认进度
        default_progress = {
            "epoch": 0,
            "totalEpochs": 100,
            "progress": 0.0,
            "loss": 0.0,
            "accuracy": 0.0,
            "learningRate": 0.001,
            "batchSize": 32,
            "samplesProcessed": 0,
            "totalSamples": 10000,
            "timeElapsed": 0,
            "estimatedTimeRemaining": 0,
            "currentPhase": "forward",
            "gpuMemoryUsage": 0.0,
            "cpuMemoryUsage": 0.0,
        }

        return SuccessResponse.create(
            data=default_progress, message="获取AGI训练进度成功"
        )
    except Exception as e:
        logger.error(f"获取AGI训练进度失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取AGI训练进度失败: {str(e)}",
        )


@router.post("/mode", response_model=SuccessResponse)
async def change_agi_mode(
    mode_request: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
):
    """切换AGI模式"""
    try:
        mode = mode_request.get("mode", "task")
        valid_modes = ["autonomous", "task", "demo"]

        if mode not in valid_modes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的模式: {mode}。有效模式: {valid_modes}",
            )

        # 如果系统服务可用，设置系统模式
        if SYSTEM_SERVICE_AVAILABLE:
            system_service = get_system_service()
            system_service.set_system_mode(mode)

        # 记录模式切换
        logger.info(f"用户 anonymous 切换AGI模式为: {mode}")

        return SuccessResponse.create(message=f"AGI模式切换为 {mode} 成功")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"切换AGI模式失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"切换AGI模式失败: {str(e)}",
        )


@router.post("/pause", response_model=SuccessResponse)
async def pause_agi(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """暂停AGI系统"""
    try:
        # 如果系统服务可用，暂停系统
        if SYSTEM_SERVICE_AVAILABLE:
            system_service = get_system_service()
            system_service.pause_system()

        # 如果训练服务可用，暂停训练
        if TRAINING_SERVICE_AVAILABLE:
            training_service = get_training_service()
            jobs = training_service.get_training_jobs("running")
            for job in jobs:
                if "AGI" in job.get("name", ""):
                    training_service.pause_training_job(job["id"])

        logger.info(f"用户 {user.username} 暂停了AGI系统")

        return SuccessResponse.create(message="AGI系统暂停成功")
    except Exception as e:
        logger.error(f"暂停AGI系统失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"暂停AGI系统失败: {str(e)}",
        )


@router.post("/resume", response_model=SuccessResponse)
async def resume_agi(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """恢复AGI系统"""
    try:
        # 如果系统服务可用，恢复系统
        if SYSTEM_SERVICE_AVAILABLE:
            system_service = get_system_service()
            system_service.resume_system()

        # 如果训练服务可用，恢复训练
        if TRAINING_SERVICE_AVAILABLE:
            training_service = get_training_service()
            jobs = training_service.get_training_jobs("paused")
            for job in jobs:
                if "AGI" in job.get("name", ""):
                    training_service.resume_training_job(job["id"])

        logger.info(f"用户 {user.username} 恢复了AGI系统")

        return SuccessResponse.create(message="AGI系统恢复成功")
    except Exception as e:
        logger.error(f"恢复AGI系统失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"恢复AGI系统失败: {str(e)}",
        )


@router.get("/logs", response_model=SuccessResponse)
async def get_agi_logs(
    lines: int = Query(100, ge=1, le=1000, description="日志行数"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取AGI系统日志"""
    try:
        # 读取日志文件
        log_file_path = "logs/agi_system.log"
        logs = []

        try:
            with open(log_file_path, "r", encoding="utf-8") as f:
                all_logs = f.readlines()
                # 获取最后lines行
                logs = all_logs[-lines:] if len(all_logs) > lines else all_logs
                # 清理每行日志
                logs = [log.strip() for log in logs if log.strip()]
        except FileNotFoundError:
            # 如果日志文件不存在，生成模拟日志
            logs = [
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - AGI系统启动",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 系统模式: task",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 硬件连接状态: 未连接",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 用户 {user.username} 登录系统",
            ]

        return SuccessResponse.create(data=logs, message="获取AGI系统日志成功")
    except Exception as e:
        logger.error(f"获取AGI系统日志失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取AGI系统日志失败: {str(e)}",
        )


@router.get("/stats", response_model=SuccessResponse)
async def get_agi_stats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取AGI系统统计"""
    try:
        # 如果训练服务可用，获取训练统计
        total_training_time = 0
        total_reasoning_time = 0
        total_learning_sessions = 0
        models_trained = 0
        average_loss = 0.0
        average_accuracy = 0.0

        if TRAINING_SERVICE_AVAILABLE:
            training_service = get_training_service()
            training_stats = training_service.get_training_stats()

            if training_stats:
                total_training_time = training_stats.get("total_training_time", 0)
                models_trained = training_stats.get("completed_jobs", 0)

        # 如果系统服务可用，获取系统统计
        hardware_uptime = 0
        system_uptime = 0
        error_count = 0
        warning_count = 0

        if SYSTEM_SERVICE_AVAILABLE:
            system_service = get_system_service()
            system_status = system_service.get_system_status()

            if system_status:
                hardware_uptime = system_status.get("hardware_uptime", 0)
                system_uptime = system_status.get("system_uptime", 0)
                error_count = system_status.get("error_count", 0)
                warning_count = system_status.get("warning_count", 0)

        stats_data = {
            "totalTrainingTime": total_training_time,
            "totalReasoningTime": total_reasoning_time,
            "totalLearningSessions": total_learning_sessions,
            "modelsTrained": models_trained,
            "averageLoss": average_loss,
            "averageAccuracy": average_accuracy,
            "hardwareUptime": hardware_uptime,
            "systemUptime": system_uptime,
            "errorCount": error_count,
            "warningCount": warning_count,
        }

        return SuccessResponse.create(data=stats_data, message="获取AGI系统统计成功")
    except Exception as e:
        logger.error(f"获取AGI系统统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取AGI系统统计失败: {str(e)}",
        )
