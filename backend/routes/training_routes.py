"""
训练路由模块
处理训练任务、数据集、GPU状态和模型类型的API请求
"""

import sys
import os

# 添加项目根目录到Python路径（当作为脚本直接运行时）
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import uuid
import random

from backend.dependencies import get_db, get_current_user

from backend.db_models.user import User

from backend.services.training_service import get_training_service

from backend.schemas.response import SuccessResponse, ErrorResponse, PaginatedResponse

router = APIRouter(prefix="/api/training", tags=["训练"])


@router.get("/jobs", response_model=SuccessResponse[List[Dict[str, Any]]])
async def get_training_jobs(
    status: Optional[str] = Query(None, description="任务状态"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取训练任务列表 - 使用真实训练服务"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 获取训练任务列表
        jobs = training_service.get_training_jobs(status)
        
        return SuccessResponse.create(
            data=jobs,
            message="获取训练任务列表成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取训练任务列表失败: {str(e)}"
        )


@router.get("/stats", response_model=SuccessResponse[Dict[str, Any]])
async def get_training_stats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取训练统计 - 使用真实训练服务"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 获取训练统计
        stats = training_service.get_training_stats()
        
        return SuccessResponse.create(
            data=stats,
            message="获取训练统计成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取训练统计失败: {str(e)}"
        )


@router.get("/datasets", response_model=SuccessResponse[List[Dict[str, Any]]])
async def get_datasets(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取数据集列表 - 使用真实训练服务"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 获取数据集列表
        datasets = training_service.get_datasets()
        
        return SuccessResponse.create(
            data=datasets,
            message="获取数据集列表成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取数据集列表失败: {str(e)}"
        )


@router.get("/gpu-status", response_model=SuccessResponse[Dict[str, Any]])
async def get_gpu_status(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取GPU状态 - 使用真实训练服务"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 获取GPU状态
        gpu_status = training_service.get_gpu_status()
        
        return SuccessResponse.create(
            data=gpu_status,
            message="获取GPU状态成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取GPU状态失败: {str(e)}"
        )


@router.get("/model-types", response_model=SuccessResponse[List[Dict[str, Any]]])
async def get_model_types(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取模型类型 - 使用真实训练服务"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 获取模型类型
        model_types = training_service.get_model_types()
        
        return SuccessResponse.create(
            data=model_types,
            message="获取模型类型成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型类型失败: {str(e)}"
        )


@router.post("/jobs", response_model=SuccessResponse[Dict[str, Any]])
async def create_training_job(
    training_request: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """创建训练任务 - 使用真实训练服务"""
    try:
        name = training_request.get("name", "")
        
        if not name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="任务名称不能为空"
            )
        
        # 获取训练服务
        training_service = get_training_service()
        
        # 创建训练任务
        result = training_service.create_training_job(training_request)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "创建训练任务失败")
            )
        
        return SuccessResponse.create(
            data=result.get("job", {}),
            message=result.get("message", "训练任务创建成功")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建训练任务失败: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=SuccessResponse[Dict[str, Any]])
async def get_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取单个训练任务 - 使用真实训练服务"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 获取训练任务
        job_data = training_service.get_training_job(job_id)
        
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"训练任务 {job_id} 不存在"
            )
        
        return SuccessResponse.create(
            data=job_data,
            message="获取训练任务成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取训练任务失败: {str(e)}"
        )


@router.put("/jobs/{job_id}", response_model=SuccessResponse[Dict[str, Any]])
async def update_training_job(
    job_id: str,
    updates: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """更新训练任务 - 使用真实训练服务"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 更新训练任务
        result = training_service.update_training_job(job_id, updates)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "更新训练任务失败")
            )
        
        return SuccessResponse.create(
            data=result,
            message=result.get("message", "训练任务更新成功")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新训练任务失败: {str(e)}"
        )


@router.delete("/jobs/{job_id}", response_model=SuccessResponse[Dict[str, Any]])
async def delete_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """删除训练任务 - 使用真实训练服务"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 删除训练任务
        result = training_service.delete_training_job(job_id)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "删除训练任务失败")
            )
        
        return SuccessResponse.create(
            data=result,
            message=result.get("message", "训练任务删除成功")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除训练任务失败: {str(e)}"
        )


@router.post("/jobs/{job_id}/pause", response_model=SuccessResponse[Dict[str, Any]])
async def pause_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """暂停训练任务"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 暂停训练任务
        result = training_service.pause_training_job(job_id)
        
        if result.get("success"):
            return SuccessResponse.create(
                data={
                    "id": job_id,
                    "status": "paused",
                },
                message=result.get("message", f"训练任务 {job_id} 已暂停")
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "暂停训练任务失败")
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"暂停训练任务失败: {str(e)}"
        )


@router.post("/jobs/{job_id}/resume", response_model=SuccessResponse[Dict[str, Any]])
async def resume_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """恢复训练任务"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 恢复训练任务
        result = training_service.resume_training_job(job_id)
        
        if result.get("success"):
            return SuccessResponse.create(
                data={
                    "id": job_id,
                    "status": "running",
                },
                message=result.get("message", f"训练任务 {job_id} 已恢复")
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "恢复训练任务失败")
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"恢复训练任务失败: {str(e)}"
        )


@router.post("/jobs/{job_id}/stop", response_model=SuccessResponse[Dict[str, Any]])
async def stop_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """停止训练任务"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 停止训练任务
        result = training_service.stop_training_job(job_id)
        
        if result.get("success"):
            return SuccessResponse.create(
                data={
                    "id": job_id,
                    "status": "stopped",  # 注意：training_service内部将状态设置为"failed"
                },
                message=result.get("message", f"训练任务 {job_id} 已停止")
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "停止训练任务失败")
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止训练任务失败: {str(e)}"
        )


@router.get("/jobs/{job_id}/logs", response_model=SuccessResponse[List[str]])
async def get_training_job_logs(
    job_id: str,
    limit: int = Query(100, ge=1, le=1000, description="日志条数限制"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取训练任务日志"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 获取训练任务日志
        logs = training_service.get_training_job_logs(job_id, limit)
        
        return SuccessResponse.create(
            data=logs,
            message="获取训练任务日志成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取训练任务日志失败: {str(e)}"
        )


@router.post("/datasets/upload", response_model=SuccessResponse[Dict[str, Any]])
async def upload_dataset(
    file: UploadFile = File(..., description="上传的数据集文件"),
    metadata: Optional[str] = Query(None, description="元数据JSON字符串"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """上传数据集"""
    try:
        # 获取训练服务
        training_service = get_training_service()
        
        # 读取文件内容
        file_content = await file.read()
        
        # 解析元数据
        metadata_dict = {}
        if metadata:
            import json
            try:
                metadata_dict = json.loads(metadata)
            except Exception:
                metadata_dict = {"description": metadata}
        
        # 上传数据集
        result = training_service.upload_dataset(
            file_content=file_content,
            filename=file.filename,
            metadata=metadata_dict
        )
        
        if result.get("success"):
            return SuccessResponse.create(
                data=result.get("dataset", {}),
                message=result.get("message", "数据集上传成功")
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "数据集上传失败")
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传数据集失败: {str(e)}"
        )


@router.post("/hyperparameter-optimization", response_model=SuccessResponse[Dict[str, Any]])
async def hyperparameter_optimization(
    optimization_request: Dict[str, Any] = Body(..., description="超参数优化请求"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """执行超参数优化
    
    请求体格式:
    {
        "model_type": "transformer",
        "dataset_id": "dataset_001",
        "hyperparameter_space": {
            "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2],
            "batch_size": [16, 32, 64, 128],
            "optimizer": ["adam", "sgd", "rmsprop"],
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "optimization_method": "bayesian",
        "num_trials": 10,
        "objective_metric": "accuracy"
    }
    """
    try:
        # 验证必需字段
        required_fields = ["model_type", "dataset_id", "hyperparameter_space"]
        for field in required_fields:
            if field not in optimization_request:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"缺少必需字段: {field}"
                )
        
        # 获取训练服务
        training_service = get_training_service()
        
        # 提取参数
        model_type = optimization_request["model_type"]
        dataset_id = optimization_request["dataset_id"]
        hyperparameter_space = optimization_request["hyperparameter_space"]
        optimization_method = optimization_request.get("optimization_method", "bayesian")
        num_trials = optimization_request.get("num_trials", 10)
        objective_metric = optimization_request.get("objective_metric", "accuracy")
        
        # 验证超参数空间格式
        if not isinstance(hyperparameter_space, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"超参数空间必须是字典格式，当前类型: {type(hyperparameter_space)}"
            )
        
        # 验证每个参数的值列表
        for param_name, param_values in hyperparameter_space.items():
            if not isinstance(param_values, list):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"超参数 '{param_name}' 的值必须是列表格式"
                )
            if len(param_values) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"超参数 '{param_name}' 的值列表不能为空"
                )
        
        # 执行超参数优化
        result = training_service.hyperparameter_optimization(
            model_type=model_type,
            dataset_id=dataset_id,
            hyperparameter_space=hyperparameter_space,
            optimization_method=optimization_method,
            num_trials=num_trials,
            objective_metric=objective_metric
        )
        
        return SuccessResponse.create(
            data=result,
            message="超参数优化执行成功"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"超参数优化失败: {str(e)}"
        )


# 注意：还有其他端点如GET /training/jobs/{job_id}, PUT /training/jobs/{job_id}, 
# DELETE /training/jobs/{job_id}, GET /training/jobs/{job_id}/logs,
# POST /training/jobs/{job_id}/pause, POST /training/jobs/{job_id}/resume,
# 完整处理
# 实际项目中应该实现完整的训练管理功能