"""
多模态处理路由模块
处理图像、视频、文本、音频等多模态数据的API请求
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import base64
import os

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.multimodal_service import get_multimodal_service
from backend.schemas.response import SuccessResponse, ErrorResponse, PaginatedResponse

router = APIRouter(prefix="/api/multimodal", tags=["多模态处理"])


@router.post("/process/text", response_model=SuccessResponse[Dict[str, Any]])
async def process_text(
    text: str = Form(..., description="要处理的文本"),
    task: str = Form("analysis", description="处理任务类型: analysis, sentiment, summary, extract"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """处理文本数据 - 使用真实多模态服务"""
    try:
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文本内容不能为空"
            )
        
        # 获取多模态服务并处理文本
        multimodal_service = get_multimodal_service()
        result = multimodal_service.process_text(text, task)
        
        # 如果处理失败，返回错误
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "多模态服务不可用")
            )
        
        # 返回处理结果
        return SuccessResponse.create(
            data={
                "text_result": result.get("data", {}),
                "processor_info": multimodal_service.get_processor_info(),
            },
            message="文本处理成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文本处理失败: {str(e)}"
        )


@router.post("/process/image", response_model=SuccessResponse[Dict[str, Any]])
async def process_image(
    file: UploadFile = File(..., description="图像文件"),
    task: str = Form("analysis", description="处理任务类型: analysis, detect, classify, segment"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """处理图像数据 - 使用真实多模态服务"""
    try:
        # 检查文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件必须是图像类型"
            )
        
        # 读取文件内容
        file_content = await file.read()
        await file.seek(0)  # 重置文件指针
        
        # 获取多模态服务并处理图像
        multimodal_service = get_multimodal_service()
        result = multimodal_service.process_image(file_content, task)
        
        # 如果处理失败，返回错误
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "多模态服务不可用")
            )
        
        # 返回处理结果
        return SuccessResponse.create(
            data={
                "image_result": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(file_content),
                    "task": task,
                    **result.get("data", {}),
                },
                "processor_info": multimodal_service.get_processor_info(),
            },
            message="图像处理成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像处理失败: {str(e)}"
        )


@router.post("/process/audio", response_model=SuccessResponse[Dict[str, Any]])
async def process_audio(
    file: UploadFile = File(..., description="音频文件"),
    task: str = Form("analysis", description="处理任务类型: analysis, transcribe, sentiment, features"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """处理音频数据"""
    try:
        # 检查文件类型
        if not file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件必须是音频类型"
            )
        
        # 获取多模态服务
        multimodal_service = get_multimodal_service()
        
        # 读取文件内容
        file_content = await file.read()
        file_size = len(file_content)
        
        # 尝试调用多模态服务的音频处理方法
        try:
            # 尝试不同的方法名
            if hasattr(multimodal_service, 'process_audio'):
                result = multimodal_service.process_audio(file_content, task, filename=file.filename, content_type=file.content_type)
            elif hasattr(multimodal_service, 'process_audio_file'):
                result = multimodal_service.process_audio_file(file_content, task, filename=file.filename, content_type=file.content_type)
            else:
                # 如果多模态服务没有音频处理方法，返回错误而不是真实数据
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="多模态服务不支持音频处理功能"
                )
            
            # 检查处理结果
            if not result.get("success", False):
                error_msg = result.get("error", "音频处理失败")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"音频处理失败: {error_msg}"
                )
            
            # 返回处理结果
            return SuccessResponse.create(
                data={
                    "audio_result": {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size": file_size,
                        "task": task,
                        "result": result.get("data", {}),
                        "processing_time": result.get("processing_time", 0.0),
                        "model_used": result.get("model_used", "unknown"),
                    }
                },
                message="音频处理成功"
            )
            
        except HTTPException:
            raise
        except Exception as service_error:
            # 服务调用失败，返回错误而不是真实数据
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"多模态服务音频处理失败: {service_error}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"音频处理服务不可用: {str(service_error)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"音频处理失败: {str(e)}"
        )


@router.post("/process/video", response_model=SuccessResponse[Dict[str, Any]])
async def process_video(
    file: UploadFile = File(..., description="视频文件"),
    task: str = Form("analysis", description="处理任务类型: analysis, detect, summarize, extract"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """处理视频数据"""
    try:
        # 检查文件类型
        if not file.content_type.startswith("video/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件必须是视频类型"
            )
        
        # 获取多模态服务
        multimodal_service = get_multimodal_service()
        
        # 读取文件内容
        file_content = await file.read()
        file_size = len(file_content)
        
        # 尝试调用多模态服务的视频处理方法
        try:
            # 尝试不同的方法名
            if hasattr(multimodal_service, 'process_video'):
                result = multimodal_service.process_video(file_content, task, filename=file.filename, content_type=file.content_type)
            elif hasattr(multimodal_service, 'process_video_file'):
                result = multimodal_service.process_video_file(file_content, task, filename=file.filename, content_type=file.content_type)
            else:
                # 如果多模态服务没有视频处理方法，返回错误而不是真实数据
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="多模态服务不支持视频处理功能"
                )
            
            # 检查处理结果
            if not result.get("success", False):
                error_msg = result.get("error", "视频处理失败")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"视频处理失败: {error_msg}"
                )
            
            # 返回处理结果
            return SuccessResponse.create(
                data={
                    "video_result": {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size": file_size,
                        "task": task,
                        "result": result.get("data", {}),
                        "processing_time": result.get("processing_time", 0.0),
                        "model_used": result.get("model_used", "unknown"),
                    }
                },
                message="视频处理成功"
            )
            
        except HTTPException:
            raise
        except Exception as service_error:
            # 服务调用失败，返回错误而不是真实数据
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"多模态服务视频处理失败: {service_error}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"视频处理服务不可用: {str(service_error)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"视频处理失败: {str(e)}"
        )


@router.post("/process/fusion", response_model=SuccessResponse[Dict[str, Any]])
async def process_multimodal_fusion(
    text: Optional[str] = Form(None, description="文本输入"),
    image_file: Optional[UploadFile] = File(None, description="图像文件"),
    audio_file: Optional[UploadFile] = File(None, description="音频文件"),
    video_file: Optional[UploadFile] = File(None, description="视频文件"),
    fusion_type: str = Form("early", description="融合类型: early, late, hybrid"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """多模态融合处理"""
    try:
        modalities = []
        
        # 收集模态信息
        if text:
            modalities.append({"type": "text", "content": f"文本长度: {len(text)}字符"})
        
        if image_file:
            image_size = len(await image_file.read()) if image_file else 0
            await image_file.seek(0) if image_file else None
            modalities.append({"type": "image", "content": f"图像文件: {image_file.filename if image_file else 'N/A'}, 大小: {image_size}字节"})
        
        if audio_file:
            audio_size = len(await audio_file.read()) if audio_file else 0
            await audio_file.seek(0) if audio_file else None
            modalities.append({"type": "audio", "content": f"音频文件: {audio_file.filename if audio_file else 'N/A'}, 大小: {audio_size}字节"})
        
        if video_file:
            video_size = len(await video_file.read()) if video_file else 0
            await video_file.seek(0) if video_file else None
            modalities.append({"type": "video", "content": f"视频文件: {video_file.filename if video_file else 'N/A'}, 大小: {video_size}字节"})
        
        if not modalities:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="至少需要提供一个模态的输入"
            )
        
        # 获取多模态服务
        multimodal_service = get_multimodal_service()
        
        # 准备多模态输入数据
        multimodal_inputs = {}
        
        if text:
            multimodal_inputs["text"] = text
        
        # 读取文件内容
        if image_file:
            image_content = await image_file.read()
            multimodal_inputs["image"] = {
                "content": image_content,
                "filename": image_file.filename,
                "content_type": image_file.content_type
            }
        
        if audio_file:
            audio_content = await audio_file.read()
            multimodal_inputs["audio"] = {
                "content": audio_content,
                "filename": audio_file.filename,
                "content_type": audio_file.content_type
            }
        
        if video_file:
            video_content = await video_file.read()
            multimodal_inputs["video"] = {
                "content": video_content,
                "filename": video_file.filename,
                "content_type": video_file.content_type
            }
        
        # 尝试调用多模态服务的融合处理方法
        try:
            # 尝试不同的方法名
            if hasattr(multimodal_service, 'fuse_modalities'):
                result = multimodal_service.fuse_modalities(multimodal_inputs, fusion_type)
            elif hasattr(multimodal_service, 'multimodal_fusion'):
                result = multimodal_service.multimodal_fusion(multimodal_inputs, fusion_type)
            elif hasattr(multimodal_service, 'process_fusion'):
                result = multimodal_service.process_fusion(multimodal_inputs, fusion_type)
            else:
                # 如果多模态服务没有融合处理方法，返回错误而不是真实数据
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="多模态服务不支持融合处理功能"
                )
            
            # 检查处理结果
            if not result.get("success", False):
                error_msg = result.get("error", "多模态融合处理失败")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"多模态融合处理失败: {error_msg}"
                )
            
            # 返回处理结果
            return SuccessResponse.create(
                data={
                    "fusion_result": {
                        "modalities_count": len(modalities),
                        "fusion_type": fusion_type,
                        "result": result.get("data", {}),
                        "processing_time": result.get("processing_time", 0.0),
                        "model_used": result.get("model_used", "unknown"),
                    }
                },
                message="多模态融合处理成功"
            )
            
        except HTTPException:
            raise
        except Exception as service_error:
            # 服务调用失败，返回错误而不是真实数据
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"多模态服务融合处理失败: {service_error}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"多模态融合处理服务不可用: {str(service_error)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"多模态融合处理失败: {str(e)}"
        )


@router.get("/capabilities", response_model=SuccessResponse[Dict[str, Any]])
async def get_multimodal_capabilities(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取多模态处理能力"""
    try:
        # 获取多模态服务
        multimodal_service = get_multimodal_service()
        
        # 尝试从服务获取能力信息
        capabilities = {}
        
        try:
            # 尝试不同的方法名
            if hasattr(multimodal_service, 'get_capabilities'):
                service_capabilities = multimodal_service.get_capabilities()
                if service_capabilities and isinstance(service_capabilities, dict):
                    capabilities = service_capabilities
                else:
                    # 如果服务返回无效数据，使用默认配置
                    capabilities = _get_default_capabilities()
            elif hasattr(multimodal_service, 'capabilities'):
                # 如果服务有capabilities属性
                service_capabilities = multimodal_service.capabilities
                if service_capabilities and isinstance(service_capabilities, dict):
                    capabilities = service_capabilities
                else:
                    capabilities = _get_default_capabilities()
            else:
                # 如果服务没有能力获取方法，使用默认配置
                capabilities = _get_default_capabilities()
                
        except Exception as service_error:
            # 服务调用失败，使用默认配置
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"从多模态服务获取能力信息失败: {service_error}，使用默认配置")
            capabilities = _get_default_capabilities()
        
        return SuccessResponse.create(
            data={
                "capabilities": capabilities
            },
            message="获取多模态处理能力成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取多模态能力失败: {str(e)}"
        )


def _get_default_capabilities() -> Dict[str, Any]:
    """获取默认的多模态处理能力配置"""
    return {
        "text_processing": {
            "supported_tasks": ["analysis", "sentiment", "summary", "extract"],
            "max_length": 100000,
            "languages": ["zh", "en", "ja", "ko"],
        },
        "image_processing": {
            "supported_formats": ["jpg", "jpeg", "png", "gif", "bmp"],
            "max_size_mb": 50,
            "supported_tasks": ["analysis", "detect", "classify", "segment"],
        },
        "audio_processing": {
            "supported_formats": ["wav", "mp3", "ogg", "flac"],
            "max_duration_seconds": 300,
            "supported_tasks": ["analysis", "transcribe", "sentiment", "features"],
        },
        "video_processing": {
            "supported_formats": ["mp4", "avi", "mov", "mkv"],
            "max_duration_seconds": 600,
            "max_size_mb": 500,
            "supported_tasks": ["analysis", "detect", "summarize", "extract"],
        },
        "fusion_capabilities": {
            "fusion_types": ["early", "late", "hybrid"],
            "max_modalities": 4,
            "cross_modal_reasoning": True,
        },
    }