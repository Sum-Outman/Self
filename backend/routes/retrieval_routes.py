"""
跨模态检索路由模块
提供文本到图像、图像到文本等跨模态检索API
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import base64
import io

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.retrieval_service import get_retrieval_service

router = APIRouter(prefix="/api/retrieval", tags=["跨模态检索"])


@router.post("/text-to-image", response_model=Dict[str, Any])
async def retrieve_images_by_text(
    query_text: str = Form(..., description="查询文本"),
    image_files: List[UploadFile] = File(..., description="候选图像文件列表"),
    top_k: int = Form(5, description="返回的Top-K结果数量", ge=1, le=20),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """文本到图像检索：根据文本查询查找最相关的图像"""
    try:
        if not query_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查询文本不能为空"
            )
        
        if not image_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="至少需要一个候选图像文件"
            )
        
        # 读取候选图像文件
        image_candidates = []
        for image_file in image_files:
            if not image_file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"文件 {image_file.filename} 必须是图像类型"
                )
            
            image_content = await image_file.read()
            await image_file.seek(0)  # 重置文件指针
            image_candidates.append(image_content)
        
        # 获取检索服务
        retrieval_service = get_retrieval_service()
        
        # 执行检索
        result = retrieval_service.retrieve_image_by_text(
            query_text=query_text,
            image_candidates=image_candidates,
            top_k=top_k
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "检索服务不可用")
            )
        
        # 构建响应
        response_results = []
        for i, item in enumerate(result["results"]):
            # 获取对应的图像文件信息
            image_index = item["image_index"]
            image_file = image_files[image_index]
            
            # 将图像转换为Base64用于显示（前3个结果）
            image_base64 = None
            if i < 3:  # 只转换前3个结果以避免响应过大
                image_content = image_candidates[image_index]
                image_base64 = base64.b64encode(image_content).decode('utf-8')
            
            response_results.append({
                "rank": item["rank"],
                "similarity": item["similarity"],
                "filename": image_file.filename,
                "content_type": image_file.content_type,
                "size": item["image_size"],
                "image_preview": image_base64 if image_base64 else None,
            })
        
        return {
            "success": True,
            "query": {
                "text": query_text,
                "type": "text",
            },
            "target": {
                "type": "image",
                "count": len(image_files),
            },
            "results": response_results,
            "retrieval_metadata": {
                "top_k": result["top_k"],
                "total_candidates": result["total_candidates"],
                "query_type": result["query_type"],
                "target_type": result["target_type"],
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文本到图像检索失败: {str(e)}"
        )


@router.post("/image-to-text", response_model=Dict[str, Any])
async def retrieve_texts_by_image(
    query_image: UploadFile = File(..., description="查询图像文件"),
    candidate_texts: List[str] = Form(..., description="候选文本列表"),
    top_k: int = Form(5, description="返回的Top-K结果数量", ge=1, le=20),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """图像到文本检索：根据图像查询查找最相关的文本"""
    try:
        # 检查查询图像
        if not query_image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查询文件必须是图像类型"
            )
        
        if not candidate_texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="至少需要一个候选文本"
            )
        
        # 读取查询图像
        query_image_content = await query_image.read()
        await query_image.seek(0)
        
        # 获取检索服务
        retrieval_service = get_retrieval_service()
        
        # 执行检索
        result = retrieval_service.retrieve_text_by_image(
            image_bytes=query_image_content,
            text_candidates=candidate_texts,
            top_k=top_k
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "检索服务不可用")
            )
        
        # 构建响应
        response_results = []
        for item in result["results"]:
            response_results.append({
                "rank": item["rank"],
                "similarity": item["similarity"],
                "text": item["text"],
                "text_preview": item["text"][:200] + ("..." if len(item["text"]) > 200 else ""),
            })
        
        # 将查询图像转换为Base64用于显示
        query_image_base64 = base64.b64encode(query_image_content).decode('utf-8')
        
        return {
            "success": True,
            "query": {
                "type": "image",
                "filename": query_image.filename,
                "content_type": query_image.content_type,
                "size": len(query_image_content),
                "preview": query_image_base64,
            },
            "target": {
                "type": "text",
                "count": len(candidate_texts),
            },
            "results": response_results,
            "retrieval_metadata": {
                "top_k": result["top_k"],
                "total_candidates": result["total_candidates"],
                "query_type": result["query_type"],
                "target_type": result["target_type"],
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像到文本检索失败: {str(e)}"
        )


@router.post("/cross-modal-similarity", response_model=Dict[str, Any])
async def compute_cross_modal_similarity(
    modality_a: str = Form(..., description="模态A类型: text, image", pattern="^(text|image)$"),
    content_a: str = Form(..., description="模态A内容：文本或Base64编码的图像"),
    modality_b: str = Form(..., description="模态B类型: text, image", pattern="^(text|image)$"),
    content_b: str = Form(..., description="模态B内容：文本或Base64编码的图像"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """计算跨模态相似度"""
    try:
        # 验证模态类型
        valid_modalities = ["text", "image"]
        if modality_a not in valid_modalities or modality_b not in valid_modalities:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"模态类型必须是: {valid_modalities}"
            )
        
        # 处理内容A
        if modality_a == "text":
            content_a_processed = content_a
        else:  # image
            # 解码Base64图像
            try:
                # 移除可能的data:image前缀
                if "," in content_a:
                    content_a = content_a.split(",")[1]
                
                content_a_processed = base64.b64decode(content_a)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"模态A内容Base64解码失败: {str(e)}"
                )
        
        # 处理内容B
        if modality_b == "text":
            content_b_processed = content_b
        else:  # image
            try:
                if "," in content_b:
                    content_b = content_b.split(",")[1]
                
                content_b_processed = base64.b64decode(content_b)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"模态B内容Base64解码失败: {str(e)}"
                )
        
        # 获取检索服务
        retrieval_service = get_retrieval_service()
        
        # 计算相似度
        result = retrieval_service.cross_modal_similarity(
            modality_a=modality_a,
            content_a=content_a_processed,
            modality_b=modality_b,
            content_b=content_b_processed
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "相似度计算失败")
            )
        
        return {
            "success": True,
            "modality_pair": result["modality_pair"],
            "similarity": result["similarity"],
            "interpretation": result["interpretation"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"跨模态相似度计算失败: {str(e)}"
        )


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_retrieval_capabilities(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取检索服务能力信息"""
    try:
        retrieval_service = get_retrieval_service()
        service_info = retrieval_service.get_service_info()
        
        capabilities = {
            "text_to_image": {
                "enabled": service_info["capabilities"]["text_to_image"],
                "description": "根据文本查询检索相关图像",
                "max_candidates": 100,
                "max_top_k": 20,
            },
            "image_to_text": {
                "enabled": service_info["capabilities"]["image_to_text"],
                "description": "根据图像查询检索相关文本",
                "max_candidates": 100,
                "max_top_k": 20,
            },
            "cross_modal_similarity": {
                "enabled": service_info["capabilities"]["cross_modal_similarity"],
                "description": "计算跨模态内容相似度",
                "supported_modality_pairs": ["text-image", "image-text"],
            },
            "supported_modalities": service_info["capabilities"]["supported_modalities"],
            "service_status": service_info["status"],
            "model_info": {
                "contrastive_model_loaded": service_info["contrastive_model_loaded"],
                "text_encoder_loaded": service_info["text_encoder_loaded"],
                "image_encoder_loaded": service_info["image_encoder_loaded"],
                "device": service_info["device"],
            },
        }
        
        return {
            "success": True,
            "capabilities": capabilities,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取检索能力信息失败: {str(e)}"
        )


@router.post("/batch-similarity", response_model=Dict[str, Any])
async def compute_batch_cross_modal_similarity(
    query_modality: str = Form(..., description="查询模态类型: text, image", pattern="^(text|image)$"),
    query_content: str = Form(..., description="查询内容：文本或Base64编码的图像"),
    target_modality: str = Form(..., description="目标模态类型: text, image", pattern="^(text|image)$"),
    target_contents: List[str] = Form(..., description="目标内容列表"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """批量计算跨模态相似度"""
    try:
        if not target_contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="至少需要一个目标内容"
            )
        
        # 获取检索服务
        retrieval_service = get_retrieval_service()
        
        # 处理查询内容
        if query_modality == "text":
            query_processed = query_content
        else:  # image
            try:
                if "," in query_content:
                    query_content = query_content.split(",")[1]
                query_processed = base64.b64decode(query_content)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"查询内容Base64解码失败: {str(e)}"
                )
        
        # 处理目标内容
        target_processed_list = []
        for i, target in enumerate(target_contents):
            if target_modality == "text":
                target_processed_list.append(target)
            else:  # image
                try:
                    if "," in target:
                        target = target.split(",")[1]
                    target_processed = base64.b64decode(target)
                    target_processed_list.append(target_processed)
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"目标内容#{i+1} Base64解码失败: {str(e)}"
                    )
        
        # 批量计算相似度
        similarities = []
        for i, target_processed in enumerate(target_processed_list):
            result = retrieval_service.cross_modal_similarity(
                modality_a=query_modality,
                content_a=query_processed,
                modality_b=target_modality,
                content_b=target_processed
            )
            
            if result.get("success", False):
                similarities.append({
                    "index": i,
                    "similarity": result["similarity"],
                    "interpretation": result["interpretation"],
                })
            else:
                similarities.append({
                    "index": i,
                    "similarity": 0.0,
                    "interpretation": "计算失败",
                    "error": result.get("error", "未知错误"),
                })
        
        # 排序结果
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "success": True,
            "query_modality": query_modality,
            "target_modality": target_modality,
            "total_targets": len(target_contents),
            "similarities": similarities,
            "top_similarity": similarities[0]["similarity"] if similarities else 0.0,
            "average_similarity": sum(s["similarity"] for s in similarities) / len(similarities) if similarities else 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量相似度计算失败: {str(e)}"
        )