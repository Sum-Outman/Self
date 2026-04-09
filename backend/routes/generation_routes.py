"""
跨模态生成路由模块
提供文本到图像、图像到文本等跨模态生成API
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import base64

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.generation_service import get_generation_service

router = APIRouter(prefix="/api/generation", tags=["跨模态生成"])


@router.post("/text-to-image", response_model=Dict[str, Any])
async def generate_image_from_text(
    prompt: str = Form(..., description="生成提示文本"),
    num_images: int = Form(1, description="生成图像数量", ge=1, le=4),
    image_size: int = Form(64, description="生成图像大小", ge=32, le=256),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """文本到图像生成：根据文本提示生成图像"""
    try:
        if not prompt.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="生成提示不能为空"
            )

        # 获取生成服务
        generation_service = get_generation_service()

        # 执行生成
        result = generation_service.generate_text_to_image(
            prompt=prompt, num_images=num_images, image_size=image_size
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "生成服务不可用"),
            )

        # 构建响应
        response_images = []
        for img in result["generated_images"]:
            response_images.append(
                {
                    "index": img["index"],
                    "width": img["size"][1] if len(img["size"]) > 1 else img["size"][0],
                    "height": (
                        img["size"][0] if len(img["size"]) > 1 else img["size"][0]
                    ),
                    "format": img["format"],
                    "base64_data": img["base64"],
                    "data_url": f"data:image/{img['format'].lower()};base64,{img['base64']}",
                }
            )

        return {
            "success": True,
            "generation_type": "text_to_image",
            "prompt": prompt,
            "generated_images": response_images,
            "generation_metadata": {
                "num_requested": num_images,
                "num_generated": result["num_generated"],
                "image_size": image_size,
                "model_info": result.get("model_info", {}),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文本到图像生成失败: {str(e)}",
        )


@router.post("/image-to-text", response_model=Dict[str, Any])
async def generate_text_from_image(
    image_file: UploadFile = File(..., description="输入图像文件"),
    max_length: int = Form(50, description="生成文本最大长度", ge=10, le=200),
    temperature: float = Form(1.0, description="生成温度", ge=0.1, le=2.0),
    top_k: int = Form(50, description="Top-K采样参数", ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """图像到文本生成：根据图像生成文本描述"""
    try:
        # 检查文件类型
        if not image_file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="文件必须是图像类型"
            )

        # 读取图像
        image_content = await image_file.read()
        await image_file.seek(0)

        # 获取生成服务
        generation_service = get_generation_service()

        # 执行生成
        result = generation_service.generate_image_to_text(
            image_bytes=image_content,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "生成服务不可用"),
            )

        # 将输入图像转换为Base64用于显示
        image_base64 = base64.b64encode(image_content).decode("utf-8")

        return {
            "success": True,
            "generation_type": "image_to_text",
            "input_image": {
                "filename": image_file.filename,
                "content_type": image_file.content_type,
                "size": len(image_content),
                "preview": image_base64,
            },
            "generated_text": result["generated_text"],
            "generation_metadata": {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "image_info": result.get("image_info", {}),
                "generation_params": result.get("generation_params", {}),
                "model_info": result.get("model_info", {}),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像到文本生成失败: {str(e)}",
        )


@router.post("/cross-modal", response_model=Dict[str, Any])
async def generate_cross_modal(
    source_modality: str = Form(
        ..., description="源模态类型: text, image", pattern="^(text|image)$"
    ),
    target_modality: str = Form(
        ..., description="目标模态类型: text, image", pattern="^(text|image)$"
    ),
    source_content: str = Form(..., description="源内容：文本或Base64编码的图像"),
    generation_params: Optional[str] = Form("{}", description="生成参数JSON字符串"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """通用跨模态生成接口"""
    try:
        import json

        # 解析生成参数
        try:
            params_dict = json.loads(generation_params)
        except json.JSONDecodeError:
            params_dict = {}

        # 获取生成服务
        generation_service = get_generation_service()

        # 处理源内容
        if source_modality == "text":
            source_processed = source_content
        else:  # image
            # 解码Base64图像
            try:
                # 移除可能的data:image前缀
                if "," in source_content:
                    source_content = source_content.split(",")[1]

                source_processed = base64.b64decode(source_content)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"源内容Base64解码失败: {str(e)}",
                )

        # 执行生成
        result = generation_service.generate_cross_modal(
            source_modality=source_modality,
            source_content=source_processed,
            target_modality=target_modality,
            generation_params=params_dict,
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "生成服务不可用"),
            )

        # 构建响应
        response_data = {
            "success": True,
            "conversion": f"{source_modality}_to_{target_modality}",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"跨模态生成失败: {str(e)}",
        )


@router.post("/batch-generate", response_model=Dict[str, Any])
async def batch_generate_cross_modal(
    source_modality: str = Form(
        ..., description="源模态类型: text, image", pattern="^(text|image)$"
    ),
    target_modality: str = Form(
        ..., description="目标模态类型: text, image", pattern="^(text|image)$"
    ),
    source_contents: List[str] = Form(..., description="源内容列表"),
    generation_params: Optional[str] = Form("{}", description="生成参数JSON字符串"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """批量跨模态生成"""
    try:
        import json

        if not source_contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="至少需要一个源内容"
            )

        # 解析生成参数
        try:
            params_dict = json.loads(generation_params)
        except json.JSONDecodeError:
            params_dict = {}

        # 处理源内容
        processed_contents = []
        for i, content in enumerate(source_contents):
            if source_modality == "text":
                processed_contents.append(content)
            else:  # image
                try:
                    if "," in content:
                        content = content.split(",")[1]
                    processed = base64.b64decode(content)
                    processed_contents.append(processed)
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"源内容#{i + 1} Base64解码失败: {str(e)}",
                    )

        # 获取生成服务
        generation_service = get_generation_service()

        # 执行批量生成
        result = generation_service.batch_generate(
            source_modality=source_modality,
            source_contents=processed_contents,
            target_modality=target_modality,
            generation_params=params_dict,
        )

        # 构建响应
        response_results = []
        for item in result["results"]:
            response_results.append(
                {
                    "index": item["index"],
                    "success": item["success"],
                    "result": item.get("result", {}),
                    "error": item.get("error"),
                }
            )

        return {
            "success": result["success"],
            "conversion": f"{source_modality}_to_{target_modality}",
            "batch_summary": {
                "total": result["total"],
                "successful": result["successful"],
                "failed": result["failed"],
                "success_rate": result["successful"] / max(result["total"], 1),
            },
            "results": response_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量生成失败: {str(e)}",
        )


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_generation_capabilities(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取生成服务能力信息"""
    try:
        generation_service = get_generation_service()
        service_info = generation_service.get_service_info()

        capabilities = {
            "text_to_image": {
                "enabled": service_info["capabilities"]["text_to_image"],
                "description": "根据文本提示生成图像",
                "max_images": service_info["capabilities"]["text_to_image_params"][
                    "max_images"
                ],
                "supported_sizes": service_info["capabilities"]["text_to_image_params"][
                    "image_sizes"
                ],
                "model_architecture": service_info["model_architecture"][
                    "text_to_image"
                ],
            },
            "image_to_text": {
                "enabled": service_info["capabilities"]["image_to_text"],
                "description": "根据图像生成文本描述",
                "max_length": service_info["capabilities"]["image_to_text_params"][
                    "max_length"
                ],
                "temperature_range": service_info["capabilities"][
                    "image_to_text_params"
                ]["temperature_range"],
                "top_k_range": service_info["capabilities"]["image_to_text_params"][
                    "top_k_range"
                ],
                "model_architecture": service_info["model_architecture"][
                    "image_to_text"
                ],
            },
            "supported_modalities": service_info["capabilities"][
                "supported_modalities"
            ],
            "supported_conversions": ["text->image", "image->text"],
            "service_status": service_info["status"],
            "model_info": {
                "generation_manager_loaded": service_info["generation_manager_loaded"],
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
            detail=f"获取生成能力信息失败: {str(e)}",
        )


@router.post("/creative-variations", response_model=Dict[str, Any])
async def generate_creative_variations(
    source_type: str = Form(
        ..., description="源类型: text, image", pattern="^(text|image)$"
    ),
    source_content: str = Form(..., description="源内容"),
    num_variations: int = Form(3, description="生成变体数量", ge=1, le=10),
    variation_strength: float = Form(0.5, description="变体强度", ge=0.1, le=1.0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """生成创意变体（增强版生成）"""
    try:
        # 获取生成服务
        generation_service = get_generation_service()

        if source_type == "text":
            # 文本变体：生成多个相关图像
            result = generation_service.generate_text_to_image(
                prompt=source_content, num_images=num_variations, image_size=64
            )

            variations = []
            if result.get("success", False):
                for img in result["generated_images"]:
                    variations.append(
                        {
                            "type": "image",
                            "content": img["base64"],
                            "format": img["format"],
                            "size": img["size"],
                        }
                    )

            return {
                "success": result.get("success", False),
                "source_type": "text",
                "source_content_preview": source_content[:100]
                + ("..." if len(source_content) > 100 else ""),
                "variations": variations,
                "num_variations": len(variations),
                "generation_params": {
                    "num_variations": num_variations,
                    "variation_strength": variation_strength,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        else:  # image
            # 图像变体：生成多个文本描述
            try:
                if "," in source_content:
                    source_content = source_content.split(",")[1]
                image_bytes = base64.b64decode(source_content)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"图像Base64解码失败: {str(e)}",
                )

            # 使用不同的温度生成多个描述
            variations = []
            for i in range(num_variations):
                # 变化温度以产生不同变体
                temperature = 0.5 + (variation_strength * i * 0.2)

                result = generation_service.generate_image_to_text(
                    image_bytes=image_bytes,
                    max_length=50,
                    temperature=temperature,
                    top_k=50,
                )

                if result.get("success", False):
                    variations.append(
                        {
                            "type": "text",
                            "content": result["generated_text"],
                            "temperature": temperature,
                            "index": i,
                        }
                    )

            # 将源图像转换为Base64用于显示
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            return {
                "success": len(variations) > 0,
                "source_type": "image",
                "source_image_preview": (
                    image_base64[:100] + "..."
                    if len(image_base64) > 100
                    else image_base64
                ),
                "variations": variations,
                "num_variations": len(variations),
                "generation_params": {
                    "num_variations": num_variations,
                    "variation_strength": variation_strength,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创意变体生成失败: {str(e)}",
        )
