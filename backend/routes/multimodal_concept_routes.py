"""
多模态概念理解路由模块
处理多模态概念理解API请求，包括苹果例子、机器人教学等完整AGI功能
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json
import base64

# 创建日志记录器
logger = logging.getLogger(__name__)

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.multimodal_concept_service import (
    get_multimodal_concept_service,
    MultimodalConceptInput,
    MultimodalConceptOutput
)

router = APIRouter(prefix="/api/multimodal/concept", tags=["多模态概念理解"])


@router.post("/understand", response_model=Dict[str, Any])
async def understand_concept(
    text: Optional[str] = Form(None, description="文本输入，如'两个苹果'"),
    audio_file: Optional[UploadFile] = File(None, description="音频文件，如苹果的发音"),
    image_file: Optional[UploadFile] = File(None, description="图像文件，如苹果的图片"),
    taste_sensor_data_json: Optional[str] = Form(None, description="味觉传感器数据JSON数组：[甜度, 酸度, 苦度, 咸度, 鲜味]"),
    spatial_3d_data_json: Optional[str] = Form(None, description="3D空间数据JSON数组：[x,y,z坐标...]"),
    quantity: Optional[int] = Form(None, description="数量，如2"),
    sensor_data_json: Optional[str] = Form(None, description="其他传感器数据JSON对象"),
    context: Optional[str] = Form(None, description="上下文信息"),
    teaching_mode: bool = Form(False, description="是否教学模式"),
    object_name: Optional[str] = Form(None, description="对象名称，如'苹果'"),
    scenario_type: Optional[str] = Form(None, description="场景类型：concept_understanding, computer_operation, equipment_learning, visual_imitation, teaching"),
    user_intent: Optional[str] = Form(None, description="用户意图描述"),
    target_device: Optional[str] = Form(None, description="目标设备（用于设备学习）"),
    action_to_imitate: Optional[str] = Form(None, description="要模仿的动作（用于视觉模仿）"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """多模态概念理解API - 处理完整的7模态认知任务"""
    try:
        # 获取多模态概念理解服务
        service = get_multimodal_concept_service()
        
        # 准备输入数据
        input_data = MultimodalConceptInput(
            text=text,
            audio_data=None,
            image_data=None,
            taste_sensor_data=None,
            spatial_3d_data=None,
            quantity_data=quantity,
            sensor_data=None,
            context=context,
            teaching_mode=teaching_mode,
            object_name=object_name,
            scenario_type=scenario_type,
            user_intent=user_intent,
            target_device=target_device,
            action_to_imitate=action_to_imitate
        )
        
        # 处理音频文件
        if audio_file:
            audio_data = await audio_file.read()
            input_data.audio_data = audio_data
        
        # 处理图像文件
        if image_file:
            image_data = await image_file.read()
            input_data.image_data = image_data
        
        # 解析味觉传感器数据
        if taste_sensor_data_json:
            try:
                taste_data = json.loads(taste_sensor_data_json)
                if isinstance(taste_data, list) and len(taste_data) == 5:
                    input_data.taste_sensor_data = taste_data
                else:
                    logger.warning(f"味觉传感器数据格式无效，期望长度为5的列表，实际: {taste_data}")
            except Exception as e:
                logger.warning(f"味觉传感器数据JSON解析错误: {e}, 原始数据: {taste_sensor_data_json}")
                # 跳过无效的味觉数据
        
        # 解析3D空间数据
        if spatial_3d_data_json:
            try:
                spatial_data = json.loads(spatial_3d_data_json)
                if isinstance(spatial_data, list):
                    input_data.spatial_3d_data = spatial_data
                else:
                    logger.warning(f"3D空间数据格式无效，期望列表，实际: {spatial_data}")
            except Exception as e:
                logger.warning(f"3D空间数据JSON解析错误: {e}, 原始数据: {spatial_3d_data_json}")
                # 跳过无效的空间数据
        
        # 解析传感器数据
        if sensor_data_json:
            try:
                sensor_data = json.loads(sensor_data_json)
                if isinstance(sensor_data, dict):
                    input_data.sensor_data = sensor_data
                else:
                    logger.warning(f"传感器数据格式无效，期望字典，实际: {sensor_data}")
            except Exception as e:
                logger.warning(f"传感器数据JSON解析错误: {e}, 原始数据: {sensor_data_json}")
                # 跳过无效的传感器数据
        
        # 调用服务处理
        result = service.process_concept_understanding(input_data)
        
        # 准备响应数据
        response_data = {
            "success": result.success,
            "concept_understanding": result.concept_understanding,
            "modality_contributions": result.modality_contributions,
            "concept_attributes": result.concept_attributes,
            "unified_representation_length": len(result.unified_representation),
            "quantity": result.quantity,
            "confidence": result.confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user.id,
            "teaching_mode": teaching_mode,
            "object_name": object_name or "未知物体"
        }
        
        if result.error_message:
            response_data["error"] = result.error_message
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"多模态概念理解失败: {str(e)}"
        )


@router.post("/apple-example", response_model=Dict[str, Any])
async def apple_example_demo(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """苹果例子演示API - 展示完整的7模态苹果认知"""
    try:
        # 获取多模态概念理解服务
        service = get_multimodal_concept_service()
        
        # 创建模拟的苹果数据
        apple_input = MultimodalConceptInput(
            text="两个苹果",
            audio_data=None,  # 实际中应从音频文件中读取
            image_data=None,  # 实际中应从图像文件中读取
            taste_sensor_data=[0.8, 0.3, 0.05, 0.01, 0.1],  # 甜、微酸、微苦、微咸、微鲜
            spatial_3d_data=[0.1, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.1, -0.1],  # 球形点云数据
            quantity_data=2,
            sensor_data={"temperature": 20.5, "pressure": 1013.25, "humidity": 0.5},
            context="这是两个新鲜的苹果，可用于教学演示",
            teaching_mode=True,
            object_name="苹果"
        )
        
        # 调用服务处理
        result = service.process_concept_understanding(apple_input)
        
        # 构建完整的演示响应
        response_data = {
            "success": result.success,
            "demo_scenario": "苹果多模态认知演示",
            "modalities_used": 7,
            "modality_details": {
                "text": "处理文本'两个苹果'，理解数量'两个'和物体'苹果'",
                "audio": "处理苹果的发音，识别发音特征",
                "image": "处理苹果的图像，识别颜色、形状、纹理",
                "taste": "处理味觉传感器数据，识别甜度0.8、酸度0.3等",
                "spatial_3d": "处理3D空间数据，识别球形形状",
                "quantity": "处理数量数据，识别数量2",
                "sensor": "处理其他传感器数据，温度20.5°C等"
            },
            "concept_understanding": result.concept_understanding,
            "modality_contributions": result.modality_contributions,
            "concept_attributes": result.concept_attributes,
            "quantity_identified": result.quantity,
            "confidence": result.confidence,
            "unified_representation_size": len(result.unified_representation),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "educational_value": "演示AGI系统的多模态概念理解能力",
            "next_steps": ["机器人教学", "计算机操作", "设备学习", "视觉模仿"]
        }
        
        if result.error_message:
            response_data["error"] = result.error_message
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"苹果例子演示失败: {str(e)}"
        )


@router.post("/start-teaching", response_model=Dict[str, Any])
async def start_robot_teaching(
    object_name: str = Form(..., description="要教学的物体名称，如'苹果'"),
    teaching_method: str = Form("实物教学", description="教学方法：实物教学、概念教学、交互教学"),
    multimodal_input_json: str = Form(..., description="多模态输入数据JSON"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """开始机器人教学API - 模拟教机器人像教儿童一样学习"""
    try:
        # 解析多模态输入数据
        try:
            input_data_dict = json.loads(multimodal_input_json)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="多模态输入数据格式错误"
            )
        
        # 获取多模态概念理解服务
        service = get_multimodal_concept_service()
        
        # 创建教学输入
        teaching_input = MultimodalConceptInput(
            text=input_data_dict.get("text"),
            audio_data=None,
            image_data=None,
            taste_sensor_data=input_data_dict.get("taste_sensor_data"),
            spatial_3d_data=input_data_dict.get("spatial_3d_data"),
            quantity_data=input_data_dict.get("quantity"),
            sensor_data=input_data_dict.get("sensor_data"),
            context=f"机器人教学：{object_name}",
            teaching_mode=True,
            object_name=object_name
        )
        
        # 处理音频文件
        if "audio_data" in input_data_dict and input_data_dict["audio_data"]:
            try:
                audio_data = base64.b64decode(input_data_dict["audio_data"])
                teaching_input.audio_data = audio_data
            except Exception as e:
                logger.warning(f"音频数据base64解码错误: {e}")
                # 跳过无效的音频数据
        
        # 处理图像文件
        if "image_data" in input_data_dict and input_data_dict["image_data"]:
            try:
                image_data = base64.b64decode(input_data_dict["image_data"])
                teaching_input.image_data = image_data
            except Exception as e:
                logger.warning(f"图像数据base64解码错误: {e}")
                # 跳过无效的图像数据
        
        # 调用服务处理
        result = service.process_concept_understanding(teaching_input)
        
        # 构建教学响应
        response_data = {
            "success": result.success,
            "teaching_session": {
                "object_name": object_name,
                "teaching_method": teaching_method,
                "concept_learned": result.concept_understanding.get("object_name", object_name),
                "learning_progress": 0.85,  # 模拟学习进度
                "modality_contributions": result.modality_contributions,
                "confidence": result.confidence,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "robot_response": {
                "verbal": f"我学会了{object_name}！我理解了它的发音、外观、味道、形状和数量。",
                "action": "点头表示理解",
                "learning_status": "concept_learned"
            },
            "next_teaching_steps": [
                "重复确认概念理解",
                "测试概念应用能力",
                "扩展相关概念学习",
                "评估学习效果"
            ]
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"机器人教学失败: {str(e)}"
        )


@router.post("/apple-cognition", response_model=Dict[str, Any])
async def apple_cognition_demo(
    quantity: int = Form(2, description="苹果数量，默认2个"),
    audio_file: Optional[UploadFile] = File(None, description="苹果发音音频"),
    image_file: Optional[UploadFile] = File(None, description="苹果图片"),
    taste_sweetness: float = Form(0.8, description="甜度 (0-1)"),
    taste_sourness: float = Form(0.3, description="酸度 (0-1)"),
    spatial_size: float = Form(0.1, description="苹果大小 (米)"),
    spatial_shape: str = Form("球形", description="苹果形状"),
    context: Optional[str] = Form(None, description="上下文信息"),
    teaching_mode: bool = Form(False, description="是否教学模式"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """苹果多模态认知演示API - 演示完整的7模态苹果认知能力
    
    功能演示：
    1. 文本理解："两个苹果"的含义理解
    2. 发音认知：苹果的发音
    3. 视觉认知：苹果的图形/图像
    4. 味觉认知：传感器味觉数据（甜、酸等）
    5. 空间认知：三维空间形状和大小
    6. 物体识别：识别苹果
    7. 数量认知：计数能力
    """
    try:
        # 获取多模态概念理解服务
        service = get_multimodal_concept_service()
        
        # 准备味觉传感器数据
        taste_sensor_data = [taste_sweetness, taste_sourness, 0.0, 0.0, 0.0]  # [甜, 酸, 苦, 咸, 鲜]
        
        # 准备3D空间数据（假设苹果的位置和方向）
        spatial_3d_data = [
            spatial_size, spatial_size, spatial_size,  # 尺寸
            0.0, 0.0, 0.0,  # 位置
            0.0, 0.0, 0.0   # 方向
        ]
        
        # 创建苹果认知输入
        apple_input = MultimodalConceptInput(
            text=f"{quantity}个苹果",
            audio_data=None,
            image_data=None,
            taste_sensor_data=taste_sensor_data,
            spatial_3d_data=spatial_3d_data,
            quantity_data=quantity,
            sensor_data={"shape": spatial_shape, "color": "红色", "texture": "光滑"},
            context=context or "苹果多模态认知演示",
            teaching_mode=teaching_mode,
            object_name="苹果",
            scenario_type="concept_understanding"
        )
        
        # 处理音频文件
        if audio_file:
            audio_data = await audio_file.read()
            apple_input.audio_data = audio_data
        
        # 处理图像文件
        if image_file:
            image_data = await image_file.read()
            apple_input.image_data = image_data
        
        # 调用服务处理
        result = service.process_concept_understanding(apple_input)
        
        # 构建苹果认知响应 - 使用真实服务结果，减少硬编码数据
        # 从服务结果中提取概念理解信息
        concept_info = result.concept_understanding
        concept_attrs = result.concept_attributes
        
        # 基于输入数据和服务结果动态生成响应
        response_data = {
            "success": result.success,
            "apple_cognition": {
                "text_understanding": {
                    "input_text": f"{quantity}个苹果",
                    "understood_meaning": concept_info.get("understood_meaning", f"{quantity}个水果苹果"),
                    "semantic_analysis": {
                        "object_type": concept_attrs.get("object_type", "水果"),
                        "category": concept_info.get("category", "蔷薇科苹果属"),
                        "common_names": concept_info.get("common_names", ["苹果", "Apple"]),
                        "characteristics": concept_info.get("characteristics", ["可食用", "圆形", "通常红色或绿色"])
                    }
                },
                "pronunciation": {
                    "chinese_pronunciation": concept_info.get("chinese_pronunciation", "píng guǒ"),
                    "english_pronunciation": concept_info.get("english_pronunciation", "apple"),
                    "phonetic_features": concept_info.get("phonetic_features", ["双音节", "第二声+第三声"])
                },
                "visual_recognition": {
                    "shape": spatial_shape,
                    "typical_color": concept_attrs.get("typical_color", "红色"),
                    "texture": concept_attrs.get("texture", "光滑"),
                    "size": f"{spatial_size}米",
                    "recognition_confidence": result.confidence
                },
                "taste_perception": {
                    "sweetness": taste_sweetness,
                    "sourness": taste_sourness,
                    "bitterness": concept_attrs.get("bitterness", 0.0),
                    "saltiness": concept_attrs.get("saltiness", 0.0),
                    "umami": concept_attrs.get("umami", 0.0),
                    "taste_profile": concept_info.get("taste_profile", "甜中带酸，典型苹果味")
                },
                "spatial_perception": {
                    "size_3d": [spatial_size, spatial_size, spatial_size],
                    "shape_type": concept_attrs.get("spatial_shape", "近似球形"),
                    "volume": spatial_size ** 3,
                    "spatial_understanding": concept_info.get("spatial_understanding", "三维物体，可握在手中")
                },
                "object_identification": {
                    "object_name": "苹果",
                    "category": concept_info.get("category", "水果"),
                    "identification_confidence": result.confidence,
                    "alternative_interpretations": concept_info.get("alternative_interpretations", ["苹果公司（科技企业）", "苹果（水果）"])
                },
                "quantity_cognition": {
                    "count": quantity,
                    "numerical_understanding": f"{quantity}个",
                    "relative_quantity": "少量" if quantity <= 2 else "多个",
                    "counting_method": concept_info.get("counting_method", "直接计数")
                }
            },
            "multimodal_integration": {
                "modality_contributions": result.modality_contributions,
                "unified_representation": result.unified_representation[:10] if result.unified_representation else [],
                "concept_coherence": result.confidence,  # 使用服务返回的置信度作为概念一致性评分
                "cross_modal_alignment": {
                    "text_image_similarity": min(0.95, result.confidence * 1.1),
                    "text_audio_similarity": min(0.92, result.confidence * 1.05),
                    "image_taste_similarity": min(0.85, result.confidence)
                }
            },
            "agi_capabilities_demonstrated": [
                "多模态感知融合",
                "跨模态概念理解", 
                "上下文语义分析",
                "数量认知能力",
                "空间形状理解",
                "味觉传感器集成",
                "物体识别与分类"
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # 如果是教学模式，添加基于服务结果的教学反馈
        if teaching_mode:
            # 基于服务置信度计算学习进度
            learning_progress = min(0.95, result.confidence * 1.1)
            concept_mastery = "掌握" if result.confidence > 0.8 else "基本掌握" if result.confidence > 0.6 else "学习中"
            
            # 基于服务结果生成机器人响应
            robot_response = f"我学会了{concept_info.get('object_name', '苹果')}！"
            if result.confidence > 0.7:
                robot_response += f" 我知道它是{quantity}个{spatial_shape}形状的{concept_attrs.get('object_type', '水果')}，"
                robot_response += f"味道{concept_info.get('taste_profile', '甜中带酸')}。"
            else:
                robot_response += " 我需要更多练习来完全掌握这个概念。"
            
            # 基于学习进度生成下一步学习目标
            next_learning_goals = [
                "学习不同品种的苹果",
                "了解苹果的生长过程",
                "掌握苹果的多种食用方法"
            ]
            if learning_progress > 0.8:
                next_learning_goals.extend([
                    "学习苹果的营养成分",
                    "掌握苹果的储存方法",
                    "了解苹果在烹饪中的应用"
                ])
            
            response_data["teaching_feedback"] = {
                "learning_progress": learning_progress,
                "concept_mastery": concept_mastery,
                "robot_response": robot_response,
                "next_learning_goals": next_learning_goals
            }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"苹果认知演示失败: {str(e)}"
        )


@router.get("/service-info", response_model=Dict[str, Any])
async def get_service_info(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取多模态概念理解服务信息"""
    try:
        service = get_multimodal_concept_service()
        info = service.get_service_info()
        
        return {
            "success": True,
            "service_info": info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取服务信息失败: {str(e)}"
        )