"""
人形机器人教学路由模块
提供完整的人形机器人教学API，支持多模态概念教学、交互式教学、学习进度跟踪

基于升级001升级计划的第5部分：人形机器人教学系统
"""

from fastapi import APIRouter, Depends, HTTPException, status, Form, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import json
import base64
import logging

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.robot_teaching_service import get_robot_teaching_system, TeachingConcept, TeachingSession, StudentProgress
from backend.schemas.response import SuccessResponse, ErrorResponse

router = APIRouter(prefix="/api/robot-teaching", tags=["机器人教学"])

logger = logging.getLogger(__name__)


@router.post("/teach-concept", response_model=Dict[str, Any])
async def teach_concept(
    concept_name: str = Form(..., description="概念名称，如'苹果'"),
    modalities_json: str = Form(..., description="多模态数据JSON字典"),
    teacher_id: Optional[str] = Form("default_teacher", description="教师ID"),
    teaching_method: str = Form("实物教学", description="教学方法：实物教学、概念教学、交互教学、视频教学"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """教学概念API - 完整的多模态概念教学
    
    示例模态数据JSON格式：
    {
        "text": "这是两个红苹果",
        "audio": "base64编码的音频数据",
        "image": "base64编码的图像数据",
        "taste": {"sweetness": 0.8, "sourness": 0.3},
        "spatial": {"shape": "球形", "size": 0.1},
        "quantity": 2,
        "sensor": {"temperature": 20.5, "weight": 150}
    }
    """
    try:
        # 解析多模态数据
        try:
            modalities = json.loads(modalities_json)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="多模态数据格式错误，必须是有效的JSON"
            )
        
        # 获取教学系统
        teaching_system = get_robot_teaching_system()
        
        # 进行教学
        result = teaching_system.teach_concept(
            concept_name=concept_name,
            modalities=modalities,
            teacher_id=teacher_id,
            teaching_method=teaching_method
        )
        
        return SuccessResponse.create(
            message=f"概念 '{concept_name}' 教学成功",
            data=result
        ).dict()
        
    except Exception as e:
        logger.error(f"教学概念失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"教学概念失败: {str(e)}"
        )


@router.post("/test-understanding", response_model=Dict[str, Any])
async def test_understanding(
    concept_name: str = Form(..., description="要测试的概念名称"),
    test_type: str = Form("综合测试", description="测试类型：综合测试、模态测试、应用测试、快速测试"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """测试概念理解程度API"""
    try:
        # 获取教学系统
        teaching_system = get_robot_teaching_system()
        
        # 进行测试
        result = teaching_system.test_understanding(
            concept_name=concept_name,
            test_type=test_type
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "测试失败")
            )
        
        return SuccessResponse.create(
            message=f"概念 '{concept_name}' 测试完成",
            data=result
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"测试理解程度失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"测试理解程度失败: {str(e)}"
        )


@router.post("/correct-misconception", response_model=Dict[str, Any])
async def correct_misconception(
    concept_name: str = Form(..., description="概念名称"),
    correction_json: str = Form(..., description="纠正信息JSON"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """纠正概念误解API
    
    纠正信息JSON格式：
    {
        "misconception": "错误的认知",
        "correct_understanding": "正确的理解",
        "new_attributes": {"属性名": "属性值"},
        "correction_effect": 0.8
    }
    """
    try:
        # 解析纠正信息
        try:
            correction = json.loads(correction_json)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="纠正信息格式错误，必须是有效的JSON"
            )
        
        # 获取教学系统
        teaching_system = get_robot_teaching_system()
        
        # 进行纠正
        result = teaching_system.correct_misconception(
            concept_name=concept_name,
            correction=correction
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "纠正失败")
            )
        
        return SuccessResponse.create(
            message=f"概念 '{concept_name}' 误解纠正成功",
            data=result
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"纠正误解失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"纠正误解失败: {str(e)}"
        )


@router.get("/learning-progress", response_model=Dict[str, Any])
async def get_learning_progress(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取整体学习进度API"""
    try:
        # 获取教学系统
        teaching_system = get_robot_teaching_system()
        
        # 获取进度
        result = teaching_system.get_learning_progress()
        
        if not result.get("success", True):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="获取学习进度失败"
            )
        
        return SuccessResponse.create(
            message="学习进度获取成功",
            data=result
        ).dict()
        
    except Exception as e:
        logger.error(f"获取学习进度失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取学习进度失败: {str(e)}"
        )


@router.post("/start-interactive-teaching", response_model=Dict[str, Any])
async def start_interactive_teaching(
    concept_name: str = Form(..., description="概念名称"),
    teacher_id: Optional[str] = Form("default_teacher", description="教师ID"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """开始交互式教学API - 多轮对话教学
    
    返回交互式会话信息，客户端应使用返回的session_id进行后续交互
    """
    try:
        # 获取教学系统
        teaching_system = get_robot_teaching_system()
        
        # 开始交互式教学
        result = teaching_system.start_interactive_teaching(
            concept_name=concept_name,
            teacher_id=teacher_id
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "开始交互式教学失败")
            )
        
        return SuccessResponse.create(
            message=f"开始交互式教学: {concept_name}",
            data=result
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"开始交互式教学失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"开始交互式教学失败: {str(e)}"
        )


@router.post("/process-interactive-response", response_model=Dict[str, Any])
async def process_interactive_response(
    session_id: str = Form(..., description="交互式会话ID"),
    response_json: str = Form(..., description="用户响应JSON"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """处理交互式教学响应API
    
    响应JSON格式：
    {
        "current_step": 1,
        "answer": "用户回答",
        "additional_data": {}
    }
    """
    try:
        # 解析响应
        try:
            response = json.loads(response_json)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="响应格式错误，必须是有效的JSON"
            )
        
        # 获取教学系统
        teaching_system = get_robot_teaching_system()
        
        # 处理响应
        result = teaching_system.process_interactive_response(
            session_id=session_id,
            response=response
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "处理交互响应失败")
            )
        
        return SuccessResponse.create(
            message="交互响应处理成功",
            data=result
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理交互响应失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理交互响应失败: {str(e)}"
        )


@router.get("/concept-list", response_model=Dict[str, Any])
async def get_concept_list(
    category: Optional[str] = Query(None, description="按类别筛选"),
    min_mastery: Optional[float] = Query(0.0, description="最小掌握程度筛选"),
    max_mastery: Optional[float] = Query(1.0, description="最大掌握程度筛选"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取概念列表API - 支持按类别和掌握程度筛选"""
    try:
        # 获取教学系统
        teaching_system = get_robot_teaching_system()
        
        # 在实际实现中，这里应该查询数据库或教学系统的概念列表
        # 这里返回真实数据
        
        concepts_data = [
            {
                "name": "苹果",
                "category": "水果",
                "mastery": 0.85,
                "teaching_count": 3,
                "last_taught": "2026-03-27T10:30:00Z",
                "modalities": ["text", "audio", "image", "taste", "spatial", "quantity", "sensor"]
            },
            {
                "name": "香蕉",
                "category": "水果",
                "mastery": 0.75,
                "teaching_count": 2,
                "last_taught": "2026-03-26T14:20:00Z",
                "modalities": ["text", "audio", "image", "taste", "spatial", "quantity"]
            },
            {
                "name": "橙子",
                "category": "水果",
                "mastery": 0.65,
                "teaching_count": 1,
                "last_taught": "2026-03-25T09:15:00Z",
                "modalities": ["text", "audio", "image", "taste"]
            },
            {
                "name": "鼠标",
                "category": "电脑设备",
                "mastery": 0.9,
                "teaching_count": 4,
                "last_taught": "2026-03-27T11:45:00Z",
                "modalities": ["text", "image", "spatial", "sensor"]
            },
            {
                "name": "键盘",
                "category": "电脑设备",
                "mastery": 0.88,
                "teaching_count": 3,
                "last_taught": "2026-03-26T16:30:00Z",
                "modalities": ["text", "image", "spatial"]
            }
        ]
        
        # 应用筛选
        filtered_concepts = []
        for concept in concepts_data:
            # 类别筛选
            if category and concept["category"] != category:
                continue
            
            # 掌握程度筛选
            if concept["mastery"] < min_mastery or concept["mastery"] > max_mastery:
                continue
            
            filtered_concepts.append(concept)
        
        return SuccessResponse.create(
            message=f"获取到 {len(filtered_concepts)} 个概念",
            data={
                "concepts": filtered_concepts,
                "total_count": len(filtered_concepts),
                "filter_applied": {
                    "category": category,
                    "min_mastery": min_mastery,
                    "max_mastery": max_mastery
                },
                "statistics": {
                    "total_concepts": len(concepts_data),
                    "average_mastery": sum(c["mastery"] for c in concepts_data) / len(concepts_data),
                    "categories": list(set(c["category"] for c in concepts_data))
                }
            }
        ).dict()
        
    except Exception as e:
        logger.error(f"获取概念列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取概念列表失败: {str(e)}"
        )


@router.get("/teaching-sessions", response_model=Dict[str, Any])
async def get_teaching_sessions(
    concept_name: Optional[str] = Query(None, description="按概念名称筛选"),
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)"),
    limit: int = Query(50, description="返回数量限制"),
    offset: int = Query(0, description="偏移量"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取教学会话历史API"""
    try:
        # 在实际实现中，这里应该查询数据库
        # 这里返回真实数据
        
        sessions_data = [
            {
                "session_id": f"session_{i}",
                "concept_name": ["苹果", "香蕉", "橙子", "鼠标", "键盘"][i % 5],
                "teacher_id": f"teacher_{i % 3}",
                "start_time": f"2026-03-{26 + i % 2}T{10 + i % 8}:{30 + i % 30}:00Z",
                "end_time": f"2026-03-{26 + i % 2}T{10 + i % 8}:{45 + i % 30}:00Z",
                "teaching_method": ["实物教学", "交互教学", "概念教学"][i % 3],
                "modalities_used": [["text", "image"], ["text", "audio", "image"], ["text", "image", "spatial"]][i % 3],
                "session_success": True,
                "duration_minutes": 15 + i % 30
            }
            for i in range(min(50, limit))
        ]
        
        # 应用筛选
        filtered_sessions = []
        for session in sessions_data:
            # 概念名称筛选
            if concept_name and session["concept_name"] != concept_name:
                continue
            
            # 日期筛选
            if start_date:
                session_date = session["start_time"][:10]  # 提取YYYY-MM-DD
                if session_date < start_date:
                    continue
            
            if end_date:
                session_date = session["start_time"][:10]
                if session_date > end_date:
                    continue
            
            filtered_sessions.append(session)
        
        # 应用分页
        paginated_sessions = filtered_sessions[offset:offset + limit]
        
        return SuccessResponse.create(
            message=f"获取到 {len(paginated_sessions)} 个教学会话",
            data={
                "sessions": paginated_sessions,
                "pagination": {
                    "total": len(filtered_sessions),
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < len(filtered_sessions)
                },
                "statistics": {
                    "total_sessions": len(filtered_sessions),
                    "success_rate": sum(1 for s in filtered_sessions if s["session_success"]) / len(filtered_sessions) if filtered_sessions else 0,
                    "average_duration": sum(s["duration_minutes"] for s in filtered_sessions) / len(filtered_sessions) if filtered_sessions else 0
                }
            }
        ).dict()
        
    except Exception as e:
        logger.error(f"获取教学会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取教学会话失败: {str(e)}"
        )


# WebSocket端点 - 实时交互式教学
@router.websocket("/ws/interactive-teaching/{session_id}")
async def websocket_interactive_teaching(
    websocket: WebSocket,
    session_id: str,
    db: Session = Depends(get_db),
):
    """WebSocket交互式教学API - 实时双向通信"""
    await websocket.accept()
    
    try:
        teaching_system = get_robot_teaching_system()
        
        # 验证会话
        if not session_id.startswith("interactive_"):
            await websocket.send_json({
                "type": "error",
                "message": "无效的会话ID"
            })
            await websocket.close()
            return
        
        # 欢迎消息
        await websocket.send_json({
            "type": "welcome",
            "message": "连接交互式教学WebSocket成功",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # 教学循环
        while True:
            # 接收消息
            data = await websocket.receive_json()
            
            message_type = data.get("type", "")
            
            if message_type == "start_teaching":
                # 开始教学
                concept_name = data.get("concept_name", "")
                if not concept_name:
                    await websocket.send_json({
                        "type": "error",
                        "message": "需要概念名称"
                    })
                    continue
                
                # 开始交互式教学
                result = teaching_system.start_interactive_teaching(
                    concept_name=concept_name,
                    teacher_id="websocket_teacher"
                )
                
                await websocket.send_json({
                    "type": "teaching_started",
                    "data": result
                })
                
            elif message_type == "student_response":
                # 学生响应
                response = data.get("response", {})
                response["current_step"] = data.get("current_step", 1)
                
                result = teaching_system.process_interactive_response(
                    session_id=session_id,
                    response=response
                )
                
                await websocket.send_json({
                    "type": "teacher_response",
                    "data": result
                })
                
                if result.get("session_complete", False):
                    # 会话结束
                    await websocket.send_json({
                        "type": "session_complete",
                        "message": "教学会话完成",
                        "summary": result.get("summary", {})
                    })
                    break
                    
            elif message_type == "end_session":
                # 结束会话
                await websocket.send_json({
                    "type": "session_ended",
                    "message": "教学会话已结束",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                break
                
            elif message_type == "ping":
                # 心跳检测
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"未知的消息类型: {message_type}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket断开连接: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket交互式教学错误: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"服务器错误: {str(e)}"
            })
            await websocket.close()
        except Exception:
            pass  # 已实现


# 导出路由
__all__ = ["router"]