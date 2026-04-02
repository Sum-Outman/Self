"""
视觉动作模仿路由模块
提供视觉观察模仿人类动作的API

基于升级001升级计划的第8部分：视觉动作模仿系统
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
from backend.services.visual_imitation_service import (
    get_visual_imitation_service,
    VisualImitationService,
    ImitationMode,
    BodyPart,
    ActionType
)
from backend.schemas.response import SuccessResponse, ErrorResponse

router = APIRouter(prefix="/api/visual-imitation", tags=["视觉动作模仿"])

logger = logging.getLogger(__name__)


@router.post("/start-session", response_model=Dict[str, Any])
async def start_imitation_session(
    target_action: Optional[str] = Form(None, description="目标动作类型（wave_hand, walk, grasp_object, sit_down, stand_up, jump, bend_over, turn_around）"),
    imitation_mode: str = Form("record_and_replay", description="模仿模式：real_time, record_and_replay, keyframe_learning, style_transfer"),
    session_notes: str = Form("", description="会话笔记"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """开始模仿会话API"""
    try:
        # 获取视觉模仿服务
        service = get_visual_imitation_service()
        
        # 解析参数
        target_action_enum = None
        if target_action:
            try:
                target_action_enum = ActionType(target_action)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"无效的动作类型: {target_action}"
                )
        
        try:
            mode_enum = ImitationMode(imitation_mode)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的模仿模式: {imitation_mode}"
            )
        
        # 开始模仿会话
        result = service.start_imitation_session(
            target_action=target_action_enum,
            imitation_mode=mode_enum,
            session_notes=session_notes
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "开始模仿会话失败")
            )
        
        return SuccessResponse.create(
            message="模仿会话开始成功",
            data=result
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"开始模仿会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"开始模仿会话失败: {str(e)}"
        )


@router.post("/record-pose", response_model=Dict[str, Any])
async def record_human_pose(
    session_id: str = Form(..., description="会话ID"),
    pose_data_json: str = Form(..., description="姿态数据JSON"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """录制人类姿态API
    
    姿态数据JSON格式：
    {
        "timestamp": 1234567890.123,
        "joints": {
            "left_wrist": {"x": 0.1, "y": 0.2, "z": 0.3, "confidence": 0.9},
            "right_wrist": {"x": 0.1, "y": 0.2, "z": 0.3, "confidence": 0.9},
            ...
        },
        "overall_confidence": 0.85
    }
    """
    try:
        # 解析姿态数据
        try:
            pose_data = json.loads(pose_data_json)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="姿态数据格式错误，必须是有效的JSON"
            )
        
        # 获取视觉模仿服务
        service = get_visual_imitation_service()
        
        # 录制姿态
        result = service.record_human_pose(session_id, pose_data)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "录制姿态失败")
            )
        
        return SuccessResponse.create(
            message="姿态录制成功",
            data=result
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"录制姿态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"录制姿态失败: {str(e)}"
        )


@router.post("/analyze-sequence", response_model=Dict[str, Any])
async def analyze_pose_sequence(
    session_id: str = Form(..., description="会话ID"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """分析姿态序列API"""
    try:
        # 获取视觉模仿服务
        service = get_visual_imitation_service()
        
        # 分析姿态序列
        result = service.analyze_pose_sequence(session_id)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "分析姿态序列失败")
            )
        
        return SuccessResponse.create(
            message="姿态序列分析完成",
            data=result
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析姿态序列失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析姿态序列失败: {str(e)}"
        )


@router.post("/generate-trajectory", response_model=Dict[str, Any])
async def generate_robot_trajectory(
    session_id: str = Form(..., description="会话ID"),
    adaptation_level: float = Form(0.5, description="适应级别（0.0=完全复制，1.0=最大适应）"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """生成机器人运动轨迹API"""
    try:
        # 验证适应级别范围
        if adaptation_level < 0.0 or adaptation_level > 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="适应级别必须在0.0到1.0之间"
            )
        
        # 获取视觉模仿服务
        service = get_visual_imitation_service()
        
        # 生成机器人轨迹
        result = service.generate_robot_trajectory(session_id, adaptation_level)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "生成机器人轨迹失败")
            )
        
        adaptation_text = "完全复制" if adaptation_level < 0.3 else "部分适应" if adaptation_level < 0.7 else "高度适应"
        return SuccessResponse.create(
            message=f"机器人轨迹生成完成（{adaptation_text}）",
            data=result
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成机器人轨迹失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成机器人轨迹失败: {str(e)}"
        )


@router.post("/evaluate-accuracy", response_model=Dict[str, Any])
async def evaluate_imitation_accuracy(
    session_id: str = Form(..., description="会话ID"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """评估模仿准确度API"""
    try:
        # 获取视觉模仿服务
        service = get_visual_imitation_service()
        
        # 评估模仿准确度
        result = service.evaluate_imitation_accuracy(session_id)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "评估模仿准确度失败")
            )
        
        # 根据评分给出反馈
        overall_score = result.get("overall_score", 0.0)
        if overall_score >= 0.8:
            feedback = "优秀模仿，接近完美"
        elif overall_score >= 0.6:
            feedback = "良好模仿，基本准确"
        elif overall_score >= 0.4:
            feedback = "一般模仿，需要改进"
        else:
            feedback = "需要重新学习和练习"
        
        result["feedback"] = feedback
        
        return SuccessResponse.create(
            message=f"模仿评估完成：{feedback}",
            data=result
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"评估模仿准确度失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"评估模仿准确度失败: {str(e)}"
        )


@router.get("/available-actions", response_model=Dict[str, Any])
async def get_available_actions(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取可用动作列表API"""
    try:
        # 获取视觉模仿服务
        service = get_visual_imitation_service()
        
        # 获取可用动作
        result = service.get_available_actions()
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="获取可用动作失败"
            )
        
        return SuccessResponse.create(
            message=f"获取到 {result.get('total_actions', 0)} 个可用动作",
            data=result
        ).dict()
        
    except Exception as e:
        logger.error(f"获取可用动作失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取可用动作失败: {str(e)}"
        )


@router.get("/imitation-sessions", response_model=Dict[str, Any])
async def get_imitation_sessions(
    limit: int = Query(50, description="返回数量限制"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取模仿会话记录API"""
    try:
        # 获取视觉模仿服务
        service = get_visual_imitation_service()
        
        # 获取模仿会话
        result = service.get_imitation_sessions(limit)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="获取模仿会话失败"
            )
        
        return SuccessResponse.create(
            message=f"获取到 {result.get('total_sessions', 0)} 个模仿会话",
            data=result
        ).dict()
        
    except Exception as e:
        logger.error(f"获取模仿会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模仿会话失败: {str(e)}"
        )


@router.get("/session-details", response_model=Dict[str, Any])
async def get_session_details(
    session_id: str = Query(..., description="会话ID"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取会话详情API"""
    try:
        # 获取视觉模仿服务
        service = get_visual_imitation_service()
        
        # 通过获取所有会话并筛选的方式
        all_sessions_result = service.get_imitation_sessions(limit=1000)
        
        if not all_sessions_result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="获取会话数据失败"
            )
        
        # 查找指定会话
        target_session = None
        for session in all_sessions_result.get("sessions", []):
            if session.get("session_id") == session_id:
                target_session = session
                break
        
        if not target_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"会话 '{session_id}' 未找到"
            )
        
        return SuccessResponse.create(
            message="会话详情获取成功",
            data={
                "session": target_session,
                "session_id": session_id
            }
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取会话详情失败: {str(e)}"
        )


@router.get("/available-body-parts", response_model=Dict[str, Any])
async def get_available_body_parts(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取可用的身体部位列表API"""
    try:
        # 返回所有身体部位
        body_parts = [
            {"value": "head", "label": "头部", "description": "头部关节"},
            {"value": "neck", "label": "颈部", "description": "颈部关节"},
            {"value": "left_shoulder", "label": "左肩", "description": "左肩关节"},
            {"value": "right_shoulder", "label": "右肩", "description": "右肩关节"},
            {"value": "left_elbow", "label": "左肘", "description": "左肘关节"},
            {"value": "right_elbow", "label": "右肘", "description": "右肘关节"},
            {"value": "left_wrist", "label": "左手腕", "description": "左手腕关节"},
            {"value": "right_wrist", "label": "右手腕", "description": "右手腕关节"},
            {"value": "left_hip", "label": "左髋", "description": "左髋关节"},
            {"value": "right_hip", "label": "右髋", "description": "右髋关节"},
            {"value": "left_knee", "label": "左膝", "description": "左膝关节"},
            {"value": "right_knee", "label": "右膝", "description": "右膝关节"},
            {"value": "left_ankle", "label": "左踝", "description": "左踝关节"},
            {"value": "right_ankle", "label": "右踝", "description": "右踝关节"}
        ]
        
        return SuccessResponse.create(
            message=f"获取到 {len(body_parts)} 个身体部位",
            data={
                "body_parts": body_parts,
                "total_count": len(body_parts)
            }
        ).dict()
        
    except Exception as e:
        logger.error(f"获取身体部位失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取身体部位失败: {str(e)}"
        )


@router.get("/available-imitation-modes", response_model=Dict[str, Any])
async def get_available_imitation_modes(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取可用的模仿模式列表API"""
    try:
        # 返回所有模仿模式
        imitation_modes = [
            {"value": "real_time", "label": "实时模仿", "description": "实时观察并立即模仿"},
            {"value": "record_and_replay", "label": "录制回放", "description": "先录制再回放模仿"},
            {"value": "keyframe_learning", "label": "关键帧学习", "description": "学习关键姿势帧"},
            {"value": "style_transfer", "label": "风格迁移", "description": "迁移动作风格"}
        ]
        
        return SuccessResponse.create(
            message=f"获取到 {len(imitation_modes)} 种模仿模式",
            data={
                "imitation_modes": imitation_modes,
                "total_count": len(imitation_modes)
            }
        ).dict()
        
    except Exception as e:
        logger.error(f"获取模仿模式失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模仿模式失败: {str(e)}"
        )


@router.post("/batch-process", response_model=Dict[str, Any])
async def batch_process_imitation(
    session_id: str = Form(..., description="会话ID"),
    steps_json: str = Form("[]", description="处理步骤JSON数组（可选步骤：analyze, generate, evaluate）"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """批量处理模仿任务API"""
    try:
        # 解析处理步骤
        try:
            steps = json.loads(steps_json)
        except json.JSONDecodeError:
            steps = ["analyze", "generate", "evaluate"]  # 默认步骤
        
        if not isinstance(steps, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="处理步骤必须是JSON数组"
            )
        
        # 获取视觉模仿服务
        service = get_visual_imitation_service()
        
        # 执行批处理
        results = {}
        
        for step in steps:
            if step == "analyze":
                results["analyze"] = service.analyze_pose_sequence(session_id)
            elif step == "generate":
                results["generate"] = service.generate_robot_trajectory(session_id, 0.5)
            elif step == "evaluate":
                results["evaluate"] = service.evaluate_imitation_accuracy(session_id)
            else:
                results[step] = {"success": False, "error": f"未知步骤: {step}"}
        
        # 统计成功步骤
        successful_steps = [step for step, result in results.items() if result.get("success", False)]
        
        return SuccessResponse.create(
            message=f"批处理完成，成功步骤: {len(successful_steps)}/{len(steps)}",
            data={
                "session_id": session_id,
                "steps_requested": steps,
                "steps_completed": successful_steps,
                "results": results
            }
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批处理模仿任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批处理模仿任务失败: {str(e)}"
        )


# WebSocket端点 - 实时视觉模仿
@router.websocket("/ws/real-time-imitation")
async def websocket_real_time_imitation(
    websocket: WebSocket,
    db: Session = Depends(get_db),
):
    """WebSocket实时视觉模仿API - 实时双向通信"""
    await websocket.accept()
    
    try:
        service = get_visual_imitation_service()
        
        # 欢迎消息
        await websocket.send_json({
            "type": "welcome",
            "message": "连接实时视觉模仿WebSocket成功",
            "supported_operations": ["start_session", "record_pose", "analyze", "generate", "evaluate", "get_actions"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        current_session_id = None
        
        # 实时模仿循环
        while True:
            # 接收消息
            data = await websocket.receive_json()
            
            message_type = data.get("type", "")
            
            if message_type == "start_session":
                # 开始新会话
                target_action = data.get("target_action")
                imitation_mode = data.get("imitation_mode", "record_and_replay")
                session_notes = data.get("session_notes", "")
                
                result = service.start_imitation_session(
                    target_action=ActionType(target_action) if target_action else None,
                    imitation_mode=ImitationMode(imitation_mode),
                    session_notes=session_notes
                )
                
                if result.get("success", False):
                    current_session_id = result["session_id"]
                    await websocket.send_json({
                        "type": "session_started",
                        "data": result
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": result.get("error", "开始会话失败")
                    })
                
            elif message_type == "record_pose":
                # 录制姿态
                if not current_session_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "请先开始一个会话"
                    })
                    continue
                
                pose_data = data.get("pose_data", {})
                result = service.record_human_pose(current_session_id, pose_data)
                
                await websocket.send_json({
                    "type": "pose_recorded",
                    "data": result
                })
                
            elif message_type == "analyze":
                # 分析姿态序列
                if not current_session_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "请先开始一个会话并录制姿态"
                    })
                    continue
                
                result = service.analyze_pose_sequence(current_session_id)
                
                await websocket.send_json({
                    "type": "analysis_result",
                    "data": result
                })
                
            elif message_type == "generate":
                # 生成机器人轨迹
                if not current_session_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "请先开始一个会话"
                    })
                    continue
                
                adaptation_level = data.get("adaptation_level", 0.5)
                result = service.generate_robot_trajectory(current_session_id, adaptation_level)
                
                await websocket.send_json({
                    "type": "trajectory_generated",
                    "data": result
                })
                
            elif message_type == "evaluate":
                # 评估模仿准确度
                if not current_session_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "请先开始一个会话"
                    })
                    continue
                
                result = service.evaluate_imitation_accuracy(current_session_id)
                
                await websocket.send_json({
                    "type": "evaluation_result",
                    "data": result
                })
                
            elif message_type == "get_actions":
                # 获取可用动作
                result = service.get_available_actions()
                
                await websocket.send_json({
                    "type": "actions_list",
                    "data": result
                })
                
            elif message_type == "end_session":
                # 结束会话
                if current_session_id:
                    await websocket.send_json({
                        "type": "session_ended",
                        "message": f"会话 {current_session_id} 已结束",
                        "session_id": current_session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    current_session_id = None
                else:
                    await websocket.send_json({
                        "type": "session_ended",
                        "message": "没有活动会话",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
            elif message_type == "ping":
                # 心跳检测
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
            elif message_type == "close":
                # 关闭连接
                await websocket.send_json({
                    "type": "connection_closed",
                    "message": "连接已关闭",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                break
                
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"未知的消息类型: {message_type}"
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket实时视觉模仿断开连接")
    except Exception as e:
        logger.error(f"WebSocket实时视觉模仿错误: {e}")
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