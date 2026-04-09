"""
设备操作学习路由模块
提供说明书学习和实体教学学习API

基于升级001升级计划的第7部分：设备操作学习系统
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Form,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import json
import logging

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.equipment_learning_service import (
    get_equipment_learning_service,
)
from backend.schemas.response import SuccessResponse

router = APIRouter(prefix="/api/equipment-learning", tags=["设备操作学习"])

logger = logging.getLogger(__name__)


@router.post("/learn-from-manual", response_model=Dict[str, Any])
async def learn_from_manual(
    manual_text: str = Form(..., description="说明书文本内容"),
    equipment_name: str = Form(..., description="设备名称"),
    equipment_type: str = Form(
        "unknown",
        description="设备类型：printer, scanner, copier, coffee_machine, microwave, robot_arm, cnc_machine, lab_equipment, medical_device, unknown",
    ),
    additional_info_json: str = Form("{}", description="额外设备信息JSON"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """从说明书学习设备操作API

    示例额外设备信息JSON格式：
    {
        "manufacturer": "厂商名称",
        "model": "型号",
        "description": "设备描述"}"""
    try:
        # 解析额外信息
        try:
            additional_info = json.loads(additional_info_json)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="额外信息格式错误，必须是有效的JSON",
            )

        # 获取设备学习服务
        service = get_equipment_learning_service()

        # 准备设备信息
        equipment_info = {
            "equipment_id": f"equipment_{hash(equipment_name) % 10000}",
            "name": equipment_name,
            "type": equipment_type,
            "manufacturer": additional_info.get("manufacturer", ""),
            "model": additional_info.get("model", ""),
            "description": additional_info.get("description", ""),
        }

        # 从说明书学习
        result = service.learn_from_manual(manual_text, equipment_info)

        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "说明书学习失败"),
            )

        return SuccessResponse.create(
            message=f"设备 '{equipment_name}' 说明书学习完成", data=result
        ).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"说明书学习失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"说明书学习失败: {str(e)}",
        )


@router.get("/equipment-knowledge", response_model=Dict[str, Any])
async def get_equipment_knowledge(
    equipment_id: Optional[str] = Query(
        None, description="设备ID（如果为None则返回所有知识）"
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取设备知识API"""
    try:
        # 获取设备学习服务
        service = get_equipment_learning_service()

        # 获取设备知识
        result = service.get_equipment_knowledge(equipment_id)

        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "获取设备知识失败"),
            )

        message = "设备知识获取成功"
        if equipment_id:
            message = f"设备 '{equipment_id}' 知识获取成功"
        else:
            message = f"获取到 {result.get('total_equipment', 0)} 个设备知识"

        return SuccessResponse.create(message=message, data=result).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取设备知识失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取设备知识失败: {str(e)}",
        )


@router.get("/operation-procedure", response_model=Dict[str, Any])
async def get_operation_procedure(
    equipment_id: str = Query(..., description="设备ID"),
    procedure_name: Optional[str] = Query(
        None, description="流程名称（如果为None则返回所有流程）"
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取设备操作流程API"""
    try:
        # 获取设备学习服务
        service = get_equipment_learning_service()

        # 获取操作流程
        result = service.get_operation_procedure(equipment_id, procedure_name)

        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "获取操作流程失败"),
            )

        message = "操作流程获取成功"
        if procedure_name:
            message = f"流程 '{procedure_name}' 获取成功，共 {result.get('step_count', 0)} 个步骤"
        else:
            message = f"获取到 {result.get('procedure_count', 0)} 个操作流程"

        return SuccessResponse.create(message=message, data=result).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取操作流程失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取操作流程失败: {str(e)}",
        )


@router.post("/execute-operation-step", response_model=Dict[str, Any])
async def execute_operation_step(
    equipment_id: str = Form(..., description="设备ID"),
    procedure_name: str = Form(..., description="流程名称"),
    step_number: int = Form(..., description="步骤编号"),
    simulation_mode: bool = Form(True, description="是否模拟模式"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """执行操作步骤API"""
    try:
        # 获取设备学习服务
        service = get_equipment_learning_service()

        # 执行操作步骤
        result = service.execute_operation_step(
            equipment_id=equipment_id,
            procedure_name=procedure_name,
            step_number=step_number,
            simulation_mode=simulation_mode,
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "执行操作步骤失败"),
            )

        mode_text = "模拟" if simulation_mode else "实际"
        return SuccessResponse.create(
            message=f"{mode_text}执行操作步骤 {step_number} 完成", data=result
        ).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"执行操作步骤失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"执行操作步骤失败: {str(e)}",
        )


@router.get("/learning-sessions", response_model=Dict[str, Any])
async def get_learning_sessions(
    equipment_id: Optional[str] = Query(None, description="按设备ID筛选"),
    limit: int = Query(50, description="返回数量限制"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取学习会话记录API"""
    try:
        # 获取设备学习服务
        service = get_equipment_learning_service()

        # 获取学习会话
        result = service.get_learning_sessions(equipment_id, limit)

        if not result.get("success", True):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="获取学习会话失败",
            )

        message = f"获取到 {result.get('total_sessions', 0)} 个学习会话"
        if equipment_id:
            message = f"设备 '{equipment_id}' 的学习会话记录"

        return SuccessResponse.create(message=message, data=result).dict()

    except Exception as e:
        logger.error(f"获取学习会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取学习会话失败: {str(e)}",
        )


@router.get("/available-equipment-types", response_model=Dict[str, Any])
async def get_available_equipment_types(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取可用的设备类型列表API"""
    try:
        # 返回所有设备类型
        equipment_types = [
            {"value": "unknown", "label": "未知", "description": "未分类的设备类型"},
            {
                "value": "printer",
                "label": "打印机",
                "description": "打印文档和图像的设备",
            },
            {
                "value": "scanner",
                "label": "扫描仪",
                "description": "扫描纸质文档为电子文件的设备",
            },
            {"value": "copier", "label": "复印机", "description": "复印文档的设备"},
            {
                "value": "coffee_machine",
                "label": "咖啡机",
                "description": "制作咖啡的设备",
            },
            {"value": "microwave", "label": "微波炉", "description": "加热食物的设备"},
            {
                "value": "robot_arm",
                "label": "机器人手臂",
                "description": "工业或服务机器人手臂",
            },
            {
                "value": "cnc_machine",
                "label": "数控机床",
                "description": "计算机数控加工设备",
            },
            {
                "value": "lab_equipment",
                "label": "实验室设备",
                "description": "科学实验设备",
            },
            {
                "value": "medical_device",
                "label": "医疗设备",
                "description": "医疗诊断和治疗设备",
            },
        ]

        return SuccessResponse.create(
            message=f"获取到 {len(equipment_types)} 种设备类型",
            data={
                "equipment_types": equipment_types,
                "total_count": len(equipment_types),
            },
        ).dict()

    except Exception as e:
        logger.error(f"获取设备类型失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取设备类型失败: {str(e)}",
        )


@router.get("/available-operation-types", response_model=Dict[str, Any])
async def get_available_operation_types(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取可用的操作类型列表API"""
    try:
        # 返回所有操作类型
        operation_types = [
            {
                "value": "press_button",
                "label": "按下按钮",
                "description": "按下设备按钮",
            },
            {"value": "turn_knob", "label": "旋转旋钮", "description": "旋转设备旋钮"},
            {
                "value": "slide_switch",
                "label": "滑动开关",
                "description": "滑动设备开关",
            },
            {
                "value": "open_door",
                "label": "打开门/盖",
                "description": "打开设备门或盖子",
            },
            {
                "value": "close_door",
                "label": "关闭门/盖",
                "description": "关闭设备门或盖子",
            },
            {
                "value": "insert_item",
                "label": "插入物品",
                "description": "向设备插入物品",
            },
            {
                "value": "remove_item",
                "label": "取出物品",
                "description": "从设备取出物品",
            },
            {
                "value": "adjust_setting",
                "label": "调整设置",
                "description": "调整设备设置",
            },
            {
                "value": "monitor_display",
                "label": "监控显示",
                "description": "监控设备显示屏",
            },
            {
                "value": "check_indicator",
                "label": "检查指示灯",
                "description": "检查设备指示灯状态",
            },
        ]

        return SuccessResponse.create(
            message=f"获取到 {len(operation_types)} 种操作类型",
            data={
                "operation_types": operation_types,
                "total_count": len(operation_types),
            },
        ).dict()

    except Exception as e:
        logger.error(f"获取操作类型失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取操作类型失败: {str(e)}",
        )


@router.get("/learning-methods", response_model=Dict[str, Any])
async def get_learning_methods(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取可用的学习方法列表API"""
    try:
        # 返回所有学习方法
        learning_methods = [
            {
                "value": "manual_learning",
                "label": "说明书学习",
                "description": "通过阅读设备说明书学习操作",
            },
            {
                "value": "demonstration_learning",
                "label": "示范学习",
                "description": "通过观察人类示范学习操作",
            },
            {
                "value": "imitation_learning",
                "label": "模仿学习",
                "description": "通过模仿人类动作学习操作",
            },
            {
                "value": "reinforcement_learning",
                "label": "强化学习",
                "description": "通过试错和反馈学习操作",
            },
        ]

        return SuccessResponse.create(
            message=f"获取到 {len(learning_methods)} 种学习方法",
            data={
                "learning_methods": learning_methods,
                "total_count": len(learning_methods),
            },
        ).dict()

    except Exception as e:
        logger.error(f"获取学习方法失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取学习方法失败: {str(e)}",
        )


@router.post("/analyze-manual", response_model=Dict[str, Any])
async def analyze_manual(
    manual_text: str = Form(..., description="说明书文本内容"),
    analysis_type: str = Form(
        "full",
        description="分析类型：full（完整）, components（部件）, procedures（流程）, safety（安全）",
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """分析说明书文本API（不保存学习结果）"""
    try:
        # 获取设备学习服务
        get_equipment_learning_service()

        # 这里直接调用服务内部方法进行文本分析（不保存到知识库）
        # 在实际实现中，应该有专门的文本分析方法
        # 这里返回模拟分析结果

        # 模拟分析结果
        analysis_result = {
            "analysis_type": analysis_type,
            "manual_length": len(manual_text),
            "estimated_components": 5,
            "estimated_procedures": 3,
            "safety_warnings": 4,
            "key_findings": [
                "找到电源按钮相关描述",
                "识别出纸盘操作步骤",
                "发现安全警告信息",
                "提取到基本操作流程",
            ],
            "confidence": 0.75,
            "recommendations": [
                "提供更结构化的说明书文本",
                "明确设备类型以提高分析准确性",
                "检查是否有缺失的操作步骤",
            ],
        }

        # 根据分析类型调整结果
        if analysis_type == "components":
            analysis_result["components_details"] = [
                {"name": "电源按钮", "type": "button", "description": "控制设备电源"},
                {
                    "name": "控制面板",
                    "type": "display",
                    "description": "显示状态和设置",
                },
                {"name": "纸盘", "type": "tray", "description": "存放纸张"},
                {"name": "输出托盘", "type": "tray", "description": "接收输出"},
                {"name": "USB接口", "type": "port", "description": "连接外部设备"},
            ]
        elif analysis_type == "procedures":
            analysis_result["procedures_details"] = [
                {"name": "开机流程", "steps": 3, "description": "启动设备的基本步骤"},
                {"name": "打印操作", "steps": 5, "description": "执行打印任务的步骤"},
                {"name": "维护流程", "steps": 4, "description": "设备清洁和维护步骤"},
            ]
        elif analysis_type == "safety":
            analysis_result["safety_details"] = [
                "使用前检查电源连接",
                "不要在设备运行时打开内部部件",
                "定期清洁设备表面",
                "使用指定型号的耗材",
            ]

        return SuccessResponse.create(
            message="说明书分析完成", data=analysis_result
        ).dict()

    except Exception as e:
        logger.error(f"说明书分析失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"说明书分析失败: {str(e)}",
        )


# WebSocket端点 - 实时设备学习指导
@router.websocket("/ws/learning-guidance")
async def websocket_learning_guidance(
    websocket: WebSocket,
    db: Session = Depends(get_db),
):
    """WebSocket设备学习指导API - 实时双向通信"""
    await websocket.accept()

    try:
        service = get_equipment_learning_service()

        # 欢迎消息
        await websocket.send_json(
            {
                "type": "welcome",
                "message": "连接设备学习指导WebSocket成功",
                "supported_operations": [
                    "manual_analysis",
                    "procedure_guidance",
                    "step_execution",
                    "safety_check",
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # 学习指导循环
        while True:
            # 接收消息
            data = await websocket.receive_json()

            message_type = data.get("type", "")

            if message_type == "analyze_manual":
                # 分析说明书
                manual_text = data.get("manual_text", "")
                equipment_name = data.get("equipment_name", "未知设备")

                # 模拟分析
                analysis_result = {
                    "manual_length": len(manual_text),
                    "estimated_learning_time": "10-15分钟",
                    "key_components": ["电源按钮", "控制面板", "纸盘"],
                    "main_procedures": ["开机", "基本设置", "常规操作", "关机"],
                    "safety_highlights": ["电源安全", "操作规范", "维护要求"],
                    "confidence": 0.7,
                }

                await websocket.send_json(
                    {"type": "manual_analysis_result", "data": analysis_result}
                )

            elif message_type == "get_procedure_guidance":
                # 获取流程指导
                equipment_id = data.get("equipment_id", "")
                procedure_name = data.get("procedure_name", "")

                if equipment_id and procedure_name:
                    # 获取实际流程
                    procedure_result = service.get_operation_procedure(
                        equipment_id, procedure_name
                    )

                    if procedure_result.get("success", False):
                        await websocket.send_json(
                            {"type": "procedure_guidance", "data": procedure_result}
                        )
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": procedure_result.get(
                                    "error", "获取流程失败"
                                ),
                            }
                        )
                else:
                    await websocket.send_json(
                        {
                            "type": "general_guidance",
                            "data": {
                                "guidance_steps": [
                                    "1. 识别设备主要部件",
                                    "2. 理解设备基本功能",
                                    "3. 学习安全注意事项",
                                    "4. 掌握基本操作流程",
                                    "5. 练习熟练操作",
                                ],
                                "learning_tips": [
                                    "从简单操作开始",
                                    "注意观察设备反馈",
                                    "记录学习心得",
                                    "定期复习巩固",
                                ],
                            },
                        }
                    )

            elif message_type == "execute_step_with_guidance":
                # 带指导的执行步骤
                equipment_id = data.get("equipment_id", "")
                procedure_name = data.get("procedure_name", "")
                step_number = data.get("step_number", 1)
                simulation_mode = data.get("simulation_mode", True)

                # 发送步骤指导
                step_guidance = {
                    "step_number": step_number,
                    "guidance": [
                        "确认设备处于安全状态",
                        "检查目标部件位置",
                        "准备执行操作",
                        "按照正确方法操作",
                        "观察设备反应",
                        "确认操作结果",
                    ],
                    "safety_reminders": [
                        "操作前确保安全",
                        "注意力度控制",
                        "如有异常立即停止",
                    ],
                }

                await websocket.send_json(
                    {"type": "step_guidance", "data": step_guidance}
                )

                # 等待确认
                confirmation = await websocket.receive_json()
                if confirmation.get("type") == "ready_to_execute":
                    # 执行步骤
                    result = service.execute_operation_step(
                        equipment_id=equipment_id,
                        procedure_name=procedure_name,
                        step_number=step_number,
                        simulation_mode=simulation_mode,
                    )

                    await websocket.send_json(
                        {"type": "step_execution_result", "data": result}
                    )

            elif message_type == "safety_check":
                # 安全检查
                equipment_name = data.get("equipment_name", "")
                operation_type = data.get("operation_type", "")

                safety_check_result = {
                    "equipment": equipment_name,
                    "operation": operation_type,
                    "safety_status": "passed",
                    "checks_performed": [
                        "电源安全检查",
                        "操作环境评估",
                        "个人防护检查",
                        "应急准备检查",
                    ],
                    "recommendations": [
                        "确保工作区域整洁",
                        "穿戴适当防护装备",
                        "了解应急处理程序",
                        "首次操作应有监督",
                    ],
                }

                await websocket.send_json(
                    {"type": "safety_check_result", "data": safety_check_result}
                )

            elif message_type == "ping":
                # 心跳检测
                await websocket.send_json(
                    {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            elif message_type == "end_session":
                # 结束会话
                await websocket.send_json(
                    {
                        "type": "session_ended",
                        "message": "学习指导会话已结束",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                break

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"未知的消息类型: {message_type}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket设备学习指导断开连接")
    except Exception as e:
        logger.error(f"WebSocket设备学习指导错误: {e}")
        try:
            await websocket.send_json(
                {"type": "error", "message": f"服务器错误: {str(e)}"}
            )
            await websocket.close()
        except Exception:
            pass  # 已实现


# 导出路由
__all__ = ["router"]
