"""
电脑操作路由模块
提供实体机器人电脑操作和命令行控制API

基于升级001升级计划的第6部分：电脑操作能力完善
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Form,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import json
import base64
import logging

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.computer_operation_service import (
    get_computer_operation_service,
    OperationMode,
    KeyboardOperation,
    MouseOperation,
    CommandOperation,
)
from backend.schemas.response import SuccessResponse

router = APIRouter(prefix="/api/computer-operation", tags=["电脑操作"])

logger = logging.getLogger(__name__)


@router.post("/analyze-screen", response_model=Dict[str, Any])
async def analyze_screen(
    screenshot_file: Optional[UploadFile] = File(
        None, description="屏幕截图文件（可选）"
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """分析屏幕内容API - OCR和元素检测"""
    try:
        # 获取电脑操作服务
        service = get_computer_operation_service()

        # 处理截图文件
        screenshot_data = None
        if screenshot_file:
            screenshot_data = await screenshot_file.read()

        # 分析屏幕
        result = service.analyze_screen(screenshot_data)

        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "屏幕分析失败"),
            )

        return SuccessResponse.create(message="屏幕分析完成", data=result).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"屏幕分析失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"屏幕分析失败: {str(e)}",
        )


@router.post("/keyboard-operation", response_model=Dict[str, Any])
async def keyboard_operation(
    keys_json: str = Form("[]", description='按键列表JSON，如["ctrl", "c"]'),
    text: Optional[str] = Form(None, description="要输入的文本"),
    delay_between_keys: float = Form(0.1, description="按键间延迟（秒）"),
    press_duration: float = Form(0.05, description="按键持续时间（秒）"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """键盘操作API"""
    try:
        # 解析按键列表
        try:
            keys = json.loads(keys_json)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="按键列表格式错误，必须是有效的JSON数组",
            )

        # 获取电脑操作服务
        service = get_computer_operation_service()

        # 创建键盘操作
        operation = KeyboardOperation(
            keys=keys,
            text=text,
            delay_between_keys=delay_between_keys,
            press_duration=press_duration,
        )

        # 执行键盘操作
        result = service.perform_keyboard_operation(operation)

        return SuccessResponse.create(
            message="键盘操作执行完成",
            data={
                "success": result.success,
                "operation_id": result.operation_id,
                "operation_type": result.operation_type,
                "execution_time": result.execution_time,
                "details": result.details,
                "error": result.error_message,
            },
        ).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"键盘操作失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"键盘操作失败: {str(e)}",
        )


@router.post("/mouse-operation", response_model=Dict[str, Any])
async def mouse_operation(
    action: str = Form(
        ...,
        description="鼠标操作：click, double_click, right_click, drag, move, scroll",
    ),
    x: int = Form(..., description="X坐标"),
    y: int = Form(..., description="Y坐标"),
    button: str = Form("left", description="鼠标按钮：left, right, middle"),
    clicks: int = Form(1, description="点击次数"),
    duration: float = Form(0.5, description="操作持续时间（秒）"),
    scroll_amount: int = Form(0, description="滚动量（仅用于scroll操作）"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """鼠标操作API"""
    try:
        # 验证操作类型
        valid_actions = [
            "click",
            "double_click",
            "right_click",
            "drag",
            "move",
            "scroll",
        ]
        if action not in valid_actions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的鼠标操作，必须是: {', '.join(valid_actions)}",
            )

        # 获取电脑操作服务
        service = get_computer_operation_service()

        # 创建鼠标操作
        operation = MouseOperation(
            action=action,
            position=(x, y),
            button=button,
            clicks=clicks,
            duration=duration,
            scroll_amount=scroll_amount,
        )

        # 执行鼠标操作
        result = service.perform_mouse_operation(operation)

        return SuccessResponse.create(
            message="鼠标操作执行完成",
            data={
                "success": result.success,
                "operation_id": result.operation_id,
                "operation_type": result.operation_type,
                "execution_time": result.execution_time,
                "details": result.details,
                "error": result.error_message,
            },
        ).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"鼠标操作失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"鼠标操作失败: {str(e)}",
        )


@router.post("/execute-command", response_model=Dict[str, Any])
async def execute_command(
    command: str = Form(..., description="命令字符串"),
    args_json: str = Form("[]", description="参数列表JSON"),
    working_dir: Optional[str] = Form(None, description="工作目录"),
    timeout: int = Form(30, description="超时时间（秒）"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """命令行执行API"""
    try:
        # 解析参数列表
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="参数列表格式错误，必须是有效的JSON数组",
            )

        # 获取电脑操作服务
        service = get_computer_operation_service()

        # 创建命令操作
        operation = CommandOperation(
            command=command, args=args, working_dir=working_dir, timeout=timeout
        )

        # 执行命令
        result = service.execute_command(operation)

        return SuccessResponse.create(
            message="命令执行完成",
            data={
                "success": result.success,
                "operation_id": result.operation_id,
                "operation_type": result.operation_type,
                "execution_time": result.execution_time,
                "details": result.details,
                "error": result.error_message,
            },
        ).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"命令执行失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"命令执行失败: {str(e)}",
        )


@router.post("/translate-natural-language", response_model=Dict[str, Any])
async def translate_natural_language(
    natural_language: str = Form(
        ..., description="自然语言描述，如'列出当前目录的文件'"
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """自然语言到命令行翻译API"""
    try:
        # 获取电脑操作服务
        service = get_computer_operation_service()

        # 翻译自然语言
        result = service.translate_natural_language_to_command(natural_language)

        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "翻译失败"),
            )

        return SuccessResponse.create(message="自然语言翻译完成", data=result).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"自然语言翻译失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"自然语言翻译失败: {str(e)}",
        )


@router.post("/execute-task", response_model=Dict[str, Any])
async def execute_task(
    task_name: str = Form(..., description="任务名称，如'file_management.create_file'"),
    params_json: str = Form("{}", description="任务参数JSON"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """执行预定义任务API"""
    try:
        # 解析任务参数
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="参数格式错误，必须是有效的JSON对象",
            )

        # 获取电脑操作服务
        service = get_computer_operation_service()

        # 执行任务
        result = service.execute_task(task_name, params)

        return SuccessResponse.create(
            message="任务执行完成",
            data={
                "success": result.success,
                "operation_id": result.operation_id,
                "operation_type": result.operation_type,
                "execution_time": result.execution_time,
                "details": result.details,
                "error": result.error_message,
            },
        ).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"任务执行失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"任务执行失败: {str(e)}",
        )


@router.get("/operation-history", response_model=Dict[str, Any])
async def get_operation_history(
    limit: int = 50,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取操作历史API"""
    try:
        # 获取电脑操作服务
        service = get_computer_operation_service()

        # 获取操作历史
        history = service.get_operation_history(limit)

        return SuccessResponse.create(
            message=f"获取到 {len(history)} 条操作记录",
            data={"history": history, "total_count": len(history)},
        ).dict()

    except Exception as e:
        logger.error(f"获取操作历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取操作历史失败: {str(e)}",
        )


@router.get("/command-history", response_model=Dict[str, Any])
async def get_command_history(
    limit: int = 50,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取命令历史API"""
    try:
        # 获取电脑操作服务
        service = get_computer_operation_service()

        # 获取命令历史
        history = service.get_command_history(limit)

        return SuccessResponse.create(
            message=f"获取到 {len(history)} 条命令记录",
            data={"history": history, "total_count": len(history)},
        ).dict()

    except Exception as e:
        logger.error(f"获取命令历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取命令历史失败: {str(e)}",
        )


@router.get("/system-info", response_model=Dict[str, Any])
async def get_system_info(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取系统信息API"""
    try:
        # 获取电脑操作服务
        service = get_computer_operation_service()

        # 获取系统信息
        info = service.get_system_info()

        if not info.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=info.get("error", "获取系统信息失败"),
            )

        return SuccessResponse.create(message="系统信息获取成功", data=info).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统信息失败: {str(e)}",
        )


@router.post("/set-operation-mode", response_model=Dict[str, Any])
async def set_operation_mode(
    mode: str = Form(
        ..., description="操作模式：physical_robot, virtual_control, command_line"
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """设置操作模式API"""
    try:
        # 验证操作模式
        valid_modes = ["physical_robot", "virtual_control", "command_line"]
        if mode not in valid_modes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的操作模式，必须是: {', '.join(valid_modes)}",
            )

        # 获取电脑操作服务
        service = get_computer_operation_service()

        # 设置操作模式
        if mode == "physical_robot":
            service.operation_mode = OperationMode.PHYSICAL_ROBOT
        elif mode == "virtual_control":
            service.operation_mode = OperationMode.VIRTUAL_CONTROL
        elif mode == "command_line":
            service.operation_mode = OperationMode.COMMAND_LINE

        return SuccessResponse.create(
            message=f"操作模式已设置为: {mode}",
            data={
                "mode": mode,
                "description": {
                    "physical_robot": "实体机器人操作模式",
                    "virtual_control": "虚拟控制（软件模拟）模式",
                    "command_line": "命令行控制模式",
                }[mode],
            },
        ).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置操作模式失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置操作模式失败: {str(e)}",
        )


@router.get("/available-tasks", response_model=Dict[str, Any])
async def get_available_tasks(
    category: Optional[str] = Query(None, description="按类别筛选"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取可用任务列表API"""
    try:
        # 获取电脑操作服务
        get_computer_operation_service()

        # 在实际实现中，这里应该从服务获取任务库
        # 这里返回真实数据

        tasks = [
            {
                "id": "file_management.create_file",
                "name": "创建文件",
                "category": "文件管理",
                "description": "创建一个新的文本文件",
                "steps": 8,
                "estimated_time": "10秒",
            },
            {
                "id": "file_management.delete_file",
                "name": "删除文件",
                "category": "文件管理",
                "description": "删除指定的文件",
                "steps": 1,
                "estimated_time": "2秒",
            },
            {
                "id": "web_browsing.open_browser",
                "name": "打开浏览器",
                "category": "网页浏览",
                "description": "打开Chrome浏览器",
                "steps": 4,
                "estimated_time": "5秒",
            },
            {
                "id": "web_browsing.search_web",
                "name": "搜索网页",
                "category": "网页浏览",
                "description": "在Google中搜索内容",
                "steps": 6,
                "estimated_time": "8秒",
            },
            {
                "id": "document_editing.create_document",
                "name": "创建文档",
                "category": "文档编辑",
                "description": "创建Word文档并输入内容",
                "steps": 5,
                "estimated_time": "10秒",
            },
            {
                "id": "system_info.get_system_info",
                "name": "获取系统信息",
                "category": "系统信息",
                "description": "获取CPU、内存、磁盘使用情况",
                "steps": 1,
                "estimated_time": "3秒",
            },
        ]

        # 应用筛选
        filtered_tasks = []
        for task in tasks:
            if category and task["category"] != category:
                continue
            filtered_tasks.append(task)

        return SuccessResponse.create(
            message=f"获取到 {len(filtered_tasks)} 个可用任务",
            data={
                "tasks": filtered_tasks,
                "total_count": len(filtered_tasks),
                "categories": list(set(t["category"] for t in tasks)),
            },
        ).dict()

    except Exception as e:
        logger.error(f"获取可用任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取可用任务失败: {str(e)}",
        )


# WebSocket端点 - 实时电脑操作
@router.websocket("/ws/real-time-operation")
async def websocket_real_time_operation(
    websocket: WebSocket,
    db: Session = Depends(get_db),
):
    """WebSocket实时电脑操作API - 实时双向通信"""
    await websocket.accept()

    try:
        service = get_computer_operation_service()

        # 欢迎消息
        await websocket.send_json(
            {
                "type": "welcome",
                "message": "连接实时电脑操作WebSocket成功",
                "supported_operations": [
                    "keyboard",
                    "mouse",
                    "command",
                    "screen_analysis",
                ],
                "current_mode": service.operation_mode.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # 操作循环
        while True:
            # 接收消息
            data = await websocket.receive_json()

            message_type = data.get("type", "")

            if message_type == "analyze_screen":
                # 分析屏幕
                screenshot_data = data.get("screenshot_data")
                if screenshot_data:
                    # 解码base64图像数据
                    try:
                        screenshot_bytes = base64.b64decode(screenshot_data)
                        result = service.analyze_screen(screenshot_bytes)
                    except Exception:
                        result = service.analyze_screen(None)
                else:
                    result = service.analyze_screen(None)

                await websocket.send_json(
                    {"type": "screen_analysis_result", "data": result}
                )

            elif message_type == "keyboard_operation":
                # 键盘操作
                keys = data.get("keys", [])
                text = data.get("text", "")
                delay = data.get("delay", 0.1)

                operation = KeyboardOperation(
                    keys=keys, text=text, delay_between_keys=delay
                )

                result = service.perform_keyboard_operation(operation)

                await websocket.send_json(
                    {
                        "type": "keyboard_operation_result",
                        "data": {
                            "success": result.success,
                            "operation_id": result.operation_id,
                            "details": result.details,
                            "error": result.error_message,
                        },
                    }
                )

            elif message_type == "mouse_operation":
                # 鼠标操作
                action = data.get("action", "click")
                x = data.get("x", 100)
                y = data.get("y", 100)
                button = data.get("button", "left")

                operation = MouseOperation(
                    action=action, position=(x, y), button=button
                )

                result = service.perform_mouse_operation(operation)

                await websocket.send_json(
                    {
                        "type": "mouse_operation_result",
                        "data": {
                            "success": result.success,
                            "operation_id": result.operation_id,
                            "details": result.details,
                            "error": result.error_message,
                        },
                    }
                )

            elif message_type == "execute_command":
                # 执行命令
                command = data.get("command", "")
                args = data.get("args", [])

                operation = CommandOperation(command=command, args=args)

                result = service.execute_command(operation)

                await websocket.send_json(
                    {
                        "type": "command_execution_result",
                        "data": {
                            "success": result.success,
                            "operation_id": result.operation_id,
                            "details": result.details,
                            "error": result.error_message,
                        },
                    }
                )

            elif message_type == "ping":
                # 心跳检测
                await websocket.send_json(
                    {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            elif message_type == "disconnect":
                # 断开连接
                await websocket.send_json(
                    {
                        "type": "disconnect_ack",
                        "message": "连接已断开",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                break

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"未知的消息类型: {message_type}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket断开连接")
    except Exception as e:
        logger.error(f"WebSocket实时操作错误: {e}")
        try:
            await websocket.send_json(
                {"type": "error", "message": f"服务器错误: {str(e)}"}
            )
            await websocket.close()
        except Exception:
            pass  # 已实现


# 导出路由
__all__ = ["router"]
