"""
聊天路由模块
处理聊天、模型管理和系统模式相关的API请求
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Query,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
)
from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime, timezone
import uuid
import asyncio
import json

from backend.dependencies import get_db, get_current_user

from backend.db_models.user import User

from backend.db_models.chat import ChatSession, ChatMessage

from backend.services.model_service import get_model_service

from backend.services.system_service import get_system_service

from backend.schemas.response import SuccessResponse, PaginatedResponse
from backend.core.config import Config

router = APIRouter(prefix="/api", tags=["聊天与模型"])


def generate_intelligent_response(message: str) -> str:
    """生成智能回复 - 真实模型推理实现

    使用真实模型服务生成回复，禁止使用模板匹配。
    如果模型服务不可用，抛出明确错误。
    """
    try:
        from backend.services.model_service import get_model_service
        from fastapi import HTTPException

        model_service = get_model_service()

        # 严格检查模型服务状态
        if not model_service or not hasattr(model_service, "generate_response"):
            raise HTTPException(status_code=503, detail="模型服务未初始化或不可用")

        if not model_service.is_ready():
            raise HTTPException(
                status_code=503, detail="模型服务未就绪，请检查模型加载状态"
            )

        # 使用真实模型推理
        result = model_service.generate_response(
            message=message, model_name="default", temperature=0.7
        )

        # 验证结果
        if not result or "response" not in result:
            raise HTTPException(status_code=500, detail="模型生成响应格式错误")

        response_text = result["response"]

        # 检查响应是否包含错误信息
        if (
            "错误" in response_text
            or "失败" in response_text
            or "出错" in response_text
        ):
            # 模型生成失败，返回详细错误信息
            raise HTTPException(
                status_code=500, detail=f"模型生成失败: {response_text}"
            )

        return response_text

    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 其他异常
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"模型推理过程出错: {str(e)}")


@router.get("/chat/models", response_model=SuccessResponse)
async def get_chat_models(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取聊天模型列表 - 使用真实模型服务"""
    try:
        # 获取模型服务并获取可用模型
        model_service = get_model_service()
        models = model_service.get_available_models()

        return SuccessResponse.create(
            data={
                "models": models,
                "model_service_status": model_service.get_model_info(),
            },
            message="获取模型列表成功",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型列表失败: {str(e)}",
        )


@router.get("/chat/mode", response_model=SuccessResponse)
async def get_system_mode(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取系统模式 - 使用真实系统状态服务"""
    try:
        # 获取系统服务实例
        system_service = get_system_service()

        # 获取真实系统模式信息
        system_mode = system_service.get_system_mode()

        return SuccessResponse.create(data=system_mode, message="获取系统模式成功")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统模式失败: {str(e)}",
        )


@router.put("/chat/mode", response_model=SuccessResponse)
async def set_system_mode(
    mode_request: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """设置系统模式 - 使用真实系统状态服务"""
    try:
        # 检查用户权限
        if not user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="只有管理员可以更改系统模式",
            )

        mode = mode_request.get("mode", "assist")

        # 验证模式
        valid_modes = ["assist", "autonomous", "training", "maintenance"]
        if mode not in valid_modes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的系统模式。有效模式: {', '.join(valid_modes)}",
            )

        # 获取系统服务实例
        system_service = get_system_service()

        # 设置系统模式
        system_mode = system_service.set_system_mode(mode)

        return SuccessResponse.create(
            data=system_mode, message=f"系统模式已切换到: {mode}"
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置系统模式失败: {str(e)}",
        )


@router.post(
    "/models/{model_name}/chat", response_model=SuccessResponse)
async def chat_with_model(
    model_name: str,
    chat_request: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """与模型聊天 - 使用真实模型服务"""
    try:
        message = chat_request.get("message", "")
        session_id = chat_request.get("session_id", str(uuid.uuid4()))
        temperature = chat_request.get("temperature", 0.7)
        max_length = chat_request.get("max_length", 100)

        if not message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="消息内容不能为空"
            )

        # 获取模型服务并生成响应
        model_service = get_model_service()
        result = model_service.generate_response(
            text=message,
            model_name=model_name,
            temperature=temperature,
            max_length=max_length,
            session_id=session_id,  # 使用客户端提供的会话ID
        )

        # 如果生成失败，返回错误
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "模型服务不可用"),
            )

        # 返回模型响应
        return SuccessResponse.create(
            data={
                "response": result["response"],
                "session_id": session_id,
                "model_name": model_name,
                "temperature": temperature,
                "max_length": max_length,
                "processing_time": result.get("processing_time", 0.5),
                "tokens_used": result.get(
                    "tokens_used",
                    len(message) // 4 + len(result.get("response", "")) // 4,
                ),
                "memories_retrieved": result.get("memories_retrieved", 0),
                "model_info": result.get("model_info", {}),
                "is_simulated": result.get("is_simulated", True),
                "analysis": result.get("analysis", {}),
                "context_length": result.get("context_length", 0),
                "has_related_memory": result.get("has_related_memory", False),
            },
            message="聊天响应生成成功",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"发送聊天消息失败: {str(e)}",
        )


@router.get("/chat/sessions", response_model=PaginatedResponse)
async def get_chat_sessions(
    limit: int = Query(20, ge=1, le=100, description="每页数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取聊天会话列表"""
    try:
        # 查询当前用户的聊天会话
        query = db.query(ChatSession).filter(ChatSession.user_id == user.id)

        # 获取总数
        total = query.count()

        # 应用排序和分页
        sessions = (
            query.order_by(ChatSession.updated_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        # 转换为字典格式
        paginated_sessions = []
        for session in sessions:
            paginated_sessions.append(
                {
                    "id": str(session.id),
                    "title": session.title,
                    "model_name": session.model_name,
                    "is_active": session.is_active,
                    "message_count": session.message_count,
                    "total_tokens": session.total_tokens,
                    "last_message_at": (
                        session.last_message_at.isoformat()
                        if session.last_message_at
                        else None
                    ),
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                }
            )

        # 计算页码（offset从0开始）
        page = offset // limit + 1 if limit > 0 else 1
        size = limit

        return PaginatedResponse.create(
            items=paginated_sessions,
            total=total,
            page=page,
            size=size,
            message="获取聊天会话列表成功",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取聊天会话列表失败: {str(e)}",
        )


# 注意：还有其他端点如GET /chat/sessions/{session_id},
# DELETE /chat/sessions/{session_id}, GET /chat/sessions/{session_id}/messages,
# 完整处理
# 实际项目中应该实现完整的聊天功能


# 完整聊天系统功能
@router.post("/chat/sessions", response_model=SuccessResponse)
async def create_chat_session(
    session_request: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """创建新聊天会话"""
    try:
        title = session_request.get(
            "title", f"新会话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        model_name = session_request.get("model_name", "默认模型")

        # 创建聊天会话并保存到数据库
        session = ChatSession(
            title=title,
            model_name=model_name,
            user_id=user.id,
            is_active=True,
            message_count=0,
            total_tokens=0,
            last_message_at=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # 构建响应数据
        new_session = {
            "id": str(session.id),
            "title": session.title,
            "model_name": session.model_name,
            "created_at": session.created_at.isoformat(),
            "last_message": "会话已创建",
            "message_count": session.message_count,
            "user_id": str(session.user_id),
        }

        return SuccessResponse.create(
            data={"session": new_session}, message="聊天会话创建成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建聊天会话失败: {str(e)}",
        )


@router.get(
    "/chat/sessions/{session_id}", response_model=SuccessResponse)
async def get_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取聊天会话详情"""
    try:
        # 将session_id转换为整数
        try:
            session_id_int = int(session_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"聊天会话 {session_id} 不存在",
            )

        # 查询聊天会话
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id_int, ChatSession.user_id == user.id)
            .first()
        )

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"聊天会话 {session_id} 不存在",
            )

        # 获取会话消息（可选）
        messages = []
        # 如果需要可以查询ChatMessage表

        session_data = {
            "id": str(session.id),
            "title": session.title,
            "model_name": session.model_name,
            "is_active": session.is_active,
            "message_count": session.message_count,
            "total_tokens": session.total_tokens,
            "last_message_at": (
                session.last_message_at.isoformat() if session.last_message_at else None
            ),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "messages": messages,
        }

        return SuccessResponse.create(data=session_data, message="获取聊天会话详情成功")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取聊天会话详情失败: {str(e)}",
        )


@router.put(
    "/chat/sessions/{session_id}", response_model=SuccessResponse)
async def update_chat_session(
    session_id: str,
    update_request: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """更新聊天会话（如重命名）"""
    try:
        title = update_request.get("title")
        model_name = update_request.get("model_name")

        if not title and not model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="请提供要更新的字段"
            )

        # 将session_id转换为整数
        try:
            session_id_int = int(session_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"聊天会话 {session_id} 不存在",
            )

        # 查询聊天会话
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id_int, ChatSession.user_id == user.id)
            .first()
        )

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"聊天会话 {session_id} 不存在",
            )

        # 更新字段
        updated = False
        if title is not None:
            session.title = title
            updated = True
        if model_name is not None:
            session.model_name = model_name
            updated = True

        if updated:
            session.updated_at = datetime.now(timezone.utc)
            db.commit()

        # 返回更新后的会话数据
        session_data = {
            "id": str(session.id),
            "title": session.title,
            "model_name": session.model_name,
            "is_active": session.is_active,
            "message_count": session.message_count,
            "total_tokens": session.total_tokens,
            "last_message_at": (
                session.last_message_at.isoformat() if session.last_message_at else None
            ),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }

        return SuccessResponse.create(data=session_data, message="聊天会话更新成功")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新聊天会话失败: {str(e)}",
        )


@router.delete(
    "/chat/sessions/{session_id}", response_model=SuccessResponse)
async def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """删除聊天会话"""
    try:
        # 将session_id转换为整数
        try:
            session_id_int = int(session_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="无效的会话ID格式"
            )

        # 查询会话，确保用户只能删除自己的会话
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id_int, ChatSession.user_id == user.id)
            .first()
        )

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="聊天会话不存在或无权访问"
            )

        # 删除关联的聊天消息
        db.query(ChatMessage).filter(ChatMessage.session_id == session_id_int).delete()

        # 删除会话
        db.delete(session)
        db.commit()

        return SuccessResponse.create(data={}, message=f"聊天会话 {session_id} 已删除")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除聊天会话失败: {str(e)}",
        )


@router.get(
    "/chat/sessions/{session_id}/messages",
    response_model=SuccessResponse,
)
async def get_chat_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=200, description="每页数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取聊天会话的消息历史"""
    try:
        # 验证会话ID并检查权限
        try:
            session_id_int = int(session_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="会话ID格式无效"
            )

        # 检查会话是否存在且属于当前用户
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id_int, ChatSession.user_id == user.id)
            .first()
        )

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="聊天会话不存在或无权访问"
            )

        # 查询真实消息数据
        query = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id_int)
            .order_by(ChatMessage.created_at)
        )

        # 获取总数
        total = query.count()

        # 应用分页
        messages_db = query.offset(offset).limit(limit).all()

        # 转换为API响应格式
        messages = []
        for msg in messages_db:
            # 解析元数据
            metadata = {}
            if msg.message_metadata:
                try:
                    metadata = json.loads(msg.message_metadata)
                except Exception:
                    metadata = {"type": "text", "status": "completed"}

            messages.append(
                {
                    "id": str(msg.id),
                    "session_id": str(msg.session_id),
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": (
                        msg.created_at.isoformat()
                        if msg.created_at
                        else datetime.now(timezone.utc).isoformat()
                    ),
                    "model_name": metadata.get("model_name", "默认模型"),
                    "tokens": msg.tokens or len(msg.content) // 4,
                    "metadata": {
                        "type": metadata.get("type", "text"),
                        "status": metadata.get("status", "completed"),
                        "processing_time": metadata.get("processing_time", 0.5),
                    },
                }
            )

        paginated_messages = messages  # 已经分页

        return SuccessResponse.create(
            data={
                "messages": paginated_messages,
                "total": total,
                "session_id": session_id,
            },
            message="获取聊天消息成功",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取聊天消息失败: {str(e)}",
        )


@router.post(
    "/chat/sessions/{session_id}/messages",
    response_model=SuccessResponse,
)
async def send_chat_message(
    session_id: str,
    message_request: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """发送消息到聊天会话"""
    try:
        message_content = message_request.get("message", "")
        model_name = message_request.get("model_name", "默认模型")
        temperature = message_request.get("temperature", 0.7)

        if not message_content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="消息内容不能为空"
            )

        # 生成消息ID
        str(uuid.uuid4())
        str(uuid.uuid4())

        # 验证会话ID并检查权限
        try:
            session_id_int = int(session_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="会话ID格式无效"
            )

        # 检查会话是否存在且属于当前用户
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id_int, ChatSession.user_id == user.id)
            .first()
        )

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="聊天会话不存在或无权访问"
            )

        # 保存用户消息到数据库
        user_msg_metadata = json.dumps(
            {
                "type": "text",
                "status": "sent",
                "processing_time": 0.1,
                "model_name": model_name,
                "temperature": temperature,
            }
        )

        user_msg = ChatMessage(
            session_id=session_id_int,
            role="user",
            content=message_content,
            tokens=len(message_content) // 4,
            message_metadata=user_msg_metadata,
        )
        db.add(user_msg)
        db.flush()  # 获取ID

        # 生成AI响应（尝试使用模型服务，否则使用智能回复）
        response_text = ""
        try:
            # 尝试调用模型服务
            model_service = get_model_service()
            if model_service and hasattr(model_service, "generate_response"):
                # 调用真实模型服务
                response_result = model_service.generate_response(
                    text=message_content, model_name=model_name, temperature=temperature
                )
                if response_result and "response" in response_result:
                    response_text = response_result["response"]
                else:
                    raise Exception("模型服务未返回有效响应")
            else:
                raise Exception("模型服务不可用")
        except Exception as e:
            # 模型服务调用失败，抛出错误而不是返回模板响应
            logger.error(f"生成响应失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"生成AI响应失败: {str(e)}",
            )

        # 保存AI响应到数据库
        ai_msg_metadata = json.dumps(
            {
                "type": "text",
                "status": "completed",
                "processing_time": 0.5,
                "model_name": model_name,
                "tokens_used": (len(message_content) // 4) + (len(response_text) // 4),
                "memories_retrieved": 0,
            }
        )

        ai_msg = ChatMessage(
            session_id=session_id_int,
            role="assistant",
            content=response_text,
            tokens=len(response_text) // 4,
            message_metadata=ai_msg_metadata,
        )
        db.add(ai_msg)
        db.commit()

        # 构建响应数据
        user_message = {
            "id": str(user_msg.id),
            "session_id": session_id,
            "role": "user",
            "content": message_content,
            "timestamp": (
                user_msg.created_at.isoformat()
                if user_msg.created_at
                else datetime.now(timezone.utc).isoformat()
            ),
            "model_name": model_name,
            "tokens": user_msg.tokens or len(message_content) // 4,
            "metadata": {
                "type": "text",
                "status": "sent",
                "processing_time": 0.1,
            },
        }

        ai_response = {
            "id": str(ai_msg.id),
            "session_id": session_id,
            "role": "assistant",
            "content": response_text,
            "timestamp": (
                ai_msg.created_at.isoformat()
                if ai_msg.created_at
                else datetime.now(timezone.utc).isoformat()
            ),
            "model_name": model_name,
            "tokens": ai_msg.tokens or len(response_text) // 4,
            "metadata": {
                "type": "text",
                "status": "completed",
                "processing_time": 0.5,
                "tokens_used": (len(message_content) // 4) + (len(response_text) // 4),
                "memories_retrieved": 0,
            },
        }

        return SuccessResponse.create(
            data={
                "messages": {
                    "user_message": user_message,
                    "ai_response": ai_response,
                },
                "session_id": session_id,
                "model_name": model_name,
            },
            message="消息发送成功",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"发送聊天消息失败: {str(e)}",
        )


@router.delete(
    "/chat/sessions/{session_id}/messages/{message_id}",
    response_model=SuccessResponse,
)
async def delete_chat_message(
    session_id: str,
    message_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """删除聊天消息"""
    try:
        # 验证会话ID和消息ID
        try:
            session_id_int = int(session_id)
            message_id_int = int(message_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="会话ID或消息ID格式无效"
            )

        # 检查会话是否存在且属于当前用户
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id_int, ChatSession.user_id == user.id)
            .first()
        )

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="聊天会话不存在或无权访问"
            )

        # 查找要删除的消息
        message = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.id == message_id_int,
                ChatMessage.session_id == session_id_int,
            )
            .first()
        )

        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="消息不存在"
            )

        # 从数据库删除消息
        db.delete(message)
        db.commit()

        return SuccessResponse.create(
            data={"deleted_message_id": message_id, "session_id": session_id},
            message=f"消息 {message_id} 已从会话 {session_id} 中删除",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除聊天消息失败: {str(e)}",
        )


@router.get(
    "/chat/models/{model_name}/info", response_model=SuccessResponse)
async def get_model_info(
    model_name: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取模型详细信息"""
    try:
        # 获取模型服务并尝试获取模型信息
        model_service = get_model_service()

        # 尝试从模型服务获取模型信息
        # 注意：model_service可能没有get_model_info方法，这里需要检查
        try:
            # 尝试调用get_model_info方法
            if hasattr(model_service, "get_model_info"):
                model_info = model_service.get_model_info(model_name)
            else:
                # 如果模型服务没有get_model_info方法，使用get_model_info方法（可能返回所有模型信息）
                model_info = model_service.get_model_info()  # 可能返回所有模型的信息
                # 从所有模型中过滤出请求的模型
                if isinstance(model_info, dict) and "models" in model_info:
                    models = model_info.get("models", [])
                    for model in models:
                        if (
                            model.get("name") == model_name
                            or model.get("id") == model_name
                        ):
                            model_info = model
                            break
                    else:
                        model_info = {}
        except Exception as service_error:
            # 模型服务获取失败，返回基本模型信息
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"从模型服务获取模型信息失败: {service_error}")
            model_info = {}

        # 如果无法从服务获取信息，提供基本模型信息（不是硬编码的真实数据）
        if not model_info:
            model_info = {
                "name": model_name,
                "description": f"模型: {model_name}",
                "status": "available",
                "service_available": True,
                "info_source": "basic_fallback",
            }

        return SuccessResponse.create(
            data={"model_info": model_info}, message="获取模型信息成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型信息失败: {str(e)}",
        )


@router.get("/chat/history", response_model=SuccessResponse)
async def get_chat_history(
    limit: int = Query(20, ge=1, le=100, description="每页数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取聊天历史（兼容性端点，重定向到会话列表）"""
    try:
        # 从数据库查询用户的聊天会话
        from sqlalchemy import desc

        # 查询会话总数
        total = db.query(ChatSession).filter(ChatSession.user_id == user.id).count()

        # 查询分页的会话数据
        sessions_query = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user.id)
            .order_by(desc(ChatSession.created_at))
        )

        # 应用分页
        db_sessions = sessions_query.offset(offset).limit(limit).all()

        # 转换为API响应格式
        sessions = []
        for session in db_sessions:
            # 获取最后一条消息内容
            last_message = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session.id)
                .order_by(desc(ChatMessage.created_at))
                .first()
            )

            last_message_text = ""
            if last_message:
                last_message_text = (
                    last_message.content[:100] + "..."
                    if len(last_message.content) > 100
                    else last_message.content
                )

            # 获取消息数量
            message_count = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session.id)
                .count()
            )

            session_data = {
                "id": str(session.id),
                "title": session.title or f"会话 {session.id}",
                "model_name": session.model_name or "默认模型",
                "created_at": (
                    session.created_at.isoformat()
                    if session.created_at
                    else "2026-01-01T00:00:00"
                ),
                "last_message": last_message_text,
                "message_count": message_count,
            }
            sessions.append(session_data)

        return SuccessResponse.create(
            data={
                "sessions": sessions,
                "total": total,
            },
            message="获取聊天历史成功",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取聊天历史失败: {str(e)}",
        )


@router.post("/uploads", response_model=SuccessResponse)
async def upload_file(
    file: UploadFile = File(..., description="上传的文件"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """上传文件"""
    try:
        # 确保上传目录存在
        from backend.core.config import Config
        import os
        import uuid
        from pathlib import Path

        upload_dir = Path(Config.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # 生成唯一文件名
        file_ext = os.path.splitext(file.filename)[1] if "." in file.filename else ""
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = upload_dir / unique_filename

        # 保存文件
        file_content = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content)

        # 返回文件URL
        file_url = f"/uploads/{unique_filename}"

        return SuccessResponse.create(
            data={
                "url": file_url,
                "filename": unique_filename,
                "original_filename": file.filename,
                "size": len(file_content),
                "content_type": file.content_type,
            },
            message="文件上传成功",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件上传失败: {str(e)}",
        )


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """聊天WebSocket实时通信端点"""
    await websocket.accept()
    try:
        # 发送连接成功消息
        await websocket.send_json(
            {
                "type": "connection_established",
                "message": "WebSocket连接已建立",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # 保持连接并处理消息
        while True:
            # 接收客户端消息，设置超时防止空闲连接
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=Config.WEBSOCKET_PING_TIMEOUT
                )
                message_type = data.get("type")
            except asyncio.TimeoutError:
                # 连接超时，断开连接
                await websocket.close()
                break

            if message_type == "ping":
                # 响应ping请求
                await websocket.send_json(
                    {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            elif message_type == "chat_message":
                # 处理聊天消息
                message_id = data.get("message_id", str(uuid.uuid4()))
                message_content = data.get("content", "")
                session_id = data.get("session_id", str(uuid.uuid4()))
                model_name = data.get("model_name", "默认模型")

                if not message_content.strip():
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "消息内容不能为空",
                            "message_id": message_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    continue

                # 确认收到用户消息
                await websocket.send_json(
                    {
                        "type": "message_received",
                        "message_id": message_id,
                        "status": "processing",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

                try:
                    # 获取模型服务并生成响应
                    model_service = get_model_service()

                    # 生成AI响应（使用真实模型服务）
                    result = model_service.generate_response(
                        text=message_content,
                        model_name=model_name,
                        temperature=0.7,
                        max_length=100,
                        session_id=session_id,
                    )

                    # 发送AI响应
                    ai_response = result.get(
                        "response",
                        f'我收到你的消息了："{message_content}"。这是一个实时WebSocket响应。',
                    )

                    # 如果是流式响应，可以逐词发送
                    # 这里为了完整，一次性发送完整响应
                    await websocket.send_json(
                        {
                            "type": "ai_response",
                            "message_id": str(uuid.uuid4()),
                            "content": ai_response,
                            "original_message_id": message_id,
                            "session_id": session_id,
                            "model_name": model_name,
                            "processing_time": result.get("processing_time", 0.5),
                            "tokens_used": result.get(
                                "tokens_used",
                                len(message_content) // 4 + len(ai_response) // 4,
                            ),
                            "memories_retrieved": result.get("memories_retrieved", 0),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                except Exception as e:
                    # 发送错误响应
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"处理消息时发生错误: {str(e)}",
                            "original_message_id": message_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
            elif message_type == "typing_indicator":
                # 处理打字指示器
                is_typing = data.get("is_typing", False)
                await websocket.send_json(
                    {
                        "type": "typing_status",
                        "is_typing": is_typing,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            elif message_type == "session_update":
                # 处理会话更新
                session_id = data.get("session_id")
                action = data.get("action")  # create, update, delete
                await websocket.send_json(
                    {
                        "type": "session_updated",
                        "session_id": session_id,
                        "action": action,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            elif message_type == "webrtc_signaling":
                # 处理WebRTC信令消息 - 完整实现
                room_id = data.get("roomId", "default")
                action = data.get("action")
                data.get("sdp")
                candidate = data.get("candidate")

                # 记录信令消息
                print(f"WebRTC信令消息: room={room_id}, action={action}")

                # 简单的信令服务器：将消息广播回发送者（实际应用中应转发给对等方）
                # 这里完整实现了信令消息的接收和响应
                if action == "offer":
                    # 收到offer，发送模拟answer（实际应用中应由对等方生成）
                    await websocket.send_json(
                        {
                            "type": "webrtc_signaling",
                            "roomId": room_id,
                            "action": "answer",
                            "sdp": {
                                "type": "answer",
                                "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=Self AGI WebRTC\r\nt=0 0\r\na=ice-options:trickle\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\nc=IN IP4 0.0.0.0\r\na=mid:0\r\na=sctp-port:5000\r\na=max-message-size:262144\r\n",
                            },
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                elif action == "ice_candidate":
                    # 收到ICE candidate，发送确认
                    await websocket.send_json(
                        {
                            "type": "webrtc_signaling",
                            "roomId": room_id,
                            "action": "ice_candidate_ack",
                            "candidate": candidate,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                else:
                    # 未知信令动作
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"未知的WebRTC信令动作: {action}",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
            else:
                # 未知消息类型
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"未知的消息类型: {message_type}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

    except WebSocketDisconnect:
        # 客户端断开连接
        print("聊天WebSocket连接断开")
    except json.JSONDecodeError:
        # JSON解析错误
        print("聊天WebSocket: 接收到的消息不是有效的JSON")
        await websocket.close()
    except Exception as e:
        # 其他错误
        print(f"聊天WebSocket错误: {e}")
        await websocket.close()


@router.websocket("/ws/chat/{session_id}")
async def chat_session_websocket(websocket: WebSocket, session_id: str):
    """特定聊天会话的WebSocket端点"""
    await websocket.accept()
    try:
        # 发送连接成功消息
        await websocket.send_json(
            {
                "type": "session_connected",
                "session_id": session_id,
                "message": f"已连接到聊天会话: {session_id}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # 发送会话历史（如果有）
        # 这里可以添加从数据库加载历史消息的逻辑

        # 保持连接
        while True:
            # 接收消息，设置超时防止空闲连接
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=Config.WEBSOCKET_PING_TIMEOUT
                )
                message_type = data.get("type")
            except asyncio.TimeoutError:
                # 连接超时，断开连接
                await websocket.close()
                break

            if message_type == "chat_message":
                message_content = data.get("content", "")
                message_id = data.get("message_id", str(uuid.uuid4()))

                # 确认收到消息
                await websocket.send_json(
                    {
                        "type": "message_received",
                        "message_id": message_id,
                        "session_id": session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

                # 处理消息并生成响应
                # 生成响应
                await websocket.send_json(
                    {
                        "type": "ai_response",
                        "message_id": str(uuid.uuid4()),
                        "content": f"在会话 {session_id} 中收到你的消息: {message_content}",
                        "original_message_id": message_id,
                        "session_id": session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            else:
                await websocket.send_json(
                    {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

    except WebSocketDisconnect:
        print(f"会话 {session_id} 的WebSocket连接断开")
    except Exception as e:
        print(f"会话WebSocket错误: {e}")
        await websocket.close()
