"""
示范学习路由模块
处理示范数据录制、回放、管理和训练的API请求
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Query,
    WebSocket,
    WebSocketDisconnect,
    Body,
)
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging
import time
from pydantic import BaseModel

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.db_models.robot import Robot
from backend.db_models.demonstration import (
    Demonstration,
    DemonstrationType,
    DemonstrationStatus,
)
from backend.services.demonstration_service import (
    DemonstrationManager,
    FrameData,
)

logger = logging.getLogger(__name__)


# Pydantic模型
class FrameRequest(BaseModel):
    """帧数据请求模型"""

    joint_positions: Dict[str, float] = {}
    joint_velocities: Optional[Dict[str, float]] = None
    joint_torques: Optional[Dict[str, float]] = None
    control_commands: Optional[Dict[str, Any]] = None
    sensor_data: Optional[Dict[str, Any]] = None
    imu_data: Optional[Dict[str, Any]] = None
    camera_data: Optional[str] = None
    camera_format: Optional[str] = "jpeg"
    environment_state: Optional[Dict[str, Any]] = None


router = APIRouter(prefix="/api/demonstration", tags=["示范学习"])

# 示范管理器实例
demonstration_manager = DemonstrationManager()


@router.get("/list", response_model=Dict[str, Any])
async def list_demonstrations(
    robot_id: Optional[int] = Query(None, description="机器人ID"),
    demonstration_type: Optional[DemonstrationType] = Query(
        None, description="示范类型"
    ),
    status_filter: Optional[DemonstrationStatus] = Query(None, description="状态过滤"),
    skip: int = Query(0, ge=0, description="跳过记录数"),
    limit: int = Query(100, ge=1, le=1000, description="返回记录数"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取示范列表"""
    try:
        query = db.query(Demonstration).filter(Demonstration.user_id == user.id)

        if robot_id:
            query = query.filter(Demonstration.robot_id == robot_id)

        if demonstration_type:
            query = query.filter(Demonstration.demonstration_type == demonstration_type)

        if status_filter:
            query = query.filter(Demonstration.status == status_filter)

        # 按创建时间降序排序
        query = query.order_by(Demonstration.created_at.desc())

        total = query.count()
        demonstrations = query.offset(skip).limit(limit).all()

        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "total": total,
                "demonstrations": [d.to_dict() for d in demonstrations],
                "pagination": {
                    "skip": skip,
                    "limit": limit,
                    "has_more": skip + len(demonstrations) < total,
                },
            },
        }
    except Exception as e:
        logger.error(f"获取示范列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取示范列表失败: {str(e)}",
        )


@router.get("/{demonstration_id}", response_model=Dict[str, Any])
async def get_demonstration(
    demonstration_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取示范详情"""
    try:
        demonstration = (
            db.query(Demonstration)
            .filter(
                Demonstration.id == demonstration_id, Demonstration.user_id == user.id
            )
            .first()
        )

        if not demonstration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"示范 {demonstration_id} 不存在或无权访问",
            )

        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": demonstration.to_dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取示范详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取示范详情失败: {str(e)}",
        )


@router.post("/create", response_model=Dict[str, Any])
async def create_demonstration(
    name: str = Query(..., description="示范名称"),
    description: str = Query("", description="示范描述"),
    robot_id: int = Query(..., description="机器人ID"),
    demonstration_type: DemonstrationType = Query(
        DemonstrationType.JOINT_CONTROL, description="示范类型"
    ),
    config: Optional[Dict[str, Any]] = Body(None, description="示范配置"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """创建示范（开始录制准备）"""
    try:
        # 验证机器人是否存在且属于用户
        robot = (
            db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"机器人 {robot_id} 不存在或无权访问",
            )

        # 创建示范记录
        demonstration = Demonstration(
            name=name,
            description=description,
            demonstration_type=demonstration_type,
            robot_id=robot_id,
            user_id=user.id,
            status=DemonstrationStatus.RECORDING,
            config=config or {},
            recorded_at=datetime.now(timezone.utc),
        )

        db.add(demonstration)
        db.commit()
        db.refresh(demonstration)

        # 创建录制器
        recorder = demonstration_manager.create_recorder(db, robot_id, user.id)

        logger.info(f"创建示范成功: {name} (ID: {demonstration.id})")

        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "demonstration": demonstration.to_dict(),
                "recorder_status": recorder.get_status(),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建示范失败: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建示范失败: {str(e)}",
        )


@router.post("/{demonstration_id}/start", response_model=Dict[str, Any])
async def start_recording(
    demonstration_id: int,
    config: Optional[Dict[str, Any]] = Body(None, description="录制配置"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """开始录制示范数据"""
    try:
        # 获取示范记录
        demonstration = (
            db.query(Demonstration)
            .filter(
                Demonstration.id == demonstration_id, Demonstration.user_id == user.id
            )
            .first()
        )

        if not demonstration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"示范 {demonstration_id} 不存在或无权访问",
            )

        if demonstration.status != DemonstrationStatus.RECORDING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"示范状态不支持录制: {demonstration.status.value}",
            )

        # 创建或获取录制器
        recorder = demonstration_manager.get_recorder(demonstration_id)
        if not recorder:
            recorder = demonstration_manager.create_recorder(
                db, demonstration.robot_id, user.id
            )
            demonstration_manager.register_recorder(demonstration_id, recorder)

        # 开始录制
        result = recorder.start_recording(
            name=demonstration.name,
            description=demonstration.description,
            demonstration_type=demonstration.demonstration_type,
            config=config or demonstration.config,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="开始录制失败"
            )

        # 更新示范记录
        demonstration.recorder_id = demonstration_id
        db.add(demonstration)
        db.commit()

        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "demonstration": demonstration.to_dict(),
                "recorder_status": recorder.get_status(),
                "message": "录制已开始",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"开始录制失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"开始录制失败: {str(e)}",
        )


@router.post("/{demonstration_id}/pause", response_model=Dict[str, Any])
async def pause_recording(
    demonstration_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """暂停录制"""
    try:
        # 获取录制器
        recorder = demonstration_manager.get_recorder(demonstration_id)
        if not recorder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"录制器不存在: {demonstration_id}",
            )

        success = recorder.pause_recording()

        return {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "recorder_status": recorder.get_status(),
                "message": "录制已暂停" if success else "暂停失败",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停录制失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"暂停录制失败: {str(e)}",
        )


@router.post("/{demonstration_id}/resume", response_model=Dict[str, Any])
async def resume_recording(
    demonstration_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """恢复录制"""
    try:
        recorder = demonstration_manager.get_recorder(demonstration_id)
        if not recorder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"录制器不存在: {demonstration_id}",
            )

        success = recorder.resume_recording()

        return {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "recorder_status": recorder.get_status(),
                "message": "录制已恢复" if success else "恢复失败",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复录制失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"恢复录制失败: {str(e)}",
        )


@router.post("/{demonstration_id}/stop", response_model=Dict[str, Any])
async def stop_recording(
    demonstration_id: int,
    save: bool = Query(True, description="是否保存数据"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """停止录制"""
    try:
        recorder = demonstration_manager.get_recorder(demonstration_id)
        if not recorder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"录制器不存在: {demonstration_id}",
            )

        # 停止录制
        result = recorder.stop_recording(save=save)

        # 更新示范记录
        if result and save:
            demonstration = (
                db.query(Demonstration)
                .filter(Demonstration.id == demonstration_id)
                .first()
            )

            if demonstration:
                demonstration.status = DemonstrationStatus.COMPLETED
                demonstration.progress = 1.0
                demonstration.updated_at = datetime.now(timezone.utc)
                db.add(demonstration)
                db.commit()

        # 注销录制器
        demonstration_manager.unregister_recorder(demonstration_id)

        return {
            "success": result is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "demonstration": result.to_dict() if result else None,
                "message": "录制已停止并保存" if save else "录制已停止，数据未保存",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止录制失败: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止录制失败: {str(e)}",
        )


@router.post("/{demonstration_id}/record-frame", response_model=Dict[str, Any])
async def record_frame(
    demonstration_id: int,
    frame_request: FrameRequest = Body(..., description="帧数据"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """录制一帧数据"""
    try:
        recorder = demonstration_manager.get_recorder(demonstration_id)
        if not recorder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"录制器不存在: {demonstration_id}",
            )

        # 解码相机数据（如果需要）
        camera_bytes = None
        if frame_request.camera_data:
            import base64

            camera_bytes = base64.b64decode(frame_request.camera_data)

        # 创建帧数据
        frame_data = FrameData(
            timestamp=time.time(),
            joint_positions=frame_request.joint_positions,
            joint_velocities=frame_request.joint_velocities,
            joint_torques=frame_request.joint_torques,
            control_commands=frame_request.control_commands,
            sensor_data=frame_request.sensor_data,
            imu_data=frame_request.imu_data,
            camera_data=camera_bytes,
            camera_format=frame_request.camera_format,
            environment_state=frame_request.environment_state,
        )

        # 录制帧
        success = recorder.record_frame(frame_data)

        return {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "recorder_status": recorder.get_status(),
                "frame_count": recorder.frame_count,
                "message": "帧录制成功" if success else "帧录制失败",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"录制帧失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"录制帧失败: {str(e)}",
        )


@router.post("/{demonstration_id}/play", response_model=Dict[str, Any])
async def start_playback(
    demonstration_id: int,
    start_frame: int = Query(0, ge=0, description="起始帧索引"),
    playback_speed: float = Query(1.0, ge=0.1, le=10.0, description="播放速度"),
    loop: bool = Query(False, description="是否循环播放"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """开始播放示范"""
    try:
        # 验证示范存在且属于用户
        demonstration = (
            db.query(Demonstration)
            .filter(
                Demonstration.id == demonstration_id,
                Demonstration.user_id == user.id,
                Demonstration.status == DemonstrationStatus.COMPLETED,
            )
            .first()
        )

        if not demonstration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"示范不存在或未完成: {demonstration_id}",
            )

        # 创建播放器
        player = demonstration_manager.create_player(db, demonstration_id)
        success = player.load_demonstration()

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="加载示范失败"
            )

        # 开始播放
        success = player.start_playback(start_frame, playback_speed, loop)

        if success:
            demonstration_manager.register_player(demonstration_id, player)

        return {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "player_status": player.get_status(),
                "message": "播放已开始" if success else "播放开始失败",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"开始播放失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"开始播放失败: {str(e)}",
        )


@router.get("/{demonstration_id}/current-frame", response_model=Dict[str, Any])
async def get_current_frame(
    demonstration_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取当前播放帧"""
    try:
        player = demonstration_manager.get_player(demonstration_id)
        if not player:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"播放器不存在: {demonstration_id}",
            )

        # 更新帧索引
        player.update_frame_index()

        # 获取当前帧数据
        frame_data = player.get_current_frame()

        return {
            "success": frame_data is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "frame_data": frame_data.__dict__ if frame_data else None,
                "player_status": player.get_status(),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取当前帧失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取当前帧失败: {str(e)}",
        )


@router.post("/{demonstration_id}/pause-playback", response_model=Dict[str, Any])
async def pause_playback(
    demonstration_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """暂停播放"""
    try:
        player = demonstration_manager.get_player(demonstration_id)
        if not player:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"播放器不存在: {demonstration_id}",
            )

        success = player.pause_playback()

        return {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "player_status": player.get_status(),
                "message": "播放已暂停" if success else "暂停失败",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停播放失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"暂停播放失败: {str(e)}",
        )


@router.post("/{demonstration_id}/resume-playback", response_model=Dict[str, Any])
async def resume_playback(
    demonstration_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """恢复播放"""
    try:
        player = demonstration_manager.get_player(demonstration_id)
        if not player:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"播放器不存在: {demonstration_id}",
            )

        success = player.resume_playback()

        return {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "player_status": player.get_status(),
                "message": "播放已恢复" if success else "恢复失败",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复播放失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"恢复播放失败: {str(e)}",
        )


@router.post("/{demonstration_id}/stop-playback", response_model=Dict[str, Any])
async def stop_playback(
    demonstration_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """停止播放"""
    try:
        player = demonstration_manager.get_player(demonstration_id)
        if not player:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"播放器不存在: {demonstration_id}",
            )

        success = player.stop_playback()

        # 注销播放器
        demonstration_manager.unregister_player(demonstration_id)

        return {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"message": "播放已停止"},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止播放失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止播放失败: {str(e)}",
        )


@router.delete("/{demonstration_id}", response_model=Dict[str, Any])
async def delete_demonstration(
    demonstration_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """删除示范"""
    try:
        demonstration = (
            db.query(Demonstration)
            .filter(
                Demonstration.id == demonstration_id, Demonstration.user_id == user.id
            )
            .first()
        )

        if not demonstration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"示范 {demonstration_id} 不存在或无权访问",
            )

        # 删除相关数据（级联删除）
        db.delete(demonstration)
        db.commit()

        # 注销相关的录制器和播放器
        demonstration_manager.unregister_recorder(demonstration_id)
        demonstration_manager.unregister_player(demonstration_id)

        logger.info(f"删除示范成功: {demonstration.name} (ID: {demonstration_id})")

        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"message": f"示范 {demonstration_id} 已删除"},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除示范失败: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除示范失败: {str(e)}",
        )


@router.websocket("/{demonstration_id}/ws")
async def demonstration_websocket(
    websocket: WebSocket,
    demonstration_id: int,
    db: Session = Depends(get_db),
):
    """示范数据WebSocket接口"""
    await websocket.accept()

    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "record_frame":
                # 录制帧数据
                recorder = demonstration_manager.get_recorder(demonstration_id)
                if recorder:
                    frame_data = FrameData(
                        timestamp=time.time(),
                        joint_positions=data.get("joint_positions", {}),
                        joint_velocities=data.get("joint_velocities"),
                        joint_torques=data.get("joint_torques"),
                        control_commands=data.get("control_commands"),
                        sensor_data=data.get("sensor_data"),
                        imu_data=data.get("imu_data"),
                        camera_data=data.get("camera_data"),
                        camera_format=data.get("camera_format"),
                        environment_state=data.get("environment_state"),
                    )

                    success = recorder.record_frame(frame_data)

                    await websocket.send_json(
                        {
                            "type": "frame_recorded",
                            "success": success,
                            "frame_count": recorder.frame_count,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

            elif message_type == "get_current_frame":
                # 获取当前播放帧
                player = demonstration_manager.get_player(demonstration_id)
                if player:
                    player.update_frame_index()
                    frame_data = player.get_current_frame()

                    await websocket.send_json(
                        {
                            "type": "current_frame",
                            "frame_data": frame_data.__dict__ if frame_data else None,
                            "player_status": player.get_status(),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

            elif message_type == "control":
                # 控制命令
                action = data.get("action")
                if action == "pause_recording":
                    recorder = demonstration_manager.get_recorder(demonstration_id)
                    if recorder:
                        success = recorder.pause_recording()
                        await websocket.send_json(
                            {
                                "type": "control_response",
                                "action": action,
                                "success": success,
                            }
                        )

                elif action == "resume_recording":
                    recorder = demonstration_manager.get_recorder(demonstration_id)
                    if recorder:
                        success = recorder.resume_recording()
                        await websocket.send_json(
                            {
                                "type": "control_response",
                                "action": action,
                                "success": success,
                            }
                        )

                elif action == "pause_playback":
                    player = demonstration_manager.get_player(demonstration_id)
                    if player:
                        success = player.pause_playback()
                        await websocket.send_json(
                            {
                                "type": "control_response",
                                "action": action,
                                "success": success,
                            }
                        )

                elif action == "resume_playback":
                    player = demonstration_manager.get_player(demonstration_id)
                    if player:
                        success = player.resume_playback()
                        await websocket.send_json(
                            {
                                "type": "control_response",
                                "action": action,
                                "success": success,
                            }
                        )

    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: demonstration_id={demonstration_id}")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        await websocket.close()


# 添加time模块导入
