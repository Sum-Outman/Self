"""
示范学习服务
处理示范数据的录制、回放、处理和分析
"""

import logging
import time
from typing import Dict, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session

from ..db_models.demonstration import (
    Demonstration,
    DemonstrationFrame,
    CameraFrame,
    DemonstrationType,
    DemonstrationStatus,
)

logger = logging.getLogger(__name__)


class RecordingMode(Enum):
    """录制模式枚举"""

    MANUAL = "manual"  # 手动录制
    AUTO_TRIGGER = "auto_trigger"  # 自动触发录制
    CONTINUOUS = "continuous"  # 连续录制
    EVENT_BASED = "event_based"  # 事件驱动录制


@dataclass
class FrameData:
    """帧数据结构"""

    timestamp: float
    joint_positions: Dict[str, float]
    joint_velocities: Optional[Dict[str, float]] = None
    joint_torques: Optional[Dict[str, float]] = None
    control_commands: Optional[Dict[str, Any]] = None
    sensor_data: Optional[Dict[str, Any]] = None
    imu_data: Optional[Dict[str, Any]] = None
    camera_data: Optional[bytes] = None
    camera_format: Optional[str] = None
    environment_state: Optional[Dict[str, Any]] = None


class DemonstrationRecorder:
    """示范录制器"""

    def __init__(self, db: Session, robot_id: int, user_id: int):
        """
        初始化示范录制器

        参数:
            db: 数据库会话
            robot_id: 机器人ID
            user_id: 用户ID
        """
        self.db = db
        self.robot_id = robot_id
        self.user_id = user_id
        self.recording = False
        self.paused = False
        self.demonstration = None
        self.frames = []
        self.start_time = None
        self.frame_count = 0
        self.sampling_interval = 0.033  # 默认30Hz采样率
        self.last_sample_time = 0
        self.config = {}

    def start_recording(
        self,
        name: str,
        description: str = "",
        demonstration_type: DemonstrationType = DemonstrationType.JOINT_CONTROL,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Demonstration]:
        """
        开始录制示范

        参数:
            name: 示范名称
            description: 示范描述
            demonstration_type: 示范类型
            config: 录制配置

        返回:
            创建的Demonstration对象或None
        """
        try:
            if self.recording:
                logger.warning("已经在录制中")
                return self.demonstration

            # 创建示范记录
            self.demonstration = Demonstration(
                name=name,
                description=description,
                demonstration_type=demonstration_type,
                robot_id=self.robot_id,
                user_id=self.user_id,
                status=DemonstrationStatus.RECORDING,
                config=config or {},
                recorded_at=datetime.now(timezone.utc),
            )

            self.db.add(self.demonstration)
            self.db.flush()  # 获取ID

            # 初始化录制状态
            self.recording = True
            self.paused = False
            self.frames = []
            self.start_time = time.time()
            self.frame_count = 0
            self.last_sample_time = 0

            # 从配置中获取采样率
            if config and "sampling_rate" in config:
                sampling_rate = config["sampling_rate"]
                if sampling_rate > 0:
                    self.sampling_interval = 1.0 / sampling_rate

            logger.info(f"开始录制示范: {name} (ID: {self.demonstration.id})")
            return self.demonstration

        except Exception as e:
            logger.error(f"开始录制失败: {e}")
            self.db.rollback()
            return None  # 返回None

    def pause_recording(self) -> bool:
        """暂停录制"""
        if not self.recording:
            logger.warning("未在录制中，无法暂停")
            return False

        self.paused = True
        logger.info("录制已暂停")
        return True

    def resume_recording(self) -> bool:
        """恢复录制"""
        if not self.recording:
            logger.warning("未在录制中，无法恢复")
            return False

        self.paused = False
        logger.info("录制已恢复")
        return True

    def record_frame(self, frame_data: FrameData) -> bool:
        """
        录制一帧数据

        参数:
            frame_data: 帧数据

        返回:
            是否成功
        """
        try:
            if not self.recording or self.paused:
                return False

            # 检查是否达到采样间隔
            current_time = time.time()
            elapsed = current_time - self.last_sample_time
            if elapsed < self.sampling_interval:
                return True  # 跳过这帧

            self.last_sample_time = current_time

            # 创建帧记录
            frame = DemonstrationFrame(
                demonstration_id=self.demonstration.id,
                frame_index=self.frame_count,
                timestamp=frame_data.timestamp,
                relative_time=frame_data.timestamp - self.start_time,
                joint_positions=frame_data.joint_positions,
                joint_velocities=frame_data.joint_velocities,
                joint_torques=frame_data.joint_torques,
                control_commands=frame_data.control_commands,
                sensor_data=frame_data.sensor_data,
                imu_data=frame_data.imu_data,
                environment_state=frame_data.environment_state,
            )

            self.db.add(frame)
            self.frame_count += 1

            # 如果有相机数据，创建相机帧
            if frame_data.camera_data:
                camera_frame = CameraFrame(
                    demonstration_id=self.demonstration.id,
                    image_data=frame_data.camera_data,
                    image_format=frame_data.camera_format or "jpeg",
                    timestamp=frame_data.timestamp,
                )
                self.db.add(camera_frame)
                self.db.flush()  # 获取相机帧ID
                frame.camera_data_ref = camera_frame.id

            # 定期提交，避免事务过大
            if self.frame_count % 50 == 0:
                self.db.commit()
                logger.debug(f"已录制 {self.frame_count} 帧")

            return True

        except Exception as e:
            logger.error(f"录制帧失败: {e}")
            self.db.rollback()
            return False

    def stop_recording(self, save: bool = True) -> Optional[Demonstration]:
        """
        停止录制

        参数:
            save: 是否保存数据

        返回:
            更新后的Demonstration对象或None
        """
        try:
            if not self.recording:
                logger.warning("未在录制中，无法停止")
                return None  # 返回None

            self.recording = False
            self.paused = False

            if save and self.demonstration:
                # 更新示范记录
                recording_duration = time.time() - self.start_time
                self.demonstration.recording_duration = recording_duration
                self.demonstration.frame_count = self.frame_count
                self.demonstration.data_size = self._calculate_data_size()
                self.demonstration.status = DemonstrationStatus.COMPLETED
                self.demonstration.progress = 1.0
                self.demonstration.updated_at = datetime.now(timezone.utc)

                # 提交所有更改
                self.db.commit()

                logger.info(
                    f"录制完成: {self.demonstration.name}, "
                    f"时长: {recording_duration:.2f}s, "
                    f"帧数: {self.frame_count}"
                )

                return self.demonstration
            else:
                # 不保存数据，回滚
                self.db.rollback()
                logger.info("录制已取消，数据未保存")
                return None  # 返回None

        except Exception as e:
            logger.error(f"停止录制失败: {e}")
            self.db.rollback()
            return None  # 返回None

    def _calculate_data_size(self) -> int:
        """估算数据大小"""
        # 简单的估算：每帧大约1KB
        return self.frame_count * 1024

    def get_status(self) -> Dict[str, Any]:
        """获取录制器状态"""
        return {
            "recording": self.recording,
            "paused": self.paused,
            "frame_count": self.frame_count,
            "recording_duration": (
                time.time() - self.start_time if self.start_time else 0
            ),
            "demonstration_id": self.demonstration.id if self.demonstration else None,
            "sampling_interval": self.sampling_interval,
            "sampling_rate": (
                1.0 / self.sampling_interval if self.sampling_interval > 0 else 0
            ),
        }


class DemonstrationPlayer:
    """示范播放器"""

    def __init__(self, db: Session, demonstration_id: int):
        """
        初始化示范播放器

        参数:
            db: 数据库会话
            demonstration_id: 示范ID
        """
        self.db = db
        self.demonstration_id = demonstration_id
        self.playing = False
        self.paused = False
        self.current_frame_index = 0
        self.playback_speed = 1.0
        self.loop = False
        self.demonstration = None
        self.frames = []
        self.total_frames = 0
        self.start_time = None

    def load_demonstration(self) -> bool:
        """加载示范数据"""
        try:
            # 加载示范
            self.demonstration = (
                self.db.query(Demonstration)
                .filter(
                    Demonstration.id == self.demonstration_id,
                    Demonstration.status == DemonstrationStatus.COMPLETED,
                )
                .first()
            )

            if not self.demonstration:
                logger.error(f"示范不存在或未完成: {self.demonstration_id}")
                return False

            # 加载帧数据
            self.frames = (
                self.db.query(DemonstrationFrame)
                .filter(DemonstrationFrame.demonstration_id == self.demonstration_id)
                .order_by(DemonstrationFrame.frame_index)
                .all()
            )

            self.total_frames = len(self.frames)

            if self.total_frames == 0:
                logger.error(f"示范没有帧数据: {self.demonstration_id}")
                return False

            logger.info(
                f"加载示范成功: {self.demonstration.name}, 帧数: {self.total_frames}"
            )
            return True

        except Exception as e:
            logger.error(f"加载示范失败: {e}")
            return False

    def start_playback(
        self, start_frame: int = 0, playback_speed: float = 1.0, loop: bool = False
    ) -> bool:
        """
        开始播放

        参数:
            start_frame: 起始帧索引
            playback_speed: 播放速度
            loop: 是否循环播放

        返回:
            是否成功
        """
        try:
            if not self.demonstration:
                if not self.load_demonstration():
                    return False

            if self.playing:
                logger.warning("已经在播放中")
                return False

            self.playing = True
            self.paused = False
            self.current_frame_index = max(0, min(start_frame, self.total_frames - 1))
            self.playback_speed = max(0.1, min(playback_speed, 10.0))  # 限制速度范围
            self.loop = loop
            self.start_time = time.time()

            logger.info(
                f"开始播放示范: {self.demonstration.name}, "
                f"起始帧: {self.current_frame_index}, "
                f"速度: {self.playback_speed}x"
            )

            return True

        except Exception as e:
            logger.error(f"开始播放失败: {e}")
            return False

    def pause_playback(self) -> bool:
        """暂停播放"""
        if not self.playing:
            logger.warning("未在播放中，无法暂停")
            return False

        self.paused = True
        logger.info("播放已暂停")
        return True

    def resume_playback(self) -> bool:
        """恢复播放"""
        if not self.playing:
            logger.warning("未在播放中，无法恢复")
            return False

        self.paused = False
        self.start_time = time.time()  # 重置开始时间
        logger.info("播放已恢复")
        return True

    def stop_playback(self) -> bool:
        """停止播放"""
        if not self.playing:
            logger.warning("未在播放中，无法停止")
            return False

        self.playing = False
        self.paused = False
        self.current_frame_index = 0
        logger.info("播放已停止")
        return True

    def get_current_frame(self) -> Optional[FrameData]:
        """获取当前帧数据"""
        if not self.playing or self.paused:
            return None  # 返回None

        if self.current_frame_index >= self.total_frames:
            if self.loop:
                self.current_frame_index = 0
                self.start_time = time.time()
            else:
                self.stop_playback()
                return None  # 返回None

        if self.current_frame_index < len(self.frames):
            frame = self.frames[self.current_frame_index]

            # 转换为FrameData格式
            frame_data = FrameData(
                timestamp=frame.timestamp,
                joint_positions=frame.joint_positions or {},
                joint_velocities=frame.joint_velocities,
                joint_torques=frame.joint_torques,
                control_commands=frame.control_commands,
                sensor_data=frame.sensor_data,
                imu_data=frame.imu_data,
                environment_state=frame.environment_state,
            )

            return frame_data

        return None  # 返回None

    def update_frame_index(self):
        """更新当前帧索引（基于时间）"""
        if not self.playing or self.paused:
            return

        current_time = time.time()
        elapsed_time = (current_time - self.start_time) * self.playback_speed

        # 根据时间计算应该播放的帧
        # 假设原始录制是均匀采样的
        if self.total_frames > 0 and self.demonstration:
            total_duration = self.demonstration.recording_duration or 1.0
            target_frame = int(elapsed_time / total_duration * self.total_frames)
            self.current_frame_index = min(target_frame, self.total_frames - 1)

    def set_frame(self, frame_index: int) -> bool:
        """
        跳转到指定帧

        参数:
            frame_index: 帧索引

        返回:
            是否成功
        """
        if not self.demonstration:
            return False

        if 0 <= frame_index < self.total_frames:
            self.current_frame_index = frame_index
            if self.playing:
                # 更新开始时间以保持同步
                if self.demonstration.recording_duration:
                    target_time = (
                        frame_index
                        / self.total_frames
                        * self.demonstration.recording_duration
                    )
                    self.start_time = time.time() - target_time / self.playback_speed

            return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """获取播放器状态"""
        if not self.demonstration:
            return {"loaded": False}

        return {
            "loaded": True,
            "playing": self.playing,
            "paused": self.paused,
            "current_frame": self.current_frame_index,
            "total_frames": self.total_frames,
            "playback_speed": self.playback_speed,
            "loop": self.loop,
            "demonstration": {
                "id": self.demonstration.id,
                "name": self.demonstration.name,
                "duration": self.demonstration.recording_duration or 0,
                "frame_count": self.demonstration.frame_count,
            },
        }


class DemonstrationManager:
    """示范管理器（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.recorders = {}  # demonstration_id -> recorder
            cls._instance.players = {}  # demonstration_id -> player
        return cls._instance

    def create_recorder(
        self, db: Session, robot_id: int, user_id: int
    ) -> DemonstrationRecorder:
        """创建示范录制器"""
        recorder = DemonstrationRecorder(db, robot_id, user_id)
        return recorder

    def create_player(self, db: Session, demonstration_id: int) -> DemonstrationPlayer:
        """创建示范播放器"""
        player = DemonstrationPlayer(db, demonstration_id)
        return player

    def get_recorder(self, demonstration_id: int) -> Optional[DemonstrationRecorder]:
        """获取示范录制器"""
        return self.recorders.get(demonstration_id)

    def get_player(self, demonstration_id: int) -> Optional[DemonstrationPlayer]:
        """获取示范播放器"""
        return self.players.get(demonstration_id)

    def register_recorder(self, demonstration_id: int, recorder: DemonstrationRecorder):
        """注册示范录制器"""
        self.recorders[demonstration_id] = recorder

    def register_player(self, demonstration_id: int, player: DemonstrationPlayer):
        """注册示范播放器"""
        self.players[demonstration_id] = player

    def unregister_recorder(self, demonstration_id: int):
        """注销示范录制器"""
        if demonstration_id in self.recorders:
            del self.recorders[demonstration_id]

    def unregister_player(self, demonstration_id: int):
        """注销示范播放器"""
        if demonstration_id in self.players:
            del self.players[demonstration_id]
