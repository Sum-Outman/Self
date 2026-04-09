"""
示范学习数据库模型
包含示范数据录制、回放、训练等相关的数据库模型
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Text,
    ForeignKey,
    JSON,
    LargeBinary,
    Enum as SQLEnum,
    Table,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum as PyEnum

from ..core.database import Base


class DemonstrationType(PyEnum):
    """示范类型枚举"""

    JOINT_CONTROL = "joint_control"  # 关节控制示范
    TRAJECTORY_FOLLOWING = "trajectory_following"  # 轨迹跟踪示范
    TASK_EXECUTION = "task_execution"  # 任务执行示范
    MANUAL_CONTROL = "manual_control"  # 手动控制示范
    AUTONOMOUS_LEARNING = "autonomous_learning"  # 自主学习示范


class DemonstrationStatus(PyEnum):
    """示范状态枚举"""

    RECORDING = "recording"  # 录制中
    PAUSED = "paused"  # 已暂停
    COMPLETED = "completed"  # 已完成
    PROCESSING = "processing"  # 处理中
    TRAINING = "training"  # 训练中
    READY = "ready"  # 就绪
    ERROR = "error"  # 错误


class DemonstrationFormat(PyEnum):
    """示范数据格式枚举"""

    JSON_RAW = "json_raw"  # 原始JSON格式
    BINARY_COMPRESSED = "binary_compressed"  # 二进制压缩格式
    NPZ_ARRAY = "npz_array"  # NumPy数组格式
    ROS_BAG = "ros_bag"  # ROS bag格式


class Demonstration(Base):
    """示范主表"""

    __tablename__ = "demonstrations"

    id = Column(Integer, primary_key=True, index=True)

    # 基本信息
    name = Column(String(255), nullable=False, comment="示范名称")
    description = Column(Text, comment="示范描述")
    demonstration_type = Column(
        SQLEnum(DemonstrationType), nullable=False, comment="示范类型"
    )
    format_type = Column(
        SQLEnum(DemonstrationFormat),
        default=DemonstrationFormat.JSON_RAW,
        comment="数据格式",
    )

    # 关联信息
    robot_id = Column(
        Integer, ForeignKey("robots.id"), nullable=False, comment="所属机器人ID"
    )
    user_id = Column(
        Integer, ForeignKey("users.id"), nullable=False, comment="创建用户ID"
    )

    # 状态信息
    status = Column(
        SQLEnum(DemonstrationStatus),
        default=DemonstrationStatus.RECORDING,
        comment="示范状态",
    )
    progress = Column(Float, default=0.0, comment="进度 (0-1)")
    error_message = Column(Text, comment="错误信息")

    # 元数据
    recording_duration = Column(Float, default=0.0, comment="录制时长 (秒)")
    frame_count = Column(Integer, default=0, comment="帧数")
    data_size = Column(Integer, default=0, comment="数据大小 (字节)")
    sampling_rate = Column(Float, default=30.0, comment="采样率 (Hz)")
    tags = Column(JSON, default=list, comment="标签列表")

    # 配置信息
    config = Column(JSON, default=dict, comment="示范配置")
    sensor_config = Column(JSON, default=dict, comment="传感器配置")
    joint_config = Column(JSON, default=dict, comment="关节配置")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    recorded_at = Column(DateTime, comment="录制时间")
    processed_at = Column(DateTime, comment="处理完成时间")

    # 关系
    robot = relationship("Robot", back_populates="demonstrations")
    user = relationship("User", back_populates="demonstrations")
    frames = relationship(
        "DemonstrationFrame",
        back_populates="demonstration",
        cascade="all, delete-orphan",
    )
    tasks = relationship(
        "DemonstrationTask",
        secondary="task_demonstrations",
        back_populates="demonstrations",
    )
    training_results = relationship(
        "TrainingResult", back_populates="demonstration", cascade="all, delete-orphan"
    )

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "demonstration_type": self.demonstration_type.value,
            "format_type": self.format_type.value,
            "robot_id": self.robot_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "progress": self.progress,
            "error_message": self.error_message,
            "recording_duration": self.recording_duration,
            "frame_count": self.frame_count,
            "data_size": self.data_size,
            "sampling_rate": self.sampling_rate,
            "tags": self.tags,
            "config": self.config,
            "sensor_config": self.sensor_config,
            "joint_config": self.joint_config,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "processed_at": (
                self.processed_at.isoformat() if self.processed_at else None
            ),
        }


class DemonstrationFrame(Base):
    """示范帧数据表"""

    __tablename__ = "demonstration_frames"

    id = Column(Integer, primary_key=True, index=True)

    # 关联信息
    demonstration_id = Column(
        Integer, ForeignKey("demonstrations.id"), nullable=False, comment="所属示范ID"
    )

    # 帧信息
    frame_index = Column(Integer, nullable=False, comment="帧索引")
    timestamp = Column(Float, nullable=False, comment="时间戳 (秒)")
    relative_time = Column(Float, comment="相对时间 (秒)")

    # 关节数据
    joint_positions = Column(JSON, comment="关节位置数据")
    joint_velocities = Column(JSON, comment="关节速度数据")
    joint_torques = Column(JSON, comment="关节扭矩数据")

    # 控制命令
    control_commands = Column(JSON, comment="控制命令数据")
    target_positions = Column(JSON, comment="目标位置数据")
    target_velocities = Column(JSON, comment="目标速度数据")

    # 传感器数据
    sensor_data = Column(JSON, comment="传感器数据")
    imu_data = Column(JSON, comment="IMU数据")
    camera_data_ref = Column(
        Integer, ForeignKey("camera_frames.id"), comment="关联的相机帧ID"
    )

    # 环境状态
    environment_state = Column(JSON, comment="环境状态")
    robot_state = Column(JSON, comment="机器人整体状态")

    # 元数据
    frame_type = Column(
        String(50), default="control", comment="帧类型: control, observation, mixed"
    )
    compression_ratio = Column(Float, default=1.0, comment="压缩率")
    data_quality = Column(Float, default=1.0, comment="数据质量 (0-1)")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 关系
    demonstration = relationship("Demonstration", back_populates="frames")
    camera_frame = relationship("CameraFrame", back_populates="demonstration_frames")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "demonstration_id": self.demonstration_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "relative_time": self.relative_time,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "joint_torques": self.joint_torques,
            "control_commands": self.control_commands,
            "target_positions": self.target_positions,
            "target_velocities": self.target_velocities,
            "sensor_data": self.sensor_data,
            "imu_data": self.imu_data,
            "camera_data_ref": self.camera_data_ref,
            "environment_state": self.environment_state,
            "robot_state": self.robot_state,
            "frame_type": self.frame_type,
            "compression_ratio": self.compression_ratio,
            "data_quality": self.data_quality,
            "created_at": self.created_at.isoformat(),
        }


class CameraFrame(Base):
    """相机帧数据表（存储图像等大尺寸数据）"""

    __tablename__ = "camera_frames"

    id = Column(Integer, primary_key=True, index=True)

    # 关联信息
    demonstration_id = Column(
        Integer, ForeignKey("demonstrations.id"), nullable=False, comment="所属示范ID"
    )

    # 图像数据
    image_data = Column(LargeBinary, comment="图像原始数据")
    image_format = Column(
        String(20), default="jpeg", comment="图像格式: jpeg, png, raw"
    )
    image_width = Column(Integer, comment="图像宽度")
    image_height = Column(Integer, comment="图像高度")
    image_channels = Column(Integer, default=3, comment="图像通道数")

    # 深度数据
    depth_data = Column(LargeBinary, comment="深度图数据")
    depth_format = Column(String(20), comment="深度图格式: float32, uint16")
    depth_scale = Column(Float, default=0.001, comment="深度图缩放因子")

    # 时间戳
    timestamp = Column(Float, nullable=False, comment="时间戳 (秒)")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 关系
    demonstration = relationship("Demonstration", backref="camera_frames")
    demonstration_frames = relationship(
        "DemonstrationFrame", back_populates="camera_frame"
    )

    def to_dict(self):
        """转换为字典（不包括大尺寸二进制数据）"""
        return {
            "id": self.id,
            "demonstration_id": self.demonstration_id,
            "image_format": self.image_format,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_channels": self.image_channels,
            "depth_format": self.depth_format,
            "depth_scale": self.depth_scale,
            "timestamp": self.timestamp,
            "created_at": self.created_at.isoformat(),
        }


class DemonstrationTask(Base):
    """示范任务表"""

    __tablename__ = "demonstration_tasks"

    id = Column(Integer, primary_key=True, index=True)

    # 基本信息
    name = Column(String(255), nullable=False, comment="任务名称")
    description = Column(Text, comment="任务描述")
    task_type = Column(String(50), nullable=False, comment="任务类型")

    # 配置信息
    config = Column(JSON, default=dict, comment="任务配置")
    success_criteria = Column(JSON, default=dict, comment="成功标准")

    # 关联信息
    robot_id = Column(
        Integer, ForeignKey("robots.id"), nullable=False, comment="所属机器人ID"
    )
    user_id = Column(
        Integer, ForeignKey("users.id"), nullable=False, comment="创建用户ID"
    )

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # 关系
    robot = relationship("Robot", back_populates="demonstration_tasks")
    user = relationship("User", back_populates="demonstration_tasks")
    demonstrations = relationship(
        "Demonstration", secondary="task_demonstrations", back_populates="tasks"
    )

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "config": self.config,
            "success_criteria": self.success_criteria,
            "robot_id": self.robot_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class TrainingResult(Base):
    """训练结果表"""

    __tablename__ = "training_results"

    id = Column(Integer, primary_key=True, index=True)

    # 关联信息
    demonstration_id = Column(
        Integer, ForeignKey("demonstrations.id"), nullable=False, comment="所属示范ID"
    )
    model_id = Column(Integer, ForeignKey("agi_models.id"), comment="训练出的模型ID")

    # 训练信息
    training_method = Column(String(50), nullable=False, comment="训练方法")
    training_duration = Column(Float, default=0.0, comment="训练时长 (秒)")
    training_iterations = Column(Integer, default=0, comment="训练迭代次数")

    # 性能指标
    metrics = Column(JSON, default=dict, comment="性能指标")
    loss_history = Column(JSON, default=dict, comment="损失历史")
    accuracy_history = Column(JSON, default=dict, comment="准确率历史")

    # 模型数据
    model_weights = Column(LargeBinary, comment="模型权重")
    model_config = Column(JSON, comment="模型配置")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, comment="完成时间")

    # 关系
    demonstration = relationship("Demonstration", back_populates="training_results")
    model = relationship("AGIModel", back_populates="training_results")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "demonstration_id": self.demonstration_id,
            "model_id": self.model_id,
            "training_method": self.training_method,
            "training_duration": self.training_duration,
            "training_iterations": self.training_iterations,
            "metrics": self.metrics,
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
            "model_config": self.model_config,
            "created_at": self.created_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }


# 关联表
task_demonstrations = Table(
    "task_demonstrations",
    Base.metadata,
    Column("task_id", Integer, ForeignKey("demonstration_tasks.id"), primary_key=True),
    Column(
        "demonstration_id", Integer, ForeignKey("demonstrations.id"), primary_key=True
    ),
    Column("created_at", DateTime, default=datetime.utcnow),
)


# 在现有模型中添加关系
# 需要在Robot模型中添加:
# demonstrations = relationship("Demonstration", back_populates="robot", cascade="all, delete-orphan")
# demonstration_tasks = relationship("DemonstrationTask", back_populates="robot", cascade="all, delete-orphan")

# 需要在User模型中添加:
# demonstrations = relationship("Demonstration", back_populates="user", cascade="all, delete-orphan")
# demonstration_tasks = relationship("DemonstrationTask", back_populates="user", cascade="all, delete-orphan")

# 在Demonstration模型中添加:
# tasks = relationship("DemonstrationTask", secondary="task_demonstrations", back_populates="demonstrations")
# training_results = relationship("TrainingResult", back_populates="demonstration", cascade="all, delete-orphan")

# 在AGIModel模型中需要添加:
# training_results = relationship("TrainingResult", back_populates="model", cascade="all, delete-orphan")
