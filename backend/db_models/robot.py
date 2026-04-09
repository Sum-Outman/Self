"""
机器人相关数据库模型
包含机器人配置、状态、硬件信息等模型
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Float,
    Text,
    ForeignKey,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum as PyEnum

from ..core.database import Base


class RobotType(PyEnum):
    """机器人类型枚举"""

    HUMANOID = "humanoid"  # 人形机器人
    MOBILE_ROBOT = "mobile_robot"  # 移动机器人
    MANIPULATOR = "manipulator"  # 机械臂
    AERIAL_DRONE = "aerial_drone"  # 无人机
    UNDERWATER_ROBOT = "underwater_robot"  # 水下机器人
    WHEELED_ROBOT = "wheeled_robot"  # 轮式机器人
    TRACKED_ROBOT = "tracked_robot"  # 履带式机器人
    CUSTOM = "custom"  # 自定义类型


class RobotStatus(PyEnum):
    """机器人状态枚举"""

    ONLINE = "online"  # 在线
    OFFLINE = "offline"  # 离线
    ERROR = "error"  # 错误
    BUSY = "busy"  # 繁忙
    IDLE = "idle"  # 空闲
    MAINTENANCE = "maintenance"  # 维护中
    SIMULATION = "simulation"  # 仿真模式


class ControlMode(PyEnum):
    """控制模式枚举"""

    POSITION = "position"  # 位置控制
    VELOCITY = "velocity"  # 速度控制
    TORQUE = "torque"  # 力矩控制
    IMPEDANCE = "impedance"  # 阻抗控制
    FORCE = "force"  # 力控制
    TRAJECTORY = "trajectory"  # 轨迹跟踪


class Robot(Base):
    """机器人主表"""

    __tablename__ = "robots"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True, comment="机器人名称")
    description = Column(Text, comment="机器人描述")

    # 类型信息
    robot_type = Column(
        SQLEnum(RobotType),
        nullable=False,
        default=RobotType.HUMANOID,
        comment="机器人类型",
    )
    model = Column(String(255), comment="机器人型号")
    manufacturer = Column(String(255), comment="制造商")

    # 状态信息
    status = Column(
        SQLEnum(RobotStatus),
        nullable=False,
        default=RobotStatus.OFFLINE,
        comment="机器人状态",
    )
    last_seen = Column(DateTime, comment="最后在线时间")
    battery_level = Column(Float, default=100.0, comment="电池电量 (%)")
    cpu_temperature = Column(Float, comment="CPU温度 (°C)")

    # 连接信息
    connection_type = Column(
        String(50),
        default="simulation",
        comment="连接类型: simulation, serial, ethernet, wifi, bluetooth",
    )
    connection_params = Column(JSON, default=dict, comment="连接参数")
    ip_address = Column(String(45), comment="IP地址")
    port = Column(Integer, comment="端口号")

    # 配置信息
    configuration = Column(JSON, default=dict, comment="机器人配置")
    urdf_path = Column(String(500), comment="URDF模型路径")
    simulation_engine = Column(
        String(50), default="gazebo", comment="仿真引擎: gazebo, pybullet, vrep"
    )
    control_mode = Column(
        SQLEnum(ControlMode), default=ControlMode.POSITION, comment="控制模式"
    )

    # 硬件能力
    capabilities = Column(JSON, default=dict, comment="机器人能力配置")
    joint_count = Column(Integer, default=0, comment="关节数量")
    sensor_count = Column(Integer, default=0, comment="传感器数量")

    # 权限和归属
    user_id = Column(
        Integer, ForeignKey("users.id"), nullable=False, comment="所属用户ID"
    )
    is_public = Column(Boolean, default=False, comment="是否公开")
    is_default = Column(Boolean, default=False, comment="是否默认机器人")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # 关系
    user = relationship("User", back_populates="robots")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "robot_type": self.robot_type.value,
            "model": self.model,
            "manufacturer": self.manufacturer,
            "status": self.status.value,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "battery_level": self.battery_level,
            "cpu_temperature": self.cpu_temperature,
            "connection_type": self.connection_type,
            "connection_params": self.connection_params,
            "ip_address": self.ip_address,
            "port": self.port,
            "configuration": self.configuration,
            "urdf_path": self.urdf_path,
            "simulation_engine": self.simulation_engine,
            "control_mode": self.control_mode.value,
            "capabilities": self.capabilities,
            "joint_count": self.joint_count,
            "sensor_count": self.sensor_count,
            "user_id": self.user_id,
            "is_public": self.is_public,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class RobotJoint(Base):
    """机器人关节表"""

    __tablename__ = "robot_joints"

    id = Column(Integer, primary_key=True, index=True)
    robot_id = Column(
        Integer, ForeignKey("robots.id"), nullable=False, comment="所属机器人ID"
    )
    name = Column(String(100), nullable=False, comment="关节名称")
    joint_type = Column(
        String(50), default="revolute", comment="关节类型: revolute, prismatic, fixed"
    )

    # 物理参数
    min_position = Column(Float, default=-3.14159, comment="最小位置 (rad)")
    max_position = Column(Float, default=3.14159, comment="最大位置 (rad)")
    max_velocity = Column(Float, default=1.0, comment="最大速度 (rad/s)")
    max_torque = Column(Float, default=10.0, comment="最大扭矩 (N·m)")

    # 校准参数
    offset = Column(Float, default=0.0, comment="零点偏移 (rad)")
    direction = Column(Float, default=1.0, comment="方向 (+1或-1)")

    # 状态信息
    current_position = Column(Float, default=0.0, comment="当前位置 (rad)")
    current_velocity = Column(Float, default=0.0, comment="当前速度 (rad/s)")
    current_torque = Column(Float, default=0.0, comment="当前扭矩 (N·m)")
    temperature = Column(Float, default=25.0, comment="温度 (°C)")

    # 元数据
    description = Column(Text, comment="关节描述")
    parent_link = Column(String(100), comment="父连杆")
    child_link = Column(String(100), comment="子连杆")
    axis = Column(String(50), default="z", comment="关节轴: x, y, z")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # 关系
    robot = relationship("Robot", back_populates="joints")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "robot_id": self.robot_id,
            "name": self.name,
            "joint_type": self.joint_type,
            "min_position": self.min_position,
            "max_position": self.max_position,
            "max_velocity": self.max_velocity,
            "max_torque": self.max_torque,
            "offset": self.offset,
            "direction": self.direction,
            "current_position": self.current_position,
            "current_velocity": self.current_velocity,
            "current_torque": self.current_torque,
            "temperature": self.temperature,
            "description": self.description,
            "parent_link": self.parent_link,
            "child_link": self.child_link,
            "axis": self.axis,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class RobotSensor(Base):
    """机器人传感器表"""

    __tablename__ = "robot_sensors"

    id = Column(Integer, primary_key=True, index=True)
    robot_id = Column(
        Integer, ForeignKey("robots.id"), nullable=False, comment="所属机器人ID"
    )
    name = Column(String(100), nullable=False, comment="传感器名称")
    sensor_type = Column(
        String(50),
        nullable=False,
        comment="传感器类型: imu, camera, lidar, force, touch",
    )

    # 配置信息
    model = Column(String(100), comment="传感器型号")
    manufacturer = Column(String(100), comment="制造商")
    sampling_rate = Column(Float, comment="采样率 (Hz)")
    accuracy = Column(Float, comment="精度")
    range_min = Column(Float, comment="最小量程")
    range_max = Column(Float, comment="最大量程")

    # 位置和方向
    position_x = Column(Float, default=0.0, comment="X位置 (m)")
    position_y = Column(Float, default=0.0, comment="Y位置 (m)")
    position_z = Column(Float, default=0.0, comment="Z位置 (m)")
    orientation_x = Column(Float, default=0.0, comment="X方向 (quaternion)")
    orientation_y = Column(Float, default=0.0, comment="Y方向 (quaternion)")
    orientation_z = Column(Float, default=0.0, comment="Z方向 (quaternion)")
    orientation_w = Column(Float, default=1.0, comment="W方向 (quaternion)")

    # 状态信息
    status = Column(
        String(50), default="online", comment="状态: online, offline, error"
    )
    last_data = Column(JSON, comment="最后数据")
    last_update = Column(DateTime, comment="最后更新时间")

    # 元数据
    description = Column(Text, comment="传感器描述")
    calibration_data = Column(JSON, comment="校准数据")

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # 关系
    robot = relationship("Robot", back_populates="sensors")

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "robot_id": self.robot_id,
            "name": self.name,
            "sensor_type": self.sensor_type,
            "model": self.model,
            "manufacturer": self.manufacturer,
            "sampling_rate": self.sampling_rate,
            "accuracy": self.accuracy,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "position_z": self.position_z,
            "orientation_x": self.orientation_x,
            "orientation_y": self.orientation_y,
            "orientation_z": self.orientation_z,
            "orientation_w": self.orientation_w,
            "status": self.status,
            "last_data": self.last_data,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "description": self.description,
            "calibration_data": self.calibration_data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# 在Robot模型中添加关系
Robot.joints = relationship(
    "RobotJoint", back_populates="robot", cascade="all, delete-orphan"
)
Robot.sensors = relationship(
    "RobotSensor", back_populates="robot", cascade="all, delete-orphan"
)
Robot.demonstrations = relationship(
    "Demonstration", back_populates="robot", cascade="all, delete-orphan"
)
Robot.demonstration_tasks = relationship(
    "DemonstrationTask", back_populates="robot", cascade="all, delete-orphan"
)
Robot.market_listings = relationship(
    "RobotMarketListing", back_populates="robot", cascade="all, delete-orphan"
)

# 在User模型中需要添加robots关系（需要更新user.py）
# 我们将在初始化数据库时处理
