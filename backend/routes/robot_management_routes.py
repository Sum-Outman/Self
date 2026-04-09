"""
多机器人管理API路由
包含机器人CRUD操作、状态管理、配置管理等功能
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from backend.dependencies.database import get_db
from backend.db_models.user import User
from backend.db_models.robot import (
    Robot,
    RobotJoint,
    RobotSensor,
    RobotType,
    RobotStatus,
    ControlMode,
)
from backend.dependencies.auth import get_current_user

router = APIRouter(prefix="/api/robots", tags=["机器人管理"])


# ========== 机器人CRUD操作 ==========


@router.get("/", response_model=Dict[str, Any])
async def get_robots(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    robot_type: Optional[RobotType] = None,
    status_filter: Optional[RobotStatus] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取机器人列表"""
    try:
        query = db.query(Robot).filter(Robot.user_id == user.id)

        if robot_type:
            query = query.filter(Robot.robot_type == robot_type)

        if status_filter:
            query = query.filter(Robot.status == status_filter)

        total = query.count()
        robots = query.offset(skip).limit(limit).all()

        return {
            "success": True,
            "data": {
                "robots": [robot.to_dict() for robot in robots],
                "total": total,
                "skip": skip,
                "limit": limit,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取机器人列表失败: {str(e)}",
        )


@router.get("/{robot_id}", response_model=Dict[str, Any])
async def get_robot(
    robot_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取单个机器人详情"""
    try:
        robot = (
            db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="机器人不存在或无权访问"
            )

        # 获取关节和传感器数据
        joints = db.query(RobotJoint).filter(RobotJoint.robot_id == robot_id).all()
        sensors = db.query(RobotSensor).filter(RobotSensor.robot_id == robot_id).all()

        robot_data = robot.to_dict()
        robot_data["joints"] = [joint.to_dict() for joint in joints]
        robot_data["sensors"] = [sensor.to_dict() for sensor in sensors]

        return {"success": True, "data": robot_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取机器人详情失败: {str(e)}",
        )


@router.post("/", response_model=Dict[str, Any])
async def create_robot(
    name: str = Query(..., description="机器人名称"),
    robot_type: RobotType = Query(RobotType.HUMANOID, description="机器人类型"),
    description: Optional[str] = Query(None, description="机器人描述"),
    model: Optional[str] = Query(None, description="机器人型号"),
    manufacturer: Optional[str] = Query(None, description="制造商"),
    configuration: Optional[Dict[str, Any]] = Body(None, description="机器人配置"),
    urdf_path: Optional[str] = Query(None, description="URDF模型路径"),
    simulation_engine: Optional[str] = Query("gazebo", description="仿真引擎"),
    control_mode: ControlMode = Query(ControlMode.POSITION, description="控制模式"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """创建新机器人"""
    try:
        # 检查名称是否已存在
        existing = (
            db.query(Robot).filter(Robot.name == name, Robot.user_id == user.id).first()
        )

        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="机器人名称已存在"
            )

        # 创建机器人
        robot = Robot(
            name=name,
            description=description,
            robot_type=robot_type,
            model=model,
            manufacturer=manufacturer,
            configuration=configuration or {},
            urdf_path=urdf_path,
            simulation_engine=simulation_engine,
            control_mode=control_mode,
            user_id=user.id,
            status=RobotStatus.OFFLINE,
            battery_level=100.0,
            joint_count=0,
            sensor_count=0,
        )

        db.add(robot)
        db.commit()
        db.refresh(robot)

        return {"success": True, "data": robot.to_dict(), "message": "机器人创建成功"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建机器人失败: {str(e)}",
        )


@router.put("/{robot_id}", response_model=Dict[str, Any])
async def update_robot(
    robot_id: int,
    name: Optional[str] = Query(None, description="机器人名称"),
    description: Optional[str] = Query(None, description="机器人描述"),
    status: Optional[RobotStatus] = Query(None, description="机器人状态"),
    configuration: Optional[Dict[str, Any]] = Body(None, description="机器人配置"),
    urdf_path: Optional[str] = Query(None, description="URDF模型路径"),
    simulation_engine: Optional[str] = Query(None, description="仿真引擎"),
    control_mode: Optional[ControlMode] = Query(None, description="控制模式"),
    battery_level: Optional[float] = Query(None, ge=0, le=100, description="电池电量"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """更新机器人信息"""
    try:
        robot = (
            db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="机器人不存在或无权访问"
            )

        # 更新字段
        if name is not None:
            # 检查名称是否与其他机器人冲突
            existing = (
                db.query(Robot)
                .filter(
                    Robot.name == name, Robot.user_id == user.id, Robot.id != robot_id
                )
                .first()
            )
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="机器人名称已存在"
                )
            robot.name = name

        if description is not None:
            robot.description = description

        if status is not None:
            robot.status = status

        if configuration is not None:
            robot.configuration = configuration

        if urdf_path is not None:
            robot.urdf_path = urdf_path

        if simulation_engine is not None:
            robot.simulation_engine = simulation_engine

        if control_mode is not None:
            robot.control_mode = control_mode

        if battery_level is not None:
            robot.battery_level = battery_level

        robot.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(robot)

        return {"success": True, "data": robot.to_dict(), "message": "机器人更新成功"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新机器人失败: {str(e)}",
        )


@router.delete("/{robot_id}", response_model=Dict[str, Any])
async def delete_robot(
    robot_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """删除机器人"""
    try:
        robot = (
            db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="机器人不存在或无权访问"
            )

        db.delete(robot)
        db.commit()

        return {"success": True, "message": "机器人删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除机器人失败: {str(e)}",
        )


# ========== 机器人操作 ==========


@router.post("/{robot_id}/connect", response_model=Dict[str, Any])
async def connect_robot(
    robot_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """连接机器人"""
    try:
        robot = (
            db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="机器人不存在或无权访问"
            )

        # 更新状态
        robot.status = RobotStatus.ONLINE
        robot.last_seen = datetime.now(timezone.utc)
        db.commit()

        return {
            "success": True,
            "message": f"机器人 {robot.name} 已连接",
            "data": {
                "robot_id": robot.id,
                "name": robot.name,
                "status": robot.status.value,
                "last_seen": robot.last_seen.isoformat() if robot.last_seen else None,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"连接机器人失败: {str(e)}",
        )


@router.post("/{robot_id}/disconnect", response_model=Dict[str, Any])
async def disconnect_robot(
    robot_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """断开机器人连接"""
    try:
        robot = (
            db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="机器人不存在或无权访问"
            )

        # 更新状态
        robot.status = RobotStatus.OFFLINE
        db.commit()

        return {
            "success": True,
            "message": f"机器人 {robot.name} 已断开连接",
            "data": {
                "robot_id": robot.id,
                "name": robot.name,
                "status": robot.status.value,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"断开机器人连接失败: {str(e)}",
        )


@router.get("/{robot_id}/status", response_model=Dict[str, Any])
async def get_robot_status(
    robot_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取机器人状态"""
    try:
        robot = (
            db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="机器人不存在或无权访问"
            )

        # 获取关节状态
        joints = db.query(RobotJoint).filter(RobotJoint.robot_id == robot_id).all()
        sensors = db.query(RobotSensor).filter(RobotSensor.robot_id == robot_id).all()

        return {
            "success": True,
            "data": {
                "robot": robot.to_dict(),
                "joints": [joint.to_dict() for joint in joints],
                "sensors": [sensor.to_dict() for sensor in sensors],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取机器人状态失败: {str(e)}",
        )


# ========== 关节和传感器管理 ==========


@router.get("/{robot_id}/joints", response_model=Dict[str, Any])
async def get_robot_joints(
    robot_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取机器人关节列表"""
    try:
        robot = (
            db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="机器人不存在或无权访问"
            )

        joints = db.query(RobotJoint).filter(RobotJoint.robot_id == robot_id).all()

        return {
            "success": True,
            "data": {
                "joints": [joint.to_dict() for joint in joints],
                "total": len(joints),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取机器人关节列表失败: {str(e)}",
        )


@router.get("/{robot_id}/sensors", response_model=Dict[str, Any])
async def get_robot_sensors(
    robot_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取机器人传感器列表"""
    try:
        robot = (
            db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="机器人不存在或无权访问"
            )

        sensors = db.query(RobotSensor).filter(RobotSensor.robot_id == robot_id).all()

        return {
            "success": True,
            "data": {
                "sensors": [sensor.to_dict() for sensor in sensors],
                "total": len(sensors),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取机器人传感器列表失败: {str(e)}",
        )


@router.post("/{robot_id}/joints", response_model=Dict[str, Any])
async def create_robot_joint(
    robot_id: int,
    name: str = Query(..., description="关节名称"),
    joint_type: str = Query(
        "revolute", description="关节类型: revolute, prismatic, fixed"
    ),
    min_position: float = Query(-3.14159, description="最小位置 (rad)"),
    max_position: float = Query(3.14159, description="最大位置 (rad)"),
    max_velocity: float = Query(1.0, description="最大速度 (rad/s)"),
    max_torque: float = Query(10.0, description="最大扭矩 (N·m)"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """为机器人添加关节"""
    try:
        robot = (
            db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="机器人不存在或无权访问"
            )

        # 检查关节是否已存在
        existing = (
            db.query(RobotJoint)
            .filter(RobotJoint.robot_id == robot_id, RobotJoint.name == name)
            .first()
        )

        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="关节名称已存在"
            )

        # 创建关节
        joint = RobotJoint(
            robot_id=robot_id,
            name=name,
            joint_type=joint_type,
            min_position=min_position,
            max_position=max_position,
            max_velocity=max_velocity,
            max_torque=max_torque,
            offset=0.0,
            direction=1.0,
            current_position=0.0,
            current_velocity=0.0,
            current_torque=0.0,
            temperature=25.0,
        )

        db.add(joint)

        # 更新机器人关节计数
        robot.joint_count = (
            db.query(RobotJoint).filter(RobotJoint.robot_id == robot_id).count()
        )

        db.commit()
        db.refresh(joint)

        return {"success": True, "data": joint.to_dict(), "message": "关节创建成功"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建关节失败: {str(e)}",
        )


# ========== 默认机器人设置 ==========


@router.post("/{robot_id}/set-default", response_model=Dict[str, Any])
async def set_default_robot(
    robot_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """设置默认机器人"""
    try:
        # 清除当前默认机器人
        db.query(Robot).filter(
            Robot.user_id == user.id, Robot.is_default == True
        ).update({"is_default": False})

        # 设置新的默认机器人
        robot = (
            db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="机器人不存在或无权访问"
            )

        robot.is_default = True
        db.commit()

        return {
            "success": True,
            "message": f"已将 {robot.name} 设置为默认机器人",
            "data": robot.to_dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置默认机器人失败: {str(e)}",
        )


@router.get("/default", response_model=Dict[str, Any])
async def get_default_robot(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取用户的默认机器人"""
    try:
        robot = (
            db.query(Robot)
            .filter(Robot.user_id == user.id, Robot.is_default == True)
            .first()
        )

        if not robot:
            # 如果没有默认机器人，返回第一个机器人
            robot = db.query(Robot).filter(Robot.user_id == user.id).first()

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="用户没有机器人"
            )

        return {"success": True, "data": robot.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取默认机器人失败: {str(e)}",
        )
