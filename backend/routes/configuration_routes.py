#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人配置导入/导出API路由
支持URDF文件导入、配置文件导入导出等功能
"""

import os
import tempfile
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session
import json
import yaml

from backend.dependencies import get_db
from backend.services.robot_configuration_service import (
    RobotConfigurationService,
    URDFParser,
)
from backend.dependencies.auth import get_current_user
from backend.db_models.user import User
from backend.db_models.robot import Robot

router = APIRouter(prefix="/api/configuration", tags=["机器人配置"])


def get_configuration_service(
    db: Session = Depends(get_db),
) -> RobotConfigurationService:
    """获取机器人配置服务"""
    return RobotConfigurationService(db)


@router.post("/import/urdf", response_model=Dict[str, Any])
async def import_urdf(
    urdf_file: UploadFile = File(..., description="URDF文件"),
    robot_name: Optional[str] = Form(
        None, description="机器人名称（可选，默认使用URDF中的名称）"
    ),
    description: Optional[str] = Form(None, description="机器人描述（可选）"),
    service: RobotConfigurationService = Depends(get_configuration_service),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """导入URDF文件并创建机器人配置"""
    try:
        # 检查文件类型
        if not urdf_file.filename.lower().endswith((".urdf", ".xml")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持URDF或XML文件"
            )

        # 读取URDF内容
        urdf_content = await urdf_file.read()
        urdf_content_str = urdf_content.decode("utf-8")

        # 导入URDF
        success, message, robot = service.import_urdf(
            urdf_content_str, user.id, robot_name, description
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        return {
            "success": True,
            "message": message,
            "robot_id": robot.id,
            "robot_name": robot.name,
            "robot_type": robot.robot_type.value,
            "joint_count": robot.joint_count,
            "sensor_count": robot.sensor_count,
            "timestamp": robot.created_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URDF导入失败: {str(e)}",
        )


@router.post("/import/config", response_model=Dict[str, Any])
async def import_config_file(
    config_file: UploadFile = File(..., description="配置文件（JSON或YAML）"),
    format: str = Form("auto", description="文件格式: auto, json, yaml"),
    service: RobotConfigurationService = Depends(get_configuration_service),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """导入配置文件（JSON/YAML）并创建机器人配置"""
    try:
        # 检查文件类型
        filename = config_file.filename.lower()
        if not (
            filename.endswith((".json", ".jsonc", ".yaml", ".yml")) or format != "auto"
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持JSON或YAML文件"
            )

        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp_file:
            content = await config_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # 导入配置文件
            success, message, robot = service.import_config_file(
                tmp_file_path, user.id, format
            )

            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=message
                )

            return {
                "success": True,
                "message": message,
                "robot_id": robot.id,
                "robot_name": robot.name,
                "robot_type": robot.robot_type.value,
                "joint_count": robot.joint_count,
                "sensor_count": robot.sensor_count,
                "timestamp": robot.created_at.isoformat(),
            }

        finally:
            # 删除临时文件
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"配置文件导入失败: {str(e)}",
        )


@router.get("/export/urdf/{robot_id}", response_model=Dict[str, Any])
async def export_urdf(
    robot_id: int,
    service: RobotConfigurationService = Depends(get_configuration_service),
    user: User = Depends(get_current_user),
):
    """导出机器人的URDF文件"""
    try:
        # 验证用户权限
        robot = (
            service.db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"机器人 {robot_id} 不存在或无权访问",
            )

        # 导出URDF
        success, message, urdf_content = service.export_urdf(robot_id)

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        return {
            "success": True,
            "message": message,
            "robot_id": robot_id,
            "robot_name": robot.name,
            "urdf_content": urdf_content,
            "filename": f"{robot.name}.urdf",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URDF导出失败: {str(e)}",
        )


@router.get("/export/config/{robot_id}", response_model=Dict[str, Any])
async def export_config(
    robot_id: int,
    format: str = "json",
    service: RobotConfigurationService = Depends(get_configuration_service),
    user: User = Depends(get_current_user),
):
    """导出机器人配置为JSON或YAML格式"""
    try:
        # 验证用户权限
        robot = (
            service.db.query(Robot)
            .filter(Robot.id == robot_id, Robot.user_id == user.id)
            .first()
        )

        if not robot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"机器人 {robot_id} 不存在或无权访问",
            )

        # 导出配置
        success, message, config = service.export_config_file(robot_id, format)

        if not success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

        # 根据格式序列化
        if format.lower() == "json":
            content = json.dumps(config, indent=2, ensure_ascii=False)
            content_type = "application/json"
            filename = f"{robot.name}_config.json"
        elif format.lower() == "yaml":
            content = yaml.dump(config, allow_unicode=True, default_flow_style=False)
            content_type = "application/x-yaml"
            filename = f"{robot.name}_config.yaml"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的格式: {format}",
            )

        return {
            "success": True,
            "message": message,
            "robot_id": robot_id,
            "robot_name": robot.name,
            "config_content": content,
            "content_type": content_type,
            "filename": filename,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"配置导出失败: {str(e)}",
        )


@router.post("/validate/urdf", response_model=Dict[str, Any])
async def validate_urdf(
    urdf_file: UploadFile = File(..., description="URDF文件"),
):
    """验证URDF文件的有效性"""
    try:
        # 检查文件类型
        if not urdf_file.filename.lower().endswith((".urdf", ".xml")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持URDF或XML文件"
            )

        # 读取URDF内容
        urdf_content = await urdf_file.read()
        urdf_content_str = urdf_content.decode("utf-8")

        try:
            # 解析URDF
            parser = URDFParser(urdf_content_str)
            robot_info = parser.get_robot_info()

            return {
                "success": True,
                "message": "URDF文件有效",
                "robot_info": robot_info,
                "joint_count": len(parser.joints),
                "link_count": len(parser.links),
                "sensor_count": len(parser.sensors),
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"URDF文件无效: {str(e)}",
                "robot_info": None,
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URDF验证失败: {str(e)}",
        )


@router.get("/templates/{robot_type}", response_model=Dict[str, Any])
async def get_configuration_template(
    robot_type: str,
):
    """获取指定机器人类型的配置模板"""
    try:
        # 机器人类型模板
        templates = {
            "humanoid": {
                "name": "人形机器人模板",
                "description": "标准人形机器人配置模板",
                "robot_type": "humanoid",
                "model": "Humanoid_Standard",
                "manufacturer": "Self AGI",
                "connection_type": "simulation",
                "connection_params": {
                    "host": "localhost",
                    "port": 11311,
                    "use_sim_time": True,
                },
                "simulation_engine": "gazebo",
                "control_mode": "position",
                "capabilities": {
                    "walking": True,
                    "balancing": True,
                    "object_manipulation": True,
                },
                "joints": [
                    {
                        "name": "head_yaw",
                        "joint_type": "revolute",
                        "min_position": -1.57,
                        "max_position": 1.57,
                        "max_velocity": 1.0,
                        "max_torque": 5.0,
                        "parent_link": "torso",
                        "child_link": "head",
                        "axis": "z",
                        "description": "头部偏航关节",
                    },
                    {
                        "name": "head_pitch",
                        "joint_type": "revolute",
                        "min_position": -0.79,
                        "max_position": 0.79,
                        "max_velocity": 1.0,
                        "max_torque": 5.0,
                        "parent_link": "head",
                        "child_link": "neck",
                        "axis": "y",
                        "description": "头部俯仰关节",
                    },
                ],
                "sensors": [
                    {
                        "name": "head_camera",
                        "sensor_type": "camera",
                        "model": "RGB_Camera",
                        "sampling_rate": 30.0,
                        "position_x": 0.0,
                        "position_y": 0.0,
                        "position_z": 0.1,
                        "orientation_x": 0.0,
                        "orientation_y": 0.0,
                        "orientation_z": 0.0,
                        "orientation_w": 1.0,
                        "description": "头部摄像头",
                    },
                    {
                        "name": "imu",
                        "sensor_type": "imu",
                        "model": "9DOF_IMU",
                        "sampling_rate": 100.0,
                        "position_x": 0.0,
                        "position_y": 0.0,
                        "position_z": 0.05,
                        "description": "惯性测量单元",
                    },
                ],
            },
            "mobile_robot": {
                "name": "移动机器人模板",
                "description": "标准移动机器人配置模板",
                "robot_type": "mobile_robot",
                "model": "Mobile_Robot_Standard",
                "manufacturer": "Self AGI",
                "connection_type": "serial",
                "connection_params": {"port": "/dev/ttyUSB0", "baudrate": 115200},
                "control_mode": "velocity",
                "capabilities": {
                    "navigation": True,
                    "mapping": True,
                    "object_detection": True,
                },
                "joints": [
                    {
                        "name": "left_wheel",
                        "joint_type": "revolute",
                        "max_velocity": 10.0,
                        "max_torque": 2.0,
                        "description": "左轮驱动关节",
                    },
                    {
                        "name": "right_wheel",
                        "joint_type": "revolute",
                        "max_velocity": 10.0,
                        "max_torque": 2.0,
                        "description": "右轮驱动关节",
                    },
                ],
                "sensors": [
                    {
                        "name": "lidar",
                        "sensor_type": "lidar",
                        "model": "2D_LiDAR",
                        "sampling_rate": 10.0,
                        "description": "激光雷达",
                    },
                    {
                        "name": "front_camera",
                        "sensor_type": "camera",
                        "model": "RGB_Camera",
                        "sampling_rate": 30.0,
                        "description": "前向摄像头",
                    },
                ],
            },
            "manipulator": {
                "name": "机械臂模板",
                "description": "标准6自由度机械臂配置模板",
                "robot_type": "manipulator",
                "model": "6DOF_Manipulator",
                "manufacturer": "Self AGI",
                "connection_type": "ethernet",
                "connection_params": {"ip": "192.168.1.100", "port": 502},
                "control_mode": "position",
                "capabilities": {
                    "pick_and_place": True,
                    "trajectory_tracking": True,
                    "force_control": True,
                },
                "joints": [
                    {
                        "name": "joint1",
                        "joint_type": "revolute",
                        "min_position": -3.14,
                        "max_position": 3.14,
                        "max_velocity": 1.57,
                        "max_torque": 20.0,
                        "description": "基座旋转关节",
                    },
                    {
                        "name": "joint2",
                        "joint_type": "revolute",
                        "min_position": -1.57,
                        "max_position": 1.57,
                        "max_velocity": 1.57,
                        "max_torque": 20.0,
                        "description": "肩部俯仰关节",
                    },
                    {
                        "name": "joint3",
                        "joint_type": "revolute",
                        "min_position": -1.57,
                        "max_position": 1.57,
                        "max_velocity": 1.57,
                        "max_torque": 15.0,
                        "description": "肘部俯仰关节",
                    },
                    {
                        "name": "joint4",
                        "joint_type": "revolute",
                        "min_position": -3.14,
                        "max_position": 3.14,
                        "max_velocity": 2.09,
                        "max_torque": 10.0,
                        "description": "手腕旋转关节",
                    },
                    {
                        "name": "joint5",
                        "joint_type": "revolute",
                        "min_position": -1.57,
                        "max_position": 1.57,
                        "max_velocity": 2.09,
                        "max_torque": 10.0,
                        "description": "手腕俯仰关节",
                    },
                    {
                        "name": "joint6",
                        "joint_type": "revolute",
                        "min_position": -3.14,
                        "max_position": 3.14,
                        "max_velocity": 2.09,
                        "max_torque": 5.0,
                        "description": "末端旋转关节",
                    },
                ],
                "sensors": [
                    {
                        "name": "force_torque_sensor",
                        "sensor_type": "force",
                        "model": "6DOF_FT_Sensor",
                        "sampling_rate": 100.0,
                        "description": "力扭矩传感器",
                    }
                ],
            },
        }

        # 获取模板
        template = templates.get(robot_type.lower())
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"找不到机器人类型 '{robot_type}' 的模板",
            )

        return {
            "success": True,
            "robot_type": robot_type,
            "template": template,
            "message": f"获取 {robot_type} 模板成功",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模板失败: {str(e)}",
        )
