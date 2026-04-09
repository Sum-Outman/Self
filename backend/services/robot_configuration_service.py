#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人配置导入/导出服务
支持URDF文件导入和导出，以及JSON/YAML配置文件处理
"""

import os
import logging
import xml.etree.ElementTree as ET
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from sqlalchemy.orm import Session

from ..db_models.robot import (
    Robot,
    RobotJoint,
    RobotSensor,
    RobotType,
    RobotStatus,
    ControlMode,
)

logger = logging.getLogger(__name__)


class URDFParser:
    """URDF文件解析器"""

    NAMESPACE = "{http://www.ros.org/wiki/urdf}"

    def __init__(self, urdf_content: str):
        """初始化URDF解析器"""
        self.urdf_content = urdf_content
        self.root = ET.fromstring(urdf_content)
        self.joints = []
        self.links = []
        self.sensors = []
        self.materials = []
        self._parse()

    def _parse(self):
        """解析URDF内容"""
        # 查找所有关节
        for joint_elem in self.root.findall(f"./{self.NAMESPACE}joint"):
            joint = self._parse_joint(joint_elem)
            if joint:
                self.joints.append(joint)

        # 查找所有连杆
        for link_elem in self.root.findall(f"./{self.NAMESPACE}link"):
            link = self._parse_link(link_elem)
            if link:
                self.links.append(link)

        # 查找传感器（URDF中通常没有传感器，但可能有自定义扩展）
        for sensor_elem in self.root.findall(f".//{self.NAMESPACE}sensor"):
            sensor = self._parse_sensor(sensor_elem)
            if sensor:
                self.sensors.append(sensor)

        # 查找材料
        for material_elem in self.root.findall(f".//{self.NAMESPACE}material"):
            material = self._parse_material(material_elem)
            if material:
                self.materials.append(material)

    def _parse_joint(self, joint_elem) -> Optional[Dict[str, Any]]:
        """解析关节元素"""
        try:
            joint = {
                "name": joint_elem.get("name", ""),
                "type": joint_elem.get("type", "fixed"),
                "parent_link": None,
                "child_link": None,
                "axis": [0, 0, 1],  # 默认Z轴
                "origin": None,
                "limits": None,
                "dynamics": None,
            }

            # 解析父连杆
            parent_elem = joint_elem.find(f"{self.NAMESPACE}parent")
            if parent_elem is not None:
                joint["parent_link"] = parent_elem.get("link")

            # 解析子连杆
            child_elem = joint_elem.find(f"{self.NAMESPACE}child")
            if child_elem is not None:
                joint["child_link"] = child_elem.get("link")

            # 解析原点
            origin_elem = joint_elem.find(f"{self.NAMESPACE}origin")
            if origin_elem is not None:
                xyz = origin_elem.get("xyz", "0 0 0").split()
                rpy = origin_elem.get("rpy", "0 0 0").split()
                joint["origin"] = {
                    "xyz": [float(val) for val in xyz],
                    "rpy": [float(val) for val in rpy],
                }

            # 解析轴
            axis_elem = joint_elem.find(f"{self.NAMESPACE}axis")
            if axis_elem is not None:
                xyz = axis_elem.get("xyz", "0 0 1").split()
                joint["axis"] = [float(val) for val in xyz]

            # 解析限制
            limit_elem = joint_elem.find(f"{self.NAMESPACE}limit")
            if limit_elem is not None:
                joint["limits"] = {
                    "lower": float(limit_elem.get("lower", "-3.14159")),
                    "upper": float(limit_elem.get("upper", "3.14159")),
                    "velocity": float(limit_elem.get("velocity", "1.0")),
                    "effort": float(limit_elem.get("effort", "10.0")),
                }

            # 解析动力学
            dynamics_elem = joint_elem.find(f"{self.NAMESPACE}dynamics")
            if dynamics_elem is not None:
                joint["dynamics"] = {
                    "damping": float(dynamics_elem.get("damping", "0.0")),
                    "friction": float(dynamics_elem.get("friction", "0.0")),
                }

            return joint

        except Exception as e:
            logger.error(f"解析关节失败: {e}")
            return None  # 返回None

    def _parse_link(self, link_elem) -> Optional[Dict[str, Any]]:
        """解析连杆元素"""
        try:
            link = {
                "name": link_elem.get("name", ""),
                "inertial": None,
                "visual": [],
                "collision": [],
            }

            # 解析惯性
            inertial_elem = link_elem.find(f"{self.NAMESPACE}inertial")
            if inertial_elem is not None:
                link["inertial"] = self._parse_inertial(inertial_elem)

            # 解析视觉元素
            for visual_elem in link_elem.findall(f"{self.NAMESPACE}visual"):
                visual = self._parse_visual(visual_elem)
                if visual:
                    link["visual"].append(visual)

            # 解析碰撞元素
            for collision_elem in link_elem.findall(f"{self.NAMESPACE}collision"):
                collision = self._parse_collision(collision_elem)
                if collision:
                    link["collision"].append(collision)

            return link

        except Exception as e:
            logger.error(f"解析连杆失败: {e}")
            return None  # 返回None

    def _parse_inertial(self, inertial_elem) -> Dict[str, Any]:
        """解析惯性元素"""
        inertial = {"mass": 0.0, "origin": None, "inertia": None}

        # 解析质量
        mass_elem = inertial_elem.find(f"{self.NAMESPACE}mass")
        if mass_elem is not None:
            inertial["mass"] = float(mass_elem.get("value", "0.0"))

        # 解析原点
        origin_elem = inertial_elem.find(f"{self.NAMESPACE}origin")
        if origin_elem is not None:
            xyz = origin_elem.get("xyz", "0 0 0").split()
            rpy = origin_elem.get("rpy", "0 0 0").split()
            inertial["origin"] = {
                "xyz": [float(val) for val in xyz],
                "rpy": [float(val) for val in rpy],
            }

        # 解析惯性矩阵
        inertia_elem = inertial_elem.find(f"{self.NAMESPACE}inertia")
        if inertia_elem is not None:
            inertial["inertia"] = {
                "ixx": float(inertia_elem.get("ixx", "0.0")),
                "ixy": float(inertia_elem.get("ixy", "0.0")),
                "ixz": float(inertia_elem.get("ixz", "0.0")),
                "iyy": float(inertia_elem.get("iyy", "0.0")),
                "iyz": float(inertia_elem.get("iyz", "0.0")),
                "izz": float(inertia_elem.get("izz", "0.0")),
            }

        return inertial

    def _parse_visual(self, visual_elem) -> Dict[str, Any]:
        """解析视觉元素"""
        visual = {
            "name": visual_elem.get("name", ""),
            "origin": None,
            "geometry": None,
            "material": None,
        }

        # 解析原点
        origin_elem = visual_elem.find(f"{self.NAMESPACE}origin")
        if origin_elem is not None:
            xyz = origin_elem.get("xyz", "0 0 0").split()
            rpy = origin_elem.get("rpy", "0 0 0").split()
            visual["origin"] = {
                "xyz": [float(val) for val in xyz],
                "rpy": [float(val) for val in rpy],
            }

        # 解析几何
        geometry_elem = visual_elem.find(f"{self.NAMESPACE}geometry")
        if geometry_elem is not None:
            visual["geometry"] = self._parse_geometry(geometry_elem)

        # 解析材质
        material_elem = visual_elem.find(f"{self.NAMESPACE}material")
        if material_elem is not None:
            visual["material"] = self._parse_material(material_elem)

        return visual

    def _parse_collision(self, collision_elem) -> Dict[str, Any]:
        """解析碰撞元素"""
        collision = {
            "name": collision_elem.get("name", ""),
            "origin": None,
            "geometry": None,
        }

        # 解析原点
        origin_elem = collision_elem.find(f"{self.NAMESPACE}origin")
        if origin_elem is not None:
            xyz = origin_elem.get("xyz", "0 0 0").split()
            rpy = origin_elem.get("rpy", "0 0 0").split()
            collision["origin"] = {
                "xyz": [float(val) for val in xyz],
                "rpy": [float(val) for val in rpy],
            }

        # 解析几何
        geometry_elem = collision_elem.find(f"{self.NAMESPACE}geometry")
        if geometry_elem is not None:
            collision["geometry"] = self._parse_geometry(geometry_elem)

        return collision

    def _parse_geometry(self, geometry_elem) -> Dict[str, Any]:
        """解析几何元素"""
        geometry = {"type": "unknown"}

        # 检查几何类型
        for child in geometry_elem:
            tag = child.tag.replace(self.NAMESPACE, "")
            if tag in ["box", "cylinder", "sphere", "mesh"]:
                geometry["type"] = tag
                geometry["params"] = {}

                if tag == "box":
                    size = child.get("size", "1 1 1").split()
                    geometry["params"]["size"] = [float(val) for val in size]

                elif tag == "cylinder":
                    geometry["params"]["radius"] = float(child.get("radius", "0.5"))
                    geometry["params"]["length"] = float(child.get("length", "1.0"))

                elif tag == "sphere":
                    geometry["params"]["radius"] = float(child.get("radius", "0.5"))

                elif tag == "mesh":
                    geometry["params"]["filename"] = child.get("filename", "")
                    scale = child.get("scale", "1 1 1").split()
                    geometry["params"]["scale"] = [float(val) for val in scale]

                break

        return geometry

    def _parse_material(self, material_elem) -> Dict[str, Any]:
        """解析材质元素"""
        material = {
            "name": material_elem.get("name", ""),
            "color": None,
            "texture": None,
        }

        # 解析颜色
        color_elem = material_elem.find(f"{self.NAMESPACE}color")
        if color_elem is not None:
            rgba = color_elem.get("rgba", "1 1 1 1").split()
            material["color"] = [float(val) for val in rgba]

        # 解析纹理
        texture_elem = material_elem.find(f"{self.NAMESPACE}texture")
        if texture_elem is not None:
            material["texture"] = texture_elem.get("filename", "")

        return material

    def _parse_sensor(self, sensor_elem) -> Optional[Dict[str, Any]]:
        """解析传感器元素（URDF扩展）"""
        try:
            sensor_type = sensor_elem.get("type", "unknown")
            sensor = {
                "name": sensor_elem.get("name", ""),
                "type": sensor_type,
                "origin": None,
                "parent_link": None,
                "parameters": {},
            }

            # 解析父连杆
            parent_link_elem = sensor_elem.find(f"{self.NAMESPACE}parent_link")
            if parent_link_elem is not None:
                sensor["parent_link"] = parent_link_elem.text

            # 解析原点
            origin_elem = sensor_elem.find(f"{self.NAMESPACE}origin")
            if origin_elem is not None:
                xyz = origin_elem.get("xyz", "0 0 0").split()
                rpy = origin_elem.get("rpy", "0 0 0").split()
                sensor["origin"] = {
                    "xyz": [float(val) for val in xyz],
                    "rpy": [float(val) for val in rpy],
                }

            # 解析参数
            for param_elem in sensor_elem.findall(f"{self.NAMESPACE}parameter"):
                name = param_elem.get("name", "")
                value = param_elem.get("value", "")
                sensor["parameters"][name] = value

            return sensor

        except Exception as e:
            logger.error(f"解析传感器失败: {e}")
            return None  # 返回None

    def get_robot_info(self) -> Dict[str, Any]:
        """从URDF获取机器人信息"""
        robot_name = self.root.get("name", "unknown_robot")

        # 尝试从机器人名称推断类型
        robot_type = self._infer_robot_type()

        return {
            "name": robot_name,
            "robot_type": robot_type,
            "joint_count": len(self.joints),
            "link_count": len(self.links),
            "sensor_count": len(self.sensors),
            "description": f"从URDF导入的机器人: {robot_name}",
        }

    def _infer_robot_type(self) -> str:
        """推断机器人类型"""
        joint_types = [j["type"] for j in self.joints]
        joint_count = len(self.joints)

        # 根据关节数量和类型推断
        if joint_count >= 20:
            return RobotType.HUMANOID.value  # 人形机器人通常有20+关节
        elif any(jt == "prismatic" for jt in joint_types):
            return RobotType.MANIPULATOR.value  # 机械臂可能有棱柱关节
        elif joint_count == 0:
            return RobotType.MOBILE_ROBOT.value  # 移动机器人可能没有关节
        else:
            return RobotType.CUSTOM.value

    def to_robot_config(self) -> Dict[str, Any]:
        """将URDF转换为机器人配置"""
        robot_info = self.get_robot_info()

        # 构建关节配置
        joints_config = []
        for joint in self.joints:
            if joint["type"] == "fixed":
                continue  # 跳过固定关节

            joint_config = {
                "name": joint["name"],
                "joint_type": joint["type"],
                "min_position": -3.14159,
                "max_position": 3.14159,
                "max_velocity": 1.0,
                "max_torque": 10.0,
                "parent_link": joint.get("parent_link", ""),
                "child_link": joint.get("child_link", ""),
                "axis": joint.get("axis", [0, 0, 1]),
                "origin": joint.get("origin"),
            }

            # 应用限制
            if joint.get("limits"):
                limits = joint["limits"]
                joint_config["min_position"] = limits.get("lower", -3.14159)
                joint_config["max_position"] = limits.get("upper", 3.14159)
                joint_config["max_velocity"] = limits.get("velocity", 1.0)
                joint_config["max_torque"] = limits.get("effort", 10.0)

            joints_config.append(joint_config)

        # 构建传感器配置
        sensors_config = []
        for sensor in self.sensors:
            sensor_config = {
                "name": sensor["name"],
                "sensor_type": sensor["type"],
                "parent_link": sensor.get("parent_link", ""),
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "parameters": sensor.get("parameters", {}),
            }

            # 设置位置
            if sensor.get("origin"):
                origin = sensor["origin"]
                sensor_config["position"] = origin.get("xyz", [0.0, 0.0, 0.0])
                # 完整处理）
                rpy = origin.get("rpy", [0.0, 0.0, 0.0])
                sensor_config["orientation"] = self._rpy_to_quaternion(rpy)

            sensors_config.append(sensor_config)

        # 构建连杆配置
        links_config = []
        for link in self.links:
            link_config = {
                "name": link["name"],
                "inertial": link.get("inertial"),
                "visuals": link.get("visual", []),
                "collisions": link.get("collision", []),
            }
            links_config.append(link_config)

        return {
            "robot_info": robot_info,
            "joints": joints_config,
            "sensors": sensors_config,
            "links": links_config,
            "materials": self.materials,
        }


class RobotConfigurationService:
    """机器人配置服务"""

    def __init__(self, db: Session):
        self.db = db

    def import_urdf(
        self,
        urdf_content: str,
        user_id: int,
        robot_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Robot]]:
        """
        导入URDF文件并创建机器人记录

        参数:
            urdf_content: URDF文件内容
            user_id: 用户ID
            robot_name: 机器人名称（如果为None，则使用URDF中的名称）
            description: 机器人描述

        返回:
            (成功状态, 消息, 机器人对象)
        """
        try:
            # 解析URDF
            parser = URDFParser(urdf_content)
            robot_config = parser.to_robot_config()
            robot_info = robot_config["robot_info"]

            # 确定机器人名称
            if robot_name is None:
                robot_name = robot_info["name"]

            if description is None:
                description = robot_info["description"]

            # 检查是否已存在同名机器人
            existing_robot = (
                self.db.query(Robot)
                .filter(Robot.name == robot_name, Robot.user_id == user_id)
                .first()
            )

            if existing_robot:
                return False, f"机器人 '{robot_name}' 已存在", None

            # 创建机器人记录
            robot = Robot(
                name=robot_name,
                description=description,
                robot_type=RobotType(robot_info["robot_type"]),
                model="URDF导入",
                manufacturer="未知",
                status=RobotStatus.OFFLINE,
                connection_type="simulation",
                connection_params={},
                configuration=robot_config,
                urdf_path=None,  # 暂时不保存URDF文件路径
                simulation_engine="gazebo",
                control_mode=ControlMode.POSITION,
                capabilities={
                    "urdf_imported": True,
                    "joint_count": robot_info["joint_count"],
                    "sensor_count": robot_info["sensor_count"],
                },
                joint_count=robot_info["joint_count"],
                sensor_count=robot_info["sensor_count"],
                user_id=user_id,
                is_public=False,
                is_default=False,
            )

            self.db.add(robot)
            self.db.flush()  # 获取机器人ID

            # 创建关节记录
            joint_records = []
            for joint_config in robot_config["joints"]:
                joint = RobotJoint(
                    robot_id=robot.id,
                    name=joint_config["name"],
                    joint_type=joint_config["joint_type"],
                    min_position=joint_config.get("min_position", -3.14159),
                    max_position=joint_config.get("max_position", 3.14159),
                    max_velocity=joint_config.get("max_velocity", 1.0),
                    max_torque=joint_config.get("max_torque", 10.0),
                    parent_link=joint_config.get("parent_link", ""),
                    child_link=joint_config.get("child_link", ""),
                    axis=str(joint_config.get("axis", [0, 0, 1])),
                    description=f"从URDF导入的关节: {joint_config['name']}",
                )
                joint_records.append(joint)

            self.db.add_all(joint_records)

            # 创建传感器记录
            sensor_records = []
            for sensor_config in robot_config["sensors"]:
                sensor = RobotSensor(
                    robot_id=robot.id,
                    name=sensor_config["name"],
                    sensor_type=sensor_config["sensor_type"],
                    position_x=sensor_config["position"][0],
                    position_y=sensor_config["position"][1],
                    position_z=sensor_config["position"][2],
                    orientation_x=sensor_config["orientation"][0],
                    orientation_y=sensor_config["orientation"][1],
                    orientation_z=sensor_config["orientation"][2],
                    orientation_w=sensor_config["orientation"][3],
                    status="offline",
                    description=f"从URDF导入的传感器: {sensor_config['name']}",
                )
                sensor_records.append(sensor)

            self.db.add_all(sensor_records)

            # 提交事务
            self.db.commit()

            logger.info(f"URDF导入成功: {robot_name} (ID: {robot.id})")
            return True, "URDF导入成功", robot

        except Exception as e:
            self.db.rollback()
            logger.error(f"URDF导入失败: {e}")
            return False, f"URDF导入失败: {str(e)}", None

    def export_urdf(self, robot_id: int) -> Tuple[bool, str, Optional[str]]:
        """
        导出机器人为URDF格式

        参数:
            robot_id: 机器人ID

        返回:
            (成功状态, 消息, URDF内容)
        """
        try:
            # 获取机器人
            robot = self.db.query(Robot).filter(Robot.id == robot_id).first()
            if not robot:
                return False, f"机器人 {robot_id} 不存在", None

            # 获取关节
            joints = (
                self.db.query(RobotJoint).filter(RobotJoint.robot_id == robot_id).all()
            )

            # 获取传感器
            sensors = (
                self.db.query(RobotSensor)
                .filter(RobotSensor.robot_id == robot_id)
                .all()
            )

            # 生成URDF
            urdf_content = self._generate_urdf(robot, joints, sensors)

            return True, "URDF导出成功", urdf_content

        except Exception as e:
            logger.error(f"URDF导出失败: {e}")
            return False, f"URDF导出失败: {str(e)}", None

    def _generate_urdf(
        self, robot: Robot, joints: List[RobotJoint], sensors: List[RobotSensor]
    ) -> str:
        """生成URDF内容"""
        lines = []

        # URDF头部
        lines.append('<?xml version="1.0"?>')
        lines.append('<robot name="{}">'.format(robot.name))
        lines.append("")

        # 材料定义（如果有）
        lines.append("  <!-- 材料定义 -->")
        lines.append('  <material name="blue">')
        lines.append('    <color rgba="0 0 0.8 1"/>')
        lines.append("  </material>")
        lines.append("")

        # 基础连杆（假设有一个基础连杆）
        lines.append("  <!-- 基础连杆 -->")
        lines.append('  <link name="base_link">')
        lines.append("    <inertial>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append('      <mass value="1.0"/>')
        lines.append(
            '      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>'
        )
        lines.append("    </inertial>")
        lines.append("    <visual>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append("      <geometry>")
        lines.append('        <box size="0.1 0.1 0.1"/>')
        lines.append("      </geometry>")
        lines.append('      <material name="blue"/>')
        lines.append("    </visual>")
        lines.append("    <collision>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append("      <geometry>")
        lines.append('        <box size="0.1 0.1 0.1"/>')
        lines.append("      </geometry>")
        lines.append("    </collision>")
        lines.append("  </link>")
        lines.append("")

        # 关节
        lines.append("  <!-- 关节定义 -->")
        for joint in joints:
            lines.append(
                '  <joint name="{}" type="{}">'.format(joint.name, joint.joint_type)
            )
            lines.append(
                '    <parent link="{}"/>'.format(
                    joint.parent_link if joint.parent_link else "base_link"
                )
            )
            lines.append('    <child link="{}-link"/>'.format(joint.name))

            # 原点
            lines.append('    <origin xyz="0 0 0" rpy="0 0 0"/>')

            # 轴
            axis_str = (
                " ".join([str(coord) for coord in eval(joint.axis)])
                if joint.axis
                else "0 0 1"
            )
            lines.append('    <axis xyz="{}"/>'.format(axis_str))

            # 限制
            lines.append(
                '    <limit lower="{:.6f}" upper="{:.6f}" velocity="{:.6f}" effort="{:.6f}"/>'.format(
                    joint.min_position,
                    joint.max_position,
                    joint.max_velocity,
                    joint.max_torque,
                )
            )

            lines.append("  </joint>")
            lines.append("")

            # 连杆
            lines.append('  <link name="{}-link">'.format(joint.name))
            lines.append("    <inertial>")
            lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
            lines.append('      <mass value="0.1"/>')
            lines.append(
                '      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>'
            )
            lines.append("    </inertial>")
            lines.append("    <visual>")
            lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
            lines.append("      <geometry>")
            lines.append('        <cylinder radius="0.02" length="0.1"/>')
            lines.append("      </geometry>")
            lines.append('      <material name="blue"/>')
            lines.append("    </visual>")
            lines.append("    <collision>")
            lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
            lines.append("      <geometry>")
            lines.append('        <cylinder radius="0.02" length="0.1"/>')
            lines.append("      </geometry>")
            lines.append("    </collision>")
            lines.append("  </link>")
            lines.append("")

        # 传感器（作为URDF扩展）
        if sensors:
            lines.append("  <!-- 传感器定义（URDF扩展） -->")
            for sensor in sensors:
                lines.append(
                    '  <sensor name="{}" type="{}">'.format(
                        sensor.name, sensor.sensor_type
                    )
                )
                lines.append(
                    "    <parent_link>{}</parent_link>".format(
                        sensor.parent_link if sensor.parent_link else "base_link"
                    )
                )
                lines.append(
                    '    <origin xyz="{:.6f} {:.6f} {:.6f}" rpy="0 0 0"/>'.format(
                        sensor.position_x, sensor.position_y, sensor.position_z
                    )
                )
                lines.append("  </sensor>")
                lines.append("")

        # URDF尾部
        lines.append("</robot>")

        return "\n".join(lines)

    def import_config_file(
        self, file_path: str, user_id: int, format: str = "auto"
    ) -> Tuple[bool, str, Optional[Robot]]:
        """
        导入配置文件（JSON/YAML）

        参数:
            file_path: 配置文件路径
            user_id: 用户ID
            format: 文件格式（auto, json, yaml）

        返回:
            (成功状态, 消息, 机器人对象)
        """
        try:
            if not os.path.exists(file_path):
                return False, f"文件不存在: {file_path}", None

            # 确定文件格式
            if format == "auto":
                ext = os.path.splitext(file_path)[1].lower()
                if ext in [".json", ".jsonc"]:
                    format = "json"
                elif ext in [".yaml", ".yml"]:
                    format = "yaml"
                else:
                    return False, f"不支持的文件格式: {ext}", None

            # 加载配置文件
            with open(file_path, "r", encoding="utf-8") as f:
                if format == "json":
                    config = json.load(f)
                elif format == "yaml":
                    config = yaml.safe_load(f)
                else:
                    return False, f"不支持的格式: {format}", None

            # 验证配置结构
            if not self._validate_config(config):
                return False, "配置文件结构无效", None

            # 创建机器人记录
            robot_name = config.get(
                "name", f"config_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            description = config.get("description", "从配置文件导入的机器人")

            # 检查是否已存在
            existing_robot = (
                self.db.query(Robot)
                .filter(Robot.name == robot_name, Robot.user_id == user_id)
                .first()
            )

            if existing_robot:
                return False, f"机器人 '{robot_name}' 已存在", None

            # 创建机器人
            robot = Robot(
                name=robot_name,
                description=description,
                robot_type=RobotType(config.get("robot_type", RobotType.CUSTOM.value)),
                model=config.get("model", "配置导入"),
                manufacturer=config.get("manufacturer", "未知"),
                status=RobotStatus.OFFLINE,
                connection_type=config.get("connection_type", "simulation"),
                connection_params=config.get("connection_params", {}),
                configuration=config.get("configuration", {}),
                urdf_path=config.get("urdf_path"),
                simulation_engine=config.get("simulation_engine", "gazebo"),
                control_mode=ControlMode(
                    config.get("control_mode", ControlMode.POSITION.value)
                ),
                capabilities=config.get("capabilities", {}),
                joint_count=len(config.get("joints", [])),
                sensor_count=len(config.get("sensors", [])),
                user_id=user_id,
                is_public=config.get("is_public", False),
                is_default=config.get("is_default", False),
            )

            self.db.add(robot)
            self.db.flush()

            # 创建关节记录
            joint_records = []
            for joint_config in config.get("joints", []):
                joint = RobotJoint(
                    robot_id=robot.id,
                    name=joint_config.get("name", f"joint_{len(joint_records)}"),
                    joint_type=joint_config.get("joint_type", "revolute"),
                    min_position=joint_config.get("min_position", -3.14159),
                    max_position=joint_config.get("max_position", 3.14159),
                    max_velocity=joint_config.get("max_velocity", 1.0),
                    max_torque=joint_config.get("max_torque", 10.0),
                    offset=joint_config.get("offset", 0.0),
                    direction=joint_config.get("direction", 1.0),
                    parent_link=joint_config.get("parent_link", ""),
                    child_link=joint_config.get("child_link", ""),
                    axis=str(joint_config.get("axis", "z")),
                    description=joint_config.get("description", ""),
                )
                joint_records.append(joint)

            if joint_records:
                self.db.add_all(joint_records)

            # 创建传感器记录
            sensor_records = []
            for sensor_config in config.get("sensors", []):
                sensor = RobotSensor(
                    robot_id=robot.id,
                    name=sensor_config.get("name", f"sensor_{len(sensor_records)}"),
                    sensor_type=sensor_config.get("sensor_type", "imu"),
                    model=sensor_config.get("model", ""),
                    manufacturer=sensor_config.get("manufacturer", ""),
                    sampling_rate=sensor_config.get("sampling_rate", 100.0),
                    position_x=sensor_config.get("position_x", 0.0),
                    position_y=sensor_config.get("position_y", 0.0),
                    position_z=sensor_config.get("position_z", 0.0),
                    orientation_x=sensor_config.get("orientation_x", 0.0),
                    orientation_y=sensor_config.get("orientation_y", 0.0),
                    orientation_z=sensor_config.get("orientation_z", 0.0),
                    orientation_w=sensor_config.get("orientation_w", 1.0),
                    status="offline",
                    description=sensor_config.get("description", ""),
                )
                sensor_records.append(sensor)

            if sensor_records:
                self.db.add_all(sensor_records)

            self.db.commit()

            logger.info(f"配置文件导入成功: {robot_name} (ID: {robot.id})")
            return True, "配置文件导入成功", robot

        except Exception as e:
            self.db.rollback()
            logger.error(f"配置文件导入失败: {e}")
            return False, f"配置文件导入失败: {str(e)}", None

    def export_config_file(
        self, robot_id: int, format: str = "json"
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        导出机器人配置为字典（可用于JSON/YAML序列化）

        参数:
            robot_id: 机器人ID
            format: 格式（json或yaml）

        返回:
            (成功状态, 消息, 配置字典)
        """
        try:
            # 获取机器人
            robot = self.db.query(Robot).filter(Robot.id == robot_id).first()
            if not robot:
                return False, f"机器人 {robot_id} 不存在", None

            # 获取关节
            joints = (
                self.db.query(RobotJoint).filter(RobotJoint.robot_id == robot_id).all()
            )

            # 获取传感器
            sensors = (
                self.db.query(RobotSensor)
                .filter(RobotSensor.robot_id == robot_id)
                .all()
            )

            # 构建配置字典
            config = {
                "name": robot.name,
                "description": robot.description,
                "robot_type": robot.robot_type.value,
                "model": robot.model,
                "manufacturer": robot.manufacturer,
                "connection_type": robot.connection_type,
                "connection_params": robot.connection_params,
                "configuration": robot.configuration,
                "urdf_path": robot.urdf_path,
                "simulation_engine": robot.simulation_engine,
                "control_mode": robot.control_mode.value,
                "capabilities": robot.capabilities,
                "joints": [joint.to_dict() for joint in joints],
                "sensors": [sensor.to_dict() for sensor in sensors],
                "is_public": robot.is_public,
                "is_default": robot.is_default,
            }

            return True, "配置导出成功", config

        except Exception as e:
            logger.error(f"配置导出失败: {e}")
            return False, f"配置导出失败: {str(e)}", None

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置文件结构"""
        required_fields = ["name"]
        for field in required_fields:
            if field not in config:
                logger.error(f"配置缺少必需字段: {field}")
                return False

        # 验证关节配置
        joints = config.get("joints", [])
        for joint in joints:
            if "name" not in joint:
                logger.error("关节配置缺少名称字段")
                return False

        # 验证传感器配置
        sensors = config.get("sensors", [])
        for sensor in sensors:
            if "name" not in sensor or "sensor_type" not in sensor:
                logger.error("传感器配置缺少名称或类型字段")
                return False

        return True


# 工具函数
def rpy_to_quaternion(rpy):
    """将RPY（滚转、俯仰、偏航）欧拉角转换为四元数

    使用完整的转换公式，支持任意旋转顺序（默认ZYX顺序）

    参数:
        rpy: 包含滚转(roll)、俯仰(pitch)、偏航(yaw)的列表或元组，单位为弧度

    返回:
        四元数 [x, y, z, w]
    """
    import math

    # 确保输入是列表或元组
    if not isinstance(rpy, (list, tuple)) or len(rpy) != 3:
        raise ValueError("RPY输入必须是包含3个元素的列表或元组")

    roll, pitch, yaw = rpy

    # 计算半角
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # ZYX顺序（偏航->俯仰->滚转）的四元数乘法
    # q = q_yaw * q_pitch * q_roll

    # 计算四元数分量
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # 确保单位四元数（数值稳定性）
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm > 0:
        w /= norm
        x /= norm
        y /= norm
        z /= norm

    return [x, y, z, w]


def quaternion_to_rpy(q):
    """将四元数转换为RPY（滚转、俯仰、偏航）欧拉角

    参数:
        q: 四元数 [x, y, z, w]

    返回:
        [roll, pitch, yaw] 单位为弧度
    """
    import math

    x, y, z, w = q

    # 计算滚转（x轴旋转）
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # 计算俯仰（y轴旋转）
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        # 使用90度处理奇点
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # 计算偏航（z轴旋转）
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]


def normalize_quaternion(q):
    """归一化四元数

    参数:
        q: 四元数 [x, y, z, w]

    返回:
        归一化的四元数
    """
    import math

    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)

    if norm == 0:
        return [0.0, 0.0, 0.0, 1.0]

    return [x / norm, y / norm, z / norm, w / norm]
