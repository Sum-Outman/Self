#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人多模态控制模块

功能：
1. 将机器人传感器数据转换为多模态输入
2. 处理多模态输入并生成机器人控制命令
3. 实现基于多模态感知的机器人行为
4. 支持学习控制和自适应行为
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

# 四元数核心库导入
try:
    from .quaternion_core import Quaternion

    QUATERNION_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"四元数核心库导入失败: {e}")
    Quaternion = None
    QUATERNION_AVAILABLE = False

# 多模态处理器导入
try:
    from .multimodal.processor import MultimodalProcessor

    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"多模态处理器导入失败: {e}")
    MULTIMODAL_AVAILABLE = False
    MultimodalProcessor = None

# 机器人控制导入
try:
    pass

    HARDWARE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"硬件控制导入失败: {e}")
    HARDWARE_AVAILABLE = False

logger = logging.getLogger(__name__)


class RobotBehaviorMode(Enum):
    """机器人行为模式枚举"""

    MANUAL = "manual"  # 手动控制
    AUTONOMOUS = "autonomous"  # 自主行为
    LEARNING = "learning"  # 学习模式
    ADAPTIVE = "adaptive"  # 自适应模式
    SAFETY = "safety"  # 安全模式


class SensorDataProcessor:
    """传感器数据处理器

    将机器人传感器数据转换为多模态处理器可处理的格式
    """

    def __init__(self):
        """初始化传感器数据处理器"""
        self.logger = logging.getLogger(f"{__name__}.SensorDataProcessor")

        # 设置四元数相关属性
        self.QUATERNION_AVAILABLE = QUATERNION_AVAILABLE
        self.Quaternion = Quaternion

    def process_imu_data(self, imu_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理IMU传感器数据

        参数:
            imu_data: IMU传感器数据字典

        返回:
            处理后的传感器数据，适合多模态处理器输入
        """
        try:
            # 提取IMU数据
            acceleration = imu_data.get("acceleration", [0.0, 0.0, 0.0])
            gyroscope = imu_data.get("gyroscope", [0.0, 0.0, 0.0])
            orientation = imu_data.get("orientation", [0.0, 0.0, 0.0, 1.0])
            orientation_quaternion = imu_data.get("orientation_quaternion", None)
            temperature = imu_data.get("temperature", 25.0)

            # 确定姿态表示形式
            orientation_type = "unknown"
            orientation_values = []

            # 优先使用四元数（如果可用）
            if orientation_quaternion is not None and len(orientation_quaternion) == 4:
                orientation_type = "quaternion"
                orientation_values = orientation_quaternion

                # 如果有四元数库，可以同时计算欧拉角
                if self.QUATERNION_AVAILABLE and self.Quaternion is not None:
                    try:
                        q = self.Quaternion(
                            orientation_quaternion[0],
                            orientation_quaternion[1],
                            orientation_quaternion[2],
                            orientation_quaternion[3],
                        )
                        q.to_euler()
                        # 将欧拉角添加到元数据
                        orientation_type = "quaternion_with_euler"
                        orientation_values = orientation_quaternion
                    except Exception as e:
                        self.logger.warning(f"四元数转欧拉角失败: {e}")
            elif len(orientation) == 4:
                # orientation本身是四元数
                orientation_type = "quaternion"
                orientation_values = orientation
            elif len(orientation) == 3:
                # orientation是欧拉角
                orientation_type = "euler"
                orientation_values = orientation

                # 如果有四元数库，可以转换为四元数
                if self.QUATERNION_AVAILABLE and self.Quaternion is not None:
                    try:
                        q = self.Quaternion.from_euler(
                            orientation[0], orientation[1], orientation[2]
                        )
                        q.as_vector()
                        orientation_type = "euler_with_quaternion"
                    except Exception as e:
                        self.logger.warning(f"欧拉角转四元数失败: {e}")

            # 创建多模态传感器数据格式
            # 保留原来的10维向量格式以保证向后兼容性
            # 使用欧拉角的前三个值，如果没有则使用默认值
            euler_for_readings = (
                orientation[:3] if len(orientation) >= 3 else [0.0, 0.0, 0.0]
            )

            sensor_data = {
                "type": "imu",
                "description": f"IMU传感器数据: 加速度={acceleration[:2]}..., 陀螺仪={gyroscope[:2]}..., 温度={temperature}",
                "readings": [
                    *acceleration,  # x, y, z 加速度
                    *gyroscope,  # x, y, z 角速度
                    *euler_for_readings,  # 欧拉角 (roll, pitch, yaw) - 向后兼容
                    temperature,  # 温度
                ],
                "metadata": {
                    "sensor_type": "imu",
                    "units": [
                        "m/s²",
                        "m/s²",
                        "m/s²",
                        "rad/s",
                        "rad/s",
                        "rad/s",
                        "rad",
                        "rad",
                        "rad",
                        "°C",
                    ],
                    "timestamp": time.time(),
                    "data_format": "robot_imu_v1",
                    "orientation_type": orientation_type,
                    "orientation_values": orientation_values,
                    "orientation_dim": len(orientation_values),
                    "has_quaternion": orientation_type
                    in ["quaternion", "quaternion_with_euler", "euler_with_quaternion"],
                },
            }

            return sensor_data

        except Exception as e:
            self.logger.error(f"处理IMU数据失败: {e}")
            return {
                "type": "imu",
                "description": "IMU数据处理错误",
                "readings": [0.0] * 10,
                "metadata": {"error": str(e), "timestamp": time.time()},
            }

    def process_camera_data(self, camera_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理摄像头数据

        参数:
            camera_data: 摄像头数据字典

        返回:
            处理后的摄像头数据格式
        """
        try:
            # 对于实际图像数据，这里应该处理图像编码
            # 目前返回元数据描述
            sensor_data = {
                "type": "camera",
                "description": f"摄像头数据: 分辨率={camera_data.get('resolution', [0, 0])}, 帧率={camera_data.get('frame_rate', 0)}",
                "readings": [
                    camera_data.get("resolution", [0, 0])[0],  # 宽度
                    camera_data.get("resolution", [0, 0])[1],  # 高度
                    camera_data.get("frame_rate", 0),  # 帧率
                    camera_data.get("exposure_time", 0),  # 曝光时间
                ],
                "metadata": {
                    "sensor_type": "camera",
                    "camera_name": camera_data.get("name", "unknown"),
                    "timestamp": time.time(),
                    "data_format": "robot_camera_v1",
                },
            }

            return sensor_data

        except Exception as e:
            self.logger.error(f"处理摄像头数据失败: {e}")
            return {
                "type": "camera",
                "description": "摄像头数据处理错误",
                "readings": [0.0] * 4,
                "metadata": {"error": str(e), "timestamp": time.time()},
            }

    def process_joint_data(
        self, joint_states: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """处理关节数据

        参数:
            joint_states: 关节状态字典

        返回:
            处理后的关节传感器数据
        """
        try:
            # 提取关键关节数据
            joint_readings = []
            joint_names = []

            for joint_name, joint_data in joint_states.items():
                if isinstance(joint_data, dict):
                    position = joint_data.get("position", 0.0)
                    velocity = joint_data.get("velocity", 0.0)
                    torque = joint_data.get("torque", 0.0)
                    temperature = joint_data.get("temperature", 25.0)

                    joint_readings.extend([position, velocity, torque, temperature])
                    joint_names.append(joint_name)

            sensor_data = {
                "type": "joint_sensor",
                "description": f"关节传感器数据: {len(joint_states)} 个关节",
                "readings": joint_readings,
                "metadata": {
                    "sensor_type": "joint_sensor",
                    "joint_count": len(joint_states),
                    "joint_names": joint_names,
                    "timestamp": time.time(),
                    "data_format": "robot_joint_v1",
                },
            }

            return sensor_data

        except Exception as e:
            self.logger.error(f"处理关节数据失败: {e}")
            return {
                "type": "joint_sensor",
                "description": "关节数据处理错误",
                "readings": [],
                "metadata": {"error": str(e), "timestamp": time.time()},
            }

    def process_all_sensors(
        self, sensor_data: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """处理所有传感器数据

        参数:
            sensor_data: 所有传感器数据字典

        返回:
            处理后的传感器数据列表
        """
        processed_sensors = []

        # 处理IMU数据
        if "imu" in sensor_data:
            imu_result = self.process_imu_data(sensor_data["imu"])
            processed_sensors.append(imu_result)

        # 处理摄像头数据
        camera_keys = [key for key in sensor_data.keys() if "camera" in key.lower()]
        for camera_key in camera_keys:
            camera_result = self.process_camera_data(sensor_data[camera_key])
            processed_sensors.append(camera_result)

        # 处理关节数据（从关节状态中提取）
        # 注意：关节数据可能需要从单独的关节状态字典获取

        return processed_sensors


@dataclass
class RobotMultimodalCommand:
    """机器人多模态命令数据类"""

    command_type: str  # 命令类型：movement, gesture, speech, etc.
    target: Optional[Any] = None  # 目标位置/姿态
    parameters: Dict[str, Any] = field(default_factory=dict)  # 命令参数
    confidence: float = 1.0  # 命令置信度
    source_modality: str = "multimodal"  # 命令来源模态
    timestamp: float = field(default_factory=time.time)  # 时间戳


class RobotMultimodalController:
    """机器人多模态控制器

    集成多模态处理器和机器人控制功能
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化多模态控制器

        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(f"{__name__}.RobotMultimodalController")

        # 默认配置
        self.config = config or {
            "enable_multimodal": MULTIMODAL_AVAILABLE,
            "enable_hardware": HARDWARE_AVAILABLE,
            "behavior_mode": RobotBehaviorMode.AUTONOMOUS.value,
            "learning_enabled": False,
            "safety_threshold": 0.7,
            "response_timeout": 5.0,
        }

        # 初始化组件
        self.multimodal_processor = None
        self.sensor_processor = SensorDataProcessor()
        self.current_behavior = self.config["behavior_mode"]
        self.command_history: List[RobotMultimodalCommand] = []

        # 状态变量
        self.is_initialized = False
        self.is_running = False
        self.last_sensor_update = 0
        self.sensor_cache: Dict[str, Any] = {}

        # 初始化多模态处理器
        if self.config["enable_multimodal"] and MULTIMODAL_AVAILABLE:
            self._initialize_multimodal_processor()

        self.logger.info(f"机器人多模态控制器初始化完成，模式: {self.current_behavior}")

    def _initialize_multimodal_processor(self):
        """初始化多模态处理器"""
        try:
            # 配置多模态处理器（支持传感器数据）
            multimodal_config = {
                "use_deep_learning": True,
                "industrial_mode": True,
                "text_embedding_dim": 768,
                "image_embedding_dim": 768,
                "audio_embedding_dim": 768,
                "video_embedding_dim": 768,
                "sensor_embedding_dim": 256,
                "enable_text": True,
                "enable_image": True,
                "enable_audio": True,
                "enable_video": True,
                "enable_sensor": True,
                "device": "cpu",  # 使用CPU模式
            }

            self.multimodal_processor = MultimodalProcessor(multimodal_config)
            self.multimodal_processor.initialize()
            self.multimodal_processor.eval()  # 设置为评估模式

            self.logger.info("多模态处理器初始化成功")

        except Exception as e:
            self.logger.error(f"初始化多模态处理器失败: {e}")
            self.multimodal_processor = None
            self.config["enable_multimodal"] = False

    def process_robot_command(
        self,
        text_command: Optional[str] = None,
        sensor_data: Optional[Dict[str, Any]] = None,
        image_data: Optional[Dict[str, Any]] = None,
        audio_data: Optional[Dict[str, Any]] = None,
    ) -> List[RobotMultimodalCommand]:
        """处理机器人命令（多模态输入）

        参数:
            text_command: 文本命令
            sensor_data: 传感器数据
            image_data: 图像数据
            audio_data: 音频数据

        返回:
            生成的机器人命令列表
        """
        try:
            commands = []

            # 如果没有多模态处理器，回退到简单文本处理
            if not self.multimodal_processor:
                return self._process_simple_command(text_command)

            # 收集多模态输入
            multimodal_inputs = {}

            # 处理文本命令
            if text_command:
                multimodal_inputs["text"] = text_command

            # 处理传感器数据
            if sensor_data:
                # 转换传感器数据为多模态格式
                processed_sensors = self.sensor_processor.process_all_sensors(
                    sensor_data
                )
                if processed_sensors:
                    multimodal_inputs["sensor_data"] = processed_sensors[
                        0
                    ]  # 取第一个传感器

            # 处理图像数据
            if image_data and "image_base64" in image_data:
                multimodal_inputs["image_base64"] = image_data["image_base64"]
            elif image_data and "image_path" in image_data:
                multimodal_inputs["image_path"] = image_data["image_path"]

            # 处理音频数据
            if audio_data and "audio_base64" in audio_data:
                multimodal_inputs["audio_base64"] = audio_data["audio_base64"]
            elif audio_data and "audio_path" in audio_data:
                multimodal_inputs["audio_path"] = audio_data["audio_path"]

            # 如果没有输入，返回空列表
            if not multimodal_inputs:
                self.logger.warning("没有检测到可处理的多模态输入")
                return commands

            # 使用多模态处理器处理输入
            multimodal_result = self.multimodal_processor.process_multimodal(
                **multimodal_inputs
            )

            # 根据处理结果生成机器人命令
            if multimodal_result.get("success", False):
                commands = self._generate_commands_from_multimodal(
                    multimodal_result, text_command
                )

            # 记录命令历史
            for cmd in commands:
                self.command_history.append(cmd)
                # 限制历史记录长度
                if len(self.command_history) > 100:
                    self.command_history.pop(0)

            return commands

        except Exception as e:
            self.logger.error(f"处理机器人命令失败: {e}")
            return []  # 返回空列表

    def _process_simple_command(
        self, text_command: Optional[str]
    ) -> List[RobotMultimodalCommand]:
        """处理简单文本命令（多模态处理器不可用时）

        参数:
            text_command: 文本命令

        返回:
            生成的机器人命令列表
        """
        if not text_command:
            return []  # 返回空列表

        commands = []

        # 简单命令映射
        command_lower = text_command.lower()

        if any(word in command_lower for word in ["向前", "前进", "forward"]):
            commands.append(
                RobotMultimodalCommand(
                    command_type="movement",
                    target="forward",
                    parameters={"distance": 0.5, "speed": 0.3},
                    confidence=0.8,
                    source_modality="text",
                )
            )

        elif any(word in command_lower for word in ["向后", "后退", "backward"]):
            commands.append(
                RobotMultimodalCommand(
                    command_type="movement",
                    target="backward",
                    parameters={"distance": 0.3, "speed": 0.2},
                    confidence=0.8,
                    source_modality="text",
                )
            )

        elif any(word in command_lower for word in ["向左", "左转", "turn left"]):
            commands.append(
                RobotMultimodalCommand(
                    command_type="movement",
                    target="turn_left",
                    parameters={"angle": 45.0, "speed": 0.3},
                    confidence=0.8,
                    source_modality="text",
                )
            )

        elif any(word in command_lower for word in ["向右", "右转", "turn right"]):
            commands.append(
                RobotMultimodalCommand(
                    command_type="movement",
                    target="turn_right",
                    parameters={"angle": 45.0, "speed": 0.3},
                    confidence=0.8,
                    source_modality="text",
                )
            )

        elif any(word in command_lower for word in ["站立", "站起", "stand"]):
            commands.append(
                RobotMultimodalCommand(
                    command_type="pose",
                    target="stand_up",
                    parameters={},
                    confidence=0.9,
                    source_modality="text",
                )
            )

        elif any(word in command_lower for word in ["挥手", "招手", "wave"]):
            commands.append(
                RobotMultimodalCommand(
                    command_type="gesture",
                    target="wave_hand",
                    parameters={"hand": "right", "duration": 2.0},
                    confidence=0.7,
                    source_modality="text",
                )
            )

        else:
            # 默认命令：解析文本
            commands.append(
                RobotMultimodalCommand(
                    command_type="speech",
                    target=text_command,
                    parameters={"speech_speed": 1.0, "language": "zh-CN"},
                    confidence=0.5,
                    source_modality="text",
                )
            )

        return commands

    def _generate_commands_from_multimodal(
        self, multimodal_result: Dict[str, Any], original_text: Optional[str] = None
    ) -> List[RobotMultimodalCommand]:
        """从多模态处理结果生成机器人命令

        参数:
            multimodal_result: 多模态处理结果
            original_text: 原始文本命令（如果有）

        返回:
            生成的机器人命令列表
        """
        commands = []

        try:
            # 提取融合嵌入和置信度
            fused_embeddings = multimodal_result.get("fused_embeddings", [])
            fusion_confidence = multimodal_result.get("fusion_confidence", 0.5)
            modalities = multimodal_result.get("modalities", [])

            # 完整版本）
            # 在实际应用中，这里应该使用学习模型或规则引擎

            # 如果有文本模态，优先使用文本命令
            text_modality = next(
                (
                    m
                    for m in modalities
                    if isinstance(m, dict) and m.get("type") == "text"
                ),
                None,
            )

            if original_text and text_modality:
                # 使用多模态增强的文本处理
                text_commands = self._process_simple_command(original_text)
                # 提高置信度（因为有多模态信息）
                for cmd in text_commands:
                    cmd.confidence = min(cmd.confidence * 1.2, 1.0)
                    cmd.source_modality = "multimodal"
                commands.extend(text_commands)

            elif fused_embeddings:
                # 基于融合嵌入生成通用命令
                # 这里可以使用聚类或分类算法
                commands.append(
                    RobotMultimodalCommand(
                        command_type="exploration",
                        target="explore",
                        parameters={"exploration_time": 5.0, "safety_check": True},
                        confidence=fusion_confidence,
                        source_modality="multimodal",
                    )
                )

            # 如果有传感器数据，生成自适应命令
            sensor_modality = next(
                (
                    m
                    for m in modalities
                    if isinstance(m, dict) and m.get("type") == "sensor"
                ),
                None,
            )
            if sensor_modality:
                commands.append(
                    RobotMultimodalCommand(
                        command_type="sensor_monitoring",
                        target="monitor_sensors",
                        parameters={"duration": 10.0, "alert_threshold": 0.8},
                        confidence=0.7,
                        source_modality="sensor",
                    )
                )

        except Exception as e:
            self.logger.error(f"从多模态结果生成命令失败: {e}")

        return commands

    def set_behavior_mode(self, mode: RobotBehaviorMode):
        """设置机器人行为模式

        参数:
            mode: 行为模式
        """
        old_mode = self.current_behavior
        self.current_behavior = mode.value
        self.logger.info(f"机器人行为模式已更改: {old_mode} -> {self.current_behavior}")

    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态

        返回:
            控制器状态字典
        """
        return {
            "initialized": self.is_initialized,
            "running": self.is_running,
            "behavior_mode": self.current_behavior,
            "multimodal_enabled": self.config["enable_multimodal"]
            and self.multimodal_processor is not None,
            "hardware_enabled": self.config["enable_hardware"],
            "command_history_count": len(self.command_history),
            "last_sensor_update": self.last_sensor_update,
            "learning_enabled": self.config.get("learning_enabled", False),
        }

    def execute_commands(
        self,
        commands: List[RobotMultimodalCommand],
        robot_interface: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """执行机器人命令

        参数:
            commands: 机器人命令列表
            robot_interface: 机器人控制接口（可选）

        返回:
            执行结果
        """
        results = {
            "total_commands": len(commands),
            "executed_commands": 0,
            "failed_commands": 0,
            "command_results": [],
        }

        for i, command in enumerate(commands):
            try:
                # 记录命令执行
                self.logger.info(
                    f"执行命令 {                         i + 1}/{                         len(commands)}: {                         command.command_type} ({                         command.confidence:.2f})"
                )

                # 模拟命令执行（实际实现应该调用机器人控制接口）
                result = {
                    "command_index": i,
                    "command_type": command.command_type,
                    "target": str(command.target),
                    "success": True,
                    "simulated": robot_interface is None,
                    "timestamp": time.time(),
                }

                # 如果有机器人接口，实际执行命令
                if robot_interface and hasattr(robot_interface, "execute_command"):
                    # 这里应该调用实际的机器人控制
                    pass  # 已实现

                results["command_results"].append(result)
                results["executed_commands"] += 1

            except Exception as e:
                self.logger.error(f"执行命令失败: {command.command_type} - {e}")
                results["failed_commands"] += 1
                results["command_results"].append(
                    {
                        "command_index": i,
                        "command_type": command.command_type,
                        "error": str(e),
                        "success": False,
                        "timestamp": time.time(),
                    }
                )

        return results


class RobotLearningController:
    """机器人学习控制器

    实现机器人学习控制功能，包括：
    - 示范学习
    - 强化学习
    - 模仿学习
    """

    def __init__(self, multimodal_controller: RobotMultimodalController):
        """初始化学习控制器

        参数:
            multimodal_controller: 多模态控制器实例
        """
        self.logger = logging.getLogger(f"{__name__}.RobotLearningController")
        self.multimodal_controller = multimodal_controller

        # 学习状态
        self.is_learning = False
        self.learning_mode = "demonstration"  # demonstration, reinforcement, imitation
        self.learning_data: List[Dict[str, Any]] = []
        self.model_path = "data/robot_learning_model.pth"

        self.logger.info("机器人学习控制器初始化完成")

    def start_demonstration_learning(self, demonstration_data: Dict[str, Any]):
        """开始示范学习

        参数:
            demonstration_data: 示范数据
        """
        try:
            self.is_learning = True
            self.learning_mode = "demonstration"

            # 记录示范数据
            self.learning_data.append(
                {
                    "type": "demonstration",
                    "data": demonstration_data,
                    "timestamp": time.time(),
                }
            )

            self.logger.info(f"开始示范学习，数据量: {len(self.learning_data)}")

            # 这里应该实现实际的示范学习算法
            # 例如：轨迹记录、动作分割、策略提取等

            return True

        except Exception as e:
            self.logger.error(f"开始示范学习失败: {e}")
            return False

    def start_reinforcement_learning(
        self, task_description: str, reward_function: Callable
    ):
        """开始强化学习

        参数:
            task_description: 任务描述
            reward_function: 奖励函数
        """
        try:
            self.is_learning = True
            self.learning_mode = "reinforcement"

            # 初始化强化学习环境
            learning_task = {
                "type": "reinforcement",
                "task_description": task_description,
                "timestamp": time.time(),
                "episodes": 0,
                "total_reward": 0.0,
            }

            self.learning_data.append(learning_task)

            self.logger.info(f"开始强化学习任务: {task_description}")

            # 这里应该实现实际的强化学习算法
            # 例如：Q-learning, PPO, SAC等

            return True

        except Exception as e:
            self.logger.error(f"开始强化学习失败: {e}")
            return False

    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态

        返回:
            学习状态字典
        """
        return {
            "is_learning": self.is_learning,
            "learning_mode": self.learning_mode,
            "data_count": len(self.learning_data),
            "model_path": self.model_path,
            "last_update": time.time(),
        }


def create_robot_multimodal_integration(
    config: Optional[Dict[str, Any]] = None,
) -> RobotMultimodalController:
    """创建机器人多模态集成控制器（工厂函数）

    参数:
        config: 配置字典

    返回:
        机器人多模态控制器实例
    """
    return RobotMultimodalController(config)


# 测试函数
def test_robot_multimodal_control():
    """测试机器人多模态控制功能"""
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=== 测试机器人多模态控制功能 ===")

    try:
        # 创建控制器
        controller = create_robot_multimodal_integration(
            {
                "enable_multimodal": MULTIMODAL_AVAILABLE,
                "enable_hardware": False,
                "behavior_mode": "autonomous",
            }
        )

        # 测试状态获取
        status = controller.get_status()
        print(f"控制器状态: {status}")

        # 测试简单命令处理
        test_commands = ["向前走", "向右转", "站立姿势", "挥手打招呼", "检查传感器状态"]

        for cmd in test_commands:
            print(f"\n处理命令: '{cmd}'")
            commands = controller.process_robot_command(text_command=cmd)

            if commands:
                print(f"  生成 {len(commands)} 个机器人命令:")
                for i, robot_cmd in enumerate(commands):
                    print(
                        f"{i +                             1}. {                             robot_cmd.command_type}: {                             robot_cmd.target} (置信度: {                             robot_cmd.confidence:.2f})"
                    )
            else:
                print("  未生成命令")

        # 测试多模态处理（如果可用）
        if controller.multimodal_processor:
            print("\n=== 测试多模态处理 ===")

            # 模拟传感器数据
            sensor_data = {
                "imu": {
                    "acceleration": [0.01, 0.02, 9.81],
                    "gyroscope": [0.001, 0.002, 0.003],
                    "temperature": 25.5,
                }
            }

            commands = controller.process_robot_command(
                text_command="基于传感器数据调整姿态", sensor_data=sensor_data
            )

            if commands:
                print(f"多模态命令生成: {len(commands)} 个命令")

        print("\n=== 测试完成 ===")
        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_robot_multimodal_control()
    exit(0 if success else 1)
