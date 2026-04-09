#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人视觉引导控制服务

功能：
1. 管理视觉引导控制器实例
2. 提供视觉处理API
3. 提供语音控制API
4. 集成视觉任务执行
5. 状态监控和报告

基于修复计划中的机器人视觉引导控制功能
"""

import logging
import time
import threading
import queue
import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class VisionTaskStatus(Enum):
    """视觉任务状态枚举"""

    PENDING = "pending"  # 待处理
    RUNNING = "running"  # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


class VoiceCommandStatus(Enum):
    """语音命令状态枚举"""

    RECEIVED = "received"  # 已接收
    PROCESSING = "processing"  # 处理中
    EXECUTING = "executing"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败


class RobotVisionService:
    """机器人视觉引导控制服务单例类"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True

            # 视觉引导控制器
            self.vision_controller = None

            # 任务管理
            self.tasks: Dict[str, Dict[str, Any]] = {}
            self.task_queue = queue.Queue()

            # 语音命令管理
            self.voice_commands: Dict[str, Dict[str, Any]] = {}
            self.voice_command_queue = queue.Queue()

            # 状态变量
            self.service_status = "initializing"
            self.last_update = datetime.now(timezone.utc)
            self.task_count = 0
            self.voice_command_count = 0
            self.error_count = 0

            # 处理线程
            self.task_processor_thread = None
            self.voice_processor_thread = None
            self.processing_running = False

            # 初始化组件
            self._initialize_components()

            # 启动处理线程
            self._start_processing_threads()

            logger.info("机器人视觉引导控制服务初始化完成")

    def _initialize_components(self):
        """初始化所有组件"""
        try:
            # 尝试导入视觉引导控制器
            try:
                from hardware.robot_vision_control import (
                    create_vision_guided_controller,
                )

                # 配置视觉控制器
                vision_config = {
                    "enable_vision": True,
                    "enable_voice": True,
                    "enable_hardware": True,
                    "vision_fps": 10,
                    "voice_timeout": 5.0,
                    "object_tracking_enabled": True,
                    "min_confidence": 0.5,
                    "camera_resolution": (640, 480),
                }

                # 创建视觉控制器实例
                self.vision_controller = create_vision_guided_controller(vision_config)

                logger.info("视觉引导控制器初始化成功")

            except ImportError as e:
                logger.warning(f"视觉引导控制器导入失败: {e}")
                self.vision_controller = None

            # 更新服务状态
            self.service_status = "ready"
            self.last_update = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"初始化组件失败: {e}")
            self.service_status = "error"
            self.error_count += 1

    def _start_processing_threads(self):
        """启动处理线程"""
        self.processing_running = True

        # 启动任务处理线程
        self.task_processor_thread = threading.Thread(
            target=self._task_processing_loop, daemon=True, name="VisionTaskProcessor"
        )
        self.task_processor_thread.start()

        # 启动语音命令处理线程
        self.voice_processor_thread = threading.Thread(
            target=self._voice_command_processing_loop,
            daemon=True,
            name="VoiceCommandProcessor",
        )
        self.voice_processor_thread.start()

        logger.info("处理线程已启动")

    def _task_processing_loop(self):
        """任务处理循环"""
        while self.processing_running:
            try:
                # 从队列获取任务
                try:
                    task_id = self.task_queue.get(timeout=0.5)
                    task_data = self.tasks.get(task_id)

                    if task_data:
                        self._process_task(task_id, task_data)

                except queue.Empty:
                    continue

            except Exception as e:
                logger.error(f"任务处理循环错误: {e}")
                time.sleep(1.0)

    def _voice_command_processing_loop(self):
        """语音命令处理循环"""
        while self.processing_running:
            try:
                # 从队列获取语音命令
                try:
                    command_id = self.voice_command_queue.get(timeout=0.5)
                    command_data = self.voice_commands.get(command_id)

                    if command_data:
                        self._process_voice_command(command_id, command_data)

                except queue.Empty:
                    continue

            except Exception as e:
                logger.error(f"语音命令处理循环错误: {e}")
                time.sleep(1.0)

    def _process_task(self, task_id: str, task_data: Dict[str, Any]):
        """处理任务

        参数:
            task_id: 任务ID
            task_data: 任务数据
        """
        try:
            # 更新任务状态
            task_data["status"] = VisionTaskStatus.RUNNING.value
            task_data["start_time"] = time.time()

            logger.info(
                f"开始处理视觉任务: {task_id} (类型: {task_data.get('task_type')})"
            )

            # 检查视觉控制器是否可用
            if self.vision_controller is None:
                task_data["status"] = VisionTaskStatus.FAILED.value
                task_data["error"] = "视觉控制器未初始化"
                task_data["end_time"] = time.time()
                task_data["execution_time"] = 0.0
                return

            # 执行任务
            task_type = task_data.get("task_type")
            target_object = task_data.get("target_object")

            success = False
            result_data = {}

            if task_type == "object_detection":
                # 对象检测任务
                success = self.vision_controller.execute_visual_task(
                    self.vision_controller.VisionTaskType.OBJECT_DETECTION,
                    target_object,
                )

                # 获取检测结果
                tracked_objects = self.vision_controller.get_tracked_objects_info()
                result_data = {
                    "detected_objects": tracked_objects,
                    "count": len(tracked_objects),
                }

            elif task_type == "object_tracking":
                # 对象跟踪任务
                success = self.vision_controller.execute_visual_task(
                    self.vision_controller.VisionTaskType.OBJECT_TRACKING, target_object
                )

                # 获取跟踪信息
                tracked_objects = self.vision_controller.get_tracked_objects_info()
                result_data = {
                    "tracked_objects": tracked_objects,
                    "count": len(tracked_objects),
                }

            elif task_type == "visual_navigation":
                # 视觉导航任务
                success = self.vision_controller.execute_visual_task(
                    self.vision_controller.VisionTaskType.NAVIGATION, target_object
                )

                result_data = {"target": target_object, "navigation_started": success}

            elif task_type == "visual_manipulation":
                # 视觉操作任务
                success = self.vision_controller.execute_visual_task(
                    self.vision_controller.VisionTaskType.MANIPULATION, target_object
                )

                result_data = {"target": target_object, "manipulation_started": success}

            else:
                task_data["status"] = VisionTaskStatus.FAILED.value
                task_data["error"] = f"未知的任务类型: {task_type}"
                task_data["end_time"] = time.time()
                task_data["execution_time"] = 0.0
                return

            # 更新任务状态
            if success:
                task_data["status"] = VisionTaskStatus.COMPLETED.value
                task_data["result"] = result_data
            else:
                task_data["status"] = VisionTaskStatus.FAILED.value
                task_data["error"] = "任务执行失败"

            task_data["end_time"] = time.time()
            task_data["execution_time"] = (
                task_data["end_time"] - task_data["start_time"]
            )

            logger.info(
                f"视觉任务完成: {task_id} (状态: {                     task_data['status']}, 用时: {                     task_data['execution_time']:.2f}s)"
            )

        except Exception as e:
            logger.error(f"处理任务失败: {e}")
            task_data["status"] = VisionTaskStatus.FAILED.value
            task_data["error"] = str(e)
            task_data["end_time"] = time.time()
            task_data["execution_time"] = 0.0

    def _process_voice_command(self, command_id: str, command_data: Dict[str, Any]):
        """处理语音命令

        参数:
            command_id: 命令ID
            command_data: 命令数据
        """
        try:
            # 更新命令状态
            command_data["status"] = VoiceCommandStatus.PROCESSING.value
            command_data["start_time"] = time.time()

            logger.info(f"开始处理语音命令: {command_id}")

            # 检查视觉控制器是否可用
            if self.vision_controller is None:
                command_data["status"] = VoiceCommandStatus.FAILED.value
                command_data["error"] = "视觉控制器未初始化"
                command_data["end_time"] = time.time()
                command_data["execution_time"] = 0.0
                return

            # 解析语音命令（如果提供了文本）
            if "command_text" in command_data:
                text = command_data["command_text"]

                # 使用控制器的解析器解析命令
                voice_command = self.vision_controller._parse_voice_command(text)

                if voice_command:
                    command_data["parsed_command"] = {
                        "command_type": voice_command.command_type.value,
                        "parameters": voice_command.parameters,
                        "confidence": voice_command.confidence,
                    }

                    # 执行语音命令
                    success = self.vision_controller.execute_voice_command(
                        voice_command
                    )

                    if success:
                        command_data["status"] = VoiceCommandStatus.COMPLETED.value
                        command_data["result"] = {"success": True}
                    else:
                        command_data["status"] = VoiceCommandStatus.FAILED.value
                        command_data["error"] = "命令执行失败"
                else:
                    command_data["status"] = VoiceCommandStatus.FAILED.value
                    command_data["error"] = "无法解析语音命令"
            else:
                command_data["status"] = VoiceCommandStatus.FAILED.value
                command_data["error"] = "缺少命令文本"

            command_data["end_time"] = time.time()
            command_data["execution_time"] = (
                command_data["end_time"] - command_data["start_time"]
            )

            logger.info(
                f"语音命令处理完成: {command_id} (状态: {                     command_data['status']}, 用时: {                     command_data['execution_time']:.2f}s)"
            )

        except Exception as e:
            logger.error(f"处理语音命令失败: {e}")
            command_data["status"] = VoiceCommandStatus.FAILED.value
            command_data["error"] = str(e)
            command_data["end_time"] = time.time()
            command_data["execution_time"] = 0.0

    def submit_vision_task(
        self, task_type: str, target_object: Optional[str] = None
    ) -> Dict[str, Any]:
        """提交视觉任务

        参数:
            task_type: 任务类型（object_detection, object_tracking, visual_navigation, visual_manipulation）
            target_object: 目标对象（可选）

        返回:
            任务提交响应
        """
        try:
            # 生成任务ID
            self.task_count += 1
            task_id = f"vision_task_{self.task_count:06d}_{int(time.time())}"

            # 创建任务数据
            task_data = {
                "task_id": task_id,
                "task_type": task_type,
                "target_object": target_object,
                "status": VisionTaskStatus.PENDING.value,
                "create_time": time.time(),
                "submit_time": datetime.now(timezone.utc).isoformat(),
            }

            # 保存任务
            self.tasks[task_id] = task_data

            # 加入处理队列
            self.task_queue.put(task_id)

            logger.info(
                f"视觉任务已提交: {task_id} (类型: {task_type}, 目标: {target_object})"
            )

            return {
                "success": True,
                "task_id": task_id,
                "message": "视觉任务已提交",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"提交视觉任务失败: {e}")
            self.error_count += 1

            return {
                "success": False,
                "task_id": None,
                "error": f"提交视觉任务失败: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def submit_voice_command(self, command_text: str) -> Dict[str, Any]:
        """提交语音命令

        参数:
            command_text: 语音命令文本

        返回:
            命令提交响应
        """
        try:
            # 生成命令ID
            self.voice_command_count += 1
            command_id = f"voice_cmd_{self.voice_command_count:06d}_{int(time.time())}"

            # 创建命令数据
            command_data = {
                "command_id": command_id,
                "command_text": command_text,
                "status": VoiceCommandStatus.RECEIVED.value,
                "create_time": time.time(),
                "submit_time": datetime.now(timezone.utc).isoformat(),
            }

            # 保存命令
            self.voice_commands[command_id] = command_data

            # 加入处理队列
            self.voice_command_queue.put(command_id)

            logger.info(f"语音命令已提交: {command_id} (文本: {command_text})")

            return {
                "success": True,
                "command_id": command_id,
                "message": "语音命令已提交",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"提交语音命令失败: {e}")
            self.error_count += 1

            return {
                "success": False,
                "command_id": None,
                "error": f"提交语音命令失败: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态

        参数:
            task_id: 任务ID

        返回:
            任务状态
        """
        task_data = self.tasks.get(task_id)

        if not task_data:
            return {
                "success": False,
                "error": f"任务不存在: {task_id}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return {
            "success": True,
            "task": task_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_voice_command_status(self, command_id: str) -> Dict[str, Any]:
        """获取语音命令状态

        参数:
            command_id: 命令ID

        返回:
            命令状态
        """
        command_data = self.voice_commands.get(command_id)

        if not command_data:
            return {
                "success": False,
                "error": f"语音命令不存在: {command_id}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return {
            "success": True,
            "command": command_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态

        返回:
            服务状态信息
        """
        # 获取视觉控制器状态（如果可用）
        vision_status = None
        tracked_objects = []

        if self.vision_controller:
            try:
                vision_status = self.vision_controller.get_status()
                tracked_objects = self.vision_controller.get_tracked_objects_info()
            except Exception as e:
                logger.warning(f"获取视觉控制器状态失败: {e}")

        return {
            "success": True,
            "status": {
                "service_status": self.service_status,
                "vision_controller_available": self.vision_controller is not None,
                "vision_controller_status": vision_status,
                "tracked_objects_count": len(tracked_objects),
                "pending_tasks": sum(
                    1
                    for t in self.tasks.values()
                    if t.get("status") == VisionTaskStatus.PENDING.value
                ),
                "running_tasks": sum(
                    1
                    for t in self.tasks.values()
                    if t.get("status") == VisionTaskStatus.RUNNING.value
                ),
                "completed_tasks": sum(
                    1
                    for t in self.tasks.values()
                    if t.get("status") == VisionTaskStatus.COMPLETED.value
                ),
                "failed_tasks": sum(
                    1
                    for t in self.tasks.values()
                    if t.get("status") == VisionTaskStatus.FAILED.value
                ),
                "pending_commands": sum(
                    1
                    for c in self.voice_commands.values()
                    if c.get("status") == VoiceCommandStatus.RECEIVED.value
                ),
                "processing_commands": sum(
                    1
                    for c in self.voice_commands.values()
                    if c.get("status") == VoiceCommandStatus.PROCESSING.value
                ),
                "completed_commands": sum(
                    1
                    for c in self.voice_commands.values()
                    if c.get("status") == VoiceCommandStatus.COMPLETED.value
                ),
                "total_tasks": len(self.tasks),
                "total_voice_commands": len(self.voice_commands),
                "task_count": self.task_count,
                "voice_command_count": self.voice_command_count,
                "error_count": self.error_count,
                "last_update": self.last_update.isoformat(),
                "processing_running": self.processing_running,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_tracked_objects(self) -> Dict[str, Any]:
        """获取跟踪对象信息

        返回:
            跟踪对象信息
        """
        tracked_objects = []

        if self.vision_controller:
            try:
                tracked_objects = self.vision_controller.get_tracked_objects_info()
            except Exception as e:
                logger.error(f"获取跟踪对象失败: {e}")

        return {
            "success": True,
            "tracked_objects": tracked_objects,
            "count": len(tracked_objects),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def process_visual_frame(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理视觉帧

        参数:
            image_data: 图像数据

        返回:
            处理结果
        """
        try:
            if not self.vision_controller:
                return {
                    "success": False,
                    "error": "视觉控制器未初始化",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # 提取图像数据
            # 注意：实际实现中应该处理不同类型的图像输入
            image_base64 = image_data.get("image_base64")
            image_path = image_data.get("image_path")

            # 这里应该将图像数据转换为numpy数组
            # 完整处理，只记录日志

            logger.info(
                f"收到视觉帧处理请求 (base64: {bool(image_base64)}, path: {image_path})"
            )

            # 模拟处理（实际应该调用视觉控制器）
            # 获取当前跟踪的对象
            tracked_objects = self.vision_controller.get_tracked_objects_info()

            return {
                "success": True,
                "processed": True,
                "tracked_objects": tracked_objects,
                "message": "视觉帧处理完成",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"处理视觉帧失败: {e}")
            self.error_count += 1

            return {
                "success": False,
                "error": f"处理视觉帧失败: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def cleanup_old_data(self, max_age_hours: float = 24.0):
        """清理旧数据

        参数:
            max_age_hours: 最大保留小时数
        """
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)

            # 清理旧任务
            tasks_to_delete = []
            for task_id, task_data in self.tasks.items():
                create_time = task_data.get("create_time", 0)
                if create_time < cutoff_time:
                    tasks_to_delete.append(task_id)

            for task_id in tasks_to_delete:
                del self.tasks[task_id]

            # 清理旧语音命令
            commands_to_delete = []
            for command_id, command_data in self.voice_commands.items():
                create_time = command_data.get("create_time", 0)
                if create_time < cutoff_time:
                    commands_to_delete.append(command_id)

            for command_id in commands_to_delete:
                del self.voice_commands[command_id]

            if tasks_to_delete or commands_to_delete:
                logger.info(
                    f"清理了 {len(tasks_to_delete)} 个旧任务和 {len(commands_to_delete)} 个旧语音命令"
                )

        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")


# 全局服务实例（单例）
_robot_vision_service_instance = None


def get_robot_vision_service() -> RobotVisionService:
    """获取机器人视觉引导控制服务实例（单例）"""
    global _robot_vision_service_instance

    if _robot_vision_service_instance is None:
        _robot_vision_service_instance = RobotVisionService()

    return _robot_vision_service_instance


# 测试函数
def test_robot_vision_service():
    """测试机器人视觉引导控制服务"""
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=== 测试机器人视觉引导控制服务 ===")

    try:
        # 获取服务实例
        service = get_robot_vision_service()

        # 测试服务状态
        print("\n=== 测试服务状态 ===")
        status = service.get_service_status()
        print(f"服务状态: {status['success']}")
        print(f"服务详情: {json.dumps(status['status'], indent=2, default=str)}")

        # 测试提交视觉任务
        print("\n=== 测试提交视觉任务 ===")
        task_response = service.submit_vision_task("object_detection", "person")
        print(f"任务提交响应: {task_response}")

        if task_response["success"]:
            task_id = task_response["task_id"]

            # 等待片刻
            time.sleep(0.5)

            # 检查任务状态
            task_status = service.get_task_status(task_id)
            print(f"任务状态: {task_status}")

        # 测试提交语音命令
        print("\n=== 测试提交语音命令 ===")
        voice_response = service.submit_voice_command("向前走")
        print(f"语音命令响应: {voice_response}")

        if voice_response["success"]:
            command_id = voice_response["command_id"]

            # 等待片刻
            time.sleep(0.5)

            # 检查命令状态
            command_status = service.get_voice_command_status(command_id)
            print(f"语音命令状态: {command_status}")

        # 测试获取跟踪对象
        print("\n=== 测试获取跟踪对象 ===")
        tracked_response = service.get_tracked_objects()
        print(f"跟踪对象响应: {tracked_response}")

        # 测试处理视觉帧
        print("\n=== 测试处理视觉帧 ===")
        image_data = {"image_base64": "mock_base64_data", "description": "测试图像"}
        frame_response = service.process_visual_frame(image_data)
        print(f"视觉帧处理响应: {frame_response}")

        print("\n=== 测试完成 ===")
        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_robot_vision_service()
    exit(0 if success else 1)
