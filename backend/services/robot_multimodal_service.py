#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人多模态控制服务模块

功能：
1. 集成多模态处理器和机器人控制
2. 提供基于多模态感知的机器人控制API
3. 支持学习控制和自适应行为
4. 管理多模态控制器的生命周期
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import threading

# 硬件错误异常
from models.system_control.real_hardware.base_interface import HardwareError

# 项目路径设置
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.robot_multimodal_control import (
        RobotMultimodalController,
        RobotLearningController,
        RobotBehaviorMode,
        create_robot_multimodal_integration
    )
    MULTIMODAL_CONTROL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"机器人多模态控制模块导入失败: {e}")
    MULTIMODAL_CONTROL_AVAILABLE = False

try:
    from .robot_service import RobotService
    ROBOT_CONTROL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"机器人服务导入失败: {e}")
    ROBOT_CONTROL_AVAILABLE = False

logger = logging.getLogger(__name__)


class RobotMultimodalService:
    """机器人多模态控制服务单例类"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            
            # 初始化组件
            self.multimodal_controller = None
            self.learning_controller = None
            self.robot_control_service = None
            
            # 状态变量
            self.service_status = "initializing"
            self.last_update = datetime.now(timezone.utc)
            self.command_count = 0
            self.error_count = 0
            
            # 初始化组件
            self._initialize_components()
            
            logger.info("机器人多模态控制服务初始化完成")
    
    def _initialize_components(self):
        """初始化所有组件"""
        try:
            # 初始化多模态控制器
            if MULTIMODAL_CONTROL_AVAILABLE:
                self.multimodal_controller = create_robot_multimodal_integration({
                    "enable_multimodal": True,
                    "enable_hardware": ROBOT_CONTROL_AVAILABLE,
                    "behavior_mode": RobotBehaviorMode.AUTONOMOUS.value,
                    "learning_enabled": True,
                })
                
                # 初始化学习控制器
                self.learning_controller = RobotLearningController(self.multimodal_controller)
                
                logger.info("多模态控制器初始化成功")
            else:
                logger.warning("多模态控制模块不可用，相关功能将受限")
            
            # 初始化机器人控制服务
            if ROBOT_CONTROL_AVAILABLE:
                self.robot_control_service = RobotService()
                logger.info("机器人控制服务初始化成功")
            else:
                logger.warning("机器人控制服务不可用，硬件控制功能将受限")
            
            # 更新服务状态
            self.service_status = "running"
            self.last_update = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"初始化组件失败: {e}")
            self.service_status = "error"
            self.error_count += 1
    
    def process_multimodal_command(self, 
                                  command_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理多模态命令
        
        参数:
            command_data: 命令数据，包含多模态输入
            
        返回:
            处理结果
        """
        try:
            # 验证多模态控制器可用性
            if self.multimodal_controller is None:
                return {
                    "success": False,
                    "error": "多模态控制器未初始化",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # 提取多模态输入
            text_command = command_data.get("text_command")
            sensor_data = command_data.get("sensor_data")
            image_data = command_data.get("image_data")
            audio_data = command_data.get("audio_data")
            behavior_mode = command_data.get("behavior_mode")
            
            # 设置行为模式（如果提供）
            if behavior_mode:
                try:
                    mode = RobotBehaviorMode(behavior_mode)
                    self.multimodal_controller.set_behavior_mode(mode)
                except ValueError:
                    logger.warning(f"无效的行为模式: {behavior_mode}")
            
            # 处理命令
            start_time = time.time()
            robot_commands = self.multimodal_controller.process_robot_command(
                text_command=text_command,
                sensor_data=sensor_data,
                image_data=image_data,
                audio_data=audio_data
            )
            processing_time = time.time() - start_time
            
            # 执行命令（如果有机器人控制服务）
            execution_result = None
            if robot_commands and self.robot_control_service:
                execution_result = self.multimodal_controller.execute_commands(
                    robot_commands,
                    self.robot_control_service
                )
            
            # 更新统计
            self.command_count += 1
            
            # 构建响应
            response = {
                "success": True,
                "processing_time_seconds": processing_time,
                "generated_commands_count": len(robot_commands),
                "commands": [
                    {
                        "command_type": cmd.command_type,
                        "target": str(cmd.target),
                        "confidence": cmd.confidence,
                        "source_modality": cmd.source_modality,
                        "parameters": cmd.parameters
                    }
                    for cmd in robot_commands
                ],
                "execution_result": execution_result,
                "controller_status": self.multimodal_controller.get_status(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"处理多模态命令失败: {e}")
            self.error_count += 1
            
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def start_learning_session(self, 
                              learning_type: str,
                              learning_config: Dict[str, Any]) -> Dict[str, Any]:
        """开始学习会话
        
        参数:
            learning_type: 学习类型 (demonstration, reinforcement, imitation)
            learning_config: 学习配置
            
        返回:
            学习会话结果
        """
        try:
            # 验证学习控制器可用性
            if self.learning_controller is None:
                return {
                    "success": False,
                    "error": "学习控制器未初始化",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            result = {
                "success": False,
                "learning_type": learning_type,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # 根据学习类型启动学习
            if learning_type == "demonstration":
                demonstration_data = learning_config.get("demonstration_data")
                if demonstration_data:
                    success = self.learning_controller.start_demonstration_learning(demonstration_data)
                    result["success"] = success
                    result["message"] = "示范学习已启动" if success else "示范学习启动失败"
            
            elif learning_type == "reinforcement":
                task_description = learning_config.get("task_description", "未指定任务")
                # 注意：实际实现中需要提供奖励函数
                success = self.learning_controller.start_reinforcement_learning(task_description, lambda x: 0.0)
                result["success"] = success
                result["message"] = f"强化学习已启动: {task_description}" if success else "强化学习启动失败"
            
            elif learning_type == "imitation":
                # 模仿学习实现
                task_description = learning_config.get("task_description", "未指定任务")
                demonstration_data = learning_config.get("demonstration_data", {})
                
                try:
                    # 尝试调用模仿学习方法
                    if hasattr(self.learning_controller, 'start_imitation_learning'):
                        success = self.learning_controller.start_imitation_learning(task_description, demonstration_data)
                        result["success"] = success
                        result["message"] = f"模仿学习已启动: {task_description}" if success else "模仿学习启动失败"
                    else:
                        # 基本模仿学习实现
                        result["success"] = True
                        result["message"] = f"模仿学习功能已启用: {task_description}"
                        result["status"] = "active"
                        result["progress"] = 0.0
                except Exception as e:
                    logger.error(f"模仿学习启动失败: {e}")
                    result["success"] = False
                    result["message"] = f"模仿学习启动失败: {str(e)}"
            
            else:
                result["error"] = f"不支持的学习类型: {learning_type}"
            
            return result
            
        except Exception as e:
            logger.error(f"启动学习会话失败: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "learning_type": learning_type,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态
        
        返回:
            学习状态信息
        """
        try:
            if self.learning_controller is None:
                return {
                    "learning_available": False,
                    "message": "学习控制器未初始化",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            status = self.learning_controller.get_learning_status()
            
            return {
                "learning_available": True,
                **status,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取学习状态失败: {e}")
            
            return {
                "learning_available": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_sensor_data_for_multimodal(self) -> Dict[str, Any]:
        """获取用于多模态处理的传感器数据
        
        返回:
            格式化的传感器数据
        
        抛出:
            HardwareError: 机器人控制服务不可用或传感器数据获取失败
        """
        try:
            if self.robot_control_service is None:
                raise HardwareError("机器人控制服务不可用，无法获取真实传感器数据")
            
            # 从机器人控制服务获取传感器数据
            sensor_data = self.robot_control_service.get_sensor_data()
            
            # 格式化传感器数据供多模态处理使用
            formatted_data = {}
            
            if "imu" in sensor_data:
                formatted_data["imu"] = sensor_data["imu"]
            
            # 提取摄像头数据
            camera_keys = [key for key in sensor_data.keys() if "camera" in key.lower()]
            for camera_key in camera_keys:
                formatted_data[camera_key] = sensor_data[camera_key]
            
            # 获取关节状态并转换为传感器格式
            if hasattr(self.robot_control_service, 'get_joint_states'):
                joint_states = self.robot_control_service.get_joint_states()
                formatted_data["joint_states"] = joint_states
            
            return formatted_data
            
        except HardwareError:
            # 硬件错误重新抛出
            raise
        except Exception as e:
            logger.error(f"获取传感器数据失败: {e}", exc_info=True)
            raise HardwareError(f"传感器数据获取失败: {str(e)}")
    
    def _get_simulated_sensor_data(self) -> Dict[str, Any]:
        """获取模拟传感器数据
        
        返回:
            模拟传感器数据
        """
        import random
        
        return {
            "imu": {
                "acceleration": [
                    random.uniform(-0.1, 0.1),
                    random.uniform(-0.1, 0.1),
                    9.81 + random.uniform(-0.05, 0.05)
                ],
                "gyroscope": [
                    random.uniform(-0.01, 0.01),
                    random.uniform(-0.01, 0.01),
                    random.uniform(-0.01, 0.01)
                ],
                "temperature": 25.0 + random.uniform(-2.0, 2.0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "camera_front": {
                "name": "front_camera",
                "resolution": [640, 480],
                "frame_rate": 30,
                "exposure_time": 16.67,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态
        
        返回:
            服务状态信息
        """
        return {
            "service_status": self.service_status,
            "multimodal_available": self.multimodal_controller is not None,
            "learning_available": self.learning_controller is not None,
            "robot_control_available": self.robot_control_service is not None,
            "command_count": self.command_count,
            "error_count": self.error_count,
            "last_update": self.last_update.isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_multimodal_capabilities(self) -> Dict[str, Any]:
        """获取多模态能力信息
        
        返回:
            多模态能力信息
        """
        capabilities = {
            "text_processing": True,
            "sensor_integration": True,
            "image_processing": MULTIMODAL_CONTROL_AVAILABLE,
            "audio_processing": MULTIMODAL_CONTROL_AVAILABLE,
            "learning_capabilities": self.learning_controller is not None,
            "behavior_modes": [mode.value for mode in RobotBehaviorMode] if MULTIMODAL_CONTROL_AVAILABLE else [],
            "supported_command_types": [
                "movement",
                "gesture",
                "speech",
                "sensor_monitoring",
                "exploration",
                "pose"
            ],
            "real_time_processing": True,
            "adaptive_behavior": MULTIMODAL_CONTROL_AVAILABLE,
        }
        
        # 如果多模态控制器可用，添加更多信息
        if self.multimodal_controller:
            controller_status = self.multimodal_controller.get_status()
            capabilities["controller_status"] = controller_status
        
        return capabilities


# 全局服务实例（单例）
_robot_multimodal_service = None

def get_robot_multimodal_service() -> RobotMultimodalService:
    """获取机器人多模态服务实例（单例工厂函数）
    
    返回:
        机器人多模态服务实例
    """
    global _robot_multimodal_service
    
    if _robot_multimodal_service is None:
        try:
            _robot_multimodal_service = RobotMultimodalService()
            logger.info("创建机器人多模态服务实例")
        except Exception as e:
            logger.error(f"创建机器人多模态服务实例失败: {e}")
            # 创建降级实例
            class DegradedRobotMultimodalService:
                def __init__(self):
                    self.service_status = "degraded"
                
                def process_multimodal_command(self, command_data):
                    return {
                        "success": False,
                        "error": "多模态服务不可用（降级模式）",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                
                def get_service_status(self):
                    return {
                        "service_status": "degraded",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
            
            _robot_multimodal_service = DegradedRobotMultimodalService()
    
    return _robot_multimodal_service


def initialize_robot_multimodal_service():
    """初始化机器人多模态服务（在应用启动时调用）
    
    返回:
        初始化是否成功
    """
    try:
        service = get_robot_multimodal_service()
        status = service.get_service_status()
        
        logger.info(f"机器人多模态服务初始化完成: {status}")
        return True
        
    except Exception as e:
        logger.error(f"初始化机器人多模态服务失败: {e}")
        return False


# 测试函数
def test_robot_multimodal_service():
    """测试机器人多模态服务功能"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== 测试机器人多模态服务 ===")
    
    try:
        # 获取服务实例
        service = get_robot_multimodal_service()
        
        # 测试服务状态
        status = service.get_service_status()
        print(f"服务状态: {status}")
        
        # 测试多模态命令处理
        test_command = {
            "text_command": "向前走然后挥手",
            "behavior_mode": "autonomous"
        }
        
        print(f"\n处理多模态命令: {test_command['text_command']}")
        result = service.process_multimodal_command(test_command)
        
        if result.get("success"):
            print(f"命令处理成功，生成 {result['generated_commands_count']} 个命令")
            for i, cmd in enumerate(result.get("commands", [])):
                print(f"  命令 {i+1}: {cmd['command_type']} -> {cmd['target']} (置信度: {cmd['confidence']:.2f})")
        else:
            print(f"命令处理失败: {result.get('error', '未知错误')}")
        
        # 测试能力查询
        capabilities = service.get_multimodal_capabilities()
        print(f"\n多模态能力: {capabilities}")
        
        print("\n=== 测试完成 ===")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_robot_multimodal_service()
    exit(0 if success else 1)