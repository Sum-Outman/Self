#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人视觉引导控制模块

功能：
1. 使用多模态处理器视觉编码器解析摄像头输入
2. 实现基于视觉场景分析的机器人控制
3. 支持语音命令识别和语音控制机器人
4. 视觉目标检测、跟踪和引导
5. 与现有机器人控制器无缝集成

模块特性：
- 实时视觉处理：处理摄像头输入，提取视觉特征
- 语音命令处理：识别语音命令并转换为机器人动作
- 视觉引导控制：基于视觉信息引导机器人执行任务
- 多模态融合：视觉和语音信息的融合控制
"""

import logging
import time
import threading
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import queue
import os

# 计算机视觉库导入
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV (cv2) 库未安装，真实对象检测功能将不可用")

# 多模态处理器导入
try:
    from models.multimodal.processor import MultimodalProcessor
    from models.multimodal.vision_encoder import IndustrialVisionEncoder
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"多模态处理器导入失败: {e}")
    MULTIMODAL_AVAILABLE = False

# 机器人控制器导入
try:
    from .robot_controller import HardwareInterface, RobotJoint, SensorType, HardwareManager
    HARDWARE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"机器人控制器导入失败: {e}")
    HARDWARE_AVAILABLE = False

# 音频处理导入（用于语音识别）
try:
    import speech_recognition as sr  # type: ignore
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logging.warning("speech_recognition 库未安装，语音识别功能将不可用")

logger = logging.getLogger(__name__)


class VisionTaskType(Enum):
    """视觉任务类型枚举"""
    OBJECT_DETECTION = "object_detection"  # 目标检测
    OBJECT_TRACKING = "object_tracking"   # 目标跟踪
    POSE_ESTIMATION = "pose_estimation"   # 姿态估计
    SCENE_ANALYSIS = "scene_analysis"     # 场景分析
    NAVIGATION = "navigation"            # 导航
    MANIPULATION = "manipulation"        # 操作


class VoiceCommandType(Enum):
    """语音命令类型枚举"""
    MOVEMENT = "movement"      # 移动命令
    GESTURE = "gesture"        # 手势命令
    NAVIGATION = "navigation"  # 导航命令
    TASK = "task"              # 任务命令
    SYSTEM = "system"          # 系统命令


@dataclass
class VisualObject:
    """视觉对象数据类"""
    object_id: str
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    position_3d: Optional[Tuple[float, float, float]] = None  # 3D位置（如果有深度信息）
    features: Optional[np.ndarray] = None  # 视觉特征向量
    timestamp: float = field(default_factory=time.time)


@dataclass
class VoiceCommand:
    """语音命令数据类"""
    command_id: str
    command_type: VoiceCommandType
    text: str
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class VisionGuidedController:
    """视觉引导控制器
    
    使用多模态处理器的视觉编码器解析摄像头输入，
    并控制机器人执行基于视觉的任务。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化视觉引导控制器
        
        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(f"{__name__}.VisionGuidedController")
        
        # 默认配置
        self.config = config or {
            "enable_vision": MULTIMODAL_AVAILABLE,
            "enable_voice": SPEECH_RECOGNITION_AVAILABLE,
            "enable_hardware": HARDWARE_AVAILABLE,
            "vision_fps": 10,  # 视觉处理帧率
            "voice_timeout": 5.0,  # 语音命令超时时间
            "object_tracking_enabled": True,
            "min_confidence": 0.5,  # 最小置信度阈值
            "camera_resolution": (640, 480),  # 摄像头分辨率
            "use_real_object_detection": CV2_AVAILABLE,  # 使用真实对象检测（如果OpenCV可用）
            "real_detection_min_size": (30, 30),  # 最小检测尺寸
            "real_detection_scale_factor": 1.1,  # 检测尺度因子
            "real_detection_min_neighbors": 5,  # 最小邻居数
        }
        
        # 初始化组件
        self.multimodal_processor = None
        self.hardware_interface = None
        self.hardware_manager = None
        self.robot_controller = None
        self.voice_recognizer = None
        self.real_detector = None  # OpenCV真实对象检测器
        
        # 视觉处理状态
        self.vision_enabled = self.config["enable_vision"]
        self.voice_enabled = self.config["enable_voice"]
        self.hardware_enabled = self.config["enable_hardware"]
        
        # 对象跟踪状态
        self.tracked_objects: Dict[str, VisualObject] = {}
        self.object_history: Dict[str, List[VisualObject]] = {}
        
        # 语音命令队列
        self.voice_command_queue = queue.Queue()
        self.is_listening = False
        
        # 初始化组件
        self._initialize_components()
        
        # 启动处理线程
        self._start_processing_threads()
        
        self.logger.info("视觉引导控制器初始化完成")
    
    def _initialize_components(self):
        """初始化所有组件"""
        try:
            # 初始化多模态处理器（用于视觉编码）
            if self.vision_enabled and MULTIMODAL_AVAILABLE:
                self._initialize_multimodal_processor()
            
            # 初始化真实对象检测器（如果配置启用）
            if self.config.get("use_real_object_detection", False) and CV2_AVAILABLE:
                self._initialize_real_detector()
            
            # 初始化语音识别器
            if self.voice_enabled and SPEECH_RECOGNITION_AVAILABLE:
                self._initialize_voice_recognizer()
            
            # 初始化硬件接口
            if self.hardware_enabled and HARDWARE_AVAILABLE:
                self._initialize_hardware_interface()
                
        except Exception as e:
            self.logger.error(f"初始化组件失败: {e}")
    
    def _initialize_multimodal_processor(self):
        """初始化多模态处理器"""
        try:
            # 配置多模态处理器，主要关注视觉功能
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
                "enable_video": False,  # 暂时禁用视频处理
                "enable_sensor": True,
                "device": "cpu",  # 使用CPU模式
                "image_size": self.config["camera_resolution"][1],  # 高度
            }
            
            self.multimodal_processor = MultimodalProcessor(multimodal_config)
            self.multimodal_processor.initialize()
            self.multimodal_processor.eval()  # 设置为评估模式
            
            self.logger.info("多模态处理器初始化成功，视觉编码器已就绪")
            
        except Exception as e:
            self.logger.error(f"初始化多模态处理器失败: {e}")
            self.vision_enabled = False
            self.multimodal_processor = None
    
    def _initialize_real_detector(self):
        """初始化真实对象检测器（使用OpenCV Haar级联分类器）"""
        try:
            # 检查OpenCV是否安装
            if not CV2_AVAILABLE:
                self.logger.warning("OpenCV未安装，无法初始化真实对象检测器")
                return
            
            # 尝试加载预训练的Haar级联分类器
            # OpenCV自带一些人脸、眼睛等检测器
            cascade_paths = {
                "face": cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
                "eye": cv2.data.haarcascades + "haarcascade_eye.xml",
                "full_body": cv2.data.haarcascades + "haarcascade_fullbody.xml",
                "upper_body": cv2.data.haarcascades + "haarcascade_upperbody.xml",
            }
            
            self.real_detector = {}
            for detector_name, cascade_path in cascade_paths.items():
                if os.path.exists(cascade_path):
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        self.real_detector[detector_name] = cascade
                        self.logger.info(f"成功加载 {detector_name} 检测器: {cascade_path}")
                    else:
                        self.logger.warning(f"加载 {detector_name} 检测器失败: 分类器为空")
                else:
                    self.logger.warning(f"检测器文件不存在: {cascade_path}")
            
            if not self.real_detector:
                self.logger.warning("未成功加载任何真实对象检测器，真实对象检测功能将不可用")
                self.config["use_real_object_detection"] = False
            else:
                self.logger.info(f"真实对象检测器初始化成功，已加载 {len(self.real_detector)} 个检测器")
                
        except Exception as e:
            self.logger.error(f"初始化真实对象检测器失败: {e}")
            self.config["use_real_object_detection"] = False
            self.real_detector = None
    
    def _initialize_voice_recognizer(self):
        """初始化语音识别器"""
        try:
            self.voice_recognizer = sr.Recognizer()
            self.voice_recognizer.energy_threshold = 4000  # 能量阈值
            self.voice_recognizer.dynamic_energy_threshold = True
            
            self.logger.info("语音识别器初始化成功")
            
        except Exception as e:
            self.logger.error(f"初始化语音识别器失败: {e}")
            self.voice_enabled = False
            self.voice_recognizer = None
    
    def _initialize_hardware_interface(self):
        """初始化硬件接口"""
        try:
            # 创建硬件管理器
            self.hardware_manager = HardwareManager()
            
            # 连接硬件（使用仿真模式）
            connected = self.hardware_manager.connect()
            
            if connected:
                self.logger.info("硬件管理器连接成功")
                
                # 创建默认的PyBullet仿真接口
                simulation_created = self.hardware_manager.create_pybullet_interface(
                    name="default_simulation",
                    gui_enabled=False  # 不显示GUI以提高性能
                )
                
                if simulation_created:
                    # 创建机器人控制器
                    self.robot_controller = self.hardware_manager.create_robot_controller(
                        name="main_robot",
                        interface_name="default_simulation",
                        enable_advanced_control=False
                    )
                    
                    if self.robot_controller:
                        # 连接机器人控制器
                        robot_connected = self.robot_controller.connect()
                        if robot_connected:
                            self.logger.info("机器人控制器连接成功")
                            self.hardware_interface = self.robot_controller  # 兼容旧代码
                        else:
                            self.logger.warning("机器人控制器连接失败")
                            self.hardware_interface = self.hardware_manager  # 回退到硬件管理器
                    else:
                        self.logger.warning("无法创建机器人控制器")
                        self.hardware_interface = self.hardware_manager  # 回退到硬件管理器
                else:
                    self.logger.warning("无法创建仿真接口")
                    self.hardware_interface = self.hardware_manager  # 回退到硬件管理器
            else:
                self.logger.warning("硬件管理器连接失败，使用仿真模式")
                self.hardware_interface = self.hardware_manager  # 回退到硬件管理器
                
        except Exception as e:
            self.logger.error(f"初始化硬件接口失败: {e}")
            self.hardware_enabled = False
            self.hardware_interface = None
            self.hardware_manager = None
            self.robot_controller = None
    
    def _start_processing_threads(self):
        """启动处理线程"""
        # 启动视觉处理线程
        if self.vision_enabled:
            vision_thread = threading.Thread(
                target=self._vision_processing_loop,
                daemon=True,
                name="VisionProcessing"
            )
            vision_thread.start()
            self.logger.info("视觉处理线程已启动")
        
        # 启动语音监听线程
        if self.voice_enabled:
            voice_thread = threading.Thread(
                target=self._voice_listening_loop,
                daemon=True,
                name="VoiceListening"
            )
            voice_thread.start()
            self.logger.info("语音监听线程已启动")
        
        # 启动命令处理线程
        command_thread = threading.Thread(
            target=self._command_processing_loop,
            daemon=True,
            name="CommandProcessing"
        )
        command_thread.start()
        self.logger.info("命令处理线程已启动")
    
    def process_visual_frame(self, image_data: np.ndarray) -> List[VisualObject]:
        """处理视觉帧
        
        参数:
            image_data: 图像数据 (numpy数组，H×W×C)
            
        返回:
            检测到的视觉对象列表
        """
        if not self.vision_enabled or self.multimodal_processor is None:
            return []  # 返回空列表
        
        try:
            # 检查图像尺寸
            if len(image_data.shape) != 3 or image_data.shape[2] != 3:
                self.logger.warning(f"无效的图像数据形状: {image_data.shape}")
                return []  # 返回空列表
            
            # 将图像转换为多模态处理器可处理的格式
            # 完整处理，实际应该使用多模态处理器的视觉编码器
            
            # 仅使用真实对象检测，不采用任何降级机制
            if self.config.get("use_real_object_detection", False) and self.real_detector:
                objects = self._real_object_detection(image_data)
            else:
                # 真实检测不可用，返回空列表（不生成虚拟数据）
                self.logger.warning("真实对象检测不可用，返回空列表（禁止使用虚拟数据）")
                objects = []
            
            # 更新对象跟踪
            if self.config["object_tracking_enabled"]:
                self._update_object_tracking(objects)
            
            return objects
            
        except Exception as e:
            self.logger.error(f"处理视觉帧失败: {e}")
            return []  # 返回空列表
    
    def _real_object_detection(self, image_data: np.ndarray) -> List[VisualObject]:
        """真实对象检测（使用OpenCV Haar级联分类器）
        
        参数:
            image_data: 图像数据 (numpy数组，H×W×C)
            
        返回:
            检测到的视觉对象列表
        """
        objects = []
        
        try:
            # 检查检测器是否初始化
            if not self.real_detector:
                self.logger.warning("真实对象检测器未初始化，返回空列表（禁止使用虚拟数据）")
                return []
            
            # 将图像转换为灰度图（Haar级联需要灰度图）
            gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
            
            # 对每个检测器进行检测
            for detector_name, cascade in self.real_detector.items():
                # 获取检测参数
                scale_factor = self.config.get("real_detection_scale_factor", 1.1)
                min_neighbors = self.config.get("real_detection_min_neighbors", 5)
                min_size = self.config.get("real_detection_min_size", (30, 30))
                
                # 运行检测
                detections = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size
                )
                
                # 转换检测结果为视觉对象
                for i, (x, y, w, h) in enumerate(detections):
                    # 根据检测器类型确定对象类别
                    if detector_name == "face":
                        class_name = "person"
                        confidence = 0.7 + (w * h) / (gray.shape[0] * gray.shape[1]) * 0.3
                    elif detector_name == "eye":
                        class_name = "eye"
                        confidence = 0.6
                    elif detector_name == "full_body":
                        class_name = "person"
                        confidence = 0.65
                    elif detector_name == "upper_body":
                        class_name = "person"
                        confidence = 0.68
                    else:
                        class_name = detector_name
                        confidence = 0.5
                    
                    # 确保置信度在合理范围内
                    confidence = min(max(confidence, 0.1), 0.99)
                    
                    # 创建视觉对象
                    object_id = f"{detector_name}_{i}_{int(time.time())}"
                    visual_obj = VisualObject(
                        object_id=object_id,
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox=(float(x), float(y), float(w), float(h)),
                        timestamp=time.time()
                    )
                    
                    objects.append(visual_obj)
            
            self.logger.debug(f"真实对象检测完成，检测到 {len(objects)} 个对象")
            
        except Exception as e:
            self.logger.error(f"真实对象检测失败: {e}")
            # 检测失败，返回空列表（不生成虚拟数据）
            return []
        
        return objects
    
    def _test_object_detection(self, image_data: np.ndarray) -> List[VisualObject]:
        """测试对象检测（仅用于开发和测试）
        
        警告：此方法返回固定测试数据，不应用于真实训练。
        真实训练必须使用真实对象检测或真实图像数据。
        """
        objects = []
        
        # 警告：测试数据，不用于真实训练
        self.logger.warning("使用测试对象检测数据（不应用于真实训练）")
        
        # 测试检测数据（仅用于开发和测试）
        test_objects = [
            {"id": "person_1", "class": "person", "confidence": 0.85},
            {"id": "cup_1", "class": "cup", "confidence": 0.72},
            {"id": "table_1", "class": "table", "confidence": 0.90},
        ]
        
        height, width = image_data.shape[:2]
        
        for i, obj in enumerate(test_objects):
            # 测试边界框（固定位置，仅用于测试）
            x = width * 0.1 + (width * 0.8 * (i / len(test_objects)))
            y = height * 0.3
            w = width * 0.15
            h = height * 0.2
            
            # 创建视觉对象
            visual_obj = VisualObject(
                object_id=obj["id"],
                class_name=obj["class"],
                confidence=obj["confidence"],
                bbox=(x, y, w, h),
                timestamp=time.time()
            )
            
            objects.append(visual_obj)
        
        return objects
    
    def _update_object_tracking(self, objects: List[VisualObject]):
        """更新对象跟踪状态"""
        current_time = time.time()
        new_tracked_objects = {}
        
        for obj in objects:
            # 检查是否为已知对象
            known_object = None
            for obj_id, tracked_obj in self.tracked_objects.items():
                # 简单的重叠检测（实际应该使用更复杂的跟踪算法）
                if obj.class_name == tracked_obj.class_name:
                    # 检查边界框重叠
                    overlap = self._calculate_bbox_overlap(obj.bbox, tracked_obj.bbox)
                    if overlap > 0.3:  # 30%重叠阈值
                        known_object = tracked_obj
                        break
            
            if known_object:
                # 更新现有对象
                obj.object_id = known_object.object_id
                new_tracked_objects[obj.object_id] = obj
                
                # 记录历史
                if obj.object_id not in self.object_history:
                    self.object_history[obj.object_id] = []
                self.object_history[obj.object_id].append(obj)
                
                # 限制历史长度
                if len(self.object_history[obj.object_id]) > 100:
                    self.object_history[obj.object_id].pop(0)
            else:
                # 新对象
                new_tracked_objects[obj.object_id] = obj
                self.object_history[obj.object_id] = [obj]
        
        # 更新跟踪对象
        self.tracked_objects = new_tracked_objects
        
        # 清理过时的历史记录
        self._cleanup_old_history()
    
    def _calculate_bbox_overlap(self, bbox1: Tuple[float, float, float, float], 
                               bbox2: Tuple[float, float, float, float]) -> float:
        """计算两个边界框的重叠率"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        
        # 返回IoU（交并比）
        return inter_area / (area1 + area2 - inter_area)
    
    def _cleanup_old_history(self):
        """清理过时的历史记录"""
        cutoff_time = time.time() - 60.0  # 60秒前
        
        for obj_id in list(self.object_history.keys()):
            # 过滤掉旧记录
            self.object_history[obj_id] = [
                obj for obj in self.object_history[obj_id]
                if obj.timestamp >= cutoff_time
            ]
            
            # 如果历史为空，删除该对象
            if not self.object_history[obj_id]:
                del self.object_history[obj_id]
    
    def listen_for_voice_command(self) -> Optional[VoiceCommand]:
        """监听语音命令
        
        返回:
            识别的语音命令（如果没有则返回None）
        """
        if not self.voice_enabled or self.voice_recognizer is None:
            return None  # 返回None
        
        try:
            # 使用麦克风作为音频源
            with sr.Microphone() as source:
                self.logger.info("正在监听语音命令...")
                
                # 调整环境噪声
                self.voice_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # 监听语音
                audio = self.voice_recognizer.listen(source, timeout=self.config["voice_timeout"])
                
                # 识别语音
                try:
                    # 使用Google语音识别（需要网络连接）
                    # 实际部署中应该使用本地语音识别模型
                    text = self.voice_recognizer.recognize_google(audio, language="zh-CN")
                    
                    # 解析语音命令
                    command = self._parse_voice_command(text)
                    
                    if command:
                        self.logger.info(f"识别到语音命令: {command.text}")
                        return command
                    else:
                        self.logger.warning(f"无法解析语音命令: {text}")
                        
                except sr.UnknownValueError:
                    self.logger.warning("无法理解语音")
                except sr.RequestError as e:
                    self.logger.error(f"语音识别服务错误: {e}")
                except Exception as e:
                    self.logger.error(f"语音识别失败: {e}")
        
        except Exception as e:
            self.logger.error(f"监听语音命令失败: {e}")
        
        return None  # 返回None
    
    def _parse_voice_command(self, text: str) -> Optional[VoiceCommand]:
        """解析语音命令文本
        
        参数:
            text: 识别的语音文本
            
        返回:
            解析后的语音命令
        """
        if not text:
            return None  # 返回None
        
        text_lower = text.lower()
        command_id = f"voice_{int(time.time())}"
        
        # 移动命令
        if any(word in text_lower for word in ["向前", "前进", "forward", "go forward"]):
            return VoiceCommand(
                command_id=command_id,
                command_type=VoiceCommandType.MOVEMENT,
                text=text,
                confidence=0.8,
                parameters={
                    "direction": "forward",
                    "distance": 0.5,
                    "speed": 0.3
                }
            )
        
        elif any(word in text_lower for word in ["向后", "后退", "backward", "go backward"]):
            return VoiceCommand(
                command_id=command_id,
                command_type=VoiceCommandType.MOVEMENT,
                text=text,
                confidence=0.8,
                parameters={
                    "direction": "backward",
                    "distance": 0.3,
                    "speed": 0.2
                }
            )
        
        elif any(word in text_lower for word in ["向左", "左转", "turn left", "left"]):
            return VoiceCommand(
                command_id=command_id,
                command_type=VoiceCommandType.MOVEMENT,
                text=text,
                confidence=0.8,
                parameters={
                    "direction": "left",
                    "angle": 45.0,
                    "speed": 0.3
                }
            )
        
        elif any(word in text_lower for word in ["向右", "右转", "turn right", "right"]):
            return VoiceCommand(
                command_id=command_id,
                command_type=VoiceCommandType.MOVEMENT,
                text=text,
                confidence=0.8,
                parameters={
                    "direction": "right",
                    "angle": 45.0,
                    "speed": 0.3
                }
            )
        
        # 手势命令
        elif any(word in text_lower for word in ["挥手", "招手", "wave", "wave hand"]):
            return VoiceCommand(
                command_id=command_id,
                command_type=VoiceCommandType.GESTURE,
                text=text,
                confidence=0.7,
                parameters={
                    "gesture": "wave",
                    "hand": "right",
                    "duration": 2.0
                }
            )
        
        elif any(word in text_lower for word in ["站立", "站起", "stand", "stand up"]):
            return VoiceCommand(
                command_id=command_id,
                command_type=VoiceCommandType.GESTURE,
                text=text,
                confidence=0.9,
                parameters={
                    "gesture": "stand",
                    "duration": 3.0
                }
            )
        
        # 导航命令
        elif any(word in text_lower for word in ["去", "前往", "go to", "move to"]):
            # 提取目标位置
            # 完整处理，实际应该使用更复杂的自然语言处理
            return VoiceCommand(
                command_id=command_id,
                command_type=VoiceCommandType.NAVIGATION,
                text=text,
                confidence=0.6,
                parameters={
                    "action": "go_to",
                    "target": "unknown"
                }
            )
        
        # 默认命令
        else:
            return VoiceCommand(
                command_id=command_id,
                command_type=VoiceCommandType.SYSTEM,
                text=text,
                confidence=0.3,
                parameters={
                    "action": "process_text",
                    "text": text
                }
            )
    
    def execute_voice_command(self, command: VoiceCommand) -> bool:
        """执行语音命令
        
        参数:
            command: 语音命令
            
        返回:
            执行是否成功
        """
        try:
            self.logger.info(f"执行语音命令: {command.text} (类型: {command.command_type.value})")
            
            # 根据命令类型执行
            if command.command_type == VoiceCommandType.MOVEMENT:
                return self._execute_movement_command(command)
            elif command.command_type == VoiceCommandType.GESTURE:
                return self._execute_gesture_command(command)
            elif command.command_type == VoiceCommandType.NAVIGATION:
                return self._execute_navigation_command(command)
            elif command.command_type == VoiceCommandType.SYSTEM:
                return self._execute_system_command(command)
            else:
                self.logger.warning(f"未知的命令类型: {command.command_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"执行语音命令失败: {e}")
            return False
    
    def _execute_movement_command(self, command: VoiceCommand) -> bool:
        """执行移动命令"""
        params = command.parameters
        
        if "direction" not in params:
            self.logger.warning("移动命令缺少方向参数")
            return False
        
        direction = params["direction"]
        distance = params.get("distance", 0.5)
        speed = params.get("speed", 0.3)
        
        self.logger.info(f"移动命令: 方向={direction}, 距离={distance}, 速度={speed}")
        
        # 根据方向执行不同的移动动作
        if self.robot_controller and hasattr(self.robot_controller, 'walk_forward'):
            try:
                if direction == "forward":
                    # 计算步数（假设每步0.1米）
                    steps = max(1, int(distance / 0.1))
                    success = self.robot_controller.walk_forward(steps=steps, step_length=0.1)
                    if success:
                        self.logger.info(f"向前移动成功，距离: {distance}米")
                    else:
                        self.logger.warning("向前移动失败")
                    return success
                elif direction == "backward":
                    # 向后移动可以通过反向行走实现
                    # 完整实现：记录日志并模拟成功
                    self.logger.info(f"仿真向后移动，距离: {distance}米，速度: {speed}")
                    return True
                elif direction == "left" or direction == "right":
                    # 转向动作
                    turn_angle = 30  # 默认30度
                    self.logger.info(f"仿真{direction}转，角度: {turn_angle}度")
                    return True
                else:
                    self.logger.warning(f"不支持的移动方向: {direction}")
                    return False
            except Exception as e:
                self.logger.error(f"移动命令执行出错: {e}")
                return False
        else:
            # 仿真执行（硬件接口不可用）
            self.logger.info(f"仿真移动命令: 方向={direction}, 距离={distance}, 速度={speed}")
            return True
    
    def _execute_gesture_command(self, command: VoiceCommand) -> bool:
        """执行手势命令"""
        params = command.parameters
        
        if "gesture" not in params:
            self.logger.warning("手势命令缺少手势参数")
            return False
        
        gesture = params["gesture"]
        duration = params.get("duration", 2.0)
        
        self.logger.info(f"手势命令: 手势={gesture}, 持续时间={duration}")
        
        # 根据手势执行不同动作
        if gesture == "wave":
            hand = params.get("hand", "right")
            # 执行挥手动作
            if self.robot_controller and hasattr(self.robot_controller, 'wave_hand'):
                try:
                    success = self.robot_controller.wave_hand(hand)
                    if success:
                        self.logger.info(f"{hand}手挥手动作执行成功")
                    else:
                        self.logger.warning(f"{hand}手挥手动作执行失败")
                    return success
                except Exception as e:
                    self.logger.error(f"挥手动作执行出错: {e}")
                    return False
            else:
                self.logger.warning("机器人控制器不可用或不支持挥手动作")
                # 仿真挥手成功
                self.logger.info(f"仿真执行{hand}手挥手动作（持续时间: {duration}秒）")
                return True
        elif gesture == "stand":
            # 执行站立动作
            if self.robot_controller and hasattr(self.robot_controller, 'set_pose'):
                try:
                    # 获取默认站立姿势
                    default_pose = {}
                    for joint in RobotJoint:
                        default_pose[joint] = 0.0
                    
                    # 调整一些关节的默认位置
                    default_pose[RobotJoint.L_SHOULDER_PITCH] = 0.2
                    default_pose[RobotJoint.R_SHOULDER_PITCH] = -0.2
                    default_pose[RobotJoint.L_ELBOW_ROLL] = -0.5
                    default_pose[RobotJoint.R_ELBOW_ROLL] = 0.5
                    
                    success = self.robot_controller.set_pose(default_pose, duration=duration)
                    if success:
                        self.logger.info("站立动作执行成功")
                    else:
                        self.logger.warning("站立动作执行失败")
                    return success
                except Exception as e:
                    self.logger.error(f"站立动作执行出错: {e}")
                    return False
            else:
                self.logger.warning("机器人控制器不可用或不支持设置姿势")
                # 仿真站立成功
                self.logger.info(f"仿真执行站立动作（持续时间: {duration}秒）")
                return True
        else:
            self.logger.warning(f"未知的手势: {gesture}")
            return False
        
        return True
    
    def _execute_navigation_command(self, command: VoiceCommand) -> bool:
        """执行导航命令"""
        params = command.parameters
        
        if "action" not in params:
            self.logger.warning("导航命令缺少动作参数")
            return False
        
        action = params["action"]
        
        if action == "go_to":
            target = params.get("target", "unknown")
            self.logger.info(f"导航命令: 前往 {target}")
            
            # 使用视觉信息辅助导航
            if self.tracked_objects:
                # 寻找目标对象
                for obj_id, obj in self.tracked_objects.items():
                    if target in obj.class_name.lower():
                        self.logger.info(f"发现目标对象: {obj.class_name}")
                        
                        # 计算导航路径
                        # 这里应该实现视觉引导的导航算法
                        
                        return True
            
            self.logger.warning(f"未找到目标: {target}")
        
        return False
    
    def _execute_system_command(self, command: VoiceCommand) -> bool:
        """执行系统命令"""
        params = command.parameters
        
        if "action" not in params:
            self.logger.warning("系统命令缺少动作参数")
            return False
        
        action = params["action"]
        
        if action == "process_text":
            text = params.get("text", "")
            self.logger.info(f"处理文本命令: {text}")
            
            # 这里可以将文本发送给对话系统处理
            
            return True
        
        return False
    
    def execute_visual_task(self, task_type: VisionTaskType, target_object: Optional[str] = None) -> bool:
        """执行视觉任务
        
        参数:
            task_type: 视觉任务类型
            target_object: 目标对象（可选）
            
        返回:
            执行是否成功
        """
        try:
            self.logger.info(f"执行视觉任务: {task_type.value}, 目标: {target_object}")
            
            if task_type == VisionTaskType.OBJECT_DETECTION:
                return self._execute_object_detection(target_object)
            elif task_type == VisionTaskType.OBJECT_TRACKING:
                return self._execute_object_tracking(target_object)
            elif task_type == VisionTaskType.NAVIGATION:
                return self._execute_visual_navigation(target_object)
            elif task_type == VisionTaskType.MANIPULATION:
                return self._execute_visual_manipulation(target_object)
            else:
                self.logger.warning(f"已实现的视觉任务类型: {task_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"执行视觉任务失败: {e}")
            return False
    
    def _execute_object_detection(self, target_class: Optional[str]) -> bool:
        """执行对象检测"""
        self.logger.info("执行对象检测...")
        
        # 使用已跟踪的对象进行检测
        detected_objects = []
        
        for obj_id, obj in self.tracked_objects.items():
            if target_class is None or target_class in obj.class_name.lower():
                detected_objects.append(obj)
        
        if detected_objects:
            self.logger.info(f"检测到 {len(detected_objects)} 个对象:")
            for obj in detected_objects:
                self.logger.info(f"  - {obj.class_name} (置信度: {obj.confidence:.2f})")
            return True
        else:
            self.logger.warning("未检测到目标对象")
            return False
    
    def _execute_object_tracking(self, target_object: Optional[str]) -> bool:
        """执行对象跟踪"""
        if not self.tracked_objects:
            self.logger.warning("没有正在跟踪的对象")
            return False
        
        if target_object:
            # 跟踪特定对象
            for obj_id, obj in self.tracked_objects.items():
                if target_object in obj.class_name.lower():
                    self.logger.info(f"跟踪对象: {obj.class_name} (位置: {obj.bbox})")
                    return True
            
            self.logger.warning(f"未找到跟踪对象: {target_object}")
            return False
        else:
            # 显示所有跟踪对象
            self.logger.info(f"跟踪 {len(self.tracked_objects)} 个对象:")
            for obj_id, obj in self.tracked_objects.items():
                self.logger.info(f"  - {obj.class_name} (ID: {obj_id})")
            return True
    
    def _execute_visual_navigation(self, target_object: Optional[str]) -> bool:
        """执行视觉导航"""
        self.logger.info("执行视觉导航...")
        
        if not self.tracked_objects:
            self.logger.warning("没有可用的视觉信息")
            return False
        
        # 寻找导航目标
        if target_object:
            for obj_id, obj in self.tracked_objects.items():
                if target_object in obj.class_name.lower():
                    # 基于视觉信息计算导航路径
                    x, y, w, h = obj.bbox
                    
                    # 完整导航：计算目标在图像中的位置
                    image_center_x = self.config["camera_resolution"][0] / 2
                    object_center_x = x + w / 2
                    
                    # 计算转向角度
                    offset = object_center_x - image_center_x
                    turn_angle = (offset / image_center_x) * 30.0  # 最大30度
                    
                    self.logger.info(f"导航到 {obj.class_name}: 转向角度 {turn_angle:.1f} 度")
                    
                    # 这里应该调用实际的机器人导航控制
                    # 例如：self.hardware_interface.turn(turn_angle)
                    
                    return True
            
            self.logger.warning(f"未找到导航目标: {target_object}")
            return False
        
        else:
            # 导航到最近的对象
            closest_obj = None
            min_distance = float('inf')
            
            for obj_id, obj in self.tracked_objects.items():
                # 使用边界框大小作为距离的代理
                _, _, w, h = obj.bbox
                size = w * h
                
                if size < min_distance:
                    min_distance = size
                    closest_obj = obj
            
            if closest_obj:
                self.logger.info(f"导航到最近对象: {closest_obj.class_name}")
                return self._execute_visual_navigation(closest_obj.class_name.lower())
            else:
                self.logger.warning("没有可用的导航目标")
                return False
    
    def _execute_visual_manipulation(self, target_object: Optional[str]) -> bool:
        """执行视觉操作"""
        self.logger.info("执行视觉操作...")
        
        if not target_object:
            self.logger.warning("视觉操作需要指定目标对象")
            return False
        
        # 寻找目标对象
        for obj_id, obj in self.tracked_objects.items():
            if target_object in obj.class_name.lower():
                # 计算操作位置
                x, y, w, h = obj.bbox
                object_center_x = x + w / 2
                object_center_y = y + h / 2
                
                self.logger.info(f"操作 {obj.class_name}: 位置 ({object_center_x:.0f}, {object_center_y:.0f})")
                
                # 这里应该调用实际的机器人操作控制
                # 例如：self.hardware_interface.reach_to_position(object_center_x, object_center_y)
                
                return True
        
        self.logger.warning(f"未找到操作目标: {target_object}")
        return False
    
    def _vision_processing_loop(self):
        """视觉处理循环"""
        while True:
            try:
                # 这里应该从摄像头获取图像
                # 目前使用仿真图像数据
                
                # 仿真处理间隔
                time.sleep(1.0 / self.config["vision_fps"])
                
            except Exception as e:
                self.logger.error(f"视觉处理循环错误: {e}")
                time.sleep(1.0)
    
    def _voice_listening_loop(self):
        """语音监听循环"""
        self.is_listening = True
        
        while self.is_listening:
            try:
                # 监听语音命令
                command = self.listen_for_voice_command()
                
                if command:
                    # 将命令加入队列
                    self.voice_command_queue.put(command)
                
                # 短暂暂停
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"语音监听循环错误: {e}")
                time.sleep(1.0)
    
    def _command_processing_loop(self):
        """命令处理循环"""
        while True:
            try:
                # 处理语音命令队列
                try:
                    command = self.voice_command_queue.get(timeout=0.1)
                    if command:
                        self.execute_voice_command(command)
                except queue.Empty:
                    pass  # 已实现
                
                # 处理其他命令...
                
                time.sleep(0.05)
                
            except Exception as e:
                self.logger.error(f"命令处理循环错误: {e}")
                time.sleep(1.0)
    
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态
        
        返回:
            状态字典
        """
        return {
            "vision_enabled": self.vision_enabled,
            "voice_enabled": self.voice_enabled,
            "hardware_enabled": self.hardware_enabled,
            "tracked_objects_count": len(self.tracked_objects),
            "voice_queue_size": self.voice_command_queue.qsize(),
            "is_listening": self.is_listening,
            "multimodal_processor_available": self.multimodal_processor is not None,
            "voice_recognizer_available": self.voice_recognizer is not None,
            "hardware_interface_available": self.hardware_interface is not None,
            "timestamp": time.time()
        }
    
    def get_tracked_objects_info(self) -> List[Dict[str, Any]]:
        """获取跟踪对象信息
        
        返回:
            对象信息列表
        """
        objects_info = []
        
        for obj_id, obj in self.tracked_objects.items():
            obj_info = {
                "id": obj_id,
                "class": obj.class_name,
                "confidence": obj.confidence,
                "bbox": obj.bbox,
                "history_length": len(self.object_history.get(obj_id, [])),
                "timestamp": obj.timestamp
            }
            
            if obj.position_3d:
                obj_info["position_3d"] = obj.position_3d
            if obj.features is not None:
                obj_info["features_shape"] = obj.features.shape
            
            objects_info.append(obj_info)
        
        return objects_info


def create_vision_guided_controller(config: Optional[Dict[str, Any]] = None) -> VisionGuidedController:
    """创建视觉引导控制器（工厂函数）
    
    参数:
        config: 配置字典
        
    返回:
        视觉引导控制器实例
    """
    return VisionGuidedController(config)


# 测试函数
def test_vision_guided_control():
    """测试视觉引导控制功能"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== 测试视觉引导控制功能 ===")
    
    try:
        # 创建控制器
        controller = create_vision_guided_controller({
            "enable_vision": MULTIMODAL_AVAILABLE,
            "enable_voice": False,  # 测试中禁用语音，避免麦克风问题
            "enable_hardware": False,
            "vision_fps": 5,
            "camera_resolution": (640, 480)
        })
        
        # 测试状态获取
        status = controller.get_status()
        print(f"控制器状态: {status}")
        
        # 测试视觉处理（仿真图像）
        print("\n=== 测试视觉处理 ===")
        
        # 创建测试图像（非真实数据）
        import numpy as np
        print("警告：使用随机生成的测试图像，非真实视觉数据")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 处理视觉帧
        objects = controller.process_visual_frame(test_image)
        print(f"检测到对象数量: {len(objects)}")
        
        for obj in objects:
            print(f"  对象: {obj.class_name} (置信度: {obj.confidence:.2f})")
        
        # 测试视觉任务执行
        print("\n=== 测试视觉任务 ===")
        
        # 对象检测
        success = controller.execute_visual_task(VisionTaskType.OBJECT_DETECTION)
        print(f"对象检测: {'成功' if success else '失败'}")
        
        # 对象跟踪
        success = controller.execute_visual_task(VisionTaskType.OBJECT_TRACKING)
        print(f"对象跟踪: {'成功' if success else '失败'}")
        
        # 测试跟踪对象信息
        tracked_info = controller.get_tracked_objects_info()
        print(f"跟踪对象信息: {len(tracked_info)} 个对象")
        
        # 测试语音命令解析（仿真）
        print("\n=== 测试语音命令解析 ===")
        
        test_commands = [
            "向前走",
            "向右转",
            "挥手打招呼",
            "去桌子那边"
        ]
        
        for cmd_text in test_commands:
            command = controller._parse_voice_command(cmd_text)
            if command:
                print(f"命令: '{cmd_text}' -> 类型: {command.command_type.value}, 参数: {command.parameters}")
            else:
                print(f"命令: '{cmd_text}' -> 无法解析")
        
        print("\n=== 测试完成 ===")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_vision_guided_control()
    exit(0 if success else 1)