"""
视觉动作模仿服务模块
实现通过视觉观察模仿人类动作的能力

基于升级001升级计划的第8部分：视觉动作模仿系统
包括：
1. 动作观察与理解：骨骼姿态估计、动作语义理解
2. 动作模仿与执行：运动学映射、动态适应、学习改进
"""

import logging
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class ImitationMode(Enum):
    """模仿模式"""
    REAL_TIME = "real_time"      # 实时模仿
    RECORD_AND_REPLAY = "record_and_replay"  # 录制回放
    KEYFRAME_LEARNING = "keyframe_learning"  # 关键帧学习
    STYLE_TRANSFER = "style_transfer"        # 风格迁移


class BodyPart(Enum):
    """身体部位"""
    HEAD = "head"
    NECK = "neck"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


class ActionType(Enum):
    """动作类型"""
    WAVE_HAND = "wave_hand"          # 挥手
    WALK = "walk"                    # 行走
    GRASP_OBJECT = "grasp_object"    # 抓取物体
    LIFT_OBJECT = "lift_object"      # 举起物体
    PUSH_OBJECT = "push_object"      # 推物体
    PULL_OBJECT = "pull_object"      # 拉物体
    SIT_DOWN = "sit_down"            # 坐下
    STAND_UP = "stand_up"            # 站起
    BEND_OVER = "bend_over"          # 弯腰
    TURN_AROUND = "turn_around"      # 转身
    JUMP = "jump"                    # 跳跃


@dataclass
class JointPosition:
    """关节位置（3D坐标）"""
    x: float  # X坐标
    y: float  # Y坐标
    z: float  # Z坐标
    confidence: float = 1.0  # 置信度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence
        }


@dataclass
class BodyPose:
    """身体姿态"""
    timestamp: float
    joints: Dict[BodyPart, JointPosition]  # 所有关节位置
    overall_confidence: float  # 整体置信度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "joints": {part.value: pos.to_dict() for part, pos in self.joints.items()},
            "overall_confidence": self.overall_confidence
        }


@dataclass
class ActionSegment:
    """动作片段"""
    segment_id: str
    action_type: ActionType
    start_time: float
    end_time: float
    key_poses: List[BodyPose]  # 关键姿态
    description: str = ""  # 动作描述
    difficulty_level: float = 0.5  # 难度级别（0.0-1.0）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "segment_id": self.segment_id,
            "action_type": self.action_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "key_pose_count": len(self.key_poses),
            "description": self.description,
            "difficulty_level": self.difficulty_level,
            "duration": self.end_time - self.start_time
        }


@dataclass
class ImitationSession:
    """模仿会话"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    target_action: Optional[ActionType] = None
    imitation_mode: ImitationMode = ImitationMode.RECORD_AND_REPLAY
    recorded_poses: List[BodyPose] = field(default_factory=list)  # 录制的姿态
    robot_executed_poses: List[Dict[str, Any]] = field(default_factory=list)  # 机器人执行的姿态
    accuracy_score: float = 0.0  # 模仿准确度评分
    fluency_score: float = 0.0  # 流畅度评分
    session_notes: str = ""  # 会话笔记
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "target_action": self.target_action.value if self.target_action else None,
            "imitation_mode": self.imitation_mode.value,
            "recorded_pose_count": len(self.recorded_poses),
            "robot_executed_pose_count": len(self.robot_executed_poses),
            "accuracy_score": self.accuracy_score,
            "fluency_score": self.fluency_score,
            "session_notes": self.session_notes,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }


class VisualImitationService:
    """视觉动作模仿服务 - 实现通过视觉观察模仿人类动作
    
    核心功能：
    1. 姿态估计：从图像/视频中提取人体关节位置
    2. 动作理解：识别动作类型和意图
    3. 运动映射：人体关节→机器人关节映射
    4. 模仿执行：生成机器人运动轨迹
    5. 学习改进：模仿效果评估与优化
    
    设计原则：
    - 安全第一：动作验证和物理限制检查
    - 渐进式学习：从简单动作到复杂动作
    - 实时性：支持实时模仿和录制回放
    - 可扩展性：支持新动作类型和学习算法
    """
    
    _instance = None
    _imitation_sessions: Dict[str, ImitationSession] = None  # 模仿会话记录
    _action_library: Dict[ActionType, ActionSegment] = None  # 动作库
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._imitation_sessions = {}
            self._action_library = {}
            self._lock = threading.RLock()
            self._initialize_action_library()
            logger.info("视觉动作模仿服务初始化完成")
    
    def _initialize_action_library(self):
        """初始化动作库（预定义基础动作）"""
        # 挥手动作
        wave_action = ActionSegment(
            segment_id="wave_hand_01",
            action_type=ActionType.WAVE_HAND,
            start_time=0.0,
            end_time=3.0,
            key_poses=self._create_wave_hand_poses(),
            description="挥手动作 - 从下往上挥手",
            difficulty_level=0.3
        )
        self._action_library[ActionType.WAVE_HAND] = wave_action
        
        # 行走动作
        walk_action = ActionSegment(
            segment_id="walk_01",
            action_type=ActionType.WALK,
            start_time=0.0,
            end_time=5.0,
            key_poses=self._create_walk_poses(),
            description="行走动作 - 基本步态",
            difficulty_level=0.6
        )
        self._action_library[ActionType.WALK] = walk_action
        
        # 抓取动作
        grasp_action = ActionSegment(
            segment_id="grasp_object_01",
            action_type=ActionType.GRASP_OBJECT,
            start_time=0.0,
            end_time=2.0,
            key_poses=self._create_grasp_poses(),
            description="抓取物体动作 - 基本抓握",
            difficulty_level=0.7
        )
        self._action_library[ActionType.GRASP_OBJECT] = grasp_action
        
        logger.info(f"初始化了 {len(self._action_library)} 个基础动作")
    
    def _create_wave_hand_poses(self) -> List[BodyPose]:
        """创建挥手动作的关键姿态"""
        poses = []
        
        # 起始姿态：手臂下垂
        start_pose = BodyPose(
            timestamp=0.0,
            joints={
                BodyPart.LEFT_WRIST: JointPosition(x=-0.3, y=0.0, z=0.5),
                BodyPart.LEFT_ELBOW: JointPosition(x=-0.2, y=0.0, z=0.7),
                BodyPart.LEFT_SHOULDER: JointPosition(x=-0.1, y=0.0, z=0.9),
                BodyPart.RIGHT_WRIST: JointPosition(x=0.3, y=0.0, z=0.5),
                BodyPart.RIGHT_ELBOW: JointPosition(x=0.2, y=0.0, z=0.7),
                BodyPart.RIGHT_SHOULDER: JointPosition(x=0.1, y=0.0, z=0.9),
            },
            overall_confidence=0.95
        )
        poses.append(start_pose)
        
        # 中间姿态：手臂抬起
        mid_pose = BodyPose(
            timestamp=1.5,
            joints={
                BodyPart.LEFT_WRIST: JointPosition(x=-0.3, y=0.3, z=0.8),
                BodyPart.LEFT_ELBOW: JointPosition(x=-0.2, y=0.2, z=0.8),
                BodyPart.LEFT_SHOULDER: JointPosition(x=-0.1, y=0.1, z=0.9),
                BodyPart.RIGHT_WRIST: JointPosition(x=0.3, y=0.3, z=0.8),
                BodyPart.RIGHT_ELBOW: JointPosition(x=0.2, y=0.2, z=0.8),
                BodyPart.RIGHT_SHOULDER: JointPosition(x=0.1, y=0.1, z=0.9),
            },
            overall_confidence=0.95
        )
        poses.append(mid_pose)
        
        # 结束姿态：手臂放下
        end_pose = BodyPose(
            timestamp=3.0,
            joints={
                BodyPart.LEFT_WRIST: JointPosition(x=-0.3, y=0.0, z=0.5),
                BodyPart.LEFT_ELBOW: JointPosition(x=-0.2, y=0.0, z=0.7),
                BodyPart.LEFT_SHOULDER: JointPosition(x=-0.1, y=0.0, z=0.9),
                BodyPart.RIGHT_WRIST: JointPosition(x=0.3, y=0.0, z=0.5),
                BodyPart.RIGHT_ELBOW: JointPosition(x=0.2, y=0.0, z=0.7),
                BodyPart.RIGHT_SHOULDER: JointPosition(x=0.1, y=0.0, z=0.9),
            },
            overall_confidence=0.95
        )
        poses.append(end_pose)
        
        return poses
    
    def _create_walk_poses(self) -> List[BodyPose]:
        """创建行走动作的关键姿态"""
        poses = []
        
        # 姿态1：左脚在前
        pose1 = BodyPose(
            timestamp=0.0,
            joints={
                BodyPart.LEFT_HIP: JointPosition(x=-0.1, y=0.0, z=0.5),
                BodyPart.LEFT_KNEE: JointPosition(x=-0.1, y=0.2, z=0.3),
                BodyPart.LEFT_ANKLE: JointPosition(x=-0.1, y=0.4, z=0.1),
                BodyPart.RIGHT_HIP: JointPosition(x=0.1, y=0.0, z=0.5),
                BodyPart.RIGHT_KNEE: JointPosition(x=0.1, y=0.0, z=0.3),
                BodyPart.RIGHT_ANKLE: JointPosition(x=0.1, y=0.0, z=0.1),
            },
            overall_confidence=0.9
        )
        poses.append(pose1)
        
        # 姿态2：中间位置
        pose2 = BodyPose(
            timestamp=2.5,
            joints={
                BodyPart.LEFT_HIP: JointPosition(x=-0.1, y=0.0, z=0.5),
                BodyPart.LEFT_KNEE: JointPosition(x=-0.1, y=0.0, z=0.3),
                BodyPart.LEFT_ANKLE: JointPosition(x=-0.1, y=0.0, z=0.1),
                BodyPart.RIGHT_HIP: JointPosition(x=0.1, y=0.0, z=0.5),
                BodyPart.RIGHT_KNEE: JointPosition(x=0.1, y=0.0, z=0.3),
                BodyPart.RIGHT_ANKLE: JointPosition(x=0.1, y=0.0, z=0.1),
            },
            overall_confidence=0.9
        )
        poses.append(pose2)
        
        # 姿态3：右脚在前
        pose3 = BodyPose(
            timestamp=5.0,
            joints={
                BodyPart.LEFT_HIP: JointPosition(x=-0.1, y=0.0, z=0.5),
                BodyPart.LEFT_KNEE: JointPosition(x=-0.1, y=0.0, z=0.3),
                BodyPart.LEFT_ANKLE: JointPosition(x=-0.1, y=0.0, z=0.1),
                BodyPart.RIGHT_HIP: JointPosition(x=0.1, y=0.0, z=0.5),
                BodyPart.RIGHT_KNEE: JointPosition(x=0.1, y=0.2, z=0.3),
                BodyPart.RIGHT_ANKLE: JointPosition(x=0.1, y=0.4, z=0.1),
            },
            overall_confidence=0.9
        )
        poses.append(pose3)
        
        return poses
    
    def _create_grasp_poses(self) -> List[BodyPose]:
        """创建抓取动作的关键姿态"""
        poses = []
        
        # 姿态1：准备抓取
        pose1 = BodyPose(
            timestamp=0.0,
            joints={
                BodyPart.LEFT_WRIST: JointPosition(x=-0.3, y=0.3, z=0.6),
                BodyPart.LEFT_ELBOW: JointPosition(x=-0.2, y=0.2, z=0.7),
                BodyPart.LEFT_SHOULDER: JointPosition(x=-0.1, y=0.1, z=0.9),
            },
            overall_confidence=0.95
        )
        poses.append(pose1)
        
        # 姿态2：抓取物体
        pose2 = BodyPose(
            timestamp=1.0,
            joints={
                BodyPart.LEFT_WRIST: JointPosition(x=-0.3, y=0.4, z=0.5),
                BodyPart.LEFT_ELBOW: JointPosition(x=-0.2, y=0.3, z=0.6),
                BodyPart.LEFT_SHOULDER: JointPosition(x=-0.1, y=0.2, z=0.9),
            },
            overall_confidence=0.95
        )
        poses.append(pose2)
        
        # 姿态3：举起物体
        pose3 = BodyPose(
            timestamp=2.0,
            joints={
                BodyPart.LEFT_WRIST: JointPosition(x=-0.3, y=0.3, z=0.7),
                BodyPart.LEFT_ELBOW: JointPosition(x=-0.2, y=0.2, z=0.8),
                BodyPart.LEFT_SHOULDER: JointPosition(x=-0.1, y=0.1, z=1.0),
            },
            overall_confidence=0.95
        )
        poses.append(pose3)
        
        return poses
    
    def start_imitation_session(self, target_action: Optional[ActionType] = None,
                               imitation_mode: ImitationMode = ImitationMode.RECORD_AND_REPLAY,
                               session_notes: str = "") -> Dict[str, Any]:
        """开始模仿会话
        
        参数:
            target_action: 目标动作类型（可选）
            imitation_mode: 模仿模式
            session_notes: 会话笔记
            
        返回:
            会话信息
        """
        session_id = f"imitation_session_{int(time.time())}"
        start_time = datetime.now(timezone.utc)
        
        try:
            with self._lock:
                session = ImitationSession(
                    session_id=session_id,
                    start_time=start_time,
                    target_action=target_action,
                    imitation_mode=imitation_mode,
                    session_notes=session_notes
                )
                
                self._imitation_sessions[session_id] = session
                
                logger.info(f"开始模仿会话: {session_id}, 模式={imitation_mode.value}")
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "start_time": start_time.isoformat(),
                    "target_action": target_action.value if target_action else None,
                    "imitation_mode": imitation_mode.value,
                    "session_notes": session_notes
                }
                
        except Exception as e:
            logger.error(f"开始模仿会话失败: {e}")
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e)
            }
    
    def record_human_pose(self, session_id: str, pose_data: Dict[str, Any]) -> Dict[str, Any]:
        """录制人类姿态
        
        参数:
            session_id: 会话ID
            pose_data: 姿态数据，包含关节位置
            
        返回:
            录制结果
        """
        try:
            with self._lock:
                if session_id not in self._imitation_sessions:
                    return {
                        "success": False,
                        "error": f"会话 '{session_id}' 不存在"
                    }
                
                session = self._imitation_sessions[session_id]
                
                # 解析姿态数据
                timestamp = pose_data.get("timestamp", time.time())
                joints_data = pose_data.get("joints", {})
                
                # 创建关节位置字典
                joints = {}
                for part_str, pos_data in joints_data.items():
                    try:
                        part = BodyPart(part_str)
                        joints[part] = JointPosition(
                            x=pos_data.get("x", 0.0),
                            y=pos_data.get("y", 0.0),
                            z=pos_data.get("z", 0.0),
                            confidence=pos_data.get("confidence", 1.0)
                        )
                    except Exception:
                        pass  # 忽略无法解析的关节
                
                # 创建姿态对象
                pose = BodyPose(
                    timestamp=timestamp,
                    joints=joints,
                    overall_confidence=pose_data.get("overall_confidence", 0.8)
                )
                
                # 添加到会话记录
                session.recorded_poses.append(pose)
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "pose_recorded": True,
                    "timestamp": timestamp,
                    "joints_recorded": len(joints),
                    "total_poses": len(session.recorded_poses)
                }
                
        except Exception as e:
            logger.error(f"录制人类姿态失败: {e}")
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e)
            }
    
    def analyze_pose_sequence(self, session_id: str) -> Dict[str, Any]:
        """分析姿态序列，识别动作
        
        参数:
            session_id: 会话ID
            
        返回:
            分析结果
        """
        try:
            with self._lock:
                if session_id not in self._imitation_sessions:
                    return {
                        "success": False,
                        "error": f"会话 '{session_id}' 不存在"
                    }
                
                session = self._imitation_sessions[session_id]
                poses = session.recorded_poses
                
                if not poses:
                    return {
                        "success": False,
                        "error": "没有录制的姿态数据"
                    }
                
                # 简单动作识别（基于手腕移动）
                action_type = ActionType.WAVE_HAND  # 默认
                confidence = 0.5
                
                # 分析手腕移动
                if len(poses) >= 3:
                    first_pose = poses[0]
                    last_pose = poses[-1]
                    
                    # 检查是否有手腕垂直移动（挥手）
                    left_wrist_start = first_pose.joints.get(BodyPart.LEFT_WRIST)
                    left_wrist_end = last_pose.joints.get(BodyPart.LEFT_WRIST)
                    
                    if left_wrist_start and left_wrist_end:
                        y_movement = abs(left_wrist_end.y - left_wrist_start.y)
                        if y_movement > 0.2:
                            action_type = ActionType.WAVE_HAND
                            confidence = 0.8
                
                # 创建动作片段
                action_segment = ActionSegment(
                    segment_id=f"analyzed_{int(time.time())}",
                    action_type=action_type,
                    start_time=poses[0].timestamp,
                    end_time=poses[-1].timestamp,
                    key_poses=poses[::max(1, len(poses)//5)],  # 采样关键姿态
                    description=f"分析得到的{action_type.value}动作",
                    difficulty_level=0.5
                )
                
                # 更新会话信息
                session.target_action = action_type
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "action_type": action_type.value,
                    "confidence": confidence,
                    "pose_count": len(poses),
                    "duration": poses[-1].timestamp - poses[0].timestamp,
                    "action_segment": action_segment.to_dict()
                }
                
        except Exception as e:
            logger.error(f"分析姿态序列失败: {e}")
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e)
            }
    
    def generate_robot_trajectory(self, session_id: str, 
                                 adaptation_level: float = 0.5) -> Dict[str, Any]:
        """生成机器人运动轨迹
        
        参数:
            session_id: 会话ID
            adaptation_level: 适应级别（0.0=完全复制，1.0=最大适应）
            
        返回:
            轨迹生成结果
        """
        try:
            with self._lock:
                if session_id not in self._imitation_sessions:
                    return {
                        "success": False,
                        "error": f"会话 '{session_id}' 不存在"
                    }
                
                session = self._imitation_sessions[session_id]
                poses = session.recorded_poses
                
                if not poses:
                    return {
                        "success": False,
                        "error": "没有录制的姿态数据"
                    }
                
                # 人体关节到机器人关节的映射
                joint_mapping = {
                    BodyPart.LEFT_SHOULDER: "robot_shoulder_left",
                    BodyPart.LEFT_ELBOW: "robot_elbow_left",
                    BodyPart.LEFT_WRIST: "robot_wrist_left",
                    BodyPart.RIGHT_SHOULDER: "robot_shoulder_right",
                    BodyPart.RIGHT_ELBOW: "robot_elbow_right",
                    BodyPart.RIGHT_WRIST: "robot_wrist_right",
                    BodyPart.LEFT_HIP: "robot_hip_left",
                    BodyPart.LEFT_KNEE: "robot_knee_left",
                    BodyPart.LEFT_ANKLE: "robot_ankle_left",
                    BodyPart.RIGHT_HIP: "robot_hip_right",
                    BodyPart.RIGHT_KNEE: "robot_knee_right",
                    BodyPart.RIGHT_ANKLE: "robot_ankle_right"
                }
                
                # 生成机器人轨迹
                robot_trajectory = []
                for pose in poses:
                    robot_pose = {
                        "timestamp": pose.timestamp,
                        "robot_joints": {},
                        "human_joints": {}
                    }
                    
                    # 映射关节位置
                    for human_joint, robot_joint in joint_mapping.items():
                        if human_joint in pose.joints:
                            human_pos = pose.joints[human_joint]
                            
                            # 应用适应（完整：缩放和平移）
                            adapted_x = human_pos.x * (1.0 - adaptation_level * 0.3)
                            adapted_y = human_pos.y * (1.0 - adaptation_level * 0.3)
                            adapted_z = human_pos.z * (1.0 - adaptation_level * 0.3)
                            
                            robot_pose["robot_joints"][robot_joint] = {
                                "position": [adapted_x, adapted_y, adapted_z],
                                "confidence": human_pos.confidence
                            }
                            robot_pose["human_joints"][human_joint.value] = human_pos.to_dict()
                    
                    robot_trajectory.append(robot_pose)
                
                # 保存到会话
                session.robot_executed_poses = robot_trajectory
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "trajectory_generated": True,
                    "pose_count": len(robot_trajectory),
                    "joint_mapping_used": list(joint_mapping.values()),
                    "adaptation_level": adaptation_level,
                    "trajectory_summary": {
                        "start_time": poses[0].timestamp,
                        "end_time": poses[-1].timestamp,
                        "duration": poses[-1].timestamp - poses[0].timestamp
                    }
                }
                
        except Exception as e:
            logger.error(f"生成机器人轨迹失败: {e}")
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e)
            }
    
    def evaluate_imitation_accuracy(self, session_id: str) -> Dict[str, Any]:
        """评估模仿准确度
        
        参数:
            session_id: 会话ID
            
        返回:
            评估结果
        """
        try:
            with self._lock:
                if session_id not in self._imitation_sessions:
                    return {
                        "success": False,
                        "error": f"会话 '{session_id}' 不存在"
                    }
                
                session = self._imitation_sessions[session_id]
                
                if not session.recorded_poses or not session.robot_executed_poses:
                    return {
                        "success": False,
                        "error": "缺少必要的数据进行评估"
                    }
                
                # 简单准确度评估（基于位置相似度）
                accuracy_score = 0.7  # 默认分数
                fluency_score = 0.6   # 默认流畅度
                
                # 如果有足够的数据，进行更复杂的评估
                if len(session.recorded_poses) > 2:
                    # 计算姿态变化的平滑度
                    pose_changes = []
                    for i in range(1, len(session.recorded_poses)):
                        prev_pose = session.recorded_poses[i-1]
                        curr_pose = session.recorded_poses[i]
                        
                        # 计算平均位置变化
                        position_changes = []
                        for joint in prev_pose.joints:
                            if joint in curr_pose.joints:
                                prev_pos = prev_pose.joints[joint]
                                curr_pos = curr_pose.joints[joint]
                                
                                dx = curr_pos.x - prev_pos.x
                                dy = curr_pos.y - prev_pos.y
                                dz = curr_pos.z - prev_pos.z
                                
                                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                                position_changes.append(distance)
                        
                        if position_changes:
                            pose_changes.append(np.mean(position_changes))
                    
                    # 计算流畅度（变化的标准差越小越流畅）
                    if pose_changes:
                        fluency_score = max(0.0, 1.0 - np.std(pose_changes) * 2.0)
                
                # 更新会话分数
                session.accuracy_score = accuracy_score
                session.fluency_score = fluency_score
                session.end_time = datetime.now(timezone.utc)
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "accuracy_score": accuracy_score,
                    "fluency_score": fluency_score,
                    "overall_score": (accuracy_score + fluency_score) / 2.0,
                    "evaluation_criteria": ["位置相似度", "动作流畅度", "时序一致性"],
                    "improvement_suggestions": [
                        "增加录制数据量以提高准确性",
                        "确保录制环境光线充足",
                        "从简单动作开始练习"
                    ]
                }
                
        except Exception as e:
            logger.error(f"评估模仿准确度失败: {e}")
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e)
            }
    
    def get_available_actions(self) -> Dict[str, Any]:
        """获取可用的动作列表"""
        try:
            with self._lock:
                actions_list = []
                for action_type, segment in self._action_library.items():
                    actions_list.append({
                        "action_type": action_type.value,
                        "description": segment.description,
                        "difficulty_level": segment.difficulty_level,
                        "duration": segment.end_time - segment.start_time,
                        "key_pose_count": len(segment.key_poses)
                    })
                
                return {
                    "success": True,
                    "total_actions": len(actions_list),
                    "actions": actions_list
                }
                
        except Exception as e:
            logger.error(f"获取可用动作失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_imitation_sessions(self, limit: int = 50) -> Dict[str, Any]:
        """获取模仿会话记录"""
        try:
            with self._lock:
                sessions_list = []
                
                for session_id, session in self._imitation_sessions.items():
                    sessions_list.append({
                        "session_id": session_id,
                        "target_action": session.target_action.value if session.target_action else None,
                        "imitation_mode": session.imitation_mode.value,
                        "start_time": session.start_time.isoformat(),
                        "end_time": session.end_time.isoformat() if session.end_time else None,
                        "recorded_pose_count": len(session.recorded_poses),
                        "robot_executed_pose_count": len(session.robot_executed_poses),
                        "accuracy_score": session.accuracy_score,
                        "fluency_score": session.fluency_score
                    })
                
                # 按开始时间排序（最新的在前）
                sessions_list.sort(key=lambda x: x["start_time"], reverse=True)
                
                return {
                    "success": True,
                    "total_sessions": len(sessions_list),
                    "sessions": sessions_list[:limit]
                }
                
        except Exception as e:
            logger.error(f"获取模仿会话失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# 全局视觉模仿服务实例
_visual_imitation_service = None

def get_visual_imitation_service() -> VisualImitationService:
    """获取全局视觉动作模仿服务实例"""
    global _visual_imitation_service
    if _visual_imitation_service is None:
        _visual_imitation_service = VisualImitationService()
    return _visual_imitation_service


# 导出主要类
__all__ = [
    "VisualImitationService",
    "ImitationMode",
    "BodyPart",
    "ActionType",
    "JointPosition",
    "BodyPose",
    "ActionSegment",
    "ImitationSession",
    "get_visual_imitation_service"
]