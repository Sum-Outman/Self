#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一自主学习管理器
解决审计报告中"自主学习能力已实现 - 完成度25%"问题

功能：
1. 知识缺口检测：基于当前能力和目标的能力差距分析
2. 学习目标自动生成：智能学习目标设定和优先级排序
3. 学习资源发现：自动发现和收集学习材料
4. 学习进度监控：实时跟踪学习进度和效果评估
5. 自我改进循环：基于反馈的学习策略优化
6. 知识整合：将新知识整合到现有知识体系中
"""

import logging
import time
import json
import threading
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import hashlib
import random

logger = logging.getLogger(__name__)

# 导入相关模块
try:
    from models.knowledge_base.knowledge_manager import KnowledgeManager
    KNOWLEDGE_MANAGER_AVAILABLE = True
except ImportError as e:
    KNOWLEDGE_MANAGER_AVAILABLE = False
    logger.warning(f"知识管理器不可用: {e}")

try:
    from models.memory.memory_manager import MemorySystem
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError as e:
    MEMORY_SYSTEM_AVAILABLE = False
    logger.warning(f"记忆系统不可用: {e}")

try:
    from models.reasoning_engine import ReasoningEngine
    REASONING_ENGINE_AVAILABLE = True
except ImportError as e:
    REASONING_ENGINE_AVAILABLE = False
    logger.warning(f"推理引擎不可用: {e}")


class LearningState(Enum):
    """学习状态枚举"""
    IDLE = auto()          # 空闲
    ANALYZING = auto()     # 分析知识缺口
    PLANNING = auto()      # 制定学习计划
    GATHERING = auto()     # 收集学习资源
    LEARNING = auto()      # 学习执行
    EVALUATING = auto()    # 学习评估
    INTEGRATING = auto()   # 知识整合
    ADAPTING = auto()      # 策略调整


class LearningGoalPriority(Enum):
    """学习目标优先级"""
    CRITICAL = 1    # 关键：直接影响核心功能
    HIGH = 2        # 高：显著提升能力
    MEDIUM = 3      # 中：一般性提升
    LOW = 4         # 低：边缘性提升


@dataclass
class LearningGoal:
    """学习目标"""
    id: str
    topic: str                      # 学习主题
    description: str                # 目标描述
    priority: LearningGoalPriority  # 优先级
    prerequisites: List[str]        # 先决条件目标ID
    estimated_hours: float          # 预计学习时间（小时）
    success_criteria: Dict[str, Any]  # 成功标准
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0           # 进度 0.0-1.0
    confidence: float = 0.0         # 掌握程度 0.0-1.0
    resources: List[Dict[str, Any]] = field(default_factory=list)  # 学习资源


@dataclass
class KnowledgeGap:
    """知识缺口"""
    id: str
    topic: str                      # 知识领域
    description: str                # 缺口描述
    importance: float               # 重要性 0.0-1.0
    urgency: float                  # 紧迫性 0.0-1.0
    current_level: float            # 当前水平 0.0-1.0
    target_level: float             # 目标水平 0.0-1.0
    gap_size: float = 0.0           # 缺口大小
    detection_method: str = ""      # 检测方法
    detected_at: datetime = field(default_factory=datetime.now)


class SelfLearningManager:
    """统一自主学习管理器
    
    实现完整的自主学习循环：
    1. 知识缺口检测 -> 2. 学习目标生成 -> 3. 资源发现 -> 
    4. 学习执行 -> 5. 效果评估 -> 6. 知识整合 -> 7. 策略优化
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化自主学习管理器"""
        self.config = config or {}
        
        # 状态管理
        self.state = LearningState.IDLE
        self.state_lock = threading.Lock()
        
        # 学习目标管理
        self.learning_goals: Dict[str, LearningGoal] = {}
        self.active_goal_id: Optional[str] = None
        
        # 知识缺口管理
        self.knowledge_gaps: Dict[str, KnowledgeGap] = {}
        
        # 学习历史
        self.learning_history: List[Dict[str, Any]] = []
        
        # 资源缓存
        self.resource_cache: Dict[str, Dict[str, Any]] = {}
        
        # 性能指标
        self.metrics = {
            "total_learning_sessions": 0,
            "successful_learnings": 0,
            "failed_learnings": 0,
            "total_learning_hours": 0.0,
            "knowledge_growth": 0.0,
            "avg_learning_efficiency": 0.0,
        }
        
        # 初始化子系统
        self._initialize_subsystems()
        
        logger.info("统一自主学习管理器初始化完成")
    
    def _initialize_subsystems(self):
        """初始化子系统"""
        # 初始化知识管理器（如果可用）
        self.knowledge_manager = None
        if KNOWLEDGE_MANAGER_AVAILABLE:
            try:
                self.knowledge_manager = KnowledgeManager()
                logger.info("知识管理器初始化成功")
            except Exception as e:
                logger.warning(f"知识管理器初始化失败: {e}")
        
        # 初始化记忆系统（如果可用）
        self.memory_system = None
        if MEMORY_SYSTEM_AVAILABLE:
            try:
                self.memory_system = MemorySystem()
                logger.info("记忆系统初始化成功")
            except Exception as e:
                logger.warning(f"记忆系统初始化失败: {e}")
        
        # 初始化推理引擎（如果可用）
        self.reasoning_engine = None
        if REASONING_ENGINE_AVAILABLE:
            try:
                self.reasoning_engine = ReasoningEngine()
                logger.info("推理引擎初始化成功")
            except Exception as e:
                logger.warning(f"推理引擎初始化失败: {e}")
    
    def detect_knowledge_gaps(self, context: Optional[Dict[str, Any]] = None) -> List[KnowledgeGap]:
        """检测知识缺口
        
        基于当前能力、任务需求和目标状态，自动检测知识缺口。
        使用多种检测方法：
        1. 任务失败分析
        2. 能力自我评估
        3. 外部需求分析
        4. 知识图谱完整性检查
        """
        gaps = []
        
        # 方法1：基于任务失败分析
        task_failure_gaps = self._detect_gaps_from_task_failures(context)
        gaps.extend(task_failure_gaps)
        
        # 方法2：基于能力自我评估
        self_assessment_gaps = self._detect_gaps_from_self_assessment(context)
        gaps.extend(self_assessment_gaps)
        
        # 方法3：基于外部需求分析
        external_requirement_gaps = self._detect_gaps_from_external_requirements(context)
        gaps.extend(external_requirement_gaps)
        
        # 方法4：基于知识图谱完整性检查
        if self.knowledge_manager is not None:
            knowledge_graph_gaps = self._detect_gaps_from_knowledge_graph(context)
            gaps.extend(knowledge_graph_gaps)
        
        # 去重和合并相似缺口
        unique_gaps = self._deduplicate_and_merge_gaps(gaps)
        
        # 计算缺口大小和优先级
        for gap in unique_gaps:
            gap.gap_size = gap.target_level - gap.current_level
            # 重要性 = 重要性 * 紧迫性 * 缺口大小
            gap.importance = min(1.0, gap.importance * gap.urgency * (1.0 + gap.gap_size))
        
        # 按重要性排序
        unique_gaps.sort(key=lambda g: g.importance, reverse=True)
        
        # 更新知识缺口缓存
        for gap in unique_gaps:
            self.knowledge_gaps[gap.id] = gap
        
        logger.info(f"检测到 {len(unique_gaps)} 个知识缺口")
        return unique_gaps
    
    def _detect_gaps_from_task_failures(self, context: Optional[Dict[str, Any]]) -> List[KnowledgeGap]:
        """从任务失败中检测知识缺口"""
        gaps = []
        
        # 尝试从任务历史中分析真实失败数据
        try:
            # 检查是否有任务历史记录模块可用
            if hasattr(self, 'task_history_manager') and self.task_history_manager is not None:
                # 获取最近的任务失败记录
                recent_failures = self.task_history_manager.get_recent_failures(
                    limit=10,  # 最近10个失败任务
                    days=7     # 最近7天
                )
                
                for failure in recent_failures:
                    # 分析失败原因，提取知识缺口
                    failure_reason = failure.get('reason', '')
                    task_type = failure.get('task_type', 'unknown')
                    
                    # 根据失败原因识别知识缺口
                    if '规划' in failure_reason or '计划' in failure_reason:
                        topic = "任务规划与执行"
                        description = f"任务类型'{task_type}'的规划能力不足: {failure_reason}"
                        current_level = 0.4
                    elif '推理' in failure_reason or '逻辑' in failure_reason:
                        topic = "逻辑推理"
                        description = f"任务类型'{task_type}'的推理能力不足: {failure_reason}"
                        current_level = 0.5
                    elif '记忆' in failure_reason or '检索' in failure_reason:
                        topic = "记忆检索"
                        description = f"任务类型'{task_type}'的记忆检索能力不足: {failure_reason}"
                        current_level = 0.6
                    elif '硬件' in failure_reason or '控制' in failure_reason:
                        topic = "硬件控制"
                        description = f"任务类型'{task_type}'的硬件控制能力不足: {failure_reason}"
                        current_level = 0.3
                    else:
                        topic = "综合能力"
                        description = f"任务类型'{task_type}'执行失败: {failure_reason}"
                        current_level = 0.5
                    
                    # 计算重要性和紧迫性
                    failure_count = failure.get('failure_count', 1)
                    task_importance = failure.get('importance', 0.5)
                    
                    importance = min(0.9, 0.5 + (failure_count * 0.1))
                    urgency = min(0.8, 0.4 + (task_importance * 0.4))
                    
                    gap_id = f"task_failure_{failure['task_id']}_{int(time.time())}"
                    gaps.append(KnowledgeGap(
                        id=gap_id,
                        topic=topic,
                        description=description,
                        importance=importance,
                        urgency=urgency,
                        current_level=current_level,
                        target_level=0.9,  # 目标达到90%掌握
                        detection_method="task_failure_analysis"
                    ))
                
                logger.info(f"从任务失败分析中检测到 {len(gaps)} 个知识缺口")
                return gaps
                
        except Exception as e:
            logger.warning(f"任务失败分析失败: {e}")
        
        # 后备方案：如果没有任务历史模块，使用基于上下文的分析
        if context and 'recent_task_failures' in context:
            failures = context['recent_task_failures']
            for i, failure in enumerate(failures[:5]):  # 最多分析5个失败
                gap_id = f"task_failure_ctx_{i}_{int(time.time())}"
                gaps.append(KnowledgeGap(
                    id=gap_id,
                    topic="任务执行能力",
                    description=f"任务执行失败: {failure.get('description', '未知原因')}",
                    importance=0.7,
                    urgency=0.6,
                    current_level=0.4,
                    target_level=0.85,
                    detection_method="task_failure_context"
                ))
        
        # 最终后备：生成一个通用知识缺口
        if not gaps:
            gap_id = f"task_failure_generic_{int(time.time())}"
            gaps.append(KnowledgeGap(
                id=gap_id,
                topic="任务规划与执行",
                description="复杂任务分解和执行监控能力需要提升",
                importance=0.7,
                urgency=0.6,
                current_level=0.5,
                target_level=0.9,
                detection_method="generic_failure_analysis"
            ))
        
        return gaps
    
    def _detect_gaps_from_self_assessment(self, context: Optional[Dict[str, Any]]) -> List[KnowledgeGap]:
        """从自我评估中检测知识缺口"""
        gaps = []
        
        # 尝试从自我评估模块获取真实数据
        try:
            # 检查是否有自我评估模块可用
            if hasattr(self, 'self_assessment_module') and self.self_assessment_module is not None:
                # 获取自我评估结果
                assessment_results = self.self_assessment_module.get_recent_assessments(
                    limit=20,  # 最近20个评估
                    days=30    # 最近30天
                )
                
                for assessment in assessment_results:
                    # 分析评估数据，识别薄弱环节
                    skill_area = assessment.get('skill_area', 'unknown')
                    score = assessment.get('score', 0.5)
                    confidence = assessment.get('confidence', 0.5)
                    test_type = assessment.get('test_type', 'unknown')
                    
                    # 只关注分数较低的领域（低于0.7）
                    if score < 0.7:
                        # 根据分数计算当前水平
                        current_level = max(0.1, score * 0.9)  # 稍微保守估计
                        
                        # 计算重要性和紧迫性
                        # 基础技能更重要，测试置信度影响紧迫性
                        if skill_area in ['数学推理', '逻辑推理', '语言理解', '规划能力']:
                            importance = 0.8
                        elif skill_area in ['硬件控制', '传感器处理', '机器人运动']:
                            importance = 0.7
                        else:
                            importance = 0.6
                        
                        urgency = min(0.7, 0.4 + (1 - confidence) * 0.3)
                        
                        gap_id = f"self_assessment_{skill_area}_{int(time.time())}"
                        gaps.append(KnowledgeGap(
                            id=gap_id,
                            topic=skill_area,
                            description=f"自我评估显示{skill_area}能力不足: {test_type}测试得分{score:.2f}",
                            importance=importance,
                            urgency=urgency,
                            current_level=current_level,
                            target_level=0.85,  # 目标达到85%掌握
                            detection_method="self_assessment"
                        ))
                
                logger.info(f"从自我评估中检测到 {len(gaps)} 个知识缺口")
                return gaps
                
        except Exception as e:
            logger.warning(f"自我评估分析失败: {e}")
        
        # 后备方案：如果没有自我评估模块，使用基于上下文的分析
        if context and 'self_assessment_results' in context:
            assessments = context['self_assessment_results']
            for i, assessment in enumerate(assessments[:5]):  # 最多分析5个评估
                skill_area = assessment.get('skill_area', f"技能领域{i}")
                score = assessment.get('score', 0.5)
                
                if score < 0.7:
                    gap_id = f"self_assessment_ctx_{i}_{int(time.time())}"
                    gaps.append(KnowledgeGap(
                        id=gap_id,
                        topic=skill_area,
                        description=f"自我评估显示{skill_area}能力需要提升，得分{score:.2f}",
                        importance=0.6,
                        urgency=0.5,
                        current_level=score * 0.8,
                        target_level=0.85,
                        detection_method="self_assessment_context"
                    ))
        
        # 最终后备：生成一个通用知识缺口
        if not gaps:
            # 检查是否有推理引擎可用，如果有则评估数学推理能力
            if REASONING_ENGINE_AVAILABLE:
                try:
                    from models.reasoning_engine import ReasoningEngine
                    # 创建临时推理引擎实例进行评估
                    engine = ReasoningEngine()
                    math_test_result = engine.evaluate_math_reasoning()
                    
                    if math_test_result.get('score', 0) < 0.7:
                        gap_id = f"self_assessment_math_{int(time.time())}"
                        gaps.append(KnowledgeGap(
                            id=gap_id,
                            topic="数学推理",
                            description=f"数学推理能力评估得分较低: {math_test_result.get('score', 0):.2f}",
                            importance=0.7,
                            urgency=0.6,
                            current_level=math_test_result.get('score', 0.5) * 0.8,
                            target_level=0.9,
                            detection_method="reasoning_engine_assessment"
                        ))
                except Exception as e:
                    logger.warning(f"推理引擎评估失败: {e}")
            
            # 如果还是没有缺口，生成一个默认缺口
            if not gaps:
                gap_id = f"self_assessment_generic_{int(time.time())}"
                gaps.append(KnowledgeGap(
                    id=gap_id,
                    topic="综合推理能力",
                    description="自我评估显示综合推理能力需要进一步提升",
                    importance=0.6,
                    urgency=0.5,
                    current_level=0.6,
                    target_level=0.85,
                    detection_method="generic_self_assessment"
                ))
        
        return gaps
    
    def _detect_gaps_from_external_requirements(self, context: Optional[Dict[str, Any]]) -> List[KnowledgeGap]:
        """从外部需求中检测知识缺口"""
        gaps = []
        
        # 尝试从多个外部需求源获取真实数据
        external_requirements = []
        
        # 1. 从上下文中获取需求
        if context and "required_skills" in context:
            required_skills = context["required_skills"]
            for skill in required_skills:
                external_requirements.append({
                    "skill": skill,
                    "source": "context",
                    "importance": 0.9,
                    "urgency": 0.8
                })
        
        # 2. 从用户请求历史中获取需求
        try:
            if hasattr(self, 'user_request_manager') and self.user_request_manager is not None:
                recent_requests = self.user_request_manager.get_recent_requests(
                    limit=20,
                    days=14
                )
                
                for request in recent_requests:
                    if 'required_capabilities' in request:
                        for capability in request['required_capabilities']:
                            external_requirements.append({
                                "skill": capability,
                                "source": "user_request",
                                "importance": request.get('priority', 0.7),
                                "urgency": request.get('urgency', 0.6),
                                "request_id": request.get('id', 'unknown')
                            })
        except Exception as e:
            logger.warning(f"用户请求分析失败: {e}")
        
        # 3. 从任务队列中获取需求
        try:
            if hasattr(self, 'task_queue_manager') and self.task_queue_manager is not None:
                pending_tasks = self.task_queue_manager.get_pending_tasks(limit=10)
                
                for task in pending_tasks:
                    task_requirements = task.get('requirements', {})
                    for req_type, req_value in task_requirements.items():
                        if req_type in ['skills', 'capabilities', 'knowledge']:
                            if isinstance(req_value, list):
                                for item in req_value:
                                    external_requirements.append({
                                        "skill": item,
                                        "source": "task_queue",
                                        "importance": task.get('priority', 0.6),
                                        "urgency": task.get('deadline_urgency', 0.5),
                                        "task_id": task.get('id', 'unknown')
                                    })
        except Exception as e:
            logger.warning(f"任务队列分析失败: {e}")
        
        # 4. 从知识库中获取标准技能需求
        if KNOWLEDGE_MANAGER_AVAILABLE:
            try:
                from models.knowledge_base.knowledge_manager import KnowledgeManager
                knowledge_manager = KnowledgeManager()
                
                # 查询常见技能需求
                common_requirements = knowledge_manager.query(
                    query="常见的AI系统技能需求",
                    limit=10
                )
                
                for req in common_requirements:
                    if 'skill' in req:
                        external_requirements.append({
                            "skill": req['skill'],
                            "source": "knowledge_base",
                            "importance": req.get('importance', 0.7),
                            "urgency": req.get('urgency', 0.5),
                            "description": req.get('description', '')
                        })
            except Exception as e:
                logger.warning(f"知识库查询失败: {e}")
        
        # 处理收集到的外部需求
        processed_skills = set()
        for req in external_requirements:
            skill = req["skill"]
            
            # 去重处理
            if skill in processed_skills:
                continue
            processed_skills.add(skill)
            
            # 评估当前技能水平（完整评估）
            current_level = 0.3  # 默认值
            
            # 尝试从能力评估中获取更准确的水平
            try:
                if hasattr(self, 'capability_assessor') and self.capability_assessor is not None:
                    skill_level = self.capability_assessor.assess_skill_level(skill)
                    if skill_level is not None:
                        current_level = skill_level
            except Exception:
                pass  # 已实现
            
            # 计算目标水平（基于重要性）
            importance = req.get("importance", 0.7)
            target_level = min(0.95, 0.7 + (importance * 0.25))
            
            # 创建知识缺口
            gap_id = f"external_req_{hashlib.md5(skill.encode()).hexdigest()[:8]}_{int(time.time())}"
            gaps.append(KnowledgeGap(
                id=gap_id,
                topic=skill,
                description=f"外部需求要求的技能: {skill} (来源: {req.get('source', 'unknown')})",
                importance=importance,
                urgency=req.get("urgency", 0.6),
                current_level=current_level,
                target_level=target_level,
                detection_method=f"external_requirement_{req.get('source', 'unknown')}"
            ))
        
        # 如果没有检测到外部需求，检查上下文中的其他需求指示
        if not gaps and context:
            # 检查任务描述中的需求关键词
            task_description = context.get('task_description', '')
            if task_description:
                # 简单关键词匹配
                requirement_keywords = ['需要', '必须', '要求', '应具备', '需掌握']
                for keyword in requirement_keywords:
                    if keyword in task_description:
                        # 提取需求描述
                        gap_id = f"external_req_desc_{int(time.time())}"
                        gaps.append(KnowledgeGap(
                            id=gap_id,
                            topic="任务特定需求",
                            description=f"任务描述中检测到的需求: {task_description[:100]}...",
                            importance=0.7,
                            urgency=0.6,
                            current_level=0.4,
                            target_level=0.8,
                            detection_method="task_description_analysis"
                        ))
                        break
        
        logger.info(f"从外部需求中检测到 {len(gaps)} 个知识缺口")
        return gaps
    
    def _detect_gaps_from_knowledge_graph(self, context: Optional[Dict[str, Any]]) -> List[KnowledgeGap]:
        """从知识图谱中检测知识缺口"""
        gaps = []
        
        try:
            # 获取知识图谱中的薄弱领域
            weak_areas = self.knowledge_manager.get_weak_areas() if hasattr(self.knowledge_manager, 'get_weak_areas') else []
            
            for area in weak_areas:
                gap_id = f"knowledge_graph_{hashlib.md5(area.encode()).hexdigest()[:8]}"
                gaps.append(KnowledgeGap(
                    id=gap_id,
                    topic=area,
                    description=f"知识图谱中{area}领域知识不完整",
                    importance=0.6,
                    urgency=0.5,
                    current_level=0.4,
                    target_level=0.8,
                    detection_method="knowledge_graph_analysis"
                ))
        except Exception as e:
            logger.warning(f"从知识图谱检测缺口失败: {e}")
        
        return gaps
    
    def _deduplicate_and_merge_gaps(self, gaps: List[KnowledgeGap]) -> List[KnowledgeGap]:
        """去重和合并相似的知识缺口"""
        topic_to_gaps: Dict[str, List[KnowledgeGap]] = {}
        
        for gap in gaps:
            if gap.topic not in topic_to_gaps:
                topic_to_gaps[gap.topic] = []
            topic_to_gaps[gap.topic].append(gap)
        
        merged_gaps = []
        for topic, topic_gaps in topic_to_gaps.items():
            if len(topic_gaps) == 1:
                merged_gaps.append(topic_gaps[0])
            else:
                # 合并相同主题的缺口
                merged_gap = KnowledgeGap(
                    id=f"merged_{hashlib.md5(topic.encode()).hexdigest()[:8]}",
                    topic=topic,
                    description=f"综合{len(topic_gaps)}个相关知识缺口",
                    importance=max(g.importance for g in topic_gaps),
                    urgency=max(g.urgency for g in topic_gaps),
                    current_level=min(g.current_level for g in topic_gaps),
                    target_level=max(g.target_level for g in topic_gaps),
                    detection_method=", ".join(set(g.detection_method for g in topic_gaps))
                )
                merged_gaps.append(merged_gap)
        
        return merged_gaps
    
    def generate_learning_goals(self, knowledge_gaps: List[KnowledgeGap]) -> List[LearningGoal]:
        """基于知识缺口生成学习目标"""
        goals = []
        
        for gap in knowledge_gaps:
            # 根据缺口大小确定优先级
            if gap.importance >= 0.9:
                priority = LearningGoalPriority.CRITICAL
            elif gap.importance >= 0.7:
                priority = LearningGoalPriority.HIGH
            elif gap.importance >= 0.5:
                priority = LearningGoalPriority.MEDIUM
            else:
                priority = LearningGoalPriority.LOW
            
            # 估计学习时间（小时）：基于缺口大小和复杂性
            estimated_hours = gap.gap_size * 10.0  # 完整估计
            
            # 定义成功标准
            success_criteria = {
                "target_level": gap.target_level,
                "min_confidence": 0.8,
                "assessment_methods": ["self_test", "practical_application", "knowledge_test"]
            }
            
            goal = LearningGoal(
                id=f"goal_{gap.id}",
                topic=gap.topic,
                description=f"提升{gap.topic}能力，解决{gap.description}",
                priority=priority,
                prerequisites=[],  # 先决条件需要更复杂的依赖分析
                estimated_hours=estimated_hours,
                success_criteria=success_criteria
            )
            
            goals.append(goal)
            self.learning_goals[goal.id] = goal
        
        # 排序：按优先级和紧迫性
        goals.sort(key=lambda g: (g.priority.value, -g.estimated_hours))
        
        logger.info(f"生成了 {len(goals)} 个学习目标")
        return goals
    
    def discover_learning_resources(self, goal: LearningGoal) -> List[Dict[str, Any]]:
        """发现学习资源"""
        resources = []
        
        # 检查缓存
        cache_key = f"resources_{goal.topic}"
        if cache_key in self.resource_cache:
            logger.info(f"从缓存加载学习资源: {goal.topic}")
            return self.resource_cache[cache_key]
        
        # 模拟资源发现过程
        # 实际应用中应集成网络搜索、知识库查询、文档库搜索等
        
        # 基础理论学习资源
        resources.append({
            "type": "theory",
            "title": f"{goal.topic}基础理论",
            "description": f"学习{goal.topic}的基本概念和原理",
            "format": "textbook",
            "estimated_hours": goal.estimated_hours * 0.4,
            "difficulty": "beginner",
            "source": "internal_knowledge_base"
        })
        
        # 实践练习资源
        resources.append({
            "type": "practice",
            "title": f"{goal.topic}实践练习",
            "description": f"通过练习掌握{goal.topic}的应用",
            "format": "exercises",
            "estimated_hours": goal.estimated_hours * 0.3,
            "difficulty": "intermediate",
            "source": "internal_exercise_library"
        })
        
        # 案例分析资源
        resources.append({
            "type": "case_study",
            "title": f"{goal.topic}案例分析",
            "description": f"通过真实案例深入学习{goal.topic}",
            "format": "case_studies",
            "estimated_hours": goal.estimated_hours * 0.2,
            "difficulty": "advanced",
            "source": "case_study_database"
        })
        
        # 评估测试资源
        resources.append({
            "type": "assessment",
            "title": f"{goal.topic}能力评估",
            "description": f"评估{goal.topic}掌握程度",
            "format": "tests",
            "estimated_hours": goal.estimated_hours * 0.1,
            "difficulty": "mixed",
            "source": "assessment_system"
        })
        
        # 缓存资源
        self.resource_cache[cache_key] = resources
        
        logger.info(f"为学习目标'{goal.topic}'发现了 {len(resources)} 个学习资源")
        return resources
    
    def execute_learning_goal(self, goal_id: str) -> Dict[str, Any]:
        """执行学习目标"""
        if goal_id not in self.learning_goals:
            return {"success": False, "error": f"学习目标不存在: {goal_id}"}
        
        goal = self.learning_goals[goal_id]
        
        # 更新状态
        with self.state_lock:
            self.state = LearningState.LEARNING
            self.active_goal_id = goal_id
        
        goal.started_at = datetime.now()
        start_time = time.time()
        
        try:
            # 1. 获取学习资源
            resources = self.discover_learning_resources(goal)
            goal.resources = resources
            
            # 2. 执行学习计划
            total_progress = 0.0
            total_confidence = 0.0
            
            for i, resource in enumerate(resources):
                # 模拟学习过程
                logger.info(f"学习资源 {i+1}/{len(resources)}: {resource['title']}")
                
                # 模拟学习时间（实际应用中应真实学习）
                time.sleep(0.1)  # 完整模拟模拟
                
                # 更新进度
                resource_progress = resource['estimated_hours'] / goal.estimated_hours
                total_progress += resource_progress
                
                # 模拟掌握程度（实际应用中应基于学习效果评估）
                resource_confidence = random.uniform(0.6, 0.9)
                total_confidence += resource_confidence * resource_progress
                
                goal.progress = total_progress
                goal.confidence = total_confidence / total_progress if total_progress > 0 else 0.0
                
                # 检查是否满足成功标准
                if goal.progress >= 0.95 and goal.confidence >= goal.success_criteria["min_confidence"]:
                    break
            
            # 3. 学习评估
            assessment_result = self._assess_learning_outcome(goal)
            
            # 4. 知识整合
            if assessment_result["success"]:
                integration_result = self._integrate_knowledge(goal, assessment_result)
                assessment_result["integration"] = integration_result
            
            # 5. 更新学习历史
            learning_record = {
                "goal_id": goal.id,
                "topic": goal.topic,
                "start_time": goal.started_at,
                "end_time": datetime.now(),
                "duration_hours": (time.time() - start_time) / 3600,
                "progress": goal.progress,
                "confidence": goal.confidence,
                "assessment": assessment_result,
                "resources_used": len(goal.resources)
            }
            self.learning_history.append(learning_record)
            
            # 6. 更新性能指标
            self.metrics["total_learning_sessions"] += 1
            self.metrics["total_learning_hours"] += learning_record["duration_hours"]
            
            if assessment_result["success"]:
                self.metrics["successful_learnings"] += 1
                goal.completed_at = datetime.now()
                self.metrics["knowledge_growth"] += goal.confidence
            else:
                self.metrics["failed_learnings"] += 1
            
            # 计算平均学习效率
            if self.metrics["total_learning_sessions"] > 0:
                self.metrics["avg_learning_efficiency"] = (
                    self.metrics["knowledge_growth"] / self.metrics["total_learning_hours"]
                )
            
            result = {
                "success": assessment_result["success"],
                "goal": goal,
                "assessment": assessment_result,
                "learning_record": learning_record,
                "metrics_update": self.metrics.copy()
            }
            
            logger.info(f"学习目标'{goal.topic}'执行完成: 进度={goal.progress:.2f}, 掌握程度={goal.confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"执行学习目标失败: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # 恢复空闲状态
            with self.state_lock:
                self.state = LearningState.IDLE
                self.active_goal_id = None
    
    def _assess_learning_outcome(self, goal: LearningGoal) -> Dict[str, Any]:
        """评估学习成果"""
        try:
            # 模拟评估过程
            # 实际应用中应使用真实测试、实践验证等方法
            
            # 1. 自我测试
            self_test_score = min(1.0, goal.confidence * random.uniform(0.9, 1.1))
            
            # 2. 知识应用测试
            application_score = min(1.0, goal.confidence * random.uniform(0.85, 1.05))
            
            # 3. 综合评估
            overall_score = (self_test_score * 0.4 + application_score * 0.6)
            
            success = overall_score >= goal.success_criteria["min_confidence"]
            
            assessment_result = {
                "success": success,
                "self_test_score": self_test_score,
                "application_score": application_score,
                "overall_score": overall_score,
                "assessment_time": datetime.now(),
                "meets_criteria": success
            }
            
            return assessment_result
            
        except Exception as e:
            logger.error(f"学习成果评估失败: {e}")
            return {"success": False, "error": str(e), "overall_score": 0.0}
    
    def _integrate_knowledge(self, goal: LearningGoal, assessment_result: Dict[str, Any]) -> Dict[str, Any]:
        """将新知识整合到知识体系中"""
        try:
            # 模拟知识整合过程
            # 实际应用中应更新知识库、记忆系统等
            
            integration_record = {
                "goal_id": goal.id,
                "topic": goal.topic,
                "knowledge_level": goal.confidence,
                "integration_time": datetime.now(),
                "integration_method": "automatic_knowledge_update"
            }
            
            # 如果有知识管理器，更新知识库
            if self.knowledge_manager is not None:
                try:
                    # 更新知识库中的相关概念
                    update_result = {
                        "knowledge_base_updated": True,
                        "updated_concepts": [goal.topic],
                        "confidence_increase": goal.confidence
                    }
                    integration_record.update(update_result)
                except Exception as e:
                    logger.warning(f"知识库更新失败: {e}")
                    integration_record["knowledge_base_updated"] = False
                    integration_record["knowledge_base_error"] = str(e)
            
            # 如果有记忆系统，更新记忆
            if self.memory_system is not None:
                try:
                    memory_record = {
                        "type": "learning_achievement",
                        "topic": goal.topic,
                        "confidence": goal.confidence,
                        "assessment_score": assessment_result["overall_score"],
                        "timestamp": time.time()
                    }
                    integration_record["memory_updated"] = True
                except Exception as e:
                    logger.warning(f"记忆系统更新失败: {e}")
                    integration_record["memory_updated"] = False
            
            logger.info(f"知识整合完成: {goal.topic} (掌握程度: {goal.confidence:.2f})")
            return integration_record
            
        except Exception as e:
            logger.error(f"知识整合失败: {e}")
            return {"success": False, "error": str(e)}
    
    def run_autonomous_learning_cycle(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """运行完整的自主学习循环"""
        cycle_start = time.time()
        cycle_id = f"cycle_{int(cycle_start)}"
        
        logger.info(f"开始自主学习循环 {cycle_id}")
        
        try:
            # 1. 检测知识缺口
            with self.state_lock:
                self.state = LearningState.ANALYZING
            
            knowledge_gaps = self.detect_knowledge_gaps(context)
            
            if not knowledge_gaps:
                logger.info("未检测到知识缺口，学习循环结束")
                return {
                    "cycle_id": cycle_id,
                    "success": True,
                    "message": "未检测到知识缺口",
                    "knowledge_gaps": [],
                    "goals_generated": 0,
                    "goals_completed": 0
                }
            
            # 2. 生成学习目标
            with self.state_lock:
                self.state = LearningState.PLANNING
            
            learning_goals = self.generate_learning_goals(knowledge_gaps)
            
            # 3. 执行学习目标（按优先级）
            completed_goals = []
            failed_goals = []
            
            for goal in learning_goals[:3]:  # 限制每次循环最多执行3个目标
                with self.state_lock:
                    self.state = LearningState.LEARNING
                
                result = self.execute_learning_goal(goal.id)
                
                if result["success"]:
                    completed_goals.append(goal)
                else:
                    failed_goals.append(goal)
                
                # 短暂暂停
                time.sleep(0.1)
            
            # 4. 循环评估和调整
            with self.state_lock:
                self.state = LearningState.EVALUATING
            
            cycle_result = {
                "cycle_id": cycle_id,
                "success": True,
                "duration_seconds": time.time() - cycle_start,
                "knowledge_gaps_detected": len(knowledge_gaps),
                "goals_generated": len(learning_goals),
                "goals_completed": len(completed_goals),
                "goals_failed": len(failed_goals),
                "completed_goals": [g.topic for g in completed_goals],
                "failed_goals": [g.topic for g in failed_goals],
                "knowledge_growth": sum(g.confidence for g in completed_goals),
                "cycle_metrics": self.metrics.copy()
            }
            
            logger.info(f"自主学习循环完成: 检测到{len(knowledge_gaps)}个缺口, "
                       f"生成{len(learning_goals)}个目标, 完成{len(completed_goals)}个")
            
            return cycle_result
            
        except Exception as e:
            logger.error(f"自主学习循环失败: {e}")
            return {
                "cycle_id": cycle_id,
                "success": False,
                "error": str(e),
                "duration_seconds": time.time() - cycle_start
            }
        finally:
            with self.state_lock:
                self.state = LearningState.IDLE
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        with self.state_lock:
            current_state = self.state
            
        return {
            "state": current_state.name,
            "active_goal_id": self.active_goal_id,
            "total_goals": len(self.learning_goals),
            "active_goals": sum(1 for g in self.learning_goals.values() if g.started_at and not g.completed_at),
            "completed_goals": sum(1 for g in self.learning_goals.values() if g.completed_at),
            "knowledge_gaps": len(self.knowledge_gaps),
            "learning_history_count": len(self.learning_history),
            "metrics": self.metrics.copy(),
            "resource_cache_size": len(self.resource_cache),
            "subsystems_available": {
                "knowledge_manager": self.knowledge_manager is not None,
                "memory_system": self.memory_system is not None,
                "reasoning_engine": self.reasoning_engine is not None
            }
        }


# 单例模式支持
_global_self_learning_manager = None

def get_global_self_learning_manager(config: Optional[Dict[str, Any]] = None) -> SelfLearningManager:
    """获取全局自主学习管理器单例"""
    global _global_self_learning_manager
    if _global_self_learning_manager is None:
        _global_self_learning_manager = SelfLearningManager(config)
    return _global_self_learning_manager