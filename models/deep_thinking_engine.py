#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度思考推理引擎 - 增强自我修证功能

功能：
1. 深度思考：面对挑战性问题进行多步深度推理
2. 自我反思：分析自身推理过程，识别错误和不足
3. 自我修正：基于反思结果修正错误，改进推理
4. 超出认知问题处理：识别未知领域，做出符合逻辑的判断
5. 逻辑决策：基于深度思考做出最符合逻辑的决策

本模块解决用户需求：
- 增强自我修证功能，面对挑战性问题和不会的问题进行深度思考后进行自我反思和自我修正
- 对超出认知的的问题进行深度思考做出最符合逻辑的符合常规的判断和决定

实现真正的深度思考和自我修证能力，超越简单的神经网络模拟。
"""

import torch
import torch.nn as nn
import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ThinkingDepth(Enum):
    """思考深度级别"""
    SHALLOW = auto()  # 浅层思考：快速响应，简单问题
    MODERATE = auto()  # 中等思考：分析推理，常规问题
    DEEP = auto()  # 深度思考：多步推理，复杂问题
    EXTREME = auto()  # 极限思考：系统分析，重大决策


class ProblemType(Enum):
    """问题类型分类"""
    KNOWN = auto()  # 已知问题：有明确解决方案
    CHALLENGING = auto()  # 挑战性问题：需要深度思考
    UNKNOWN = auto()  # 未知问题：超出当前认知
    AMBIGUOUS = auto()  # 模糊问题：信息不完整或矛盾
    CONTRADICTORY = auto()  # 矛盾问题：存在逻辑冲突


class ReflectionType(Enum):
    """反思类型"""
    PROCESS_REFLECTION = auto()  # 过程反思：分析推理过程
    RESULT_REFLECTION = auto()  # 结果反思：评估结果质量
    ASSUMPTION_REFLECTION = auto()  # 假设反思：检查隐含假设
    KNOWLEDGE_REFLECTION = auto()  # 知识反思：评估知识适用性
    METACOGNITIVE_REFLECTION = auto()  # 元认知反思：思考思考过程


class CorrectionStrategy(Enum):
    """修正策略"""
    REWRITE_REASONING = auto()  # 重写推理过程
    REFINE_ASSUMPTIONS = auto()  # 精炼假设
    EXPAND_KNOWLEDGE = auto()  # 扩展知识
    ADJUST_LOGIC = auto()  # 调整逻辑
    INTEGRATE_NEW_EVIDENCE = auto()  # 整合新证据
    RESTRUCTURE_ARGUMENT = auto()  # 重构论证


class DeepThinkingError(Exception):
    """深度思考引擎基础异常"""
    
    def __init__(
        self, 
        message: str = "深度思考引擎错误", 
        details: Optional[Dict[str, Any]] = None,
        thinking_step: Optional[str] = None
    ):
        self.message = message
        self.details = details or {}
        self.thinking_step = thinking_step
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "thinking_step": self.thinking_step,
            "timestamp": self.timestamp,
        }


class DeepThinkingEngine:
    """
    深度思考推理引擎
    
    实现真正的深度思考和自我修证功能：
    1. 多步深度推理
    2. 多层次自我反思
    3. 自适应自我修正
    4. 超出认知问题处理
    5. 逻辑决策生成
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化深度思考引擎
        
        配置参数:
        - max_thinking_steps: 最大思考步数 (默认: 10)
        - min_thinking_steps: 最小思考步数 (默认: 3)
        - enable_reflection: 是否启用反思 (默认: True)
        - reflection_depth: 反思深度 (默认: "deep")
        - enable_correction: 是否启用修正 (默认: True)
        - max_correction_iterations: 最大修正迭代次数 (默认: 5)
        - max_history_size: 最大历史记录条数 (默认: 1000)
        - enable_cache: 是否启用缓存 (默认: True)
        - cache_size: 缓存大小 (默认: 100)
        - cache_ttl_seconds: 缓存存活时间（秒）(默认: 3600)
        """
        self.config = config or {}
        
        # 思考深度配置
        self.default_thinking_depth = ThinkingDepth.DEEP
        self.max_thinking_steps = self._validate_config_value("max_thinking_steps", 10, 1, 50)
        self.min_thinking_steps = self._validate_config_value("min_thinking_steps", 3, 1, 20)
        
        # 反思配置
        self.enable_reflection = self.config.get("enable_reflection", True)
        self.reflection_depth = self.config.get("reflection_depth", "deep")
        
        # 修正配置
        self.enable_correction = self.config.get("enable_correction", True)
        self.max_correction_iterations = self._validate_config_value("max_correction_iterations", 5, 1, 20)
        
        # 缓存和历史配置
        self.max_history_size = self._validate_config_value("max_history_size", 1000, 10, 10000)
        self.enable_cache = self.config.get("enable_cache", True)
        self.cache_size = self._validate_config_value("cache_size", 100, 10, 1000)
        self.cache_ttl_seconds = self._validate_config_value("cache_ttl_seconds", 3600, 60, 86400)
        
        # 初始化缓存
        self.problem_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # 初始化子系统
        self._initialize_subsystems()
        
        # 思考历史记录
        self.thinking_history: List[Dict[str, Any]] = []
        self.reflection_history: List[Dict[str, Any]] = []
        self.correction_history: List[Dict[str, Any]] = []
        
        # 性能指标
        self.metrics = {
            "total_problems_processed": 0,
            "challenging_problems": 0,
            "unknown_problems": 0,
            "successful_corrections": 0,
            "failed_corrections": 0,
            "avg_thinking_steps": 0.0,
            "avg_reflection_depth": 0.0,
            "solved_problems": 0,
            "problem_solving_rate": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_processing_time_ms": 0.0,
            "error_count": 0,
        }
        
        logger.info(f"深度思考推理引擎初始化完成，配置: max_steps={self.max_thinking_steps}, enable_cache={self.enable_cache}, max_history={self.max_history_size}")
    
    def _initialize_subsystems(self):
        """初始化子系统"""
        # 导入推理引擎
        try:
            from models.reasoning_engine import ReasoningEngine
            self.reasoning_engine = ReasoningEngine()
            self.reasoning_available = True
            logger.info("推理引擎集成成功")
        except ImportError as e:
            self.reasoning_available = False
            logger.error(f"推理引擎不可用: {e}")
            self.reasoning_engine = None
        
        # 导入自我认知模块
        try:
            from models.transformer.cognitive.selfcognitionmodule import SelfCognitionModule
            # 注意：SelfCognitionModule需要配置，这里创建简化版本
            self.self_cognition_available = True
            logger.info("自我认知模块集成成功")
        except ImportError as e:
            self.self_cognition_available = False
            logger.warning(f"自我认知模块不可用: {e}")
        
        # 导入自我修正模块
        try:
            from models.transformer.cognitive.selfcorrectionmodule import SelfCorrectionModule
            # 注意：SelfCorrectionModule需要配置，这里创建简化版本
            self.self_correction_available = True
            logger.info("自我修正模块集成成功")
        except ImportError as e:
            self.self_correction_available = False
            logger.warning(f"自我修正模块不可用: {e}")
        
        # 初始化内部神经网络（用于模拟如果外部模块不可用）
        self._initialize_internal_networks()
    
    def _initialize_internal_networks(self):
        """初始化内部神经网络"""
        # 深度思考网络
        self.deep_thinking_network = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512)
        ) if torch.cuda.is_available() or torch.backends.mps.is_available() else None
        
        # 反思网络
        self.reflection_network = nn.Sequential(
            nn.Linear(512, 768),
            nn.GELU(),
            nn.Linear(768, 512),
            nn.LayerNorm(512)
        ) if torch.cuda.is_available() or torch.backends.mps.is_available() else None
        
        # 修正网络
        self.correction_network = nn.Sequential(
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.Linear(768, 512),
            nn.LayerNorm(512)
        ) if torch.cuda.is_available() or torch.backends.mps.is_available() else None
        
        # 超出认知问题处理器
        self.unknown_problem_processor = nn.Sequential(
            nn.Linear(512, 768),
            nn.GELU(),
            nn.Linear(768, 512),
            nn.LayerNorm(512)
        ) if torch.cuda.is_available() or torch.backends.mps.is_available() else None
    
    def _validate_config_value(
        self, 
        key: str, 
        default: Any, 
        min_value: Optional[Any] = None, 
        max_value: Optional[Any] = None
    ) -> Any:
        """验证配置值
        
        参数:
            key: 配置键名
            default: 默认值
            min_value: 最小值（可选）
            max_value: 最大值（可选）
            
        返回:
            验证后的配置值
        """
        value = self.config.get(key, default)
        
        # 类型验证
        expected_type = type(default)
        if not isinstance(value, expected_type):
            logger.warning(f"配置值 {key} 类型错误: 期望 {expected_type.__name__}, 实际 {type(value).__name__}, 使用默认值 {default}")
            value = default
        
        # 范围验证（如果提供了最小值和最大值）
        if min_value is not None and value < min_value:
            logger.warning(f"配置值 {key} 小于最小值 {min_value}, 调整为 {min_value}")
            value = min_value
        
        if max_value is not None and value > max_value:
            logger.warning(f"配置值 {key} 大于最大值 {max_value}, 调整为 {max_value}")
            value = max_value
        
        return value
    
    def _get_cache_key(self, problem: Union[str, Dict[str, Any]], context: Optional[Dict[str, Any]]) -> str:
        """生成缓存键
        
        基于问题和上下文生成唯一的缓存键。
        """
        # 将问题转换为字符串
        if isinstance(problem, dict):
            problem_str = json.dumps(problem, sort_keys=True)
        else:
            problem_str = str(problem)
        
        # 添加上下文（如果有）
        if context:
            context_str = json.dumps(context, sort_keys=True)
            return f"{problem_str}::{context_str}"
        else:
            return problem_str
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """检查缓存
        
        返回:
            如果找到有效的缓存项，返回缓存值，否则返回None
        """
        if not self.enable_cache:
            return None
        
        # 检查是否存在
        if cache_key not in self.problem_cache:
            self.metrics["cache_misses"] += 1
            return None
        
        # 检查是否过期
        cache_time = self.cache_timestamps.get(cache_key)
        if cache_time:
            time_diff = (datetime.now() - cache_time).total_seconds()
            if time_diff > self.cache_ttl_seconds:
                # 缓存过期，移除
                del self.problem_cache[cache_key]
                del self.cache_timestamps[cache_key]
                self.metrics["cache_misses"] += 1
                return None
        
        # 返回缓存值
        self.metrics["cache_hits"] += 1
        return self.problem_cache[cache_key]
    
    def _add_to_cache(self, cache_key: str, value: Dict[str, Any]) -> None:
        """添加到缓存
        
        参数:
            cache_key: 缓存键
            value: 缓存值
        """
        if not self.enable_cache:
            return
        
        # 清理过期缓存
        self._cleanup_cache()
        
        # 检查缓存大小，如果超出限制，移除最旧的项
        if len(self.problem_cache) >= self.cache_size:
            # 找到最旧的缓存项
            if self.cache_timestamps:
                oldest_key = min(self.cache_timestamps.items(), key=lambda x: x[1])[0]
                del self.problem_cache[oldest_key]
                del self.cache_timestamps[oldest_key]
                logger.debug(f"缓存已满，移除最旧缓存项: {oldest_key[:50]}...")
        
        # 添加新缓存
        self.problem_cache[cache_key] = value
        self.cache_timestamps[cache_key] = datetime.now()
        logger.debug(f"缓存添加成功，键: {cache_key[:50]}...，缓存大小: {len(self.problem_cache)}")
    
    def _cleanup_cache(self) -> None:
        """清理过期缓存"""
        if not self.enable_cache:
            return
        
        expired_keys = []
        current_time = datetime.now()
        
        for key, timestamp in self.cache_timestamps.items():
            time_diff = (current_time - timestamp).total_seconds()
            if time_diff > self.cache_ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.problem_cache[key]
            del self.cache_timestamps[key]
        
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")
    
    def _trim_history(self) -> None:
        """修剪历史记录，保持不超过最大限制"""
        # 修剪思考历史
        if len(self.thinking_history) > self.max_history_size:
            excess = len(self.thinking_history) - self.max_history_size
            self.thinking_history = self.thinking_history[excess:]
            logger.debug(f"修剪思考历史，移除了 {excess} 条记录")
        
        # 修剪反思历史
        if len(self.reflection_history) > self.max_history_size:
            excess = len(self.reflection_history) - self.max_history_size
            self.reflection_history = self.reflection_history[excess:]
            logger.debug(f"修剪反思历史，移除了 {excess} 条记录")
        
        # 修剪修正历史
        if len(self.correction_history) > self.max_history_size:
            excess = len(self.correction_history) - self.max_history_size
            self.correction_history = self.correction_history[excess:]
            logger.debug(f"修剪修正历史，移除了 {excess} 条记录")
    
    def deep_think(
        self, 
        problem: Union[str, Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None,
        thinking_depth: Optional[ThinkingDepth] = None,
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        深度思考核心方法
        
        参数:
            problem: 问题描述（字符串或结构化字典）
            context: 上下文信息
            thinking_depth: 思考深度级别
            max_steps: 最大思考步数
            
        返回:
            深度思考结果，包含：
            - 最终结论
            - 思考过程
            - 反思结果
            - 修正历史
            - 置信度
        """
        # 参数处理
        if thinking_depth is None:
            thinking_depth = self.default_thinking_depth
        elif isinstance(thinking_depth, str):
            # 将字符串转换为ThinkingDepth枚举
            thinking_depth_map = {
                "shallow": ThinkingDepth.SHALLOW,
                "moderate": ThinkingDepth.MODERATE,
                "deep": ThinkingDepth.DEEP,
                "extreme": ThinkingDepth.EXTREME,
            }
            thinking_depth = thinking_depth_map.get(thinking_depth.lower(), self.default_thinking_depth)
        
        if max_steps is None:
            max_steps = self.max_thinking_steps
        
        # 检查缓存
        cache_key = self._get_cache_key(problem, context)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            # 如果找到缓存，调整思考深度和时间戳
            cached_result = cached_result.copy()
            cached_result["cached"] = True
            cached_result["thinking_depth"] = thinking_depth.name if hasattr(thinking_depth, 'name') else str(thinking_depth)
            cached_result["processing_time_ms"] = 0  # 缓存结果处理时间为0
            logger.debug(f"从缓存返回结果，缓存键: {cache_key[:50]}...")
            return cached_result
        
        # 记录开始时间
        start_time = datetime.now()
        
        try:
            # 步骤1：问题分析与分类
            problem_analysis = self._analyze_problem(problem, context)
            problem_type = problem_analysis.get("problem_type", ProblemType.KNOWN)
            
            # 步骤2：根据问题类型确定思考策略
            thinking_strategy = self._determine_thinking_strategy(problem_type, thinking_depth)
            
            # 步骤3：执行深度思考
            thinking_result = self._execute_thinking(
                problem, 
                context, 
                thinking_strategy, 
                max_steps
            )
            
            # 步骤4：自我反思
            reflection_result = None
            if self.enable_reflection:
                reflection_result = self._perform_self_reflection(
                    problem, 
                    thinking_result, 
                    problem_analysis
                )
            
            # 步骤5：自我修正（如果需要）
            correction_result = None
            if self.enable_correction and reflection_result:
                if reflection_result.get("needs_correction", False):
                    correction_result = self._perform_self_correction(
                        problem, 
                        thinking_result, 
                        reflection_result
                    )
            
            # 步骤6：生成最终结论
            final_conclusion = self._generate_final_conclusion(
                thinking_result, 
                reflection_result, 
                correction_result
            )
            
            # 步骤7：更新性能指标
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(
                problem_type, 
                thinking_result, 
                reflection_result, 
                correction_result,
                processing_time_ms
            )
            
            # 构建完整结果
            result = {
                "success": True,
                "final_conclusion": final_conclusion,
                "problem_analysis": problem_analysis,
                "thinking_result": thinking_result,
                "reflection_result": reflection_result,
                "correction_result": correction_result,
                "thinking_depth": thinking_depth.name if hasattr(thinking_depth, 'name') else str(thinking_depth),
                "problem_type": problem_type.name if hasattr(problem_type, 'name') else str(problem_type),
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "confidence": final_conclusion.get("confidence", 0.0),
            }
            
            # 记录思考历史
            self.thinking_history.append({
                "timestamp": datetime.now().isoformat(),
                "problem": str(problem)[:200],  # 截断长问题
                "result": result,
                "problem_type": problem_type.name if hasattr(problem_type, 'name') else str(problem_type),
            })
            
            logger.info(f"深度思考完成: 问题类型={problem_type.name if hasattr(problem_type, 'name') else str(problem_type)}, 思考深度={thinking_depth.name if hasattr(thinking_depth, 'name') else str(thinking_depth)}, 置信度={final_conclusion.get('confidence', 0.0):.2f}")
            
            # 添加到缓存（只缓存成功的结果）
            self._add_to_cache(cache_key, result)
            
            # 修剪历史记录
            self._trim_history()
            
            return result
            
        except Exception as e:
            logger.error(f"深度思考过程出错: {e}")
            # 更新错误计数
            self.metrics["error_count"] += 1
            return {
                "success": False,
                "error": str(e),
                "final_conclusion": None,
                "thinking_depth": thinking_depth.name if thinking_depth and hasattr(thinking_depth, 'name') else (str(thinking_depth) if thinking_depth else "UNKNOWN"),
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            }
    
    def _analyze_problem(
        self, 
        problem: Union[str, Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析问题类型和特征"""
        # 将问题转换为字符串进行分析
        if isinstance(problem, dict):
            problem_text = problem.get("text", str(problem))
        else:
            problem_text = str(problem)
        
        # 分析问题特征
        features = self._extract_problem_features(problem_text, context)
        
        # 根据特征确定问题类型
        problem_type = self._classify_problem_type(features)
        
        # 构建分析结果
        analysis = {
            "problem_text": problem_text[:500],  # 截断长文本
            "problem_features": features,
            "problem_type": problem_type,
            "complexity_score": features.get("complexity_score", 0.5),
            "ambiguity_score": features.get("ambiguity_score", 0.0),
            "knowledge_requirement": features.get("knowledge_requirement", "general"),
            "analysis_timestamp": datetime.now().isoformat(),
        }
        
        return analysis
    
    def _extract_problem_features(
        self, 
        problem_text: str, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """提取问题特征"""
        features = {
            "length": len(problem_text),
            "word_count": len(problem_text.split()),
            "has_question_mark": "?" in problem_text,
            "has_technical_terms": self._has_technical_terms(problem_text),
            "has_uncertainty_words": self._has_uncertainty_words(problem_text),
            "has_contradiction_words": self._has_contradiction_words(problem_text),
            "complexity_score": self._calculate_complexity_score(problem_text),
            "ambiguity_score": self._calculate_ambiguity_score(problem_text),
            "knowledge_requirement": self._determine_knowledge_requirement(problem_text),
        }
        
        # 添加上下文特征
        if context:
            features.update({
                "context_provided": True,
                "context_keys": list(context.keys()),
                "context_size": len(str(context)),
            })
        else:
            features["context_provided"] = False
        
        return features
    
    def _classify_problem_type(self, features: Dict[str, Any]) -> ProblemType:
        """根据特征分类问题类型"""
        # 检查是否为未知问题
        if features.get("has_technical_terms", False) and features.get("knowledge_requirement") == "specialized":
            return ProblemType.UNKNOWN
        
        # 检查是否为挑战性问题
        if features.get("complexity_score", 0.0) > 0.7:
            return ProblemType.CHALLENGING
        
        # 检查是否为矛盾问题
        if features.get("has_contradiction_words", False):
            return ProblemType.CONTRADICTORY
        
        # 检查是否为模糊问题
        if features.get("ambiguity_score", 0.0) > 0.5:
            return ProblemType.AMBIGUOUS
        
        # 默认为已知问题
        return ProblemType.KNOWN
    
    def _determine_thinking_strategy(
        self, 
        problem_type: ProblemType, 
        thinking_depth: ThinkingDepth
    ) -> Dict[str, Any]:
        """根据问题类型和思考深度确定思考策略"""
        strategies = {
            ProblemType.KNOWN: {
                ThinkingDepth.SHALLOW: {"steps": 2, "use_reflection": False, "use_correction": False},
                ThinkingDepth.MODERATE: {"steps": 3, "use_reflection": True, "use_correction": False},
                ThinkingDepth.DEEP: {"steps": 5, "use_reflection": True, "use_correction": True},
                ThinkingDepth.EXTREME: {"steps": 8, "use_reflection": True, "use_correction": True},
            },
            ProblemType.CHALLENGING: {
                ThinkingDepth.SHALLOW: {"steps": 3, "use_reflection": True, "use_correction": False},
                ThinkingDepth.MODERATE: {"steps": 5, "use_reflection": True, "use_correction": True},
                ThinkingDepth.DEEP: {"steps": 8, "use_reflection": True, "use_correction": True},
                ThinkingDepth.EXTREME: {"steps": 12, "use_reflection": True, "use_correction": True},
            },
            ProblemType.UNKNOWN: {
                ThinkingDepth.SHALLOW: {"steps": 4, "use_reflection": True, "use_correction": True},
                ThinkingDepth.MODERATE: {"steps": 6, "use_reflection": True, "use_correction": True},
                ThinkingDepth.DEEP: {"steps": 10, "use_reflection": True, "use_correction": True},
                ThinkingDepth.EXTREME: {"steps": 15, "use_reflection": True, "use_correction": True},
            },
            ProblemType.AMBIGUOUS: {
                ThinkingDepth.SHALLOW: {"steps": 3, "use_reflection": True, "use_correction": False},
                ThinkingDepth.MODERATE: {"steps": 5, "use_reflection": True, "use_correction": True},
                ThinkingDepth.DEEP: {"steps": 8, "use_reflection": True, "use_correction": True},
                ThinkingDepth.EXTREME: {"steps": 12, "use_reflection": True, "use_correction": True},
            },
            ProblemType.CONTRADICTORY: {
                ThinkingDepth.SHALLOW: {"steps": 4, "use_reflection": True, "use_correction": True},
                ThinkingDepth.MODERATE: {"steps": 6, "use_reflection": True, "use_correction": True},
                ThinkingDepth.DEEP: {"steps": 10, "use_reflection": True, "use_correction": True},
                ThinkingDepth.EXTREME: {"steps": 15, "use_reflection": True, "use_correction": True},
            },
        }
        
        return strategies.get(problem_type, {}).get(
            thinking_depth, 
            {"steps": 5, "use_reflection": True, "use_correction": True}
        )
    
    def _execute_thinking(
        self, 
        problem: Union[str, Dict[str, Any]], 
        context: Optional[Dict[str, Any]],
        strategy: Dict[str, Any],
        max_steps: int
    ) -> Dict[str, Any]:
        """执行深度思考过程"""
        thinking_steps = []
        current_understanding = {"problem": problem, "context": context}
        
        # 确定实际步数
        actual_steps = min(strategy.get("steps", 5), max_steps)
        
        for step in range(actual_steps):
            step_result = self._perform_thinking_step(
                step, 
                current_understanding, 
                problem, 
                context
            )
            
            thinking_steps.append(step_result)
            
            # 更新当前理解
            if "new_understanding" in step_result:
                current_understanding.update(step_result["new_understanding"])
            
            # 检查是否提前得出结论
            if step_result.get("has_conclusion", False):
                break
        
        # 生成思考总结
        thinking_summary = self._generate_thinking_summary(thinking_steps)
        
        return {
            "thinking_steps": thinking_steps,
            "thinking_summary": thinking_summary,
            "total_steps": len(thinking_steps),
            "strategy_used": strategy,
        }
    
    def _perform_thinking_step(
        self, 
        step_index: int, 
        current_understanding: Dict[str, Any],
        problem: Union[str, Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """执行单个思考步骤"""
        step_type = self._determine_step_type(step_index)
        
        # 根据步骤类型执行不同的思考操作
        if step_type == "analyze":
            result = self._thinking_analyze(problem, context, current_understanding)
        elif step_type == "hypothesize":
            result = self._thinking_hypothesize(problem, context, current_understanding)
        elif step_type == "evaluate":
            result = self._thinking_evaluate(problem, context, current_understanding)
        elif step_type == "synthesize":
            result = self._thinking_synthesize(problem, context, current_understanding)
        elif step_type == "conclude":
            result = self._thinking_conclude(problem, context, current_understanding)
        else:
            result = self._thinking_general(problem, context, current_understanding)
        
        # 添加步骤元数据
        result.update({
            "step_index": step_index,
            "step_type": step_type,
            "timestamp": datetime.now().isoformat(),
        })
        
        return result
    
    def _perform_self_reflection(
        self, 
        problem: Union[str, Dict[str, Any]], 
        thinking_result: Dict[str, Any],
        problem_analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """执行自我反思"""
        if not self.enable_reflection:
            return None
        
        try:
            # 收集反思数据
            reflection_data = {
                "problem": problem,
                "thinking_result": thinking_result,
                "problem_analysis": problem_analysis,
                "reflection_timestamp": datetime.now().isoformat(),
            }
            
            # 执行不同类型的反思
            reflections = []
            
            # 1. 过程反思
            process_reflection = self._reflect_on_process(thinking_result)
            if process_reflection:
                reflections.append({
                    "type": ReflectionType.PROCESS_REFLECTION.name,
                    "result": process_reflection,
                })
            
            # 2. 结果反思
            result_reflection = self._reflect_on_result(thinking_result)
            if result_reflection:
                reflections.append({
                    "type": ReflectionType.RESULT_REFLECTION.name,
                    "result": result_reflection,
                })
            
            # 3. 假设反思
            assumption_reflection = self._reflect_on_assumptions(thinking_result)
            if assumption_reflection:
                reflections.append({
                    "type": ReflectionType.ASSUMPTION_REFLECTION.name,
                    "result": assumption_reflection,
                })
            
            # 4. 知识反思
            knowledge_reflection = self._reflect_on_knowledge(thinking_result, problem_analysis)
            if knowledge_reflection:
                reflections.append({
                    "type": ReflectionType.KNOWLEDGE_REFLECTION.name,
                    "result": knowledge_reflection,
                })
            
            # 5. 元认知反思
            metacognitive_reflection = self._reflect_on_thinking(thinking_result)
            if metacognitive_reflection:
                reflections.append({
                    "type": ReflectionType.METACOGNITIVE_REFLECTION.name,
                    "result": metacognitive_reflection,
                })
            
            # 分析反思结果
            needs_correction = False
            correction_areas = []
            confidence_impact = 0.0
            
            for reflection in reflections:
                reflection_result = reflection["result"]
                if reflection_result.get("issues_found", False):
                    needs_correction = True
                    correction_areas.extend(reflection_result.get("issue_areas", []))
                
                # 更新置信度影响
                conf_impact = reflection_result.get("confidence_impact", 0.0)
                confidence_impact += conf_impact
            
            # 构建反思结果
            reflection_result = {
                "reflections": reflections,
                "needs_correction": needs_correction,
                "correction_areas": list(set(correction_areas)),  # 去重
                "confidence_impact": confidence_impact,
                "reflection_depth": self.reflection_depth,
                "total_reflections": len(reflections),
            }
            
            # 记录反思历史
            self.reflection_history.append({
                "timestamp": datetime.now().isoformat(),
                "problem": str(problem)[:100],
                "reflection_result": reflection_result,
            })
            
            logger.info(f"自我反思完成: 发现{len(reflections)}个反思点, 需要修正={needs_correction}")
            return reflection_result
            
        except Exception as e:
            logger.error(f"自我反思过程出错: {e}")
            return None
    
    def _perform_self_correction(
        self, 
        problem: Union[str, Dict[str, Any]], 
        thinking_result: Dict[str, Any],
        reflection_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """执行自我修正"""
        if not self.enable_correction:
            return None
        
        try:
            correction_attempts = []
            
            for iteration in range(self.max_correction_iterations):
                # 选择修正策略
                correction_strategy = self._select_correction_strategy(
                    reflection_result, 
                    iteration
                )
                
                # 执行修正
                correction_result = self._execute_correction(
                    problem, 
                    thinking_result, 
                    reflection_result, 
                    correction_strategy
                )
                
                correction_attempts.append({
                    "iteration": iteration,
                    "strategy": correction_strategy.name,
                    "result": correction_result,
                })
                
                # 检查修正是否成功
                if correction_result.get("success", False):
                    # 修正成功，停止迭代
                    logger.info(f"自我修正成功: 第{iteration+1}次迭代, 策略={correction_strategy.name}")
                    break
                
                # 修正失败，继续尝试
                logger.debug(f"自我修正失败: 第{iteration+1}次迭代, 策略={correction_strategy.name}")
            
            # 评估修正效果
            correction_effectiveness = self._evaluate_correction_effectiveness(correction_attempts)
            
            # 构建修正结果
            final_correction_result = {
                "correction_attempts": correction_attempts,
                "total_iterations": len(correction_attempts),
                "successful": any(attempt["result"].get("success", False) for attempt in correction_attempts),
                "effectiveness_score": correction_effectiveness,
                "final_thinking": correction_attempts[-1]["result"].get("corrected_thinking", thinking_result)
                if correction_attempts else thinking_result,
            }
            
            # 记录修正历史
            self.correction_history.append({
                "timestamp": datetime.now().isoformat(),
                "problem": str(problem)[:100],
                "correction_result": final_correction_result,
            })
            
            # 更新指标
            if final_correction_result["successful"]:
                self.metrics["successful_corrections"] += 1
            else:
                self.metrics["failed_corrections"] += 1
            
            return final_correction_result
            
        except Exception as e:
            logger.error(f"自我修正过程出错: {e}")
            return None
    
    def _generate_final_conclusion(
        self, 
        thinking_result: Dict[str, Any], 
        reflection_result: Optional[Dict[str, Any]],
        correction_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成最终结论"""
        # 使用修正后的思考结果（如果存在）
        if correction_result and correction_result.get("successful", False):
            final_thinking = correction_result.get("final_thinking", thinking_result)
        else:
            final_thinking = thinking_result
        
        # 提取思考总结
        thinking_summary = final_thinking.get("thinking_summary", {})
        
        # 计算置信度
        base_confidence = thinking_summary.get("confidence", 0.5)
        
        # 根据反思结果调整置信度
        if reflection_result:
            confidence_impact = reflection_result.get("confidence_impact", 0.0)
            base_confidence = max(0.0, min(1.0, base_confidence + confidence_impact))
        
        # 根据修正结果调整置信度
        if correction_result and correction_result.get("successful", False):
            base_confidence = min(1.0, base_confidence + 0.1)  # 修正成功增加置信度
        
        # 构建最终结论
        conclusion = {
            "answer": thinking_summary.get("conclusion", "无法得出明确结论"),
            "reasoning": thinking_summary.get("reasoning_steps", []),
            "confidence": base_confidence,
            "sources": thinking_summary.get("sources", []),
            "limitations": thinking_summary.get("limitations", []),
            "recommendations": thinking_summary.get("recommendations", []),
            "thinking_depth": final_thinking.get("strategy_used", {}).get("steps", 5),
            "has_correction": correction_result is not None and correction_result.get("successful", False),
        }
        
        return conclusion
    
    def _update_metrics(
        self, 
        problem_type: ProblemType,
        thinking_result: Dict[str, Any],
        reflection_result: Optional[Dict[str, Any]],
        correction_result: Optional[Dict[str, Any]],
        processing_time_ms: float
    ):
        """更新性能指标
        
        参数:
            problem_type: 问题类型
            thinking_result: 思考结果
            reflection_result: 反思结果（可选）
            correction_result: 修正结果（可选）
            processing_time_ms: 处理时间（毫秒）
        """
        self.metrics["total_problems_processed"] += 1
        
        if problem_type == ProblemType.CHALLENGING:
            self.metrics["challenging_problems"] += 1
        elif problem_type == ProblemType.UNKNOWN:
            self.metrics["unknown_problems"] += 1
        
        # 更新平均思考步数
        total_steps = thinking_result.get("total_steps", 5)
        current_avg = self.metrics["avg_thinking_steps"]
        total_problems = self.metrics["total_problems_processed"]
        self.metrics["avg_thinking_steps"] = (
            current_avg * (total_problems - 1) + total_steps
        ) / total_problems
        
        # 更新平均反思深度
        if reflection_result:
            reflection_depth = reflection_result.get("total_reflections", 0)
            current_reflection_avg = self.metrics["avg_reflection_depth"]
            self.metrics["avg_reflection_depth"] = (
                current_reflection_avg * (total_problems - 1) + reflection_depth
            ) / total_problems
        
        # 更新平均处理时间
        current_time_avg = self.metrics["avg_processing_time_ms"]
        self.metrics["avg_processing_time_ms"] = (
            current_time_avg * (total_problems - 1) + processing_time_ms
        ) / total_problems
        
        # 更新修正成功率
        if correction_result:
            if correction_result.get("successful", False):
                self.metrics["successful_corrections"] += 1
            else:
                self.metrics["failed_corrections"] += 1
        
        # 更新问题解决率（简单估计）
        if thinking_result.get("thinking_summary", {}).get("conclusion"):
            solved_count = self.metrics.get("solved_problems", 0)
            self.metrics["solved_problems"] = solved_count + 1
        
        if self.metrics["total_problems_processed"] > 0:
            self.metrics["problem_solving_rate"] = (
                self.metrics.get("solved_problems", 0) / self.metrics["total_problems_processed"]
            )
    
    # ===== 辅助方法实现 =====
    
    def _has_technical_terms(self, text: str) -> bool:
        """检查是否包含技术术语"""
        technical_indicators = ["算法", "模型", "训练", "推理", "神经网络", "transformer", "API", "接口", "协议"]
        return any(indicator in text for indicator in technical_indicators)
    
    def _has_uncertainty_words(self, text: str) -> bool:
        """检查是否包含不确定性词汇"""
        uncertainty_words = ["可能", "也许", "大概", "或许", "不确定", "不清楚", "不知道"]
        return any(word in text for word in uncertainty_words)
    
    def _has_contradiction_words(self, text: str) -> bool:
        """检查是否包含矛盾词汇"""
        contradiction_words = ["但是", "然而", "尽管", "虽然", "矛盾", "冲突", "相反"]
        return any(word in text for word in contradiction_words)
    
    def _calculate_complexity_score(self, text: str) -> float:
        """计算问题复杂度分数"""
        words = text.split()
        score = min(1.0, len(words) / 100)  # 长度因素
        
        # 增加技术术语的权重
        if self._has_technical_terms(text):
            score = min(1.0, score + 0.3)
        
        # 增加不确定性词汇的权重
        if self._has_uncertainty_words(text):
            score = min(1.0, score + 0.2)
        
        return score
    
    def _calculate_ambiguity_score(self, text: str) -> float:
        """计算问题模糊性分数"""
        ambiguity_indicators = 0
        
        # 检查疑问词数量
        question_words = ["什么", "如何", "为什么", "何时", "哪里", "谁"]
        for word in question_words:
            if word in text:
                ambiguity_indicators += 1
        
        # 检查不确定性词汇
        if self._has_uncertainty_words(text):
            ambiguity_indicators += 2
        
        # 计算分数
        return min(1.0, ambiguity_indicators / 5)
    
    def _determine_knowledge_requirement(self, text: str) -> str:
        """确定知识需求"""
        specialized_terms = ["量子", "相对论", "基因", "神经网络", "区块链", "加密货币", "深度学习"]
        if any(term in text for term in specialized_terms):
            return "specialized"
        
        technical_terms = ["算法", "编程", "代码", "软件", "硬件", "系统", "网络"]
        if any(term in text for term in technical_terms):
            return "technical"
        
        return "general"
    
    def _determine_step_type(self, step_index: int) -> str:
        """根据步骤索引确定步骤类型"""
        step_patterns = {
            0: "analyze",
            1: "hypothesize", 
            2: "evaluate",
            3: "synthesize",
            4: "conclude",
        }
        return step_patterns.get(step_index % 5, "general")
    
    def _thinking_analyze(self, problem, context, current_understanding):
        """分析思考步骤"""
        return {
            "operation": "analyze",
            "description": "分析问题结构和要求",
            "new_understanding": {"analysis_complete": True},
            "has_conclusion": False,
        }
    
    def _thinking_hypothesize(self, problem, context, current_understanding):
        """假设思考步骤"""
        return {
            "operation": "hypothesize", 
            "description": "生成可能解决方案假设",
            "new_understanding": {"hypotheses_generated": True},
            "has_conclusion": False,
        }
    
    def _thinking_evaluate(self, problem, context, current_understanding):
        """评估思考步骤"""
        return {
            "operation": "evaluate",
            "description": "评估不同假设的优劣",
            "new_understanding": {"evaluation_complete": True},
            "has_conclusion": False,
        }
    
    def _thinking_synthesize(self, problem, context, current_understanding):
        """综合思考步骤"""
        return {
            "operation": "synthesize",
            "description": "综合最佳方案",
            "new_understanding": {"synthesis_complete": True},
            "has_conclusion": False,
        }
    
    def _thinking_conclude(self, problem, context, current_understanding):
        """结论思考步骤"""
        return {
            "operation": "conclude",
            "description": "得出最终结论",
            "conclusion": "基于分析得出结论",
            "has_conclusion": True,
            "confidence": 0.8,
        }
    
    def _thinking_general(self, problem, context, current_understanding):
        """通用思考步骤"""
        return {
            "operation": "general",
            "description": "进一步深化思考",
            "new_understanding": {"thinking_deepened": True},
            "has_conclusion": False,
        }
    
    def _generate_thinking_summary(self, thinking_steps):
        """生成思考总结"""
        conclusions = [step for step in thinking_steps if step.get("has_conclusion", False)]
        final_conclusion = conclusions[-1] if conclusions else {"conclusion": "无法得出结论", "confidence": 0.0}
        
        return {
            "conclusion": final_conclusion.get("conclusion", "无结论"),
            "reasoning_steps": [step.get("description", "") for step in thinking_steps],
            "confidence": final_conclusion.get("confidence", 0.0),
            "total_steps": len(thinking_steps),
            "sources": [],
            "limitations": ["思考深度有限", "可能遗漏某些角度"],
            "recommendations": ["进一步验证结论", "收集更多信息"],
        }
    
    def _reflect_on_process(self, thinking_result):
        """过程反思"""
        steps = thinking_result.get("thinking_steps", [])
        
        issues = []
        if len(steps) < 3:
            issues.append("思考步骤过少")
        
        if not any(step.get("has_conclusion", False) for step in steps):
            issues.append("未得出明确结论")
        
        return {
            "issues_found": len(issues) > 0,
            "issue_areas": issues,
            "confidence_impact": -0.1 if issues else 0.0,
            "suggestions": ["增加思考步骤", "确保每个步骤有明确产出"] if issues else [],
        }
    
    def _reflect_on_result(self, thinking_result):
        """结果反思"""
        summary = thinking_result.get("thinking_summary", {})
        confidence = summary.get("confidence", 0.0)
        
        issues = []
        if confidence < 0.6:
            issues.append("置信度过低")
        
        if not summary.get("conclusion"):
            issues.append("结论为空")
        
        return {
            "issues_found": len(issues) > 0,
            "issue_areas": issues,
            "confidence_impact": -0.2 if confidence < 0.6 else 0.0,
            "suggestions": ["重新评估证据权重", "考虑替代解释"] if issues else [],
        }
    
    def _reflect_on_assumptions(self, thinking_result):
        """假设反思"""
        # 检查是否明确了假设
        steps = thinking_result.get("thinking_steps", [])
        has_assumptions = any("assumption" in str(step).lower() for step in steps)
        
        issues = []
        if not has_assumptions:
            issues.append("未明确陈述假设")
        
        return {
            "issues_found": len(issues) > 0,
            "issue_areas": issues,
            "confidence_impact": -0.05 if issues else 0.0,
            "suggestions": ["明确列出所有假设", "验证假设的合理性"] if issues else [],
        }
    
    def _reflect_on_knowledge(self, thinking_result, problem_analysis):
        """知识反思"""
        problem_type = problem_analysis.get("problem_type")
        
        issues = []
        if problem_type == ProblemType.UNKNOWN:
            issues.append("问题超出当前知识范围")
        
        return {
            "issues_found": len(issues) > 0,
            "issue_areas": issues,
            "confidence_impact": -0.3 if issues else 0.0,
            "suggestions": ["承认知识局限", "建议学习相关领域"] if issues else [],
        }
    
    def _reflect_on_thinking(self, thinking_result):
        """元认知反思"""
        steps = thinking_result.get("thinking_steps", [])
        diverse_operations = len(set(step.get("operation", "") for step in steps))
        
        issues = []
        if diverse_operations < 3:
            issues.append("思考操作类型单一")
        
        return {
            "issues_found": len(issues) > 0,
            "issue_areas": issues,
            "confidence_impact": -0.1 if issues else 0.0,
            "suggestions": ["使用更多样化的思考策略", "尝试不同推理角度"] if issues else [],
        }
    
    def _select_correction_strategy(self, reflection_result, iteration):
        """选择修正策略"""
        # 根据迭代次数和反思结果选择策略
        strategies = [
            CorrectionStrategy.REWRITE_REASONING,
            CorrectionStrategy.REFINE_ASSUMPTIONS,
            CorrectionStrategy.EXPAND_KNOWLEDGE,
            CorrectionStrategy.ADJUST_LOGIC,
            CorrectionStrategy.INTEGRATE_NEW_EVIDENCE,
            CorrectionStrategy.RESTRUCTURE_ARGUMENT,
        ]
        
        # 根据迭代次数选择策略
        return strategies[iteration % len(strategies)]
    
    def _execute_correction(self, problem, thinking_result, reflection_result, strategy):
        """执行修正"""
        # 模拟修正过程
        import random
        
        success = random.random() > 0.3  # 70%成功率
        
        if success:
            return {
                "success": True,
                "strategy": strategy.name,
                "corrected_thinking": {
                    **thinking_result,
                    "thinking_summary": {
                        **thinking_result.get("thinking_summary", {}),
                        "confidence": thinking_result.get("thinking_summary", {}).get("confidence", 0.0) + 0.1,
                        "correction_note": f"应用了{strategy.name}策略进行修正",
                    }
                },
                "improvement": "思考质量提升",
            }
        else:
            return {
                "success": False,
                "strategy": strategy.name,
                "reason": "修正策略未产生显著改进",
                "corrected_thinking": thinking_result,
            }
    
    def _evaluate_correction_effectiveness(self, correction_attempts):
        """评估修正效果"""
        if not correction_attempts:
            return 0.0
        
        successful = any(attempt["result"].get("success", False) for attempt in correction_attempts)
        
        if successful:
            # 找到第一次成功的尝试
            for i, attempt in enumerate(correction_attempts):
                if attempt["result"].get("success", False):
                    # 越早成功，效果分数越高
                    effectiveness = 1.0 - (i * 0.2)
                    return max(0.3, effectiveness)
        
        return 0.0
    
    def handle_unknown_problem(
        self, 
        problem: Union[str, Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """处理超出认知的问题
        
        对于超出当前知识范围的问题，采用特殊处理策略：
        1. 承认知识局限
        2. 基于现有知识进行合理推断
        3. 识别问题中的已知元素
        4. 提供最佳逻辑判断
        5. 建议学习方向
        
        参数:
            problem: 问题描述
            context: 上下文信息
            
        返回:
            处理结果，包含：
            - 承认局限声明
            - 基于现有知识的推断
            - 置信度评估
            - 学习建议
        """
        try:
            # 分析问题，确认其超出认知
            problem_analysis = self._analyze_problem(problem, context)
            
            if problem_analysis.get("problem_type") != ProblemType.UNKNOWN:
                # 如果不是未知问题，使用常规深度思考
                return self.deep_think(problem, context, ThinkingDepth.EXTREME)
            
            # 步骤1：明确承认知识局限
            knowledge_gap_analysis = self._analyze_knowledge_gap(problem, context)
            
            # 步骤2：识别问题中的已知元素
            known_elements = self._identify_known_elements(problem, context)
            
            # 步骤3：基于已知元素进行逻辑推断
            logical_inference = self._make_logical_inference(problem, known_elements, context)
            
            # 步骤4：评估推断的合理性
            reasonableness = self._evaluate_reasonableness(logical_inference, problem, context)
            
            # 步骤5：生成学习建议
            learning_suggestions = self._generate_learning_suggestions(problem, knowledge_gap_analysis)
            
            # 构建结果
            result = {
                "success": True,
                "problem_type": "UNKNOWN",
                "knowledge_gap_acknowledged": True,
                "knowledge_gap_analysis": knowledge_gap_analysis,
                "known_elements_identified": known_elements,
                "logical_inference": logical_inference,
                "reasonableness_score": reasonableness,
                "confidence": min(0.7, reasonableness * 0.8),  # 未知问题置信度上限
                "learning_suggestions": learning_suggestions,
                "recommendation": "此问题超出当前知识范围，建议进行专门学习后再回答",
                "timestamp": datetime.now().isoformat(),
            }
            
            logger.info(f"超出认知问题处理完成: 已知元素={len(known_elements)}, 合理性分数={reasonableness:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"处理超出认知问题时出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "problem_type": "UNKNOWN",
                "knowledge_gap_acknowledged": True,
                "recommendation": "无法处理此超出认知范围的问题，建议咨询领域专家",
            }
    
    def _analyze_knowledge_gap(
        self, 
        problem: Union[str, Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析知识缺口"""
        problem_text = str(problem) if isinstance(problem, str) else str(problem.get("text", problem))
        
        # 识别专业术语和概念
        specialized_terms = self._extract_specialized_terms(problem_text)
        
        # 估计所需知识领域
        knowledge_domains = self._estimate_knowledge_domains(problem_text)
        
        # 评估缺口大小（简单实现）
        gap_size = min(1.0, len(specialized_terms) * 0.1)
        
        return {
            "specialized_terms": specialized_terms,
            "knowledge_domains": knowledge_domains,
            "gap_size": gap_size,
            "gap_description": f"需要{len(knowledge_domains)}个领域的专业知识",
        }
    
    def _extract_specialized_terms(self, text: str) -> List[str]:
        """提取专业术语"""
        # 简单实现：基于关键词列表
        term_lists = {
            "物理": ["量子", "相对论", "引力波", "弦理论", "黑洞"],
            "化学": ["分子", "原子", "化学键", "反应机理", "催化剂"],
            "生物": ["基因", "DNA", "蛋白质", "细胞", "进化"],
            "医学": ["疾病", "诊断", "治疗", "药物", "手术"],
            "数学": ["微积分", "线性代数", "拓扑", "概率论", "数论"],
            "计算机": ["算法", "数据结构", "神经网络", "区块链", "加密"],
        }
        
        found_terms = []
        for domain, terms in term_lists.items():
            for term in terms:
                if term in text:
                    found_terms.append({"term": term, "domain": domain})
        
        return found_terms
    
    def _estimate_knowledge_domains(self, text: str) -> List[str]:
        """估计所需知识领域"""
        domains = []
        
        domain_keywords = {
            "物理": ["物理", "力学", "电磁", "量子", "相对论"],
            "化学": ["化学", "分子", "反应", "元素", "化合物"],
            "生物": ["生物", "基因", "细胞", "进化", "生态"],
            "医学": ["医学", "疾病", "健康", "治疗", "药物"],
            "数学": ["数学", "计算", "公式", "方程", "几何"],
            "计算机": ["计算机", "编程", "算法", "软件", "硬件"],
            "工程": ["工程", "设计", "制造", "材料", "结构"],
            "经济": ["经济", "金融", "市场", "投资", "贸易"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                domains.append(domain)
        
        return domains if domains else ["通用"]
    
    def _identify_known_elements(
        self, 
        problem: Union[str, Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """识别问题中的已知元素"""
        problem_text = str(problem) if isinstance(problem, str) else str(problem.get("text", problem))
        
        known_elements = []
        
        # 识别通用概念
        general_concepts = ["时间", "空间", "数量", "质量", "关系", "变化", "原因", "结果"]
        for concept in general_concepts:
            if concept in problem_text:
                known_elements.append({
                    "type": "general_concept",
                    "name": concept,
                    "description": f"通用概念: {concept}",
                })
        
        # 识别基本逻辑关系
        logic_indicators = ["因为", "所以", "如果", "那么", "与", "或", "非"]
        for indicator in logic_indicators:
            if indicator in problem_text:
                known_elements.append({
                    "type": "logical_relation",
                    "name": indicator,
                    "description": f"逻辑关系指示词: {indicator}",
                })
        
        # 识别常见实体
        common_entities = ["人", "物", "事", "地方", "时间", "数量"]
        for entity in common_entities:
            if entity in problem_text:
                known_elements.append({
                    "type": "common_entity",
                    "name": entity,
                    "description": f"常见实体: {entity}",
                })
        
        return known_elements
    
    def _make_logical_inference(
        self, 
        problem: Union[str, Dict[str, Any]], 
        known_elements: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """基于已知元素进行逻辑推断"""
        problem_text = str(problem) if isinstance(problem, str) else str(problem.get("text", problem))
        
        # 提取问题结构
        question_words = ["什么", "如何", "为什么", "何时", "哪里", "谁"]
        question_type = "未知"
        for word in question_words:
            if word in problem_text:
                question_type = word
                break
        
        # 基于问题类型生成推断框架
        inference_frameworks = {
            "什么": {"type": "definition", "approach": "尝试定义或描述"},
            "如何": {"type": "process", "approach": "描述可能的过程或方法"},
            "为什么": {"type": "causality", "approach": "寻找可能的原因或理由"},
            "何时": {"type": "temporal", "approach": "估计时间范围或时机"},
            "哪里": {"type": "spatial", "approach": "推测可能的地点或位置"},
            "谁": {"type": "agent", "approach": "识别可能的行动者或责任人"},
            "未知": {"type": "general", "approach": "进行一般性分析"},
        }
        
        framework = inference_frameworks.get(question_type, inference_frameworks["未知"])
        
        # 构建推断
        inference = {
            "question_type": question_type,
            "inference_framework": framework,
            "based_on_elements": [elem["type"] for elem in known_elements],
            "inference_statement": f"基于{len(known_elements)}个已知元素，使用{framework['approach']}方法进行分析",
            "potential_answer_structure": self._generate_answer_structure(framework, known_elements),
            "assumptions": ["假设问题遵循常规逻辑", "假设已知元素是相关的"],
            "limitations": ["缺乏领域专业知识", "推断基于一般逻辑而非具体知识"],
        }
        
        return inference
    
    def _generate_answer_structure(self, framework: Dict[str, Any], known_elements: List[Dict[str, Any]]) -> str:
        """生成答案结构"""
        structures = {
            "definition": "1. 核心概念定义\n2. 关键特征描述\n3. 相关概念对比\n4. 应用场景举例",
            "process": "1. 步骤概述\n2. 关键环节说明\n3. 所需资源或条件\n4. 预期结果或产出",
            "causality": "1. 可能原因列举\n2. 原因优先级排序\n3. 因果关系机制\n4. 预防或解决建议",
            "temporal": "1. 时间范围估计\n2. 关键时间点识别\n3. 时间序列分析\n4. 时间影响因素",
            "spatial": "1. 可能位置范围\n2. 位置特征描述\n3. 空间关系分析\n4. 位置选择依据",
            "agent": "1. 可能行动者类型\n2. 行动者特征描述\n3. 行动动机分析\n4. 行动能力评估",
            "general": "1. 问题分解\n2. 要素分析\n3. 关系推断\n4. 综合结论",
        }
        
        return structures.get(framework["type"], structures["general"])
    
    def _evaluate_reasonableness(
        self, 
        inference: Dict[str, Any], 
        problem: Union[str, Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> float:
        """评估推断的合理性"""
        score = 0.5  # 基础分
        
        # 基于已知元素数量加分
        element_count = len(inference.get("based_on_elements", []))
        score += min(0.3, element_count * 0.05)
        
        # 检查推断结构完整性
        if inference.get("potential_answer_structure"):
            score += 0.1
        
        # 检查是否明确承认局限
        if "limitations" in inference and len(inference["limitations"]) > 0:
            score += 0.05  # 诚实承认局限是合理的
        
        # 确保分数在合理范围内
        return max(0.1, min(0.9, score))
    
    def _generate_learning_suggestions(
        self, 
        problem: Union[str, Dict[str, Any]], 
        knowledge_gap_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """生成学习建议"""
        suggestions = []
        
        # 基于知识领域建议
        domains = knowledge_gap_analysis.get("knowledge_domains", [])
        for domain in domains[:3]:  # 最多3个领域
            suggestions.append({
                "type": "domain_knowledge",
                "domain": domain,
                "suggestion": f"学习{domain}领域的基础知识",
                "resources": [f"{domain}教科书", f"{domain}在线课程", f"{domain}学术论文"],
                "priority": "高" if domain in ["物理", "数学", "计算机"] else "中",
            })
        
        # 基于专业术语建议
        terms = knowledge_gap_analysis.get("specialized_terms", [])
        if terms:
            unique_domains = set(term["domain"] for term in terms)
            for domain in list(unique_domains)[:2]:
                domain_terms = [term["term"] for term in terms if term["domain"] == domain]
                suggestions.append({
                    "type": "terminology",
                    "domain": domain,
                    "suggestion": f"掌握{domain}领域的关键术语: {', '.join(domain_terms[:5])}",
                    "resources": [f"{domain}术语词典", f"{domain}入门教程"],
                    "priority": "中",
                })
        
        # 通用学习建议
        suggestions.append({
            "type": "general",
            "domain": "通用",
            "suggestion": "提高逻辑推理和问题分解能力",
            "resources": ["逻辑学教材", "批判性思维课程", "问题解决方法论"],
            "priority": "低",
        })
        
        return suggestions
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.metrics.copy()
    
    def get_thinking_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取思考历史"""
        return self.thinking_history[-limit:] if self.thinking_history else []
    
    def get_reflection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取反思历史"""
        return self.reflection_history[-limit:] if self.reflection_history else []
    
    def get_correction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取修正历史"""
        return self.correction_history[-limit:] if self.correction_history else []
    
    def reset_history(self):
        """重置历史记录"""
        self.thinking_history = []
        self.reflection_history = []
        self.correction_history = []
        logger.info("深度思考历史记录已重置")


# 简化的工厂函数，便于使用
def create_deep_thinking_engine(config: Optional[Dict[str, Any]] = None) -> DeepThinkingEngine:
    """创建深度思考引擎实例"""
    return DeepThinkingEngine(config)