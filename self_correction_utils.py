#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自我修证辅助工具模块

提供自我修证功能的公共常量、函数和工具：
1. 思考深度映射和转换
2. 问题类型识别
3. 缓存键生成
4. 性能指标计算

本模块旨在减少代码重复，提高自我修证功能的一致性。
"""

import hashlib
import json
from typing import Dict, Any, Optional, Union, Tuple, List
from enum import Enum

# 如果DeepThinkingEngine可用，导入相关枚举
try:
    from models.deep_thinking_engine import (
        ThinkingDepth,
        ProblemType,
        ReflectionType,
        CorrectionStrategy,
    )
    DEEP_THINKING_AVAILABLE = True
except ImportError:
    # 定义占位枚举（当DeepThinkingEngine不可用时）
    class ThinkingDepth(Enum):
        """思考深度级别"""
        SHALLOW = "shallow"
        MODERATE = "moderate"
        DEEP = "deep"
        EXTREME = "extreme"
    
    class ProblemType(Enum):
        """问题类型分类"""
        KNOWN = "known"
        CHALLENGING = "challenging"
        UNKNOWN = "unknown"
        AMBIGUOUS = "ambiguous"
        CONTRADICTORY = "contradictory"
    
    DEEP_THINKING_AVAILABLE = False


# 公共常量
DEFAULT_THINKING_DEPTH = "moderate"
DEFAULT_MAX_THINKING_STEPS = 10
DEFAULT_CACHE_SIZE = 100
DEFAULT_CACHE_TTL_SECONDS = 3600  # 1小时

# 思考深度到步数映射（用于训练器）
THINKING_DEPTH_TO_STEPS = {
    "shallow": 3,
    "moderate": 5,
    "deep": 8,
    "extreme": 12,
}

# 字符串到ThinkingDepth枚举映射
STRING_TO_THINKING_DEPTH = {
    "shallow": ThinkingDepth.SHALLOW,
    "moderate": ThinkingDepth.MODERATE,
    "deep": ThinkingDepth.DEEP,
    "extreme": ThinkingDepth.EXTREME,
}

# ThinkingDepth枚举到字符串映射
THINKING_DEPTH_TO_STRING = {
    ThinkingDepth.SHALLOW: "shallow",
    ThinkingDepth.MODERATE: "moderate",
    ThinkingDepth.DEEP: "deep",
    ThinkingDepth.EXTREME: "extreme",
}

# 默认损失组件权重
DEFAULT_LOSS_COMPONENTS = {
    "reflection_weight": 0.1,
    "correction_weight": 0.2,
    "alignment_weight": 0.05,
    "depth_reward_weight": 0.01,
}

# 默认自我修证训练配置
DEFAULT_SELF_CORRECTION_CONFIG = {
    "thinking_depth": DEFAULT_THINKING_DEPTH,
    "enable_reflection": True,
    "enable_correction": True,
    "correction_loss_weight": 0.1,
    "max_thinking_steps": DEFAULT_MAX_THINKING_STEPS,
    "use_deep_thinking_engine": True,
    "integration_mode": "loss_based",
    "dynamic_thinking_adjustment": True,
    "thinking_cache_enabled": True,
    "cache_max_size": DEFAULT_CACHE_SIZE,
    "cache_ttl_steps": 1000,
    "signal_generation_mode": "automatic",
    "loss_components": DEFAULT_LOSS_COMPONENTS,
}


def normalize_thinking_depth(
    thinking_depth: Union[str, ThinkingDepth, None],
    default: Union[str, ThinkingDepth] = DEFAULT_THINKING_DEPTH
) -> Union[str, ThinkingDepth]:
    """
    标准化思考深度输入
    
    支持多种输入格式：
    - 字符串: "shallow", "moderate", "deep", "extreme"
    - ThinkingDepth枚举
    - None: 返回默认值
    
    参数:
        thinking_depth: 思考深度输入
        default: 默认值
        
    返回:
        标准化的思考深度（如果输入是字符串则返回字符串，如果是枚举则返回枚举）
    """
    if thinking_depth is None:
        return default
    
    if isinstance(thinking_depth, ThinkingDepth):
        return thinking_depth
    
    if isinstance(thinking_depth, str):
        # 标准化字符串
        thinking_depth = thinking_depth.lower().strip()
        if thinking_depth in STRING_TO_THINKING_DEPTH:
            return thinking_depth  # 返回标准化字符串
        else:
            # 尝试转换为枚举（如果可用）
            if DEEP_THINKING_AVAILABLE:
                return STRING_TO_THINKING_DEPTH.get(thinking_depth, default)
            else:
                return default
    
    # 其他类型，返回默认值
    return default


def thinking_depth_to_steps(
    thinking_depth: Union[str, ThinkingDepth],
    default_steps: int = DEFAULT_MAX_THINKING_STEPS
) -> int:
    """
    将思考深度转换为步数
    
    参数:
        thinking_depth: 思考深度（字符串或枚举）
        default_steps: 默认步数
        
    返回:
        对应的思考步数
    """
    # 首先标准化思考深度
    normalized_depth = normalize_thinking_depth(thinking_depth)
    
    if isinstance(normalized_depth, ThinkingDepth):
        # 如果是枚举，转换为字符串
        depth_str = THINKING_DEPTH_TO_STRING.get(normalized_depth, DEFAULT_THINKING_DEPTH)
    else:
        depth_str = normalized_depth
    
    # 返回对应的步数
    return THINKING_DEPTH_TO_STEPS.get(depth_str, default_steps)


def get_default_loss_components() -> Dict[str, float]:
    """
    获取默认损失组件权重
    
    返回:
        默认损失组件权重字典
    """
    return DEFAULT_LOSS_COMPONENTS.copy()


def get_default_self_correction_config() -> Dict[str, Any]:
    """
    获取默认自我修证训练配置
    
    返回:
        默认自我修证训练配置字典
    """
    return DEFAULT_SELF_CORRECTION_CONFIG.copy()


def generate_cache_key(
    data: Union[str, Dict[str, Any], List[Any]],
    prefix: str = "thinking_cache"
) -> str:
    """
    生成缓存键
    
    参数:
        data: 缓存数据（字符串、字典或列表）
        prefix: 缓存键前缀
        
    返回:
        唯一的缓存键字符串
    """
    # 将数据转换为字符串表示
    if isinstance(data, dict) or isinstance(data, list):
        data_str = json.dumps(data, sort_keys=True)  # 排序键以确保一致性
    else:
        data_str = str(data)
    
    # 生成哈希
    hash_obj = hashlib.md5(data_str.encode('utf-8'))
    data_hash = hash_obj.hexdigest()[:12]  # 使用前12个字符
    
    # 返回带前缀的缓存键
    return f"{prefix}_{data_hash}"


def estimate_batch_complexity(batch: Dict[str, Any]) -> float:
    """
    估计批次复杂度
    
    基于批次特征估计复杂度分数（0.0到1.0之间）
    
    参数:
        batch: 训练批次数据
        
    返回:
        复杂度分数（0.0到1.0之间）
    """
    complexity_factors = []
    
    # 因子1: 批次大小
    if "batch_size" in batch:
        batch_size = batch["batch_size"]
        # 归一化批次大小（假设最大批次大小为32）
        batch_size_factor = min(1.0, batch_size / 32.0)
        complexity_factors.append(batch_size_factor * 0.2)
    
    # 因子2: 序列长度
    if "input_ids" in batch and hasattr(batch["input_ids"], "shape"):
        seq_len = batch["input_ids"].shape[-1] if len(batch["input_ids"].shape) > 1 else 1
        # 归一化序列长度（假设最大序列长度为512）
        seq_len_factor = min(1.0, seq_len / 512.0)
        complexity_factors.append(seq_len_factor * 0.3)
    
    # 因子3: 字段数量
    field_count = len(batch)
    # 归一化字段数量（假设最多20个字段）
    field_count_factor = min(1.0, field_count / 20.0)
    complexity_factors.append(field_count_factor * 0.2)
    
    # 因子4: 数据类型多样性
    tensor_count = sum(1 for v in batch.values() if hasattr(v, "shape"))
    tensor_factor = min(1.0, tensor_count / max(1, field_count))
    complexity_factors.append(tensor_factor * 0.3)
    
    # 计算平均复杂度
    if complexity_factors:
        complexity = sum(complexity_factors)
    else:
        complexity = 0.5  # 默认中等复杂度
    
    # 确保在0.0到1.0之间
    return max(0.0, min(1.0, complexity))


def validate_self_correction_config(
    config: Dict[str, Any],
    raise_on_error: bool = False
) -> Tuple[bool, List[str]]:
    """
    验证自我修证配置
    
    参数:
        config: 自我修证配置字典
        raise_on_error: 是否在错误时抛出异常
        
    返回:
        (是否有效, 错误消息列表)
    """
    errors = []
    
    # 检查必要字段
    required_fields = ["thinking_depth", "correction_loss_weight"]
    for field in required_fields:
        if field not in config:
            errors.append(f"缺少必要字段: {field}")
    
    # 验证思考深度
    if "thinking_depth" in config:
        thinking_depth = config["thinking_depth"]
        valid_depths = ["shallow", "moderate", "deep", "extreme"]
        if isinstance(thinking_depth, str) and thinking_depth.lower() not in valid_depths:
            errors.append(f"无效的思考深度: {thinking_depth}，有效值: {valid_depths}")
    
    # 验证权重
    if "correction_loss_weight" in config:
        weight = config["correction_loss_weight"]
        if not isinstance(weight, (int, float)) or weight < 0 or weight > 1:
            errors.append(f"修正损失权重必须在0到1之间: {weight}")
    
    # 验证损失组件（如果存在）
    if "loss_components" in config and isinstance(config["loss_components"], dict):
        for key, value in config["loss_components"].items():
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"损失组件 {key} 权重必须为非负数: {value}")
    
    # 验证缓存配置（如果存在）
    if "cache_max_size" in config:
        cache_size = config["cache_max_size"]
        if not isinstance(cache_size, int) or cache_size < 0:
            errors.append(f"缓存大小必须为非负整数: {cache_size}")
    
    # 如果有错误且需要抛出异常
    if errors and raise_on_error:
        raise ValueError(f"自我修证配置验证失败: {', '.join(errors)}")
    
    return len(errors) == 0, errors


def create_scheduler_function(
    warmup_steps: int = 500,
    cool_down_steps: int = 2000,
    max_weight: float = 0.5,
    min_weight: float = 0.01,
    base_weight: float = 0.1
):
    """
    创建自适应损失权重调度器函数
    
    参数:
        warmup_steps: 预热步数
        cool_down_steps: 冷却开始步数
        max_weight: 最大权重乘数
        min_weight: 最小权重乘数
        base_weight: 基础权重
        
    返回:
        调度器函数 f(step) -> weight
    """
    def scheduler(step: int) -> float:
        """自适应损失权重调度器"""
        if step < warmup_steps:
            # 预热阶段：线性增加乘数
            multiplier = min_weight + (max_weight - min_weight) * (step / warmup_steps)
        elif step < cool_down_steps:
            # 稳定阶段：保持最大乘数
            multiplier = max_weight
        else:
            # 冷却阶段：线性减少到最小乘数
            decay_steps = step - cool_down_steps
            decay_factor = max(0.0, 1.0 - decay_steps / 10000)  # 在10000步内衰减
            multiplier = min_weight + (max_weight - min_weight) * decay_factor
        
        # 返回总权重 = 基础权重 × 乘数
        return base_weight * multiplier
    
    return scheduler


def calculate_self_correction_metrics(
    thinking_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    计算自我修证性能指标
    
    参数:
        thinking_results: 深度思考结果列表
        
    返回:
        性能指标字典
    """
    if not thinking_results:
        return {"total_results": 0}
    
    metrics = {
        "total_results": len(thinking_results),
        "successful_results": 0,
        "average_confidence": 0.0,
        "average_thinking_steps": 0.0,
        "cache_hit_rate": 0.0,
        "reflection_enabled_rate": 0.0,
        "correction_enabled_rate": 0.0,
    }
    
    total_confidence = 0.0
    total_steps = 0
    cache_hits = 0
    reflection_enabled = 0
    correction_enabled = 0
    
    for result in thinking_results:
        if result.get("success", False):
            metrics["successful_results"] += 1
        
        # 置信度
        confidence = result.get("final_conclusion", {}).get("confidence", 0.0)
        total_confidence += confidence
        
        # 思考步数
        steps = result.get("thinking_result", {}).get("total_steps", 0)
        total_steps += steps
        
        # 缓存命中
        if result.get("cached", False):
            cache_hits += 1
        
        # 反思和修正启用状态
        if result.get("reflection_result"):
            reflection_enabled += 1
        if result.get("correction_result"):
            correction_enabled += 1
    
    # 计算平均值
    if metrics["successful_results"] > 0:
        metrics["average_confidence"] = total_confidence / metrics["successful_results"]
        metrics["average_thinking_steps"] = total_steps / metrics["successful_results"]
    
    if len(thinking_results) > 0:
        metrics["cache_hit_rate"] = cache_hits / len(thinking_results)
        metrics["reflection_enabled_rate"] = reflection_enabled / len(thinking_results)
        metrics["correction_enabled_rate"] = correction_enabled / len(thinking_results)
    
    return metrics


if __name__ == "__main__":
    # 模块测试
    print("自我修证辅助工具模块测试")
    print("=" * 60)
    
    # 测试思考深度转换
    print("思考深度转换测试:")
    for depth in ["shallow", "moderate", "deep", "extreme", "invalid"]:
        normalized = normalize_thinking_depth(depth)
        steps = thinking_depth_to_steps(depth)
        print(f"  {depth:10} -> 标准化: {normalized}, 步数: {steps}")
    
    # 测试缓存键生成
    print("\n缓存键生成测试:")
    test_data = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    cache_key = generate_cache_key(test_data)
    print(f"  测试数据: {test_data}")
    print(f"  缓存键: {cache_key}")
    
    # 测试批次复杂度估计
    print("\n批次复杂度估计测试:")
    test_batch = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]],
        "labels": [[1, 2, 3], [4, 5, 6]],
    }
    complexity = estimate_batch_complexity(test_batch)
    print(f"  测试批次: {len(test_batch)} 个字段")
    print(f"  估计复杂度: {complexity:.2f}")
    
    # 测试配置验证
    print("\n配置验证测试:")
    test_config = {
        "thinking_depth": "moderate",
        "correction_loss_weight": 0.1,
        "loss_components": {"reflection_weight": 0.1, "correction_weight": 0.2},
    }
    is_valid, errors = validate_self_correction_config(test_config)
    print(f"  测试配置: {test_config}")
    print(f"  是否有效: {is_valid}")
    if errors:
        print(f"  错误: {errors}")
    
    print("\n" + "=" * 60)
    print("模块测试完成")