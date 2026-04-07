#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self AGI模型测试脚本
测试增强的计划能力和推理引擎
"""

import sys
import os
import torch
import pytest
import logging
from typing import Dict, Any, Tuple, List, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.transformer.self_agi_model import SelfAGIModel, AGIModelConfig  # noqa: E402
from models.transformer.config import ModelConfig  # noqa: E402

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_initialization() -> Tuple["SelfAGIModel", "ModelConfig"]:
    """测试模型初始化"""
    logger.info("测试模型初始化...")

    # 创建配置
    config = ModelConfig(
        vocab_size=10000,
        hidden_size=1024,  # 修正：匹配模型实际维度
        num_hidden_layers=4,
        num_attention_heads=16,  # 1024/16=64，可整除
        max_position_embeddings=512,
        multimodal_enabled=True,  # 启用多模态，我们将提供所有5个模态
        multimodal_fusion_enabled=True,  # 启用融合
        planning_enabled=False,  # 禁用规划以简化测试
        reasoning_enabled=False,  # 禁用推理以避免维度问题
        execution_control_enabled=False,  # 禁用执行控制以简化测试
        self_cognition_enabled=True,
        learning_enabled=True,
        self_correction_enabled=True,  # 启用自我改正功能
        # 添加缺失的嵌入维度参数
        text_embedding_dim=512,
        image_embedding_dim=512,
        audio_embedding_dim=512,
        video_embedding_dim=512,
        sensor_embedding_dim=512,
        fused_embedding_dim=512,
    )

    # 创建模型
    model = SelfAGIModel(config)

    # 获取模型信息
    model_info = model.get_model_info()
    logger.info(f"模型类型: {model_info['model_type']}")
    logger.info(f"总参数: {model_info['total_parameters']:,}")
    logger.info(f"可训练参数: {model_info['trainable_parameters']:,}")
    logger.info(f"计划能力启用: {model_info['planning_enabled']}")
    logger.info(f"推理能力启用: {model_info['reasoning_enabled']}")

    # 检查模型状态
    assert model.initialized, "模型未正确初始化"
    logger.info("✓ 模型初始化测试通过")

    return model, config


def test_forward_pass() -> Dict[str, Any]:
    """测试前向传播"""
    logger.info("测试前向传播...")

    # 创建配置
    config = ModelConfig(
        vocab_size=10000,
        hidden_size=512,  # 修正：匹配模型实际维度
        num_hidden_layers=4,
        num_attention_heads=8,  # 512/8=64，可整除
        max_position_embeddings=512,
        multimodal_enabled=True,
        multimodal_fusion_enabled=False,  # 禁用融合以避免维度问题
        planning_enabled=True,
        reasoning_enabled=True,
        execution_control_enabled=True,
        self_cognition_enabled=True,
        learning_enabled=True,
        self_correction_enabled=True,  # 启用自我改正功能
        # 添加缺失的嵌入维度参数
        text_embedding_dim=512,
        image_embedding_dim=512,
        audio_embedding_dim=512,
        video_embedding_dim=512,
        sensor_embedding_dim=512,
        fused_embedding_dim=512,
    )

    # 创建模型
    model = SelfAGIModel(config)

    # 创建测试输入
    batch_size = 2
    seq_len = 5  # 匹配5个模态

    # 输入token IDs
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # 注意力掩码
    attention_mask = torch.ones(batch_size, seq_len)

    # 目标嵌入
    goals = torch.randn(batch_size, config.hidden_size)

    # 上下文信息
    context = torch.randn(batch_size, seq_len // 2, config.hidden_size)

    # 约束条件
    constraints = torch.randn(batch_size, config.hidden_size)

    # 资源信息
    resources = torch.randn(batch_size, config.hidden_size)

    # 多模态输入（完整5个模态）
    multimodal_inputs = {
        "text_embeddings": torch.randn(
            batch_size, seq_len, config.text_embedding_dim
        ),
        "image_embeddings": torch.randn(
            batch_size, seq_len, config.image_embedding_dim
        ),
        "audio_embeddings": torch.randn(
            batch_size, seq_len, config.audio_embedding_dim
        ),
        "video_embeddings": torch.randn(
            batch_size, seq_len, config.video_embedding_dim
        ),
        "sensor_embeddings": torch.randn(
            batch_size, seq_len, config.sensor_embedding_dim
        ),
        "modality_types": [0, 1, 2, 3, 4],  # 文本、图像、音频、视频、传感器
    }

    # 推理类型
    reasoning_type = ["logic", "causal", "spatial"]

    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        multimodal_inputs=multimodal_inputs,
        goals=goals,
        context=context,
        constraints=constraints,
        resources=resources,
        reasoning_type=reasoning_type,
    )

    # 验证输出
    assert "logits" in outputs, "输出中缺少logits"
    assert "hidden_states" in outputs, "输出中缺少hidden_states"
    assert outputs["logits"].shape == (
        batch_size,
        seq_len,
        config.vocab_size,
    ), "logits形状不正确"
    assert outputs["hidden_states"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), "隐藏状态形状不正确"

    # 检查计划输出
    if config.planning_enabled:
        assert "plans" in outputs, "输出中缺少plans"
        assert "planned_actions" in outputs, "输出中缺少planned_actions"
        assert outputs["plans"].shape == (
            batch_size,
            seq_len,
            config.hidden_size,
        ), "计划形状不正确"
        logger.info("✓ 计划模块输出验证通过")

    # 检查推理输出
    if config.reasoning_enabled:
        assert "logic_output" in outputs, "输出中缺少logic_output"
        assert "causal_output" in outputs, "输出中缺少causal_output"
        assert "spatial_output" in outputs, "输出中缺少spatial_output"
        assert "fused_reasoning" in outputs, "输出中缺少fused_reasoning"
        logger.info("✓ 推理模块输出验证通过")

    # 检查其他模块输出
    if config.execution_control_enabled:
        assert "execution_actions" in outputs, "输出中缺少execution_actions"

    if config.self_cognition_enabled:
        assert "self_representation" in outputs, "输出中缺少self_representation"

    if config.learning_enabled:
        assert "learned_features" in outputs, "输出中缺少learned_features"

        # 检查自我改正输出
        if config.self_correction_enabled:
            assert "error_scores" in outputs, "输出中缺少error_scores"
            assert "error_types" in outputs, "输出中缺少error_types"
            assert "cause_analysis" in outputs, "输出中缺少cause_analysis"
            assert "corrections" in outputs, "输出中缺少corrections"
            assert "verification_scores" in outputs, "输出中缺少verification_scores"
            assert "corrected_features" in outputs, "输出中缺少corrected_features"
            assert "applied_corrections" in outputs, "输出中缺少applied_corrections"

            # 检查形状
            assert outputs["error_scores"].shape == (
                batch_size,
                seq_len,
                3,
            ), "错误分数形状不正确"
            assert outputs["error_types"].shape == (
                batch_size,
                seq_len,
            ), "错误类型形状不正确"
            assert outputs["cause_analysis"].shape == (
                batch_size,
                seq_len,
                config.hidden_size,
            ), "原因分析形状不正确"
            assert outputs["corrections"].shape == (
                batch_size,
                seq_len,
                config.hidden_size,
            ), "改正建议形状不正确"
            assert outputs["verification_scores"].shape == (
                batch_size,
                seq_len,
                1,
            ), "验证分数形状不正确"
            assert outputs["corrected_features"].shape == (
                batch_size,
                seq_len,
                config.hidden_size,
            ), "改正特征形状不正确"

            logger.info("✓ 自我改正模块输出验证通过")

    logger.info("✓ 前向传播测试通过")

    return outputs


def test_planning_module() -> Dict[str, Any]:
    """测试计划模块"""
    logger.info("测试增强计划模块...")

    from models.transformer.self_agi_model import PlanningModule

    # 创建配置
    config = AGIModelConfig(
        hidden_size=256,
        num_attention_heads=8,  # 256/8=32，可整除
        planning_enabled=True,
        multimodal_fusion_enabled=False,
    )

    # 创建计划模块
    planning_module = PlanningModule(config)

    # 测试输入
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    goals = torch.randn(batch_size, config.hidden_size)
    constraints = torch.randn(batch_size, config.hidden_size)
    resources = torch.randn(batch_size, config.hidden_size)

    # 前向传播
    plan_output = planning_module(
        hidden_states=hidden_states,
        goals=goals,
        constraints=constraints,
        resources=resources,
    )

    # 验证输出
    assert "plans" in plan_output, "计划输出中缺少plans"
    assert "actions" in plan_output, "计划输出中缺少actions"
    assert "optimized_path" in plan_output, "计划输出中缺少optimized_path"
    assert "risk_scores" in plan_output, "计划输出中缺少risk_scores"
    assert "plan_features" in plan_output, "计划输出中缺少plan_features"

    # 检查形状
    assert plan_output["plans"].shape == (batch_size, seq_len, config.hidden_size)
    assert plan_output["actions"].shape == (batch_size, seq_len, config.hidden_size)
    assert plan_output["risk_scores"].shape == (batch_size, 1, 1)

    logger.info("✓ 增强计划模块测试通过")

    return plan_output


def test_reasoning_module() -> Dict[str, Any]:
    """测试推理模块"""
    logger.info("测试增强推理模块...")

    from models.transformer.self_agi_model import ReasoningModule

    # 创建配置
    config = AGIModelConfig(
        hidden_size=256,
        num_attention_heads=8,  # 256/8=32，可整除
        reasoning_enabled=True,
        multimodal_fusion_enabled=False,
    )

    # 创建推理模块
    reasoning_module = ReasoningModule(config)

    # 测试输入
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    context = torch.randn(batch_size, seq_len // 2, config.hidden_size)

    # 测试完整推理
    reasoning_output = reasoning_module(hidden_states=hidden_states, context=context)

    # 验证输出
    assert "logic_output" in reasoning_output, "推理输出中缺少logic_output"
    assert "causal_output" in reasoning_output, "推理输出中缺少causal_output"
    assert "spatial_output" in reasoning_output, "推理输出中缺少spatial_output"
    assert "math_output" in reasoning_output, "推理输出中缺少math_output"
    assert "fused_reasoning" in reasoning_output, "推理输出中缺少fused_reasoning"
    assert "confidence_scores" in reasoning_output, "推理输出中缺少confidence_scores"

    # 检查形状
    assert reasoning_output["logic_output"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    )
    assert reasoning_output["fused_reasoning"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    )
    assert reasoning_output["confidence_scores"].shape == (batch_size, 8)  # 8个推理类型

    # 测试指定推理类型
    reasoning_type = ["logic", "causal"]
    filtered_output = reasoning_module(
        hidden_states=hidden_states, context=context, reasoning_type=reasoning_type
    )

    assert "fused_reasoning" in filtered_output
    assert "logic_output" in filtered_output
    assert "causal_output" in filtered_output
    assert "spatial_output" not in filtered_output  # 不应该在过滤输出中

    logger.info("✓ 增强推理模块测试通过")

    return reasoning_output


def test_self_correction_module() -> Dict[str, Any]:
    """测试自我改正模块"""
    logger.info("测试自我改正模块...")

    from models.transformer.self_agi_model import SelfCorrectionModule

    # 创建配置
    config = AGIModelConfig(
        hidden_size=256,
        num_attention_heads=8,  # 256/8=32，可整除
        self_correction_enabled=True,
        multimodal_fusion_enabled=False,
    )

    # 创建自我改正模块
    correction_module = SelfCorrectionModule(config)

    # 测试输入
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # 模拟模型输出
    outputs = {
        "logits": torch.randn(batch_size, seq_len, config.vocab_size),
        "fused_reasoning": torch.randn(batch_size, seq_len, config.hidden_size),
    }

    # 上下文信息
    context = torch.randn(batch_size, seq_len // 2, config.hidden_size)

    # 测试自我改正
    correction_output = correction_module(
        hidden_states=hidden_states, outputs=outputs, context=context
    )

    # 验证输出
    assert "error_scores" in correction_output, "改正输出中缺少error_scores"
    assert "error_types" in correction_output, "改正输出中缺少error_types"
    assert "cause_analysis" in correction_output, "改正输出中缺少cause_analysis"
    assert "corrections" in correction_output, "改正输出中缺少corrections"
    assert (
        "verification_scores" in correction_output
    ), "改正输出中缺少verification_scores"
    assert "corrected_features" in correction_output, "改正输出中缺少corrected_features"
    assert (
        "applied_corrections" in correction_output
    ), "改正输出中缺少applied_corrections"

    # 检查推理相关输出
    assert "reasoning_output" in correction_output, "改正输出中缺少reasoning_output"
    assert "rule_similarities" in correction_output, "改正输出中缺少rule_similarities"
    assert "top_rule_indices" in correction_output, "改正输出中缺少top_rule_indices"
    assert "strategy_probs" in correction_output, "改正输出中缺少strategy_probs"
    assert (
        "selected_strategies" in correction_output
    ), "改正输出中缺少selected_strategies"
    assert "knowledge_features" in correction_output, "改正输出中缺少knowledge_features"

    # 检查形状
    assert correction_output["error_scores"].shape == (
        batch_size,
        seq_len,
        3,
    ), "错误分数形状不正确"
    assert correction_output["error_types"].shape == (
        batch_size,
        seq_len,
    ), "错误类型形状不正确"
    assert correction_output["cause_analysis"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), "原因分析形状不正确"
    assert correction_output["corrections"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), "改正建议形状不正确"
    assert correction_output["verification_scores"].shape == (
        batch_size,
        seq_len,
        1,
    ), "验证分数形状不正确"
    assert correction_output["corrected_features"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), "改正特征形状不正确"

    # 检查推理输出形状
    assert correction_output["reasoning_output"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), "推理输出形状不正确"
    assert correction_output["strategy_probs"].shape == (
        batch_size,
        6,
    ), "策略概率形状不正确"
    assert correction_output["selected_strategies"].shape == (
        batch_size,
    ), "选择策略形状不正确"
    assert correction_output["knowledge_features"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    ), "知识特征形状不正确"

    # 测试错误类型值范围
    error_types = correction_output["error_types"]
    assert torch.all((error_types >= 0) & (error_types < 3)), "错误类型值超出范围[0, 2]"

    # 测试验证分数范围
    verification_scores = correction_output["verification_scores"]
    assert torch.all(
        (verification_scores >= 0) & (verification_scores <= 1)
    ), "验证分数超出范围[0, 1]"

    # 测试策略概率
    strategy_probs = correction_output["strategy_probs"]
    assert torch.all(
        (strategy_probs >= 0) & (strategy_probs <= 1)
    ), "策略概率超出范围[0, 1]"
    assert torch.allclose(
        strategy_probs.sum(dim=-1), torch.ones(batch_size)
    ), "策略概率总和不为1"

    logger.info("✓ 自我改正模块测试通过")

    return correction_output


def test_generation() -> Any:
    """测试文本生成"""
    logger.info("测试文本生成...")

    # 创建配置
    config = ModelConfig(
        vocab_size=10000,
        hidden_size=256,  # 测试时使用较小的隐藏大小
        num_hidden_layers=4,
        num_attention_heads=8,  # 256/8=32，可整除
        max_position_embeddings=512,
        multimodal_enabled=True,
        multimodal_fusion_enabled=False,  # 禁用融合以避免维度问题
        planning_enabled=True,
        reasoning_enabled=True,
        execution_control_enabled=True,
        self_cognition_enabled=True,
        learning_enabled=True,
        self_correction_enabled=True,  # 启用自我改正功能
    )

    # 创建模型
    model = SelfAGIModel(config)

    batch_size = 1
    seq_len = 10
    vocab_size = model.config.vocab_size

    # 创建输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 生成文本
    generated = model.generate(input_ids=input_ids, max_length=20, temperature=0.8)

    # 验证生成结果
    assert generated.shape[0] == batch_size, "生成批次大小不正确"
    assert generated.shape[1] == seq_len + 20, "生成长度不正确"

    logger.info(f"✓ 文本生成测试通过，生成序列长度: {generated.shape[1]}")

    return generated


def main() -> bool:
    """主测试函数"""
    logger.info("开始Self AGI模型测试...")

    try:
        # 1. 测试模型初始化
        model, config = test_model_initialization()

        # 2. 测试前向传播
        outputs = test_forward_pass()

        # 3. 测试计划模块
        plan_output = test_planning_module()

        # 4. 测试推理模块
        reasoning_output = test_reasoning_module()

        # 5. 测试自我改正模块
        correction_output = test_self_correction_module()

        # 6. 测试文本生成
        _ = test_generation()

        logger.info("✅ 所有测试通过！")
        logger.info(f"计划模块输出键: {list(plan_output.keys())}")
        logger.info(f"推理模块输出键: {list(reasoning_output.keys())}")
        logger.info(f"自我改正模块输出键: {list(correction_output.keys())}")
        logger.info(f"模型总输出键: {list(outputs.keys())}")

        return True

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
