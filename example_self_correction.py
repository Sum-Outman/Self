#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自我修证功能使用示例

本示例展示如何：
1. 使用深度思考推理引擎进行深度思考和自我修证
2. 使用训练系统的自我修证训练模式
3. 处理超出认知的问题

遵循项目代码规范，提供完整可运行的示例。
"""

import logging
import sys
from typing import Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_deep_thinking_engine():
    """示例1：深度思考推理引擎使用"""
    print("=" * 60)
    print("示例1: 深度思考推理引擎")
    print("=" * 60)
    
    try:
        # 导入深度思考引擎
        from models.deep_thinking_engine import DeepThinkingEngine
        
        # 创建引擎实例
        engine = DeepThinkingEngine({
            "max_thinking_steps": 8,
            "enable_reflection": True,
            "enable_correction": True,
        })
        
        # 示例问题
        problems = [
            "如何设计一个高效的AGI系统架构？",
            "量子计算对传统密码学有什么影响？",  # 超出认知的问题
            "如果时间旅行是可能的，会引发哪些伦理问题？",
        ]
        
        for i, problem in enumerate(problems, 1):
            print(f"\n问题 {i}: {problem}")
            print("-" * 40)
            
            # 深度思考
            result = engine.deep_think(problem, thinking_depth="deep")
            
            if result["success"]:
                conclusion = result["final_conclusion"]
                print(f"结论: {conclusion.get('answer', '无结论')[:100]}...")
                print(f"置信度: {conclusion.get('confidence', 0.0):.2f}")
                print(f"思考深度: {result.get('thinking_depth', '未知')}")
                print(f"问题类型: {result.get('problem_type', '未知')}")
                
                # 如果有学习建议
                if conclusion.get('has_correction', False):
                    print("✓ 经过自我修正")
            else:
                print(f"思考失败: {result.get('error', '未知错误')}")
        
        # 处理超出认知的问题
        print("\n" + "=" * 60)
        print("处理超出认知的问题示例")
        print("=" * 60)
        
        unknown_problem = "如何解释M理论中的11维时空结构？"
        print(f"问题: {unknown_problem}")
        
        unknown_result = engine.handle_unknown_problem(unknown_problem)
        
        if unknown_result["success"]:
            print(f"知识缺口确认: {unknown_result['knowledge_gap_acknowledged']}")
            print(f"合理性分数: {unknown_result.get('reasonableness_score', 0.0):.2f}")
            print(f"置信度: {unknown_result.get('confidence', 0.0):.2f}")
            print(f"建议: {unknown_result.get('recommendation', '无建议')}")
            
            # 显示学习建议
            suggestions = unknown_result.get('learning_suggestions', [])
            if suggestions:
                print("学习建议:")
                for suggestion in suggestions[:2]:  # 只显示前2个
                    print(f"  - {suggestion.get('suggestion', '')}")
        
        # 显示性能指标
        metrics = engine.get_metrics()
        print(f"\n引擎性能指标:")
        print(f"  处理问题总数: {metrics.get('total_problems_processed', 0)}")
        print(f"  挑战性问题: {metrics.get('challenging_problems', 0)}")
        print(f"  未知问题: {metrics.get('unknown_problems', 0)}")
        print(f"  平均思考步数: {metrics.get('avg_thinking_steps', 0.0):.1f}")
        
    except ImportError as e:
        print(f"导入深度思考引擎失败: {e}")
        print("请确保 models/deep_thinking_engine.py 存在")
    except Exception as e:
        print(f"深度思考引擎示例出错: {e}")


def example_self_correction_training():
    """示例2：自我修证训练模式使用"""
    print("\n" + "=" * 60)
    print("示例2: 自我修证训练模式")
    print("=" * 60)
    
    try:
        # 导入训练器
        from training.trainer import AGITrainer
        
        print("创建AGI训练器实例...")
        
        # 创建训练器配置
        from training.trainer import TrainingConfig
        
        config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-4,
            num_epochs=3,
            use_gpu=False,  # 为了示例，使用CPU
            logging_steps=10,
            save_steps=50,
        )
        
        # 创建训练器实例（需要模型参数，这里使用占位模型）
        # 注意：实际使用需要传入真实的模型
        import torch.nn as nn
        
        # 创建简单占位模型
        class PlaceholderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = PlaceholderModel()
        
        # 创建训练器实例
        trainer = AGITrainer(
            model=model,
            config=config,
            train_dataset=None,
            eval_dataset=None,
            from_scratch=False
        )
        
        print("设置自我修证训练模式...")
        
        # 设置自我修证训练模式
        trainer.set_training_mode("self_correction", {
            "self_correction": {
                "thinking_depth": "moderate",
                "correction_loss_weight": 0.15,
                "enable_reflection": True,
                "enable_correction": True,
                "integration_mode": "hybrid",
            }
        })
        
        print("自我修证训练模式已启用")
        print(f"训练模式: {trainer.training_mode}")
        print(f"自我修证训练启用: {trainer.self_correction_training_enabled}")
        
        if trainer.deep_thinking_engine is not None:
            print("深度思考引擎已初始化")
        else:
            print("深度思考引擎不可用，将使用简化版本")
        
        # 显示训练器配置
        print("\n训练器配置摘要:")
        print(f"  批量大小: {trainer.config.batch_size}")
        print(f"  学习率: {trainer.config.learning_rate}")
        print(f"  训练模式: {trainer.training_mode}")
        print(f"  修正损失权重: {getattr(trainer, 'correction_loss_weight', 0.1)}")
        
        # 模拟训练步骤（不实际训练）
        print("\n注意: 这是一个配置示例，实际训练需要数据集和模型。")
        print("要实际运行训练，需要准备数据集并调用 trainer.train() 方法。")
        
    except ImportError as e:
        print(f"导入训练器失败: {e}")
        print("请确保 training/trainer.py 存在且可导入")
    except Exception as e:
        print(f"自我修证训练示例出错: {e}")


def example_integration_with_existing_modules():
    """示例3：与现有模块集成"""
    print("\n" + "=" * 60)
    print("示例3: 与现有模块集成")
    print("=" * 60)
    
    try:
        # 尝试导入现有自我修正模块
        from models.transformer.cognitive.selfcorrectionmodule import SelfCorrectionModule
        from models.transformer.cognitive.selfcognitionmodule import SelfCognitionModule
        
        print("现有模块导入成功:")
        print(f"  - SelfCorrectionModule: 自我修正模块")
        print(f"  - SelfCognitionModule: 自我认知模块")
        print("\n深度思考引擎可与这些模块协同工作:")
        print("  1. 深度思考引擎提供高层反思和修正策略")
        print("  2. SelfCorrectionModule提供具体的错误检测和改正")
        print("  3. SelfCognitionModule提供自我认知和评估")
        print("\n集成方式:")
        print("  - 在训练中使用 self_correction 训练模式")
        print("  - 在推理中调用 deep_think() 方法处理复杂问题")
        print("  - 使用 handle_unknown_problem() 处理超出认知的问题")
        
    except ImportError as e:
        print(f"导入现有模块失败: {e}")
        print("部分模块可能不存在，但深度思考引擎仍可独立工作")


def practical_usage_scenarios():
    """示例4：实际使用场景"""
    print("\n" + "=" * 60)
    print("示例4: 实际使用场景")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "复杂问题求解",
            "description": "面对复杂、模糊或多方面的问题时",
            "approach": "使用 deep_think() 进行深度多步推理",
            "benefit": "得到更全面、深思熟虑的结论",
        },
        {
            "name": "错误检测与修正",
            "description": "在代码生成、决策制定等任务中",
            "approach": "结合自我修正模块和深度思考",
            "benefit": "自动检测并修正错误，提高输出质量",
        },
        {
            "name": "知识缺口处理",
            "description": "遇到超出当前知识范围的问题时",
            "approach": "使用 handle_unknown_problem()",
            "benefit": "诚实承认局限，提供合理推断和学习建议",
        },
        {
            "name": "持续自我改进",
            "description": "在训练过程中集成自我修证",
            "approach": "使用 self_correction 训练模式",
            "benefit": "模型学会自我反思和修正，持续改进",
        },
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n场景 {i}: {scenario['name']}")
        print(f"  描述: {scenario['description']}")
        print(f"  方法: {scenario['approach']}")
        print(f"  优势: {scenario['benefit']}")


def main():
    """主函数"""
    print("自我修证功能使用示例")
    print("=" * 60)
    print("本示例展示增强后的自我修证功能，包括:")
    print("1. 深度思考推理引擎")
    print("2. 自我反思和自我修正机制")
    print("3. 超出认知问题处理")
    print("4. 集成到训练系统的自我修证训练模式")
    print("=" * 60)
    
    try:
        # 运行各个示例
        example_deep_thinking_engine()
        example_self_correction_training()
        example_integration_with_existing_modules()
        practical_usage_scenarios()
        
        print("\n" + "=" * 60)
        print("示例运行完成")
        print("=" * 60)
        print("\n关键实现要点:")
        print("1. 深度思考引擎: models/deep_thinking_engine.py")
        print("2. 自我修证训练: training/trainer.py 中的 self_correction 模式")
        print("3. 辅助方法: _generate_thinking_signals, _compute_self_correction_loss")
        print("4. 现有模块: SelfCorrectionModule, SelfCognitionModule")
        print("\n使用步骤:")
        print("1. 导入深度思考引擎: from models.deep_thinking_engine import DeepThinkingEngine")
        print("2. 创建实例: engine = DeepThinkingEngine(config)")
        print("3. 深度思考: result = engine.deep_think(problem)")
        print("4. 或设置训练模式: trainer.set_training_mode('self_correction', config)")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())