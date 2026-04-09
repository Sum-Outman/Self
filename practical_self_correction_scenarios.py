#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实际应用场景示例：自我修证功能

本示例展示自我修证功能在实际应用中的各种场景：
1. 不同信号生成模式对比
2. 缓存机制效果演示
3. 动态思考深度调整
4. 损失组件配置影响
5. 调度器行为可视化

这些示例帮助理解如何在实际项目中配置和使用自我修证功能。
"""

import logging
import sys
import time
from typing import Dict, Any, List
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 降低日志级别以简化输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PracticalSelfCorrectionScenarios:
    """实际应用场景展示类"""
    
    def __init__(self):
        """初始化场景展示器"""
        self.scenarios = []
        self.results = {}
    
    def scenario_1_signal_generation_modes(self):
        """场景1：不同信号生成模式对比"""
        print("=" * 70)
        print("场景1: 不同信号生成模式对比")
        print("=" * 70)
        
        try:
            from models.deep_thinking_engine import DeepThinkingEngine
            from training.trainer import AGITrainer, TrainingConfig
            
            # 创建简单模型用于训练器
            import torch.nn as nn
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)
                
                def forward(self, x):
                    return {"logits": self.linear(x)}
            
            model = SimpleModel()
            
            # 测试三种信号生成模式
            modes = ["text_based", "metadata_based", "automatic"]
            
            for mode in modes:
                print(f"\n--- 信号生成模式: {mode} ---")
                
                # 创建训练配置
                config = TrainingConfig(
                    batch_size=2,
                    learning_rate=1e-4,
                    num_epochs=1,
                    use_gpu=False,
                )
                
                # 创建训练器
                trainer = AGITrainer(
                    model=model,
                    config=config,
                    train_dataset=None,
                    eval_dataset=None,
                    from_scratch=False
                )
                
                # 启用自我修证，指定信号生成模式
                trainer.set_training_mode("self_correction", {
                    "self_correction": {
                        "thinking_depth": "moderate",
                        "correction_loss_weight": 0.1,
                        "signal_generation_mode": mode,
                        "enable_cache": False,  # 禁用缓存以简化测试
                    }
                })
                
                # 验证配置
                print(f"  训练器信号生成模式: {trainer.signal_generation_mode}")
                print(f"  自我修证训练启用: {trainer.self_correction_training_enabled}")
                
                # 测试信号生成
                import torch
                batch = {
                    "input_ids": torch.zeros((2, 10), dtype=torch.long),
                    "attention_mask": torch.ones((2, 10), dtype=torch.long),
                }
                
                # 注意：实际信号生成需要深度思考引擎
                if trainer.deep_thinking_engine is not None:
                    print(f"  深度思考引擎可用")
                else:
                    print(f"  深度思考引擎不可用")
                
            print(f"\n总结: 不同信号生成模式适用于不同的数据格式:")
            print(f"  - text_based: 适用于文本密集型任务（如NLP）")
            print(f"  - metadata_based: 适用于复杂元数据分析")
            print(f"  - automatic: 自动选择最适合的模式")
            
        except ImportError as e:
            print(f"导入失败: {e}")
            print("请确保相关模块已安装")
        except Exception as e:
            print(f"场景1执行出错: {e}")
    
    def scenario_2_cache_mechanism_demo(self):
        """场景2：缓存机制效果演示"""
        print("\n" + "=" * 70)
        print("场景2: 缓存机制效果演示")
        print("=" * 70)
        
        try:
            from models.deep_thinking_engine import DeepThinkingEngine
            
            # 创建带缓存的深度思考引擎
            cache_config = {
                "max_thinking_steps": 5,
                "enable_reflection": True,
                "enable_correction": True,
                "enable_cache": True,
                "cache_size": 10,  # 小缓存以便观察
            }
            
            engine = DeepThinkingEngine(cache_config)
            
            # 测试问题
            test_problems = [
                "什么是机器学习？",
                "什么是深度学习？",
                "什么是机器学习？",  # 重复问题，应该命中缓存
                "什么是强化学习？",
                "什么是机器学习？",  # 再次重复，应该命中缓存
            ]
            
            cache_hits = 0
            cache_misses = 0
            
            for i, problem in enumerate(test_problems, 1):
                print(f"\n问题 {i}: {problem}")
                
                # 测量思考时间
                start_time = time.time()
                result = engine.deep_think(problem, thinking_depth="moderate")
                end_time = time.time()
                
                thinking_time = end_time - start_time
                
                if result["success"]:
                    confidence = result.get("final_conclusion", {}).get("confidence", 0.0)
                    
                    # 检查是否从缓存返回
                    if i > 1 and problem in [test_problems[0], test_problems[2]]:
                        # 重复问题，可能来自缓存
                        print(f"  置信度: {confidence:.2f}")
                        print(f"  思考时间: {thinking_time:.3f}秒")
                        if thinking_time < 0.1:  # 非常快，可能是缓存
                            print(f"  ✓ 可能命中缓存 (时间: {thinking_time:.3f}s)")
                            cache_hits += 1
                        else:
                            cache_misses += 1
                    else:
                        print(f"  置信度: {confidence:.2f}")
                        print(f"  思考时间: {thinking_time:.3f}秒")
                        cache_misses += 1
                else:
                    print(f"  思考失败: {result.get('error', '未知错误')}")
            
            # 获取缓存指标
            metrics = engine.get_metrics()
            print(f"\n缓存性能指标:")
            print(f"  总处理问题: {metrics.get('total_problems_processed', 0)}")
            print(f"  缓存命中估算: {cache_hits}")
            print(f"  缓存未命中: {cache_misses}")
            print(f"  估计命中率: {cache_hits/(cache_hits+cache_misses)*100:.1f}%")
            
            print(f"\n总结: 缓存机制可以显著加速重复问题的处理")
            print(f"  - 第一次处理: 完整思考流程")
            print(f"  - 重复问题: 从缓存快速返回结果")
            print(f"  - 缓存大小: {cache_config['cache_size']} 个条目")
            
        except ImportError as e:
            print(f"导入失败: {e}")
        except Exception as e:
            print(f"场景2执行出错: {e}")
    
    def scenario_3_dynamic_thinking_adjustment(self):
        """场景3：动态思考深度调整演示"""
        print("\n" + "=" * 70)
        print("场景3: 动态思考深度调整演示")
        print("=" * 70)
        
        try:
            from training.trainer import AGITrainer, TrainingConfig
            
            # 创建简单模型
            import torch.nn as nn
            import torch
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)
                
                def forward(self, x):
                    return {"logits": self.linear(x)}
            
            model = SimpleModel()
            
            # 创建训练配置，启用动态调整
            config = TrainingConfig(
                batch_size=2,
                learning_rate=1e-4,
                num_epochs=1,
                use_gpu=False,
            )
            
            trainer = AGITrainer(
                model=model,
                config=config,
                train_dataset=None,
                eval_dataset=None,
                from_scratch=False
            )
            
            # 启用自我修证，启用动态调整
            trainer.set_training_mode("self_correction", {
                "self_correction": {
                    "thinking_depth": "moderate",
                    "correction_loss_weight": 0.1,
                    "dynamic_thinking_adjustment": True,
                    "max_thinking_steps": 10,
                }
            })
            
            print(f"基础配置:")
            print(f"  基础思考深度: moderate")
            print(f"  基础思考步数: {trainer.thinking_depth_steps}")
            print(f"  动态调整启用: {trainer.dynamic_thinking_adjustment}")
            
            # 模拟不同复杂度的批次
            simple_batch = {"input_ids": torch.zeros((2, 5), dtype=torch.long)}
            complex_batch = {
                "input_ids": torch.zeros((2, 100), dtype=torch.long),
                "attention_mask": torch.ones((2, 100), dtype=torch.long),
                "labels": torch.zeros((2, 100), dtype=torch.long),
            }
            
            # 测试批次复杂度估计（如果可用）
            if hasattr(trainer, '_estimate_batch_complexity'):
                simple_complexity = trainer._estimate_batch_complexity(simple_batch)
                complex_complexity = trainer._estimate_batch_complexity(complex_batch)
                
                print(f"\n批次复杂度估计:")
                print(f"  简单批次复杂度: {simple_complexity:.2f}")
                print(f"  复杂批次复杂度: {complex_complexity:.2f}")
                print(f"  复杂度差异: {complex_complexity - simple_complexity:.2f}")
                
                # 动态调整逻辑
                print(f"\n动态调整逻辑:")
                print(f"  当批次复杂度 > 0.7 且训练进度 > 0.5 时:")
                print(f"    - 思考深度: moderate -> extreme")
                print(f"    - 思考步数: 增加50%")
                print(f"  当批次复杂度 > 0.5 时:")
                print(f"    - 思考深度: 保持 moderate 或 deep")
                print(f"    - 思考步数: 保持基础值")
                print(f"  当训练进度 < 0.3 时:")
                print(f"    - 思考深度: moderate -> shallow")
                print(f"    - 思考步数: 减少30%")
            
            print(f"\n总结: 动态思考深度调整根据批次复杂度和训练进度优化资源使用")
            print(f"  - 简单批次/训练早期: 减少思考深度以加速")
            print(f"  - 复杂批次/训练后期: 增加思考深度以提高质量")
            print(f"  - 中等复杂度: 保持平衡配置")
            
        except ImportError as e:
            print(f"导入失败: {e}")
        except Exception as e:
            print(f"场景3执行出错: {e}")
    
    def scenario_4_loss_components_analysis(self):
        """场景4：损失组件配置分析"""
        print("\n" + "=" * 70)
        print("场景4: 损失组件配置分析")
        print("=" * 70)
        
        try:
            from training.trainer import AGITrainer, TrainingConfig
            
            # 创建简单模型
            import torch.nn as nn
            import torch
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)
                
                def forward(self, x):
                    return {"logits": self.linear(x)}
            
            model = SimpleModel()
            
            # 测试不同的损失组件配置
            configs = [
                {
                    "name": "仅反思",
                    "config": {
                        "enable_reflection": True,
                        "enable_correction": False,
                        "reflection_weight": 0.3,
                        "correction_weight": 0.0,
                    },
                    "description": "专注于反思能力训练"
                },
                {
                    "name": "仅修正",
                    "config": {
                        "enable_reflection": False,
                        "enable_correction": True,
                        "reflection_weight": 0.0,
                        "correction_weight": 0.3,
                    },
                    "description": "专注于修正能力训练"
                },
                {
                    "name": "平衡配置",
                    "config": {
                        "enable_reflection": True,
                        "enable_correction": True,
                        "reflection_weight": 0.1,
                        "correction_weight": 0.2,
                    },
                    "description": "反思和修正平衡训练"
                },
                {
                    "name": "强化对齐",
                    "config": {
                        "enable_reflection": True,
                        "enable_correction": True,
                        "reflection_weight": 0.1,
                        "correction_weight": 0.2,
                        "alignment_weight": 0.15,
                    },
                    "description": "强化自我认知对齐"
                },
            ]
            
            for config_info in configs:
                print(f"\n--- 配置: {config_info['name']} ---")
                print(f"描述: {config_info['description']}")
                
                # 创建训练器
                trainer_config = TrainingConfig(
                    batch_size=2,
                    learning_rate=1e-4,
                    num_epochs=1,
                    use_gpu=False,
                )
                
                trainer = AGITrainer(
                    model=model,
                    config=trainer_config,
                    train_dataset=None,
                    eval_dataset=None,
                    from_scratch=False
                )
                
                # 启用自我修证
                trainer.set_training_mode("self_correction", {
                    "self_correction": {
                        "thinking_depth": "shallow",
                        "correction_loss_weight": 0.1,
                        "loss_components": {
                            "reflection_weight": config_info["config"]["reflection_weight"],
                            "correction_weight": config_info["config"]["correction_weight"],
                            "alignment_weight": config_info["config"].get("alignment_weight", 0.05),
                        },
                        "enable_reflection": config_info["config"]["enable_reflection"],
                        "enable_correction": config_info["config"]["enable_correction"],
                    }
                })
                
                # 显示配置
                print(f"  反思组件权重: {trainer.reflection_component_weight}")
                print(f"  修正组件权重: {trainer.correction_component_weight}")
                print(f"  对齐组件权重: {trainer.alignment_component_weight}")
                print(f"  深度奖励权重: {trainer.depth_reward_component_weight}")
                
                # 模拟损失计算
                outputs = {"logits": torch.randn((2, 5))}
                batch = {"input_ids": torch.zeros((2, 10), dtype=torch.long)}
                thinking_signals = {
                    "confidence": 0.8,
                    "thinking_steps": 3,
                    "reflection_signals": {"confidence_impact": 0.2},
                    "correction_signals": {"effectiveness_score": 0.7},
                }
                
                if trainer.self_correction_training_enabled:
                    loss = trainer._compute_self_correction_loss(outputs, batch, thinking_signals)
                    print(f"  示例损失值: {loss.item():.4f}")
            
            print(f"\n总结: 不同损失组件配置适用于不同训练目标")
            print(f"  - 仅反思: 提高模型自我反思能力")
            print(f"  - 仅修正: 提高模型错误修正能力")
            print(f"  - 平衡配置: 综合训练反思和修正能力")
            print(f"  - 强化对齐: 加强自我认知一致性")
            
        except ImportError as e:
            print(f"导入失败: {e}")
        except Exception as e:
            print(f"场景4执行出错: {e}")
    
    def scenario_5_scheduler_visualization(self):
        """场景5：调度器行为可视化"""
        print("\n" + "=" * 70)
        print("场景5: 调度器行为可视化")
        print("=" * 70)
        
        try:
            from training.trainer import AGITrainer, TrainingConfig
            
            # 创建简单模型
            import torch.nn as nn
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)
                
                def forward(self, x):
                    return {"logits": self.linear(x)}
            
            model = SimpleModel()
            
            # 测试不同的调度器配置
            scheduler_configs = [
                {
                    "name": "标准调度",
                    "config": {
                        "warmup_steps": 100,
                        "cool_down_steps": 500,
                        "max_correction_weight": 0.3,
                        "min_correction_weight": 0.01,
                    },
                    "description": "标准预热-稳定-冷却调度"
                },
                {
                    "name": "快速预热",
                    "config": {
                        "warmup_steps": 50,
                        "cool_down_steps": 300,
                        "max_correction_weight": 0.4,
                        "min_correction_weight": 0.02,
                    },
                    "description": "快速预热，较早开始冷却"
                },
                {
                    "name": "长稳定期",
                    "config": {
                        "warmup_steps": 100,
                        "cool_down_steps": 1000,
                        "max_correction_weight": 0.25,
                        "min_correction_weight": 0.01,
                    },
                    "description": "长稳定期，缓慢冷却"
                },
            ]
            
            print("调度器行为模拟（总修正损失权重 = 基础权重 × 调度器乘数）:")
            
            for scheduler_info in scheduler_configs:
                print(f"\n--- 调度器: {scheduler_info['name']} ---")
                print(f"描述: {scheduler_info['description']}")
                
                # 创建训练器
                config = TrainingConfig(
                    batch_size=2,
                    learning_rate=1e-4,
                    num_epochs=1,
                    use_gpu=False,
                )
                
                trainer = AGITrainer(
                    model=model,
                    config=config,
                    train_dataset=None,
                    eval_dataset=None,
                    from_scratch=False
                )
                
                # 启用自我修证，配置调度器
                trainer.set_training_mode("self_correction", {
                    "self_correction": {
                        "thinking_depth": "moderate",
                        "correction_loss_weight": 0.1,  # 基础权重
                        "adaptive_scheduling": scheduler_info["config"]
                    }
                })
                
                # 模拟不同训练步数的权重
                test_steps = [0, 25, 50, 100, 250, 500, 750, 1000]
                weights = []
                
                for step in test_steps:
                    weight = trainer.correction_loss_scheduler(step)
                    weights.append(weight)
                    print(f"  步数 {step:4d}: 权重 = {weight:.4f}")
                
                # 简单分析
                max_weight = max(weights)
                min_weight = min(weights)
                avg_weight = sum(weights) / len(weights)
                
                print(f"  权重范围: {min_weight:.4f} - {max_weight:.4f}")
                print(f"  平均权重: {avg_weight:.4f}")
                
                # 可视化指示
                if scheduler_info["name"] == "快速预热":
                    print(f"  → 特征: 快速达到峰值，较早开始衰减")
                elif scheduler_info["name"] == "长稳定期":
                    print(f"  → 特征: 长稳定平台期，缓慢衰减")
                else:
                    print(f"  → 特征: 标准三阶段调度")
            
            print(f"\n总结: 调度器控制自我修证损失权重在训练过程中的变化")
            print(f"  - 预热阶段: 权重从最小值增加到最大值")
            print(f"  - 稳定阶段: 权重保持最大值")
            print(f"  - 冷却阶段: 权重逐渐减少到最小值")
            print(f"  - 目标: 平衡训练稳定性和自我修证效果")
            
        except ImportError as e:
            print(f"导入失败: {e}")
        except Exception as e:
            print(f"场景5执行出错: {e}")
    
    def run_all_scenarios(self):
        """运行所有场景"""
        print("自我修证功能实际应用场景展示")
        print("=" * 70)
        print("本展示演示自我修证功能在实际应用中的配置和使用方法。")
        print("每个场景展示一个关键功能及其实际应用价值。")
        print("=" * 70)
        
        try:
            # 运行各个场景
            self.scenario_1_signal_generation_modes()
            self.scenario_2_cache_mechanism_demo()
            self.scenario_3_dynamic_thinking_adjustment()
            self.scenario_4_loss_components_analysis()
            self.scenario_5_scheduler_visualization()
            
            print("\n" + "=" * 70)
            print("所有场景演示完成")
            print("=" * 70)
            
            print("\n关键要点总结:")
            print("1. 信号生成模式: 根据数据特性选择合适模式")
            print("2. 缓存机制: 加速重复问题处理，减少计算开销")
            print("3. 动态调整: 根据批次复杂度和训练进度优化资源")
            print("4. 损失组件: 根据训练目标配置不同权重")
            print("5. 调度器: 控制自我修证损失在训练中的变化")
            
            print("\n实际应用建议:")
            print("1. 初期: 使用automatic模式，中等思考深度")
            print("2. 优化: 根据任务特点调整损失组件权重")
            print("3. 生产: 启用缓存和动态调整以提高效率")
            print("4. 监控: 跟踪思考深度、缓存命中率等指标")
            
        except Exception as e:
            print(f"场景演示出错: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    # 检查是否安装了必要的包
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
    except ImportError:
        print("错误: PyTorch未安装。请先安装PyTorch。")
        print("安装命令: pip install torch")
        return 1
    
    # 创建场景展示器并运行
    scenarios = PracticalSelfCorrectionScenarios()
    scenarios.run_all_scenarios()
    
    return 0


if __name__ == "__main__":
    # 在if __name__中导入torch以避免在导入测试时提前导入
    import torch
    
    sys.exit(main())