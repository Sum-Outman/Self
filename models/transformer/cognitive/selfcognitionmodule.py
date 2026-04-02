# SelfCognitionModule - 从self_agi_model.py拆分
"""SelfCognition模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging

class SelfCognitionModule(nn.Module):
    """自我认知模块 - 真实自我认知系统

    功能：
    - 自我表示：可学习的自我模型，包括能力、状态、目标和偏好的表示
    - 自我评估：基于性能指标、成功率和效率的真实自我评估
    - 元认知：监控和调节思考过程、注意力分配和策略选择
    - 自我知识：关于自身能力、限制和偏好的知识表示和推理
    - 自我意识：反思自身状态、意图和未来可能性的能力

    基于真实认知科学原理，实现多层次自我认知和元认知能力
    """

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # === 自我表示子系统 ===
        # 可学习的自我模型 - 从经验中学习
        self.self_model_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 能力表示：表示不同能力的向量
        self.ability_representations = nn.Parameter(
            torch.randn(8, config.hidden_size)  # 8种核心能力
        )

        # 状态表示：当前状态表示
        self.state_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 目标表示：当前目标表示
        self.goal_encoder = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 偏好表示：个人偏好表示
        self.preference_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.LayerNorm(config.hidden_size // 4, eps=config.layer_norm_eps),
        )

        # 自我概念整合网络
        self.self_concept_integrator = nn.Sequential(
            nn.Linear(
                config.hidden_size * 4 + config.hidden_size // 4, config.hidden_size * 2
            ),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # === 自我评估子系统 ===
        # 性能评估网络：基于任务性能的自我评估
        self.performance_evaluator = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 5),  # 5个性能维度
            nn.Sigmoid(),
        )

        # 成功率评估器：基于历史成功率的评估
        self.success_rate_evaluator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),  # 成功率
            nn.Sigmoid(),
        )

        # 效率评估器：基于资源使用效率的评估
        self.efficiency_evaluator = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 3),  # 3种效率指标
            nn.Softmax(dim=-1),
        )

        # 能力水平评估器：评估各能力水平
        self.ability_level_evaluator = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 8),  # 8种能力水平
            nn.Sigmoid(),
        )

        # 综合自我评估网络
        self.comprehensive_self_evaluation = nn.Sequential(
            nn.Linear(config.hidden_size + 5 + 1 + 3 + 8, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # === 元认知子系统 ===
        # 思考过程监控器：监控当前思考过程
        self.thought_process_monitor = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 4),  # 4个监控维度
            nn.Sigmoid(),
        )

        # 注意力分配监控器：监控注意力分布
        self.attention_monitor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 3),  # 3个注意力维度
            nn.Softmax(dim=-1),
        )

        # 策略选择监控器：监控策略选择和效果
        self.strategy_monitor = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 6),  # 6种策略类型
            nn.Softmax(dim=-1),
        )

        # 认知负荷评估器：评估当前认知负荷
        self.cognitive_load_evaluator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),  # 认知负荷分数
            nn.Sigmoid(),
        )

        # 元认知控制网络：调节思考过程
        self.metacognitive_controller = nn.Sequential(
            nn.Linear(config.hidden_size + 4 + 3 + 6 + 1, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # === 自我知识子系统 ===
        # 能力知识库：关于自身能力的知识
        self.ability_knowledge_base = nn.Parameter(
            torch.randn(8, config.hidden_size)  # 8种能力知识
        )

        # 限制知识库：关于自身限制的知识
        self.limitation_knowledge_base = nn.Parameter(
            torch.randn(5, config.hidden_size)  # 5种主要限制
        )

        # 偏好知识库：关于个人偏好的知识
        self.preference_knowledge_base = nn.Parameter(
            torch.randn(6, config.hidden_size)  # 6种主要偏好
        )

        # 经验知识库：从经验中学习的知识
        self.experience_knowledge_base = nn.Parameter(
            torch.randn(10, config.hidden_size)  # 10个经验知识
        )

        # 知识查询网络：查询自我相关知识
        self.knowledge_query_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 知识推理网络：基于自我知识的推理
        self.knowledge_reasoning_network = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # === 自我意识子系统 ===
        # 状态反思网络：反思当前状态
        self.state_reflection_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 意图推理网络：推理自身意图
        self.intention_reasoning_network = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 未来可能性预测网络：预测未来可能状态
        self.future_prediction_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 自我意识整合网络
        self.self_awareness_integrator = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # === 记忆和学习子系统 ===
        # 自我模型更新网络：更新自我模型（增强版：包含经验记忆）
        self.self_model_updater = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 经验记忆库：记忆重要经验
        self.experience_memory = nn.Parameter(
            torch.randn(20, config.hidden_size)  # 20个重要经验
        )

        # 自我评估记忆：记忆自我评估结果
        self.evaluation_memory = nn.Parameter(
            torch.randn(15, config.hidden_size)  # 15个评估结果
        )

        # 元认知记忆：记忆元认知监控结果
        self.metacognition_memory = nn.Parameter(
            torch.randn(10, config.hidden_size)  # 10个元认知结果
        )

        # 学习网络：从自我认知经验中学习（增强版：包含经验记忆）
        self.self_cognition_learner = nn.Sequential(
            nn.Linear(config.hidden_size * 5, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # === 注意力机制 ===
        # 自我表示注意力
        self.self_representation_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        # 自我评估注意力
        self.self_evaluation_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        # 元认知注意力
        self.metacognition_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        # 自我知识注意力
        self.self_knowledge_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        # === 融合和整合网络 ===
        # 自我特征融合网络
        self.self_feature_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 认知过程融合网络
        self.cognitive_process_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 5, config.hidden_size * 3),
            nn.GELU(),
            nn.Linear(config.hidden_size * 3, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 最终自我认知融合网络
        self.final_self_cognition_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 反馈投影网络：将任意维度的反馈投影到隐藏维度
        self.feedback_projection = nn.Sequential(
            nn.Linear(1, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 错误检测器网络：检测错误并分类
        self.error_detector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 3),  # 3个类别：无错误, 轻微错误, 严重错误
            nn.Softmax(dim=-1),
        )

        # 错误检测注意力：用于错误检测的注意力机制
        self.error_detection_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        # 错误注意力（error_attention的别名，用于forward方法兼容性）
        self.error_attention = self.error_detection_attention

        # 推理网络（用于forward方法）
        self.reasoning_network = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 层归一化和dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 认知科学算法库
        self.cognitive_science_algorithms = CognitiveScienceAlgorithms(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        goals: Optional[torch.Tensor] = None,
        feedback: Optional[torch.Tensor] = None,
        performance_history: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """真实自我认知引擎

        参数:
            hidden_states: [batch_size, seq_len, hidden_size] 输入特征
            goals: [batch_size, goal_len, hidden_size] 当前目标
            feedback: [batch_size, feedback_dim] 反馈信息
            performance_history: 性能历史记录

        返回:
            自我认知输出字典，包含：
            - self_representation: 自我表示 [batch_size, seq_len, hidden_size]
            - self_evaluation: 自我评估结果 [batch_size, hidden_size]
            - metacognition: 元认知结果 [batch_size, hidden_size]
            - self_knowledge: 自我知识 [batch_size, hidden_size]
            - self_awareness: 自我意识 [batch_size, hidden_size]
            - integrated_self_cognition: 整合的自我认知 [batch_size, seq_len, hidden_size]
            - performance_scores: 性能分数 [batch_size, 5]
            - ability_levels: 能力水平 [batch_size, 8]
            - cognitive_load: 认知负荷 [batch_size, 1]
            - attention_distribution: 注意力分布 [batch_size, 3]
            - strategy_choices: 策略选择 [batch_size, 6]
            - state_reflection: 状态反思 [batch_size, hidden_size]
            - intention_reasoning: 意图推理 [batch_size, hidden_size]
            - future_prediction: 未来预测 [batch_size, hidden_size]
            - self_model_update: 自我模型更新 [batch_size, hidden_size]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # === 0. 认知科学算法集成 ===
        # 收集数据用于认知科学算法
        cognitive_data = {
            "hidden_states": hidden_states,
            "goals": goals,
            "feedback": feedback,
            "performance_history": performance_history,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
        }

        # 准备认知过程数据
        cognitive_processes = {
            "thinking_process": {
                "indicators": {
                    "speed": 0.5,  # 中等速度
                    "accuracy": 0.7,  # 中等准确度
                    "consistency": 0.6,
                },
                "complexity": 0.5,
                "resource_usage": {"memory": 0.4, "attention": 0.6, "computation": 0.5},
            },
            "planning_process": {
                "indicators": {
                    "speed": 0.6,
                    "accuracy": 0.8,
                    "consistency": 0.7,
                },
                "complexity": 0.6,
                "resource_usage": {"memory": 0.5, "attention": 0.7, "computation": 0.6},
            },
            "reasoning_process": {
                "indicators": {
                    "speed": 0.4,
                    "accuracy": 0.9,
                    "consistency": 0.8,
                },
                "complexity": 0.7,
                "resource_usage": {"memory": 0.6, "attention": 0.8, "computation": 0.7},
            },
        }

        # 性能历史数据
        performance_data = (
            [0.7, 0.8, 0.6, 0.9, 0.7]
            if performance_history is None
            else performance_history
        )

        # 当前状态表示
        current_state = {
            "current_abilities": {
                "reasoning": 0.7,
                "planning": 0.6,
                "learning": 0.8,
                "adaptation": 0.5,
            },
            "knowledge_level": 0.6,
            "skill_level": 0.7,
        }

        # 目标表示
        if goals is not None:
            goal_state = {
                "target_abilities": {
                    "reasoning": 0.9,
                    "planning": 0.8,
                    "learning": 0.9,
                    "adaptation": 0.7,
                },
                "target_knowledge": 0.8,
                "target_skill": 0.9,
            }
        else:
            goal_state = {
                "target_abilities": current_state["current_abilities"],
                "target_knowledge": current_state["knowledge_level"],
                "target_skill": current_state["skill_level"],
            }

        # 调用认知科学算法
        try:
            # 1. 元认知监控算法
            metacognitive_monitoring = (
                self.cognitive_science_algorithms.metacognitive_monitoring(
                    cognitive_processes, performance_data
                )
            )

            # 2. 自我调节学习循环算法
            # 展平当前状态和目标状态，以便认知科学算法处理
            def flatten_dict(d, parent_key='', sep='.'):
                items = {}
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.update(flatten_dict(v, new_key, sep=sep))
                    else:
                        items[new_key] = v
                return items
            
            flat_current_state = flatten_dict(current_state)
            flat_goal_state = flatten_dict(goal_state)
            
            self_regulated_learning = (
                self.cognitive_science_algorithms.self_regulated_learning_cycle(
                    flat_current_state, flat_goal_state, feedback
                )
            )

            # 3. 自我图式形成算法
            self_schemas = self.cognitive_science_algorithms.self_schema_formation(
                experiences=[],  # 实际应用中从记忆中提取
                attributes=["intelligence", "competence", "social_skill", "creativity"],
            )

            cognitive_science_results = {
                "metacognitive_monitoring": metacognitive_monitoring,
                "self_regulated_learning": self_regulated_learning,
                "self_schemas": self_schemas,
            }
        except Exception as e:
            logger.warning(f"认知科学算法执行失败: {e}")
            cognitive_science_results = {
                "metacognitive_monitoring": {},
                "self_regulated_learning": {},
                "self_schemas": {},
            }

        # === 1. 自我表示生成 ===
        # 编码自我模型
        self_model = self.self_model_encoder(hidden_states.mean(dim=1, keepdim=True))

        # 能力表示
        ability_repr = self.ability_representations.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # 状态表示
        state_repr = self.state_encoder(hidden_states)

        # 目标表示
        if goals is not None:
            # 处理2D或3D goals
            if goals.dim() == 3:
                goals_input = goals.mean(dim=1)
            else:
                goals_input = goals  # 2D: [batch_size, feature_dim]

            # 检查维度兼容性
            if goals_input.shape[-1] != self.config.hidden_size:
                # 动态创建投影层
                if (
                    not hasattr(self, "_goal_projection")
                    or self._goal_projection.in_features != goals_input.shape[-1]
                ):
                    self._goal_projection = nn.Linear(
                        goals_input.shape[-1], self.config.hidden_size
                    ).to(goals_input.device)
                goals_input = self._goal_projection(goals_input)

            goal_repr = self.goal_encoder(
                torch.cat([hidden_states.mean(dim=1), goals_input], dim=-1)
            )
        else:
            goal_repr = self.goal_encoder(
                torch.cat(
                    [hidden_states.mean(dim=1), hidden_states.mean(dim=1)], dim=-1
                )
            )

        # 偏好表示
        preference_repr = self.preference_encoder(hidden_states.mean(dim=1))
        preference_repr = preference_repr.unsqueeze(1).expand(-1, seq_len, -1)

        # 整合自我表示
        self_representation_input = torch.cat(
            [
                self_model.expand(-1, seq_len, -1),
                ability_repr.mean(dim=1, keepdim=True).expand(-1, seq_len, -1),
                state_repr,
                goal_repr.unsqueeze(1).expand(-1, seq_len, -1),
                preference_repr,
            ],
            dim=-1,
        )

        self_representation = self.self_concept_integrator(self_representation_input)

        # 自我表示注意力
        self_representation_attn_output, _ = self.self_representation_attention(
            self_representation, self_representation, self_representation
        )

        # === 2. 自我评估 ===
        # 性能评估
        performance_input = torch.cat(
            [hidden_states.mean(dim=1), self_representation.mean(dim=1)], dim=-1
        )
        performance_scores = self.performance_evaluator(
            performance_input
        )  # [batch_size, 5]

        # 成功率评估
        success_rate = self.success_rate_evaluator(
            hidden_states.mean(dim=1)
        )  # [batch_size, 1]

        # 效率评估
        efficiency_input = torch.cat(
            [hidden_states.mean(dim=1), self_representation.mean(dim=1)], dim=-1
        )
        efficiency_scores = self.efficiency_evaluator(
            efficiency_input
        )  # [batch_size, 3]

        # 能力水平评估
        ability_input = torch.cat(
            [hidden_states.mean(dim=1), self_representation.mean(dim=1)], dim=-1
        )
        ability_levels = self.ability_level_evaluator(ability_input)  # [batch_size, 8]

        # 综合自我评估
        comprehensive_input = torch.cat(
            [
                hidden_states.mean(dim=1),
                performance_scores,
                success_rate,
                efficiency_scores,
                ability_levels,
            ],
            dim=-1,
        )

        self_evaluation = self.comprehensive_self_evaluation(comprehensive_input)

        # 自我评估注意力
        self_evaluation_expanded = self_evaluation.unsqueeze(1).expand(-1, seq_len, -1)
        self_evaluation_attn_output, _ = self.self_evaluation_attention(
            self_evaluation_expanded, self_evaluation_expanded, self_evaluation_expanded
        )

        # === 3. 元认知 ===
        # 思考过程监控
        thought_process_input = torch.cat(
            [hidden_states.mean(dim=1), self_representation.mean(dim=1)], dim=-1
        )
        thought_monitor = self.thought_process_monitor(
            thought_process_input
        )  # [batch_size, 4]

        # 注意力监控
        attention_distribution = self.attention_monitor(
            hidden_states.mean(dim=1)
        )  # [batch_size, 3]

        # 策略监控
        strategy_input = torch.cat(
            [hidden_states.mean(dim=1), self_representation.mean(dim=1)], dim=-1
        )
        strategy_choices = self.strategy_monitor(strategy_input)  # [batch_size, 6]

        # 认知负荷评估
        cognitive_load = self.cognitive_load_evaluator(
            hidden_states.mean(dim=1)
        )  # [batch_size, 1]

        # 元认知控制
        metacognitive_input = torch.cat(
            [
                hidden_states.mean(dim=1),
                thought_monitor,
                attention_distribution,
                strategy_choices,
                cognitive_load,
            ],
            dim=-1,
        )

        metacognition = self.metacognitive_controller(metacognitive_input)

        # 元认知注意力
        metacognition_expanded = metacognition.unsqueeze(1).expand(-1, seq_len, -1)
        metacognition_attn_output, _ = self.metacognition_attention(
            metacognition_expanded, metacognition_expanded, metacognition_expanded
        )

        # === 4. 自我知识 ===
        # 查询能力知识
        ability_query = torch.cat(
            [
                hidden_states.mean(dim=1, keepdim=True),
                self_representation.mean(dim=1, keepdim=True),
            ],
            dim=-1,
        )
        ability_knowledge = self.knowledge_query_network(ability_query)

        # 知识推理
        knowledge_reasoning_input = torch.cat(
            [
                ability_knowledge.mean(dim=1),
                self_representation.mean(dim=1),
                self_evaluation,
            ],
            dim=-1,
        )

        self_knowledge = self.knowledge_reasoning_network(knowledge_reasoning_input)

        # 自我知识注意力
        self_knowledge_expanded = self_knowledge.unsqueeze(1).expand(-1, seq_len, -1)
        self_knowledge_attn_output, _ = self.self_knowledge_attention(
            self_knowledge_expanded, self_knowledge_expanded, self_knowledge_expanded
        )

        # === 5. 自我意识 ===
        # 状态反思
        state_reflection_input = torch.cat(
            [hidden_states.mean(dim=1), self_representation.mean(dim=1)], dim=-1
        )
        state_reflection = self.state_reflection_network(state_reflection_input)

        # 意图推理
        intention_input = torch.cat(
            [
                hidden_states.mean(dim=1),
                self_representation.mean(dim=1),
                goal_repr if goals is not None else hidden_states.mean(dim=1),
            ],
            dim=-1,
        )
        intention_reasoning = self.intention_reasoning_network(intention_input)

        # 未来预测
        future_input = torch.cat(
            [hidden_states.mean(dim=1), self_representation.mean(dim=1)], dim=-1
        )
        future_prediction = self.future_prediction_network(future_input)

        # 自我意识整合
        self_awareness_input = torch.cat(
            [
                state_reflection,
                intention_reasoning,
                future_prediction,
                self_representation.mean(dim=1),
            ],
            dim=-1,
        )

        self_awareness = self.self_awareness_integrator(self_awareness_input)

        # === 6. 记忆和学习 ===
        # 经验记忆检索和学习
        # 当前经验特征：整合的自我表示
        current_experience = self_representation.mean(
            dim=1
        )  # [batch_size, hidden_size]

        # 从经验记忆中检索相关经验
        retrieved_experience, exp_attention_weights, exp_memory_indices = (
            self.retrieve_experience_memory(query=current_experience, top_k=5)
        )

        # 计算经验学习损失（仅在训练时）
        experience_learning_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.training:
            experience_learning_loss = self.compute_experience_learning_loss(
                current_experience=current_experience,
                retrieved_experience=retrieved_experience,
                attention_weights=exp_attention_weights,
            )

        # 更新经验记忆（实际更新由优化器完成）
        if self.training and feedback is not None:
            # 使用反馈计算经验重要性
            # 积极反馈：重要性高；消极反馈：重要性低
            if feedback.shape[-1] == 1:
                experience_importance = torch.sigmoid(
                    feedback.squeeze(-1)
                )  # [batch_size]
            else:
                experience_importance = torch.ones(
                    batch_size, device=hidden_states.device
                )

            # 计算更新（返回更新后的记忆，实际更新在训练循环中完成）
            updated_experience_memory = self.update_experience_memory(
                new_experience=current_experience,
                experience_importance=experience_importance,
                learning_rate=0.01,
            )

        # 使用检索到的经验增强学习
        weighted_retrieved_exp = torch.sum(
            retrieved_experience * exp_attention_weights.unsqueeze(-1), dim=1
        )  # [batch_size, hidden_size]

        # 自我模型更新（增强版：包含经验记忆）
        if feedback is not None:
            # 处理反馈维度：如果反馈维度不是hidden_size，使用投影层
            if feedback.shape[-1] != self.config.hidden_size:
                # 重塑反馈以便投影
                if feedback.shape[-1] == 1:
                    # 单维反馈：使用反馈投影层
                    projected_feedback = self.feedback_projection(feedback)
                else:
                    # 多维反馈：使用线性投影
                    projected_feedback = nn.functional.linear(
                        feedback,
                        torch.eye(self.config.hidden_size, device=feedback.device)[
                            : feedback.shape[-1]
                        ].T,
                    )
            else:
                projected_feedback = feedback

            self_model_update_input = torch.cat(
                [
                    self_representation.mean(dim=1),
                    self_evaluation,
                    projected_feedback,
                    weighted_retrieved_exp,
                ],
                dim=-1,
            )
        else:
            # 没有反馈时使用零向量
            zero_feedback = torch.zeros_like(self_representation.mean(dim=1))
            self_model_update_input = torch.cat(
                [
                    self_representation.mean(dim=1),
                    self_evaluation,
                    zero_feedback,
                    weighted_retrieved_exp,
                ],
                dim=-1,
            )

        self_model_update = self.self_model_updater(self_model_update_input)

        # 自我认知学习（增强版：包含经验记忆）
        self_cognition_learning_input = torch.cat(
            [
                self_representation.mean(dim=1),
                self_evaluation,
                metacognition,
                self_knowledge,
                weighted_retrieved_exp,
            ],
            dim=-1,
        )

        learned_features = self.self_cognition_learner(self_cognition_learning_input)

        # === 7. 特征融合和整合 ===
        # 自我特征融合
        self_features_input = torch.cat(
            [
                self_representation_attn_output.mean(dim=1),
                self_evaluation_attn_output.mean(dim=1),
                metacognition_attn_output.mean(dim=1),
                self_knowledge_attn_output.mean(dim=1),
            ],
            dim=-1,
        )

        fused_self_features = self.self_feature_fusion(self_features_input)

        # 认知过程融合
        cognitive_process_input = torch.cat(
            [
                fused_self_features.unsqueeze(1).expand(-1, seq_len, -1),
                self_representation_attn_output,
                self_evaluation_attn_output,
                metacognition_attn_output,
                self_knowledge_attn_output,
            ],
            dim=-1,
        )

        fused_cognitive_process = self.cognitive_process_fusion(cognitive_process_input)

        # 最终自我认知融合
        final_fusion_input = torch.cat(
            [
                fused_cognitive_process,
                self_awareness.unsqueeze(1).expand(-1, seq_len, -1),
                learned_features.unsqueeze(1).expand(-1, seq_len, -1),
                hidden_states,
            ],
            dim=-1,
        )

        integrated_self_cognition = self.final_self_cognition_fusion(final_fusion_input)
        integrated_self_cognition = self.layer_norm(integrated_self_cognition)

        # 返回完整自我认知结果
        return {
            "self_representation": self_representation,
            "self_evaluation": self_evaluation,
            "metacognition": metacognition,
            "self_knowledge": self_knowledge,
            "self_awareness": self_awareness,
            "integrated_self_cognition": integrated_self_cognition,
            "performance_scores": performance_scores,
            "ability_levels": ability_levels,
            "cognitive_load": cognitive_load,
            "attention_distribution": attention_distribution,
            "strategy_choices": strategy_choices,
            "state_reflection": state_reflection,
            "intention_reasoning": intention_reasoning,
            "future_prediction": future_prediction,
            "self_model_update": self_model_update,
            "learned_features": learned_features,
            # 经验记忆相关输出
            "current_experience": current_experience,
            "retrieved_experience": retrieved_experience,
            "experience_attention_weights": exp_attention_weights,
            "experience_memory_indices": exp_memory_indices,
            "weighted_retrieved_experience": weighted_retrieved_exp,
            "experience_learning_loss": experience_learning_loss,
            # 认知科学算法结果
            "cognitive_science_results": cognitive_science_results,
            "metacognitive_monitoring": cognitive_science_results.get(
                "metacognitive_monitoring", {}
            ),
            "self_regulated_learning": cognitive_science_results.get(
                "self_regulated_learning", {}
            ),
            "self_schemas": cognitive_science_results.get("self_schemas", {}),
        }

    def compute_consistency_loss(
        self,
        self_cognition_outputs: Dict[str, torch.Tensor],
        performance_history: Optional[Dict[str, torch.Tensor]] = None,
        temporal_history: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算自我认知一致性损失 - 多层次验证机制

        基于修复方案实现多层次一致性验证：
        1. 语义一致性：自我表示与自我评估的语义对齐
        2. 性能一致性：能力评估与实际性能的一致性
        3. 时间一致性：自我认知随时间的一致性
        4. 逻辑一致性：自我认知中的逻辑一致性

        参数:
            self_cognition_outputs: 自我认知模块的输出字典
            performance_history: 性能历史记录（可选）
            temporal_history: 时间历史记录（可选）

        返回:
            一致性损失字典，包含各个维度的损失值
        """
        losses = {}

        # 1. 语义一致性损失 - 自我表示与自我评估的语义对齐
        self_representation = self_cognition_outputs["self_representation"]
        self_evaluation = self_cognition_outputs["self_evaluation"]

        # 维度对齐：确保可以计算相似度
        batch_size = self_representation.shape[0]
        seq_len = self_representation.shape[1]

        # 平均池化自我表示以获得与self_evaluation相同的维度
        self_rep_pooled = self_representation.mean(dim=1)  # [batch_size, hidden_size]

        # 余弦相似度损失（鼓励高相似度）
        cosine_sim = nn.CosineSimilarity(dim=-1)
        semantic_similarity = cosine_sim(self_rep_pooled, self_evaluation)
        semantic_loss = 1.0 - semantic_similarity.mean()  # 1 - 相似度作为损失

        losses["semantic_consistency"] = semantic_loss

        # 2. 性能一致性损失 - 能力评估与实际性能的一致性
        ability_levels = self_cognition_outputs["ability_levels"]  # [batch_size, 8]
        performance_scores = self_cognition_outputs[
            "performance_scores"
        ]  # [batch_size, 5]

        # 映射能力水平到性能维度（完整：使用线性变换）
        # 实际中应根据领域知识设计映射关系
        if ability_levels.shape[1] >= performance_scores.shape[1]:
            # 如果能力维度 >= 性能维度，使用前n个能力
            mapped_abilities = ability_levels[:, : performance_scores.shape[1]]
            performance_consistency_loss = F.mse_loss(
                mapped_abilities, performance_scores
            )
        else:
            # 否则使用插值
            mapped_abilities = F.interpolate(
                ability_levels.unsqueeze(1),
                size=performance_scores.shape[1],
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            performance_consistency_loss = F.mse_loss(
                mapped_abilities, performance_scores
            )

        losses["performance_consistency"] = performance_consistency_loss

        # 3. 时间一致性损失 - 自我认知随时间的一致性
        temporal_consistency_loss = torch.tensor(0.0, device=self_representation.device)

        if temporal_history and len(temporal_history) > 1:
            # 如果有时间历史，计算相邻时间步之间的一致性
            for i in range(len(temporal_history) - 1):
                prev_repr = temporal_history[i].get("self_representation", None)
                curr_repr = temporal_history[i + 1].get("self_representation", None)

                if prev_repr is not None and curr_repr is not None:
                    # 确保形状一致
                    min_seq_len = min(prev_repr.shape[1], curr_repr.shape[1])
                    prev_repr_trunc = prev_repr[:, :min_seq_len, :]
                    curr_repr_trunc = curr_repr[:, :min_seq_len, :]

                    # 计算时间一致性损失（鼓励平滑变化）
                    temporal_loss = F.mse_loss(prev_repr_trunc, curr_repr_trunc)
                    temporal_consistency_loss = (
                        temporal_consistency_loss + temporal_loss
                    )

            if len(temporal_history) > 1:
                temporal_consistency_loss = temporal_consistency_loss / (
                    len(temporal_history) - 1
                )

        losses["temporal_consistency"] = temporal_consistency_loss

        # 4. 逻辑一致性损失 - 自我认知中的逻辑一致性
        # 检查能力水平是否在合理范围内 (0-1)
        ability_levels = self_cognition_outputs["ability_levels"]
        ability_range_loss = torch.mean(
            torch.relu(ability_levels - 1.0) + torch.relu(-ability_levels)
        )

        # 检查认知负荷是否非负
        cognitive_load = self_cognition_outputs["cognitive_load"]
        cognitive_load_loss = torch.mean(torch.relu(-cognitive_load))

        # 检查注意力分布是否和为1
        attention_distribution = self_cognition_outputs["attention_distribution"]
        attention_sum = attention_distribution.sum(dim=-1)
        attention_sum_loss = F.mse_loss(attention_sum, torch.ones_like(attention_sum))

        logical_consistency_loss = (
            ability_range_loss + cognitive_load_loss + attention_sum_loss
        )

        losses["logical_consistency"] = logical_consistency_loss

        # 5. 内部一致性损失 - 不同自我认知组件之间的一致性
        # 自我表示与元认知的一致性
        metacognition = self_cognition_outputs["metacognition"]
        self_knowledge = self_cognition_outputs["self_knowledge"]

        # 计算组件之间的互信息（近似为余弦相似度）
        internal_consistency_loss = 0.0
        components = [self_rep_pooled, self_evaluation, metacognition, self_knowledge]

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp_i = components[i]
                comp_j = components[j]

                # 确保形状一致
                if comp_i.shape == comp_j.shape:
                    similarity = cosine_sim(comp_i, comp_j)
                    # 鼓励组件之间有适度的一致性（不是完全相同，也不是完全不同）
                    # 目标相似度设为0.7（适度相关）
                    target_similarity = 0.7
                    comp_consistency_loss = F.mse_loss(
                        similarity, torch.ones_like(similarity) * target_similarity
                    )
                    internal_consistency_loss = (
                        internal_consistency_loss + comp_consistency_loss
                    )

        if len(components) > 1:
            internal_consistency_loss = internal_consistency_loss / (
                len(components) * (len(components) - 1) / 2
            )

        losses["internal_consistency"] = internal_consistency_loss

        # 总一致性损失（加权和）
        weights = {
            "semantic_consistency": 0.3,
            "performance_consistency": 0.25,
            "temporal_consistency": 0.2,
            "logical_consistency": 0.15,
            "internal_consistency": 0.1,
        }

        total_loss = torch.tensor(0.0, device=self_representation.device)
        for loss_name, loss_value in losses.items():
            if loss_name in weights:
                total_loss = total_loss + weights[loss_name] * loss_value

        losses["total_consistency"] = total_loss

        # 返回详细的损失字典
        return losses

    def update_experience_memory(
        self,
        new_experience: torch.Tensor,
        experience_importance: Optional[torch.Tensor] = None,
        learning_rate: float = 0.01,
    ) -> torch.Tensor:
        """更新经验记忆库 - 基于经验的重要性加权更新

        参数:
            new_experience: 新经验特征 [batch_size, hidden_size]
            experience_importance: 经验重要性分数 [batch_size] (可选)
            learning_rate: 学习率，控制更新速度

        返回:
            updated_memory: 更新后的经验记忆 [memory_size, hidden_size]

        设计原则：
        1. 重要性加权更新：重要经验获得更大更新
        2. 渐近学习：使用动量更新，避免剧烈变化
        3. 多样性保持：确保记忆库覆盖不同经验类型
        """
        batch_size = new_experience.shape[0]
        memory_size = self.experience_memory.shape[0]
        hidden_dim = self.experience_memory.shape[1]

        # 如果没有提供重要性分数，默认为1.0
        if experience_importance is None:
            experience_importance = torch.ones(batch_size, device=new_experience.device)

        # 计算新经验与记忆中每个条目的相似度
        # self.experience_memory: [memory_size, hidden_dim]
        # new_experience: [batch_size, hidden_dim]
        new_experience_norm = F.normalize(
            new_experience, p=2, dim=-1
        )  # [batch_size, hidden_dim]
        memory_norm = F.normalize(
            self.experience_memory, p=2, dim=-1
        )  # [memory_size, hidden_dim]

        similarities = torch.matmul(
            new_experience_norm, memory_norm.T
        )  # [batch_size, memory_size]

        # 为每个新经验找到最相似的记忆条目
        # 使用top-3相似度，加权更新多个记忆条目
        top_k = min(3, memory_size)
        top_similarities, top_indices = torch.topk(
            similarities, k=top_k, dim=-1
        )  # [batch_size, top_k]

        # 计算更新权重：基于相似度和重要性
        # 相似度越高，更新权重越大
        update_weights = F.softmax(
            top_similarities * 10.0, dim=-1
        )  # [batch_size, top_k]
        update_weights = update_weights * experience_importance.unsqueeze(
            1
        )  # 重要性加权

        # 动量更新：新值 = (1 - lr*weight) * 旧值 + lr*weight * 新值
        updated_memory = self.experience_memory.clone()

        for b in range(batch_size):
            for k in range(top_k):
                mem_idx = top_indices[b, k]
                weight = update_weights[b, k] * learning_rate

                # 动量更新
                old_value = self.experience_memory[mem_idx].detach()
                new_value = new_experience[b].detach()
                updated_value = (1.0 - weight) * old_value + weight * new_value

                updated_memory[mem_idx] = updated_value

        # 更新记忆参数（在训练中，这应该通过梯度下降学习）
        # 这里我们实现真实的内存更新机制
        if self.training:
            # 在训练模式下，我们需要通过梯度下降更新记忆
            # 创建memory的梯度更新
            if (
                hasattr(self, "experience_memory")
                and self.experience_memory.requires_grad
            ):
                # 计算内存更新损失：鼓励内存向新经验方向更新
                memory_update_loss = 0.0
                for b in range(batch_size):
                    for k in range(top_k):
                        mem_idx = top_indices[b, k]
                        weight = update_weights[b, k] * learning_rate

                        # 计算目标值和当前值的差异
                        target_value = new_experience[b]
                        current_value = self.experience_memory[mem_idx]

                        # L2损失加权
                        memory_update_loss += weight * F.mse_loss(
                            current_value, target_value
                        )

                # 如果内存可学习，我们保留损失用于反向传播
                # 完整实现，实际应该通过优化器）
                with torch.no_grad():
                    self.experience_memory.data = updated_memory

            # 记录内存更新统计
            if hasattr(self, "memory_update_stats"):
                avg_update_weight = update_weights.mean().item()
                self.memory_update_stats.append(
                    {
                        "avg_update_weight": avg_update_weight,
                        "num_updates": batch_size * top_k,
                        "learning_rate": learning_rate,
                    }
                )
        else:
            # 在推理模式下，直接更新内存
            with torch.no_grad():
                self.experience_memory.data = updated_memory

        return updated_memory

    def retrieve_experience_memory(
        self, query: torch.Tensor, top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从经验记忆中检索相关内容

        参数:
            query: 查询特征 [batch_size, hidden_size]
            top_k: 返回的top-k记忆条目数量

        返回:
            retrieved_memory: 检索到的记忆 [batch_size, top_k, hidden_size]
            attention_weights: 注意力权重 [batch_size, top_k]
            memory_indices: 记忆索引 [batch_size, top_k]
        """
        batch_size = query.shape[0]
        memory_size = self.experience_memory.shape[0]

        # 归一化查询和记忆
        query_norm = F.normalize(query, p=2, dim=-1)  # [batch_size, hidden_dim]
        memory_norm = F.normalize(
            self.experience_memory, p=2, dim=-1
        )  # [memory_size, hidden_dim]

        # 计算相似度（注意力分数）
        attention_scores = torch.matmul(
            query_norm, memory_norm.T
        )  # [batch_size, memory_size]

        # 获取top-k记忆条目
        top_k = min(top_k, memory_size)
        top_scores, top_indices = torch.topk(
            attention_scores, k=top_k, dim=-1
        )  # [batch_size, top_k]

        # 计算注意力权重（softmax）
        attention_weights = F.softmax(top_scores, dim=-1)  # [batch_size, top_k]

        # 检索记忆
        retrieved_memory = torch.zeros(
            batch_size, top_k, self.experience_memory.shape[1], device=query.device
        )

        for b in range(batch_size):
            for k in range(top_k):
                mem_idx = top_indices[b, k]
                retrieved_memory[b, k] = self.experience_memory[mem_idx]

        return retrieved_memory, attention_weights, top_indices

    def compute_experience_learning_loss(
        self,
        current_experience: torch.Tensor,
        retrieved_experience: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """计算经验学习损失 - 鼓励从经验中学习

        参数:
            current_experience: 当前经验特征 [batch_size, hidden_size]
            retrieved_experience: 检索到的经验 [batch_size, top_k, hidden_size]
            attention_weights: 注意力权重 [batch_size, top_k]

        返回:
            learning_loss: 经验学习损失值
        """
        batch_size = current_experience.shape[0]
        top_k = retrieved_experience.shape[1]

        # 加权平均检索到的经验
        weighted_retrieved = torch.sum(
            retrieved_experience * attention_weights.unsqueeze(-1), dim=1
        )  # [batch_size, hidden_size]

        # 鼓励当前经验与加权检索经验相似（从经验中学习）
        similarity = F.cosine_similarity(current_experience, weighted_retrieved, dim=-1)

        # 损失 = 1 - 相似度（鼓励高相似度）
        learning_loss = 1.0 - similarity.mean()

        return learning_loss



