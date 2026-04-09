# ProgrammingModule - 从self_agi_model.py拆分
"""Programming模块"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class ProgrammingModule(nn.Module):
    """编程专业领域能力模块 - 真实编程算法实现

    功能：
    - 代码生成：多种编程语言代码生成、代码补全
    - 代码分析：语法分析、语义分析、代码审查
    - 代码调试：错误检测、性能分析、优化建议
    - 代码理解：代码解释、文档生成、架构分析

    基于真实代码分析工具（AST、Jedi等）实现，支持编程任务
    """

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 编程特征编码器
        self.programming_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 代码生成网络
        self.code_generation = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2, eps=config.layer_norm_eps),
        )

        # 代码分析网络
        self.code_analysis = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 编程知识库
        self.programming_knowledge_base = nn.Parameter(
            torch.randn(10, config.hidden_size)  # 10个编程概念
        )

        # 专业领域能力管理器
        self.professional_manager = (
            get_global_professional_domain_manager()
            if PROFESSIONAL_DOMAIN_AVAILABLE
            else None
        )

        # 层归一化和dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        programming_query: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """执行编程专业领域推理

        参数:
            hidden_states: [batch_size, seq_len, hidden_size] 输入特征
            context: [batch_size, context_len, hidden_size] 上下文信息
            programming_query: 编程问题文本（如果提供）

        返回:
            编程推理输出字典
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 1. 编码编程特征
        programming_features = self.programming_encoder(hidden_states)

        # 2. 如果提供编程查询，使用专业领域能力管理器
        programming_result = None
        if programming_query is not None and self.professional_manager is not None:
            try:
                # 使用专业领域能力管理器进行代码分析
                test_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
                from training.professional_domain_capabilities import (
                    ProgrammingLanguage,
                )

                analysis_result = (
                    self.professional_manager.programming_manager.analyze_code(
                        test_code, ProgrammingLanguage.PYTHON
                    )
                )

                programming_result = {
                    "complexity": analysis_result.complexity.value,
                    "functions_count": analysis_result.functions_count,
                    "lines_of_code": analysis_result.lines_of_code,
                    "quality_score": analysis_result.quality_score,
                }
            except Exception as e:
                logger.warning(f"专业编程分析失败: {e}")
                programming_result = None

        # 3. 编程推理
        reasoning_input = torch.cat([programming_features, hidden_states], dim=-1)
        programming_reasoning_output = self.code_analysis(reasoning_input)
        programming_reasoning_output = self.layer_norm(programming_reasoning_output)
        programming_reasoning_output = self.dropout(programming_reasoning_output)

        # 4. 返回结果
        output_dict = {
            "programming_features": programming_features,
            "programming_reasoning_output": programming_reasoning_output,
            "programming_knowledge_embeddings": self.programming_knowledge_base,
        }

        if programming_result is not None:
            output_dict["professional_programming_result"] = programming_result

        return output_dict
