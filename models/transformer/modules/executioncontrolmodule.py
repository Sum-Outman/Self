# ExecutionControlModule - 从self_agi_model.py拆分
"""ExecutionControl模块"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class ExecutionControlModule(nn.Module):
    """执行控制模块"""

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 控制网络
        self.control_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # 动作选择
        self.action_selector = nn.Linear(config.hidden_size, config.hidden_size)

        # 系统控制器
        self.system_controller = nn.Linear(config.hidden_size, config.hidden_size)

        # 层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, plans: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """执行控制

        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            plans: 计划输入

        返回:
            控制输出字典
        """
        # 整合计划信息
        if plans is not None:
            control_input = hidden_states + plans
        else:
            control_input = hidden_states

        # 控制网络
        control_features = self.control_network(control_input)
        control_features = self.layer_norm(control_features)

        # 动作选择
        actions = self.action_selector(control_features)

        # 系统控制
        system_control = self.system_controller(control_features)

        return {
            "control_features": control_features,
            "actions": actions,
            "system_control": system_control,
        }
