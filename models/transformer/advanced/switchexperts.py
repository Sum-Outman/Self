# SwitchExperts - 从self_agi_model.py拆分
"""SwitchExperts模块"""

import torch
import torch.nn as nn


class SwitchExperts(nn.Module):
    """Switch Transformers专家网络"""

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts

        # 专家网络 (更深的网络结构)
        self.experts = nn.ModuleList(
            [self.create_expert_network() for _ in range(self.num_experts)]
        )

        # 路由器
        self.router = SwitchRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            capacity_factor=config.expert_capacity_factor,
        )

        # 专家丢弃概率
        self.expert_dropout = nn.Dropout(config.expert_dropout)

        logger.info(
            f"初始化SwitchExperts: 专家数={self.num_experts}, 隐藏大小={self.hidden_size}"
        )

    def create_expert_network(self) -> nn.Module:
        """创建单个专家网络 (更深的版本)"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(self.config.expert_dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(self.config.expert_dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(self.config.expert_dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Switch Experts前向传播"""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 路由决策
        router_logits, expert_mask, load_balance_loss = self.router(hidden_states)

        # 展平以处理专家分配
        hidden_states_flat = hidden_states.reshape(
            -1, hidden_size
        )  # [batch*seq, hidden]
        expert_mask_flat = expert_mask.reshape(
            -1, self.num_experts
        )  # [batch*seq, num_experts]

        # 初始化输出
        outputs_flat = torch.zeros_like(hidden_states_flat)

        # 处理每个专家
        for expert_idx in range(self.num_experts):
            # 获取分配给当前专家的token掩码
            expert_token_mask = expert_mask_flat[:, expert_idx].bool()

            if expert_token_mask.any():
                # 提取分配给该专家的token
                expert_input = hidden_states_flat[expert_token_mask]

                # 专家处理
                expert_output = self.experts[expert_idx](expert_input)
                expert_output = self.expert_dropout(expert_output)

                # 写回输出
                outputs_flat[expert_token_mask] = expert_output

        # 恢复原始形状
        outputs = outputs_flat.reshape(batch_size, seq_len, hidden_size)

        # 如果需要，返回负载平衡损失
        if self.training:
            return outputs, load_balance_loss
        else:
            return outputs


# ============================================================================
# DoRA (权重分解的低秩适应) 实现
# ============================================================================
