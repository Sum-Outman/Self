# SwitchRouter - 从self_agi_model.py拆分
"""SwitchRouter模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitchRouter(nn.Module):
    """Switch Transformers路由器 - 每个token只路由到一个专家"""

    def __init__(
        self, hidden_size: int, num_experts: int, capacity_factor: float = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # 路由器网络
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # 负载平衡损失参数
        self.load_balancing_lambda = 0.01

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Switch路由前向传播"""
        batch_size, seq_len, _ = hidden_states.shape

        # 计算路由logits
        router_logits = self.router(hidden_states)  # [batch, seq, num_experts]

        # 每个token选择top-1专家 (Switch风格)
        routing_weights = F.softmax(router_logits, dim=-1)
        expert_index = torch.argmax(routing_weights, dim=-1)  # [batch, seq]

        # 计算专家掩码
        expert_mask = F.one_hot(expert_index, num_classes=self.num_experts).float()

        # 计算负载平衡损失
        if self.training:
            load_balance_loss = self.compute_load_balance_loss(
                routing_weights, expert_mask
            )
        else:
            load_balance_loss = torch.tensor(0.0, device=hidden_states.device)

        return router_logits, expert_mask, load_balance_loss

    def compute_load_balance_loss(
        self, routing_weights: torch.Tensor, expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算负载平衡损失"""
        # 路由器概率
        router_prob = routing_weights.mean(dim=(0, 1))  # [num_experts]

        # 专家使用概率
        expert_usage = expert_mask.mean(dim=(0, 1))  # [num_experts]

        # 负载平衡损失 (交叉熵)
        load_balance_loss = F.cross_entropy(
            router_prob.unsqueeze(0), expert_usage.unsqueeze(0)
        )

        return load_balance_loss * self.load_balancing_lambda



