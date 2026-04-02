# MixtureOfExpertsLayer - 从self_agi_model.py拆分
"""MixtureOfExpertsLayer模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging

class MixtureOfExpertsLayer(nn.Module):
    """混合专家层 - 基于MoE架构

    参考论文:
    - Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017)
    - Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus et al., 2021)
    - GLaM: Efficient Scaling of Language Models with Mixture-of-Experts (Du et al., 2021)

    关键特性:
    1. 稀疏激活：每个token只激活少数专家
    2. 负载平衡：确保专家使用均衡
    3. 容量因子：处理专家容量限制
    4. 可扩展性：支持大量专家
    """

    def __init__(self, config: 'AGIModelConfig'):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # MoE配置
        self.num_experts = config.num_experts
        self.expert_capacity_factor = config.expert_capacity_factor
        self.router_type = config.router_type
        self.top_k = config.top_k
        self.load_balancing_lambda = config.load_balancing_lambda

        # 专家网络 (每个专家是一个小型前馈网络)
        self.experts = nn.ModuleList(
            [self._create_expert() for _ in range(self.num_experts)]
        )

        # 路由器网络 (决定token分配给哪个专家)
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # 专家dropout
        self.expert_dropout = nn.Dropout(config.expert_dropout)

        # 层归一化
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

        logger.info(
            f"初始化混合专家层: 专家数={self.num_experts}, top_k={self.top_k}, "
            f"路由器类型={self.router_type}, 负载平衡lambda={self.load_balancing_lambda}"
        )

    def _create_expert(self) -> nn.Module:
        """创建单个专家网络"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(self.config.expert_dropout),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """前向传播

        实现稀疏混合专家层，包含负载平衡损失计算。

        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] 可选

        返回:
            训练模式: (输出张量, 损失字典)
            评估模式: 输出张量
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 计算路由器logits
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]

        # 路由器门控 (top-k选择)
        if self.router_type == "topk":
            # 标准top-k选择
            topk_values, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
            router_weights = torch.softmax(topk_values, dim=-1)
        elif self.router_type == "noisy_topk":
            # 带噪声的top-k (增强探索)
            noise = torch.randn_like(router_logits) * 0.01
            noisy_logits = router_logits + noise
            topk_values, topk_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
            router_weights = torch.softmax(topk_values, dim=-1)
        else:
            # 学习路由器 (通过softmax)
            router_weights = torch.softmax(router_logits, dim=-1)
            topk_values, topk_indices = torch.topk(router_weights, self.top_k, dim=-1)

        # 扁平化处理以便批量处理
        flat_hidden_states = hidden_states.view(
            -1, self.hidden_size
        )  # [batch_size*seq_len, hidden_size]
        flat_topk_indices = topk_indices.view(
            -1, self.top_k
        )  # [batch_size*seq_len, top_k]
        flat_router_weights = router_weights.view(
            -1, self.top_k
        )  # [batch_size*seq_len, top_k]

        # 初始化输出
        output = torch.zeros_like(flat_hidden_states)

        # 计算每个token的专家容量
        total_tokens = batch_size * seq_len
        expert_capacity = int(
            total_tokens * self.expert_capacity_factor / self.num_experts
        )
        expert_capacity = max(expert_capacity, 1)  # 至少为1

        # 跟踪专家使用情况用于负载平衡
        expert_usage = torch.zeros(self.num_experts, device=hidden_states.device)

        # 为每个专家处理分配到的token
        for expert_idx in range(self.num_experts):
            # 找出分配给当前专家的token
            expert_mask = (flat_topk_indices == expert_idx).any(
                dim=-1
            )  # [batch_size*seq_len]

            if not expert_mask.any():
                continue  # 没有token分配给这个专家

            # 限制专家容量 (如果需要)
            if expert_mask.sum() > expert_capacity:
                # 随机选择容量限制内的token
                selected_indices = torch.nonzero(expert_mask, as_tuple=True)[0]
                selected_indices = selected_indices[
                    torch.randperm(len(selected_indices))[:expert_capacity]
                ]
                expert_mask = torch.zeros_like(expert_mask, dtype=torch.bool)
                expert_mask[selected_indices] = True

            # 更新专家使用情况
            expert_usage[expert_idx] = expert_mask.sum().item()

            if expert_mask.any():
                # 获取分配给当前专家的token
                expert_tokens = flat_hidden_states[
                    expert_mask
                ]  # [num_tokens, hidden_size]

                # 获取对应权重
                token_weights = []
                for i, mask in enumerate(expert_mask):
                    if mask:
                        # 找到当前专家在top-k中的位置
                        pos = (flat_topk_indices[i] == expert_idx).nonzero(
                            as_tuple=True
                        )[0]
                        weight = flat_router_weights[i, pos].sum()  # 可能多个位置
                        token_weights.append(weight)

                token_weights = (
                    torch.stack(token_weights)
                    if token_weights
                    else torch.tensor([], device=hidden_states.device)
                )

                # 专家处理
                expert_output = self.experts[expert_idx](
                    expert_tokens
                )  # [num_tokens, hidden_size]

                # 应用路由器权重
                weighted_output = expert_output * token_weights.unsqueeze(-1)

                # 累加到输出
                output[expert_mask] += weighted_output

        # 负载平衡损失计算 (Switch Transformers风格)
        load_balance_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.training and self.load_balancing_lambda > 0:
            # 计算路由器概率分布
            router_probs = F.softmax(
                router_logits, dim=-1
            )  # [batch_size, seq_len, num_experts]

            # 计算专家分配掩码 (one-hot编码)
            expert_mask_one_hot = F.one_hot(
                topk_indices, num_classes=self.num_experts
            ).float()
            # 对于top_k > 1的情况，需要聚合
            if self.top_k > 1:
                expert_mask_one_hot = expert_mask_one_hot.sum(dim=-2)  # 聚合top_k维度
                # 归一化到[0, 1]
                expert_mask_one_hot = torch.clamp(expert_mask_one_hot, 0, 1)

            # 计算负载平衡损失 (交叉熵风格)
            # 路由器概率的均值
            router_prob_mean = router_probs.mean(dim=(0, 1))  # [num_experts]
            # 专家使用概率
            expert_usage_prob = expert_mask_one_hot.mean(dim=(0, 1))  # [num_experts]

            # 避免零除和NaN
            eps = 1e-8
            router_prob_mean = router_prob_mean + eps
            expert_usage_prob = expert_usage_prob + eps

            # 负载平衡损失 (KL散度风格)
            load_balance_loss = (
                expert_usage_prob * torch.log(expert_usage_prob / router_prob_mean)
            ).sum()
            load_balance_loss = load_balance_loss * self.load_balancing_lambda

        # 恢复原始形状
        output = output.view(batch_size, seq_len, self.hidden_size)

        # 应用dropout
        output = self.expert_dropout(output)

        # 残差连接和层归一化
        output = hidden_states + output
        output = self.layer_norm(output)

        # 返回结果
        if self.training:
            losses = {"load_balance_loss": load_balance_loss}
            return output, losses
        else:
            return output



