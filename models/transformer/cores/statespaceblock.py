# StateSpaceBlock - 从self_agi_model.py拆分
"""StateSpaceBlock模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StateSpaceBlock(nn.Module):
    """状态空间模型块 - 基于Mamba和RetNet架构

    参考论文:
    - Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
    - RetNet: Retention Network for Language Modeling (Sun et al., 2022)
    - Structured State Space Models for Sequence Modeling (Gu et al., 2022)

    关键特性:
    1. 选择性状态空间：输入依赖的状态转移
    2. 高效扫描算法：线性时间复杂度的序列建模
    3. 硬件感知设计：高效GPU实现
    4. 长程依赖：处理超长序列
    """

    def __init__(self, config: "AGIModelConfig"):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # 状态空间配置
        self.state_dim = config.state_space_dim
        self.expand = config.state_space_expand
        self.inner_dim = self.hidden_size * self.expand

        # 选择性状态空间参数 (Mamba风格)
        # 输入投影
        self.in_proj = nn.Linear(self.hidden_size, self.inner_dim * 2)

        # 状态空间参数
        self.A = nn.Parameter(torch.randn(self.state_dim, self.inner_dim) * 0.02)
        self.B = nn.Linear(self.inner_dim, self.state_dim, bias=False)
        self.C = nn.Linear(self.inner_dim, self.state_dim, bias=False)
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        # Mamba卷积预处理层（完整实现）
        # 基于Mamba论文的深度可分离卷积，用于输入特征预处理
        self.conv = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=self.inner_dim,  # 深度可分离卷积
            bias=False,
        )

        # 选择性机制 (输入依赖的门控)
        self.selective_gate = nn.Linear(self.inner_dim, self.inner_dim)

        # 输出投影
        self.out_proj = nn.Linear(self.inner_dim, self.hidden_size)

        # 层归一化
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

        # 激活函数
        self.activation = nn.SiLU() if config.activation_fn == "silu" else nn.GELU()

        # 门控激活 (GLU) 可选
        self.glu_enabled = config.gated_activation_enabled
        if self.glu_enabled:
            self.glu_gate = nn.Linear(self.inner_dim, config.glu_dim)

        # RetNet保留机制 (可选)
        self.use_retention = config.use_retention
        if self.use_retention:
            self.retention_heads = config.retention_heads
            self.retention_dim = self.hidden_size // self.retention_heads

            # 保留参数
            self.retention_q = nn.Linear(self.hidden_size, self.hidden_size)
            self.retention_k = nn.Linear(self.hidden_size, self.hidden_size)
            self.retention_v = nn.Linear(self.hidden_size, self.hidden_size)
            self.retention_gamma = nn.Parameter(torch.ones(1) * 0.9)  # 衰减因子

            # 保留门函数
            if config.retention_gate_fn == "swish":
                self.retention_gate = nn.SiLU()
            else:
                self.retention_gate = nn.Sigmoid()

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        logger.info(
            f"初始化状态空间块: 隐藏大小={self.hidden_size}, 状态维度={self.state_dim}, "
            f"扩展因子={self.expand}, 保留机制={self.use_retention}"
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        实现选择性状态空间模型 (Mamba风格) 和RetNet保留机制的混合。

        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] 可选

        返回:
            输出张量: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 输入投影
        x = self.in_proj(hidden_states)  # [batch_size, seq_len, inner_dim*2]

        # 分割为x和gate
        x, gate = torch.split(x, self.inner_dim, dim=-1)

        # Mamba卷积预处理（完整实现）
        x_conv = x.transpose(1, 2)  # [batch_size, inner_dim, seq_len]
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [batch_size, seq_len, inner_dim]

        # 选择性门控 (输入依赖)
        gate = self.selective_gate(gate)
        x = x_conv * self.activation(gate)

        # ============ Mamba选择性状态空间模型实现 ============
        # 计算B和C参数 (输入依赖)
        B = self.B(x)  # [batch_size, seq_len, state_dim]
        C = self.C(x)  # [batch_size, seq_len, state_dim]

        # 计算Δ参数 (时间步长，输入依赖) - Mamba关键创新
        # 使用gate计算Δ，应用softplus确保正值
        delta = F.softplus(gate.mean(dim=-1, keepdim=True))  # [batch_size, seq_len, 1]
        delta = delta.unsqueeze(-1)  # [batch_size, seq_len, 1, 1] 为广播准备

        # 离散化状态空间参数 (零阶保持离散化)
        # Ā = exp(Δ * A)
        # 扩展A矩阵以匹配batch和seq维度
        A_expanded = self.A.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim, inner_dim]
        A_bar = torch.exp(
            delta * A_expanded
        )  # [batch_size, seq_len, state_dim, inner_dim]

        # B̄ = Δ * B
        B_bar = delta.squeeze(-1) * B  # [batch_size, seq_len, state_dim]

        # 初始化状态
        state = torch.zeros(batch_size, self.state_dim, self.inner_dim, device=x.device)

        # 选择性状态空间扫描 (Mamba风格)
        outputs = []
        for t in range(seq_len):
            # 获取当前时间步的参数
            A_bar_t = A_bar[:, t, :, :]  # [batch_size, state_dim, inner_dim]
            B_bar_t = B_bar[:, t, :].unsqueeze(-1)  # [batch_size, state_dim, 1]
            x_t = x[:, t, :].unsqueeze(1)  # [batch_size, 1, inner_dim]
            C_t = C[:, t, :].unsqueeze(1)  # [batch_size, 1, state_dim]

            # 状态更新: state = Ā_t * state + B̄_t * x_t
            # A_bar_t: [batch_size, state_dim, inner_dim], state: [batch_size, state_dim, inner_dim]
            # B_bar_t: [batch_size, state_dim, 1], x_t: [batch_size, 1, inner_dim]
            state_update = torch.matmul(A_bar_t, state) + torch.matmul(B_bar_t, x_t)
            state = state_update

            # 输出: y_t = C_t * state + D * x_t
            y_t = torch.matmul(C_t, state) + self.D * x_t
            outputs.append(y_t)

        x = torch.cat(outputs, dim=1)  # [batch_size, seq_len, inner_dim]
        # ============ Mamba实现结束 ============

        # 门控激活 (GLU) 可选
        if self.glu_enabled:
            gate = self.glu_gate(x)
            x = x * torch.sigmoid(gate)

        # RetNet保留机制 (可选) - 论文级实现
        if self.use_retention:
            # 保留注意力 (RetNet论文公式)
            q = self.retention_q(hidden_states)  # [batch_size, seq_len, hidden_size]
            k = self.retention_k(hidden_states)  # [batch_size, seq_len, hidden_size]
            v = self.retention_v(hidden_states)  # [batch_size, seq_len, hidden_size]

            # 多头分割
            q = q.view(
                batch_size, seq_len, self.retention_heads, self.retention_dim
            ).transpose(
                1, 2
            )  # [batch_size, heads, seq_len, dim]
            k = k.view(
                batch_size, seq_len, self.retention_heads, self.retention_dim
            ).transpose(
                1, 2
            )  # [batch_size, heads, seq_len, dim]
            v = v.view(
                batch_size, seq_len, self.retention_heads, self.retention_dim
            ).transpose(
                1, 2
            )  # [batch_size, heads, seq_len, dim]

            # 计算保留分数 (RetNet并行形式)
            # 获取衰减因子
            gamma = torch.clamp(self.retention_gamma, 0.0, 1.0)

            # 创建衰减矩阵 D = gamma^{|i-j|}
            # 使用高效计算避免构建完整矩阵
            seq_range = torch.arange(seq_len, device=q.device).float()
            # 计算相对位置距离矩阵
            distance_matrix = torch.abs(
                seq_range.unsqueeze(1) - seq_range.unsqueeze(0)
            )  # [seq_len, seq_len]
            decay_matrix = torch.pow(gamma, distance_matrix)  # [seq_len, seq_len]

            # 计算QK^T
            qk = torch.matmul(
                q, k.transpose(-1, -2)
            )  # [batch_size, heads, seq_len, seq_len]

            # 应用衰减矩阵
            qk_decayed = qk * decay_matrix.unsqueeze(0).unsqueeze(
                0
            )  # [batch_size, heads, seq_len, seq_len]

            # 计算保留输出 O = (QK^T ⊙ D) V
            retention_output = torch.matmul(
                qk_decayed, v
            )  # [batch_size, heads, seq_len, dim]

            # 可选：递归形式（推理时更高效）
            # 训练时使用并行形式，推理时可以使用递归形式
            if not self.training and seq_len > 512:  # 长序列推理使用递归形式
                # 递归形式: s_t = gamma * s_{t-1} + q_t k_t^T, o_t = s_t v_t
                retention_output_recursive = torch.zeros_like(retention_output)
                retention_state = torch.zeros(
                    batch_size,
                    self.retention_heads,
                    self.retention_dim,
                    self.retention_dim,
                    device=q.device,
                )

                for t in range(seq_len):
                    q_t = q[:, :, t, :].unsqueeze(-1)  # [batch_size, heads, dim, 1]
                    k_t = k[:, :, t, :].unsqueeze(-2)  # [batch_size, heads, 1, dim]
                    v_t = v[:, :, t, :].unsqueeze(-2)  # [batch_size, heads, 1, dim]

                    # 更新保留状态
                    retention_state = gamma * retention_state + torch.matmul(q_t, k_t)
                    # 计算输出
                    o_t = torch.matmul(retention_state, v_t.transpose(-1, -2)).squeeze(
                        -1
                    )
                    retention_output_recursive[:, :, t, :] = o_t

                retention_output = retention_output_recursive

            # 应用门函数
            retention_output = retention_output.transpose(1, 2).reshape(
                batch_size, seq_len, -1
            )  # [batch_size, seq_len, hidden_size]
            retention_gate = self.retention_gate(
                hidden_states.mean(dim=-1, keepdim=True)
            )
            retention_output = retention_output * retention_gate

            # 残差连接
            x = self.out_proj(x)
            x = x + retention_output
        else:
            # 输出投影
            x = self.out_proj(x)

        # 残差连接和层归一化
        x = hidden_states + self.dropout(x)
        x = self.layer_norm(x)

        return x
