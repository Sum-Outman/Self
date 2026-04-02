# SystemControlModule - 从self_agi_model.py拆分
"""SystemControl模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

class SystemControlModule(nn.Module):
    """系统控制模块 - 实现多系统协调和资源管理

    功能：
    - 多系统协调：协调多个子系统的工作
    - 资源管理：CPU、GPU、内存、网络资源管理
    - 故障处理：系统故障检测和恢复
    - 性能监控：实时监控系统性能指标

    基于神经网络控制器，支持自适应系统管理
    """

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 系统状态编码器
        self.system_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 资源分配网络
        self.resource_allocation = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 4),  # CPU, GPU, 内存, 网络
            nn.Softmax(dim=-1),  # 资源分配比例
        )

        # 协调网络
        self.coordination_network = nn.Sequential(
            nn.Linear(
                config.hidden_size + 2 * (config.hidden_size // 3),
                config.hidden_size * 2,
            ),  # 1280 = 768 + 2*256
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 故障检测网络
        self.fault_detection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid(),  # 故障概率
        )

        # 性能监控网络
        self.performance_monitor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 3),  # 利用率、延迟、吞吐量
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, system_metrics: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """前向传播

        参数:
            hidden_states: 输入特征 [batch_size, seq_len, hidden_size]
            system_metrics: 系统指标 [batch_size, metrics_dim] (可选)

        返回:
            系统控制输出字典
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 系统状态编码
        system_features = self.system_encoder(hidden_states)

        # 资源分配
        if system_metrics is not None:
            # 处理system_metrics：可能是张量或字典
            if isinstance(system_metrics, dict):
                # 从字典中提取所有张量值并拼接
                metric_tensors = []
                for key, value in system_metrics.items():
                    if isinstance(value, torch.Tensor):
                        # 确保value是2D: [batch_size, feature_dim]
                        if value.dim() == 1:
                            value = value.unsqueeze(-1)  # [batch_size, 1]
                        metric_tensors.append(value)

                if metric_tensors:
                    # 拼接所有指标: [batch_size, total_metrics]
                    system_metrics_tensor = torch.cat(metric_tensors, dim=-1)
                else:
                    # 没有有效张量，使用默认值
                    system_metrics_tensor = torch.zeros(
                        batch_size, 3, device=hidden_states.device
                    )
            else:
                # system_metrics已经是张量
                system_metrics_tensor = system_metrics

            # 现在system_metrics_tensor是张量
            # 检查维度兼容性
            metrics_dim = system_metrics_tensor.shape[-1]
            if metrics_dim != hidden_dim:
                # 动态创建投影层，将指标维度投影到hidden_dim
                if (
                    not hasattr(self, "_metrics_projection")
                    or self._metrics_projection.in_features != metrics_dim
                ):
                    self._metrics_projection = nn.Linear(metrics_dim, hidden_dim).to(
                        system_metrics_tensor.device
                    )
                system_metrics_tensor = self._metrics_projection(system_metrics_tensor)

            # 扩展系统指标以匹配序列长度
            metrics_expanded = system_metrics_tensor.unsqueeze(1).expand(
                -1, seq_len, -1
            )

            resource_input = torch.cat([system_features, metrics_expanded], dim=-1)
            resource_allocation = self.resource_allocation(resource_input.mean(dim=1))
        else:
            resource_allocation = (
                torch.ones(batch_size, 4, device=hidden_states.device) * 0.25
            )

        # 系统协调（假设有多个子系统）
        # 为标准，我们假设有3个子系统需要协调
        subsystem_features = []
        for i in range(3):
            # 为每个子系统生成不同的特征（通过不同的线性变换）
            subsystem_feature = system_features[
                :, :, i * (hidden_dim // 3) : (i + 1) * (hidden_dim // 3)
            ]
            if subsystem_feature.size(-1) < hidden_dim // 3:
                subsystem_feature = F.pad(
                    subsystem_feature, (0, hidden_dim // 3 - subsystem_feature.size(-1))
                )
            subsystem_features.append(subsystem_feature.mean(dim=1))

        # 构建协调输入
        coordination_input = torch.cat(
            [system_features.mean(dim=1), subsystem_features[0], subsystem_features[1]],
            dim=-1,
        )

        coordination_output = self.coordination_network(coordination_input)

        # 故障检测
        fault_probability = self.fault_detection(system_features.mean(dim=1))

        # 性能监控
        performance_metrics = self.performance_monitor(system_features.mean(dim=1))

        return {
            "system_features": system_features,
            "resource_allocation": resource_allocation,
            "coordination_output": coordination_output,
            "fault_probability": fault_probability,
            "performance_metrics": performance_metrics,
            "subsystem_features": subsystem_features,
        }



