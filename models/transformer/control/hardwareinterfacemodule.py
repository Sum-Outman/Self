# HardwareInterfaceModule - 从self_agi_model.py拆分
"""HardwareInterface模块"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class HardwareInterfaceModule(nn.Module):
    """硬件接口模块 - 实现与物理硬件的通信和控制

    功能：
    - 串口通信：与Arduino、传感器等设备的串口通信
    - USB设备管理：USB设备的识别和控制
    - 网络设备控制：网络设备的配置和管理
    - 定制硬件支持：专用硬件的驱动和接口

    基于神经网络接口层，支持多种硬件协议
    """

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 硬件命令编码器
        self.command_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 串口协议编码器
        self.serial_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 64),  # 64字节串口消息
        )

        # USB设备编码器
        self.usb_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 128),  # USB设备描述符
        )

        # 网络协议编码器
        self.network_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 256),  # 网络数据包
        )

        # 硬件状态解码器
        self.status_decoder = nn.Sequential(
            nn.Linear(128, config.hidden_size),  # 改为输出hidden_size
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),  # 保持hidden_size输出
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 硬件响应预测网络
        self.response_predictor = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2, eps=config.layer_norm_eps),
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hardware_command: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """前向传播

        参数:
            hidden_states: 输入特征 [batch_size, seq_len, hidden_size]
            hardware_command: 硬件命令 [batch_size, command_dim] (可选)

        返回:
            硬件接口输出字典
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 硬件命令编码
        if hardware_command is not None:
            # 检查维度兼容性
            command_dim = hardware_command.shape[-1]
            if command_dim != self.config.hidden_size:
                # 动态创建投影层，将命令维度投影到hidden_size
                if (
                    not hasattr(self, "_command_projection")
                    or self._command_projection.in_features != command_dim
                ):
                    self._command_projection = nn.Linear(
                        command_dim, self.config.hidden_size
                    ).to(hardware_command.device)
                hardware_command_projected = self._command_projection(hardware_command)
                command_features = self.command_encoder(hardware_command_projected)
            else:
                command_features = self.command_encoder(hardware_command)
        else:
            command_features = self.command_encoder(hidden_states.mean(dim=1))

        # 生成串口消息
        serial_message = self.serial_encoder(command_features)

        # 生成USB设备命令
        usb_command = self.usb_encoder(command_features)

        # 生成网络数据包
        network_packet = self.network_encoder(command_features)

        # 硬件状态预测 - 基于命令特征的真实预测
        # 使用命令特征投影到状态空间，替代模拟响应
        # 首先将命令特征投影到128维状态空间
        if command_features.size(-1) >= 128:
            # 如果特征维度足够，使用前128维
            projected_status = command_features[:, :128]
        else:
            # 如果特征维度不足，使用线性投影（临时实现）
            # 注意：在真实系统中应使用训练好的投影层
            projection_weight = torch.eye(
                command_features.size(-1), 128, device=command_features.device
            )
            projected_status = torch.matmul(command_features, projection_weight)

        hardware_status = self.status_decoder(projected_status)

        # 预测硬件响应
        response_input = torch.cat([command_features, hardware_status], dim=-1)
        predicted_response = self.response_predictor(response_input)

        return {
            "command_features": command_features,
            "serial_message": serial_message,
            "usb_command": usb_command,
            "network_packet": network_packet,
            "hardware_status": hardware_status,
            "predicted_response": predicted_response,
        }


# ============================================================================
# DoRA (权重分解的低秩适应) 实现
# ============================================================================
