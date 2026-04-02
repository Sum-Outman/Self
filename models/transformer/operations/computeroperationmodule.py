# ComputerOperationModule - 从self_agi_model.py拆分
"""ComputerOperation模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

class ComputerOperationModule(nn.Module):
    """计算机操作模块 - 控制机器人操作电脑

    功能：
    - 键盘操作：生成键盘按键序列
    - 鼠标操作：控制鼠标移动、点击、滚动
    - 命令行控制：生成和执行命令行命令
    - 前端网页控制：通过Web界面控制电脑
    - 屏幕理解：理解屏幕内容并做出相应操作

    支持两种操作模式：
    1. 实体机器人操作：通过机器人手臂操作物理键盘和鼠标
    2. 软件控制：直接通过命令行或API控制电脑
    """

    def __init__(self, config: AGIModelConfig):
        super().__init__()
        self.config = config

        # 键盘操作编码器 - 将意图转换为键盘操作
        self.keyboard_operation_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
        )

        # 键盘按键预测器 - 预测按键序列
        self.key_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 128),  # 128个常见按键
            nn.Softmax(dim=-1),
        )

        # 鼠标操作编码器 - 控制鼠标移动和点击
        self.mouse_operation_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 鼠标动作预测器 - 预测鼠标动作（移动、点击、滚动）
        self.mouse_action_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(
                config.hidden_size // 2, 6
            ),  # 6种鼠标动作：移动、左键点击、右键点击、中键点击、滚动上、滚动下
            nn.Softmax(dim=-1),
        )

        # 鼠标位置回归器 - 预测鼠标坐标
        self.mouse_position_regressor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 2),  # (x, y)坐标
            nn.Sigmoid(),  # 归一化到0-1（屏幕相对位置）
        )

        # 命令行命令生成器 - 生成命令行命令
        self.command_generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 命令序列解码器 - 生成命令序列
        self.command_decoder = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,
            batch_first=True,
        )

        # 屏幕理解编码器 - 理解屏幕内容
        self.screen_understanding_encoder = nn.Sequential(
            nn.Linear(config.image_embedding_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        # 操作决策网络 - 决定执行什么操作
        self.operation_decision_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),  # 意图 + 屏幕状态
            nn.GELU(),
            nn.Linear(config.hidden_size, 4),  # 4种操作类型：键盘、鼠标、命令行、无操作
            nn.Softmax(dim=-1),
        )

        # 操作序列规划器 - 规划操作序列
        self.operation_sequence_planner = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,
            batch_first=True,
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        user_intent: torch.Tensor,
        screen_state: Optional[torch.Tensor] = None,
        current_computer_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """前向传播 - 生成计算机操作指令

        参数:
            user_intent: 用户意图 [batch_size, hidden_dim]
            screen_state: 屏幕状态（截图特征）[batch_size, image_dim] (可选)
            current_computer_state: 当前计算机状态字典 (可选)

        返回:
            包含计算机操作指令的字典
        """
        results = {
            "keyboard_operations": None,
            "mouse_operations": None,
            "command_line_commands": None,
            "operation_decision": None,
            "operation_sequence": None,
            "screen_understanding": None,
            "operation_confidence": 0.0,
        }

        batch_size = user_intent.shape[0]

        # 1. 操作决策
        if screen_state is not None:
            # 编码屏幕状态
            screen_encoded = self.screen_understanding_encoder(screen_state)
            results["screen_understanding"] = screen_encoded

            # 结合用户意图和屏幕状态做决策
            decision_input = torch.cat([user_intent, screen_encoded], dim=-1)
        else:
            decision_input = user_intent

        operation_decision = self.operation_decision_net(decision_input)
        results["operation_decision"] = operation_decision

        # 2. 键盘操作生成
        keyboard_encoded = self.keyboard_operation_encoder(user_intent)
        key_predictions = self.key_predictor(keyboard_encoded)
        results["keyboard_operations"] = {
            "key_predictions": key_predictions,
            "key_sequence": self._decode_key_sequence(key_predictions),
        }

        # 3. 鼠标操作生成
        mouse_encoded = self.mouse_operation_encoder(user_intent)
        mouse_actions = self.mouse_action_predictor(mouse_encoded)
        mouse_positions = self.mouse_position_regressor(mouse_encoded)
        results["mouse_operations"] = {
            "mouse_actions": mouse_actions,
            "mouse_positions": mouse_positions,
            "click_positions": (
                mouse_positions if mouse_actions[:, 1:4].sum() > 0.5 else None
            ),  # 如果有点击动作
        }

        # 4. 命令行命令生成
        command_encoded = self.command_generator(user_intent)
        command_sequence, _ = self.command_decoder(command_encoded.unsqueeze(1))
        results["command_line_commands"] = {
            "command_features": command_sequence,
            "command_embeddings": command_encoded,
        }

        # 5. 操作序列规划
        operation_sequence_input = torch.cat(
            [
                keyboard_encoded.unsqueeze(1),
                mouse_encoded.unsqueeze(1),
                command_encoded.unsqueeze(1),
            ],
            dim=1,
        )

        operation_sequence, (hidden_state, cell_state) = (
            self.operation_sequence_planner(operation_sequence_input)
        )
        results["operation_sequence"] = {
            "sequence_features": operation_sequence,
            "hidden_state": hidden_state,
            "cell_state": cell_state,
        }

        # 6. 计算操作置信度
        # 基于决策概率和操作复杂度
        decision_confidence = operation_decision.max(dim=-1)[0].mean().item()
        results["operation_confidence"] = decision_confidence * 0.8  # 调整置信度

        return results

    def _decode_key_sequence(self, key_predictions: torch.Tensor) -> List[List[str]]:
        """解码按键预测为按键序列"""
        batch_size = key_predictions.shape[0]
        key_sequences = []

        # 完整解码：选择概率最高的前3个按键
        for i in range(batch_size):
            probs = key_predictions[i]
            top_k = 3
            top_indices = torch.topk(probs, top_k).indices.tolist()

            # 将索引映射为按键字符（标准映射）
            key_chars = []
            for idx in top_indices:
                if idx < 26:
                    key_chars.append(chr(ord("a") + idx))
                elif idx < 52:
                    key_chars.append(chr(ord("A") + idx - 26))
                elif idx < 62:
                    key_chars.append(chr(ord("0") + idx - 52))
                elif idx == 62:
                    key_chars.append(" ")
                elif idx == 63:
                    key_chars.append("\n")
                else:
                    key_chars.append(f"KEY_{idx}")

            key_sequences.append(key_chars)

        return key_sequences


# 设备操作学习模块

