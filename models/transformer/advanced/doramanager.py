# DoRAManager - 从self_agi_model.py拆分
"""DoRAManager模块"""

import torch.nn as nn


class DoRAManager:
    """DoRA管理器 - 管理模型中的DoRA适配器

    功能:
    - 为模型的线性层添加DoRA适配器
    - 启用/禁用DoRA训练
    - 合并DoRA权重到基础模型
    """

    def __init__(self, model: nn.Module, config: AGIModelConfig):
        self.model = model
        self.config = config
        self.dora_layers = {}

    def inject_dora(self):
        """为模型注入DoRA适配器"""
        if not self.config.dora_enabled:
            logger.info("DoRA未启用，跳过注入")
            return

        logger.info(
            f"为模型注入DoRA适配器: 秩={self.config.dora_rank}, alpha={self.config.dora_alpha}"
        )

        # 遍历模型的所有线性层
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 跳过某些特定层（如输出层）
                if "lm_head" in name or "output" in name:
                    continue

                # 创建DoRA适配器
                dora_layer = DoRALinear(
                    base_layer=module,
                    rank=self.config.dora_rank,
                    alpha=self.config.dora_alpha,
                )

                # 替换原始层
                # 需要更新父模块中的引用
                parent = self._get_parent_module(self.model, name)
                child_name = name.split(".")[-1]
                setattr(parent, child_name, dora_layer)

                # 记录DoRA层
                self.dora_layers[name] = dora_layer

                logger.debug(f"为层 {name} 注入DoRA适配器")

    def _get_parent_module(self, model: nn.Module, full_name: str) -> nn.Module:
        """获取父模块"""
        name_parts = full_name.split(".")
        parent = model

        for part in name_parts[:-1]:
            parent = getattr(parent, part)

        return parent

    def enable_dora_training(self):
        """启用DoRA训练模式"""
        for name, layer in self.dora_layers.items():
            layer.train()
            logger.debug(f"启用DoRA训练: {name}")

    def disable_dora_training(self):
        """禁用DoRA训练模式"""
        for name, layer in self.dora_layers.items():
            layer.eval()
            logger.debug(f"禁用DoRA训练: {name}")

    def merge_dora_weights(self):
        """合并所有DoRA权重到基础模型中"""
        logger.info("合并所有DoRA权重到基础模型中")

        for name, layer in self.dora_layers.items():
            layer.merge_weights()

        logger.info("DoRA权重合并完成")


# 传感器模块
