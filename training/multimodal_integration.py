#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态训练集成模块
将多模态训练框架集成到现有AGI训练系统中

功能：
1. 创建多模态AGI训练器
2. 配置转换
3. 训练流程适配
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# 导入现有AGI训练系统
from training.trainer import AGITrainer, TrainingConfig
from training.multimodal_trainer import MultimodalTrainer, create_default_config


class MultimodalAGITrainer(AGITrainer):
    """多模态AGI训练器 - 集成多模态训练能力
    
    扩展标准AGI训练器以支持多模态训练功能：
    1. 多模态数据处理
    2. 跨模态融合
    3. 渐进式训练策略
    4. 多任务学习
    """
    
    def __init__(
        self,
        multimodal_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[TrainingConfig] = None,
        from_scratch: bool = True,
    ):
        """初始化多模态AGI训练器
        
        参数:
            multimodal_config: 多模态训练配置
            training_config: AGI训练配置
            from_scratch: 是否从零开始训练
        """
        # 创建多模态训练器
        self.multimodal_config = multimodal_config or create_default_config()
        self.multimodal_trainer = MultimodalTrainer(self.multimodal_config)
        
        # 获取多模态处理器作为模型
        model = self.multimodal_trainer.multimodal_processor
        
        # 创建或使用提供的训练配置
        if training_config is None:
            training_config = self._create_training_config_from_multimodal()
        
        # 初始化父类
        super().__init__(
            model=model,
            config=training_config,
            from_scratch=from_scratch
        )
        
        # 保存多模态训练器引用
        self.multimodal_trainer = self.multimodal_trainer
        self.multimodal_processor = model
        
        logger.info("多模态AGI训练器初始化完成")
        logger.info(f"训练模式: {self.training_mode}")
        logger.info(f"设备: {self.device}")
        logger.info(f"多模态配置: {list(self.multimodal_config.keys())}")
    
    def _create_training_config_from_multimodal(self) -> TrainingConfig:
        """从多模态配置创建训练配置"""
        # 使用多模态配置中的参数
        return TrainingConfig(
            batch_size=self.multimodal_config.get('phase1_batch', 32),
            learning_rate=self.multimodal_config.get('phase1_lr', 1e-4),
            num_epochs=self.multimodal_config.get('num_epochs', 20),
            warmup_steps=1000,
            weight_decay=self.multimodal_config.get('weight_decay', 0.01),
            gradient_accumulation_steps=self.multimodal_config.get('gradient_accumulation_steps', 1),
            max_grad_norm=self.multimodal_config.get('max_grad_norm', 1.0),
            logging_steps=self.multimodal_config.get('logging_steps', 10),
            save_steps=self.multimodal_config.get('save_steps', 100),
            eval_steps=500,
            fp16=self.multimodal_config.get('fp16', True),
            use_gpu=self.multimodal_config.get('use_gpu', True),
            gpu_ids=self.multimodal_config.get('gpu_ids', [0]),
            checkpoint_dir=self.multimodal_config.get('checkpoint_dir', 'checkpoints/multimodal'),
            log_dir=self.multimodal_config.get('log_dir', 'logs/multimodal'),
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch - 重写以支持多模态训练"""
        # 设置多模态处理器为训练模式
        self.multimodal_processor.train()
        
        # 调用父类训练方法
        metrics = super().train_epoch(epoch)
        
        # 添加多模态特定指标
        metrics.update({
            "multimodal_mode": True,
            "training_phase": self.multimodal_trainer.progressive_scheduler.get_current_phase(),
        })
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型 - 重写以支持多模态评估"""
        # 设置多模态处理器为评估模式
        self.multimodal_processor.eval()
        
        # 调用父类评估方法
        metrics = super().evaluate()
        
        # 添加多模态特定指标
        metrics.update({
            "multimodal_mode": True,
            "evaluation_phase": "full_multimodal",
        })
        
        return metrics
    
    def enable_multimodal_training(self):
        """启用多模态训练模式"""
        self.training_mode = "multimodal"
        self.logger.info("启用多模态训练模式")
    
    def enable_progressive_training(self):
        """启用渐进式训练策略"""
        # 使用多模态训练器的渐进式调度器
        self.multimodal_trainer.progressive_scheduler.enable()
        self.logger.info("启用渐进式训练策略")
    
    def get_multimodal_status(self) -> Dict[str, Any]:
        """获取多模态训练状态"""
        base_status = self.get_training_status()
        
        multimodal_status = {
            "multimodal_config_keys": list(self.multimodal_config.keys()),
            "multimodal_processor_initialized": hasattr(self.multimodal_processor, 'initialized'),
            "progressive_training_phase": self.multimodal_trainer.progressive_scheduler.current_phase,
            "enabled_tasks": self.multimodal_trainer.loss_fn.tasks if hasattr(self.multimodal_trainer.loss_fn, 'tasks') else [],
        }
        
        base_status.update(multimodal_status)
        return base_status


def create_multimodal_agi_trainer(
    config_dict: Optional[Dict[str, Any]] = None
) -> MultimodalAGITrainer:
    """创建多模态AGI训练器（工厂函数）
    
    参数:
        config_dict: 配置字典，将合并到默认配置中
        
    返回:
        MultimodalAGITrainer: 多模态AGI训练器实例
    """
    # 合并配置
    default_config = create_default_config()
    if config_dict:
        default_config.update(config_dict)
    
    # 创建多模态AGI训练器
    trainer = MultimodalAGITrainer(
        multimodal_config=default_config,
        from_scratch=True
    )
    
    # 启用多模态训练模式
    trainer.enable_multimodal_training()
    
    logger.info("多模态AGI训练器创建成功")
    logger.info(f"配置参数数量: {len(default_config)}")
    
    return trainer


def demo_multimodal_integration():
    """演示多模态集成功能"""
    print("=" * 80)
    print("多模态AGI训练系统集成演示")
    print("基于《多模态实现修复方案.md》实现工业级集成")
    print("=" * 80)
    
    try:
        # 创建多模态AGI训练器
        trainer = create_multimodal_agi_trainer({
            "use_gpu": False,  # 演示使用CPU
            "disable_file_logging": True,  # 禁用文件日志
            "train_samples": 50,
            "eval_samples": 10,
            "num_epochs": 1,  # 仅演示1个epoch
        })
        
        # 获取训练状态
        status = trainer.get_multimodal_status()
        print("训练器状态:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("\n多模态集成演示完成!")
        print("训练器已准备好进行多模态AGI训练")
        
        return trainer
        
    except Exception as e:
        logger.error(f"多模态集成演示失败: {e}")
        raise


if __name__ == "__main__":
    # 运行演示
    demo_multimodal_integration()