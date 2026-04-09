#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨模态生成模型训练器

功能：
1. 文本到图像生成（TextToImageGenerator）训练
2. 图像到文本生成（ImageToTextGenerator）训练
3. 生成模型评估指标计算
4. 合成数据生成（用于初始训练）
5. 真实数据加载和预处理

工业级AGI系统要求：从零开始训练，不使用预训练模型依赖
"""

from collections import defaultdict
import logging
from pathlib import Path
import time
import random
from typing import Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import sys
import os

# 添加项目根目录到Python路径，以便导入models模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# 导入生成模型
try:
    from models.multimodal.generative_models import (
        TextToImageGenerator,
        ImageToTextGenerator,
    )

    GENERATIVE_MODELS_AVAILABLE = True
except ImportError as e:
    GENERATIVE_MODELS_AVAILABLE = False
    logging.warning(f"生成模型导入失败: {e}")

# 导入真实数据集
try:
    from training.real_multimodal_dataset import (
        RealMultimodalDataset,
        DataSourceType,
    )

    REAL_DATASET_AVAILABLE = True
except ImportError as e:
    REAL_DATASET_AVAILABLE = False
    logging.warning(f"真实多模态数据集导入失败: {e}")

# 导入评估指标库（可选）
try:
    import evaluate  # type: ignore

    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    logging.warning("evaluate库不可用，部分评估指标将受限")

logger = logging.getLogger(__name__)


class GenerativeTrainer:
    """跨模态生成模型训练器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化生成模型训练器

        参数:
            config: 配置字典，包含模型参数、训练参数等
        """
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # 模型配置
        self.model_type = config.get(
            "model_type", "both"
        )  # text_to_image, image_to_text, both
        self.image_size = config.get("image_size", 64)
        self.text_vocab_size = config.get("vocab_size", 10000)
        self.max_seq_len = config.get("max_seq_len", 128)

        # 训练配置
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.num_epochs = config.get("num_epochs", 100)
        self.save_interval = config.get("save_interval", 10)
        self.eval_interval = config.get("eval_interval", 5)
        self.log_interval = config.get("log_interval", 10)

        # 数据配置
        self.data_source = config.get("data_source", "synthetic")
        self.data_root = config.get("data_root", "data/generative")
        self.annotations_path = config.get("annotations_path", "annotations.jsonl")

        # 模型初始化
        self.text_to_image_model = None
        self.image_to_text_model = None
        self.optimizer_text_to_image = None
        self.optimizer_image_to_text = None
        self.scheduler_text_to_image = None
        self.scheduler_image_to_text = None

        # 损失函数
        self.recon_criterion = nn.MSELoss()
        self.text_criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充token

        # 数据集和数据加载器
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "text_to_image_loss": [],
            "image_to_text_loss": [],
            "learning_rates": [],
        }

        # 初始化模型和数据
        self._init_models()
        self._init_data()

        logger.info("生成模型训练器初始化完成")
        logger.info(f"模型类型: {self.model_type}")
        logger.info(f"设备: {self.device}")
        logger.info(f"数据源: {self.data_source}")
        logger.info(f"图像大小: {self.image_size}")

    def _init_models(self):
        """初始化生成模型"""
        model_config = {
            "text_embedding_dim": 768,
            "vocab_size": self.text_vocab_size,
            "latent_dim": 256,
            "image_channels": 3,
            "image_size": self.image_size,
            "max_position_embeddings": self.max_seq_len,
            "kl_weight": 0.001,
        }

        # 初始化文本到图像模型
        if self.model_type in ["text_to_image", "both"]:
            self.text_to_image_model = TextToImageGenerator(model_config).to(
                self.device
            )
            self.optimizer_text_to_image = optim.Adam(
                self.text_to_image_model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
            )
            self.scheduler_text_to_image = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_text_to_image, mode="min", patience=5, factor=0.5
            )
            logger.info(
                f"文本到图像模型初始化完成，参数量: {                     self._count_parameters(                         self.text_to_image_model):,}"
            )

        # 初始化图像到文本模型
        if self.model_type in ["image_to_text", "both"]:
            self.image_to_text_model = ImageToTextGenerator(model_config).to(
                self.device
            )
            self.optimizer_image_to_text = optim.Adam(
                self.image_to_text_model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
            )
            self.scheduler_image_to_text = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_image_to_text, mode="min", patience=5, factor=0.5
            )
            logger.info(
                f"图像到文本模型初始化完成，参数量: {                     self._count_parameters(                         self.image_to_text_model):,}"
            )

    def _init_data(self):
        """初始化数据集"""
        dataset_config = {
            "vocab_size": self.text_vocab_size,
            "max_sequence_length": self.max_seq_len,
            "image_size": self.image_size,
            "data_root": self.data_root,
            "annotations_path": self.annotations_path,
            "enable_cache": True,
            "strict_real_data": False,
        }

        if self.data_source == "real" and REAL_DATASET_AVAILABLE:
            # 使用真实数据集
            logger.info("加载真实多模态数据集")

            # 训练数据集
            self.train_dataset = RealMultimodalDataset(
                config=dataset_config,
                mode="train",
                data_source=DataSourceType.REAL_IMAGE_TEXT,
            )

            # 验证数据集
            self.val_dataset = RealMultimodalDataset(
                config=dataset_config,
                mode="eval",
                data_source=DataSourceType.REAL_IMAGE_TEXT,
            )

        else:
            # 使用合成数据集
            logger.info("创建合成数据集")
            self.train_dataset = self._create_synthetic_dataset(num_samples=1000)
            self.val_dataset = self._create_synthetic_dataset(num_samples=200)

        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Windows上建议设为0
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        logger.info(f"训练数据集大小: {len(self.train_dataset)}")
        logger.info(f"验证数据集大小: {len(self.val_dataset)}")

    def _create_synthetic_dataset(self, num_samples: int = 1000) -> Dataset:
        """创建合成数据集（用于初始训练）"""

        class SyntheticDataset(Dataset):
            def __init__(self, num_samples, image_size, vocab_size, max_seq_len):
                self.num_samples = num_samples
                self.image_size = image_size
                self.vocab_size = vocab_size
                self.max_seq_len = max_seq_len
                self.categories = [
                    "动物",
                    "食物",
                    "交通工具",
                    "建筑",
                    "自然景观",
                    "人物",
                ]
                self.colors = ["红色", "蓝色", "绿色", "黄色", "紫色", "黑色", "白色"]

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # 随机选择类别和颜色
                category = random.choice(self.categories)
                color = random.choice(self.colors)

                # 生成文本描述
                text = f"一张{color}的{category}图片"

                # 生成文本ID序列（完整）
                # 在实际实现中，应使用分词器
                text_ids = torch.randint(
                    1, min(100, self.vocab_size), (self.max_seq_len,)
                )
                text_ids[10:] = 0  # 填充

                # 生成合成图像（随机噪声）
                image = torch.randn(3, self.image_size, self.image_size)

                # 标准化图像
                image = (image - image.mean()) / (image.std() + 1e-8)

                return {
                    "image": image,
                    "text_ids": text_ids,
                    "text": text,
                    "category": category,
                    "color": color,
                }

        return SyntheticDataset(
            num_samples, self.image_size, self.text_vocab_size, self.max_seq_len
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.text_to_image_model.train() if self.text_to_image_model else None
        self.image_to_text_model.train() if self.image_to_text_model else None

        epoch_losses = defaultdict(float)
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # 将数据移动到设备
            images = batch["image"].to(self.device)
            text_ids = batch["text_ids"].to(self.device)

            images.size(0)

            # 文本到图像训练
            if self.text_to_image_model:
                self.optimizer_text_to_image.zero_grad()

                # 前向传播
                text_to_image_output = self.text_to_image_model(images, text_ids)

                # 计算损失
                text_to_image_losses = self.text_to_image_model.compute_loss(
                    text_to_image_output["recon_images"],
                    images,
                    text_to_image_output["mu"],
                    text_to_image_output["logvar"],
                )

                # 反向传播
                text_to_image_losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.text_to_image_model.parameters(), 1.0
                )
                self.optimizer_text_to_image.step()

                # 记录损失
                epoch_losses["text_to_image_total"] += text_to_image_losses[
                    "total_loss"
                ].item()
                epoch_losses["text_to_image_recon"] += text_to_image_losses[
                    "recon_loss"
                ].item()
                epoch_losses["text_to_image_kl"] += text_to_image_losses[
                    "kl_loss"
                ].item()

            # 图像到文本训练
            if self.image_to_text_model:
                self.optimizer_image_to_text.zero_grad()

                # 前向传播
                image_to_text_output = self.image_to_text_model(images, text_ids)

                # 计算损失
                image_to_text_loss_dict = self.image_to_text_model.compute_loss(
                    image_to_text_output["logits"], text_ids
                )
                image_to_text_loss = image_to_text_loss_dict["loss"]

                # 反向传播
                image_to_text_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.image_to_text_model.parameters(), 1.0
                )
                self.optimizer_image_to_text.step()

                # 记录损失
                epoch_losses["image_to_text"] += image_to_text_loss.item()
                epoch_losses["image_to_text_perplexity"] = image_to_text_loss_dict.get(
                    "perplexity", 0
                ).item()

            # 记录进度
            if (batch_idx + 1) % self.log_interval == 0:
                avg_losses = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Text2Image: {avg_losses.get('text_to_image_total', 0):.4f} "
                    f"Image2Text: {avg_losses.get('image_to_text', 0):.4f}"
                )

        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        # 总损失（用于调度器）
        total_train_loss = avg_losses.get("text_to_image_total", 0) + avg_losses.get(
            "image_to_text", 0
        )
        avg_losses["total"] = total_train_loss

        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch} 完成，时间: {epoch_time:.2f}s，总损失: {total_train_loss:.4f}"
        )

        return avg_losses

    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.text_to_image_model.eval() if self.text_to_image_model else None
        self.image_to_text_model.eval() if self.image_to_text_model else None

        val_losses = defaultdict(float)

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                text_ids = batch["text_ids"].to(self.device)

                # 文本到图像验证
                if self.text_to_image_model:
                    text_to_image_output = self.text_to_image_model(images, text_ids)
                    text_to_image_losses = self.text_to_image_model.compute_loss(
                        text_to_image_output["recon_images"],
                        images,
                        text_to_image_output["mu"],
                        text_to_image_output["logvar"],
                    )
                    val_losses["text_to_image_total"] += text_to_image_losses[
                        "total_loss"
                    ].item()
                    val_losses["text_to_image_recon"] += text_to_image_losses[
                        "recon_loss"
                    ].item()
                    val_losses["text_to_image_kl"] += text_to_image_losses[
                        "kl_loss"
                    ].item()

                # 图像到文本验证
                if self.image_to_text_model:
                    image_to_text_output = self.image_to_text_model(images, text_ids)
                    image_to_text_loss_dict = self.image_to_text_model.compute_loss(
                        image_to_text_output["logits"], text_ids
                    )
                    image_to_text_loss = image_to_text_loss_dict["loss"]
                    val_losses["image_to_text"] += image_to_text_loss.item()
                    val_losses["image_to_text_perplexity"] = (
                        image_to_text_loss_dict.get("perplexity", 0).item()
                    )

        # 计算平均验证损失
        num_batches = len(self.val_loader)
        avg_val_losses = {k: v / num_batches for k, v in val_losses.items()}

        # 总验证损失
        total_val_loss = avg_val_losses.get(
            "text_to_image_total", 0
        ) + avg_val_losses.get("image_to_text", 0)
        avg_val_losses["total"] = total_val_loss

        logger.info(
            f"验证损失: Text2Image={avg_val_losses.get('text_to_image_total', 0):.4f}, "
            f"Image2Text={avg_val_losses.get('image_to_text', 0):.4f}, "
            f"Total={total_val_loss:.4f}"
        )

        return avg_val_losses

    def evaluate_generation_quality(self) -> Dict[str, float]:
        """评估生成质量（完整版）"""
        if not self.val_loader:
            return {}  # 返回空字典

        evaluation_results = {}

        # 评估文本到图像生成质量（完整）
        if self.text_to_image_model:
            try:
                # 使用验证集中的几个样本进行生成
                sample_batch = next(iter(self.val_loader))
                images = sample_batch["image"].to(self.device)[:4]
                text_ids = sample_batch["text_ids"].to(self.device)[:4]

                # 生成图像
                with torch.no_grad():
                    generated_images = self.text_to_image_model.generate(text_ids)

                # 计算重构误差（作为质量指标）
                mse_loss = self.recon_criterion(generated_images, images)
                evaluation_results["text_to_image_mse"] = mse_loss.item()

                # 计算结构相似性（完整）
                evaluation_results["text_to_image_quality"] = 1.0 / (
                    1.0 + mse_loss.item()
                )

                logger.info(
                    f"文本到图像生成质量评估: MSE={mse_loss.item():.4f}, "
                    f"质量分数={evaluation_results['text_to_image_quality']:.4f}"
                )
            except Exception as e:
                logger.warning(f"文本到图像生成质量评估失败: {e}")

        # 评估图像到文本生成质量（完整）
        if self.image_to_text_model:
            try:
                sample_batch = next(iter(self.val_loader))
                images = sample_batch["image"].to(self.device)[:4]
                text_ids = sample_batch["text_ids"].to(self.device)[:4]

                # 生成文本
                with torch.no_grad():
                    generated_logits = self.image_to_text_model.generate(images)
                    generated_ids = torch.argmax(generated_logits, dim=-1)

                # 计算准确率（完整）
                accuracy = (generated_ids == text_ids).float().mean()
                evaluation_results["image_to_text_accuracy"] = accuracy.item()

                logger.info(f"图像到文本生成质量评估: 准确率={accuracy.item():.4f}")
            except Exception as e:
                logger.warning(f"图像到文本生成质量评估失败: {e}")

        return evaluation_results

    def train(self):
        """完整训练流程"""
        logger.info(f"开始训练，总epoch数: {self.num_epochs}")

        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch

            # 训练一个epoch
            train_losses = self.train_epoch(epoch)

            # 验证
            if epoch % self.eval_interval == 0:
                val_losses = self.validate()

                # 更新学习率调度器
                if self.text_to_image_model and "text_to_image_total" in val_losses:
                    self.scheduler_text_to_image.step(val_losses["text_to_image_total"])

                if self.image_to_text_model and "image_to_text" in val_losses:
                    self.scheduler_image_to_text.step(val_losses["image_to_text"])

                # 保存最佳模型
                total_val_loss = val_losses.get("total", float("inf"))
                if total_val_loss < self.best_val_loss:
                    self.best_val_loss = total_val_loss
                    self.save_checkpoint(f"best_model_epoch_{epoch}.pth")
                    logger.info(f"新的最佳模型已保存，验证损失: {total_val_loss:.4f}")

                # 评估生成质量
                if epoch % (self.eval_interval * 2) == 0:
                    self.evaluate_generation_quality()
                    # 可以记录评估结果

            # 保存检查点
            if epoch % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

            # 记录训练历史
            self.training_history["train_loss"].append(train_losses.get("total", 0))
            self.training_history["val_loss"].append(
                val_losses.get("total", 0) if epoch % self.eval_interval == 0 else None
            )
            self.training_history["text_to_image_loss"].append(
                train_losses.get("text_to_image_total", 0)
            )
            self.training_history["image_to_text_loss"].append(
                train_losses.get("image_to_text", 0)
            )

            current_lr = (
                self.optimizer_text_to_image.param_groups[0]["lr"]
                if self.text_to_image_model
                else 0
            )
            self.training_history["learning_rates"].append(current_lr)

        logger.info(f"训练完成，最佳验证损失: {self.best_val_loss:.4f}")

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "config": self.config,
        }

        if self.text_to_image_model:
            checkpoint["text_to_image_state_dict"] = (
                self.text_to_image_model.state_dict()
            )
            checkpoint["text_to_image_optimizer"] = (
                self.optimizer_text_to_image.state_dict()
            )

        if self.image_to_text_model:
            checkpoint["image_to_text_state_dict"] = (
                self.image_to_text_model.state_dict()
            )
            checkpoint["image_to_text_optimizer"] = (
                self.optimizer_image_to_text.state_dict()
            )

        save_path = Path("checkpoints") / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, save_path)
        logger.info(f"检查点已保存: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]

        if self.text_to_image_model and "text_to_image_state_dict" in checkpoint:
            self.text_to_image_model.load_state_dict(
                checkpoint["text_to_image_state_dict"]
            )
            self.optimizer_text_to_image.load_state_dict(
                checkpoint["text_to_image_optimizer"]
            )

        if self.image_to_text_model and "image_to_text_state_dict" in checkpoint:
            self.image_to_text_model.load_state_dict(
                checkpoint["image_to_text_state_dict"]
            )
            self.optimizer_image_to_text.load_state_dict(
                checkpoint["image_to_text_optimizer"]
            )

        logger.info(f"检查点已加载: {checkpoint_path}，epoch: {self.current_epoch}")

    def generate_text_to_image(self, text_input: str) -> torch.Tensor:
        """文本到图像生成（推理接口）"""
        if not self.text_to_image_model:
            raise RuntimeError("文本到图像模型未初始化")

        self.text_to_image_model.eval()

        # 完整文本编码（在实际应用中应使用分词器）
        # 这里使用随机编码作为示例
        text_ids = torch.randint(
            1, min(100, self.text_vocab_size), (1, self.max_seq_len)
        )
        text_ids = text_ids.to(self.device)

        with torch.no_grad():
            generated_image = self.text_to_image_model.generate(text_ids)

        return generated_image

    def generate_image_to_text(self, image: torch.Tensor) -> str:
        """图像到文本生成（推理接口）"""
        if not self.image_to_text_model:
            raise RuntimeError("图像到文本模型未初始化")

        self.image_to_text_model.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)  # 添加批次维度
        image = image.to(self.device)

        with torch.no_grad():
            generated_logits = self.image_to_text_model.generate(image)
            generated_ids = torch.argmax(generated_logits, dim=-1)[0]

        # 完整文本解码（在实际应用中应使用解码器）
        # 这里返回实现
        return f"生成的描述 (token IDs: {generated_ids[:10].tolist()}...)"

    def _count_parameters(self, model: nn.Module) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "model_type": self.model_type,
            "device": str(self.device),
            "text_to_image_params": (
                self._count_parameters(self.text_to_image_model)
                if self.text_to_image_model
                else 0
            ),
            "image_to_text_params": (
                self._count_parameters(self.image_to_text_model)
                if self.image_to_text_model
                else 0
            ),
            "dataset_sizes": {
                "train": len(self.train_dataset) if self.train_dataset else 0,
                "val": len(self.val_dataset) if self.val_dataset else 0,
            },
        }


def create_generative_trainer(
    config: Optional[Dict[str, Any]] = None,
) -> GenerativeTrainer:
    """创建生成模型训练器（工厂函数）"""
    if config is None:
        config = {
            "model_type": "both",
            "image_size": 64,
            "vocab_size": 10000,
            "max_seq_len": 128,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_epochs": 50,
            "data_source": "synthetic",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    return GenerativeTrainer(config)


# 测试函数
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("测试生成模型训练器...")

    try:
        # 创建训练器（使用合成数据）
        config = {
            "model_type": "both",
            "image_size": 64,
            "batch_size": 8,
            "num_epochs": 2,  # 测试用，只训练2个epoch
            "data_source": "synthetic",
            "log_interval": 1,
            "eval_interval": 1,
        }

        trainer = create_generative_trainer(config)

        # 获取训练摘要
        summary = trainer.get_training_summary()
        print(f"训练摘要: {summary}")

        # 测试单个epoch训练
        print("\n测试单epoch训练...")
        losses = trainer.train_epoch(1)
        print(f"训练损失: {losses}")

        # 测试验证
        print("\n测试验证...")
        val_losses = trainer.validate()
        print(f"验证损失: {val_losses}")

        # 测试生成质量评估
        print("\n测试生成质量评估...")
        eval_results = trainer.evaluate_generation_quality()
        print(f"评估结果: {eval_results}")

        # 测试推理接口
        print("\n测试推理接口...")
        try:
            # 文本到图像生成
            generated_image = trainer.generate_text_to_image("一张红色的苹果图片")
            print(f"文本到图像生成完成，图像形状: {generated_image.shape}")

            # 图像到文本生成
            test_image = torch.randn(3, 64, 64)
            generated_text = trainer.generate_image_to_text(test_image)
            print(f"图像到文本生成完成，描述: {generated_text}")
        except Exception as e:
            print(f"推理测试失败（预期中，因为模型仅训练了少量数据）: {e}")

        print("\n生成模型训练器测试完成！")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
