#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨模态生成模型模块

包含：
1. TextToImageGenerator - 文本到图像生成（条件VAE）
2. ImageToTextGenerator - 图像到文本生成（图像描述生成）

工业级AGI系统要求：从零开始训练，不使用预训练模型依赖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# 导入现有的编码器
from .text_encoder import IndustrialTextEncoder
from .vision_encoder import IndustrialVisionEncoder


class TextToImageGenerator(nn.Module):
    """文本到图像生成器 - 基于条件变分自编码器（Conditional VAE）
    
    架构：
    1. 文本编码器：将文本编码为条件向量
    2. VAE编码器：将图像编码为潜在变量（均值和对数方差）
    3. VAE解码器：从潜在变量生成图像，以文本条件为条件
    4. 重参数化技巧：从分布中采样
    
    训练目标：
    - 重构损失（图像重建）
    - KL散度损失（潜在空间正则化）
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 文本编码器（条件编码）
        text_embedding_dim = config.get("text_embedding_dim", 768)
        vocab_size = config.get("vocab_size", 100000)
        num_layers = config.get("num_layers", 6)  # 生成任务可以使用较少的层数
        
        self.text_encoder = IndustrialTextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            num_layers=num_layers,
            max_position_embeddings=config.get("max_position_embeddings", 512)
        )
        
        # VAE参数
        self.latent_dim = config.get("latent_dim", 256)
        self.image_channels = config.get("image_channels", 3)
        self.image_size = config.get("image_size", 64)  # 生成较小图像以降低复杂度
        
        # VAE编码器：图像 -> 潜在变量
        self.encoder = self._build_encoder(text_embedding_dim)
        
        # VAE解码器：潜在变量 -> 图像
        self.decoder = self._build_decoder(text_embedding_dim)
        
        # 潜在空间投影层
        self.fc_mu = nn.Linear(512 + text_embedding_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(512 + text_embedding_dim, self.latent_dim)
        self.fc_z = nn.Linear(self.latent_dim + text_embedding_dim, self.latent_dim + text_embedding_dim)
        
        logger.info(f"初始化TextToImageGenerator: 潜在维度={self.latent_dim}, 图像大小={self.image_size}")
    
    def _build_encoder(self, condition_dim: int) -> nn.Module:
        """构建VAE编码器"""
        return nn.Sequential(
            # 输入: [batch, 3, image_size, image_size]
            nn.Conv2d(self.image_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),  # 输出: [batch, 512]
        )
    
    def _build_decoder(self, condition_dim: int) -> nn.Module:
        """构建VAE解码器"""
        # 计算起始特征图大小
        # 64x64图像经过4次stride=2下采样后为4x4
        start_size = self.image_size // 16
        
        return nn.Sequential(
            # 起始投影: [batch, latent_dim + condition_dim] -> [batch, 512 * start_size * start_size]
            nn.Linear(self.latent_dim + condition_dim, 512 * start_size * start_size),
            nn.Unflatten(1, (512, start_size, start_size)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 转置卷积上采样
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出值在[-1, 1]范围内
        )
    
    def encode_text(self, text_input: torch.Tensor) -> torch.Tensor:
        """编码文本为条件向量"""
        text_features = self.text_encoder(text_input)
        # 如果文本特征是3D (batch_size, seq_len, embedding_dim)，在序列维度上取平均
        if text_features.dim() == 3:
            text_features = text_features.mean(dim=1)
        return text_features
    
    def encode(self, images: torch.Tensor, text_condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码图像为潜在变量（带文本条件）"""
        # 提取图像特征
        image_features = self.encoder(images)
        
        # 拼接图像特征和文本条件
        combined = torch.cat([image_features, text_condition], dim=-1)
        
        # 计算潜在分布的参数
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧：从分布中采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, text_condition: torch.Tensor) -> torch.Tensor:
        """从潜在变量解码图像（带文本条件）"""
        # 拼接潜在变量和文本条件
        combined = torch.cat([z, text_condition], dim=-1)
        z_projected = self.fc_z(combined)
        
        # 解码为图像
        recon_images = self.decoder(z_projected)
        return recon_images
    
    def forward(self, images: torch.Tensor, text_input: torch.Tensor) -> Dict[str, Any]:
        """完整的前向传播：编码 -> 重参数化 -> 解码"""
        # 编码文本
        text_condition = self.encode_text(text_input)
        
        # 编码图像
        mu, logvar = self.encode(images, text_condition)
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # 解码图像
        recon_images = self.decode(z, text_condition)
        
        return {
            "recon_images": recon_images,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "text_condition": text_condition,
        }
    
    def generate(self, text_input: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """生成图像（推理时使用）"""
        batch_size = text_input.size(0) if text_input.dim() > 1 else 1
        
        # 编码文本
        text_condition = self.encode_text(text_input)
        
        # 从标准正态分布采样
        z = torch.randn(batch_size, self.latent_dim, device=text_input.device)
        
        # 解码图像
        generated_images = self.decode(z, text_condition)
        
        return generated_images
    
    def compute_loss(self, recon_images: torch.Tensor, original_images: torch.Tensor,
                    mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算VAE损失：重构损失 + KL散度"""
        batch_size = original_images.size(0)
        
        # 重构损失（均方误差）
        recon_loss = F.mse_loss(recon_images, original_images, reduction='sum') / batch_size
        
        # KL散度损失（负KL散度，最小化）
        # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # 总损失
        total_loss = recon_loss + self.config.get("kl_weight", 0.001) * kl_loss
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


class ImageToTextGenerator(nn.Module):
    """图像到文本生成器 - 基于编码器-解码器架构
    
    架构：
    1. 图像编码器：将图像编码为特征向量
    2. 文本解码器：Transformer解码器，以图像特征为条件生成文本
    
    训练目标：
    - 交叉熵损失（文本生成）
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 图像编码器
        image_embedding_dim = config.get("image_embedding_dim", 768)
        image_size = config.get("image_size", 224)
        patch_size = config.get("patch_size", 16)
        num_layers = config.get("num_layers", 6)
        
        self.image_encoder = IndustrialVisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=image_embedding_dim,
            num_layers=num_layers
        )
        
        # 文本解码器参数
        self.vocab_size = config.get("vocab_size", 100000)
        self.text_embedding_dim = config.get("text_embedding_dim", 768)
        self.max_seq_len = config.get("max_seq_len", 128)
        
        # 文本嵌入层
        self.text_embedding = nn.Embedding(self.vocab_size, self.text_embedding_dim)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.text_embedding_dim)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.text_embedding_dim,
            nhead=8,
            dim_feedforward=self.text_embedding_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(self.text_embedding_dim, self.vocab_size)
        
        # 图像特征到解码器初始状态的投影
        self.image_projection = nn.Linear(image_embedding_dim, self.text_embedding_dim)
        
        logger.info(f"初始化ImageToTextGenerator: 词汇表大小={self.vocab_size}, 最大序列长度={self.max_seq_len}")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像为特征向量"""
        image_features = self.image_encoder(images)
        
        # 图像编码器输出可能是[batch, num_patches+1, dim]或[batch, dim]
        if image_features.dim() == 3:
            # 取CLS token或平均池化
            image_features = image_features[:, 0, :]  # CLS token
        
        # 投影到解码器维度
        projected_features = self.image_projection(image_features)
        
        return projected_features
    
    def generate_text(self, images: torch.Tensor, max_length: int = 50, 
                     temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """生成文本描述（推理时使用）"""
        batch_size = images.size(0)
        device = images.device
        
        # 编码图像
        image_features = self.encode_image(images)  # [batch, text_embedding_dim]
        
        # 初始输入：开始token（假设0是开始token）
        start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # 生成序列
        generated_tokens = start_token
        
        for i in range(max_length - 1):
            # 创建位置ID
            seq_len = generated_tokens.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
            
            # 文本嵌入
            token_embeds = self.text_embedding(generated_tokens)
            pos_embeds = self.position_embedding(positions)
            text_embeds = token_embeds + pos_embeds
            
            # 创建掩码（防止关注未来token）
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            
            # 准备图像特征作为编码器输出
            memory = image_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, text_embedding_dim]
            
            # Transformer解码
            decoder_output = self.transformer_decoder(
                tgt=text_embeds,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=None
            )
            
            # 预测下一个token
            logits = self.output_projection(decoder_output[:, -1:, :])  # 只取最后一个位置
            
            # 应用温度采样
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-k采样
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                # 从top-k中采样
                probs = F.softmax(top_k_logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), 1)
                next_token = top_k_indices.gather(-1, next_token.unsqueeze(-1))
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), 1)
            
            # 添加到生成序列
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            
            # 检查是否遇到结束token（假设1是结束token）
            if (next_token == 1).all():
                break
        
        return generated_tokens
    
    def forward(self, images: torch.Tensor, text_input: torch.Tensor) -> Dict[str, Any]:
        """前向传播：训练时使用教师强制"""
        batch_size = text_input.size(0)
        seq_len = text_input.size(1)
        device = images.device
        
        # 编码图像
        image_features = self.encode_image(images)  # [batch, text_embedding_dim]
        
        # 创建位置ID
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        
        # 文本嵌入
        token_embeds = self.text_embedding(text_input)
        pos_embeds = self.position_embedding(positions)
        text_embeds = token_embeds + pos_embeds
        
        # 创建掩码（防止关注未来token）
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        # 准备图像特征作为编码器输出
        memory = image_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, text_embedding_dim]
        
        # Transformer解码
        decoder_output = self.transformer_decoder(
            tgt=text_embeds,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None
        )
        
        # 预测logits
        logits = self.output_projection(decoder_output)
        
        return {
            "logits": logits,
            "decoder_output": decoder_output,
            "image_features": image_features,
        }
    
    def compute_loss(self, logits: torch.Tensor, target_text: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算文本生成损失"""
        # 调整形状：从[batch, seq_len, vocab_size]到[batch*seq_len, vocab_size]
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_text.reshape(-1)
        
        # 交叉熵损失
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)  # 忽略填充token（假设0是填充）
        
        return {
            "loss": loss,
            "perplexity": torch.exp(loss),
        }


class CrossModalGenerationManager(nn.Module):
    """跨模态生成管理器 - 统一管理文本到图像和图像到文本生成"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 初始化两个生成器
        self.text_to_image = TextToImageGenerator(config)
        self.image_to_text = ImageToTextGenerator(config)
        
        logger.info("初始化CrossModalGenerationManager: 包含文本到图像和图像到文本生成器")
    
    def forward(self, mode: str, **kwargs) -> Dict[str, Any]:
        """根据模式调用相应的生成器"""
        if mode == "text_to_image":
            return self.text_to_image(**kwargs)
        elif mode == "image_to_text":
            return self.image_to_text(**kwargs)
        else:
            raise ValueError(f"不支持的生成模式: {mode}")
    
    def generate(self, mode: str, **kwargs) -> torch.Tensor:
        """生成新内容"""
        if mode == "text_to_image":
            return self.text_to_image.generate(**kwargs)
        elif mode == "image_to_text":
            return self.image_to_text.generate_text(**kwargs)
        else:
            raise ValueError(f"不支持的生成模式: {mode}")
    
    def compute_loss(self, mode: str, **kwargs) -> Dict[str, torch.Tensor]:
        """计算损失"""
        if mode == "text_to_image":
            return self.text_to_image.compute_loss(**kwargs)
        elif mode == "image_to_text":
            return self.image_to_text.compute_loss(**kwargs)
        else:
            raise ValueError(f"不支持的生成模式: {mode}")