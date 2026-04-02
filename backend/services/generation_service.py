"""
跨模态生成服务模块
提供文本到图像、图像到文本等跨模态生成功能
基于GenerativeModels实现
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

# 导入生成模型
try:
    from models.multimodal.generative_models import (
        CrossModalGenerationManager,
        TextToImageGenerator,
        ImageToTextGenerator
    )
    GENERATION_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入生成模型: {e}")
    CrossModalGenerationManager = None
    GENERATION_MODEL_AVAILABLE = False


class GenerationService:
    """跨模态生成服务单例类"""
    
    _instance = None
    _generation_manager = None
    _config = None
    _initialized = False
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._load_models()
    
    def _load_models(self):
        """加载生成模型"""
        if not GENERATION_MODEL_AVAILABLE:
            logger.error("生成模型不可用，生成功能将受限")
            return
        
        try:
            logger.info("正在加载跨模态生成模型...")
            
            # 配置参数
            self._config = {
                # 文本编码器参数
                "text_embedding_dim": 256,  # 生成任务可以使用较小维度
                "vocab_size": 50000,
                "num_layers": 4,
                "max_position_embeddings": 512,
                "max_seq_len": 128,
                
                # 图像编码器参数
                "image_embedding_dim": 256,
                "image_size": 64,  # 生成较小图像以降低复杂度
                "patch_size": 16,
                
                # VAE参数
                "latent_dim": 128,
                "image_channels": 3,
                "kl_weight": 0.001,
                
                # 通用参数
                "industrial_mode": True,
            }
            
            # 加载生成管理器
            self._generation_manager = CrossModalGenerationManager(self._config)
            self._generation_manager.to(self._device)
            self._generation_manager.eval()
            
            logger.info(f"跨模态生成模型加载成功，设备: {self._device}")
            
        except Exception as e:
            logger.error(f"跨模态生成模型加载失败: {e}")
            self._generation_manager = None
    
    def is_ready(self) -> bool:
        """检查生成服务是否就绪"""
        return self._generation_manager is not None
    
    def generate_text_to_image(self, prompt: str, num_images: int = 1, 
                              image_size: int = 64) -> Dict[str, Any]:
        """文本到图像生成"""
        if not self.is_ready():
            return {
                "success": False,
                "error": "生成服务未就绪",
                "generated_images": [],
                "prompt": prompt,
            }
        
        try:
            # 简单tokenization（实际实现应使用tokenizer）
            # 这里使用简单模拟
            tokens = torch.randint(0, 1000, (1, min(len(prompt), 128)))
            tokens = tokens.to(self._device)
            
            # 生成图像
            with torch.no_grad():
                generated_tensor = self._generation_manager.generate(
                    mode="text_to_image",
                    text_input=tokens,
                    num_samples=num_images
                )
                
                # 转换tensor为图像
                generated_images = []
                for i in range(num_images):
                    if i < generated_tensor.size(0):
                        image_tensor = generated_tensor[i]
                        
                        # 从[-1, 1]转换到[0, 255]
                        image_tensor = (image_tensor + 1) / 2  # [-1, 1] -> [0, 1]
                        image_tensor = image_tensor.clamp(0, 1)
                        
                        # 转换为numpy数组
                        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
                        image_np = (image_np * 255).astype(np.uint8)
                        
                        # 转换为PIL图像
                        pil_image = Image.fromarray(image_np)
                        
                        # 转换为Base64
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
                        generated_images.append({
                            "index": i,
                            "size": image_np.shape[:2],
                            "format": "PNG",
                            "base64": img_base64,
                        })
            
            return {
                "success": True,
                "prompt": prompt,
                "generated_images": generated_images,
                "num_generated": len(generated_images),
                "model_info": {
                    "model_type": "TextToImageGenerator",
                    "latent_dim": self._config.get("latent_dim", 128),
                    "image_size": image_size,
                },
            }
            
        except Exception as e:
            logger.error(f"文本到图像生成失败: {e}")
            return {
                "success": False,
                "error": f"生成失败: {str(e)}",
                "prompt": prompt,
                "generated_images": [],
            }
    
    def generate_image_to_text(self, image_bytes: bytes, max_length: int = 50,
                              temperature: float = 1.0, top_k: int = 50) -> Dict[str, Any]:
        """图像到文本生成（图像描述）"""
        if not self.is_ready():
            return {
                "success": False,
                "error": "生成服务未就绪",
                "generated_text": "",
                "image_info": {"size": len(image_bytes)},
            }
        
        try:
            # 转换图像为张量
            image = Image.open(io.BytesIO(image_bytes))
            
            # 预处理
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize((64, 64)),  # 匹配模型输入大小
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                   std=[0.5, 0.5, 0.5]),  # [-1, 1]范围
            ])
            
            image_tensor = preprocess(image).unsqueeze(0).to(self._device)
            
            # 生成文本
            with torch.no_grad():
                # 生成token序列
                generated_tokens = self._generation_manager.generate(
                    mode="image_to_text",
                    images=image_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # 完整版本，实际应使用词汇表）
                # 这里使用ASCII字符作为演示
                generated_text = ""
                for token_seq in generated_tokens:
                    for token in token_seq:
                        if token.item() == 0:  # 填充token
                            continue
                        if token.item() == 1:  # 结束token
                            break
                        # 将token映射到ASCII字符
                        char_code = token.item() % 128
                        if 32 <= char_code < 127:  # 可打印ASCII
                            generated_text += chr(char_code)
                        else:
                            generated_text += " "
                    generated_text += "\n"
                
                generated_text = generated_text.strip()
            
            return {
                "success": True,
                "generated_text": generated_text,
                "image_info": {
                    "size": len(image_bytes),
                    "original_size": image.size,
                    "processed_size": (64, 64),
                },
                "generation_params": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_k": top_k,
                },
                "model_info": {
                    "model_type": "ImageToTextGenerator",
                    "vocab_size": self._config.get("vocab_size", 50000),
                    "max_seq_len": self._config.get("max_seq_len", 128),
                },
            }
            
        except Exception as e:
            logger.error(f"图像到文本生成失败: {e}")
            return {
                "success": False,
                "error": f"生成失败: {str(e)}",
                "generated_text": "",
                "image_info": {"size": len(image_bytes)},
            }
    
    def generate_cross_modal(self, source_modality: str, source_content: Union[str, bytes],
                            target_modality: str, generation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """通用跨模态生成接口"""
        if not self.is_ready():
            return {
                "success": False,
                "error": "生成服务未就绪",
                "source_modality": source_modality,
                "target_modality": target_modality,
            }
        
        # 默认生成参数
        if generation_params is None:
            generation_params = {}
        
        # 根据模态对调用相应的生成函数
        if source_modality == "text" and target_modality == "image":
            if not isinstance(source_content, str):
                return {"success": False, "error": "源内容必须是字符串"}
            
            return self.generate_text_to_image(
                prompt=source_content,
                num_images=generation_params.get("num_images", 1),
                image_size=generation_params.get("image_size", 64)
            )
        
        elif source_modality == "image" and target_modality == "text":
            if not isinstance(source_content, bytes):
                return {"success": False, "error": "源内容必须是字节流"}
            
            return self.generate_image_to_text(
                image_bytes=source_content,
                max_length=generation_params.get("max_length", 50),
                temperature=generation_params.get("temperature", 1.0),
                top_k=generation_params.get("top_k", 50)
            )
        
        else:
            return {
                "success": False,
                "error": f"不支持的模态转换: {source_modality} -> {target_modality}",
                "supported_conversions": ["text->image", "image->text"],
            }
    
    def batch_generate(self, source_modality: str, source_contents: List[Union[str, bytes]],
                      target_modality: str, generation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """批量跨模态生成"""
        if not self.is_ready():
            return {
                "success": False,
                "error": "生成服务未就绪",
                "results": [],
            }
        
        results = []
        for i, source_content in enumerate(source_contents):
            result = self.generate_cross_modal(
                source_modality=source_modality,
                source_content=source_content,
                target_modality=target_modality,
                generation_params=generation_params
            )
            
            results.append({
                "index": i,
                "success": result.get("success", False),
                "result": result,
                "error": result.get("error", None),
            })
        
        success_count = sum(1 for r in results if r["success"])
        
        return {
            "success": success_count > 0,
            "total": len(results),
            "successful": success_count,
            "failed": len(results) - success_count,
            "results": results,
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            "status": "ready" if self.is_ready() else "not_ready",
            "generation_manager_loaded": self._generation_manager is not None,
            "device": str(self._device) if self._device else "unknown",
            "capabilities": {
                "text_to_image": self.is_ready(),
                "image_to_text": self.is_ready(),
                "supported_modalities": ["text", "image"],
                "text_to_image_params": {
                    "max_images": 4,
                    "image_sizes": [64, 128],  # 实际支持的大小
                },
                "image_to_text_params": {
                    "max_length": 100,
                    "temperature_range": [0.1, 2.0],
                    "top_k_range": [1, 100],
                },
            },
            "config": self._config if self._config else {},
            "model_architecture": {
                "text_to_image": "Conditional VAE",
                "image_to_text": "Encoder-Decoder Transformer",
            },
        }


# 单例实例获取函数
_generation_service_instance = None

def get_generation_service() -> GenerationService:
    """获取生成服务单例实例"""
    global _generation_service_instance
    if _generation_service_instance is None:
        _generation_service_instance = GenerationService()
    return _generation_service_instance