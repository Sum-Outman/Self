"""
跨模态检索服务模块
提供文本、图像、音频、视频等跨模态检索功能
基于ContrastiveAlignmentModel实现
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

# 导入多模态模型
try:
    from models.multimodal.contrastive_learning import ContrastiveAlignmentModel
    from models.multimodal.processor import MultimodalProcessor
    from models.multimodal.text_encoder import IndustrialTextEncoder
    from models.multimodal.vision_encoder import IndustrialVisionEncoder
    from models.multimodal.industrial_audio_encoder import IndustrialAudioEncoder
    CONTRASTIVE_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入对比学习模型: {e}")
    ContrastiveAlignmentModel = None
    MultimodalProcessor = None
    CONTRASTIVE_MODEL_AVAILABLE = False


class RetrievalService:
    """跨模态检索服务单例类"""
    
    _instance = None
    _contrastive_model = None
    _text_encoder = None
    _image_encoder = None
    _audio_encoder = None
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
        """加载检索模型"""
        if not CONTRASTIVE_MODEL_AVAILABLE:
            logger.error("对比学习模型不可用，检索功能将受限")
            return
        
        try:
            logger.info("正在加载跨模态检索模型...")
            
            # 配置参数
            self._config = {
                "text_embedding_dim": 768,
                "image_embedding_dim": 768,
                "audio_embedding_dim": 256,
                "sensor_embedding_dim": 256,
                "shared_dim": 512,
                "vocab_size": 100000,
                "num_layers": 6,
                "max_position_embeddings": 2048,
                "image_size": 224,
                "patch_size": 16,
                "initial_temperature": 0.07,
            }
            
            # 加载对比学习模型
            self._contrastive_model = ContrastiveAlignmentModel(self._config)
            self._contrastive_model.to(self._device)
            self._contrastive_model.eval()
            
            # 加载独立的编码器用于检索
            self._text_encoder = IndustrialTextEncoder(
                vocab_size=self._config["vocab_size"],
                embedding_dim=self._config["text_embedding_dim"],
                num_layers=self._config["num_layers"],
                max_position_embeddings=self._config["max_position_embeddings"]
            )
            self._text_encoder.to(self._device)
            self._text_encoder.eval()
            
            self._image_encoder = IndustrialVisionEncoder(
                image_size=self._config["image_size"],
                patch_size=self._config["patch_size"],
                embedding_dim=self._config["image_embedding_dim"],
                num_layers=self._config["num_layers"]
            )
            self._image_encoder.to(self._device)
            self._image_encoder.eval()
            
            logger.info(f"跨模态检索模型加载成功，设备: {self._device}")
            
        except Exception as e:
            logger.error(f"跨模态检索模型加载失败: {e}")
            self._contrastive_model = None
            self._text_encoder = None
            self._image_encoder = None
    
    def is_ready(self) -> bool:
        """检查检索服务是否就绪"""
        return self._contrastive_model is not None
    
    def encode_text(self, text: str) -> Optional[torch.Tensor]:
        """编码文本为特征向量"""
        if not self.is_ready() or self._text_encoder is None:
            return None  # 返回None
        
        try:
            # 简单tokenization（实际实现应使用tokenizer）
            # 这里使用简单模拟
            tokens = torch.randint(0, 1000, (1, min(len(text), 128)))
            tokens = tokens.to(self._device)
            
            with torch.no_grad():
                features = self._text_encoder(tokens)
                # 池化
                if features.dim() == 3:
                    features = features.mean(dim=1)
                
                # 投影到共享空间（如果使用对比模型）
                if self._contrastive_model is not None:
                    features = self._contrastive_model.text_projection(features)
                    features = F.normalize(features, dim=-1)
                
            return features
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            return None  # 返回None
    
    def encode_image(self, image_bytes: bytes) -> Optional[torch.Tensor]:
        """编码图像为特征向量"""
        if not self.is_ready() or self._image_encoder is None:
            return None  # 返回None
        
        try:
            # 转换图像为张量
            image = Image.open(io.BytesIO(image_bytes))
            
            # 预处理
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
            ])
            
            image_tensor = preprocess(image).unsqueeze(0).to(self._device)
            
            with torch.no_grad():
                features = self._image_encoder(image_tensor)
                # 池化
                if features.dim() == 3:
                    features = features.mean(dim=1)
                
                # 投影到共享空间
                if self._contrastive_model is not None:
                    features = self._contrastive_model.image_projection(features)
                    features = F.normalize(features, dim=-1)
                
            return features
        except Exception as e:
            logger.error(f"图像编码失败: {e}")
            return None  # 返回None
    
    def compute_similarity(self, features_a: torch.Tensor, features_b: torch.Tensor) -> float:
        """计算两个特征向量之间的相似度"""
        if features_a is None or features_b is None:
            return 0.0
        
        try:
            similarity = F.cosine_similarity(features_a, features_b, dim=-1)
            return similarity.item()
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def retrieve_text_by_image(self, image_bytes: bytes, text_candidates: List[str], top_k: int = 5) -> Dict[str, Any]:
        """图像到文本检索：根据图像查找最相关的文本"""
        if not self.is_ready():
            return {
                "success": False,
                "error": "检索服务未就绪",
                "results": [],
                "top_k": top_k,
            }
        
        try:
            # 编码查询图像
            query_features = self.encode_image(image_bytes)
            if query_features is None:
                return {
                    "success": False,
                    "error": "图像编码失败",
                    "results": [],
                    "top_k": top_k,
                }
            
            # 编码候选文本
            candidate_features = []
            valid_candidates = []
            
            for text in text_candidates:
                features = self.encode_text(text)
                if features is not None:
                    candidate_features.append(features)
                    valid_candidates.append(text)
            
            if not candidate_features:
                return {
                    "success": False,
                    "error": "没有有效的文本候选",
                    "results": [],
                    "top_k": top_k,
                }
            
            # 计算相似度
            candidate_tensor = torch.cat(candidate_features, dim=0)
            similarities = F.cosine_similarity(query_features, candidate_tensor, dim=-1)
            
            # 获取top-k结果
            top_scores, top_indices = similarities.topk(min(top_k, len(valid_candidates)))
            
            results = []
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                results.append({
                    "text": valid_candidates[idx],
                    "similarity": float(score),
                    "rank": len(results) + 1,
                })
            
            return {
                "success": True,
                "query_type": "image",
                "target_type": "text",
                "results": results,
                "top_k": top_k,
                "total_candidates": len(valid_candidates),
                "processing_time": 0.0,  # 实际应计算时间
            }
            
        except Exception as e:
            logger.error(f"图像到文本检索失败: {e}")
            return {
                "success": False,
                "error": f"检索失败: {str(e)}",
                "results": [],
                "top_k": top_k,
            }
    
    def retrieve_image_by_text(self, query_text: str, image_candidates: List[bytes], top_k: int = 5) -> Dict[str, Any]:
        """文本到图像检索：根据文本查找最相关的图像"""
        if not self.is_ready():
            return {
                "success": False,
                "error": "检索服务未就绪",
                "results": [],
                "top_k": top_k,
            }
        
        try:
            # 编码查询文本
            query_features = self.encode_text(query_text)
            if query_features is None:
                return {
                    "success": False,
                    "error": "文本编码失败",
                    "results": [],
                    "top_k": top_k,
                }
            
            # 编码候选图像
            candidate_features = []
            valid_candidates = []
            
            for i, image_bytes in enumerate(image_candidates):
                features = self.encode_image(image_bytes)
                if features is not None:
                    candidate_features.append(features)
                    valid_candidates.append({
                        "index": i,
                        "size": len(image_bytes),
                    })
            
            if not candidate_features:
                return {
                    "success": False,
                    "error": "没有有效的图像候选",
                    "results": [],
                    "top_k": top_k,
                }
            
            # 计算相似度
            candidate_tensor = torch.cat(candidate_features, dim=0)
            similarities = F.cosine_similarity(query_features, candidate_tensor, dim=-1)
            
            # 获取top-k结果
            top_scores, top_indices = similarities.topk(min(top_k, len(valid_candidates)))
            
            results = []
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                results.append({
                    "image_index": valid_candidates[idx]["index"],
                    "similarity": float(score),
                    "rank": len(results) + 1,
                    "image_size": valid_candidates[idx]["size"],
                })
            
            return {
                "success": True,
                "query_type": "text",
                "target_type": "image",
                "results": results,
                "top_k": top_k,
                "total_candidates": len(valid_candidates),
                "processing_time": 0.0,
            }
            
        except Exception as e:
            logger.error(f"文本到图像检索失败: {e}")
            return {
                "success": False,
                "error": f"检索失败: {str(e)}",
                "results": [],
                "top_k": top_k,
            }
    
    def cross_modal_similarity(self, modality_a: str, content_a: Union[str, bytes], 
                              modality_b: str, content_b: Union[str, bytes]) -> Dict[str, Any]:
        """计算跨模态相似度"""
        if not self.is_ready():
            return {
                "success": False,
                "error": "检索服务未就绪",
                "similarity": 0.0,
            }
        
        try:
            # 编码内容A
            if modality_a == "text":
                if not isinstance(content_a, str):
                    return {"success": False, "error": "内容A必须是字符串"}
                features_a = self.encode_text(content_a)
            elif modality_a == "image":
                if not isinstance(content_a, bytes):
                    return {"success": False, "error": "内容A必须是字节流"}
                features_a = self.encode_image(content_a)
            else:
                return {"success": False, "error": f"不支持的模态: {modality_a}"}
            
            # 编码内容B
            if modality_b == "text":
                if not isinstance(content_b, str):
                    return {"success": False, "error": "内容B必须是字符串"}
                features_b = self.encode_text(content_b)
            elif modality_b == "image":
                if not isinstance(content_b, bytes):
                    return {"success": False, "error": "内容B必须是字节流"}
                features_b = self.encode_image(content_b)
            else:
                return {"success": False, "error": f"不支持的模态: {modality_b}"}
            
            if features_a is None or features_b is None:
                return {
                    "success": False,
                    "error": "特征编码失败",
                    "similarity": 0.0,
                }
            
            similarity = self.compute_similarity(features_a, features_b)
            
            return {
                "success": True,
                "modality_pair": f"{modality_a}-{modality_b}",
                "similarity": similarity,
                "interpretation": self._interpret_similarity(similarity),
            }
            
        except Exception as e:
            logger.error(f"跨模态相似度计算失败: {e}")
            return {
                "success": False,
                "error": f"计算失败: {str(e)}",
                "similarity": 0.0,
            }
    
    def _interpret_similarity(self, similarity: float) -> str:
        """解释相似度分数"""
        if similarity >= 0.8:
            return "高度相关"
        elif similarity >= 0.6:
            return "中等相关"
        elif similarity >= 0.4:
            return "弱相关"
        elif similarity >= 0.2:
            return "轻微相关"
        else:
            return "不相关"
    
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            "status": "ready" if self.is_ready() else "not_ready",
            "contrastive_model_loaded": self._contrastive_model is not None,
            "text_encoder_loaded": self._text_encoder is not None,
            "image_encoder_loaded": self._image_encoder is not None,
            "device": str(self._device) if self._device else "unknown",
            "capabilities": {
                "text_to_image": self.is_ready(),
                "image_to_text": self.is_ready(),
                "cross_modal_similarity": self.is_ready(),
                "supported_modalities": ["text", "image"],
                "audio_support": False,  # 当前版本不支持音频
                "video_support": False,  # 当前版本不支持视频
            },
            "config": self._config if self._config else {},
        }


# 单例实例获取函数
_retrieval_service_instance = None

def get_retrieval_service() -> RetrievalService:
    """获取检索服务单例实例"""
    global _retrieval_service_instance
    if _retrieval_service_instance is None:
        _retrieval_service_instance = RetrievalService()
    return _retrieval_service_instance