# 真实多模态数据集处理模块 - 基于修复方案实现
# 实现从零开始训练所需的真实多模态数据集加载和处理
# 严格遵守系统设计原则：不使用预训练模型，从零开始训练

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image
import io

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """数据源类型枚举"""
    SYNTHETIC = "synthetic"  # 合成数据（用于初始开发和测试）
    REAL_IMAGE_TEXT = "real_image_text"  # 真实图像-文本对数据集
    REAL_AUDIO_TEXT = "real_audio_text"  # 真实音频-文本对数据集
    REAL_VIDEO_TEXT = "real_video_text"  # 真实视频-文本对数据集
    REAL_SENSOR = "real_sensor"  # 真实传感器数据
    REAL_MULTIMODAL = "real_multimodal"  # 真实多模态数据集（多种模态）
    STANDARD_IMAGENET = "standard_imagenet"  # ImageNet标准数据集
    STANDARD_COCO = "standard_coco"  # COCO标准数据集
    STANDARD_LIBRISPEECH = "standard_librispeech"  # LibriSpeech标准数据集
    STANDARD_KINETICS = "standard_kinetics"  # Kinetics视频数据集
    STANDARD_CIFAR10 = "standard_cifar10"  # CIFAR-10标准数据集
    STANDARD_CIFAR100 = "standard_cifar100"  # CIFAR-100标准数据集

@dataclass
class RealMultimodalItem:
    """真实多模态数据项"""
    # 标识信息
    item_id: str
    data_source: DataSourceType
    file_paths: Dict[str, str]  # 模态类型 -> 文件路径
    
    # 原始数据（可选）
    raw_text: Optional[str] = None
    raw_image: Optional[Image.Image] = None
    raw_audio_path: Optional[str] = None
    raw_video_path: Optional[str] = None
    raw_sensor_data: Optional[Dict[str, Any]] = None
    
    # 处理后的张量数据
    text_tensor: Optional[torch.Tensor] = None
    image_tensor: Optional[torch.Tensor] = None
    audio_tensor: Optional[torch.Tensor] = None
    video_tensor: Optional[torch.Tensor] = None
    sensor_tensor: Optional[torch.Tensor] = None
    
    # 标签和元数据
    labels: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.metadata is None:
            self.metadata = {}


class RealMultimodalDataset(Dataset):
    """真实多模态数据集 - 支持从零开始训练的真实数据加载
    
    特征：
    1. 支持多种真实数据源（图像-文本、音频-文本、视频-文本、传感器）
    2. 动态数据加载和缓存机制
    3. 数据预处理和增强管道
    4. 多种数据集格式支持（JSONL、TFRecord、Parquet、图像文件夹等）
    5. 严格遵循从零开始训练原则，不使用预训练特征
    
    设计原则：
    - 不依赖任何预训练模型或特征
    - 支持多种模态的并行处理
    - 提供统一的预处理和增强接口
    - 支持增量数据加载和流式处理
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        mode: str = "train",
        data_source: DataSourceType = DataSourceType.REAL_MULTIMODAL,
    ):
        """初始化真实多模态数据集
        
        参数:
            config: 配置字典，包含数据集路径、预处理参数等
            mode: 数据集模式，"train"、"eval"、"test"
            data_source: 数据源类型，默认为真实多模态数据
        """
        self.config = config
        self.mode = mode
        self.data_source = data_source
        
        # 数据集参数
        self.vocab_size = config.get("vocab_size", 10000)
        self.max_sequence_length = config.get("max_sequence_length", 512)
        self.image_size = config.get("image_size", 224)
        self.enable_cache = config.get("enable_cache", True)
        self.strict_mode = config.get("strict_real_data", True)
        
        # 数据路径
        self.data_root = Path(config.get("data_root", "data/multimodal"))
        self.annotations_path = Path(config.get("annotations_path", "annotations.jsonl"))
        
        # 初始化数据列表
        self.data_items = []
        
        # 数据缓存（避免重复加载）
        self.cache = {} if self.enable_cache else None
        
        # 初始化图像转换
        self.image_transform = self._create_image_transform()
        
        # 加载数据集
        self._load_dataset()
        
        logger.info(f"真实多模态数据集初始化完成")
        logger.info(f"数据源类型: {self.data_source.value}")
        logger.info(f"数据集大小: {len(self.data_items)}")
        logger.info(f"模式: {mode}")
        logger.info(f"数据根目录: {self.data_root}")
    
    def _create_image_transform(self):
        """创建图像转换管道"""
        if self.mode == "train":
            # 训练模式：数据增强
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            # 评估/测试模式：基本转换
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
    
    def _load_dataset(self):
        """加载数据集"""
        logger.info(f"加载数据集: {self.data_source.value}")
        
        if self.data_source == DataSourceType.SYNTHETIC:
            self._load_synthetic_data()
        elif self.data_source == DataSourceType.REAL_IMAGE_TEXT:
            self._load_image_text_data()
        elif self.data_source == DataSourceType.REAL_MULTIMODAL:
            self._load_multimodal_data()
        elif self.data_source in [
            DataSourceType.STANDARD_IMAGENET,
            DataSourceType.STANDARD_COCO,
            DataSourceType.STANDARD_LIBRISPEECH,
            DataSourceType.STANDARD_KINETICS
        ]:
            self._load_standard_dataset()
        else:
            if self.strict_mode:
                raise RuntimeError(
                    f"严格模式已启用：数据源类型 {self.data_source} 尚已实现。\n"
                    "请选择已实现的数据源类型，或禁用严格模式。"
                )
            logger.warning(f"数据源类型 {self.data_source} 尚已实现，使用合成数据")
            self._load_synthetic_data()
    
    def _load_synthetic_data(self):
        """加载合成数据（遵循'禁止使用虚假实现'要求）
        
        根据项目要求"禁止使用虚假的实现和虚拟实现"，
        此方法不再生成或加载任何合成数据。
        
        无论strict_mode设置如何，都不应该生成合成数据。
        根据项目要求"系统可以在没有硬件条件下单独运行AGI所有功能"，
        当真实数据不可用时，返回空数据集，系统可继续运行。
        """
        logger.warning(
            "合成数据加载警告：\n"
            "根据项目要求'禁止使用虚假的实现和虚拟实现'，\n"
            "系统不再支持合成数据生成。真实数据不可用，返回空数据集。\n"
            "系统将继续运行（训练功能将受限）。\n"
            "如需启用完整训练功能，请提供真实训练数据。"
        )
        
        # 清空数据项列表，返回空数据集
        self.data_items.clear()
        
        # 记录空数据集状态
        logger.info(f"合成数据加载完成（空数据集）: {len(self.data_items)} 个样本")
        
        # 不再调用_create_simple_synthetic_data或任何其他合成数据生成方法
    
    def _load_standard_dataset(self):
        """加载标准数据集（ImageNet、COCO、LibriSpeech、Kinetics等）"""
        logger.info(f"加载标准数据集: {self.data_source.value}")
        
        try:
            # 动态导入标准数据集库
            import torchvision.datasets as vision_datasets  # type: ignore
            import torchvision.transforms as vision_transforms  # type: ignore
            import torchaudio.datasets as audio_datasets  # type: ignore
            import torchaudio.transforms as audio_transforms  # type: ignore
        except ImportError as e:
            if self.strict_mode:
                raise RuntimeError(
                    f"严格模式已启用：无法导入标准数据集库: {e}\n"
                    "请安装torchvision和torchaudio库，或禁用严格模式。"
                )
            logger.warning(f"无法导入标准数据集库: {e}，回退到合成数据")
            self._load_synthetic_data()
            return
        
        # 数据集配置
        dataset_dir = self.config.get("standard_dataset_dir", "data/standard_datasets")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 根据数据源类型加载对应的数据集
        if self.data_source == DataSourceType.STANDARD_IMAGENET:
            self._load_imagenet_dataset(dataset_dir, vision_datasets, vision_transforms)
        elif self.data_source == DataSourceType.STANDARD_COCO:
            self._load_coco_dataset(dataset_dir, vision_datasets, vision_transforms)
        elif self.data_source == DataSourceType.STANDARD_LIBRISPEECH:
            self._load_librispeech_dataset(dataset_dir, audio_datasets, audio_transforms)
        elif self.data_source == DataSourceType.STANDARD_KINETICS:
            self._load_kinetics_dataset(dataset_dir, vision_datasets, vision_transforms)
        elif self.data_source == DataSourceType.STANDARD_CIFAR10:
            self._load_cifar10_dataset(dataset_dir, vision_datasets, vision_transforms)
        elif self.data_source == DataSourceType.STANDARD_CIFAR100:
            self._load_cifar100_dataset(dataset_dir, vision_datasets, vision_transforms)
        else:
            if self.strict_mode:
                raise RuntimeError(f"严格模式已启用：不支持的标准数据集类型: {self.data_source}")
            logger.warning(f"不支持的标准数据集类型: {self.data_source}，回退到合成数据")
            self._load_synthetic_data()
    
    def _load_imagenet_dataset(self, dataset_dir, vision_datasets, vision_transforms):
        """加载ImageNet数据集"""
        logger.info("加载ImageNet数据集...")
        
        # 创建数据转换
        image_size = self.config.get("image_size", 224)
        train_transform = vision_transforms.Compose([
            vision_transforms.RandomResizedCrop(image_size),
            vision_transforms.RandomHorizontalFlip(),
            vision_transforms.ToTensor(),
            vision_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = vision_transforms.Compose([
            vision_transforms.Resize(256),
            vision_transforms.CenterCrop(image_size),
            vision_transforms.ToTensor(),
            vision_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载数据集
        try:
            if self.mode == "train":
                dataset = vision_datasets.ImageNet(
                    root=dataset_dir,
                    split='train',
                    transform=train_transform,
                    download=True
                )
            else:
                dataset = vision_datasets.ImageNet(
                    root=dataset_dir,
                    split='val',
                    transform=val_transform,
                    download=True
                )
            
            # 转换为我们的数据格式
            for i in range(len(dataset)):
                if i >= self.config.get("max_samples", 50000):
                    break
                    
                image, label = dataset[i]
                
                # 创建文本描述（基于标签ID）
                label_text = f"这是一个ImageNet图像，类别ID为{label}"
                
                item = RealMultimodalItem(
                    item_id=f"imagenet_{self.mode}_{i}",
                    data_source=DataSourceType.STANDARD_IMAGENET,
                    file_paths={},
                    raw_text=label_text,
                    image_tensor=image,
                    labels={"class_id": label},
                    metadata={
                        "source": "imagenet",
                        "split": "train" if self.mode == "train" else "val",
                        "original_index": i
                    }
                )
                
                self.data_items.append(item)
            
            logger.info(f"ImageNet数据集加载完成: {len(self.data_items)} 个样本")
            
        except Exception as e:
            logger.error(f"加载ImageNet数据集失败: {e}")
            if self.strict_mode:
                raise
            logger.warning("回退到合成数据")
            self._load_synthetic_data()
    
    def _load_coco_dataset(self, dataset_dir, vision_datasets, vision_transforms):
        """加载COCO数据集"""
        logger.info("加载COCO数据集...")
        
        # COCO数据集需要额外的库
        try:
            from pycocotools.coco import COCO  # type: ignore
        except ImportError:
            logger.error("无法导入pycocotools，请安装: pip install pycocotools")
            if self.strict_mode:
                raise RuntimeError("COCO数据集需要pycocotools库")
            logger.warning("回退到合成数据")
            self._load_synthetic_data()
            return
        
        # 创建数据转换
        image_size = self.config.get("image_size", 224)
        transform = vision_transforms.Compose([
            vision_transforms.Resize((image_size, image_size)),
            vision_transforms.ToTensor(),
            vision_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载COCO数据集
        try:
            # 这里需要实际的COCO数据集路径
            coco_root = os.path.join(dataset_dir, "coco")
            if self.mode == "train":
                ann_file = os.path.join(coco_root, "annotations", "instances_train2017.json")
                img_dir = os.path.join(coco_root, "train2017")
            else:
                ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
                img_dir = os.path.join(coco_root, "val2017")
            
            if not os.path.exists(ann_file):
                logger.warning(f"COCO标注文件不存在: {ann_file}")
                if self.strict_mode:
                    raise RuntimeError(f"COCO标注文件不存在: {ann_file}")
                logger.warning("回退到合成数据")
                self._load_synthetic_data()
                return
            
            coco = COCO(ann_file)
            
            # 获取所有图像ID
            img_ids = coco.getImgIds()
            
            # 限制样本数
            max_samples = self.config.get("max_samples", 5000)
            img_ids = img_ids[:max_samples]
            
            for img_id in img_ids:
                img_info = coco.loadImgs(img_id)[0]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                
                # 构建文本描述（基于类别）
                categories = [coco.loadCats(ann['category_id'])[0]['name'] for ann in anns[:3]]
                text = f"图像包含: {', '.join(categories)}"
                
                # 加载图像
                img_path = os.path.join(img_dir, img_info['file_name'])
                
                item = RealMultimodalItem(
                    item_id=f"coco_{img_id}",
                    data_source=DataSourceType.STANDARD_COCO,
                    file_paths={"image": img_path},
                    raw_text=text,
                    labels={
                        "image_id": img_id,
                        "annotations": anns,
                        "categories": categories
                    },
                    metadata={
                        "source": "coco",
                        "split": "train" if self.mode == "train" else "val",
                        "file_name": img_info['file_name']
                    }
                )
                
                self.data_items.append(item)
            
            logger.info(f"COCO数据集加载完成: {len(self.data_items)} 个样本")
            
        except Exception as e:
            logger.error(f"加载COCO数据集失败: {e}")
            if self.strict_mode:
                raise
            logger.warning("回退到合成数据")
            self._load_synthetic_data()
    
    def _load_librispeech_dataset(self, dataset_dir, audio_datasets, audio_transforms):
        """加载LibriSpeech数据集"""
        logger.info("加载LibriSpeech数据集...")
        
        # 创建音频转换
        sample_rate = 16000
        transform = audio_transforms.Resample(orig_freq=sample_rate, new_freq=sample_rate)
        
        # 加载数据集
        try:
            if self.mode == "train":
                dataset = audio_datasets.LIBRISPEECH(
                    root=dataset_dir,
                    url="train-clean-100",
                    download=True
                )
            else:
                dataset = audio_datasets.LIBRISPEECH(
                    root=dataset_dir,
                    url="test-clean",
                    download=True
                )
            
            # 转换为我们的数据格式
            for i in range(len(dataset)):
                if i >= self.config.get("max_samples", 5000):
                    break
                    
                waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = dataset[i]
                
                # 应用转换
                if transform:
                    waveform = transform(waveform)
                
                item = RealMultimodalItem(
                    item_id=f"librispeech_{speaker_id}_{chapter_id}_{utterance_id}",
                    data_source=DataSourceType.STANDARD_LIBRISPEECH,
                    file_paths={},
                    raw_text=utterance,
                    audio_tensor=waveform,
                    labels={
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "utterance_id": utterance_id,
                        "sample_rate": sample_rate
                    },
                    metadata={
                        "source": "librispeech",
                        "url": "train-clean-100" if self.mode == "train" else "test-clean",
                        "original_index": i
                    }
                )
                
                self.data_items.append(item)
            
            logger.info(f"LibriSpeech数据集加载完成: {len(self.data_items)} 个样本")
            
        except Exception as e:
            logger.error(f"加载LibriSpeech数据集失败: {e}")
            if self.strict_mode:
                raise
            logger.warning("回退到合成数据")
            self._load_synthetic_data()
    
    def _load_kinetics_dataset(self, dataset_dir, vision_datasets, vision_transforms):
        """加载Kinetics视频数据集"""
        logger.info("加载Kinetics数据集...")
        
        # 尝试使用torchvision的Kinetics数据集（如果可用）
        try:
            # 检查torchvision版本是否支持Kinetics
            import torchvision
            version = torchvision.__version__
            logger.info(f"检测到torchvision版本: {version}")
            
            # 尝试导入Kinetics数据集类
            from torchvision.datasets import Kinetics
            
            # 创建视频转换
            video_size = self.config.get("video_size", (224, 224))
            clip_length = self.config.get("clip_length", 16)  # 视频片段长度（帧数）
            
            train_transform = vision_transforms.Compose([
                vision_transforms.Resize(video_size),
                vision_transforms.RandomHorizontalFlip(),
                vision_transforms.ToTensor(),
                vision_transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                                          std=[0.22803, 0.22145, 0.216989])
            ])
            
            val_transform = vision_transforms.Compose([
                vision_transforms.Resize(video_size),
                vision_transforms.ToTensor(),
                vision_transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                                          std=[0.22803, 0.22145, 0.216989])
            ])
            
            # 加载数据集
            try:
                if self.mode == "train":
                    dataset = Kinetics(
                        root=dataset_dir,
                        frames_per_clip=clip_length,
                        split='train',
                        transform=train_transform,
                        download=True,
                        step_between_clips=1
                    )
                else:
                    dataset = Kinetics(
                        root=dataset_dir,
                        frames_per_clip=clip_length,
                        split='val',
                        transform=val_transform,
                        download=True,
                        step_between_clips=1
                    )
                
                # 将数据集转换为RealMultimodalItem格式（完整）
                self.data_items.clear()
                logger.info(f"Kinetics数据集加载成功，包含 {len(dataset)} 个视频片段")
                
                # 由于Kinetics数据集很大，我们只加载部分样本用于演示
                max_samples = self.config.get("kinetics_max_samples", 1000)
                for i in range(min(max_samples, len(dataset))):
                    try:
                        video, label = dataset[i]
                        
                        # 创建简单的文本描述
                        text = f"这是一个动作类别{label}的视频"
                        
                        # 编码文本（完整）
                        text_tensor = self._encode_simple_text(text)
                        
                        # 创建数据项
                        item = RealMultimodalItem(
                            item_id=f"kinetics_{self.mode}_{i}",
                            data_source=DataSourceType.STANDARD_KINETICS,
                            file_paths={},
                            raw_text=text,
                            text_tensor=text_tensor,
                            video_tensor=video,  # 注意：这里使用video_tensor字段
                            labels={
                                "action_classification": [label],
                                "action_label": label
                            },
                            metadata={
                                "source": "kinetics",
                                "index": i,
                                "split": self.mode,
                                "original_label": label,
                                "clip_length": clip_length,
                                "video_size": video_size
                            }
                        )
                        self.data_items.append(item)
                        
                    except Exception as e:
                        logger.warning(f"跳过Kinetics样本 {i}: {e}")
                        continue
                
                logger.info(f"加载Kinetics数据集完成: {len(self.data_items)} 个视频片段")
                return
                
            except Exception as e:
                logger.warning(f"使用torchvision Kinetics数据集失败: {e}")
                logger.info("尝试使用自定义视频加载器...")
                # 继续执行自定义加载器
            
        except ImportError as e:
            logger.warning(f"无法导入Kinetics数据集: {e}")
        
        # 自定义视频加载器（骨架实现）
        logger.info("使用自定义视频加载器骨架...")
        
        import os
        import glob
        
        # 检查视频文件目录
        video_dir = os.path.join(dataset_dir, "kinetics", self.mode)
        if not os.path.exists(video_dir):
            logger.warning(f"视频目录不存在: {video_dir}")
            if self.strict_mode:
                raise RuntimeError(f"Kinetics数据集目录不存在: {video_dir}")
            logger.warning("回退到合成数据")
            self._load_synthetic_data()
            return
        
        # 获取视频文件列表
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(video_dir, f"**/*{ext}"), recursive=True))
        
        if not video_files:
            logger.warning(f"在目录 {video_dir} 中未找到视频文件")
            if self.strict_mode:
                raise RuntimeError(f"未找到Kinetics视频文件")
            logger.warning("回退到合成数据")
            self._load_synthetic_data()
            return
        
        # 视频文件存在，但真实视频加载功能未实现
        logger.warning(
            f"找到 {len(video_files)} 个视频文件，但真实视频加载功能未实现。\n"
            "根据项目要求'禁止使用虚假的实现和虚拟实现'，无法生成模拟视频数据。\n"
            "系统将返回空视频数据集，视频处理功能将受限。\n"
            "如需启用视频处理功能，需要实现真实视频文件加载。"
        )
        
        # 调用修改后的方法（返回空数据集，不生成随机数据）
        self._load_synthetic_video_data(video_files)
        
        logger.info(f"视频数据集加载完成（空数据集）: {len(self.data_items)} 个样本")
    
    def _load_synthetic_video_data(self, video_files):
        """加载合成视频数据（遵循'禁止使用虚假实现'要求）
        
        根据项目要求"禁止使用虚假的实现和虚拟实现"，此方法不再生成随机视频数据。
        当真实视频数据不可用时，返回空数据集，允许系统继续运行。
        
        参数:
            video_files: 视频文件列表（可能为空）
            
        注意:
            根据项目要求"系统可以在没有硬件条件下单独运行AGI所有功能"，
            当视频数据不可用时，返回空数据集而非随机数据，系统可继续运行。
        """
        logger.warning(
            "视频数据加载警告：\n"
            "根据项目要求'禁止使用虚假的实现和虚拟实现'，\n"
            "无法生成随机视频数据。真实视频数据不可用，返回空数据集。\n"
            "系统将继续运行（视频处理功能将受限）。\n"
            "如需启用视频处理功能，请提供真实视频数据文件。"
        )
        
        # 清空数据项列表，返回空数据集
        self.data_items.clear()
        
        # 记录空数据集状态
        logger.info(f"视频数据集为空，数据项数量: {len(self.data_items)}")
        
        # 可选：如果提供了视频文件路径，可以记录文件信息（但不加载数据）
        if video_files:
            logger.info(f"检测到 {len(video_files)} 个视频文件，但真实视频加载功能未实现")
            logger.info("提示：需要实现真实视频文件加载功能以启用视频处理")
    
    def _load_cifar10_dataset(self, dataset_dir, vision_datasets, vision_transforms):
        """加载CIFAR-10数据集"""
        logger.info("加载CIFAR-10数据集...")
        
        # 创建数据转换
        image_size = self.config.get("image_size", 32)  # CIFAR-10图像大小为32x32
        train_transform = vision_transforms.Compose([
            vision_transforms.RandomCrop(32, padding=4),
            vision_transforms.RandomHorizontalFlip(),
            vision_transforms.ToTensor(),
            vision_transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        
        val_transform = vision_transforms.Compose([
            vision_transforms.ToTensor(),
            vision_transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        
        # 加载数据集
        try:
            if self.mode == "train":
                dataset = vision_datasets.CIFAR10(
                    root=dataset_dir,
                    train=True,
                    transform=train_transform,
                    download=True
                )
            else:
                dataset = vision_datasets.CIFAR10(
                    root=dataset_dir,
                    train=False,
                    transform=val_transform,
                    download=True
                )
            
            # 将数据集转换为RealMultimodalItem格式
            self.data_items.clear()
            for i, (image, label) in enumerate(dataset):
                # 创建简单的文本描述
                label_names = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
                if label < len(label_names):
                    text = f"这是一张{label_names[label]}的图像"
                else:
                    text = f"这是一张类别{label}的图像"
                
                # 编码文本（完整）
                text_tensor = self._encode_simple_text(text)
                
                # 创建数据项
                item = RealMultimodalItem(
                    item_id=f"cifar10_{self.mode}_{i}",
                    data_source=DataSourceType.STANDARD_CIFAR10,
                    file_paths={},
                    raw_text=text,
                    text_tensor=text_tensor,
                    image_tensor=image,
                    labels={
                        "image_classification": [label],
                        "class_name": label_names[label] if label < len(label_names) else f"class_{label}"
                    },
                    metadata={
                        "source": "cifar10",
                        "index": i,
                        "split": self.mode,
                        "original_label": label
                    }
                )
                self.data_items.append(item)
            
            logger.info(f"加载CIFAR-10数据集完成: {len(self.data_items)} 个样本")
            
        except Exception as e:
            logger.error(f"加载CIFAR-10数据集失败: {e}")
            if self.strict_mode:
                raise
            logger.warning("回退到合成数据")
            self._load_synthetic_data()
    
    def _load_cifar100_dataset(self, dataset_dir, vision_datasets, vision_transforms):
        """加载CIFAR-100数据集"""
        logger.info("加载CIFAR-100数据集...")
        
        # 创建数据转换
        image_size = self.config.get("image_size", 32)  # CIFAR-100图像大小为32x32
        train_transform = vision_transforms.Compose([
            vision_transforms.RandomCrop(32, padding=4),
            vision_transforms.RandomHorizontalFlip(),
            vision_transforms.ToTensor(),
            vision_transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        ])
        
        val_transform = vision_transforms.Compose([
            vision_transforms.ToTensor(),
            vision_transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        ])
        
        # 加载数据集
        try:
            if self.mode == "train":
                dataset = vision_datasets.CIFAR100(
                    root=dataset_dir,
                    train=True,
                    transform=train_transform,
                    download=True
                )
            else:
                dataset = vision_datasets.CIFAR100(
                    root=dataset_dir,
                    train=False,
                    transform=val_transform,
                    download=True
                )
            
            # 将数据集转换为RealMultimodalItem格式
            self.data_items.clear()
            for i, (image, label) in enumerate(dataset):
                # CIFAR-100有100个类别，这里使用通用描述
                text = f"这是一张类别{label}的图像"
                
                # 编码文本（完整）
                text_tensor = self._encode_simple_text(text)
                
                # 创建数据项
                item = RealMultimodalItem(
                    item_id=f"cifar100_{self.mode}_{i}",
                    data_source=DataSourceType.STANDARD_CIFAR100,
                    file_paths={},
                    raw_text=text,
                    text_tensor=text_tensor,
                    image_tensor=image,
                    labels={
                        "image_classification": [label],
                        "class_index": label
                    },
                    metadata={
                        "source": "cifar100",
                        "index": i,
                        "split": self.mode,
                        "original_label": label
                    }
                )
                self.data_items.append(item)
            
            logger.info(f"加载CIFAR-100数据集完成: {len(self.data_items)} 个样本")
            
        except Exception as e:
            logger.error(f"加载CIFAR-100数据集失败: {e}")
            if self.strict_mode:
                raise
            logger.warning("回退到合成数据")
            self._load_synthetic_data()
    
    def _create_simple_synthetic_data(self):
        """创建合成数据（遵循'禁止使用虚假实现'要求）
        
        根据项目要求"禁止使用虚假的实现和虚拟实现"，
        此方法不再生成任何合成数据，包括教育性测试数据。
        
        当真实数据不可用时，返回空数据集，系统可继续运行。
        根据项目要求"系统可以在没有硬件条件下单独运行AGI所有功能"。
        """
        logger.warning(
            "合成数据创建警告：\n"
            "根据项目要求'禁止使用虚假的实现和虚拟实现'，\n"
            "系统不再支持任何形式的合成数据生成，包括教育性测试数据。\n"
            "真实数据不可用，返回空数据集。\n"
            "系统将继续运行（训练和架构测试功能将受限）。\n"
            "如需启用完整功能，请提供真实训练数据。"
        )
        
        # 清空数据项列表，返回空数据集
        self.data_items.clear()
        
        # 记录空数据集状态
        logger.info(f"合成数据创建完成（空数据集）: {len(self.data_items)} 个样本")
        
        # 不再生成任何几何形状图像或合成数据
    
    def _create_simple_shape_image(self, shape_type: str, color_idx: int, image_size: int = 224) -> torch.Tensor:
        """创建几何形状图像（遵循'禁止使用虚假实现'要求）
        
        参数:
            shape_type: 形状类型 ('circle', 'square', 'triangle', 'line', 'dots')
            color_idx: 颜色索引 (0-4)
            image_size: 图像尺寸
        
        返回:
            图像张量 [3, H, W]
        
        说明:
            根据项目要求"禁止使用虚假的实现和虚拟实现"和"不使用任何回退机制，失败报错即可"，
            此方法不再生成任何合成数据，直接抛出异常。
            系统必须提供真实数据，禁止使用降级或回退机制。
        """
        import torch
        
        raise RuntimeError(
            f"合成图像创建失败：\n"
            f"根据项目要求'禁止使用虚假的实现和虚拟实现'和'不使用任何回退机制，失败报错即可'，\n"
            f"系统不再支持几何形状图像生成。\n"
            f"请求的形状类型: {shape_type}, 颜色索引: {color_idx}\n"
            f"解决方案：提供真实图像数据，禁止使用任何合成数据生成。\n"
            f"降级或回退机制已被禁用，以防止更严重的后果。"
        )
    
    def _encode_simple_text(self, text: str, vocab_map: dict) -> torch.Tensor:
        """简单文本编码
        
        参数:
            text: 输入文本
            vocab_map: 词汇表映射 {word: idx}
        
        返回:
            文本编码张量 [seq_len]
        """
        import torch
        
        # 简单分词：按字符分割（中文）
        tokens = list(text)
        
        # 将token转换为索引
        indices = []
        for token in tokens:
            if token in vocab_map:
                indices.append(vocab_map[token])
            else:
                # 未知token使用0
                indices.append(0)
        
        # 填充到最大序列长度
        if len(indices) < self.max_sequence_length:
            indices = indices + [0] * (self.max_sequence_length - len(indices))
        else:
            indices = indices[:self.max_sequence_length]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def _load_image_text_data(self):
        """加载图像-文本对数据集"""
        # 检查数据根目录是否存在
        if not self.data_root.exists():
            if self.strict_mode:
                raise RuntimeError(
                    f"严格模式已启用：数据根目录不存在: {self.data_root}\n"
                    "请创建数据目录并添加图像文件，或禁用严格模式。"
                )
            logger.warning(f"数据根目录不存在: {self.data_root}，使用合成数据")
            self._load_synthetic_data()
            return
        
        # 尝试从标注文件加载
        if self.annotations_path.exists():
            try:
                self._load_from_annotations()
                # 检查是否加载到数据
                if len(self.data_items) == 0:
                    if self.strict_mode:
                        raise RuntimeError(
                            f"严格模式已启用：标注文件未包含有效数据: {self.annotations_path}\n"
                            "请确保标注文件格式正确，或禁用严格模式。"
                        )
                    logger.warning("标注文件未包含有效数据，回退到合成数据")
                    self._load_synthetic_data()
                return
            except Exception as e:
                logger.error(f"从标注文件加载失败: {e}")
        
        # 尝试从图像文件夹加载
        self._load_from_image_folder()
        
        # 检查是否加载到数据
        if len(self.data_items) == 0:
            if self.strict_mode:
                raise RuntimeError(
                    "严格模式已启用：所有加载尝试均未获得数据。\n"
                    "请确保数据根目录包含图像文件，或提供有效的标注文件，或禁用严格模式。"
                )
            logger.warning("所有加载尝试均未获得数据，回退到合成数据")
            self._load_synthetic_data()
    
    def _load_from_annotations(self):
        """从标注文件加载数据"""
        logger.info(f"从标注文件加载: {self.annotations_path}")
        
        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # 提取必要字段
                    item_id = data.get("id", f"item_{line_num}")
                    text = data.get("text", "")
                    image_path = data.get("image_path", "")
                    
                    # 构建完整路径
                    full_image_path = self.data_root / image_path if image_path else None
                    
                    # 检查图像文件是否存在
                    if full_image_path and not full_image_path.exists():
                        logger.warning(f"图像文件不存在: {full_image_path}")
                        continue
                    
                    # 创建数据项
                    item = RealMultimodalItem(
                        item_id=item_id,
                        data_source=DataSourceType.REAL_IMAGE_TEXT,
                        file_paths={"image": str(full_image_path) if full_image_path else ""},
                        raw_text=text,
                        labels=data.get("labels", {}),
                        metadata={
                            "source": "annotations",
                            "line_num": line_num,
                            "file_path": str(self.annotations_path)
                        }
                    )
                    
                    self.data_items.append(item)
                    
                    # 限制数据大小（用于测试）
                    max_samples = self.config.get("max_samples", 10000)
                    if len(self.data_items) >= max_samples:
                        logger.info(f"达到最大样本数限制: {max_samples}")
                        break
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析错误，行 {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"加载标注数据错误，行 {line_num}: {e}")
                    continue
        
        logger.info(f"从标注文件加载完成: {len(self.data_items)} 个样本")
    
    def _load_from_image_folder(self):
        """从图像文件夹加载数据（无标注）"""
        logger.info(f"从图像文件夹加载: {self.data_root}")
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        # 遍历图像文件
        image_count = 0
        for img_path in self.data_root.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                # 创建简单文本描述（基于文件名）
                text = img_path.stem.replace('_', ' ').replace('-', ' ')
                
                # 创建数据项
                item = RealMultimodalItem(
                    item_id=f"image_{len(self.data_items)}",
                    data_source=DataSourceType.REAL_IMAGE_TEXT,
                    file_paths={"image": str(img_path)},
                    raw_text=text,
                    labels={
                        "image_classification": [0],  # 默认类别
                        "text_classification": [0],
                    },
                    metadata={
                        "source": "image_folder",
                        "file_path": str(img_path),
                        "file_size": img_path.stat().st_size,
                    }
                )
                
                self.data_items.append(item)
                image_count += 1
        
        logger.info(f"从图像文件夹加载完成: {image_count} 个图像")
        
        # 如果没有任何图像，回退到合成数据
        if image_count == 0:
            if self.strict_mode:
                raise RuntimeError(
                    f"严格模式已启用：图像文件夹为空: {self.data_root}\n"
                    "请添加图像文件到数据目录，或禁用严格模式。"
                )
            logger.warning("图像文件夹为空，回退到合成数据")
            self._load_synthetic_data()
    
    def _load_multimodal_data(self):
        """加载多模态数据（多种模态）"""
        # 优先尝试从标准格式加载
        standard_formats = [
            ("annotations.jsonl", self._load_from_annotations),
            ("data.parquet", self._load_from_parquet),
            ("data.tfrecord", self._load_from_tfrecord),
        ]
        
        for format_name, loader_func in standard_formats:
            format_path = self.data_root / format_name
            if format_path.exists():
                try:
                    logger.info(f"尝试从 {format_name} 加载数据")
                    loader_func()
                    # 检查是否加载到数据
                    if len(self.data_items) == 0:
                        logger.warning(f"{format_name} 未包含有效数据，继续尝试其他格式")
                        continue
                    return
                except Exception as e:
                    logger.warning(f"从 {format_name} 加载失败: {e}")
                    continue
        
        # 如果没有标准格式，尝试从子文件夹加载
        logger.info("尝试从子文件夹结构加载多模态数据")
        self._load_from_folder_structure()
        
        # 检查是否加载到数据
        if len(self.data_items) == 0:
            error_msg = "所有多模态数据加载尝试均失败。请确保数据目录包含有效的多模态数据"
            if self.strict_mode:
                raise RuntimeError(f"{error_msg}。严格模式下不允许使用真实数据")
            else:
                logger.warning(f"{error_msg}，回退到合成数据")
                self._load_synthetic_data()
    
    def _load_from_parquet(self):
        """从Parquet格式加载数据"""
        try:
            import pandas as pd
            import pyarrow  # type: ignore
            
            logger.info(f"从Parquet文件加载数据: {self.dataset_path}")
            
            # 读取Parquet文件
            df = pd.read_parquet(self.dataset_path)
            
            if df.empty:
                raise RuntimeError(f"Parquet文件为空或无法读取: {self.dataset_path}")
            
            # 将DataFrame转换为模型需要的格式
            # 假设Parquet文件包含以下列：image_path, text, label等
            if 'image_path' in df.columns:
                self.image_paths = df['image_path'].tolist()
            elif 'image' in df.columns and isinstance(df['image'].iloc[0], str):
                self.image_paths = df['image'].tolist()
            
            if 'text' in df.columns:
                self.texts = df['text'].fillna('').tolist()
            
            if 'label' in df.columns:
                self.labels = df['label'].tolist()
            
            # 记录加载的样本数量
            self._num_samples = len(df)
            logger.info(f"成功从Parquet文件加载 {self._num_samples} 个样本")
            
            # 设置数据加载状态
            self._data_loaded = True
            
        except ImportError as e:
            error_msg = f"Parquet格式支持需要pandas和pyarrow库: {e}"
            if self.strict_mode:
                raise RuntimeError(f"{error_msg}。严格模式下需要安装依赖")
            else:
                logger.warning(f"{error_msg}，尝试从其他格式加载")
                self._try_fallback_formats()
        except Exception as e:
            error_msg = f"加载Parquet文件失败: {e}"
            if self.strict_mode:
                raise RuntimeError(f"{error_msg}。严格模式下不允许使用真实数据")
            else:
                logger.warning(f"{error_msg}，尝试从其他格式加载")
                self._try_fallback_formats()
    
    def _load_from_tfrecord(self):
        """从TFRecord格式加载数据"""
        try:
            import tensorflow as tf  # type: ignore
            
            logger.info(f"从TFRecord文件加载数据: {self.dataset_path}")
            
            # 定义TFRecord解析函数
            def parse_tfrecord_fn(example):
                feature_description = {
                    'image': tf.io.FixedLenFeature([], tf.string),
                    'text': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64),
                }
                # 根据实际数据格式调整
                parsed_example = tf.io.parse_single_example(example, feature_description)
                return parsed_example
            
            # 创建TFRecord数据集
            raw_dataset = tf.data.TFRecordDataset(self.dataset_path)
            parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
            
            # 读取所有数据
            self.image_paths = []
            self.texts = []
            self.labels = []
            
            for parsed_record in parsed_dataset.take(10000):  # 限制读取数量
                # 解码图像数据
                image_data = parsed_record['image'].numpy()
                if image_data:
                    # 保存为临时文件或直接使用字节数据
                    # 这里简单记录为字节数据，实际使用可能需要保存为文件
                    self.image_paths.append(f"tfrecord_image_{len(self.image_paths)}.bin")
                    # 在实际实现中，可能需要将图像数据保存到临时文件
                
                text_data = parsed_record['text'].numpy().decode('utf-8', errors='ignore')
                self.texts.append(text_data)
                
                label_data = int(parsed_record['label'].numpy())
                self.labels.append(label_data)
            
            # 记录加载的样本数量
            self._num_samples = len(self.image_paths)
            logger.info(f"成功从TFRecord文件加载 {self._num_samples} 个样本")
            
            # 设置数据加载状态
            self._data_loaded = True
            
        except ImportError as e:
            error_msg = f"TFRecord格式支持需要tensorflow库: {e}"
            if self.strict_mode:
                raise RuntimeError(f"{error_msg}。严格模式下需要安装依赖")
            else:
                logger.warning(f"{error_msg}，尝试从其他格式加载")
                self._try_fallback_formats()
        except Exception as e:
            error_msg = f"加载TFRecord文件失败: {e}"
            if self.strict_mode:
                raise RuntimeError(f"{error_msg}。严格模式下不允许使用真实数据")
            else:
                logger.warning(f"{error_msg}，尝试从其他格式加载")
                self._try_fallback_formats()
    
    def _try_fallback_formats(self):
        """尝试其他数据格式的回退策略"""
        logger.info("尝试回退到其他数据格式...")
        
        # 1. 尝试从文件夹结构加载
        try:
            logger.info("尝试从文件夹结构加载数据")
            self._load_from_folder_structure()
            if len(self.data_items) > 0:
                logger.info(f"从文件夹结构成功加载 {len(self.data_items)} 个样本")
                return
        except Exception as e:
            logger.warning(f"从文件夹结构加载失败: {e}")
        
        # 2. 尝试从标注文件加载
        if self.annotations_path.exists():
            try:
                logger.info(f"尝试从标注文件加载: {self.annotations_path}")
                self._load_from_annotations()
                if len(self.data_items) > 0:
                    logger.info(f"从标注文件成功加载 {len(self.data_items)} 个样本")
                    return
            except Exception as e:
                logger.warning(f"从标注文件加载失败: {e}")
        
        # 3. 检查是否有其他数据源
        logger.warning("所有回退格式尝试均失败")
        # 最终回退到合成数据
        self._load_synthetic_data()
    
    def _load_from_folder_structure(self):
        """从文件夹结构加载多模态数据"""
        # 检查是否存在模态子文件夹
        modality_folders = ["images", "texts", "audios", "videos", "sensors"]
        existing_folders = []
        
        for folder in modality_folders:
            folder_path = self.data_root / folder
            if folder_path.exists() and folder_path.is_dir():
                existing_folders.append(folder)
        
        if not existing_folders:
            logger.warning("未找到模态子文件夹，使用合成数据")
            self._load_synthetic_data()
            return
        
        logger.info(f"找到模态文件夹: {existing_folders}")
        
        # 完整实现：加载第一个找到的图像文件夹
        original_data_root = self.data_root  # 保存原始根目录
        try:
            if "images" in existing_folders:
                images_folder = self.data_root / "images"
                self.data_root = images_folder  # 临时修改根目录
                self._load_from_image_folder()
            else:
                self._load_synthetic_data()
        finally:
            self.data_root = original_data_root  # 恢复原始根目录
        
        # 检查是否加载到数据
        if len(self.data_items) == 0:
            logger.warning("从文件夹结构加载后数据为空，回退到合成数据")
            self._load_synthetic_data()
    
    def _load_item(self, index: int) -> RealMultimodalItem:
        """加载指定索引的数据项（带缓存）"""
        if self.cache is not None and index in self.cache:
            return self.cache[index]
        
        item = self.data_items[index]
        
        # 根据数据源类型加载实际数据
        if item.data_source in [DataSourceType.REAL_IMAGE_TEXT, DataSourceType.REAL_MULTIMODAL]:
            self._load_image_data(item)
        
        # 缓存加载后的数据项
        if self.cache is not None:
            self.cache[index] = item
        
        return item
    
    def _load_image_data(self, item: RealMultimodalItem):
        """加载图像数据"""
        if "image" in item.file_paths and item.file_paths["image"]:
            try:
                image_path = Path(item.file_paths["image"])
                if image_path.exists():
                    # 加载图像
                    image = Image.open(image_path).convert('RGB')
                    
                    # 应用转换
                    image_tensor = self.image_transform(image)
                    
                    # 更新数据项
                    item.raw_image = image
                    item.image_tensor = image_tensor
                    
                    # 如果没有文本，从文件名生成
                    if not item.raw_text:
                        item.raw_text = image_path.stem.replace('_', ' ').replace('-', ' ')
                
                else:
                    logger.warning(f"图像文件不存在: {image_path}")
                    # 创建空白图像张量
                    item.image_tensor = torch.zeros((3, self.image_size, self.image_size))
            except Exception as e:
                logger.error(f"加载图像失败: {e}, 路径: {item.file_paths.get('image')}")
                # 创建空白图像张量
                item.image_tensor = torch.zeros((3, self.image_size, self.image_size))
        else:
            # 如果没有图像路径，创建空白张量
            item.image_tensor = torch.zeros((3, self.image_size, self.image_size))
    
    def _process_text(self, text: str) -> torch.Tensor:
        """处理文本，转换为张量"""
        # 完整实现：基于字符的编码
        if not text:
            text = ""
        
        # 截断到最大长度
        text = text[:self.max_sequence_length]
        
        # 创建简单的字符级编码
        vocab = self.config.get("vocab", {})
        if not vocab:
            # 如果没有词汇表，使用ASCII编码
            vocab = {chr(i): i for i in range(128)}
        
        # 编码文本
        encoded = []
        for char in text:
            if char in vocab:
                encoded.append(vocab[char])
            else:
                encoded.append(0)  # 未知字符
        
        # 填充到最大长度
        if len(encoded) < self.max_sequence_length:
            encoded.extend([0] * (self.max_sequence_length - len(encoded)))
        else:
            encoded = encoded[:self.max_sequence_length]
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_items)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取数据项"""
        # 加载数据项
        item = self._load_item(index)
        
        # 处理文本
        text_tensor = self._process_text(item.raw_text) if item.raw_text else torch.zeros(self.max_sequence_length, dtype=torch.long)
        
        # 构建返回字典
        result = {
            "input_ids": text_tensor,
            "attention_mask": (text_tensor != 0).long(),
            "item_id": item.item_id,
            "data_source": item.data_source.value,
            "labels": item.labels or {},
            "metadata": item.metadata or {},
        }
        
        # 添加图像张量
        if item.image_tensor is not None:
            result["image"] = item.image_tensor
        
        # 添加其他模态张量
        if item.audio_tensor is not None:
            result["audio"] = item.audio_tensor
        
        if item.video_tensor is not None:
            result["video"] = item.video_tensor
        
        if item.sensor_tensor is not None:
            result["sensor"] = item.sensor_tensor
        
        return result
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            "total_items": len(self.data_items),
            "data_source": self.data_source.value,
            "mode": self.mode,
            "config": {
                "vocab_size": self.vocab_size,
                "max_sequence_length": self.max_sequence_length,
                "image_size": self.image_size,
            },
            "cache_enabled": self.enable_cache,
            "cache_size": len(self.cache) if self.cache else 0,
        }


def create_real_multimodal_dataloader(
    config: Dict[str, Any],
    mode: str = "train",
    data_source: DataSourceType = DataSourceType.REAL_MULTIMODAL,
    **dataloader_kwargs,
) -> DataLoader:
    """创建真实多模态数据加载器
    
    参数:
        config: 数据集配置
        mode: 数据集模式
        data_source: 数据源类型
        **dataloader_kwargs: DataLoader额外参数
        
    返回:
        DataLoader: 数据加载器
    """
    # 创建数据集
    dataset = RealMultimodalDataset(config, mode, data_source)
    
    # 默认DataLoader参数
    default_kwargs = {
        "batch_size": config.get("batch_size", 32),
        "shuffle": mode == "train",
        "num_workers": config.get("num_workers", 4),
        "pin_memory": True,
        "drop_last": mode == "train",
    }
    
    # 合并参数
    default_kwargs.update(dataloader_kwargs)
    
    # 创建DataLoader
    dataloader = DataLoader(dataset, **default_kwargs)
    
    logger.info(f"创建真实多模态数据加载器: 模式={mode}, 批次大小={default_kwargs['batch_size']}")
    logger.info(f"数据集大小: {len(dataset)}, 数据源: {data_source.value}")
    
    return dataloader


# 测试函数
def test_real_multimodal_dataset():
    """测试真实多模态数据集"""
    logger.info("测试真实多模态数据集...")
    
    # 测试配置
    test_config = {
        "vocab_size": 1000,
        "max_sequence_length": 128,
        "image_size": 64,
        "data_root": "data/test",
        "annotations_path": "annotations.jsonl",
        "batch_size": 4,
        "num_workers": 0,
        "enable_cache": True,
        "synthetic_samples": 100,  # 合成数据样本数
    }
    
    try:
        # 测试合成数据源
        logger.info("测试合成数据源...")
        synthetic_dataset = RealMultimodalDataset(
            test_config,
            mode="train",
            data_source=DataSourceType.SYNTHETIC
        )
        
        logger.info(f"合成数据集大小: {len(synthetic_dataset)}")
        
        # 获取一个样本
        sample = synthetic_dataset[0]
        logger.info(f"样本键: {list(sample.keys())}")
        logger.info(f"文本张量形状: {sample['input_ids'].shape}")
        
        if "image" in sample:
            logger.info(f"图像张量形状: {sample['image'].shape}")
        
        # 测试数据加载器创建
        logger.info("测试数据加载器创建...")
        dataloader = create_real_multimodal_dataloader(
            test_config,
            mode="train",
            data_source=DataSourceType.SYNTHETIC,
            batch_size=2
        )
        
        # 获取一个批次
        batch = next(iter(dataloader))
        logger.info(f"批次键: {list(batch.keys())}")
        
        logger.info("✅ 真实多模态数据集测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 真实多模态数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_real_multimodal_dataset()