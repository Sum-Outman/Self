#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据下载脚本

功能：
1. 下载小型公开数据集用于初始训练
2. 准备数据格式以供RealMultimodalDataset使用
3. 确保系统有真实数据可用

注意事项：
- 使用公开许可的数据集（CC-BY, MIT, Apache 2.0）
- 确保数据质量
- 遵循数据集的许可要求
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import json
import shutil
import zipfile
import tarfile
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataDownloader:
    """真实数据下载器"""
    
    def __init__(self, data_root: str = "data/real_datasets"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # 数据集配置
        self.datasets_config = {
            "cifar10": {
                "name": "CIFAR-10",
                "description": "10类彩色图像数据集，32x32像素",
                "license": "MIT",
                "num_classes": 10,
                "image_size": 32,
                "download_function": self.download_cifar10,
            },
            "mnist": {
                "name": "MNIST",
                "description": "手写数字数据集，28x28灰度图像",
                "license": "Creative Commons Attribution-Share Alike 3.0",
                "num_classes": 10,
                "image_size": 28,
                "download_function": self.download_mnist,
            },
            "fashion_mnist": {
                "name": "Fashion-MNIST",
                "description": "时尚物品数据集，28x28灰度图像",
                "license": "MIT",
                "num_classes": 10,
                "image_size": 28,
                "download_function": self.download_fashion_mnist,
            },
            "caltech101": {
                "name": "Caltech-101",
                "description": "101类物体图像数据集",
                "license": "Free for research and educational use",
                "num_classes": 101,
                "image_size": 300,
                "download_function": self.download_caltech101,
            },
        }
    
    def list_available_datasets(self) -> List[str]:
        """列出可用的数据集"""
        return list(self.datasets_config.keys())
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> Dict[str, Any]:
        """下载数据集"""
        if dataset_name not in self.datasets_config:
            raise ValueError(f"未知的数据集: {dataset_name}。可用数据集: {self.list_available_datasets()}")
        
        config = self.datasets_config[dataset_name]
        logger.info(f"开始下载数据集: {config['name']}")
        logger.info(f"描述: {config['description']}")
        logger.info(f"许可: {config['license']}")
        
        # 调用下载函数
        result = config["download_function"](force_download)
        
        logger.info(f"数据集下载完成: {dataset_name}")
        return result
    
    def download_cifar10(self, force_download: bool = False) -> Dict[str, Any]:
        """下载CIFAR-10数据集"""
        dataset_dir = self.data_root / "cifar10"
        dataset_dir.mkdir(exist_ok=True)
        
        # 定义转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 下载训练集
        train_dataset = datasets.CIFAR10(
            root=str(dataset_dir),
            train=True,
            download=True,
            transform=transform
        )
        
        # 下载测试集
        test_dataset = datasets.CIFAR10(
            root=str(dataset_dir),
            train=False,
            download=True,
            transform=transform
        )
        
        # 创建标注文件（JSONL格式）
        annotations_path = dataset_dir / "annotations.jsonl"
        class_names = train_dataset.classes
        
        with open(annotations_path, 'w', encoding='utf-8') as f:
            # 训练集标注
            for i in range(len(train_dataset)):
                image, label = train_dataset[i]
                
                # 保存图像
                img_path = dataset_dir / "images" / f"train_{i:05d}.png"
                img_path.parent.mkdir(exist_ok=True)
                
                # 转换tensor为PIL图像并保存
                img_np = image.numpy().transpose(1, 2, 0)
                img_np = (img_np * 0.5 + 0.5) * 255  # 反归一化
                img_np = img_np.astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_pil.save(img_path)
                
                # 写入标注
                annotation = {
                    "id": f"cifar10_train_{i:05d}",
                    "image_path": f"images/train_{i:05d}.png",
                    "text": f"这是一张{class_names[label]}的图像",
                    "labels": {
                        "category": class_names[label],
                        "category_id": int(label),
                        "multilabel": [int(label)],
                        "is_real_data": True
                    },
                    "metadata": {
                        "source": "CIFAR-10",
                        "license": "MIT",
                        "dataset": "cifar10",
                        "split": "train",
                        "original_index": i,
                        "image_size": [32, 32]
                    }
                }
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
            
            # 测试集标注
            for i in range(len(test_dataset)):
                image, label = test_dataset[i]
                
                # 保存图像
                img_path = dataset_dir / "images" / f"test_{i:05d}.png"
                img_path.parent.mkdir(exist_ok=True)
                
                # 转换tensor为PIL图像并保存
                img_np = image.numpy().transpose(1, 2, 0)
                img_np = (img_np * 0.5 + 0.5) * 255  # 反归一化
                img_np = img_np.astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_pil.save(img_path)
                
                # 写入标注
                annotation = {
                    "id": f"cifar10_test_{i:05d}",
                    "image_path": f"images/test_{i:05d}.png",
                    "text": f"这是一张{class_names[label]}的图像",
                    "labels": {
                        "category": class_names[label],
                        "category_id": int(label),
                        "multilabel": [int(label)],
                        "is_real_data": True
                    },
                    "metadata": {
                        "source": "CIFAR-10",
                        "license": "MIT",
                        "dataset": "cifar10",
                        "split": "test",
                        "original_index": i,
                        "image_size": [32, 32]
                    }
                }
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
        
        # 创建数据集信息文件
        dataset_info = {
            "name": "CIFAR-10",
            "description": "10类彩色图像数据集，32x32像素",
            "license": "MIT",
            "num_classes": 10,
            "class_names": class_names,
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "total_samples": len(train_dataset) + len(test_dataset),
            "image_size": [32, 32],
            "channels": 3,
            "created_at": "2026-04-01",
            "is_real_data": True
        }
        
        with open(dataset_dir / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"CIFAR-10数据集下载完成: {len(train_dataset)}训练 + {len(test_dataset)}测试样本")
        
        return {
            "success": True,
            "dataset": "cifar10",
            "path": str(dataset_dir),
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "annotations_file": str(annotations_path),
            "info_file": str(dataset_dir / "dataset_info.json")
        }
    
    def download_mnist(self, force_download: bool = False) -> Dict[str, Any]:
        """下载MNIST数据集"""
        dataset_dir = self.data_root / "mnist"
        dataset_dir.mkdir(exist_ok=True)
        
        # 定义转换（灰度转RGB以便与图像编码器兼容）
        transform = transforms.Compose([
            transforms.Grayscale(3),  # 转换为3通道
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 下载训练集
        train_dataset = datasets.MNIST(
            root=str(dataset_dir),
            train=True,
            download=True,
            transform=transform
        )
        
        # 下载测试集
        test_dataset = datasets.MNIST(
            root=str(dataset_dir),
            train=False,
            download=True,
            transform=transform
        )
        
        # 创建标注文件（JSONL格式）
        annotations_path = dataset_dir / "annotations.jsonl"
        
        with open(annotations_path, 'w', encoding='utf-8') as f:
            # 训练集标注
            for i in range(min(len(train_dataset), 10000)):  # 限制样本数量
                image, label = train_dataset[i]
                
                # 保存图像
                img_path = dataset_dir / "images" / f"train_{i:05d}.png"
                img_path.parent.mkdir(exist_ok=True)
                
                # 转换tensor为PIL图像并保存
                img_np = image.numpy().transpose(1, 2, 0)
                img_np = (img_np * 0.5 + 0.5) * 255  # 反归一化
                img_np = img_np.astype(np.uint8)[:, :, 0]  # 取第一个通道
                img_pil = Image.fromarray(img_np, mode='L')
                img_pil.save(img_path)
                
                # 写入标注
                annotation = {
                    "id": f"mnist_train_{i:05d}",
                    "image_path": f"images/train_{i:05d}.png",
                    "text": f"这是一个手写数字{label}的图像",
                    "labels": {
                        "category": str(label),
                        "category_id": int(label),
                        "multilabel": [int(label)],
                        "is_real_data": True
                    },
                    "metadata": {
                        "source": "MNIST",
                        "license": "Creative Commons Attribution-Share Alike 3.0",
                        "dataset": "mnist",
                        "split": "train",
                        "original_index": i,
                        "image_size": [28, 28]
                    }
                }
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
            
            # 测试集标注
            for i in range(min(len(test_dataset), 2000)):  # 限制样本数量
                image, label = test_dataset[i]
                
                # 保存图像
                img_path = dataset_dir / "images" / f"test_{i:05d}.png"
                img_path.parent.mkdir(exist_ok=True)
                
                # 转换tensor为PIL图像并保存
                img_np = image.numpy().transpose(1, 2, 0)
                img_np = (img_np * 0.5 + 0.5) * 255  # 反归一化
                img_np = img_np.astype(np.uint8)[:, :, 0]  # 取第一个通道
                img_pil = Image.fromarray(img_np, mode='L')
                img_pil.save(img_path)
                
                # 写入标注
                annotation = {
                    "id": f"mnist_test_{i:05d}",
                    "image_path": f"images/test_{i:05d}.png",
                    "text": f"这是一个手写数字{label}的图像",
                    "labels": {
                        "category": str(label),
                        "category_id": int(label),
                        "multilabel": [int(label)],
                        "is_real_data": True
                    },
                    "metadata": {
                        "source": "MNIST",
                        "license": "Creative Commons Attribution-Share Alike 3.0",
                        "dataset": "mnist",
                        "split": "test",
                        "original_index": i,
                        "image_size": [28, 28]
                    }
                }
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
        
        # 创建数据集信息文件
        dataset_info = {
            "name": "MNIST",
            "description": "手写数字数据集，28x28灰度图像",
            "license": "Creative Commons Attribution-Share Alike 3.0",
            "num_classes": 10,
            "class_names": [str(i) for i in range(10)],
            "train_samples": min(len(train_dataset), 10000),
            "test_samples": min(len(test_dataset), 2000),
            "total_samples": min(len(train_dataset), 10000) + min(len(test_dataset), 2000),
            "image_size": [28, 28],
            "channels": 1,
            "created_at": "2026-04-01",
            "is_real_data": True
        }
        
        with open(dataset_dir / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"MNIST数据集下载完成: {min(len(train_dataset), 10000)}训练 + {min(len(test_dataset), 2000)}测试样本")
        
        return {
            "success": True,
            "dataset": "mnist",
            "path": str(dataset_dir),
            "train_samples": min(len(train_dataset), 10000),
            "test_samples": min(len(test_dataset), 2000),
            "annotations_file": str(annotations_path),
            "info_file": str(dataset_dir / "dataset_info.json")
        }
    
    def download_fashion_mnist(self, force_download: bool = False) -> Dict[str, Any]:
        """下载Fashion-MNIST数据集"""
        dataset_dir = self.data_root / "fashion_mnist"
        dataset_dir.mkdir(exist_ok=True)
        
        # 定义转换（灰度转RGB以便与图像编码器兼容）
        transform = transforms.Compose([
            transforms.Grayscale(3),  # 转换为3通道
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 下载训练集
        train_dataset = datasets.FashionMNIST(
            root=str(dataset_dir),
            train=True,
            download=True,
            transform=transform
        )
        
        # 下载测试集
        test_dataset = datasets.FashionMNIST(
            root=str(dataset_dir),
            train=False,
            download=True,
            transform=transform
        )
        
        # 类别名称
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        # 创建标注文件（JSONL格式）
        annotations_path = dataset_dir / "annotations.jsonl"
        
        with open(annotations_path, 'w', encoding='utf-8') as f:
            # 训练集标注
            for i in range(min(len(train_dataset), 10000)):  # 限制样本数量
                image, label = train_dataset[i]
                
                # 保存图像
                img_path = dataset_dir / "images" / f"train_{i:05d}.png"
                img_path.parent.mkdir(exist_ok=True)
                
                # 转换tensor为PIL图像并保存
                img_np = image.numpy().transpose(1, 2, 0)
                img_np = (img_np * 0.5 + 0.5) * 255  # 反归一化
                img_np = img_np.astype(np.uint8)[:, :, 0]  # 取第一个通道
                img_pil = Image.fromarray(img_np, mode='L')
                img_pil.save(img_path)
                
                # 写入标注
                annotation = {
                    "id": f"fashion_mnist_train_{i:05d}",
                    "image_path": f"images/train_{i:05d}.png",
                    "text": f"这是一张{class_names[label]}的图像",
                    "labels": {
                        "category": class_names[label],
                        "category_id": int(label),
                        "multilabel": [int(label)],
                        "is_real_data": True
                    },
                    "metadata": {
                        "source": "Fashion-MNIST",
                        "license": "MIT",
                        "dataset": "fashion_mnist",
                        "split": "train",
                        "original_index": i,
                        "image_size": [28, 28]
                    }
                }
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
            
            # 测试集标注
            for i in range(min(len(test_dataset), 2000)):  # 限制样本数量
                image, label = test_dataset[i]
                
                # 保存图像
                img_path = dataset_dir / "images" / f"test_{i:05d}.png"
                img_path.parent.mkdir(exist_ok=True)
                
                # 转换tensor为PIL图像并保存
                img_np = image.numpy().transpose(1, 2, 0)
                img_np = (img_np * 0.5 + 0.5) * 255  # 反归一化
                img_np = img_np.astype(np.uint8)[:, :, 0]  # 取第一个通道
                img_pil = Image.fromarray(img_np, mode='L')
                img_pil.save(img_path)
                
                # 写入标注
                annotation = {
                    "id": f"fashion_mnist_test_{i:05d}",
                    "image_path": f"images/test_{i:05d}.png",
                    "text": f"这是一张{class_names[label]}的图像",
                    "labels": {
                        "category": class_names[label],
                        "category_id": int(label),
                        "multilabel": [int(label)],
                        "is_real_data": True
                    },
                    "metadata": {
                        "source": "Fashion-MNIST",
                        "license": "MIT",
                        "dataset": "fashion_mnist",
                        "split": "test",
                        "original_index": i,
                        "image_size": [28, 28]
                    }
                }
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
        
        # 创建数据集信息文件
        dataset_info = {
            "name": "Fashion-MNIST",
            "description": "时尚物品数据集，28x28灰度图像",
            "license": "MIT",
            "num_classes": 10,
            "class_names": class_names,
            "train_samples": min(len(train_dataset), 10000),
            "test_samples": min(len(test_dataset), 2000),
            "total_samples": min(len(train_dataset), 10000) + min(len(test_dataset), 2000),
            "image_size": [28, 28],
            "channels": 1,
            "created_at": "2026-04-01",
            "is_real_data": True
        }
        
        with open(dataset_dir / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Fashion-MNIST数据集下载完成: {min(len(train_dataset), 10000)}训练 + {min(len(test_dataset), 2000)}测试样本")
        
        return {
            "success": True,
            "dataset": "fashion_mnist",
            "path": str(dataset_dir),
            "train_samples": min(len(train_dataset), 10000),
            "test_samples": min(len(test_dataset), 2000),
            "annotations_file": str(annotations_path),
            "info_file": str(dataset_dir / "dataset_info.json")
        }
    
    def download_caltech101(self, force_download: bool = False) -> Dict[str, Any]:
        """下载Caltech-101数据集"""
        dataset_dir = self.data_root / "caltech101"
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            # 下载数据集
            dataset = datasets.Caltech101(
                root=str(dataset_dir),
                download=True,
            )
        except Exception as e:
            logger.error(f"下载Caltech-101失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Caltech-101下载失败，可能需要手动下载"
            }
        
        # 创建标注文件（JSONL格式）
        annotations_path = dataset_dir / "annotations.jsonl"
        
        with open(annotations_path, 'w', encoding='utf-8') as f:
            for i in range(min(len(dataset), 5000)):  # 限制样本数量
                try:
                    image, label = dataset[i]
                    
                    # 保存图像
                    img_path = dataset_dir / "images" / f"image_{i:05d}.png"
                    img_path.parent.mkdir(exist_ok=True)
                    
                    # 保存图像
                    image.save(img_path)
                    
                    # 获取类别名称
                    category = dataset.categories[label] if hasattr(dataset, 'categories') else f"class_{label}"
                    
                    # 写入标注
                    annotation = {
                        "id": f"caltech101_{i:05d}",
                        "image_path": f"images/image_{i:05d}.png",
                        "text": f"这是一张{category}的图像",
                        "labels": {
                            "category": category,
                            "category_id": int(label),
                            "multilabel": [int(label)],
                            "is_real_data": True
                        },
                        "metadata": {
                            "source": "Caltech-101",
                            "license": "Free for research and educational use",
                            "dataset": "caltech101",
                            "original_index": i,
                            "image_size": list(image.size)
                        }
                    }
                    f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
                except Exception as e:
                    logger.warning(f"处理Caltech-101样本{i}失败: {e}")
                    continue
        
        # 创建数据集信息文件
        dataset_info = {
            "name": "Caltech-101",
            "description": "101类物体图像数据集",
            "license": "Free for research and educational use",
            "num_classes": 101,
            "samples_processed": min(len(dataset), 5000),
            "total_samples": len(dataset),
            "created_at": "2026-04-01",
            "is_real_data": True
        }
        
        with open(dataset_dir / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Caltech-101数据集处理完成: {min(len(dataset), 5000)}样本")
        
        return {
            "success": True,
            "dataset": "caltech101",
            "path": str(dataset_dir),
            "samples": min(len(dataset), 5000),
            "annotations_file": str(annotations_path),
            "info_file": str(dataset_dir / "dataset_info.json")
        }
    
    def prepare_for_self_agi(self, dataset_name: str) -> Dict[str, Any]:
        """为Self AGI系统准备数据集"""
        result = self.download_dataset(dataset_name)
        
        if not result.get("success", False):
            return result
        
        # 创建与RealMultimodalDataset兼容的链接
        multimodal_dir = project_root / "data" / "multimodal"
        multimodal_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建符号链接或复制数据
        dataset_path = Path(result["path"])
        
        # 创建链接到multimodal目录
        link_path = multimodal_dir / dataset_name
        if link_path.exists():
            logger.info(f"链接已存在: {link_path}")
        else:
            try:
                # 尝试创建符号链接
                link_path.symlink_to(dataset_path, target_is_directory=True)
                logger.info(f"创建符号链接: {link_path} -> {dataset_path}")
            except Exception as e:
                logger.warning(f"无法创建符号链接: {e}，将复制数据")
                # 复制数据
                import shutil
                shutil.copytree(dataset_path, link_path)
                logger.info(f"复制数据: {dataset_path} -> {link_path}")
        
        # 更新结果
        result["multimodal_path"] = str(link_path)
        
        # 创建配置示例
        config_example = {
            "data_source": "real_image_text",
            "data_root": str(link_path),
            "annotations_path": str(link_path / "annotations.jsonl"),
            "image_size": 224,
            "vocab_size": 10000,
            "max_sequence_length": 512,
            "enable_cache": True,
            "strict_real_data": True
        }
        
        config_path = link_path / "config_example.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_example, f, ensure_ascii=False, indent=2)
        
        result["config_example"] = str(config_path)
        
        logger.info(f"数据集已准备就绪，可用于Self AGI训练")
        logger.info(f"配置示例: {config_path}")
        
        return result


def main():
    parser = argparse.ArgumentParser(description="下载真实数据集供Self AGI系统使用")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="要下载的数据集名称（可用: cifar10, mnist, fashion_mnist, caltech101）"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出可用的数据集"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有可用的数据集"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="为Self AGI系统准备数据集（创建链接和配置）"
    )
    
    args = parser.parse_args()
    
    downloader = RealDataDownloader()
    
    if args.list:
        print("可用的数据集:")
        for dataset_name in downloader.list_available_datasets():
            config = downloader.datasets_config[dataset_name]
            print(f"  {dataset_name}: {config['name']} - {config['description']}")
        return
    
    if args.all:
        results = []
        for dataset_name in downloader.list_available_datasets():
            print(f"\n下载数据集: {dataset_name}")
            try:
                if args.prepare:
                    result = downloader.prepare_for_self_agi(dataset_name)
                else:
                    result = downloader.download_dataset(dataset_name)
                results.append((dataset_name, result))
            except Exception as e:
                print(f"下载{dataset_name}失败: {e}")
                results.append((dataset_name, {"success": False, "error": str(e)}))
        
        print("\n下载完成摘要:")
        for dataset_name, result in results:
            status = "成功" if result.get("success", False) else "失败"
            print(f"  {dataset_name}: {status}")
            if result.get("success", False):
                print(f"    路径: {result.get('path', 'N/A')}")
                print(f"    样本数: {result.get('train_samples', result.get('samples', 'N/A'))}")
    else:
        try:
            if args.prepare:
                result = downloader.prepare_for_self_agi(args.dataset)
            else:
                result = downloader.download_dataset(args.dataset)
            
            if result.get("success", False):
                print(f"\n数据集下载成功: {args.dataset}")
                print(f"路径: {result.get('path')}")
                print(f"标注文件: {result.get('annotations_file')}")
                print(f"信息文件: {result.get('info_file')}")
                if 'multimodal_path' in result:
                    print(f"MultiModal路径: {result.get('multimodal_path')}")
            else:
                print(f"\n数据集下载失败: {args.dataset}")
                print(f"错误: {result.get('error', '未知错误')}")
                
        except Exception as e:
            print(f"下载过程中发生错误: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()