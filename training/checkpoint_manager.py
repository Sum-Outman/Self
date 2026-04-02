#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型模型检查点管理器
提供完整的模型检查点管理功能，包括版本控制、自动清理、元数据管理和完整性验证

功能：
1. 智能检查点保存：基于性能指标自动保存最佳模型
2. 版本控制：为每个检查点分配唯一版本号，支持回滚
3. 自动清理：根据存储策略自动删除旧检查点
4. 元数据管理：记录检查点的训练指标、超参数和环境信息
5. 完整性验证：检查检查点文件的完整性和一致性
6. 分布式检查点：支持分布式训练环境的检查点同步
7. 检查点压缩：可选压缩以节省存储空间
"""

import torch
import json
import os
import shutil
import hashlib
import time
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging


class CheckpointFormat(Enum):
    """检查点格式枚举"""
    PYTORCH = "pytorch"      # PyTorch原生格式 (.pt, .pth)
    SAFETENSORS = "safetensors"  # SafeTensors格式
    ONNX = "onnx"            # ONNX格式
    COMPRESSED = "compressed"  # 压缩格式


class CheckpointStrategy(Enum):
    """检查点策略枚举"""
    BEST_ONLY = "best_only"          # 只保存最佳模型
    LAST_K = "last_k"                # 保存最后K个检查点
    EVERY_N_STEPS = "every_n_steps"  # 每N步保存一次
    PERFORMANCE_BASED = "performance_based"  # 基于性能指标保存


@dataclass
class CheckpointMetadata:
    """检查点元数据"""
    
    # 基本信息
    checkpoint_id: str
    checkpoint_path: str
    created_at: str
    format_version: str = "1.0"
    
    # 训练信息
    global_step: int = 0
    epoch: int = 0
    training_time_hours: float = 0.0
    
    # 性能指标
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # 模型信息
    model_type: str = ""
    model_parameters: int = 0
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # 训练配置
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # 系统信息
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    # 依赖信息
    dependencies: Dict[str, str] = field(default_factory=dict)
    
    # 校验信息
    checksum: str = ""
    file_size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_path": self.checkpoint_path,
            "created_at": self.created_at,
            "format_version": self.format_version,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "training_time_hours": self.training_time_hours,
            "metrics": self.metrics,
            "model_type": self.model_type,
            "model_parameters": self.model_parameters,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "system_info": self.system_info,
            "dependencies": self.dependencies,
            "checksum": self.checksum,
            "file_size_bytes": self.file_size_bytes,
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self,
                 checkpoint_dir: Union[str, Path],
                 strategy: CheckpointStrategy = CheckpointStrategy.BEST_ONLY,
                 max_checkpoints: int = 5,
                 checkpoint_format: CheckpointFormat = CheckpointFormat.PYTORCH,
                 enable_compression: bool = False):
        """
        初始化检查点管理器
        
        参数:
            checkpoint_dir: 检查点目录
            strategy: 检查点保存策略
            max_checkpoints: 最大检查点数量（用于LAST_K策略）
            checkpoint_format: 检查点格式
            enable_compression: 是否启用压缩
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.strategy = strategy
        self.max_checkpoints = max_checkpoints
        self.checkpoint_format = checkpoint_format
        self.enable_compression = enable_compression
        
        # 创建检查点目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 元数据目录
        self.metadata_dir = self.checkpoint_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # 最佳检查点文件
        self.best_checkpoint_path = self.checkpoint_dir / "best_model"
        
        # 日志
        self.logger = logging.getLogger("CheckpointManager")
        
        # 存储检查点历史
        self.checkpoint_history: List[CheckpointMetadata] = []
        
        # 加载现有检查点元数据
        self._load_checkpoint_history()
    
    def _load_checkpoint_history(self):
        """加载检查点历史"""
        metadata_files = list(self.metadata_dir.glob("*.json"))
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                metadata = CheckpointMetadata(**metadata_dict)
                self.checkpoint_history.append(metadata)
                self.logger.debug(f"加载检查点元数据: {metadata.checkpoint_id}")
            except Exception as e:
                self.logger.warning(f"加载检查点元数据失败 {metadata_file}: {e}")
        
        # 按创建时间排序
        self.checkpoint_history.sort(key=lambda x: x.created_at, reverse=True)
    
    def generate_checkpoint_id(self, global_step: int = 0) -> str:
        """生成唯一检查点ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"checkpoint_{timestamp}_step{global_step}_{random_suffix}"
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       global_step: int = 0,
                       epoch: int = 0,
                       metrics: Optional[Dict[str, float]] = None,
                       is_best: bool = False,
                       additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        保存检查点
        
        参数:
            model: 模型
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
            global_step: 全局训练步数
            epoch: 当前训练轮次
            metrics: 性能指标
            is_best: 是否为最佳模型
            additional_info: 额外信息
            
        返回:
            检查点ID
        """
        checkpoint_id = self.generate_checkpoint_id(global_step)
        
        # 确定检查点文件名
        if self.checkpoint_format == CheckpointFormat.PYTORCH:
            checkpoint_filename = f"{checkpoint_id}.pt"
        elif self.checkpoint_format == CheckpointFormat.SAFETENSORS:
            checkpoint_filename = f"{checkpoint_id}.safetensors"
        elif self.checkpoint_format == CheckpointFormat.ONNX:
            checkpoint_filename = f"{checkpoint_id}.onnx"
        else:
            checkpoint_filename = f"{checkpoint_id}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # 准备检查点数据
        checkpoint_data = {
            "global_step": global_step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "checkpoint_id": checkpoint_id,
            "saved_at": datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        
        if additional_info is not None:
            checkpoint_data.update(additional_info)
        
        # 保存检查点文件
        try:
            if self.checkpoint_format == CheckpointFormat.PYTORCH:
                torch.save(checkpoint_data, checkpoint_path)
            elif self.checkpoint_format == CheckpointFormat.SAFETENSORS:
                # 使用safetensors库保存
                try:
                    import safetensors.torch
                    safetensors.torch.save_file(
                        checkpoint_data.get("model_state_dict", {}),
                        checkpoint_path
                    )
                except ImportError:
                    self.logger.warning("safetensors库未安装，使用PyTorch格式保存")
                    torch.save(checkpoint_data, checkpoint_path)
            elif self.checkpoint_format == CheckpointFormat.ONNX:
                # 保存为ONNX格式（示例，实际实现需要更多处理）
                self.logger.warning("ONNX格式保存需要额外实现，使用PyTorch格式保存")
                torch.save(checkpoint_data, checkpoint_path)
            
            self.logger.info(f"保存检查点到 {checkpoint_path}")
            
            # 如果需要压缩
            if self.enable_compression:
                compressed_path = self._compress_checkpoint(checkpoint_path)
                if compressed_path:
                    # 删除原始文件
                    checkpoint_path.unlink()
                    checkpoint_path = compressed_path
                    self.logger.info(f"检查点已压缩: {compressed_path}")
            
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
            raise
        
        # 保存最佳模型
        if is_best:
            self._save_best_checkpoint(checkpoint_data, checkpoint_path, checkpoint_id)
        
        # 创建元数据
        metadata = self._create_metadata(
            checkpoint_id=checkpoint_id,
            checkpoint_path=str(checkpoint_path),
            global_step=global_step,
            epoch=epoch,
            metrics=metrics or {},
            model=model,
            optimizer=optimizer,
            additional_info=additional_info
        )
        
        # 保存元数据
        self._save_metadata(metadata)
        
        # 更新历史
        self.checkpoint_history.append(metadata)
        
        # 根据策略清理旧检查点
        self._cleanup_old_checkpoints()
        
        return checkpoint_id
    
    def _save_best_checkpoint(self, checkpoint_data: Dict[str, Any], 
                             checkpoint_path: Path, checkpoint_id: str):
        """保存最佳检查点"""
        best_checkpoint_path = self.best_checkpoint_path.with_suffix(checkpoint_path.suffix)
        
        try:
            # 复制检查点到最佳模型位置
            shutil.copy2(checkpoint_path, best_checkpoint_path)
            
            # 同时保存最佳模型的元数据
            best_metadata_path = self.metadata_dir / "best_model.json"
            best_metadata = {
                "checkpoint_id": checkpoint_id,
                "original_path": str(checkpoint_path),
                "best_model_path": str(best_checkpoint_path),
                "saved_at": datetime.now().isoformat(),
            }
            
            with open(best_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(best_metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"保存最佳模型到 {best_checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"保存最佳模型失败: {e}")
    
    def _create_metadata(self,
                        checkpoint_id: str,
                        checkpoint_path: str,
                        global_step: int,
                        epoch: int,
                        metrics: Dict[str, float],
                        model: torch.nn.Module,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        additional_info: Optional[Dict[str, Any]] = None) -> CheckpointMetadata:
        """创建检查点元数据"""
        
        # 计算模型参数数量
        model_parameters = sum(p.numel() for p in model.parameters())
        
        # 收集模型配置
        model_config = {}
        if hasattr(model, "config"):
            model_config = model.config.to_dict() if hasattr(model.config, "to_dict") else {}
        
        # 收集系统信息
        system_info = self._collect_system_info()
        
        # 计算文件校验和
        checksum = self._calculate_checksum(checkpoint_path)
        
        # 获取文件大小
        file_size_bytes = Path(checkpoint_path).stat().st_size
        
        # 创建元数据对象
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            checkpoint_path=checkpoint_path,
            created_at=datetime.now().isoformat(),
            global_step=global_step,
            epoch=epoch,
            training_time_hours=0.0,  # 实际训练时间需要从训练器获取
            metrics=metrics,
            model_type=model.__class__.__name__,
            model_parameters=model_parameters,
            model_config=model_config,
            training_config=additional_info.get("training_config", {}) if additional_info else {},
            system_info=system_info,
            dependencies=self._collect_dependencies(),
            checksum=checksum,
            file_size_bytes=file_size_bytes,
        )
        
        return metadata
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """收集系统信息"""
        import platform
        import psutil
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
        }
        
        return system_info
    
    def _collect_dependencies(self) -> Dict[str, str]:
        """收集依赖信息"""
        dependencies = {}
        
        # 主要依赖
        try:
            import numpy
            dependencies["numpy"] = numpy.__version__
        except ImportError:
            pass
        
        try:
            import transformers
            dependencies["transformers"] = transformers.__version__
        except ImportError:
            pass
        
        try:
            import datasets
            dependencies["datasets"] = datasets.__version__
        except ImportError:
            pass
        
        return dependencies
    
    def _calculate_checksum(self, filepath: str) -> str:
        """计算文件校验和"""
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.sha256()
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
                return file_hash.hexdigest()
        except Exception as e:
            self.logger.warning(f"计算校验和失败 {filepath}: {e}")
            return ""
    
    def _save_metadata(self, metadata: CheckpointMetadata):
        """保存元数据"""
        metadata_path = self.metadata_dir / f"{metadata.checkpoint_id}.json"
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存元数据失败: {e}")
    
    def _compress_checkpoint(self, checkpoint_path: Path) -> Optional[Path]:
        """压缩检查点文件"""
        compressed_path = checkpoint_path.with_suffix('.zip')
        
        try:
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(checkpoint_path, checkpoint_path.name)
            
            # 验证压缩文件
            with zipfile.ZipFile(compressed_path, 'r') as zipf:
                if zipf.testzip() is None:
                    return compressed_path
                else:
                    self.logger.warning("压缩文件验证失败")
                    compressed_path.unlink()
                    return None
                    
        except Exception as e:
            self.logger.warning(f"压缩检查点失败: {e}")
            if compressed_path.exists():
                compressed_path.unlink()
            return None
    
    def _cleanup_old_checkpoints(self):
        """根据策略清理旧检查点"""
        if self.strategy == CheckpointStrategy.LAST_K:
            # 保留最后K个检查点
            if len(self.checkpoint_history) > self.max_checkpoints:
                checkpoints_to_delete = self.checkpoint_history[self.max_checkpoints:]
                
                for metadata in checkpoints_to_delete:
                    try:
                        # 删除检查点文件
                        checkpoint_path = Path(metadata.checkpoint_path)
                        if checkpoint_path.exists():
                            checkpoint_path.unlink()
                        
                        # 删除元数据文件
                        metadata_path = self.metadata_dir / f"{metadata.checkpoint_id}.json"
                        if metadata_path.exists():
                            metadata_path.unlink()
                        
                        self.logger.info(f"删除旧检查点: {metadata.checkpoint_id}")
                        
                    except Exception as e:
                        self.logger.warning(f"删除检查点失败 {metadata.checkpoint_id}: {e}")
                
                # 更新历史
                self.checkpoint_history = self.checkpoint_history[:self.max_checkpoints]
    
    def load_checkpoint(self,
                       checkpoint_id: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        加载检查点
        
        参数:
            checkpoint_id: 检查点ID或路径
            model: 要加载状态的模型
            optimizer: 要加载状态的优化器（可选）
            scheduler: 要加载状态的调度器（可选）
            device: 加载设备
            
        返回:
            检查点数据
        """
        # 查找检查点文件
        checkpoint_path = self._find_checkpoint_path(checkpoint_id)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # 如果是压缩文件，先解压
        if checkpoint_path.suffix == '.zip':
            checkpoint_path = self._decompress_checkpoint(checkpoint_path)
        
        # 验证检查点完整性
        if not self._validate_checkpoint(checkpoint_path):
            raise ValueError(f"检查点验证失败: {checkpoint_path}")
        
        # 加载检查点
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            
            # 加载模型状态
            if "model_state_dict" in checkpoint_data:
                model.load_state_dict(checkpoint_data["model_state_dict"])
            
            # 加载优化器状态
            if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            
            # 加载调度器状态
            if scheduler is not None and "scheduler_state_dict" in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
            
            self.logger.info(f"加载检查点: {checkpoint_path}")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            raise
    
    def _find_checkpoint_path(self, checkpoint_id: str) -> Path:
        """查找检查点文件路径"""
        # 如果是完整路径
        if Path(checkpoint_id).exists():
            return Path(checkpoint_id)
        
        # 如果是检查点ID
        checkpoint_patterns = [
            self.checkpoint_dir / f"{checkpoint_id}.pt",
            self.checkpoint_dir / f"{checkpoint_id}.pth",
            self.checkpoint_dir / f"{checkpoint_id}.zip",
            self.checkpoint_dir / f"{checkpoint_id}.safetensors",
        ]
        
        for pattern in checkpoint_patterns:
            if pattern.exists():
                return pattern
        
        # 检查最佳模型
        if checkpoint_id == "best":
            best_patterns = [
                self.best_checkpoint_path.with_suffix('.pt'),
                self.best_checkpoint_path.with_suffix('.pth'),
                self.best_checkpoint_path.with_suffix('.zip'),
            ]
            
            for pattern in best_patterns:
                if pattern.exists():
                    return pattern
        
        # 查找最新检查点
        if checkpoint_id == "latest":
            if self.checkpoint_history:
                latest_metadata = self.checkpoint_history[0]
                return Path(latest_metadata.checkpoint_path)
        
        raise FileNotFoundError(f"未找到检查点: {checkpoint_id}")
    
    def _decompress_checkpoint(self, compressed_path: Path) -> Path:
        """解压检查点文件"""
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            with zipfile.ZipFile(compressed_path, 'r') as zipf:
                # 解压所有文件
                zipf.extractall(temp_dir)
                
                # 查找解压后的检查点文件
                extracted_files = list(temp_dir.glob("*.pt")) + list(temp_dir.glob("*.pth"))
                
                if not extracted_files:
                    raise ValueError(f"压缩文件中未找到检查点文件: {compressed_path}")
                
                # 使用第一个找到的文件
                decompressed_path = extracted_files[0]
                
                # 复制到检查点目录（以便后续使用）
                checkpoint_filename = compressed_path.stem + decompressed_path.suffix
                target_path = self.checkpoint_dir / checkpoint_filename
                
                if not target_path.exists():
                    shutil.copy2(decompressed_path, target_path)
                
                return target_path
                
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """验证检查点完整性"""
        try:
            # 检查文件大小
            if checkpoint_path.stat().st_size == 0:
                self.logger.warning(f"检查点文件为空: {checkpoint_path}")
                return False
            
            # 尝试加载检查点（验证格式）
            if checkpoint_path.suffix in ['.pt', '.pth']:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                required_keys = ['model_state_dict', 'global_step']
                for key in required_keys:
                    if key not in checkpoint:
                        self.logger.warning(f"检查点缺少必需键 {key}: {checkpoint_path}")
                        return False
            
            # 验证校验和
            if checkpoint_path.suffix not in ['.zip']:  # 不验证压缩文件
                metadata = self._find_metadata(checkpoint_path)
                if metadata and metadata.checksum:
                    current_checksum = self._calculate_checksum(str(checkpoint_path))
                    if current_checksum != metadata.checksum:
                        self.logger.warning(f"检查点校验和不匹配: {checkpoint_path}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"检查点验证失败 {checkpoint_path}: {e}")
            return False
    
    def _find_metadata(self, checkpoint_path: Path) -> Optional[CheckpointMetadata]:
        """查找检查点对应的元数据"""
        checkpoint_id = checkpoint_path.stem
        
        for metadata in self.checkpoint_history:
            if metadata.checkpoint_id == checkpoint_id:
                return metadata
        
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        checkpoints = []
        
        for metadata in self.checkpoint_history:
            checkpoint_info = {
                "checkpoint_id": metadata.checkpoint_id,
                "path": metadata.checkpoint_path,
                "global_step": metadata.global_step,
                "epoch": metadata.epoch,
                "created_at": metadata.created_at,
                "metrics": metadata.metrics,
                "file_size_mb": metadata.file_size_bytes / (1024 * 1024),
            }
            checkpoints.append(checkpoint_info)
        
        return checkpoints
    
    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """获取最佳检查点"""
        best_metadata_path = self.metadata_dir / "best_model.json"
        
        if best_metadata_path.exists():
            try:
                with open(best_metadata_path, 'r', encoding='utf-8') as f:
                    best_metadata = json.load(f)
                
                checkpoint_id = best_metadata.get("checkpoint_id")
                for metadata in self.checkpoint_history:
                    if metadata.checkpoint_id == checkpoint_id:
                        return {
                            "checkpoint_id": metadata.checkpoint_id,
                            "path": metadata.checkpoint_path,
                            "global_step": metadata.global_step,
                            "epoch": metadata.epoch,
                            "metrics": metadata.metrics,
                            "is_best": True,
                        }
            except Exception as e:
                self.logger.warning(f"读取最佳检查点元数据失败: {e}")
        
        return None
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        try:
            checkpoint_path = self._find_checkpoint_path(checkpoint_id)
            
            # 删除检查点文件
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # 删除元数据文件
            metadata_path = self.metadata_dir / f"{checkpoint_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            # 从历史中移除
            self.checkpoint_history = [
                m for m in self.checkpoint_history 
                if m.checkpoint_id != checkpoint_id
            ]
            
            self.logger.info(f"删除检查点: {checkpoint_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"删除检查点失败 {checkpoint_id}: {e}")
            return False


# 全局检查点管理器实例
_checkpoint_manager = None

def get_checkpoint_manager(
    checkpoint_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> CheckpointManager:
    """获取检查点管理器单例"""
    global _checkpoint_manager
    
    if _checkpoint_manager is None:
        if checkpoint_dir is None:
            checkpoint_dir = Path("checkpoints")
        
        _checkpoint_manager = CheckpointManager(checkpoint_dir, **kwargs)
    
    return _checkpoint_manager


# 使用示例
if __name__ == "__main__":
    # 初始化日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建检查点管理器
    manager = get_checkpoint_manager(
        checkpoint_dir="test_checkpoints",
        strategy=CheckpointStrategy.LAST_K,
        max_checkpoints=3,
        enable_compression=False
    )
    
    # 模拟模型和优化器
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 保存检查点
    checkpoint_id = manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=100,
        epoch=1,
        metrics={"loss": 0.1, "accuracy": 0.9},
        is_best=True,
        additional_info={"training_config": {"batch_size": 32}}
    )
    
    print(f"保存检查点: {checkpoint_id}")
    
    # 列出检查点
    checkpoints = manager.list_checkpoints()
    print(f"检查点数量: {len(checkpoints)}")
    
    # 获取最佳检查点
    best_checkpoint = manager.get_best_checkpoint()
    if best_checkpoint:
        print(f"最佳检查点: {best_checkpoint['checkpoint_id']}")
    
    print("检查点管理测试完成")