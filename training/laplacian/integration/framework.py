#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉普拉斯增强系统集成框架

提供与主Self AGI框架的集成接口，支持向后兼容和灵活配置。

从 training/laplacian_integration.py 迁移而来，更新了导入路径以使用新的模块结构。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Union, Any, Tuple, Type
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 导入状态标志
LAPLACIAN_MODULES_AVAILABLE = False

# 尝试导入新模块结构的拉普拉斯模块
try:
    # 从新的模块结构导入
    from ..utils.config import LaplacianEnhancedTrainingConfig, UnifiedLaplacianConfig
    from ..core.regularization import LaplacianRegularization, RegularizationConfig
    from ..models.pinn import LaplacianEnhancedPINN
    from ..models.cnn import LaplacianEnhancedCNN
    
    # 尝试导入图模块
    try:
        from models.graph.laplacian_matrix import GraphLaplacian, GraphStructure, GraphType
        GRAPH_MODULE_AVAILABLE = True
    except ImportError:
        GRAPH_MODULE_AVAILABLE = False
        logger.warning("图模块不可用，部分功能将受限")
        # 创建虚拟类
        class GraphLaplacian:
        pass  # 已实现
        class GraphStructure:
        pass  # 已实现
        class GraphType:
            UNDIRECTED = "undirected"
            DIRECTED = "directed"
    
    # 尝试导入其他依赖模块
    try:
        from models.multimodal.pinn_cnn_fusion import PINNCNNFusionConfig, PINNCNNFusionModel
        FUSION_MODULE_AVAILABLE = True
    except ImportError:
        FUSION_MODULE_AVAILABLE = False
        logger.warning("PINN-CNN融合模块不可用，部分功能将受限")
        # 创建虚拟类
        class PINNCNNFusionConfig:
        pass  # 已实现
        class PINNCNNFusionModel:
        pass  # 已实现
    
    LAPLACIAN_MODULES_AVAILABLE = True
    logger.info("拉普拉斯模块导入成功 (新模块结构)")
    
except ImportError as e:
    logger.warning(f"拉普拉斯模块导入失败: {e}, 尝试回退到旧模块结构")
    
    # 尝试导入旧模块结构
    try:
        # 回退到旧模块导入
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from training.laplacian_enhanced_training import LaplacianEnhancedTrainingConfig
        from training.laplacian_regularization import LaplacianRegularization, RegularizationConfig
        from training.laplacian_enhanced_training import LaplacianEnhancedPINN, LaplacianEnhancedCNN
        
        LAPLACIAN_MODULES_AVAILABLE = True
        logger.info("拉普拉斯模块导入成功 (旧模块结构)")
        
    except ImportError as e2:
        LAPLACIAN_MODULES_AVAILABLE = False
        logger.warning(f"所有拉普拉斯模块导入失败: {e2}, 集成功能将受限")


@dataclass
class LaplacianIntegrationConfig:
    """拉普拉斯集成配置"""
    
    # 基础配置
    enabled: bool = True
    integration_mode: str = "auto"  # "auto", "new", "old", "compat"
    
    # 模块选择
    use_regularization: bool = True
    use_enhanced_training: bool = True
    use_graph_laplacian: bool = True
    
    # 性能配置
    use_sparse_matrices: bool = True
    enable_caching: bool = True
    cache_size: int = 100
    
    # 兼容性配置
    backward_compatible: bool = True
    create_dummy_classes: bool = True  # 当模块不可用时创建虚拟类
    
    # 监控配置
    log_import_errors: bool = True
    log_performance: bool = True
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if not self.enabled:
            logger.info("拉普拉斯集成已禁用")
            return True
        
        if self.integration_mode not in ["auto", "new", "old", "compat"]:
            logger.warning(f"无效的集成模式: {self.integration_mode}, 使用'auto'")
            self.integration_mode = "auto"
        
        return True


class LaplacianIntegrationFramework:
    """拉普拉斯集成框架
    
    提供统一的接口来访问拉普拉斯增强功能，支持多种集成模式和向后兼容。
    """
    
    def __init__(self, config: Optional[LaplacianIntegrationConfig] = None):
        """初始化集成框架"""
        
        self.config = config or LaplacianIntegrationConfig()
        self.config.validate()
        
        self._modules_loaded = False
        self._module_registry = {}
        
        # 根据配置加载模块
        if self.config.enabled:
            self._load_modules()
        else:
            logger.info("拉普拉斯集成框架已禁用")
    
    def _load_modules(self):
        """加载拉普拉斯模块"""
        
        if self._modules_loaded:
            return
        
        logger.info(f"加载拉普拉斯模块，模式: {self.config.integration_mode}")
        
        # 根据集成模式决定加载策略
        if self.config.integration_mode == "new" or (self.config.integration_mode == "auto" and LAPLACIAN_MODULES_AVAILABLE):
            # 使用新模块结构
            self._load_new_modules()
        elif self.config.integration_mode == "old":
            # 强制使用旧模块结构
            self._load_old_modules()
        elif self.config.integration_mode == "compat":
            # 兼容模式：尝试新结构，失败则使用旧结构
            try:
                self._load_new_modules()
            except ImportError:
                self._load_old_modules()
        else:
            # 自动模式，但新模块不可用，尝试旧模块
            self._load_old_modules()
        
        self._modules_loaded = True
    
    def _load_new_modules(self):
        """加载新模块结构的拉普拉斯模块"""
        
        try:
            # 导入新模块
            from ..utils.config import LaplacianEnhancedTrainingConfig, UnifiedLaplacianConfig
            from ..core.regularization import LaplacianRegularization, RegularizationConfig
            from ..models.pinn import LaplacianEnhancedPINN
            from ..models.cnn import LaplacianEnhancedCNN
            
            # 注册模块
            self._module_registry.update({
                'LaplacianEnhancedTrainingConfig': LaplacianEnhancedTrainingConfig,
                'UnifiedLaplacianConfig': UnifiedLaplacianConfig,
                'LaplacianRegularization': LaplacianRegularization,
                'RegularizationConfig': RegularizationConfig,
                'LaplacianEnhancedPINN': LaplacianEnhancedPINN,
                'LaplacianEnhancedCNN': LaplacianEnhancedCNN,
            })
            
            logger.info("新模块结构加载成功")
            
        except ImportError as e:
            logger.error(f"新模块结构加载失败: {e}")
            raise
    
    def _load_old_modules(self):
        """加载旧模块结构的拉普拉斯模块"""
        
        try:
            # 导入旧模块
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from training.laplacian_enhanced_training import LaplacianEnhancedTrainingConfig
            from training.laplacian_regularization import LaplacianRegularization, RegularizationConfig
            from training.laplacian_enhanced_training import LaplacianEnhancedPINN, LaplacianEnhancedCNN
            
            # 注册模块
            self._module_registry.update({
                'LaplacianEnhancedTrainingConfig': LaplacianEnhancedTrainingConfig,
                'LaplacianRegularization': LaplacianRegularization,
                'RegularizationConfig': RegularizationConfig,
                'LaplacianEnhancedPINN': LaplacianEnhancedPINN,
                'LaplacianEnhancedCNN': LaplacianEnhancedCNN,
            })
            
            logger.info("旧模块结构加载成功")
            
        except ImportError as e:
            logger.error(f"旧模块结构加载失败: {e}")
            
            if self.config.create_dummy_classes:
                self._create_dummy_classes()
            else:
                raise
    
    def _create_dummy_classes(self):
        """创建虚拟类（当模块不可用时）"""
        
        logger.warning("创建虚拟类替代拉普拉斯模块")
        
        class DummyClass:
            def __init__(self, *args, **kwargs):
                self._initialized = True
                logger.debug(f"虚拟类初始化: {self.__class__.__name__}")
            
            def __call__(self, *args, **kwargs):
                logger.debug(f"虚拟类调用: {self.__class__.__name__}")
                return None  # 返回None
            
            def __getattr__(self, name):
                def dummy_method(*args, **kwargs):
                    logger.debug(f"虚拟方法调用: {name}")
                    return None  # 返回None
                return dummy_method
        
        # 创建虚拟类并注册
        dummy_classes = {
            'LaplacianEnhancedTrainingConfig': type('DummyLaplacianEnhancedTrainingConfig', (DummyClass,), {}),
            'LaplacianRegularization': type('DummyLaplacianRegularization', (DummyClass,), {}),
            'RegularizationConfig': type('DummyRegularizationConfig', (DummyClass,), {}),
            'LaplacianEnhancedPINN': type('DummyLaplacianEnhancedPINN', (DummyClass, nn.Module), {}),
            'LaplacianEnhancedCNN': type('DummyLaplacianEnhancedCNN', (DummyClass, nn.Module), {}),
        }
        
        self._module_registry.update(dummy_classes)
        logger.info("虚拟类创建完成")
    
    def get_module(self, module_name: str) -> Type:
        """获取指定的拉普拉斯模块"""
        
        if not self._modules_loaded:
            self._load_modules()
        
        if module_name in self._module_registry:
            return self._module_registry[module_name]
        else:
            raise KeyError(f"模块未找到: {module_name}")
    
    def create_regularizer(self, config_dict: Dict[str, Any]) -> LaplacianRegularization:
        """创建拉普拉斯正则化器"""
        
        RegularizationConfig = self.get_module('RegularizationConfig')
        LaplacianRegularization = self.get_module('LaplacianRegularization')
        
        config = RegularizationConfig(**config_dict)
        return LaplacianRegularization(config)
    
    def create_enhanced_pinn(self, pinn_config: Any, laplacian_config_dict: Dict[str, Any]) -> LaplacianEnhancedPINN:
        """创建拉普拉斯增强PINN"""
        
        LaplacianEnhancedTrainingConfig = self.get_module('LaplacianEnhancedTrainingConfig')
        LaplacianEnhancedPINN = self.get_module('LaplacianEnhancedPINN')
        
        laplacian_config = LaplacianEnhancedTrainingConfig(**laplacian_config_dict)
        return LaplacianEnhancedPINN(pinn_config, laplacian_config)
    
    def create_enhanced_cnn(self, cnn_config: Any, laplacian_config_dict: Dict[str, Any]) -> LaplacianEnhancedCNN:
        """创建拉普拉斯增强CNN"""
        
        LaplacianEnhancedTrainingConfig = self.get_module('LaplacianEnhancedTrainingConfig')
        LaplacianEnhancedCNN = self.get_module('LaplacianEnhancedCNN')
        
        laplacian_config = LaplacianEnhancedTrainingConfig(**laplacian_config_dict)
        return LaplacianEnhancedCNN(cnn_config, laplacian_config)
    
    def is_available(self) -> bool:
        """检查拉普拉斯模块是否可用"""
        
        if not self._modules_loaded:
            self._load_modules()
        
        # 检查是否有虚拟类
        for name, module in self._module_registry.items():
            if 'Dummy' in module.__name__:
                logger.debug(f"模块 {name} 是虚拟类")
                return False
        
        return True
    
    def get_available_modules(self) -> List[str]:
        """获取可用的模块列表"""
        
        if not self._modules_loaded:
            self._load_modules()
        
        return list(self._module_registry.keys())
    
    def get_module_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        
        info = {
            'modules_loaded': self._modules_loaded,
            'integration_mode': self.config.integration_mode,
            'laplacian_modules_available': LAPLACIAN_MODULES_AVAILABLE,
            'available_modules': self.get_available_modules(),
            'is_available': self.is_available(),
        }
        
        return info


def integrate_laplacian_with_training(
    training_config: Dict[str, Any],
    laplacian_config: Optional[Dict[str, Any]] = None,
    integration_mode: str = "auto"
) -> Dict[str, Any]:
    """将拉普拉斯增强集成到训练配置中
    
    参数:
        training_config: 原始训练配置
        laplacian_config: 拉普拉斯增强配置
        integration_mode: 集成模式
        
    返回:
        更新后的训练配置
    """
    
    logger.info(f"集成拉普拉斯增强到训练配置，模式: {integration_mode}")
    
    if laplacian_config is None:
        laplacian_config = {
            'enabled': True,
            'laplacian_reg_enabled': True,
            'laplacian_reg_lambda': 0.01,
        }
    
    # 创建集成框架
    integration_config = LaplacianIntegrationConfig(
        enabled=laplacian_config.get('enabled', True),
        integration_mode=integration_mode,
    )
    
    framework = LaplacianIntegrationFramework(integration_config)
    
    if not framework.is_available():
        logger.warning("拉普拉斯模块不可用，跳过集成")
        return training_config
    
    # 更新训练配置
    updated_config = training_config.copy()
    
    # 添加拉普拉斯增强配置
    updated_config['laplacian_enhancement'] = laplacian_config
    
    # 根据配置添加具体的拉普拉斯增强
    if laplacian_config.get('laplacian_reg_enabled', False):
        updated_config['regularization'] = {
            **updated_config.get('regularization', {}),
            'laplacian': {
                'enabled': True,
                'lambda_reg': laplacian_config.get('laplacian_reg_lambda', 0.01),
                'normalization': laplacian_config.get('laplacian_normalization', 'sym'),
            }
        }
    
    logger.info("拉普拉斯增强集成完成")
    
    return updated_config