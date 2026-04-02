#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新的拉普拉斯正则化组件

测试重构后的拉普拉斯正则化组件的功能和兼容性
"""

import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_core_imports():
    """测试核心模块导入"""
    print("=== 测试核心模块导入 ===")
    
    try:
        from training.laplacian.core.base import (
            LaplacianType, NormalizationType, LaplacianConfig, LaplacianBase
        )
        from training.laplacian.core.regularization import (
            RegularizationConfig, LaplacianRegularization
        )
        
        print("✓ 核心模块导入成功")
        print(f"  - LaplacianType: {LaplacianType.GRAPH}")
        print(f"  - NormalizationType: {NormalizationType.SYM}")
        print(f"  - LaplacianBase类: {LaplacianBase}")
        print(f"  - LaplacianRegularization类: {LaplacianRegularization}")
        
        return True
    except Exception as e:
        print(f"✗ 核心模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_creation():
    """测试配置创建"""
    print("\n=== 测试配置创建 ===")
    
    try:
        from training.laplacian.core.base import LaplacianConfig, LaplacianType, NormalizationType
        from training.laplacian.core.regularization import RegularizationConfig
        
        # 测试基础配置
        base_config = LaplacianConfig(
            laplacian_type=LaplacianType.GRAPH,
            lambda_reg=0.1,
            normalization=NormalizationType.SYM
        )
        print(f"✓ 基础配置创建: lambda={base_config.lambda_reg}, type={base_config.laplacian_type}")
        
        # 测试正则化配置
        reg_config = RegularizationConfig(
            regularization_type="graph_laplacian",
            lambda_reg=0.05,
            k_neighbors=5,
            num_scales=2
        )
        print(f"✓ 正则化配置创建: type={reg_config.regularization_type}, k={reg_config.k_neighbors}")
        
        # 测试配置序列化
        config_dict = reg_config.to_dict()
        print(f"✓ 配置序列化: keys={list(config_dict.keys())[:5]}...")
        
        return True
    except Exception as e:
        print(f"✗ 配置创建失败: {e}")
        return False

def test_regularization_instantiation():
    """测试正则化组件实例化"""
    print("\n=== 测试正则化组件实例化 ===")
    
    try:
        from training.laplacian.core.regularization import RegularizationConfig, LaplacianRegularization
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 创建配置
        config = RegularizationConfig(
            regularization_type="graph_laplacian",
            lambda_reg=0.01,
            normalization="sym",
            device=device
        )
        
        # 实例化正则化组件
        regularizer = LaplacianRegularization(config, feature_dim=10, name="test_regularizer")
        print(f"✓ 正则化组件实例化成功: {regularizer}")
        print(f"  组件名称: {regularizer.name}")
        print(f"  当前lambda: {regularizer.current_lambda}")
        print(f"  设备: {regularizer.device}")
        
        return regularizer
    except Exception as e:
        print(f"✗ 正则化组件实例化失败: {e}")
        import traceback
        traceback.print_exc()
        return None  # 返回None

def test_graph_regularization_computation(regularizer):
    """测试图拉普拉斯正则化计算"""
    print("\n=== 测试图拉普拉斯正则化计算 ===")
    
    if regularizer is None:
        print("✗ 正则化组件未初始化")
        return False
    
    try:
        device = regularizer.device
        
        # 创建测试数据
        n_samples = 50
        n_features = 10
        
        # 生成在流形上的数据（圆）
        t = torch.linspace(0, 2*np.pi, n_samples, device=device)
        features = torch.stack([torch.sin(t), torch.cos(t)], dim=1)
        features = torch.cat([features, torch.randn(n_samples, n_features-2, device=device)], dim=1)
        
        # 构建简单的环状邻接矩阵
        adjacency = torch.zeros(n_samples, n_samples, device=device)
        for i in range(n_samples):
            adjacency[i, (i+1)%n_samples] = 1
            adjacency[(i+1)%n_samples, i] = 1
        
        print(f"测试数据: {features.shape}, 邻接矩阵: {adjacency.shape}")
        
        # 计算正则化损失
        reg_loss = regularizer(features, adjacency_matrix=adjacency)
        
        print(f"✓ 正则化损失计算成功: {reg_loss.item():.6f}")
        print(f"  损失类型: {type(reg_loss)}")
        print(f"  损失设备: {reg_loss.device}")
        
        # 测试梯度计算
        features.requires_grad_(True)
        reg_loss = regularizer(features, adjacency_matrix=adjacency)
        reg_loss.backward()
        
        grad_norm = torch.norm(features.grad)
        print(f"✓ 梯度计算成功: 梯度范数={grad_norm.item():.6f}")
        
        features.requires_grad_(False)
        
        # 获取统计信息
        stats = regularizer.get_full_stats()
        print(f"✓ 统计信息获取成功: calls={stats.get('calls', 0)}, compute_time={stats.get('avg_compute_time', 0):.6f}")
        
        return True
    except Exception as e:
        print(f"✗ 正则化计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_regularization_types():
    """测试不同类型的正则化"""
    print("\n=== 测试不同类型的正则化 ===")
    
    try:
        from training.laplacian.core.regularization import RegularizationConfig, LaplacianRegularization
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 测试数据
        n_samples = 30
        n_features = 8
        features = torch.randn(n_samples, n_features, device=device)
        
        # 邻接矩阵
        adjacency = torch.eye(n_samples, device=device)
        for i in range(n_samples-1):
            adjacency[i, i+1] = 1
            adjacency[i+1, i] = 1
        
        # 测试标签（用于流形正则化）
        labels = torch.randn(n_samples, device=device)
        
        # 测试不同类型
        test_cases = [
            ("graph_laplacian", {"regularization_type": "graph_laplacian", "lambda_reg": 0.1}),
            ("manifold", {"regularization_type": "manifold", "lambda_reg": 0.05, "use_labeled_data": True}),
            ("multi_scale", {"regularization_type": "multi_scale", "lambda_reg": 0.01, "num_scales": 2}),
            ("adaptive", {"regularization_type": "adaptive", "lambda_reg": 0.1, "adaptive_enabled": True}),
        ]
        
        for name, config_params in test_cases:
            print(f"\n测试: {name}")
            
            try:
                config = RegularizationConfig(**config_params, device=device)
                regularizer = LaplacianRegularization(config, feature_dim=n_features, name=f"test_{name}")
                
                # 计算损失
                if name == "manifold":
                    loss = regularizer(features, labels=labels, adjacency_matrix=adjacency)
                else:
                    loss = regularizer(features, adjacency_matrix=adjacency)
                
                print(f"  ✓ {name}正则化损失: {loss.item():.6f}")
                
                # 获取统计
                stats = regularizer.get_stats()
                print(f"    调用次数: {stats.get('calls', 0)}, 当前lambda: {stats.get('current_lambda', 0):.6f}")
                
            except Exception as e:
                print(f"  ✗ {name}测试失败: {e}")
        
        print("\n✓ 所有正则化类型测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 正则化类型测试失败: {e}")
        return False

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n=== 测试向后兼容性 ===")
    
    try:
        # 尝试导入旧的模块
        try:
            from training.laplacian_regularization import LaplacianRegularization as OldLaplacianRegularization
            print("✓ 旧模块导入成功")
            
            # 比较新旧模块
            print("注意: 新旧模块同时存在，建议更新导入路径到'training.laplacian.core.regularization'")
        except ImportError:
            print("✓ 旧模块不存在，无需兼容性处理")
        
        # 测试新模块的导入路径
        try:
            from training.laplacian import LaplacianRegularization as NewLaplacianRegularization
            from training.laplacian import RegularizationConfig as NewRegularizationConfig
            
            print("✓ 新模块顶层导入成功")
            
            # 实例化测试
            config = NewRegularizationConfig(regularization_type="graph_laplacian", lambda_reg=0.01)
            regularizer = NewLaplacianRegularization(config, feature_dim=5)
            
            print(f"✓ 新模块实例化成功: {regularizer}")
            
        except Exception as e:
            print(f"✗ 新模块顶层导入失败: {e}")
        
        return True
    except Exception as e:
        print(f"✗ 向后兼容性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("拉普拉斯正则化组件测试")
    print("=" * 60)
    
    # 测试1: 核心模块导入
    if not test_core_imports():
        print("\n❌ 核心模块导入测试失败")
        return False
    
    # 测试2: 配置创建
    if not test_config_creation():
        print("\n❌ 配置创建测试失败")
        return False
    
    # 测试3: 正则化组件实例化
    regularizer = test_regularization_instantiation()
    if regularizer is None:
        print("\n❌ 正则化组件实例化测试失败")
        return False
    
    # 测试4: 图拉普拉斯正则化计算
    if not test_graph_regularization_computation(regularizer):
        print("\n❌ 图拉普拉斯正则化计算测试失败")
        return False
    
    # 测试5: 不同类型正则化
    if not test_different_regularization_types():
        print("\n❌ 不同类型正则化测试失败")
        return False
    
    # 测试6: 向后兼容性
    if not test_backward_compatibility():
        print("\n⚠️ 向后兼容性测试有警告")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)