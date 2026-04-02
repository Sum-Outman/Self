# Physics-Informed Neural Networks (PINN) | 物理信息神经网络

This document provides detailed documentation of the Physics-Informed Neural Network (PINN) framework in Self AGI, including physics constraint loss functions, PDE residual automatic differentiation, and multi-physics coupling.

本文档详细介绍 Self AGI 中的物理信息神经网络（PINN）框架，包括物理约束损失函数、PDE 残差自动微分和多物理场耦合。

## Table of Contents | 目录
- [Overview | 概述](#overview--概述)
- [Mathematical Principles | 数学原理](#mathematical-principles--数学原理)
- [PINN Configuration | PINN 配置](#pinn-configuration--pinn-配置)
- [Physics Constraints | 物理约束](#physics-constraints--物理约束)
- [Supported PDE Types | 支持的 PDE 类型](#supported-pde-types--支持的-pde-类型)
- [Usage Examples | 使用示例](#usage-examples--使用示例)
- [Best Practices | 最佳实践](#best-practices--最佳实践)

---

## Overview | 概述

Physics-Informed Neural Networks (PINNs) are neural networks trained to satisfy given physical laws described by partial differential equations (PDEs), enabling data-efficient and physics-constrained learning.

物理信息神经网络（PINN）是经过训练以满足由偏微分方程（PDE）描述的给定物理定律的神经网络，实现了数据高效且受物理约束的学习。

### Key Features | 核心特性

- **Physics Constraint Loss | 物理约束损失**: Loss functions enforcing physical laws
- **PDE Residual Automatic Differentiation | PDE 残差自动微分**: Automatic computation of PDE residuals
- **Boundary and Initial Conditions | 边界条件和初始条件**: Handling of BCs and ICs
- **Multi-Physics Coupling | 多物理场耦合**: Support for coupled physics domains
- **Adaptive Weighting | 自适应权重**: Dynamic adjustment of loss weights

### Industrial-Grade Quality Standards | 工业级质量标准

- **Numerical Stability | 数值稳定性**: Double-precision computation, stable gradients
- **Computational Efficiency | 计算效率**: GPU acceleration, auto-differentiation optimization
- **Memory Efficiency | 内存效率**: Support for large-scale physics field simulation
- **Scalability | 可扩展性**: Modular design, easy to extend

---

## Mathematical Principles | 数学原理

### Basic PINN Equation | 基本 PINN 方程

The fundamental PINN loss function combines data loss and physics loss:

```
L = L_data + λ * L_physics
```

where:
- L_data is the loss on available data
- L_physics is the physics constraint loss
- λ is the weight balancing the two losses

### PDE Residual | PDE 残差

For a PDE of the form:

```
N(u) = f
```

where N is a differential operator, u is the solution, and f is a source term.

The PDE residual is:

```
R(u) = N(u) - f
```

### Automatic Differentiation | 自动微分

PINNs use automatic differentiation (via PyTorch autograd) to compute derivatives:

```python
# Compute first derivative
u_x = torch.autograd.grad(u, x, create_graph=True)[0]

# Compute second derivative
u_xx = torch.autograd.grad(u_x, x, create_graph=True)[0]
```

---

## PINN Configuration | PINN 配置

### Configuration Class | 配置类

```python
from models.physics.pinn_framework import PINNConfig

# Create PINN configuration
config = PINNConfig(
    # General configuration
    input_dim=3,                    # Input dimension (e.g., x, y, t)
    output_dim=1,                   # Output dimension (e.g., u)
    hidden_dim=64,                  # Hidden layer dimension
    num_layers=5,                   # Number of network layers
    activation="tanh",              # Activation function
    
    # Physics constraint configuration
    physics_weight=1.0,             # Physics loss weight
    data_weight=1.0,                # Data loss weight
    bc_weight=1.0,                  # Boundary condition weight
    ic_weight=1.0,                  # Initial condition weight
    
    # PDE configuration
    pde_type="burgers",             # PDE type
    pde_order=2,                    # PDE order
    use_autograd=True,               # Use automatic differentiation
    
    # Training configuration
    adaptive_weighting=True,         # Use adaptive weighting
    weight_update_freq=100,          # Weight update frequency
    grad_clip=1.0,                   # Gradient clipping
    
    # Performance configuration
    use_gpu=True,                    # Use GPU
    dtype=torch.float64,             # Data type (float64 recommended)
    parallel_computation=False,       # Use parallel computation
    
    # Mixed precision training
    use_mixed_precision=False,        # Use mixed precision training
    mixed_precision_dtype=torch.float16,
    amp_enabled=True,
    grad_scaler_enabled=True,
    
    # Distributed training
    distributed_training=False,
    world_size=1,
    local_rank=0,
    backend="nccl",
    
    # Incremental cache configuration
    enable_incremental_cache=True,
    max_cache_size=1000,
    cache_eviction_policy="adaptive",
    adaptive_cache_sizing=True,
    cache_enabled_for_pde=True,
    cache_enabled_for_gradients=True,
    cache_enabled_for_losses=True
)
```

---

## Physics Constraints | 物理约束

### Physics Constraint Base Class | 物理约束基类

```python
from models.physics.pinn_framework import PhysicsConstraint

class CustomPhysicsConstraint(PhysicsConstraint):
    """Custom physics constraint implementation"""
    
    def compute_residual(self, x, u, derivatives):
        """
        Compute PDE residual
        
        Args:
            x: Input coordinates
            u: Network output
            derivatives: Dictionary of derivatives
        
        Returns:
            PDE residual
        """
        # Get derivatives
        u_t = derivatives.get('u_t')  # du/dt
        u_x = derivatives.get('u_x')  # du/dx
        u_xx = derivatives.get('u_xx') # d²u/dx²
        
        # Compute residual for custom PDE
        residual = u_t + u * u_x - 0.01 * u_xx
        
        return residual
    
    def compute_boundary_loss(self, x_bc, u_bc):
        """Compute boundary condition loss"""
        return F.mse_loss(u_bc, self.boundary_values)
    
    def compute_initial_loss(self, x_ic, u_ic):
        """Compute initial condition loss"""
        return F.mse_loss(u_ic, self.initial_values)
```

### Built-in Physics Constraints | 内置物理约束

The framework includes several built-in physics constraints:

1. **Burgers' Equation | 伯格斯方程**: Fluid dynamics
2. **Navier-Stokes | 纳维-斯托克斯方程**: Fluid flow
3. **Heat Equation | 热方程**: Heat transfer
4. **Wave Equation | 波动方程**: Wave propagation
5. **Schrödinger Equation | 薛定谔方程**: Quantum mechanics
6. **Maxwell Equations | 麦克斯韦方程**: Electromagnetism
7. **Elasticity | 弹性方程**: Structural mechanics
8. **Reaction-Diffusion | 反应扩散方程**: Chemical reactions

---

## Supported PDE Types | 支持的 PDE 类型

### PDE Type Enum | PDE 类型枚举

```python
from models.physics.pinn_framework import PDEType

# Available PDE types
pde_types = [
    PDEType.BURGERS,           # Burgers' equation
    PDEType.NAVIER_STOKES,     # Navier-Stokes equations
    PDEType.HEAT,               # Heat equation
    PDEType.WAVE,               # Wave equation
    PDEType.SCHRODINGER,        # Schrödinger equation
    PDEType.MAXWELL,            # Maxwell equations
    PDEType.ELASTICITY,         # Elasticity equations
    PDEType.REACTION_DIFFUSION  # Reaction-diffusion equation
]
```

### Burgers' Equation | 伯格斯方程

**Equation | 方程**:
```
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
```

**Usage | 使用**:
```python
from models.physics.pinn_framework import PINNFramework, PDEType

# Initialize PINN for Burgers' equation
pinn = PINNFramework(
    pde_type=PDEType.BURGERS,
    viscosity=0.01,  # Kinematic viscosity
    config=config
)

# Train PINN
pinn.train(training_data, num_epochs=1000)
```

### Heat Equation | 热方程

**Equation | 方程**:
```
∂u/∂t = α ∂²u/∂x²
```

**Usage | 使用**:
```python
pinn = PINNFramework(
    pde_type=PDEType.HEAT,
    thermal_diffusivity=0.1,
    config=config
)
```

### Wave Equation | 波动方程

**Equation | 方程**:
```
∂²u/∂t² = c² ∂²u/∂x²
```

**Usage | 使用**:
```python
pinn = PINNFramework(
    pde_type=PDEType.WAVE,
    wave_speed=1.0,
    config=config
)
```

---

## Usage Examples | 使用示例

### Complete PINN Workflow | 完整 PINN 工作流

```python
import torch
import numpy as np
from models.physics.pinn_framework import (
    PINNFramework,
    PINNConfig,
    PDEType
)

# Step 1: Create configuration
config = PINNConfig(
    input_dim=2,           # x and t
    output_dim=1,          # u
    hidden_dim=64,
    num_layers=5,
    activation="tanh",
    physics_weight=1.0,
    adaptive_weighting=True,
    use_gpu=torch.cuda.is_available()
)

# Step 2: Initialize PINN framework
pinn = PINNFramework(
    pde_type=PDEType.BURGERS,
    viscosity=0.01,
    config=config
)

# Step 3: Generate or load data
# Collocation points for physics loss
x_physics = torch.linspace(-1, 1, 1000).requires_grad_(True)
t_physics = torch.linspace(0, 1, 1000).requires_grad_(True)

# Boundary points
x_boundary = torch.tensor([-1.0, 1.0]).requires_grad_(True)
t_boundary = torch.linspace(0, 1, 100).requires_grad_(True)

# Initial condition points
x_initial = torch.linspace(-1, 1, 100).requires_grad_(True)
t_initial = torch.zeros_like(x_initial).requires_grad_(True)

# Step 4: Set up training data
training_data = {
    'physics': {'x': x_physics, 't': t_physics},
    'boundary': {'x': x_boundary, 't': t_boundary},
    'initial': {'x': x_initial, 't': t_initial, 'u': -torch.sin(torch.pi * x_initial)}
}

# Step 5: Train PINN
results = pinn.train(
    training_data=training_data,
    num_epochs=5000,
    learning_rate=1e-3,
    verbose=True
)

# Step 6: Evaluate and visualize
x_test = torch.linspace(-1, 1, 200)
t_test = torch.linspace(0, 1, 100)
X, T = torch.meshgrid(x_test, t_test, indexing='ij')

u_pred = pinn.predict(X.flatten(), T.flatten())
u_pred = u_pred.reshape(X.shape)

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.contourf(T.numpy(), X.numpy(), u_pred.detach().numpy(), levels=50)
plt.colorbar(label='u(x,t)')
plt.xlabel('t')
plt.ylabel('x')
plt.title('PINN Solution of Burgers Equation')
plt.show()
```

### Multi-Physics Coupling | 多物理场耦合

```python
from models.physics.pinn_framework import MultiPhysicsPINN

# Define coupled physics
coupled_physics = MultiPhysicsPINN(
    physics_domains=[
        {'type': 'heat_transfer', 'name': 'temperature'},
        {'type': 'fluid_flow', 'name': 'velocity'},
        {'type': 'structural_mechanics', 'name': 'stress'}
    ],
    coupling_terms=[
        {'from': 'temperature', 'to': 'stress', 'type': 'thermal_expansion'},
        {'from': 'velocity', 'to': 'temperature', 'type': 'convection'}
    ]
)

# Train coupled physics model
coupled_physics.train(coupled_training_data)
```

---

## Best Practices | 最佳实践

1. **Use Double Precision | 使用双精度**: Float64 for numerical stability
2. **Adaptive Weighting | 自适应权重**: Balance physics and data losses dynamically
3. **Gradient Clipping | 梯度裁剪**: Prevent exploding gradients
4. **Incremental Cache | 增量缓存**: Cache frequent computations
5. **Good Initialization | 良好初始化**: Use appropriate weight initialization
6. **Monitor Residuals | 监控残差**: Track PDE residual convergence
7. **Validate with Analytics | 分析验证**: Compare with analytical solutions when possible
8. **GPU Acceleration | GPU 加速**: Use GPU for large-scale problems

---

*Last Updated: March 31, 2026*  
*最后更新: 2026年3月31日*
