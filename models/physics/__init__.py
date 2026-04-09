# 物理建模和PINN模块
"""
物理信息神经网络(PINN)和物理建模模块

功能：
1. 物理信息神经网络(PINN)基础框架
2. 机器人动力学物理建模
3. 传感器数据物理约束
4. 环境交互物理仿真
5. 多物理场耦合建模

模块列表：
- pinn_framework: PINN基础框架
- robot_dynamics: 机器人动力学建模
- sensor_physics: 传感器物理约束
- environment_simulation: 环境交互仿真
- multiphysics_coupling: 多物理场耦合
"""

# 导入所有模块
from .pinn_framework import (
    PINNConfig,
    PINNModel,
    PhysicsConstraint,
    PDEResidualConstraint,
    BoundaryConditionConstraint,
    InitialConditionConstraint,
    DataConstraint,
    burgers_equation,
    heat_equation,
    wave_equation,
)

from .robot_dynamics import (
    RobotConfig,
    LagrangianDynamics,
    HamiltonianDynamics,
    RobotPINN,
)

from .sensor_physics import (
    SensorConfig,
    IMUModel,
    ForceTorqueSensor,
    CameraModel,
    SensorFusionPINN,
)

from .environment_simulation import (
    EnvironmentConfig,
    RigidBody,
    CollisionDetector,
    ContactSolver,
    PhysicsSimulator,
    FluidSimulator,
)

from .multiphysics_coupling import (
    MultiPhysicsConfig,
    PhysicsField,
    StructuralField,
    ThermalField,
    FluidField,
    MonolithicCoupling,
    PartitionedCoupling,
    MultiPhysicsPINN,
)

# 导出所有类
__all__ = [
    # pinn_framework
    "PINNConfig",
    "PINNModel",
    "PhysicsConstraint",
    "PDEResidualConstraint",
    "BoundaryConditionConstraint",
    "InitialConditionConstraint",
    "DataConstraint",
    "burgers_equation",
    "heat_equation",
    "wave_equation",
    # robot_dynamics
    "RobotConfig",
    "LagrangianDynamics",
    "HamiltonianDynamics",
    "RobotPINN",
    # sensor_physics
    "SensorConfig",
    "IMUModel",
    "ForceTorqueSensor",
    "CameraModel",
    "SensorFusionPINN",
    # environment_simulation
    "EnvironmentConfig",
    "RigidBody",
    "CollisionDetector",
    "ContactSolver",
    "PhysicsSimulator",
    "FluidSimulator",
    # multiphysics_coupling
    "MultiPhysicsConfig",
    "PhysicsField",
    "StructuralField",
    "ThermalField",
    "FluidField",
    "MonolithicCoupling",
    "PartitionedCoupling",
    "MultiPhysicsPINN",
]
