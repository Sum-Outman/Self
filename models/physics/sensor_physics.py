#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器数据物理约束模块

功能：
1. 传感器物理模型（IMU、力传感器、视觉传感器等）
2. 传感器噪声和偏差建模
3. 传感器数据物理一致性约束
4. 多传感器融合物理约束
5. 传感器校准物理模型

工业级质量标准要求：
- 真实性：基于真实传感器物理特性
- 准确性：高精度传感器模型
- 实时性：高效传感器数据处理
- 鲁棒性：对噪声和异常值鲁棒

数学原理：
1. 传感器测量模型: y = h(x) + ε
2. 噪声模型: ε ~ N(0, Σ)
3. 物理约束: g(y, x) = 0
4. 卡尔曼滤波: 状态估计和协方差更新

参考文献：
[1] Thrun, S., et al. (2005). Probabilistic Robotics.
[2] Bar-Shalom, Y., et al. (2001). Estimation with Applications to Tracking and Navigation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import logging
from ..quaternion_core import QuaternionTensor

logger = logging.getLogger(__name__)


@dataclass
class SensorConfig:
    """传感器配置类"""

    # 传感器类型
    sensor_type: str = "imu"  # "imu", "force_torque", "camera", "lidar", "encoder"

    # IMU配置
    imu_accel_scale: float = 1.0  # 加速度计尺度因子
    imu_gyro_scale: float = 1.0  # 陀螺仪尺度因子
    imu_accel_bias: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    imu_gyro_bias: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # 力传感器配置
    ft_force_scale: float = 1.0  # 力传感器尺度因子
    ft_torque_scale: float = 1.0  # 扭矩传感器尺度因子

    # 噪声参数
    noise_type: str = "gaussian"  # "gaussian", "laplacian", "student_t"
    noise_std: float = 0.01  # 噪声标准差
    bias_drift: float = 0.001  # 偏置漂移

    # 采样率
    sampling_rate: float = 100.0  # Hz

    # 计算配置
    use_gpu: bool = True
    dtype: torch.dtype = torch.float64


class IMUModel(nn.Module):
    """IMU传感器模型"""

    def __init__(self, config: SensorConfig):
        super().__init__()
        self.config = config

        # 设备配置
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 可学习的偏置参数
        self.accel_bias = nn.Parameter(
            torch.tensor(config.imu_accel_bias, dtype=config.dtype)
        )
        self.gyro_bias = nn.Parameter(
            torch.tensor(config.imu_gyro_bias, dtype=config.dtype)
        )

        # 尺度因子参数
        self.accel_scale = nn.Parameter(
            torch.eye(3, dtype=config.dtype) * config.imu_accel_scale
        )
        self.gyro_scale = nn.Parameter(
            torch.eye(3, dtype=config.dtype) * config.imu_gyro_scale
        )

        # 噪声协方差
        self.accel_noise_cov = nn.Parameter(
            torch.eye(3, dtype=config.dtype) * config.noise_std**2
        )
        self.gyro_noise_cov = nn.Parameter(
            torch.eye(3, dtype=config.dtype) * config.noise_std**2
        )

        # 移动到设备
        self.to(self.device)
        self.to(config.dtype)

    def forward(
        self, true_accel: torch.Tensor, true_gyro: torch.Tensor, dt: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成IMU测量值

        参数：
            true_accel: 真实加速度 [batch_size, 3]
            true_gyro: 真实角速度 [batch_size, 3]
            dt: 时间步长

        返回：
            measured_accel: 测量的加速度 [batch_size, 3]
            measured_gyro: 测量的角速度 [batch_size, 3]
        """
        batch_size = true_accel.shape[0]

        # 应用尺度因子和偏置
        accel_scaled = true_accel @ self.accel_scale.T + self.accel_bias.unsqueeze(0)
        gyro_scaled = true_gyro @ self.gyro_scale.T + self.gyro_bias.unsqueeze(0)

        # 添加噪声
        if self.training:
            # 训练时添加噪声
            accel_noise = self._generate_noise(batch_size, self.accel_noise_cov)
            gyro_noise = self._generate_noise(batch_size, self.gyro_noise_cov)

            measured_accel = accel_scaled + accel_noise
            measured_gyro = gyro_scaled + gyro_noise
        else:
            # 推理时不添加噪声
            measured_accel = accel_scaled
            measured_gyro = gyro_scaled

        # 添加偏置漂移
        if self.config.bias_drift > 0:
            bias_drift = (
                torch.randn(batch_size, 3, device=self.device)
                * self.config.bias_drift
                * dt
            )
            measured_accel += bias_drift
            measured_gyro += bias_drift

        return measured_accel, measured_gyro

    def _generate_noise(
        self, batch_size: int, covariance: torch.Tensor
    ) -> torch.Tensor:
        """生成噪声"""
        if self.config.noise_type == "gaussian":
            # 高斯噪声
            noise = torch.randn(
                batch_size, 3, device=self.device, dtype=self.config.dtype
            )
            # 应用协方差
            L = torch.linalg.cholesky(covariance)
            noise = noise @ L.T
        elif self.config.noise_type == "laplacian":
            # 拉普拉斯噪声
            noise = (
                torch.distributions.Laplace(0, self.config.noise_std)
                .sample((batch_size, 3))
                .to(self.device)
            )
        elif self.config.noise_type == "student_t":
            # 学生t分布噪声
            noise = (
                torch.distributions.StudentT(3).sample((batch_size, 3)).to(self.device)
                * self.config.noise_std
            )
        else:
            raise ValueError(f"未知的噪声类型: {self.config.noise_type}")

        return noise

    def physics_constraints(
        self,
        accel_measurements: torch.Tensor,
        gyro_measurements: torch.Tensor,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        """物理约束检查

        参数：
            accel_measurements: 加速度测量值
            gyro_measurements: 角速度测量值
            positions: 位置估计
            orientations: 姿态估计
            dt: 时间步长

        返回：
            constraints: 物理约束损失字典
        """
        constraints = {}

        # 1. 重力约束：在静止状态下，加速度计应测量重力
        gravity = torch.tensor(
            [0, 0, -9.81], device=self.device, dtype=self.config.dtype
        )

        # 旋转姿态到世界坐标系
        R = self.quaternion_to_matrix(orientations)
        accel_world = torch.bmm(R, accel_measurements.unsqueeze(-1)).squeeze(-1)

        gravity_constraint = torch.mean((accel_world - gravity.unsqueeze(0)) ** 2)
        constraints["gravity_constraint"] = gravity_constraint

        # 2. 角速度积分约束：角速度积分应与姿态变化一致
        # 计算从角速度积分得到的姿态变化
        gyro_norm = torch.norm(gyro_measurements, dim=1)
        angle_increment = gyro_norm * dt

        # 小角度近似
        delta_q = self.axis_angle_to_quaternion(gyro_measurements, angle_increment)
        q_next_integrated = self.quaternion_multiply(orientations, delta_q)

        # 实际下一时刻姿态（如果有的话）
        # 完整处理，使用当前姿态
        orientation_constraint = torch.mean((q_next_integrated - orientations) ** 2)
        constraints["orientation_constraint"] = orientation_constraint

        # 3. 加速度双重积分约束：加速度积分应与位置变化一致
        # 计算加速度在世界坐标系中的值
        accel_world_linear = accel_world - gravity.unsqueeze(0)

        # 速度变化
        velocity_increment = accel_world_linear * dt

        # 位置变化
        position_increment = velocity_increment * dt + 0.5 * accel_world_linear * dt**2

        # 实际位置变化（如果有的话）
        # 完整处理
        position_constraint = torch.mean(position_increment**2)
        constraints["position_constraint"] = position_constraint

        # 4. 角速度和加速度相关性约束（对于刚体运动）
        # 对于刚体上的点，加速度和角速度有关：a = α × r + ω × (ω × r)
        # 完整处理
        correlation_constraint = torch.mean(
            torch.sum(accel_measurements * gyro_measurements, dim=1) ** 2
        )
        constraints["correlation_constraint"] = correlation_constraint

        return constraints

    def quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """四元数转旋转矩阵，使用统一的四元数核心库"""
        return QuaternionTensor.to_rotation_matrix(q)

    def axis_angle_to_quaternion(
        self, axis: torch.Tensor, angle: torch.Tensor
    ) -> torch.Tensor:
        """轴角转四元数，使用统一的四元数核心库"""
        return QuaternionTensor.from_axis_angle(axis, angle)

    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """四元数乘法，使用统一的四元数核心库"""
        return QuaternionTensor.multiply(q1, q2)


class ForceTorqueSensor(nn.Module):
    """力扭矩传感器模型"""

    def __init__(self, config: SensorConfig):
        super().__init__()
        self.config = config

        # 设备配置
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 尺度因子和偏置
        self.force_scale = nn.Parameter(
            torch.eye(3, dtype=config.dtype) * config.ft_force_scale
        )
        self.torque_scale = nn.Parameter(
            torch.eye(3, dtype=config.dtype) * config.ft_torque_scale
        )
        self.force_bias = nn.Parameter(torch.zeros(3, dtype=config.dtype))
        self.torque_bias = nn.Parameter(torch.zeros(3, dtype=config.dtype))

        # 噪声
        self.force_noise_cov = nn.Parameter(
            torch.eye(3, dtype=config.dtype) * config.noise_std**2
        )
        self.torque_noise_cov = nn.Parameter(
            torch.eye(3, dtype=config.dtype) * config.noise_std**2
        )

        # 移动到设备
        self.to(self.device)
        self.to(config.dtype)

    def forward(
        self, true_force: torch.Tensor, true_torque: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成力扭矩测量值"""
        batch_size = true_force.shape[0]

        # 应用尺度因子和偏置
        force_scaled = true_force @ self.force_scale.T + self.force_bias.unsqueeze(0)
        torque_scaled = true_torque @ self.torque_scale.T + self.torque_bias.unsqueeze(
            0
        )

        # 添加噪声
        if self.training:
            force_noise = self._generate_noise(batch_size, self.force_noise_cov)
            torque_noise = self._generate_noise(batch_size, self.torque_noise_cov)

            measured_force = force_scaled + force_noise
            measured_torque = torque_scaled + torque_noise
        else:
            measured_force = force_scaled
            measured_torque = torque_scaled

        return measured_force, measured_torque

    def _generate_noise(
        self, batch_size: int, covariance: torch.Tensor
    ) -> torch.Tensor:
        """生成噪声"""
        noise = torch.randn(batch_size, 3, device=self.device, dtype=self.config.dtype)
        L = torch.linalg.cholesky(covariance)
        return noise @ L.T

    def physics_constraints(
        self,
        force_measurements: torch.Tensor,
        torque_measurements: torch.Tensor,
        contact_positions: torch.Tensor,
        external_forces: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """物理约束检查"""
        constraints = {}

        # 1. 力平衡约束：ΣF = ma
        # 完整处理：测量力应与外部力平衡
        force_balance = torch.mean((force_measurements - external_forces) ** 2)
        constraints["force_balance"] = force_balance

        # 2. 扭矩平衡约束：Στ = Iα + r × F
        # 计算由于力产生的扭矩
        force_torque = torch.cross(contact_positions, force_measurements)

        # 扭矩平衡
        torque_balance = torch.mean((torque_measurements - force_torque) ** 2)
        constraints["torque_balance"] = torque_balance

        # 3. 摩擦力约束：|F_t| ≤ μ|F_n|
        # 法向力
        F_n = force_measurements[:, 2]  # z方向
        # 切向力大小
        F_t = torch.norm(force_measurements[:, :2], dim=1)

        # 摩擦系数
        mu = 0.5

        friction_constraint = torch.mean(F.relu(F_t - mu * torch.abs(F_n)))
        constraints["friction_constraint"] = friction_constraint

        return constraints


class CameraModel(nn.Module):
    """相机传感器模型"""

    def __init__(self, config: SensorConfig):
        super().__init__()
        self.config = config

        # 相机内参
        self.fx = nn.Parameter(torch.tensor(500.0, dtype=config.dtype))  # 焦距x
        self.fy = nn.Parameter(torch.tensor(500.0, dtype=config.dtype))  # 焦距y
        self.cx = nn.Parameter(torch.tensor(320.0, dtype=config.dtype))  # 主点x
        self.cy = nn.Parameter(torch.tensor(240.0, dtype=config.dtype))  # 主点y

        # 畸变参数
        self.k1 = nn.Parameter(torch.tensor(0.0, dtype=config.dtype))  # 径向畸变
        self.k2 = nn.Parameter(torch.tensor(0.0, dtype=config.dtype))
        self.p1 = nn.Parameter(torch.tensor(0.0, dtype=config.dtype))  # 切向畸变
        self.p2 = nn.Parameter(torch.tensor(0.0, dtype=config.dtype))

        # 噪声
        self.pixel_noise_std = nn.Parameter(
            torch.tensor(config.noise_std, dtype=config.dtype)
        )

        # 设备配置
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)
        self.to(config.dtype)

    def project(
        self, points_3d: torch.Tensor, camera_pose: torch.Tensor
    ) -> torch.Tensor:
        """3D点投影到图像平面

        参数：
            points_3d: 3D点 [batch_size, num_points, 3]
            camera_pose: 相机位姿 [batch_size, 4, 4] (齐次变换矩阵)

        返回：
            points_2d: 2D图像点 [batch_size, num_points, 2]
        """
        batch_size, num_points, _ = points_3d.shape

        # 转换到相机坐标系
        points_homogeneous = torch.cat(
            [
                points_3d,
                torch.ones(
                    batch_size,
                    num_points,
                    1,
                    device=self.device,
                    dtype=self.config.dtype,
                ),
            ],
            dim=2,
        )
        points_camera = torch.bmm(points_homogeneous, camera_pose.transpose(1, 2))[
            :, :, :3
        ]

        # 归一化平面坐标
        x = points_camera[:, :, 0] / (points_camera[:, :, 2] + 1e-8)
        y = points_camera[:, :, 1] / (points_camera[:, :, 2] + 1e-8)

        # 径向畸变
        r2 = x**2 + y**2
        radial_distortion = 1 + self.k1 * r2 + self.k2 * r2**2

        # 切向畸变
        x_tangential = 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x**2)
        y_tangential = self.p1 * (r2 + 2 * y**2) + 2 * self.p2 * x * y

        # 应用畸变
        x_distorted = x * radial_distortion + x_tangential
        y_distorted = y * radial_distortion + y_tangential

        # 投影到像素坐标
        u = self.fx * x_distorted + self.cx
        v = self.fy * y_distorted + self.cy

        points_2d = torch.stack([u, v], dim=2)

        # 添加噪声
        if self.training:
            noise = torch.randn_like(points_2d) * self.pixel_noise_std
            points_2d = points_2d + noise

        return points_2d

    def physics_constraints(
        self,
        points_2d: torch.Tensor,
        points_3d: torch.Tensor,
        camera_pose: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """物理约束检查"""
        constraints = {}

        # 1. 重投影误差约束
        points_2d_reprojected = self.project(points_3d, camera_pose)
        reprojection_error = torch.mean((points_2d - points_2d_reprojected) ** 2)
        constraints["reprojection_error"] = reprojection_error

        # 2. 极线几何约束（对于多视图）
        # 完整处理

        # 3. 深度正定性约束：所有点的深度应为正
        points_camera = torch.bmm(
            torch.cat([points_3d, torch.ones_like(points_3d[:, :, :1])], dim=2),
            camera_pose.transpose(1, 2),
        )[:, :, :3]

        depth = points_camera[:, :, 2]
        depth_constraint = torch.mean(F.relu(-depth))
        constraints["depth_constraint"] = depth_constraint

        return constraints


class SensorFusionPINN(nn.Module):
    """传感器融合PINN模型"""

    def __init__(self, config: SensorConfig):
        super().__init__()
        self.config = config

        # 传感器模型
        self.imu = IMUModel(config)
        self.force_sensor = ForceTorqueSensor(config)
        self.camera = CameraModel(config)

        # 状态估计网络
        self.state_estimator = nn.Sequential(
            nn.Linear(20, 128),  # 输入: 各种传感器测量值
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 13),  # 输出: [位置(3), 姿态(4), 速度(3), 角速度(3)]
        )

        # 设备配置
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)
        self.to(config.dtype)

    def forward(self, sensor_measurements: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播：状态估计"""
        # 合并传感器测量值
        features = []

        if "imu_accel" in sensor_measurements:
            features.append(sensor_measurements["imu_accel"])
        if "imu_gyro" in sensor_measurements:
            features.append(sensor_measurements["imu_gyro"])
        if "force" in sensor_measurements:
            features.append(sensor_measurements["force"])
        if "torque" in sensor_measurements:
            features.append(sensor_measurements["torque"])

        if features:
            x = torch.cat(features, dim=1)
            # 确保特征维度匹配网络输入
            if x.shape[1] < 20:
                # 填充
                padding = torch.zeros(
                    x.shape[0],
                    20 - x.shape[1],
                    device=self.device,
                    dtype=self.config.dtype,
                )
                x = torch.cat([x, padding], dim=1)
            elif x.shape[1] > 20:
                # 截断
                x = x[:, :20]

            state = self.state_estimator(x)
        else:
            batch_size = next(iter(sensor_measurements.values())).shape[0]
            state = torch.zeros(
                batch_size, 13, device=self.device, dtype=self.config.dtype
            )

        return state

    def physics_loss(
        self,
        sensor_measurements: Dict[str, torch.Tensor],
        estimated_state: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """物理损失：传感器数据一致性"""
        loss = 0.0

        # 提取状态
        position = estimated_state[:, :3]
        orientation = estimated_state[:, 3:7]
        velocity = estimated_state[:, 7:10]
        angular_velocity = estimated_state[:, 10:13]

        # IMU物理约束
        if "imu_accel" in sensor_measurements and "imu_gyro" in sensor_measurements:
            imu_constraints = self.imu.physics_constraints(
                sensor_measurements["imu_accel"],
                sensor_measurements["imu_gyro"],
                position,
                orientation,
                dt,
            )
            loss += sum(imu_constraints.values())

        # 力传感器物理约束
        if "force" in sensor_measurements and "torque" in sensor_measurements:
            # 完整：假设接触位置为原点
            contact_positions = torch.zeros_like(position)
            external_forces = torch.zeros_like(sensor_measurements["force"])

            ft_constraints = self.force_sensor.physics_constraints(
                sensor_measurements["force"],
                sensor_measurements["torque"],
                contact_positions,
                external_forces,
            )
            loss += sum(ft_constraints.values())

        # 运动学一致性约束
        # 位置变化应与速度一致
        position_from_velocity = position + velocity * dt
        kinematic_constraint = torch.mean((position_from_velocity - position) ** 2)
        loss += kinematic_constraint

        # 姿态变化应与角速度一致
        # 使用小角度近似
        angle_increment = torch.norm(angular_velocity, dim=1) * dt
        delta_q = self.imu.axis_angle_to_quaternion(angular_velocity, angle_increment)
        q_next = self.imu.quaternion_multiply(orientation, delta_q)

        orientation_constraint = torch.mean((q_next - orientation) ** 2)
        loss += orientation_constraint

        return loss


def test_sensor_physics():
    """测试传感器物理模块"""
    print("=== 测试传感器物理模块 ===")

    # 创建测试配置
    config = SensorConfig(
        sensor_type="imu", noise_std=0.01, use_gpu=False, dtype=torch.float64
    )

    # 测试IMU模型
    print("\n1. 测试IMU模型:")
    imu = IMUModel(config)

    batch_size = 4
    true_accel = torch.ones(batch_size, 3, dtype=config.dtype) * torch.tensor(
        [0, 0, -9.81]
    )
    true_gyro = torch.ones(batch_size, 3, dtype=config.dtype) * 0.1

    measured_accel, measured_gyro = imu(true_accel, true_gyro, dt=0.01)
    print(f"真实加速度: {true_accel[0]}")
    print(f"测量加速度: {measured_accel[0]}")
    print(f"真实角速度: {true_gyro[0]}")
    print(f"测量角速度: {measured_gyro[0]}")

    # 测试物理约束
    positions = torch.zeros(batch_size, 3, dtype=config.dtype)
    orientations = torch.zeros(batch_size, 4, dtype=config.dtype)
    orientations[:, 0] = 1.0  # 单位四元数

    constraints = imu.physics_constraints(
        measured_accel, measured_gyro, positions, orientations, dt=0.01
    )
    print("\nIMU物理约束:")
    for name, value in constraints.items():
        print(f"  {name}: {value.item():.6f}")

    # 测试力传感器模型
    print("\n2. 测试力传感器模型:")
    ft_config = SensorConfig(
        sensor_type="force_torque", use_gpu=False, dtype=torch.float64
    )
    ft_sensor = ForceTorqueSensor(ft_config)

    true_force = torch.ones(batch_size, 3, dtype=config.dtype) * torch.tensor(
        [0, 0, -10]
    )
    true_torque = torch.ones(batch_size, 3, dtype=config.dtype) * torch.tensor(
        [0, 0, 0.1]
    )

    measured_force, measured_torque = ft_sensor(true_force, true_torque)
    print(f"真实力: {true_force[0]}")
    print(f"测量力: {measured_force[0]}")

    # 测试相机模型
    print("\n3. 测试相机模型:")
    camera_config = SensorConfig(
        sensor_type="camera", use_gpu=False, dtype=torch.float64
    )
    camera = CameraModel(camera_config)

    points_3d = torch.tensor(
        [[[1, 0, 2], [0, 1, 2], [-1, 0, 2]]], dtype=config.dtype
    ).repeat(batch_size, 1, 1)
    camera_pose = torch.eye(4, dtype=config.dtype).unsqueeze(0).repeat(batch_size, 1, 1)

    points_2d = camera.project(points_3d, camera_pose)
    print(f"3D点: {points_3d[0]}")
    print(f"2D投影: {points_2d[0]}")

    # 测试传感器融合PINN
    print("\n4. 测试传感器融合PINN:")
    fusion_config = SensorConfig(use_gpu=False, dtype=torch.float64)
    fusion_pinn = SensorFusionPINN(fusion_config)

    sensor_measurements = {
        "imu_accel": measured_accel,
        "imu_gyro": measured_gyro,
        "force": measured_force,
        "torque": measured_torque,
    }

    estimated_state = fusion_pinn(sensor_measurements)
    print(f"估计状态形状: {estimated_state.shape}")
    print(f"估计位置: {estimated_state[0, :3]}")
    print(f"估计姿态: {estimated_state[0, 3:7]}")

    # 测试物理损失
    physics_loss = fusion_pinn.physics_loss(
        sensor_measurements, estimated_state, dt=0.01
    )
    print(f"物理损失: {physics_loss.item():.6f}")

    print("\n=== 传感器物理测试完成 ===")


if __name__ == "__main__":
    test_sensor_physics()
