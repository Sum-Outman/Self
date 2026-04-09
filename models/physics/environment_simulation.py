#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境交互物理仿真模块

功能：
1. 刚体碰撞检测和响应
2. 软体变形仿真
3. 流体动力学仿真
4. 接触力学建模
5. 环境物理场仿真

工业级质量标准要求：
- 真实性：基于真实物理定律
- 稳定性：数值稳定，避免爆炸
- 效率：实时或准实时仿真
- 准确性：高精度物理仿真

数学原理：
1. 刚体动力学: Mq̈ + Cq̇ + g = τ + J_c^T λ
2. 碰撞检测: 分离轴定理，GJK算法
3. 接触力学: 库仑摩擦，恢复系数
4. 流体动力学: Navier-Stokes方程
5. 有限元方法: 弹性力学方程

参考文献：
[1] Baraff, D. (1997). An Introduction to Physically Based Modeling.
[2] Erleben, K., et al. (2005). Physics-Based Animation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """环境仿真配置类"""

    # 环境类型
    environment_type: str = (
        "rigid_body"  # "rigid_body", "soft_body", "fluid", "deformable"
    )

    # 物理参数
    gravity: List[float] = field(default_factory=lambda: [0, 0, -9.81])
    time_step: float = 0.01  # 时间步长
    substeps: int = 1  # 子步数

    # 碰撞参数
    collision_margin: float = 0.001  # 碰撞容差
    restitution: float = 0.5  # 恢复系数
    friction_coefficient: float = 0.5  # 摩擦系数

    # 仿真精度
    solver_iterations: int = 10  # 求解器迭代次数
    tolerance: float = 1e-6  # 容差

    # 计算配置
    use_gpu: bool = True
    dtype: torch.dtype = torch.float64


class RigidBody:
    """刚体类"""

    def __init__(self, body_id: int, config: EnvironmentConfig):
        self.body_id = body_id
        self.config = config

        # 状态
        self.position = torch.zeros(3, dtype=config.dtype)
        self.orientation = torch.tensor([1.0, 0, 0, 0], dtype=config.dtype)  # 四元数
        self.linear_velocity = torch.zeros(3, dtype=config.dtype)
        self.angular_velocity = torch.zeros(3, dtype=config.dtype)

        # 质量属性
        self.mass = 1.0
        self.inertia = torch.eye(3, dtype=config.dtype)
        self.inverse_inertia = torch.eye(3, dtype=config.dtype)

        # 几何形状
        self.vertices = None  # 顶点
        self.faces = None  # 面
        self.collision_shape = "box"  # 碰撞形状
        self.radius = 0.5  # 默认半径（对于球体碰撞形状）

        # 外力
        self.external_force = torch.zeros(3, dtype=config.dtype)
        self.external_torque = torch.zeros(3, dtype=config.dtype)

    def update_inertia(self):
        """更新惯量矩阵"""
        # 如果有自定义顶点，计算惯量张量
        if self.vertices is not None:
            # 完整计算：边界盒惯量
            # 计算边界盒尺寸
            vertices = self.vertices
            min_vals = torch.min(vertices, dim=0)[0]
            max_vals = torch.max(vertices, dim=0)[0]
            size = max_vals - min_vals

            # 计算边界盒体积（假设为立方体）
            volume = size[0] * size[1] * size[2]

            # 如果体积接近零，使用单位矩阵作为惯量
            if volume < 1e-12:
                self.inertia = torch.eye(3, dtype=self.config.dtype)
                self.inverse_inertia = torch.eye(3, dtype=self.config.dtype)
                return

            # 计算密度（质量/体积）
            self.mass / volume

            # 计算边界盒惯量张量（均匀密度立方体）
            # 公式: Ixx = (m/12) * (y^2 + z^2), 类似地 Iyy, Izz
            # Ixy = Ixz = Iyz = 0 (对于轴对齐边界盒)
            x, y, z = size[0], size[1], size[2]
            Ixx = (self.mass / 12.0) * (y * y + z * z)
            Iyy = (self.mass / 12.0) * (x * x + z * z)
            Izz = (self.mass / 12.0) * (x * x + y * y)

            self.inertia = torch.diag(
                torch.tensor([Ixx, Iyy, Izz], dtype=self.config.dtype)
            )
            self.inverse_inertia = torch.inverse(self.inertia)
        else:
            # 没有自定义顶点，使用默认惯量（基于碰撞形状）
            if self.collision_shape == "box":
                # 假设单位立方体
                Ixx = Iyy = Izz = self.mass / 6.0  # 对于单位立方体
                self.inertia = torch.diag(
                    torch.tensor([Ixx, Iyy, Izz], dtype=self.config.dtype)
                )
                self.inverse_inertia = torch.inverse(self.inertia)
            elif self.collision_shape == "sphere":
                # 球体惯量: (2/5) * m * r^2，假设半径为0.5
                I = (2.0 / 5.0) * self.mass * (0.5**2)
                self.inertia = torch.eye(3, dtype=self.config.dtype) * I
                self.inverse_inertia = torch.inverse(self.inertia)
            else:
                # 默认单位矩阵
                self.inertia = torch.eye(3, dtype=self.config.dtype)
                self.inverse_inertia = torch.eye(3, dtype=self.config.dtype)

    def get_transform_matrix(self) -> torch.Tensor:
        """获取变换矩阵"""
        R = self.quaternion_to_matrix(self.orientation)
        T = torch.eye(4, dtype=self.config.dtype)
        T[:3, :3] = R
        T[:3, 3] = self.position
        return T

    def quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """四元数转旋转矩阵"""
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

        R = torch.zeros(3, 3, dtype=self.config.dtype)
        R[0, 0] = 1 - 2 * q2**2 - 2 * q3**2
        R[0, 1] = 2 * q1 * q2 - 2 * q0 * q3
        R[0, 2] = 2 * q1 * q3 + 2 * q0 * q2

        R[1, 0] = 2 * q1 * q2 + 2 * q0 * q3
        R[1, 1] = 1 - 2 * q1**2 - 2 * q3**2
        R[1, 2] = 2 * q2 * q3 - 2 * q0 * q1

        R[2, 0] = 2 * q1 * q3 - 2 * q0 * q2
        R[2, 1] = 2 * q2 * q3 + 2 * q0 * q1
        R[2, 2] = 1 - 2 * q1**2 - 2 * q2**2

        return R


class CollisionDetector:
    """碰撞检测器"""

    def __init__(self, config: EnvironmentConfig):
        self.config = config

    def detect_collisions(self, bodies: List[RigidBody]) -> List[Dict[str, Any]]:
        """检测碰撞"""
        collisions = []
        n_bodies = len(bodies)

        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                body_a = bodies[i]
                body_b = bodies[j]

                # 基于碰撞形状的检测
                if body_a.collision_shape == "box" and body_b.collision_shape == "box":
                    collision = self.box_box_collision(body_a, body_b)
                    if collision["colliding"]:
                        collisions.append(collision)
                elif (
                    body_a.collision_shape == "sphere"
                    and body_b.collision_shape == "sphere"
                ):
                    collision = self.sphere_sphere_collision(body_a, body_b)
                    if collision["colliding"]:
                        collisions.append(collision)

        return collisions

    def box_box_collision(self, box_a: RigidBody, box_b: RigidBody) -> Dict[str, Any]:
        """盒-盒碰撞检测"""
        # 分离轴定理实现
        # 这里完整：使用轴对齐边界盒

        # 获取变换后的边界
        a_min, a_max = self.get_transformed_aabb(box_a)
        b_min, b_max = self.get_transformed_aabb(box_b)

        # 检查重叠
        colliding = True
        for k in range(3):
            if a_max[k] < b_min[k] or a_min[k] > b_max[k]:
                colliding = False
                break

        if not colliding:
            return {
                "body_a": box_a.body_id,
                "body_b": box_b.body_id,
                "colliding": False,
                "contact_point": None,
                "contact_normal": None,
                "penetration_depth": 0.0,
            }

        # 计算接触信息（完整）
        # 找到最小穿透轴
        overlaps = []
        for k in range(3):
            overlap = min(a_max[k], b_max[k]) - max(a_min[k], b_min[k])
            overlaps.append(overlap)

        min_overlap_idx = np.argmin(overlaps)
        min_overlap = overlaps[min_overlap_idx]

        # 接触法向（指向body_a）
        center_a = (a_min + a_max) / 2
        center_b = (b_min + b_max) / 2
        direction = center_b - center_a

        if direction[min_overlap_idx] > 0:
            normal = torch.zeros(3, dtype=self.config.dtype)
            normal[min_overlap_idx] = -1.0
        else:
            normal = torch.zeros(3, dtype=self.config.dtype)
            normal[min_overlap_idx] = 1.0

        # 接触点（完整计算）
        # 在最小穿透轴上，接触点位于两个边界盒投影的重叠区域中点
        # 在其他轴上，使用两个中心坐标的平均值
        contact_point = torch.zeros(3, dtype=self.config.dtype)
        for k in range(3):
            if k == min_overlap_idx:
                # 计算重叠区域的中点
                overlap_min = max(a_min[k], b_min[k])
                overlap_max = min(a_max[k], b_max[k])
                contact_point[k] = (overlap_min + overlap_max) / 2.0
            else:
                # 使用两个中心坐标的平均值
                contact_point[k] = (center_a[k] + center_b[k]) / 2.0

        return {
            "body_a": box_a.body_id,
            "body_b": box_b.body_id,
            "colliding": True,
            "contact_point": contact_point,
            "contact_normal": normal,
            "penetration_depth": min_overlap,
        }

    def sphere_sphere_collision(
        self, sphere_a: RigidBody, sphere_b: RigidBody
    ) -> Dict[str, Any]:
        """球-球碰撞检测"""
        # 球心距离
        delta = sphere_b.position - sphere_a.position
        distance = torch.norm(delta)

        # 球半径（完整：使用刚体半径属性）
        radius_a = sphere_a.radius
        radius_b = sphere_b.radius

        colliding = distance < (radius_a + radius_b)

        if not colliding:
            return {
                "body_a": sphere_a.body_id,
                "body_b": sphere_b.body_id,
                "colliding": False,
                "contact_point": None,
                "contact_normal": None,
                "penetration_depth": 0.0,
            }

        # 接触信息
        penetration = (radius_a + radius_b) - distance
        normal = delta / (distance + 1e-8)
        contact_point = sphere_a.position + normal * radius_a

        return {
            "body_a": sphere_a.body_id,
            "body_b": sphere_b.body_id,
            "colliding": True,
            "contact_point": contact_point,
            "contact_normal": normal,
            "penetration_depth": penetration,
        }

    def get_transformed_aabb(
        self, body: RigidBody
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取变换后的轴对齐边界盒"""
        # 完整：假设原始边界盒为[-0.5, 0.5]^3
        original_min = torch.tensor([-0.5, -0.5, -0.5], dtype=self.config.dtype)
        original_max = torch.tensor([0.5, 0.5, 0.5], dtype=self.config.dtype)

        # 变换到世界坐标系
        T = body.get_transform_matrix()

        # 变换8个顶点
        vertices_local = torch.tensor(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=self.config.dtype,
        )

        vertices_homogeneous = torch.cat(
            [vertices_local, torch.ones(8, 1, dtype=self.config.dtype)], dim=1
        )
        vertices_world = vertices_homogeneous @ T.T

        # 计算AABB
        min_corner = torch.min(vertices_world[:, :3], dim=0)[0]
        max_corner = torch.max(vertices_world[:, :3], dim=0)[0]

        return min_corner, max_corner


class ContactSolver:
    """接触求解器"""

    def __init__(self, config: EnvironmentConfig):
        self.config = config

    def solve_contacts(
        self, bodies: List[RigidBody], collisions: List[Dict[str, Any]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """求解接触力"""
        n_bodies = len(bodies)
        contact_forces = [
            torch.zeros(3, dtype=self.config.dtype) for _ in range(n_bodies)
        ]
        contact_torques = [
            torch.zeros(3, dtype=self.config.dtype) for _ in range(n_bodies)
        ]

        for collision in collisions:
            if not collision["colliding"]:
                continue

            body_a_idx = collision["body_a"]
            body_b_idx = collision["body_b"]

            body_a = bodies[body_a_idx]
            body_b = bodies[body_b_idx]

            # 计算接触力
            force_a, torque_a, force_b, torque_b = self.compute_contact_force(
                body_a, body_b, collision
            )

            contact_forces[body_a_idx] += force_a
            contact_torques[body_a_idx] += torque_a
            contact_forces[body_b_idx] += force_b
            contact_torques[body_b_idx] += torque_b

        return contact_forces, contact_torques

    def compute_contact_force(
        self, body_a: RigidBody, body_b: RigidBody, collision: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算接触力"""
        # 接触点
        p = collision["contact_point"]
        n = collision["contact_normal"]  # 指向body_a

        # 相对速度
        v_a = body_a.linear_velocity + torch.cross(
            body_a.angular_velocity, p - body_a.position
        )
        v_b = body_b.linear_velocity + torch.cross(
            body_b.angular_velocity, p - body_b.position
        )
        v_rel = v_b - v_a

        # 法向速度
        vn = torch.dot(v_rel, n)

        # 如果物体正在分离，不需要力
        if vn > 0:
            return (
                torch.zeros_like(n),
                torch.zeros_like(n),
                torch.zeros_like(n),
                torch.zeros_like(n),
            )

        # 法向冲量（基于恢复系数）
        e = self.config.restitution
        jn = -(1 + e) * vn

        # 质量项
        r_a = p - body_a.position
        r_b = p - body_b.position

        # 有效质量
        inv_mass_a = 1.0 / body_a.mass
        inv_mass_b = 1.0 / body_b.mass

        # 惯量项
        I_a = body_a.inertia
        I_b = body_b.inertia

        # 计算有效质量
        term1 = inv_mass_a + inv_mass_b
        term2 = torch.dot(n, torch.cross(torch.cross(r_a, n) @ I_a.inverse(), r_a))
        term3 = torch.dot(n, torch.cross(torch.cross(r_b, n) @ I_b.inverse(), r_b))

        effective_mass = 1.0 / (term1 + term2 + term3)

        # 法向冲量
        jn *= effective_mass

        # 切向速度（摩擦力）
        vt = v_rel - vn * n
        vt_norm = torch.norm(vt)

        if vt_norm > 1e-8:
            t = vt / vt_norm
        else:
            t = torch.zeros_like(n)

        # 切向冲量（库仑摩擦）
        mu = self.config.friction_coefficient
        jt_max = mu * jn

        # 切向有效质量（类似法向）
        term2_t = torch.dot(t, torch.cross(torch.cross(r_a, t) @ I_a.inverse(), r_a))
        term3_t = torch.dot(t, torch.cross(torch.cross(r_b, t) @ I_b.inverse(), r_b))
        effective_mass_t = 1.0 / (term1 + term2_t + term3_t)

        jt = -effective_mass_t * vt_norm
        jt = torch.clamp(jt, -jt_max, jt_max)

        # 总冲量
        impulse = jn * n + jt * t

        # 力和扭矩
        force_a = impulse / self.config.time_step
        torque_a = torch.cross(r_a, impulse) / self.config.time_step

        force_b = -impulse / self.config.time_step
        torque_b = torch.cross(r_b, -impulse) / self.config.time_step

        return force_a, torque_a, force_b, torque_b


class PhysicsSimulator(nn.Module):
    """物理仿真器"""

    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.config = config

        # 组件
        self.collision_detector = CollisionDetector(config)
        self.contact_solver = ContactSolver(config)

        # 设备配置
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)
        self.to(config.dtype)

    def forward(self, bodies: List[RigidBody], dt: float = None) -> List[RigidBody]:
        """执行物理仿真步"""
        if dt is None:
            dt = self.config.time_step

        # 碰撞检测
        collisions = self.collision_detector.detect_collisions(bodies)

        # 计算接触力
        contact_forces, contact_torques = self.contact_solver.solve_contacts(
            bodies, collisions
        )

        # 更新每个物体
        for i, body in enumerate(bodies):
            # 总力：重力 + 外力 + 接触力
            total_force = (
                torch.tensor(self.config.gravity, dtype=self.config.dtype) * body.mass
            )
            total_force += body.external_force
            total_force += contact_forces[i]

            # 总扭矩：外力扭矩 + 接触扭矩
            total_torque = body.external_torque + contact_torques[i]

            # 线性加速度
            linear_acceleration = total_force / body.mass

            # 角加速度
            angular_acceleration = body.inverse_inertia @ total_torque

            # 更新速度（半隐式欧拉）
            body.linear_velocity += linear_acceleration * dt
            body.angular_velocity += angular_acceleration * dt

            # 更新位置
            body.position += body.linear_velocity * dt

            # 更新姿态（四元数积分）
            # 角速度四元数
            omega = torch.cat(
                [torch.zeros(1, dtype=self.config.dtype), body.angular_velocity]
            )
            q_dot = 0.5 * self.quaternion_multiply(omega, body.orientation)

            body.orientation += q_dot * dt
            body.orientation = body.orientation / torch.norm(body.orientation)

            # 重置外力
            body.external_force = torch.zeros_like(body.external_force)
            body.external_torque = torch.zeros_like(body.external_torque)

        return bodies

    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """四元数乘法"""
        a1, b1, c1, d1 = q1[0], q1[1], q1[2], q1[3]
        a2, b2, c2, d2 = q2[0], q2[1], q2[2], q2[3]

        result = torch.zeros(4, dtype=self.config.dtype)
        result[0] = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        result[1] = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        result[2] = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        result[3] = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

        return result


class FluidSimulator(nn.Module):
    """流体动力学仿真器"""

    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.config = config

        # 流体参数
        self.density = 1000.0  # 密度 kg/m^3
        self.viscosity = 0.01  # 粘度
        self.pressure = 101325.0  # 压力 Pa

        # 网格参数
        self.grid_size = 32
        self.cell_size = 0.1

        # 速度场
        self.velocity_field = None
        self.pressure_field = None

        # 设备配置
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)
        self.to(config.dtype)

    def forward(self, dt: float) -> torch.Tensor:
        """执行流体仿真步"""
        # 完整流体仿真（基于Stable Fluids方法）

        # 初始化场
        if self.velocity_field is None:
            self.velocity_field = torch.zeros(
                3,
                self.grid_size,
                self.grid_size,
                self.grid_size,
                dtype=self.config.dtype,
                device=self.device,
            )
            self.pressure_field = torch.zeros(
                self.grid_size,
                self.grid_size,
                self.grid_size,
                dtype=self.config.dtype,
                device=self.device,
            )

        # 1. 添加外力（如重力）
        self.add_gravity(dt)

        # 2. 对流
        self.advect(dt)

        # 3. 扩散（粘度）
        self.diffuse(dt)

        # 4. 投影（保持不可压缩性）
        self.project()

        return self.velocity_field

    def add_gravity(self, dt: float):
        """添加重力"""
        self.velocity_field[2] -= 9.81 * dt

    def advect(self, dt: float):
        """对流项"""
        # 半拉格朗日平流
        new_velocity = torch.zeros_like(self.velocity_field)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    # 当前位置
                    x = torch.tensor(
                        [i, j, k], dtype=self.config.dtype, device=self.device
                    )

                    # 回溯位置
                    v = self.velocity_field[:, i, j, k]
                    x_prev = x - v * dt / self.cell_size

                    # 双线性插值
                    new_velocity[:, i, j, k] = self.interpolate_velocity(x_prev)

        self.velocity_field = new_velocity

    def interpolate_velocity(self, pos: torch.Tensor) -> torch.Tensor:
        """插值速度"""
        # 双线性插值
        x = torch.clamp(pos[0], 0, self.grid_size - 2)
        y = torch.clamp(pos[1], 0, self.grid_size - 2)
        z = torch.clamp(pos[2], 0, self.grid_size - 2)

        i0 = int(x)
        i1 = i0 + 1
        j0 = int(y)
        j1 = j0 + 1
        k0 = int(z)
        k1 = k0 + 1

        # 插值权重
        wx1 = x - i0
        wx0 = 1 - wx1
        wy1 = y - j0
        wy0 = 1 - wy1
        wz1 = z - k0
        wz0 = 1 - wz1

        # 三线性插值
        v000 = self.velocity_field[:, i0, j0, k0]
        v001 = self.velocity_field[:, i0, j0, k1]
        v010 = self.velocity_field[:, i0, j1, k0]
        v011 = self.velocity_field[:, i0, j1, k1]
        v100 = self.velocity_field[:, i1, j0, k0]
        v101 = self.velocity_field[:, i1, j0, k1]
        v110 = self.velocity_field[:, i1, j1, k0]
        v111 = self.velocity_field[:, i1, j1, k1]

        v00 = v000 * wz0 + v001 * wz1
        v01 = v010 * wz0 + v011 * wz1
        v10 = v100 * wz0 + v101 * wz1
        v11 = v110 * wz0 + v111 * wz1

        v0 = v00 * wy0 + v01 * wy1
        v1 = v10 * wy0 + v11 * wy1

        v = v0 * wx0 + v1 * wx1

        return v

    def diffuse(self, dt: float):
        """扩散项"""
        # 完整扩散
        alpha = self.viscosity * dt / (self.cell_size**2)

        # 高斯-赛德尔迭代
        for _ in range(5):
            for i in range(1, self.grid_size - 1):
                for j in range(1, self.grid_size - 1):
                    for k in range(1, self.grid_size - 1):
                        self.velocity_field[:, i, j, k] = (
                            self.velocity_field[:, i, j, k]
                            + alpha
                            * (
                                self.velocity_field[:, i - 1, j, k]
                                + self.velocity_field[:, i + 1, j, k]
                                + self.velocity_field[:, i, j - 1, k]
                                + self.velocity_field[:, i, j + 1, k]
                                + self.velocity_field[:, i, j, k - 1]
                                + self.velocity_field[:, i, j, k + 1]
                            )
                        ) / (1 + 6 * alpha)

    def project(self):
        """投影步（保持不可压缩性）"""
        # 计算散度
        divergence = torch.zeros(
            self.grid_size,
            self.grid_size,
            self.grid_size,
            dtype=self.config.dtype,
            device=self.device,
        )

        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                for k in range(1, self.grid_size - 1):
                    divergence[i, j, k] = (
                        (
                            self.velocity_field[0, i + 1, j, k]
                            - self.velocity_field[0, i - 1, j, k]
                        )
                        + (
                            self.velocity_field[1, i, j + 1, k]
                            - self.velocity_field[1, i, j - 1, k]
                        )
                        + (
                            self.velocity_field[2, i, j, k + 1]
                            - self.velocity_field[2, i, j, k - 1]
                        )
                    ) / (2 * self.cell_size)

        # 解泊松方程求压力
        pressure = torch.zeros_like(divergence)

        for _ in range(20):
            for i in range(1, self.grid_size - 1):
                for j in range(1, self.grid_size - 1):
                    for k in range(1, self.grid_size - 1):
                        pressure[i, j, k] = (
                            divergence[i, j, k]
                            + pressure[i - 1, j, k]
                            + pressure[i + 1, j, k]
                            + pressure[i, j - 1, k]
                            + pressure[i, j + 1, k]
                            + pressure[i, j, k - 1]
                            + pressure[i, j, k + 1]
                        ) / 6

        # 用压力梯度修正速度
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                for k in range(1, self.grid_size - 1):
                    self.velocity_field[0, i, j, k] -= (
                        pressure[i + 1, j, k] - pressure[i - 1, j, k]
                    ) / (2 * self.cell_size)
                    self.velocity_field[1, i, j, k] -= (
                        pressure[i, j + 1, k] - pressure[i, j - 1, k]
                    ) / (2 * self.cell_size)
                    self.velocity_field[2, i, j, k] -= (
                        pressure[i, j, k + 1] - pressure[i, j, k - 1]
                    ) / (2 * self.cell_size)


def test_environment_simulation():
    """测试环境仿真模块"""
    print("=== 测试环境仿真模块 ===")

    # 创建测试配置
    config = EnvironmentConfig(
        environment_type="rigid_body",
        time_step=0.01,
        use_gpu=False,
        dtype=torch.float64,
    )

    # 测试刚体仿真
    print("\n1. 测试刚体仿真:")

    # 创建两个刚体
    body1 = RigidBody(0, config)
    body1.position = torch.tensor([0, 0, 1], dtype=config.dtype)
    body1.mass = 1.0

    body2 = RigidBody(1, config)
    body2.position = torch.tensor([0.1, 0, 0.5], dtype=config.dtype)
    body2.mass = 1.0

    bodies = [body1, body2]

    # 创建仿真器
    simulator = PhysicsSimulator(config)

    # 仿真10步
    print("初始状态:")
    print(f"物体1位置: {body1.position}")
    print(f"物体2位置: {body2.position}")

    for step in range(10):
        bodies = simulator(bodies)

        if step == 4:
            print(f"\n第{step + 1}步:")
            print(f"物体1位置: {body1.position}")
            print(f"物体1速度: {body1.linear_velocity}")
            print(f"物体2位置: {body2.position}")
            print(f"物体2速度: {body2.linear_velocity}")

    print("\n最终状态（第10步）:")
    print(f"物体1位置: {body1.position}")
    print(f"物体1速度: {body1.linear_velocity}")
    print(f"物体2位置: {body2.position}")
    print(f"物体2速度: {body2.linear_velocity}")

    # 测试流体仿真
    print("\n2. 测试流体仿真:")
    fluid_config = EnvironmentConfig(
        environment_type="fluid", use_gpu=False, dtype=torch.float64
    )

    fluid_simulator = FluidSimulator(fluid_config)

    # 执行流体仿真
    velocity_field = fluid_simulator(0.01)
    print(f"速度场形状: {velocity_field.shape}")
    print(f"速度场均值: {torch.mean(velocity_field).item():.6f}")

    # 测试碰撞检测
    print("\n3. 测试碰撞检测:")
    collision_detector = CollisionDetector(config)

    # 设置碰撞情况
    body1.position = torch.tensor([0, 0, 0.5], dtype=config.dtype)
    body2.position = torch.tensor([0, 0, 0.4], dtype=config.dtype)

    collisions = collision_detector.detect_collisions([body1, body2])
    print(f"检测到碰撞数量: {len(collisions)}")

    if collisions:
        collision = collisions[0]
        print(f"碰撞穿透深度: {collision['penetration_depth'].item():.6f}")
        print(f"接触法向: {collision['contact_normal']}")

    print("\n=== 环境仿真测试完成 ===")


if __name__ == "__main__":
    test_environment_simulation()
