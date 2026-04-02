#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四元数核心库 - Self AGI 系统四元数全面引入实施方案核心模块

功能：
1. 四元数基础数据结构（NumPy和PyTorch双版本）
2. 四元数基本运算（乘法、共轭、归一化、点积、SLERP等）
3. 四元数转换函数（欧拉角、旋转矩阵、轴角表示）
4. 四元数距离度量和损失函数
5. 四元数神经网络层基础组件

工业级质量标准要求：
- 数值稳定性：双精度计算，避免奇异性，万向节锁消除
- 计算效率：GPU加速支持，向量化运算
- 内存优化：连续内存布局，量化支持
- 兼容性：NumPy和PyTorch双版本，向后兼容现有系统

数学原理：
1. 四元数表示：q = w + xi + yj + zk, 其中w为实部，(x,y,z)为虚部
2. 旋转表示：单位四元数表示SO(3)旋转群
3. 双倍覆盖：q和-q表示同一旋转
4. 球面线性插值（SLERP）：保持恒定角速度的旋转插值

参考标准：
[1] Ken Shoemake (1985). Animating Rotation with Quaternion Curves.
[2] Joan Solà (2017). Quaternion kinematics for the error-state Kalman filter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple, Optional, Any
import math
import warnings


class Quaternion:
    """四元数类（NumPy版本）"""
    
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        初始化四元数
        
        参数:
            w: 实部
            x, y, z: 虚部
        """
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    @classmethod
    def from_vector(cls, vec: np.ndarray):
        """从向量创建四元数 [w, x, y, z]"""
        if vec.shape[-1] != 4:
            raise ValueError(f"四元数向量必须是4维，但形状为{vec.shape}")
        if vec.ndim == 1:
            return cls(vec[0], vec[1], vec[2], vec[3])
        else:
            # 批量创建
            return np.apply_along_axis(lambda v: cls(v[0], v[1], v[2], v[3]), -1, vec)
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float, order: str = 'zyx'):
        """
        从欧拉角创建四元数
        
        参数:
            roll, pitch, yaw: 欧拉角（弧度）
            order: 旋转顺序（'zyx', 'xyz', etc.）
        
        返回:
            Quaternion对象
        """
        # 使用现有函数
        from backend.services.robot_configuration_service import rpy_to_quaternion
        q_vec = rpy_to_quaternion([roll, pitch, yaw])
        return cls(q_vec[3], q_vec[0], q_vec[1], q_vec[2])  # 注意顺序：现有函数返回[x,y,z,w]
    
    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float):
        """
        从轴角表示创建四元数
        
        参数:
            axis: 旋转轴（3维向量）
            angle: 旋转角度（弧度）
        """
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        sin_half = np.sin(angle / 2)
        cos_half = np.cos(angle / 2)
        
        return cls(cos_half, axis[0]*sin_half, axis[1]*sin_half, axis[2]*sin_half)
    
    @classmethod
    def from_rotation_matrix(cls, R: np.ndarray):
        """
        从旋转矩阵创建四元数
        """
        # 确保是3x3矩阵
        if R.shape != (3, 3):
            raise ValueError(f"旋转矩阵必须是3x3，但形状为{R.shape}")
        
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        q = cls(w, x, y, z)
        return q.normalize()
    
    @classmethod
    def identity(cls):
        """单位四元数（无旋转）"""
        return cls(1.0, 0.0, 0.0, 0.0)
    
    @classmethod
    def random(cls):
        """均匀随机单位四元数"""
        u1, u2, u3 = np.random.uniform(0, 1, 3)
        
        w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        
        return cls(w, x, y, z)
    
    def as_vector(self):
        """返回向量表示 [w, x, y, z]"""
        return np.array([self.w, self.x, self.y, self.z])
    
    def conjugate(self):
        """共轭四元数"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm(self):
        """四元数模长"""
        return np.sqrt(self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z)
    
    def normalize(self):
        """归一化为单位四元数"""
        n = self.norm()
        if n < 1e-8:
            return Quaternion.identity()
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def inverse(self):
        """四元数逆（对于单位四元数，逆等于共轭）"""
        n2 = self.norm() ** 2
        if n2 < 1e-8:
            return Quaternion.identity()
        return Quaternion(self.w/n2, -self.x/n2, -self.y/n2, -self.z/n2)
    
    def __mul__(self, other):
        """四元数乘法"""
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.w, self.x, self.y, self.z
            w2, x2, y2, z2 = other.w, other.x, other.y, other.z
            
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float)):
            # 标量乘法
            return Quaternion(self.w*other, self.x*other, self.y*other, self.z*other)
        else:
            raise TypeError(f"不支持的类型: {type(other)}")
    
    def __add__(self, other):
        """四元数加法"""
        if isinstance(other, Quaternion):
            return Quaternion(self.w+other.w, self.x+other.x, self.y+other.y, self.z+other.z)
        else:
            raise TypeError(f"不支持的类型: {type(other)}")
    
    def __sub__(self, other):
        """四元数减法"""
        if isinstance(other, Quaternion):
            return Quaternion(self.w-other.w, self.x-other.x, self.y-other.y, self.z-other.z)
        else:
            raise TypeError(f"不支持的类型: {type(other)}")
    
    def __neg__(self):
        """四元数取负"""
        return Quaternion(-self.w, -self.x, -self.y, -self.z)
    
    def dot(self, other):
        """四元数点积"""
        if isinstance(other, Quaternion):
            return self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z
        else:
            raise TypeError(f"不支持的类型: {type(other)}")
    
    def angle_to(self, other):
        """计算到另一个四元数的角度（弧度）"""
        dot = self.dot(other)
        dot = np.clip(dot, -1.0, 1.0)
        return 2 * np.arccos(abs(dot))
    
    def rotate_vector(self, v: np.ndarray):
        """用四元数旋转向量"""
        if v.shape != (3,):
            raise ValueError(f"向量必须是3维，但形状为{v.shape}")
        
        # 将向量转换为纯四元数
        v_q = Quaternion(0.0, v[0], v[1], v[2])
        
        # 旋转: q * v * q⁻¹
        rotated = self * v_q * self.inverse()
        
        return np.array([rotated.x, rotated.y, rotated.z])
    
    def to_euler(self, order: str = 'zyx'):
        """转换为欧拉角（弧度）"""
        # 使用现有函数
        from backend.services.robot_configuration_service import quaternion_to_rpy
        q_vec = [self.x, self.y, self.z, self.w]  # 注意顺序
        rpy = quaternion_to_rpy(q_vec)
        return rpy  # [roll, pitch, yaw]
    
    def to_rotation_matrix(self):
        """转换为旋转矩阵"""
        w, x, y, z = self.w, self.x, self.y, self.z
        
        R = np.zeros((3, 3))
        
        R[0, 0] = 1 - 2*y*y - 2*z*z
        R[0, 1] = 2*x*y - 2*w*z
        R[0, 2] = 2*x*z + 2*w*y
        
        R[1, 0] = 2*x*y + 2*w*z
        R[1, 1] = 1 - 2*x*x - 2*z*z
        R[1, 2] = 2*y*z - 2*w*x
        
        R[2, 0] = 2*x*z - 2*w*y
        R[2, 1] = 2*y*z + 2*w*x
        R[2, 2] = 1 - 2*x*x - 2*y*y
        
        return R
    
    def to_axis_angle(self):
        """转换为轴角表示"""
        angle = 2 * np.arccos(np.clip(self.w, -1.0, 1.0))
        
        if angle < 1e-6:
            return np.array([1.0, 0.0, 0.0]), 0.0
        
        axis = np.array([self.x, self.y, self.z]) / np.sin(angle/2)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        
        return axis, angle
    
    def slerp(self, other, t: float):
        """球面线性插值（SLERP）"""
        if not isinstance(other, Quaternion):
            raise TypeError(f"不支持的类型: {type(other)}")
        
        # 计算点积
        dot = self.dot(other)
        
        # 确保使用最短路径
        if dot < 0:
            q1 = -self
            dot = -dot
        else:
            q1 = self
        
        # 如果角度很小，使用线性插值
        if dot > 0.9995:
            result = q1 * (1 - t) + other * t
            return result.normalize()
        
        # 计算角度
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        
        # 计算插值权重
        q1_weight = np.sin(theta_0 - theta) / np.sin(theta_0)
        q2_weight = np.sin(theta) / np.sin(theta_0)
        
        # SLERP插值
        result = q1 * q1_weight + other * q2_weight
        
        return result.normalize()
    
    def __str__(self):
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"
    
    def __repr__(self):
        return self.__str__()


class QuaternionTensor:
    """四元数张量类（PyTorch版本）"""
    
    @staticmethod
    def identity(batch_size: int = 1, device=None, dtype=torch.float32):
        """批量单位四元数"""
        q = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        q[:, 0] = 1.0  # w分量
        return q
    
    @staticmethod
    def random(batch_size: int = 1, device=None, dtype=torch.float32):
        """均匀随机单位四元数"""
        u1 = torch.rand(batch_size, 1, device=device, dtype=dtype)
        u2 = torch.rand(batch_size, 1, device=device, dtype=dtype)
        u3 = torch.rand(batch_size, 1, device=device, dtype=dtype)
        
        w = torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2)
        x = torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2)
        y = torch.sqrt(u1) * torch.sin(2 * np.pi * u3)
        z = torch.sqrt(u1) * torch.cos(2 * np.pi * u3)
        
        return torch.cat([w, x, y, z], dim=1)
    
    @staticmethod
    def from_euler(rpy: torch.Tensor, order: str = 'zyx'):
        """
        从欧拉角创建四元数张量
        
        参数:
            rpy: [batch_size, 3] 欧拉角（弧度）
            order: 旋转顺序
        
        返回:
            quaternions: [batch_size, 4] 四元数
        """
        batch_size = rpy.shape[0]
        device = rpy.device
        dtype = rpy.dtype
        
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        
        # 计算半角
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        
        # ZYX顺序
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        q = torch.stack([w, x, y, z], dim=1)
        
        # 归一化
        norm = torch.norm(q, dim=1, keepdim=True) + 1e-8
        q = q / norm
        
        return q
    
    @staticmethod
    def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor):
        """
        从轴角创建四元数张量
        
        参数:
            axis: [batch_size, 3] 旋转轴
            angle: [batch_size] 旋转角度（弧度）
        
        返回:
            quaternions: [batch_size, 4] 四元数
        """
        batch_size = axis.shape[0]
        device = axis.device
        dtype = axis.dtype
        
        # 归一化轴
        axis_norm = torch.norm(axis, dim=1, keepdim=True) + 1e-8
        axis_normalized = axis / axis_norm
        
        # 计算四元数
        sin_half = torch.sin(angle / 2).unsqueeze(1)
        cos_half = torch.cos(angle / 2).unsqueeze(1)
        
        w = cos_half
        xyz = axis_normalized * sin_half
        
        q = torch.cat([w, xyz], dim=1)
        
        return q
    
    @staticmethod
    def from_rotation_matrix(R: torch.Tensor):
        """
        从旋转矩阵创建四元数张量（批量）
        
        参数:
            R: [batch_size, 3, 3] 旋转矩阵
        
        返回:
            quaternions: [batch_size, 4] 四元数
        """
        batch_size = R.shape[0]
        device = R.device
        dtype = R.dtype
        
        quaternions = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        
        for i in range(batch_size):
            R_i = R[i]
            trace = torch.trace(R_i)
            
            if trace > 0:
                S = torch.sqrt(trace + 1.0) * 2
                w = 0.25 * S
                x = (R_i[2, 1] - R_i[1, 2]) / S
                y = (R_i[0, 2] - R_i[2, 0]) / S
                z = (R_i[1, 0] - R_i[0, 1]) / S
            elif R_i[0, 0] > R_i[1, 1] and R_i[0, 0] > R_i[2, 2]:
                S = torch.sqrt(1.0 + R_i[0, 0] - R_i[1, 1] - R_i[2, 2]) * 2
                w = (R_i[2, 1] - R_i[1, 2]) / S
                x = 0.25 * S
                y = (R_i[0, 1] + R_i[1, 0]) / S
                z = (R_i[0, 2] + R_i[2, 0]) / S
            elif R_i[1, 1] > R_i[2, 2]:
                S = torch.sqrt(1.0 + R_i[1, 1] - R_i[0, 0] - R_i[2, 2]) * 2
                w = (R_i[0, 2] - R_i[2, 0]) / S
                x = (R_i[0, 1] + R_i[1, 0]) / S
                y = 0.25 * S
                z = (R_i[1, 2] + R_i[2, 1]) / S
            else:
                S = torch.sqrt(1.0 + R_i[2, 2] - R_i[0, 0] - R_i[1, 1]) * 2
                w = (R_i[1, 0] - R_i[0, 1]) / S
                x = (R_i[0, 2] + R_i[2, 0]) / S
                y = (R_i[1, 2] + R_i[2, 1]) / S
                z = 0.25 * S
            
            quaternions[i] = torch.tensor([w, x, y, z], device=device, dtype=dtype)
        
        # 归一化
        norm = torch.norm(quaternions, dim=1, keepdim=True) + 1e-8
        quaternions = quaternions / norm
        
        return quaternions
    
    @staticmethod
    def normalize(q: torch.Tensor):
        """归一化四元数张量"""
        norm = torch.norm(q, dim=1, keepdim=True) + 1e-8
        return q / norm
    
    @staticmethod
    def conjugate(q: torch.Tensor):
        """四元数共轭"""
        q_conj = q.clone()
        q_conj[:, 1:] = -q_conj[:, 1:]  # 虚部取负
        return q_conj
    
    @staticmethod
    def inverse(q: torch.Tensor):
        """四元数逆"""
        norm_sq = torch.sum(q * q, dim=1, keepdim=True) + 1e-8
        q_conj = QuaternionTensor.conjugate(q)
        return q_conj / norm_sq
    
    @staticmethod
    def multiply(q1: torch.Tensor, q2: torch.Tensor):
        """
        四元数乘法（批量）
        
        参数:
            q1: [batch_size, 4] 四元数
            q2: [batch_size, 4] 四元数
        
        返回:
            result: [batch_size, 4] 四元数乘积
        """
        batch_size = q1.shape[0]
        
        # 分离分量
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        # 四元数乘法公式
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        result = torch.stack([w, x, y, z], dim=1)
        
        return result
    
    @staticmethod
    def dot(q1: torch.Tensor, q2: torch.Tensor):
        """四元数点积"""
        return torch.sum(q1 * q2, dim=1)
    
    @staticmethod
    def angle_between(q1: torch.Tensor, q2: torch.Tensor):
        """计算两个四元数之间的角度（弧度）"""
        dot = QuaternionTensor.dot(q1, q2)
        dot = torch.clamp(dot, -1.0, 1.0)
        return 2 * torch.acos(torch.abs(dot))
    
    @staticmethod
    def rotate_vector(q: torch.Tensor, v: torch.Tensor):
        """
        用四元数旋转向量（批量）
        
        参数:
            q: [batch_size, 4] 四元数
            v: [batch_size, 3] 向量
        
        返回:
            rotated: [batch_size, 3] 旋转后的向量
        """
        batch_size = q.shape[0]
        
        # 将向量转换为纯四元数
        v_q = torch.zeros(batch_size, 4, device=q.device, dtype=q.dtype)
        v_q[:, 1:] = v
        
        # 计算 q * v * q⁻¹
        q_inv = QuaternionTensor.inverse(q)
        rotated_q = QuaternionTensor.multiply(QuaternionTensor.multiply(q, v_q), q_inv)
        
        return rotated_q[:, 1:]  # 返回向量部分
    
    @staticmethod
    def to_euler(q: torch.Tensor, order: str = 'zyx'):
        """
        四元数转欧拉角（批量）
        
        参数:
            q: [batch_size, 4] 四元数
            order: 旋转顺序
        
        返回:
            rpy: [batch_size, 3] 欧拉角（弧度）
        """
        batch_size = q.shape[0]
        device = q.device
        dtype = q.dtype
        
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # 滚转（roll）
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # 俯仰（pitch）
        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.where(
            torch.abs(sinp) >= 0.9999,
            torch.sign(sinp) * torch.pi / 2,
            torch.asin(sinp)
        )
        
        # 偏航（yaw）
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return torch.stack([roll, pitch, yaw], dim=1)
    
    @staticmethod
    def to_rotation_matrix(q: torch.Tensor):
        """
        四元数转旋转矩阵（批量）
        
        参数:
            q: [batch_size, 4] 四元数
        
        返回:
            R: [batch_size, 3, 3] 旋转矩阵
        """
        batch_size = q.shape[0]
        device = q.device
        dtype = q.dtype
        
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        R = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        
        # 计算矩阵元素
        R[:, 0, 0] = 1 - 2*y*y - 2*z*z
        R[:, 0, 1] = 2*x*y - 2*w*z
        R[:, 0, 2] = 2*x*z + 2*w*y
        
        R[:, 1, 0] = 2*x*y + 2*w*z
        R[:, 1, 1] = 1 - 2*x*x - 2*z*z
        R[:, 1, 2] = 2*y*z - 2*w*x
        
        R[:, 2, 0] = 2*x*z - 2*w*y
        R[:, 2, 1] = 2*y*z + 2*w*x
        R[:, 2, 2] = 1 - 2*x*x - 2*y*y
        
        return R
    
    @staticmethod
    def slerp(q1: torch.Tensor, q2: torch.Tensor, t: Union[float, torch.Tensor]):
        """
        球面线性插值（SLERP）
        
        参数:
            q1: [batch_size, 4] 起始四元数
            q2: [batch_size, 4] 终止四元数
            t: 插值参数（0到1之间）
        
        返回:
            interpolated: [batch_size, 4] 插值四元数
        """
        batch_size = q1.shape[0]
        device = q1.device
        dtype = q1.dtype
        
        # 计算点积
        dot = QuaternionTensor.dot(q1, q2)
        
        # 确保使用最短路径
        mask = dot < 0
        q2_adj = q2.clone()
        q2_adj[mask] = -q2_adj[mask]
        dot[mask] = -dot[mask]
        
        # 限制点积范围
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # 计算角度
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        # 避免除零
        epsilon = 1e-8
        sin_theta = torch.where(sin_theta < epsilon, epsilon, sin_theta)
        
        # 处理小角度情况（线性插值）
        linear_mask = sin_theta < 1e-3
        non_linear_mask = ~linear_mask
        
        result = torch.zeros_like(q1)
        
        # 线性插值部分
        if linear_mask.any():
            result[linear_mask] = (1 - t) * q1[linear_mask] + t * q2_adj[linear_mask]
            result[linear_mask] = QuaternionTensor.normalize(result[linear_mask])
        
        # 非线性插值部分（SLERP）
        if non_linear_mask.any():
            q1_nonlinear = q1[non_linear_mask]
            q2_nonlinear = q2_adj[non_linear_mask]
            dot_nonlinear = dot[non_linear_mask]
            theta_nonlinear = theta[non_linear_mask]
            sin_theta_nonlinear = sin_theta[non_linear_mask]
            
            # 计算权重
            if isinstance(t, torch.Tensor):
                t_nonlinear = t.expand_as(dot_nonlinear)[non_linear_mask]
            else:
                t_nonlinear = t
            
            w1 = torch.sin((1 - t_nonlinear) * theta_nonlinear) / sin_theta_nonlinear
            w2 = torch.sin(t_nonlinear * theta_nonlinear) / sin_theta_nonlinear
            
            # SLERP插值
            interpolated = w1.unsqueeze(1) * q1_nonlinear + w2.unsqueeze(1) * q2_nonlinear
            
            result[non_linear_mask] = interpolated
        
        return result


# ============================================================================
# 工具函数
# ============================================================================

def quaternion_angle_loss(pred_q: torch.Tensor, target_q: torch.Tensor):
    """
    四元数角度损失函数
    
    参数:
        pred_q: [batch_size, 4] 预测四元数
        target_q: [batch_size, 4] 目标四元数
    
    返回:
        loss: 角度损失
    """
    # 计算点积
    dot = torch.sum(pred_q * target_q, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # 计算角度误差
    angle = 2 * torch.acos(torch.abs(dot))
    
    return angle.mean()

def quaternion_dot_loss(pred_q: torch.Tensor, target_q: torch.Tensor):
    """
    四元数点积损失函数
    
    参数:
        pred_q: [batch_size, 4] 预测四元数
        target_q: [batch_size, 4] 目标四元数
    
    返回:
        loss: 点积损失
    """
    # 归一化四元数
    pred_q_norm = F.normalize(pred_q, dim=1)
    target_q_norm = F.normalize(target_q, dim=1)
    
    # 计算点积
    dot = torch.sum(pred_q_norm * target_q_norm, dim=1)
    
    # 点积应在[-1, 1]之间，越大表示越相似
    loss = 1.0 - dot.mean()
    
    return loss

def quaternion_double_cover_loss(pred_q: torch.Tensor, target_q: torch.Tensor):
    """
    四元数双倍覆盖损失函数（考虑q和-q表示同一旋转）
    
    参数:
        pred_q: [batch_size, 4] 预测四元数
        target_q: [batch_size, 4] 目标四元数
    
    返回:
        loss: 双倍覆盖损失
    """
    # 计算两种情况的点积
    dot1 = torch.sum(pred_q * target_q, dim=1)
    dot2 = torch.sum(pred_q * (-target_q), dim=1)
    
    # 取绝对值较大的点积（表示更接近的旋转）
    dot = torch.max(torch.abs(dot1), torch.abs(dot2))
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # 计算角度误差
    angle = 2 * torch.acos(dot)
    
    return angle.mean()

def euler_to_quaternion(rpy: np.ndarray, order: str = 'zyx'):
    """
    欧拉角转四元数（NumPy版本）
    
    参数:
        rpy: [3,] 或 [batch_size, 3] 欧拉角（弧度）
        order: 旋转顺序
    
    返回:
        quaternions: [4,] 或 [batch_size, 4] 四元数
    """
    if rpy.ndim == 1:
        # 单个四元数
        return Quaternion.from_euler(rpy[0], rpy[1], rpy[2], order)
    else:
        # 批量处理
        batch_size = rpy.shape[0]
        quaternions = []
        for i in range(batch_size):
            q = Quaternion.from_euler(rpy[i, 0], rpy[i, 1], rpy[i, 2], order)
            quaternions.append(q.as_vector())
        return np.array(quaternions)

def quaternion_to_euler(q: np.ndarray, order: str = 'zyx'):
    """
    四元数转欧拉角（NumPy版本）
    
    参数:
        q: [4,] 或 [batch_size, 4] 四元数
        order: 旋转顺序
    
    返回:
        rpy: [3,] 或 [batch_size, 3] 欧拉角（弧度）
    """
    if q.ndim == 1:
        # 单个四元数
        quat = Quaternion(q[0], q[1], q[2], q[3])
        return quat.to_euler(order)
    else:
        # 批量处理
        batch_size = q.shape[0]
        rpy_list = []
        for i in range(batch_size):
            quat = Quaternion(q[i, 0], q[i, 1], q[i, 2], q[i, 3])
            rpy_list.append(quat.to_euler(order))
        return np.array(rpy_list)

def normalize_quaternion_numpy(q: np.ndarray):
    """
    归一化四元数（NumPy版本）
    
    参数:
        q: [4,] 或 [batch_size, 4] 四元数
    
    返回:
        normalized: 归一化四元数
    """
    if q.ndim == 1:
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm
    else:
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        norm = np.where(norm < 1e-8, 1.0, norm)
        return q / norm


# ============================================================================
# 兼容性函数（向后兼容现有系统）
# ============================================================================

def rpy_to_quaternion(rpy):
    """向后兼容现有系统的RPY转四元数函数"""
    return Quaternion.from_euler(rpy[0], rpy[1], rpy[2]).as_vector()

def quaternion_to_rpy(q):
    """向后兼容现有系统的四元数转RPY函数"""
    quat = Quaternion(q[3], q[0], q[1], q[2])  # 注意顺序转换
    return quat.to_euler()

def normalize_quaternion(q):
    """向后兼容现有系统的四元数归一化函数"""
    return normalize_quaternion_numpy(q)


# ============================================================================
# 高级四元数运算函数
# ============================================================================

def quaternion_exp(v: np.ndarray) -> np.ndarray:
    """
    四元数指数映射（李代数到李群）
    
    参数:
        v: [3,] 或 [batch_size, 3] 旋转向量（角轴表示，模长为角度）
    
    返回:
        q: [4,] 或 [batch_size, 4] 四元数
    """
    if v.ndim == 1:
        theta = np.linalg.norm(v)
        if theta < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = v / theta
        sin_half = np.sin(theta / 2)
        cos_half = np.cos(theta / 2)
        return np.array([cos_half, axis[0]*sin_half, axis[1]*sin_half, axis[2]*sin_half])
    else:
        batch_size = v.shape[0]
        theta = np.linalg.norm(v, axis=1, keepdims=True)
        axis = v / (theta + 1e-8)
        sin_half = np.sin(theta / 2)
        cos_half = np.cos(theta / 2)
        q = np.zeros((batch_size, 4))
        q[:, 0] = cos_half.squeeze()
        q[:, 1:] = axis * sin_half
        return q

def quaternion_log(q: np.ndarray) -> np.ndarray:
    """
    四元数对数映射（李群到李代数）
    
    参数:
        q: [4,] 或 [batch_size, 4] 四元数
    
    返回:
        v: [3,] 或 [batch_size, 3] 旋转向量（角轴表示，模长为角度）
    """
    if q.ndim == 1:
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        w = q_norm[0]
        if abs(w) > 0.9999:
            return np.zeros(3)
        theta = 2 * np.arccos(np.clip(w, -1.0, 1.0))
        sin_half = np.sin(theta / 2)
        if sin_half < 1e-8:
            return np.zeros(3)
        axis = q_norm[1:] / sin_half
        return axis * theta
    else:
        batch_size = q.shape[0]
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        w = q_norm[:, 0]
        theta = 2 * np.arccos(np.clip(w, -1.0, 1.0))
        sin_half = np.sin(theta / 2)
        axis = q_norm[:, 1:] / (sin_half[:, np.newaxis] + 1e-8)
        # 处理小角度情况
        mask = sin_half < 1e-8
        axis[mask] = 0
        theta[mask] = 0
        return axis * theta[:, np.newaxis]

def quaternion_from_angular_velocity(omega: np.ndarray, dt: float) -> np.ndarray:
    """
    从角速度计算四元数增量
    
    参数:
        omega: [3,] 或 [batch_size, 3] 角速度（弧度/秒）
        dt: 时间间隔（秒）
    
    返回:
        delta_q: [4,] 或 [batch_size, 4] 四元数增量
    """
    if omega.ndim == 1:
        theta = np.linalg.norm(omega) * dt
        if theta < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = omega / (np.linalg.norm(omega) + 1e-8)
        return quaternion_exp(axis * theta)
    else:
        theta = np.linalg.norm(omega, axis=1) * dt
        axis = omega / (np.linalg.norm(omega, axis=1, keepdims=True) + 1e-8)
        return quaternion_exp(axis * theta[:, np.newaxis])

def quaternion_weighted_average(quaternions: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    加权平均四元数（用于滤波和融合）
    
    参数:
        quaternions: [n, 4] 四元数数组
        weights: [n,] 权重数组，默认为均匀权重
    
    返回:
        avg_q: [4,] 平均四元数
    """
    n = quaternions.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = weights / np.sum(weights)
    
    # 归一化所有四元数
    q_norm = normalize_quaternion_numpy(quaternions)
    
    # 初始化平均四元数
    avg_q = q_norm[0]
    
    # 迭代平均（使用梯度下降在球面上）
    max_iter = 100
    tolerance = 1e-8
    
    for _ in range(max_iter):
        # 计算误差向量
        errors = []
        for i in range(n):
            q_i = Quaternion(q_norm[i, 0], q_norm[i, 1], q_norm[i, 2], q_norm[i, 3])
            avg_quat = Quaternion(avg_q[0], avg_q[1], avg_q[2], avg_q[3])
            # 计算相对旋转
            delta_q = avg_quat.inverse() * q_i
            # 转换为旋转向量
            v = quaternion_log(delta_q.as_vector())
            errors.append(v * weights[i])
        
        error_sum = np.sum(errors, axis=0)
        error_norm = np.linalg.norm(error_sum)
        
        if error_norm < tolerance:
            break
        
        # 更新平均四元数
        delta_q_vec = quaternion_exp(error_sum * 0.5)
        delta_q = Quaternion(delta_q_vec[0], delta_q_vec[1], delta_q_vec[2], delta_q_vec[3])
        avg_quat = Quaternion(avg_q[0], avg_q[1], avg_q[2], avg_q[3])
        avg_quat = avg_quat * delta_q
        avg_q = avg_quat.normalize().as_vector()
    
    return avg_q

def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    计算两个四元数之间的测地距离
    
    参数:
        q1, q2: [4,] 四元数
    
    返回:
        distance: 角度距离（弧度）
    """
    q1_norm = normalize_quaternion_numpy(q1)
    q2_norm = normalize_quaternion_numpy(q2)
    dot = np.dot(q1_norm, q2_norm)
    dot = np.clip(dot, -1.0, 1.0)
    return 2 * np.arccos(abs(dot))

class QuaternionNormalization(nn.Module):
    """四元数归一化层（强制模长为1）"""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [batch_size, 4] 或 [batch_size, *, 4] 四元数张量
        
        返回:
            normalized: 归一化四元数
        """
        # 计算模长
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=self.eps)
        
        return x / norm
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps})"

class QuaternionDistance(nn.Module):
    """四元数距离计算层"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        计算批量四元数之间的角度距离
        
        参数:
            q1, q2: [batch_size, 4] 四元数
        
        返回:
            distances: [batch_size] 角度距离（弧度）
        """
        # 归一化
        q1_norm = F.normalize(q1, dim=-1)
        q2_norm = F.normalize(q2, dim=-1)
        
        # 计算点积
        dot = torch.sum(q1_norm * q2_norm, dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # 计算角度距离
        return 2 * torch.acos(torch.abs(dot))


# ============================================================================
# 测试函数
# ============================================================================

def test_quaternion_core():
    """测试四元数核心库功能"""
    print("测试四元数核心库...")
    
    # 测试Quaternion类
    q1 = Quaternion(1, 0, 0, 0)
    q2 = Quaternion(0, 1, 0, 0)
    
    # 乘法测试
    q3 = q1 * q2
    assert np.allclose(q3.as_vector(), [0, 1, 0, 0]), "乘法测试失败"
    
    # 欧拉角转换测试
    rpy = [0.1, 0.2, 0.3]
    q_from_euler = Quaternion.from_euler(*rpy)
    rpy_back = q_from_euler.to_euler()
    assert np.allclose(rpy, rpy_back, rtol=1e-5), "欧拉角转换测试失败"
    
    # SLERP测试
    q_start = Quaternion.from_euler(0, 0, 0)
    q_end = Quaternion.from_euler(0, 0, np.pi/2)
    q_mid = q_start.slerp(q_end, 0.5)
    angle = q_start.angle_to(q_mid)
    assert np.isclose(angle, np.pi/4, rtol=1e-5), "SLERP测试失败"
    
    # 测试QuaternionTensor类
    batch_size = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 随机四元数
    q_tensor = QuaternionTensor.random(batch_size, device)
    assert q_tensor.shape == (batch_size, 4), "四元数张量形状错误"
    
    # 归一化测试
    q_norm = QuaternionTensor.normalize(q_tensor)
    norms = torch.norm(q_norm, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5), "归一化测试失败"
    
    # 旋转测试
    vectors = torch.randn(batch_size, 3, device=device)
    rotated = QuaternionTensor.rotate_vector(q_norm, vectors)
    assert rotated.shape == (batch_size, 3), "旋转向量形状错误"
    
    print("所有测试通过！")
    
    return True


if __name__ == "__main__":
    # 运行测试
    test_quaternion_core()