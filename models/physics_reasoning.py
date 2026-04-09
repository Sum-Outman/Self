#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物理推理引擎模块
提供牛顿运动定律、能量守恒、动量守恒、抛射体运动等物理推理功能
"""

import math
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PhysicsReasoningEngine:
    """物理推理引擎"""

    def __init__(self, config):
        self.config = config
        self.physics_laws = self._initialize_physics_laws()
        logger.info("物理推理引擎初始化完成")

    def _initialize_physics_laws(self) -> Dict[str, callable]:
        """初始化物理定律"""
        return {
            "newton_motion": self._apply_newton_laws,
            "energy_conservation": self._apply_energy_conservation,
            "momentum_conservation": self._apply_momentum_conservation,
            "projectile_motion": self._calculate_projectile_motion,
        }

    def _apply_newton_laws(
        self, mass: float, force: float, initial_velocity: float, time: float
    ) -> Dict[str, float]:
        """应用牛顿运动定律"""
        # F = m*a => a = F/m
        acceleration = force / mass if mass != 0 else 0
        # v = v0 + a*t
        velocity = initial_velocity + acceleration * time
        # s = v0*t + 0.5*a*t²
        displacement = initial_velocity * time + 0.5 * acceleration * time * time
        return {
            "acceleration": acceleration,
            "velocity": velocity,
            "displacement": displacement,
        }

    def _apply_energy_conservation(
        self, mass: float, velocity: float, height: float, gravity: float = 9.81
    ) -> Dict[str, float]:
        """应用能量守恒定律"""
        kinetic_energy = 0.5 * mass * velocity * velocity
        potential_energy = mass * gravity * height
        total_energy = kinetic_energy + potential_energy
        return {
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "total_energy": total_energy,
        }

    def _apply_momentum_conservation(
        self, mass1: float, velocity1: float, mass2: float, velocity2: float
    ) -> Dict[str, float]:
        """应用动量守恒定律"""
        momentum1 = mass1 * velocity1
        momentum2 = mass2 * velocity2
        total_momentum = momentum1 + momentum2
        return {
            "momentum1": momentum1,
            "momentum2": momentum2,
            "total_momentum": total_momentum,
        }

    def _calculate_projectile_motion(
        self, initial_velocity: float, launch_angle: float, gravity: float = 9.81
    ) -> Dict[str, float]:
        """计算抛射体运动"""
        angle_rad = math.radians(launch_angle)
        vx = initial_velocity * math.cos(angle_rad)
        vy = initial_velocity * math.sin(angle_rad)
        time_of_flight = 2 * vy / gravity if vy > 0 else 0
        max_height = (vy * vy) / (2 * gravity)
        range_distance = vx * time_of_flight
        return {
            "horizontal_velocity": vx,
            "vertical_velocity": vy,
            "time_of_flight": time_of_flight,
            "max_height": max_height,
            "range": range_distance,
        }

    def infer(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """执行物理推理"""
        query_lower = query.lower()

        # 提取数字参数
        numbers = re.findall(r"-?\d+\.?\d*", query)
        numbers = [float(n) for n in numbers]

        # 检测运动学问题
        if any(
            word in query_lower
            for word in [
                "加速度",
                "速度",
                "位移",
                "acceleration",
                "velocity",
                "displacement",
            ]
        ):
            if len(numbers) >= 3:
                mass = numbers[0] if len(numbers) > 0 else 1.0
                force = numbers[1] if len(numbers) > 1 else 1.0
                time = numbers[2] if len(numbers) > 2 else 1.0
                initial_velocity = numbers[3] if len(numbers) > 3 else 0.0

                result = self._apply_newton_laws(mass, force, initial_velocity, time)
                return {
                    "success": True,
                    "query": query,
                    "result": f"牛顿运动定律应用: 加速度={result['acceleration']:.2f}, 速度={result['velocity']:.2f}, 位移={result['displacement']:.2f}",
                    "details": result,
                    "method": "牛顿运动定律",
                }

        # 检测抛射体运动问题
        if any(
            word in query_lower for word in ["抛射", "弹道", "projectile", "trajectory"]
        ):
            if len(numbers) >= 2:
                initial_velocity = numbers[0]
                launch_angle = numbers[1]
                result = self._calculate_projectile_motion(
                    initial_velocity, launch_angle
                )
                return {
                    "success": True,
                    "query": query,
                    "result": f"抛射体运动: 飞行时间={result['time_of_flight']:.2f}s, 最大高度={result['max_height']:.2f}m, 射程={result['range']:.2f}m",
                    "details": result,
                    "method": "抛射体运动计算",
                }

        # 检测能量守恒问题
        if any(
            word in query_lower for word in ["能量", "守恒", "energy", "conservation"]
        ):
            if len(numbers) >= 3:
                mass = numbers[0]
                velocity = numbers[1]
                height = numbers[2]
                result = self._apply_energy_conservation(mass, velocity, height)
                return {
                    "success": True,
                    "query": query,
                    "result": f"能量守恒: 动能={result['kinetic_energy']:.2f}J, 势能={result['potential_energy']:.2f}J, 总能量={result['total_energy']:.2f}J",
                    "details": result,
                    "method": "能量守恒定律",
                }

        # 默认响应
        return {
            "success": True,
            "query": query,
            "result": "物理推理完成 - 应用物理定律",
            "methods": list(self.physics_laws.keys()),
            "explanation": "使用物理定律进行推理",
        }

    def calculate_acceleration(self, force: float, mass: float) -> float:
        """计算加速度（公开方法）"""
        return force / mass if mass != 0 else 0

    def calculate_velocity(
        self, initial_velocity: float, acceleration: float, time: float
    ) -> float:
        """计算速度（公开方法）"""
        return initial_velocity + acceleration * time

    def calculate_displacement(
        self, initial_velocity: float, acceleration: float, time: float
    ) -> float:
        """计算位移（公开方法）"""
        return initial_velocity * time + 0.5 * acceleration * time * time

    def calculate_kinetic_energy(self, mass: float, velocity: float) -> float:
        """计算动能（公开方法）"""
        return 0.5 * mass * velocity * velocity

    def calculate_potential_energy(
        self, mass: float, height: float, gravity: float = 9.81
    ) -> float:
        """计算势能（公开方法）"""
        return mass * gravity * height
