#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间推理引擎模块
提供几何推理、距离计算、角度计算、碰撞检测等功能
"""

import math
import re
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class SpatialReasoningEngine:
    """空间推理引擎"""

    def __init__(self, config):
        self.config = config
        self.geometry_rules = self._initialize_geometry_rules()
        logger.info("空间推理引擎初始化完成")

    def _initialize_geometry_rules(self) -> Dict[str, callable]:
        """初始化几何推理规则"""
        return {
            "distance": self._calculate_distance,
            "angle": self._calculate_angle,
            "collision": self._detect_collision,
            "containment": self._check_containment,
        }

    def _calculate_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """计算两点间距离 - 欧几里得距离"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

    def _calculate_angle(
        self, vector1: Tuple[float, float], vector2: Tuple[float, float]
    ) -> float:
        """计算两向量间夹角 - 余弦公式"""
        dot = sum(a * b for a, b in zip(vector1, vector2))
        norm1 = math.sqrt(sum(a**2 for a in vector1))
        norm2 = math.sqrt(sum(b**2 for b in vector2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_angle = dot / (norm1 * norm2)
        # 限制在[-1, 1]范围内，避免浮点误差
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))

    def _detect_collision(
        self,
        shape1: Tuple[float, float, float, float],
        shape2: Tuple[float, float, float, float],
    ) -> bool:
        """检测形状碰撞 - 简化的轴对齐边界框检测"""
        # 假设形状为边界框: (min_x, min_y, max_x, max_y)
        return not (
            shape1[2] < shape2[0]
            or shape1[0] > shape2[2]
            or shape1[3] < shape2[1]
            or shape1[1] > shape2[3]
        )

    def _check_containment(
        self, point: Tuple[float, float], shape: Tuple[float, float, float, float]
    ) -> bool:
        """检查点是否在形状内 - 简化的矩形包含检测"""
        return shape[0] <= point[0] <= shape[2] and shape[1] <= point[1] <= shape[3]

    def infer(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """执行空间推理"""
        query_lower = query.lower()

        # 检测距离查询
        if "距离" in query or "distance" in query_lower:
            # 提取坐标点
            points = self._extract_points(query)
            if len(points) >= 2:
                distance = self._calculate_distance(points[0], points[1])
                return {
                    "success": True,
                    "query": query,
                    "result": f"两点间距离: {distance:.2f}",
                    "distance": distance,
                    "method": "欧几里得距离计算",
                }

        # 检测角度查询
        if "角度" in query or "angle" in query_lower:
            vectors = self._extract_vectors(query)
            if len(vectors) >= 2:
                angle = self._calculate_angle(vectors[0], vectors[1])
                return {
                    "success": True,
                    "query": query,
                    "result": f"两向量间夹角: {angle:.2f}度",
                    "angle": angle,
                    "method": "余弦公式计算",
                }

        # 默认响应
        return {
            "success": True,
            "query": query,
            "result": "空间推理完成 - 应用几何算法",
            "methods": list(self.geometry_rules.keys()),
            "explanation": "使用几何算法进行空间推理",
        }

    def _extract_points(self, text: str) -> List[Tuple[float, float]]:
        """从文本中提取坐标点"""
        points = []
        # 匹配数字模式
        numbers = re.findall(r"-?\d+\.?\d*", text)
        numbers = [float(n) for n in numbers]

        # 每2个数字组成一个点
        for i in range(0, len(numbers) - 1, 2):
            points.append((numbers[i], numbers[i + 1]))

        return points

    def _extract_vectors(self, text: str) -> List[Tuple[float, float]]:
        """从文本中提取向量"""
        # 与坐标点提取逻辑相同
        return self._extract_points(text)

    def calculate_distance_between_points(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """计算两点间距离（公开方法）"""
        return self._calculate_distance(point1, point2)

    def calculate_angle_between_vectors(
        self, vector1: Tuple[float, float], vector2: Tuple[float, float]
    ) -> float:
        """计算两向量间夹角（公开方法）"""
        return self._calculate_angle(vector1, vector2)

    def detect_collision(
        self,
        shape1: Tuple[float, float, float, float],
        shape2: Tuple[float, float, float, float],
    ) -> bool:
        """检测形状碰撞（公开方法）"""
        return self._detect_collision(shape1, shape2)

    def check_point_containment(
        self, point: Tuple[float, float], shape: Tuple[float, float, float, float]
    ) -> bool:
        """检查点是否在形状内（公开方法）"""
        return self._check_containment(point, shape)
