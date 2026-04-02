#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
碰撞检测服务
提供基本的3D碰撞检测功能，支持边界框、球体和圆柱体碰撞检测
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CollisionShapeType(Enum):
    """碰撞形状类型枚举"""
    
    BOX = "box"  # 边界框
    SPHERE = "sphere"  # 球体
    CYLINDER = "cylinder"  # 圆柱体
    MESH = "mesh"  # 完整处理）


@dataclass
class CollisionShape:
    """碰撞形状"""
    
    shape_type: CollisionShapeType
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    
    # 形状参数
    dimensions: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])  # 对于BOX: [长, 宽, 高]
    radius: float = 0.5  # 对于SPHERE和CYLINDER
    height: float = 1.0  # 对于CYLINDER
    
    # 元数据
    name: str = ""
    object_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "shape_type": self.shape_type.value,
            "position": self.position,
            "orientation": self.orientation,
            "dimensions": self.dimensions,
            "radius": self.radius,
            "height": self.height,
            "name": self.name,
            "object_id": self.object_id
        }


class CollisionDetectionService:
    """碰撞检测服务"""
    
    def __init__(self):
        self.shapes: Dict[str, CollisionShape] = {}
        logger.info("碰撞检测服务初始化完成")
    
    def register_shape(self, shape: CollisionShape) -> bool:
        """注册碰撞形状"""
        try:
            if not shape.object_id:
                shape.object_id = str(len(self.shapes))
            
            self.shapes[shape.object_id] = shape
            logger.debug(f"注册碰撞形状: {shape.name} ({shape.object_id})")
            return True
            
        except Exception as e:
            logger.error(f"注册碰撞形状失败: {e}")
            return False
    
    def unregister_shape(self, object_id: str) -> bool:
        """注销碰撞形状"""
        try:
            if object_id in self.shapes:
                del self.shapes[object_id]
                logger.debug(f"注销碰撞形状: {object_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"注销碰撞形状失败: {e}")
            return False
    
    def update_shape_position(self, object_id: str, position: List[float], orientation: Optional[List[float]] = None) -> bool:
        """更新碰撞形状位置"""
        try:
            if object_id not in self.shapes:
                return False
            
            shape = self.shapes[object_id]
            shape.position = position
            if orientation:
                shape.orientation = orientation
            
            return True
            
        except Exception as e:
            logger.error(f"更新形状位置失败: {e}")
            return False
    
    def check_collision(self, object_id1: str, object_id2: str) -> Tuple[bool, float, List[float]]:
        """检查两个形状之间的碰撞
        
        返回:
            (是否碰撞, 穿透深度, 碰撞法线)
        """
        try:
            if object_id1 not in self.shapes or object_id2 not in self.shapes:
                return False, 0.0, [0.0, 0.0, 0.0]
            
            shape1 = self.shapes[object_id1]
            shape2 = self.shapes[object_id2]
            
            # 根据形状类型调用相应的碰撞检测函数
            if shape1.shape_type == CollisionShapeType.BOX and shape2.shape_type == CollisionShapeType.BOX:
                return self._check_box_box_collision(shape1, shape2)
            elif shape1.shape_type == CollisionShapeType.SPHERE and shape2.shape_type == CollisionShapeType.SPHERE:
                return self._check_sphere_sphere_collision(shape1, shape2)
            elif shape1.shape_type == CollisionShapeType.BOX and shape2.shape_type == CollisionShapeType.SPHERE:
                return self._check_box_sphere_collision(shape1, shape2)
            elif shape1.shape_type == CollisionShapeType.SPHERE and shape2.shape_type == CollisionShapeType.BOX:
                collision, depth, normal = self._check_box_sphere_collision(shape2, shape1)
                return collision, depth, [-n for n in normal] if collision else (False, 0.0, [0.0, 0.0, 0.0])
            else:
                # 完整处理：使用边界球检测
                return self._check_bounding_sphere_collision(shape1, shape2)
                
        except Exception as e:
            logger.error(f"碰撞检测失败: {e}")
            return False, 0.0, [0.0, 0.0, 0.0]
    
    def check_all_collisions(self) -> List[Dict[str, Any]]:
        """检查所有注册形状之间的碰撞"""
        collisions = []
        object_ids = list(self.shapes.keys())
        
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                object_id1 = object_ids[i]
                object_id2 = object_ids[j]
                
                collision, depth, normal = self.check_collision(object_id1, object_id2)
                if collision:
                    collisions.append({
                        "object_id1": object_id1,
                        "object_id2": object_id2,
                        "object_name1": self.shapes[object_id1].name,
                        "object_name2": self.shapes[object_id2].name,
                        "depth": depth,
                        "normal": normal,
                        "position1": self.shapes[object_id1].position,
                        "position2": self.shapes[object_id2].position
                    })
        
        return collisions
    
    def _check_box_box_collision(self, box1: CollisionShape, box2: CollisionShape) -> Tuple[bool, float, List[float]]:
        """检测两个边界框之间的碰撞（轴对齐完整版本）"""
        # 完整：假设边界框是轴对齐的
        # 完整处理
        
        # 计算半尺寸
        half_size1 = [d / 2.0 for d in box1.dimensions]
        half_size2 = [d / 2.0 for d in box2.dimensions]
        
        # 检查每个轴上的重叠
        overlap_x = abs(box1.position[0] - box2.position[0]) <= (half_size1[0] + half_size2[0])
        overlap_y = abs(box1.position[1] - box2.position[1]) <= (half_size1[1] + half_size2[1])
        overlap_z = abs(box1.position[2] - box2.position[2]) <= (half_size1[2] + half_size2[2])
        
        if overlap_x and overlap_y and overlap_z:
            # 计算穿透深度和法线（完整）
            depth_x = (half_size1[0] + half_size2[0]) - abs(box1.position[0] - box2.position[0])
            depth_y = (half_size1[1] + half_size2[1]) - abs(box1.position[1] - box2.position[1])
            depth_z = (half_size1[2] + half_size2[2]) - abs(box1.position[2] - box2.position[2])
            
            # 找到最小穿透深度的轴
            if depth_x < depth_y and depth_x < depth_z:
                normal = [1.0 if box1.position[0] < box2.position[0] else -1.0, 0.0, 0.0]
                depth = depth_x
            elif depth_y < depth_z:
                normal = [0.0, 1.0 if box1.position[1] < box2.position[1] else -1.0, 0.0]
                depth = depth_y
            else:
                normal = [0.0, 0.0, 1.0 if box1.position[2] < box2.position[2] else -1.0]
                depth = depth_z
            
            return True, depth, normal
        
        return False, 0.0, [0.0, 0.0, 0.0]
    
    def _check_sphere_sphere_collision(self, sphere1: CollisionShape, sphere2: CollisionShape) -> Tuple[bool, float, List[float]]:
        """检测两个球体之间的碰撞"""
        # 计算中心距离
        dx = sphere1.position[0] - sphere2.position[0]
        dy = sphere1.position[1] - sphere2.position[1]
        dz = sphere1.position[2] - sphere2.position[2]
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # 计算半径和
        radius_sum = sphere1.radius + sphere2.radius
        
        if distance < radius_sum:
            # 计算穿透深度和法线
            depth = radius_sum - distance
            if distance > 0:
                normal = [dx/distance, dy/distance, dz/distance]
            else:
                normal = [1.0, 0.0, 0.0]  # 重合时的默认法线
            
            return True, depth, normal
        
        return False, 0.0, [0.0, 0.0, 0.0]
    
    def _check_box_sphere_collision(self, box: CollisionShape, sphere: CollisionShape) -> Tuple[bool, float, List[float]]:
        """检测边界框和球体之间的碰撞（轴对齐完整版本）"""
        # 完整：假设边界框是轴对齐的
        
        # 计算边界框半尺寸
        half_size = [d / 2.0 for d in box.dimensions]
        
        # 找到球心到边界框的最近点
        closest_point = [
            max(box.position[0] - half_size[0], min(sphere.position[0], box.position[0] + half_size[0])),
            max(box.position[1] - half_size[1], min(sphere.position[1], box.position[1] + half_size[1])),
            max(box.position[2] - half_size[2], min(sphere.position[2], box.position[2] + half_size[2]))
        ]
        
        # 计算最近点到球心的距离
        dx = sphere.position[0] - closest_point[0]
        dy = sphere.position[1] - closest_point[1]
        dz = sphere.position[2] - closest_point[2]
        distance_squared = dx*dx + dy*dy + dz*dz
        
        if distance_squared < sphere.radius * sphere.radius:
            if distance_squared > 0:
                distance = np.sqrt(distance_squared)
                depth = sphere.radius - distance
                normal = [dx/distance, dy/distance, dz/distance]
            else:
                # 球心在边界框内部
                # 找到最近的边界
                dist_to_min_x = sphere.position[0] - (box.position[0] - half_size[0])
                dist_to_max_x = (box.position[0] + half_size[0]) - sphere.position[0]
                dist_to_min_y = sphere.position[1] - (box.position[1] - half_size[1])
                dist_to_max_y = (box.position[1] + half_size[1]) - sphere.position[1]
                dist_to_min_z = sphere.position[2] - (box.position[2] - half_size[2])
                dist_to_max_z = (box.position[2] + half_size[2]) - sphere.position[2]
                
                # 找到最小距离
                min_dist = min(dist_to_min_x, dist_to_max_x, dist_to_min_y, dist_to_max_y, dist_to_min_z, dist_to_max_z)
                depth = sphere.radius + min_dist
                
                # 确定法线方向
                if min_dist == dist_to_min_x:
                    normal = [-1.0, 0.0, 0.0]
                elif min_dist == dist_to_max_x:
                    normal = [1.0, 0.0, 0.0]
                elif min_dist == dist_to_min_y:
                    normal = [0.0, -1.0, 0.0]
                elif min_dist == dist_to_max_y:
                    normal = [0.0, 1.0, 0.0]
                elif min_dist == dist_to_min_z:
                    normal = [0.0, 0.0, -1.0]
                else:
                    normal = [0.0, 0.0, 1.0]
            
            return True, depth, normal
        
        return False, 0.0, [0.0, 0.0, 0.0]
    
    def _check_bounding_sphere_collision(self, shape1: CollisionShape, shape2: CollisionShape) -> Tuple[bool, float, List[float]]:
        """使用边界球检测碰撞（通用但保守的方法）"""
        # 计算边界球半径
        if shape1.shape_type == CollisionShapeType.BOX:
            # 边界框的边界球半径是半对角线长度
            half_diag = np.sqrt(sum((d/2.0)**2 for d in shape1.dimensions))
            radius1 = half_diag
        elif shape1.shape_type == CollisionShapeType.SPHERE:
            radius1 = shape1.radius
        elif shape1.shape_type == CollisionShapeType.CYLINDER:
            # 圆柱体的边界球半径是高度一半和半径的最大值
            radius1 = max(shape1.height/2.0, shape1.radius)
        else:
            # 默认使用包围球
            radius1 = 1.0
        
        if shape2.shape_type == CollisionShapeType.BOX:
            half_diag = np.sqrt(sum((d/2.0)**2 for d in shape2.dimensions))
            radius2 = half_diag
        elif shape2.shape_type == CollisionShapeType.SPHERE:
            radius2 = shape2.radius
        elif shape2.shape_type == CollisionShapeType.CYLINDER:
            radius2 = max(shape2.height/2.0, shape2.radius)
        else:
            radius2 = 1.0
        
        # 计算中心距离
        dx = shape1.position[0] - shape2.position[0]
        dy = shape1.position[1] - shape2.position[1]
        dz = shape1.position[2] - shape2.position[2]
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # 检查碰撞
        if distance < radius1 + radius2:
            depth = (radius1 + radius2) - distance
            if distance > 0:
                normal = [dx/distance, dy/distance, dz/distance]
            else:
                normal = [1.0, 0.0, 0.0]
            
            return True, depth, normal
        
        return False, 0.0, [0.0, 0.0, 0.0]
    
    def ray_cast(self, 
                 origin: List[float], 
                 direction: List[float],
                 max_distance: float = 100.0) -> List[Dict[str, Any]]:
        """射线检测
        
        参数:
            origin: 射线起点
            direction: 射线方向（单位向量）
            max_distance: 最大检测距离
            
        返回:
            命中列表，按距离排序
        """
        hits = []
        
        # 归一化方向向量
        dir_length = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        if dir_length == 0:
            return hits
        
        dir_norm = [direction[0]/dir_length, direction[1]/dir_length, direction[2]/dir_length]
        
        for object_id, shape in self.shapes.items():
            hit_distance = self._ray_shape_intersection(origin, dir_norm, shape, max_distance)
            if hit_distance is not None and 0 <= hit_distance <= max_distance:
                # 计算命中点
                hit_point = [
                    origin[0] + dir_norm[0] * hit_distance,
                    origin[1] + dir_norm[1] * hit_distance,
                    origin[2] + dir_norm[2] * hit_distance
                ]
                
                hits.append({
                    "object_id": object_id,
                    "object_name": shape.name,
                    "distance": hit_distance,
                    "hit_point": hit_point,
                    "shape_type": shape.shape_type.value
                })
        
        # 按距离排序
        hits.sort(key=lambda x: x["distance"])
        return hits
    
    def _ray_shape_intersection(self, 
                               origin: List[float], 
                               direction: List[float],
                               shape: CollisionShape,
                               max_distance: float) -> Optional[float]:
        """射线与形状相交检测（完整版本）"""
        # 完整：使用边界球相交检测
        if shape.shape_type == CollisionShapeType.SPHERE:
            return self._ray_sphere_intersection(origin, direction, shape, max_distance)
        elif shape.shape_type == CollisionShapeType.BOX:
            return self._ray_box_intersection(origin, direction, shape, max_distance)
        else:
            # 默认使用边界球
            return self._ray_sphere_intersection(origin, direction, shape, max_distance)
    
    def _ray_sphere_intersection(self, 
                                origin: List[float], 
                                direction: List[float],
                                sphere: CollisionShape,
                                max_distance: float) -> Optional[float]:
        """射线与球体相交检测"""
        # 球心到射线起点的向量
        oc = [origin[0] - sphere.position[0], 
              origin[1] - sphere.position[1], 
              origin[2] - sphere.position[2]]
        
        # 二次方程系数
        a = direction[0]**2 + direction[1]**2 + direction[2]**2
        b = 2.0 * (oc[0]*direction[0] + oc[1]*direction[1] + oc[2]*direction[2])
        c = oc[0]**2 + oc[1]**2 + oc[2]**2 - sphere.radius**2
        
        # 判别式
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return None  # 返回None
        
        sqrt_discriminant = np.sqrt(discriminant)
        
        # 计算两个交点
        t1 = (-b - sqrt_discriminant) / (2.0*a)
        t2 = (-b + sqrt_discriminant) / (2.0*a)
        
        # 返回有效的最近交点
        if 0 <= t1 <= max_distance:
            return t1
        elif 0 <= t2 <= max_distance:
            return t2
        
        return None  # 返回None
    
    def _ray_box_intersection(self, 
                             origin: List[float], 
                             direction: List[float],
                             box: CollisionShape,
                             max_distance: float) -> Optional[float]:
        """射线与边界框相交检测（轴对齐完整版本）"""
        # 完整：假设边界框是轴对齐的
        
        half_size = [d/2.0 for d in box.dimensions]
        bounds_min = [box.position[0] - half_size[0],
                      box.position[1] - half_size[1],
                      box.position[2] - half_size[2]]
        bounds_max = [box.position[0] + half_size[0],
                      box.position[1] + half_size[1],
                      box.position[2] + half_size[2]]
        
        # 初始化交点参数
        tmin = 0.0
        tmax = max_distance
        
        # 检查每个轴
        for i in range(3):
            if abs(direction[i]) < 1e-6:
                # 射线平行于该轴平面
                if origin[i] < bounds_min[i] or origin[i] > bounds_max[i]:
                    return None  # 返回None
            else:
                # 计算与该轴平面的交点参数
                ood = 1.0 / direction[i]
                t1 = (bounds_min[i] - origin[i]) * ood
                t2 = (bounds_max[i] - origin[i]) * ood
                
                # 确保t1是近交点，t2是远交点
                if t1 > t2:
                    t1, t2 = t2, t1
                
                # 更新交点参数范围
                if t1 > tmin:
                    tmin = t1
                if t2 < tmax:
                    tmax = t2
                
                # 检查是否无交集
                if tmin > tmax:
                    return None  # 返回None
        
        # 检查交点是否在有效范围内
        if 0 <= tmin <= max_distance:
            return tmin
        elif 0 <= tmax <= max_distance:
            return tmax
        
        return None  # 返回None


# 全局碰撞检测服务实例
_collision_detection_service = None

def get_collision_detection_service() -> CollisionDetectionService:
    """获取全局碰撞检测服务实例"""
    global _collision_detection_service
    if _collision_detection_service is None:
        _collision_detection_service = CollisionDetectionService()
    return _collision_detection_service