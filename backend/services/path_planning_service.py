#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径规划服务
提供机器人路径规划和可视化功能，支持A*、RRT等算法
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import heapq
import random
import math
import time

logger = logging.getLogger(__name__)


class PathPlanningAlgorithm(Enum):
    """路径规划算法枚举"""

    ASTAR = "astar"  # A*算法
    DIJKSTRA = "dijkstra"  # Dijkstra算法
    RRT = "rrt"  # 快速随机扩展树
    RRT_STAR = "rrt_star"  # RRT*优化算法
    PRM = "prm"  # 概率路线图


class EnvironmentType(Enum):
    """环境类型枚举"""

    GRID_2D = "grid_2d"  # 2D网格环境
    GRID_3D = "grid_3d"  # 3D网格环境
    CONTINUOUS_2D = "continuous_2d"  # 2D连续环境
    CONTINUOUS_3D = "continuous_3d"  # 3D连续环境
    VOXEL = "voxel"  # 体素环境


@dataclass
class GridCell:
    """网格单元"""

    x: int
    y: int
    z: int = 0
    cost: float = 1.0  # 通过成本
    is_obstacle: bool = False  # 是否为障碍物
    is_start: bool = False  # 是否为起点
    is_goal: bool = False  # 是否为终点
    visited: bool = False  # 是否已访问
    parent: Optional[Tuple[int, int, int]] = None  # 父节点
    g_cost: float = float("inf")  # 从起点到当前节点的实际成本
    h_cost: float = 0.0  # 启发式成本
    f_cost: float = float("inf")  # 总成本 (g + h)


@dataclass
class PlanningResult:
    """规划结果"""

    success: bool
    path: List[List[float]]  # 路径点列表 [x, y, z]
    path_length: float
    computation_time: float
    nodes_explored: int
    algorithm: PathPlanningAlgorithm
    environment_type: EnvironmentType
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "path": self.path,
            "path_length": self.path_length,
            "computation_time": self.computation_time,
            "nodes_explored": self.nodes_explored,
            "algorithm": self.algorithm.value,
            "environment_type": self.environment_type.value,
            "message": self.message,
        }


@dataclass
class Obstacle:
    """障碍物"""

    position: List[float]  # 位置 [x, y, z]
    dimensions: List[float]  # 尺寸 [长, 宽, 高]
    shape_type: str = "box"  # 形状类型：box, sphere, cylinder
    name: str = ""


class PathPlanningService:
    """路径规划服务"""

    def __init__(self):
        self.grids: Dict[str, List[List[List[GridCell]]]] = {}
        self.obstacles: Dict[str, List[Obstacle]] = {}
        logger.info("路径规划服务初始化完成")

    def create_grid_environment(
        self,
        env_id: str,
        width: int,
        height: int,
        depth: int = 1,
        cell_size: float = 1.0,
    ) -> bool:
        """创建网格环境

        参数:
            env_id: 环境ID
            width: 宽度（单元格数）
            height: 高度（单元格数）
            depth: 深度（单元格数，3D环境使用）
            cell_size: 单元格大小

        返回:
            是否创建成功
        """
        try:
            # 创建3D网格
            grid = []
            for z in range(depth):
                layer = []
                for y in range(height):
                    row = []
                    for x in range(width):
                        cell = GridCell(x=x, y=y, z=z)
                        row.append(cell)
                    layer.append(row)
                grid.append(layer)

            self.grids[env_id] = grid
            logger.info(f"创建网格环境: {env_id} ({width}x{height}x{depth})")
            # 清除缓存，因为环境已更改
            self._cache.clear()
            return True

        except Exception as e:
            logger.error(f"创建网格环境失败: {e}")
            return False

    def add_obstacle(
        self,
        env_id: str,
        position: List[float],
        dimensions: List[float],
        shape_type: str = "box",
        name: str = "",
    ) -> bool:
        """添加障碍物到环境"""
        try:
            if env_id not in self.obstacles:
                self.obstacles[env_id] = []

            obstacle = Obstacle(
                position=position,
                dimensions=dimensions,
                shape_type=shape_type,
                name=name,
            )

            self.obstacles[env_id].append(obstacle)

            # 更新网格环境中的障碍物标记
            self._update_grid_with_obstacle(env_id, obstacle)

            logger.info(f"添加障碍物到环境 {env_id}: {name}")
            # 清除缓存，因为环境已更改
            self._cache.clear()
            return True

        except Exception as e:
            logger.error(f"添加障碍物失败: {e}")
            return False

    def _update_grid_with_obstacle(self, env_id: str, obstacle: Obstacle):
        """更新网格环境中的障碍物标记"""
        if env_id not in self.grids:
            return

        grid = self.grids[env_id]
        if not grid:
            return

        # 获取网格尺寸
        depth = len(grid)
        height = len(grid[0])
        width = len(grid[0][0])

        # 计算障碍物在网格中的范围
        x_start = max(0, int((obstacle.position[0] - obstacle.dimensions[0] / 2)))
        x_end = min(width - 1, int((obstacle.position[0] + obstacle.dimensions[0] / 2)))
        y_start = max(0, int((obstacle.position[1] - obstacle.dimensions[1] / 2)))
        y_end = min(
            height - 1, int((obstacle.position[1] + obstacle.dimensions[1] / 2))
        )
        z_start = max(0, int((obstacle.position[2] - obstacle.dimensions[2] / 2)))
        z_end = min(depth - 1, int((obstacle.position[2] + obstacle.dimensions[2] / 2)))

        # 标记障碍物单元格
        for z in range(z_start, z_end + 1):
            for y in range(y_start, y_end + 1):
                for x in range(x_start, x_end + 1):
                    grid[z][y][x].is_obstacle = True

    def plan_path(
        self,
        env_id: str,
        start: List[float],
        goal: List[float],
        algorithm: PathPlanningAlgorithm = PathPlanningAlgorithm.ASTAR,
        max_iterations: int = 10000,
        step_size: float = 1.0,
    ) -> PlanningResult:
        """规划路径

        参数:
            env_id: 环境ID
            start: 起点 [x, y, z]
            goal: 终点 [x, y, z]
            algorithm: 规划算法
            max_iterations: 最大迭代次数
            step_size: 步长（连续环境使用）

        返回:
            规划结果
        """
        start_time = time.time()

        # 生成缓存键
        cache_key = f"{env_id}:{             tuple(start)}:{             tuple(goal)}:{             algorithm.value}:{max_iterations}:{step_size}"

        # 检查缓存
        if cache_key in self._cache:
            cached_result, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                logger.debug(f"缓存命中: {cache_key}")
                cached_result.computation_time = time.time() - start_time
                return cached_result

        try:
            # 根据算法选择规划方法
            if algorithm in [
                PathPlanningAlgorithm.ASTAR,
                PathPlanningAlgorithm.DIJKSTRA,
            ]:
                result = self._plan_grid_path(
                    env_id, start, goal, algorithm, max_iterations
                )
            elif algorithm in [
                PathPlanningAlgorithm.RRT,
                PathPlanningAlgorithm.RRT_STAR,
            ]:
                result = self._plan_rrt_path(
                    env_id, start, goal, algorithm, max_iterations, step_size
                )
            else:
                # 默认使用A*算法
                result = self._plan_grid_path(
                    env_id, start, goal, PathPlanningAlgorithm.ASTAR, max_iterations
                )

            result.computation_time = time.time() - start_time
            # 存储到缓存
            self._cache[cache_key] = (result, time.time())
            return result

        except Exception as e:
            logger.error(f"路径规划失败: {e}")
            computation_time = time.time() - start_time
            return PlanningResult(
                success=False,
                path=[],
                path_length=0.0,
                computation_time=computation_time,
                nodes_explored=0,
                algorithm=algorithm,
                environment_type=EnvironmentType.GRID_3D,
                message=f"路径规划失败: {str(e)}",
            )

    def _plan_grid_path(
        self,
        env_id: str,
        start: List[float],
        goal: List[float],
        algorithm: PathPlanningAlgorithm,
        max_iterations: int,
    ) -> PlanningResult:
        """网格环境路径规划（A*或Dijkstra）"""

        if env_id not in self.grids:
            return PlanningResult(
                success=False,
                path=[],
                path_length=0.0,
                computation_time=0.0,
                nodes_explored=0,
                algorithm=algorithm,
                environment_type=EnvironmentType.GRID_3D,
                message=f"环境 {env_id} 不存在",
            )

        grid = self.grids[env_id]
        len(grid)
        len(grid[0])
        len(grid[0][0])

        # 将坐标转换为网格索引
        start_idx = (int(start[0]), int(start[1]), int(start[2]))
        goal_idx = (int(goal[0]), int(goal[1]), int(goal[2]))

        # 验证起点和终点
        if not self._is_valid_cell(grid, start_idx):
            return PlanningResult(
                success=False,
                path=[],
                path_length=0.0,
                computation_time=0.0,
                nodes_explored=0,
                algorithm=algorithm,
                environment_type=EnvironmentType.GRID_3D,
                message="起点位置无效",
            )

        if not self._is_valid_cell(grid, goal_idx):
            return PlanningResult(
                success=False,
                path=[],
                path_length=0.0,
                computation_time=0.0,
                nodes_explored=0,
                algorithm=algorithm,
                environment_type=EnvironmentType.GRID_3D,
                message="终点位置无效",
            )

        # 重置网格
        self._reset_grid(grid)

        # 设置起点和终点
        grid[start_idx[2]][start_idx[1]][start_idx[0]].is_start = True
        grid[start_idx[2]][start_idx[1]][start_idx[0]].g_cost = 0
        grid[goal_idx[2]][goal_idx[1]][goal_idx[0]].is_goal = True

        # 初始化开放列表
        open_list = []
        start_cell = grid[start_idx[2]][start_idx[1]][start_idx[0]]

        # 计算启发式成本
        if algorithm == PathPlanningAlgorithm.ASTAR:
            start_cell.h_cost = self._heuristic(start_idx, goal_idx)
        start_cell.f_cost = start_cell.g_cost + start_cell.h_cost

        heapq.heappush(open_list, (start_cell.f_cost, id(start_cell), start_cell))

        nodes_explored = 0

        # 主循环
        while open_list and nodes_explored < max_iterations:
            _, _, current_cell = heapq.heappop(open_list)

            # 检查是否到达目标
            current_idx = (current_cell.x, current_cell.y, current_cell.z)
            if current_idx == goal_idx:
                # 重建路径
                path = self._reconstruct_path(grid, current_cell)
                path_length = self._calculate_path_length(path)

                return PlanningResult(
                    success=True,
                    path=path,
                    path_length=path_length,
                    computation_time=0.0,  # 将在外部设置
                    nodes_explored=nodes_explored,
                    algorithm=algorithm,
                    environment_type=EnvironmentType.GRID_3D,
                    message="路径规划成功",
                )

            # 标记为已访问
            current_cell.visited = True
            nodes_explored += 1

            # 探索邻居
            neighbors = self._get_neighbors(grid, current_idx)
            for neighbor_idx in neighbors:
                neighbor_cell = grid[neighbor_idx[2]][neighbor_idx[1]][neighbor_idx[0]]

                # 跳过障碍物和已访问的单元格
                if neighbor_cell.is_obstacle or neighbor_cell.visited:
                    continue

                # 计算新的g成本
                tentative_g_cost = current_cell.g_cost + self._cost_between(
                    current_idx, neighbor_idx
                )

                if tentative_g_cost < neighbor_cell.g_cost:
                    # 更新邻居单元格
                    neighbor_cell.parent = current_idx
                    neighbor_cell.g_cost = tentative_g_cost

                    if algorithm == PathPlanningAlgorithm.ASTAR:
                        neighbor_cell.h_cost = self._heuristic(neighbor_idx, goal_idx)
                    neighbor_cell.f_cost = neighbor_cell.g_cost + neighbor_cell.h_cost

                    # 添加到开放列表
                    heapq.heappush(
                        open_list,
                        (neighbor_cell.f_cost, id(neighbor_cell), neighbor_cell),
                    )

        # 没有找到路径
        return PlanningResult(
            success=False,
            path=[],
            path_length=0.0,
            computation_time=0.0,
            nodes_explored=nodes_explored,
            algorithm=algorithm,
            environment_type=EnvironmentType.GRID_3D,
            message=f"在 {nodes_explored} 次迭代后未找到路径",
        )

    def _plan_rrt_path(
        self,
        env_id: str,
        start: List[float],
        goal: List[float],
        algorithm: PathPlanningAlgorithm,
        max_iterations: int,
        step_size: float,
    ) -> PlanningResult:
        """RRT路径规划（连续环境）"""

        # RRT数据结构
        class RRTNode:
            def __init__(self, position: List[float], parent=None):
                self.position = position
                self.parent = parent
                self.children = []

        # 创建根节点
        root = RRTNode(start)
        nodes = [root]

        goal_reached = False
        goal_node = None
        nodes_explored = 0

        # 主循环
        for i in range(max_iterations):
            # 随机采样
            if random.random() < 0.1:  # 10%概率采样到目标点
                target = goal
            else:
                # 在环境边界内随机采样
                # 完整：假设环境边界为[-10, 10]的立方体
                target = [
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(0, 5),
                ]

            # 寻找最近的节点
            nearest_node = root
            min_distance = float("inf")
            for node in nodes:
                distance = self._distance(node.position, target)
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node

            # 向目标方向扩展
            direction = [
                target[0] - nearest_node.position[0],
                target[1] - nearest_node.position[1],
                target[2] - nearest_node.position[2],
            ]

            # 归一化方向向量
            dir_length = math.sqrt(
                direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2
            )
            if dir_length > 0:
                direction = [
                    direction[0] / dir_length,
                    direction[1] / dir_length,
                    direction[2] / dir_length,
                ]

            # 计算新位置
            new_position = [
                nearest_node.position[0] + direction[0] * step_size,
                nearest_node.position[1] + direction[1] * step_size,
                nearest_node.position[2] + direction[2] * step_size,
            ]

            # 检查碰撞
            if self._check_collision(env_id, new_position):
                continue

            # 创建新节点
            new_node = RRTNode(new_position, nearest_node)
            nearest_node.children.append(new_node)
            nodes.append(new_node)
            nodes_explored += 1

            # 检查是否到达目标
            if self._distance(new_position, goal) < step_size:
                goal_reached = True
                goal_node = new_node
                break

        # 重建路径
        if goal_reached and goal_node:
            path = []
            current_node = goal_node

            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent

            path.reverse()  # 从起点到终点

            # 计算路径长度
            path_length = 0.0
            for i in range(len(path) - 1):
                path_length += self._distance(path[i], path[i + 1])

            return PlanningResult(
                success=True,
                path=path,
                path_length=path_length,
                computation_time=0.0,
                nodes_explored=nodes_explored,
                algorithm=algorithm,
                environment_type=EnvironmentType.CONTINUOUS_3D,
                message="RRT路径规划成功",
            )
        else:
            return PlanningResult(
                success=False,
                path=[],
                path_length=0.0,
                computation_time=0.0,
                nodes_explored=nodes_explored,
                algorithm=algorithm,
                environment_type=EnvironmentType.CONTINUOUS_3D,
                message=f"RRT在 {nodes_explored} 次迭代后未找到路径",
            )

    def _is_valid_cell(
        self, grid: List[List[List[GridCell]]], idx: Tuple[int, int, int]
    ) -> bool:
        """检查网格索引是否有效"""
        x, y, z = idx
        depth = len(grid)
        height = len(grid[0])
        width = len(grid[0][0])

        return 0 <= z < depth and 0 <= y < height and 0 <= x < width

    def _reset_grid(self, grid: List[List[List[GridCell]]]):
        """重置网格状态"""
        for layer in grid:
            for row in layer:
                for cell in row:
                    cell.visited = False
                    cell.parent = None
                    cell.g_cost = float("inf")
                    cell.h_cost = 0.0
                    cell.f_cost = float("inf")
                    cell.is_start = False
                    cell.is_goal = False

    def _heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """启发式函数（曼哈顿距离）"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def _distance(self, a: List[float], b: List[float]) -> float:
        """计算两点之间的欧几里得距离"""
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _cost_between(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """计算两个单元格之间的成本"""
        # 完整：使用欧几里得距离
        return self._distance([a[0], a[1], a[2]], [b[0], b[1], b[2]])

    def _get_neighbors(
        self, grid: List[List[List[GridCell]]], idx: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """获取邻居单元格（8方向或26方向）"""
        x, y, z = idx
        depth = len(grid)
        height = len(grid[0])
        width = len(grid[0][0])

        neighbors = []

        # 8方向邻居（同一层）
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # 跳过自身
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbors.append((nx, ny, z))

        # 3D：添加上下层邻居
        if depth > 1:
            for dz in [-1, 1]:
                nz = z + dz
                if 0 <= nz < depth:
                    neighbors.append((x, y, nz))

        return neighbors

    def _reconstruct_path(
        self, grid: List[List[List[GridCell]]], goal_cell: GridCell
    ) -> List[List[float]]:
        """重建路径"""
        path = []
        current_cell = goal_cell

        while current_cell:
            path.append(
                [float(current_cell.x), float(current_cell.y), float(current_cell.z)]
            )
            if current_cell.parent:
                x, y, z = current_cell.parent
                current_cell = grid[z][y][x]
            else:
                current_cell = None

        path.reverse()
        return path

    def _calculate_path_length(self, path: List[List[float]]) -> float:
        """计算路径长度"""
        if len(path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += self._distance(path[i], path[i + 1])

        return total_length

    def _check_collision(self, env_id: str, position: List[float]) -> bool:
        """检查位置是否与障碍物碰撞"""
        if env_id not in self.obstacles:
            return False

        for obstacle in self.obstacles[env_id]:
            if self._point_in_obstacle(position, obstacle):
                return True

        return False

    def _point_in_obstacle(self, point: List[float], obstacle: Obstacle) -> bool:
        """检查点是否在障碍物内部"""
        if obstacle.shape_type == "box":
            # 检查点是否在长方体内部
            half_dims = [d / 2 for d in obstacle.dimensions]
            return (
                obstacle.position[0] - half_dims[0]
                <= point[0]
                <= obstacle.position[0] + half_dims[0]
                and obstacle.position[1] - half_dims[1]
                <= point[1]
                <= obstacle.position[1] + half_dims[1]
                and obstacle.position[2] - half_dims[2]
                <= point[2]
                <= obstacle.position[2] + half_dims[2]
            )
        elif obstacle.shape_type == "sphere":
            # 检查点是否在球体内部
            distance = self._distance(point, obstacle.position)
            return distance <= obstacle.dimensions[0] / 2  # 假设dimensions[0]是直径
        else:
            # 完整处理：使用边界框
            half_dims = [d / 2 for d in obstacle.dimensions]
            return (
                obstacle.position[0] - half_dims[0]
                <= point[0]
                <= obstacle.position[0] + half_dims[0]
                and obstacle.position[1] - half_dims[1]
                <= point[1]
                <= obstacle.position[1] + half_dims[1]
                and obstacle.position[2] - half_dims[2]
                <= point[2]
                <= obstacle.position[2] + half_dims[2]
            )

    def get_environment_info(self, env_id: str) -> Dict[str, Any]:
        """获取环境信息"""
        if env_id not in self.grids:
            return {"exists": False}

        grid = self.grids[env_id]
        depth = len(grid)
        height = len(grid[0])
        width = len(grid[0][0])

        # 统计障碍物数量
        obstacle_count = 0
        for layer in grid:
            for row in layer:
                for cell in row:
                    if cell.is_obstacle:
                        obstacle_count += 1

        # 获取障碍物列表
        obstacles = []
        if env_id in self.obstacles:
            for obs in self.obstacles[env_id]:
                obstacles.append(
                    {
                        "position": obs.position,
                        "dimensions": obs.dimensions,
                        "shape_type": obs.shape_type,
                        "name": obs.name,
                    }
                )

        return {
            "exists": True,
            "dimensions": {"width": width, "height": height, "depth": depth},
            "grid_size": width * height * depth,
            "obstacle_count": obstacle_count,
            "obstacles": obstacles,
        }

    def get_visualization_data(self, env_id: str) -> Dict[str, Any]:
        """获取可视化数据"""
        info = self.get_environment_info(env_id)
        if not info["exists"]:
            return info

        # 添加可视化特定的数据
        grid = self.grids[env_id]
        depth = len(grid)
        height = len(grid[0])
        width = len(grid[0][0])

        # 构建网格数据结构
        grid_data = []
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    cell = grid[z][y][x]
                    if cell.is_obstacle:
                        grid_data.append({"position": [x, y, z], "type": "obstacle"})
                    elif cell.is_start:
                        grid_data.append({"position": [x, y, z], "type": "start"})
                    elif cell.is_goal:
                        grid_data.append({"position": [x, y, z], "type": "goal"})
                    elif cell.visited:
                        grid_data.append({"position": [x, y, z], "type": "visited"})

        info["grid_data"] = grid_data
        info["cell_count"] = len(grid_data)

        return info


# 全局路径规划服务实例
_path_planning_service = None


def get_path_planning_service() -> PathPlanningService:
    """获取全局路径规划服务实例"""
    global _path_planning_service
    if _path_planning_service is None:
        _path_planning_service = PathPlanningService()
    return _path_planning_service
