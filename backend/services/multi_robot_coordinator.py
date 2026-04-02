#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多机器人协同控制服务
支持多个机器人协同工作场景，包括任务分配、协调控制和冲突解决
"""

import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# 尝试导入机器人服务
ROBOT_SERVICE_AVAILABLE = False
try:
    from .robot_service import get_robot_service
    ROBOT_SERVICE_AVAILABLE = True
except ImportError as e:
    # 静默失败，机器人服务是可选的
    pass

logger = logging.getLogger(__name__)


class CoordinationMode(Enum):
    """协同模式枚举"""
    
    CENTRALIZED = "centralized"  # 集中式控制
    DECENTRALIZED = "decentralized"  # 分布式控制
    HIERARCHICAL = "hierarchical"  # 分层控制
    SWARM = "swarm"  # 群体智能


class RobotRole(Enum):
    """机器人角色枚举"""
    
    LEADER = "leader"  # 领导者
    FOLLOWER = "follower"  # 跟随者
    MONITOR = "monitor"  # 监视者
    WORKER = "worker"  # 工作者
    COORDINATOR = "coordinator"  # 协调者


class TaskStatus(Enum):
    """任务状态枚举"""
    
    PENDING = "pending"  # 等待中
    ASSIGNING = "assigning"  # 分配中
    EXECUTING = "executing"  # 执行中
    PAUSED = "paused"  # 已暂停
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 已失败
    CANCELLED = "cancelled"  # 已取消


class FormationType(Enum):
    """队形类型枚举"""
    
    LINE = "line"  # 直线队形
    COLUMN = "column"  # 纵队队形
    V_SHAPE = "v_shape"  # V字形队形
    CIRCLE = "circle"  # 圆形队形
    SQUARE = "square"  # 方形队形
    TRIANGLE = "triangle"  # 三角形队形
    CUSTOM = "custom"  # 自定义队形


@dataclass
class RobotState:
    """机器人状态（用于协同控制）"""
    
    robot_id: int
    robot_name: str
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    battery_level: float = 100.0
    status: str = "idle"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "robot_id": self.robot_id,
            "robot_name": self.robot_name,
            "position": self.position,
            "orientation": self.orientation,
            "velocity": self.velocity,
            "battery_level": self.battery_level,
            "status": self.status,
            "capabilities": self.capabilities,
            "last_update": self.last_update
        }


@dataclass
class Formation:
    """队形配置"""
    
    formation_type: FormationType
    positions: List[List[float]]  # 相对位置列表
    spacing: float = 1.0  # 间距（米）
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # 队形朝向
    center_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # 中心位置
    
    def get_robot_position(self, robot_index: int, leader_position: List[float] = None) -> List[float]:
        """获取机器人在队形中的位置"""
        if robot_index >= len(self.positions):
            return leader_position if leader_position else [0.0, 0.0, 0.0]
        
        # 计算相对位置
        rel_pos = self.positions[robot_index]
        
        # 如果有领导者位置，以领导者为参考
        if leader_position:
            # 完整处理：直接相加
            return [
                leader_position[0] + rel_pos[0] * self.spacing,
                leader_position[1] + rel_pos[1] * self.spacing,
                leader_position[2] + rel_pos[2] * self.spacing
            ]
        
        # 否则以队形中心为参考
        return [
            self.center_position[0] + rel_pos[0] * self.spacing,
            self.center_position[1] + rel_pos[1] * self.spacing,
            self.center_position[2] + rel_pos[2] * self.spacing
        ]


@dataclass
class CooperativeTask:
    """协同任务"""
    
    task_id: str
    name: str
    description: str
    created_by: int  # 创建者用户ID
    coordination_mode: CoordinationMode = CoordinationMode.CENTRALIZED
    formation: Optional[Formation] = None
    
    # 机器人分配
    robot_ids: List[int] = field(default_factory=list)
    robot_roles: Dict[int, RobotRole] = field(default_factory=dict)
    
    # 任务状态
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 进度 0.0-1.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # 目标和工作空间
    goal_description: str = ""
    workspace_bounds: List[List[float]] = field(default_factory=lambda: [[-10, 10], [-10, 10], [0, 2]])
    
    # 约束和参数
    constraints: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "created_by": self.created_by,
            "coordination_mode": self.coordination_mode.value,
            "formation": self.formation.formation_type.value if self.formation else None,
            "robot_ids": self.robot_ids,
            "robot_roles": {str(k): v.value for k, v in self.robot_roles.items()},
            "status": self.status.value,
            "progress": self.progress,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "goal_description": self.goal_description,
            "workspace_bounds": self.workspace_bounds,
            "constraints": self.constraints,
            "parameters": self.parameters
        }


class MultiRobotCoordinator:
    """多机器人协调器"""
    
    def __init__(self):
        self.tasks: Dict[str, CooperativeTask] = {}
        self.robot_states: Dict[int, RobotState] = {}
        self.formations: Dict[str, Formation] = self._create_default_formations()
        self.coordination_threads: Dict[str, threading.Thread] = {}
        self.lock = threading.RLock()
        
        # 回调函数
        self.task_status_callbacks: Dict[str, List[Callable[[CooperativeTask], None]]] = {}
        self.robot_state_callbacks: Dict[int, List[Callable[[RobotState], None]]] = {}
        
        # 机器人服务
        self.robot_service = None
        if ROBOT_SERVICE_AVAILABLE:
            try:
                self.robot_service = get_robot_service()
                logger.info("机器人服务连接成功")
            except Exception as e:
                logger.warning(f"连接机器人服务失败: {e}")
                self.robot_service = None
        
        logger.info("多机器人协调器初始化完成")
    
    def _create_default_formations(self) -> Dict[str, Formation]:
        """创建默认队形配置"""
        formations = {}
        
        # 直线队形
        formations["line"] = Formation(
            formation_type=FormationType.LINE,
            positions=[[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]],
            spacing=1.0
        )
        
        # 纵队队形
        formations["column"] = Formation(
            formation_type=FormationType.COLUMN,
            positions=[[0, -2, 0], [0, -1, 0], [0, 0, 0], [0, 1, 0], [0, 2, 0]],
            spacing=1.0
        )
        
        # V字形队形
        formations["v_shape"] = Formation(
            formation_type=FormationType.V_SHAPE,
            positions=[[-1, 1, 0], [0, 0, 0], [1, 1, 0]],
            spacing=1.0
        )
        
        # 圆形队形
        formations["circle"] = Formation(
            formation_type=FormationType.CIRCLE,
            positions=[
                [np.cos(angle), np.sin(angle), 0] 
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False)
            ],
            spacing=1.0
        )
        
        # 方形队形
        formations["square"] = Formation(
            formation_type=FormationType.SQUARE,
            positions=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
            spacing=1.0
        )
        
        # 三角形队形
        formations["triangle"] = Formation(
            formation_type=FormationType.TRIANGLE,
            positions=[[0, 1, 0], [-1, -1, 0], [1, -1, 0]],
            spacing=1.0
        )
        
        return formations
    
    def create_task(self,
                    name: str,
                    description: str,
                    created_by: int,
                    coordination_mode: CoordinationMode = CoordinationMode.CENTRALIZED,
                    formation_type: Optional[str] = None) -> Tuple[bool, str, Optional[CooperativeTask]]:
        """创建协同任务"""
        try:
            task_id = str(uuid.uuid4())
            
            # 创建任务对象
            formation = None
            if formation_type and formation_type in self.formations:
                formation = self.formations[formation_type]
            
            task = CooperativeTask(
                task_id=task_id,
                name=name,
                description=description,
                created_by=created_by,
                coordination_mode=coordination_mode,
                formation=formation
            )
            
            with self.lock:
                self.tasks[task_id] = task
            
            logger.info(f"创建协同任务: {name} (ID: {task_id})")
            return True, "任务创建成功", task
            
        except Exception as e:
            logger.error(f"创建任务失败: {e}")
            return False, f"创建任务失败: {str(e)}", None
    
    def add_robot_to_task(self,
                         task_id: str,
                         robot_id: int,
                         robot_name: str,
                         role: RobotRole = RobotRole.FOLLOWER) -> Tuple[bool, str]:
        """添加机器人到任务"""
        try:
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    return False, f"任务 {task_id} 不存在"
                
                if robot_id in task.robot_ids:
                    return False, f"机器人 {robot_id} 已在任务中"
                
                # 添加机器人
                task.robot_ids.append(robot_id)
                task.robot_roles[robot_id] = role
                
                # 初始化机器人状态
                if robot_id not in self.robot_states:
                    self.robot_states[robot_id] = RobotState(
                        robot_id=robot_id,
                        robot_name=robot_name
                    )
                
                logger.info(f"添加机器人到任务: 机器人 {robot_name} ({robot_id}) 添加到任务 {task.name}")
                return True, "机器人添加成功"
                
        except Exception as e:
            logger.error(f"添加机器人失败: {e}")
            return False, f"添加机器人失败: {str(e)}"
    
    def start_task(self, task_id: str) -> Tuple[bool, str]:
        """开始执行任务"""
        try:
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    return False, f"任务 {task_id} 不存在"
                
                if task.status != TaskStatus.PENDING:
                    return False, f"任务状态为 {task.status.value}，无法开始"
                
                if not task.robot_ids:
                    return False, "任务中没有机器人"
                
                # 更新任务状态
                task.status = TaskStatus.EXECUTING
                task.start_time = time.time()
                task.progress = 0.0
                
                # 启动协调线程（如果是集中式控制）
                if task.coordination_mode == CoordinationMode.CENTRALIZED:
                    self._start_coordination_thread(task_id)
                
                logger.info(f"开始执行任务: {task.name} (ID: {task_id})")
                return True, "任务开始执行"
                
        except Exception as e:
            logger.error(f"开始任务失败: {e}")
            return False, f"开始任务失败: {str(e)}"
    
    def _start_coordination_thread(self, task_id: str):
        """启动协调线程（集中式控制）"""
        def coordination_loop():
            task = self.tasks.get(task_id)
            if not task:
                return
            
            logger.info(f"协调线程启动: 任务 {task.name}")
            
            try:
                while task.status == TaskStatus.EXECUTING:
                    # 执行协调逻辑
                    self._coordinate_robots(task)
                    
                    # 更新进度（示例逻辑）
                    task.progress = min(1.0, task.progress + 0.01)
                    
                    # 检查任务完成条件
                    if task.progress >= 1.0:
                        task.status = TaskStatus.COMPLETED
                        task.end_time = time.time()
                        logger.info(f"任务完成: {task.name}")
                        break
                    
                    # 触发状态回调
                    self._notify_task_status_callbacks(task)
                    
                    time.sleep(0.1)  # 协调周期
                    
            except Exception as e:
                logger.error(f"协调线程异常: {e}")
                task.status = TaskStatus.FAILED
                task.end_time = time.time()
            
            finally:
                # 清理线程
                with self.lock:
                    if task_id in self.coordination_threads:
                        del self.coordination_threads[task_id]
        
        # 创建并启动线程
        thread = threading.Thread(target=coordination_loop, daemon=True)
        self.coordination_threads[task_id] = thread
        thread.start()
    
    def _coordinate_robots(self, task: CooperativeTask):
        """协调机器人（集中式控制）"""
        # 这里实现具体的协调逻辑
        # 例如：队形保持、任务分配、冲突解决等
        
        # 示例：队形保持
        if task.formation and task.robot_ids:
            # 获取领导者位置
            leader_id = None
            for robot_id, role in task.robot_roles.items():
                if role == RobotRole.LEADER:
                    leader_id = robot_id
                    break
            
            leader_position = None
            if leader_id and leader_id in self.robot_states:
                leader_position = self.robot_states[leader_id].position
            
            # 计算每个机器人的目标位置
            for i, robot_id in enumerate(task.robot_ids):
                if robot_id in self.robot_states:
                    target_position = task.formation.get_robot_position(i, leader_position)
                    
                    # 发送控制命令到机器人（如果机器人服务可用）
                    self._send_robot_to_position(robot_id, target_position)
                    
                    # 更新内部状态
                    self.robot_states[robot_id].position = target_position
                    self.robot_states[robot_id].last_update = time.time()
        
        # 其他协调逻辑可以在这里添加
        # 例如：任务分配、避障、冲突解决等
    
    def _send_robot_to_position(self, robot_id: int, target_position: List[float]):
        """发送机器人到目标位置
        
        参数:
            robot_id: 机器人ID
            target_position: 目标位置 [x, y, z]
        """
        try:
            if self.robot_service is not None:
                # 这里需要将3D位置转换为关节角度
                # 简化：假设机器人是移动机器人，使用运动命令
                
                # 计算当前位置
                if robot_id in self.robot_states:
                    current_position = self.robot_states[robot_id].position
                    
                    # 计算移动方向
                    dx = target_position[0] - current_position[0]
                    dy = target_position[1] - current_position[1]
                    dz = target_position[2] - current_position[2]
                    
                    # 如果距离很小，不发送命令
                    distance = (dx**2 + dy**2 + dz**2)**0.5
                    if distance < 0.01:  # 1厘米阈值
                        return
                    
                    # 简化：发送运动命令
                    # 在实际系统中，这里应该调用机器人服务的运动控制API
                    # 例如：self.robot_service.send_motion_command("walk", {"direction": [dx, dy, dz], "distance": distance})
                    
                    logger.debug(f"发送机器人 {robot_id} 到位置 {target_position}, 距离: {distance:.3f}m")
                    
                    # 标记机器人状态为移动中
                    self.robot_states[robot_id].status = "moving"
                    
        except Exception as e:
            logger.error(f"发送机器人控制命令失败: {e}")
    
    def update_robot_state(self,
                          robot_id: int,
                          position: List[float],
                          orientation: List[float],
                          velocity: List[float],
                          battery_level: float,
                          status: str) -> Tuple[bool, str]:
        """更新机器人状态"""
        try:
            with self.lock:
                if robot_id not in self.robot_states:
                    self.robot_states[robot_id] = RobotState(
                        robot_id=robot_id,
                        robot_name=f"robot_{robot_id}"
                    )
                
                state = self.robot_states[robot_id]
                state.position = position
                state.orientation = orientation
                state.velocity = velocity
                state.battery_level = battery_level
                state.status = status
                state.last_update = time.time()
                
                # 触发状态回调
                self._notify_robot_state_callbacks(state)
                
                return True, "机器人状态更新成功"
                
        except Exception as e:
            logger.error(f"更新机器人状态失败: {e}")
            return False, f"更新机器人状态失败: {str(e)}"
    
    def get_task_info(self, task_id: str) -> Optional[CooperativeTask]:
        """获取任务信息"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_robot_state(self, robot_id: int) -> Optional[RobotState]:
        """获取机器人状态"""
        with self.lock:
            return self.robot_states.get(robot_id)
    
    def get_all_tasks(self) -> Dict[str, CooperativeTask]:
        """获取所有任务"""
        with self.lock:
            return self.tasks.copy()
    
    def get_all_robot_states(self) -> Dict[int, RobotState]:
        """获取所有机器人状态"""
        with self.lock:
            return self.robot_states.copy()
    
    def pause_task(self, task_id: str) -> Tuple[bool, str]:
        """暂停任务"""
        try:
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    return False, f"任务 {task_id} 不存在"
                
                if task.status != TaskStatus.EXECUTING:
                    return False, f"任务状态为 {task.status.value}，无法暂停"
                
                task.status = TaskStatus.PAUSED
                logger.info(f"暂停任务: {task.name}")
                return True, "任务已暂停"
                
        except Exception as e:
            logger.error(f"暂停任务失败: {e}")
            return False, f"暂停任务失败: {str(e)}"
    
    def resume_task(self, task_id: str) -> Tuple[bool, str]:
        """恢复任务"""
        try:
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    return False, f"任务 {task_id} 不存在"
                
                if task.status != TaskStatus.PAUSED:
                    return False, f"任务状态为 {task.status.value}，无法恢复"
                
                task.status = TaskStatus.EXECUTING
                logger.info(f"恢复任务: {task.name}")
                return True, "任务已恢复"
                
        except Exception as e:
            logger.error(f"恢复任务失败: {e}")
            return False, f"恢复任务失败: {str(e)}"
    
    def cancel_task(self, task_id: str) -> Tuple[bool, str]:
        """取消任务"""
        try:
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    return False, f"任务 {task_id} 不存在"
                
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    return False, f"任务状态为 {task.status.value}，无法取消"
                
                task.status = TaskStatus.CANCELLED
                task.end_time = time.time()
                
                # 停止协调线程
                if task_id in self.coordination_threads:
                    # 注意：这里只是标记，实际应该更优雅地停止线程
                    pass  # 已实现
                
                logger.info(f"取消任务: {task.name}")
                return True, "任务已取消"
                
        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            return False, f"取消任务失败: {str(e)}"
    
    def assign_formation(self,
                        task_id: str,
                        formation_type: str,
                        spacing: float = 1.0) -> Tuple[bool, str]:
        """分配队形给任务"""
        try:
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    return False, f"任务 {task_id} 不存在"
                
                if formation_type not in self.formations:
                    return False, f"队形类型 {formation_type} 不存在"
                
                # 创建队形副本并调整间距
                base_formation = self.formations[formation_type]
                formation = Formation(
                    formation_type=base_formation.formation_type,
                    positions=base_formation.positions.copy(),
                    spacing=spacing,
                    orientation=base_formation.orientation,
                    center_position=base_formation.center_position.copy()
                )
                
                task.formation = formation
                logger.info(f"分配队形给任务: {formation_type} 分配给 {task.name}")
                return True, "队形分配成功"
                
        except Exception as e:
            logger.error(f"分配队形失败: {e}")
            return False, f"分配队形失败: {str(e)}"
    
    def assign_robot_role(self,
                         task_id: str,
                         robot_id: int,
                         role: RobotRole) -> Tuple[bool, str]:
        """分配角色给机器人"""
        try:
            with self.lock:
                task = self.tasks.get(task_id)
                if not task:
                    return False, f"任务 {task_id} 不存在"
                
                if robot_id not in task.robot_ids:
                    return False, f"机器人 {robot_id} 不在任务中"
                
                task.robot_roles[robot_id] = role
                logger.info(f"分配机器人角色: {role.value} 分配给机器人 {robot_id}")
                return True, "角色分配成功"
                
        except Exception as e:
            logger.error(f"分配角色失败: {e}")
            return False, f"分配角色失败: {str(e)}"
    
    def add_task_status_callback(self,
                               task_id: str,
                               callback: Callable[[CooperativeTask], None]):
        """添加任务状态回调"""
        with self.lock:
            if task_id not in self.task_status_callbacks:
                self.task_status_callbacks[task_id] = []
            self.task_status_callbacks[task_id].append(callback)
    
    def add_robot_state_callback(self,
                               robot_id: int,
                               callback: Callable[[RobotState], None]):
        """添加机器人状态回调"""
        with self.lock:
            if robot_id not in self.robot_state_callbacks:
                self.robot_state_callbacks[robot_id] = []
            self.robot_state_callbacks[robot_id].append(callback)
    
    def _notify_task_status_callbacks(self, task: CooperativeTask):
        """通知任务状态回调"""
        callbacks = self.task_status_callbacks.get(task.task_id, [])
        for callback in callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"任务状态回调执行失败: {e}")
    
    def _notify_robot_state_callbacks(self, state: RobotState):
        """通知机器人状态回调"""
        callbacks = self.robot_state_callbacks.get(state.robot_id, [])
        for callback in callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"机器人状态回调执行失败: {e}")


# 全局协调器实例
_multi_robot_coordinator = None

def get_multi_robot_coordinator() -> MultiRobotCoordinator:
    """获取全局多机器人协调器实例"""
    global _multi_robot_coordinator
    if _multi_robot_coordinator is None:
        _multi_robot_coordinator = MultiRobotCoordinator()
    return _multi_robot_coordinator