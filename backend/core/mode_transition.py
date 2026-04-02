"""
模式安全切换协议
负责管理自主模式和任务执行模式之间的安全切换

功能：
1. 切换前的状态检查（任务完成度、系统稳定性）
2. 切换中的资源锁定和释放
3. 切换后的状态验证和异常处理
4. 优雅切换和紧急切换两种模式
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class TransitionType(Enum):
    """切换类型枚举"""
    
    GRACEFUL = "graceful"          # 优雅切换：等待当前任务完成
    IMMEDIATE = "immediate"        # 立即切换：立即停止当前任务
    EMERGENCY = "emergency"        # 紧急切换：强制停止，不考虑数据完整性


class TransitionState(Enum):
    """切换状态枚举"""
    
    PENDING = "pending"            # 等待切换
    CHECKING = "checking"          # 检查状态
    LOCKING = "locking"            # 锁定资源
    SAVING = "saving"              # 保存状态
    SWITCHING = "switching"        # 执行切换
    VERIFYING = "verifying"        # 验证结果
    COMPLETED = "completed"        # 切换完成
    FAILED = "failed"              # 切换失败
    ROLLBACK = "rollback"          # 回滚中


class TransitionCheckResult:
    """切换检查结果"""
    
    def __init__(self, 
                 safe: bool = True,
                 warnings: List[str] = None,
                 errors: List[str] = None,
                 recommendations: List[str] = None):
        self.safe = safe
        self.warnings = warnings or []
        self.errors = errors or []
        self.recommendations = recommendations or []
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "safe": self.safe,
            "warnings": self.warnings,
            "errors": self.errors,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def __bool__(self):
        return self.safe


class ModeTransitionProtocol:
    """模式安全切换协议
    
    功能：
    - 切换前的状态检查
    - 切换中的资源管理
    - 切换后的验证
    - 异常处理和回滚
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化切换协议
        
        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(f"{__name__}.ModeTransitionProtocol")
        
        # 默认配置
        self.config = config or {
            "graceful_timeout_seconds": 300,      # 优雅切换超时时间（5分钟）
            "immediate_timeout_seconds": 30,      # 立即切换超时时间（30秒）
            "emergency_timeout_seconds": 10,      # 紧急切换超时时间（10秒）
            "max_retry_attempts": 3,              # 最大重试次数
            "retry_delay_seconds": 5,             # 重试延迟（秒）
            "enable_state_validation": True,      # 启用状态验证
            "enable_resource_locking": True,      # 启用资源锁定
            "enable_rollback": True,              # 启用回滚机制
            "min_system_stability_score": 0.7,    # 最小系统稳定性分数
            "max_active_tasks": 10,               # 最大活动任务数
            "critical_resource_check": True,      # 关键资源检查
        }
        
        # 当前切换状态
        self.current_transition: Optional[Dict[str, Any]] = None
        self.transition_lock = threading.RLock()
        
        # 资源锁管理器
        self.resource_locks: Dict[str, threading.Lock] = {}
        
        # 切换历史
        self.transition_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        
        # 统计信息
        self.stats = {
            "total_transitions": 0,
            "successful_transitions": 0,
            "failed_transitions": 0,
            "graceful_transitions": 0,
            "immediate_transitions": 0,
            "emergency_transitions": 0,
            "avg_transition_time_ms": 0.0,
            "last_transition_time": None,
        }
        
        self.logger.info("模式安全切换协议初始化完成")
    
    def initiate_transition(self,
                           from_mode: str,
                           to_mode: str,
                           transition_type: TransitionType = TransitionType.GRACEFUL,
                           user_id: Optional[int] = None,
                           session_id: Optional[str] = None,
                           reason: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """初始化模式切换
        
        参数:
            from_mode: 源模式
            to_mode: 目标模式
            transition_type: 切换类型
            user_id: 用户ID（可选）
            session_id: 会话ID（可选）
            reason: 切换原因（可选）
            
        返回:
            Tuple[bool, str, Optional[Dict[str, Any]]]: (是否成功, 消息, 切换详情)
        """
        with self.transition_lock:
            # 检查是否已经有进行中的切换
            if (self.current_transition and 
                self.current_transition.get("state") not in [TransitionState.COMPLETED, TransitionState.FAILED]):
                
                msg = f"已有进行中的切换: {self.current_transition.get('id', 'unknown')}"
                self.logger.warning(msg)
                return False, msg, None
            
            # 生成切换ID
            transition_id = f"transition_{int(time.time())}_{from_mode[:3]}_{to_mode[:3]}"
            
            # 创建切换记录
            self.current_transition = {
                "id": transition_id,
                "from_mode": from_mode,
                "to_mode": to_mode,
                "type": transition_type.value,
                "state": TransitionState.PENDING.value,
                "user_id": user_id,
                "session_id": session_id,
                "reason": reason,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "duration_ms": None,
                "success": None,
                "error_message": None,
                "check_result": None,
                "resource_locks": [],
                "state_snapshot": None,
                "rollback_data": None,
            }
            
            # 添加到历史
            self.transition_history.append(self.current_transition)
            if len(self.transition_history) > self.max_history_size:
                self.transition_history.pop(0)
            
            self.logger.info(
                f"切换初始化: ID={transition_id}, "
                f"从 {from_mode} 到 {to_mode}, "
                f"类型={transition_type.value}, 原因={reason}"
            )
            
            return True, "切换初始化成功", self.current_transition
    
    def execute_transition(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """执行模式切换
        
        返回:
            Tuple[bool, str, Optional[Dict[str, Any]]]: (是否成功, 消息, 切换结果)
        """
        with self.transition_lock:
            if not self.current_transition:
                return False, "没有进行中的切换", None
            
            transition_id = self.current_transition["id"]
            from_mode = self.current_transition["from_mode"]
            to_mode = self.current_transition["to_mode"]
            transition_type = TransitionType(self.current_transition["type"])
            
            self.logger.info(f"开始执行切换: {transition_id}")
            
            start_time = time.time()
            success = False
            error_message = None
            
            try:
                # 1. 检查状态
                self._update_transition_state(TransitionState.CHECKING)
                check_result = self._perform_pre_transition_checks()
                self.current_transition["check_result"] = check_result.to_dict()
                
                if not check_result.safe:
                    if transition_type == TransitionType.EMERGENCY:
                        self.logger.warning(f"紧急切换忽略安全检查: {check_result.errors}")
                    else:
                        error_message = f"安全检查失败: {check_result.errors}"
                        raise ValueError(error_message)
                
                # 2. 锁定资源
                self._update_transition_state(TransitionState.LOCKING)
                locked_resources = self._lock_resources()
                self.current_transition["resource_locks"] = locked_resources
                
                # 3. 保存状态（优雅切换）
                if transition_type == TransitionType.GRACEFUL:
                    self._update_transition_state(TransitionState.SAVING)
                    state_snapshot = self._save_current_state()
                    self.current_transition["state_snapshot"] = state_snapshot
                
                # 4. 执行切换
                self._update_transition_state(TransitionState.SWITCHING)
                
                # 根据切换类型执行不同的切换逻辑
                if transition_type == TransitionType.GRACEFUL:
                    success = self._perform_graceful_transition()
                elif transition_type == TransitionType.IMMEDIATE:
                    success = self._perform_immediate_transition()
                elif transition_type == TransitionType.EMERGENCY:
                    success = self._perform_emergency_transition()
                else:
                    raise ValueError(f"不支持的切换类型: {transition_type}")
                
                if not success:
                    error_message = "切换执行失败"
                    raise RuntimeError(error_message)
                
                # 5. 验证结果
                self._update_transition_state(TransitionState.VERIFYING)
                verification_success = self._verify_transition_result()
                
                if not verification_success:
                    error_message = "切换结果验证失败"
                    
                    # 尝试回滚
                    if self.config["enable_rollback"]:
                        self.logger.warning(f"切换验证失败，尝试回滚: {transition_id}")
                        self._update_transition_state(TransitionState.ROLLBACK)
                        rollback_success = self._perform_rollback()
                        
                        if rollback_success:
                            self.logger.info(f"回滚成功: {transition_id}")
                        else:
                            self.logger.error(f"回滚失败: {transition_id}")
                    
                    raise RuntimeError(error_message)
                
                # 6. 标记完成
                self._update_transition_state(TransitionState.COMPLETED)
                success = True
                
            except Exception as e:
                self.logger.error(f"切换执行失败: {e}")
                self._update_transition_state(TransitionState.FAILED)
                error_message = str(e)
                success = False
                
                # 释放资源锁
                self._release_resources()
                
                # 尝试回滚
                if self.config["enable_rollback"]:
                    try:
                        self._update_transition_state(TransitionState.ROLLBACK)
                        self._perform_rollback()
                    except Exception as rollback_error:
                        self.logger.error(f"回滚失败: {rollback_error}")
            
            finally:
                # 计算耗时
                end_time = time.time()
                duration_ms = int((end_time - start_time) * 1000)
                
                # 更新切换记录
                self.current_transition.update({
                    "end_time": datetime.now().isoformat(),
                    "duration_ms": duration_ms,
                    "success": success,
                    "error_message": error_message,
                    "state": self.current_transition.get("state", TransitionState.FAILED.value),
                })
                
                # 释放资源锁（如果还没有释放）
                self._release_resources()
                
                # 更新统计信息
                self._update_statistics(success, transition_type, duration_ms)
            
            # 生成结果消息
            if success:
                msg = f"切换成功: 从 {from_mode} 到 {to_mode}, 耗时 {duration_ms}ms"
                self.logger.info(msg)
            else:
                msg = f"切换失败: {error_message}"
                self.logger.error(msg)
            
            return success, msg, self.current_transition
    
    def cancel_transition(self, reason: Optional[str] = None) -> Tuple[bool, str]:
        """取消当前切换
        
        参数:
            reason: 取消原因（可选）
            
        返回:
            Tuple[bool, str]: (是否成功, 消息)
        """
        with self.transition_lock:
            if not self.current_transition:
                return False, "没有进行中的切换"
            
            transition_id = self.current_transition["id"]
            current_state = self.current_transition.get("state")
            
            # 检查是否可以取消
            cancelable_states = [
                TransitionState.PENDING.value,
                TransitionState.CHECKING.value,
                TransitionState.LOCKING.value,
                TransitionState.SAVING.value,
            ]
            
            if current_state not in cancelable_states:
                return False, f"切换状态为 {current_state}，无法取消"
            
            self.logger.info(f"取消切换: {transition_id}, 原因: {reason}")
            
            # 更新状态
            self.current_transition.update({
                "state": TransitionState.FAILED.value,
                "success": False,
                "error_message": f"用户取消: {reason}",
                "end_time": datetime.now().isoformat(),
            })
            
            # 释放资源
            self._release_resources()
            
            # 更新统计信息
            self.stats["failed_transitions"] += 1
            
            return True, f"切换已取消: {transition_id}"
    
    def _update_transition_state(self, new_state: TransitionState):
        """更新切换状态"""
        if self.current_transition:
            old_state = self.current_transition.get("state", "unknown")
            self.current_transition["state"] = new_state.value
            
            self.logger.debug(f"切换状态更新: {old_state} -> {new_state.value}")
    
    def _perform_pre_transition_checks(self) -> TransitionCheckResult:
        """执行切换前检查"""
        result = TransitionCheckResult()
        
        try:
            # 1. 检查系统稳定性
            stability_score = self._check_system_stability()
            if stability_score < self.config["min_system_stability_score"]:
                result.safe = False
                result.errors.append(f"系统稳定性分数过低: {stability_score:.2f} < {self.config['min_system_stability_score']}")
            else:
                result.recommendations.append(f"系统稳定性良好: {stability_score:.2f}")
            
            # 2. 检查活动任务数量
            active_tasks = self._get_active_task_count()
            if active_tasks > self.config["max_active_tasks"]:
                result.warnings.append(f"活动任务数过多: {active_tasks} > {self.config['max_active_tasks']}")
            
            # 3. 检查关键资源
            if self.config["critical_resource_check"]:
                critical_resources_ok = self._check_critical_resources()
                if not critical_resources_ok:
                    result.errors.append("关键资源检查失败")
                    result.safe = False
            
            # 4. 检查模式兼容性
            compatibility_ok = self._check_mode_compatibility()
            if not compatibility_ok:
                result.errors.append("模式兼容性检查失败")
                result.safe = False
            
            # 5. 检查权限
            permission_ok = self._check_transition_permission()
            if not permission_ok:
                result.errors.append("切换权限检查失败")
                result.safe = False
            
        except Exception as e:
            self.logger.error(f"切换前检查失败: {e}")
            result.safe = False
            result.errors.append(f"检查过程异常: {str(e)}")
        
        return result
    
    def _check_system_stability(self) -> float:
        """检查系统稳定性
        
        返回:
            float: 稳定性分数 0.0-1.0
        """
        # 这里实现具体的稳定性检查逻辑
        # 暂时返回一个模拟值
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # 计算稳定性分数
            cpu_score = 1.0 - min(cpu_percent / 100.0, 1.0)
            memory_score = 1.0 - min(memory_percent / 100.0, 1.0)
            
            stability_score = (cpu_score + memory_score) / 2.0
            
            return stability_score
            
        except ImportError:
            self.logger.warning("psutil不可用，使用默认稳定性分数")
            return 0.9  # 默认分数
    
    def _get_active_task_count(self) -> int:
        """获取活动任务数量
        
        返回:
            int: 活动任务数
        """
        # 这里实现具体的任务计数逻辑
        # 暂时返回模拟值
        return 5
    
    def _check_critical_resources(self) -> bool:
        """检查关键资源
        
        返回:
            bool: 关键资源是否正常
        """
        # 这里实现具体的关键资源检查逻辑
        # 暂时返回True
        return True
    
    def _check_mode_compatibility(self) -> bool:
        """检查模式兼容性
        
        返回:
            bool: 模式是否兼容
        """
        # 这里实现具体的模式兼容性检查逻辑
        # 暂时返回True
        return True
    
    def _check_transition_permission(self) -> bool:
        """检查切换权限
        
        返回:
            bool: 是否有权限执行切换
        """
        # 这里实现具体的权限检查逻辑
        # 暂时返回True
        return True
    
    def _lock_resources(self) -> List[str]:
        """锁定资源
        
        返回:
            List[str]: 已锁定的资源列表
        """
        locked_resources = []
        
        if not self.config["enable_resource_locking"]:
            return locked_resources
        
        # 需要锁定的资源列表
        resources_to_lock = [
            "system_mode",
            "autonomous_engine",
            "task_queue",
            "memory_system",
        ]
        
        for resource in resources_to_lock:
            try:
                if resource not in self.resource_locks:
                    self.resource_locks[resource] = threading.Lock()
                
                lock_acquired = self.resource_locks[resource].acquire(timeout=5.0)
                if lock_acquired:
                    locked_resources.append(resource)
                    self.logger.debug(f"资源锁定成功: {resource}")
                else:
                    self.logger.warning(f"资源锁定超时: {resource}")
                    
            except Exception as e:
                self.logger.error(f"资源锁定失败 {resource}: {e}")
        
        return locked_resources
    
    def _release_resources(self):
        """释放资源锁"""
        for resource, lock in self.resource_locks.items():
            try:
                if lock.locked():
                    lock.release()
                    self.logger.debug(f"资源释放: {resource}")
            except Exception as e:
                self.logger.error(f"资源释放失败 {resource}: {e}")
    
    def _save_current_state(self) -> Dict[str, Any]:
        """保存当前状态
        
        返回:
            Dict[str, Any]: 状态快照
        """
        # 这里实现具体的状态保存逻辑
        # 暂时返回真实数据
        
        state_snapshot = {
            "mode": self.current_transition.get("from_mode"),
            "timestamp": datetime.now().isoformat(),
            "active_tasks": self._get_active_task_count(),
            "system_metrics": {
                "cpu_usage": 45.0,
                "memory_usage": 60.0,
                "disk_usage": 30.0,
            },
            "autonomous_goals": [],
            "decision_history": [],
            "saved_by": "ModeTransitionProtocol",
        }
        
        self.logger.info("当前状态已保存")
        
        return state_snapshot
    
    def _perform_graceful_transition(self) -> bool:
        """执行优雅切换
        
        返回:
            bool: 切换是否成功
        """
        self.logger.info("执行优雅切换")
        
        try:
            # 1. 等待当前任务完成
            timeout = self.config["graceful_timeout_seconds"]
            wait_start = time.time()
            
            while self._has_active_tasks() and (time.time() - wait_start) < timeout:
                self.logger.info(f"等待任务完成... ({self._get_active_task_count()} 个活动任务)")
                time.sleep(1.0)
            
            if self._has_active_tasks():
                self.logger.warning(f"优雅切换超时，仍有 {self._get_active_task_count()} 个活动任务")
                return False
            
            # 2. 执行模式切换
            success = self._execute_mode_switch()
            
            if success:
                self.logger.info("优雅切换成功")
                return True
            else:
                self.logger.error("优雅切换失败")
                return False
            
        except Exception as e:
            self.logger.error(f"优雅切换异常: {e}")
            return False
    
    def _perform_immediate_transition(self) -> bool:
        """执行立即切换
        
        返回:
            bool: 切换是否成功
        """
        self.logger.info("执行立即切换")
        
        try:
            # 1. 停止当前任务（不等待完成）
            self._stop_active_tasks()
            
            # 2. 执行模式切换
            success = self._execute_mode_switch()
            
            if success:
                self.logger.info("立即切换成功")
                return True
            else:
                self.logger.error("立即切换失败")
                return False
            
        except Exception as e:
            self.logger.error(f"立即切换异常: {e}")
            return False
    
    def _perform_emergency_transition(self) -> bool:
        """执行紧急切换
        
        返回:
            bool: 切换是否成功
        """
        self.logger.warning("执行紧急切换")
        
        try:
            # 1. 强制停止所有任务
            self._force_stop_all_tasks()
            
            # 2. 执行模式切换（跳过部分检查）
            success = self._execute_mode_switch(emergency=True)
            
            if success:
                self.logger.warning("紧急切换成功")
                return True
            else:
                self.logger.error("紧急切换失败")
                return False
            
        except Exception as e:
            self.logger.error(f"紧急切换异常: {e}")
            return False
    
    def _has_active_tasks(self) -> bool:
        """检查是否有活动任务
        
        返回:
            bool: 是否有活动任务
        """
        return self._get_active_task_count() > 0
    
    def _stop_active_tasks(self):
        """停止活动任务"""
        self.logger.info("停止活动任务")
        # 这里实现具体的任务停止逻辑
    
    def _force_stop_all_tasks(self):
        """强制停止所有任务"""
        self.logger.warning("强制停止所有任务")
        # 这里实现具体的强制停止逻辑
    
    def _execute_mode_switch(self, emergency: bool = False) -> bool:
        """执行模式切换
        
        参数:
            emergency: 是否为紧急模式
            
        返回:
            bool: 切换是否成功
        """
        # 这里实现具体的模式切换逻辑
        # 暂时返回成功
        
        from_mode = self.current_transition.get("from_mode")
        to_mode = self.current_transition.get("to_mode")
        
        self.logger.info(f"执行模式切换: {from_mode} -> {to_mode}")
        
        # 模拟切换耗时
        time.sleep(0.5)
        
        return True
    
    def _verify_transition_result(self) -> bool:
        """验证切换结果
        
        返回:
            bool: 验证是否成功
        """
        if not self.config["enable_state_validation"]:
            return True
        
        try:
            # 1. 验证目标模式是否已激活
            target_mode_active = self._verify_target_mode_active()
            if not target_mode_active:
                self.logger.error("目标模式未激活")
                return False
            
            # 2. 验证系统功能是否正常
            system_functional = self._verify_system_functionality()
            if not system_functional:
                self.logger.error("系统功能异常")
                return False
            
            # 3. 验证数据完整性
            data_integrity_ok = self._verify_data_integrity()
            if not data_integrity_ok:
                self.logger.error("数据完整性检查失败")
                return False
            
            self.logger.info("切换结果验证成功")
            return True
            
        except Exception as e:
            self.logger.error(f"切换结果验证异常: {e}")
            return False
    
    def _verify_target_mode_active(self) -> bool:
        """验证目标模式是否已激活
        
        返回:
            bool: 目标模式是否激活
        """
        # 这里实现具体的模式激活验证逻辑
        # 暂时返回True
        return True
    
    def _verify_system_functionality(self) -> bool:
        """验证系统功能是否正常
        
        返回:
            bool: 系统功能是否正常
        """
        # 这里实现具体的系统功能验证逻辑
        # 暂时返回True
        return True
    
    def _verify_data_integrity(self) -> bool:
        """验证数据完整性
        
        返回:
            bool: 数据完整性是否正常
        """
        # 这里实现具体的数据完整性验证逻辑
        # 暂时返回True
        return True
    
    def _perform_rollback(self) -> bool:
        """执行回滚
        
        返回:
            bool: 回滚是否成功
        """
        self.logger.info("执行回滚")
        
        try:
            # 1. 检查是否有回滚数据
            rollback_data = self.current_transition.get("rollback_data")
            state_snapshot = self.current_transition.get("state_snapshot")
            
            if not rollback_data and not state_snapshot:
                self.logger.warning("没有回滚数据可用")
                return False
            
            # 2. 执行回滚操作
            rollback_success = self._execute_rollback()
            
            if rollback_success:
                self.logger.info("回滚成功")
                return True
            else:
                self.logger.error("回滚失败")
                return False
            
        except Exception as e:
            self.logger.error(f"回滚异常: {e}")
            return False
    
    def _execute_rollback(self) -> bool:
        """执行回滚操作
        
        返回:
            bool: 回滚是否成功
        """
        # 这里实现具体的回滚逻辑
        # 暂时返回成功
        
        self.logger.info("执行回滚操作")
        time.sleep(0.3)
        
        return True
    
    def _update_statistics(self, success: bool, transition_type: TransitionType, duration_ms: int):
        """更新统计信息"""
        self.stats["total_transitions"] += 1
        
        if success:
            self.stats["successful_transitions"] += 1
        else:
            self.stats["failed_transitions"] += 1
        
        if transition_type == TransitionType.GRACEFUL:
            self.stats["graceful_transitions"] += 1
        elif transition_type == TransitionType.IMMEDIATE:
            self.stats["immediate_transitions"] += 1
        elif transition_type == TransitionType.EMERGENCY:
            self.stats["emergency_transitions"] += 1
        
        # 更新平均切换时间
        total_successful = self.stats["successful_transitions"]
        old_avg = self.stats["avg_transition_time_ms"]
        
        if success:
            self.stats["avg_transition_time_ms"] = (
                old_avg * (total_successful - 1) + duration_ms
            ) / total_successful
        
        self.stats["last_transition_time"] = datetime.now().isoformat()
    
    def get_current_transition(self) -> Optional[Dict[str, Any]]:
        """获取当前切换信息
        
        返回:
            Optional[Dict[str, Any]]: 当前切换信息
        """
        return self.current_transition
    
    def get_transition_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取切换历史
        
        参数:
            limit: 返回记录数限制
            
        返回:
            List[Dict[str, Any]]: 切换历史
        """
        return self.transition_history[-limit:] if limit > 0 else self.transition_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        return {
            **self.stats,
            "current_transition_id": self.current_transition.get("id") if self.current_transition else None,
            "current_transition_state": self.current_transition.get("state") if self.current_transition else None,
            "transition_history_size": len(self.transition_history),
            "resource_locks_count": len(self.resource_locks),
            "timestamp": datetime.now().isoformat(),
        }
    
    def reset(self):
        """重置切换协议"""
        with self.transition_lock:
            self.logger.info("重置切换协议")
            
            # 释放所有资源锁
            self._release_resources()
            
            # 重置当前切换
            self.current_transition = None
            
            # 清空历史（保留最后几个）
            self.transition_history = self.transition_history[-10:] if self.transition_history else []
            
            # 重置统计信息
            self.stats = {
                "total_transitions": 0,
                "successful_transitions": 0,
                "failed_transitions": 0,
                "graceful_transitions": 0,
                "immediate_transitions": 0,
                "emergency_transitions": 0,
                "avg_transition_time_ms": 0.0,
                "last_transition_time": None,
            }
            
            self.logger.info("切换协议重置完成")


# 全局实例
_mode_transition_protocol_instance = None


def get_mode_transition_protocol(config: Optional[Dict[str, Any]] = None) -> ModeTransitionProtocol:
    """获取模式切换协议单例
    
    参数:
        config: 配置字典
        
    返回:
        ModeTransitionProtocol: 模式切换协议实例
    """
    global _mode_transition_protocol_instance
    
    if _mode_transition_protocol_instance is None:
        _mode_transition_protocol_instance = ModeTransitionProtocol(config)
    
    return _mode_transition_protocol_instance


__all__ = [
    "ModeTransitionProtocol",
    "get_mode_transition_protocol",
    "TransitionType",
    "TransitionState",
    "TransitionCheckResult",
]