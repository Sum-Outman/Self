"""
模式切换权限控制
集成到现有权限系统，提供模式切换的细粒度权限控制

功能：
1. 模式切换权限分级（用户、管理员、系统）
2. 操作审计和日志记录
3. 异常操作报警
4. 权限检查和验证
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps

from fastapi import HTTPException, status, Depends, Request
from sqlalchemy.orm import Session

from ..core.permissions import Permission, PermissionManager, UserRole
from ..dependencies.database import get_db
from ..dependencies.auth import get_current_user
from ..db_models.user import User

logger = logging.getLogger(__name__)


class ModePermission(Enum):
    """模式切换权限枚举"""
    
    # 模式查看权限
    MODE_VIEW = "mode:view"                  # 查看模式状态
    MODE_HISTORY_VIEW = "mode:history:view"  # 查看模式切换历史
    
    # 模式控制权限
    MODE_SWITCH = "mode:switch"              # 切换模式（基本）
    MODE_SWITCH_GRACEFUL = "mode:switch:graceful"  # 优雅切换模式
    MODE_SWITCH_IMMEDIATE = "mode:switch:immediate"  # 立即切换模式
    MODE_SWITCH_EMERGENCY = "mode:switch:emergency"  # 紧急切换模式
    
    # 模式配置权限
    MODE_CONFIG_VIEW = "mode:config:view"    # 查看模式配置
    MODE_CONFIG_UPDATE = "mode:config:update"  # 更新模式配置
    
    # 自主目标权限
    MODE_GOAL_VIEW = "mode:goal:view"        # 查看自主目标
    MODE_GOAL_CREATE = "mode:goal:create"    # 创建自主目标
    MODE_GOAL_UPDATE = "mode:goal:update"    # 更新自主目标
    MODE_GOAL_DELETE = "mode:goal:delete"    # 删除自主目标
    MODE_GOAL_ACTIVATE = "mode:goal:activate"  # 激活自主目标
    MODE_GOAL_COMPLETE = "mode:goal:complete"  # 完成自主目标
    
    # 决策管理权限
    MODE_DECISION_VIEW = "mode:decision:view"  # 查看决策历史
    MODE_DECISION_MAKE = "mode:decision:make"  # 制定决策
    MODE_DECISION_RESET = "mode:decision:reset"  # 重置决策引擎
    
    # 系统级权限
    MODE_SYSTEM_CONTROL = "mode:system:control"  # 系统级模式控制
    MODE_AUDIT_VIEW = "mode:audit:view"      # 查看审计日志
    MODE_FORCE_SWITCH = "mode:force:switch"  # 强制切换模式（绕过安全检查）


class ModePermissionManager:
    """模式权限管理器
    
    功能：
    - 权限检查和验证
    - 操作审计和日志记录
    - 异常操作检测和报警
    - 权限继承和角色管理
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化模式权限管理器
        
        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(f"{__name__}.ModePermissionManager")
        
        # 默认配置
        self.config = config or {
            "enable_permission_check": True,          # 启用权限检查
            "enable_audit_logging": True,             # 启用审计日志
            "enable_anomaly_detection": True,         # 启用异常检测
            "audit_log_retention_days": 90,           # 审计日志保留天数
            "max_failed_attempts": 5,                 # 最大失败尝试次数
            "lockout_duration_minutes": 15,           # 锁定持续时间（分钟）
            "require_reason_for_switch": True,        # 切换时需要提供原因
            "require_approval_for_emergency": True,   # 紧急切换需要审批
            "enable_role_based_access": True,         # 启用基于角色的访问控制
        }
        
        # 权限到角色的映射
        self.role_permissions = self._initialize_role_permissions()
        
        # 审计日志
        self.audit_log: List[Dict[str, Any]] = []
        self.max_audit_log_size = 10000
        
        # 失败尝试计数器
        self.failed_attempts: Dict[str, Dict[str, Any]] = {}
        
        # 权限管理器实例
        self.permission_manager = PermissionManager()
        
        # 统计信息
        self.stats = {
            "total_permission_checks": 0,
            "failed_permission_checks": 0,
            "audit_log_entries": 0,
            "anomaly_detections": 0,
            "lockouts_triggered": 0,
            "last_audit_time": None,
        }
        
        self.logger.info("模式权限管理器初始化完成")
    
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[ModePermission]]:
        """初始化角色权限映射
        
        返回:
            Dict[UserRole, Set[ModePermission]]: 角色到权限的映射
        """
        role_permissions = {}
        
        # 系统角色 - 所有权限
        role_permissions[UserRole.SYSTEM] = set(ModePermission)
        
        # 管理员角色 - 大部分权限，除了系统级强制操作
        admin_permissions = set(ModePermission)
        admin_permissions.discard(ModePermission.MODE_FORCE_SWITCH)  # 管理员不能强制切换
        role_permissions[UserRole.ADMIN] = admin_permissions
        
        # 高级用户角色 - 基本的模式控制和查看权限
        advanced_user_permissions = {
            ModePermission.MODE_VIEW,
            ModePermission.MODE_HISTORY_VIEW,
            ModePermission.MODE_SWITCH,
            ModePermission.MODE_SWITCH_GRACEFUL,
            ModePermission.MODE_CONFIG_VIEW,
            ModePermission.MODE_GOAL_VIEW,
            ModePermission.MODE_GOAL_CREATE,
            ModePermission.MODE_DECISION_VIEW,
        }
        role_permissions[UserRole.ADVANCED_USER] = advanced_user_permissions
        
        # 普通用户角色 - 仅查看权限和基本切换
        user_permissions = {
            ModePermission.MODE_VIEW,
            ModePermission.MODE_SWITCH,  # 只能进行基本切换
            ModePermission.MODE_GOAL_VIEW,
            ModePermission.MODE_DECISION_VIEW,
        }
        role_permissions[UserRole.USER] = user_permissions
        
        # 访客角色 - 仅查看权限
        guest_permissions = {
            ModePermission.MODE_VIEW,
        }
        role_permissions[UserRole.GUEST] = guest_permissions
        
        return role_permissions
    
    def check_permission(self, 
                        user: User, 
                        permission: ModePermission,
                        context: Optional[Dict[str, Any]] = None) -> bool:
        """检查用户权限
        
        参数:
            user: 用户对象
            permission: 需要的权限
            context: 上下文信息（可选）
            
        返回:
            bool: 是否有权限
        """
        if not self.config["enable_permission_check"]:
            return True
        
        self.stats["total_permission_checks"] += 1
        
        try:
            # 获取用户角色
            user_role = UserRole(user.role)
            
            # 检查用户是否被锁定
            if self._is_user_locked(user.id):
                self.logger.warning(f"用户被锁定，拒绝权限检查: user_id={user.id}")
                self.stats["failed_permission_checks"] += 1
                return False
            
            # 检查角色权限
            role_has_permission = permission in self.role_permissions.get(user_role, set())
            
            # 检查额外权限（如果用户有额外权限）
            extra_permissions = self._get_user_extra_permissions(user.id)
            extra_has_permission = permission in extra_permissions
            
            has_permission = role_has_permission or extra_has_permission
            
            # 记录审计日志
            if self.config["enable_audit_logging"]:
                self._log_audit_entry(
                    user_id=user.id,
                    username=user.username,
                    permission=permission.value,
                    has_permission=has_permission,
                    context=context,
                )
            
            if not has_permission:
                self.stats["failed_permission_checks"] += 1
                
                # 记录失败尝试
                self._record_failed_attempt(user.id, permission, context)
                
                # 检查异常行为
                if self.config["enable_anomaly_detection"]:
                    self._check_for_anomalies(user.id, permission, context)
            
            return has_permission
            
        except Exception as e:
            self.logger.error(f"权限检查失败: {e}")
            
            # 安全失败：默认拒绝
            self.stats["failed_permission_checks"] += 1
            return False
    
    def check_mode_switch_permission(self,
                                    user: User,
                                    from_mode: str,
                                    to_mode: str,
                                    transition_type: str,
                                    reason: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """检查模式切换权限
        
        参数:
            user: 用户对象
            from_mode: 源模式
            to_mode: 目标模式
            transition_type: 切换类型
            reason: 切换原因（可选）
            
        返回:
            Tuple[bool, Optional[str]]: (是否有权限, 错误消息)
        """
        # 确定需要的权限
        if transition_type == "graceful":
            required_permission = ModePermission.MODE_SWITCH_GRACEFUL
        elif transition_type == "immediate":
            required_permission = ModePermission.MODE_SWITCH_IMMEDIATE
        elif transition_type == "emergency":
            required_permission = ModePermission.MODE_SWITCH_EMERGENCY
            
            # 紧急切换需要额外检查
            if self.config["require_approval_for_emergency"]:
                has_approval = self._check_emergency_approval(user.id, from_mode, to_mode, reason)
                if not has_approval:
                    return False, "紧急切换需要审批"
        else:
            required_permission = ModePermission.MODE_SWITCH
        
        # 检查是否需要提供原因
        if self.config["require_reason_for_switch"] and not reason:
            return False, "切换模式需要提供原因"
        
        # 检查特定模式切换的权限（例如：从自主模式切换到任务模式可能需要额外权限）
        context = {
            "from_mode": from_mode,
            "to_mode": to_mode,
            "transition_type": transition_type,
            "reason": reason,
        }
        
        has_permission = self.check_permission(user, required_permission, context)
        
        if not has_permission:
            error_msg = f"没有权限执行 {transition_type} 切换: {from_mode} -> {to_mode}"
            return False, error_msg
        
        return True, None
    
    def _get_user_extra_permissions(self, user_id: int) -> Set[ModePermission]:
        """获取用户额外权限
        
        参数:
            user_id: 用户ID
            
        返回:
            Set[ModePermission]: 额外权限集合
        """
        # 这里实现从数据库获取用户额外权限的逻辑
        # 暂时返回空集合
        return set()
    
    def _is_user_locked(self, user_id: int) -> bool:
        """检查用户是否被锁定
        
        参数:
            user_id: 用户ID
            
        返回:
            bool: 是否被锁定
        """
        if user_id not in self.failed_attempts:
            return False
        
        user_data = self.failed_attempts[user_id]
        
        # 检查锁定是否已过期
        lockout_until = user_data.get("lockout_until")
        if lockout_until and datetime.now() < lockout_until:
            return True
        
        # 锁定已过期，清除记录
        if lockout_until and datetime.now() >= lockout_until:
            del self.failed_attempts[user_id]
        
        return False
    
    def _record_failed_attempt(self, 
                              user_id: int, 
                              permission: ModePermission,
                              context: Optional[Dict[str, Any]] = None):
        """记录失败尝试
        
        参数:
            user_id: 用户ID
            permission: 尝试的权限
            context: 上下文信息
        """
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = {
                "count": 0,
                "first_attempt": datetime.now(),
                "last_attempt": datetime.now(),
                "attempts": [],
                "lockout_until": None,
            }
        
        user_data = self.failed_attempts[user_id]
        user_data["count"] += 1
        user_data["last_attempt"] = datetime.now()
        
        # 记录尝试详情
        attempt_detail = {
            "timestamp": datetime.now().isoformat(),
            "permission": permission.value,
            "context": context,
        }
        user_data["attempts"].append(attempt_detail)
        
        # 检查是否超过最大失败尝试次数
        max_attempts = self.config["max_failed_attempts"]
        if user_data["count"] >= max_attempts:
            # 触发锁定
            lockout_duration = self.config["lockout_duration_minutes"]
            lockout_until = datetime.now() + timedelta(minutes=lockout_duration)
            user_data["lockout_until"] = lockout_until
            
            self.stats["lockouts_triggered"] += 1
            
            self.logger.warning(
                f"用户被锁定: user_id={user_id}, "
                f"失败尝试次数={user_data['count']}, "
                f"锁定直到={lockout_until}"
            )
    
    def _check_for_anomalies(self, 
                            user_id: int, 
                            permission: ModePermission,
                            context: Optional[Dict[str, Any]] = None):
        """检查异常行为
        
        参数:
            user_id: 用户ID
            permission: 权限
            context: 上下文信息
        """
        # 这里实现异常检测逻辑
        # 暂时使用简单规则
        
        if user_id not in self.failed_attempts:
            return
        
        user_data = self.failed_attempts[user_id]
        
        # 规则1：短时间内多次失败尝试
        time_window = timedelta(minutes=5)
        recent_attempts = [
            attempt for attempt in user_data["attempts"]
            if datetime.fromisoformat(attempt["timestamp"]) > datetime.now() - time_window
        ]
        
        if len(recent_attempts) > 3:
            self.logger.warning(
                f"异常行为检测: user_id={user_id}, "
                f"5分钟内 {len(recent_attempts)} 次失败尝试"
            )
            self.stats["anomaly_detections"] += 1
        
        # 规则2：尝试访问敏感权限
        sensitive_permissions = {
            ModePermission.MODE_SWITCH_EMERGENCY,
            ModePermission.MODE_FORCE_SWITCH,
            ModePermission.MODE_SYSTEM_CONTROL,
        }
        
        if permission in sensitive_permissions:
            self.logger.warning(
                f"敏感权限访问尝试: user_id={user_id}, "
                f"权限={permission.value}"
            )
            self.stats["anomaly_detections"] += 1
    
    def _check_emergency_approval(self, 
                                 user_id: int, 
                                 from_mode: str, 
                                 to_mode: str,
                                 reason: Optional[str] = None) -> bool:
        """检查紧急切换审批
        
        参数:
            user_id: 用户ID
            from_mode: 源模式
            to_mode: 目标模式
            reason: 切换原因
            
        返回:
            bool: 是否有审批
        """
        # 这里实现紧急切换审批逻辑
        # 暂时返回True（假设有审批）
        
        # 记录审批检查
        self.logger.info(
            f"紧急切换审批检查: user_id={user_id}, "
            f"从 {from_mode} 到 {to_mode}, "
            f"原因={reason}"
        )
        
        return True
    
    def _log_audit_entry(self,
                        user_id: int,
                        username: str,
                        permission: str,
                        has_permission: bool,
                        context: Optional[Dict[str, Any]] = None):
        """记录审计日志条目
        
        参数:
            user_id: 用户ID
            username: 用户名
            permission: 权限
            has_permission: 是否有权限
            context: 上下文信息
        """
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "username": username,
            "permission": permission,
            "has_permission": has_permission,
            "context": context,
            "ip_address": None,  # 可以从请求中获取
            "user_agent": None,  # 可以从请求中获取
        }
        
        self.audit_log.append(audit_entry)
        self.stats["audit_log_entries"] += 1
        
        # 限制日志大小
        if len(self.audit_log) > self.max_audit_log_size:
            self.audit_log.pop(0)
        
        self.stats["last_audit_time"] = datetime.now().isoformat()
    
    def get_audit_log(self, 
                     limit: int = 100,
                     offset: int = 0,
                     user_id: Optional[int] = None,
                     permission: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取审计日志
        
        参数:
            limit: 返回记录数限制
            offset: 偏移量
            user_id: 筛选用户ID（可选）
            permission: 筛选权限（可选）
            
        返回:
            List[Dict[str, Any]]: 审计日志
        """
        filtered_log = self.audit_log
        
        if user_id is not None:
            filtered_log = [entry for entry in filtered_log if entry["user_id"] == user_id]
        
        if permission is not None:
            filtered_log = [entry for entry in filtered_log if entry["permission"] == permission]
        
        # 按时间倒序排序
        filtered_log.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # 应用分页
        start_idx = offset
        end_idx = offset + limit
        paginated_log = filtered_log[start_idx:end_idx]
        
        return paginated_log
    
    def get_user_permissions(self, user: User) -> Dict[str, Any]:
        """获取用户权限信息
        
        参数:
            user: 用户对象
            
        返回:
            Dict[str, Any]: 权限信息
        """
        try:
            user_role = UserRole(user.role)
            role_permissions = self.role_permissions.get(user_role, set())
            
            extra_permissions = self._get_user_extra_permissions(user.id)
            all_permissions = role_permissions.union(extra_permissions)
            
            # 检查锁定状态
            is_locked = self._is_user_locked(user.id)
            
            # 获取失败尝试信息
            failed_attempts = self.failed_attempts.get(user.id, {})
            
            return {
                "user_id": user.id,
                "username": user.username,
                "role": user.role,
                "permissions": [perm.value for perm in all_permissions],
                "permission_count": len(all_permissions),
                "is_locked": is_locked,
                "failed_attempts": failed_attempts.get("count", 0),
                "lockout_until": failed_attempts.get("lockout_until"),
                "has_emergency_approval": True,  # 暂时假设有
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            self.logger.error(f"获取用户权限信息失败: {e}")
            return {
                "user_id": user.id,
                "username": user.username,
                "role": user.role,
                "permissions": [],
                "permission_count": 0,
                "is_locked": False,
                "error": str(e),
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        return {
            **self.stats,
            "audit_log_size": len(self.audit_log),
            "locked_users": sum(1 for data in self.failed_attempts.values() 
                              if data.get("lockout_until") and 
                              datetime.now() < data["lockout_until"]),
            "total_users_tracked": len(self.failed_attempts),
            "role_count": len(self.role_permissions),
            "permission_count": len(ModePermission),
            "timestamp": datetime.now().isoformat(),
        }
    
    def clear_audit_log(self):
        """清空审计日志"""
        self.audit_log.clear()
        self.stats["audit_log_entries"] = 0
        self.logger.info("审计日志已清空")
    
    def unlock_user(self, user_id: int) -> bool:
        """解锁用户
        
        参数:
            user_id: 用户ID
            
        返回:
            bool: 解锁是否成功
        """
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
            self.logger.info(f"用户已解锁: user_id={user_id}")
            return True
        
        return False
    
    def reset(self):
        """重置权限管理器"""
        self.logger.info("重置模式权限管理器")
        
        # 清空审计日志
        self.audit_log.clear()
        
        # 清空失败尝试记录
        self.failed_attempts.clear()
        
        # 重置统计信息
        self.stats = {
            "total_permission_checks": 0,
            "failed_permission_checks": 0,
            "audit_log_entries": 0,
            "anomaly_detections": 0,
            "lockouts_triggered": 0,
            "last_audit_time": None,
        }
        
        self.logger.info("模式权限管理器重置完成")


# FastAPI依赖项
def get_mode_permission_manager() -> ModePermissionManager:
    """获取模式权限管理器依赖"""
    # 这里可以使用全局单例或从应用状态获取
    # 暂时创建新实例
    return ModePermissionManager()


def require_mode_permission(permission: ModePermission):
    """模式权限检查装饰器
    
    参数:
        permission: 需要的权限
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取依赖
            request = kwargs.get('request')
            current_user = kwargs.get('current_user')
            db = kwargs.get('db')
            permission_manager = kwargs.get('permission_manager')
            
            if not all([request, current_user, permission_manager]):
                # 尝试从其他地方获取
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                    elif isinstance(arg, User):
                        current_user = arg
                    elif isinstance(arg, Session):
                        db = arg
                    elif isinstance(arg, ModePermissionManager):
                        permission_manager = arg
            
            if not permission_manager:
                permission_manager = get_mode_permission_manager()
            
            # 检查权限
            has_permission = permission_manager.check_permission(
                current_user, 
                permission,
                context={"endpoint": func.__name__, "method": request.method if request else "unknown"}
            )
            
            if not has_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"没有权限执行此操作: {permission.value}"
                )
            
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        
        return wrapper
    return decorator


# 特定权限的快捷装饰器
def require_mode_view(func: Callable):
    """需要模式查看权限"""
    return require_mode_permission(ModePermission.MODE_VIEW)(func)

def require_mode_switch(func: Callable):
    """需要模式切换权限"""
    return require_mode_permission(ModePermission.MODE_SWITCH)(func)

def require_mode_config_update(func: Callable):
    """需要模式配置更新权限"""
    return require_mode_permission(ModePermission.MODE_CONFIG_UPDATE)(func)

def require_mode_system_control(func: Callable):
    """需要系统级模式控制权限"""
    return require_mode_permission(ModePermission.MODE_SYSTEM_CONTROL)(func)


# 全局实例
_mode_permission_manager_instance = None


def get_global_mode_permission_manager(config: Optional[Dict[str, Any]] = None) -> ModePermissionManager:
    """获取全局模式权限管理器单例
    
    参数:
        config: 配置字典
        
    返回:
        ModePermissionManager: 模式权限管理器实例
    """
    global _mode_permission_manager_instance
    
    if _mode_permission_manager_instance is None:
        _mode_permission_manager_instance = ModePermissionManager(config)
    
    return _mode_permission_manager_instance


__all__ = [
    "ModePermissionManager",
    "get_global_mode_permission_manager",
    "ModePermission",
    "require_mode_permission",
    "require_mode_view",
    "require_mode_switch",
    "require_mode_config_update",
    "require_mode_system_control",
]