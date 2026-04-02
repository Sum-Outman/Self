"""
权限管理系统
定义用户角色和权限，提供权限检查功能
"""

from enum import Enum
from typing import Set, List, Dict, Any
from functools import wraps
from fastapi import HTTPException, status, Depends
from sqlalchemy.orm import Session

from ..dependencies.database import get_db
from ..db_models.user import User
from ..dependencies.auth import get_current_user


class Permission(Enum):
    """权限枚举"""
    
    # 用户管理权限
    USER_VIEW = "user:view"              # 查看用户列表
    USER_CREATE = "user:create"          # 创建用户
    USER_UPDATE = "user:update"          # 更新用户信息
    USER_DELETE = "user:delete"          # 删除用户
    USER_MANAGE_ROLES = "user:manage_roles"  # 管理用户角色
    
    # API密钥管理权限
    APIKEY_VIEW = "apikey:view"          # 查看API密钥
    APIKEY_CREATE = "apikey:create"      # 创建API密钥
    APIKEY_UPDATE = "apikey:update"      # 更新API密钥
    APIKEY_DELETE = "apikey:delete"      # 删除API密钥
    
    # 机器人管理权限
    ROBOT_VIEW = "robot:view"            # 查看机器人
    ROBOT_CREATE = "robot:create"        # 创建机器人
    ROBOT_UPDATE = "robot:update"        # 更新机器人
    ROBOT_DELETE = "robot:delete"        # 删除机器人
    ROBOT_CONTROL = "robot:control"      # 控制机器人
    ROBOT_TRAIN = "robot:train"          # 训练机器人
    ROBOT_DEBUG = "robot:debug"          # 调试机器人
    
    # 知识库权限
    KNOWLEDGE_VIEW = "knowledge:view"    # 查看知识库
    KNOWLEDGE_UPLOAD = "knowledge:upload" # 上传知识
    KNOWLEDGE_UPDATE = "knowledge:update" # 更新知识
    KNOWLEDGE_DELETE = "knowledge:delete" # 删除知识
    KNOWLEDGE_SEARCH = "knowledge:search" # 搜索知识
    
    # 训练管理权限
    TRAINING_VIEW = "training:view"      # 查看训练任务
    TRAINING_START = "training:start"    # 启动训练
    TRAINING_STOP = "training:stop"      # 停止训练
    TRAINING_DELETE = "training:delete"  # 删除训练任务
    
    # 系统管理权限
    SYSTEM_MONITOR = "system:monitor"    # 监控系统状态
    SYSTEM_CONFIG = "system:config"      # 配置系统
    SYSTEM_BACKUP = "system:backup"      # 系统备份
    SYSTEM_RESTORE = "system:restore"    # 系统恢复
    
    # 机器人市场权限
    MARKET_VIEW = "market:view"          # 查看市场
    MARKET_UPLOAD = "market:upload"      # 上传到市场
    MARKET_DOWNLOAD = "market:download"  # 从市场下载
    MARKET_RATE = "market:rate"          # 评分机器人
    MARKET_COMMENT = "market:comment"    # 评论机器人
    
    # 管理员专属权限
    ADMIN_ALL = "admin:all"              # 所有权限


class UserRole(Enum):
    """用户角色枚举"""
    
    VIEWER = "viewer"      # 观察者：只读访问
    USER = "user"         # 普通用户：基本操作权限
    MANAGER = "manager"   # 经理：管理用户和内容
    ADMIN = "admin"       # 管理员：所有权限


# 角色权限映射
ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    UserRole.VIEWER.value: {
        Permission.USER_VIEW.value,
        Permission.ROBOT_VIEW.value,
        Permission.KNOWLEDGE_VIEW.value,
        Permission.KNOWLEDGE_SEARCH.value,
        Permission.TRAINING_VIEW.value,
        Permission.MARKET_VIEW.value,
    },
    
    UserRole.USER.value: {
        Permission.USER_VIEW.value,
        Permission.USER_UPDATE.value,
        Permission.APIKEY_VIEW.value,
        Permission.APIKEY_CREATE.value,
        Permission.APIKEY_UPDATE.value,
        Permission.APIKEY_DELETE.value,
        Permission.ROBOT_VIEW.value,
        Permission.ROBOT_CREATE.value,
        Permission.ROBOT_UPDATE.value,
        Permission.ROBOT_DELETE.value,
        Permission.ROBOT_CONTROL.value,
        Permission.KNOWLEDGE_VIEW.value,
        Permission.KNOWLEDGE_UPLOAD.value,
        Permission.KNOWLEDGE_SEARCH.value,
        Permission.TRAINING_VIEW.value,
        Permission.TRAINING_START.value,
        Permission.MARKET_VIEW.value,
        Permission.MARKET_DOWNLOAD.value,
        Permission.MARKET_RATE.value,
        Permission.MARKET_COMMENT.value,
    },
    
    UserRole.MANAGER.value: {
        Permission.USER_VIEW.value,
        Permission.USER_UPDATE.value,
        Permission.USER_MANAGE_ROLES.value,
        Permission.APIKEY_VIEW.value,
        Permission.APIKEY_CREATE.value,
        Permission.APIKEY_UPDATE.value,
        Permission.APIKEY_DELETE.value,
        Permission.ROBOT_VIEW.value,
        Permission.ROBOT_CREATE.value,
        Permission.ROBOT_UPDATE.value,
        Permission.ROBOT_DELETE.value,
        Permission.ROBOT_CONTROL.value,
        Permission.ROBOT_TRAIN.value,
        Permission.KNOWLEDGE_VIEW.value,
        Permission.KNOWLEDGE_UPLOAD.value,
        Permission.KNOWLEDGE_UPDATE.value,
        Permission.KNOWLEDGE_DELETE.value,
        Permission.KNOWLEDGE_SEARCH.value,
        Permission.TRAINING_VIEW.value,
        Permission.TRAINING_START.value,
        Permission.TRAINING_STOP.value,
        Permission.MARKET_VIEW.value,
        Permission.MARKET_UPLOAD.value,
        Permission.MARKET_DOWNLOAD.value,
        Permission.MARKET_RATE.value,
        Permission.MARKET_COMMENT.value,
        Permission.SYSTEM_MONITOR.value,
    },
    
    UserRole.ADMIN.value: {
        Permission.ADMIN_ALL.value,
        Permission.USER_VIEW.value,
        Permission.USER_CREATE.value,
        Permission.USER_UPDATE.value,
        Permission.USER_DELETE.value,
        Permission.USER_MANAGE_ROLES.value,
        Permission.APIKEY_VIEW.value,
        Permission.APIKEY_CREATE.value,
        Permission.APIKEY_UPDATE.value,
        Permission.APIKEY_DELETE.value,
        Permission.ROBOT_VIEW.value,
        Permission.ROBOT_CREATE.value,
        Permission.ROBOT_UPDATE.value,
        Permission.ROBOT_DELETE.value,
        Permission.ROBOT_CONTROL.value,
        Permission.ROBOT_TRAIN.value,
        Permission.ROBOT_DEBUG.value,
        Permission.KNOWLEDGE_VIEW.value,
        Permission.KNOWLEDGE_UPLOAD.value,
        Permission.KNOWLEDGE_UPDATE.value,
        Permission.KNOWLEDGE_DELETE.value,
        Permission.KNOWLEDGE_SEARCH.value,
        Permission.TRAINING_VIEW.value,
        Permission.TRAINING_START.value,
        Permission.TRAINING_STOP.value,
        Permission.TRAINING_DELETE.value,
        Permission.SYSTEM_MONITOR.value,
        Permission.SYSTEM_CONFIG.value,
        Permission.SYSTEM_BACKUP.value,
        Permission.SYSTEM_RESTORE.value,
        Permission.MARKET_VIEW.value,
        Permission.MARKET_UPLOAD.value,
        Permission.MARKET_DOWNLOAD.value,
        Permission.MARKET_RATE.value,
        Permission.MARKET_COMMENT.value,
    }
}


class PermissionManager:
    """权限管理器"""
    
    @staticmethod
    def has_permission(user_role: str, permission: Permission) -> bool:
        """检查用户角色是否具有指定权限"""
        if user_role not in ROLE_PERMISSIONS:
            return False
        
        permissions_set = ROLE_PERMISSIONS[user_role]
        
        # 如果用户拥有admin:all权限，则允许所有操作
        if Permission.ADMIN_ALL.value in permissions_set:
            return True
        
        return permission.value in permissions_set
    
    @staticmethod
    def get_user_permissions(user_role: str) -> List[str]:
        """获取用户角色的所有权限"""
        if user_role not in ROLE_PERMISSIONS:
            return []  # 返回空列表
        
        permissions_set = ROLE_PERMISSIONS[user_role]
        
        # 如果用户拥有admin:all权限，返回所有权限
        if Permission.ADMIN_ALL.value in permissions_set:
            return [p.value for p in Permission]
        
        return list(permissions_set)
    
    @staticmethod
    def check_permission(user: User, permission: Permission, db: Session = None) -> bool:
        """检查用户是否具有指定权限"""
        user_role = user.role or "user"
        
        # 兼容旧的is_admin字段
        if user.is_admin:
            return True
        
        return PermissionManager.has_permission(user_role, permission)
    
    @staticmethod
    def require_permission(permission: Permission):
        """FastAPI依赖项：要求特定权限"""
        def permission_dependency(
            user: User = Depends(get_current_user),
            db: Session = Depends(get_db),
        ):
            if not PermissionManager.check_permission(user, permission, db):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="权限不足"
                )
            return user
        
        return permission_dependency


# 快捷依赖项定义
# 用户管理权限
require_user_view = PermissionManager.require_permission(Permission.USER_VIEW)
require_user_create = PermissionManager.require_permission(Permission.USER_CREATE)
require_user_update = PermissionManager.require_permission(Permission.USER_UPDATE)
require_user_delete = PermissionManager.require_permission(Permission.USER_DELETE)
require_user_manage_roles = PermissionManager.require_permission(Permission.USER_MANAGE_ROLES)

# API密钥管理权限
require_apikey_view = PermissionManager.require_permission(Permission.APIKEY_VIEW)
require_apikey_create = PermissionManager.require_permission(Permission.APIKEY_CREATE)
require_apikey_update = PermissionManager.require_permission(Permission.APIKEY_UPDATE)
require_apikey_delete = PermissionManager.require_permission(Permission.APIKEY_DELETE)

# 机器人管理权限
require_robot_view = PermissionManager.require_permission(Permission.ROBOT_VIEW)
require_robot_create = PermissionManager.require_permission(Permission.ROBOT_CREATE)
require_robot_update = PermissionManager.require_permission(Permission.ROBOT_UPDATE)
require_robot_delete = PermissionManager.require_permission(Permission.ROBOT_DELETE)
require_robot_control = PermissionManager.require_permission(Permission.ROBOT_CONTROL)
require_robot_train = PermissionManager.require_permission(Permission.ROBOT_TRAIN)
require_robot_debug = PermissionManager.require_permission(Permission.ROBOT_DEBUG)

# 知识库权限
require_knowledge_view = PermissionManager.require_permission(Permission.KNOWLEDGE_VIEW)
require_knowledge_upload = PermissionManager.require_permission(Permission.KNOWLEDGE_UPLOAD)
require_knowledge_update = PermissionManager.require_permission(Permission.KNOWLEDGE_UPDATE)
require_knowledge_delete = PermissionManager.require_permission(Permission.KNOWLEDGE_DELETE)
require_knowledge_search = PermissionManager.require_permission(Permission.KNOWLEDGE_SEARCH)

# 训练管理权限
require_training_view = PermissionManager.require_permission(Permission.TRAINING_VIEW)
require_training_start = PermissionManager.require_permission(Permission.TRAINING_START)
require_training_stop = PermissionManager.require_permission(Permission.TRAINING_STOP)
require_training_delete = PermissionManager.require_permission(Permission.TRAINING_DELETE)

# 系统管理权限
require_system_monitor = PermissionManager.require_permission(Permission.SYSTEM_MONITOR)
require_system_config = PermissionManager.require_permission(Permission.SYSTEM_CONFIG)
require_system_backup = PermissionManager.require_permission(Permission.SYSTEM_BACKUP)
require_system_restore = PermissionManager.require_permission(Permission.SYSTEM_RESTORE)

# 机器人市场权限
require_market_view = PermissionManager.require_permission(Permission.MARKET_VIEW)
require_market_upload = PermissionManager.require_permission(Permission.MARKET_UPLOAD)
require_market_download = PermissionManager.require_permission(Permission.MARKET_DOWNLOAD)
require_market_rate = PermissionManager.require_permission(Permission.MARKET_RATE)
require_market_comment = PermissionManager.require_permission(Permission.MARKET_COMMENT)

# 管理员权限
require_admin = PermissionManager.require_permission(Permission.ADMIN_ALL)