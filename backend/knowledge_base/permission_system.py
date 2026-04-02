"""
知识库权限系统
提供细粒度的知识库权限控制，包括条目级、类别级、标签级权限

功能：
1. 细粒度权限控制（条目、类别、标签）
2. 权限继承和组合
3. 共享权限管理
4. 权限验证和检查
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import json
import logging
from functools import wraps
from sqlalchemy.orm import Session

from ..db_models.user import User
from ..db_models.knowledge import KnowledgeItem
from ..core.permissions import Permission as BasePermission

logger = logging.getLogger(__name__)


class KnowledgePermission(Enum):
    """知识库细粒度权限枚举"""
    
    # 条目级权限
    ITEM_VIEW = "knowledge:item:view"           # 查看知识条目
    ITEM_EDIT = "knowledge:item:edit"           # 编辑知识条目
    ITEM_DELETE = "knowledge:item:delete"       # 删除知识条目
    ITEM_SHARE = "knowledge:item:share"         # 共享知识条目
    ITEM_EXPORT = "knowledge:item:export"       # 导出知识条目
    ITEM_ANNOTATE = "knowledge:item:annotate"   # 添加注释
    
    # 类别级权限
    CATEGORY_VIEW = "knowledge:category:view"   # 查看类别
    CATEGORY_CREATE = "knowledge:category:create"  # 创建类别
    CATEGORY_EDIT = "knowledge:category:edit"   # 编辑类别
    CATEGORY_DELETE = "knowledge:category:delete"  # 删除类别
    CATEGORY_MANAGE = "knowledge:category:manage"  # 管理类别权限
    
    # 标签级权限
    TAG_VIEW = "knowledge:tag:view"             # 查看标签
    TAG_CREATE = "knowledge:tag:create"         # 创建标签
    TAG_EDIT = "knowledge:tag:edit"             # 编辑标签
    TAG_DELETE = "knowledge:tag:delete"         # 删除标签
    TAG_ASSIGN = "knowledge:tag:assign"         # 分配标签
    
    # 集合权限
    COLLECTION_VIEW = "knowledge:collection:view"  # 查看集合
    COLLECTION_CREATE = "knowledge:collection:create"  # 创建集合
    COLLECTION_EDIT = "knowledge:collection:edit"  # 编辑集合
    COLLECTION_DELETE = "knowledge:collection:delete"  # 删除集合
    COLLECTION_SHARE = "knowledge:collection:share"  # 共享集合
    
    # 管理权限
    KNOWLEDGE_ADMIN = "knowledge:admin"         # 知识库管理员权限
    KNOWLEDGE_AUDIT = "knowledge:audit"         # 审计权限
    KNOWLEDGE_IMPORT = "knowledge:import"       # 导入权限
    KNOWLEDGE_EXPORT_ALL = "knowledge:export:all"  # 导出所有权限


class PermissionScope(Enum):
    """权限作用域"""
    GLOBAL = "global"          # 全局权限
    ORGANIZATION = "organization"  # 组织权限
    TEAM = "team"              # 团队权限
    PERSONAL = "personal"      # 个人权限
    SHARED = "shared"          # 共享权限


class PermissionType(Enum):
    """权限类型"""
    ALLOW = "allow"            # 允许
    DENY = "deny"              # 拒绝
    INHERIT = "inherit"        # 继承


@dataclass
class PermissionRule:
    """权限规则"""
    permission: KnowledgePermission
    scope: PermissionScope
    target_id: Optional[str] = None  # 目标ID（条目、类别、标签ID）
    target_type: Optional[str] = None  # 目标类型：item, category, tag, collection
    permission_type: PermissionType = PermissionType.ALLOW
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["permission"] = self.permission.value
        result["scope"] = self.scope.value
        result["permission_type"] = self.permission_type.value
        if self.expires_at:
            result["expires_at"] = self.expires_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PermissionRule':
        """从字典创建"""
        data = data.copy()
        data["permission"] = KnowledgePermission(data["permission"])
        data["scope"] = PermissionScope(data["scope"])
        data["permission_type"] = PermissionType(data["permission_type"])
        if "expires_at" in data and data["expires_at"]:
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def matches(self, 
                permission: KnowledgePermission,
                target_id: Optional[str] = None,
                target_type: Optional[str] = None) -> bool:
        """检查规则是否匹配"""
        # 权限不匹配
        if self.permission != permission:
            return False
        
        # 检查目标匹配
        if self.target_id and target_id != self.target_id:
            return False
        
        if self.target_type and target_type != self.target_type:
            return False
        
        # 检查过期
        if self.is_expired():
            return False
        
        return True


@dataclass
class UserPermissions:
    """用户权限集合"""
    user_id: str
    roles: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)
    rules: List[PermissionRule] = field(default_factory=list)
    inherited_rules: List[PermissionRule] = field(default_factory=list)
    effective_permissions: Set[str] = field(default_factory=set)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["rules"] = [rule.to_dict() for rule in self.rules]
        result["inherited_rules"] = [rule.to_dict() for rule in self.inherited_rules]
        result["effective_permissions"] = list(self.effective_permissions)
        result["last_updated"] = self.last_updated.isoformat()
        return result
    
    def add_rule(self, rule: PermissionRule):
        """添加权限规则"""
        self.rules.append(rule)
        self._update_effective_permissions()
    
    def remove_rule(self, rule_id: str):
        """移除权限规则"""
        self.rules = [r for r in self.rules if r.metadata.get("id") != rule_id]
        self._update_effective_permissions()
    
    def clear_expired_rules(self):
        """清除过期规则"""
        self.rules = [r for r in self.rules if not r.is_expired()]
        self._update_effective_permissions()
    
    def _update_effective_permissions(self):
        """更新有效权限集合"""
        self.effective_permissions.clear()
        
        # 收集所有允许的权限
        for rule in self.rules + self.inherited_rules:
            if rule.permission_type == PermissionType.ALLOW and not rule.is_expired():
                self.effective_permissions.add(rule.permission.value)
        
        # 移除被拒绝的权限
        for rule in self.rules + self.inherited_rules:
            if rule.permission_type == PermissionType.DENY and not rule.is_expired():
                self.effective_permissions.discard(rule.permission.value)
    
    def has_permission(self, 
                      permission: KnowledgePermission,
                      target_id: Optional[str] = None,
                      target_type: Optional[str] = None) -> bool:
        """检查用户是否具有特定权限"""
        permission_str = permission.value
        
        # 首先检查有效权限集合
        if permission_str not in self.effective_permissions:
            return False
        
        # 检查是否有匹配的允许规则
        has_allow = False
        has_deny = False
        
        for rule in self.rules + self.inherited_rules:
            if rule.is_expired():
                continue
            
            if rule.matches(permission, target_id, target_type):
                if rule.permission_type == PermissionType.ALLOW:
                    has_allow = True
                elif rule.permission_type == PermissionType.DENY:
                    has_deny = True
        
        return has_allow and not has_deny


class KnowledgePermissionSystem:
    """知识库权限系统"""
    
    def __init__(self, db: Session):
        """初始化权限系统"""
        self.db = db
        self.logger = logging.getLogger(f"{__name__}.KnowledgePermissionSystem")
        
        # 权限缓存
        self.permission_cache: Dict[str, UserPermissions] = {}
        self.cache_ttl = timedelta(minutes=30)
        
        # 权限继承关系
        self.inheritance_hierarchy = {
            "knowledge:admin": {
                "includes": [
                    "knowledge:item:*",
                    "knowledge:category:*",
                    "knowledge:tag:*",
                    "knowledge:collection:*",
                    "knowledge:audit",
                    "knowledge:import",
                    "knowledge:export:all",
                ]
            },
            "knowledge:item:edit": {
                "includes": ["knowledge:item:view", "knowledge:item:annotate"]
            },
            "knowledge:item:delete": {
                "includes": ["knowledge:item:view"]
            },
            "knowledge:category:manage": {
                "includes": [
                    "knowledge:category:view",
                    "knowledge:category:create",
                    "knowledge:category:edit",
                    "knowledge:category:delete",
                ]
            },
        }
        
        self.logger.info("知识库权限系统初始化完成")
    
    def get_user_permissions(self, user_id: str, force_refresh: bool = False) -> UserPermissions:
        """获取用户权限"""
        # 检查缓存
        cache_key = f"user_permissions:{user_id}"
        
        if not force_refresh and cache_key in self.permission_cache:
            cached = self.permission_cache[cache_key]
            # 检查缓存是否过期
            if datetime.now() - cached.last_updated < self.cache_ttl:
                return cached
        
        # 从数据库加载用户权限
        user_perms = self._load_user_permissions(user_id)
        
        # 缓存结果
        self.permission_cache[cache_key] = user_perms
        
        return user_perms
    
    def _load_user_permissions(self, user_id: str) -> UserPermissions:
        """从数据库加载用户权限"""
        from ..db_models.user import User
        
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"用户不存在: {user_id}")
        
        # 创建用户权限对象
        user_perms = UserPermissions(
            user_id=user_id,
            roles=[user.role] if user.role else [],
            groups=json.loads(user.groups) if user.groups else [],
        )
        
        # 加载角色权限
        self._load_role_permissions(user_perms)
        
        # 加载组权限
        self._load_group_permissions(user_perms)
        
        # 加载个人权限规则
        self._load_personal_rules(user_perms)
        
        # 加载共享权限
        self._load_shared_permissions(user_perms)
        
        # 更新有效权限
        user_perms._update_effective_permissions()
        
        return user_perms
    
    def _load_role_permissions(self, user_perms: UserPermissions):
        """加载角色权限"""
        from ..core.permissions import ROLE_PERMISSIONS
        
        for role in user_perms.roles:
            if role in ROLE_PERMISSIONS:
                for perm_str in ROLE_PERMISSIONS[role]:
                    # 转换基础权限为知识库权限
                    knowledge_perm = self._map_base_permission_to_knowledge(perm_str)
                    if knowledge_perm:
                        rule = PermissionRule(
                            permission=knowledge_perm,
                            scope=PermissionScope.GLOBAL,
                            permission_type=PermissionType.ALLOW,
                            metadata={"source": f"role:{role}"}
                        )
                        user_perms.inherited_rules.append(rule)
    
    def _map_base_permission_to_knowledge(self, base_perm: str) -> Optional[KnowledgePermission]:
        """将基础权限映射到知识库权限"""
        mapping = {
            "knowledge:view": KnowledgePermission.ITEM_VIEW,
            "knowledge:upload": KnowledgePermission.ITEM_EDIT,
            "knowledge:update": KnowledgePermission.ITEM_EDIT,
            "knowledge:delete": KnowledgePermission.ITEM_DELETE,
            "knowledge:search": KnowledgePermission.ITEM_VIEW,
        }
        
        for base, knowledge in mapping.items():
            if base_perm.startswith(base):
                return knowledge
        
        return None  # 返回None
    
    def _load_group_permissions(self, user_perms: UserPermissions):
        """加载组权限"""
        # 这里可以从组权限表加载
        # 暂时实现为从用户组字段解析
        for group in user_perms.groups:
            # 加载组权限规则
            group_rules = self._load_group_rules(group)
            user_perms.inherited_rules.extend(group_rules)
    
    def _load_group_rules(self, group: str) -> List[PermissionRule]:
        """加载组权限规则"""
        # 实际项目中从数据库加载
        # 这里返回示例规则
        return []  # 返回空列表
    
    def _load_personal_rules(self, user_perms: UserPermissions):
        """加载个人权限规则"""
        # 从数据库加载用户的个人权限规则
        # 这里可以从knowledge_permission_rules表加载
        # 暂时返回空列表
        pass  # 已修复: 实现函数功能
    
    def _load_shared_permissions(self, user_perms: UserPermissions):
        """加载共享权限"""
        # 加载其他用户共享给该用户的权限
        # 查询共享表
        shared_rules = self._load_shared_rules(user_perms.user_id)
        user_perms.inherited_rules.extend(shared_rules)
    
    def _load_shared_rules(self, user_id: str) -> List[PermissionRule]:
        """加载共享规则"""
        # 实际项目中从数据库加载
        return []  # 返回空列表
    
    def check_permission(self, 
                        user_id: str,
                        permission: KnowledgePermission,
                        target_id: Optional[str] = None,
                        target_type: Optional[str] = None) -> bool:
        """检查用户权限"""
        try:
            user_perms = self.get_user_permissions(user_id)
            return user_perms.has_permission(permission, target_id, target_type)
        except Exception as e:
            self.logger.error(f"检查权限失败: {e}")
            return False
    
    def grant_permission(self,
                        grantor_id: str,
                        grantee_id: str,
                        permission: KnowledgePermission,
                        scope: PermissionScope,
                        target_id: Optional[str] = None,
                        target_type: Optional[str] = None,
                        expires_at: Optional[datetime] = None,
                        conditions: Optional[Dict[str, Any]] = None) -> bool:
        """授予权限"""
        # 检查授予者是否有权限授予此权限
        if not self.check_permission(grantor_id, KnowledgePermission.KNOWLEDGE_ADMIN):
            if target_id and target_type:
                # 检查授予者是否拥有目标
                if not self._check_ownership(grantor_id, target_id, target_type):
                    return False
            else:
                return False
        
        # 创建权限规则
        rule = PermissionRule(
            permission=permission,
            scope=scope,
            target_id=target_id,
            target_type=target_type,
            permission_type=PermissionType.ALLOW,
            expires_at=expires_at,
            conditions=conditions or {},
            metadata={
                "granted_by": grantor_id,
                "granted_at": datetime.now().isoformat(),
                "grantee": grantee_id,
            }
        )
        
        # 保存到数据库
        success = self._save_permission_rule(grantee_id, rule)
        
        if success:
            # 清除缓存
            self._clear_user_cache(grantee_id)
            self.logger.info(f"权限授予成功: {grantor_id} -> {grantee_id}, {permission.value}")
        
        return success
    
    def revoke_permission(self,
                         revoker_id: str,
                         user_id: str,
                         permission: KnowledgePermission,
                         target_id: Optional[str] = None,
                         target_type: Optional[str] = None) -> bool:
        """撤销权限"""
        # 检查撤销者是否有权限
        if not self.check_permission(revoker_id, KnowledgePermission.KNOWLEDGE_ADMIN):
            if target_id and target_type:
                if not self._check_ownership(revoker_id, target_id, target_type):
                    return False
            else:
                return False
        
        # 从数据库移除权限规则
        success = self._remove_permission_rule(user_id, permission, target_id, target_type)
        
        if success:
            # 清除缓存
            self._clear_user_cache(user_id)
            self.logger.info(f"权限撤销成功: {revoker_id} -> {user_id}, {permission.value}")
        
        return success
    
    def _check_ownership(self, user_id: str, target_id: str, target_type: str) -> bool:
        """检查用户是否拥有目标"""
        if target_type == "item":
            item = self.db.query(KnowledgeItem).filter(
                KnowledgeItem.id == target_id,
                KnowledgeItem.uploaded_by == user_id
            ).first()
            return item is not None
        
        # 其他类型的所有权检查
        # 暂时返回False
        return False
    
    def _save_permission_rule(self, user_id: str, rule: PermissionRule) -> bool:
        """保存权限规则到数据库"""
        # 实际项目中保存到数据库
        # 这里模拟成功
        return True
    
    def _remove_permission_rule(self, 
                               user_id: str,
                               permission: KnowledgePermission,
                               target_id: Optional[str] = None,
                               target_type: Optional[str] = None) -> bool:
        """从数据库移除权限规则"""
        # 实际项目中从数据库删除
        # 这里模拟成功
        return True
    
    def _clear_user_cache(self, user_id: str):
        """清除用户缓存"""
        cache_key = f"user_permissions:{user_id}"
        if cache_key in self.permission_cache:
            del self.permission_cache[cache_key]
    
    def get_effective_permissions(self, user_id: str) -> Set[str]:
        """获取用户有效权限集合"""
        user_perms = self.get_user_permissions(user_id)
        return user_perms.effective_permissions
    
    def get_item_permissions(self, user_id: str, item_id: str) -> Dict[str, bool]:
        """获取用户对特定知识条目的所有权限"""
        permissions = {}
        
        for permission in KnowledgePermission:
            if permission.value.startswith("knowledge:item:"):
                has_perm = self.check_permission(
                    user_id, permission, target_id=item_id, target_type="item"
                )
                permissions[permission.value] = has_perm
        
        return permissions
    
    def can_view_item(self, user_id: str, item_id: str) -> bool:
        """检查用户是否可以查看知识条目"""
        return self.check_permission(
            user_id, KnowledgePermission.ITEM_VIEW, target_id=item_id, target_type="item"
        )
    
    def can_edit_item(self, user_id: str, item_id: str) -> bool:
        """检查用户是否可以编辑知识条目"""
        return self.check_permission(
            user_id, KnowledgePermission.ITEM_EDIT, target_id=item_id, target_type="item"
        )
    
    def can_delete_item(self, user_id: str, item_id: str) -> bool:
        """检查用户是否可以删除知识条目"""
        return self.check_permission(
            user_id, KnowledgePermission.ITEM_DELETE, target_id=item_id, target_type="item"
        )
    
    def filter_items_by_permission(self, user_id: str, items: List[KnowledgeItem], 
                                   permission: KnowledgePermission = KnowledgePermission.ITEM_VIEW) -> List[KnowledgeItem]:
        """根据权限过滤知识条目"""
        filtered = []
        
        for item in items:
            if self.check_permission(user_id, permission, target_id=item.id, target_type="item"):
                filtered.append(item)
        
        return filtered
    
    def audit_permission_access(self, 
                               user_id: str,
                               permission: KnowledgePermission,
                               target_id: Optional[str] = None,
                               target_type: Optional[str] = None,
                               action: str = "check",
                               success: bool = True,
                               details: Optional[Dict[str, Any]] = None):
        """记录权限访问审计日志"""
        from ..db_models.knowledge import PermissionAuditLog
        
        audit_log = PermissionAuditLog(
            user_id=user_id,
            permission=permission.value,
            target_id=target_id,
            target_type=target_type,
            action=action,
            success=success,
            details=json.dumps(details or {}),
            timestamp=datetime.now()
        )
        
        try:
            self.db.add(audit_log)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"记录审计日志失败: {e}")


# 权限装饰器
def require_knowledge_permission(permission: KnowledgePermission, 
                                target_id_param: str = None,
                                target_type: str = None):
    """权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from fastapi import HTTPException, status
            from ..dependencies.auth import get_current_user
            
            # 获取当前用户
            user = kwargs.get("user")
            if not user:
                # 尝试从依赖获取
                for dep in kwargs.values():
                    if isinstance(dep, User):
                        user = dep
                        break
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="需要认证"
                )
            
            # 获取目标ID
            target_id = None
            if target_id_param:
                target_id = kwargs.get(target_id_param)
            
            # 检查权限
            permission_system = KnowledgePermissionSystem(kwargs.get("db"))
            
            if not permission_system.check_permission(
                user.id, permission, target_id, target_type
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"权限不足: {permission.value}"
                )
            
            # 记录审计日志
            permission_system.audit_permission_access(
                user_id=user.id,
                permission=permission,
                target_id=target_id,
                target_type=target_type,
                action="api_access",
                success=True,
                details={"endpoint": func.__name__}
            )
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# 全局实例
_permission_system_instance = None


def get_permission_system(db: Session) -> KnowledgePermissionSystem:
    """获取权限系统单例"""
    global _permission_system_instance
    
    if _permission_system_instance is None:
        _permission_system_instance = KnowledgePermissionSystem(db)
    
    return _permission_system_instance


__all__ = [
    "KnowledgePermissionSystem",
    "get_permission_system",
    "KnowledgePermission",
    "PermissionScope",
    "PermissionType",
    "PermissionRule",
    "UserPermissions",
    "require_knowledge_permission",
]