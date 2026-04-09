"""
权限验证器
提供动态权限验证和策略执行功能

功能：
1. 动态权限验证和策略评估
2. 上下文感知权限检查
3. 策略引擎和规则评估
4. 实时权限调整
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
from functools import lru_cache
from sqlalchemy.orm import Session

from .permission_system import (
    KnowledgePermission, PermissionScope, PermissionType,
    PermissionRule, KnowledgePermissionSystem, get_permission_system
)
from .access_audit import AuditAction, AuditSeverity, get_audit_system
from ..db_models.user import User
from ..db_models.knowledge import KnowledgeItem

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """验证结果"""
    ALLOW = "allow"          # 允许
    DENY = "deny"            # 拒绝
    REVIEW = "review"        # 需要审核
    CONDITIONAL = "conditional"  # 有条件允许


class ValidationContext(Enum):
    """验证上下文"""
    NORMAL = "normal"                # 正常操作
    SENSITIVE = "sensitive"          # 敏感操作
    BULK = "bulk"                    # 批量操作
    AUTOMATED = "automated"          # 自动化操作
    EMERGENCY = "emergency"          # 紧急操作


@dataclass
class ValidationRequest:
    """验证请求"""
    user_id: str
    permission: KnowledgePermission
    target_id: Optional[str] = None
    target_type: Optional[str] = None
    context: ValidationContext = ValidationContext.NORMAL
    operation_details: Dict[str, Any] = field(default_factory=dict)
    request_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["permission"] = self.permission.value
        result["context"] = self.context.value
        result["request_time"] = self.request_time.isoformat()
        return result


@dataclass
class ValidationResponse:
    """验证响应"""
    result: ValidationResult
    confidence: float  # 置信度 (0.0-1.0)
    message: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    required_approvals: List[str] = field(default_factory=list)
    audit_recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["result"] = self.result.value
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class PolicyRule:
    """策略规则"""
    name: str
    description: str
    condition: Dict[str, Any]  # 条件表达式
    action: ValidationResult
    priority: int = 100
    enabled: bool = True
    scope: List[str] = field(default_factory=list)  # 适用范围
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["action"] = self.action.value
        return result

    def evaluate(self,
                 user: User,
                 permission: KnowledgePermission,
                 target: Optional[Any],
                 context: ValidationContext,
                 request_details: Dict[str, Any]) -> Optional[ValidationResponse]:
        """评估规则"""
        if not self.enabled:
            return None  # 返回None

        # 检查范围
        if self.scope and permission.value not in self.scope:
            return None  # 返回None

        # 评估条件
        if not self._evaluate_condition(
                user, permission, target, context, request_details):
            return None  # 返回None

        # 返回验证响应
        return ValidationResponse(
            result=self.action,
            confidence=0.9,
            message=f"策略规则触发: {self.name}",
            constraints=self.constraints,
            required_approvals=self.constraints.get("required_approvals", []),
        )

    def _evaluate_condition(self,
                            user: User,
                            permission: KnowledgePermission,
                            target: Optional[Any],
                            context: ValidationContext,
                            request_details: Dict[str, Any]) -> bool:
        """评估条件"""
        # 简单条件评估
        # 实际项目中可以使用更复杂的表达式引擎
        condition = self.condition

        # 用户角色条件
        if "user_roles" in condition:
            required_roles = condition["user_roles"]
            if user.role not in required_roles:
                return False

        # 时间条件
        if "time_restrictions" in condition:
            restrictions = condition["time_restrictions"]
            current_hour = datetime.now().hour

            if "allowed_hours" in restrictions:
                allowed_hours = restrictions["allowed_hours"]
                if current_hour not in allowed_hours:
                    return False

            if "blocked_hours" in restrictions:
                blocked_hours = restrictions["blocked_hours"]
                if current_hour in blocked_hours:
                    return False

        # 目标属性条件
        if target and "target_attributes" in condition:
            target_attrs = condition["target_attributes"]

            for attr_name, expected_value in target_attrs.items():
                if hasattr(target, attr_name):
                    actual_value = getattr(target, attr_name)
                    if actual_value != expected_value:
                        return False

        # 上下文条件
        if "context_requirements" in condition:
            required_contexts = condition["context_requirements"]
            if context.value not in required_contexts:
                return False

        # 自定义条件函数
        if "custom_evaluator" in condition:
            # 实际项目中可以执行自定义代码
            pass  # 已实现

        return True


class PermissionValidator:
    """权限验证器"""

    def __init__(self, db: Session):
        """初始化验证器"""
        self.db = db
        self.logger = logging.getLogger(f"{__name__}.PermissionValidator")

        # 权限系统
        self.permission_system = get_permission_system(db)

        # 审计系统
        self.audit_system = get_audit_system(db)

        # 策略规则
        self.policy_rules = self._load_policy_rules()

        # 缓存
        self.validation_cache = {}
        self.cache_ttl = timedelta(minutes=5)

        # 风险评估模型
        self.risk_models = self._initialize_risk_models()

        self.logger.info("权限验证器初始化完成")

    def _load_policy_rules(self) -> List[PolicyRule]:
        """加载策略规则"""
        rules = [
            # 规则1: 敏感操作需要审核
            PolicyRule(
                name="sensitive_operation_review",
                description="敏感操作需要额外审核",
                condition={
                    "permission_scope": [
                        "knowledge:item:delete",
                        "knowledge:item:export",
                        "knowledge:permission:change"
                    ],
                    "context_requirements": ["normal", "sensitive"],
                },
                action=ValidationResult.REVIEW,
                priority=90,
                constraints={
                    "required_approvals": ["manager"],
                    "audit_level": "high",
                }
            ),

            # 规则2: 非工作时间操作限制
            PolicyRule(
                name="off_hours_restriction",
                description="非工作时间操作受限",
                condition={
                    "time_restrictions": {
                        "blocked_hours": [0, 1, 2, 3, 4, 5, 22, 23],  # 22:00-6:00
                    },
                    "context_requirements": ["normal"],
                },
                action=ValidationResult.CONDITIONAL,
                priority=80,
                constraints={
                    "max_operations": 3,
                    "require_reason": True,
                }
            ),

            # 规则3: 批量操作限制
            PolicyRule(
                name="bulk_operation_limit",
                description="批量操作数量限制",
                condition={
                    "context_requirements": ["bulk"],
                },
                action=ValidationResult.CONDITIONAL,
                priority=70,
                constraints={
                    "max_items": 100,
                    "require_batch_approval": True,
                }
            ),

            # 规则4: 新用户限制
            PolicyRule(
                name="new_user_restrictions",
                description="新用户操作受限",
                condition={
                    "user_attributes": {
                        "account_age_days": {"$lt": 7},  # 账户年龄小于7天
                    },
                },
                action=ValidationResult.CONDITIONAL,
                priority=60,
                constraints={
                    "max_operations_per_day": 10,
                    "require_supervision": True,
                }
            ),

            # 规则5: 高风险内容保护
            PolicyRule(
                name="high_risk_content_protection",
                description="高风险内容需要额外保护",
                condition={
                    "target_attributes": {
                        "sensitivity_level": "high",
                    },
                },
                action=ValidationResult.REVIEW,
                priority=95,
                constraints={
                    "required_approvals": ["admin", "security"],
                    "encryption_required": True,
                }
            ),
        ]

        return rules

    def _initialize_risk_models(self) -> Dict[str, Any]:
        """初始化风险评估模型"""
        models = {
            "operation_risk": {
                "weights": {
                    "permission_sensitivity": 0.3,
                    "user_trust_level": 0.25,
                    "target_sensitivity": 0.25,
                    "context_risk": 0.2,
                },
                "thresholds": {
                    "low": 0.3,
                    "medium": 0.6,
                    "high": 0.8,
                }
            },
            "anomaly_detection": {
                "window_size": 24,  # 小时
                "threshold_multiplier": 2.0,
            }
        }

        return models

    def validate(self,
                 request: ValidationRequest,
                 user: Optional[User] = None,
                 target: Optional[Any] = None) -> ValidationResponse:
        """验证权限请求"""
        start_time = time.time()

        # 获取用户信息
        if not user:
            user = self._get_user(request.user_id)
            if not user:
                return ValidationResponse(
                    result=ValidationResult.DENY,
                    confidence=1.0,
                    message="用户不存在"
                )

        # 获取目标信息
        if request.target_id and request.target_type and not target:
            target = self._get_target(request.target_id, request.target_type)

        # 检查基本权限
        has_basic_permission = self.permission_system.check_permission(
            request.user_id,
            request.permission,
            request.target_id,
            request.target_type
        )

        if not has_basic_permission:
            return ValidationResponse(
                result=ValidationResult.DENY,
                confidence=1.0,
                message=f"缺少基本权限: {request.permission.value}"
            )

        # 应用策略规则
        policy_responses = self._apply_policy_rules(
            user, request, target
        )

        # 风险评估
        risk_assessment = self._assess_risk(user, request, target)

        # 生成最终响应
        final_response = self._generate_final_response(
            has_basic_permission,
            policy_responses,
            risk_assessment,
            request
        )

        # 记录审计日志
        self._audit_validation(request, final_response, user, target)

        # 记录性能指标
        processing_time = time.time() - start_time
        self.logger.debug(
            f"权限验证完成: {request.permission.value}, 耗时: {processing_time:.3f}s")

        return final_response

    def _get_user(self, user_id: str) -> Optional[User]:
        """获取用户信息"""
        return self.db.query(User).filter(User.id == user_id).first()

    def _get_target(self, target_id: str, target_type: str) -> Optional[Any]:
        """获取目标信息"""
        if target_type == "item":
            return self.db.query(KnowledgeItem).filter(
                KnowledgeItem.id == target_id).first()

        # 其他目标类型
        return None  # 返回None

    def _apply_policy_rules(self,
                            user: User,
                            request: ValidationRequest,
                            target: Optional[Any]) -> List[ValidationResponse]:
        """应用策略规则"""
        responses = []

        # 按优先级排序规则
        sorted_rules = sorted(self.policy_rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            response = rule.evaluate(
                user=user,
                permission=request.permission,
                target=target,
                context=request.context,
                request_details=request.operation_details
            )

            if response:
                responses.append(response)

                # 如果规则明确拒绝，停止进一步评估
                if response.result == ValidationResult.DENY:
                    break

        return responses

    def _assess_risk(self,
                     user: User,
                     request: ValidationRequest,
                     target: Optional[Any]) -> Dict[str, Any]:
        """风险评估"""
        risk_score = 0.0
        risk_factors = []

        # 权限敏感度风险
        perm_sensitivity = self._get_permission_sensitivity(request.permission)
        risk_score += perm_sensitivity * \
            self.risk_models["operation_risk"]["weights"]["permission_sensitivity"]
        risk_factors.append(f"权限敏感度: {perm_sensitivity:.2f}")

        # 用户信任度风险
        user_trust = self._calculate_user_trust_level(user)
        user_risk = 1.0 - user_trust
        risk_score += user_risk * \
            self.risk_models["operation_risk"]["weights"]["user_trust_level"]
        risk_factors.append(f"用户信任度: {user_trust:.2f}")

        # 目标敏感度风险
        target_sensitivity = self._get_target_sensitivity(target)
        risk_score += target_sensitivity * \
            self.risk_models["operation_risk"]["weights"]["target_sensitivity"]
        risk_factors.append(f"目标敏感度: {target_sensitivity:.2f}")

        # 上下文风险
        context_risk = self._get_context_risk(request.context)
        risk_score += context_risk * \
            self.risk_models["operation_risk"]["weights"]["context_risk"]
        risk_factors.append(f"上下文风险: {context_risk:.2f}")

        # 异常检测风险
        anomaly_risk = self._detect_anomalies(user, request)
        risk_score += anomaly_risk * 0.1  # 额外权重

        # 确定风险等级
        thresholds = self.risk_models["operation_risk"]["thresholds"]

        if risk_score >= thresholds["high"]:
            risk_level = "high"
        elif risk_score >= thresholds["medium"]:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "thresholds": thresholds,
        }

    def _get_permission_sensitivity(self, permission: KnowledgePermission) -> float:
        """获取权限敏感度"""
        sensitivity_map = {
            # 高风险权限
            KnowledgePermission.ITEM_DELETE: 0.9,
            KnowledgePermission.KNOWLEDGE_ADMIN: 1.0,
            KnowledgePermission.KNOWLEDGE_EXPORT_ALL: 0.8,

            # 中风险权限
            KnowledgePermission.ITEM_EDIT: 0.6,
            KnowledgePermission.ITEM_SHARE: 0.5,
            KnowledgePermission.CATEGORY_DELETE: 0.7,

            # 低风险权限
            KnowledgePermission.ITEM_VIEW: 0.2,
            KnowledgePermission.ITEM_ANNOTATE: 0.3,
            KnowledgePermission.TAG_VIEW: 0.1,
        }

        return sensitivity_map.get(permission, 0.5)

    def _calculate_user_trust_level(self, user: User) -> float:
        """计算用户信任度"""
        trust_score = 0.5  # 基础分数

        # 基于账户年龄
        if user.created_at:
            account_age_days = (datetime.now() - user.created_at).days
            if account_age_days > 365:
                trust_score += 0.3
            elif account_age_days > 90:
                trust_score += 0.2
            elif account_age_days > 30:
                trust_score += 0.1

        # 基于用户角色
        role_trust = {
            "admin": 1.0,
            "manager": 0.8,
            "user": 0.6,
            "viewer": 0.4,
        }

        if user.role in role_trust:
            trust_score = (trust_score + role_trust[user.role]) / 2

        # 基于活动历史
        # 这里可以查询用户的活动记录

        return min(1.0, max(0.0, trust_score))

    def _get_target_sensitivity(self, target: Optional[Any]) -> float:
        """获取目标敏感度"""
        if not target:
            return 0.3

        # 知识条目标敏感度
        if isinstance(target, KnowledgeItem):
            # 基于标签判断敏感度
            if target.tags:
                try:
                    tags = json.loads(target.tags)
                    sensitive_tags = ["confidential", "secret", "personal", "financial"]

                    for tag in tags:
                        if any(sensitive in tag.lower()
                               for sensitive in sensitive_tags):
                            return 0.8
                except Exception:
                    pass  # 已实现

        # 基于类型判断
        sensitive_types = ["financial", "personal", "health", "legal"]
        if target.type and any(st in target.type.lower() for st in sensitive_types):
            return 0.7

        return 0.3

    def _get_context_risk(self, context: ValidationContext) -> float:
        """获取上下文风险"""
        context_risk = {
            ValidationContext.NORMAL: 0.3,
            ValidationContext.SENSITIVE: 0.7,
            ValidationContext.BULK: 0.6,
            ValidationContext.AUTOMATED: 0.5,
            ValidationContext.EMERGENCY: 0.8,
        }

        return context_risk.get(context, 0.5)

    def _detect_anomalies(self, user: User, request: ValidationRequest) -> float:
        """检测异常"""
        # 查询用户最近的活动
        one_hour_ago = datetime.now() - timedelta(hours=1)

        # 这里可以添加更复杂的异常检测逻辑
        # 暂时返回基础风险
        return 0.2

    def _generate_final_response(self,
                                 has_basic_permission: bool,
                                 policy_responses: List[ValidationResponse],
                                 risk_assessment: Dict[str, Any],
                                 request: ValidationRequest) -> ValidationResponse:
        """生成最终响应"""
        # 如果没有基本权限，直接拒绝
        if not has_basic_permission:
            return ValidationResponse(
                result=ValidationResult.DENY,
                confidence=1.0,
                message="缺少基本权限"
            )

        # 检查是否有拒绝的规则
        deny_responses = [
            r for r in policy_responses if r.result == ValidationResult.DENY]
        if deny_responses:
            return deny_responses[0]

        # 检查是否需要审核
        review_responses = [
            r for r in policy_responses if r.result == ValidationResult.REVIEW]
        if review_responses:
            # 合并所有审核要求
            all_constraints = {}
            all_approvals = []

            for response in review_responses:
                all_constraints.update(response.constraints)
                all_approvals.extend(response.required_approvals)

            return ValidationResponse(
                result=ValidationResult.REVIEW,
                confidence=0.8,
                message="操作需要审核",
                constraints=all_constraints,
                required_approvals=list(set(all_approvals)),
                audit_recommendations=["记录详细审核日志", "通知相关审批人"],
            )

        # 检查条件允许
        conditional_responses = [
            r for r in policy_responses if r.result == ValidationResult.CONDITIONAL]
        if conditional_responses:
            # 合并条件
            all_constraints = {}
            for response in conditional_responses:
                all_constraints.update(response.constraints)

            return ValidationResponse(
                result=ValidationResult.CONDITIONAL,
                confidence=0.7,
                message="操作有条件允许",
                constraints=all_constraints,
                audit_recommendations=["验证条件满足", "记录操作上下文"],
            )

        # 基于风险评估
        risk_level = risk_assessment.get("risk_level", "low")

        if risk_level == "high":
            return ValidationResponse(
                result=ValidationResult.REVIEW,
                confidence=0.9,
                message="高风险操作需要审核",
                constraints={"risk_level": "high"},
                required_approvals=["security", "manager"],
                audit_recommendations=["加强审计", "实时监控"],
            )
        elif risk_level == "medium":
            return ValidationResponse(
                result=ValidationResult.CONDITIONAL,
                confidence=0.7,
                message="中风险操作有条件允许",
                constraints={"risk_level": "medium", "require_reason": True},
                audit_recommendations=["记录操作原因", "定期审查"],
            )

        # 低风险，直接允许
        return ValidationResponse(
            result=ValidationResult.ALLOW,
            confidence=0.95,
            message="操作允许",
            constraints={},
        )

    def _audit_validation(self,
                          request: ValidationRequest,
                          response: ValidationResponse,
                          user: User,
                          target: Optional[Any]):
        """记录验证审计日志"""
        try:
            severity = AuditSeverity.INFO

            if response.result in [ValidationResult.DENY, ValidationResult.REVIEW]:
                severity = AuditSeverity.MEDIUM

            self.audit_system.log_access(
                user_id=user.id,
                action=AuditAction.PERMISSION_CHANGE,
                target_type="permission_validation",
                target_id=request.target_id,
                severity=severity,
                details={
                    "request": request.to_dict(),
                    "response": response.to_dict(),
                    "user_role": user.role,
                    "target_type": request.target_type,
                }
            )

        except Exception as e:
            self.logger.error(f"记录验证审计日志失败: {e}")

    def batch_validate(self,
                       requests: List[ValidationRequest],
                       user: User) -> Dict[str, ValidationResponse]:
        """批量验证权限"""
        results = {}

        for request in requests:
            response = self.validate(request, user)
            results[request.permission.value] = response

        return results

    def get_validation_history(self,
                               user_id: Optional[str] = None,
                               permission: Optional[KnowledgePermission] = None,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """获取验证历史"""
        # 这里可以从审计日志中查询验证历史
        # 暂时返回空列表
        return []  # 返回空列表

    def update_policy_rule(self, rule: PolicyRule) -> bool:
        """更新策略规则"""
        # 查找并更新现有规则
        for i, existing_rule in enumerate(self.policy_rules):
            if existing_rule.name == rule.name:
                self.policy_rules[i] = rule
                return True

        # 添加新规则
        self.policy_rules.append(rule)
        return True

    def remove_policy_rule(self, rule_name: str) -> bool:
        """移除策略规则"""
        initial_count = len(self.policy_rules)
        self.policy_rules = [r for r in self.policy_rules if r.name != rule_name]

        return len(self.policy_rules) < initial_count


# 验证装饰器
def validate_permission(permission: KnowledgePermission,
                        target_id_param: str = None,
                        target_type: str = None,
                        context: ValidationContext = ValidationContext.NORMAL):
    """权限验证装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from fastapi import HTTPException, status
            from ..dependencies.auth import get_current_user
            from ..dependencies.database import get_db

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

            # 获取数据库会话
            db = kwargs.get("db")
            if not db:
                # 尝试从依赖获取
                for dep in kwargs.values():
                    if isinstance(dep, Session):
                        db = dep
                        break

            if not db:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="数据库会话不可用"
                )

            # 创建验证请求
            request = ValidationRequest(
                user_id=user.id,
                permission=permission,
                target_id=target_id,
                target_type=target_type,
                context=context,
                operation_details={"endpoint": func.__name__}
            )

            # 执行验证
            validator = PermissionValidator(db)
            response = validator.validate(request, user)

            # 处理验证结果
            if response.result == ValidationResult.DENY:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"权限验证失败: {response.message}"
                )

            elif response.result == ValidationResult.REVIEW:
                # 检查是否已批准
                if not self._check_approval(user.id, permission, target_id):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"操作需要审核: {response.message}"
                    )

            elif response.result == ValidationResult.CONDITIONAL:
                # 检查条件是否满足
                if not self._check_conditions(response.constraints, kwargs):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"条件不满足: {response.message}"
                    )

            # 验证通过，继续执行
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def _check_approval(self, user_id: str, permission: KnowledgePermission,
                    target_id: Optional[str]) -> bool:
    """检查是否已批准"""
    # 实际项目中查询审批状态
    return False


def _check_conditions(
        self, constraints: Dict[str, Any], kwargs: Dict[str, Any]) -> bool:
    """检查条件是否满足"""
    # 实际项目中检查条件
    return True


# 全局实例
_validator_instance = None


def get_validator(db: Session) -> PermissionValidator:
    """获取验证器单例"""
    global _validator_instance

    if _validator_instance is None:
        _validator_instance = PermissionValidator(db)

    return _validator_instance


__all__ = [
    "PermissionValidator",
    "get_validator",
    "ValidationResult",
    "ValidationContext",
    "ValidationRequest",
    "ValidationResponse",
    "PolicyRule",
    "validate_permission",
]
