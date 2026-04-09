"""
知识库访问审计系统
记录和分析知识库访问日志，提供安全审计和异常检测功能

功能：
1. 访问日志记录和管理
2. 安全审计和合规性检查
3. 异常访问检测
4. 审计报告生成
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ..db_models.knowledge import AccessAuditLog
from ..db_models.user import User

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """审计动作枚举"""

    VIEW = "view"  # 查看
    CREATE = "create"  # 创建
    UPDATE = "update"  # 更新
    DELETE = "delete"  # 删除
    SEARCH = "search"  # 搜索
    EXPORT = "export"  # 导出
    IMPORT = "import"  # 导入
    SHARE = "share"  # 共享
    PERMISSION_CHANGE = "permission_change"  # 权限变更
    LOGIN = "login"  # 登录
    LOGOUT = "logout"  # 登出


class AuditSeverity(Enum):
    """审计严重程度"""

    INFO = "info"  # 信息
    LOW = "low"  # 低风险
    MEDIUM = "medium"  # 中风险
    HIGH = "high"  # 高风险
    CRITICAL = "critical"  # 严重风险


@dataclass
class AuditRecord:
    """审计记录"""

    id: str
    user_id: str
    action: AuditAction
    target_type: str
    target_id: Optional[str] = None
    target_name: Optional[str] = None
    severity: AuditSeverity = AuditSeverity.INFO
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["action"] = self.action.value
        result["severity"] = self.severity.value
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class AuditStatistics:
    """审计统计"""

    total_records: int = 0
    records_by_action: Dict[str, int] = field(default_factory=dict)
    records_by_user: Dict[str, int] = field(default_factory=dict)
    records_by_severity: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    average_records_per_day: float = 0.0
    peak_hour: Optional[int] = None
    most_active_user: Optional[str] = None
    most_accessed_item: Optional[str] = None
    suspicious_activities: int = 0
    time_period_days: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class AnomalyDetectionResult:
    """异常检测结果"""

    anomaly_type: str
    confidence: float
    description: str
    affected_users: List[str]
    affected_items: List[str]
    severity: AuditSeverity
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["severity"] = self.severity.value
        result["timestamp"] = self.timestamp.isoformat()
        return result


class AccessAuditSystem:
    """访问审计系统"""

    def __init__(self, db: Session):
        """初始化审计系统"""
        self.db = db
        self.logger = logging.getLogger(f"{__name__}.AccessAuditSystem")

        # 异常检测配置
        self.anomaly_config = {
            "failed_login_threshold": 5,  # 连续失败登录次数阈值
            "access_rate_threshold": 100,  # 每小时访问次数阈值
            "unusual_hour_access_threshold": 10,  # 非工作时间访问阈值
            "data_export_threshold": 50,  # 单次导出数据量阈值
            "sensitive_operation_threshold": 3,  # 敏感操作次数阈值
        }

        self.logger.info("访问审计系统初始化完成")

    def log_access(
        self,
        user_id: str,
        action: AuditAction,
        target_type: str,
        target_id: Optional[str] = None,
        target_name: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> bool:
        """记录访问日志"""
        try:
            audit_log = AccessAuditLog(
                user_id=user_id,
                action=action.value,
                target_type=target_type,
                target_id=target_id,
                target_name=target_name,
                severity=severity.value,
                ip_address=ip_address,
                user_agent=user_agent,
                details=json.dumps(details or {}),
                success=success,
                error_message=error_message,
                timestamp=datetime.now(),
            )

            self.db.add(audit_log)
            self.db.commit()

            self.logger.debug(f"审计日志记录成功: {user_id} -> {action.value}")

            # 检查异常
            self._check_anomalies(user_id, action, target_id, target_type, ip_address)

            return True

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"记录审计日志失败: {e}")
            return False

    def _check_anomalies(
        self,
        user_id: str,
        action: AuditAction,
        target_id: Optional[str],
        target_type: str,
        ip_address: Optional[str],
    ):
        """检查异常行为"""
        anomalies = []

        # 检查失败登录
        if action == AuditAction.LOGIN and not self._is_successful():
            anomalies.extend(self._detect_failed_login_anomaly(user_id, ip_address))

        # 检查访问频率
        anomalies.extend(self._detect_access_rate_anomaly(user_id, ip_address))

        # 检查非工作时间访问
        anomalies.extend(self._detect_unusual_hour_access_anomaly(user_id))

        # 检查敏感操作
        if action in [
            AuditAction.DELETE,
            AuditAction.EXPORT,
            AuditAction.PERMISSION_CHANGE,
        ]:
            anomalies.extend(self._detect_sensitive_operation_anomaly(user_id, action))

        # 记录检测到的异常
        for anomaly in anomalies:
            self._log_anomaly(anomaly)

    def _detect_failed_login_anomaly(
        self, user_id: str, ip_address: Optional[str]
    ) -> List[AnomalyDetectionResult]:
        """检测失败登录异常"""
        threshold = self.anomaly_config["failed_login_threshold"]

        # 查询最近1小时内的失败登录
        one_hour_ago = datetime.now() - timedelta(hours=1)

        failed_logins = (
            self.db.query(AccessAuditLog)
            .filter(
                and_(
                    AccessAuditLog.user_id == user_id,
                    AccessAuditLog.action == AuditAction.LOGIN.value,
                    AccessAuditLog.success == False,
                    AccessAuditLog.timestamp >= one_hour_ago,
                )
            )
            .count()
        )

        if failed_logins >= threshold:
            return [
                AnomalyDetectionResult(
                    anomaly_type="failed_login_anomaly",
                    confidence=min(1.0, failed_logins / threshold),
                    description=f"用户 {user_id} 在1小时内失败登录 {failed_logins} 次",
                    affected_users=[user_id],
                    affected_items=[],
                    severity=AuditSeverity.HIGH,
                    recommendations=[
                        "检查账户安全",
                        "启用双因素认证",
                        "锁定可疑账户",
                    ],
                )
            ]

        return []  # 返回空列表

    def _detect_access_rate_anomaly(
        self, user_id: str, ip_address: Optional[str]
    ) -> List[AnomalyDetectionResult]:
        """检测访问频率异常"""
        threshold = self.anomaly_config["access_rate_threshold"]

        # 查询最近1小时内的访问记录
        one_hour_ago = datetime.now() - timedelta(hours=1)

        access_count = (
            self.db.query(AccessAuditLog)
            .filter(
                and_(
                    AccessAuditLog.user_id == user_id,
                    AccessAuditLog.timestamp >= one_hour_ago,
                    AccessAuditLog.action.in_(
                        [AuditAction.VIEW.value, AuditAction.SEARCH.value]
                    ),
                )
            )
            .count()
        )

        if access_count >= threshold:
            return [
                AnomalyDetectionResult(
                    anomaly_type="access_rate_anomaly",
                    confidence=min(1.0, access_count / threshold),
                    description=f"用户 {user_id} 在1小时内访问 {access_count} 次，超过阈值 {threshold}",
                    affected_users=[user_id],
                    affected_items=[],
                    severity=AuditSeverity.MEDIUM,
                    recommendations=[
                        "检查用户行为是否正常",
                        "限制访问频率",
                        "发送安全通知",
                    ],
                )
            ]

        return []  # 返回空列表

    def _detect_unusual_hour_access_anomaly(
        self, user_id: str
    ) -> List[AnomalyDetectionResult]:
        """检测非工作时间访问异常"""
        threshold = self.anomaly_config["unusual_hour_access_threshold"]

        # 定义工作时间（9:00-18:00）
        current_hour = datetime.now().hour

        if 9 <= current_hour <= 18:
            return []  # 返回空列表  # 工作时间

        # 查询最近24小时内的非工作时间访问
        twenty_four_hours_ago = datetime.now() - timedelta(hours=24)

        unusual_access = (
            self.db.query(AccessAuditLog)
            .filter(
                and_(
                    AccessAuditLog.user_id == user_id,
                    AccessAuditLog.timestamp >= twenty_four_hours_ago,
                    or_(
                        func.extract("hour", AccessAuditLog.timestamp) < 9,
                        func.extract("hour", AccessAuditLog.timestamp) > 18,
                    ),
                )
            )
            .count()
        )

        if unusual_access >= threshold:
            return [
                AnomalyDetectionResult(
                    anomaly_type="unusual_hour_access_anomaly",
                    confidence=min(1.0, unusual_access / threshold),
                    description=f"用户 {user_id} 在非工作时间访问 {unusual_access} 次",
                    affected_users=[user_id],
                    affected_items=[],
                    severity=AuditSeverity.LOW,
                    recommendations=[
                        "检查访问是否合理",
                        "记录详细访问日志",
                        "发送安全提醒",
                    ],
                )
            ]

        return []  # 返回空列表

    def _detect_sensitive_operation_anomaly(
        self, user_id: str, action: AuditAction
    ) -> List[AnomalyDetectionResult]:
        """检测敏感操作异常"""
        threshold = self.anomaly_config["sensitive_operation_threshold"]

        # 查询最近1小时内的敏感操作
        one_hour_ago = datetime.now() - timedelta(hours=1)

        sensitive_ops = (
            self.db.query(AccessAuditLog)
            .filter(
                and_(
                    AccessAuditLog.user_id == user_id,
                    AccessAuditLog.action == action.value,
                    AccessAuditLog.timestamp >= one_hour_ago,
                )
            )
            .count()
        )

        if sensitive_ops >= threshold:
            action_name = action.value.replace("_", " ").title()

            return [
                AnomalyDetectionResult(
                    anomaly_type="sensitive_operation_anomaly",
                    confidence=min(1.0, sensitive_ops / threshold),
                    description=f"用户 {user_id} 在1小时内执行 {sensitive_ops} 次{action_name}操作",
                    affected_users=[user_id],
                    affected_items=[],
                    severity=AuditSeverity.HIGH,
                    recommendations=[
                        "立即审核操作日志",
                        "暂停用户权限",
                        "通知安全团队",
                    ],
                )
            ]

        return []  # 返回空列表

    def _log_anomaly(self, anomaly: AnomalyDetectionResult):
        """记录异常检测结果"""
        try:
            # 将异常记录到审计日志
            self.log_access(
                user_id="system",
                action=AuditAction.PERMISSION_CHANGE,
                target_type="security",
                severity=anomaly.severity,
                details=anomaly.to_dict(),
                success=True,
                error_message=None,
            )

            self.logger.warning(
                f"检测到异常: {anomaly.anomaly_type} - {anomaly.description}"
            )

        except Exception as e:
            self.logger.error(f"记录异常失败: {e}")

    def get_statistics(self, days: int = 30) -> AuditStatistics:
        """获取审计统计信息"""
        start_date = datetime.now() - timedelta(days=days)

        # 查询总记录数
        total_records = (
            self.db.query(AccessAuditLog)
            .filter(AccessAuditLog.timestamp >= start_date)
            .count()
        )

        # 按动作统计
        records_by_action = {}
        actions = (
            self.db.query(AccessAuditLog.action, func.count(AccessAuditLog.id))
            .filter(AccessAuditLog.timestamp >= start_date)
            .group_by(AccessAuditLog.action)
            .all()
        )

        for action, count in actions:
            records_by_action[action] = count

        # 按用户统计
        records_by_user = {}
        users = (
            self.db.query(AccessAuditLog.user_id, func.count(AccessAuditLog.id))
            .filter(AccessAuditLog.timestamp >= start_date)
            .group_by(AccessAuditLog.user_id)
            .all()
        )

        for user_id, count in users:
            records_by_user[user_id] = count

        # 按严重程度统计
        records_by_severity = {}
        severities = (
            self.db.query(AccessAuditLog.severity, func.count(AccessAuditLog.id))
            .filter(AccessAuditLog.timestamp >= start_date)
            .group_by(AccessAuditLog.severity)
            .all()
        )

        for severity, count in severities:
            records_by_severity[severity] = count

        # 计算成功率
        success_count = (
            self.db.query(AccessAuditLog)
            .filter(
                and_(
                    AccessAuditLog.timestamp >= start_date,
                    AccessAuditLog.success == True,
                )
            )
            .count()
        )

        success_rate = success_count / total_records if total_records > 0 else 0.0

        # 计算日均记录数
        average_records_per_day = total_records / days if days > 0 else 0.0

        # 查找高峰小时
        peak_hour_query = (
            self.db.query(
                func.extract("hour", AccessAuditLog.timestamp).label("hour"),
                func.count(AccessAuditLog.id).label("count"),
            )
            .filter(AccessAuditLog.timestamp >= start_date)
            .group_by("hour")
            .order_by(func.count(AccessAuditLog.id).desc())
            .first()
        )

        peak_hour = int(peak_hour_query[0]) if peak_hour_query else None

        # 最活跃用户
        most_active_user = (
            max(records_by_user.items(), key=lambda x: x[1])[0]
            if records_by_user
            else None
        )

        # 最常访问的项目
        most_accessed_item_query = (
            self.db.query(
                AccessAuditLog.target_id, func.count(AccessAuditLog.id).label("count")
            )
            .filter(
                and_(
                    AccessAuditLog.timestamp >= start_date,
                    AccessAuditLog.target_id.isnot(None),
                )
            )
            .group_by(AccessAuditLog.target_id)
            .order_by(func.count(AccessAuditLog.id).desc())
            .first()
        )

        most_accessed_item = (
            most_accessed_item_query[0] if most_accessed_item_query else None
        )

        # 可疑活动数量（高风险和中风险）
        suspicious_activities = (
            self.db.query(AccessAuditLog)
            .filter(
                and_(
                    AccessAuditLog.timestamp >= start_date,
                    AccessAuditLog.severity.in_(
                        [AuditSeverity.HIGH.value, AuditSeverity.CRITICAL.value]
                    ),
                )
            )
            .count()
        )

        return AuditStatistics(
            total_records=total_records,
            records_by_action=records_by_action,
            records_by_user=records_by_user,
            records_by_severity=records_by_severity,
            success_rate=success_rate,
            average_records_per_day=average_records_per_day,
            peak_hour=peak_hour,
            most_active_user=most_active_user,
            most_accessed_item=most_accessed_item,
            suspicious_activities=suspicious_activities,
            time_period_days=days,
        )

    def search_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        severity: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """搜索审计日志"""
        query = self.db.query(AccessAuditLog)

        # 应用过滤器
        if start_date:
            query = query.filter(AccessAuditLog.timestamp >= start_date)

        if end_date:
            query = query.filter(AccessAuditLog.timestamp <= end_date)

        if user_id:
            query = query.filter(AccessAuditLog.user_id == user_id)

        if action:
            query = query.filter(AccessAuditLog.action == action)

        if target_type:
            query = query.filter(AccessAuditLog.target_type == target_type)

        if target_id:
            query = query.filter(AccessAuditLog.target_id == target_id)

        if severity:
            query = query.filter(AccessAuditLog.severity == severity)

        if success is not None:
            query = query.filter(AccessAuditLog.success == success)

        # 总数
        total = query.count()

        # 分页查询
        logs = (
            query.order_by(AccessAuditLog.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        # 转换为字典
        log_dicts = []
        for log in logs:
            log_dict = {
                "id": log.id,
                "user_id": log.user_id,
                "action": log.action,
                "target_type": log.target_type,
                "target_id": log.target_id,
                "target_name": log.target_name,
                "severity": log.severity,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "details": json.loads(log.details) if log.details else {},
                "success": log.success,
                "error_message": log.error_message,
                "timestamp": log.timestamp.isoformat(),
            }
            log_dicts.append(log_dict)

        return log_dicts, total

    def generate_report(
        self,
        report_type: str = "daily",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """生成审计报告"""
        if not start_date:
            if report_type == "daily":
                start_date = datetime.now() - timedelta(days=1)
                end_date = datetime.now()
            elif report_type == "weekly":
                start_date = datetime.now() - timedelta(days=7)
                end_date = datetime.now()
            elif report_type == "monthly":
                start_date = datetime.now() - timedelta(days=30)
                end_date = datetime.now()
            else:
                start_date = datetime.now() - timedelta(days=1)
                end_date = datetime.now()

        # 获取统计信息
        days = (end_date - start_date).days
        stats = self.get_statistics(days)

        # 获取日志
        logs, total_logs = self.search_logs(start_date, end_date, limit=100)

        # 检测异常
        anomalies = self._detect_anomalies_for_period(start_date, end_date)

        # 生成报告
        report = {
            "report_type": report_type,
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
            "statistics": stats.to_dict(),
            "summary": {
                "total_operations": stats.total_records,
                "success_rate": f"{stats.success_rate:.2%}",
                "suspicious_activities": stats.suspicious_activities,
                "most_active_user": stats.most_active_user,
                "peak_activity_hour": stats.peak_hour,
            },
            "anomalies": [anomaly.to_dict() for anomaly in anomalies],
            "sample_logs": logs[:10],  # 前10条日志作为样本
            "generated_at": datetime.now().isoformat(),
            "recommendations": self._generate_recommendations(stats, anomalies),
        }

        return report

    def _detect_anomalies_for_period(
        self, start_date: datetime, end_date: datetime
    ) -> List[AnomalyDetectionResult]:
        """检测指定时间段内的异常"""
        anomalies = []

        # 这里可以调用各种异常检测方法
        # 暂时返回空列表
        return anomalies

    def _generate_recommendations(
        self, stats: AuditStatistics, anomalies: List[AnomalyDetectionResult]
    ) -> List[str]:
        """生成建议"""
        recommendations = []

        # 基于统计数据的建议
        if stats.success_rate < 0.95:
            recommendations.append(
                f"操作成功率较低 ({stats.success_rate:.2%})，建议检查系统稳定性"
            )

        if stats.suspicious_activities > 0:
            recommendations.append(
                f"检测到 {stats.suspicious_activities} 次可疑活动，建议进行安全审查"
            )

        if stats.average_records_per_day > 1000:
            recommendations.append(
                f"日均审计记录数较高 ({stats.average_records_per_day:.0f})，考虑优化日志记录策略"
            )

        # 基于异常的建议
        for anomaly in anomalies:
            if anomaly.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
                recommendations.append(
                    f"严重异常检测: {anomaly.description}，建议立即处理"
                )

        # 通用建议
        if not recommendations:
            recommendations.append("审计系统运行正常，无需特别处理")

        return recommendations

    def cleanup_old_logs(self, days_to_keep: int = 90) -> int:
        """清理旧日志"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        try:
            # 删除旧日志
            deleted_count = (
                self.db.query(AccessAuditLog)
                .filter(AccessAuditLog.timestamp < cutoff_date)
                .delete()
            )

            self.db.commit()

            self.logger.info(
                f"清理审计日志: 删除 {deleted_count} 条 {days_to_keep} 天前的记录"
            )

            return deleted_count

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"清理审计日志失败: {e}")
            return 0


# 审计装饰器
def audit_access(
    action: AuditAction,
    target_type: str,
    target_id_param: str = None,
    severity: AuditSeverity = AuditSeverity.INFO,
):
    """审计访问装饰器"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from fastapi import Request

            # 获取当前用户
            user = kwargs.get("user")
            if not user:
                # 尝试从依赖获取
                for dep in kwargs.values():
                    if isinstance(dep, User):
                        user = dep
                        break

            # 获取目标ID
            target_id = None
            if target_id_param:
                target_id = kwargs.get(target_id_param)

            # 获取请求信息
            request = kwargs.get("request")
            ip_address = None
            user_agent = None

            if request and isinstance(request, Request):
                ip_address = request.client.host if request.client else "unknown"
                user_agent = request.headers.get("user-agent", "unknown")

            # 调用原始函数
            try:
                result = await func(*args, **kwargs)
                success = True
                error_message = None
            except Exception as e:
                success = False
                error_message = str(e)
                raise e

            # 记录审计日志
            try:
                # 获取数据库会话
                db = kwargs.get("db")
                if db:
                    audit_system = AccessAuditSystem(db)
                    audit_system.log_access(
                        user_id=user.id if user else "anonymous",
                        action=action,
                        target_type=target_type,
                        target_id=target_id,
                        severity=severity,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        details={
                            "endpoint": func.__name__,
                            "result": "success" if success else "error",
                        },
                        success=success,
                        error_message=error_message,
                    )
            except Exception as audit_error:
                # 审计失败不应影响主要功能
                logger.error(f"记录审计日志失败: {audit_error}")

            return result

        return wrapper

    return decorator


# 全局实例
_audit_system_instance = None


def get_audit_system(db: Session) -> AccessAuditSystem:
    """获取审计系统单例"""
    global _audit_system_instance

    if _audit_system_instance is None:
        _audit_system_instance = AccessAuditSystem(db)

    return _audit_system_instance


__all__ = [
    "AccessAuditSystem",
    "get_audit_system",
    "AuditAction",
    "AuditSeverity",
    "AuditRecord",
    "AuditStatistics",
    "AnomalyDetectionResult",
    "audit_access",
]
