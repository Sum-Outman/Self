"""
监控相关的Pydantic模型
用于API请求和响应验证
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class ErrorType(str, Enum):
    """错误类型"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    PERFORMANCE = "performance"


class ErrorInfo(BaseModel):
    """错误信息"""
    type: ErrorType
    message: str
    stack: Optional[str] = None
    component_stack: Optional[str] = None
    timestamp: str
    url: str
    user_agent: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class PerformanceMetric(BaseModel):
    """性能指标"""
    name: str
    value: float
    unit: str
    timestamp: str
    tags: Optional[Dict[str, str]] = None


class ErrorReportRequest(BaseModel):
    """错误报告请求"""
    errors: List[ErrorInfo]
    performance_metrics: List[PerformanceMetric] = []
    timestamp: str
    session_id: str


class ErrorReportResponse(BaseModel):
    """错误报告响应"""
    success: bool
    message: str
    timestamp: str
    reported_count: int


class PerformanceMetricsRequest(BaseModel):
    """性能指标报告请求"""
    metrics: List[PerformanceMetric]
    timestamp: str
    session_id: str


class PerformanceMetricsResponse(BaseModel):
    """性能指标报告响应"""
    success: bool
    message: str
    timestamp: str
    reported_count: int


class AlertSeverity(str, Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """告警类型"""
    SYSTEM = "system"
    API = "api"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    CUSTOM = "custom"


class AlertRequest(BaseModel):
    """告警请求"""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    source: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class AlertResponse(BaseModel):
    """告警响应"""
    success: bool
    message: str
    timestamp: str
    alert_id: Optional[str] = None


class SystemMetricsResponse(BaseModel):
    """系统指标响应"""
    success: bool
    timestamp: str
    metrics: Dict[str, Any]
    error: Optional[str] = None


class AlertStatus(str, Enum):
    """告警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class Alert(BaseModel):
    """告警信息"""
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    source: str
    timestamp: str
    status: AlertStatus
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class MonitoringConfig(BaseModel):
    """监控配置"""
    enabled: bool = True
    sample_rate: float = Field(ge=0.0, le=1.0, default=1.0)
    alert_thresholds: Dict[str, float] = {
        "error_rate": 5.0,  # 错误率阈值（%）
        "response_time": 1000,  # 响应时间阈值（ms）
        "cpu_usage": 90.0,  # CPU使用率阈值（%）
        "memory_usage": 90.0,  # 内存使用率阈值（%）
    }
    notification_channels: List[str] = ["log"]
    retention_days: int = 30


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    service: str
    version: str
    uptime: float
    dependencies: Dict[str, str]