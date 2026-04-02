# System Monitoring | 系统监控

This guide covers monitoring the Self AGI system, including health checks, performance metrics, logging, and alerting.

本指南涵盖监控 Self AGI 系统，包括健康检查、性能指标、日志记录和告警。

## Monitoring Overview | 监控概述

### Monitoring Goals | 监控目标
- **Availability**: Ensure system is available and responsive
- **Performance**: Monitor system performance and resource usage
- **Reliability**: Detect and prevent failures
- **Security**: Monitor for security issues and anomalies

- **可用性**: 确保系统可用且响应迅速
- **性能**: 监控系统性能和资源使用
- **可靠性**: 检测和预防故障
- **安全性**: 监控安全问题和异常

### Monitoring Levels | 监控级别
1. **Infrastructure Monitoring**: CPU, memory, disk, network
2. **Application Monitoring**: Application performance and errors
3. **Business Monitoring**: User activity, API usage, business metrics
4. **Security Monitoring**: Security events, access patterns, threats

1. **基础设施监控**: CPU、内存、磁盘、网络
2. **应用监控**: 应用程序性能和错误
3. **业务监控**: 用户活动、API使用、业务指标
4. **安全监控**: 安全事件、访问模式、威胁

## Health Checks | 健康检查

### System Health Endpoints | 系统健康端点
```bash
# Overall system health
GET /api/health

# Database health
GET /api/health/database

# Redis health
GET /api/health/redis

# Model health
GET /api/health/model

# Hardware health
GET /api/health/hardware
```

### Health Check Implementation | 健康检查实现
```python
from fastapi import APIRouter, HTTPException
from monitoring.health import HealthChecker

router = APIRouter()
health_checker = HealthChecker()

@router.get("/health")
async def system_health():
    """Check overall system health."""
    health_status = health_checker.check_all()
    
    if not health_status["overall"]:
        raise HTTPException(
            status_code=503,
            detail="System unhealthy",
            headers={"Retry-After": "30"}
        )
    
    return health_status

@router.get("/health/database")
async def database_health():
    """Check database health."""
    db_status = health_checker.check_database()
    
    if not db_status["healthy"]:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable"
        )
    
    return db_status
```

## Performance Metrics | 性能指标

### Key Performance Indicators | 关键性能指标

#### System Metrics | 系统指标
- **CPU Usage**: Percentage of CPU used
- **Memory Usage**: RAM usage and available memory
- **Disk Usage**: Disk space usage and I/O operations
- **Network Traffic**: Network bandwidth usage

- **CPU使用率**: CPU使用百分比
- **内存使用**: RAM使用情况和可用内存
- **磁盘使用**: 磁盘空间使用和I/O操作
- **网络流量**: 网络带宽使用

#### Application Metrics | 应用指标
- **Response Time**: API response times (p50, p95, p99)
- **Request Rate**: Requests per second/minute
- **Error Rate**: Percentage of failed requests
- **Queue Length**: Length of task queues

- **响应时间**: API响应时间（p50、p95、p99）
- **请求率**: 每秒/分钟请求数
- **错误率**: 失败请求百分比
- **队列长度**: 任务队列长度

#### AGI Model Metrics | AGI模型指标
- **Inference Latency**: Model inference time
- **Token Generation Rate**: Tokens generated per second
- **Model Accuracy**: Model accuracy on validation tasks
- **Memory Usage**: Model memory consumption

- **推理延迟**: 模型推理时间
- **令牌生成率**: 每秒生成的令牌数
- **模型准确率**: 模型在验证任务上的准确率
- **内存使用**: 模型内存消耗

### Metrics Collection | 指标收集
```python
from monitoring.metrics import MetricsCollector

# Initialize metrics collector
metrics_collector = MetricsCollector()

# Collect system metrics
system_metrics = metrics_collector.collect_system_metrics()

# Collect application metrics
app_metrics = metrics_collector.collect_application_metrics()

# Collect model metrics
model_metrics = metrics_collector.collect_model_metrics()

# Export metrics for monitoring systems
prometheus_metrics = metrics_collector.export_prometheus()
```

## Logging | 日志记录

### Log Levels | 日志级别
- **DEBUG**: Detailed information for debugging
- **INFO**: General information about system operation
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failed operations
- **CRITICAL**: Critical errors requiring immediate attention

- **DEBUG**: 用于调试的详细信息
- **INFO**: 关于系统操作的一般信息
- **WARNING**: 潜在问题的警告消息
- **ERROR**: 失败操作的错误消息
- **CRITICAL**: 需要立即关注的关键错误

### Structured Logging | 结构化日志
```python
import logging
from monitoring.logging import StructuredLogger

# Initialize structured logger
logger = StructuredLogger(
    name="self_agi",
    level=logging.INFO,
    format="json"
)

# Log with context
logger.info(
    "API request processed",
    extra={
        "endpoint": "/api/chat/send",
        "method": "POST",
        "duration_ms": 150,
        "user_id": "user_123",
        "model": "self_agi_v1"
    }
)

# Log errors with stack trace
try:
    result = risky_operation()
except Exception as e:
    logger.error(
        "Operation failed",
        exc_info=True,
        extra={
            "operation": "risky_operation",
            "parameters": {"param1": "value1"}
        }
    )
```

### Log Management | 日志管理
```python
from monitoring.logging import LogManager

# Initialize log manager
log_manager = LogManager(
    log_dir="./logs",
    max_size=104857600,  # 100MB
    backup_count=10,
    rotation_interval="1 day"
)

# Configure logging
log_manager.configure_logging()

# Rotate logs manually
log_manager.rotate_logs()

# Archive old logs
log_manager.archive_logs(older_than_days=30)
```

## Alerting | 告警

### Alert Rules | 告警规则
```python
from monitoring.alerting import AlertManager

# Initialize alert manager
alert_manager = AlertManager()

# Define alert rules
alert_manager.add_rule(
    name="high_cpu_usage",
    condition=lambda metrics: metrics["cpu_percent"] > 80,
    severity="warning",
    message="CPU usage above 80%",
    cooldown_minutes=5
)

alert_manager.add_rule(
    name="api_error_rate_high",
    condition=lambda metrics: metrics["api_error_rate"] > 5,
    severity="critical",
    message="API error rate above 5%",
    cooldown_minutes=1
)

alert_manager.add_rule(
    name="low_disk_space",
    condition=lambda metrics: metrics["disk_free_percent"] < 10,
    severity="warning",
    message="Disk space below 10%",
    cooldown_minutes=60
)
```

### Alert Notifications | 告警通知
```python
from monitoring.notifications import NotificationManager

# Initialize notification manager
notification_manager = NotificationManager()

# Configure notification channels
notification_manager.add_channel(
    name="email",
    type="email",
    config={
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "alerts@example.com",
        "password": "password",
        "recipients": ["admin@example.com", "devops@example.com"]
    }
)

notification_manager.add_channel(
    name="slack",
    type="slack",
    config={
        "webhook_url": "https://hooks.slack.com/services/...",
        "channel": "#alerts"
    }
)

# Send alert notification
notification_manager.notify(
    alert={
        "name": "high_cpu_usage",
        "severity": "warning",
        "message": "CPU usage at 85%",
        "timestamp": "2026-03-30T10:30:00Z",
        "metrics": {"cpu_percent": 85}
    },
    channels=["email", "slack"]
)
```

## Monitoring Dashboards | 监控仪表板

### Dashboard Configuration | 仪表板配置
```python
from monitoring.dashboard import DashboardBuilder

# Initialize dashboard builder
dashboard_builder = DashboardBuilder()

# Create system dashboard
system_dashboard = dashboard_builder.create_dashboard(
    title="System Dashboard",
    panels=[
        {
            "title": "CPU Usage",
            "type": "line",
            "query": "system_cpu_percent",
            "interval": "1m"
        },
        {
            "title": "Memory Usage",
            "type": "gauge",
            "query": "system_memory_percent",
            "thresholds": [70, 90]
        },
        {
            "title": "API Response Time",
            "type": "histogram",
            "query": "api_response_time_seconds",
            "buckets": [0.1, 0.5, 1.0, 2.0, 5.0]
        },
        {
            "title": "Error Rate",
            "type": "bar",
            "query": "api_error_rate",
            "interval": "5m"
        }
    ]
)

# Export dashboard
dashboard_json = system_dashboard.export_json()
```

### Real-time Monitoring | 实时监控
```python
from monitoring.realtime import RealTimeMonitor

# Initialize real-time monitor
real_time_monitor = RealTimeMonitor(
    update_interval=1,  # seconds
    max_data_points=1000
)

# Add metrics to monitor
real_time_monitor.add_metric(
    name="cpu_usage",
    collector=lambda: psutil.cpu_percent(interval=1)
)

real_time_monitor.add_metric(
    name="memory_usage",
    collector=lambda: psutil.virtual_memory().percent
)

real_time_monitor.add_metric(
    name="active_users",
    collector=lambda: get_active_user_count()
)

# Start monitoring
real_time_monitor.start()

# Get current metrics
current_metrics = real_time_monitor.get_metrics()

# Stop monitoring
real_time_monitor.stop()
```

## Performance Analysis | 性能分析

### Performance Profiling | 性能剖析
```python
from monitoring.profiling import PerformanceProfiler

# Initialize profiler
profiler = PerformanceProfiler()

# Profile function
@profiler.profile()
def expensive_operation():
    # Perform expensive operation
    result = complex_computation()
    return result

# Profile code block
with profiler.profile_block("data_processing"):
    process_large_dataset()

# Get profiling results
profiling_results = profiler.get_results()

# Generate profiling report
profiler.generate_report(output_file="profiling_report.html")
```

### Bottleneck Detection | 瓶颈检测
```python
from monitoring.analysis import BottleneckDetector

# Initialize bottleneck detector
bottleneck_detector = BottleneckDetector()

# Analyze system for bottlenecks
bottlenecks = bottleneck_detector.detect_bottlenecks(
    metrics_data=historical_metrics,
    thresholds={
        "cpu_percent": 70,
        "memory_percent": 80,
        "disk_iops": 1000,
        "api_response_time": 2.0  # seconds
    }
)

# Get recommendations
recommendations = bottleneck_detector.get_recommendations(bottlenecks)

# Generate report
report = bottleneck_detector.generate_report(bottlenecks, recommendations)
```

## Capacity Planning | 容量规划

### Resource Forecasting | 资源预测
```python
from monitoring.capacity import CapacityPlanner

# Initialize capacity planner
capacity_planner = CapacityPlanner()

# Analyze historical data
historical_analysis = capacity_planner.analyze_historical_data(
    metrics_data=historical_metrics,
    period_days=30
)

# Forecast future requirements
forecast = capacity_planner.forecast(
    historical_data=historical_analysis,
    growth_rate=0.1,  # 10% monthly growth
    forecast_months=6
)

# Generate capacity plan
capacity_plan = capacity_planner.generate_plan(
    forecast=forecast,
    current_resources=current_resources,
    target_utilization=0.7  # 70% target utilization
)

# Get resource recommendations
recommendations = capacity_plan["recommendations"]
```

### Scaling Recommendations | 扩展建议
```python
from monitoring.scaling import ScalingAdvisor

# Initialize scaling advisor
scaling_advisor = ScalingAdvisor()

# Analyze scaling needs
scaling_analysis = scaling_advisor.analyze(
    current_metrics=current_metrics,
    historical_patterns=historical_patterns,
    business_forecast=business_forecast
)

# Get scaling recommendations
recommendations = scaling_advisor.get_recommendations(
    analysis=scaling_analysis,
    scaling_strategy="auto"  # or "manual", "scheduled"
)

# Implement scaling
if recommendations["scale_up"]:
    scaling_advisor.scale_up(
        service=recommendations["service"],
        amount=recommendations["amount"]
    )
```

## Security Monitoring | 安全监控

### Security Event Monitoring | 安全事件监控
```python
from monitoring.security import SecurityMonitor

# Initialize security monitor
security_monitor = SecurityMonitor()

# Monitor authentication events
security_monitor.monitor_authentication(
    events=auth_events,
    rules={
        "failed_attempts": {
            "threshold": 5,
            "window_minutes": 15,
            "action": "block_ip"
        },
        "suspicious_location": {
            "action": "require_2fa"
        }
    }
)

# Monitor API access
security_monitor.monitor_api_access(
    access_logs=api_logs,
    rules={
        "rate_limiting": {
            "requests_per_minute": 100,
            "action": "throttle"
        },
        "unusual_patterns": {
            "action": "alert"
        }
    }
)

# Monitor data access
security_monitor.monitor_data_access(
    access_logs=data_logs,
    sensitive_tables=["users", "api_keys", "payment_info"]
)
```

### Threat Detection | 威胁检测
```python
from monitoring.threat_detection import ThreatDetector

# Initialize threat detector
threat_detector = ThreatDetector()

# Detect anomalies
anomalies = threat_detector.detect_anomalies(
    data=system_metrics,
    algorithms=["isolation_forest", "autoencoder", "statistical"]
)

# Classify threats
threats = threat_detector.classify_threats(
    anomalies=anomalies,
    threat_database=threat_database
)

# Generate threat report
threat_report = threat_detector.generate_report(threats)

# Take mitigation actions
for threat in threats["critical"]:
    threat_detector.mitigate_threat(threat)
```

## Incident Response | 事件响应

### Incident Management | 事件管理
```python
from monitoring.incident import IncidentManager

# Initialize incident manager
incident_manager = IncidentManager()

# Create incident
incident = incident_manager.create_incident(
    title="API Service Degradation",
    severity="high",
    description="API response times increased by 300%",
    detected_by="monitoring_system",
    affected_services=["api_gateway", "backend_service"]
)

# Update incident status
incident_manager.update_incident(
    incident_id=incident["id"],
    status="investigating",
    update="Root cause analysis in progress"
)

# Add responders
incident_manager.add_responder(
    incident_id=incident["id"],
    responder={
        "name": "John Doe",
        "role": "SRE",
        "contact": "john@example.com"
    }
)

# Resolve incident
incident_manager.resolve_incident(
    incident_id=incident["id"],
    resolution="Fixed database connection pool issue",
    root_cause="Database connection pool exhausted"
)
```

### Post-Incident Analysis | 事后分析
```python
from monitoring.postmortem import PostMortemAnalyzer

# Initialize post-mortem analyzer
postmortem_analyzer = PostMortemAnalyzer()

# Analyze incident
analysis = postmortem_analyzer.analyze(
    incident_data=incident_data,
    timeline=incident_timeline,
    metrics=incident_metrics
)

# Generate post-mortem report
report = postmortem_analyzer.generate_report(
    analysis=analysis,
    template="standard"
)

# Extract lessons learned
lessons_learned = postmortem_analyzer.extract_lessons_learned(analysis)

# Create action items
action_items = postmortem_analyzer.create_action_items(
    lessons_learned=lessons_learned,
    priority="high"
)
```

## Monitoring Tools Integration | 监控工具集成

### Prometheus Integration | Prometheus集成
```python
from monitoring.integrations.prometheus import PrometheusExporter

# Initialize Prometheus exporter
prometheus_exporter = PrometheusExporter()

# Export metrics to Prometheus
prometheus_exporter.export_metrics(
    metrics=system_metrics,
    job_name="self_agi",
    instance="production"
)

# Create Prometheus alerts
prometheus_exporter.create_alerts(
    alert_rules=alert_rules,
    namespace="self_agi"
)
```

### Grafana Integration | Grafana集成
```python
from monitoring.integrations.grafana import GrafanaDashboardManager

# Initialize Grafana dashboard manager
grafana_manager = GrafanaDashboardManager(
    grafana_url="http://grafana:3000",
    api_key="your_grafana_api_key"
)

# Create Grafana dashboard
dashboard_id = grafana_manager.create_dashboard(
    dashboard_json=dashboard_json,
    folder="Self AGI",
    overwrite=True
)

# Update dashboard
grafana_manager.update_dashboard(
    dashboard_id=dashboard_id,
    dashboard_json=updated_dashboard_json
)
```

### ELK Stack Integration | ELK Stack集成
```python
from monitoring.integrations.elk import ELKLogger

# Initialize ELK logger
elk_logger = ELKLogger(
    elasticsearch_host="elasticsearch:9200",
    index_name="self-agi-logs"
)

# Send logs to ELK
elk_logger.send_logs(
    logs=structured_logs,
    index_pattern="self-agi-logs-*"
)

# Create Kibana visualizations
elk_logger.create_visualizations(
    logs_index="self-agi-logs-*",
    visualizations=visualization_definitions
)
```

## Best Practices | 最佳实践

### Monitoring Best Practices | 监控最佳实践
1. **Monitor Everything**: Monitor all components and services
2. **Set Meaningful Alerts**: Avoid alert fatigue with meaningful thresholds
3. **Use Dashboards**: Create comprehensive dashboards for different roles
4. **Regular Reviews**: Regularly review and update monitoring configuration
5. **Test Monitoring**: Test monitoring systems regularly

1. **监控一切**: 监控所有组件和服务
2. **设置有意义的告警**: 使用有意义的阈值避免告警疲劳
3. **使用仪表板**: 为不同角色创建全面的仪表板
4. **定期审查**: 定期审查和更新监控配置
5. **测试监控**: 定期测试监控系统

### Alerting Best Practices | 告警最佳实践
1. **Prioritize Alerts**: Classify alerts by severity and impact
2. **Actionable Alerts**: Ensure alerts provide enough information for action
3. **Alert Routing**: Route alerts to appropriate teams
4. **Alert Documentation**: Document alert procedures and runbooks
5. **Review and Refine**: Regularly review and refine alert rules

1. **优先级告警**: 按严重性和影响对告警分类
2. **可操作告警**: 确保告警提供足够的信息以采取行动
3. **告警路由**: 将告警路由到适当的团队
4. **告警文档**: 记录告警流程和操作手册
5. **审查和改进**: 定期审查和改进告警规则

## Next Steps | 后续步骤

After setting up monitoring:

设置监控后：

1. **Review Metrics**: Review collected metrics and adjust as needed
2. **Optimize Alerts**: Optimize alert thresholds and rules
3. **Create Dashboards**: Create dashboards for different use cases
4. **Automate Responses**: Automate responses to common issues
5. **Continuous Improvement**: Continuously improve monitoring setup

1. **审查指标**: 审查收集的指标并根据需要调整
2. **优化告警**: 优化告警阈值和规则
3. **创建仪表板**: 为不同用例创建仪表板
4. **自动化响应**: 自动化对常见问题的响应
5. **持续改进**: 持续改进监控设置

---

*Last Updated: March 30, 2026*  
*最后更新: 2026年3月30日*