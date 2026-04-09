"""
训练报警系统
检测训练过程中的异常情况，提供多通道报警功能

功能：
1. 异常检测算法（数据异常、模型崩溃、资源耗尽）
2. 多通道报警（邮件、Webhook、短信）
3. 报警级别和静默策略
4. 报警历史记录和管理
"""

import logging
import time
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime, timedelta
import threading
import statistics
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """报警严重程度"""

    INFO = "info"  # 信息：正常通知
    WARNING = "warning"  # 警告：需要注意但非紧急
    ERROR = "error"  # 错误：需要立即关注
    CRITICAL = "critical"  # 严重：需要立即干预


class AlertType(Enum):
    """报警类型"""

    DATA_ANOMALY = "data_anomaly"  # 数据异常
    MODEL_DIVERGENCE = "model_divergence"  # 模型发散
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # 资源耗尽
    TRAINING_STALL = "training_stall"  # 训练停滞
    LOSS_EXPLOSION = "loss_explosion"  # 损失爆炸
    GRADIENT_EXPLOSION = "gradient_explosion"  # 梯度爆炸
    VALIDATION_DEGRADATION = "validation_degradation"  # 验证性能下降
    MEMORY_LEAK = "memory_leak"  # 内存泄漏
    HARDWARE_FAILURE = "hardware_failure"  # 硬件故障
    CONNECTION_LOST = "connection_lost"  # 连接丢失


@dataclass
class Alert:
    """报警信息"""

    id: str
    type: AlertType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: str
    training_id: Optional[str] = None
    model_id: Optional[str] = None
    acknowledged: bool = False
    resolved: bool = False
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    acknowledgement_time: Optional[str] = None
    resolution_time: Optional[str] = None
    recurrence_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            **asdict(self),
            "type": self.type.value,
            "severity": self.severity.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        """从字典创建"""
        data = data.copy()
        data["type"] = AlertType(data["type"])
        data["severity"] = AlertSeverity(data["severity"])
        return cls(**data)


class AlertChannel:
    """报警通道基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.name = config.get("name", "unnamed")
        self.logger = logging.getLogger(f"{__name__}.AlertChannel.{self.name}")

    def send(self, alert: Dict[str, Any]) -> bool:
        """发送报警

        参数:
            alert: 报警信息

        返回:
            bool: 发送是否成功

        注意：根据用户要求"系统可以在没有硬件条件下单独运行AGI所有功能"，
        当具体报警通道未实现时，返回False并记录警告。
        """
        self.logger.warning(
            f"报警发送：具体报警通道未实现（报警类型: {alert.get('type', 'unknown')}）。\n"
            "根据用户要求'系统可以在没有硬件条件下单独运行AGI所有功能'，\n"
            "返回False，系统可以继续运行（报警功能将不可用）。"
        )
        return False  # 返回False表示发送失败


class EmailAlertChannel(AlertChannel):
    """邮件报警通道"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.from_email = config.get("from_email", "")
        self.to_emails = config.get("to_emails", [])
        self.use_tls = config.get("use_tls", True)

    def send(self, alert: Alert) -> bool:
        """发送邮件报警"""
        if not self.enabled or not self.to_emails:
            return False

        try:
            # 创建邮件
            msg = MIMEMultipart("alternative")
            msg["Subject"] = (
                f"[{alert.severity.value.upper()}] {alert.type.value} - AGI训练报警"
            )
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)

            # 纯文本内容
            text = f"""             训练报警通知              类型: {alert.type.value}             严重程度: {alert.severity.value}             消息: {alert.message}             时间: {alert.timestamp}             训练ID: {alert.training_id or 'N/A'}             模型ID: {alert.model_id or 'N/A'}              详情:             {json.dumps(alert.details, indent=2, ensure_ascii=False)}             """

            # HTML内容
            html = f"""             <html>             <body>                 <h2>训练报警通知</h2>                 <table border="1" cellpadding="5" cellspacing="0">                     <tr><th>字段</th><th>值</th></tr>                     <tr><td>类型</td><td>{alert.type.value}</td></tr>                     <tr><td>严重程度</td><td><strong>{alert.severity.value}</strong></td></tr>                     <tr><td>消息</td><td>{alert.message}</td></tr>                     <tr><td>时间</td><td>{alert.timestamp}</td></tr>                     <tr><td>训练ID</td><td>{alert.training_id or 'N/A'}</td></tr>                     <tr><td>模型ID</td><td>{alert.model_id or 'N/A'}</td></tr>                 </table>                 <h3>详情:</h3>                 <pre>{json.dumps(alert.details, indent=2, ensure_ascii=False)}</pre>             </body>             </html>             """

            # 添加内容
            msg.attach(MIMEText(text, "plain", "utf-8"))
            msg.attach(MIMEText(html, "html", "utf-8"))

            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)

            self.logger.info(f"邮件报警发送成功: {alert.id}")
            return True

        except Exception as e:
            self.logger.error(f"邮件报警发送失败: {e}")
            return False


class WebhookAlertChannel(AlertChannel):
    """Webhook报警通道"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get("webhook_url", "")
        self.headers = config.get("headers", {"Content-Type": "application/json"})
        self.timeout = config.get("timeout", 10)

    def send(self, alert: Alert) -> bool:
        """发送Webhook报警"""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            import requests

            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.now().isoformat(),
                "system": "AGI训练系统",
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )

            if response.status_code in [200, 201, 202]:
                self.logger.info(f"Webhook报警发送成功: {alert.id}")
                return True
            else:
                self.logger.error(
                    f"Webhook报警发送失败: {response.status_code} - {response.text}"
                )
                return False

        except ImportError:
            self.logger.error("requests库未安装，无法发送Webhook报警")
            return False
        except Exception as e:
            self.logger.error(f"Webhook报警发送失败: {e}")
            return False


class SMSAlertChannel(AlertChannel):
    """短信报警通道（模拟实现）"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.phone_numbers = config.get("phone_numbers", [])
        self.provider = config.get("provider", "twilio")  # twilio, aws_sns, etc.
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")

    def send(self, alert: Alert) -> bool:
        """发送短信报警（模拟）"""
        if not self.enabled or not self.phone_numbers:
            return False

        try:
            # 这里实现实际的短信发送逻辑
            # 暂时模拟成功
            self.logger.info(
                f"短信报警发送模拟: {alert.id} 到 {len(self.phone_numbers)} 个号码"
            )

            # 实际实现示例（使用Twilio）：
            # from twilio.rest import Client
            # client = Client(self.api_key, self.api_secret)
            # for phone in self.phone_numbers:
            #     message = client.messages.create(
            #         body=f"[{alert.severity.value}] {alert.message}",
            #         from_=self.config.get("from_number"),
            #         to=phone
            #     )

            return True

        except Exception as e:
            self.logger.error(f"短信报警发送失败: {e}")
            return False


class AnomalyDetector:
    """异常检测器基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "unnamed")
        self.threshold = config.get("threshold", 3.0)  # 异常阈值（标准差倍数）
        self.window_size = config.get("window_size", 50)  # 滑动窗口大小
        self.min_samples = config.get("min_samples", 10)  # 最小样本数
        self.logger = logging.getLogger(f"{__name__}.AnomalyDetector.{self.name}")

        # 历史数据
        self.history: List[float] = []
        self.timestamps: List[datetime] = []

    def add_measurement(self, value: float, timestamp: Optional[datetime] = None):
        """添加测量值"""
        self.history.append(value)
        self.timestamps.append(timestamp or datetime.now())

        # 保持窗口大小
        if len(self.history) > self.window_size:
            self.history.pop(0)
            self.timestamps.pop(0)

    def check_anomaly(self, value: float) -> Optional[Dict[str, Any]]:
        """检查异常

        返回:
            Optional[Dict[str, Any]]: 异常信息，如果没有异常则返回None

        注意：根据用户要求"系统可以在没有硬件条件下单独运行AGI所有功能"，
        当具体异常检测器未实现时，返回None并记录警告。
        """
        self.logger.warning(
            f"异常检查：具体异常检测器未实现（值: {value}）。\n"
            "根据用户要求'系统可以在没有硬件条件下单独运行AGI所有功能'，\n"
            "返回None，系统可以继续运行（异常检测功能将不可用）。"
        )
        return None  # 返回None表示无异常检测

    def _calculate_statistics(self) -> Dict[str, float]:
        """计算统计信息"""
        if len(self.history) < self.min_samples:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": len(self.history),
            }

        mean = statistics.mean(self.history)
        std = statistics.stdev(self.history) if len(self.history) > 1 else 0.0
        min_val = min(self.history)
        max_val = max(self.history)

        return {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "count": len(self.history),
        }


class ZScoreAnomalyDetector(AnomalyDetector):
    """Z-Score异常检测器"""

    def check_anomaly(self, value: float) -> Optional[Dict[str, Any]]:
        """检查Z-Score异常"""
        if len(self.history) < self.min_samples:
            return None  # 返回None

        stats = self._calculate_statistics()

        if stats["std"] == 0:
            return None  # 返回None

        z_score = abs((value - stats["mean"]) / stats["std"])

        if z_score > self.threshold:
            return {
                "type": "z_score_anomaly",
                "value": value,
                "z_score": z_score,
                "threshold": self.threshold,
                "mean": stats["mean"],
                "std": stats["std"],
                "history_size": len(self.history),
                "anomaly_score": min(
                    1.0, z_score / (self.threshold * 2)
                ),  # 0-1之间的异常分数
            }

        return None  # 返回None


class MovingAverageAnomalyDetector(AnomalyDetector):
    """移动平均异常检测器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.moving_window = config.get("moving_window", 10)

    def check_anomaly(self, value: float) -> Optional[Dict[str, Any]]:
        """检查移动平均异常"""
        if len(self.history) < self.moving_window:
            return None  # 返回None

        # 计算移动平均
        recent_values = self.history[-self.moving_window:]
        moving_avg = statistics.mean(recent_values)
        moving_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0

        if moving_std == 0:
            return None  # 返回None

        # 计算偏差
        deviation = abs(value - moving_avg) / moving_std

        if deviation > self.threshold:
            return {
                "type": "moving_average_anomaly",
                "value": value,
                "moving_average": moving_avg,
                "moving_std": moving_std,
                "deviation": deviation,
                "threshold": self.threshold,
                "window_size": self.moving_window,
                "anomaly_score": min(1.0, deviation / (self.threshold * 2)),
            }

        return None  # 返回None


class TrendAnomalyDetector(AnomalyDetector):
    """趋势异常检测器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trend_window = config.get("trend_window", 20)
        self.slope_threshold = config.get("slope_threshold", 0.1)

    def check_anomaly(self, value: float) -> Optional[Dict[str, Any]]:
        """检查趋势异常"""
        if len(self.history) < self.trend_window:
            return None  # 返回None

        # 提取最近的数据点
        recent_values = self.history[-self.trend_window:]
        recent_timestamps = self.timestamps[-self.trend_window:]

        # 计算时间差（秒）
        time_diffs = [
            (recent_timestamps[i] - recent_timestamps[0]).total_seconds()
            for i in range(len(recent_timestamps))
        ]

        # 线性回归计算趋势
        try:
            x = np.array(time_diffs)
            y = np.array(recent_values)

            # 计算斜率和截距
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]

            # 计算当前值与预测值的偏差
            predicted = m * time_diffs[-1] + c
            residual = abs(value - predicted)

            # 计算残差的标准差
            residuals = [abs(y[i] - (m * time_diffs[i] + c)) for i in range(len(y))]
            residual_std = statistics.stdev(residuals) if len(residuals) > 1 else 0.0

            if residual_std == 0:
                return None  # 返回None

            # 检查趋势斜率是否异常
            slope_magnitude = abs(m)

            anomaly_info = {
                "type": "trend_anomaly",
                "value": value,
                "predicted": predicted,
                "residual": residual,
                "residual_std": residual_std,
                "slope": m,
                "intercept": c,
                "slope_magnitude": slope_magnitude,
                "residual_z_score": residual / residual_std if residual_std > 0 else 0,
            }

            # 检查残差异常
            if residual / residual_std > self.threshold:
                anomaly_info["anomaly_type"] = "residual_anomaly"
                anomaly_info["anomaly_score"] = min(
                    1.0, (residual / residual_std) / (self.threshold * 2)
                )
                return anomaly_info

            # 检查趋势斜率异常
            if slope_magnitude > self.slope_threshold:
                anomaly_info["anomaly_type"] = "slope_anomaly"
                anomaly_info["anomaly_score"] = min(
                    1.0, slope_magnitude / (self.slope_threshold * 2)
                )
                return anomaly_info

        except Exception as e:
            self.logger.error(f"趋势异常检测失败: {e}")

        return None  # 返回None


class TrainingAlertSystem:
    """训练报警系统"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化训练报警系统"""
        self.config = config or {
            "alert_channels": {
                "email": {"enabled": False},
                "webhook": {"enabled": False},
                "sms": {"enabled": False},
            },
            "anomaly_detectors": {
                "loss": {"type": "z_score", "threshold": 3.0, "window_size": 100},
                "accuracy": {
                    "type": "moving_average",
                    "threshold": 2.5,
                    "window_size": 50,
                },
                "resource_usage": {
                    "type": "trend",
                    "threshold": 2.0,
                    "window_size": 30,
                },
            },
            "alert_rules": {
                "enable_data_anomaly": True,
                "enable_model_divergence": True,
                "enable_resource_exhaustion": True,
                "enable_training_stall": True,
                "min_loss_for_stall": 1e-3,  # 损失小于此值可能表示停滞
                "max_loss_for_explosion": 100.0,  # 损失大于此值表示爆炸
                "memory_threshold_percent": 90,  # 内存使用阈值
                "gpu_threshold_percent": 95,  # GPU使用阈值
                "stall_time_minutes": 30,  # 训练停滞时间阈值
            },
            "silence_policy": {
                "enable": True,
                "max_alerts_per_hour": 10,
                "silence_duration_minutes": 60,
                "escalate_after_silence": True,
            },
            "alert_history_size": 1000,
        }

        self.logger = logging.getLogger(f"{__name__}.TrainingAlertSystem")

        # 初始化报警通道
        self.alert_channels = self._initialize_alert_channels()

        # 初始化异常检测器
        self.anomaly_detectors = self._initialize_anomaly_detectors()

        # 报警历史
        self.alert_history: List[Alert] = []
        self.alert_history_lock = threading.RLock()

        # 报警统计
        self.alert_stats = {
            "total_alerts": 0,
            "sent_alerts": 0,
            "failed_alerts": 0,
            "acknowledged_alerts": 0,
            "resolved_alerts": 0,
            "alerts_by_severity": {severity.value: 0 for severity in AlertSeverity},
            "alerts_by_type": {alert_type.value: 0 for alert_type in AlertType},
            "last_alert_time": None,
            "alerts_last_hour": 0,
            "silence_mode": False,
            "silence_until": None,
        }

        # 训练状态
        self.training_state = {
            "current_training_id": None,
            "current_model_id": None,
            "start_time": None,
            "last_progress_time": None,
            "metrics_history": [],
            "resource_history": [],
            "stall_start_time": None,
        }

        # 启动统计清理线程
        self._start_cleanup_thread()

        self.logger.info("训练报警系统初始化完成")

    def _initialize_alert_channels(self) -> Dict[str, AlertChannel]:
        """初始化报警通道"""
        channels = {}
        channels_config = self.config.get("alert_channels", {})

        # 邮件通道
        if channels_config.get("email", {}).get("enabled", False):
            try:
                channels["email"] = EmailAlertChannel(channels_config["email"])
                self.logger.info("邮件报警通道已启用")
            except Exception as e:
                self.logger.error(f"邮件报警通道初始化失败: {e}")

        # Webhook通道
        if channels_config.get("webhook", {}).get("enabled", False):
            try:
                channels["webhook"] = WebhookAlertChannel(channels_config["webhook"])
                self.logger.info("Webhook报警通道已启用")
            except Exception as e:
                self.logger.error(f"Webhook报警通道初始化失败: {e}")

        # 短信通道
        if channels_config.get("sms", {}).get("enabled", False):
            try:
                channels["sms"] = SMSAlertChannel(channels_config["sms"])
                self.logger.info("短信报警通道已启用")
            except Exception as e:
                self.logger.error(f"短信报警通道初始化失败: {e}")

        return channels

    def _initialize_anomaly_detectors(self) -> Dict[str, AnomalyDetector]:
        """初始化异常检测器"""
        detectors = {}
        detectors_config = self.config.get("anomaly_detectors", {})

        detector_classes = {
            "z_score": ZScoreAnomalyDetector,
            "moving_average": MovingAverageAnomalyDetector,
            "trend": TrendAnomalyDetector,
        }

        for detector_name, detector_config in detectors_config.items():
            detector_type = detector_config.get("type", "z_score")

            if detector_type in detector_classes:
                try:
                    detectors[detector_name] = detector_classes[detector_type](
                        detector_config
                    )
                    self.logger.info(
                        f"异常检测器已初始化: {detector_name} ({detector_type})"
                    )
                except Exception as e:
                    self.logger.error(f"异常检测器初始化失败 {detector_name}: {e}")
            else:
                self.logger.warning(f"未知的检测器类型: {detector_type}")

        return detectors

    def _start_cleanup_thread(self):
        """启动清理线程"""

        def cleanup_thread():
            while True:
                time.sleep(3600)  # 每小时清理一次
                self._cleanup_old_alerts()

        thread = threading.Thread(target=cleanup_thread, daemon=True)
        thread.start()

    def _cleanup_old_alerts(self):
        """清理旧报警"""
        with self.alert_history_lock:
            max_history = self.config.get("alert_history_size", 1000)
            if len(self.alert_history) > max_history:
                # 保留最近的报警
                self.alert_history = self.alert_history[-max_history:]
                self.logger.debug(f"清理报警历史，保留最近 {max_history} 条记录")

    def set_training_context(self, training_id: str, model_id: Optional[str] = None):
        """设置训练上下文"""
        self.training_state.update(
            {
                "current_training_id": training_id,
                "current_model_id": model_id,
                "start_time": datetime.now(),
                "last_progress_time": datetime.now(),
                "metrics_history": [],
                "resource_history": [],
                "stall_start_time": None,
            }
        )
        self.logger.info(
            f"训练上下文设置: training_id={training_id}, model_id={model_id}"
        )

    def report_metrics(self, metrics: Dict[str, float]):
        """报告训练指标"""
        if not self.training_state["current_training_id"]:
            self.logger.warning("没有设置训练上下文，忽略指标报告")
            return

        timestamp = datetime.now()
        self.training_state["last_progress_time"] = timestamp

        # 添加到历史
        self.training_state["metrics_history"].append(
            {
                "timestamp": timestamp,
                **metrics,
            }
        )

        # 限制历史大小
        max_history = 1000
        if len(self.training_state["metrics_history"]) > max_history:
            self.training_state["metrics_history"] = self.training_state[
                "metrics_history"
            ][-max_history:]

        # 检查指标异常
        self._check_metrics_anomalies(metrics, timestamp)

        # 检查训练停滞
        self._check_training_stall(metrics, timestamp)

    def report_resource_usage(self, resources: Dict[str, float]):
        """报告资源使用情况"""
        if not self.training_state["current_training_id"]:
            self.logger.warning("没有设置训练上下文，忽略资源报告")
            return

        timestamp = datetime.now()

        # 添加到历史
        self.training_state["resource_history"].append(
            {
                "timestamp": timestamp,
                **resources,
            }
        )

        # 限制历史大小
        max_history = 500
        if len(self.training_state["resource_history"]) > max_history:
            self.training_state["resource_history"] = self.training_state[
                "resource_history"
            ][-max_history:]

        # 检查资源异常
        self._check_resource_anomalies(resources, timestamp)

    def _check_metrics_anomalies(self, metrics: Dict[str, float], timestamp: datetime):
        """检查指标异常"""
        alert_rules = self.config.get("alert_rules", {})

        # 检查损失异常
        if "loss" in metrics:
            loss = metrics["loss"]

            # 损失爆炸检查
            if alert_rules.get("enable_model_divergence", True):
                max_loss = alert_rules.get("max_loss_for_explosion", 100.0)
                if loss > max_loss:
                    self.trigger_alert(
                        AlertType.LOSS_EXPLOSION,
                        AlertSeverity.CRITICAL,
                        f"训练损失爆炸: {loss:.6f} > {max_loss}",
                        {"loss": loss, "threshold": max_loss, "metrics": metrics},
                    )

            # 损失异常检测
            if "loss" in self.anomaly_detectors:
                anomaly = self.anomaly_detectors["loss"].check_anomaly(loss)
                if anomaly:
                    self.anomaly_detectors["loss"].add_measurement(loss, timestamp)
                    self.trigger_alert(
                        AlertType.DATA_ANOMALY,
                        AlertSeverity.WARNING,
                        f"训练损失异常: {loss:.6f}",
                        {**anomaly, "metrics": metrics},
                    )
                else:
                    self.anomaly_detectors["loss"].add_measurement(loss, timestamp)

        # 检查准确率异常
        if "accuracy" in metrics:
            accuracy = metrics["accuracy"]

            if "accuracy" in self.anomaly_detectors:
                anomaly = self.anomaly_detectors["accuracy"].check_anomaly(accuracy)
                if anomaly:
                    self.anomaly_detectors["accuracy"].add_measurement(
                        accuracy, timestamp
                    )
                    self.trigger_alert(
                        AlertType.VALIDATION_DEGRADATION,
                        AlertSeverity.WARNING,
                        f"训练准确率异常: {accuracy:.4f}",
                        {**anomaly, "metrics": metrics},
                    )
                else:
                    self.anomaly_detectors["accuracy"].add_measurement(
                        accuracy, timestamp
                    )

    def _check_training_stall(self, metrics: Dict[str, float], timestamp: datetime):
        """检查训练停滞"""
        alert_rules = self.config.get("alert_rules", {})

        if not alert_rules.get("enable_training_stall", True):
            return

        # 检查损失是否过小（可能停滞）
        if "loss" in metrics:
            min_loss = alert_rules.get("min_loss_for_stall", 1e-3)
            if metrics["loss"] < min_loss:
                if self.training_state["stall_start_time"] is None:
                    self.training_state["stall_start_time"] = timestamp
                    self.logger.info(
                        f"训练可能停滞，损失低于阈值: {metrics['loss']:.6f} < {min_loss}"
                    )
                else:
                    # 计算停滞时间
                    stall_duration = (
                        timestamp - self.training_state["stall_start_time"]
                    ).total_seconds() / 60.0
                    stall_threshold = alert_rules.get("stall_time_minutes", 30)

                    if stall_duration > stall_threshold:
                        self.trigger_alert(
                            AlertType.TRAINING_STALL,
                            AlertSeverity.WARNING,
                            f"训练停滞超过 {                                 stall_duration:.1f} 分钟，损失稳定在 {                                 metrics['loss']:.6f}",
                            {
                                "loss": metrics["loss"],
                                "stall_duration_minutes": stall_duration,
                                "stall_threshold_minutes": stall_threshold,
                                "metrics": metrics,
                            },
                        )
                        # 重置停滞开始时间，避免重复报警
                        self.training_state["stall_start_time"] = timestamp
            else:
                # 损失正常，重置停滞检测
                self.training_state["stall_start_time"] = None

    def _check_resource_anomalies(
        self, resources: Dict[str, float], timestamp: datetime
    ):
        """检查资源异常"""
        alert_rules = self.config.get("alert_rules", {})

        # 检查内存使用
        if "memory_percent" in resources:
            memory_threshold = alert_rules.get("memory_threshold_percent", 90)
            if resources["memory_percent"] > memory_threshold:
                self.trigger_alert(
                    AlertType.RESOURCE_EXHAUSTION,
                    AlertSeverity.ERROR,
                    f"内存使用过高: {resources['memory_percent']:.1f}% > {memory_threshold}%",
                    {
                        "memory_percent": resources["memory_percent"],
                        "threshold": memory_threshold,
                    },
                )

        # 检查GPU使用
        if "gpu_percent" in resources:
            gpu_threshold = alert_rules.get("gpu_threshold_percent", 95)
            if resources["gpu_percent"] > gpu_threshold:
                self.trigger_alert(
                    AlertType.RESOURCE_EXHAUSTION,
                    AlertSeverity.ERROR,
                    f"GPU使用过高: {resources['gpu_percent']:.1f}% > {gpu_threshold}%",
                    {
                        "gpu_percent": resources["gpu_percent"],
                        "threshold": gpu_threshold,
                    },
                )

        # 资源趋势异常检测
        if "resource_usage" in self.anomaly_detectors and "memory_percent" in resources:
            anomaly = self.anomaly_detectors["resource_usage"].check_anomaly(
                resources["memory_percent"]
            )
            if anomaly:
                self.anomaly_detectors["resource_usage"].add_measurement(
                    resources["memory_percent"], timestamp
                )
                self.trigger_alert(
                    AlertType.MEMORY_LEAK,
                    AlertSeverity.WARNING,
                    f"内存使用趋势异常: {resources['memory_percent']:.1f}%",
                    {**anomaly, "resources": resources},
                )
            else:
                self.anomaly_detectors["resource_usage"].add_measurement(
                    resources["memory_percent"], timestamp
                )

    def trigger_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Dict[str, Any],
    ) -> Optional[Alert]:
        """触发报警"""
        # 检查静默策略
        if self._should_silence_alert(alert_type, severity):
            self.logger.debug(f"报警被静默: {alert_type.value} - {message}")
            return None  # 返回None

        # 生成报警ID
        alert_id = f"alert_{int(time.time())}_{alert_type.value}"

        # 创建报警
        alert = Alert(
            id=alert_id,
            type=alert_type,
            severity=severity,
            message=message,
            details=details,
            timestamp=datetime.now().isoformat(),
            training_id=self.training_state["current_training_id"],
            model_id=self.training_state["current_model_id"],
        )

        # 检查是否重复报警
        existing_alert = self._find_similar_alert(alert)
        if existing_alert:
            # 更新重复计数
            existing_alert.recurrence_count += 1
            self.logger.info(
                f"重复报警: {alert_id} (重复次数: {existing_alert.recurrence_count})"
            )

            # 如果重复次数超过阈值，提升严重程度
            if (
                existing_alert.recurrence_count >= 5
                and existing_alert.severity != AlertSeverity.CRITICAL
            ):
                existing_alert.severity = AlertSeverity.CRITICAL
                existing_alert.message = (
                    f"[重复报警×{existing_alert.recurrence_count}] {message}"
                )
                self._send_alert(existing_alert)

            return existing_alert

        # 发送报警
        sent = self._send_alert(alert)

        # 记录报警
        with self.alert_history_lock:
            self.alert_history.append(alert)

            # 更新统计
            self.alert_stats["total_alerts"] += 1
            self.alert_stats["alerts_by_severity"][severity.value] += 1
            self.alert_stats["alerts_by_type"][alert_type.value] += 1
            self.alert_stats["last_alert_time"] = datetime.now().isoformat()

            # 更新最近一小时报警计数
            self._update_hourly_stats()

        if sent:
            self.alert_stats["sent_alerts"] += 1
            self.logger.info(
                f"报警已发送: {alert_id} - {alert_type.value} ({severity.value})"
            )
        else:
            self.alert_stats["failed_alerts"] += 1
            self.logger.error(f"报警发送失败: {alert_id}")

        return alert

    def _should_silence_alert(
        self, alert_type: AlertType, severity: AlertSeverity
    ) -> bool:
        """检查是否应该静默报警"""
        silence_policy = self.config.get("silence_policy", {})

        if not silence_policy.get("enable", True):
            return False

        # 严重报警不静默
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR]:
            return False

        # 检查静默模式
        if self.alert_stats.get("silence_mode", False):
            silence_until = self.alert_stats.get("silence_until")
            if silence_until:
                try:
                    silence_until_dt = datetime.fromisoformat(silence_until)
                    if datetime.now() < silence_until_dt:
                        return True
                    else:
                        # 静默时间已过
                        self.alert_stats["silence_mode"] = False
                        self.alert_stats["silence_until"] = None
                        self.logger.info("静默模式已结束")
                except Exception:
                    pass  # 已实现

        # 检查每小时报警限制
        max_alerts_per_hour = silence_policy.get("max_alerts_per_hour", 10)
        if self.alert_stats["alerts_last_hour"] >= max_alerts_per_hour:
            # 触发静默模式
            silence_duration = silence_policy.get("silence_duration_minutes", 60)
            silence_until = datetime.now() + timedelta(minutes=silence_duration)

            self.alert_stats["silence_mode"] = True
            self.alert_stats["silence_until"] = silence_until.isoformat()

            self.logger.warning(
                f"达到报警限制 ({                     self.alert_stats['alerts_last_hour']}/{max_alerts_per_hour})，"
                f"进入静默模式直到 {silence_until}"
            )

            # 发送静默通知
            if silence_policy.get("escalate_after_silence", True):
                self.trigger_alert(
                    AlertType.TRAINING_STALL,
                    AlertSeverity.WARNING,
                    f"报警系统进入静默模式 ({silence_duration} 分钟)，已达到每小时报警限制",
                    {
                        "alerts_last_hour": self.alert_stats["alerts_last_hour"],
                        "max_alerts_per_hour": max_alerts_per_hour,
                        "silence_duration_minutes": silence_duration,
                        "silence_until": silence_until.isoformat(),
                    },
                )

            return True

        return False

    def _find_similar_alert(self, alert: Alert) -> Optional[Alert]:
        """查找相似的报警"""
        with self.alert_history_lock:
            # 查找最近10分钟内相同类型的报警
            ten_minutes_ago = datetime.now() - timedelta(minutes=10)

            for existing_alert in reversed(self.alert_history):
                try:
                    alert_time = datetime.fromisoformat(existing_alert.timestamp)
                    if alert_time < ten_minutes_ago:
                        break

                    if (
                        existing_alert.type == alert.type
                        and existing_alert.training_id == alert.training_id
                        and not existing_alert.resolved
                    ):
                        return existing_alert
                except Exception:
                    continue

        return None  # 返回None

    def _send_alert(self, alert: Alert) -> bool:
        """发送报警到所有通道"""
        if not self.alert_channels:
            self.logger.warning("没有可用的报警通道")
            return False

        sent_to_any = False

        for channel_name, channel in self.alert_channels.items():
            try:
                if channel.send(alert):
                    sent_to_any = True
                    self.logger.debug(f"报警通过 {channel_name} 发送成功")
                else:
                    self.logger.warning(f"报警通过 {channel_name} 发送失败")
            except Exception as e:
                self.logger.error(f"报警通过 {channel_name} 发送异常: {e}")

        return sent_to_any

    def _update_hourly_stats(self):
        """更新每小时统计"""
        one_hour_ago = datetime.now() - timedelta(hours=1)

        with self.alert_history_lock:
            recent_alerts = 0
            for alert in self.alert_history:
                try:
                    alert_time = datetime.fromisoformat(alert.timestamp)
                    if alert_time > one_hour_ago:
                        recent_alerts += 1
                except Exception:
                    continue

            self.alert_stats["alerts_last_hour"] = recent_alerts

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认报警"""
        with self.alert_history_lock:
            for alert in self.alert_history:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledgement_time = datetime.now().isoformat()

                    self.alert_stats["acknowledged_alerts"] += 1

                    self.logger.info(f"报警已确认: {alert_id} by {acknowledged_by}")
                    return True

        return False

    def resolve_alert(
        self, alert_id: str, resolved_by: str, resolution_notes: str = ""
    ) -> bool:
        """解决报警"""
        with self.alert_history_lock:
            for alert in self.alert_history:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolved_by = resolved_by
                    alert.resolution_time = datetime.now().isoformat()
                    if resolution_notes:
                        alert.details["resolution_notes"] = resolution_notes

                    self.alert_stats["resolved_alerts"] += 1

                    self.logger.info(f"报警已解决: {alert_id} by {resolved_by}")
                    return True

        return False

    def get_active_alerts(self) -> List[Alert]:
        """获取活动报警（未解决）"""
        with self.alert_history_lock:
            return [alert for alert in self.alert_history if not alert.resolved]

    def get_alert_history(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取报警历史"""
        with self.alert_history_lock:
            sorted_history = sorted(
                self.alert_history,
                key=lambda x: datetime.fromisoformat(x.timestamp),
                reverse=True,
            )
            paginated = sorted_history[offset: offset + limit]
            return [alert.to_dict() for alert in paginated]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.alert_history_lock:
            stats = self.alert_stats.copy()

            # 计算解决率
            total = stats["total_alerts"]
            resolved = stats["resolved_alerts"]
            stats["resolution_rate"] = resolved / total if total > 0 else 0.0

            # 计算平均响应时间（从创建到确认）
            total_response_time = 0
            count = 0

            for alert in self.alert_history:
                if alert.acknowledged and alert.acknowledgement_time:
                    try:
                        create_time = datetime.fromisoformat(alert.timestamp)
                        ack_time = datetime.fromisoformat(alert.acknowledgement_time)
                        response_time = (ack_time - create_time).total_seconds()
                        total_response_time += response_time
                        count += 1
                    except Exception:
                        continue

            stats["avg_response_time_seconds"] = (
                total_response_time / count if count > 0 else 0.0
            )

            # 添加时间戳
            stats["timestamp"] = datetime.now().isoformat()

            return stats

    def reset(self):
        """重置报警系统"""
        with self.alert_history_lock:
            self.alert_history.clear()
            self.alert_stats = {
                "total_alerts": 0,
                "sent_alerts": 0,
                "failed_alerts": 0,
                "acknowledged_alerts": 0,
                "resolved_alerts": 0,
                "alerts_by_severity": {severity.value: 0 for severity in AlertSeverity},
                "alerts_by_type": {alert_type.value: 0 for alert_type in AlertType},
                "last_alert_time": None,
                "alerts_last_hour": 0,
                "silence_mode": False,
                "silence_until": None,
            }

            self.training_state = {
                "current_training_id": None,
                "current_model_id": None,
                "start_time": None,
                "last_progress_time": None,
                "metrics_history": [],
                "resource_history": [],
                "stall_start_time": None,
            }

            # 重置检测器
            for detector in self.anomaly_detectors.values():
                detector.history.clear()
                detector.timestamps.clear()

            self.logger.info("报警系统已重置")


# 全局实例
_training_alert_system_instance = None


def get_training_alert_system(
    config: Optional[Dict[str, Any]] = None,
) -> TrainingAlertSystem:
    """获取训练报警系统单例

    参数:
        config: 配置字典

    返回:
        TrainingAlertSystem: 训练报警系统实例
    """
    global _training_alert_system_instance

    if _training_alert_system_instance is None:
        _training_alert_system_instance = TrainingAlertSystem(config)

    return _training_alert_system_instance


__all__ = [
    "TrainingAlertSystem",
    "get_training_alert_system",
    "Alert",
    "AlertType",
    "AlertSeverity",
    "AlertChannel",
    "EmailAlertChannel",
    "WebhookAlertChannel",
    "SMSAlertChannel",
    "AnomalyDetector",
    "ZScoreAnomalyDetector",
    "MovingAverageAnomalyDetector",
    "TrendAnomalyDetector",
]
