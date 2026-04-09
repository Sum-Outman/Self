"""
配置模块
包含应用配置和安全验证
"""

import os
import logging

logger = logging.getLogger(__name__)


class Config:
    """应用配置"""

    # 数据库配置
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "sqlite:///./self_agi.db",
    )
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    MONGODB_URL = os.getenv(
        "MONGODB_URL", "mongodb://admin:your_mongodb_password@localhost:27017/self_agi"
    )
    RABBITMQ_URL = os.getenv(
        "RABBITMQ_URL", "amqp://admin:your_rabbitmq_password@localhost:5672"
    )

    # 服务端口配置
    PORT = int(os.getenv("PORT", "8000"))
    HOST = os.getenv("HOST", "0.0.0.0")
    WS_PORT = int(os.getenv("WS_PORT", "8080"))
    MONITORING_PORT = int(os.getenv("MONITORING_PORT", "8081"))

    # 安全配置
    # SECRET_KEY必须设置，生产环境不允许使用默认值
    _secret_key = os.getenv("SECRET_KEY")
    if _secret_key is None:
        env = os.getenv("ENVIRONMENT", "development")
        if env == "production":
            raise ValueError(
                "生产环境错误：SECRET_KEY必须设置，不能使用默认值。"
                "请设置环境变量SECRET_KEY，并使用强随机字符串。"
            )
        else:
            # 开发环境：生成随机密钥，但记录严重警告
            import secrets

            _secret_key = secrets.token_urlsafe(64)
            logger.critical(
                "安全警告：SECRET_KEY未设置，已生成随机值。"
                "开发环境可以使用此值，但生产环境必须设置环境变量SECRET_KEY。"
                f"当前生成值：{_secret_key[:16]}..."
            )
    SECRET_KEY = _secret_key

    # API密钥配置
    API_KEY_PREFIX = os.getenv("API_KEY_PREFIX", "sk_")  # API密钥前缀，可配置

    @classmethod
    def validate_config(cls):
        """验证配置，特别检查生产环境安全性"""
        env = os.getenv("ENVIRONMENT", "development")

        # SECRET_KEY安全检查（已在新逻辑中处理，此处仅验证强度）
        if env == "production":
            # 生产环境安全检查

            # 检查SECRET_KEY强度（至少32个字符）
            if len(cls.SECRET_KEY) < 32:
                logger.warning(
                    "生产环境警告：SECRET_KEY长度小于32个字符，建议使用更长的密钥"
                )

            if cls.DATABASE_URL.startswith("sqlite"):
                logger.warning(
                    "生产环境警告：使用SQLite数据库，建议使用PostgreSQL或MySQL"
                )

            # 检查数据库URL中的默认密码
            if "your_mongodb_password" in cls.MONGODB_URL:
                logger.warning(
                    "生产环境警告：MongoDB URL中包含默认密码，请设置环境变量MONGODB_URL"
                )

            if "your_rabbitmq_password" in cls.RABBITMQ_URL:
                logger.warning(
                    "生产环境警告：RabbitMQ URL中包含默认密码，请设置环境变量RABBITMQ_URL"
                )

            # 检查Redis URL（开发环境默认没有密码）
            if cls.REDIS_URL == "redis://localhost:6379/0":
                logger.warning(
                    "生产环境警告：Redis使用默认URL且无密码，建议设置密码并使用环境变量REDIS_URL"
                )

            # 检查其他敏感配置 - 确保所有敏感变量都已定义
            sensitive_defaults = [
                ("WECHAT_APP_ID", cls.WECHAT_APP_ID, "your_wechat_app_id"),
                ("WECHAT_MCH_ID", cls.WECHAT_MCH_ID, "your_wechat_mch_id"),
                ("ALIPAY_APP_ID", cls.ALIPAY_APP_ID, "your_alipay_app_id"),
                (
                    "ALIPAY_PRIVATE_KEY",
                    cls.ALIPAY_PRIVATE_KEY,
                    "your_alipay_private_key",
                ),
            ]

            for var_name, value, default in sensitive_defaults:
                if value == default:
                    logger.warning(
                        f"生产环境警告：{var_name}使用默认值，请设置环境变量{var_name}"
                    )

            # 新增安全配置验证
            # SSL/TLS配置检查
            if cls.SSL_ENABLED:
                if not cls.SSL_CERT_PATH or not cls.SSL_KEY_PATH:
                    logger.warning(
                        "生产环境警告：SSL已启用但SSL证书或密钥路径未设置。"
                        "请设置SSL_CERT_PATH和SSL_KEY_PATH环境变量。"
                    )
            
            # 密码策略检查
            if cls.PASSWORD_MIN_LENGTH < 12:
                logger.warning(
                    "生产环境警告：密码最小长度小于12个字符，建议至少12个字符"
                )
            
            # API密钥安全检查
            if cls.API_KEY_MIN_LENGTH < 32:
                logger.warning(
                    "生产环境警告：API密钥最小长度小于32个字符，建议至少32个字符"
                )
            
            # 登录安全配置检查
            if cls.LOGIN_MAX_ATTEMPTS > 10:
                logger.warning(
                    "生产环境警告：登录最大尝试次数过高（大于10次），建议设置为5-10次"
                )
            
            # 数据验证配置检查
            if cls.MAX_UPLOAD_FILE_SIZE_MB > 1024:
                logger.warning(
                    "生产环境警告：最大上传文件大小超过1GB，可能存在安全风险"
                )
            
            # 网络安全配置检查
            if not cls.ENABLE_HSTS:
                logger.warning(
                    "生产环境警告：HSTS未启用，建议在生产环境启用HSTS"
                )
            
            # 记录安全配置状态
            logger.info("生产环境配置安全检查完成（包含新增安全配置验证）")
        else:
            # 开发环境：记录配置验证状态
            logger.debug("开发环境配置验证完成")

    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))  # 每分钟请求数

    # 支付配置
    WECHAT_APP_ID = os.getenv("WECHAT_APP_ID", "your_wechat_app_id")
    WECHAT_MCH_ID = os.getenv("WECHAT_MCH_ID", "your_wechat_mch_id")
    ALIPAY_APP_ID = os.getenv("ALIPAY_APP_ID", "your_alipay_app_id")
    ALIPAY_PRIVATE_KEY = os.getenv("ALIPAY_PRIVATE_KEY", "your_alipay_private_key")

    # 文件存储
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    MODEL_DIR = os.getenv("MODEL_DIR", "models")
    LOG_DIR = os.getenv("LOG_DIR", "logs")

    # 训练配置
    TRAINING_GPU_ENABLED = os.getenv("TRAINING_GPU_ENABLED", "true").lower() == "true"
    TRAINING_MODEL_DIR = os.getenv("TRAINING_MODEL_DIR", "./models")

    # 硬件配置
    HARDWARE_SERIAL_PORT = os.getenv("HARDWARE_SERIAL_PORT", "COM3")
    HARDWARE_BAUDRATE = int(os.getenv("HARDWARE_BAUDRATE", "115200"))
    HARDWARE_ROS_NAMESPACE = os.getenv("HARDWARE_ROS_NAMESPACE", "/self_agi")

    # 速率限制配置
    # 默认限制格式：["100/minute", "1000/hour", "10000/day"]
    RATE_LIMITS = os.getenv("RATE_LIMITS", "100/minute, 1000/hour, 10000/day").split(
        ", "
    )

    # 特定端点的速率限制配置
    # 格式：{"endpoint": ["limit1", "limit2"]}
    ENDPOINT_RATE_LIMITS = {
        "/api/auth/login": ["10/minute", "50/hour"],  # 登录接口更严格
        "/api/auth/register": ["5/minute", "30/hour"],  # 注册接口更严格
        "/api/keys/create": ["20/minute", "200/hour"],  # API密钥创建
        "/api/payment/wechat": ["30/minute", "300/hour"],  # 微信支付
        "/api/payment/alipay": ["30/minute", "300/hour"],  # 支付宝支付
    }

    # 白名单IP（不受速率限制）
    RATE_LIMIT_WHITELIST = os.getenv("RATE_LIMIT_WHITELIST", "127.0.0.1,::1").split(
        ", "
    )

    # 高并发配置
    # 数据库连接池配置
    DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "10"))
    DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
    DATABASE_POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
    DATABASE_POOL_RECYCLE = int(os.getenv("DATABASE_POOL_RECYCLE", "3600"))

    # 异步任务配置
    ASYNC_TASK_MAX_WORKERS = int(os.getenv("ASYNC_TASK_MAX_WORKERS", "20"))
    ASYNC_TASK_QUEUE_SIZE = int(os.getenv("ASYNC_TASK_QUEUE_SIZE", "1000"))
    ASYNC_TASK_TIMEOUT = int(os.getenv("ASYNC_TASK_TIMEOUT", "300"))

    # 缓存配置
    CACHE_DEFAULT_TIMEOUT = int(os.getenv("CACHE_DEFAULT_TIMEOUT", "300"))
    CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "10000"))
    CACHE_REDIS_PREFIX = os.getenv("CACHE_REDIS_PREFIX", "self_agi_cache:")

    # WebSocket配置
    WEBSOCKET_MAX_CONNECTIONS = int(os.getenv("WEBSOCKET_MAX_CONNECTIONS", "1000"))
    WEBSOCKET_PING_INTERVAL = int(os.getenv("WEBSOCKET_PING_INTERVAL", "30"))
    WEBSOCKET_PING_TIMEOUT = int(os.getenv("WEBSOCKET_PING_TIMEOUT", "10"))

    # 性能监控配置
    PERFORMANCE_MONITOR_INTERVAL = float(
        os.getenv("PERFORMANCE_MONITOR_INTERVAL", "5.0")
    )
    PERFORMANCE_ALERT_THRESHOLD_CPU = float(
        os.getenv("PERFORMANCE_ALERT_THRESHOLD_CPU", "80.0")
    )
    PERFORMANCE_ALERT_THRESHOLD_MEMORY = float(
        os.getenv("PERFORMANCE_ALERT_THRESHOLD_MEMORY", "85.0")
    )
    PERFORMANCE_ALERT_THRESHOLD_DISK = float(
        os.getenv("PERFORMANCE_ALERT_THRESHOLD_DISK", "90.0")
    )

    # Gunicorn配置（生产环境）
    GUNICORN_WORKERS = int(os.getenv("GUNICORN_WORKERS", "4"))
    GUNICORN_THREADS = int(os.getenv("GUNICORN_THREADS", "4"))
    GUNICORN_WORKER_CONNECTIONS = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", "1000"))
    GUNICORN_MAX_REQUESTS = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
    GUNICORN_MAX_REQUESTS_JITTER = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

    # 数据库查询优化配置
    SLOW_QUERY_THRESHOLD = float(
        os.getenv("SLOW_QUERY_THRESHOLD", "100.0")
    )  # 慢查询阈值（毫秒）
    MAX_QUERY_HISTORY = int(
        os.getenv("MAX_QUERY_HISTORY", "1000")
    )  # 最大查询历史记录数
    DATABASE_MONITOR_INTERVAL = float(
        os.getenv("DATABASE_MONITOR_INTERVAL", "60.0")
    )  # 数据库监控间隔（秒）
    QUERY_CACHE_ENABLED = (
        os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true"
    )  # 查询缓存是否启用
    QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "300"))  # 查询缓存TTL（秒）
    MAX_DATABASE_CONNECTIONS = int(
        os.getenv("MAX_DATABASE_CONNECTIONS", "100")
    )  # 最大数据库连接数

    # ============================================
    # 安全配置 - 生产环境强化
    # ============================================
    
    # SSL/TLS配置
    SSL_ENABLED = os.getenv("SSL_ENABLED", "false").lower() == "true"
    SSL_CERT_PATH = os.getenv("SSL_CERT_PATH", "")
    SSL_KEY_PATH = os.getenv("SSL_KEY_PATH", "")
    
    # 会话安全配置
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "true").lower() == "true"
    SESSION_COOKIE_HTTPONLY = os.getenv("SESSION_COOKIE_HTTPONLY", "true").lower() == "true"
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "lax")  # lax, strict, none
    
    # 密码策略配置
    PASSWORD_MIN_LENGTH = int(os.getenv("PASSWORD_MIN_LENGTH", "12"))
    PASSWORD_REQUIRE_UPPERCASE = os.getenv("PASSWORD_REQUIRE_UPPERCASE", "true").lower() == "true"
    PASSWORD_REQUIRE_LOWERCASE = os.getenv("PASSWORD_REQUIRE_LOWERCASE", "true").lower() == "true"
    PASSWORD_REQUIRE_DIGITS = os.getenv("PASSWORD_REQUIRE_DIGITS", "true").lower() == "true"
    PASSWORD_REQUIRE_SYMBOLS = os.getenv("PASSWORD_REQUIRE_SYMBOLS", "true").lower() == "true"
    PASSWORD_HISTORY_SIZE = int(os.getenv("PASSWORD_HISTORY_SIZE", "5"))  # 记住的密码历史数量
    PASSWORD_MAX_AGE_DAYS = int(os.getenv("PASSWORD_MAX_AGE_DAYS", "90"))  # 密码最大使用天数
    
    # API密钥安全配置
    API_KEY_MIN_LENGTH = int(os.getenv("API_KEY_MIN_LENGTH", "32"))
    API_KEY_EXPIRE_DAYS = int(os.getenv("API_KEY_EXPIRE_DAYS", "365"))
    API_KEY_ROTATION_DAYS = int(os.getenv("API_KEY_ROTATION_DAYS", "90"))  # 建议密钥轮换天数
    
    # 登录安全配置
    LOGIN_MAX_ATTEMPTS = int(os.getenv("LOGIN_MAX_ATTEMPTS", "5"))
    LOGIN_LOCKOUT_MINUTES = int(os.getenv("LOGIN_LOCKOUT_MINUTES", "15"))
    LOGIN_IP_BLOCK_THRESHOLD = int(os.getenv("LOGIN_IP_BLOCK_THRESHOLD", "10"))  # 同一IP最大失败尝试次数
    LOGIN_IP_BLOCK_HOURS = int(os.getenv("LOGIN_IP_BLOCK_HOURS", "24"))  # IP封锁小时数
    
    # 数据验证配置
    MAX_REQUEST_SIZE_MB = int(os.getenv("MAX_REQUEST_SIZE_MB", "100"))  # 最大请求大小（MB）
    MAX_UPLOAD_FILE_SIZE_MB = int(os.getenv("MAX_UPLOAD_FILE_SIZE_MB", "500"))  # 最大上传文件大小（MB）
    ALLOWED_UPLOAD_EXTENSIONS = os.getenv(
        "ALLOWED_UPLOAD_EXTENSIONS", 
        ".txt,.pdf,.doc,.docx,.xls,.xlsx,.jpg,.jpeg,.png,.gif,.mp4,.avi,.zip,.rar"
    ).split(",")
    
    # 审计日志配置
    AUDIT_LOG_ENABLED = os.getenv("AUDIT_LOG_ENABLED", "true").lower() == "true"
    AUDIT_LOG_RETENTION_DAYS = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "365"))
    
    # 网络安全配置
    ENABLE_HSTS = os.getenv("ENABLE_HSTS", "true").lower() == "true"
    HSTS_MAX_AGE = int(os.getenv("HSTS_MAX_AGE", "31536000"))  # 1年
    ENABLE_XSS_PROTECTION = os.getenv("ENABLE_XSS_PROTECTION", "true").lower() == "true"
    ENABLE_CONTENT_TYPE_OPTIONS = os.getenv("ENABLE_CONTENT_TYPE_OPTIONS", "true").lower() == "true"
    ENABLE_FRAME_OPTIONS = os.getenv("ENABLE_FRAME_OPTIONS", "true").lower() == "true"
    FRAME_OPTIONS_VALUE = os.getenv("FRAME_OPTIONS_VALUE", "SAMEORIGIN")  # DENY, SAMEORIGIN
