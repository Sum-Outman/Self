# Gunicorn 生产服务器配置
# 用于支持高并发 (10,000+ 并发连接)

import multiprocessing

# 导入配置
try:
    from backend.core.config import Config

    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    print("警告: 无法导入Config类，使用默认Gunicorn配置")

# 服务器套接字
bind = "0.0.0.0:8000"

# 工作进程数
cpu_count = multiprocessing.cpu_count()
if USE_CONFIG:
    workers = Config.GUNICORN_WORKERS
else:
    # 公式: workers = (2 * CPU核心数) + 1
    workers = (2 * cpu_count) + 1

worker_class = "uvicorn.workers.UvicornWorker"

# 每个工作进程的线程数
if USE_CONFIG:
    threads = Config.GUNICORN_THREADS
else:
    threads = 4

# 连接设置
backlog = 2048  # 等待连接队列大小

if USE_CONFIG:
    max_requests = Config.GUNICORN_MAX_REQUESTS
    max_requests_jitter = Config.GUNICORN_MAX_REQUESTS_JITTER
else:
    max_requests = 1000  # 每个工作进程处理的最大请求数，防止内存泄漏
    max_requests_jitter = 50  # 随机抖动，防止所有工作进程同时重启

# 超时设置
timeout = 120  # 请求超时时间（秒）
keepalive = 2  # 保持连接时间（秒）

# 进程名称
proc_name = "self_agi_backend"

# 日志设置
accesslog = "./logs/gunicorn_access.log"
errorlog = "./logs/gunicorn_error.log"
loglevel = "info"

# 访问日志格式
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# 进程ID文件
pidfile = "./logs/gunicorn.pid"

# 用户/组（生产环境应使用非root用户）
# user = "self_agi"
# group = "self_agi"

# 安全性
limit_request_line = 4096  # 最大请求行大小
limit_request_fields = 100  # 最大请求头字段数
limit_request_field_size = 8190  # 单个请求头字段最大大小

# 性能调优
if USE_CONFIG:
    worker_connections = Config.GUNICORN_WORKER_CONNECTIONS
else:
    worker_connections = 1000  # 每个工作进程的最大并发连接数

# 预加载应用（减少内存使用和启动时间）
preload_app = True

# 守护进程模式（生产环境使用）
daemon = False  # 设置为True以守护进程模式运行

# 环境变量
raw_env = [
    "PYTHONPATH=.",
]

print("Gunicorn 配置加载完成:")
print(f"  - 绑定地址: {bind}")
print(f"  - 工作进程数: {workers} (CPU核心数: {cpu_count})")
print(f"  - 工作进程类: {worker_class}")
print(f"  - 线程数: {threads}")
print(f"  - 最大并发连接数: {workers * worker_connections}")
print(f"  - 预加载应用: {preload_app}")
if USE_CONFIG:
    print("  - 配置来源: Config类")
else:
    print("  - 配置来源: 默认值")
