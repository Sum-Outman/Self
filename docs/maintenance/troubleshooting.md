# Troubleshooting | 故障排除

This guide covers common issues, errors, and solutions for the Self AGI system. Use this guide to diagnose and resolve problems quickly.

本指南涵盖 Self AGI 系统的常见问题、错误和解决方案。使用本指南快速诊断和解决问题。

## Troubleshooting Overview | 故障排除概述

### Troubleshooting Process | 故障排除流程
1. **Identify Problem**: Understand what's not working
2. **Gather Information**: Collect logs, error messages, and system state
3. **Diagnose Root Cause**: Identify the underlying issue
4. **Apply Solution**: Implement appropriate fix
5. **Verify Resolution**: Confirm problem is resolved
6. **Document Solution**: Record solution for future reference

1. **识别问题**: 理解什么不起作用
2. **收集信息**: 收集日志、错误消息和系统状态
3. **诊断根本原因**: 识别潜在问题
4. **应用解决方案**: 实施适当的修复
5. **验证解决**: 确认问题已解决
6. **记录解决方案**: 记录解决方案以备将来参考

### Troubleshooting Tools | 故障排除工具
- **Log Files**: System and application logs
- **Monitoring Dashboards**: Real-time system metrics
- **Diagnostic Commands**: Command-line tools for system inspection
- **Debug Mode**: Enhanced logging for troubleshooting
- **Health Checks**: API endpoints for system health verification

- **日志文件**: 系统和应用程序日志
- **监控仪表板**: 实时系统指标
- **诊断命令**: 用于系统检查的命令行工具
- **调试模式**: 用于故障排除的增强日志记录
- **健康检查**: 用于系统健康验证的API端点

## Common Issues | 常见问题

### Installation Issues | 安装问题

#### Dependency Installation Failures | 依赖安装失败
```bash
# Error: Package installation failed
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied

# Solution: Use virtual environment or user install
# 解决方案：使用虚拟环境或用户安装
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Alternative: Install with user flag
# 备选方案：使用用户标志安装
pip install --user -r requirements.txt
```

#### Environment Configuration Issues | 环境配置问题
```bash
# Error: Missing environment variables
ERROR: Environment variable DATABASE_URL is not set

# Solution: Set required environment variables
# 解决方案：设置必需的环境变量
# Create .env file
# 创建 .env 文件
echo "DATABASE_URL=postgresql://user:password@localhost:5432/self_agi" > .env
echo "REDIS_URL=redis://localhost:6379/0" >> .env
echo "SECRET_KEY=your-secret-key-here" >> .env

# Load environment variables
# 加载环境变量
source .env  # On Windows: use set or export
```

#### Port Conflicts | 端口冲突
```bash
# Error: Port already in use
ERROR: Address already in use: 8000

# Solution: Use different port or kill existing process
# 解决方案：使用不同端口或终止现有进程

# Option 1: Change port
# 选项1：更改端口
uvicorn main:app --port 8080

# Option 2: Find and kill process using port
# 选项2：查找并终止使用端口的进程
# On Linux/Mac:
lsof -ti:8000 | xargs kill -9
# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Startup Issues | 启动问题

#### Database Connection Issues | 数据库连接问题
```python
# Error: Database connection failed
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not connect to server: Connection refused

# Solution: Check database service and connection settings
# 解决方案：检查数据库服务和连接设置

# 1. Check if database is running
# 1. 检查数据库是否在运行
# PostgreSQL
sudo systemctl status postgresql
# Start if not running
sudo systemctl start postgresql

# 2. Verify connection settings
# 2. 验证连接设置
import psycopg2
try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="self_agi",
        user="postgres",
        password="your_password"
    )
    print("Database connection successful")
except Exception as e:
    print(f"Database connection failed: {e}")

# 3. Create database if missing
# 3. 如果缺少则创建数据库
createdb -U postgres self_agi
```

#### Redis Connection Issues | Redis 连接问题
```python
# Error: Redis connection failed
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.

# Solution: Start Redis service and verify connection
# 解决方案：启动Redis服务并验证连接

# 1. Start Redis service
# 1. 启动Redis服务
# On Ubuntu/Debian:
sudo systemctl start redis-server
# On Windows:
redis-server

# 2. Test Redis connection
# 2. 测试Redis连接
import redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("Redis connection successful")
except Exception as e:
    print(f"Redis connection failed: {e}")

# 3. Check Redis configuration
# 3. 检查Redis配置
redis-cli config get bind
redis-cli config get requirepass
```

#### Model Loading Issues | 模型加载问题
```python
# Error: Model file not found
FileNotFoundError: [Errno 2] No such file or directory: 'models/self_agi_v1.pt'

# Solution: Download or train model
# 解决方案：下载或训练模型

# 1. Download pre-trained model
# 1. 下载预训练模型
from models.download import download_model
download_model(model_name="self_agi_v1", save_path="./models")

# 2. Train model from scratch
# 2. 从头开始训练模型
from training.trainer import train_model
train_model(config_path="./config/training_config.yaml")

# 3. Check model file permissions
# 3. 检查模型文件权限
import os
if os.path.exists("models/self_agi_v1.pt"):
    print("Model file exists")
    print(f"Permissions: {oct(os.stat('models/self_agi_v1.pt').st_mode)[-3:]}")
else:
    print("Model file not found")
```

### Runtime Issues | 运行时问题

#### Memory Issues | 内存问题
```python
# Error: Out of memory
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 8.00 GiB total capacity; 5.80 GiB already allocated; 0 bytes free; 6.20 GiB reserved in total by PyTorch)

# Solution: Reduce memory usage
# 解决方案：减少内存使用

# 1. Reduce batch size
# 1. 减少批次大小
training_config.batch_size = 8  # Reduce from 32

# 2. Use gradient accumulation
# 2. 使用梯度累积
training_config.gradient_accumulation_steps = 4

# 3. Use mixed precision training
# 3. 使用混合精度训练
training_config.mixed_precision = True

# 4. Clear GPU memory
# 4. 清除GPU内存
import torch
torch.cuda.empty_cache()

# 5. Monitor memory usage
# 5. 监控内存使用
from monitoring.memory import MemoryMonitor
monitor = MemoryMonitor()
memory_info = monitor.get_memory_info()
print(f"GPU Memory: {memory_info['gpu_used']}/{memory_info['gpu_total']}")
print(f"RAM: {memory_info['ram_used']}/{memory_info['ram_total']}")
```

#### Performance Issues | 性能问题

##### Slow Inference | 推理速度慢
```python
# Issue: Model inference is slow
# 问题：模型推理速度慢

# Solution: Optimize inference performance
# 解决方案：优化推理性能

# 1. Enable model optimization
# 1. 启用模型优化
import torch
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    output = model(input_data)

# 2. Use half precision
# 2. 使用半精度
model.half()  # Convert model to half precision
input_data = input_data.half()

# 3. Use GPU if available
# 3. 使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_data = input_data.to(device)

# 4. Use inference optimization
# 4. 使用推理优化
from inference.optimization import optimize_inference
optimized_model = optimize_inference(
    model=model,
    optimization_level="O2"  # O1 (basic), O2 (balanced), O3 (aggressive)
)
```

##### High Latency | 高延迟
```python
# Issue: High API response latency
# 问题：API响应延迟高

# Solution: Identify and reduce latency sources
# 解决方案：识别并减少延迟源

# 1. Profile API endpoints
# 1. 剖析API端点
from profiling.api_profiler import APIProfiler
profiler = APIProfiler()
latency_report = profiler.profile_endpoint("/api/chat/send")

# 2. Optimize database queries
# 2. 优化数据库查询
# Enable query logging
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Use database indexes
# 使用数据库索引
CREATE INDEX idx_user_messages ON messages(user_id, created_at);

# 3. Implement caching
# 3. 实现缓存
from caching.response_cache import ResponseCache
cache = ResponseCache(ttl=300)  # 5 minutes

@cache.cached()
def expensive_operation(user_id):
    # Expensive database operation
    return result

# 4. Use connection pooling
# 4. 使用连接池
# In database configuration
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 40
DATABASE_POOL_RECYCLE = 3600
```

#### Concurrency Issues | 并发问题

##### Deadlocks | 死锁
```python
# Issue: Database deadlocks
# 问题：数据库死锁
ERROR: deadlock detected

# Solution: Prevent and handle deadlocks
# 解决方案：预防和处理死锁

# 1. Use transaction with timeout
# 1. 使用带超时的事务
from sqlalchemy import text
with engine.connect() as conn:
    # Set lock timeout
    conn.execute(text("SET lock_timeout = '5s'"))
    
    # Execute transaction
    with conn.begin():
        conn.execute(update_statement)

# 2. Acquire locks in consistent order
# 2. 以一致顺序获取锁
# Always lock resources in the same order
# 始终以相同顺序锁定资源
def update_resources(resource1, resource2):
    # Lock in consistent order (e.g., by ID)
    resources = sorted([resource1, resource2], key=lambda x: x.id)
    for resource in resources:
        resource.lock()
    # Perform operations
    for resource in resources:
        resource.unlock()

# 3. Use retry logic for deadlocks
# 3. 对死锁使用重试逻辑
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import sqlalchemy.exc

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(sqlalchemy.exc.OperationalError)
)
def update_with_retry():
    # Database operation that may cause deadlock
    pass
```

##### Race Conditions | 竞态条件
```python
# Issue: Race conditions in concurrent operations
# 问题：并发操作中的竞态条件

# Solution: Use synchronization mechanisms
# 解决方案：使用同步机制

# 1. Use database transactions
# 1. 使用数据库事务
from sqlalchemy.orm import Session

def transfer_funds(session: Session, from_account, to_account, amount):
    with session.begin():
        # Check balance within transaction
        if from_account.balance >= amount:
            from_account.balance -= amount
            to_account.balance += amount
            session.add_all([from_account, to_account])
        else:
            raise ValueError("Insufficient funds")

# 2. Use distributed locks
# 2. 使用分布式锁
from locking.distributed_lock import DistributedLock

lock = DistributedLock(redis_client, "account_lock:12345")
with lock.acquire(timeout=10):
    # Critical section
    account = get_account(12345)
    account.balance += 100
    save_account(account)

# 3. Use optimistic concurrency control
# 3. 使用乐观并发控制
def update_with_version(entity_id, new_data):
    # Get current version
    entity = session.query(Entity).get(entity_id)
    current_version = entity.version
    
    # Update with version check
    result = session.query(Entity).filter(
        Entity.id == entity_id,
        Entity.version == current_version
    ).update({
        **new_data,
        'version': current_version + 1
    })
    
    if result == 0:
        raise ConcurrentModificationError("Entity was modified by another transaction")
```

### Hardware Integration Issues | 硬件集成问题

#### Robot Communication Issues | 机器人通信问题
```python
# Error: Robot connection failed
ConnectionError: Failed to connect to robot at 192.168.1.100:9090

# Solution: Troubleshoot robot communication
# 解决方案：排除机器人通信故障

# 1. Check network connectivity
# 1. 检查网络连接
import socket
import subprocess

# Ping robot
result = subprocess.run(['ping', '-c', '4', '192.168.1.100'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("Robot is reachable")
else:
    print("Robot is not reachable")

# 2. Check port availability
# 2. 检查端口可用性
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('192.168.1.100', 9090))
if result == 0:
    print("Port 9090 is open")
else:
    print(f"Port 9090 is closed (error: {result})")

# 3. Verify robot configuration
# 3. 验证机器人配置
# Check robot ROS2 nodes
import rclpy
from rclpy.node import Node

try:
    rclpy.init()
    node = Node('test_node')
    # List available nodes
    node_names = node.get_node_names()
    print(f"Available nodes: {node_names}")
    
    # Check robot services
    service_names = node.get_service_names_and_types()
    print(f"Available services: {service_names}")
except Exception as e:
    print(f"ROS2 connection failed: {e}")
```

#### Sensor Data Issues | 传感器数据问题
```python
# Issue: Invalid or missing sensor data
# 问题：无效或缺失的传感器数据

# Solution: Validate and handle sensor data
# 解决方案：验证和处理传感器数据

# 1. Validate sensor readings
# 1. 验证传感器读数
from sensors.validation import SensorValidator

validator = SensorValidator()

def process_sensor_data(data):
    # Check data validity
    if not validator.is_valid(data):
        raise ValueError("Invalid sensor data")
    
    # Check for missing values
    if validator.has_missing_values(data):
        # Handle missing values
        data = validator.fill_missing_values(data)
    
    # Check for outliers
    if validator.has_outliers(data):
        # Remove or correct outliers
        data = validator.remove_outliers(data)
    
    return data

# 2. Implement sensor calibration
# 2. 实现传感器校准
from sensors.calibration import SensorCalibrator

calibrator = SensorCalibrator()

# Load calibration data
calibrator.load_calibration("camera_calibration.json")

# Calibrate sensor
calibrated_data = calibrator.calibrate(raw_sensor_data)

# 3. Handle sensor failures
# 3. 处理传感器故障
from sensors.failure_detection import SensorFailureDetector

detector = SensorFailureDetector()

# Monitor sensor health
sensor_health = detector.check_sensor_health(sensor_id="camera_01")

if sensor_health["status"] == "failed":
    # Switch to backup sensor
    switch_to_backup_sensor("camera_01", "camera_02")
    
    # Log failure
    logger.error(f"Sensor {sensor_id} failed: {sensor_health['reason']}")
    
    # Notify maintenance
    notify_maintenance(sensor_id, sensor_health)
```

#### Actuator Control Issues | 执行器控制问题
```python
# Issue: Actuator not responding or moving incorrectly
# 问题：执行器无响应或移动不正确

# Solution: Debug actuator control
# 解决方案：调试执行器控制

# 1. Test actuator communication
# 1. 测试执行器通信
from actuators.testing import ActuatorTester

tester = ActuatorTester()

# Test basic movement
test_result = tester.test_actuator(
    actuator_id="joint_01",
    test_type="basic_movement",
    parameters={"position": 0.5, "velocity": 0.1}
)

if test_result["success"]:
    print("Actuator communication successful")
else:
    print(f"Actuator test failed: {test_result['error']}")

# 2. Check actuator limits and constraints
# 2. 检查执行器限制和约束
from actuators.safety import ActuatorSafetyChecker

safety_checker = ActuatorSafetyChecker()

# Check if command is safe
command = {"position": 1.5, "velocity": 2.0}
is_safe, reason = safety_checker.is_command_safe(
    actuator_id="joint_01",
    command=command
)

if not is_safe:
    print(f"Command unsafe: {reason}")
    # Adjust command to safe limits
    safe_command = safety_checker.get_safe_command(
        actuator_id="joint_01",
        desired_command=command
    )

# 3. Monitor actuator performance
# 3. 监控执行器性能
from actuators.monitoring import ActuatorMonitor

monitor = ActuatorMonitor()

# Get actuator status
status = monitor.get_status(actuator_id="joint_01")
print(f"Position: {status['position']}")
print(f"Velocity: {status['velocity']}")
print(f"Temperature: {status['temperature']}")
print(f"Load: {status['load']}")

if status["temperature"] > 80:  # degrees Celsius
    print("Actuator overheating - reducing load")
    reduce_actuator_load("joint_01")
```

### Model Training Issues | 模型训练问题

#### Training Convergence Issues | 训练收敛问题

##### Training Not Converging | 训练不收敛
```python
# Issue: Training loss not decreasing
# 问题：训练损失不下降

# Solution: Adjust training parameters
# 解决方案：调整训练参数

# 1. Check learning rate
# 1. 检查学习率
from training.analysis import LearningRateAnalyzer

analyzer = LearningRateAnalyzer()
optimal_lr = analyzer.find_optimal_lr(
    model=model,
    train_loader=train_loader,
    lr_range=(1e-5, 1e-1)
)

print(f"Optimal learning rate: {optimal_lr}")

# Update learning rate
optimizer.param_groups[0]['lr'] = optimal_lr

# 2. Use learning rate scheduling
# 2. 使用学习率调度
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=100,  # Number of epochs
    eta_min=1e-6  # Minimum learning rate
)

# Use in training loop
for epoch in range(num_epochs):
    # Training steps
    train_epoch()
    
    # Update learning rate
    scheduler.step()
    
    # Print current learning rate
    print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']}")

# 3. Check gradient flow
# 3. 检查梯度流
from training.analysis import GradientAnalyzer

grad_analyzer = GradientAnalyzer()

# Analyze gradients
gradient_report = grad_analyzer.analyze_gradients(model)

print(f"Gradient norms: {gradient_report['norms']}")
print(f"Vanishing gradients: {gradient_report['vanishing']}")
print(f"Exploding gradients: {gradient_report['exploding']}")

# Fix gradient issues
if gradient_report['vanishing']:
    # Use skip connections or gradient clipping
    model.enable_skip_connections()
    
if gradient_report['exploding']:
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

##### Overfitting | 过拟合
```python
# Issue: Model overfits to training data
# 问题：模型过拟合训练数据

# Solution: Apply regularization techniques
# 解决方案：应用正则化技术

# 1. Add dropout
# 1. 添加dropout
from models.transformer.self_agi_model import SelfAGIModel

model_config = {
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'dropout_rate': 0.1,  # Add dropout
    'attention_dropout_rate': 0.1
}

model = SelfAGIModel(config=model_config)

# 2. Use weight decay
# 2. 使用权重衰减
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 regularization
)

# 3. Implement early stopping
# 3. 实现早停
from training.early_stopping import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,  # Stop if no improvement for 10 epochs
    min_delta=0.001,  # Minimum change to qualify as improvement
    mode='min'  # Minimize validation loss
)

for epoch in range(num_epochs):
    # Training
    train_loss = train_epoch()
    
    # Validation
    val_loss = validate_epoch()
    
    # Check early stopping
    early_stopping(val_loss)
    
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

# 4. Use data augmentation
# 4. 使用数据增强
from training.data_augmentation import DataAugmentor

augmentor = DataAugmentor()

# Augment training data
augmented_dataset = augmentor.augment_dataset(
    dataset=train_dataset,
    augmentations=['random_flip', 'color_jitter', 'random_rotation'],
    augmentation_probability=0.5
)
```

#### Distributed Training Issues | 分布式训练问题

##### Node Communication Issues | 节点通信问题
```python
# Issue: Distributed training nodes cannot communicate
# 问题：分布式训练节点无法通信

# Solution: Configure network and communication
# 解决方案：配置网络和通信

# 1. Check network configuration
# 1. 检查网络配置
import socket
import torch.distributed as dist

def check_network():
    # Get local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Local IP: {local_ip}")
    
    # Check connectivity to master node
    master_addr = "192.168.1.100"
    master_port = 29500
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    
    try:
        sock.connect((master_addr, master_port))
        print(f"Connected to master at {master_addr}:{master_port}")
        sock.close()
        return True
    except Exception as e:
        print(f"Failed to connect to master: {e}")
        return False

# 2. Initialize distributed training properly
# 2. 正确初始化分布式训练
import os

def init_distributed():
    # Set environment variables
    os.environ['MASTER_ADDR'] = '192.168.1.100'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = '4'  # Total number of processes
    os.environ['RANK'] = '0'  # Global rank of this process
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # or 'gloo' for CPU
        init_method='env://',
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK'])
    )
    
    # Get process information
    print(f"Initialized process {dist.get_rank()} of {dist.get_world_size()}")

# 3. Handle communication errors
# 3. 处理通信错误
from training.distributed import DistributedTrainingManager

manager = DistributedTrainingManager()

try:
    # Perform distributed operation
    manager.all_reduce(tensors)
except RuntimeError as e:
    if "connection closed" in str(e).lower():
        print("Connection lost, attempting to reconnect...")
        manager.reconnect()
        
        # Retry operation
        manager.all_reduce(tensors)
    else:
        raise
```

##### Synchronization Issues | 同步问题
```python
# Issue: Nodes out of sync during training
# 问题：训练期间节点不同步

# Solution: Ensure proper synchronization
# 解决方案：确保适当的同步

# 1. Use barrier for synchronization
# 1. 使用屏障进行同步
import torch.distributed as dist

def synchronized_operation():
    # All nodes must reach this point before continuing
    dist.barrier()
    
    # Perform operation that requires synchronization
    perform_operation()
    
    # Synchronize again if needed
    dist.barrier()

# 2. Synchronize model parameters
# 2. 同步模型参数
def sync_model_parameters(model):
    # Average parameters across all nodes
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= dist.get_world_size()
    
    # Ensure all nodes have the same parameters
    dist.barrier()

# 3. Handle straggler nodes
# 3. 处理落后节点
from training.distributed import StragglerHandler

handler = StragglerHandler(timeout=30)  # 30 second timeout

def distributed_training_step():
    # Start timing
    start_time = time.time()
    
    try:
        # Perform training step with timeout
        result = handler.execute_with_timeout(
            function=train_step,
            args=(model, data),
            timeout=30
        )
        
        # Synchronize results
        dist.barrier()
        
        return result
    except TimeoutError:
        print("Node timed out, excluding from this step")
        
        # Mark node as straggler
        handler.mark_straggler(dist.get_rank())
        
        # Continue without straggler
        return None
```

### API Issues | API问题

#### Authentication Issues | 认证问题
```python
# Error: Authentication failed
HTTP 401: Unauthorized - Invalid API key

# Solution: Check and fix authentication
# 解决方案：检查并修复认证

# 1. Verify API key
# 1. 验证API密钥
from authentication.api_auth import APIAuthenticator

authenticator = APIAuthenticator()

def verify_api_key(api_key):
    # Check if API key exists
    if not authenticator.key_exists(api_key):
        return False, "API key not found"
    
    # Check if API key is valid
    if not authenticator.is_key_valid(api_key):
        return False, "API key expired or revoked"
    
    # Check rate limits
    if authenticator.is_rate_limited(api_key):
        return False, "Rate limit exceeded"
    
    return True, "Authentication successful"

# 2. Generate new API key
# 2. 生成新的API密钥
from authentication.key_manager import APIKeyManager

key_manager = APIKeyManager()

# Generate new key
new_key = key_manager.generate_key(
    user_id="user_123",
    permissions=["chat:send", "training:start"],
    expires_in=30  # days
)

print(f"New API key: {new_key['key']}")
print(f"Expires: {new_key['expires_at']}")

# 3. Reset API key
# 3. 重置API密钥
def reset_api_key(user_id):
    # Revoke existing keys
    key_manager.revoke_user_keys(user_id)
    
    # Generate new key
    new_key = key_manager.generate_key(
        user_id=user_id,
        permissions=["full_access"],
        expires_in=30
    )
    
    # Send email with new key
    send_key_email(user_id, new_key)
    
    return new_key
```

#### Rate Limiting Issues | 速率限制问题
```python
# Error: Rate limit exceeded
HTTP 429: Too Many Requests - Rate limit exceeded. Try again in 60 seconds.

# Solution: Manage rate limits
# 解决方案：管理速率限制

# 1. Check current usage
# 1. 检查当前使用情况
from rate_limiting.rate_limiter import RateLimiter

limiter = RateLimiter()

def check_rate_limit(api_key, endpoint):
    # Get current usage
    usage = limiter.get_usage(api_key, endpoint)
    
    print(f"Requests made: {usage['count']}")
    print(f"Limit: {usage['limit']}")
    print(f"Reset in: {usage['reset_in']} seconds")
    
    if usage['remaining'] <= 0:
        return False, f"Rate limit exceeded. Reset in {usage['reset_in']} seconds"
    
    return True, f"Remaining: {usage['remaining']} requests"

# 2. Implement backoff strategy
# 2. 实现退避策略
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(requests.exceptions.HTTPError)
)
def make_api_request_with_backoff(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises HTTPError for 4xx/5xx
    return response.json()

# 3. Upgrade rate limits
# 3. 升级速率限制
from billing.plan_manager import PlanManager

plan_manager = PlanManager()

def upgrade_rate_limit(user_id, new_plan):
    # Check if user can upgrade
    can_upgrade, reason = plan_manager.can_upgrade(user_id, new_plan)
    
    if not can_upgrade:
        return False, reason
    
    # Upgrade plan
    success = plan_manager.upgrade_plan(user_id, new_plan)
    
    if success:
        # Update rate limits
        new_limits = plan_manager.get_plan_limits(new_plan)
        limiter.update_limits(user_id, new_limits)
        
        return True, f"Upgraded to {new_plan} plan"
    else:
        return False, "Upgrade failed"
```

## Debugging Techniques | 调试技术

### Log Analysis | 日志分析
```python
# Analyze application logs
# 分析应用程序日志
from debugging.log_analyzer import LogAnalyzer

analyzer = LogAnalyzer(log_file="./logs/app.log")

# Search for errors
errors = analyzer.search(pattern="ERROR", limit=100)
print(f"Found {len(errors)} errors")

# Analyze error patterns
error_patterns = analyzer.analyze_patterns(errors)
for pattern, count in error_patterns.items():
    print(f"{pattern}: {count} occurrences")

# Get error timeline
error_timeline = analyzer.get_timeline(errors)

# Generate log report
report = analyzer.generate_report()
report.save("./logs/analysis_report.html")
```

### Debug Mode | 调试模式
```python
# Enable debug mode
# 启用调试模式
import os
os.environ["DEBUG"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"

# Configure debug logging
# 配置调试日志
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/debug.log"),
        logging.StreamHandler()
    ]
)

# Add debug middleware
# 添加调试中间件
from fastapi import FastAPI
from debug.middleware import DebugMiddleware

app = FastAPI()
app.add_middleware(DebugMiddleware)

# Enable SQL query logging
# 启用SQL查询日志
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### Interactive Debugging | 交互式调试
```python
# Use Python debugger
# 使用Python调试器
import pdb

def problematic_function():
    # Set breakpoint
    pdb.set_trace()
    
    # Code to debug
    result = complex_operation()
    
    return result

# Use IPython for enhanced debugging
# 使用IPython进行增强调试
from IPython import embed

def debug_interactively():
    # Enter interactive shell
    embed()
    
    # Examine variables
    # locals() and globals() are available
    
# Use remote debugging
# 使用远程调试
import debugpy

# Start debug server
debugpy.listen(("0.0.0.0", 5678))
print("Debug server started on port 5678")

# Wait for debugger to attach
debugpy.wait_for_client()
print("Debugger attached")

# Set breakpoints in code
debugpy.breakpoint()
```

## Recovery Procedures | 恢复程序

### Database Recovery | 数据库恢复
```python
# Database backup and restore
# 数据库备份和恢复
from database.recovery import DatabaseRecovery

recovery = DatabaseRecovery()

# Create backup
backup_file = recovery.create_backup(
    database="self_agi",
    backup_type="full",  # full, incremental, differential
    compress=True
)
print(f"Backup created: {backup_file}")

# Restore from backup
recovery.restore_backup(
    backup_file=backup_file,
    database="self_agi",
    create_database=True
)
print("Database restored")

# Point-in-time recovery
recovery.point_in_time_recovery(
    database="self_agi",
    recovery_time="2026-03-30 10:30:00"
)
print("Database recovered to specified time")

# Check database integrity
integrity_check = recovery.check_integrity("self_agi")
if integrity_check["healthy"]:
    print("Database integrity verified")
else:
    print(f"Database integrity issues: {integrity_check['issues']}")
    
    # Repair database if needed
    if integrity_check["repairable"]:
        recovery.repair_database("self_agi")
        print("Database repaired")
```

### System Recovery | 系统恢复
```python
# System recovery procedures
# 系统恢复程序
from system.recovery import SystemRecovery

system_recovery = SystemRecovery()

# 1. Restart services
# 1. 重启服务
services = ["self_agi_api", "self_agi_model", "self_agi_worker"]
for service in services:
    system_recovery.restart_service(service)
    print(f"Restarted {service}")

# 2. Clear cache
# 2. 清除缓存
system_recovery.clear_cache(
    cache_types=["redis", "memory", "disk"]
)
print("Cache cleared")

# 3. Reset to known good state
# 3. 重置到已知良好状态
system_recovery.reset_to_good_state(
    backup_id="backup_20260330",
    preserve_data=True
)
print("System reset to known good state")

# 4. Failover to backup system
# 4. 故障转移到备份系统
if system_recovery.is_primary_failed():
    system_recovery.failover_to_backup()
    print("Failed over to backup system")
    
    # Notify administrators
    system_recovery.notify_failover()
```

### Data Recovery | 数据恢复
```python
# Data recovery procedures
# 数据恢复程序
from data.recovery import DataRecovery

data_recovery = DataRecovery()

# Recover lost data
recovery_result = data_recovery.recover_data(
    data_type="user_messages",
    time_range=("2026-03-30 00:00:00", "2026-03-30 12:00:00"),
    source="backup"
)

if recovery_result["success"]:
    print(f"Recovered {recovery_result['recovered_count']} records")
    
    # Verify recovered data
    verification = data_recovery.verify_recovered_data(
        recovered_data=recovery_result["data"],
        original_checksum=recovery_result["original_checksum"]
    )
    
    if verification["verified"]:
        print("Data verification successful")
        
        # Import recovered data
        data_recovery.import_data(recovery_result["data"])
        print("Data imported successfully")
    else:
        print(f"Data verification failed: {verification['reason']}")
else:
    print(f"Data recovery failed: {recovery_result['error']}")

# Recover corrupted files
corrupted_files = data_recovery.find_corrupted_files()
for file in corrupted_files:
    print(f"Corrupted file: {file}")
    
    # Attempt recovery
    recovery = data_recovery.recover_file(file)
    
    if recovery["success"]:
        print(f"File recovered: {recovery['recovered_path']}")
    else:
        print(f"File recovery failed: {recovery['error']}")
```

## Prevention Strategies | 预防策略

### Proactive Monitoring | 主动监控
```python
# Implement proactive monitoring
# 实施主动监控
from monitoring.proactive import ProactiveMonitor

proactive_monitor = ProactiveMonitor()

# Set up health checks
proactive_monitor.setup_health_checks(
    checks=[
        {
            "name": "database_connection",
            "check_function": check_database_connection,
            "interval": 60,  # seconds
            "threshold": 3  # failures before alert
        },
        {
            "name": "api_response_time",
            "check_function": check_api_response_time,
            "interval": 30,
            "threshold": 5
        },
        {
            "name": "disk_space",
            "check_function": check_disk_space,
            "interval": 300,
            "threshold": 1
        }
    ]
)

# Start proactive monitoring
proactive_monitor.start()

# Add predictive alerts
proactive_monitor.add_predictive_alert(
    metric="memory_usage",
    prediction_horizon=3600,  # 1 hour
    threshold=0.9,  # 90%
    alert_message="Memory usage predicted to exceed 90% in 1 hour"
)
```

### Regular Maintenance | 定期维护
```python
# Schedule regular maintenance
# 安排定期维护
from maintenance.scheduler import MaintenanceScheduler

scheduler = MaintenanceScheduler()

# Schedule database maintenance
scheduler.schedule_task(
    name="database_vacuum",
    task_function=vacuum_database,
    schedule="0 2 * * *",  # Daily at 2 AM
    description="Clean up and optimize database"
)

# Schedule log rotation
scheduler.schedule_task(
    name="log_rotation",
    task_function=rotate_logs,
    schedule="0 0 * * *",  # Daily at midnight
    description="Rotate and archive log files"
)

# Schedule backup
scheduler.schedule_task(
    name="system_backup",
    task_function=create_system_backup,
    schedule="0 1 * * 0",  # Weekly on Sunday at 1 AM
    description="Create full system backup"
)

# Schedule dependency updates
scheduler.schedule_task(
    name="dependency_updates",
    task_function=update_dependencies,
    schedule="0 3 * * 0",  # Weekly on Sunday at 3 AM
    description="Check and update system dependencies"
)

# Start maintenance scheduler
scheduler.start()
```

### Testing and Validation | 测试和验证
```python
# Implement comprehensive testing
# 实施全面测试
from testing.validation import SystemValidator

validator = SystemValidator()

# Run validation tests
validation_results = validator.run_tests(
    test_types=[
        "unit",      # Unit tests
        "integration", # Integration tests
        "system",    # System tests
        "performance", # Performance tests
        "security"   # Security tests
    ],
    environment="staging"  # or "production", "development"
)

# Analyze test results
analysis = validator.analyze_results(validation_results)

print(f"Tests passed: {analysis['passed']}/{analysis['total']}")
print(f"Success rate: {analysis['success_rate']:.2%}")

if analysis["critical_failures"] > 0:
    print(f"Critical failures: {analysis['critical_failures']}")
    
    # Block deployment if critical failures
    validator.block_deployment(
        reason=f"{analysis['critical_failures']} critical test failures"
    )

# Generate test report
report = validator.generate_report(validation_results)
report.save("./tests/validation_report.html")
```

## Getting Help | 获取帮助

### Support Resources | 支持资源
- **Documentation**: Complete documentation at [docs.self-agi.com](https://docs.self-agi.com)
- **Community Forum**: Ask questions and share solutions at [forum.self-agi.com](https://forum.self-agi.com)
- **GitHub Issues**: Report bugs and request features at [github.com/Sum-Outman/Self/issues](https://github.com/Sum-Outman/Self/issues)
- **Email Support**: Contact support at support@self-agi.com
- **Emergency Contact**: For critical issues, contact emergency@self-agi.com

- **文档**: 完整的文档在 [docs.self-agi.com](https://docs.self-agi.com)
- **社区论坛**: 在 [forum.self-agi.com](https://forum.self-agi.com) 提问和分享解决方案
- **GitHub Issues**: 在 [github.com/Sum-Outman/Self/issues](https://github.com/Sum-Outman/Self/issues) 报告错误和请求功能
- **邮件支持**: 联系 support@self-agi.com
- **紧急联系人**: 对于关键问题，联系 emergency@self-agi.com

### Diagnostic Information to Provide | 要提供的诊断信息
When seeking help, provide the following information:
- System version and configuration
- Error messages and stack traces
- Log files (with sensitive information redacted)
- Steps to reproduce the issue
- Environment details (OS, Python version, dependencies)
- What you've tried already

寻求帮助时，提供以下信息：
- 系统版本和配置
- 错误消息和堆栈跟踪
- 日志文件（已编辑敏感信息）
- 重现问题的步骤
- 环境详情（操作系统、Python版本、依赖项）
- 已经尝试过的解决方案

---

*Last Updated: March 30, 2026*  
*最后更新: 2026年3月30日*