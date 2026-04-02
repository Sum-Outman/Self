# 服务管理器 / Service Manager

## 概述 / Overview

### 中文
服务管理器模块提供完整的跨平台系统服务控制能力，支持Windows服务、Linux systemd服务、macOS launchd服务等。实现了服务启动、停止、重启、启用/禁用、计划任务管理、启动项管理等功能。

### English
The Service Manager module provides complete cross-platform system service control capabilities, supporting Windows services, Linux systemd services, macOS launchd services, etc. Implements service start, stop, restart, enable/disable, scheduled task management, startup item management, and more.

---

## 核心功能 / Core Features

### 中文
1. **跨平台服务管理**：Windows、Linux、macOS完整支持
2. **服务操作**：启动、停止、重启、暂停、恢复服务
3. **服务配置**：启用/禁用自动启动、修改启动类型
4. **计划任务**：创建、列出、删除计划任务（Windows计划任务、Linux cron、macOS launchd）
5. **启动项管理**：管理系统启动项
6. **服务监控**：实时监控关键服务状态

### English
1. **Cross-Platform Service Management**: Complete Windows, Linux, macOS support
2. **Service Operations**: Start, stop, restart, pause, resume services
3. **Service Configuration**: Enable/disable auto-start, modify startup type
4. **Scheduled Tasks**: Create, list, delete scheduled tasks (Windows Task Scheduler, Linux cron, macOS launchd)
5. **Startup Item Management**: Manage system startup items
6. **Service Monitoring**: Real-time monitoring of critical service status

---

## 核心类 / Core Classes

### ServiceStatus
```python
class ServiceStatus(Enum):
    STOPPED = "stopped"        # 已停止 / Stopped
    STARTING = "starting"      # 启动中 / Starting
    RUNNING = "running"        # 运行中 / Running
    STOPPING = "stopping"      # 停止中 / Stopping
    PAUSED = "paused"          # 已暂停 / Paused
    ERROR = "error"            # 错误 / Error
    UNKNOWN = "unknown"        # 未知 / Unknown
```

### ServiceStartType
```python
class ServiceStartType(Enum):
    AUTOMATIC = "automatic"    # 自动 / Automatic
    MANUAL = "manual"          # 手动 / Manual
    DISABLED = "disabled"      # 禁用 / Disabled
    BOOT = "boot"              # 启动时 / Boot
    SYSTEM = "system"          # 系统 / System
```

### ServiceType
```python
class ServiceType(Enum):
    WIN32 = "win32"            # Windows服务 / Windows Service
    SYSTEMD = "systemd"        # Linux systemd服务 / Linux systemd Service
    UPSTART = "upstart"        # Linux upstart服务 / Linux upstart Service
    LAUNCHD = "launchd"        # macOS launchd服务 / macOS launchd Service
    SYSV = "sysv"              # Linux SysV init服务 / Linux SysV init Service
    DOCKER = "docker"          # Docker容器服务 / Docker Container Service
```

### ServiceInfo
```python
@dataclass
class ServiceInfo:
    name: str                      # 服务名称 / Service name
    display_name: str              # 显示名称 / Display name
    status: ServiceStatus          # 状态 / Status
    start_type: ServiceStartType   # 启动类型 / Startup type
    service_type: ServiceType      # 服务类型 / Service type
    pid: Optional[int] = None      # 进程ID / Process ID
    exit_code: Optional[int] = None # 退出代码 / Exit code
    description: str = ""          # 描述 / Description
    binary_path: str = ""          # 二进制路径 / Binary path
    dependencies: List[str] = field(default_factory=list) # 依赖 / Dependencies
    account: str = ""              # 账户 / Account
    started_time: Optional[float] = None # 启动时间 / Started time
    cpu_usage: float = 0.0         # CPU使用率 / CPU usage
    memory_usage: int = 0          # 内存使用 / Memory usage
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ScheduledTask
```python
@dataclass
class ScheduledTask:
    task_id: str                   # 任务ID / Task ID
    name: str                      # 名称 / Name
    command: str                   # 命令 / Command
    schedule: str                  # 调度（cron或Windows格式） / Schedule (cron or Windows format)
    enabled: bool = True           # 是否启用 / Is enabled
    last_run_time: Optional[float] = None # 上次运行时间 / Last run time
    next_run_time: Optional[float] = None # 下次运行时间 / Next run time
    last_run_result: Optional[int] = None # 上次运行结果 / Last run result
    creator: str = ""              # 创建者 / Creator
    description: str = ""          # 描述 / Description
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### StartupItem
```python
@dataclass
class StartupItem:
    item_id: str                   # 项目ID / Item ID
    name: str                      # 名称 / Name
    command: str                   # 命令 / Command
    location: str                  # 位置 / Location
    enabled: bool = True           # 是否启用 / Is enabled
    user: str = ""                 # 用户 / User
    description: str = ""          # 描述 / Description
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 主要方法 / Main Methods

### 服务操作 / Service Operations

#### 获取服务信息 / Get Service Info
```python
def get_service_info(service_name: str) -> Optional[ServiceInfo]
```
获取指定服务的详细信息。

Get detailed information for a specific service.

#### 启动服务 / Start Service
```python
def start_service(service_name: str) -> bool
```
启动指定服务。

Start a specific service.

#### 停止服务 / Stop Service
```python
def stop_service(service_name: str, force: bool = False) -> bool
```
停止指定服务，支持强制停止。

Stop a specific service, supports forceful stop.

#### 重启服务 / Restart Service
```python
def restart_service(service_name: str) -> bool
```
重启指定服务。

Restart a specific service.

#### 启用服务 / Enable Service
```python
def enable_service(service_name: str) -> bool
```
启用服务自动启动。

Enable service auto-start.

#### 禁用服务 / Disable Service
```python
def disable_service(service_name: str) -> bool
```
禁用服务自动启动。

Disable service auto-start.

#### 列出服务 / List Services
```python
def list_services(filter_by_status: Optional[ServiceStatus] = None) -> List[ServiceInfo]
```
列出所有服务，可按状态过滤。

List all services, can filter by status.

### 计划任务 / Scheduled Tasks

#### 创建计划任务 / Create Scheduled Task
```python
def create_scheduled_task(task: ScheduledTask) -> bool
```
创建新的计划任务。

Create a new scheduled task.

#### 列出计划任务 / List Scheduled Tasks
```python
def list_scheduled_tasks() -> List[ScheduledTask]
```
列出所有计划任务。

List all scheduled tasks.

### 服务监控 / Service Monitoring

#### 启动服务监控 / Start Service Monitoring
```python
def start_service_monitoring(interval: float = 30.0)
```
启动服务监控，指定检查间隔。

Start service monitoring with specified check interval.

#### 停止服务监控 / Stop Service Monitoring
```python
def stop_service_monitoring()
```
停止服务监控。

Stop service monitoring.

---

## 使用示例 / Usage Examples

### 中文
```python
from models.system_control.service_manager import get_service_manager

manager = get_service_manager()

# 列出所有服务 / List all services
services = manager.list_services()
print(f"发现 {len(services)} 个服务")

# 获取特定服务信息 / Get specific service info
if platform.system() == "Windows":
    service_info = manager.get_service_info("EventLog")
elif platform.system() == "Linux":
    service_info = manager.get_service_info("systemd-journald")

if service_info:
    print(f"服务: {service_info.name}")
    print(f"状态: {service_info.status.value}")
    print(f"启动类型: {service_info.start_type.value}")

# 启动服务监控 / Start service monitoring
manager.start_service_monitoring(interval=10.0)
print("服务监控已启动")

# 列出计划任务 / List scheduled tasks
tasks = manager.list_scheduled_tasks()
print(f"发现 {len(tasks)} 个计划任务")

# 清理 / Cleanup
manager.cleanup()
```

### English
```python
from models.system_control.service_manager import get_service_manager

manager = get_service_manager()

# List all services
services = manager.list_services()
print(f"Found {len(services)} services")

# Get specific service info
if platform.system() == "Windows":
    service_info = manager.get_service_info("EventLog")
elif platform.system() == "Linux":
    service_info = manager.get_service_info("systemd-journald")

if service_info:
    print(f"Service: {service_info.name}")
    print(f"Status: {service_info.status.value}")
    print(f"Startup type: {service_info.start_type.value}")

# Start service monitoring
manager.start_service_monitoring(interval=10.0)
print("Service monitoring started")

# List scheduled tasks
tasks = manager.list_scheduled_tasks()
print(f"Found {len(tasks)} scheduled tasks")

# Cleanup
manager.cleanup()
```

---

## 相关模块 / Related Modules

- [文件系统管理器](./filesystem-manager.md) - 文件系统管理
- [进程管理器](./process-manager.md) - 进程管理
- [网络控制器](./network-controller.md) - 网络控制
