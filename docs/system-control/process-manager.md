# 进程管理器 / Process Manager

## 概述 / Overview

### 中文
进程管理器模块提供完整的跨平台进程控制能力，支持Windows、Linux和macOS系统。实现了进程创建、监控、终止、资源限制和进程间通信等功能。

### English
The Process Manager module provides complete cross-platform process control capabilities, supporting Windows, Linux, and macOS systems. Implements process creation, monitoring, termination, resource limits, and inter-process communication.

---

## 核心功能 / Core Features

### 中文
1. **进程操作**：创建、启动、暂停、恢复、终止进程
2. **进程监控**：实时监控进程状态、资源使用情况
3. **资源限制**：CPU时间、内存、文件大小等资源限制
4. **进程间通信**：管道、队列、共享内存等IPC机制
5. **进程树管理**：父子进程关系管理
6. **守护进程管理**：后台进程管理

### English
1. **Process Operations**: Create, start, pause, resume, terminate processes
2. **Process Monitoring**: Real-time process status and resource usage monitoring
3. **Resource Limits**: CPU time, memory, file size, and other resource limits
4. **Inter-Process Communication**: Pipes, queues, shared memory, and other IPC mechanisms
5. **Process Tree Management**: Parent-child process relationship management
6. **Daemon Process Management**: Background process management

---

## 核心类 / Core Classes

### ProcessStatus
```python
class ProcessStatus(Enum):
    CREATED = "created"      # 已创建 / Created
    STARTING = "starting"    # 启动中 / Starting
    RUNNING = "running"      # 运行中 / Running
    SUSPENDED = "suspended"  # 已暂停 / Suspended
    TERMINATING = "terminating"  # 终止中 / Terminating
    TERMINATED = "terminated"    # 已终止 / Terminated
    ERROR = "error"          # 错误 / Error
    ZOMBIE = "zombie"        # 僵尸进程 / Zombie
```

### ProcessPriority
```python
class ProcessPriority(Enum):
    IDLE = "idle"           # 空闲 / Idle
    BELOW_NORMAL = "below_normal"  # 低于正常 / Below Normal
    NORMAL = "normal"       # 正常 / Normal
    ABOVE_NORMAL = "above_normal"  # 高于正常 / Above Normal
    HIGH = "high"           # 高 / High
    REALTIME = "realtime"   # 实时 / Realtime
```

### ProcessInfo
```python
@dataclass
class ProcessInfo:
    pid: int
    name: str
    command_line: str
    status: ProcessStatus
    start_time: float
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_rss: int = 0
    memory_vms: int = 0
    num_threads: int = 0
    num_fds: int = 0
    priority: ProcessPriority = ProcessPriority.NORMAL
    exit_code: Optional[int] = None
    exit_time: Optional[float] = None
    parent_pid: Optional[int] = None
    children_pids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 主要方法 / Main Methods

### 进程操作 / Process Operations

#### 启动进程 / Start Process
```python
def start_process(
    command: Union[str, List[str]],
    name: str = "",
    working_dir: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    resource_limits: Optional[ProcessResourceLimits] = None
) -> Optional[int]
```
启动新进程，返回进程ID。

Start a new process, returns the process ID.

#### 获取进程信息 / Get Process Info
```python
def get_process_info(pid: int) -> Optional[ProcessInfo]
```
获取指定进程的详细信息。

Get detailed information for the specified process.

#### 列出进程 / List Processes
```python
def list_processes(filter_by_user: Optional[int] = None) -> List[ProcessInfo]
```
列出系统中所有进程。

List all processes in the system.

#### 终止进程 / Terminate Process
```python
def terminate_process(pid: int, force: bool = False) -> bool
```
终止指定进程，支持优雅终止和强制终止。

Terminate the specified process, supporting graceful and forceful termination.

#### 暂停/恢复进程 / Pause/Resume Process
```python
def suspend_process(pid: int) -> bool
def resume_process(pid: int) -> bool
```
暂停或恢复进程执行。

Pause or resume process execution.

---

## 使用示例 / Usage Examples

### 中文
```python
from models.system_control.process_manager import get_process_manager

manager = get_process_manager()

# 启动进程 / Start process
pid = manager.start_process("ping 127.0.0.1 -n 5", name="ping_test")

if pid:
    print(f"Process started: PID={pid}")
    
    # 获取进程信息 / Get process info
    time.sleep(1)
    proc_info = manager.get_process_info(pid)
    if proc_info:
        print(f"CPU: {proc_info.cpu_percent}%, Memory: {proc_info.memory_percent}%")
    
    # 等待完成 / Wait for completion
    time.sleep(6)

# 列出所有进程 / List all processes
processes = manager.list_processes()
print(f"Total processes: {len(processes)}")
```

### English
```python
from models.system_control.process_manager import get_process_manager

manager = get_process_manager()

# Start process
pid = manager.start_process("ping 127.0.0.1 -n 5", name="ping_test")

if pid:
    print(f"Process started: PID={pid}")
    
    # Get process info
    time.sleep(1)
    proc_info = manager.get_process_info(pid)
    if proc_info:
        print(f"CPU: {proc_info.cpu_percent}%, Memory: {proc_info.memory_percent}%")
    
    # Wait for completion
    time.sleep(6)

# List all processes
processes = manager.list_processes()
print(f"Total processes: {len(processes)}")
```

---

## 相关模块 / Related Modules

- [文件系统管理器](./filesystem-manager.md) - 文件系统管理
- [网络控制器](./network-controller.md) - 网络控制
- [服务管理器](./service-manager.md) - 服务管理
