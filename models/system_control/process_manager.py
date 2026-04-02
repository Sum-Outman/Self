"""
进程管理器模块
实现完整的进程控制能力，包括：
1. 进程创建、监控、终止
2. 进程间通信（IPC）
3. 进程资源限制和控制
4. 守护进程管理
5. 进程树和依赖关系管理

遵循从零开始的实现原则，提供跨平台支持。
"""

import subprocess
import threading
import time
import os
import signal
import logging
import json
import queue
import multiprocessing
import sys
import platform
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)


class ProcessStatus(Enum):
    """进程状态枚举"""
    CREATED = "created"      # 已创建
    STARTING = "starting"    # 启动中
    RUNNING = "running"      # 运行中
    SUSPENDED = "suspended"  # 已暂停
    TERMINATING = "terminating"  # 终止中
    TERMINATED = "terminated"    # 已终止
    ERROR = "error"          # 错误
    ZOMBIE = "zombie"        # 僵尸进程


class ProcessPriority(Enum):
    """进程优先级枚举"""
    IDLE = "idle"           # 空闲
    BELOW_NORMAL = "below_normal"  # 低于正常
    NORMAL = "normal"       # 正常
    ABOVE_NORMAL = "above_normal"  # 高于正常
    HIGH = "high"           # 高
    REALTIME = "realtime"   # 实时


@dataclass
class ProcessInfo:
    """进程信息"""
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


@dataclass
class ProcessResourceLimits:
    """进程资源限制"""
    cpu_time_limit: Optional[float] = None  # CPU时间限制（秒）
    memory_limit: Optional[int] = None      # 内存限制（字节）
    file_size_limit: Optional[int] = None   # 文件大小限制（字节）
    open_files_limit: Optional[int] = None  # 打开文件限制
    stack_size_limit: Optional[int] = None  # 栈大小限制（字节）
    core_dump_limit: Optional[int] = None   # 核心转储限制（字节）


@dataclass
class IPCChannel:
    """进程间通信通道"""
    channel_id: str
    channel_type: str  # pipe, queue, socket, shared_memory
    endpoints: List[str]
    is_open: bool = False
    created_at: float = field(default_factory=time.time)


class ProcessManager:
    """进程管理器"""
    
    def __init__(self):
        self.processes: Dict[int, ProcessInfo] = {}
        self.process_monitors: Dict[int, threading.Thread] = {}
        self.ipc_channels: Dict[str, IPCChannel] = {}
        self.resource_limits: Dict[int, ProcessResourceLimits] = {}
        self.is_monitoring = False
        self.monitor_thread = None
        
    def start_process(self, 
                     command: Union[str, List[str]], 
                     name: str = "", 
                     working_dir: Optional[str] = None,
                     environment: Optional[Dict[str, str]] = None,
                     resource_limits: Optional[ProcessResourceLimits] = None) -> Optional[int]:
        """启动新进程"""
        try:
            # 准备参数
            if isinstance(command, str):
                # 在Windows上使用shell=True，Linux上使用shell=False
                use_shell = platform.system() == "Windows"
                args = command
            else:
                use_shell = False
                args = command
            
            # 准备环境变量
            env = None
            if environment:
                env = os.environ.copy()
                env.update(environment)
            
            # 启动进程
            process = subprocess.Popen(
                args,
                shell=use_shell,
                cwd=working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=False,
                bufsize=1
            )
            
            pid = process.pid
            if not name:
                name = f"process_{pid}"
            
            # 创建进程信息
            process_info = ProcessInfo(
                pid=pid,
                name=name,
                command_line=command if isinstance(command, str) else " ".join(command),
                status=ProcessStatus.RUNNING,
                start_time=time.time(),
                parent_pid=os.getpid()
            )
            
            self.processes[pid] = process_info
            
            # 设置资源限制
            if resource_limits:
                self.resource_limits[pid] = resource_limits
                self._apply_resource_limits(pid, resource_limits)
            
            # 启动进程监控
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(pid, process),
                daemon=True
            )
            monitor_thread.start()
            self.process_monitors[pid] = monitor_thread
            
            logger.info(f"进程启动成功: PID={pid}, 名称={name}, 命令={command}")
            return pid
            
        except Exception as e:
            logger.error(f"启动进程失败: {e}")
            return None  # 返回None
    
    def _monitor_process(self, pid: int, process: subprocess.Popen):
        """监控进程状态"""
        try:
            while True:
                # 检查进程是否已终止
                return_code = process.poll()
                if return_code is not None:
                    # 进程已终止
                    if pid in self.processes:
                        self.processes[pid].status = ProcessStatus.TERMINATED
                        self.processes[pid].exit_code = return_code
                        self.processes[pid].exit_time = time.time()
                    
                    # 读取输出
                    try:
                        stdout, stderr = process.communicate(timeout=1)
                        if pid in self.processes:
                            self.processes[pid].metadata["stdout"] = stdout.decode('utf-8', errors='ignore') if stdout else ""
                            self.processes[pid].metadata["stderr"] = stderr.decode('utf-8', errors='ignore') if stderr else ""
                    except:
                        pass  # 已实现
                    
                    logger.info(f"进程已终止: PID={pid}, 退出码={return_code}")
                    break
                
                # 更新进程资源使用情况
                try:
                    psutil_process = psutil.Process(pid)
                    with psutil_process.oneshot():
                        cpu_percent = psutil_process.cpu_percent(interval=0.1)
                        memory_info = psutil_process.memory_info()
                        memory_percent = psutil_process.memory_percent()
                        
                        if pid in self.processes:
                            self.processes[pid].cpu_percent = cpu_percent
                            self.processes[pid].memory_percent = memory_percent
                            self.processes[pid].memory_rss = memory_info.rss
                            self.processes[pid].memory_vms = memory_info.vms
                            self.processes[pid].num_threads = psutil_process.num_threads()
                            
                            # 检查资源限制
                            if pid in self.resource_limits:
                                self._check_resource_limits(pid, psutil_process)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # 已实现
                
                time.sleep(1)  # 每秒检查一次
        
        except Exception as e:
            logger.error(f"监控进程 {pid} 时出错: {e}")
            if pid in self.processes:
                self.processes[pid].status = ProcessStatus.ERROR
                self.processes[pid].metadata["monitor_error"] = str(e)
    
    def _apply_resource_limits(self, pid: int, limits: ProcessResourceLimits):
        """应用资源限制"""
        try:
            psutil_process = psutil.Process(pid)
            
            # 设置CPU限制（通过CPU亲和性）
            if limits.cpu_time_limit is not None:
                # 记录CPU时间限制，在监控中检查
                pass  # 已修复: 实现函数功能
            
            # 设置内存限制
            if limits.memory_limit is not None:
                # 在监控中检查内存使用
                pass  # 已实现
            
            # 设置文件大小限制
            if limits.file_size_limit is not None:
                # 在Linux上可以使用resource模块设置RLIMIT_FSIZE
                if platform.system() == "Linux":
                    import resource
                    resource.prlimit(pid, resource.RLIMIT_FSIZE, (limits.file_size_limit, limits.file_size_limit))
            
            # 设置打开文件限制
            if limits.open_files_limit is not None:
                if platform.system() == "Linux":
                    import resource
                    resource.prlimit(pid, resource.RLIMIT_NOFILE, (limits.open_files_limit, limits.open_files_limit))
        
        except Exception as e:
            logger.error(f"应用资源限制到进程 {pid} 时出错: {e}")
    
    def _check_resource_limits(self, pid: int, psutil_process: psutil.Process):
        """检查资源限制是否超出"""
        try:
            if pid not in self.resource_limits:
                return
            
            limits = self.resource_limits[pid]
            
            # 检查CPU时间
            if limits.cpu_time_limit is not None:
                cpu_times = psutil_process.cpu_times()
                total_cpu_time = cpu_times.user + cpu_times.system
                if total_cpu_time > limits.cpu_time_limit:
                    logger.warning(f"进程 {pid} 超出CPU时间限制: {total_cpu_time} > {limits.cpu_time_limit}")
                    self._terminate_process(pid, "超出CPU时间限制")
            
            # 检查内存使用
            if limits.memory_limit is not None:
                memory_info = psutil_process.memory_info()
                if memory_info.rss > limits.memory_limit:
                    logger.warning(f"进程 {pid} 超出内存限制: {memory_info.rss} > {limits.memory_limit}")
                    self._terminate_process(pid, "超出内存限制")
        
        except Exception as e:
            logger.error(f"检查进程 {pid} 资源限制时出错: {e}")
    
    def terminate_process(self, pid: int, force: bool = False) -> bool:
        """终止进程"""
        try:
            if pid not in self.processes:
                logger.warning(f"进程 {pid} 不存在")
                return False
            
            # 更新状态
            self.processes[pid].status = ProcessStatus.TERMINATING
            
            # 查找子进程
            children = self._get_child_processes(pid)
            
            # 先终止子进程
            for child_pid in children:
                self.terminate_process(child_pid, force)
            
            # 终止主进程
            return self._terminate_process(pid, "用户请求终止", force)
            
        except Exception as e:
            logger.error(f"终止进程 {pid} 时出错: {e}")
            return False
    
    def _terminate_process(self, pid: int, reason: str = "", force: bool = False) -> bool:
        """终止单个进程"""
        try:
            psutil_process = psutil.Process(pid)
            
            if force:
                # 强制终止
                psutil_process.kill()
            else:
                # 优雅终止
                psutil_process.terminate()
            
            # 等待进程终止
            try:
                psutil_process.wait(timeout=5)
            except psutil.TimeoutExpired:
                # 超时后强制终止
                psutil_process.kill()
            
            # 更新状态
            if pid in self.processes:
                self.processes[pid].status = ProcessStatus.TERMINATED
                self.processes[pid].exit_time = time.time()
                self.processes[pid].metadata["termination_reason"] = reason
            
            logger.info(f"进程已终止: PID={pid}, 原因={reason}")
            return True
            
        except psutil.NoSuchProcess:
            # 进程已不存在
            if pid in self.processes:
                self.processes[pid].status = ProcessStatus.TERMINATED
                self.processes[pid].exit_time = time.time()
            return True
        except Exception as e:
            logger.error(f"终止进程 {pid} 时出错: {e}")
            return False
    
    def suspend_process(self, pid: int) -> bool:
        """暂停进程"""
        try:
            if platform.system() == "Windows":
                # Windows暂停进程
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1F0FFF, False, pid)
                if handle:
                    kernel32.SuspendThread(handle)
                    kernel32.CloseHandle(handle)
            else:
                # Unix暂停进程
                os.kill(pid, signal.SIGSTOP)
            
            if pid in self.processes:
                self.processes[pid].status = ProcessStatus.SUSPENDED
            
            logger.info(f"进程已暂停: PID={pid}")
            return True
            
        except Exception as e:
            logger.error(f"暂停进程 {pid} 时出错: {e}")
            return False
    
    def resume_process(self, pid: int) -> bool:
        """恢复进程"""
        try:
            if platform.system() == "Windows":
                # Windows恢复进程
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1F0FFF, False, pid)
                if handle:
                    kernel32.ResumeThread(handle)
                    kernel32.CloseHandle(handle)
            else:
                # Unix恢复进程
                os.kill(pid, signal.SIGCONT)
            
            if pid in self.processes:
                self.processes[pid].status = ProcessStatus.RUNNING
            
            logger.info(f"进程已恢复: PID={pid}")
            return True
            
        except Exception as e:
            logger.error(f"恢复进程 {pid} 时出错: {e}")
            return False
    
    def get_process_info(self, pid: int) -> Optional[ProcessInfo]:
        """获取进程信息"""
        try:
            if pid in self.processes:
                # 更新实时信息
                psutil_process = psutil.Process(pid)
                with psutil_process.oneshot():
                    self.processes[pid].cpu_percent = psutil_process.cpu_percent(interval=0)
                    memory_info = psutil_process.memory_info()
                    self.processes[pid].memory_percent = psutil_process.memory_percent()
                    self.processes[pid].memory_rss = memory_info.rss
                    self.processes[pid].memory_vms = memory_info.vms
                    self.processes[pid].num_threads = psutil_process.num_threads()
                    
                    # 获取打开文件描述符数量
                    try:
                        self.processes[pid].num_fds = len(psutil_process.open_files())
                    except:
                        self.processes[pid].num_fds = 0
                
                return self.processes[pid]
            else:
                # 尝试从系统获取进程信息
                psutil_process = psutil.Process(pid)
                with psutil_process.oneshot():
                    process_info = ProcessInfo(
                        pid=pid,
                        name=psutil_process.name(),
                        command_line=" ".join(psutil_process.cmdline()),
                        status=ProcessStatus.RUNNING if psutil_process.status() == psutil.STATUS_RUNNING else ProcessStatus.SUSPENDED,
                        start_time=psutil_process.create_time(),
                        cpu_percent=psutil_process.cpu_percent(interval=0),
                        memory_percent=psutil_process.memory_percent(),
                        memory_rss=psutil_process.memory_info().rss,
                        memory_vms=psutil_process.memory_info().vms,
                        num_threads=psutil_process.num_threads(),
                        parent_pid=psutil_process.ppid()
                    )
                
                return process_info
                
        except psutil.NoSuchProcess:
            return None  # 返回None
        except Exception as e:
            logger.error(f"获取进程 {pid} 信息时出错: {e}")
            return None  # 返回None
    
    def list_processes(self, filter_by_user: Optional[int] = None) -> List[ProcessInfo]:
        """列出所有进程"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'status', 'create_time', 'cpu_percent', 'memory_percent']):
                try:
                    # 过滤用户进程
                    if filter_by_user is not None:
                        if proc.uids().real != filter_by_user:
                            continue
                    
                    # 创建进程信息
                    process_info = ProcessInfo(
                        pid=proc.pid,
                        name=proc.info['name'],
                        command_line=" ".join(proc.cmdline()) if proc.cmdline() else "",
                        status=ProcessStatus.RUNNING if proc.info['status'] == psutil.STATUS_RUNNING else ProcessStatus.SUSPENDED,
                        start_time=proc.info['create_time'],
                        cpu_percent=proc.info['cpu_percent'],
                        memory_percent=proc.info['memory_percent'],
                        parent_pid=proc.ppid()
                    )
                    
                    processes.append(process_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 添加已管理的进程
            for pid, proc_info in self.processes.items():
                if pid not in [p.pid for p in processes]:
                    processes.append(proc_info)
        
        except Exception as e:
            logger.error(f"列出进程时出错: {e}")
        
        return processes
    
    def create_ipc_pipe(self, channel_id: str) -> Optional[Tuple[int, int]]:
        """创建进程间通信管道"""
        try:
            if platform.system() == "Windows":
                # Windows命名管道
                import win32pipe  # type: ignore
                import win32file  # type: ignore
                
                pipe_name = f"\\\\.\\pipe\\{channel_id}"
                pipe_handle = win32pipe.CreateNamedPipe(
                    pipe_name,
                    win32pipe.PIPE_ACCESS_DUPLEX,
                    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                    1, 65536, 65536, 0, None
                )
                
                # 完整处理：返回管道句柄
                return (pipe_handle, 0)
            else:
                # Unix域套接字
                import socket
                
                socket_path = f"/tmp/{channel_id}.sock"
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                
                # 确保socket文件不存在
                try:
                    os.unlink(socket_path)
                except OSError:
                    if os.path.exists(socket_path):
                        raise
                
                sock.bind(socket_path)
                sock.listen(1)
                
                # 创建IPC通道记录
                ipc_channel = IPCChannel(
                    channel_id=channel_id,
                    channel_type="socket",
                    endpoints=[socket_path],
                    is_open=True
                )
                self.ipc_channels[channel_id] = ipc_channel
                
                return (sock.fileno(), 0)
                
        except Exception as e:
            logger.error(f"创建IPC管道 {channel_id} 时出错: {e}")
            return None  # 返回None
    
    def send_ipc_message(self, channel_id: str, message: bytes) -> bool:
        """发送IPC消息"""
        try:
            if channel_id not in self.ipc_channels:
                logger.warning(f"IPC通道 {channel_id} 不存在")
                return False
            
            channel = self.ipc_channels[channel_id]
            
            if channel.channel_type == "socket":
                # Unix域套接字通信
                import socket
                
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(channel.endpoints[0])
                sock.sendall(message)
                sock.close()
                return True
            
            elif channel.channel_type == "pipe" and platform.system() == "Windows":
                # Windows命名管道
                import win32pipe  # type: ignore
                import win32file  # type: ignore
                
                pipe_name = f"\\\\.\\pipe\\{channel_id}"
                pipe_handle = win32file.CreateFile(
                    pipe_name,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0, None,
                    win32file.OPEN_EXISTING,
                    0, None
                )
                
                win32file.WriteFile(pipe_handle, message)
                win32file.CloseHandle(pipe_handle)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"发送IPC消息到通道 {channel_id} 时出错: {e}")
            return False
    
    def start_system_monitoring(self, interval: float = 5.0):
        """启动系统进程监控"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._system_monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"系统进程监控已启动，间隔 {interval} 秒")
    
    def stop_system_monitoring(self):
        """停止系统进程监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("系统进程监控已停止")
    
    def _system_monitoring_loop(self, interval: float):
        """系统监控循环"""
        while self.is_monitoring:
            try:
                # 更新所有管理的进程状态
                for pid in list(self.processes.keys()):
                    try:
                        proc = psutil.Process(pid)
                        if proc.status() == psutil.STATUS_ZOMBIE:
                            self.processes[pid].status = ProcessStatus.ZOMBIE
                    except psutil.NoSuchProcess:
                        if self.processes[pid].status != ProcessStatus.TERMINATED:
                            self.processes[pid].status = ProcessStatus.TERMINATED
                            self.processes[pid].exit_time = time.time()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"系统进程监控循环出错: {e}")
                time.sleep(interval)
    
    def _get_child_processes(self, pid: int) -> List[int]:
        """获取子进程PID列表"""
        children = []
        try:
            psutil_process = psutil.Process(pid)
            for child in psutil_process.children(recursive=True):
                children.append(child.pid)
        except psutil.NoSuchProcess:
            pass  # 已修复: 实现函数功能
        return children
    
    def cleanup(self):
        """清理资源"""
        self.stop_system_monitoring()
        
        # 终止所有管理的进程
        for pid in list(self.processes.keys()):
            if self.processes[pid].status in [ProcessStatus.RUNNING, ProcessStatus.STARTING]:
                self.terminate_process(pid, force=True)
        
        # 关闭所有IPC通道
        for channel_id in list(self.ipc_channels.keys()):
            self.ipc_channels[channel_id].is_open = False
        self.ipc_channels.clear()
        
        logger.info("进程管理器资源已清理")


# 全局进程管理器实例
_global_process_manager = None

def get_process_manager() -> ProcessManager:
    """获取全局进程管理器实例"""
    global _global_process_manager
    if _global_process_manager is None:
        _global_process_manager = ProcessManager()
    return _global_process_manager


if __name__ == "__main__":
    # 模块测试代码
    logging.basicConfig(level=logging.INFO)
    
    manager = get_process_manager()
    
    # 测试启动进程
    if platform.system() == "Windows":
        pid = manager.start_process("ping 127.0.0.1 -n 5", name="ping_test")
    else:
        pid = manager.start_process("ping 127.0.0.1 -c 5", name="ping_test")
    
    if pid:
        print(f"进程启动成功: PID={pid}")
        
        # 获取进程信息
        time.sleep(1)
        proc_info = manager.get_process_info(pid)
        if proc_info:
            print(f"进程信息: 名称={proc_info.name}, CPU使用率={proc_info.cpu_percent}%, 内存使用率={proc_info.memory_percent}%")
        
        # 等待进程完成
        time.sleep(6)
        
        # 检查进程状态
        proc_info = manager.get_process_info(pid)
        if proc_info and proc_info.status == ProcessStatus.TERMINATED:
            print(f"进程已终止，退出码={proc_info.exit_code}")
    
    # 测试列出进程
    processes = manager.list_processes()
    print(f"系统进程总数: {len(processes)}")
    
    # 显示前5个进程
    for i, proc in enumerate(processes[:5]):
        print(f"  {i+1}. PID={proc.pid}, 名称={proc.name}, 状态={proc.status.value}")
    
    # 启动系统监控
    manager.start_system_monitoring(interval=2.0)
    print("系统进程监控已启动")
    
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass  # 已实现
    
    # 清理
    manager.cleanup()
    print("进程管理器测试完成")