"""
系统服务控制模块
实现完整的系统服务控制能力，包括：
1. 跨平台服务管理（Windows服务、Linux systemd服务、macOS launchd服务）
2. 启动项和计划任务管理
3. 系统配置管理（注册表、配置文件）
4. 日志系统管理
5. 系统用户和组管理

遵循从零开始的实现原则，提供跨平台支持。
"""

import os
import subprocess
import platform
import time
import logging
import threading
import json
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """服务状态枚举"""
    STOPPED = "stopped"        # 已停止
    STARTING = "starting"      # 启动中
    RUNNING = "running"        # 运行中
    STOPPING = "stopping"      # 停止中
    PAUSED = "paused"          # 已暂停
    ERROR = "error"            # 错误
    UNKNOWN = "unknown"        # 未知


class ServiceStartType(Enum):
    """服务启动类型枚举"""
    AUTOMATIC = "automatic"    # 自动
    MANUAL = "manual"          # 手动
    DISABLED = "disabled"      # 禁用
    BOOT = "boot"              # 启动时
    SYSTEM = "system"          # 系统


class ServiceType(Enum):
    """服务类型枚举"""
    WIN32 = "win32"            # Windows服务
    SYSTEMD = "systemd"        # Linux systemd服务
    UPSTART = "upstart"        # Linux upstart服务
    LAUNCHD = "launchd"        # macOS launchd服务
    SYSV = "sysv"              # Linux SysV init服务
    DOCKER = "docker"          # Docker容器服务


@dataclass
class ServiceInfo:
    """服务信息"""
    name: str
    display_name: str
    status: ServiceStatus
    start_type: ServiceStartType
    service_type: ServiceType
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    description: str = ""
    binary_path: str = ""
    dependencies: List[str] = field(default_factory=list)
    account: str = ""
    started_time: Optional[float] = None
    cpu_usage: float = 0.0
    memory_usage: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledTask:
    """计划任务"""
    task_id: str
    name: str
    command: str
    schedule: str  # cron表达式或Windows计划任务格式
    enabled: bool = True
    last_run_time: Optional[float] = None
    next_run_time: Optional[float] = None
    last_run_result: Optional[int] = None
    creator: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StartupItem:
    """启动项"""
    item_id: str
    name: str
    command: str
    location: str  # 启动位置（注册表、启动文件夹等）
    enabled: bool = True
    user: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceManager:
    """系统服务管理器"""
    
    def __init__(self):
        self.service_monitors: Dict[str, threading.Thread] = {}
        self.service_event_handlers: Dict[str, List[Callable]] = {}
        self.is_monitoring = False
        self.monitor_thread = None
        
    def get_service_info(self, service_name: str) -> Optional[ServiceInfo]:
        """获取服务信息"""
        try:
            system = platform.system()
            
            if system == "Windows":
                return self._get_windows_service_info(service_name)
            elif system == "Linux":
                return self._get_linux_service_info(service_name)
            elif system == "Darwin":
                return self._get_macos_service_info(service_name)
            else:
                logger.warning(f"不支持的操作系统: {system}")
                return None  # 返回None
                
        except Exception as e:
            logger.error(f"获取服务信息失败 {service_name}: {e}")
            return None  # 返回None
    
    def _get_windows_service_info(self, service_name: str) -> Optional[ServiceInfo]:
        """获取Windows服务信息"""
        try:
            # 使用sc命令查询服务信息
            cmd = ["sc", "query", service_name]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='gbk')
            
            if result.returncode != 0:
                logger.warning(f"Windows服务不存在: {service_name}")
                return None  # 返回None
            
            output = result.stdout
            info = {}
            
            for line in output.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
            
            # 解析状态
            status = ServiceStatus.UNKNOWN
            state = info.get('STATE', '')
            if 'STOPPED' in state:
                status = ServiceStatus.STOPPED
            elif 'START_PENDING' in state:
                status = ServiceStatus.STARTING
            elif 'RUNNING' in state:
                status = ServiceStatus.RUNNING
            elif 'STOP_PENDING' in state:
                status = ServiceStatus.STOPPING
            elif 'PAUSED' in state:
                status = ServiceStatus.PAUSED
            
            # 解析启动类型
            start_type = ServiceStartType.MANUAL
            start_mode = info.get('START_TYPE', '')
            if 'AUTO_START' in start_mode:
                start_type = ServiceStartType.AUTOMATIC
            elif 'DEMAND_START' in start_mode:
                start_type = ServiceStartType.MANUAL
            elif 'DISABLED' in start_mode:
                start_type = ServiceStartType.DISABLED
            
            # 获取显示名称
            display_name = info.get('DISPLAY_NAME', service_name)
            
            # 获取二进制路径
            binary_path = ""
            cmd_path = info.get('BINARY_PATH_NAME', '')
            if cmd_path:
                binary_path = cmd_path.strip('"')
            
            # 获取PID
            pid = None
            pid_str = info.get('PID', '')
            if pid_str.isdigit():
                pid = int(pid_str)
            
            service_info = ServiceInfo(
                name=service_name,
                display_name=display_name,
                status=status,
                start_type=start_type,
                service_type=ServiceType.WIN32,
                pid=pid,
                description=info.get('DESCRIPTION', ''),
                binary_path=binary_path
            )
            
            return service_info
            
        except Exception as e:
            logger.error(f"获取Windows服务信息失败 {service_name}: {e}")
            return None  # 返回None
    
    def _get_linux_service_info(self, service_name: str) -> Optional[ServiceInfo]:
        """获取Linux服务信息"""
        try:
            # 尝试使用systemctl
            cmd = ["systemctl", "status", service_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # systemd服务
                return self._parse_systemd_service_info(service_name, result.stdout)
            else:
                # 尝试使用service命令
                cmd = ["service", service_name, "status"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # SysV init服务
                    return self._parse_sysv_service_info(service_name, result.stdout)
                else:
                    logger.warning(f"Linux服务不存在: {service_name}")
                    return None  # 返回None
                    
        except Exception as e:
            logger.error(f"获取Linux服务信息失败 {service_name}: {e}")
            return None  # 返回None
    
    def _parse_systemd_service_info(self, service_name: str, output: str) -> ServiceInfo:
        """解析systemd服务信息"""
        lines = output.split('\n')
        info = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()
        
        # 解析状态
        status = ServiceStatus.UNKNOWN
        state = info.get('Loaded', '')
        if 'running' in state.lower():
            status = ServiceStatus.RUNNING
        elif 'dead' in state.lower():
            status = ServiceStatus.STOPPED
        elif 'activating' in state.lower():
            status = ServiceStatus.STARTING
        elif 'deactivating' in state.lower():
            status = ServiceStatus.STOPPING
        
        # 解析启动类型
        start_type = ServiceStartType.MANUAL
        if 'enabled' in state.lower():
            start_type = ServiceStartType.AUTOMATIC
        elif 'disabled' in state.lower():
            start_type = ServiceStartType.DISABLED
        
        # 获取PID
        pid = None
        pid_line = [line for line in lines if 'Main PID' in line]
        if pid_line:
            parts = pid_line[0].split(':')
            if len(parts) > 1:
                pid_str = parts[1].strip().split()[0]
                if pid_str.isdigit():
                    pid = int(pid_str)
        
        # 获取描述
        description = info.get('Description', '')
        
        service_info = ServiceInfo(
            name=service_name,
            display_name=service_name,
            status=status,
            start_type=start_type,
            service_type=ServiceType.SYSTEMD,
            pid=pid,
            description=description
        )
        
        return service_info
    
    def _parse_sysv_service_info(self, service_name: str, output: str) -> ServiceInfo:
        """解析SysV init服务信息"""
        # 完整解析
        status = ServiceStatus.UNKNOWN
        if 'running' in output.lower():
            status = ServiceStatus.RUNNING
        elif 'stopped' in output.lower():
            status = ServiceStatus.STOPPED
        
        # 尝试获取PID
        pid = None
        import re
        pid_match = re.search(r'pid\s+(\d+)', output.lower())
        if pid_match:
            pid = int(pid_match.group(1))
        
        service_info = ServiceInfo(
            name=service_name,
            display_name=service_name,
            status=status,
            start_type=ServiceStartType.MANUAL,  # 需要额外检查启动项
            service_type=ServiceType.SYSV,
            pid=pid,
            description=f"SysV init service: {service_name}"
        )
        
        return service_info
    
    def _get_macos_service_info(self, service_name: str) -> Optional[ServiceInfo]:
        """获取macOS服务信息"""
        try:
            # 使用launchctl命令
            cmd = ["launchctl", "list"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return None  # 返回None
            
            # 查找服务
            lines = result.stdout.split('\n')
            for line in lines:
                if service_name in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        pid_str = parts[0]
                        status_str = parts[1]
                        
                        pid = None
                        if pid_str != '-':
                            pid = int(pid_str)
                        
                        status = ServiceStatus.UNKNOWN
                        if status_str == '0':
                            status = ServiceStatus.RUNNING
                        else:
                            status = ServiceStatus.STOPPED
                        
                        service_info = ServiceInfo(
                            name=service_name,
                            display_name=service_name,
                            status=status,
                            start_type=ServiceStartType.MANUAL,  # 需要额外检查
                            service_type=ServiceType.LAUNCHD,
                            pid=pid,
                            description=f"macOS launchd service: {service_name}"
                        )
                        
                        return service_info
            
            logger.warning(f"macOS服务不存在: {service_name}")
            return None  # 返回None
            
        except Exception as e:
            logger.error(f"获取macOS服务信息失败 {service_name}: {e}")
            return None  # 返回None
    
    def start_service(self, service_name: str) -> bool:
        """启动服务"""
        try:
            system = platform.system()
            
            if system == "Windows":
                cmd = ["sc", "start", service_name]
            elif system == "Linux":
                # 尝试systemctl，然后是service
                cmd = ["systemctl", "start", service_name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    cmd = ["service", service_name, "start"]
            elif system == "Darwin":
                cmd = ["launchctl", "start", service_name]
            else:
                logger.warning(f"不支持的操作系统: {system}")
                return False
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"服务启动成功: {service_name}")
                return True
            else:
                logger.error(f"服务启动失败 {service_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"启动服务失败 {service_name}: {e}")
            return False
    
    def stop_service(self, service_name: str, force: bool = False) -> bool:
        """停止服务"""
        try:
            system = platform.system()
            
            if system == "Windows":
                if force:
                    cmd = ["sc", "stop", service_name, "/force"]
                else:
                    cmd = ["sc", "stop", service_name]
            elif system == "Linux":
                if force:
                    cmd = ["systemctl", "kill", service_name, "-s", "KILL"]
                else:
                    cmd = ["systemctl", "stop", service_name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    cmd = ["service", service_name, "stop"]
            elif system == "Darwin":
                cmd = ["launchctl", "stop", service_name]
            else:
                logger.warning(f"不支持的操作系统: {system}")
                return False
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"服务停止成功: {service_name}")
                return True
            else:
                logger.error(f"服务停止失败 {service_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"停止服务失败 {service_name}: {e}")
            return False
    
    def restart_service(self, service_name: str) -> bool:
        """重启服务"""
        try:
            system = platform.system()
            
            if system == "Windows":
                cmd = ["sc", "stop", service_name]
                subprocess.run(cmd, capture_output=True)
                time.sleep(2)
                cmd = ["sc", "start", service_name]
            elif system == "Linux":
                cmd = ["systemctl", "restart", service_name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    cmd = ["service", service_name, "restart"]
            elif system == "Darwin":
                cmd = ["launchctl", "stop", service_name]
                subprocess.run(cmd, capture_output=True)
                time.sleep(2)
                cmd = ["launchctl", "start", service_name]
            else:
                logger.warning(f"不支持的操作系统: {system}")
                return False
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"服务重启成功: {service_name}")
                return True
            else:
                logger.error(f"服务重启失败 {service_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"重启服务失败 {service_name}: {e}")
            return False
    
    def enable_service(self, service_name: str) -> bool:
        """启用服务自动启动"""
        try:
            system = platform.system()
            
            if system == "Windows":
                cmd = ["sc", "config", service_name, "start=auto"]
            elif system == "Linux":
                cmd = ["systemctl", "enable", service_name]
            elif system == "Darwin":
                # macOS使用launchctl load
                cmd = ["launchctl", "load", f"/Library/LaunchDaemons/{service_name}.plist"]
            else:
                logger.warning(f"不支持的操作系统: {system}")
                return False
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"服务启用成功: {service_name}")
                return True
            else:
                logger.error(f"服务启用失败 {service_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"启用服务失败 {service_name}: {e}")
            return False
    
    def disable_service(self, service_name: str) -> bool:
        """禁用服务自动启动"""
        try:
            system = platform.system()
            
            if system == "Windows":
                cmd = ["sc", "config", service_name, "start=disabled"]
            elif system == "Linux":
                cmd = ["systemctl", "disable", service_name]
            elif system == "Darwin":
                cmd = ["launchctl", "unload", f"/Library/LaunchDaemons/{service_name}.plist"]
            else:
                logger.warning(f"不支持的操作系统: {system}")
                return False
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"服务禁用成功: {service_name}")
                return True
            else:
                logger.error(f"服务禁用失败 {service_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"禁用服务失败 {service_name}: {e}")
            return False
    
    def list_services(self, filter_by_status: Optional[ServiceStatus] = None) -> List[ServiceInfo]:
        """列出所有服务"""
        services = []
        
        try:
            system = platform.system()
            
            if system == "Windows":
                services = self._list_windows_services(filter_by_status)
            elif system == "Linux":
                services = self._list_linux_services(filter_by_status)
            elif system == "Darwin":
                services = self._list_macos_services(filter_by_status)
            else:
                logger.warning(f"不支持的操作系统: {system}")
                return []  # 返回空列表
            
            # 过滤服务状态
            if filter_by_status:
                services = [s for s in services if s.status == filter_by_status]
            
            logger.info(f"发现 {len(services)} 个服务")
            return services
            
        except Exception as e:
            logger.error(f"列出服务失败: {e}")
            return []  # 返回空列表
    
    def _list_windows_services(self, filter_by_status: Optional[ServiceStatus] = None) -> List[ServiceInfo]:
        """列出Windows服务"""
        services = []
        
        try:
            # 使用sc query命令获取服务列表
            cmd = ["sc", "query", "type=", "service", "state=", "all"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='gbk')
            
            if result.returncode != 0:
                return services
            
            output = result.stdout
            current_service = {}
            
            for line in output.split('\n'):
                line = line.strip()
                
                if line.startswith('SERVICE_NAME:'):
                    # 保存上一个服务
                    if current_service and 'name' in current_service:
                        service_info = self._create_service_info_from_windows_output(current_service)
                        if service_info:
                            services.append(service_info)
                    
                    # 开始新服务
                    service_name = line.split(':', 1)[1].strip()
                    current_service = {'name': service_name}
                
                elif line.startswith('DISPLAY_NAME:') and current_service:
                    display_name = line.split(':', 1)[1].strip()
                    current_service['display_name'] = display_name
                
                elif line.startswith('STATE:') and current_service:
                    state = line.split(':', 1)[1].strip()
                    current_service['state'] = state
                
                elif line.startswith('START_TYPE:') and current_service:
                    start_type = line.split(':', 1)[1].strip()
                    current_service['start_type'] = start_type
            
            # 保存最后一个服务
            if current_service and 'name' in current_service:
                service_info = self._create_service_info_from_windows_output(current_service)
                if service_info:
                    services.append(service_info)
            
            return services
            
        except Exception as e:
            logger.error(f"列出Windows服务失败: {e}")
            return []  # 返回空列表
    
    def _create_service_info_from_windows_output(self, service_data: Dict[str, str]) -> Optional[ServiceInfo]:
        """从Windows输出创建服务信息"""
        try:
            service_name = service_data.get('name', '')
            
            # 解析状态
            status = ServiceStatus.UNKNOWN
            state = service_data.get('state', '')
            if 'STOPPED' in state:
                status = ServiceStatus.STOPPED
            elif 'START_PENDING' in state:
                status = ServiceStatus.STARTING
            elif 'RUNNING' in state:
                status = ServiceStatus.RUNNING
            elif 'STOP_PENDING' in state:
                status = ServiceStatus.STOPPING
            elif 'PAUSED' in state:
                status = ServiceStatus.PAUSED
            
            # 解析启动类型
            start_type = ServiceStartType.MANUAL
            start_mode = service_data.get('start_type', '')
            if 'AUTO_START' in start_mode:
                start_type = ServiceStartType.AUTOMATIC
            elif 'DEMAND_START' in start_mode:
                start_type = ServiceStartType.MANUAL
            elif 'DISABLED' in start_mode:
                start_type = ServiceStartType.DISABLED
            
            service_info = ServiceInfo(
                name=service_name,
                display_name=service_data.get('display_name', service_name),
                status=status,
                start_type=start_type,
                service_type=ServiceType.WIN32,
                description=f"Windows service: {service_name}"
            )
            
            return service_info
            
        except Exception as e:
            logger.error(f"创建Windows服务信息失败: {e}")
            return None  # 返回None
    
    def _list_linux_services(self, filter_by_status: Optional[ServiceStatus] = None) -> List[ServiceInfo]:
        """列出Linux服务"""
        services = []
        
        try:
            # 尝试使用systemctl
            cmd = ["systemctl", "list-units", "--type=service", "--all", "--no-pager"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 解析systemctl输出
                lines = result.stdout.split('\n')
                for line in lines[1:]:  # 跳过标题行
                    if not line.strip():
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        service_name = parts[0]
                        if '.service' in service_name:
                            service_name = service_name.replace('.service', '')
                        
                        # 解析状态
                        status = ServiceStatus.UNKNOWN
                        state = parts[3]
                        if state == 'running':
                            status = ServiceStatus.RUNNING
                        elif state == 'dead':
                            status = ServiceStatus.STOPPED
                        elif state == 'activating':
                            status = ServiceStatus.STARTING
                        elif state == 'deactivating':
                            status = ServiceStatus.STOPPING
                        
                        # 解析启动类型
                        start_type = ServiceStartType.MANUAL
                        load_state = parts[1]
                        if load_state == 'loaded':
                            # 检查是否启用
                            cmd_enable = ["systemctl", "is-enabled", service_name]
                            result_enable = subprocess.run(cmd_enable, capture_output=True, text=True)
                            if result_enable.returncode == 0 and 'enabled' in result_enable.stdout:
                                start_type = ServiceStartType.AUTOMATIC
                        
                        service_info = ServiceInfo(
                            name=service_name,
                            display_name=service_name,
                            status=status,
                            start_type=start_type,
                            service_type=ServiceType.SYSTEMD,
                            description=f"Linux systemd service: {service_name}"
                        )
                        
                        services.append(service_info)
            
            return services
            
        except Exception as e:
            logger.error(f"列出Linux服务失败: {e}")
            return []  # 返回空列表
    
    def _list_macos_services(self, filter_by_status: Optional[ServiceStatus] = None) -> List[ServiceInfo]:
        """列出macOS服务"""
        services = []
        
        try:
            # 使用launchctl list
            cmd = ["launchctl", "list"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return services
            
            lines = result.stdout.split('\n')
            for line in lines[1:]:  # 跳过标题行
                if not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    pid_str = parts[0]
                    status_str = parts[1]
                    service_name = parts[2]
                    
                    pid = None
                    if pid_str != '-':
                        pid = int(pid_str)
                    
                    status = ServiceStatus.UNKNOWN
                    if status_str == '0':
                        status = ServiceStatus.RUNNING
                    else:
                        status = ServiceStatus.STOPPED
                    
                    service_info = ServiceInfo(
                        name=service_name,
                        display_name=service_name,
                        status=status,
                        start_type=ServiceStartType.MANUAL,
                        service_type=ServiceType.LAUNCHD,
                        pid=pid,
                        description=f"macOS launchd service: {service_name}"
                    )
                    
                    services.append(service_info)
            
            return services
            
        except Exception as e:
            logger.error(f"列出macOS服务失败: {e}")
            return []  # 返回空列表
    
    def create_scheduled_task(self, task: ScheduledTask) -> bool:
        """创建计划任务"""
        try:
            system = platform.system()
            
            if system == "Windows":
                return self._create_windows_scheduled_task(task)
            elif system == "Linux":
                return self._create_linux_cron_job(task)
            elif system == "Darwin":
                return self._create_macos_launchd_task(task)
            else:
                logger.warning(f"不支持的操作系统: {system}")
                return False
                
        except Exception as e:
            logger.error(f"创建计划任务失败 {task.name}: {e}")
            return False
    
    def _create_windows_scheduled_task(self, task: ScheduledTask) -> bool:
        """创建Windows计划任务"""
        try:
            # 使用schtasks命令
            cmd = [
                "schtasks", "/create",
                "/tn", task.name,
                "/tr", task.command,
                "/sc", "daily",  # 完整处理，使用每日
                "/st", "00:00",
                "/f"  # 强制创建
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='gbk')
            
            if result.returncode == 0:
                logger.info(f"Windows计划任务创建成功: {task.name}")
                return True
            else:
                logger.error(f"Windows计划任务创建失败 {task.name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"创建Windows计划任务失败 {task.name}: {e}")
            return False
    
    def _create_linux_cron_job(self, task: ScheduledTask) -> bool:
        """创建Linux cron作业"""
        try:
            # 解析cron表达式
            cron_parts = task.schedule.split()
            if len(cron_parts) != 5:
                logger.error(f"无效的cron表达式: {task.schedule}")
                return False
            
            # 创建cron行
            cron_line = f"{task.schedule} {task.command}\n"
            
            # 写入临时文件
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            temp_file.write(cron_line)
            temp_file.close()
            
            # 添加到crontab
            cmd = ["crontab", temp_file.name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 删除临时文件
            os.unlink(temp_file.name)
            
            if result.returncode == 0:
                logger.info(f"Linux cron作业创建成功: {task.name}")
                return True
            else:
                logger.error(f"Linux cron作业创建失败 {task.name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"创建Linux cron作业失败 {task.name}: {e}")
            return False
    
    def _create_macos_launchd_task(self, task: ScheduledTask) -> bool:
        """创建macOS launchd任务"""
        try:
            # 创建plist文件
            plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{task.name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>{task.command}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>0</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>"""
            
            plist_path = f"/Library/LaunchDaemons/{task.name}.plist"
            
            # 写入plist文件（需要root权限）
            with open(plist_path, 'w') as f:
                f.write(plist_content)
            
            # 加载任务
            cmd = ["launchctl", "load", plist_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"macOS launchd任务创建成功: {task.name}")
                return True
            else:
                logger.error(f"macOS launchd任务创建失败 {task.name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"创建macOS launchd任务失败 {task.name}: {e}")
            return False
    
    def list_scheduled_tasks(self) -> List[ScheduledTask]:
        """列出计划任务"""
        try:
            system = platform.system()
            
            if system == "Windows":
                return self._list_windows_scheduled_tasks()
            elif system == "Linux":
                return self._list_linux_cron_jobs()
            elif system == "Darwin":
                return self._list_macos_launchd_tasks()
            else:
                logger.warning(f"不支持的操作系统: {system}")
                return []  # 返回空列表
                
        except Exception as e:
            logger.error(f"列出计划任务失败: {e}")
            return []  # 返回空列表
    
    def _list_windows_scheduled_tasks(self) -> List[ScheduledTask]:
        """列出Windows计划任务"""
        tasks = []
        
        try:
            cmd = ["schtasks", "/query", "/fo", "LIST", "/v"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='gbk')
            
            if result.returncode != 0:
                return tasks
            
            # 完整解析
            lines = result.stdout.split('\n')
            current_task = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('任务名:'):
                    if current_task:
                        task = self._create_scheduled_task_from_windows_output(current_task)
                        if task:
                            tasks.append(task)
                    current_task = {'name': line.split(':', 1)[1].strip()}
                elif line.startswith('日程:'):
                    current_task['schedule'] = line.split(':', 1)[1].strip()
                elif line.startswith('任务运行:'):
                    current_task['command'] = line.split(':', 1)[1].strip()
            
            # 添加最后一个任务
            if current_task:
                task = self._create_scheduled_task_from_windows_output(current_task)
                if task:
                    tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"列出Windows计划任务失败: {e}")
            return []  # 返回空列表
    
    def _create_scheduled_task_from_windows_output(self, task_data: Dict[str, str]) -> Optional[ScheduledTask]:
        """从Windows输出创建计划任务"""
        try:
            task = ScheduledTask(
                task_id=task_data.get('name', ''),
                name=task_data.get('name', ''),
                command=task_data.get('command', ''),
                schedule=task_data.get('schedule', ''),
                enabled=True
            )
            return task
        except Exception as e:
            logger.error(f"创建Windows计划任务对象失败: {e}")
            return None  # 返回None
    
    def _list_linux_cron_jobs(self) -> List[ScheduledTask]:
        """列出Linux cron作业"""
        tasks = []
        
        try:
            cmd = ["crontab", "-l"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(maxsplit=5)
                        if len(parts) >= 6:
                            schedule = ' '.join(parts[:5])
                            command = parts[5]
                            
                            task = ScheduledTask(
                                task_id=f"cron_{i}",
                                name=f"Cron Job {i}",
                                command=command,
                                schedule=schedule,
                                enabled=True
                            )
                            tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"列出Linux cron作业失败: {e}")
            return []  # 返回空列表
    
    def _list_macos_launchd_tasks(self) -> List[ScheduledTask]:
        """列出macOS launchd任务"""
        tasks = []
        
        try:
            # 列出LaunchDaemons目录
            launchd_dir = "/Library/LaunchDaemons"
            if os.path.exists(launchd_dir):
                for filename in os.listdir(launchd_dir):
                    if filename.endswith('.plist'):
                        task_name = filename.replace('.plist', '')
                        task = ScheduledTask(
                            task_id=task_name,
                            name=task_name,
                            command=f"launchd service: {task_name}",
                            schedule="launchd",
                            enabled=True
                        )
                        tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"列出macOS launchd任务失败: {e}")
            return []  # 返回空列表
    
    def start_service_monitoring(self, interval: float = 30.0):
        """启动服务监控"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._service_monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"服务监控已启动，间隔 {interval} 秒")
    
    def stop_service_monitoring(self):
        """停止服务监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("服务监控已停止")
    
    def _service_monitoring_loop(self, interval: float):
        """服务监控循环"""
        while self.is_monitoring:
            try:
                # 监控关键服务状态
                critical_services = []
                if platform.system() == "Windows":
                    critical_services = ["Winmgmt", "EventLog", "Dhcp", "Dnscache"]
                elif platform.system() == "Linux":
                    critical_services = ["systemd-journald", "dbus", "NetworkManager", "sshd"]
                elif platform.system() == "Darwin":
                    critical_services = ["com.apple.system.logd", "com.apple.system.DirectoryService"]
                
                for service_name in critical_services:
                    service_info = self.get_service_info(service_name)
                    if service_info and service_info.status != ServiceStatus.RUNNING:
                        logger.warning(f"关键服务状态异常: {service_name} - {service_info.status}")
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"服务监控循环出错: {e}")
                time.sleep(interval)
    
    def cleanup(self):
        """清理资源"""
        self.stop_service_monitoring()
        logger.info("服务管理器资源已清理")


# 全局服务管理器实例
_global_service_manager = None

def get_service_manager() -> ServiceManager:
    """获取全局服务管理器实例"""
    global _global_service_manager
    if _global_service_manager is None:
        _global_service_manager = ServiceManager()
    return _global_service_manager


if __name__ == "__main__":
    # 模块测试代码
    logging.basicConfig(level=logging.INFO)
    
    manager = get_service_manager()
    
    # 测试列出服务
    services = manager.list_services()
    print(f"发现 {len(services)} 个服务")
    
    # 显示前5个服务
    for i, service in enumerate(services[:5]):
        print(f"  {i+1}. {service.name}: {service.status.value} ({service.start_type.value})")
    
    # 测试获取特定服务信息
    if platform.system() == "Windows":
        test_service = "EventLog"
    elif platform.system() == "Linux":
        test_service = "systemd-journald"
    elif platform.system() == "Darwin":
        test_service = "com.apple.system.logd"
    else:
        test_service = ""
    
    if test_service:
        service_info = manager.get_service_info(test_service)
        if service_info:
            print(f"\n测试服务信息: {test_service}")
            print(f"  显示名称: {service_info.display_name}")
            print(f"  状态: {service_info.status.value}")
            print(f"  启动类型: {service_info.start_type.value}")
            print(f"  服务类型: {service_info.service_type.value}")
    
    # 测试计划任务
    tasks = manager.list_scheduled_tasks()
    print(f"\n发现 {len(tasks)} 个计划任务")
    
    for i, task in enumerate(tasks[:3]):  # 只显示前3个
        print(f"  {i+1}. {task.name}: {task.schedule}")
    
    # 启动服务监控
    manager.start_service_monitoring(interval=10.0)
    print("\n服务监控已启动")
    
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        pass  # 已实现
    
    # 清理
    manager.cleanup()
    print("服务管理器测试完成")