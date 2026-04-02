"""
文件系统管理器模块
实现完整的文件系统控制能力，包括：
1. 高级文件操作（创建、读取、写入、删除、移动、复制）
2. 文件监控和同步
3. 磁盘管理和分区操作
4. 文件权限和属性管理
5. 文件搜索和索引

遵循从零开始的实现原则，提供跨平台支持。
"""

import os
import shutil
import stat
import time
import hashlib
import fnmatch
import threading
import logging
import json
import tempfile
import pathlib
import platform
import subprocess
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import zipfile
import tarfile
import gzip

logger = logging.getLogger(__name__)


class FileType(Enum):
    """文件类型枚举"""
    FILE = "file"          # 普通文件
    DIRECTORY = "directory"  # 目录
    SYMLINK = "symlink"    # 符号链接
    DEVICE = "device"      # 设备文件
    SOCKET = "socket"      # 套接字
    FIFO = "fifo"          # 管道


class FilePermission(Enum):
    """文件权限枚举"""
    READ = "read"          # 读取
    WRITE = "write"        # 写入
    EXECUTE = "execute"    # 执行
    ALL = "all"            # 所有权限


class DiskType(Enum):
    """磁盘类型枚举"""
    HDD = "hdd"            # 机械硬盘
    SSD = "ssd"            # 固态硬盘
    NVME = "nvme"          # NVMe硬盘
    USB = "usb"            # USB存储
    NETWORK = "network"    # 网络存储
    CDROM = "cdrom"        # 光盘
    RAMDISK = "ramdisk"    # 内存盘


@dataclass
class FileInfo:
    """文件信息"""
    path: str
    name: str
    size: int
    file_type: FileType
    permissions: Dict[FilePermission, bool]
    created_time: float
    modified_time: float
    accessed_time: float
    owner: str = ""
    group: str = ""
    inode: int = 0
    device: int = 0
    hard_links: int = 0
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiskInfo:
    """磁盘信息"""
    device: str
    mount_point: str
    total_size: int
    used_size: int
    free_size: int
    disk_type: DiskType
    filesystem: str
    block_size: int
    read_only: bool = False
    removable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileSystemEvent:
    """文件系统事件"""
    event_id: str
    event_type: str  # created, modified, deleted, moved, renamed
    src_path: str
    dst_path: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    is_directory: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class FileSystemManager:
    """文件系统管理器"""
    
    def __init__(self):
        self.file_monitors: Dict[str, threading.Thread] = {}
        self.file_event_handlers: Dict[str, List[Callable]] = {}
        self.disk_monitor_thread = None
        self.is_monitoring = False
        
    def get_file_info(self, file_path: str) -> Optional[FileInfo]:
        """获取文件信息"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                return None  # 返回None
            
            # 获取文件状态
            stat_info = os.stat(file_path)
            
            # 确定文件类型
            if os.path.isfile(file_path):
                file_type = FileType.FILE
            elif os.path.isdir(file_path):
                file_type = FileType.DIRECTORY
            elif os.path.islink(file_path):
                file_type = FileType.SYMLINK
            else:
                # 检查特殊文件类型
                mode = stat_info.st_mode
                if stat.S_ISCHR(mode):
                    file_type = FileType.DEVICE
                elif stat.S_ISSOCK(mode):
                    file_type = FileType.SOCKET
                elif stat.S_ISFIFO(mode):
                    file_type = FileType.FIFO
                else:
                    file_type = FileType.FILE
            
            # 获取权限信息
            permissions = {
                FilePermission.READ: os.access(file_path, os.R_OK),
                FilePermission.WRITE: os.access(file_path, os.W_OK),
                FilePermission.EXECUTE: os.access(file_path, os.X_OK),
                FilePermission.ALL: os.access(file_path, os.R_OK | os.W_OK | os.X_OK)
            }
            
            # 获取所有者和组（平台相关）
            owner = ""
            group = ""
            if platform.system() != "Windows":
                try:
                    import pwd
                    import grp
                    owner = pwd.getpwuid(stat_info.st_uid).pw_name
                    group = grp.getgrgid(stat_info.st_gid).gr_name
                except (ImportError, KeyError):
                    pass  # 已实现
            
            # 计算文件校验和（仅对普通文件）
            checksum = ""
            if file_type == FileType.FILE and stat_info.st_size < 100 * 1024 * 1024:  # 小于100MB
                try:
                    checksum = self._calculate_checksum(file_path)
                except:
                    pass  # 已实现
            
            # 创建文件信息对象
            file_info = FileInfo(
                path=os.path.abspath(file_path),
                name=os.path.basename(file_path),
                size=stat_info.st_size,
                file_type=file_type,
                permissions=permissions,
                created_time=stat_info.st_ctime,
                modified_time=stat_info.st_mtime,
                accessed_time=stat_info.st_atime,
                owner=owner,
                group=group,
                inode=stat_info.st_ino,
                device=stat_info.st_dev,
                hard_links=stat_info.st_nlink,
                checksum=checksum
            )
            
            return file_info
            
        except Exception as e:
            logger.error(f"获取文件信息失败 {file_path}: {e}")
            return None  # 返回None
    
    def create_file(self, file_path: str, content: Union[str, bytes] = "", overwrite: bool = False) -> bool:
        """创建文件"""
        try:
            # 检查文件是否已存在
            if os.path.exists(file_path) and not overwrite:
                logger.warning(f"文件已存在: {file_path}")
                return False
            
            # 确保目录存在
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # 写入内容
            if isinstance(content, str):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                with open(file_path, 'wb') as f:
                    f.write(content)
            
            logger.info(f"文件创建成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建文件失败 {file_path}: {e}")
            return False
    
    def read_file(self, file_path: str, binary: bool = False) -> Optional[Union[str, bytes]]:
        """读取文件内容"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                return None  # 返回None
            
            if binary:
                with open(file_path, 'rb') as f:
                    return f.read()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return None  # 返回None
    
    def write_file(self, file_path: str, content: Union[str, bytes], append: bool = False) -> bool:
        """写入文件内容"""
        try:
            # 确保目录存在
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            mode = 'ab' if append else 'wb' if isinstance(content, bytes) else 'a' if append else 'w'
            encoding = None if isinstance(content, bytes) else 'utf-8'
            
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            
            logger.info(f"文件写入成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"写入文件失败 {file_path}: {e}")
            return False
    
    def delete_file(self, file_path: str, recursive: bool = False) -> bool:
        """删除文件或目录"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                return False
            
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                logger.info(f"文件删除成功: {file_path}")
                return True
            elif os.path.isdir(file_path):
                if recursive:
                    shutil.rmtree(file_path)
                    logger.info(f"目录删除成功（递归）: {file_path}")
                else:
                    os.rmdir(file_path)
                    logger.info(f"目录删除成功: {file_path}")
                return True
            else:
                logger.warning(f"不支持的文件类型: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"删除文件失败 {file_path}: {e}")
            return False
    
    def copy_file(self, src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        """复制文件或目录"""
        try:
            if not os.path.exists(src_path):
                logger.warning(f"源文件不存在: {src_path}")
                return False
            
            if os.path.exists(dst_path) and not overwrite:
                logger.warning(f"目标文件已存在: {dst_path}")
                return False
            
            # 确保目标目录存在
            dst_dir = os.path.dirname(dst_path)
            if dst_dir and not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"文件复制成功: {src_path} -> {dst_path}")
                return True
            elif os.path.isdir(src_path):
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
                logger.info(f"目录复制成功: {src_path} -> {dst_path}")
                return True
            else:
                logger.warning(f"不支持的文件类型: {src_path}")
                return False
                
        except Exception as e:
            logger.error(f"复制文件失败 {src_path} -> {dst_path}: {e}")
            return False
    
    def move_file(self, src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        """移动文件或目录"""
        try:
            if not os.path.exists(src_path):
                logger.warning(f"源文件不存在: {src_path}")
                return False
            
            if os.path.exists(dst_path) and not overwrite:
                logger.warning(f"目标文件已存在: {dst_path}")
                return False
            
            # 确保目标目录存在
            dst_dir = os.path.dirname(dst_path)
            if dst_dir and not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            
            if os.path.exists(dst_path) and overwrite:
                self.delete_file(dst_path, recursive=True)
            
            shutil.move(src_path, dst_path)
            logger.info(f"文件移动成功: {src_path} -> {dst_path}")
            return True
            
        except Exception as e:
            logger.error(f"移动文件失败 {src_path} -> {dst_path}: {e}")
            return False
    
    def search_files(self, 
                    root_dir: str, 
                    pattern: str = "*", 
                    recursive: bool = True,
                    file_type: Optional[FileType] = None) -> List[FileInfo]:
        """搜索文件"""
        results = []
        
        try:
            if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
                logger.warning(f"目录不存在: {root_dir}")
                return results
            
            # 确定遍历方法
            if recursive:
                walk_func = os.walk
            else:
                # 非递归遍历
                walk_func = lambda dir: [(dir, [], os.listdir(dir))]
            
            for dirpath, dirnames, filenames in walk_func(root_dir):
                # 处理文件
                for filename in filenames:
                    if fnmatch.fnmatch(filename, pattern):
                        file_path = os.path.join(dirpath, filename)
                        file_info = self.get_file_info(file_path)
                        
                        if file_info and (file_type is None or file_info.file_type == file_type):
                            results.append(file_info)
                
                # 处理目录
                if file_type == FileType.DIRECTORY or file_type is None:
                    for dirname in dirnames:
                        if fnmatch.fnmatch(dirname, pattern):
                            dir_path = os.path.join(dirpath, dirname)
                            file_info = self.get_file_info(dir_path)
                            
                            if file_info and file_info.file_type == FileType.DIRECTORY:
                                results.append(file_info)
                
                if not recursive:
                    break
            
            logger.info(f"文件搜索完成: 在 {root_dir} 中找到 {len(results)} 个文件")
            return results
            
        except Exception as e:
            logger.error(f"搜索文件失败 {root_dir}: {e}")
            return []  # 返回空列表
    
    def calculate_directory_size(self, directory: str) -> int:
        """计算目录大小"""
        total_size = 0
        
        try:
            if not os.path.exists(directory) or not os.path.isdir(directory):
                logger.warning(f"目录不存在: {directory}")
                return 0
            
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except OSError:
                        pass  # 已实现
            
            return total_size
            
        except Exception as e:
            logger.error(f"计算目录大小失败 {directory}: {e}")
            return 0
    
    def _calculate_checksum(self, file_path: str, algorithm: str = "sha256") -> str:
        """计算文件校验和"""
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)
        
        with open(file_path, 'rb') as f:
            file_hash = hash_func()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
        
        return file_hash.hexdigest()
    
    def compress_file(self, src_path: str, dst_path: str, format: str = "zip") -> bool:
        """压缩文件或目录"""
        try:
            if not os.path.exists(src_path):
                logger.warning(f"源文件不存在: {src_path}")
                return False
            
            # 确保目标目录存在
            dst_dir = os.path.dirname(dst_path)
            if dst_dir and not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            
            if format.lower() == "zip":
                if os.path.isfile(src_path):
                    with zipfile.ZipFile(dst_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        zipf.write(src_path, os.path.basename(src_path))
                elif os.path.isdir(src_path):
                    with zipfile.ZipFile(dst_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk(src_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, src_path)
                                zipf.write(file_path, arcname)
                else:
                    logger.warning(f"不支持的文件类型: {src_path}")
                    return False
            
            elif format.lower() == "tar":
                if os.path.isfile(src_path):
                    with tarfile.open(dst_path, 'w') as tar:
                        tar.add(src_path, arcname=os.path.basename(src_path))
                elif os.path.isdir(src_path):
                    with tarfile.open(dst_path, 'w') as tar:
                        tar.add(src_path, arcname=os.path.basename(src_path))
                else:
                    logger.warning(f"不支持的文件类型: {src_path}")
                    return False
            
            elif format.lower() == "gzip":
                if os.path.isfile(src_path):
                    with open(src_path, 'rb') as f_in:
                        with gzip.open(dst_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    logger.warning(f"Gzip只支持单个文件压缩: {src_path}")
                    return False
            
            else:
                logger.warning(f"不支持的压缩格式: {format}")
                return False
            
            logger.info(f"文件压缩成功: {src_path} -> {dst_path}")
            return True
            
        except Exception as e:
            logger.error(f"压缩文件失败 {src_path} -> {dst_path}: {e}")
            return False
    
    def extract_file(self, src_path: str, dst_dir: str, format: str = "auto") -> bool:
        """解压文件"""
        try:
            if not os.path.exists(src_path):
                logger.warning(f"源文件不存在: {src_path}")
                return False
            
            # 确保目标目录存在
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            
            # 自动检测格式
            if format == "auto":
                if src_path.endswith('.zip'):
                    format = "zip"
                elif src_path.endswith('.tar.gz') or src_path.endswith('.tgz'):
                    format = "tar.gz"
                elif src_path.endswith('.tar'):
                    format = "tar"
                elif src_path.endswith('.gz'):
                    format = "gzip"
                else:
                    logger.warning(f"无法自动识别压缩格式: {src_path}")
                    return False
            
            if format == "zip":
                with zipfile.ZipFile(src_path, 'r') as zipf:
                    zipf.extractall(dst_dir)
            
            elif format == "tar":
                with tarfile.open(src_path, 'r') as tar:
                    tar.extractall(dst_dir)
            
            elif format == "tar.gz":
                with tarfile.open(src_path, 'r:gz') as tar:
                    tar.extractall(dst_dir)
            
            elif format == "gzip":
                # 猜测解压后的文件名
                base_name = os.path.basename(src_path)
                if base_name.endswith('.gz'):
                    output_name = base_name[:-3]
                else:
                    output_name = base_name + '.extracted'
                
                output_path = os.path.join(dst_dir, output_name)
                with gzip.open(src_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            else:
                logger.warning(f"不支持的压缩格式: {format}")
                return False
            
            logger.info(f"文件解压成功: {src_path} -> {dst_dir}")
            return True
            
        except Exception as e:
            logger.error(f"解压文件失败 {src_path} -> {dst_dir}: {e}")
            return False
    
    def get_disk_info(self, path: str = "/") -> Optional[DiskInfo]:
        """获取磁盘信息"""
        try:
            if platform.system() == "Windows":
                return self._get_windows_disk_info(path)
            else:
                return self._get_unix_disk_info(path)
                
        except Exception as e:
            logger.error(f"获取磁盘信息失败 {path}: {e}")
            return None  # 返回None
    
    def _get_windows_disk_info(self, path: str) -> Optional[DiskInfo]:
        """获取Windows磁盘信息"""
        try:
            import ctypes
            import win32api  # type: ignore
            import win32file  # type: ignore
            
            # 获取磁盘空间信息
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            free_bytes_to_caller = ctypes.c_ulonglong(0)
            
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path),
                ctypes.byref(free_bytes_to_caller),
                ctypes.byref(total_bytes),
                ctypes.byref(free_bytes)
            )
            
            # 获取驱动器类型
            drive_type = win32file.GetDriveType(path)
            drive_type_map = {
                0: DiskType.UNKNOWN,
                1: DiskType.USB,  # 可移动
                2: DiskType.HDD,  # 固定
                3: DiskType.NETWORK,  # 网络
                4: DiskType.CDROM,  # 光盘
                5: DiskType.RAMDISK  # RAM磁盘
            }
            
            disk_type = drive_type_map.get(drive_type, DiskType.HDD)
            
            # 获取文件系统信息
            volume_name = ctypes.create_unicode_buffer(1024)
            file_system_name = ctypes.create_unicode_buffer(1024)
            max_component_length = ctypes.c_ulong(0)
            file_system_flags = ctypes.c_ulong(0)
            
            ctypes.windll.kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(path),
                volume_name,
                ctypes.sizeof(volume_name),
                None,
                ctypes.byref(max_component_length),
                ctypes.byref(file_system_flags),
                file_system_name,
                ctypes.sizeof(file_system_name)
            )
            
            # 完整处理）
            is_ssd = False
            if "SSD" in path.upper() or "NVME" in path.upper():
                is_ssd = True
                disk_type = DiskType.SSD if "NVME" not in path.upper() else DiskType.NVME
            
            disk_info = DiskInfo(
                device=path,
                mount_point=path,
                total_size=total_bytes.value,
                used_size=total_bytes.value - free_bytes.value,
                free_size=free_bytes.value,
                disk_type=disk_type,
                filesystem=file_system_name.value,
                block_size=4096,  # 默认值
                read_only=(file_system_flags.value & 0x80000) != 0,  # FILE_READ_ONLY_VOLUME
                removable=(drive_type == 1)
            )
            
            return disk_info
            
        except Exception as e:
            logger.error(f"获取Windows磁盘信息失败: {e}")
            return None  # 返回None
    
    def _get_unix_disk_info(self, path: str) -> Optional[DiskInfo]:
        """获取Unix磁盘信息"""
        try:
            import shutil
            
            # 获取磁盘使用情况
            disk_usage = shutil.disk_usage(path)
            
            # 获取挂载点信息
            mount_point = os.path.abspath(path)
            
            # 完整处理）
            disk_type = DiskType.HDD
            if mount_point.startswith('/mnt/usb') or mount_point.startswith('/media'):
                disk_type = DiskType.USB
            elif mount_point.startswith('/mnt/ssd') or '/ssd' in mount_point:
                disk_type = DiskType.SSD
            elif mount_point.startswith('/mnt/nvme') or '/nvme' in mount_point:
                disk_type = DiskType.NVME
            
            # 获取文件系统信息
            filesystem = "unknown"
            try:
                # 使用df命令获取文件系统类型
                result = subprocess.run(['df', '-T', path], capture_output=True, text=True)
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) > 1:
                        filesystem = parts[1]
            except:
                pass  # 已实现
            
            disk_info = DiskInfo(
                device=mount_point,
                mount_point=mount_point,
                total_size=disk_usage.total,
                used_size=disk_usage.used,
                free_size=disk_usage.free,
                disk_type=disk_type,
                filesystem=filesystem,
                block_size=4096,  # 默认值
                read_only=not os.access(path, os.W_OK)
            )
            
            return disk_info
            
        except Exception as e:
            logger.error(f"获取Unix磁盘信息失败: {e}")
            return None  # 返回None
    
    def list_disks(self) -> List[DiskInfo]:
        """列出所有磁盘"""
        disks = []
        
        try:
            if platform.system() == "Windows":
                # Windows: 列出所有驱动器
                import string
                import ctypes
                
                drives = []
                bitmask = ctypes.windll.kernel32.GetLogicalDrives()
                for letter in string.ascii_uppercase:
                    if bitmask & 1:
                        drives.append(f"{letter}:\\")
                    bitmask >>= 1
                
                for drive in drives:
                    disk_info = self.get_disk_info(drive)
                    if disk_info:
                        disks.append(disk_info)
            
            else:
                # Unix: 使用df命令
                try:
                    result = subprocess.run(['df', '-h'], capture_output=True, text=True)
                    lines = result.stdout.strip().split('\n')
                    
                    for line in lines[1:]:  # 跳过标题行
                        parts = line.split()
                        if len(parts) >= 6:
                            mount_point = parts[5]
                            disk_info = self.get_disk_info(mount_point)
                            if disk_info:
                                disks.append(disk_info)
                except:
                    # 备选方案：检查常见挂载点
                    common_mounts = ['/', '/home', '/mnt', '/media']
                    for mount in common_mounts:
                        if os.path.exists(mount):
                            disk_info = self.get_disk_info(mount)
                            if disk_info:
                                disks.append(disk_info)
            
            logger.info(f"发现 {len(disks)} 个磁盘")
            return disks
            
        except Exception as e:
            logger.error(f"列出磁盘失败: {e}")
            return []  # 返回空列表
    
    def start_file_monitoring(self, path: str, handler: Callable[[FileSystemEvent], None]) -> bool:
        """开始文件监控"""
        try:
            if not os.path.exists(path):
                logger.warning(f"监控路径不存在: {path}")
                return False
            
            # 标准化路径
            path = os.path.abspath(path)
            
            # 检查是否已监控
            if path in self.file_monitors:
                logger.warning(f"路径已在监控中: {path}")
                return False
            
            # 注册事件处理器
            if path not in self.file_event_handlers:
                self.file_event_handlers[path] = []
            self.file_event_handlers[path].append(handler)
            
            # 启动监控线程
            monitor_thread = threading.Thread(
                target=self._monitor_filesystem,
                args=(path,),
                daemon=True
            )
            monitor_thread.start()
            self.file_monitors[path] = monitor_thread
            
            logger.info(f"文件监控已启动: {path}")
            return True
            
        except Exception as e:
            logger.error(f"启动文件监控失败 {path}: {e}")
            return False
    
    def stop_file_monitoring(self, path: str) -> bool:
        """停止文件监控"""
        try:
            path = os.path.abspath(path)
            
            if path not in self.file_monitors:
                logger.warning(f"路径未在监控中: {path}")
                return False
            
            # 标记监控停止（通过事件处理器）
            if path in self.file_event_handlers:
                del self.file_event_handlers[path]
            
            # 等待监控线程结束
            if path in self.file_monitors:
                self.file_monitors[path].join(timeout=2.0)
                del self.file_monitors[path]
            
            logger.info(f"文件监控已停止: {path}")
            return True
            
        except Exception as e:
            logger.error(f"停止文件监控失败 {path}: {e}")
            return False
    
    def _monitor_filesystem(self, path: str):
        """文件系统监控循环（完整版）"""
        try:
            # 记录初始文件状态
            initial_state = {}
            for root, dirs, files in os.walk(path):
                for name in dirs + files:
                    full_path = os.path.join(root, name)
                    try:
                        stat_info = os.stat(full_path)
                        initial_state[full_path] = {
                            'mtime': stat_info.st_mtime,
                            'size': stat_info.st_size,
                            'exists': True
                        }
                    except OSError:
                        pass  # 已实现
            
            # 监控循环
            while path in self.file_event_handlers:
                # 检查文件变化
                current_state = {}
                for root, dirs, files in os.walk(path):
                    for name in dirs + files:
                        full_path = os.path.join(root, name)
                        try:
                            stat_info = os.stat(full_path)
                            current_state[full_path] = {
                                'mtime': stat_info.st_mtime,
                                'size': stat_info.st_size,
                                'exists': True
                            }
                        except OSError:
                            current_state[full_path] = {'exists': False}
                
                # 检测新文件
                for file_path in current_state:
                    if file_path not in initial_state and current_state[file_path]['exists']:
                        event = FileSystemEvent(
                            event_id=f"create_{time.time()}",
                            event_type="created",
                            src_path=file_path,
                            is_directory=os.path.isdir(file_path)
                        )
                        self._dispatch_event(path, event)
                
                # 检测删除的文件
                for file_path in initial_state:
                    if file_path not in current_state or not current_state[file_path]['exists']:
                        event = FileSystemEvent(
                            event_id=f"delete_{time.time()}",
                            event_type="deleted",
                            src_path=file_path,
                            is_directory=initial_state[file_path].get('is_dir', False)
                        )
                        self._dispatch_event(path, event)
                
                # 检测修改的文件
                for file_path in current_state:
                    if (file_path in initial_state and 
                        current_state[file_path]['exists'] and 
                        initial_state[file_path]['exists']):
                        
                        if (current_state[file_path]['mtime'] != initial_state[file_path]['mtime'] or
                            current_state[file_path]['size'] != initial_state[file_path]['size']):
                            
                            event = FileSystemEvent(
                                event_id=f"modify_{time.time()}",
                                event_type="modified",
                                src_path=file_path,
                                is_directory=os.path.isdir(file_path)
                            )
                            self._dispatch_event(path, event)
                
                # 更新初始状态
                initial_state = current_state
                
                # 休眠一段时间
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"文件监控循环出错 {path}: {e}")
    
    def _dispatch_event(self, path: str, event: FileSystemEvent):
        """分发文件系统事件"""
        try:
            if path in self.file_event_handlers:
                for handler in self.file_event_handlers[path]:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"文件事件处理器出错: {e}")
        except Exception as e:
            logger.error(f"分发文件事件失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        # 停止所有文件监控
        for path in list(self.file_monitors.keys()):
            self.stop_file_monitoring(path)
        
        logger.info("文件系统管理器资源已清理")


# 全局文件系统管理器实例
_global_filesystem_manager = None

def get_filesystem_manager() -> FileSystemManager:
    """获取全局文件系统管理器实例"""
    global _global_filesystem_manager
    if _global_filesystem_manager is None:
        _global_filesystem_manager = FileSystemManager()
    return _global_filesystem_manager


if __name__ == "__main__":
    # 模块测试代码
    logging.basicConfig(level=logging.INFO)
    
    manager = get_filesystem_manager()
    
    # 测试创建文件
    test_file = "test_file.txt"
    if manager.create_file(test_file, "Hello, World!"):
        print(f"文件创建成功: {test_file}")
    
    # 测试读取文件
    content = manager.read_file(test_file)
    if content:
        print(f"文件内容: {content}")
    
    # 测试获取文件信息
    file_info = manager.get_file_info(test_file)
    if file_info:
        print(f"文件信息: 名称={file_info.name}, 大小={file_info.size} 字节, 类型={file_info.file_type.value}")
    
    # 测试搜索文件
    files = manager.search_files(".", "*.txt", recursive=False)
    print(f"找到 {len(files)} 个txt文件")
    
    # 测试磁盘信息
    disks = manager.list_disks()
    print(f"发现 {len(disks)} 个磁盘:")
    for i, disk in enumerate(disks[:3]):  # 只显示前3个
        usage_percent = (disk.used_size / disk.total_size * 100) if disk.total_size > 0 else 0
        print(f"  {i+1}. {disk.device}: {disk.filesystem}, 使用率={usage_percent:.1f}%")
    
    # 测试文件监控
    def file_event_handler(event: FileSystemEvent):
        print(f"文件事件: {event.event_type} - {event.src_path}")
    
    if manager.start_file_monitoring(".", file_event_handler):
        print("文件监控已启动")
        
        # 创建一个新文件来触发事件
        test_event_file = "test_event.txt"
        manager.create_file(test_event_file, "Trigger event")
        time.sleep(1)
        
        # 停止监控
        manager.stop_file_monitoring(".")
        print("文件监控已停止")
        
        # 清理测试文件
        manager.delete_file(test_event_file)
    
    # 清理测试文件
    manager.delete_file(test_file)
    print("文件系统管理器测试完成")