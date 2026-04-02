# 文件系统管理器 / File System Manager

## 概述 / Overview

### 中文
文件系统管理器模块提供完整的跨平台文件系统控制能力，支持Windows、Linux和macOS系统。实现了文件操作、磁盘管理、文件监控、压缩解压等功能。

### English
The File System Manager module provides complete cross-platform file system control capabilities, supporting Windows, Linux, and macOS systems. Implements file operations, disk management, file monitoring, compression/decompression, and other functions.

---

## 核心功能 / Core Features

### 中文
1. **高级文件操作**：创建、读取、写入、删除、移动、复制
2. **文件监控**：实时监控文件变化和事件
3. **磁盘管理**：磁盘信息查询、分区操作
4. **文件权限和属性**：权限管理、属性设置
5. **文件搜索和索引**：快速文件搜索
6. **压缩解压**：支持zip、tar、gzip格式

### English
1. **Advanced File Operations**: Create, read, write, delete, move, copy
2. **File Monitoring**: Real-time file change and event monitoring
3. **Disk Management**: Disk information query, partition operations
4. **File Permissions and Attributes**: Permission management, attribute settings
5. **File Search and Indexing**: Fast file search
6. **Compression/Decompression**: Supports zip, tar, gzip formats

---

## 核心类 / Core Classes

### FileType
```python
class FileType(Enum):
    FILE = "file"          # 普通文件 / Regular file
    DIRECTORY = "directory"  # 目录 / Directory
    SYMLINK = "symlink"    # 符号链接 / Symbolic link
    DEVICE = "device"      # 设备文件 / Device file
    SOCKET = "socket"      # 套接字 / Socket
    FIFO = "fifo"          # 管道 / FIFO
```

### FilePermission
```python
class FilePermission(Enum):
    READ = "read"          # 读取 / Read
    WRITE = "write"        # 写入 / Write
    EXECUTE = "execute"    # 执行 / Execute
    ALL = "all"            # 所有权限 / All permissions
```

### DiskType
```python
class DiskType(Enum):
    HDD = "hdd"            # 机械硬盘 / Hard disk drive
    SSD = "ssd"            # 固态硬盘 / Solid state drive
    NVME = "nvme"          # NVMe硬盘 / NVMe drive
    USB = "usb"            # USB存储 / USB storage
    NETWORK = "network"    # 网络存储 / Network storage
    CDROM = "cdrom"        # 光盘 / CD-ROM
    RAMDISK = "ramdisk"    # 内存盘 / RAM disk
```

### FileInfo
```python
@dataclass
class FileInfo:
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
```

### DiskInfo
```python
@dataclass
class DiskInfo:
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
```

### FileSystemEvent
```python
@dataclass
class FileSystemEvent:
    event_id: str
    event_type: str  # created, modified, deleted, moved, renamed
    src_path: str
    dst_path: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    is_directory: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 主要方法 / Main Methods

### 文件操作 / File Operations

#### 获取文件信息 / Get File Info
```python
def get_file_info(file_path: str) -> Optional[FileInfo]
```
获取文件的详细信息，包括大小、类型、权限、时间戳等。

Get detailed file information including size, type, permissions, timestamps, etc.

#### 创建文件 / Create File
```python
def create_file(
    file_path: str,
    content: Union[str, bytes] = "",
    overwrite: bool = False
) -> bool
```
创建新文件，可选择覆盖现有文件。

Create a new file, optionally overwriting existing files.

#### 读取文件 / Read File
```python
def read_file(file_path: str, binary: bool = False) -> Optional[Union[str, bytes]]
```
读取文件内容，支持文本和二进制模式。

Read file content, supporting text and binary modes.

#### 写入文件 / Write File
```python
def write_file(
    file_path: str,
    content: Union[str, bytes],
    append: bool = False
) -> bool
```
写入文件内容，支持追加模式。

Write file content, supporting append mode.

#### 删除文件 / Delete File
```python
def delete_file(file_path: str, recursive: bool = False) -> bool
```
删除文件或目录，支持递归删除。

Delete files or directories, supporting recursive deletion.

#### 复制文件 / Copy File
```python
def copy_file(
    src_path: str,
    dst_path: str,
    overwrite: bool = False
) -> bool
```
复制文件或目录。

Copy files or directories.

#### 移动文件 / Move File
```python
def move_file(
    src_path: str,
    dst_path: str,
    overwrite: bool = False
) -> bool
```
移动文件或目录。

Move files or directories.

---

### 文件搜索 / File Search

#### 搜索文件 / Search Files
```python
def search_files(
    root_dir: str,
    pattern: str = "*",
    recursive: bool = True,
    file_type: Optional[FileType] = None
) -> List[FileInfo]
```
在指定目录中搜索文件，支持通配符模式。

Search for files in the specified directory, supporting wildcard patterns.

#### 计算目录大小 / Calculate Directory Size
```python
def calculate_directory_size(directory: str) -> int
```
计算目录的总大小。

Calculate the total size of a directory.

---

### 磁盘管理 / Disk Management

#### 获取磁盘信息 / Get Disk Info
```python
def get_disk_info(path: str = "/") -> Optional[DiskInfo]
```
获取指定路径的磁盘信息。

Get disk information for the specified path.

#### 列出所有磁盘 / List All Disks
```python
def list_disks() -> List[DiskInfo]
```
列出系统中所有可用的磁盘。

List all available disks in the system.

---

### 文件监控 / File Monitoring

#### 开始文件监控 / Start File Monitoring
```python
def start_file_monitoring(
    path: str,
    handler: Callable[[FileSystemEvent], None]
) -> bool
```
开始监控指定路径的文件变化。

Start monitoring file changes in the specified path.

#### 停止文件监控 / Stop File Monitoring
```python
def stop_file_monitoring(path: str) -> bool
```
停止文件监控。

Stop file monitoring.

---

### 压缩解压 / Compression/Decompression

#### 压缩文件 / Compress File
```python
def compress_file(
    src_path: str,
    dst_path: str,
    format: str = "zip"
) -> bool
```
压缩文件或目录，支持zip、tar、gzip格式。

Compress files or directories, supporting zip, tar, gzip formats.

#### 解压文件 / Extract File
```python
def extract_file(
    src_path: str,
    dst_dir: str,
    format: str = "auto"
) -> bool
```
解压文件，支持自动检测格式。

Extract files, supporting automatic format detection.

---

## 使用示例 / Usage Examples

### 基础文件操作 / Basic File Operations

#### 中文
```python
from models.system_control.filesystem_manager import get_filesystem_manager

manager = get_filesystem_manager()

# 创建文件 / Create file
manager.create_file("test.txt", "Hello, World!")

# 读取文件 / Read file
content = manager.read_file("test.txt")
print(f"Content: {content}")

# 获取文件信息 / Get file info
file_info = manager.get_file_info("test.txt")
print(f"Size: {file_info.size} bytes")

# 搜索文件 / Search files
files = manager.search_files(".", "*.txt")
print(f"Found {len(files)} txt files")

# 复制文件 / Copy file
manager.copy_file("test.txt", "test_copy.txt")

# 删除文件 / Delete file
manager.delete_file("test.txt")
```

#### English
```python
from models.system_control.filesystem_manager import get_filesystem_manager

manager = get_filesystem_manager()

# Create file
manager.create_file("test.txt", "Hello, World!")

# Read file
content = manager.read_file("test.txt")
print(f"Content: {content}")

# Get file info
file_info = manager.get_file_info("test.txt")
print(f"Size: {file_info.size} bytes")

# Search files
files = manager.search_files(".", "*.txt")
print(f"Found {len(files)} txt files")

# Copy file
manager.copy_file("test.txt", "test_copy.txt")

# Delete file
manager.delete_file("test.txt")
```

### 文件监控 / File Monitoring

#### 中文
```python
from models.system_control.filesystem_manager import get_filesystem_manager, FileSystemEvent

manager = get_filesystem_manager()

def event_handler(event: FileSystemEvent):
    print(f"File event: {event.event_type} - {event.src_path}")

# 开始监控 / Start monitoring
manager.start_file_monitoring(".", event_handler)
print("Monitoring started...")

try:
    time.sleep(30)  # 监控30秒 / Monitor for 30 seconds
except KeyboardInterrupt:
    pass

# 停止监控 / Stop monitoring
manager.stop_file_monitoring(".")
print("Monitoring stopped")
```

#### English
```python
from models.system_control.filesystem_manager import get_filesystem_manager, FileSystemEvent

manager = get_filesystem_manager()

def event_handler(event: FileSystemEvent):
    print(f"File event: {event.event_type} - {event.src_path}")

# Start monitoring
manager.start_file_monitoring(".", event_handler)
print("Monitoring started...")

try:
    time.sleep(30)  # Monitor for 30 seconds
except KeyboardInterrupt:
    pass

# Stop monitoring
manager.stop_file_monitoring(".")
print("Monitoring stopped")
```

### 磁盘管理 / Disk Management

#### 中文
```python
from models.system_control.filesystem_manager import get_filesystem_manager

manager = get_filesystem_manager()

# 列出所有磁盘 / List all disks
disks = manager.list_disks()
print(f"Found {len(disks)} disks:")
for disk in disks:
    usage_percent = (disk.used_size / disk.total_size * 100) if disk.total_size > 0 else 0
    print(f"  {disk.device}: {usage_percent:.1f}% used")

# 获取特定磁盘信息 / Get specific disk info
if platform.system() == "Windows":
    disk_info = manager.get_disk_info("C:\\")
else:
    disk_info = manager.get_disk_info("/")

if disk_info:
    print(f"\nDisk info:")
    print(f"  Total: {disk_info.total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Used: {disk_info.used_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Free: {disk_info.free_size / 1024 / 1024 / 1024:.2f} GB")
```

#### English
```python
from models.system_control.filesystem_manager import get_filesystem_manager

manager = get_filesystem_manager()

# List all disks
disks = manager.list_disks()
print(f"Found {len(disks)} disks:")
for disk in disks:
    usage_percent = (disk.used_size / disk.total_size * 100) if disk.total_size > 0 else 0
    print(f"  {disk.device}: {usage_percent:.1f}% used")

# Get specific disk info
if platform.system() == "Windows":
    disk_info = manager.get_disk_info("C:\\")
else:
    disk_info = manager.get_disk_info("/")

if disk_info:
    print(f"\nDisk info:")
    print(f"  Total: {disk_info.total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Used: {disk_info.used_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Free: {disk_info.free_size / 1024 / 1024 / 1024:.2f} GB")
```

---

## 跨平台支持 / Cross-Platform Support

### 中文

| 功能 | Windows | Linux | macOS |
|------|---------|-------|-------|
| 文件操作 | ✓ | ✓ | ✓ |
| 磁盘管理 | ✓ | ✓ | ✓ |
| 文件监控 | ✓ | ✓ | ✓ |
| 权限管理 | ✓ | ✓ | ✓ |
| 压缩解压 | ✓ | ✓ | ✓ |

### English

| Feature | Windows | Linux | macOS |
|---------|---------|-------|-------|
| File Operations | ✓ | ✓ | ✓ |
| Disk Management | ✓ | ✓ | ✓ |
| File Monitoring | ✓ | ✓ | ✓ |
| Permission Management | ✓ | ✓ | ✓ |
| Compression/Decompression | ✓ | ✓ | ✓ |

---

## 最佳实践 / Best Practices

### 中文
1. **错误处理**：始终检查返回值，处理可能的异常
2. **路径验证**：在操作前验证文件路径
3. **权限检查**：确保有足够的权限执行操作
4. **资源释放**：及时关闭文件句柄
5. **监控性能**：文件监控可能影响性能，合理设置监控范围

### English
1. **Error Handling**: Always check return values and handle possible exceptions
2. **Path Validation**: Validate file paths before operations
3. **Permission Check**: Ensure sufficient permissions for operations
4. **Resource Release**: Close file handles in time
5. **Monitor Performance**: File monitoring may affect performance, set reasonable monitoring scope

---

## 相关模块 / Related Modules

- [进程管理器](./process-manager.md) - 进程管理
- [网络控制器](./network-controller.md) - 网络控制
- [服务管理器](./service-manager.md) - 服务管理
