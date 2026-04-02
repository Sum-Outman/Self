# 网络控制器 / Network Controller

## 概述 / Overview

### 中文
网络控制器模块提供完整的跨平台网络控制能力，支持Windows、Linux和macOS系统。实现了TCP/UDP服务器和客户端、HTTP/WebSocket服务器、网络配置管理、网络监控诊断、网络安全控制等功能。

### English
The Network Controller module provides complete cross-platform network control capabilities, supporting Windows, Linux, and macOS systems. Implements TCP/UDP servers and clients, HTTP/WebSocket servers, network configuration management, network monitoring and diagnostics, and network security control.

---

## 核心功能 / Core Features

### 中文
1. **TCP服务器**：多客户端TCP服务器，支持广播和单播
2. **UDP服务器**：UDP通信服务器，支持无连接数据传输
3. **HTTP服务器**：简化HTTP服务器，支持路由配置
4. **网络配置管理**：IP地址、子网掩码、网关、DNS配置
5. **网络监控**：实时网络流量、延迟、吞吐量监控
6. **跨平台支持**：Windows、Linux、macOS完整支持

### English
1. **TCP Server**: Multi-client TCP server with broadcast and unicast support
2. **UDP Server**: UDP communication server for connectionless data transfer
3. **HTTP Server**: Simplified HTTP server with route configuration
4. **Network Configuration Management**: IP address, subnet mask, gateway, DNS configuration
5. **Network Monitoring**: Real-time network traffic, latency, throughput monitoring
6. **Cross-Platform Support**: Complete Windows, Linux, macOS support

---

## 核心类 / Core Classes

### NetworkProtocol
```python
class NetworkProtocol(Enum):
    TCP = "tcp"                    # TCP协议 / TCP Protocol
    UDP = "udp"                    # UDP协议 / UDP Protocol
    HTTP = "http"                  # HTTP协议 / HTTP Protocol
    HTTPS = "https"                # HTTPS协议 / HTTPS Protocol
    WEBSOCKET = "websocket"        # WebSocket协议 / WebSocket Protocol
    WEBSOCKET_SECURE = "websocket_secure"  # 安全WebSocket / Secure WebSocket
```

### NetworkStatus
```python
class NetworkStatus(Enum):
    DISCONNECTED = "disconnected"  # 断开连接 / Disconnected
    CONNECTING = "connecting"      # 连接中 / Connecting
    CONNECTED = "connected"        # 已连接 / Connected
    LISTENING = "listening"        # 监听中 / Listening
    ERROR = "error"                # 错误 / Error
    CLOSING = "closing"            # 关闭中 / Closing
```

### NetworkInterface
```python
@dataclass
class NetworkInterface:
    name: str                      # 接口名称 / Interface name
    ip_address: str                # IP地址 / IP address
    netmask: str                   # 子网掩码 / Subnet mask
    gateway: str                   # 网关 / Gateway
    mac_address: str               # MAC地址 / MAC address
    status: str                    # 状态 / Status
    mtu: int                       # MTU值 / MTU value
    speed_mbps: int                # 速度(Mbps) / Speed (Mbps)
    is_up: bool                    # 是否启用 / Is enabled
    is_wireless: bool              # 是否无线 / Is wireless
```

### ConnectionInfo
```python
@dataclass
class ConnectionInfo:
    connection_id: str              # 连接ID / Connection ID
    protocol: NetworkProtocol       # 协议 / Protocol
    local_address: Tuple[str, int]  # 本地地址 / Local address
    remote_address: Tuple[str, int] # 远程地址 / Remote address
    status: NetworkStatus           # 状态 / Status
    bytes_sent: int = 0             # 发送字节数 / Bytes sent
    bytes_received: int = 0         # 接收字节数 / Bytes received
    created_at: float = ...         # 创建时间 / Creation time
    last_activity: float = ...      # 最后活动时间 / Last activity time
```

### NetworkMetric
```python
@dataclass
class NetworkMetric:
    metric_id: str                  # 指标ID / Metric ID
    name: str                       # 名称 / Name
    value: float                    # 值 / Value
    unit: str = ""                  # 单位 / Unit
    timestamp: float = ...          # 时间戳 / Timestamp
    interface: str = ""             # 接口 / Interface
    protocol: str = ""              # 协议 / Protocol
```

---

## 主要方法 / Main Methods

### TCP服务器 / TCP Server

#### 创建TCP服务器 / Create TCP Server
```python
def create_tcp_server(
    server_id: str,
    host: str = "0.0.0.0",
    port: int = 8080
) -> bool
```
创建并启动TCP服务器。

Create and start a TCP server.

#### 发送数据到客户端 / Send Data to Client
```python
def send_to_client(client_id: str, data: bytes) -> bool
```
发送数据到指定客户端。

Send data to a specific client.

#### 广播数据 / Broadcast Data
```python
def broadcast(data: bytes) -> int
```
广播数据到所有客户端，返回成功发送的客户端数量。

Broadcast data to all clients, returns the number of successful clients.

### UDP服务器 / UDP Server

#### 创建UDP服务器 / Create UDP Server
```python
def create_udp_server(
    server_id: str,
    host: str = "0.0.0.0",
    port: int = 9090
) -> bool
```
创建并启动UDP服务器。

Create and start a UDP server.

#### 发送UDP数据 / Send UDP Data
```python
def send_to(address: Tuple[str, int], data: bytes) -> bool
```
发送UDP数据到指定地址。

Send UDP data to a specific address.

### HTTP服务器 / HTTP Server

#### 创建HTTP服务器 / Create HTTP Server
```python
def create_http_server(
    server_id: str,
    host: str = "0.0.0.0",
    port: int = 80
) -> bool
```
创建并启动HTTP服务器。

Create and start an HTTP server.

#### 添加路由 / Add Route
```python
def add_route(path: str, handler: Callable)
```
为HTTP服务器添加路由处理器。

Add a route handler for the HTTP server.

### 网络配置 / Network Configuration

#### 扫描网络接口 / Scan Network Interfaces
```python
def get_network_interfaces() -> List[NetworkInterface]
```
扫描系统中的所有网络接口。

Scan all network interfaces in the system.

#### 设置静态IP / Set Static IP
```python
def set_static_ip(
    interface_name: str,
    ip_address: str,
    netmask: str,
    gateway: str
) -> bool
```
设置网络接口的静态IP地址。

Set static IP address for a network interface.

#### 设置DHCP / Set DHCP
```python
def set_dhcp(interface_name: str) -> bool
```
设置网络接口使用DHCP自动获取IP。

Set network interface to use DHCP for automatic IP.

### 网络监控 / Network Monitoring

#### 启动监控 / Start Monitoring
```python
def start_monitoring(interval: float = 5.0)
```
启动网络监控，指定采样间隔。

Start network monitoring with specified sampling interval.

#### 获取网络指标 / Get Network Metrics
```python
def get_network_metrics() -> List[NetworkMetric]
```
获取当前网络指标数据。

Get current network metric data.

---

## 使用示例 / Usage Examples

### 中文
```python
from models.system_control.network_controller import get_network_controller

controller = get_network_controller()

# 创建TCP服务器 / Create TCP server
if controller.create_tcp_server("my_tcp", "0.0.0.0", 8888):
    print("TCP服务器启动成功")

# 创建UDP服务器 / Create UDP server
if controller.create_udp_server("my_udp", "0.0.0.0", 9999):
    print("UDP服务器启动成功")

# 扫描网络接口 / Scan network interfaces
interfaces = controller.get_network_interfaces()
for iface in interfaces:
    print(f"接口: {iface.name}, IP: {iface.ip_address}")

# 启动网络监控 / Start network monitoring
controller.start_monitoring(interval=2.0)

# 获取网络指标 / Get network metrics
time.sleep(5)
metrics = controller.get_network_metrics()
for metric in metrics:
    print(f"{metric.name}: {metric.value} {metric.unit}")

# 关闭 / Shutdown
controller.shutdown()
```

### English
```python
from models.system_control.network_controller import get_network_controller

controller = get_network_controller()

# Create TCP server
if controller.create_tcp_server("my_tcp", "0.0.0.0", 8888):
    print("TCP server started successfully")

# Create UDP server
if controller.create_udp_server("my_udp", "0.0.0.0", 9999):
    print("UDP server started successfully")

# Scan network interfaces
interfaces = controller.get_network_interfaces()
for iface in interfaces:
    print(f"Interface: {iface.name}, IP: {iface.ip_address}")

# Start network monitoring
controller.start_monitoring(interval=2.0)

# Get network metrics
time.sleep(5)
metrics = controller.get_network_metrics()
for metric in metrics:
    print(f"{metric.name}: {metric.value} {metric.unit}")

# Shutdown
controller.shutdown()
```

---

## 相关模块 / Related Modules

- [文件系统管理器](./filesystem-manager.md) - 文件系统管理
- [进程管理器](./process-manager.md) - 进程管理
- [服务管理器](./service-manager.md) - 服务管理
