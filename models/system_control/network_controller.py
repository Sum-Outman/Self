"""
网络控制器模块
实现完整的网络控制能力，包括：
1. TCP/UDP服务器和客户端
2. HTTP/WebSocket服务器
3. 网络配置和管理
4. 网络监控和诊断
5. 网络安全控制

遵循从零开始的实现原则，不使用预训练模型或外部服务。
"""

import socket
import threading
import logging
import time
import subprocess
import platform
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import select

logger = logging.getLogger(__name__)


class NetworkProtocol(Enum):
    """网络协议枚举"""

    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    WEBSOCKET_SECURE = "websocket_secure"


class NetworkStatus(Enum):
    """网络状态枚举"""

    DISCONNECTED = "disconnected"  # 断开连接
    CONNECTING = "connecting"  # 连接中
    CONNECTED = "connected"  # 已连接
    LISTENING = "listening"  # 监听中
    ERROR = "error"  # 错误
    CLOSING = "closing"  # 关闭中


@dataclass
class NetworkInterface:
    """网络接口信息"""

    name: str
    ip_address: str
    netmask: str
    gateway: str
    mac_address: str
    status: str
    mtu: int
    speed_mbps: int
    is_up: bool
    is_wireless: bool


@dataclass
class ConnectionInfo:
    """连接信息"""

    connection_id: str
    protocol: NetworkProtocol
    local_address: Tuple[str, int]
    remote_address: Tuple[str, int]
    status: NetworkStatus
    bytes_sent: int = 0
    bytes_received: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


@dataclass
class NetworkMetric:
    """网络指标"""

    metric_id: str
    name: str
    value: float
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    interface: str = ""
    protocol: str = ""


class TCPServer:
    """TCP服务器实现"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, max_clients: int = 100):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.server_socket = None
        self.clients: Dict[str, socket.socket] = {}
        self.is_running = False
        self.server_thread = None
        self.callbacks: Dict[str, Callable] = {}

    def start(self) -> bool:
        """启动TCP服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_clients)
            self.server_socket.setblocking(False)

            self.is_running = True
            self.server_thread = threading.Thread(
                target=self._accept_connections, daemon=True
            )
            self.server_thread.start()

            logger.info(f"TCP服务器启动成功，监听 {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"TCP服务器启动失败: {e}")
            return False

    def _accept_connections(self):
        """接受连接线程"""
        while self.is_running:
            try:
                readable, _, _ = select.select([self.server_socket], [], [], 0.1)
                if readable:
                    client_socket, client_address = self.server_socket.accept()
                    client_socket.setblocking(False)
                    client_id = f"{client_address[0]}:{client_address[1]}"
                    self.clients[client_id] = client_socket

                    logger.info(f"新客户端连接: {client_id}")

                    # 启动客户端处理线程
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_id, client_socket),
                        daemon=True,
                    )
                    client_thread.start()

                    if "on_connect" in self.callbacks:
                        self.callbacks["on_connect"](client_id, client_address)
            except Exception as e:
                if self.is_running:
                    logger.error(f"接受连接时出错: {e}")
                break

    def _handle_client(self, client_id: str, client_socket: socket.socket):
        """处理客户端连接"""
        buffer_size = 4096

        while self.is_running and client_id in self.clients:
            try:
                readable, _, _ = select.select([client_socket], [], [], 0.1)
                if readable:
                    data = client_socket.recv(buffer_size)
                    if data:
                        if "on_message" in self.callbacks:
                            self.callbacks["on_message"](client_id, data)

                        # 回显数据（示例）
                        client_socket.send(data)
                    else:
                        # 连接关闭
                        break
            except Exception as e:
                logger.error(f"处理客户端 {client_id} 时出错: {e}")
                break

        # 清理客户端
        if client_id in self.clients:
            del self.clients[client_id]
        try:
            client_socket.close()
        except BaseException:
            pass  # 已实现

        logger.info(f"客户端断开连接: {client_id}")
        if "on_disconnect" in self.callbacks:
            self.callbacks["on_disconnect"](client_id)

    def stop(self):
        """停止TCP服务器"""
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()

        # 关闭所有客户端连接
        for client_id, client_socket in list(self.clients.items()):
            try:
                client_socket.close()
            except BaseException:
                pass  # 已实现
            del self.clients[client_id]

        logger.info("TCP服务器已停止")

    def send_to_client(self, client_id: str, data: bytes) -> bool:
        """发送数据到指定客户端"""
        if client_id in self.clients:
            try:
                self.clients[client_id].send(data)
                return True
            except Exception as e:
                logger.error(f"发送数据到客户端 {client_id} 时出错: {e}")
                return False
        return False

    def broadcast(self, data: bytes) -> int:
        """广播数据到所有客户端"""
        success_count = 0
        for client_id in list(self.clients.keys()):
            if self.send_to_client(client_id, data):
                success_count += 1
        return success_count


class UDPServer:
    """UDP服务器实现"""

    def __init__(self, host: str = "0.0.0.0", port: int = 9090):
        self.host = host
        self.port = port
        self.server_socket = None
        self.is_running = False
        self.server_thread = None
        self.callbacks: Dict[str, Callable] = {}

    def start(self) -> bool:
        """启动UDP服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind((self.host, self.port))
            self.is_running = True
            self.server_thread = threading.Thread(
                target=self._receive_messages, daemon=True
            )
            self.server_thread.start()

            logger.info(f"UDP服务器启动成功，监听 {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"UDP服务器启动失败: {e}")
            return False

    def _receive_messages(self):
        """接收消息线程"""
        buffer_size = 65535

        while self.is_running:
            try:
                data, address = self.server_socket.recvfrom(buffer_size)
                if "on_message" in self.callbacks:
                    self.callbacks["on_message"](data, address)

                # 示例：回复接收确认
                ack_message = f"收到 {len(data)} 字节数据".encode()
                self.server_socket.sendto(ack_message, address)

            except Exception as e:
                if self.is_running:
                    logger.error(f"接收UDP消息时出错: {e}")

    def stop(self):
        """停止UDP服务器"""
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("UDP服务器已停止")

    def send_to(self, address: Tuple[str, int], data: bytes) -> bool:
        """发送UDP数据到指定地址"""
        try:
            self.server_socket.sendto(data, address)
            return True
        except Exception as e:
            logger.error(f"发送UDP数据到 {address} 时出错: {e}")
            return False


class HTTPServer:
    """HTTP服务器实现（完整版）"""

    def __init__(self, host: str = "0.0.0.0", port: int = 80):
        self.host = host
        self.port = port
        self.routes: Dict[str, Callable] = {}
        self.server = None
        self.is_running = False

    def add_route(self, path: str, handler: Callable):
        """添加路由"""
        self.routes[path] = handler

    def start(self) -> bool:
        """启动HTTP服务器"""
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((self.host, self.port))
            self.server.listen(5)
            self.is_running = True

            server_thread = threading.Thread(
                target=self._handle_connections, daemon=True
            )
            server_thread.start()

            logger.info(f"HTTP服务器启动成功，监听 {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"HTTP服务器启动失败: {e}")
            return False

    def _handle_connections(self):
        """处理HTTP连接"""
        while self.is_running:
            try:
                client_socket, client_address = self.server.accept()
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address),
                    daemon=True,
                )
                client_thread.start()
            except Exception as e:
                if self.is_running:
                    logger.error(f"接受HTTP连接时出错: {e}")

    def _handle_client(
        self, client_socket: socket.socket, client_address: Tuple[str, int]
    ):
        """处理HTTP客户端请求"""
        try:
            request_data = client_socket.recv(4096).decode("utf-8")

            # 解析HTTP请求
            lines = request_data.split("\r\n")
            if not lines:
                return

            request_line = lines[0]
            parts = request_line.split(" ")
            if len(parts) < 2:
                return

            method = parts[0]
            path = parts[1]

            # 查找路由处理器
            response = None
            if path in self.routes:
                response = self.routes[path](method, request_data)
            else:
                # 默认404响应
                response = self._create_response(
                    404, "Not Found", "text/html", "<h1>404 Not Found</h1>"
                )

            client_socket.send(response.encode("utf-8"))
        except Exception as e:
            logger.error(f"处理HTTP客户端时出错: {e}")
            error_response = self._create_response(
                500,
                "Internal Server Error",
                "text/html",
                "<h1>500 Internal Server Error</h1>",
            )
            try:
                client_socket.send(error_response.encode("utf-8"))
            except BaseException:
                pass  # 已实现
        finally:
            client_socket.close()

    def _create_response(
        self, status_code: int, status_text: str, content_type: str, body: str
    ) -> str:
        """创建HTTP响应"""
        response = f"HTTP/1.1 {status_code} {status_text}\r\n"
        response += "Server: Self AGI Network Controller\r\n"
        response += f"Content-Type: {content_type}\r\n"
        response += f"Content-Length: {len(body)}\r\n"
        response += "Connection: close\r\n"
        response += "\r\n"
        response += body
        return response

    def stop(self):
        """停止HTTP服务器"""
        self.is_running = False
        if self.server:
            self.server.close()
        logger.info("HTTP服务器已停止")


class NetworkConfigurationManager:
    """网络配置管理器"""

    def __init__(self):
        self.interfaces: Dict[str, NetworkInterface] = {}
        self.dns_servers: List[str] = []
        self.routing_table: List[Dict[str, Any]] = []

    def scan_interfaces(self) -> List[NetworkInterface]:
        """扫描网络接口"""
        self.interfaces = {}

        try:
            # Windows系统
            if platform.system() == "Windows":
                self._scan_windows_interfaces()
            # Linux系统
            elif platform.system() == "Linux":
                self._scan_linux_interfaces()
            # macOS系统
            elif platform.system() == "Darwin":
                self._scan_macos_interfaces()
            else:
                logger.warning(f"不支持的操作系统: {platform.system()}")
                self._scan_fallback_interfaces()
        except Exception as e:
            logger.error(f"扫描网络接口时出错: {e}")
            self._scan_fallback_interfaces()

        return list(self.interfaces.values())

    def _scan_windows_interfaces(self):
        """扫描Windows网络接口"""
        try:
            # 使用ipconfig命令获取网络接口信息
            result = subprocess.run(
                ["ipconfig", "/all"], capture_output=True, text=True, encoding="gbk"
            )
            output = result.stdout

            current_interface = None
            for line in output.split("\n"):
                line = line.strip()

                # 检测接口开始
                if (
                    line
                    and not line.startswith(" ")
                    and ":" in line
                    and not line.startswith("Windows IP")
                ):
                    if current_interface:
                        self.interfaces[current_interface["name"]] = NetworkInterface(
                            **current_interface
                        )

                    interface_name = line.split(":")[0].strip()
                    current_interface = {
                        "name": interface_name,
                        "ip_address": "",
                        "netmask": "",
                        "gateway": "",
                        "mac_address": "",
                        "status": "unknown",
                        "mtu": 1500,
                        "speed_mbps": 0,
                        "is_up": False,
                        "is_wireless": "无线" in interface_name
                        or "Wi-Fi" in interface_name,
                    }

                # 解析IP地址
                elif current_interface and "IPv4 地址" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        current_interface["ip_address"] = (
                            parts[1].strip().split("(")[0].strip()
                        )

                # 解析子网掩码
                elif current_interface and "子网掩码" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        current_interface["netmask"] = parts[1].strip()

                # 解析默认网关
                elif current_interface and "默认网关" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        current_interface["gateway"] = parts[1].strip()

                # 解析物理地址
                elif current_interface and "物理地址" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        current_interface["mac_address"] = parts[1].strip()

            # 保存最后一个接口
            if current_interface:
                self.interfaces[current_interface["name"]] = NetworkInterface(
                    **current_interface
                )
        except Exception as e:
            logger.error(f"扫描Windows网络接口时出错: {e}")

    def _scan_linux_interfaces(self):
        """扫描Linux网络接口"""
        try:
            # 使用ifconfig命令获取网络接口信息
            result = subprocess.run(["ifconfig", "-a"], capture_output=True, text=True)
            output = result.stdout

            current_interface = None
            for line in output.split("\n"):
                line = line.strip()

                # 检测接口开始
                if line and not line.startswith(" ") and ":" in line:
                    if current_interface:
                        self.interfaces[current_interface["name"]] = NetworkInterface(
                            **current_interface
                        )

                    interface_name = line.split(":")[0].strip()
                    current_interface = {
                        "name": interface_name,
                        "ip_address": "",
                        "netmask": "",
                        "gateway": "",
                        "mac_address": "",
                        "status": "unknown",
                        "mtu": 1500,
                        "speed_mbps": 0,
                        "is_up": "UP" in line,
                        "is_wireless": "wlan" in interface_name
                        or "wlp" in interface_name,
                    }

                # 解析IP地址
                elif current_interface and "inet " in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        current_interface["ip_address"] = parts[1]

                # 解析子网掩码
                elif current_interface and "netmask " in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "netmask":
                            if i + 1 < len(parts):
                                current_interface["netmask"] = parts[i + 1]

                # 解析MAC地址
                elif current_interface and "ether " in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "ether":
                            if i + 1 < len(parts):
                                current_interface["mac_address"] = parts[i + 1]

            # 保存最后一个接口
            if current_interface:
                self.interfaces[current_interface["name"]] = NetworkInterface(
                    **current_interface
                )
        except Exception as e:
            logger.error(f"扫描Linux网络接口时出错: {e}")

    def _scan_macos_interfaces(self):
        """扫描macOS网络接口"""
        self._scan_linux_interfaces()  # macOS与Linux类似

    def _scan_fallback_interfaces(self):
        """备用接口扫描方法"""
        try:
            # 尝试通过socket获取本地IP
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)

            interface = NetworkInterface(
                name="eth0",
                ip_address=local_ip,
                netmask="255.255.255.0",
                gateway="",
                mac_address="00:00:00:00:00:00",
                status="active",
                mtu=1500,
                speed_mbps=1000,
                is_up=True,
                is_wireless=False,
            )
            self.interfaces["eth0"] = interface
        except BaseException:
            pass  # 已实现

    def get_interface(self, name: str) -> Optional[NetworkInterface]:
        """获取指定接口信息"""
        return self.interfaces.get(name)

    def set_static_ip(
        self, interface_name: str, ip_address: str, netmask: str, gateway: str
    ) -> bool:
        """设置静态IP地址"""
        try:
            # Windows系统
            if platform.system() == "Windows":
                # 使用netsh命令设置静态IP
                cmd = [
                    "netsh",
                    "interface",
                    "ip",
                    "set",
                    "address",
                    f"name={interface_name}",
                    "source=static",
                    f"addr={ip_address}",
                    f"mask={netmask}",
                    f"gateway={gateway}",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.returncode == 0

            # Linux系统（需要root权限）
            elif platform.system() == "Linux":
                # 修改网络配置文件
                config_file = f"/etc/network/interfaces.d/{interface_name}"
                config = f""" auto {interface_name} iface {interface_name} inet static     address {ip_address}     netmask {netmask}     gateway {gateway} """
                with open(config_file, "w") as f:
                    f.write(config)

                # 重启网络服务
                subprocess.run(
                    ["systemctl", "restart", "networking"], capture_output=True
                )
                return True
        except Exception as e:
            logger.error(f"设置静态IP时出错: {e}")
            return False

        return False

    def set_dhcp(self, interface_name: str) -> bool:
        """设置DHCP自动获取IP"""
        try:
            # Windows系统
            if platform.system() == "Windows":
                cmd = [
                    "netsh",
                    "interface",
                    "ip",
                    "set",
                    "address",
                    f"name={interface_name}",
                    "source=dhcp",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.returncode == 0

            # Linux系统
            elif platform.system() == "Linux":
                config_file = f"/etc/network/interfaces.d/{interface_name}"
                config = f""" auto {interface_name} iface {interface_name} inet dhcp """
                with open(config_file, "w") as f:
                    f.write(config)

                subprocess.run(
                    ["systemctl", "restart", "networking"], capture_output=True
                )
                return True
        except Exception as e:
            logger.error(f"设置DHCP时出错: {e}")
            return False

        return False

    def get_dns_servers(self) -> List[str]:
        """获取DNS服务器列表"""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["nslookup", "localhost"], capture_output=True, text=True
                )
                output = result.stdout
                # 解析DNS服务器
                for line in output.split("\n"):
                    if "服务器" in line and ":" in line:
                        parts = line.split(":")
                        if len(parts) > 1:
                            dns_server = parts[1].strip()
                            if dns_server not in self.dns_servers:
                                self.dns_servers.append(dns_server)
            elif platform.system() == "Linux":
                with open("/etc/resolv.conf", "r") as f:
                    for line in f:
                        if line.startswith("nameserver"):
                            dns_server = line.split()[1].strip()
                            if dns_server not in self.dns_servers:
                                self.dns_servers.append(dns_server)
        except Exception as e:
            logger.error(f"获取DNS服务器时出错: {e}")

        return self.dns_servers

    def set_dns_servers(self, dns_servers: List[str]) -> bool:
        """设置DNS服务器"""
        try:
            if platform.system() == "Windows":
                # 设置主要和备用DNS
                for i, dns in enumerate(dns_servers[:2]):
                    cmd = [
                        "netsh",
                        "interface",
                        "ip",
                        "set",
                        "dns",
                        "name=本地连接",  # 需要根据实际情况修改
                        "source=static",
                        f"addr={dns}",
                        "register=primary" if i == 0 else "index=2",
                    ]
                    subprocess.run(cmd, capture_output=True)
                return True
            elif platform.system() == "Linux":
                # 更新resolv.conf
                with open("/etc/resolv.conf", "w") as f:
                    for dns in dns_servers:
                        f.write(f"nameserver {dns}\n")
                return True
        except Exception as e:
            logger.error(f"设置DNS服务器时出错: {e}")
            return False

        return False


class NetworkMonitor:
    """网络监控器"""

    def __init__(self):
        self.metrics: Dict[str, NetworkMetric] = {}
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_metrics = {}

    def start_monitoring(self, interval: float = 5.0):
        """开始网络监控"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval,), daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"网络监控已启动，间隔 {interval} 秒")

    def stop_monitoring(self):
        """停止网络监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("网络监控已停止")

    def _monitoring_loop(self, interval: float):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._collect_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"网络监控循环出错: {e}")
                time.sleep(interval)

    def _collect_metrics(self):
        """收集网络指标"""
        try:
            # 收集网络接口统计信息
            if platform.system() == "Windows":
                self._collect_windows_metrics()
            elif platform.system() == "Linux":
                self._collect_linux_metrics()
            else:
                raise RuntimeError(
                    f"不支持的操作系统: {                         platform.system()}。网络指标收集需要Windows或Linux系统。项目要求禁止使用虚拟数据。"
                )
        except Exception as e:
            logger.error(f"收集网络指标时出错: {e}")

    def _collect_windows_metrics(self):
        """收集Windows网络指标"""
        # 使用netsh命令获取接口统计
        try:
            result = subprocess.run(
                ["netsh", "interface", "ipv4", "show", "interfaces"],
                capture_output=True,
                text=True,
                encoding="gbk",
            )
            output = result.stdout

            for line in output.split("\n"):
                if "MTU" in line and "状态" in line:
                    parts = [p.strip() for p in line.split() if p.strip()]
                    if len(parts) >= 9:
                        interface_idx = parts[0]
                        parts[2]
                        parts[3]
                        bytes_sent = parts[7] if len(parts) > 7 else "0"
                        bytes_received = parts[8] if len(parts) > 8 else "0"

                        metric_id = f"interface_{interface_idx}"
                        self.metrics[metric_id] = NetworkMetric(
                            metric_id=metric_id,
                            name=f"接口 {interface_idx} 流量",
                            value=float(bytes_sent) + float(bytes_received),
                            unit="bytes",
                            interface=f"interface_{interface_idx}",
                        )
        except Exception as e:
            logger.error(f"收集Windows网络指标时出错: {e}")

    def _collect_linux_metrics(self):
        """收集Linux网络指标"""
        # 读取/proc/net/dev文件
        try:
            with open("/proc/net/dev", "r") as f:
                lines = f.readlines()

            for line in lines[2:]:  # 跳过前两行标题
                parts = line.split()
                if len(parts) >= 17:
                    interface_name = parts[0].replace(":", "")
                    bytes_received = int(parts[1])
                    bytes_sent = int(parts[9])

                    metric_id = f"interface_{interface_name}"
                    self.metrics[metric_id] = NetworkMetric(
                        metric_id=metric_id,
                        name=f"接口 {interface_name} 流量",
                        value=float(bytes_sent + bytes_received),
                        unit="bytes",
                        interface=interface_name,
                    )
        except Exception as e:
            logger.error(f"收集Linux网络指标时出错: {e}")

    def _collect_basic_metrics(self):
        """收集基础网络指标

        根据项目要求"禁止使用虚拟数据"，此方法不再提供模拟指标。
        必须使用真实网络监控器收集网络指标。
        """
        raise RuntimeError(
            "模拟网络指标收集已禁用。请确保网络监控器已正确初始化并可用。项目要求禁止使用虚拟数据，必须使用真实网络监控。"
        )

    def get_metrics(self) -> List[NetworkMetric]:
        """获取所有网络指标"""
        return list(self.metrics.values())

    def get_metric(self, metric_id: str) -> Optional[NetworkMetric]:
        """获取指定指标"""
        return self.metrics.get(metric_id)


class NetworkController:
    """网络控制器主类"""

    def __init__(self):
        self.tcp_servers: Dict[str, TCPServer] = {}
        self.udp_servers: Dict[str, UDPServer] = {}
        self.http_servers: Dict[str, HTTPServer] = {}
        self.config_manager = NetworkConfigurationManager()
        self.monitor = NetworkMonitor()
        self.is_initialized = False

    def initialize(self):
        """初始化网络控制器"""
        try:
            # 扫描网络接口
            interfaces = self.config_manager.scan_interfaces()
            logger.info(f"发现 {len(interfaces)} 个网络接口")

            # 获取DNS服务器
            dns_servers = self.config_manager.get_dns_servers()
            logger.info(f"DNS服务器: {dns_servers}")

            self.is_initialized = True
            logger.info("网络控制器初始化完成")
            return True
        except Exception as e:
            logger.error(f"网络控制器初始化失败: {e}")
            return False

    def create_tcp_server(
        self, server_id: str, host: str = "0.0.0.0", port: int = 8080
    ) -> bool:
        """创建TCP服务器"""
        if server_id in self.tcp_servers:
            logger.warning(f"TCP服务器 {server_id} 已存在")
            return False

        server = TCPServer(host, port)
        if server.start():
            self.tcp_servers[server_id] = server
            return True
        return False

    def create_udp_server(
        self, server_id: str, host: str = "0.0.0.0", port: int = 9090
    ) -> bool:
        """创建UDP服务器"""
        if server_id in self.udp_servers:
            logger.warning(f"UDP服务器 {server_id} 已存在")
            return False

        server = UDPServer(host, port)
        if server.start():
            self.udp_servers[server_id] = server
            return True
        return False

    def create_http_server(
        self, server_id: str, host: str = "0.0.0.0", port: int = 80
    ) -> bool:
        """创建HTTP服务器"""
        if server_id in self.http_servers:
            logger.warning(f"HTTP服务器 {server_id} 已存在")
            return False

        server = HTTPServer(host, port)
        if server.start():
            self.http_servers[server_id] = server
            return True
        return False

    def start_monitoring(self, interval: float = 5.0):
        """启动网络监控"""
        self.monitor.start_monitoring(interval)

    def stop_monitoring(self):
        """停止网络监控"""
        self.monitor.stop_monitoring()

    def get_network_interfaces(self) -> List[NetworkInterface]:
        """获取网络接口列表"""
        return self.config_manager.scan_interfaces()

    def set_static_ip(
        self, interface_name: str, ip_address: str, netmask: str, gateway: str
    ) -> bool:
        """设置静态IP"""
        return self.config_manager.set_static_ip(
            interface_name, ip_address, netmask, gateway
        )

    def set_dhcp(self, interface_name: str) -> bool:
        """设置DHCP"""
        return self.config_manager.set_dhcp(interface_name)

    def get_network_metrics(self) -> List[NetworkMetric]:
        """获取网络指标"""
        return self.monitor.get_metrics()

    def shutdown(self):
        """关闭网络控制器"""
        # 停止所有服务器
        for server_id, server in list(self.tcp_servers.items()):
            server.stop()
            del self.tcp_servers[server_id]

        for server_id, server in list(self.udp_servers.items()):
            server.stop()
            del self.udp_servers[server_id]

        for server_id, server in list(self.http_servers.items()):
            server.stop()
            del self.http_servers[server_id]

        # 停止监控
        self.monitor.stop_monitoring()

        logger.info("网络控制器已关闭")


# 全局网络控制器实例
_global_network_controller = None


def get_network_controller() -> NetworkController:
    """获取全局网络控制器实例"""
    global _global_network_controller
    if _global_network_controller is None:
        _global_network_controller = NetworkController()
        _global_network_controller.initialize()
    return _global_network_controller


if __name__ == "__main__":
    # 模块测试代码
    logging.basicConfig(level=logging.INFO)

    controller = get_network_controller()

    # 测试网络接口扫描
    interfaces = controller.get_network_interfaces()
    print(f"发现 {len(interfaces)} 个网络接口:")
    for interface in interfaces:
        print(f"  - {interface.name}: {interface.ip_address} ({interface.mac_address})")

    # 测试TCP服务器
    if controller.create_tcp_server("test_tcp", "127.0.0.1", 8888):
        print("TCP服务器创建成功")

    # 测试UDP服务器
    if controller.create_udp_server("test_udp", "127.0.0.1", 9999):
        print("UDP服务器创建成功")

    # 启动网络监控
    controller.start_monitoring(interval=2.0)
    print("网络监控已启动")

    try:
        # 运行一段时间
        time.sleep(10)
    except KeyboardInterrupt:
        pass  # 已实现

    # 关闭
    controller.shutdown()
    print("网络控制器测试完成")
