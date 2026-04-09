"""
串口控制器

功能：
- 管理串口连接和通信
- 支持多种串口协议
- 提供异步和同步通信模式
- 错误处理和重连机制
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import serial
import serial.tools.list_ports


class SerialProtocol(Enum):
    """串口协议枚举"""

    RAW = "raw"  # 原始数据
    ASCII = "ascii"  # ASCII文本
    HEX = "hex"  # 十六进制
    JSON = "json"  # JSON格式
    CUSTOM = "custom"  # 自定义协议


class SerialState(Enum):
    """串口状态枚举"""

    CLOSED = "closed"  # 完全关闭，未连接设备
    CONNECTING = "connecting"  # 正在连接中
    CONNECTED = "connected"  # 已连接但未开始工作
    WORKING = "working"  # 连接成功并开始工作（接收/发送数据）
    ERROR = "error"  # 错误状态
    DISCONNECTING = "disconnecting"  # 正在断开连接


class SerialController:
    """串口控制器

    功能：
    - 串口连接管理
    - 数据收发
    - 协议解析
    - 错误处理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化串口控制器

        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)

        # 默认配置
        self.config = config or {
            "baudrate": 9600,
            "bytesize": serial.EIGHTBITS,
            "parity": serial.PARITY_NONE,
            "stopbits": serial.STOPBITS_ONE,
            "timeout": 1.0,
            "write_timeout": 1.0,
            "xonxoff": False,
            "rtscts": False,
            "dsrdtr": False,
        }

        # 串口连接
        self.serial_connection = None
        self.is_connected = False
        self.state = SerialState.CLOSED  # 初始状态为关闭

        # 接收线程
        self.receive_thread = None
        self.receive_running = False

        # 数据缓冲区
        self.receive_buffer = bytearray()
        self.receive_callbacks = []

        # 发送队列
        self.send_queue = []
        self.send_thread = None
        self.send_running = False

        # 统计信息
        self.stats = {
            "bytes_received": 0,
            "bytes_sent": 0,
            "errors": 0,
            "last_activity": None,
        }

        self.logger.info("串口控制器初始化完成")

    def list_available_ports(self) -> List[Dict[str, Any]]:
        """列出可用串口

        返回:
            可用串口列表
        """
        ports = []
        for port in serial.tools.list_ports.comports():
            port_info = {
                "device": port.device,
                "name": port.name,
                "description": port.description,
                "hwid": port.hwid,
                "vid": port.vid,
                "pid": port.pid,
                "serial_number": port.serial_number,
                "location": port.location,
                "manufacturer": port.manufacturer,
                "product": port.product,
                "interface": port.interface,
            }
            ports.append(port_info)

        self.logger.info(f"发现 {len(ports)} 个可用串口")
        return ports

    def connect(self, port: str, baudrate: Optional[int] = None, **kwargs) -> bool:
        """连接串口

        参数:
            port: 串口设备路径
            baudrate: 波特率
            **kwargs: 其他串口参数

        返回:
            连接是否成功
        """
        if self.state in [
            SerialState.CONNECTING,
            SerialState.CONNECTED,
            SerialState.WORKING,
        ]:
            self.logger.warning(f"串口已在 {self.state.value} 状态，无需重复连接")
            return self.state in [SerialState.CONNECTED, SerialState.WORKING]

        try:
            # 更新状态为连接中
            self.state = SerialState.CONNECTING
            self.logger.info(f"正在连接串口: {port}")

            # 合并配置
            serial_config = self.config.copy()
            serial_config["port"] = port

            if baudrate:
                serial_config["baudrate"] = baudrate

            # 覆盖其他参数
            for key, value in kwargs.items():
                serial_config[key] = value

            # 创建串口连接
            self.serial_connection = serial.Serial(**serial_config)
            self.is_connected = True

            # 启动接收线程
            self.receive_running = True
            self.receive_thread = threading.Thread(
                target=self._receive_loop, daemon=True, name="SerialReceiver"
            )
            self.receive_thread.start()

            # 启动发送线程
            self.send_running = True
            self.send_thread = threading.Thread(
                target=self._send_loop, daemon=True, name="SerialSender"
            )
            self.send_thread.start()

            # 更新状态为工作状态（连接成功并开始工作）
            self.state = SerialState.WORKING

            self.logger.info(
                f"串口连接成功并开始工作: {port}, 波特率: {serial_config['baudrate']}"
            )
            return True

        except Exception as e:
            self.logger.error(f"串口连接失败: {e}")
            self.is_connected = False
            self.state = SerialState.ERROR
            return False

    def disconnect(self) -> bool:
        """断开串口连接

        返回:
            断开是否成功
        """
        if self.state == SerialState.CLOSED:
            self.logger.warning("串口已处于关闭状态")
            return True

        if self.state == SerialState.DISCONNECTING:
            self.logger.warning("串口正在断开连接中")
            return False

        try:
            # 更新状态为断开连接中
            self.state = SerialState.DISCONNECTING
            self.logger.info("正在断开串口连接...")

            # 停止接收线程
            self.receive_running = False
            if self.receive_thread and self.receive_thread.is_alive():
                self.receive_thread.join(timeout=2.0)

            # 停止发送线程
            self.send_running = False
            if self.send_thread and self.send_thread.is_alive():
                self.send_thread.join(timeout=2.0)

            # 关闭串口连接
            if self.serial_connection:
                self.serial_connection.close()
                self.serial_connection = None

            self.is_connected = False
            self.state = SerialState.CLOSED  # 更新状态为完全关闭
            self.logger.info("串口连接已断开，状态: 关闭")
            return True

        except Exception as e:
            self.logger.error(f"断开串口连接失败: {e}")
            self.state = SerialState.ERROR
            return False

    def send(
        self,
        data: Union[bytes, str, List[int]],
        protocol: SerialProtocol = SerialProtocol.RAW,
        callback: Optional[Callable] = None,
    ) -> bool:
        """发送数据

        参数:
            data: 要发送的数据
            protocol: 数据协议
            callback: 发送完成回调

        返回:
            发送是否成功
        """
        if self.state != SerialState.WORKING:
            self.logger.error(
                f"串口状态为 {self.state.value}，无法发送数据（需要WORKING状态）"
            )
            return False

        try:
            # 根据协议编码数据
            encoded_data = self._encode_data(data, protocol)

            # 添加到发送队列
            send_task = {
                "data": encoded_data,
                "protocol": protocol,
                "callback": callback,
                "timestamp": time.time(),
            }

            self.send_queue.append(send_task)
            return True

        except Exception as e:
            self.logger.error(f"准备发送数据失败: {e}")
            self.stats["errors"] += 1
            return False

    def send_sync(
        self,
        data: Union[bytes, str, List[int]],
        protocol: SerialProtocol = SerialProtocol.RAW,
        timeout: float = 5.0,
    ) -> bool:
        """同步发送数据

        参数:
            data: 要发送的数据
            protocol: 数据协议
            timeout: 超时时间

        返回:
            发送是否成功
        """
        if self.state != SerialState.WORKING:
            self.logger.error(
                f"串口状态为 {self.state.value}，无法发送数据（需要WORKING状态）"
            )
            return False

        try:
            # 根据协议编码数据
            encoded_data = self._encode_data(data, protocol)

            # 直接发送（同步）
            bytes_sent = self.serial_connection.write(encoded_data)

            if bytes_sent == len(encoded_data):
                self.stats["bytes_sent"] += bytes_sent
                self.stats["last_activity"] = time.time()
                self.logger.debug(f"同步发送 {bytes_sent} 字节数据")
                return True
            else:
                self.logger.warning(
                    f"部分数据发送失败: {bytes_sent}/{len(encoded_data)}"
                )
                self.stats["errors"] += 1
                return False

        except Exception as e:
            self.logger.error(f"同步发送数据失败: {e}")
            self.stats["errors"] += 1
            return False

    def register_receive_callback(
        self, callback: Callable[[bytes, SerialProtocol], None]
    ):
        """注册接收数据回调

        参数:
            callback: 回调函数，接收数据和协议
        """
        if callback not in self.receive_callbacks:
            self.receive_callbacks.append(callback)
            self.logger.debug("注册接收数据回调")

    def unregister_receive_callback(
        self, callback: Callable[[bytes, SerialProtocol], None]
    ):
        """注销接收数据回调

        参数:
            callback: 要注销的回调函数
        """
        if callback in self.receive_callbacks:
            self.receive_callbacks.remove(callback)
            self.logger.debug("注销接收数据回调")

    def clear_receive_buffer(self):
        """清空接收缓冲区"""
        self.receive_buffer.clear()
        self.logger.debug("清空接收缓冲区")

    def get_receive_buffer(self) -> bytes:
        """获取接收缓冲区内容

        返回:
            缓冲区数据
        """
        return bytes(self.receive_buffer)

    def _receive_loop(self):
        """接收数据循环"""
        self.logger.info("串口接收线程启动")

        while self.receive_running and self.state == SerialState.WORKING:
            try:
                if self.serial_connection and self.serial_connection.in_waiting > 0:
                    # 读取数据
                    data = self.serial_connection.read(
                        self.serial_connection.in_waiting
                    )

                    if data:
                        # 更新统计
                        self.stats["bytes_received"] += len(data)
                        self.stats["last_activity"] = time.time()

                        # 添加到缓冲区
                        self.receive_buffer.extend(data)

                        # 调用回调函数
                        for callback in self.receive_callbacks:
                            try:
                                callback(data, SerialProtocol.RAW)
                            except Exception as e:
                                self.logger.error(f"接收回调执行失败: {e}")

                        self.logger.debug(f"接收 {len(data)} 字节数据")

                # 短暂休眠以避免CPU占用过高
                time.sleep(0.001)

            except serial.SerialException as e:
                self.logger.error(f"串口接收错误: {e}")
                self.stats["errors"] += 1
                self.is_connected = False
                self.state = SerialState.ERROR
                break
            except Exception as e:
                self.logger.error(f"接收循环异常: {e}")
                self.stats["errors"] += 1
                time.sleep(0.1)

        self.logger.info("串口接收线程停止")

    def _send_loop(self):
        """发送数据循环"""
        self.logger.info("串口发送线程启动")

        while self.send_running and self.state == SerialState.WORKING:
            try:
                if self.send_queue and self.serial_connection:
                    # 获取发送任务
                    send_task = self.send_queue.pop(0)
                    data = send_task["data"]
                    callback = send_task["callback"]

                    # 发送数据
                    bytes_sent = self.serial_connection.write(data)

                    if bytes_sent == len(data):
                        # 更新统计
                        self.stats["bytes_sent"] += bytes_sent
                        self.stats["last_activity"] = time.time()

                        # 调用回调函数
                        if callback:
                            try:
                                callback(True, bytes_sent)
                            except Exception as e:
                                self.logger.error(f"发送回调执行失败: {e}")

                        self.logger.debug(f"发送 {bytes_sent} 字节数据")
                    else:
                        self.logger.warning(
                            f"部分数据发送失败: {bytes_sent}/{len(data)}"
                        )
                        self.stats["errors"] += 1

                        if callback:
                            try:
                                callback(False, bytes_sent)
                            except Exception as e:
                                self.logger.error(f"发送回调执行失败: {e}")

                # 短暂休眠以避免CPU占用过高
                time.sleep(0.001)

            except serial.SerialException as e:
                self.logger.error(f"串口发送错误: {e}")
                self.stats["errors"] += 1
                self.is_connected = False
                self.state = SerialState.ERROR
                break
            except Exception as e:
                self.logger.error(f"发送循环异常: {e}")
                self.stats["errors"] += 1
                time.sleep(0.1)

        self.logger.info("串口发送线程停止")

    def _encode_data(
        self, data: Union[bytes, str, List[int]], protocol: SerialProtocol
    ) -> bytes:
        """编码数据

        参数:
            data: 原始数据
            protocol: 协议

        返回:
            编码后的字节数据
        """
        if isinstance(data, bytes):
            return data

        elif isinstance(data, str):
            if protocol == SerialProtocol.HEX:
                # 十六进制字符串转字节
                data = data.strip().replace(" ", "")
                if len(data) % 2 != 0:
                    raise ValueError("十六进制字符串长度必须为偶数")
                return bytes.fromhex(data)
            else:
                # ASCII/UTF-8编码
                return data.encode("utf-8")

        elif isinstance(data, list):
            # 整数列表转字节
            return bytes(data)

        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

    def _decode_data(
        self, data: bytes, protocol: SerialProtocol
    ) -> Union[str, bytes, List[int]]:
        """解码数据

        参数:
            data: 字节数据
            protocol: 协议

        返回:
            解码后的数据
        """
        if protocol == SerialProtocol.RAW:
            return data

        elif protocol == SerialProtocol.ASCII:
            return data.decode("ascii", errors="ignore")

        elif protocol == SerialProtocol.HEX:
            return data.hex()

        elif protocol == SerialProtocol.JSON:
            import json

            return json.loads(data.decode("utf-8", errors="ignore"))

        else:
            return data

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        返回:
            统计信息字典
        """
        stats = self.stats.copy()
        stats["is_connected"] = self.is_connected
        stats["state"] = self.state.value  # 添加详细状态信息
        stats["receive_buffer_size"] = len(self.receive_buffer)
        stats["send_queue_size"] = len(self.send_queue)
        stats["receive_callbacks"] = len(self.receive_callbacks)

        if self.serial_connection:
            stats["port"] = self.serial_connection.port
            stats["baudrate"] = self.serial_connection.baudrate

        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "bytes_received": 0,
            "bytes_sent": 0,
            "errors": 0,
            "last_activity": None,
        }
        self.logger.info("统计信息已重置")

    def is_port_available(self, port: str) -> bool:
        """检查串口是否可用

        参数:
            port: 串口设备路径

        返回:
            是否可用
        """
        try:
            # 尝试打开串口
            test_serial = serial.Serial(
                port=port, baudrate=self.config["baudrate"], timeout=0.1
            )
            test_serial.close()
            return True
        except Exception:
            return False

    def get_state(self) -> SerialState:
        """获取当前串口状态

        返回:
            当前状态
        """
        return self.state

    def is_closed(self) -> bool:
        """检查串口是否关闭

        返回:
            是否关闭
        """
        return self.state == SerialState.CLOSED

    def is_working(self) -> bool:
        """检查串口是否在工作状态

        返回:
            是否在工作状态
        """
        return self.state == SerialState.WORKING

    def check_connection(self) -> bool:
        """检查连接状态（真实硬件检测）

        返回:
            真实连接状态
        """
        if self.serial_connection:
            try:
                # 尝试读取端口信息来检测连接
                _ = self.serial_connection.port
                return True
            except Exception:
                self.state = SerialState.ERROR
                self.is_connected = False
                return False
        return False

    def __del__(self):
        """析构函数，确保资源被清理"""
        self.disconnect()
