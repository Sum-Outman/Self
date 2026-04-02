#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口数据服务

功能：
1. 统一管理所有串口连接和数据接收
2. 调用解码器将原始数据转换为结构化数据
3. 将解码后的数据发布给AGI系统和其他订阅者
4. 提供数据缓存和历史查询

设计原则：
- 简单直接：前端只管接收原始数据，后台统一处理
- 解耦合：解码逻辑与接收逻辑分离
- 可扩展：支持多种设备和数据格式
- 实时性：支持WebSocket等实时数据流
"""

import logging
import threading
import time
import queue
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timezone
from enum import Enum
import json

# 导入解码器
from .serial_decoder import SerialDecoder, SerialProtocol, get_serial_decoder, DecodeResult

# 尝试导入串口控制器
try:
    from models.system_control import SerialController
    SERIAL_CONTROLLER_AVAILABLE = True
except ImportError:
    SERIAL_CONTROLLER_AVAILABLE = False
    SerialController = None

logger = logging.getLogger(__name__)


class DataDestination(Enum):
    """数据目的地枚举"""
    AGI_SYSTEM = "agi_system"          # AGI系统
    MEMORY_SYSTEM = "memory_system"    # 记忆系统
    DATABASE = "database"              # 数据库
    WEBSOCKET = "websocket"            # WebSocket客户端
    FILE = "file"                      # 文件存储
    CUSTOM = "custom"                  # 自定义目的地


class SerialDataService:
    """串口数据服务
    
    统一接收串口数据，解码后分发给各个目的地
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化串口数据服务
        
        参数:
            config: 配置字典
        """
        self.logger = logger
        self.config = config or {}
        
        # 串口控制器实例
        self.serial_controller: Optional[SerialController] = None
        
        # 解码器实例
        self.decoder: SerialDecoder = get_serial_decoder(
            self.config.get("decoder_config")
        )
        
        # 数据队列（原始数据）
        self.raw_data_queue = queue.Queue(maxsize=1000)
        
        # 解码后数据队列
        self.decoded_data_queue = queue.Queue(maxsize=1000)
        
        # 目的地处理器
        self.destination_handlers: Dict[DataDestination, List[Callable]] = {
            DataDestination.AGI_SYSTEM: [],
            DataDestination.MEMORY_SYSTEM: [],
            DataDestination.DATABASE: [],
            DataDestination.WEBSOCKET: [],
            DataDestination.FILE: [],
            DataDestination.CUSTOM: [],
        }
        
        # 串口配置
        self.serial_configs: Dict[str, Dict[str, Any]] = {}  # port -> config
        self.active_connections: Set[str] = set()  # 活跃连接端口
        
        # 处理线程
        self.processing_thread: Optional[threading.Thread] = None
        self.distribution_thread: Optional[threading.Thread] = None
        self.running = False
        
        # 统计信息
        self.stats = {
            "raw_data_received": 0,
            "data_decoded": 0,
            "data_distributed": 0,
            "errors": 0,
            "active_connections": 0,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        
        # 数据缓存（最近N条数据）
        self.recent_data_cache: List[Dict[str, Any]] = []
        self.max_cache_size = self.config.get("max_cache_size", 1000)
        
        self.logger.info("串口数据服务初始化完成")
    
    def start(self) -> bool:
        """启动串口数据服务
        
        返回:
            是否成功启动
        """
        try:
            if self.running:
                self.logger.warning("串口数据服务已在运行")
                return True
            
            self.logger.info("启动串口数据服务...")
            self.running = True
            
            # 启动处理线程
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="SerialDataProcessor"
            )
            self.processing_thread.start()
            
            # 启动分发线程
            self.distribution_thread = threading.Thread(
                target=self._distribution_loop,
                daemon=True,
                name="SerialDataDistributor"
            )
            self.distribution_thread.start()
            
            # 初始化串口连接（如果有配置）
            self._initialize_serial_connections()
            
            self.logger.info("串口数据服务启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"启动串口数据服务失败: {e}")
            self.running = False
            return False
    
    def stop(self) -> bool:
        """停止串口数据服务
        
        返回:
            是否成功停止
        """
        try:
            if not self.running:
                self.logger.warning("串口数据服务未运行")
                return True
            
            self.logger.info("停止串口数据服务...")
            self.running = False
            
            # 停止处理线程
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            # 停止分发线程
            if self.distribution_thread and self.distribution_thread.is_alive():
                self.distribution_thread.join(timeout=5.0)
            
            # 断开所有串口连接
            self._disconnect_all_serial()
            
            self.logger.info("串口数据服务停止成功")
            return True
            
        except Exception as e:
            self.logger.error(f"停止串口数据服务失败: {e}")
            return False
    
    def register_destination_handler(
        self,
        destination: DataDestination,
        handler: Callable[[Dict[str, Any]], bool]
    ):
        """注册目的地处理器
        
        参数:
            destination: 数据目的地
            handler: 处理器函数，接收解码后的数据，返回处理是否成功
        """
        if destination not in self.destination_handlers:
            self.destination_handlers[destination] = []
        
        self.destination_handlers[destination].append(handler)
        self.logger.info(f"注册{destination.value}目的地处理器")
    
    def unregister_destination_handler(
        self,
        destination: DataDestination,
        handler: Callable[[Dict[str, Any]], bool]
    ):
        """注销目的地处理器"""
        if destination in self.destination_handlers:
            if handler in self.destination_handlers[destination]:
                self.destination_handlers[destination].remove(handler)
                self.logger.info(f"注销{destination.value}目的地处理器")
    
    def add_serial_config(
        self,
        port: str,
        baudrate: int = 9600,
        protocol: Optional[SerialProtocol] = None,
        hint: Optional[str] = None,
        auto_connect: bool = True
    ):
        """添加串口配置
        
        参数:
            port: 串口端口（如COM3, /dev/ttyUSB0）
            baudrate: 波特率
            protocol: 数据协议（如果为None则自动检测）
            hint: 解码提示（如设备类型）
            auto_connect: 是否自动连接
        """
        config = {
            "port": port,
            "baudrate": baudrate,
            "protocol": protocol,
            "hint": hint,
            "auto_connect": auto_connect,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        self.serial_configs[port] = config
        self.logger.info(f"添加串口配置: {port} @ {baudrate} baud")
        
        if auto_connect:
            self.connect_serial(port)
    
    def connect_serial(self, port: str) -> bool:
        """连接串口
        
        参数:
            port: 串口端口
        
        返回:
            连接是否成功
        """
        try:
            if port not in self.serial_configs:
                self.logger.error(f"未找到串口配置: {port}")
                return False
            
            if not SERIAL_CONTROLLER_AVAILABLE:
                self.logger.error("串口控制器不可用")
                return False
            
            config = self.serial_configs[port]
            
            # 创建或获取串口控制器
            if self.serial_controller is None:
                self.serial_controller = SerialController({
                    "baudrate": config["baudrate"],
                    "timeout": 1.0,
                })
            
            # 连接串口
            success = self.serial_controller.connect(
                port=config["port"],
                baudrate=config["baudrate"]
            )
            
            if success:
                # 注册数据接收回调
                self.serial_controller.register_receive_callback(
                    self._serial_data_callback
                )
                
                self.active_connections.add(port)
                self.stats["active_connections"] = len(self.active_connections)
                
                self.logger.info(f"串口连接成功: {port}")
                return True
            else:
                self.logger.error(f"串口连接失败: {port}")
                return False
                
        except Exception as e:
            self.logger.error(f"连接串口失败 {port}: {e}")
            self.stats["errors"] += 1
            return False
    
    def disconnect_serial(self, port: str) -> bool:
        """断开串口连接"""
        try:
            if port not in self.active_connections:
                self.logger.warning(f"串口未连接: {port}")
                return True
            
            if self.serial_controller:
                self.serial_controller.disconnect()
            
            self.active_connections.remove(port)
            self.stats["active_connections"] = len(self.active_connections)
            
            self.logger.info(f"串口断开连接: {port}")
            return True
            
        except Exception as e:
            self.logger.error(f"断开串口连接失败 {port}: {e}")
            return False
    
    def send_serial_data(
        self,
        port: str,
        data: bytes,
        protocol: SerialProtocol = SerialProtocol.RAW
    ) -> bool:
        """发送串口数据
        
        参数:
            port: 串口端口
            data: 要发送的数据
            protocol: 数据协议
        
        返回:
            发送是否成功
        """
        try:
            if port not in self.active_connections:
                self.logger.error(f"串口未连接: {port}")
                return False
            
            if not self.serial_controller:
                self.logger.error("串口控制器未初始化")
                return False
            
            success = self.serial_controller.send(data, protocol=protocol)
            
            if success:
                self.logger.debug(f"串口数据发送成功: {port}, {len(data)} 字节")
            else:
                self.logger.warning(f"串口数据发送失败: {port}")
                self.stats["errors"] += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"发送串口数据失败 {port}: {e}")
            self.stats["errors"] += 1
            return False
    
    def receive_raw_data(
        self,
        raw_data: bytes,
        source_port: Optional[str] = None,
        protocol_hint: Optional[str] = None
    ):
        """接收原始串口数据（外部调用）
        
        参数:
            raw_data: 原始字节数据
            source_port: 数据来源端口
            protocol_hint: 协议提示
        """
        try:
            # 创建数据包
            data_packet = {
                "raw_data": raw_data,
                "source_port": source_port or "unknown",
                "protocol_hint": protocol_hint,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_length": len(raw_data),
            }
            
            # 放入原始数据队列
            self.raw_data_queue.put(data_packet, block=False)
            self.stats["raw_data_received"] += 1
            
            self.logger.debug(f"接收原始数据: {source_port}, {len(raw_data)} 字节")
            
        except queue.Full:
            self.logger.warning("原始数据队列已满，丢弃数据")
            self.stats["errors"] += 1
        except Exception as e:
            self.logger.error(f"接收原始数据失败: {e}")
            self.stats["errors"] += 1
    
    def get_recent_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近的数据
        
        参数:
            limit: 返回的数据条数限制
        
        返回:
            最近的数据列表
        """
        return self.recent_data_cache[-limit:] if self.recent_data_cache else []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats.update({
            "running": self.running,
            "active_connections_count": len(self.active_connections),
            "active_connections": list(self.active_connections),
            "serial_configs_count": len(self.serial_configs),
            "decoder_stats": self.decoder.get_stats(),
            "raw_queue_size": self.raw_data_queue.qsize(),
            "decoded_queue_size": self.decoded_data_queue.qsize(),
            "recent_data_count": len(self.recent_data_cache),
            "current_time": datetime.now(timezone.utc).isoformat(),
        })
        return stats
    
    def _initialize_serial_connections(self):
        """初始化串口连接"""
        for port, config in self.serial_configs.items():
            if config.get("auto_connect", True):
                self.connect_serial(port)
    
    def _disconnect_all_serial(self):
        """断开所有串口连接"""
        for port in list(self.active_connections):
            self.disconnect_serial(port)
    
    def _serial_data_callback(self, data: bytes, protocol: SerialProtocol):
        """串口数据回调函数（由SerialController调用）"""
        try:
            # 完整实现，实际需要跟踪哪个端口接收的数据）
            source_port = "unknown"
            if self.active_connections:
                source_port = next(iter(self.active_connections))
            
            # 接收原始数据
            self.receive_raw_data(
                raw_data=data,
                source_port=source_port,
                protocol_hint=protocol.value if protocol else None
            )
            
        except Exception as e:
            self.logger.error(f"串口数据回调异常: {e}")
            self.stats["errors"] += 1
    
    def _processing_loop(self):
        """数据处理循环（解码原始数据）"""
        self.logger.info("串口数据处理线程启动")
        
        while self.running:
            try:
                # 从队列获取原始数据
                try:
                    raw_packet = self.raw_data_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # 解码数据
                decode_result = self.decoder.decode(
                    raw_data=raw_packet["raw_data"],
                    protocol=raw_packet.get("protocol_hint"),
                    hint=raw_packet.get("protocol_hint")
                )
                
                # 创建解码后的数据包
                decoded_packet = {
                    "source_port": raw_packet["source_port"],
                    "raw_data": raw_packet["raw_data"],
                    "decode_result": decode_result.to_dict(),
                    "timestamp": raw_packet["timestamp"],
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
                
                # 放入解码数据队列
                self.decoded_data_queue.put(decoded_packet, block=False)
                self.stats["data_decoded"] += 1
                
                # 添加到缓存
                self._add_to_cache(decoded_packet)
                
                self.logger.debug(
                    f"数据解码完成: {raw_packet['source_port']}, "
                    f"成功: {decode_result.success}"
                )
                
                # 标记任务完成
                self.raw_data_queue.task_done()
                
            except queue.Full:
                self.logger.warning("解码数据队列已满，丢弃数据")
                self.stats["errors"] += 1
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"数据处理循环异常: {e}")
                self.stats["errors"] += 1
                time.sleep(0.1)
        
        self.logger.info("串口数据处理线程停止")
    
    def _distribution_loop(self):
        """数据分发循环（将解码后的数据发送到各个目的地）"""
        self.logger.info("串口数据分发线程启动")
        
        while self.running:
            try:
                # 从队列获取解码后的数据
                try:
                    decoded_packet = self.decoded_data_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # 分发到各个目的地
                distribution_results = {}
                
                for destination, handlers in self.destination_handlers.items():
                    if not handlers:
                        continue
                    
                    destination_results = []
                    for handler in handlers:
                        try:
                            success = handler(decoded_packet)
                            destination_results.append({
                                "handler": handler.__name__ if hasattr(handler, '__name__') else str(handler),
                                "success": success,
                            })
                        except Exception as e:
                            self.logger.error(f"目的地处理器执行失败 {destination}: {e}")
                            destination_results.append({
                                "handler": handler.__name__ if hasattr(handler, '__name__') else str(handler),
                                "success": False,
                                "error": str(e),
                            })
                    
                    distribution_results[destination.value] = destination_results
                
                # 更新统计
                self.stats["data_distributed"] += 1
                
                self.logger.debug(
                    f"数据分发完成: {decoded_packet['source_port']}, "
                    f"目的地: {list(distribution_results.keys())}"
                )
                
                # 标记任务完成
                self.decoded_data_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"数据分发循环异常: {e}")
                self.stats["errors"] += 1
                time.sleep(0.1)
        
        self.logger.info("串口数据分发线程停止")
    
    def _add_to_cache(self, data_packet: Dict[str, Any]):
        """添加数据到缓存"""
        self.recent_data_cache.append(data_packet)
        
        # 限制缓存大小
        if len(self.recent_data_cache) > self.max_cache_size:
            self.recent_data_cache = self.recent_data_cache[-self.max_cache_size:]


# 单例实例
_serial_data_service_instance: Optional[SerialDataService] = None


def get_serial_data_service(config: Optional[Dict[str, Any]] = None) -> SerialDataService:
    """获取串口数据服务单例实例"""
    global _serial_data_service_instance
    if _serial_data_service_instance is None:
        _serial_data_service_instance = SerialDataService(config)
    return _serial_data_service_instance


def start_serial_data_service(config: Optional[Dict[str, Any]] = None) -> bool:
    """启动串口数据服务"""
    service = get_serial_data_service(config)
    return service.start()


def stop_serial_data_service() -> bool:
    """停止串口数据服务"""
    global _serial_data_service_instance
    if _serial_data_service_instance is not None:
        return _serial_data_service_instance.stop()
    return True


def receive_serial_raw_data(
    raw_data: bytes,
    source_port: Optional[str] = None,
    protocol_hint: Optional[str] = None
):
    """接收原始串口数据（便捷函数）"""
    service = get_serial_data_service()
    service.receive_raw_data(raw_data, source_port, protocol_hint)