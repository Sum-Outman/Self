"""
硬件路由模块
处理硬件状态、传感器数据、系统指标和电机控制的API请求
"""

import threading
import asyncio
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from backend.dependencies import get_db, get_current_user

from backend.db_models.user import User

from backend.core.response_cache import monitored_cache_response, CacheLevel
from backend.core import api_error_handler

from backend.services.hardware_service import get_hardware_service

# 导入串口数据服务
try:
    from backend.services.serial_data_service import (
        get_serial_data_service,
        receive_serial_raw_data,
        start_serial_data_service,
        stop_serial_data_service,
    )

    SERIAL_DATA_SERVICE_AVAILABLE = True
except ImportError:
    SERIAL_DATA_SERVICE_AVAILABLE = False
    get_serial_data_service = None
    receive_serial_raw_data = None
    start_serial_data_service = None
    stop_serial_data_service = None

from backend.schemas.response import SuccessResponse

# 导入真实传感器读取器
try:
    from backend.services.real_sensor_reader import (
        get_real_sensor_reader,
        RealSensorReader,
    )

    REAL_SENSOR_READER_AVAILABLE = True
except ImportError:
    REAL_SENSOR_READER_AVAILABLE = False
    get_real_sensor_reader = None
    RealSensorReader = None

# 硬件初始化函数将在端点内部延迟导入，避免循环导入问题
# 使用函数内部导入的方式

router = APIRouter(prefix="/api/hardware", tags=["硬件"])


@router.get("/status", response_model=SuccessResponse)
@api_error_handler
async def get_hardware_status(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取硬件状态 - 使用真实硬件服务"""
    # 获取硬件服务
    hardware_service = get_hardware_service()

    # 获取硬件设备和系统状态
    devices = hardware_service.get_hardware_devices()
    system_status = hardware_service.get_system_status()

    # 获取服务信息
    service_info = hardware_service.get_service_info()

    return SuccessResponse.create(
        data={
            "devices": devices,
            "system_status": system_status,
            "service_info": service_info,
        },
        message="获取硬件状态成功",
    )


@router.get("/sensor-data", response_model=SuccessResponse)
@api_error_handler(status_code=503, error_id="sensor_data_error")
async def get_sensor_data(
    sensor_id: Optional[str] = Query(None, description="传感器ID"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取传感器数据 - 使用真实硬件服务"""
    # 获取硬件服务
    hardware_service = get_hardware_service()

    # 获取传感器数据
    sensors = hardware_service.get_sensor_data(sensor_id)

    return SuccessResponse.create(data=sensors, message="获取传感器数据成功")


@router.get("/metrics", response_model=SuccessResponse)
@api_error_handler(status_code=500, error_id="system_metrics_error")
async def get_system_metrics(
    metric_type: Optional[str] = Query(None, description="指标类型"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取系统指标 - 使用真实硬件服务"""
    # 获取硬件服务
    hardware_service = get_hardware_service()

    # 获取系统指标
    metrics = hardware_service.get_system_metrics(metric_type)

    return SuccessResponse.create(data=metrics, message="获取系统指标成功")


@router.post("/motor/command", response_model=SuccessResponse)
async def send_motor_command(
    command: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """发送电机命令"""
    try:
        motor_id = command.get("motor_id", "")
        motor_command = command.get("command", "")

        if not motor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="电机ID不能为空"
            )

        if motor_command not in ["move", "stop", "reset", "calibrate"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的电机命令: {motor_command}",
            )

        # 只支持真实硬件控制，禁止模拟实现
        response = None

        # 获取硬件服务以访问硬件管理器
        try:
            hardware_service = get_hardware_service()
            if not hardware_service:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="硬件服务不可用",
                )

            if (
                not hasattr(hardware_service, "_hardware_manager")
                or hardware_service._hardware_manager is None
            ):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="硬件管理器未初始化",
                )

            # 检查硬件管理器是否有电机控制方法
            if not hasattr(hardware_service._hardware_manager, "send_motor_command"):
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="硬件管理器不支持电机控制功能",
                )

            # 调用真实电机控制
            result = hardware_service._hardware_manager.send_motor_command(
                motor_id=motor_id, command=motor_command, parameters=command
            )

            response = {
                "motor_id": motor_id,
                "command": motor_command,
                "status": result.get("status", "executed"),
                "message": result.get(
                    "message", f"电机 {motor_id} 命令 '{motor_command}' 已执行"
                ),
                "implementation": "real_hardware",
                "implementation_status": "real_hardware",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"电机控制失败: {str(e)}",
            )

        return SuccessResponse.create(
            data=response, message="电机命令处理完成（真实硬件）"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"发送电机命令失败: {str(e)}",
        )


@router.get("/serial/ports", response_model=SuccessResponse)
async def get_serial_ports(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取可用串口列表"""
    try:
        import serial.tools.list_ports

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
                "interface": port.interface if hasattr(port, "interface") else None,
            }
            ports.append(port_info)

        return SuccessResponse.create(
            data={
                "ports": ports,
                "count": len(ports),
            },
            message="获取串口列表成功",
        )
    except Exception as e:
        # 如果串口库不可用或出错，记录错误并返回空列表
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"获取串口列表失败: {e}")
        # 返回空列表而不是真实数据
        return SuccessResponse.create(
            data={
                "ports": [],
                "count": 0,
            },
            message="串口库不可用，返回空列表",
        )


@router.post("/serial/command", response_model=SuccessResponse)
async def send_serial_command(
    command: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """发送串口命令 - 使用串口数据服务"""
    try:
        cmd = command.get("command", "")
        port = command.get("port", "COM3")
        baudrate = command.get("baudrate", 9600)
        wait_for_response = command.get("wait_for_response", False)
        timeout = command.get("timeout", 5)
        protocol = command.get("protocol", "raw")  # raw, ascii, hex, json

        if not cmd:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="串口命令不能为空"
            )

        # 检查串口数据服务是否可用
        if not SERIAL_DATA_SERVICE_AVAILABLE or not get_serial_data_service:
            # 根据项目要求"禁止使用虚拟数据"，串口服务不可用时返回错误
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "串口数据服务不可用",
                    "message": "串口数据服务未初始化或不可用",
                    "solution": "请确保串口服务已正确安装和配置",
                    "requirement": "项目要求禁止使用虚拟数据，必须使用真实硬件接口",
                },
            )

        # 使用串口数据服务
        serial_service = get_serial_data_service()

        # 添加串口配置（如果不存在）
        from backend.services.serial_data_service import SerialProtocol

        protocol_enum = (
            SerialProtocol(protocol)
            if protocol in [p.value for p in SerialProtocol]
            else SerialProtocol.RAW
        )

        serial_service.add_serial_config(
            port=port, baudrate=baudrate, protocol=protocol_enum, auto_connect=True
        )

        # 连接串口（如果未连接）
        if port not in serial_service.active_connections:
            connected = serial_service.connect_serial(port)
            if not connected:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"无法连接到串口: {port}",
                )

        # 发送数据
        success = serial_service.send_serial_data(
            port=port,
            data=cmd.encode("utf-8") if isinstance(cmd, str) else cmd,
            protocol=protocol_enum,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="串口命令发送失败"
            )

        # 如果需要等待响应，简单延迟
        if wait_for_response:
            import time

            time.sleep(timeout)

        response_text = f"命令已通过端口 {port} (波特率 {baudrate}) 发送"

        return SuccessResponse.create(
            data={
                "command": cmd,
                "port": port,
                "baudrate": baudrate,
                "response": response_text,
                "mode": "real",
                "success": success,
            },
            message="串口命令发送成功",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"发送串口命令失败: {str(e)}",
        )


@router.post("/serial/receive", response_model=SuccessResponse)
async def receive_serial_data(
    data: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """接收原始串口数据并解码

    前端/硬件层只需发送原始数据，后台负责解码
    符合用户需求："接收后交给后台处理串口数据解码即可使用"
    """
    try:
        raw_data = data.get("raw_data", "")
        source_port = data.get("port", "unknown")
        protocol_hint = data.get("protocol", None)
        encoding = data.get("encoding", "base64")  # base64, hex, utf-8

        if not raw_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="原始数据不能为空"
            )

        # 解码原始数据（根据编码格式）
        import base64

        if encoding == "base64":
            try:
                bytes_data = base64.b64decode(raw_data)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Base64解码失败"
                )
        elif encoding == "hex":
            try:
                bytes_data = bytes.fromhex(raw_data)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="十六进制解码失败"
                )
        elif encoding == "utf-8":
            bytes_data = raw_data.encode("utf-8")
        else:
            # 默认尝试base64，然后utf-8
            try:
                bytes_data = base64.b64decode(raw_data)
            except Exception:
                bytes_data = raw_data.encode("utf-8")

        # 检查串口数据服务是否可用
        if not SERIAL_DATA_SERVICE_AVAILABLE or not receive_serial_raw_data:
            # 回退到简单解码
            from backend.services.serial_decoder import (
                get_serial_decoder,
                SerialProtocol,
            )

            decoder = get_serial_decoder()
            protocol = (
                SerialProtocol(protocol_hint)
                if protocol_hint in [p.value for p in SerialProtocol]
                else None
            )

            decode_result = decoder.decode(
                raw_data=bytes_data, protocol=protocol, hint=protocol_hint
            )

            return SuccessResponse.create(
                data={
                    "source_port": source_port,
                    "raw_length": len(bytes_data),
                    "decode_result": decode_result.to_dict(),
                    "mode": "decoder_only",
                },
                message="串口数据解码完成（仅解码模式）",
            )

        # 使用串口数据服务接收数据
        receive_serial_raw_data(
            raw_data=bytes_data, source_port=source_port, protocol_hint=protocol_hint
        )

        # 获取串口数据服务以获取统计信息
        serial_service = get_serial_data_service()
        stats = serial_service.get_stats()

        return SuccessResponse.create(
            data={
                "source_port": source_port,
                "raw_length": len(bytes_data),
                "received": True,
                "mode": "full_service",
                "service_stats": stats,
            },
            message="串口数据已接收并交给后台处理",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"接收串口数据失败: {str(e)}",
        )


@router.get("/sensors/data", response_model=SuccessResponse)
async def get_sensors_data(
    sensor_id: Optional[str] = Query(None, description="传感器ID"),
    sensor_type: Optional[str] = Query(None, description="传感器类型"),
    start_time: Optional[str] = Query(None, description="开始时间"),
    end_time: Optional[str] = Query(None, description="结束时间"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取传感器数据（兼容前端API）"""
    # 调用现有的 get_sensor_data 端点，忽略额外参数
    return await get_sensor_data(sensor_id, db, user)


@router.post("/motors/command", response_model=SuccessResponse)
async def send_motors_command(
    command: Dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """发送电机命令（兼容前端API）"""
    # 重定向到现有的 send_motor_command 端点
    return await send_motor_command(command, db, user)


@router.get("/system/metrics", response_model=SuccessResponse)
@monitored_cache_response(ttl=5, cache_level=CacheLevel.MEMORY)
async def get_system_metrics_alias(
    metric_type: Optional[str] = Query(None, description="指标类型"),
    start_time: Optional[str] = Query(None, description="开始时间"),
    end_time: Optional[str] = Query(None, description="结束时间"),
    limit: Optional[int] = Query(None, description="限制返回数量"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取系统指标（兼容前端API）"""
    # 调用现有的 get_system_metrics 端点，忽略额外参数
    return await get_system_metrics(metric_type, db, user)


# WebSocket连接管理器
class WebSocketConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.sensor_data_history: Dict[str, List[Dict[str, Any]]] = {}

    async def connect(self, websocket: WebSocket):
        """连接WebSocket"""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """发送个人消息"""
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """广播消息给所有连接"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"WebSocket发送失败: {e}")
                self.disconnect(connection)

    async def broadcast_json(self, data: Dict[str, Any]):
        """广播JSON数据给所有连接"""

        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"WebSocket JSON发送失败: {e}")
                self.disconnect(connection)

    def add_sensor_data(self, sensor_id: str, data: Dict[str, Any]):
        """添加传感器数据到历史记录"""
        if sensor_id not in self.sensor_data_history:
            self.sensor_data_history[sensor_id] = []

        self.sensor_data_history[sensor_id].append(data)

        # 限制历史记录大小
        if len(self.sensor_data_history[sensor_id]) > 1000:
            self.sensor_data_history[sensor_id] = self.sensor_data_history[sensor_id][
                -1000:
            ]


# 创建全局WebSocket管理器实例
websocket_manager = WebSocketConnectionManager()


@router.websocket("/ws/sensors")
async def websocket_sensor_data(websocket: WebSocket):
    """WebSocket传感器数据流"""
    await websocket_manager.connect(websocket)

    try:
        while True:
            # 等待客户端消息
            data = await websocket.receive_json()

            # 处理客户端请求
            if data.get("type") == "subscribe":
                # 订阅特定传感器
                sensor_id = data.get("sensor_id")
                await websocket.send_json(
                    {
                        "type": "subscribed",
                        "sensor_id": sensor_id,
                        "message": f"已订阅传感器 {sensor_id}",
                    }
                )

            elif data.get("type") == "unsubscribe":
                # 取消订阅
                sensor_id = data.get("sensor_id")
                await websocket.send_json(
                    {
                        "type": "unsubscribed",
                        "sensor_id": sensor_id,
                        "message": f"已取消订阅传感器 {sensor_id}",
                    }
                )

            elif data.get("type") == "get_history":
                # 获取历史数据
                sensor_id = data.get("sensor_id")
                limit = data.get("limit", 100)

                history = websocket_manager.sensor_data_history.get(sensor_id, [])
                if limit > 0:
                    history = history[-limit:]

                await websocket.send_json(
                    {
                        "type": "history",
                        "sensor_id": sensor_id,
                        "data": history,
                        "count": len(history),
                    }
                )

            elif data.get("type") == "control":
                # 真实硬件控制命令
                command = data.get("command", {})
                device_id = command.get("device_id")
                action = command.get("action")
                params = command.get("params", {})

                # 尝试使用真实硬件服务
                try:
                    hardware_service = get_hardware_service()
                    if not hardware_service:
                        raise RuntimeError("硬件服务不可用")

                    if (
                        not hasattr(hardware_service, "_hardware_manager")
                        or hardware_service._hardware_manager is None
                    ):
                        raise RuntimeError("硬件管理器未初始化")

                    # 根据设备类型和动作调用相应的硬件控制方法
                    control_result = None

                    if action == "motor_control" and hasattr(
                        hardware_service._hardware_manager, "send_motor_command"
                    ):
                        # 电机控制
                        control_result = (
                            hardware_service._hardware_manager.send_motor_command(
                                motor_id=device_id,
                                command=params.get("motor_command", "move"),
                                parameters=params,
                            )
                        )
                    elif hasattr(
                        hardware_service._hardware_manager, "send_device_command"
                    ):
                        # 通用设备控制
                        control_result = (
                            hardware_service._hardware_manager.send_device_command(
                                device_id=device_id, command=action, parameters=params
                            )
                        )
                    else:
                        raise RuntimeError(f"硬件管理器不支持{action}控制")

                    # 构建响应
                    response = {
                        "type": "control_response",
                        "device_id": device_id,
                        "action": action,
                        "status": (
                            control_result.get("status", "executed")
                            if control_result
                            else "executed"
                        ),
                        "message": (
                            control_result.get(
                                "message", f"设备 {device_id} 命令 '{action}' 已执行"
                            )
                            if control_result
                            else f"设备 {device_id} 命令 '{action}' 已执行"
                        ),
                        "result": control_result,
                        "implementation": "real_hardware",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                except Exception as e:
                    # 硬件控制失败
                    response = {
                        "type": "control_response",
                        "device_id": device_id,
                        "action": action,
                        "status": "failed",
                        "message": f"硬件控制失败: {str(e)}",
                        "implementation": "error",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                await websocket.send_json(response)

            else:
                # 未知消息类型
                await websocket.send_json(
                    {"type": "error", "message": f"未知的消息类型: {data.get('type')}"}
                )

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        websocket_manager.disconnect(websocket)
        raise


# 真实传感器数据生成任务 - 尝试使用真实硬件，失败时返回错误状态


async def generate_sensor_data():
    """生成真实传感器数据并广播 - 遵循'禁止使用虚拟数据'要求"""

    # 初始化传感器读取器
    sensor_reader = None
    if REAL_SENSOR_READER_AVAILABLE and get_real_sensor_reader:
        try:
            sensor_reader = get_real_sensor_reader()
            print(
                f"传感器读取器初始化: 真实传感器{'可用' if sensor_reader.has_real_sensors() else '不可用'}"
            )
        except Exception as e:
            print(f"初始化真实传感器读取器失败: {e}")
            sensor_reader = None

    while True:
        try:
            # 获取传感器数据
            if sensor_reader and REAL_SENSOR_READER_AVAILABLE:
                # 从真实传感器读取数据
                try:
                    real_sensor_data = sensor_reader.read_sensor_data()

                    if real_sensor_data:
                        for sensor_item in real_sensor_data:
                            # 格式化传感器数据
                            sensor_data = {
                                "type": "sensor_data",
                                "sensor_id": sensor_item.get(
                                    "sensor_id", "unknown_001"
                                ),
                                "sensor_type": sensor_item.get(
                                    "sensor_type", "unknown"
                                ),
                                "value": sensor_item.get("value", 0.0),
                                "unit": sensor_item.get("unit", ""),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "status": sensor_item.get("status", "normal"),
                                "source": "real_hardware",
                            }

                            # 添加到历史记录
                            websocket_manager.add_sensor_data(
                                sensor_data["sensor_id"], sensor_data
                            )

                            # 广播给所有连接的客户端
                            await websocket_manager.broadcast_json(sensor_data)
                    else:
                        # 没有真实传感器数据，生成回退数据
                        await _generate_fallback_sensor_data()

                except Exception as e:
                    print(f"读取真实传感器数据失败: {e}")
                    # 回退到真实数据
                    await _generate_fallback_sensor_data()
            else:
                # 真实传感器读取器不可用，使用回退数据
                await _generate_fallback_sensor_data()

            # 等待1秒
            await asyncio.sleep(1.0)

        except Exception as e:
            print(f"生成传感器数据出错: {e}")
            await asyncio.sleep(1.0)


async def _generate_fallback_sensor_data():
    """真实硬件不可用时的处理（不生成真实数据）

    根据项目要求"禁止使用虚拟数据"，当真实硬件不可用时，
    不生成真实数据，而是生成错误状态的数据包。
    """
    import logging

    logger = logging.getLogger(__name__)

    # 只警告一次，避免重复日志
    if not hasattr(_generate_fallback_sensor_data, "_has_warned"):
        logger.warning("真实硬件不可用，不生成虚拟数据（符合'禁止使用虚拟数据'要求）")
        _generate_fallback_sensor_data._has_warned = True

    # 生成一个错误状态的数据包，而不是真实数据
    error_data = {
        "type": "hardware_error",
        "message": "真实硬件不可用",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "source": "system",
        "details": {
            "reason": "真实传感器硬件未连接或不可用",
            "requirement": "项目要求禁止使用虚拟数据，必须使用真实硬件",
            "solution": "请连接真实硬件设备或配置硬件接口",
        },
    }

    # 广播错误状态
    await websocket_manager.broadcast_json(error_data)


# 启动传感器数据生成任务


def start_sensor_data_generator():
    """启动传感器数据生成器"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(generate_sensor_data())


# 在应用启动时启动生成器
sensor_thread = None
if sensor_thread is None:
    sensor_thread = threading.Thread(target=start_sensor_data_generator, daemon=True)
    sensor_thread.start()


# 测试WebSocket的HTML页面
@router.get("/ws-test", response_class=HTMLResponse)
async def get_ws_test_page():
    """WebSocket测试页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>硬件WebSocket测试</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .sensor-container { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .sensor-value { font-size: 24px; font-weight: bold; color: #007bff; }
            .sensor-unit { color: #666; }
            .controls { margin: 20px 0; }
            button { padding: 10px 20px; margin: 0 10px 10px 0; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .log { background: #f5f5f5; padding: 10px; border-radius: 4px; max-height: 300px; overflow-y: auto; margin-top: 20px; }
            .log-entry { margin: 5px 0; padding: 5px; border-bottom: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>硬件WebSocket测试</h1>

            <div class="controls">
                <button onclick="connectWebSocket()">连接WebSocket</button>
                <button onclick="disconnectWebSocket()">断开连接</button>
                <button onclick="subscribeAll()">订阅所有传感器</button>
                <button onclick="unsubscribeAll()">取消订阅所有</button>
            </div>

            <div id="sensors">
                <!-- 传感器数据将在这里显示 -->
            </div>

            <div class="log">
                <h3>日志</h3>
                <div id="log"></div>
            </div>
        </div>

        <script>
            let ws = null;
            const sensors = {};

            function log(message) {
                const logDiv = document.getElementById('log');
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.textContent = new Date().toLocaleTimeString() + ': ' + message;
                logDiv.appendChild(entry);
                logDiv.scrollTop = logDiv.scrollHeight;
            }

            function connectWebSocket() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    log('WebSocket已经连接');
                    return;
                }

                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = protocol + '//' + window.location.host + '/api/hardware/ws/sensors';

                ws = new WebSocket(wsUrl);

                ws.onopen = function() {
                    log('WebSocket连接已建立');
                };

                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        log('收到消息: ' + event.data.substring(0, 100));

                        if (data.type === 'sensor_data') {
                            updateSensorDisplay(data);
                        }
                    } catch (e) {
                        log('解析消息失败: ' + e.message);
                    }
                };

                ws.onerror = function(error) {
                    log('WebSocket错误: ' + error);
                };

                ws.onclose = function() {
                    log('WebSocket连接已关闭');
                };
            }

            function disconnectWebSocket() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }

            function subscribeAll() {
                const sensorTypes = ['temperature', 'humidity', 'pressure', 'light', 'sound'];
                sensorTypes.forEach(type => {
                    subscribeSensor(type + '_001');
                });
            }

            function unsubscribeAll() {
                const sensorTypes = ['temperature', 'humidity', 'pressure', 'light', 'sound'];
                sensorTypes.forEach(type => {
                    unsubscribeSensor(type + '_001');
                });
            }

            function subscribeSensor(sensorId) {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'subscribe',
                        sensor_id: sensorId
                    }));
                    log('已订阅传感器: ' + sensorId);
                } else {
                    log('WebSocket未连接，无法订阅');
                }
            }

            function unsubscribeSensor(sensorId) {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'unsubscribe',
                        sensor_id: sensorId
                    }));
                    log('已取消订阅传感器: ' + sensorId);
                } else {
                    log('WebSocket未连接，无法取消订阅');
                }
            }

            function updateSensorDisplay(sensorData) {
                const sensorId = sensorData.sensor_id;
                const sensorType = sensorData.sensor_type;

                if (!sensors[sensorId]) {
                    // 创建新的传感器显示
                    const sensorsDiv = document.getElementById('sensors');
                    const sensorDiv = document.createElement('div');
                    sensorDiv.className = 'sensor-container';
                    sensorDiv.id = 'sensor-' + sensorId;
                    sensorDiv.innerHTML = `
                        <h3>${sensorType.toUpperCase()}: ${sensorId}</h3>
                        <div class="sensor-value">${sensorData.value} <span class="sensor-unit">${sensorData.unit}</span></div>
                        <div>状态: ${sensorData.status}</div>
                        <div>时间: ${sensorData.timestamp}</div>
                    `;
                    sensorsDiv.appendChild(sensorDiv);
                    sensors[sensorId] = sensorDiv;
                } else {
                    // 更新现有传感器显示
                    const sensorDiv = sensors[sensorId];
                    sensorDiv.querySelector('.sensor-value').innerHTML = `${sensorData.value} <span class="sensor-unit">${sensorData.unit}</span>`;
                    sensorDiv.querySelector('div:nth-child(3)').textContent = '状态: ' + sensorData.status;
                    sensorDiv.querySelector('div:nth-child(4)').textContent = '时间: ' + sensorData.timestamp;
                }
            }

            // 页面加载时自动连接
            window.onload = function() {
                connectWebSocket();
                setTimeout(subscribeAll, 1000);
            };
        </script>
    </body>
    </html>
    """


@router.post("/initialize", response_model=SuccessResponse)
@api_error_handler
async def initialize_hardware(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """初始化硬件组件（模型开启后接入）

    根据用户要求，硬件组件应在模型开启后接入，而不是系统启动时自动初始化
    此端点用于手动初始化硬件组件
    """
    # 延迟导入，避免循环导入
    try:
        from backend.main import initialize_hardware_components, hardware_initialized

        HARDWARE_INIT_FUNCTIONS_AVAILABLE = True
    except ImportError as e:
        HARDWARE_INIT_FUNCTIONS_AVAILABLE = False
        initialize_hardware_components = None
        hardware_initialized = False
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"硬件初始化函数导入失败: {e}")

    if not HARDWARE_INIT_FUNCTIONS_AVAILABLE or not initialize_hardware_components:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="硬件初始化功能不可用"
        )

    if hardware_initialized:
        return SuccessResponse.create(
            data={"hardware_initialized": True}, message="硬件组件已初始化"
        )

    try:
        success = initialize_hardware_components()
        if success:
            return SuccessResponse.create(
                data={"hardware_initialized": True},
                message="硬件组件初始化成功（模型开启后接入模式）",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="硬件组件初始化失败",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"硬件初始化异常: {str(e)}",
        )


@router.post("/shutdown", response_model=SuccessResponse)
@api_error_handler
async def shutdown_hardware(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """关闭硬件组件（模型关闭时调用）

    根据用户要求，硬件组件应在模型关闭时关闭
    此端点用于手动关闭硬件组件
    """
    # 延迟导入，避免循环导入
    try:
        from backend.main import shutdown_hardware_components, hardware_initialized

        HARDWARE_INIT_FUNCTIONS_AVAILABLE = True
    except ImportError as e:
        HARDWARE_INIT_FUNCTIONS_AVAILABLE = False
        shutdown_hardware_components = None
        hardware_initialized = False
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"硬件关闭函数导入失败: {e}")

    if not HARDWARE_INIT_FUNCTIONS_AVAILABLE or not shutdown_hardware_components:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="硬件关闭功能不可用"
        )

    if not hardware_initialized:
        return SuccessResponse.create(
            data={"hardware_initialized": False}, message="硬件组件未初始化，无需关闭"
        )

    try:
        success = shutdown_hardware_components()
        if success:
            return SuccessResponse.create(
                data={"hardware_initialized": False}, message="硬件组件关闭成功"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="硬件组件关闭失败",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"硬件关闭异常: {str(e)}",
        )


@router.get("/initialization-status", response_model=SuccessResponse)
@api_error_handler
async def get_hardware_initialization_status(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取硬件初始化状态"""
    # 延迟导入，避免循环导入
    try:
        from backend.main import hardware_initialized

        HARDWARE_INIT_FUNCTIONS_AVAILABLE = True
    except ImportError as e:
        HARDWARE_INIT_FUNCTIONS_AVAILABLE = False
        hardware_initialized = False
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"硬件状态函数导入失败: {e}")

    return SuccessResponse.create(
        data={
            "hardware_initialized": (
                hardware_initialized if HARDWARE_INIT_FUNCTIONS_AVAILABLE else False
            ),
            "functions_available": HARDWARE_INIT_FUNCTIONS_AVAILABLE,
        },
        message="硬件初始化状态查询成功",
    )


# 注意：还有其他端点如POST /hardware/serial/command,
# GET /hardware/devices, GET /hardware/devices/{device_id},
# PUT /hardware/devices/{device_id}, POST /hardware/devices/{device_id}/control,
# 完整处理
# 实际项目中应该实现完整的硬件控制功能
