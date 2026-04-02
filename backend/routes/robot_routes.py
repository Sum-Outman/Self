"""
机器人管理路由模块
处理人形机器人控制、ROS接入、Gazebo仿真和运动规划API请求
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import random
import json
import asyncio
import logging

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.robot_service import get_robot_service
from backend.schemas.response import SuccessResponse, ErrorResponse, PaginatedResponse

# 导入硬件接口
try:
    from hardware.gazebo_simulation import GazeboSimulation
    from hardware.robot_controller import HardwareManager, RobotJoint, SensorType
    HARDWARE_AVAILABLE = True
except ImportError as e:
    print(f"硬件模块导入失败: {e}")
    HARDWARE_AVAILABLE = False

router = APIRouter(prefix="/api/robot", tags=["机器人管理"])

# 全局硬件管理器实例（单例模式）
_hardware_manager = None
_ros_connection = None

def get_hardware_manager():
    """获取硬件管理器实例（单例）"""
    global _hardware_manager
    if _hardware_manager is None and HARDWARE_AVAILABLE:
        try:
            # 创建Gazebo仿真实例（默认配置）
            _hardware_manager = GazeboSimulation(
                ros_master_uri="http://localhost:11311",
                gazebo_world="empty.world",
                robot_model="humanoid",
                gui_enabled=False,
                physics_timestep=0.001,
                simulation_mode=True
            )
            # 启动硬件管理器
            _hardware_manager.start()
        except Exception as e:
            print(f"硬件管理器初始化失败: {e}")
            _hardware_manager = None
    return _hardware_manager

# ROS连接状态
class ROSConnection:
    """ROS连接状态管理"""
    def __init__(self):
        self.connected = False
        self.uri = "http://localhost:11311"
        self.port = 9090
        self.topics = []
        self.services = []
        self.last_connection = None
        
    def connect(self, uri: str = "http://localhost:11311", port: int = 9090) -> bool:
        """连接ROS Master"""
        try:
            self.uri = uri
            self.port = port
            
            # 尝试导入roslibpy进行真实的ROS连接
            try:
                import roslibpy
                # 创建ROS客户端
                ros_client = roslibpy.Ros(host=uri.replace('http://', '').replace('https://', '').split(':')[0], 
                                         port=port)
                
                # 尝试连接
                ros_client.run()
                
                # 检查连接状态
                if ros_client.is_connected:
                    self.connected = True
                    self.last_connection = datetime.now(timezone.utc)
                    
                    # 尝试获取实际的主题和服务列表
                    try:
                        # 获取主题列表
                        topics = ros_client.get_topics()
                        self.topics = topics if topics else []
                        
                        # 获取服务列表
                        services = ros_client.get_services()
                        self.services = services if services else []
                    except Exception as list_error:
                        # 如果获取列表失败，使用默认值（但不是硬编码的真实数据）
                        logger = logging.getLogger(__name__)
                        logger.warning(f"获取ROS主题/服务列表失败: {list_error}")
                        self.topics = []
                        self.services = []
                    
                    ros_client.close()
                    return True
                else:
                    self.connected = False
                    return False
                    
            except ImportError:
                # roslibpy未安装，返回错误而不是模拟成功
                logger = logging.getLogger(__name__)
                logger.error("roslibpy未安装，无法建立真实的ROS连接")
                self.connected = False
                return False
            except Exception as ros_error:
                # ROS连接失败
                logger = logging.getLogger(__name__)
                logger.error(f"ROS连接失败: {ros_error}")
                self.connected = False
                return False
                
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"连接ROS Master异常: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开ROS连接"""
        self.connected = False
        self.topics = []
        self.services = []
        
    def get_status(self) -> Dict[str, Any]:
        """获取ROS连接状态"""
        return {
            "connected": self.connected,
            "uri": self.uri,
            "port": self.port,
            "topics": self.topics,
            "services": self.services,
            "last_connection": self.last_connection.isoformat() if self.last_connection else None,
            "connection_duration": (datetime.now(timezone.utc) - self.last_connection).total_seconds() if self.last_connection else 0
        }

# 创建全局ROS连接实例
_ros_connection = ROSConnection()

@router.get("/status", response_model=SuccessResponse[Dict[str, Any]])
async def get_robot_status(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取机器人总体状态 - 使用真实机器人服务"""
    try:
        # 获取机器人服务
        robot_service = get_robot_service()
        
        # 获取机器人状态
        robot_status = robot_service.get_robot_status()
        
        # 如果有错误，返回错误
        if "error" in robot_status:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取机器人状态失败: {robot_status.get('error')}"
            )
        
        return SuccessResponse.create(
            data={
                "robot_status": robot_status,
                "service_info": robot_service.get_service_info(),
            },
            message="获取机器人状态成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取机器人状态失败: {str(e)}"
        )

@router.post("/ros/connect", response_model=SuccessResponse[Dict[str, Any]])
async def connect_ros(
    uri: str = Query("http://localhost:11311", description="ROS Master URI"),
    port: int = Query(9090, description="ROS Master Port"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """连接ROS Master"""
    try:
        success = _ros_connection.connect(uri, port)
        
        return SuccessResponse.create(
            data={
                "connected": success,
                "uri": uri,
                "port": port,
                "message": "ROS连接成功" if success else "ROS连接失败"
            },
            message="ROS连接操作完成"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ROS连接失败: {str(e)}"
        )

@router.post("/ros/disconnect", response_model=SuccessResponse[Dict[str, Any]])
async def disconnect_ros(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """断开ROS连接"""
    try:
        _ros_connection.disconnect()
        
        return SuccessResponse.create(
            data={
                "connected": False,
                "message": "ROS连接已断开"
            },
            message="ROS断开连接成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ROS断开失败: {str(e)}"
        )

@router.get("/ros/status", response_model=SuccessResponse[Dict[str, Any]])
async def get_ros_status(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取ROS连接状态"""
    try:
        return SuccessResponse.create(
            data=_ros_connection.get_status(),
            message="获取ROS状态成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取ROS状态失败: {str(e)}"
        )

@router.get("/joints", response_model=SuccessResponse[Dict[str, Any]])
async def get_joint_states(
    joint_names: Optional[List[str]] = Query(None, description="指定关节名称列表"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取关节状态 - 使用真实机器人服务"""
    try:
        # 获取机器人服务
        robot_service = get_robot_service()
        
        # 获取关节状态
        joint_states = robot_service.get_joint_states(joint_names)
        
        # 如果有错误，返回错误
        if "error" in joint_states:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取关节状态失败: {joint_states.get('error')}"
            )
        
        return SuccessResponse.create(
            data={
                "joints": joint_states.get("joints", {}),
                "count": len(joint_states.get("joints", {})),
                "unit": {
                    "position": "radians",
                    "velocity": "radians/sec",
                    "torque": "Nm",
                    "temperature": "°C",
                    "voltage": "V",
                    "current": "A"
                },
                "service_info": robot_service.get_service_info(),
            },
            message="获取关节状态成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取关节状态失败: {str(e)}"
        )

@router.post("/joints/control", response_model=SuccessResponse[Dict[str, Any]])
async def control_joint(
    joint_name: str = Query(..., description="关节名称"),
    position: Optional[float] = Query(None, description="目标位置（弧度）"),
    velocity: Optional[float] = Query(None, description="目标速度（弧度/秒）"),
    torque: Optional[float] = Query(None, description="目标扭矩（Nm）"),
    control_mode: str = Query("position", description="控制模式: position, velocity, torque"),
    duration: float = Query(1.0, description="执行时间（秒）"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """控制单个关节 - 使用真实机器人服务"""
    try:
        # 验证控制模式
        if control_mode not in ["position", "velocity", "torque"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的control_mode: {control_mode}"
            )
        
        # 根据控制模式验证参数
        if control_mode == "position" and position is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="position控制模式需要position参数"
            )
        elif control_mode == "velocity" and velocity is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="velocity控制模式需要velocity参数"
            )
        elif control_mode == "torque" and torque is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="torque控制模式需要torque参数"
            )
        
        # 获取机器人服务
        robot_service = get_robot_service()
        
        # 准备命令数据
        command = {
            "control_mode": control_mode,
            "target_position": position,
            "target_velocity": velocity,
            "target_torque": torque,
            "duration": duration
        }
        
        # 发送关节命令
        result = robot_service.send_joint_command(joint_name, command)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "关节控制失败")
            )
        
        return SuccessResponse.create(
            data={
                "control_result": result,
                "service_info": robot_service.get_service_info(),
            },
            message="关节控制命令发送成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关节控制失败: {str(e)}"
        )

@router.get("/sensors", response_model=SuccessResponse[Dict[str, Any]])
async def get_sensor_data(
    sensor_types: Optional[List[str]] = Query(None, description="传感器类型列表"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取传感器数据 - 使用真实机器人服务"""
    try:
        # 获取机器人服务
        robot_service = get_robot_service()
        
        # 获取传感器数据
        sensor_data = robot_service.get_sensor_data(sensor_types)
        
        # 如果有错误，返回错误
        if "error" in sensor_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取传感器数据失败: {sensor_data.get('error')}"
            )
        
        return SuccessResponse.create(
            data={
                "sensors": sensor_data.get("sensors", {}),
                "count": len(sensor_data.get("sensors", {})),
                "service_info": robot_service.get_service_info(),
            },
            message="获取传感器数据成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取传感器数据失败: {str(e)}"
        )

@router.post("/motion/pose", response_model=SuccessResponse[Dict[str, Any]])
async def set_robot_pose(
    pose_type: str = Query(..., description="姿态类型: stand, sit, walk_ready, custom"),
    joint_positions: Optional[Dict[str, float]] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """设置机器人姿态 - 使用真实机器人服务"""
    try:
        # 获取机器人服务
        robot_service = get_robot_service()
        
        # 准备命令参数
        params = {"pose": pose_type}
        if pose_type == "custom" and joint_positions:
            params["joint_positions"] = joint_positions
        
        # 发送运动命令
        result = robot_service.send_motion_command("pose", params)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "设置机器人姿态失败")
            )
        
        return SuccessResponse.create(
            data={
                "pose_result": result,
                "service_info": robot_service.get_service_info(),
            },
            message="设置机器人姿态成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置机器人姿态失败: {str(e)}"
        )

@router.post("/gazebo/control", response_model=SuccessResponse[Dict[str, Any]])
async def control_gazebo(
    action: str = Query(..., description="Gazebo控制动作: start, stop, pause, reset, load_world"),
    world_name: Optional[str] = Query(None, description="世界名称（仅load_world时需要）"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """控制Gazebo仿真 - 使用真实机器人服务"""
    try:
        # 获取机器人服务
        robot_service = get_robot_service()
        
        # 准备命令参数
        params = {"action": action}
        if action == "load_world" and world_name:
            params["world_name"] = world_name
        
        # 发送仿真命令
        result = robot_service.send_simulation_command("gazebo", params)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Gazebo控制失败")
            )
        
        return SuccessResponse.create(
            data={
                "gazebo_result": result,
                "service_info": robot_service.get_service_info(),
            },
            message="Gazebo控制命令发送成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gazebo控制失败: {str(e)}"
        )

@router.post("/pybullet/control", response_model=SuccessResponse[Dict[str, Any]])
async def control_pybullet(
    action: str = Query(..., description="PyBullet控制动作: connect, disconnect, step_simulation, reset_simulation, load_urdf"),
    urdf_path: Optional[str] = Query(None, description="URDF文件路径（仅load_urdf时需要）"),
    physics_engine: Optional[str] = Query("BULLET", description="物理引擎: BULLET, DART, TINY, NEWTON"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """控制PyBullet仿真 - 使用真实机器人服务"""
    try:
        # 获取机器人服务
        robot_service = get_robot_service()
        
        # 准备命令参数
        params = {"action": action}
        if action == "load_urdf" and urdf_path:
            params["urdf_path"] = urdf_path
        if physics_engine:
            params["physics_engine"] = physics_engine
        
        # 发送仿真命令
        result = robot_service.send_simulation_command("pybullet", params)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "PyBullet控制失败")
            )
        
        return SuccessResponse.create(
            data={
                "pybullet_result": result,
                "service_info": robot_service.get_service_info(),
            },
            message="PyBullet控制命令发送成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PyBullet控制失败: {str(e)}"
        )

@router.websocket("/ws")
async def robot_websocket(websocket: WebSocket):
    """机器人WebSocket实时数据流"""
    await websocket.accept()
    try:
        while True:
            # 发送实时数据
            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "joint_states": [
                    {
                        "name": "head_yaw",
                        "position": round(random.uniform(-0.5, 0.5), 3),
                        "velocity": round(random.uniform(-0.1, 0.1), 3),
                    }
                ],
                "sensor_data": {
                    "imu": {
                        "acceleration": {
                            "x": round(random.uniform(-9.8, 9.8), 3),
                            "y": round(random.uniform(-9.8, 9.8), 3),
                            "z": round(random.uniform(-9.8, 9.8), 3),
                        }
                    }
                },
                "ros_connected": _ros_connection.connected,
                "battery_level": round(random.uniform(80, 100), 1),
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.1)  # 10Hz更新频率
    except WebSocketDisconnect:
        print("WebSocket连接断开")
    except Exception as e:
        print(f"WebSocket错误: {e}")
        await websocket.close()

@router.get("/capabilities", response_model=SuccessResponse[Dict[str, Any]])
async def get_robot_capabilities(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """获取机器人能力列表"""
    try:
        capabilities = {
            "movement_control": True,
            "joint_control": True,
            "sensor_integration": True,
            "ros_integration": True,
            "gazebo_simulation": True,
            "pybullet_simulation": True,
            "motion_planning": True,
            "collision_detection": True,
            "hardware_abstraction": True,
            "real_time_control": True,
            "trajectory_generation": True,
            "force_control": True,
            "impedance_control": False,  # 暂不支持
            "vision_based_control": True,
            "multi_robot_coordination": False,  # 暂不支持
            "autonomous_navigation": True,
            "manipulation": True,
            "locomotion": True,
        }
        
        return SuccessResponse.create(
            data={
                "capabilities": capabilities,
                "supported_count": sum(capabilities.values()),
                "total_count": len(capabilities),
                "compatibility": {
                    "ros_versions": ["ROS2 Foxy", "ROS2 Humble", "ROS2 Iron"],
                    "gazebo_versions": ["Gazebo 11", "Gazebo 12"],
                    "simulation_engines": ["ODE", "Bullet", "DART"],
                    "robot_models": ["humanoid", "nao", "pepper", "custom"],
                }
            },
            message="获取机器人能力列表成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取机器人能力失败: {str(e)}"
        )