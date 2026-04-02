"""
机器人控制路由模块
处理机器人运动控制、轨迹规划和状态监控API请求
基于统一的RobotService重构
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import logging

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.robot_service import get_robot_service

router = APIRouter(prefix="/api/robot/control", tags=["机器人控制"])

logger = logging.getLogger(__name__)


@router.get("/status", summary="获取机器人控制状态")
async def get_robot_control_status():
    """获取机器人控制服务状态"""
    try:
        control_service = get_robot_service()
        service_info = control_service.get_service_info()
        
        return {
            "success": True,
            "status": "running",
            "data": service_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取机器人控制状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取机器人控制状态失败: {str(e)}"
        )


@router.get("/state", summary="获取机器人完整状态")
async def get_robot_full_state():
    """获取机器人完整状态信息"""
    try:
        control_service = get_robot_service()
        robot_state = control_service.get_robot_state()
        
        return {
            "success": True,
            "state": robot_state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取机器人状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取机器人状态失败: {str(e)}"
        )


@router.get("/joints", summary="获取关节状态")
async def get_robot_joints():
    """获取机器人关节状态"""
    try:
        control_service = get_robot_service()
        joint_states = control_service.get_joint_states()
        
        return {
            "success": True,
            "joints": joint_states,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取关节状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取关节状态失败: {str(e)}"
        )


@router.get("/sensors", summary="获取传感器数据")
async def get_robot_sensors():
    """获取机器人传感器数据"""
    try:
        control_service = get_robot_service()
        sensor_data = control_service.get_sensor_data()
        
        return {
            "success": True,
            "sensors": sensor_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取传感器数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取传感器数据失败: {str(e)}"
        )


@router.get("/system", summary="获取系统状态")
async def get_robot_system_status():
    """获取机器人系统状态"""
    try:
        control_service = get_robot_service()
        system_status = control_service.get_system_status()
        
        return {
            "success": True,
            "system": system_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统状态失败: {str(e)}"
        )


@router.post("/mode", summary="设置控制模式")
async def set_control_mode(
    mode: str = Query(..., description="控制模式: position, velocity, torque, trajectory")
):
    """设置机器人控制模式"""
    try:
        control_service = get_robot_service()
        result = control_service.set_control_mode(mode)
        
        if result.get("success", False):
            return {
                "success": True,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "设置控制模式失败")
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置控制模式失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置控制模式失败: {str(e)}"
        )


@router.post("/joints/positions", summary="设置关节位置")
async def set_joint_positions(
    positions: List[float] = Body(..., description="关节位置列表（弧度）")
):
    """设置机器人关节位置（位置控制）"""
    try:
        control_service = get_robot_service()
        result = control_service.set_joint_positions(positions)
        
        if result.get("success", False):
            return {
                "success": True,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "设置关节位置失败")
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置关节位置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置关节位置失败: {str(e)}"
        )


@router.post("/joints/velocities", summary="设置关节速度")
async def set_joint_velocities(
    velocities: List[float] = Body(..., description="关节速度列表（弧度/秒）")
):
    """设置机器人关节速度（速度控制）"""
    try:
        control_service = get_robot_service()
        result = control_service.set_joint_velocities(velocities)
        
        if result.get("success", False):
            return {
                "success": True,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "设置关节速度失败")
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置关节速度失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置关节速度失败: {str(e)}"
        )


@router.post("/trajectory", summary="添加运动轨迹")
async def add_trajectory(
    trajectory: Dict[str, Any] = Body(..., description="运动轨迹定义")
):
    """添加机器人运动轨迹"""
    try:
        control_service = get_robot_service()
        result = control_service.add_trajectory(trajectory)
        
        if result.get("success", False):
            return {
                "success": True,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "添加轨迹失败")
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加轨迹失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加轨迹失败: {str(e)}"
        )


@router.delete("/trajectory", summary="清除所有轨迹")
async def clear_trajectories():
    """清除机器人所有运动轨迹"""
    try:
        control_service = get_robot_service()
        result = control_service.clear_trajectories()
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"清除轨迹失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清除轨迹失败: {str(e)}"
        )


@router.get("/trajectory/queue", summary="获取轨迹队列状态")
async def get_trajectory_queue():
    """获取轨迹队列状态"""
    try:
        control_service = get_robot_service()
        service_info = control_service.get_service_info()
        
        queue_info = {
            "queue_size": service_info.get("trajectory_queue_size", 0),
            "has_trajectory": service_info.get("trajectory_queue_size", 0) > 0,
            "control_mode": service_info.get("control_mode", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return {
            "success": True,
            "queue": queue_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取轨迹队列状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取轨迹队列状态失败: {str(e)}"
        )


@router.post("/emergency/stop", summary="紧急停止")
async def emergency_stop():
    """执行机器人紧急停止"""
    try:
        control_service = get_robot_service()
        
        # 停止所有运动
        result_clear = control_service.clear_trajectories()
        
        # 设置关节速度为0（紧急停止）
        joint_count = 28  # 默认关节数量
        zero_velocities = [0.0] * joint_count
        result_velocities = control_service.set_joint_velocities(zero_velocities)
        
        # 切换到位置控制模式
        result_mode = control_service.set_control_mode("position")
        
        return {
            "success": True,
            "actions": {
                "trajectories_cleared": result_clear.get("success", False),
                "velocities_zeroed": result_velocities.get("success", False),
                "mode_changed": result_mode.get("success", False),
            },
            "message": "机器人紧急停止已执行",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"紧急停止失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"紧急停止失败: {str(e)}"
        )


@router.post("/reset", summary="重置机器人状态")
async def reset_robot():
    """重置机器人状态到初始位置"""
    try:
        control_service = get_robot_service()
        
        # 清除所有轨迹
        control_service.clear_trajectories()
        
        # 设置所有关节位置为0
        joint_count = 28  # 默认关节数量
        zero_positions = [0.0] * joint_count
        result = control_service.set_joint_positions(zero_positions)
        
        return {
            "success": result.get("success", False),
            "result": result,
            "message": "机器人状态已重置",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"重置机器人状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重置机器人状态失败: {str(e)}"
        )


    # ========== 力控制API端点 ==========
    
    @router.get("/force/status", summary="获取力控制系统状态")
    async def get_force_control_status():
        """获取力控制系统状态"""
        try:
            control_service = get_robot_service()
            status = control_service.get_force_control_status()
            
            return {
                "success": status.get("success", False),
                "force_control": status.get("force_control", {}),
                "available": status.get("available", False),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"获取力控制系统状态失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取力控制系统状态失败: {str(e)}"
            )
    
    @router.post("/force/start", summary="启动力控制系统")
    async def start_force_control(
        control_type: str = Query("impedance", description="控制类型: impedance, admittance, hybrid")
    ):
        """启动力控制系统"""
        try:
            control_service = get_robot_service()
            result = control_service.start_force_control(control_type)
            
            if result.get("success", False):
                return {
                    "success": True,
                    "started": result.get("started", False),
                    "control_type": control_type,
                    "message": result.get("message", "力控制系统已启动"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.get("error", "启动力控制系统失败")
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"启动力控制系统失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"启动力控制系统失败: {str(e)}"
            )
    
    @router.post("/force/stop", summary="停止力控制系统")
    async def stop_force_control():
        """停止力控制系统"""
        try:
            control_service = get_robot_service()
            result = control_service.stop_force_control()
            
            if result.get("success", False):
                return {
                    "success": True,
                    "stopped": result.get("stopped", False),
                    "message": result.get("message", "力控制系统已停止"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.get("error", "停止力控制系统失败")
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"停止力控制系统失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"停止力控制系统失败: {str(e)}"
            )
    
    @router.post("/force/params", summary="设置力控制参数")
    async def set_force_control_params(
        params: Dict[str, Any] = Body(..., description="力控制参数")
    ):
        """设置力控制参数"""
        try:
            control_service = get_robot_service()
            result = control_service.set_force_control_params(params)
            
            if result.get("success", False):
                return {
                    "success": True,
                    "params_set": result.get("params_set", False),
                    "message": result.get("message", "力控制参数已设置"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.get("error", "设置力控制参数失败")
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"设置力控制参数失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"设置力控制参数失败: {str(e)}"
            )
    
    @router.get("/force/data", summary="获取力控制数据")
    async def get_force_control_data(
        limit: int = Query(100, description="返回数据条数限制")
    ):
        """获取力控制数据日志"""
        try:
            control_service = get_robot_service()
            result = control_service.get_force_control_data(limit)
            
            if result.get("success", False):
                return {
                    "success": True,
                    "data": result.get("data", []),
                    "count": result.get("count", 0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.get("error", "获取力控制数据失败")
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取力控制数据失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取力控制数据失败: {str(e)}"
            )

    # ========== 测试端点 ==========
    
    @router.get("/test", summary="测试机器人控制API")
    async def test_robot_control_api():
        """测试机器人控制API是否正常工作"""
        try:
            control_service = get_robot_service()
            
            # 测试基本功能
            service_info = control_service.get_service_info()
            robot_state = control_service.get_robot_state()
            joint_states = control_service.get_joint_states()
            
            # 测试力控制系统
            force_control_status = control_service.get_force_control_status()
            
            test_results = {
                "service_initialized": service_info.get("initialized", False),
                "control_mode": service_info.get("control_mode", "unknown"),
                "hardware_connected": service_info.get("hardware_connected", False),
                "simulation_mode": service_info.get("simulation_mode", True),
                "joint_count": service_info.get("joint_count", 0),
                "robot_state_available": robot_state is not None,
                "joint_states_available": joint_states is not None,
                "force_control_available": force_control_status.get("available", False),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            all_tests_passed = (
                test_results["service_initialized"] and
                test_results["robot_state_available"] and
                test_results["joint_states_available"]
            )
            
            recommendations = []
            if not test_results["hardware_connected"]:
                recommendations.append("机器人未连接硬件，使用模拟模式")
                recommendations.append("如需硬件控制，请连接真实机器人或启动仿真环境")
            
            if test_results["simulation_mode"]:
                recommendations.append("当前为仿真模式，功能有限")
                recommendations.append("安装PyBullet以获得完整物理仿真: pip install pybullet")
            
            if not test_results["force_control_available"]:
                recommendations.append("力控制系统不可用，请检查force_control.py是否正确实现")
            
            return {
                "success": all_tests_passed,
                "test_results": test_results,
                "message": "机器人控制API测试完成" if all_tests_passed else "机器人控制API测试失败",
                "recommendations": recommendations,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"测试机器人控制API失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }