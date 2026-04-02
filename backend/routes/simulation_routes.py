"""
仿真环境管理路由模块
处理PyBullet和Gazebo仿真环境的API请求
包括仿真控制、状态监控、环境配置等功能
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import logging

from backend.dependencies import get_db, get_current_user
from backend.db_models.user import User
from backend.services.simulation_service import get_simulation_service

router = APIRouter(prefix="/api/simulation", tags=["仿真环境管理"])

logger = logging.getLogger(__name__)


@router.get("/status", summary="获取仿真环境状态")
async def get_simulation_status():
    """获取仿真环境状态信息"""
    try:
        simulation_service = get_simulation_service()
        service_info = simulation_service.get_service_info()
        
        return {
            "success": True,
            "status": "running",
            "data": service_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取仿真环境状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取仿真环境状态失败: {str(e)}"
        )


@router.get("/dependencies", summary="获取仿真环境依赖状态")
async def get_simulation_dependencies():
    """获取仿真环境依赖安装状态"""
    try:
        simulation_service = get_simulation_service()
        service_info = simulation_service.get_service_info()
        
        dependencies = service_info.get("dependencies", {})
        capabilities = service_info.get("capabilities", {})
        
        return {
            "success": True,
            "dependencies": dependencies,
            "capabilities": capabilities,
            "recommendations": _get_dependency_recommendations(dependencies),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取仿真环境依赖状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取仿真环境依赖状态失败: {str(e)}"
        )


@router.get("/interfaces", summary="获取仿真接口状态")
async def get_simulation_interfaces(interface_name: Optional[str] = Query(None, description="特定接口名称")):
    """获取仿真接口状态信息"""
    try:
        simulation_service = get_simulation_service()
        
        if interface_name:
            interface_status = simulation_service.get_interface_status(interface_name)
            return {
                "success": True,
                "interface": interface_name,
                "status": interface_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            all_status = simulation_service.get_interface_status()
            return {
                "success": True,
                "interfaces": all_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    except Exception as e:
        logger.error(f"获取仿真接口状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取仿真接口状态失败: {str(e)}"
        )


@router.post("/command", summary="执行仿真命令")
async def execute_simulation_command(
    command: str = Query(..., description="仿真命令：reset, step, pause, resume"),
    steps: Optional[int] = Query(1, description="步数（仅对step命令有效）"),
    interface: Optional[str] = Query(None, description="指定仿真接口，默认为当前活动接口"),
):
    """执行仿真环境命令"""
    try:
        simulation_service = get_simulation_service()
        
        # 准备命令参数
        params = {}
        if command == "step":
            params["steps"] = steps
        
        # 执行命令
        result = simulation_service.execute_simulation_command(command, params)
        
        return {
            "success": result.get("success", False),
            "command": command,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"执行仿真命令失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"执行仿真命令失败: {str(e)}"
        )


@router.post("/reset", summary="重置仿真环境")
async def reset_simulation(interface: Optional[str] = Query(None, description="指定仿真接口，默认为当前活动接口")):
    """重置仿真环境到初始状态"""
    try:
        simulation_service = get_simulation_service()
        result = simulation_service.execute_simulation_command("reset")
        
        return {
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "interface": result.get("interface", result.get("mode", "unknown")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"重置仿真环境失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重置仿真环境失败: {str(e)}"
        )


@router.post("/step", summary="步进仿真环境")
async def step_simulation(
    steps: int = Query(1, description="步进次数"),
    interface: Optional[str] = Query(None, description="指定仿真接口，默认为当前活动接口"),
):
    """步进仿真环境"""
    try:
        simulation_service = get_simulation_service()
        params = {"steps": steps}
        result = simulation_service.execute_simulation_command("step", params)
        
        return {
            "success": result.get("success", False),
            "steps": result.get("steps_executed", 0),
            "interface": result.get("interface", result.get("mode", "unknown")),
            "results": result.get("results", []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"步进仿真环境失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"步进仿真环境失败: {str(e)}"
        )


@router.get("/capabilities", summary="获取仿真能力信息")
async def get_simulation_capabilities():
    """获取仿真环境的详细能力信息"""
    try:
        simulation_service = get_simulation_service()
        capabilities = simulation_service.get_simulation_capabilities()
        
        return {
            "success": True,
            "capabilities": capabilities,
            "recommendations": _get_capability_recommendations(capabilities),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"获取仿真能力信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取仿真能力信息失败: {str(e)}"
        )


def _get_dependency_recommendations(dependencies: Dict[str, bool]) -> List[str]:
    """根据依赖状态生成安装建议"""
    recommendations = []
    
    if not dependencies.get("pybullet", False):
        recommendations.append("安装PyBullet: pip install pybullet")
    
    if not dependencies.get("roslibpy", False):
        recommendations.append("安装roslibpy: pip install roslibpy")
    
    if not dependencies.get("pybullet_simulation", False):
        recommendations.append("PyBullet仿真模块缺失，检查hardware.simulation模块")
    
    if not dependencies.get("gazebo_simulation", False):
        recommendations.append("Gazebo仿真模块缺失，检查hardware.gazebo_simulation模块")
    
    if not recommendations:
        recommendations.append("所有依赖已满足，仿真环境可用")
    
    return recommendations


def _get_capability_recommendations(capabilities: Dict[str, Any]) -> List[str]:
    """根据能力状态生成优化建议"""
    recommendations = []
    
    interfaces_available = capabilities.get("interfaces_available", [])
    active_interface = capabilities.get("active_interface", "")
    
    if not interfaces_available:
        recommendations.append("未检测到可用的仿真接口，请安装PyBullet或启动Gazebo/ROS服务")
        recommendations.append("PyBullet安装: pip install pybullet")
        recommendations.append("Gazebo/ROS配置: 启动ROS服务 (roslaunch gazebo_ros empty_world.launch)")
    
    if active_interface == "simulation":
        recommendations.append("当前使用纯模拟模式，仿真功能有限")
        if "pybullet" not in interfaces_available:
            recommendations.append("建议安装PyBullet以获得完整物理仿真功能")
    
    if "pybullet" in interfaces_available:
        recommendations.append("PyBullet可用，可启用GUI进行可视化: gui_enabled=True")
    
    if "gazebo" in interfaces_available:
        recommendations.append("Gazebo可用，建议配置真实机器人模型和环境")
    
    if not recommendations:
        recommendations.append("仿真环境配置良好，建议进行机器人控制测试")
    
    return recommendations


# 测试端点
@router.get("/test", summary="测试仿真环境API")
async def test_simulation_api():
    """测试仿真环境API是否正常工作"""
    try:
        simulation_service = get_simulation_service()
        
        # 测试基本功能
        service_info = simulation_service.get_service_info()
        capabilities = simulation_service.get_simulation_capabilities()
        interface_status = simulation_service.get_interface_status()
        
        test_results = {
            "service_initialized": service_info.get("initialized", False),
            "active_interface": service_info.get("active_interface", "unknown"),
            "interfaces_available": service_info.get("interfaces_available", []),
            "dependencies_check": all(service_info.get("dependencies", {}).values()),
            "capabilities_available": len(capabilities.get("interfaces_available", [])) > 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        all_tests_passed = (
            test_results["service_initialized"] and
            test_results["capabilities_available"]
        )
        
        return {
            "success": all_tests_passed,
            "test_results": test_results,
            "message": "仿真环境API测试完成" if all_tests_passed else "仿真环境API测试失败",
            "recommendations": _get_capability_recommendations(capabilities),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        logger.error(f"测试仿真环境API失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "仿真环境API测试失败",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }