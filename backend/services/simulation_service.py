#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真环境服务模块
管理PyBullet和Gazebo仿真环境的集成，提供统一的仿真接口
处理依赖检测、错误处理和回退机制

基于BaseService重构，集成统一日志、错误处理和服务管理
"""

import logging
import sys
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

from .base_service import BaseService, ServiceConfig, ServiceError, service_operation


class SimulationService(BaseService):
    """仿真环境服务单例类，基于BaseService重构"""
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        # 配置扩展：添加仿真服务特定配置
        if config:
            # 确保有仿真特定的额外配置
            if "extra_config" not in config.__dict__:
                config.extra_config = {}
            
            # 设置仿真默认配置
            simulation_defaults = {
                "preferred_engine": "unified",  # unified, pybullet, gazebo, simulation
                "gui_enabled": False,
                "physics_timestep": 1.0/240.0,
                "realtime_simulation": False,
                "ros_master_uri": "http://localhost:11311",
                "gazebo_world": "empty.world",
                "robot_model": "humanoid",
                "simulation_mode": True,
                "fallback_order": ["unified_pybullet", "unified_gazebo", "pybullet", "gazebo", "simulation"]
            }
            
            # 合并默认配置，但不覆盖用户配置
            for key, value in simulation_defaults.items():
                if key not in config.extra_config:
                    config.extra_config[key] = value
        
        # 调用父类初始化
        super().__init__(config)
        
        # 初始化仿真特定属性（_initialize_service会设置其他属性）
        self._simulation_interfaces = {}
        self._dependencies = {}
    
    def _initialize_service(self) -> bool:
        """初始化仿真环境服务特定资源
        
        返回:
            bool: 初始化是否成功
        """
        try:
            self.logger.info("开始初始化仿真环境服务...")
            
            # 检查依赖
            self._dependencies = self._check_dependencies()
            
            # 初始化仿真接口
            self._simulation_interfaces = {}
            self._active_interface = None
            
            # 根据配置初始化接口
            self._initialize_simulation_interfaces()
            
            self.logger.info(f"仿真环境服务初始化成功，活动接口: {self._active_interface}")
            return True
            
        except Exception as e:
            self.logger.error(f"仿真环境服务初始化失败: {e}")
            self._last_error = str(e)
            return False
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """检查仿真环境依赖"""
        dependencies = {
            "pybullet": False,
            "roslibpy": False,
            "unified_simulation": False,  # 统一仿真接口可用性
            "gazebo_simulation": False,   # 向后兼容性检查
            "pybullet_simulation": False, # 向后兼容性检查
        }
        
        # 检查pybullet
        try:
            import pybullet
            dependencies["pybullet"] = True
            self.logger.info("PyBullet可用")
        except ImportError:
            self.logger.warning("PyBullet不可用，需要安装: pip install pybullet")
        
        # 检查roslibpy
        try:
            import roslibpy
            dependencies["roslibpy"] = True
            self.logger.info("roslibpy可用")
        except ImportError:
            self.logger.warning("roslibpy不可用，需要安装: pip install roslibpy")
        
        # 检查统一仿真接口
        try:
            from hardware.unified_simulation import UnifiedSimulation
            dependencies["unified_simulation"] = True
            self.logger.info("统一仿真接口可用")
        except ImportError as e:
            self.logger.warning(f"统一仿真接口导入失败: {e}")
        
        # 检查硬件仿真模块（向后兼容性）
        try:
            from hardware.gazebo_simulation import GazeboSimulation
            dependencies["gazebo_simulation"] = True
            self.logger.info("Gazebo仿真模块可用（向后兼容）")
        except ImportError as e:
            self.logger.warning(f"Gazebo仿真模块导入失败: {e}")
        
        try:
            from hardware.simulation import PyBulletSimulation
            dependencies["pybullet_simulation"] = True
            self.logger.info("PyBullet仿真模块可用（向后兼容）")
        except ImportError as e:
            self.logger.warning(f"PyBullet仿真模块导入失败: {e}")
        
        return dependencies
    
    def _initialize_simulation_interfaces(self):
        """初始化仿真接口"""
        extra_config = self.config.extra_config
        fallback_order = extra_config.get("fallback_order", ["unified_pybullet", "unified_gazebo", "pybullet", "gazebo", "simulation"])
        
        # 按回退顺序尝试初始化接口
        for interface_type in fallback_order:
            success = False
            
            if interface_type == "unified_pybullet":
                success = self._initialize_unified_pybullet_interface()
            elif interface_type == "unified_gazebo":
                success = self._initialize_unified_gazebo_interface()
            elif interface_type == "pybullet":
                success = self._initialize_pybullet_interface()
            elif interface_type == "gazebo":
                success = self._initialize_gazebo_interface()
            elif interface_type == "simulation":
                success = self._setup_simulation_only()
            
            if success:
                self._active_interface = interface_type
                self.logger.info(f"成功使用 {interface_type} 接口")
                return
        
        # 所有接口都失败
        self.logger.warning("所有仿真接口初始化失败")
        self._active_interface = "simulation"
        self._setup_simulation_only()
    
    def _initialize_unified_pybullet_interface(self) -> bool:
        """初始化统一PyBullet仿真接口"""
        try:
            from hardware.unified_simulation import UnifiedSimulation
            
            extra_config = self.config.extra_config
            
            simulation = UnifiedSimulation(
                engine="pybullet",
                engine_config={
                    "gui_enabled": extra_config.get("gui_enabled", False),
                    "physics_timestep": extra_config.get("physics_timestep", 1.0/240.0),
                    "realtime_simulation": extra_config.get("realtime_simulation", False)
                },
                gui_enabled=extra_config.get("gui_enabled", False),
                physics_timestep=extra_config.get("physics_timestep", 1.0/240.0),
                realtime_simulation=extra_config.get("realtime_simulation", False)
            )
            
            if simulation.connect():
                if hasattr(simulation, 'backend_type') and simulation.backend_type == "pybullet":
                    self._simulation_interfaces["pybullet"] = simulation
                    self._simulation_interfaces["unified_pybullet"] = simulation
                    self.logger.info("统一PyBullet仿真接口初始化成功")
                    return True
                else:
                    self.logger.warning(f"统一仿真接口后端类型不是pybullet: {getattr(simulation, 'backend_type', 'unknown')}")
                    return False
            else:
                self.logger.warning("统一PyBullet仿真接口连接失败")
                return False
                
        except ImportError as e:
            self.logger.info(f"统一仿真接口不可用: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"统一PyBullet仿真接口初始化失败: {e}")
            return False
    
    def _initialize_unified_gazebo_interface(self) -> bool:
        """初始化统一Gazebo仿真接口"""
        try:
            from hardware.unified_simulation import UnifiedSimulation
            
            extra_config = self.config.extra_config
            
            simulation = UnifiedSimulation(
                engine="gazebo",
                engine_config={
                    "ros_master_uri": extra_config.get("ros_master_uri", "http://localhost:11311"),
                    "gazebo_world": extra_config.get("gazebo_world", "empty.world"),
                    "robot_model": extra_config.get("robot_model", "humanoid"),
                    "gui_enabled": extra_config.get("gui_enabled", False),
                    "simulation_mode": extra_config.get("simulation_mode", True)
                },
                gui_enabled=extra_config.get("gui_enabled", False),
                simulation_mode=extra_config.get("simulation_mode", True)
            )
            
            if simulation.connect():
                if hasattr(simulation, 'backend_type') and simulation.backend_type == "gazebo":
                    self._simulation_interfaces["gazebo"] = simulation
                    self._simulation_interfaces["unified_gazebo"] = simulation
                    self.logger.info("统一Gazebo仿真接口初始化成功")
                    return True
                else:
                    self.logger.warning(f"统一仿真接口后端类型不是gazebo: {getattr(simulation, 'backend_type', 'unknown')}")
                    return False
            else:
                self.logger.warning("统一Gazebo仿真接口连接失败")
                return False
                
        except ImportError as e:
            self.logger.info(f"统一仿真接口不可用: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"统一Gazebo仿真接口初始化失败: {e}")
            return False
    
    def _initialize_pybullet_interface(self) -> bool:
        """初始化原始PyBullet仿真接口（向后兼容）"""
        try:
            from hardware.simulation import PyBulletSimulation
            
            extra_config = self.config.extra_config
            
            simulation = PyBulletSimulation(
                gui_enabled=extra_config.get("gui_enabled", False),
                physics_timestep=extra_config.get("physics_timestep", 1.0/240.0),
                realtime_simulation=extra_config.get("realtime_simulation", False)
            )
            
            if simulation.connect():
                if hasattr(simulation, 'pybullet_available') and simulation.pybullet_available:
                    self._simulation_interfaces["pybullet"] = simulation
                    self.logger.info("PyBullet仿真接口初始化成功（使用原始仿真模块）")
                    return True
                else:
                    self.logger.warning("PyBullet仿真接口连接但PyBullet不可用")
                    return False
            else:
                self.logger.warning("原始PyBullet仿真接口连接失败")
                return False
                
        except ImportError as e:
            self.logger.error(f"原始PyBullet仿真模块也不可用: {e}")
            return False
        except Exception as e:
            self.logger.error(f"原始PyBullet仿真接口初始化失败: {e}")
            return False
    
    def _initialize_gazebo_interface(self) -> bool:
        """初始化原始Gazebo仿真接口（向后兼容）"""
        try:
            from hardware.gazebo_simulation import GazeboSimulation
            
            extra_config = self.config.extra_config
            
            simulation = GazeboSimulation(
                ros_master_uri=extra_config.get("ros_master_uri", "http://localhost:11311"),
                gazebo_world=extra_config.get("gazebo_world", "empty.world"),
                robot_model=extra_config.get("robot_model", "humanoid"),
                gui_enabled=extra_config.get("gui_enabled", False),
                simulation_mode=extra_config.get("simulation_mode", True)
            )
            
            if simulation.connect():
                self._simulation_interfaces["gazebo"] = simulation
                self.logger.info("Gazebo仿真接口初始化成功（使用原始仿真模块）")
                return True
            else:
                self.logger.warning("原始Gazebo仿真接口连接失败")
                return False
                
        except ImportError as e:
            self.logger.error(f"原始Gazebo仿真模块也不可用: {e}")
            return False
        except Exception as e:
            self.logger.error(f"原始Gazebo仿真接口初始化失败: {e}")
            return False
    
    def _setup_simulation_only(self) -> bool:
        """设置纯模拟模式（无实际仿真引擎）"""
        self._active_interface = "simulation"
        self.logger.info("使用纯模拟模式（无实际仿真引擎）")
        return True
    
    @service_operation(operation_name="get_active_interface")
    def get_active_interface(self) -> Optional[str]:
        """获取当前活动的仿真接口名称"""
        return self._active_interface
    
    @service_operation(operation_name="get_interface_status")
    def get_interface_status(self, interface_name: Optional[str] = None) -> Dict[str, Any]:
        """获取仿真接口状态
        
        参数:
            interface_name: 接口名称，如果为None则返回所有接口状态
            
        返回:
            Dict[str, Any]: 接口状态信息
        """
        if interface_name:
            if interface_name in self._simulation_interfaces:
                interface = self._simulation_interfaces[interface_name]
                status = {
                    "name": interface_name,
                    "active": (interface_name == self._active_interface),
                    "connected": True,
                    "backend_type": getattr(interface, 'backend_type', 'unknown') if hasattr(interface, 'backend_type') else 'unknown',
                    "engine": getattr(interface, 'engine', 'unknown') if hasattr(interface, 'engine') else 'unknown',
                    "available": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # 添加接口特定信息
                if hasattr(interface, 'get_interface_info'):
                    try:
                        interface_info = interface.get_interface_info()
                        if isinstance(interface_info, dict):
                            status.update(interface_info)
                    except Exception as e:
                        self.logger.warning(f"获取接口信息失败: {e}")
                        status["interface_info_error"] = str(e)
                
                return {interface_name: status}
            else:
                return {
                    interface_name: {
                        "name": interface_name,
                        "active": False,
                        "connected": False,
                        "available": False,
                        "error": f"接口 {interface_name} 不存在",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
        else:
            # 返回所有接口状态
            all_status = {}
            for name, interface in self._simulation_interfaces.items():
                all_status[name] = {
                    "name": name,
                    "active": (name == self._active_interface),
                    "connected": True,
                    "backend_type": getattr(interface, 'backend_type', 'unknown') if hasattr(interface, 'backend_type') else 'unknown',
                    "engine": getattr(interface, 'engine', 'unknown') if hasattr(interface, 'engine') else 'unknown',
                    "available": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # 添加活动接口信息
            if self._active_interface and self._active_interface not in all_status:
                all_status[self._active_interface] = {
                    "name": self._active_interface,
                    "active": True,
                    "connected": False,
                    "available": True,
                    "type": "simulation_only",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            return all_status
    
    @service_operation(operation_name="get_simulation_capabilities")
    def get_simulation_capabilities(self) -> Dict[str, Any]:
        """获取仿真能力信息"""
        capabilities = {
            "dependencies": self._dependencies,
            "available_interfaces": list(self._simulation_interfaces.keys()),
            "active_interface": self._active_interface,
            "interfaces_count": len(self._simulation_interfaces),
            "service_name": self.service_name,
            "service_version": "1.0.0",
            "config": self.config.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # 添加接口详细信息
        interfaces_info = {}
        for name, interface in self._simulation_interfaces.items():
            interface_info = {
                "backend_type": getattr(interface, 'backend_type', 'unknown') if hasattr(interface, 'backend_type') else 'unknown',
                "engine": getattr(interface, 'engine', 'unknown') if hasattr(interface, 'engine') else 'unknown',
                "connected": True,
                "active": (name == self._active_interface)
            }
            
            # 添加接口特定能力
            if hasattr(interface, 'get_capabilities'):
                try:
                    interface_caps = interface.get_capabilities()
                    if isinstance(interface_caps, dict):
                        interface_info.update(interface_caps)
                except Exception as e:
                    self.logger.warning(f"获取接口能力失败: {e}")
                    interface_info["capabilities_error"] = str(e)
            
            interfaces_info[name] = interface_info
        
        capabilities["interfaces_info"] = interfaces_info
        
        return capabilities
    
    @service_operation(operation_name="execute_simulation_command")
    def execute_simulation_command(self, command: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行仿真命令
        
        参数:
            command: 命令名称
            parameters: 命令参数
            
        返回:
            Dict[str, Any]: 命令执行结果
        """
        if parameters is None:
            parameters = {}
        
        try:
            # 检查活动接口
            if not self._active_interface:
                return {
                    "success": False,
                    "error": "没有活动的仿真接口",
                    "command": command,
                    "parameters": parameters,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # 如果是纯模拟模式，返回模拟响应
            if self._active_interface == "simulation":
                return {
                    "success": True,
                    "message": f"模拟执行命令: {command}",
                    "command": command,
                    "parameters": parameters,
                    "simulation_mode": True,
                    "result": {"simulated": True, "executed": False},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # 获取活动接口
            active_interface = None
            for name, interface in self._simulation_interfaces.items():
                if name == self._active_interface or name.replace("unified_", "") == self._active_interface:
                    active_interface = interface
                    break
            
            if not active_interface:
                return {
                    "success": False,
                    "error": f"活动接口 {self._active_interface} 不可用",
                    "command": command,
                    "parameters": parameters,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # 根据命令类型执行
            if command == "reset":
                # 重置仿真
                if hasattr(active_interface, 'reset'):
                    result = active_interface.reset()
                    return {
                        "success": True,
                        "message": "仿真重置成功",
                        "command": command,
                        "result": result,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "error": "当前接口不支持reset命令",
                        "command": command,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
            
            elif command == "step":
                # 执行仿真步
                if hasattr(active_interface, 'step'):
                    result = active_interface.step()
                    return {
                        "success": True,
                        "message": "仿真步执行成功",
                        "command": command,
                        "result": result,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "error": "当前接口不支持step命令",
                        "command": command,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
            
            elif command == "get_state":
                # 获取仿真状态
                if hasattr(active_interface, 'get_state'):
                    result = active_interface.get_state()
                    return {
                        "success": True,
                        "message": "获取仿真状态成功",
                        "command": command,
                        "result": result,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "error": "当前接口不支持get_state命令",
                        "command": command,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
            
            elif command == "set_state":
                # 设置仿真状态
                state = parameters.get("state", {})
                if hasattr(active_interface, 'set_state'):
                    result = active_interface.set_state(state)
                    return {
                        "success": True,
                        "message": "设置仿真状态成功",
                        "command": command,
                        "result": result,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "error": "当前接口不支持set_state命令",
                        "command": command,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
            
            else:
                # 未知命令
                return {
                    "success": False,
                    "error": f"未知命令: {command}",
                    "command": command,
                    "supported_commands": ["reset", "step", "get_state", "set_state"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"执行仿真命令失败: {e}")
            return {
                "success": False,
                "error": f"执行仿真命令失败: {e}",
                "command": command,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    @service_operation(operation_name="get_service_info")
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        # 获取父类的服务信息
        base_info = super().get_service_info()
        
        # 添加仿真特定信息
        simulation_info = {
            "service_name": "SimulationService",
            "status": "running",
            "version": "1.0.0",
            "active_interface": self._active_interface,
            "available_interfaces": list(self._simulation_interfaces.keys()),
            "interfaces_count": len(self._simulation_interfaces),
            "dependencies": self._dependencies,
            "initialized": self._initialized,
            "mock_data": False,  # 明确标记为非真实数据
        }
        
        # 合并信息
        base_info.update(simulation_info)
        return base_info
    
    def get_simulation_interface(self, interface_name: str):
        """获取仿真接口实例（用于高级操作）"""
        return self._simulation_interfaces.get(interface_name)
    
    def set_simulation_interface(self, interface_name: str, interface):
        """设置仿真接口（主要用于测试）"""
        self._simulation_interfaces[interface_name] = interface
        if interface is not None:
            self._active_interface = interface_name
        else:
            # 如果设置为None，移除接口
            if interface_name in self._simulation_interfaces:
                del self._simulation_interfaces[interface_name]
            if self._active_interface == interface_name:
                self._active_interface = None


# 向后兼容函数
def get_simulation_service() -> SimulationService:
    """获取仿真服务实例（向后兼容）"""
    return SimulationService()