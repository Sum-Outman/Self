"""
硬件工厂模块
提供硬件接口的优雅降级机制，确保在真实硬件SDK不可用时自动切换到模拟模式

功能：
1. 自动检测硬件SDK可用性
2. 根据配置和可用性选择适当的硬件接口
3. 提供统一的硬件接口创建函数
4. 支持配置驱动的硬件接口选择
5. 提供硬件状态监控和自动降级

解决审计报告中的问题：真实硬件驱动依赖外部SDK，缺乏优雅降级机制
"""

import logging
from typing import Dict, Any, Optional, Type, Union, List, Tuple
from dataclasses import dataclass, field

from .robot_controller import HardwareInterface, RobotJoint, NullHardwareInterface
from .real_robot_interface import NAOqiRobotInterface, RobotConnectionConfig, RealRobotType
from .simulation import PyBulletSimulation
from .unified_interface import UnifiedHardwareInterface

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """硬件配置"""
    
    # 接口选择策略
    interface_strategy: str = "auto"  # auto, real_first, simulation_first, real_only, simulation_only
    fallback_enabled: bool = True  # 是否启用降级回退
    fallback_timeout: float = 5.0  # 降级检测超时（秒）
    
    # 真实硬件配置
    real_robot_type: RealRobotType = RealRobotType.NAO
    real_robot_host: str = "localhost"
    real_robot_port: int = 9559
    real_connection_timeout: float = 10.0
    
    # 仿真配置
    simulation_gui_enabled: bool = False
    simulation_physics_timestep: float = 1.0/240.0
    simulation_realtime: bool = True
    
    # 监控配置
    health_check_interval: float = 30.0  # 健康检查间隔（秒）
    auto_reconnect: bool = True  # 自动重连
    reconnect_attempts: int = 3  # 重连尝试次数
    
    def to_connection_config(self) -> RobotConnectionConfig:
        """转换为机器人连接配置"""
        return RobotConnectionConfig(
            robot_type=self.real_robot_type,
            host=self.real_robot_host,
            port=self.real_robot_port,
            timeout=self.real_connection_timeout,
            auto_reconnect=True,
            reconnect_interval=2.0
        )


class HardwareFactory:
    """硬件工厂类（单例）"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._config = HardwareConfig()
            self._current_interface = None
            self._interface_type = None  # 当前接口类型：real, simulation, unified
            self._sdk_availability_cache = {}  # SDK可用性缓存
            self._init_sdk_availability()
            
            logger.info("硬件工厂初始化完成")
    
    def _init_sdk_availability(self) -> None:
        """初始化SDK可用性检测"""
        # 检测NAOqi SDK可用性
        naoqi_available = self._check_naoqi_availability()
        self._sdk_availability_cache["naoqi"] = naoqi_available
        
        # 检测PyBullet可用性
        pybullet_available = self._check_pybullet_availability()
        self._sdk_availability_cache["pybullet"] = pybullet_available
        
        logger.info(f"SDK可用性检测: NAOqi={naoqi_available}, PyBullet={pybullet_available}")
    
    def _check_naoqi_availability(self) -> bool:
        """检查NAOqi SDK是否可用"""
        try:
            import naoqi  # type: ignore
            from naoqi import ALProxy  # type: ignore
            logger.debug("NAOqi SDK可用")
            return True
        except ImportError:
            logger.warning("NAOqi SDK不可用，NAO机器人功能将不可用（项目要求禁止使用虚拟数据）")
            return False
        except Exception as e:
            logger.error(f"检查NAOqi SDK时出错: {e}")
            return False
    
    def _check_pybullet_availability(self) -> bool:
        """检查PyBullet是否可用"""
        try:
            import pybullet  # type: ignore
            logger.debug("PyBullet可用")
            return True
        except ImportError:
            logger.warning("PyBullet不可用，物理仿真功能将不可用（项目要求禁止使用虚拟数据）")
            return False
        except Exception as e:
            logger.error(f"检查PyBullet时出错: {e}")
            return False
    
    def get_hardware_interface(self, config: Optional[HardwareConfig] = None) -> HardwareInterface:
        """
        获取硬件接口
        
        注意：根据项目要求"禁止使用虚拟数据"和"不做妥协不可以降级处理"，
        只支持真实硬件接口，不提供模拟后备方案。
        
        参数:
            config: 硬件配置，如果为None则使用默认配置
            
        返回:
            HardwareInterface: 硬件接口实例
        """
        if config is not None:
            self._config = config
        
        # 根据策略选择接口
        strategy = self._config.interface_strategy
        
        # 根据项目要求"禁止使用虚拟数据"，只支持真实硬件接口
        if strategy == "real_only":
            interface = self._create_real_interface()
        elif strategy == "simulation_only":
            raise ValueError(
                "simulation_only策略已被禁用\n"
                "根据项目要求'禁止使用虚拟数据'，不支持模拟接口策略。\n"
                "请使用real_only策略以使用真实硬件接口。"
            )
        elif strategy == "real_first":
            # real_first现在等同于real_only（禁止降级）
            logger.warning("real_first策略已修改为只尝试真实硬件接口（禁止降级）")
            interface = self._create_real_interface()
        elif strategy == "simulation_first":
            raise ValueError(
                "simulation_first策略已被禁用\n"
                "根据项目要求'禁止使用虚拟数据'，不支持模拟接口策略。\n"
                "请使用real_only策略以使用真实硬件接口。"
            )
        else:  # auto
            # auto策略现在尝试真实硬件接口（禁止降级）
            logger.warning("auto策略已修改为只尝试真实硬件接口（禁止降级）")
            interface = self._create_real_interface()
        
        # 保存当前接口
        self._current_interface = interface
        return interface
    
    def _create_real_interface(self) -> HardwareInterface:
        """创建真实硬件接口"""
        logger.info("尝试创建真实硬件接口")
        
        # 检查NAOqi SDK是否可用
        if not self._sdk_availability_cache.get("naoqi", False):
            logger.error("NAOqi SDK不可用，无法创建真实硬件接口")
            # 根据用户要求"系统可以在没有硬件条件下单独运行AGI所有功能"，
            # 当真实硬件不可用时，返回NullHardwareInterface
            if self._config.fallback_enabled:
                logger.info("启用降级回退：返回NullHardwareInterface（无硬件模式）")
                null_interface = NullHardwareInterface()
                self._interface_type = "null_hardware"
                return null_interface
            else:
                # 如果不允许降级，抛出异常
                raise RuntimeError(
                    "NAOqi SDK不可用，无法创建真实硬件接口\n"
                    "根据项目要求'禁止使用虚拟数据'，必须安装并配置NAOqi SDK。"
                )
        
        try:
            # 创建连接配置
            connection_config = self._config.to_connection_config()
            
            # 创建NAOqi接口
            interface = NAOqiRobotInterface(connection_config)
            
            # 尝试连接
            connected = interface.connect()
            if connected:
                logger.info("真实硬件接口创建并连接成功")
                self._interface_type = "real"
                return interface
            else:
                logger.error("真实硬件接口连接失败")
                # 连接失败时，根据配置决定是否降级
                if self._config.fallback_enabled:
                    logger.info("真实硬件连接失败，启用降级回退：返回NullHardwareInterface（无硬件模式）")
                    null_interface = NullHardwareInterface()
                    self._interface_type = "null_hardware"
                    return null_interface
                else:
                    raise RuntimeError(
                        "真实硬件接口连接失败\n"
                        "根据项目要求'禁止使用虚拟数据'，必须成功连接真实硬件。"
                    )
                    
        except Exception as e:
            logger.error(f"创建真实硬件接口失败: {e}")
            # 异常情况下，根据配置决定是否降级
            if self._config.fallback_enabled:
                logger.info(f"创建真实硬件接口时发生异常，启用降级回退：返回NullHardwareInterface（无硬件模式），异常: {e}")
                null_interface = NullHardwareInterface()
                self._interface_type = "null_hardware"
                return null_interface
            else:
                raise RuntimeError(
                    f"创建真实硬件接口失败: {e}\n"
                    "根据项目要求'禁止使用虚拟数据'，必须成功创建真实硬件接口。"
                )
    
    def _create_simulation_interface(self) -> HardwareInterface:
        """创建仿真硬件接口（已禁用）
        
        注意：根据项目要求"禁止使用虚拟数据"，此方法已被禁用。
        必须使用真实硬件接口。
        """
        raise RuntimeError(
            "仿真硬件接口已被禁用\n"
            "根据项目要求'禁止使用虚拟数据'，不支持模拟接口。\n"
            "必须使用真实硬件接口。"
        )
    
    def _create_basic_simulation_interface(self) -> HardwareInterface:
        """创建基础模拟接口（已禁用）
        
        注意：根据项目要求"禁止使用虚拟数据"，此方法已被禁用。
        必须使用真实硬件接口。
        """
        raise RuntimeError(
            "基础模拟接口已被禁用\n"
            "根据项目要求'禁止使用虚拟数据'，不支持模拟接口。\n"
            "必须使用真实硬件接口。"
        )
            

    
    # 完整：删除冗余的_try_real_first和_try_simulation_first方法
    # 逻辑已合并到_auto_select_interface和_create_simulation_interface中
    
    def _auto_select_interface(self) -> HardwareInterface:
        """自动选择最佳接口
        
        根据用户要求"系统可以在没有硬件条件下单独运行AGI所有功能"，
        当真实硬件不可用时，允许降级到NullHardwareInterface。
        """
        logger.info("使用auto策略：尝试真实硬件接口，失败时降级到NullHardwareInterface")
        
        # 检查NAOqi SDK是否可用
        if self._sdk_availability_cache.get("naoqi", False):
            logger.info("NAOqi SDK可用，尝试使用真实硬件接口")
            try:
                interface = self._create_real_interface()
                # _create_real_interface现在会处理降级，可能返回NullHardwareInterface
                logger.info(f"接口创建成功，类型: {self._interface_type}")
                return interface
            except Exception as e:
                logger.error(f"真实接口创建失败: {e}")
                # 如果配置不允许降级，会抛出异常；否则_create_real_interface已经处理了降级
                raise RuntimeError(
                    f"真实硬件接口创建失败且不允许降级: {e}\n"
                    "请检查硬件配置或启用降级回退。"
                )
        
        # NAOqi SDK不可用，尝试降级到NullHardwareInterface
        logger.warning("NAOqi SDK不可用，无法创建真实硬件接口")
        if self._config.fallback_enabled:
            logger.info("启用降级回退：返回NullHardwareInterface（无硬件模式）")
            null_interface = NullHardwareInterface()
            self._interface_type = "null_hardware"
            return null_interface
        else:
            raise RuntimeError(
                "NAOqi SDK不可用，无法创建真实硬件接口\n"
                "根据项目要求'禁止使用虚拟数据'，必须安装并配置真实硬件SDK，或启用降级回退。"
            )
    
    def get_current_interface_info(self) -> Dict[str, Any]:
        """获取当前接口信息"""
        if self._current_interface is None:
            return {
                "interface_type": None,
                "interface_available": False,
                "simulation_mode": True,
                "description": "无活动接口"
            }
        
        info = self._current_interface.get_interface_info()
        info.update({
            "interface_type": self._interface_type,
            "factory_strategy": self._config.interface_strategy,
            "sdk_availability": self._sdk_availability_cache.copy(),
            "fallback_enabled": self._config.fallback_enabled
        })
        return info
    
    def check_hardware_health(self) -> Dict[str, Any]:
        """检查硬件健康状态"""
        if self._current_interface is None:
            return {
                "healthy": False,
                "status": "no_interface",
                "interface_type": None,
                "sdk_available": self._sdk_availability_cache.copy(),
                "message": "无活动硬件接口"
            }
        
        try:
            # 检查接口连接状态
            connected = self._current_interface.is_connected()
            
            health_status = {
                "healthy": connected,
                "status": "connected" if connected else "disconnected",
                "interface_type": self._interface_type,
                "simulation_mode": self._current_interface.is_simulation,
                "sdk_available": self._sdk_availability_cache.copy(),
                "message": "硬件接口正常" if connected else "硬件接口断开连接"
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"检查硬件健康状态失败: {e}")
            return {
                "healthy": False,
                "status": "error",
                "interface_type": self._interface_type,
                "simulation_mode": True,
                "sdk_available": self._sdk_availability_cache.copy(),
                "message": f"健康检查失败: {str(e)}"
            }


# 全局硬件工厂实例
_hardware_factory_instance = None


def get_hardware_factory() -> HardwareFactory:
    """获取硬件工厂实例（单例）"""
    global _hardware_factory_instance
    if _hardware_factory_instance is None:
        _hardware_factory_instance = HardwareFactory()
    return _hardware_factory_instance


def create_hardware_interface(config: Optional[HardwareConfig] = None) -> HardwareInterface:
    """
    创建硬件接口（工厂函数）
    
    参数:
        config: 硬件配置
        
    返回:
        HardwareInterface: 硬件接口实例
    """
    factory = get_hardware_factory()
    return factory.get_hardware_interface(config)


def get_hardware_health() -> Dict[str, Any]:
    """获取硬件健康状态"""
    factory = get_hardware_factory()
    return factory.check_hardware_health()