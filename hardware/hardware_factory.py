"""
硬件工厂模块
根据项目要求提供硬件接口管理,不支持任何降级处理

功能:
1. 自动检测硬件SDK可用性
2. 根据配置创建适当的硬件接口
3. 提供统一的硬件接口创建函数
4. 支持配置驱动的硬件接口选择
5. 提供硬件状态监控

设计原则:
1. 不连接硬件情况下AGI系统可以正常运行,硬件部分没有任何数据
2. 不采用任何降级处理,直接返回无硬件接口
3. 连接硬件后与硬件开始处理真实数据
4. 部分硬件连接就可以工作
"""

import logging
from typing import Dict, Any, Optional, Type, Union, List, Tuple
from dataclasses import dataclass, field

from .robot_controller import HardwareInterface, RobotJoint, NullHardwareInterface
from .real_robot_interface import NAOqiRobotInterface, RobotConnectionConfig, RealRobotType
from .simulation import PyBulletSimulation

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    '''硬件配置
    
    注意:根据项目要求"不使用任何降级处理",fallback_enabled已禁用.
    当硬件不可用时,系统将返回明确的"无硬件"接口,而不是降级到模拟模式.
    '''
    
    # 接口选择策略
    interface_strategy: str = "auto"  # auto, real_first, simulation_first, real_only, simulation_only
    fallback_enabled: bool = False  # 是否启用降级回退(已禁用,根据项目要求)
    fallback_timeout: float = 5.0  # 降级检测超时(秒)
    
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
    health_check_interval: float = 30.0  # 健康检查间隔(秒)
    auto_reconnect: bool = True  # 自动重连
    reconnect_attempts: int = 3  # 重连尝试次数
    
    def to_connection_config(self) -> RobotConnectionConfig:
        '''转换为机器人连接配置'''
        return RobotConnectionConfig(
            robot_type=self.real_robot_type,
            host=self.real_robot_host,
            port=self.real_robot_port,
            timeout=self.real_connection_timeout,
            auto_reconnect=True,
            reconnect_interval=2.0
        )


class HardwareFactory:
    """硬件工厂类(单例)"""
    
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
            self._interface_type = None  # 当前接口类型:real, simulation, unified
            self._sdk_availability_cache = {}  # SDK可用性缓存
            self._init_sdk_availability()
            
            logger.info("硬件工厂初始化完成")
    
    def _init_sdk_availability(self) -> None:
        '''初始化SDK可用性检测'''
        # 检测NAOqi SDK可用性
        naoqi_available = self._check_naoqi_availability()
        self._sdk_availability_cache["naoqi"] = naoqi_available
        
        # 检测PyBullet可用性
        pybullet_available = self._check_pybullet_availability()
        self._sdk_availability_cache["pybullet"] = pybullet_available
        
        logger.info(f"SDK可用性检测: NAOqi={naoqi_available}, PyBullet={pybullet_available}")
    
    def _check_naoqi_availability(self) -> bool:
        '''检查NAOqi SDK是否可用'''
        try:
            import naoqi  # type: ignore
            from naoqi import ALProxy  # type: ignore
            logger.debug("NAOqi SDK可用")
            return True
        except ImportError:
            logger.warning("NAOqi SDK不可用,NAO机器人功能将不可用(项目要求禁止使用虚拟数据)")
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
            logger.warning("PyBullet不可用,物理仿真功能将不可用(项目要求禁止使用虚拟数据)")
            return False
        except Exception as e:
            logger.error(f"检查PyBullet时出错: {e}")
            return False
    
    def get_hardware_interface(self, config: Optional[HardwareConfig] = None) -> HardwareInterface:
        """
        获取硬件接口
        
        根据项目要求:
        1. 不连接硬件情况下AGI系统可以正常运行,硬件部分没有任何数据
        2. 不采用任何降级处理,硬件不可用时返回无硬件接口
        3. 连接硬件后与硬件开始处理真实数据(禁止使用虚拟数据)
        4. 部分硬件连接就可以工作
        """

        if config is not None:
            self._config = config
        
        # 根据策略选择接口
        strategy = self._config.interface_strategy
        
        # 根据项目要求"禁止使用虚拟数据",只支持真实硬件接口
        if strategy == "real_only":
            interface = self._create_real_interface()
        elif strategy == "simulation_only":
            raise ValueError(
                "simulation_only策略已被禁用\n"
                "根据项目要求'禁止使用虚拟数据',不支持模拟接口策略.\n"
                "请使用real_only策略以使用真实硬件接口."
            )
        elif strategy == "real_first":
            # real_first现在等同于real_only(禁止降级)
            logger.warning("real_first策略已修改为只尝试真实硬件接口(禁止降级)")
            interface = self._create_real_interface()
        elif strategy == "simulation_first":
            raise ValueError(
                "simulation_first策略已被禁用\n"
                "根据项目要求'禁止使用虚拟数据',不支持模拟接口策略.\n"
                "请使用real_only策略以使用真实硬件接口."
            )
        else:  # auto
            # auto策略现在尝试真实硬件接口(禁止降级)
            logger.warning("auto策略已修改为只尝试真实硬件接口(禁止降级)")
            interface = self._create_real_interface()
        
        # 保存当前接口
        self._current_interface = interface
        return interface
    
    def _create_real_interface(self) -> HardwareInterface:
        '''创建真实硬件接口
        
        根据项目要求:
        1. 不连接硬件情况下AGI系统可以正常运行,硬件部分没有任何数据
        2. 不采用任何降级处理,硬件不可用时直接返回无硬件接口
        3. 连接硬件后与硬件开始处理真实数据
        4. 部分硬件连接就可以工作
        '''
        logger.info("尝试创建真实硬件接口")
        
        # 检查NAOqi SDK是否可用
        if not self._sdk_availability_cache.get("naoqi", False):
            logger.warning("NAOqi SDK不可用,进入无硬件模式")
            # 根据用户要求"不连接硬件情况下AGI系统可以正常运行"
            # 返回无硬件接口,系统其他部分继续运行
            null_interface = NullHardwareInterface()
            self._interface_type = "null_hardware"
            logger.info("无硬件模式:系统可在无硬件条件下运行AGI功能,硬件部分无数据")
            return null_interface
        
        try:
            # 创建连接配置
            connection_config = self._config.to_connection_config()
            
            # 创建NAOqi接口
            interface = NAOqiRobotInterface(connection_config)
            
            # 尝试连接（NAOqiRobotInterface.connect()已支持部分硬件连接）
            connected = interface.connect()
            
            if not connected:
                # 连接失败，根据项目要求返回无硬件接口
                # 注意：NAOqiRobotInterface.connect()已实现部分连接支持，
                # 所以连接失败意味着完全无法连接，应进入无硬件模式
                logger.warning("真实硬件接口连接失败,进入无硬件模式")
                null_interface = NullHardwareInterface()
                self._interface_type = "null_hardware"
                logger.info("无硬件模式:系统可在无硬件条件下运行AGI功能,硬件部分无数据")
                return null_interface
            
            # 连接成功
            logger.info("真实硬件接口创建并连接成功")
            self._interface_type = "real"
            
            # 检测部分硬件可用性（如果连接成功）
            # 注意：NAOqiRobotInterface.connect()已标记部分连接状态
            self._detect_partial_hardware(interface)
            
            return interface
                    
        except Exception as e:
            # 创建接口失败，根据项目要求返回无硬件接口
            logger.error(f"创建真实硬件接口失败: {e}")
            null_interface = NullHardwareInterface()
            self._interface_type = "null_hardware"
            logger.info("无硬件模式:系统可在无硬件条件下运行AGI功能,硬件部分无数据")
            return null_interface
    
    def _detect_partial_hardware(self, interface) -> None:
        '''检测部分硬件可用性
        
        检查接口的各个功能组件是否可用，标记不可用的部分。
        这有助于实现"部分硬件连接就可以工作"的要求。
        '''
        logger.info("检测部分硬件可用性...")
        
        # 检查接口是否支持硬件健康检测
        if hasattr(interface, 'get_hardware_health'):
            try:
                health = interface.get_hardware_health()
                logger.info(f"硬件健康状态: {health}")
                
                # 根据健康状态标记部分可用性
                if "components" in health:
                    components = health["components"]
                    unavailable_count = sum(1 for status in components.values() if status == "unavailable")
                    if unavailable_count > 0:
                        logger.info(f"检测到 {unavailable_count} 个硬件组件不可用，进入部分硬件模式")
                        interface._partial_hardware = True
                        interface._available_components = {
                            comp: status for comp, status in components.items()
                        }
            except Exception as e:
                logger.warning(f"检测硬件健康状态失败: {e}")
        
        # 检查关键代理是否可用（针对NAOqi接口）
        if hasattr(interface, 'motion_proxy'):
            if interface.motion_proxy is None:
                logger.warning("运动代理不可用，运动功能将受限")
        
        if hasattr(interface, 'memory_proxy'):
            if interface.memory_proxy is None:
                logger.warning("内存代理不可用，传感器数据获取将受限")
        
        logger.info("部分硬件检测完成")
    
    def _create_simulation_interface(self) -> HardwareInterface:
        '''创建仿真硬件接口(已禁用)
        
        注意:根据项目要求"禁止使用虚拟数据",此方法已被禁用.
        必须使用真实硬件接口.
        '''
        raise RuntimeError(
            "仿真硬件接口已被禁用\n"
            "根据项目要求'禁止使用虚拟数据',不支持模拟接口.\n"
            "必须使用真实硬件接口."
        )
    
    def _create_basic_simulation_interface(self) -> HardwareInterface:
        '''创建基础模拟接口(已禁用)
        
        注意:根据项目要求"禁止使用虚拟数据",此方法已被禁用.
        必须使用真实硬件接口.
        '''
        raise RuntimeError(
            "基础模拟接口已被禁用\n"
            "根据项目要求'禁止使用虚拟数据',不支持模拟接口.\n"
            "必须使用真实硬件接口."
        )
            

    
    # 完整:删除冗余的_try_real_first和_try_simulation_first方法
    # 逻辑已合并到_auto_select_interface和_create_simulation_interface中
    
    def _auto_select_interface(self) -> HardwareInterface:
        '''自动选择最佳接口
        
        根据项目要求:
        1. 尝试使用真实硬件接口
        2. 如果硬件不可用,进入无硬件模式(返回NullHardwareInterface)
        3. 系统可在无硬件条件下运行AGI功能
        '''
        logger.info("使用auto策略:尝试真实硬件接口,硬件不可用时进入无硬件模式")
        
        # 直接调用_create_real_interface,它会处理所有情况
        interface = self._create_real_interface()
        logger.info(f"接口创建成功,类型: {self._interface_type}")
        return interface
    
    def get_current_interface_info(self) -> Dict[str, Any]:
        '''获取当前接口信息'''
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
        '''检查硬件健康状态'''
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
            # 根据项目要求"不采用任何降级处理，直接报错"
            raise RuntimeError(f"硬件健康检查失败: {e}")


# 全局硬件工厂实例
_hardware_factory_instance = None


def get_hardware_factory() -> HardwareFactory:
    '''获取硬件工厂实例(单例)'''
    global _hardware_factory_instance
    if _hardware_factory_instance is None:
        _hardware_factory_instance = HardwareFactory()
    return _hardware_factory_instance


def create_hardware_interface(config: Optional[HardwareConfig] = None) -> HardwareInterface:
    '''创建硬件接口(工厂函数)'''
    factory = get_hardware_factory()
    return factory.get_hardware_interface(config)



