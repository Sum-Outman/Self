"""
机器人控制与示范学习集成服务
将示范学习功能集成到机器人控制流程中
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from .demonstration_service import (
    DemonstrationRecorder, DemonstrationPlayer, FrameData, RecordingMode
)
from ..db_models.robot import Robot
from ..db_models.demonstration import Demonstration, DemonstrationStatus, DemonstrationType


logger = logging.getLogger(__name__)


class RobotDemonstrationIntegration:
    """机器人控制与示范学习集成"""
    
    def __init__(self, db: Session):
        """
        初始化集成服务
        
        参数:
            db: 数据库会话
        """
        self.db = db
        self.active_recorders = {}  # robot_id -> DemonstrationRecorder
        self.active_players = {}    # robot_id -> DemonstrationPlayer
        self.robot_status_cache = {}
    
    async def start_demonstration_recording(
        self,
        robot_id: int,
        user_id: int,
        name: str,
        description: str = "",
        demonstration_type: DemonstrationType = DemonstrationType.JOINT_CONTROL,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Demonstration]:
        """
        开始录制机器人示范
        
        参数:
            robot_id: 机器人ID
            user_id: 用户ID
            name: 示范名称
            description: 示范描述
            demonstration_type: 示范类型
            config: 录制配置
            
        返回:
            创建的Demonstration对象或None
        """
        try:
            # 检查是否已在录制
            if robot_id in self.active_recorders:
                logger.warning(f"机器人 {robot_id} 已在录制中")
                return self.active_recorders[robot_id].demonstration
            
            # 创建录制器
            recorder = DemonstrationRecorder(self.db, robot_id, user_id)
            
            # 开始录制
            demonstration = recorder.start_recording(
                name=name,
                description=description,
                demonstration_type=demonstration_type,
                config=config
            )
            
            if demonstration:
                # 保存录制器
                self.active_recorders[robot_id] = recorder
                logger.info(f"开始录制机器人 {robot_id} 的示范: {name}")
                
                # 启动后台录制任务
                asyncio.create_task(self._recording_monitor(robot_id))
                
                return demonstration
            else:
                logger.error(f"开始录制失败: 机器人 {robot_id}")
                return None  # 返回None
                
        except Exception as e:
            logger.error(f"开始录制示范失败: {e}")
            return None  # 返回None
    
    async def stop_demonstration_recording(
        self,
        robot_id: int,
        save: bool = True
    ) -> Optional[Demonstration]:
        """
        停止录制机器人示范
        
        参数:
            robot_id: 机器人ID
            save: 是否保存数据
            
        返回:
            更新后的Demonstration对象或None
        """
        try:
            if robot_id not in self.active_recorders:
                logger.warning(f"机器人 {robot_id} 未在录制中")
                return None  # 返回None
            
            recorder = self.active_recorders[robot_id]
            demonstration = recorder.stop_recording(save=save)
            
            # 移除录制器
            if robot_id in self.active_recorders:
                del self.active_recorders[robot_id]
            
            if demonstration and save:
                logger.info(f"停止录制机器人 {robot_id} 的示范: {demonstration.name}")
            else:
                logger.info(f"取消录制机器人 {robot_id} 的示范")
            
            return demonstration
            
        except Exception as e:
            logger.error(f"停止录制示范失败: {e}")
            return None  # 返回None
    
    async def record_robot_state(
        self,
        robot_id: int,
        joint_positions: Dict[str, float],
        joint_velocities: Optional[Dict[str, float]] = None,
        joint_torques: Optional[Dict[str, float]] = None,
        sensor_data: Optional[Dict[str, Any]] = None,
        imu_data: Optional[Dict[str, Any]] = None,
        control_commands: Optional[Dict[str, Any]] = None,
        environment_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        录制机器人状态帧
        
        参数:
            robot_id: 机器人ID
            joint_positions: 关节位置数据
            joint_velocities: 关节速度数据
            joint_torques: 关节扭矩数据
            sensor_data: 传感器数据
            imu_data: IMU数据
            control_commands: 控制命令
            environment_state: 环境状态
            
        返回:
            是否成功
        """
        try:
            if robot_id not in self.active_recorders:
                return False
            
            recorder = self.active_recorders[robot_id]
            
            # 创建帧数据
            frame_data = FrameData(
                timestamp=time.time(),
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                joint_torques=joint_torques,
                sensor_data=sensor_data,
                imu_data=imu_data,
                control_commands=control_commands,
                environment_state=environment_state
            )
            
            # 录制帧
            success = recorder.record_frame(frame_data)
            
            return success
            
        except Exception as e:
            logger.error(f"录制机器人状态失败: {e}")
            return False
    
    async def start_demonstration_playback(
        self,
        robot_id: int,
        demonstration_id: int,
        start_frame: int = 0,
        playback_speed: float = 1.0,
        loop: bool = False
    ) -> bool:
        """
        开始播放示范数据
        
        参数:
            robot_id: 机器人ID
            demonstration_id: 示范ID
            start_frame: 起始帧索引
            playback_speed: 播放速度
            loop: 是否循环播放
            
        返回:
            是否成功
        """
        try:
            # 检查是否已在播放
            if robot_id in self.active_players:
                logger.warning(f"机器人 {robot_id} 已在播放中")
                return False
            
            # 创建播放器
            player = DemonstrationPlayer(self.db, demonstration_id)
            
            # 加载示范
            success = player.load_demonstration()
            if not success:
                logger.error(f"加载示范失败: {demonstration_id}")
                return False
            
            # 开始播放
            success = player.start_playback(start_frame, playback_speed, loop)
            if success:
                # 保存播放器
                self.active_players[robot_id] = player
                logger.info(f"开始播放机器人 {robot_id} 的示范: {demonstration_id}")
                
                # 启动后台播放任务
                asyncio.create_task(self._playback_monitor(robot_id))
                
                return True
            else:
                logger.error(f"开始播放失败: 示范 {demonstration_id}")
                return False
                
        except Exception as e:
            logger.error(f"开始播放示范失败: {e}")
            return False
    
    async def stop_demonstration_playback(self, robot_id: int) -> bool:
        """
        停止播放示范数据
        
        参数:
            robot_id: 机器人ID
            
        返回:
            是否成功
        """
        try:
            if robot_id not in self.active_players:
                logger.warning(f"机器人 {robot_id} 未在播放中")
                return False
            
            player = self.active_players[robot_id]
            success = player.stop_playback()
            
            # 移除播放器
            if robot_id in self.active_players:
                del self.active_players[robot_id]
            
            logger.info(f"停止播放机器人 {robot_id} 的示范")
            
            return success
            
        except Exception as e:
            logger.error(f"停止播放示范失败: {e}")
            return False
    
    async def get_current_playback_frame(self, robot_id: int) -> Optional[FrameData]:
        """
        获取当前播放帧数据
        
        参数:
            robot_id: 机器人ID
            
        返回:
            当前帧数据或None
        """
        try:
            if robot_id not in self.active_players:
                return None  # 返回None
            
            player = self.active_players[robot_id]
            
            # 更新帧索引
            player.update_frame_index()
            
            # 获取当前帧
            frame_data = player.get_current_frame()
            
            return frame_data
            
        except Exception as e:
            logger.error(f"获取当前播放帧失败: {e}")
            return None  # 返回None
    
    async def apply_demonstration_frame(
        self,
        robot_id: int,
        frame_data: FrameData,
        control_mode: str = "position"
    ) -> bool:
        """
        应用示范帧数据到机器人控制
        
        参数:
            robot_id: 机器人ID
            frame_data: 帧数据
            control_mode: 控制模式 (position, velocity, torque)
            
        返回:
            是否成功应用
        """
        try:
            # 这里应该调用实际的机器人控制接口
            # 目前只是模拟应用
            
            if not frame_data:
                return False
            
            # 应用关节位置控制
            if frame_data.joint_positions and control_mode == "position":
                # 这里应该调用机器人控制API
                # 例如: await robot_controller.set_joint_positions(robot_id, frame_data.joint_positions)
                logger.debug(f"应用关节位置控制: {len(frame_data.joint_positions)} 个关节")
            
            # 应用关节速度控制
            elif frame_data.joint_velocities and control_mode == "velocity":
                # 这里应该调用机器人控制API
                logger.debug(f"应用关节速度控制: {len(frame_data.joint_velocities)} 个关节")
            
            # 应用关节扭矩控制
            elif frame_data.joint_torques and control_mode == "torque":
                # 这里应该调用机器人控制API
                logger.debug(f"应用关节扭矩控制: {len(frame_data.joint_torques)} 个关节")
            
            # 应用控制命令
            if frame_data.control_commands:
                # 这里应该处理控制命令
                logger.debug(f"应用控制命令: {frame_data.control_commands}")
            
            return True
            
        except Exception as e:
            logger.error(f"应用示范帧失败: {e}")
            return False
    
    async def _recording_monitor(self, robot_id: int):
        """录制监控任务"""
        try:
            while robot_id in self.active_recorders:
                recorder = self.active_recorders[robot_id]
                
                # 检查录制状态
                status = recorder.get_status()
                
                # 更新机器人状态缓存
                self.robot_status_cache[robot_id] = {
                    "recording": True,
                    "frame_count": status["frame_count"],
                    "duration": status["recording_duration"],
                    "demonstration_id": status["demonstration_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # 定期检查
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"录制监控任务失败: {e}")
        finally:
            # 清理
            if robot_id in self.robot_status_cache:
                del self.robot_status_cache[robot_id]
    
    async def _playback_monitor(self, robot_id: int):
        """播放监控任务"""
        try:
            while robot_id in self.active_players:
                player = self.active_players[robot_id]
                
                # 获取当前帧数据
                frame_data = await self.get_current_playback_frame(robot_id)
                
                if frame_data:
                    # 应用帧数据到机器人控制
                    await self.apply_demonstration_frame(robot_id, frame_data)
                
                # 更新机器人状态缓存
                status = player.get_status()
                self.robot_status_cache[robot_id] = {
                    "playing": True,
                    "current_frame": status["current_frame"],
                    "total_frames": status["total_frames"],
                    "playback_speed": status["playback_speed"],
                    "demonstration": status["demonstration"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # 定期检查
                await asyncio.sleep(0.033)  # 大约30Hz
                
        except Exception as e:
            logger.error(f"播放监控任务失败: {e}")
        finally:
            # 清理
            if robot_id in self.robot_status_cache:
                del self.robot_status_cache[robot_id]
    
    def get_robot_demonstration_status(self, robot_id: int) -> Dict[str, Any]:
        """
        获取机器人示范状态
        
        参数:
            robot_id: 机器人ID
            
        返回:
            示范状态信息
        """
        status = {
            "recording": robot_id in self.active_recorders,
            "playing": robot_id in self.active_players,
            "robot_id": robot_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # 添加录制状态
        if robot_id in self.active_recorders:
            recorder = self.active_recorders[robot_id]
            status.update(recorder.get_status())
        
        # 添加播放状态
        if robot_id in self.active_players:
            player = self.active_players[robot_id]
            status.update(player.get_status())
        
        return status
    
    def get_all_robot_demonstration_status(self) -> Dict[int, Dict[str, Any]]:
        """
        获取所有机器人的示范状态
        
        返回:
            机器人ID到状态的映射
        """
        status_dict = {}
        
        # 获取录制中的机器人
        for robot_id in self.active_recorders:
            status_dict[robot_id] = self.get_robot_demonstration_status(robot_id)
        
        # 获取播放中的机器人
        for robot_id in self.active_players:
            if robot_id not in status_dict:
                status_dict[robot_id] = self.get_robot_demonstration_status(robot_id)
        
        return status_dict


# 全局集成服务实例
_integration_service = None

def get_robot_demonstration_integration() -> RobotDemonstrationIntegration:
    """获取机器人示范集成服务实例（单例）"""
    global _integration_service
    if _integration_service is None:
        # 注意：这里需要传入数据库会话，使用工厂函数模式
        # 在实际使用时，应该通过依赖注入传递db会话
        _integration_service = RobotDemonstrationIntegration(None)
    return _integration_service