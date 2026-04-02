"""
机器人数据库迁移脚本
为现有用户创建默认机器人配置

此脚本执行以下操作：
1. 检查robot表是否存在，如果不存在则创建
2. 为每个现有用户创建默认人形机器人
3. 为每个机器人添加示例关节配置
4. 为每个机器人添加示例传感器配置

使用方法：
python migrate_robot_tables.py
"""

import sys
import os
import logging
from sqlalchemy import inspect
from datetime import datetime

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # 父目录是项目根目录
sys.path.insert(0, project_root)

from backend.core.database import Base, engine, SessionLocal
from backend.db_models.user import User
from backend.db_models.robot import Robot, RobotJoint, RobotSensor, RobotType, RobotStatus, ControlMode

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_table_exists(table_name):
    """检查表是否存在"""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def create_robot_tables():
    """创建机器人相关表"""
    try:
        # 导入所有需要的模型以确保它们注册到Base.metadata
        from backend.db_models.robot import Robot, RobotJoint, RobotSensor
        from backend.db_models.user import User
        
        # 创建表
        logger.info("创建机器人相关数据库表...")
        Base.metadata.create_all(bind=engine, tables=[
            Robot.__table__,
            RobotJoint.__table__,
            RobotSensor.__table__
        ])
        
        # 验证表是否创建成功
        inspector = inspect(engine)
        tables_created = [
            Robot.__tablename__,
            RobotJoint.__tablename__,
            RobotSensor.__tablename__
        ]
        
        for table in tables_created:
            if check_table_exists(table):
                logger.info(f"✓ 表 {table} 创建成功")
            else:
                logger.error(f"✗ 表 {table} 创建失败")
                return False
                
        return True
    except Exception as e:
        logger.error(f"创建表失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def create_default_robot_for_user(user, db):
    """为用户创建默认机器人"""
    try:
        # 检查用户是否已有默认机器人
        existing_default = db.query(Robot).filter(
            Robot.user_id == user.id,
            Robot.is_default == True
        ).first()
        
        if existing_default:
            logger.info(f"用户 {user.username} 已有默认机器人: {existing_default.name}")
            return existing_default
        
        # 创建默认机器人
        default_robot = Robot(
            name=f"{user.username}的机器人",
            description="默认人形机器人配置",
            robot_type=RobotType.HUMANOID,
            model="Humanoid V2.0",
            manufacturer="Self AGI Systems",
            status=RobotStatus.SIMULATION,
            battery_level=95.0,
            cpu_temperature=35.0,
            connection_type="simulation",
            connection_params={
                "simulation_mode": True,
                "gazebo_world": "empty.world",
                "physics_timestep": 0.001
            },
            configuration={
                "height": 1.6,  # 米
                "weight": 45.0,  # 千克
                "dof": 28,  # 自由度
                "max_walking_speed": 1.5,  # 米/秒
                "battery_capacity": "2000mAh",
                "operating_time": "4小时"
            },
            urdf_path="humanoid.urdf",
            simulation_engine="gazebo",
            control_mode=ControlMode.POSITION,
            capabilities={
                "walking": True,
                "balancing": True,
                "manipulation": True,
                "vision": True,
                "speech_recognition": True,
                "navigation": True
            },
            joint_count=28,
            sensor_count=12,
            user_id=user.id,
            is_public=False,
            is_default=True
        )
        
        db.add(default_robot)
        db.flush()  # 获取机器人ID
        
        logger.info(f"为用户 {user.username} 创建默认机器人: {default_robot.name} (ID: {default_robot.id})")
        
        # 为机器人创建关节配置
        create_joints_for_robot(default_robot, db)
        
        # 为机器人创建传感器配置
        create_sensors_for_robot(default_robot, db)
        
        return default_robot
    except Exception as e:
        logger.error(f"为用户 {user.username} 创建默认机器人失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None  # 返回None


def create_joints_for_robot(robot, db):
    """为机器人创建关节配置"""
    try:
        # 定义人形机器人的基本关节
        joints_config = [
            # 头部关节
            {
                "name": "head_yaw",
                "joint_type": "revolute",
                "min_position": -1.57,
                "max_position": 1.57,
                "max_velocity": 1.0,
                "max_torque": 5.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "头部偏航关节",
                "parent_link": "neck_base",
                "child_link": "head",
                "axis": "z"
            },
            {
                "name": "head_pitch",
                "joint_type": "revolute",
                "min_position": -0.79,
                "max_position": 0.79,
                "max_velocity": 1.0,
                "max_torque": 5.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "头部俯仰关节",
                "parent_link": "head",
                "child_link": "head_camera",
                "axis": "y"
            },
            # 左臂关节
            {
                "name": "l_shoulder_pitch",
                "joint_type": "revolute",
                "min_position": -2.09,
                "max_position": 2.09,
                "max_velocity": 2.0,
                "max_torque": 15.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "左肩俯仰关节",
                "parent_link": "torso",
                "child_link": "l_upper_arm",
                "axis": "y"
            },
            {
                "name": "l_shoulder_roll",
                "joint_type": "revolute",
                "min_position": -0.79,
                "max_position": 1.57,
                "max_velocity": 2.0,
                "max_torque": 15.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "左肩滚转关节",
                "parent_link": "l_upper_arm",
                "child_link": "l_lower_arm",
                "axis": "x"
            },
            {
                "name": "l_elbow_pitch",
                "joint_type": "revolute",
                "min_position": -2.09,
                "max_position": 2.09,
                "max_velocity": 2.0,
                "max_torque": 10.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "左肘关节",
                "parent_link": "l_lower_arm",
                "child_link": "l_hand",
                "axis": "y"
            },
            # 右臂关节
            {
                "name": "r_shoulder_pitch",
                "joint_type": "revolute",
                "min_position": -2.09,
                "max_position": 2.09,
                "max_velocity": 2.0,
                "max_torque": 15.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "右肩俯仰关节",
                "parent_link": "torso",
                "child_link": "r_upper_arm",
                "axis": "y"
            },
            {
                "name": "r_shoulder_roll",
                "joint_type": "revolute",
                "min_position": -0.79,
                "max_position": 1.57,
                "max_velocity": 2.0,
                "max_torque": 15.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "右肩滚转关节",
                "parent_link": "r_upper_arm",
                "child_link": "r_lower_arm",
                "axis": "x"
            },
            {
                "name": "r_elbow_pitch",
                "joint_type": "revolute",
                "min_position": -2.09,
                "max_position": 2.09,
                "max_velocity": 2.0,
                "max_torque": 10.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "右肘关节",
                "parent_link": "r_lower_arm",
                "child_link": "r_hand",
                "axis": "y"
            },
            # 左腿关节
            {
                "name": "l_hip_pitch",
                "joint_type": "revolute",
                "min_position": -1.57,
                "max_position": 1.57,
                "max_velocity": 2.0,
                "max_torque": 30.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "左髋俯仰关节",
                "parent_link": "pelvis",
                "child_link": "l_thigh",
                "axis": "y"
            },
            {
                "name": "l_hip_roll",
                "joint_type": "revolute",
                "min_position": -0.79,
                "max_position": 0.79,
                "max_velocity": 2.0,
                "max_torque": 25.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "左髋滚转关节",
                "parent_link": "l_thigh",
                "child_link": "l_shin",
                "axis": "x"
            },
            {
                "name": "l_knee_pitch",
                "joint_type": "revolute",
                "min_position": -2.09,
                "max_position": 0.0,
                "max_velocity": 2.0,
                "max_torque": 20.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "左膝关节",
                "parent_link": "l_shin",
                "child_link": "l_foot",
                "axis": "y"
            },
            # 右腿关节
            {
                "name": "r_hip_pitch",
                "joint_type": "revolute",
                "min_position": -1.57,
                "max_position": 1.57,
                "max_velocity": 2.0,
                "max_torque": 30.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "右髋俯仰关节",
                "parent_link": "pelvis",
                "child_link": "r_thigh",
                "axis": "y"
            },
            {
                "name": "r_hip_roll",
                "joint_type": "revolute",
                "min_position": -0.79,
                "max_position": 0.79,
                "max_velocity": 2.0,
                "max_torque": 25.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "右髋滚转关节",
                "parent_link": "r_thigh",
                "child_link": "r_shin",
                "axis": "x"
            },
            {
                "name": "r_knee_pitch",
                "joint_type": "revolute",
                "min_position": -2.09,
                "max_position": 0.0,
                "max_velocity": 2.0,
                "max_torque": 20.0,
                "offset": 0.0,
                "direction": 1.0,
                "description": "右膝关节",
                "parent_link": "r_shin",
                "child_link": "r_foot",
                "axis": "y"
            }
        ]
        
        joints_created = 0
        for joint_config in joints_config:
            # 检查关节是否已存在
            existing_joint = db.query(RobotJoint).filter(
                RobotJoint.robot_id == robot.id,
                RobotJoint.name == joint_config["name"]
            ).first()
            
            if existing_joint:
                logger.debug(f"机器人 {robot.name} 的关节 {joint_config['name']} 已存在")
                continue
            
            # 创建关节
            joint = RobotJoint(
                robot_id=robot.id,
                **joint_config
            )
            db.add(joint)
            joints_created += 1
        
        db.flush()
        logger.info(f"为机器人 {robot.name} 创建了 {joints_created} 个关节")
        
        # 更新机器人的关节计数
        robot.joint_count = joints_created
        db.add(robot)
        
        return joints_created
    except Exception as e:
        logger.error(f"为机器人 {robot.name} 创建关节失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


def create_sensors_for_robot(robot, db):
    """为机器人创建传感器配置"""
    try:
        # 定义人形机器人的基本传感器
        sensors_config = [
            # IMU传感器
            {
                "name": "imu_torso",
                "sensor_type": "imu",
                "model": "MPU-9250",
                "manufacturer": "InvenSense",
                "sampling_rate": 100.0,
                "accuracy": 0.1,
                "range_min": -16.0,
                "range_max": 16.0,
                "position_x": 0.0,
                "position_y": 0.8,
                "position_z": 0.0,
                "orientation_x": 0.0,
                "orientation_y": 0.0,
                "orientation_z": 0.0,
                "orientation_w": 1.0,
                "status": "online",
                "description": "躯干IMU传感器",
                "calibration_data": {
                    "bias_x": 0.001,
                    "bias_y": 0.001,
                    "bias_z": 0.001,
                    "scale_factor": 1.002
                }
            },
            # 相机传感器
            {
                "name": "head_camera",
                "sensor_type": "camera",
                "model": "Realsense D435",
                "manufacturer": "Intel",
                "sampling_rate": 30.0,
                "accuracy": 0.05,
                "range_min": 0.1,
                "range_max": 10.0,
                "position_x": 0.0,
                "position_y": 1.25,
                "position_z": 0.05,
                "orientation_x": 0.0,
                "orientation_y": 0.0,
                "orientation_z": 0.0,
                "orientation_w": 1.0,
                "status": "online",
                "description": "头部RGB-D相机",
                "calibration_data": {
                    "fx": 616.591,
                    "fy": 616.591,
                    "cx": 318.169,
                    "cy": 241.873,
                    "distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
                }
            },
            # 力传感器
            {
                "name": "l_foot_force",
                "sensor_type": "force",
                "model": "FSR-406",
                "manufacturer": "Interlink Electronics",
                "sampling_rate": 100.0,
                "accuracy": 0.01,
                "range_min": 0.0,
                "range_max": 100.0,
                "position_x": -0.05,
                "position_y": -1.0,
                "position_z": 0.02,
                "orientation_x": 0.0,
                "orientation_y": 0.0,
                "orientation_z": 0.0,
                "orientation_w": 1.0,
                "status": "online",
                "description": "左脚力传感器",
                "calibration_data": {
                    "zero_offset": 0.05,
                    "sensitivity": 0.01
                }
            },
            {
                "name": "r_foot_force",
                "sensor_type": "force",
                "model": "FSR-406",
                "manufacturer": "Interlink Electronics",
                "sampling_rate": 100.0,
                "accuracy": 0.01,
                "range_min": 0.0,
                "range_max": 100.0,
                "position_x": 0.05,
                "position_y": -1.0,
                "position_z": 0.02,
                "orientation_x": 0.0,
                "orientation_y": 0.0,
                "orientation_z": 0.0,
                "orientation_w": 1.0,
                "status": "online",
                "description": "右脚力传感器",
                "calibration_data": {
                    "zero_offset": 0.05,
                    "sensitivity": 0.01
                }
            }
        ]
        
        sensors_created = 0
        for sensor_config in sensors_config:
            # 检查传感器是否已存在
            existing_sensor = db.query(RobotSensor).filter(
                RobotSensor.robot_id == robot.id,
                RobotSensor.name == sensor_config["name"]
            ).first()
            
            if existing_sensor:
                logger.debug(f"机器人 {robot.name} 的传感器 {sensor_config['name']} 已存在")
                continue
            
            # 创建传感器
            sensor = RobotSensor(
                robot_id=robot.id,
                **sensor_config
            )
            db.add(sensor)
            sensors_created += 1
        
        db.flush()
        logger.info(f"为机器人 {robot.name} 创建了 {sensors_created} 个传感器")
        
        # 更新机器人的传感器计数
        robot.sensor_count = sensors_created
        db.add(robot)
        
        return sensors_created
    except Exception as e:
        logger.error(f"为机器人 {robot.name} 创建传感器失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


def migrate():
    """执行数据库迁移"""
    logger.info("开始机器人数据库迁移...")
    
    db = SessionLocal()
    try:
        # 检查机器人表是否存在
        robot_table_exists = check_table_exists("robots")
        robot_joints_table_exists = check_table_exists("robot_joints")
        robot_sensors_table_exists = check_table_exists("robot_sensors")
        
        tables_exist = all([
            robot_table_exists,
            robot_joints_table_exists,
            robot_sensors_table_exists
        ])
        
        if not tables_exist:
            logger.info("机器人相关表不存在，开始创建...")
            if not create_robot_tables():
                logger.error("创建机器人表失败")
                return False
        else:
            logger.info("机器人相关表已存在")
        
        # 获取所有用户
        users = db.query(User).all()
        logger.info(f"找到 {len(users)} 个用户")
        
        if not users:
            logger.warning("没有找到用户，无法创建默认机器人")
            return False
        
        # 为每个用户创建默认机器人
        robots_created = 0
        for user in users:
            logger.info(f"为用户 {user.username} (ID: {user.id}) 创建默认机器人...")
            robot = create_default_robot_for_user(user, db)
            if robot:
                robots_created += 1
        
        # 提交所有更改
        db.commit()
        
        logger.info(f"迁移完成！成功为 {robots_created}/{len(users)} 个用户创建了默认机器人")
        
        # 打印统计信息
        total_robots = db.query(Robot).count()
        total_joints = db.query(RobotJoint).count()
        total_sensors = db.query(RobotSensor).count()
        
        logger.info(f"数据库统计:")
        logger.info(f"  机器人总数: {total_robots}")
        logger.info(f"  关节总数: {total_joints}")
        logger.info(f"  传感器总数: {total_sensors}")
        
        return True
        
    except Exception as e:
        logger.error(f"迁移失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
        return False
    finally:
        db.close()


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Self AGI 机器人数据库迁移工具")
    logger.info("=" * 60)
    
    try:
        success = migrate()
        if success:
            logger.info("✓ 迁移成功完成！")
            return 0
        else:
            logger.error("✗ 迁移失败")
            return 1
    except KeyboardInterrupt:
        logger.info("迁移被用户中断")
        return 1
    except Exception as e:
        logger.error(f"迁移过程中发生未预期的错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())