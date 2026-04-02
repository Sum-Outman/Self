"""
示范学习数据库迁移脚本
创建示范学习相关表并添加示例数据

使用方法：
python migrate_demonstration_tables.py
"""

import sys
import os
import logging
from sqlalchemy import inspect
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.database import Base, engine, SessionLocal
from backend.db_models.demonstration import (
    Demonstration, DemonstrationFrame, CameraFrame, DemonstrationTask, TrainingResult,
    DemonstrationType, DemonstrationStatus, DemonstrationFormat
)
from backend.db_models.robot import Robot
from backend.db_models.user import User

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


def create_demonstration_tables():
    """创建示范学习相关表"""
    try:
        # 导入所有需要的模型以确保它们注册到Base.metadata
        from backend.db_models.demonstration import (
            Demonstration, DemonstrationFrame, CameraFrame, 
            DemonstrationTask, TrainingResult
        )
        
        # 创建表
        logger.info("创建示范学习相关数据库表...")
        Base.metadata.create_all(bind=engine, tables=[
            Demonstration.__table__,
            DemonstrationFrame.__table__,
            CameraFrame.__table__,
            DemonstrationTask.__table__,
            TrainingResult.__table__,
        ])
        
        # 验证表是否创建成功
        inspector = inspect(engine)
        tables_created = [
            Demonstration.__tablename__,
            DemonstrationFrame.__tablename__,
            CameraFrame.__tablename__,
            DemonstrationTask.__tablename__,
            TrainingResult.__tablename__,
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


def create_example_demonstration_tasks(db):
    """创建示例示范任务"""
    try:
        # 示例任务配置
        example_tasks = [
            {
                "name": "关节控制示范",
                "description": "基础关节控制示范任务，学习如何控制机器人关节",
                "task_type": "joint_control",
                "config": {
                    "joints": ["head_yaw", "head_pitch", "l_shoulder_pitch", "r_shoulder_pitch"],
                    "position_range": [-1.57, 1.57],
                    "velocity_limit": 1.0,
                    "duration": 10.0
                },
                "success_criteria": {
                    "position_accuracy": 0.1,
                    "velocity_stability": 0.05,
                    "smoothness": 0.8
                }
            },
            {
                "name": "站立平衡示范",
                "description": "机器人站立平衡控制示范",
                "task_type": "balance",
                "config": {
                    "joints": ["l_hip_pitch", "r_hip_pitch", "l_knee_pitch", "r_knee_pitch", "l_ankle_pitch", "r_ankle_pitch"],
                    "balance_tolerance": 0.05,
                    "max_tilt": 0.2,
                    "duration": 30.0
                },
                "success_criteria": {
                    "balance_duration": 20.0,
                    "max_tilt": 0.1,
                    "recovery_speed": 1.0
                }
            },
            {
                "name": "手臂轨迹跟踪",
                "description": "手臂轨迹跟踪示范任务",
                "task_type": "trajectory_tracking",
                "config": {
                    "trajectory_type": "circle",
                    "radius": 0.3,
                    "speed": 0.1,
                    "joints": ["l_shoulder_pitch", "l_shoulder_roll", "l_elbow_pitch"]
                },
                "success_criteria": {
                    "tracking_error": 0.02,
                    "smoothness": 0.9,
                    "completion_time": 15.0
                }
            }
        ]
        
        # 获取所有用户
        users = db.query(User).all()
        robots = db.query(Robot).all()
        
        if not users or not robots:
            logger.warning("没有找到用户或机器人，无法创建示例任务")
            return 0
        
        tasks_created = 0
        for user in users:
            for robot in robots:
                if robot.user_id != user.id:
                    continue  # 只为用户自己的机器人创建任务
                
                for task_config in example_tasks:
                    # 检查任务是否已存在
                    existing_task = db.query(DemonstrationTask).filter(
                        DemonstrationTask.name == task_config["name"],
                        DemonstrationTask.robot_id == robot.id,
                        DemonstrationTask.user_id == user.id
                    ).first()
                    
                    if existing_task:
                        logger.debug(f"任务 {task_config['name']} 已存在")
                        continue
                    
                    # 创建任务
                    task = DemonstrationTask(
                        name=task_config["name"],
                        description=task_config["description"],
                        task_type=task_config["task_type"],
                        config=task_config["config"],
                        success_criteria=task_config["success_criteria"],
                        robot_id=robot.id,
                        user_id=user.id
                    )
                    
                    db.add(task)
                    tasks_created += 1
        
        db.flush()
        logger.info(f"创建了 {tasks_created} 个示例示范任务")
        return tasks_created
        
    except Exception as e:
        logger.error(f"创建示例任务失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


def migrate():
    """执行数据库迁移"""
    logger.info("开始示范学习数据库迁移...")
    
    db = SessionLocal()
    try:
        # 检查示范学习表是否存在
        tables_to_check = [
            "demonstrations",
            "demonstration_frames", 
            "camera_frames",
            "demonstration_tasks",
            "training_results"
        ]
        
        tables_exist = all([check_table_exists(table) for table in tables_to_check])
        
        if not tables_exist:
            logger.info("示范学习相关表不存在，开始创建...")
            if not create_demonstration_tables():
                logger.error("创建示范学习表失败")
                return False
        else:
            logger.info("示范学习相关表已存在")
        
        # 创建示例示范任务
        tasks_created = create_example_demonstration_tasks(db)
        
        # 提交所有更改
        db.commit()
        
        logger.info(f"迁移完成！成功创建了 {tasks_created} 个示例示范任务")
        
        # 打印统计信息
        total_demonstrations = db.query(Demonstration).count()
        total_frames = db.query(DemonstrationFrame).count()
        total_camera_frames = db.query(CameraFrame).count()
        total_tasks = db.query(DemonstrationTask).count()
        total_training_results = db.query(TrainingResult).count()
        
        logger.info(f"数据库统计:")
        logger.info(f"  示范总数: {total_demonstrations}")
        logger.info(f"  示范帧总数: {total_frames}")
        logger.info(f"  相机帧总数: {total_camera_frames}")
        logger.info(f"  示范任务总数: {total_tasks}")
        logger.info(f"  训练结果总数: {total_training_results}")
        
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
    logger.info("Self AGI 示范学习数据库迁移工具")
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