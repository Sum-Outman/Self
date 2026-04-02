"""
机器人市场数据库迁移脚本
创建机器人市场相关表并添加示例数据

使用方法：
python migrate_market_tables.py
"""

import sys
import os
import logging
from sqlalchemy import inspect
from datetime import datetime, timedelta, timezone
import random

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.database import Base, engine, SessionLocal
from backend.db_models.robot_market import (
    RobotMarketListing,
    RobotMarketRating,
    RobotMarketComment,
    RobotMarketDownload,
    RobotMarketFavorite,
    RobotMarketStatus,
    RobotMarketCategory
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


def create_market_tables():
    """创建机器人市场相关表"""
    try:
        # 导入所有需要的模型以确保它们注册到Base.metadata
        from backend.db_models.robot_market import (
            RobotMarketListing,
            RobotMarketRating,
            RobotMarketComment,
            RobotMarketDownload,
            RobotMarketFavorite
        )
        
        # 创建表
        logger.info("创建机器人市场相关数据库表...")
        Base.metadata.create_all(bind=engine, tables=[
            RobotMarketListing.__table__,
            RobotMarketRating.__table__,
            RobotMarketComment.__table__,
            RobotMarketDownload.__table__,
            RobotMarketFavorite.__table__,
        ])
        
        # 验证表是否创建成功
        inspector = inspect(engine)
        tables_created = [
            RobotMarketListing.__tablename__,
            RobotMarketRating.__tablename__,
            RobotMarketComment.__tablename__,
            RobotMarketDownload.__tablename__,
            RobotMarketFavorite.__tablename__,
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


def create_example_market_listings(db):
    """创建示例机器人市场列表"""
    try:
        # 获取所有机器人和用户
        robots = db.query(Robot).filter(Robot.is_public == True).all()
        users = db.query(User).all()
        
        if not robots or not users:
            logger.warning("没有可用的公共机器人或用户，无法创建示例市场列表")
            return 0
        
        logger.info(f"找到 {len(robots)} 个公共机器人和 {len(users)} 个用户")
        
        # 示例列表配置
        example_listings = [
            {
                "title": "NAO人形机器人完整配置",
                "description": "完整的NAO v6人形机器人配置，包含所有关节和传感器设置，适用于研究和教育用途。",
                "category": RobotMarketCategory.HUMANOID,
                "tags": ["nao", "humanoid", "education", "research"],
                "version": "1.0.0",
                "changelog": "初始版本发布",
                "license_type": "Creative Commons Attribution 4.0",
                "is_featured": True,
                "is_verified": True
            },
            {
                "title": "TurtleBot3移动机器人URDF",
                "description": "TurtleBot3 Burger和Waffle模型的完整URDF文件，包含激光雷达和摄像头配置。",
                "category": RobotMarketCategory.MOBILE,
                "tags": ["turtlebot3", "mobile", "ros", "urdf"],
                "version": "2.0.1",
                "changelog": "修复了URDF中的碰撞检测问题",
                "license_type": "Apache 2.0",
                "is_featured": True,
                "is_verified": True
            },
            {
                "title": "Universal Robots UR5机械臂配置",
                "description": "UR5协作机器人的完整配置，包含运动学和动力学参数，适用于工业自动化。",
                "category": RobotMarketCategory.MANIPULATOR,
                "tags": ["ur5", "manipulator", "industrial", "collaborative"],
                "version": "1.2.0",
                "changelog": "添加了力控参数",
                "license_type": "MIT",
                "is_featured": False,
                "is_verified": True
            },
            {
                "title": "DJI Phantom 4无人机模型",
                "description": "DJI Phantom 4无人机的精确3D模型和飞行控制器配置，适用于仿真和开发。",
                "category": RobotMarketCategory.AERIAL,
                "tags": ["dji", "drone", "aerial", "simulation"],
                "version": "1.1.0",
                "changelog": "更新了空气动力学参数",
                "license_type": "GPL v3",
                "is_featured": True,
                "is_verified": False
            },
            {
                "title": "Baxter研究机器人配置",
                "description": "Rethink Robotics Baxter机器人的完整配置，包含双臂协调控制和传感器融合设置。",
                "category": RobotMarketCategory.RESEARCH,
                "tags": ["baxter", "research", "dual-arm", "manipulation"],
                "version": "2.0.0",
                "changelog": "添加了示教编程功能",
                "license_type": "BSD 3-Clause",
                "is_featured": True,
                "is_verified": True
            },
            {
                "title": "Pepper社交机器人情感引擎",
                "description": "SoftBank Pepper机器人的情感引擎和社交交互配置，包含面部表情和语音识别设置。",
                "category": RobotMarketCategory.SERVICE,
                "tags": ["pepper", "social", "service", "emotion"],
                "version": "1.3.0",
                "changelog": "优化了情感识别算法",
                "license_type": "MIT",
                "is_featured": True,
                "is_verified": True
            }
        ]
        
        listings_created = 0
        
        for i, listing_config in enumerate(example_listings):
            # 选择机器人和用户
            robot = robots[i % len(robots)]
            user = users[i % len(users)]
            
            # 检查是否已存在相同标题的列表
            existing_listing = db.query(RobotMarketListing).filter(
                RobotMarketListing.title == listing_config["title"],
                RobotMarketListing.owner_id == user.id
            ).first()
            
            if existing_listing:
                logger.debug(f"列表 {listing_config['title']} 已存在")
                continue
            
            # 创建列表
            published_at = datetime.now(timezone.utc) - timedelta(days=random.randint(1, 365))
            
            listing = RobotMarketListing(
                robot_id=robot.id,
                title=listing_config["title"],
                description=listing_config["description"],
                owner_id=user.id,
                category=listing_config["category"],
                tags=listing_config["tags"],
                version=listing_config["version"],
                changelog=listing_config["changelog"],
                license_type=listing_config["license_type"],
                status=RobotMarketStatus.APPROVED,
                is_featured=listing_config["is_featured"],
                is_verified=listing_config["is_verified"],
                download_count=random.randint(50, 1000),
                view_count=random.randint(200, 5000),
                rating_count=random.randint(10, 200),
                average_rating=random.uniform(3.5, 5.0),
                created_at=published_at - timedelta(days=random.randint(1, 30)),
                updated_at=published_at,
                published_at=published_at,
                reviewed_at=published_at - timedelta(hours=1),
                reviewer_id=users[0].id if users else None
            )
            
            db.add(listing)
            listings_created += 1
            
            # 创建评分
            for j in range(min(5, listing_config["rating_count"])):
                rating_user = users[(i + j) % len(users)]
                rating = RobotMarketRating(
                    listing_id=listing.id,
                    user_id=rating_user.id,
                    rating=random.randint(3, 5),
                    comment=f"这个配置很好用，帮助我快速开始了我的项目。",
                    ease_of_use=random.randint(3, 5),
                    documentation_quality=random.randint(3, 5),
                    performance=random.randint(3, 5),
                    reliability=random.randint(3, 5),
                    created_at=published_at + timedelta(days=random.randint(1, 30))
                )
                db.add(rating)
            
            # 创建评论
            comments = [
                "这个机器人配置非常详细，文档也很完整，让我很快就上手了。",
                "URDF文件质量很高，直接导入Gazebo就能用，节省了很多时间。",
                "配置中的传感器校准数据很准确，让我的仿真结果更接近真实情况。",
                "希望作者能添加更多示例代码和教程，对新手会更友好。",
                "许可证选择很合理，让我可以在商业项目中使用，非常感谢！",
                "这个配置帮我完成了毕业设计，非常感谢作者的分享。",
                "配置更新很及时，修复了我遇到的好几个问题。",
                "希望未来能支持更多版本的ROS。"
            ]
            
            for j in range(min(3, len(comments))):
                comment_user = users[(i + j + 1) % len(users)]
                comment = RobotMarketComment(
                    listing_id=listing.id,
                    user_id=comment_user.id,
                    content=comments[j],
                    created_at=published_at + timedelta(days=random.randint(1, 60))
                )
                db.add(comment)
            
            # 创建下载记录
            for j in range(min(10, listing.download_count)):
                download_user = users[(i + j + 2) % len(users)]
                download = RobotMarketDownload(
                    listing_id=listing.id,
                    user_id=download_user.id,
                    download_type=random.choice(["config", "urdf", "full"]),
                    file_size=random.randint(1024, 10485760),  # 1KB到10MB
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    ip_address=f"192.168.1.{random.randint(1, 255)}",
                    created_at=published_at + timedelta(days=random.randint(1, 90))
                )
                db.add(download)
            
            # 创建收藏记录
            for j in range(min(5, listing.rating_count)):
                favorite_user = users[(i + j + 3) % len(users)]
                favorite = RobotMarketFavorite(
                    listing_id=listing.id,
                    user_id=favorite_user.id,
                    folder=random.choice(["default", "favorites", "research"]),
                    notes="有用的机器人配置",
                    created_at=published_at + timedelta(days=random.randint(1, 120))
                )
                db.add(favorite)
        
        db.flush()
        logger.info(f"创建了 {listings_created} 个示例机器人市场列表")
        return listings_created
        
    except Exception as e:
        logger.error(f"创建示例市场列表失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


def migrate():
    """执行数据库迁移"""
    logger.info("开始机器人市场数据库迁移...")
    
    db = SessionLocal()
    try:
        # 检查机器人市场表是否存在
        tables_to_check = [
            "robot_market_listings",
            "robot_market_ratings",
            "robot_market_comments",
            "robot_market_downloads",
            "robot_market_favorites"
        ]
        
        tables_exist = all([check_table_exists(table) for table in tables_to_check])
        
        if not tables_exist:
            logger.info("机器人市场相关表不存在，开始创建...")
            if not create_market_tables():
                logger.error("创建机器人市场表失败")
                return False
        else:
            logger.info("机器人市场相关表已存在")
        
        # 创建示例机器人市场列表
        listings_created = create_example_market_listings(db)
        
        # 提交所有更改
        db.commit()
        
        logger.info(f"迁移完成！成功创建了 {listings_created} 个示例机器人市场列表")
        
        # 打印统计信息
        total_listings = db.query(RobotMarketListing).count()
        total_ratings = db.query(RobotMarketRating).count()
        total_comments = db.query(RobotMarketComment).count()
        total_downloads = db.query(RobotMarketDownload).count()
        total_favorites = db.query(RobotMarketFavorite).count()
        
        logger.info(f"数据库统计:")
        logger.info(f"  市场列表总数: {total_listings}")
        logger.info(f"  评分总数: {total_ratings}")
        logger.info(f"  评论总数: {total_comments}")
        logger.info(f"  下载记录总数: {total_downloads}")
        logger.info(f"  收藏记录总数: {total_favorites}")
        
        # 按状态统计列表
        for status in RobotMarketStatus:
            count = db.query(RobotMarketListing).filter(
                RobotMarketListing.status == status
            ).count()
            logger.info(f"  {status.value}: {count}")
        
        # 按分类统计列表
        for category in RobotMarketCategory:
            count = db.query(RobotMarketListing).filter(
                RobotMarketListing.category == category,
                RobotMarketListing.status == RobotMarketStatus.APPROVED
            ).count()
            logger.info(f"  {category.value}: {count}")
        
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
    logger.info("Self AGI 机器人市场数据库迁移工具")
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