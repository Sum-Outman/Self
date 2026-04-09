#!/usr/bin/env python3
"""
Self AGI 数据库迁移工具
创建所有数据库表并运行迁移脚本
"""

from sqlalchemy import inspect
from core.database import Base, engine
import sys
import os
import logging

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def import_all_models():
    """导入所有数据库模型以确保它们注册到Base.metadata"""
    logger.info("导入所有数据库模型...")

    # 用户相关模型

    # AGI相关模型

    # 记忆相关模型

    # 知识库相关模型

    # 演示相关模型

    # 机器人相关模型

    # 机器人市场相关模型

    logger.info("所有数据库模型导入成功")


def get_all_table_names():
    """获取所有已注册的表名"""
    return list(Base.metadata.tables.keys())


def check_existing_tables():
    """检查数据库中已存在的表"""
    inspector = inspect(engine)
    return inspector.get_table_names()


def create_all_tables():
    """创建所有数据库表"""
    logger.info("创建所有数据库表...")
    try:
        import_all_models()
        Base.metadata.create_all(bind=engine)

        # 验证表创建
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        registered_tables = get_all_table_names()

        # 检查缺失的表
        missing_tables = [t for t in registered_tables if t not in existing_tables]

        if missing_tables:
            logger.error(f"创建表后仍然缺失的表: {missing_tables}")
            return False

        logger.info(f"成功创建了 {len(existing_tables)} 个表")
        logger.info(f"数据库表列表: {existing_tables}")
        return True

    except Exception as e:
        logger.error(f"创建数据库表失败: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def run_migration_scripts():
    """运行现有的迁移脚本"""
    logger.info("运行迁移脚本...")

    migration_scripts = [
        "migrate_auth_tables.py",
        "migrate_demonstration_tables.py",
        "migrate_robot_tables.py",
        "migrate_market_tables.py",
    ]

    success_count = 0
    for script_name in migration_scripts:
        script_path = os.path.join(script_dir, script_name)
        if os.path.exists(script_path):
            try:
                logger.info(f"运行迁移脚本: {script_name}")
                # 动态导入并执行迁移脚本
                script_module = __import__(script_name.replace(".py", ""))
                # 假设每个脚本都有一个main()函数
                if hasattr(script_module, "main"):
                    script_module.main()
                success_count += 1
                logger.info(f"迁移脚本 {script_name} 执行成功")
            except Exception as e:
                logger.error(f"运行迁移脚本 {script_name} 失败: {e}")
        else:
            logger.warning(f"迁移脚本不存在: {script_path}")

    logger.info(f"成功运行 {success_count}/{len(migration_scripts)} 个迁移脚本")
    return success_count == len(migration_scripts)


def verify_database_integrity():
    """验证数据库完整性"""
    logger.info("验证数据库完整性...")

    try:
        # 检查连接
        with engine.connect() as conn:
            conn.execute("SELECT 1")

        # 检查表结构
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        registered_tables = get_all_table_names()

        # 检查表是否匹配
        missing_in_db = [t for t in registered_tables if t not in existing_tables]
        missing_in_metadata = [t for t in existing_tables if t not in registered_tables]

        if missing_in_db:
            logger.warning(f"数据库中缺失的表: {missing_in_db}")

        if missing_in_metadata:
            logger.warning(f"已注册元数据中缺失的表: {missing_in_metadata}")

        if not missing_in_db and not missing_in_metadata:
            logger.info("数据库完整性验证通过")
            return True
        else:
            logger.warning("数据库完整性验证发现问题")
            return False

    except Exception as e:
        logger.error(f"数据库完整性验证失败: {e}")
        return False


def initialize_sample_data():
    """初始化示例数据（可选）"""
    logger.info("初始化示例数据...")

    try:
        from core.database import SessionLocal
        from db_models.user import User

        db = SessionLocal()
        try:
            # 检查是否有用户
            user_count = db.query(User).count()
            if user_count == 0:
                logger.info("数据库中没有用户，跳过示例数据初始化")
            else:
                logger.info(f"数据库中有 {user_count} 个用户")
        finally:
            db.close()

        logger.info("示例数据检查完成")
        return True
    except Exception as e:
        logger.error(f"初始化示例数据失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Self AGI 数据库迁移工具")
    logger.info("=" * 60)

    # 检查当前数据库状态
    logger.info("检查当前数据库状态...")
    existing_tables = check_existing_tables()
    logger.info(f"现有表数量: {len(existing_tables)}")

    if existing_tables:
        logger.info(f"现有表: {existing_tables}")

    # 导入所有模型以获取已注册的表
    import_all_models()
    registered_tables = get_all_table_names()
    logger.info(f"已注册的表数量: {len(registered_tables)}")

    # 检查是否需要创建表
    missing_tables = [t for t in registered_tables if t not in existing_tables]

    if missing_tables:
        logger.info(f"发现 {len(missing_tables)} 个缺失的表")

        # 确认是否继续
        logger.info("是否创建缺失的表？[Y/n]")
        response = input().strip().lower()

        if response not in ["", "y", "yes"]:
            logger.info("用户取消操作")
            return 1

        # 创建所有表
        if not create_all_tables():
            logger.error("创建表失败")
            return 1
    else:
        logger.info("所有表已存在，无需创建")

    # 运行迁移脚本
    if not run_migration_scripts():
        logger.warning("部分迁移脚本执行失败")

    # 验证数据库完整性
    verify_database_integrity()

    # 初始化示例数据（可选）
    logger.info("是否初始化示例数据？[y/N]")
    response = input().strip().lower()
    if response in ["y", "yes"]:
        initialize_sample_data()

    logger.info("=" * 60)
    logger.info("数据库迁移完成")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"迁移过程中发生错误: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)
