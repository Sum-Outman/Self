#!/usr/bin/env python3
"""
创建缺失的数据库表
"""

from sqlalchemy import inspect
from backend.core.database import Base, engine
import sys
import os
import logging

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)


# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def import_all_models():
    """导入所有数据库模型"""
    logger.info("导入所有数据库模型...")

    # 用户相关模型

    # AGI相关模型

    # 记忆相关模型

    # 知识库相关模型

    # 演示相关模型

    # 机器人相关模型

    # 机器人市场相关模型

    logger.info("所有数据库模型导入成功")


def main():
    """主函数"""
    logger.info("创建缺失的数据库表...")

    # 导入所有模型
    import_all_models()

    # 检查当前存在的表
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    logger.info(f"现有表数量: {len(existing_tables)}")

    # 获取所有已注册的表
    registered_tables = list(Base.metadata.tables.keys())
    logger.info(f"已注册的表数量: {len(registered_tables)}")

    # 找出缺失的表
    missing_tables = [t for t in registered_tables if t not in existing_tables]

    if not missing_tables:
        logger.info("所有表已存在，无需创建")
        return True

    logger.info(f"发现 {len(missing_tables)} 个缺失的表:")
    for table in missing_tables:
        logger.info(f"  - {table}")

    # 创建缺失的表
    try:
        logger.info("创建缺失的表...")
        Base.metadata.create_all(
            bind=engine,
            tables=[Base.metadata.tables[table_name] for table_name in missing_tables],
        )

        # 验证创建结果
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        still_missing = [t for t in missing_tables if t not in existing_tables]

        if still_missing:
            logger.error(f"创建表后仍然缺失的表: {still_missing}")
            return False
        else:
            logger.info(f"成功创建了 {len(missing_tables)} 个缺失的表")
            return True

    except Exception as e:
        logger.error(f"创建表失败: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
