"""
认证数据库迁移脚本
为现有用户表添加双因素认证和角色字段
创建密码重置和邮箱验证令牌表

此脚本执行以下操作：
1. 为用户表添加双因素认证相关字段
2. 为用户表添加角色字段
3. 创建密码重置令牌表
4. 创建邮箱验证令牌表

使用方法：
python migrate_auth_tables.py
"""

import sys
import os
import logging
from sqlalchemy import inspect, text
from datetime import datetime

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # 父目录是项目根目录
sys.path.insert(0, project_root)

from backend.core.database import Base, engine, SessionLocal
from backend.db_models.user import User, PasswordResetToken, EmailVerificationToken

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


def check_column_exists(table_name, column_name):
    """检查表中的列是否存在"""
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    return any(col['name'] == column_name for col in columns)


def migrate_user_table():
    """迁移用户表，添加缺失的字段"""
    try:
        with engine.connect() as conn:
            # 检查用户表是否存在
            if not check_table_exists('users'):
                logger.error("用户表不存在，请先运行主应用以创建表")
                return False
            
            # 检查并添加双因素认证字段
            columns_to_add = [
                ('two_factor_enabled', 'BOOLEAN DEFAULT FALSE'),
                ('two_factor_method', 'VARCHAR(20) DEFAULT \'email\''),
                ('totp_secret', 'VARCHAR(32)'),
                ('backup_codes', 'TEXT'),
                ('role', 'VARCHAR(20) DEFAULT \'user\''),
            ]
            
            for column_name, column_type in columns_to_add:
                if not check_column_exists('users', column_name):
                    logger.info(f"为用户表添加列: {column_name} {column_type}")
                    try:
                        conn.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {column_type}"))
                        conn.commit()
                        logger.info(f"成功添加列 {column_name}")
                    except Exception as e:
                        logger.error(f"添加列 {column_name} 失败: {e}")
                        # 继续尝试其他列
                else:
                    logger.info(f"列 {column_name} 已存在")
            
            return True
    except Exception as e:
        logger.error(f"迁移用户表失败: {e}")
        return False


def create_token_tables():
    """创建令牌表"""
    try:
        # 导入所有需要的模型以确保它们注册到Base.metadata
        from backend.db_models.user import PasswordResetToken, EmailVerificationToken
        
        # 创建令牌表
        tables_to_create = [
            ('password_reset_tokens', PasswordResetToken.__table__),
            ('email_verification_tokens', EmailVerificationToken.__table__),
        ]
        
        for table_name, table_obj in tables_to_create:
            if not check_table_exists(table_name):
                logger.info(f"创建表: {table_name}")
                table_obj.create(bind=engine, checkfirst=True)
                logger.info(f"成功创建表 {table_name}")
            else:
                logger.info(f"表 {table_name} 已存在")
        
        return True
    except Exception as e:
        logger.error(f"创建令牌表失败: {e}")
        return False


def main():
    """主迁移函数"""
    logger.info("开始认证数据库迁移...")
    
    # 迁移用户表
    logger.info("迁移用户表...")
    if not migrate_user_table():
        logger.error("用户表迁移失败")
        return False
    
    # 创建令牌表
    logger.info("创建令牌表...")
    if not create_token_tables():
        logger.error("令牌表创建失败")
        return False
    
    logger.info("认证数据库迁移完成")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)