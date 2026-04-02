"""
数据库模块
包含数据库引擎、会话工厂和基础模型
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config import Config

# 基础模型
Base = declarative_base()

# 注意：数据库模型不在这个模块中导入，以避免循环导入
# 模型应在使用前从backend.db_models导入
# 例如：在main.py的initialize_systems()函数中导入

# 数据库引擎
engine = create_engine(Config.DATABASE_URL)

# 会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 导出
__all__ = ["Base", "engine", "SessionLocal"]