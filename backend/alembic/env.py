from logging.config import fileConfig
import os
import sys

# 添加项目根目录到Python路径，以便导入后端模块
# alembic/env.py 位于 backend/alembic/env.py，所以需要上溯三级到项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# 导入后端数据库模型
try:
    from backend.core.database import Base
    from backend.core.config import Config
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保后端模块可访问。当前Python路径:")
    for p in sys.path:
        print(f"  {p}")
    raise

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 设置目标元数据，用于自动生成迁移
target_metadata = Base.metadata

# 其他配置值可以从config中获取
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    # 使用配置中的数据库URL，而不是alembic.ini中的硬编码值
    url = Config.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # 从配置获取数据库URL，覆盖alembic.ini中的设置
    configuration = config.get_section(config.config_ini_section, {})
    configuration['sqlalchemy.url'] = Config.DATABASE_URL
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()