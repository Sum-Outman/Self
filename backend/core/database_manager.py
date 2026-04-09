"""
数据库管理器 - 统一管理数据库配置、迁移和健康检查
修复重复的迁移脚本和不一致问题

功能：
1. 统一的数据库配置管理
2. 自动化迁移管理（替代多个独立的migrate_*.py脚本）
3. 数据库健康检查和修复
4. 生产环境安全验证
"""

from backend.core.database import Base, engine
from backend.core.config import Config
from alembic.command import upgrade
from alembic.config import Config as AlembicConfig
from sqlalchemy import text, inspect
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器 - 统一管理所有数据库操作"""

    def __init__(self, config: Config = None):
        """初始化数据库管理器

        参数：
            config: 配置对象，如果为None则使用默认Config
        """
        self.config = config or Config
        self.engine = engine
        self.metadata = Base.metadata

    def check_database_connection(self) -> Tuple[bool, str]:
        """检查数据库连接

        返回：
            (连接成功, 消息)
        """
        try:
            with self.engine.connect() as conn:
                # SQLite和SQLAlchemy的正确语法
                result = conn.execute(text("SELECT 1"))
                row = result.fetchone()
                if row and row[0] == 1:
                    return True, "数据库连接成功"
                else:
                    return False, "数据库连接测试返回异常结果"
        except Exception as e:
            return False, f"数据库连接失败: {e}"

    def check_table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

    def get_all_tables(self) -> List[str]:
        """获取所有表名"""
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """获取表结构"""
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        indexes = inspector.get_indexes(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)

        return {
            "name": table_name,
            "columns": columns,
            "indexes": indexes,
            "foreign_keys": foreign_keys,
        }

    def check_missing_tables(self) -> List[str]:
        """检查缺失的表（对比metadata）"""
        existing_tables = self.get_all_tables()
        expected_tables = list(self.metadata.tables.keys())

        missing_tables = []
        for table in expected_tables:
            if table not in existing_tables:
                missing_tables.append(table)

        return missing_tables

    def create_missing_tables(self) -> Tuple[bool, str]:
        """创建缺失的表"""
        try:
            missing_tables = self.check_missing_tables()
            if not missing_tables:
                return True, "所有表已存在，无需创建"

            logger.info(f"创建缺失的表: {missing_tables}")
            self.metadata.create_all(bind=self.engine)
            return True, f"成功创建 {len(missing_tables)} 个表"
        except Exception as e:
            return False, f"创建表失败: {e}"

    def run_alembic_migration(self, revision: str = "head") -> Tuple[bool, str]:
        """运行Alembic迁移

        参数：
            revision: 目标版本，默认为最新版本
        """
        try:
            # 获取alembic.ini路径
            alembic_ini_path = Path(__file__).parent.parent / "alembic.ini"
            if not alembic_ini_path.exists():
                return False, f"Alembic配置文件不存在: {alembic_ini_path}"

            # 创建Alembic配置
            alembic_cfg = AlembicConfig(str(alembic_ini_path))

            # 运行迁移
            upgrade(alembic_cfg, revision)
            return True, f"成功迁移到版本: {revision}"
        except Exception as e:
            return False, f"Alembic迁移失败: {e}"

    def check_alembic_current(self) -> Tuple[Optional[str], str]:
        """检查当前Alembic版本"""
        try:
            alembic_ini_path = Path(__file__).parent.parent / "alembic.ini"
            if not alembic_ini_path.exists():
                return None, "Alembic配置文件不存在"

            AlembicConfig(str(alembic_ini_path))

            # 获取当前版本
            from alembic.migration import MigrationContext

            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_rev = context.get_current_revision()

            return current_rev, "获取当前版本成功"
        except Exception as e:
            return None, f"获取当前版本失败: {e}"

    def cleanup_legacy_migration_scripts(self) -> Tuple[bool, str]:
        """清理遗留的独立迁移脚本

        移除多个独立的migrate_*.py脚本，统一使用Alembic
        """
        legacy_scripts = [
            "migrate_auth_tables.py",
            "migrate_demonstration_tables.py",
            "migrate_robot_tables.py",
            "migrate_market_tables.py",
        ]

        cleaned = []
        failed = []

        for script_name in legacy_scripts:
            script_path = Path(__file__).parent.parent / script_name
            if script_path.exists():
                try:
                    # 备份内容到日志
                    with open(script_path, "r", encoding="utf-8") as f:
                        f.read()
                    logger.info(f"备份遗留脚本 {script_name} 内容到日志")

                    # 删除文件
                    script_path.unlink()
                    cleaned.append(script_name)
                    logger.info(f"已删除遗留迁移脚本: {script_name}")
                except Exception as e:
                    failed.append(f"{script_name}: {e}")

        if cleaned:
            message = f"已清理 {len(cleaned)} 个遗留迁移脚本: {', '.join(cleaned)}"
            if failed:
                message += f"; 失败 {len(failed)} 个: {', '.join(failed)}"
            return True, message
        else:
            return False, "未找到遗留迁移脚本或清理失败"

    def validate_database_schema(self) -> Dict[str, Any]:
        """验证数据库架构完整性

        检查：
        1. 所有必需的表是否存在
        2. 表结构是否符合预期
        3. 索引和外键是否正确
        """
        validation_result = {
            "overall": True,
            "connection": {"status": False, "message": ""},
            "tables": {"missing": [], "extra": [], "details": {}},
            "alembic": {"current_revision": None, "message": ""},
            "recommendations": [],
        }

        # 1. 检查数据库连接
        connection_ok, connection_msg = self.check_database_connection()
        validation_result["connection"]["status"] = connection_ok
        validation_result["connection"]["message"] = connection_msg

        if not connection_ok:
            validation_result["overall"] = False
            validation_result["recommendations"].append("修复数据库连接问题")
            return validation_result

        # 2. 检查表
        existing_tables = self.get_all_tables()
        expected_tables = list(self.metadata.tables.keys())

        # 查找缺失的表
        missing_tables = []
        for table in expected_tables:
            if table not in existing_tables:
                missing_tables.append(table)

        # 查找额外的表（可能来自旧版本）
        extra_tables = []
        for table in existing_tables:
            if table not in expected_tables:
                extra_tables.append(table)

        validation_result["tables"]["missing"] = missing_tables
        validation_result["tables"]["extra"] = extra_tables

        # 详细表信息
        for table_name in existing_tables:
            try:
                schema = self.get_table_schema(table_name)
                validation_result["tables"]["details"][table_name] = {
                    "column_count": len(schema["columns"]),
                    "index_count": len(schema["indexes"]),
                    "foreign_key_count": len(schema["foreign_keys"]),
                }
            except Exception as e:
                validation_result["tables"]["details"][table_name] = {"error": str(e)}

        if missing_tables:
            validation_result["overall"] = False
            validation_result["recommendations"].append(
                f"创建缺失的表: {missing_tables}"
            )

        # 3. 检查Alembic版本
        current_rev, alembic_msg = self.check_alembic_current()
        validation_result["alembic"]["current_revision"] = current_rev
        validation_result["alembic"]["message"] = alembic_msg

        # 4. 生产环境特定检查
        if os.getenv("ENVIRONMENT") == "production":
            # 检查是否使用SQLite（生产环境不推荐）
            if self.config.DATABASE_URL.startswith("sqlite"):
                validation_result["overall"] = False
                validation_result["recommendations"].append(
                    "生产环境不推荐使用SQLite，建议切换到PostgreSQL或MySQL"
                )

            # 检查默认密码
            default_password_warnings = []
            if "your_mongodb_password" in self.config.MONGODB_URL:
                default_password_warnings.append("MongoDB使用默认密码")
            if "your_rabbitmq_password" in self.config.RABBITMQ_URL:
                default_password_warnings.append("RabbitMQ使用默认密码")

            if default_password_warnings:
                validation_result["overall"] = False
                validation_result["recommendations"].append(
                    f"生产环境安全警告: {'; '.join(default_password_warnings)}"
                )

        return validation_result

    def repair_database(self, auto_fix: bool = True) -> Dict[str, Any]:
        """修复数据库问题

        参数：
            auto_fix: 是否自动修复问题

        返回：
            修复结果报告
        """
        repair_result = {
            "overall": True,
            "actions": [],
            "errors": [],
            "warnings": [],
        }

        try:
            # 1. 验证数据库状态
            validation = self.validate_database_schema()

            if not validation["connection"]["status"]:
                repair_result["errors"].append(
                    f"数据库连接失败: {validation['connection']['message']}"
                )
                repair_result["overall"] = False
                return repair_result

            # 2. 创建缺失的表
            missing_tables = validation["tables"]["missing"]
            if missing_tables and auto_fix:
                success, message = self.create_missing_tables()
                if success:
                    repair_result["actions"].append(f"创建缺失的表: {message}")
                else:
                    repair_result["errors"].append(f"创建表失败: {message}")
                    repair_result["overall"] = False

            # 3. 运行Alembic迁移
            current_rev = validation["alembic"]["current_revision"]
            if current_rev is None or current_rev != "head":
                if auto_fix:
                    success, message = self.run_alembic_migration("head")
                    if success:
                        repair_result["actions"].append(f"Alembic迁移: {message}")
                    else:
                        repair_result["warnings"].append(f"Alembic迁移失败: {message}")
                else:
                    repair_result["warnings"].append(
                        "Alembic版本不是最新，建议运行迁移"
                    )

            # 4. 清理遗留脚本
            success, message = self.cleanup_legacy_migration_scripts()
            if success and "已清理" in message:
                repair_result["actions"].append(f"清理遗留脚本: {message}")

            # 5. 生产环境建议
            if os.getenv("ENVIRONMENT") == "production":
                # 检查并记录生产环境建议
                for recommendation in validation["recommendations"]:
                    if "生产环境" in recommendation:
                        repair_result["warnings"].append(recommendation)

            return repair_result

        except Exception as e:
            repair_result["errors"].append(f"数据库修复过程中出现异常: {e}")
            repair_result["overall"] = False
            return repair_result


def main():
    """数据库管理命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Self AGI数据库管理器")
    parser.add_argument("--check", action="store_true", help="检查数据库状态")
    parser.add_argument("--repair", action="store_true", help="修复数据库问题")
    parser.add_argument("--cleanup", action="store_true", help="清理遗留迁移脚本")
    parser.add_argument("--create-tables", action="store_true", help="创建缺失的表")
    parser.add_argument(
        "--migrate", action="store_true", help="运行Alembic迁移到最新版本"
    )
    parser.add_argument(
        "--auto-fix", action="store_true", help="自动修复问题（配合--repair使用）"
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    manager = DatabaseManager()

    if args.check:
        print("=== 数据库状态检查 ===")
        validation = manager.validate_database_schema()

        print(
            f"数据库连接: {'成功' if validation['connection']['status'] else '失败'}"
        )
        if not validation["connection"]["status"]:
            print(f"  消息: {validation['connection']['message']}")

        print("\n表检查:")
        print(f"  缺失的表: {validation['tables']['missing']}")
        print(f"  额外的表: {validation['tables']['extra']}")

        print(f"\nAlembic版本: {validation['alembic']['current_revision']}")
        print(f"  消息: {validation['alembic']['message']}")

        if validation["recommendations"]:
            print("\n建议:")
            for rec in validation["recommendations"]:
                print(f"  • {rec}")

        print(f"\n总体状态: {'正常' if validation['overall'] else '需要修复'}")

    elif args.repair:
        print("=== 数据库修复 ===")
        repair_result = manager.repair_database(auto_fix=args.auto_fix)

        if repair_result["actions"]:
            print("执行的操作:")
            for action in repair_result["actions"]:
                print(f"  完成: {action}")

        if repair_result["errors"]:
            print("错误:")
            for error in repair_result["errors"]:
                print(f"  错误: {error}")

        if repair_result["warnings"]:
            print("警告:")
            for warning in repair_result["warnings"]:
                print(f"  警告: {warning}")

        print(f"\n修复结果: {'成功' if repair_result['overall'] else '失败'}")

    elif args.cleanup:
        print("=== 清理遗留迁移脚本 ===")
        success, message = manager.cleanup_legacy_migration_scripts()
        if success:
            print(f"成功: {message}")
        else:
            print(f"失败: {message}")

    elif args.create_tables:
        print("=== 创建缺失的表 ===")
        success, message = manager.create_missing_tables()
        if success:
            print(f"成功: {message}")
        else:
            print(f"失败: {message}")

    elif args.migrate:
        print("=== 运行Alembic迁移 ===")
        success, message = manager.run_alembic_migration("head")
        if success:
            print(f"成功: {message}")
        else:
            print(f"失败: {message}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
