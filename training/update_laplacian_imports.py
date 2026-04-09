#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉普拉斯模块导入路径更新脚本

功能：
1. 搜索项目中所有导入旧拉普拉斯模块的文件
2. 更新导入路径到新的模块结构
3. 提供预览和确认机制
4. 备份原始文件

安全第一：创建备份，提供回滚方案
"""

import os
import re
import shutil
from typing import List, Tuple, Optional
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 旧模块到新模块的映射
IMPORT_MAPPINGS = {
    # 旧模块路径 -> 新模块路径
    "training.laplacian_regularization": "training.laplacian.core.regularization",
    "training.laplacian_enhanced_training": [
        "training.laplacian.utils.config",  # 配置类
        "training.laplacian.models.pinn",  # PINN模型
        "training.laplacian.models.cnn",  # CNN模型
        "training.laplacian.optimizers.laplacian_optimizer",  # 优化器
    ],
    "training.laplacian_integration": "training.laplacian.integration.framework",
    "training.laplacian_benchmark": "training.laplacian.benchmarks.performance",
    # 完整版）
    "LaplacianRegularization": "LaplacianRegularization",  # 类名不变，模块路径变
    "RegularizationConfig": "RegularizationConfig",
    "LaplacianEnhancedTrainingConfig": "LaplacianEnhancedTrainingConfig",
    "LaplacianEnhancedPINN": "LaplacianEnhancedPINN",
    "LaplacianEnhancedCNN": "LaplacianEnhancedCNN",
    "LaplacianEnhancedOptimizer": "LaplacianEnhancedOptimizer",
    "LaplacianIntegrationFramework": "LaplacianIntegrationFramework",
    "LaplacianIntegrationConfig": "LaplacianIntegrationConfig",
    "integrate_laplacian_with_training": "integrate_laplacian_with_training",
    "LaplacianBenchmark": "LaplacianBenchmark",
    "BenchmarkResult": "BenchmarkResult",
}

# 导入语句的正则表达式模式
IMPORT_PATTERNS = [
    # from module import class1, class2
    re.compile(r"^\s*from\s+([\w\.]+)\s+import\s+(.+)$"),
    # import module
    re.compile(r"^\s*import\s+([\w\.]+)(?:\s+as\s+\w+)?$"),
    # import module as alias
    re.compile(r"^\s*import\s+([\w\.]+)\s+as\s+(\w+)$"),
]


def find_files_with_old_imports(root_dir: Path) -> List[Path]:
    """查找包含旧拉普拉斯模块导入的文件"""

    python_files = []

    # 排除的目录
    exclude_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".trae"}

    # 排除的文件
    exclude_files = {
        "laplacian_regularization.py",
        "laplacian_enhanced_training.py",
        "laplacian_integration.py",
        "laplacian_benchmark.py",
        "update_laplacian_imports.py",  # 本脚本
    }

    logger.info(f"搜索目录: {root_dir}")

    for root, dirs, files in os.walk(root_dir):
        # 跳过排除的目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if not file.endswith(".py"):
                continue

            if file in exclude_files:
                continue

            file_path = Path(root) / file

            # 检查文件是否包含旧模块名称
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                    # 检查是否包含任何旧模块名称
                    for old_module in [
                        "laplacian_regularization",
                        "laplacian_enhanced_training",
                        "laplacian_integration",
                        "laplacian_benchmark",
                    ]:
                        if old_module in content:
                            python_files.append(file_path)
                            break

            except Exception as e:
                logger.warning(f"无法读取文件 {file_path}: {e}")

    logger.info(f"找到 {len(python_files)} 个可能包含旧导入的文件")
    return python_files


def parse_import_statement(line: str) -> Optional[Tuple[str, str, List[str]]]:
    """解析导入语句，返回（模块路径，导入类型，导入对象列表）"""

    for pattern in IMPORT_PATTERNS:
        match = pattern.match(line)
        if match:
            if pattern.pattern.startswith("^\\s*from"):
                # from module import class1, class2
                module_path = match.group(1)
                imports_raw = match.group(2)

                # 解析导入的对象列表
                imports = []
                current = ""
                paren_depth = 0

                for char in imports_raw:
                    if char == "(":
                        paren_depth += 1
                        current += char
                    elif char == ")":
                        paren_depth -= 1
                        current += char
                    elif char == "," and paren_depth == 0:
                        imports.append(current.strip())
                        current = ""
                    else:
                        current += char

                if current:
                    imports.append(current.strip())

                return module_path, "from_import", imports

            elif " as " in pattern.pattern:
                # import module as alias
                module_path = match.group(1)
                alias = match.group(2)
                return module_path, "import_as", [alias]

            else:
                # import module
                module_path = match.group(1)
                return module_path, "import", []

    return None  # 返回None


def update_import_line(line: str, file_path: Path) -> Tuple[str, bool]:
    """更新单行导入语句"""

    original_line = line
    result = parse_import_statement(line)

    if not result:
        return line, False

    module_path, import_type, imports = result

    # 检查是否是旧模块路径
    old_modules = [
        "training.laplacian_regularization",
        "training.laplacian_enhanced_training",
        "training.laplacian_integration",
        "training.laplacian_benchmark",
    ]

    updated = False

    for old_module in old_modules:
        if module_path == old_module or module_path.startswith(old_module + "."):
            # 找到旧模块导入
            logger.debug(f"在 {file_path.name} 中找到旧模块导入: "
                         f"{line.strip()}")

            if old_module == "training.laplacian_regularization":
                # 更新为新的正则化模块路径
                new_module = "training.laplacian.core.regularization"

                if import_type == "from_import":
                    # from training.laplacian_regularization import \
                    #     LaplacianRegularization
                    line = line.replace(old_module, new_module)
                    updated = True

                elif import_type == "import":
                    # import training.laplacian_regularization
                    # 需要更新为新的模块结构
                    line = ("# 已迁移到新模块结构: "
                            f"{new_module}\n")
                    line += f"# import {old_module}\n"
                    updated = True

            elif old_module == "training.laplacian_enhanced_training":
                # 这是一个复合模块，需要更复杂的处理
                # 完整处理：提供导入建议
                if import_type == "from_import":
                    # 检查导入的具体类
                    for imported in imports:
                        if imported in [
                            "LaplacianEnhancedTrainingConfig",
                            "UnifiedLaplacianConfig",
                        ]:
                            # 这些类现在在 utils.config 中
                            new_module = "training.laplacian.utils.config"
                            line = line.replace(old_module, new_module)
                            updated = True
                            break

                        elif imported in ["LaplacianEnhancedPINN"]:
                            # PINN模型在 models.pinn 中
                            new_module = "training.laplacian.models.pinn"
                            line = line.replace(old_module, new_module)
                            updated = True
                            break

                        elif imported in ["LaplacianEnhancedCNN"]:
                            # CNN模型在 models.cnn 中
                            new_module = "training.laplacian.models.cnn"
                            line = line.replace(old_module, new_module)
                            updated = True
                            break

                        elif imported in ["LaplacianEnhancedOptimizer"]:
                            # 优化器在 optimizers 中
                            new_module = (
                                "training.laplacian.optimizers.laplacian_optimizer"
                            )
                            line = line.replace(old_module, new_module)
                            updated = True
                            break

                    if not updated:
                        # 没有匹配的具体类，添加注释建议
                        comment = f"# 注意: {old_module} 已拆分为多个子模块\n"
                        comment += "# 请根据具体需要更新导入路径:\n"
                        comment += "# - 配置类: training.laplacian.utils.config\n"
                        comment += "# - PINN模型: training.laplacian.models.pinn\n"
                        comment += "# - CNN模型: training.laplacian.models.cnn\n"
                        comment += ("# - 优化器: "
                                    "training.laplacian.optimizers."
                                    "laplacian_optimizer\n")
                        line = comment + line
                        updated = True

            elif old_module == "training.laplacian_integration":
                new_module = "training.laplacian.integration.framework"
                line = line.replace(old_module, new_module)
                updated = True

            elif old_module == "training.laplacian_benchmark":
                new_module = "training.laplacian.benchmarks.performance"
                line = line.replace(old_module, new_module)
                updated = True

            if updated:
                logger.info(f"更新导入: {original_line.strip()} -> {line.strip()}")
                break

    return line, updated


def update_file_imports(file_path: Path, backup: bool = True) -> bool:
    """更新文件的导入语句"""

    logger.info(f"处理文件: {file_path}")

    # 创建备份
    backup_path = None
    if backup:
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        try:
            shutil.copy2(file_path, backup_path)
            logger.debug(f"创建备份: {backup_path}")
        except Exception as e:
            logger.error(f"无法创建备份 {backup_path}: {e}")
            return False

    # 读取文件内容
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"无法读取文件 {file_path}: {e}")
        return False

    # 更新每一行
    updated_lines = []
    total_updates = 0

    for i, line in enumerate(lines):
        updated_line, updated = update_import_line(line, file_path)
        updated_lines.append(updated_line)

        if updated:
            total_updates += 1

    # 如果没有更新，跳过写入
    if total_updates == 0:
        logger.debug(f"文件 {file_path.name} 没有需要更新的导入")
        if backup_path and backup_path.exists():
            os.remove(backup_path)  # 删除不必要的备份
        return False

    # 写入更新后的内容
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)

        logger.info(f"成功更新 {file_path.name}: {total_updates} 处修改")
        return True

    except Exception as e:
        logger.error(f"无法写入文件 {file_path}: {e}")

        # 恢复备份
        if backup_path and backup_path.exists():
            try:
                shutil.copy2(backup_path, file_path)
                logger.info(f"已从备份恢复文件: {file_path}")
            except Exception as restore_error:
                logger.error(f"无法恢复备份: {restore_error}")

        return False


def generate_migration_report(updated_files: List[Path]) -> None:
    """生成迁移报告"""

    report_path = PROJECT_ROOT / "laplacian_migration_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 拉普拉斯模块导入路径迁移报告\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 概述\n\n")
        f.write(f"已更新 {len(updated_files)} 个文件的导入路径。\n\n")

        f.write("## 迁移映射\n\n")
        f.write("| 旧模块路径 | 新模块路径 | 说明 |\n")
        f.write("|------------|------------|------|\n")
        f.write(
            "| `training.laplacian_regularization` | "
            "`training.laplacian.core.regularization` | 拉普拉斯正则化组件 |\n"
        )
        f.write(
            "| `training.laplacian_enhanced_training` | "
            "`training.laplacian.utils.config`<br>`training.laplacian.models.pinn`<br>"
            "`training.laplacian.models.cnn`<br>"
            "`training.laplacian.optimizers.laplacian_optimizer` "
            "| 拆分为多个子模块 |\n"
        )
        f.write(
            "| `training.laplacian_integration` | "
            "`training.laplacian.integration.framework` | 集成框架 |\n"
        )
        f.write(
            "| `training.laplacian_benchmark` | "
            "`training.laplacian.benchmarks.performance` | 性能基准测试 |\n\n"
        )

        f.write("## 更新的文件\n\n")
        for file_path in updated_files:
            rel_path = file_path.relative_to(PROJECT_ROOT)
            f.write(f"- `{rel_path}`\n")

        f.write("\n## 使用建议\n\n")
        f.write("1. **测试更新后的代码**: 运行相关测试以确保功能正常\n")
        f.write("2. **更新其他文件**: 如果发现还有未更新的导入，手动更新\n")
        f.write("3. **清理旧文件**: 确认所有引用都已更新后，可以考虑删除旧模块文件:\n")
        f.write("   - `training/laplacian_regularization.py`\n")
        f.write("   - `training/laplacian_enhanced_training.py`\n")
        f.write("   - `training/laplacian_integration.py`\n")
        f.write("   - `training/laplacian_benchmark.py`\n")
        f.write("\n4. **从新模块导入**: 后续开发请使用新的模块路径\n")

    logger.info(f"迁移报告已生成: {report_path}")


def main():
    """主函数"""

    print("=" * 60)
    print("拉普拉斯模块导入路径迁移工具")
    print("=" * 60)
    print()
    print("本工具将自动更新项目中旧拉普拉斯模块的导入路径到新结构。")
    print()

    # 确认操作
    response = input("是否继续？(y/N): ").strip().lower()
    if response != "y":
        print("操作已取消。")
        return

    print()
    print("正在搜索包含旧导入的文件...")

    # 查找文件
    files_to_update = find_files_with_old_imports(PROJECT_ROOT)

    if not files_to_update:
        print("未找到需要更新的文件。")
        return

    print(f"找到 {len(files_to_update)} 个需要检查的文件:")
    for file_path in files_to_update[:10]:  # 显示前10个
        rel_path = file_path.relative_to(PROJECT_ROOT)
        print(f"  - {rel_path}")

    if len(files_to_update) > 10:
        print(f"  ... 还有 {len(files_to_update) - 10} 个文件")

    print()

    # 预览模式
    preview = input("是否先预览更改而不实际修改？(y/N): ").strip().lower() == "y"

    if preview:
        print()
        print("预览模式 - 只显示更改，不修改文件")
        print("-" * 60)

        for file_path in files_to_update:
            print(f"\n文件: {file_path.relative_to(PROJECT_ROOT)}")
            print("-" * 40)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for i, line in enumerate(lines, 1):
                    updated_line, updated = update_import_line(line, file_path)
                    if updated:
                        print(f"行 {i}:")
                        print(f"  原: {line.rstrip()}")
                        print(f"  新: {updated_line.rstrip()}")
                        print()

            except Exception as e:
                print(f"  读取文件失败: {e}")

        print()
        print("预览完成。")

        # 询问是否应用更改
        apply = input("是否应用上述更改？(y/N): ").strip().lower() == "y"
        if not apply:
            print("操作已取消。")
            return

    # 实际更新文件
    print()
    print("开始更新文件...")
    print("-" * 60)

    updated_files = []

    for file_path in files_to_update:
        success = update_file_imports(file_path, backup=True)
        if success:
            updated_files.append(file_path)

    print()
    print("-" * 60)
    print(f"更新完成！共更新 {len(updated_files)} 个文件。")

    # 生成报告
    if updated_files:
        generate_migration_report(updated_files)

    print()
    print("重要提示:")
    print("1. 请运行相关测试以验证更改是否正常")
    print("2. 备份文件已保存为 .bak 扩展名，必要时可恢复")
    print("3. 查看生成的迁移报告以获取详细信息")
    print()
    print("=" * 60)


if __name__ == "__main__":
    import time

    main()
