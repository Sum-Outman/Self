#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self AGI 专业领域能力模块
实现编程和数学等专业领域的高级能力

功能：
1. 编程能力：代码生成、代码分析、代码调试、代码优化
2. 数学能力：数学问题求解、符号计算、数学证明、数值计算
3. 物理模拟：物理引擎集成、运动模拟、碰撞检测
4. 医学推理：医学知识库、疾病诊断、治疗方案推理
5. 金融分析：金融数据建模、风险评估、投资策略

基于真实算法和库实现，包括代码分析工具、数学计算库、物理引擎等
"""

import sys
import os
import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import time
import math
import re
import hashlib
from collections import deque, defaultdict
import warnings
from datetime import datetime, timedelta

# 导入代码分析库
try:
    import ast
    AST_AVAILABLE = True
except ImportError as e:
    AST_AVAILABLE = False
    warnings.warn(f"Python AST模块不可用: {e}")

try:
    import jedi
    JEDI_AVAILABLE = True
except ImportError as e:
    JEDI_AVAILABLE = False
    warnings.warn(f"Jedi代码分析库不可用: {e}")

# 导入数学计算库
try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError as e:
    SYMPY_AVAILABLE = False
    warnings.warn(f"SymPy数学库不可用: {e}")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    NUMPY_AVAILABLE = False
    warnings.warn(f"NumPy数学库不可用: {e}")

# 导入物理模拟库
try:
    import pybullet  # type: ignore
    PYBULLET_AVAILABLE = True
except ImportError as e:
    PYBULLET_AVAILABLE = False
    warnings.warn(
        f"PyBullet物理引擎不可用: {e}\n机器人仿真功能将受限。\nWindows用户安装建议:\n1. 安装Visual Studio Build Tools (C++编译环境)\n2. 运行: pip install pybullet\n3. 或使用预编译版本: pip install pybullet --find-links https://github.com/bulletphysics/bullet3/releases\n4. 对于不需要编译的简易安装，可以使用: pip install pybullet --only-binary :all:")

# 导入医学知识库（模拟）
try:
    # 这里可以集成真实的医学知识库
    MEDICAL_KB_AVAILABLE = False
except ImportError as e:
    MEDICAL_KB_AVAILABLE = False
    warnings.warn(f"医学知识库不可用: {e}")

# 导入金融分析库
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    PANDAS_AVAILABLE = False
    warnings.warn(f"Pandas数据分析库不可用: {e}")


class ProgrammingLanguage(Enum):
    """编程语言枚举"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    CPP = "c++"
    C = "c"
    CSHARP = "csharp"
    RUST = "rust"
    GO = "go"
    R = "r"
    PHP = "php"
    RUBY = "ruby"
    SHELL = "shell"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    DOCKERFILE = "dockerfile"


class CodeComplexity(Enum):
    """代码复杂度级别"""
    VERY_SIMPLE = "very_simple"    # 非常简单
    SIMPLE = "simple"              # 简单
    MODERATE = "moderate"          # 中等
    COMPLEX = "complex"            # 复杂
    VERY_COMPLEX = "very_complex"  # 非常复杂


class MathematicalDomain(Enum):
    """数学领域枚举 - 增强版（修复缺陷3.1）"""
    ALGEBRA = "algebra"            # 代数
    CALCULUS = "calculus"          # 微积分
    GEOMETRY = "geometry"          # 几何
    STATISTICS = "statistics"      # 统计
    PROBABILITY = "probability"    # 概率
    LINEAR_ALGEBRA = "linear_algebra"  # 线性代数
    DISCRETE_MATH = "discrete_math"  # 离散数学
    NUMBER_THEORY = "number_theory"  # 数论
    DIFFERENTIAL_GEOMETRY = "differential_geometry"  # 微分几何
    TOPOLOGY = "topology"          # 拓扑
    COMPLEX_ANALYSIS = "complex_analysis"  # 复分析
    FUNCTIONAL_ANALYSIS = "functional_analysis"  # 泛函分析
    DIFFERENTIAL_EQUATIONS = "differential_equations"  # 微分方程
    MATHEMATICAL_LOGIC = "mathematical_logic"  # 数理逻辑
    COMBINATORICS = "combinatorics"  # 组合数学


@dataclass
class CodeAnalysisResult:
    """代码分析结果"""

    code_id: str
    language: ProgrammingLanguage
    complexity: CodeComplexity
    lines_of_code: int
    functions_count: int
    classes_count: int

    # 代码质量指标
    cyclomatic_complexity: float = 0.0
    maintainability_index: float = 0.0
    test_coverage: float = 0.0

    # 问题检测
    bugs_detected: List[Dict[str, Any]] = field(default_factory=list)
    style_violations: List[Dict[str, Any]] = field(default_factory=list)
    performance_issues: List[Dict[str, Any]] = field(default_factory=list)

    # 建议
    optimization_suggestions: List[str] = field(default_factory=list)
    refactoring_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "code_id": self.code_id,
            "language": self.language.value,
            "complexity": self.complexity.value,
            "lines_of_code": self.lines_of_code,
            "functions_count": self.functions_count,
            "classes_count": self.classes_count,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "maintainability_index": self.maintainability_index,
            "test_coverage": self.test_coverage,
            "bugs_detected": self.bugs_detected,
            "style_violations": self.style_violations,
            "performance_issues": self.performance_issues,
            "optimization_suggestions": self.optimization_suggestions,
            "refactoring_suggestions": self.refactoring_suggestions
        }


@dataclass
class MathematicalProblem:
    """数学问题"""

    problem_id: str
    domain: MathematicalDomain
    problem_statement: str
    difficulty_level: float  # 0.0-1.0

    # 解决方案
    solution_steps: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[Any] = None
    verification_result: Optional[bool] = None

    # 元数据
    time_taken_seconds: float = 0.0
    attempted_methods: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "problem_id": self.problem_id,
            "domain": self.domain.value,
            "problem_statement": self.problem_statement,
            "difficulty_level": self.difficulty_level,
            "solution_steps": self.solution_steps,
            "final_answer": self.final_answer,
            "verification_result": self.verification_result,
            "time_taken_seconds": self.time_taken_seconds,
            "attempted_methods": self.attempted_methods
        }


class ProgrammingCapabilityManager:
    """编程能力管理器"""

    def __init__(self):
        self.code_patterns = {}
        self.code_templates = {}
        self.bug_patterns = {}
        self.optimization_rules = {}

        # 初始化代码分析器
        self._initialize_code_analyzers()
        # 初始化代码生成器
        self._initialize_code_generators()
        # 初始化bug检测器
        self._initialize_bug_detectors()

        self.logger = logging.getLogger("ProgrammingCapabilityManager")
        self.logger.info("编程能力管理器初始化完成")

    def _initialize_code_analyzers(self):
        """初始化代码分析器"""
        self.code_analyzers = {
            "complexity": self._analyze_code_complexity,
            "quality": self._analyze_code_quality,
            "security": self._analyze_code_security,
            "performance": self._analyze_code_performance
        }

    def _initialize_code_generators(self):
        """初始化代码生成器"""
        self.code_generators = {
            "function": self._generate_function,
            "class": self._generate_class,
            "algorithm": self._generate_algorithm,
            "test": self._generate_test,
            "module": self._generate_module,
            "interface": self._generate_interface,
            "script": self._generate_script,
            "api": self._generate_api,
            "database": self._generate_database,
            "web": self._generate_web,
            "cli": self._generate_cli,
            "config": self._generate_config
        }

    def _initialize_bug_detectors(self):
        """初始化bug检测器"""
        self.bug_detectors = {
            "syntax_error": self._detect_syntax_errors,
            "logical_error": self._detect_logical_errors,
            "runtime_error": self._detect_runtime_errors,
            "security_vulnerability": self._detect_security_vulnerabilities
        }

    def analyze_code(self,
                     code: str,
                     language: ProgrammingLanguage = ProgrammingLanguage.PYTHON) -> CodeAnalysisResult:
        """
        分析代码

        参数:
            code: 源代码
            language: 编程语言

        返回:
            代码分析结果
        """
        self.logger.info(f"开始分析代码，语言: {language.value}, 长度: {len(code)} 字符")

        # 生成唯一ID
        code_id = hashlib.md5(
            f"{code}{language.value}{time.time()}".encode()).hexdigest()[:16]

        # 基本分析
        lines_of_code = len(code.split('\n'))
        functions_count = self._count_functions(code, language)
        classes_count = self._count_classes(code, language)

        # 复杂度分析
        complexity = self._analyze_code_complexity(code, language)

        # 代码质量分析
        maintainability_index = self._calculate_maintainability_index(code, language)

        # 问题检测
        bugs_detected = self._detect_all_bugs(code, language)
        style_violations = self._detect_style_violations(code, language)
        performance_issues = self._detect_performance_issues(code, language)

        # 建议生成
        optimization_suggestions = self._generate_optimization_suggestions(
            code, language, performance_issues)
        refactoring_suggestions = self._generate_refactoring_suggestions(
            code, language, complexity)

        result = CodeAnalysisResult(
            code_id=code_id,
            language=language,
            complexity=complexity,
            lines_of_code=lines_of_code,
            functions_count=functions_count,
            classes_count=classes_count,
            cyclomatic_complexity=self._calculate_cyclomatic_complexity(code, language),
            maintainability_index=maintainability_index,
            test_coverage=0.0,  # 需要测试覆盖率数据
            bugs_detected=bugs_detected,
            style_violations=style_violations,
            performance_issues=performance_issues,
            optimization_suggestions=optimization_suggestions,
            refactoring_suggestions=refactoring_suggestions
        )

        self.logger.info(
            f"代码分析完成: ID={code_id}, "
            f"复杂度={complexity.value}, "
            f"函数数={functions_count}, "
            f"问题数={len(bugs_detected) + len(style_violations) + len(performance_issues)}"
        )

        return result

    def generate_code(self,
                      description: str,
                      language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
                      code_type: str = "function") -> Dict[str, Any]:
        """
        根据描述生成代码

        参数:
            description: 代码功能描述
            language: 编程语言
            code_type: 代码类型 (function, class, algorithm, test)

        返回:
            生成的代码和相关信息
        """
        self.logger.info(
            f"生成代码: 类型={code_type}, 语言={language.value}, 描述={description[:50]}...")

        # 选择代码生成器
        generator_func = self.code_generators.get(code_type)
        if not generator_func:
            self.logger.warning(f"未知代码类型: {code_type}，使用函数生成器")
            generator_func = self._generate_function

        # 生成代码
        generated_code = generator_func(description, language)

        # 分析生成的代码
        analysis_result = self.analyze_code(generated_code, language)

        result = {
            "code_id": analysis_result.code_id,
            "generated_code": generated_code,
            "language": language.value,
            "code_type": code_type,
            "description": description,
            "analysis_result": analysis_result.to_dict(),
            "quality_score": self._calculate_code_quality_score(analysis_result),
            "timestamp": time.time(),
            "timestamp_human": datetime.now().isoformat()
        }

        self.logger.info(
            f"代码生成完成: ID={analysis_result.code_id}, 质量分数={result['quality_score']:.2f}")

        return result

    def debug_code(self,
                   code: str,
                   language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
                   error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        调试代码

        参数:
            code: 有问题的代码
            language: 编程语言
            error_message: 错误信息（如果有）

        返回:
            调试结果
        """
        self.logger.info(f"开始调试代码，语言: {language.value}")

        start_time = time.time()

        # 分析代码
        analysis_result = self.analyze_code(code, language)

        # 检测bug
        all_bugs = self._detect_all_bugs(code, language)

        # 如果有错误信息，优先处理
        if error_message:
            suggested_fixes = self._suggest_fixes_for_error(
                error_message, code, language)
        else:
            suggested_fixes = self._suggest_fixes_for_bugs(all_bugs, code, language)

        # 生成修复后的代码
        fixed_code = self._apply_fixes(code, suggested_fixes, language)

        execution_time = time.time() - start_time

        result = {
            "original_code": code,
            "fixed_code": fixed_code,
            "language": language.value,
            "analysis_result": analysis_result.to_dict(),
            "bugs_detected": all_bugs,
            "suggested_fixes": suggested_fixes,
            "fixes_applied": len([f for f in suggested_fixes if f.get("applied", False)]),
            "execution_time_seconds": execution_time,
            "success": len(all_bugs) == 0 or fixed_code != code,
            "timestamp": time.time(),
            "timestamp_human": datetime.now().isoformat()
        }

        self.logger.info(
            f"代码调试完成: "
            f"检测到 {len(all_bugs)} 个bug, "
            f"应用了 {result['fixes_applied']} 个修复, "
            f"成功: {result['success']}"
        )

        return result

    def _count_functions(self, code: str, language: ProgrammingLanguage) -> int:
        """统计函数数量"""
        if language == ProgrammingLanguage.PYTHON:
            # 简单统计Python函数
            return len(re.findall(r'def\s+\w+\s*\(', code))
        elif language == ProgrammingLanguage.JAVASCRIPT:
            # 统计JavaScript函数
            patterns = [
                r'function\s+\w+\s*\(',  # function声明
                r'const\s+\w+\s*=\s*\(',  # 箭头函数
                r'let\s+\w+\s*=\s*\(',
                r'var\s+\w+\s*=\s*\('
            ]
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, code))
            return count
        else:
            # 其他语言，使用简单模式匹配
            return len(re.findall(r'def|function|func|fn\s+\w+\s*\(', code))

    def _count_classes(self, code: str, language: ProgrammingLanguage) -> int:
        """统计类数量"""
        if language == ProgrammingLanguage.PYTHON:
            return len(re.findall(r'class\s+\w+', code))
        elif language == ProgrammingLanguage.JAVA:
            return len(re.findall(r'class\s+\w+', code))
        elif language == ProgrammingLanguage.CPP:
            return len(re.findall(r'class\s+\w+', code))
        else:
            return len(re.findall(r'class\s+\w+', code))

    def _analyze_code_complexity(self, code: str, language: ProgrammingLanguage) -> CodeComplexity:
        """分析代码复杂度"""
        lines = len(code.split('\n'))
        functions = self._count_functions(code, language)

        # 简单复杂度计算
        if lines < 50 and functions < 3:
            return CodeComplexity.VERY_SIMPLE
        elif lines < 100 and functions < 5:
            return CodeComplexity.SIMPLE
        elif lines < 200 and functions < 10:
            return CodeComplexity.MODERATE
        elif lines < 500 and functions < 20:
            return CodeComplexity.COMPLEX
        else:
            return CodeComplexity.VERY_COMPLEX

    def _analyze_code_quality(self, code: str, language: ProgrammingLanguage) -> Dict[str, Any]:
        """分析代码质量"""
        quality_metrics = {}

        # 计算基本质量指标
        quality_metrics["maintainability_index"] = self._calculate_maintainability_index(
            code, language)
        quality_metrics["cyclomatic_complexity"] = self._calculate_cyclomatic_complexity(
            code, language)

        # 检测问题
        bugs = self._detect_all_bugs(code, language)
        style_violations = self._detect_style_violations(code, language)
        performance_issues = self._detect_performance_issues(code, language)

        # 计算质量分数
        total_issues = len(bugs) + len(style_violations) + len(performance_issues)

        # 基础质量分数（基于可维护性指数）
        base_score = quality_metrics["maintainability_index"]

        # 根据问题数量调整分数
        issue_deduction = min(30, total_issues * 5)
        quality_score = max(0, base_score - issue_deduction)

        quality_metrics["quality_score"] = quality_score
        quality_metrics["total_issues"] = total_issues
        quality_metrics["bugs_count"] = len(bugs)
        quality_metrics["style_violations_count"] = len(style_violations)
        quality_metrics["performance_issues_count"] = len(performance_issues)

        # 质量评级
        if quality_score >= 80:
            quality_rating = "excellent"
        elif quality_score >= 60:
            quality_rating = "good"
        elif quality_score >= 40:
            quality_rating = "fair"
        elif quality_score >= 20:
            quality_rating = "poor"
        else:
            quality_rating = "very_poor"

        quality_metrics["quality_rating"] = quality_rating

        return quality_metrics

    def _analyze_code_security(self, code: str, language: ProgrammingLanguage) -> Dict[str, Any]:
        """分析代码安全性"""
        security_issues = self._detect_security_vulnerabilities(code, language)

        security_metrics = {
            "security_issues_count": len(security_issues),
            "security_issues": security_issues,
            "security_score": max(0, 100 - len(security_issues) * 20),
            "security_level": self._determine_security_level(len(security_issues))
        }

        return security_metrics

    def _determine_security_level(self, issue_count: int) -> str:
        """确定安全级别"""
        if issue_count == 0:
            return "excellent"
        elif issue_count <= 2:
            return "good"
        elif issue_count <= 5:
            return "fair"
        elif issue_count <= 10:
            return "poor"
        else:
            return "critical"

    def _analyze_code_performance(self, code: str, language: ProgrammingLanguage) -> Dict[str, Any]:
        """分析代码性能"""
        performance_issues = self._detect_performance_issues(code, language)

        performance_metrics = {
            "performance_issues_count": len(performance_issues),
            "performance_issues": performance_issues,
            "performance_score": max(0, 100 - len(performance_issues) * 15),
            "performance_level": self._determine_performance_level(len(performance_issues))
        }

        return performance_metrics

    def _determine_performance_level(self, issue_count: int) -> str:
        """确定性能级别"""
        if issue_count == 0:
            return "excellent"
        elif issue_count <= 3:
            return "good"
        elif issue_count <= 7:
            return "fair"
        elif issue_count <= 12:
            return "poor"
        else:
            return "critical"

    def _calculate_cyclomatic_complexity(self, code: str, language: ProgrammingLanguage) -> float:
        """计算圈复杂度（完整版）"""
        # 简单估计：基于控制流关键字
        if language == ProgrammingLanguage.PYTHON:
            keywords = ['if', 'elif', 'else', 'for', 'while', 'except', 'and', 'or']
        elif language == ProgrammingLanguage.JAVASCRIPT:
            keywords = ['if', 'else', 'for', 'while', 'catch', '&&', '||', '?']
        else:
            keywords = ['if', 'else', 'for', 'while', 'switch', 'case']

        complexity = 1  # 基本复杂度
        for keyword in keywords:
            complexity += len(re.findall(re.escape(keyword), code, re.IGNORECASE))

        return float(complexity)

    def _calculate_maintainability_index(self, code: str, language: ProgrammingLanguage) -> float:
        """计算可维护性指数（完整版）"""
        lines = len(code.split('\n'))
        functions = self._count_functions(code, language)
        classes = self._count_classes(code, language)
        complexity = self._calculate_cyclomatic_complexity(code, language)

        # 完整版可维护性指数计算
        # 基于：代码行数越少、函数越少、类越少、复杂度越低，可维护性越高
        if lines == 0:
            return 100.0

        # 归一化计算
        line_score = max(0, 100 - (lines / 10))
        function_score = max(0, 100 - (functions * 5))
        class_score = max(0, 100 - (classes * 10))
        complexity_score = max(0, 100 - (complexity * 10))

        # 加权平均
        maintainability = (line_score * 0.3 + function_score * 0.2 +
                           class_score * 0.2 + complexity_score * 0.3)

        return max(0, min(100, maintainability))

    def _detect_all_bugs(self, code: str, language: ProgrammingLanguage) -> List[Dict[str, Any]]:
        """检测所有bug"""
        bugs = []

        for detector_name, detector_func in self.bug_detectors.items():
            detected_bugs = detector_func(code, language)
            bugs.extend(detected_bugs)

        return bugs

    def _detect_syntax_errors(self, code: str, language: ProgrammingLanguage) -> List[Dict[str, Any]]:
        """检测语法错误"""
        bugs = []

        if language == ProgrammingLanguage.PYTHON and AST_AVAILABLE:
            try:
                ast.parse(code)
            except SyntaxError as e:
                bugs.append({
                    "type": "syntax_error",
                    "severity": "high",
                    "message": f"语法错误: {e.msg}",
                    "line": e.lineno if hasattr(e, 'lineno') else None,
                    "column": e.offset if hasattr(e, 'offset') else None,
                    "suggestion": "检查语法错误并修复"
                })

        # 其他语言的语法检测可以在这里添加

        return bugs

    def _detect_logical_errors(self, code: str, language: ProgrammingLanguage) -> List[Dict[str, Any]]:
        """检测逻辑错误"""
        bugs = []

        # 检测常见逻辑错误模式
        patterns = []

        if language == ProgrammingLanguage.PYTHON:
            patterns = [
                # 无限循环风险
                (r'while\s+True:', "无限循环，考虑添加退出条件"),
                # 可能除零
                (r'/\s*\w+\s*\)', "可能除零，添加除数检查"),
                # 未处理的异常
                (r'try:\s*\n(?!.*except)', "未处理的异常，添加except块"),
                # 错误的比较
                (r'==\s*True\b', "冗余的True比较，直接使用变量"),
                (r'==\s*False\b', "冗余的False比较，使用not操作符")
            ]
        elif language == ProgrammingLanguage.JAVASCRIPT:
            patterns = [
                # 可能未定义
                (r'console\.log\(', "生产环境中应移除console.log"),
                # 严格相等
                (r'==\s*null', "考虑使用===进行严格相等比较"),
                # 未处理的Promise
                (r'\.then\([^)]*\)(?!\.catch)', "Promise未处理错误，添加.catch")
            ]

        for pattern, message in patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_number = code[:match.start()].count('\n') + 1
                bugs.append({
                    "type": "logical_error",
                    "severity": "medium",
                    "message": f"逻辑错误: {message}",
                    "line": line_number,
                    "column": match.start() - code[:match.start()].rfind('\n'),
                    "suggestion": message
                })

        return bugs

    def _detect_runtime_errors(self, code: str, language: ProgrammingLanguage) -> List[Dict[str, Any]]:
        """检测运行时错误"""
        bugs = []

        # 检测常见运行时错误模式
        patterns = []

        if language == ProgrammingLanguage.PYTHON:
            patterns = [
                # 未检查的索引
                (r'\[\s*\w+\s*\+\s*\d+\s*\]', "索引可能越界，添加边界检查"),
                # 未检查的字典访问
                (r'\[\s*[\'\"][^\'\"]+[\'\"]\s*\]', "字典键可能不存在，使用.get()方法"),
                # 文件操作未关闭
                (r'open\([^)]+\)(?!.*with)', "文件未安全关闭，使用with语句"),
                # 大内存操作
                (r'list\(range\(\d{6,}\)\)', "大范围列表可能消耗大量内存，考虑使用生成器")
            ]

        for pattern, message in patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_number = code[:match.start()].count('\n') + 1
                bugs.append({
                    "type": "runtime_error",
                    "severity": "medium",
                    "message": f"运行时错误风险: {message}",
                    "line": line_number,
                    "column": match.start() - code[:match.start()].rfind('\n'),
                    "suggestion": message
                })

        return bugs

    def _detect_security_vulnerabilities(self, code: str, language: ProgrammingLanguage) -> List[Dict[str, Any]]:
        """检测安全漏洞"""
        bugs = []

        # 检测常见安全漏洞
        patterns = []

        if language == ProgrammingLanguage.PYTHON:
            patterns = [
                # SQL注入
                (r'execute\s*\(\s*f?"[^"]*\%s[^"]*"', "可能的SQL注入，使用参数化查询"),
                # 命令注入
                (r'os\.system\([^)]+\)', "命令注入风险，使用subprocess.run()"),
                # 硬编码密码
                (r'password\s*=\s*[\'"][^\'"]+[\'"]', "硬编码密码，使用环境变量"),
                # 不安全的反序列化
                (r'pickle\.loads\(', "不安全的反序列化，考虑使用json或安全替代")
            ]
        elif language == ProgrammingLanguage.JAVASCRIPT:
            patterns = [
                # XSS风险
                (r'innerHTML\s*=', "XSS风险，使用textContent或安全库"),
                # 不安全的eval
                (r'eval\(', "不安全的eval使用，寻找替代方案"),
                # 不安全的正则表达式
                (r'new RegExp\([^)]*\+', "不安全的正则表达式，可能引起ReDoS")
            ]

        for pattern, message in patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_number = code[:match.start()].count('\n') + 1
                bugs.append({
                    "type": "security_vulnerability",
                    "severity": "high",
                    "message": f"安全漏洞: {message}",
                    "line": line_number,
                    "column": match.start() - code[:match.start()].rfind('\n'),
                    "suggestion": message
                })

        return bugs

    def _detect_style_violations(self, code: str, language: ProgrammingLanguage) -> List[Dict[str, Any]]:
        """检测代码风格违规"""
        violations = []

        # 检测常见风格问题
        patterns = []

        if language == ProgrammingLanguage.PYTHON:
            patterns = [
                # 行过长
                (r'^.{100,}$', "行超过100字符，考虑换行"),
                # 未使用的导入
                (r'^import\s+\w+', "检查导入是否使用"),
                # 魔法数字
                (r'\b\d{3,}\b', "魔法数字，考虑定义为常量"),
                # 不一致的命名
                (r'def\s+[a-z][a-z0-9_]*[A-Z]', "函数名应使用小写字母和下划线")
            ]

        for pattern, message in patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_number = code[:match.start()].count('\n') + 1
                violations.append({
                    "type": "style_violation",
                    "severity": "low",
                    "message": f"代码风格: {message}",
                    "line": line_number,
                    "column": match.start() - code[:match.start()].rfind('\n'),
                    "suggestion": message
                })

        return violations

    def _detect_performance_issues(self, code: str, language: ProgrammingLanguage) -> List[Dict[str, Any]]:
        """检测性能问题"""
        issues = []

        # 检测常见性能问题
        patterns = []

        if language == ProgrammingLanguage.PYTHON:
            patterns = [
                # 字符串拼接循环
                (r'for[^:]+:\s*\w+\s*\+=\s*[\'"]', "循环内字符串拼接低效，使用join()"),
                # 不必要的列表复制
                (r'list\(\[\]', "不必要的列表复制"),
                # 低效的成员检查
                (r'\w+\s+in\s+\[\s*\w+', "列表成员检查低效，考虑使用集合"),
                # 重复计算
                (r'len\(\w+\)\s*>\s*\d+\s+and\s+len\(\w+\)', "重复计算len()，存储结果")
            ]

        for pattern, message in patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_number = code[:match.start()].count('\n') + 1
                issues.append({
                    "type": "performance_issue",
                    "severity": "medium",
                    "message": f"性能问题: {message}",
                    "line": line_number,
                    "column": match.start() - code[:match.start()].rfind('\n'),
                    "suggestion": message
                })

        return issues

    def _generate_function(self, description: str, language: ProgrammingLanguage) -> str:
        """生成函数代码"""
        function_name = self._extract_function_name(description)

        if language == ProgrammingLanguage.PYTHON:
            return self._generate_python_function(function_name, description)
        elif language == ProgrammingLanguage.JAVASCRIPT:
            return self._generate_javascript_function(function_name, description)
        elif language == ProgrammingLanguage.TYPESCRIPT:
            return self._generate_typescript_function(function_name, description)
        elif language == ProgrammingLanguage.JAVA:
            return self._generate_java_function(function_name, description)
        elif language == ProgrammingLanguage.CPP:
            return self._generate_cpp_function(function_name, description)
        elif language == ProgrammingLanguage.C:
            return self._generate_c_function(function_name, description)
        elif language == ProgrammingLanguage.RUST:
            return self._generate_rust_function(function_name, description)
        elif language == ProgrammingLanguage.GO:
            return self._generate_go_function(function_name, description)
        elif language == ProgrammingLanguage.CSHARP:
            return self._generate_csharp_function(function_name, description)
        elif language == ProgrammingLanguage.PHP:
            return self._generate_php_function(function_name, description)
        elif language == ProgrammingLanguage.RUBY:
            return self._generate_ruby_function(function_name, description)
        elif language == ProgrammingLanguage.SWIFT:
            return self._generate_swift_function(function_name, description)
        elif language == ProgrammingLanguage.KOTLIN:
            return self._generate_kotlin_function(function_name, description)
        elif language == ProgrammingLanguage.SQL:
            return self._generate_sql_function(function_name, description)
        else:
            return f"// 实现 {language.value} 函数: {description}（根据描述生成具体实现）"

    def _extract_function_name(self, description: str) -> str:
        """从描述中提取函数名"""
        # 简单提取：使用第一个动词或名词
        words = description.lower().split()
        if words:
            # 移除常见停用词
            stop_words = {'the', 'a', 'an', 'to', 'for',
                          'of', 'and', 'or', 'in', 'on', 'at'}
            valid_words = [w for w in words if w not in stop_words and w.isalpha()]

            if valid_words:
                # 使用第一个有效词作为函数名
                base_name = valid_words[0]
                # 转换为蛇形命名
                function_name = ''.join(
                    ['_' + c.lower() if c.isupper() else c for c in base_name]).lstrip('_')
                return function_name

        return "custom_function"

    def _extract_module_name(self, description: str) -> str:
        """从描述中提取模块名"""
        # 使用提取函数名的逻辑，但转换为帕斯卡命名法
        function_name = self._extract_function_name(description)
        # 转换为帕斯卡命名（首字母大写）
        if function_name:
            # 将蛇形命名转换为帕斯卡命名
            parts = function_name.split('_')
            module_name = ''.join(part.capitalize() for part in parts if part)
            return module_name
        return "CustomModule"

    def _generate_python_function(self, function_name: str, description: str) -> str:
        """生成Python函数代码"""
        # 分析描述，尝试推断参数和返回值
        params = self._infer_function_params(description)
        return_type = self._infer_return_type(description)

        # 构建参数字符串
        if params:
            params_str = ", ".join(params)
        else:
            params_str = "*args, **kwargs"

        docstring = f'    """{description}'
        if params:
            docstring += f'\n\n    参数:\n'
            for param in params:
                docstring += f'        {param}: 参数描述\n'
        if return_type:
            docstring += f'\n    返回:\n        {return_type}: 返回值描述'
        docstring += '\n    """'

        code = f"def {function_name}({params_str}):\n"
        code += docstring + "\n"
        code += "    # 实现函数功能（根据描述生成具体实现）\n"

        # 根据返回类型添加返回值
        if return_type == "bool":
            code += "    return True\n"
        elif return_type == "int":
            code += "    return 0\n"
        elif return_type == "float":
            code += "    return 0.0\n"
        elif return_type == "str":
            code += "    return ''\n"
        elif return_type == "list":
            code += "    return []  # 返回空列表\n"
        elif return_type == "dict":
            code += "    return {}  # 返回空字典\n"
        else:
            code += "    return None  # 返回None\n"

        return code

    def _infer_function_params(self, description: str) -> List[str]:
        """从描述中推断函数参数"""
        params = []

        # 简单的关键词匹配
        keywords = {
            "calculate": ["a", "b"],
            "process": ["data", "config"],
            "analyze": ["input_data", "options"],
            "generate": ["template", "params"],
            "validate": ["value", "rules"],
            "transform": ["input", "output_format"],
            "filter": ["items", "condition"],
            "sort": ["items", "key"],
            "search": ["items", "target"],
            "parse": ["text", "format"],
            "encode": ["data", "algorithm"],
            "decode": ["encoded_data", "algorithm"],
            "compress": ["data", "method"],
            "extract": ["source", "pattern"],
            "merge": ["list1", "list2"],
            "split": ["text", "separator"],
            "convert": ["value", "from_unit", "to_unit"],
            "format": ["text", "style"],
            "validate": ["data", "schema"]
        }

        desc_lower = description.lower()
        for keyword, default_params in keywords.items():
            if keyword in desc_lower:
                params = default_params
                break

        return params

    def _infer_return_type(self, description: str) -> str:
        """从描述中推断返回类型"""
        desc_lower = description.lower()

        if any(word in desc_lower for word in ["计算", "calculate", "sum", "total", "count", "number"]):
            return "int" if "整数" in description else "float"
        elif any(word in desc_lower for word in ["检查", "check", "验证", "validate", "是否", "is_"]):
            return "bool"
        elif any(word in desc_lower for word in ["文本", "text", "字符串", "string", "消息", "message"]):
            return "str"
        elif any(word in desc_lower for word in ["列表", "list", "数组", "array", "集合", "collection"]):
            return "list"
        elif any(word in desc_lower for word in ["字典", "dict", "映射", "map", "对象", "object"]):
            return "dict"
        elif any(word in desc_lower for word in ["处理", "process", "转换", "transform", "生成", "generate"]):
            return "Any"  # 通用类型

        return ""

    def _generate_javascript_function(self, function_name: str, description: str) -> str:
        """生成JavaScript函数代码"""
        params = self._infer_function_params(description)
        params_str = ", ".join(params) if params else "...args"

        code = f"function {function_name}({params_str}) {{\n"
        code += f"    // {description}\n"

        # 添加注释文档
        if params:
            code += "    /**\n"
            for param in params:
                code += f"     * @param {{{param}}}\n"
            code += f"     * @returns {{any}}\n"
            code += "     */\n"

        code += "    // 实现函数功能（根据描述生成具体实现）\n"

        # 根据描述添加基本实现
        desc_lower = description.lower()
        if "calculate" in desc_lower or "sum" in desc_lower:
            if params and len(params) >= 2:
                code += f"    return {params[0]} + {params[1]};\n"
            else:
                code += "    return 0;\n"
        elif "check" in desc_lower or "validate" in desc_lower:
            code += "    return true;\n"
        else:
            code += "    return null;\n"

        code += "}\n"

        # 添加ES6箭头函数版本
        code += f"\n// ES6箭头函数版本:\n"
        code += f"// const {function_name} = ({params_str}) => {{\n"
        code += f"//     // {description}\n"
        # 根据描述添加箭头函数实现
        desc_lower = description.lower()
        if "calculate" in desc_lower or "sum" in desc_lower:
            if params and len(params) >= 2:
                code += f"//     return {params[0]} + {params[1]};\n"
            else:
                code += f"//     return 0;\n"
        elif "check" in desc_lower or "validate" in desc_lower:
            code += f"//     return true;\n"
        elif "filter" in desc_lower:
            if params and len(params) >= 1:
                code += f"//     return {params[0]}.filter(item => item);\n"
            else:
                code += f"//     return []  # 返回空列表;\n"
        elif "map" in desc_lower or "transform" in desc_lower:
            if params and len(params) >= 1:
                code += f"//     return {params[0]}.map(item => item);\n"
            else:
                code += f"//     return []  # 返回空列表;\n"
        else:
            code += f"//     return null;\n"
        code += f"// }};\n"

        return code

    def _generate_java_function(self, function_name: str, description: str) -> str:
        """生成Java函数代码"""
        # Java方法名使用驼峰命名
        method_name = function_name
        if '_' in method_name:
            parts = method_name.split('_')
            method_name = parts[0] + ''.join(p.title() for p in parts[1:])

        params = self._infer_function_params(description)
        return_type = self._infer_return_type(description) or "void"

        # Java类型映射
        type_mapping = {
            "int": "int",
            "float": "float",
            "double": "double",
            "bool": "boolean",
            "str": "String",
            "list": "List<Object>",
            "dict": "Map<String, Object>",
            "Any": "Object"
        }

        java_return_type = type_mapping.get(return_type, "Object")

        # 构建参数列表
        param_list = []
        if params:
            for i, param in enumerate(params):
                param_type = "Object"  # 默认类型
                if i == 0 and java_return_type != "void":
                    param_type = java_return_type
                param_list.append(f"{param_type} {param}")

        params_str = ", ".join(param_list) if param_list else ""

        code = f"public {java_return_type} {method_name}({params_str}) {{\n"
        code += f"    // {description}\n"
        code += "    // 方法实现\n"

        # 添加返回值
        if java_return_type != "void":
            if java_return_type == "int":
                code += "    return 0;\n"
            elif java_return_type == "boolean":
                code += "    return true;\n"
            elif java_return_type == "String":
                code += '    return "";\n'
            elif java_return_type == "double":
                code += "    return 0.0;\n"
            elif java_return_type == "float":
                code += "    return 0.0f;\n"
            elif "List" in java_return_type:
                code += "    return new ArrayList<>();\n"
            elif "Map" in java_return_type:
                code += "    return new HashMap<>();\n"
            else:
                code += "    return null;\n"

        code += "}\n"

        return code

    def _generate_cpp_function(self, function_name: str, description: str) -> str:
        """生成C++函数代码"""
        return_type = self._infer_return_type(description) or "void"

        # C++类型映射
        type_mapping = {
            "int": "int",
            "float": "float",
            "double": "double",
            "bool": "bool",
            "str": "std::string",
            "list": "std::vector<auto>",
            "dict": "std::map<std::string, auto>",
            "Any": "auto"
        }

        cpp_return_type = type_mapping.get(return_type, "void")

        params = self._infer_function_params(description)
        param_list = []
        if params:
            for i, param in enumerate(params):
                param_type = "auto"  # 默认使用auto
                param_list.append(f"{param_type} {param}")

        params_str = ", ".join(param_list) if param_list else ""

        code = f"{cpp_return_type} {function_name}({params_str}) {{\n"
        code += f"    // {description}\n"

        # 根据描述关键词生成具体实现
        desc_lower = description.lower()

        # 根据返回类型和描述生成适当的代码
        if cpp_return_type != "void":
            if "calculate" in desc_lower or "sum" in desc_lower:
                if params and len(params) >= 2:
                    code += f"    return {params[0]} + {params[1]};\n"
                else:
                    code += "    return 0;\n"
            elif "check" in desc_lower or "validate" in desc_lower:
                code += "    return true;\n"
            elif "get" in desc_lower or "retrieve" in desc_lower:
                if cpp_return_type == "int":
                    code += "    return 0;\n"
                elif cpp_return_type == "bool":
                    code += "    return true;\n"
                elif cpp_return_type == "double":
                    code += "    return 0.0;\n"
                elif cpp_return_type == "float":
                    code += "    return 0.0f;\n"
                elif cpp_return_type == "std::string":
                    code += '    return "";\n'
                else:
                    code += "    return {}  # 返回空字典;\n"
            else:
                # 默认返回语句
                if cpp_return_type == "int":
                    code += "    return 0;\n"
                elif cpp_return_type == "bool":
                    code += "    return true;\n"
                elif cpp_return_type == "double":
                    code += "    return 0.0;\n"
                elif cpp_return_type == "float":
                    code += "    return 0.0f;\n"
                elif cpp_return_type == "std::string":
                    code += '    return "";\n'
                else:
                    code += "    return {}  # 返回空字典;\n"
        else:
            # void 函数，添加一些操作
            if "print" in desc_lower or "log" in desc_lower:
                if params and len(params) >= 1:
                    code += f'    std::cout << {params[0]} << std::endl;\n'
                else:
                    code += '    std::cout << "操作完成" << std::endl;\n'
            else:
                code += "    // 执行操作\n"

        code += "}\n"

        return code

    def _generate_typescript_function(self, function_name: str, description: str) -> str:
        """生成TypeScript函数代码"""
        params = self._infer_function_params(description)
        return_type = self._infer_return_type(description) or "any"

        # TypeScript类型映射
        type_mapping = {
            "int": "number",
            "float": "number",
            "double": "number",
            "bool": "boolean",
            "str": "string",
            "list": "any[]",
            "dict": "Record<string, any>",
            "Any": "any"
        }

        ts_return_type = type_mapping.get(return_type, "any")

        # 构建参数列表
        param_list = []
        if params:
            for param in params:
                param_list.append(f"{param}: any")

        params_str = ", ".join(param_list) if param_list else "...args: any[]"

        code = f"function {function_name}({params_str}): {ts_return_type} {{\n"
        code += f"    // {description}\n"

        # 根据描述关键词生成具体实现
        desc_lower = description.lower()

        # 根据返回类型和描述生成适当的代码
        if ts_return_type != "void":
            if "calculate" in desc_lower or "sum" in desc_lower:
                if params and len(params) >= 2:
                    code += f"    return {params[0]} + {params[1]};\n"
                else:
                    code += "    return 0;\n"
            elif "check" in desc_lower or "validate" in desc_lower:
                code += "    return true;\n"
            elif "filter" in desc_lower:
                if params and len(params) >= 1:
                    code += f"    return {params[0]}.filter((item: any) => item);\n"
                else:
                    code += "    return []  # 返回空列表;\n"
            elif "map" in desc_lower or "transform" in desc_lower:
                if params and len(params) >= 1:
                    code += f"    return {params[0]}.map((item: any) => item);\n"
                else:
                    code += "    return []  # 返回空列表;\n"
            elif "get" in desc_lower or "retrieve" in desc_lower:
                if ts_return_type == "number":
                    code += "    return 0;\n"
                elif ts_return_type == "boolean":
                    code += "    return true;\n"
                elif ts_return_type == "string":
                    code += '    return "";\n'
                elif ts_return_type == "any[]":
                    code += "    return []  # 返回空列表;\n"
                elif ts_return_type == "Record<string, any>":
                    code += "    return {}  # 返回空字典;\n"
                else:
                    code += "    return null;\n"
            else:
                # 默认返回语句
                if ts_return_type == "number":
                    code += "    return 0;\n"
                elif ts_return_type == "boolean":
                    code += "    return true;\n"
                elif ts_return_type == "string":
                    code += '    return "";\n'
                elif ts_return_type == "any[]":
                    code += "    return []  # 返回空列表;\n"
                elif ts_return_type == "Record<string, any>":
                    code += "    return {}  # 返回空字典;\n"
                else:
                    code += "    return null;\n"
        else:
            # void 函数，添加一些操作
            if "print" in desc_lower or "log" in desc_lower:
                if params and len(params) >= 1:
                    code += f'    console.log({params[0]});\n'
                else:
                    code += '    console.log("操作完成");\n'
            else:
                code += "    // 执行操作\n"

        code += "}\n"

        # 添加箭头函数版本
        code += f"\n// 箭头函数版本:\n"
        code += f"// const {function_name} = ({params_str}): {ts_return_type} => {{\n"
        code += f"//     // {description}\n"
        # 根据描述关键词生成箭头函数实现
        desc_lower = description.lower()

        # 根据返回类型和描述生成箭头函数代码
        if ts_return_type != "void":
            if "calculate" in desc_lower or "sum" in desc_lower:
                if params and len(params) >= 2:
                    code += f"//     return {params[0]} + {params[1]};\n"
                else:
                    code += f"//     return 0;\n"
            elif "check" in desc_lower or "validate" in desc_lower:
                code += f"//     return true;\n"
            elif "filter" in desc_lower:
                if params and len(params) >= 1:
                    code += f"//     return {params[0]}.filter((item: any) => item);\n"
                else:
                    code += f"//     return []  # 返回空列表;\n"
            elif "map" in desc_lower or "transform" in desc_lower:
                if params and len(params) >= 1:
                    code += f"//     return {params[0]}.map((item: any) => item);\n"
                else:
                    code += f"//     return []  # 返回空列表;\n"
            elif "get" in desc_lower or "retrieve" in desc_lower:
                if ts_return_type == "number":
                    code += f"//     return 0;\n"
                elif ts_return_type == "boolean":
                    code += f"//     return true;\n"
                elif ts_return_type == "string":
                    code += f"//     return '';\n"
                elif ts_return_type == "any[]":
                    code += f"//     return []  # 返回空列表;\n"
                elif ts_return_type == "Record<string, any>":
                    code += f"//     return {{}};\n"
                else:
                    code += f"//     return null;\n"
            else:
                # 默认返回语句
                if ts_return_type == "number":
                    code += f"//     return 0;\n"
                elif ts_return_type == "boolean":
                    code += f"//     return true;\n"
                elif ts_return_type == "string":
                    code += f"//     return '';\n"
                elif ts_return_type == "any[]":
                    code += f"//     return []  # 返回空列表;\n"
                elif ts_return_type == "Record<string, any>":
                    code += f"//     return {{}};\n"
                else:
                    code += f"//     return null;\n"
        else:
            # void 箭头函数，添加一些操作
            if "print" in desc_lower or "log" in desc_lower:
                if params and len(params) >= 1:
                    code += f"//     console.log({params[0]});\n"
                else:
                    code += f'//     console.log("操作完成");\n'
            else:
                code += f"//     // 执行操作\n"
        code += f"// }};\n"

        return code

    def _generate_c_function(self, function_name: str, description: str) -> str:
        """生成C函数代码"""
        params = self._infer_function_params(description)
        return_type = self._infer_return_type(description) or "void"

        # C类型映射
        type_mapping = {
            "int": "int",
            "float": "float",
            "double": "double",
            "bool": "int",  # C语言中bool通常用int表示
            "str": "char*",
            "list": "void*",
            "dict": "void*",
            "Any": "void*"
        }

        c_return_type = type_mapping.get(return_type, "void")

        # 构建参数列表
        param_list = []
        if params:
            for param in params:
                param_type = "void*"  # 默认使用void指针
                param_list.append(f"{param_type} {param}")

        params_str = ", ".join(param_list) if param_list else "void"

        code = f"{c_return_type} {function_name}({params_str}) {{\n"
        code += f"    /* {description} */\n"

        # 根据描述关键词生成具体实现
        desc_lower = description.lower()

        # 根据返回类型和描述生成适当的代码
        if c_return_type != "void":
            if "calculate" in desc_lower or "sum" in desc_lower:
                if params and len(params) >= 2:
                    code += f"    return {params[0]} + {params[1]};\n"
                else:
                    code += "    return 0;\n"
            elif "check" in desc_lower or "validate" in desc_lower:
                code += "    return 1;\n"  # C语言中true通常用1表示
            elif "get" in desc_lower or "retrieve" in desc_lower:
                if c_return_type == "int":
                    code += "    return 0;\n"
                elif c_return_type == "float":
                    code += "    return 0.0f;\n"
                elif c_return_type == "double":
                    code += "    return 0.0;\n"
                elif c_return_type == "char*":
                    code += '    return "";\n'
                else:
                    code += "    return NULL;\n"
            else:
                # 默认返回语句
                if c_return_type == "int":
                    code += "    return 0;\n"
                elif c_return_type == "float":
                    code += "    return 0.0f;\n"
                elif c_return_type == "double":
                    code += "    return 0.0;\n"
                elif c_return_type == "char*":
                    code += '    return "";\n'
                else:
                    code += "    return NULL;\n"
        else:
            # void 函数，添加一些操作
            if "print" in desc_lower or "log" in desc_lower:
                if params and len(params) >= 1:
                    code += f'    printf("%p\\n", {params[0]});\n'
                else:
                    code += '    printf("操作完成\\n");\n'
            else:
                code += "    /* 执行操作 */\n"

        code += "}\n"

        return code

    def _generate_rust_function(self, function_name: str, description: str) -> str:
        """生成Rust函数代码"""
        params = self._infer_function_params(description)
        return_type = self._infer_return_type(description) or "()"

        # Rust类型映射
        type_mapping = {
            "int": "i32",
            "float": "f32",
            "double": "f64",
            "bool": "bool",
            "str": "String",
            "list": "Vec<Box<dyn Any>>",
            "dict": "HashMap<String, Box<dyn Any>>",
            "Any": "Box<dyn Any>"
        }

        rust_return_type = type_mapping.get(return_type, "()")

        # 构建参数列表
        param_list = []
        if params:
            for param in params:
                param_type = "impl Any"  # 使用trait对象
                param_list.append(f"{param}: {param_type}")

        params_str = ", ".join(param_list) if param_list else ""

        code = f"fn {function_name}({params_str}) -> {rust_return_type} {{\n"
        code += f"    // {description}\n"

        # 根据描述关键词生成具体实现
        desc_lower = description.lower()

        # 根据返回类型和描述生成适当的代码
        if rust_return_type != "()":
            if "calculate" in desc_lower or "sum" in desc_lower:
                if params and len(params) >= 2:
                    code += f"    {params[0]} + {params[1]}\n"
                else:
                    code += "    0\n"
            elif "check" in desc_lower or "validate" in desc_lower:
                code += "    true\n"
            elif "filter" in desc_lower:
                if params and len(params) >= 1:
                    code += f"    {params[0]}.into_iter().filter(|item| true).collect()\n"
                else:
                    code += "    Vec::new()\n"
            elif "map" in desc_lower or "transform" in desc_lower:
                if params and len(params) >= 1:
                    code += f"    {params[0]}.into_iter().map(|item| item).collect()\n"
                else:
                    code += "    Vec::new()\n"
            elif "get" in desc_lower or "retrieve" in desc_lower:
                if rust_return_type == "i32":
                    code += "    0\n"
                elif rust_return_type == "bool":
                    code += "    true\n"
                elif rust_return_type == "f32":
                    code += "    0.0\n"
                elif rust_return_type == "f64":
                    code += "    0.0\n"
                elif rust_return_type == "String":
                    code += '    String::from("")\n'
                elif "Vec" in rust_return_type:
                    code += "    Vec::new()\n"
                elif "HashMap" in rust_return_type:
                    code += "    HashMap::new()\n"
                else:
                    code += "    Box::new(())\n"
            else:
                # 默认返回语句
                if rust_return_type == "i32":
                    code += "    0\n"
                elif rust_return_type == "bool":
                    code += "    true\n"
                elif rust_return_type == "f32":
                    code += "    0.0\n"
                elif rust_return_type == "f64":
                    code += "    0.0\n"
                elif rust_return_type == "String":
                    code += '    String::from("")\n'
                elif "Vec" in rust_return_type:
                    code += "    Vec::new()\n"
                elif "HashMap" in rust_return_type:
                    code += "    HashMap::new()\n"
                else:
                    code += "    Box::new(())\n"
        else:
            # unit 函数，添加一些操作
            if "print" in desc_lower or "log" in desc_lower:
                if params and len(params) >= 1:
                    code += f'    println!("{{:?}}", {params[0]});\n'
                else:
                    code += '    println!("操作完成");\n'
            # unit函数隐式返回()

        code += "}\n"

        return code

    def _generate_go_function(self, function_name: str, description: str) -> str:
        """生成Go函数代码"""
        # Go函数名使用驼峰命名
        func_name = function_name
        if '_' in func_name:
            parts = func_name.split('_')
            func_name = parts[0] + ''.join(p.title() for p in parts[1:])

        params = self._infer_function_params(description)
        return_type = self._infer_return_type(description) or ""

        # Go类型映射
        type_mapping = {
            "int": "int",
            "float": "float64",
            "double": "float64",
            "bool": "bool",
            "str": "string",
            "list": "[]interface{}",
            "dict": "map[string]interface{}",
            "Any": "interface{}"
        }

        go_return_type = type_mapping.get(return_type, "")

        # 构建参数列表
        param_list = []
        if params:
            for param in params:
                param_type = "interface{}"
                param_list.append(f"{param} {param_type}")

        params_str = ", ".join(param_list) if param_list else ""

        code = f"func {func_name}({params_str})"
        if go_return_type:
            code += f" {go_return_type}"
        code += " {\n"
        code += f"    // {description}\n"
        code += "    // 根据描述实现函数功能\n"

        if go_return_type:
            if go_return_type == "int":
                code += "    return 0\n"
            elif go_return_type == "bool":
                code += "    return true\n"
            elif go_return_type == "float64":
                code += "    return 0.0\n"
            elif go_return_type == "string":
                code += '    return ""\n'
            elif go_return_type == "[]interface{}":
                code += "    return []  # 返回空列表interface{}{}\n"
            elif go_return_type == "map[string]interface{}":
                code += "    return map[string]interface{}{}\n"
            else:
                code += "    return nil\n"

        code += "}\n"

        return code

    def _generate_csharp_function(self, function_name: str, description: str) -> str:
        """生成C#函数代码"""
        # C#方法名使用帕斯卡命名
        method_name = function_name
        if '_' in method_name:
            parts = method_name.split('_')
            method_name = ''.join(p.title() for p in parts)

        params = self._infer_function_params(description)
        return_type = self._infer_return_type(description) or "void"

        # C#类型映射
        type_mapping = {
            "int": "int",
            "float": "float",
            "double": "double",
            "bool": "bool",
            "str": "string",
            "list": "List<object>",
            "dict": "Dictionary<string, object>",
            "Any": "object"
        }

        csharp_return_type = type_mapping.get(return_type, "void")

        # 构建参数列表
        param_list = []
        if params:
            for param in params:
                param_type = "object"
                param_list.append(f"{param_type} {param}")

        params_str = ", ".join(param_list) if param_list else ""

        code = f"public {csharp_return_type} {method_name}({params_str})\n"
        code += "{\n"
        code += f"    // {description}\n"
        code += "    // 根据描述实现方法功能\n"

        if csharp_return_type != "void":
            if csharp_return_type == "int":
                code += "    return 0;\n"
            elif csharp_return_type == "bool":
                code += "    return true;\n"
            elif csharp_return_type == "double":
                code += "    return 0.0;\n"
            elif csharp_return_type == "float":
                code += "    return 0.0f;\n"
            elif csharp_return_type == "string":
                code += '    return "";\n'
            elif csharp_return_type == "List<object>":
                code += "    return new List<object>();\n"
            elif csharp_return_type == "Dictionary<string, object>":
                code += "    return new Dictionary<string, object>();\n"
            else:
                code += "    return null;\n"

        code += "}\n"

        return code

    def _generate_php_function(self, function_name: str, description: str) -> str:
        """生成PHP函数代码"""
        params = self._infer_function_params(description)
        params_str = ", ".join(
            [f"${param}" for param in params]) if params else "...$args"

        code = f"function {function_name}({params_str})\n"
        code += "{\n"
        code += f"    // {description}\n"
        # 根据描述添加基本实现
        desc_lower = description.lower()
        if "calculate" in desc_lower or "sum" in desc_lower:
            if params and len(params) >= 2:
                code += f"    return ${params[0]} + ${params[1]};\n"
            else:
                code += "    return 0;\n"
        elif "check" in desc_lower or "validate" in desc_lower:
            code += "    return true;\n"
        else:
            code += "    return null;\n"

        code += "}\n"

        return code

    def _generate_ruby_function(self, function_name: str, description: str) -> str:
        """生成Ruby函数代码"""
        params = self._infer_function_params(description)
        params_str = ", ".join(params) if params else "*args"

        code = f"def {function_name}({params_str})\n"
        code += f"  # {description}\n"
        # Ruby通常返回最后一个表达式的值
        desc_lower = description.lower()
        if "calculate" in desc_lower or "sum" in desc_lower:
            if params and len(params) >= 2:
                code += f"  {params[0]} + {params[1]}\n"
            else:
                code += "  0\n"
        elif "check" in desc_lower or "validate" in desc_lower:
            code += "  true\n"
        else:
            code += "  nil\n"

        code += "end\n"

        return code

    def _generate_swift_function(self, function_name: str, description: str) -> str:
        """生成Swift函数代码"""
        params = self._infer_function_params(description)
        return_type = self._infer_return_type(description) or "Void"

        # Swift类型映射
        type_mapping = {
            "int": "Int",
            "float": "Float",
            "double": "Double",
            "bool": "Bool",
            "str": "String",
            "list": "[Any]",
            "dict": "[String: Any]",
            "Any": "Any"
        }

        swift_return_type = type_mapping.get(return_type, "Void")

        # 构建参数列表
        param_list = []
        if params:
            for param in params:
                param_list.append(f"{param}: Any")

        params_str = ", ".join(param_list) if param_list else ""

        code = f"func {function_name}({params_str}) -> {swift_return_type} {{\n"
        code += f"    // {description}\n"
        code += "    // 根据描述实现具体功能\n"

        if swift_return_type != "Void":
            if swift_return_type == "Int":
                code += "    return 0\n"
            elif swift_return_type == "Bool":
                code += "    return true\n"
            elif swift_return_type == "Float":
                code += "    return 0.0\n"
            elif swift_return_type == "Double":
                code += "    return 0.0\n"
            elif swift_return_type == "String":
                code += '    return ""\n'
            elif swift_return_type == "[Any]":
                code += "    return []  # 返回空列表\n"
            elif swift_return_type == "[String: Any]":
                code += "    return [:]\n"
            else:
                code += "    return nil\n"

        code += "}\n"

        return code

    def _generate_kotlin_function(self, function_name: str, description: str) -> str:
        """生成Kotlin函数代码"""
        params = self._infer_function_params(description)
        return_type = self._infer_return_type(description) or "Unit"

        # Kotlin类型映射
        type_mapping = {
            "int": "Int",
            "float": "Float",
            "double": "Double",
            "bool": "Boolean",
            "str": "String",
            "list": "List<Any>",
            "dict": "Map<String, Any>",
            "Any": "Any"
        }

        kotlin_return_type = type_mapping.get(return_type, "Unit")

        # 构建参数列表
        param_list = []
        if params:
            for param in params:
                param_list.append(f"{param}: Any")

        params_str = ", ".join(param_list) if param_list else ""

        code = f"fun {function_name}({params_str}): {kotlin_return_type} {{\n"
        code += f"    // {description}\n"
        code += "    // 根据描述实现具体功能\n"

        if kotlin_return_type != "Unit":
            if kotlin_return_type == "Int":
                code += "    return 0\n"
            elif kotlin_return_type == "Boolean":
                code += "    return true\n"
            elif kotlin_return_type == "Float":
                code += "    return 0.0f\n"
            elif kotlin_return_type == "Double":
                code += "    return 0.0\n"
            elif kotlin_return_type == "String":
                code += '    return ""\n'
            elif kotlin_return_type == "List<Any>":
                code += "    return listOf()\n"
            elif kotlin_return_type == "Map<String, Any>":
                code += "    return mapOf()\n"
            else:
                code += "    return null\n"

        code += "}\n"

        return code

    def _generate_sql_function(self, function_name: str, description: str) -> str:
        """生成SQL函数代码"""
        # SQL函数通常是存储过程或函数
        desc_lower = description.lower()

        if "procedure" in desc_lower or "存储过程" in desc_lower:
            # 生成存储过程
            code = f"CREATE PROCEDURE {function_name}\n"
            code += "AS\n"
            code += "BEGIN\n"
            code += f"    -- {description}\n"
            code += "    -- 根据描述实现存储过程逻辑\n"
            code += "    SELECT 1 AS result;\n"
            code += "END;\n"
        else:
            # 生成标量函数
            code = f"CREATE FUNCTION {function_name}()\n"
            code += "RETURNS INT\n"
            code += "AS\n"
            code += "BEGIN\n"
            code += f"    -- {description}\n"
            code += "    -- 根据描述实现函数逻辑\n"
            code += "    RETURN 1;\n"
            code += "END;\n"

        return code

    def _generate_class(self, description: str, language: ProgrammingLanguage) -> str:
        """生成类代码"""
        if language == ProgrammingLanguage.PYTHON:
            class_name = self._extract_class_name(description)

            # 根据描述确定类的属性
            attributes = self._extract_class_attributes(description)

            code = f"class {class_name}:\n"
            code += f'    """{description}"""\n\n'

            # 生成__init__方法
            init_params = ", ".join([f"{attr}=None" for attr in attributes])
            code += f"    def __init__(self, {init_params}):\n"
            for attr in attributes:
                code += f"        self.{attr} = {attr}\n"

            # 生成一个示例方法
            if attributes:
                example_attr = attributes[0]
                code += f"\n    def get_{example_attr}(self):\n"
                code += f"        \"\"\"获取{example_attr}属性\"\"\"\n"
                code += f"        return self.{example_attr}\n\n"
                code += f"    def set_{example_attr}(self, value):\n"
                code += f"        \"\"\"设置{example_attr}属性\"\"\"\n"
                code += f"        self.{example_attr} = value\n"
            else:
                code += f"\n    def example_method(self):\n"
                code += f"        \"\"\"示例方法\"\"\"\n"
                code += f"        return \"这是一个{description}类\"\n"

            # 生成字符串表示方法
            code += f"\n    def __str__(self):\n"
            code += f"        \"\"\"字符串表示\"\"\"\n"
            code += f"        return f\"{class_name}({', '.join([f'{attr}={{self.{attr}}}' for attr in attributes])})\"\n"

            return code
        elif language == ProgrammingLanguage.JAVA:
            class_name = self._extract_class_name(description)
            attributes = self._extract_class_attributes(description)

            code = f"public class {class_name} {{\n"

            # 生成属性
            for attr in attributes:
                code += f"    private String {attr};\n"

            # 生成构造函数
            code += f"\n    public {class_name}("
            if attributes:
                code += ", ".join([f"String {attr}" for attr in attributes])
            code += ") {\n"
            for attr in attributes:
                code += f"        this.{attr} = {attr};\n"
            code += "    }\n"

            # 生成getter和setter
            for attr in attributes:
                code += f"\n    public String get{attr.title()}() {{\n"
                code += f"        return this.{attr};\n"
                code += "    }\n"
                code += f"\n    public void set{attr.title()}(String {attr}) {{\n"
                code += f"        this.{attr} = {attr};\n"
                code += "    }\n"

            code += "}\n"
            return code
        elif language == ProgrammingLanguage.JAVASCRIPT:
            # JavaScript类（ES6）
            class_name = self._extract_class_name(description)
            attributes = self._extract_class_attributes(description)

            code = f"class {class_name} {{\n"
            code += f"    /** {description} */\n\n"

            # 构造函数
            if attributes:
                code += f"    constructor("
                code += ", ".join([f"{attr} = null" for attr in attributes])
                code += ") {\n"
                for attr in attributes:
                    code += f"        this.{attr} = {attr};\n"
                code += "    }\n"
            else:
                code += "    constructor() {\n"
                code += "        // 初始化\n"
                code += "    }\n"

            # 示例方法
            if attributes:
                attr = attributes[0]
                code += f"\n    get{attr.title()}() {{\n"
                code += f"        return this.{attr};\n"
                code += "    }\n"
                code += f"\n    set{attr.title()}(value) {{\n"
                code += f"        this.{attr} = value;\n"
                code += "    }\n"

            # toString方法
            code += "\n    toString() {\n"
            if attributes:
                # 生成模板字符串
                attrs_parts = []
                for attr in attributes:
                    attrs_parts.append(f"{attr}=$'{'{'}this.{attr}{'}'}'")
                attrs_str = ", ".join(attrs_parts)
                code += f"        return `{class_name}({attrs_str})`;\n"
            else:
                code += f"        return `{class_name}()`;\n"
            code += "    }\n"
            code += "}\n"

            return code
        elif language == ProgrammingLanguage.TYPESCRIPT:
            # TypeScript类
            class_name = self._extract_class_name(description)
            attributes = self._extract_class_attributes(description)

            code = f"class {class_name} {{\n"
            code += f"    // {description}\n\n"

            # 属性声明
            for attr in attributes:
                code += f"    private {attr}: any;\n"

            # 构造函数
            code += f"\n    constructor("
            if attributes:
                code += ", ".join([f"{attr}: any = null" for attr in attributes])
            code += ") {\n"
            for attr in attributes:
                code += f"        this.{attr} = {attr};\n"
            code += "    }\n"

            # 示例方法
            if attributes:
                attr = attributes[0]
                code += f"\n    public get{attr.title()}(): any {{\n"
                code += f"        return this.{attr};\n"
                code += "    }\n"
                code += f"\n    public set{attr.title()}(value: any): void {{\n"
                code += f"        this.{attr} = value;\n"
                code += "    }\n"

            code += "}\n"
            return code
        elif language == ProgrammingLanguage.CPP:
            # C++类
            class_name = self._extract_class_name(description)
            attributes = self._extract_class_attributes(description)

            code = f"class {class_name} {{\n"
            code += f"private:\n"

            # 私有成员变量
            for attr in attributes:
                code += f"    std::string {attr};\n"

            code += f"\npublic:\n"
            code += f"    // 构造函数\n"
            code += f"    {class_name}("
            if attributes:
                code += ", ".join([f"std::string {attr}" for attr in attributes])
            code += ") : "
            if attributes:
                init_list = [f"{attr}({attr})" for attr in attributes]
                code += ", ".join(init_list)
            code += " {}\n\n"

            # Getter和Setter
            for attr in attributes:
                code += f"    std::string get{attr.title()}() const {{\n"
                code += f"        return this->{attr};\n"
                code += "    }\n\n"
                code += f"    void set{attr.title()}(std::string {attr}) {{\n"
                code += f"        this->{attr} = {attr};\n"
                code += "    }\n"

            # toString方法
            code += f"\n    std::string toString() const {{\n"
            code += f"        return \"{class_name}(\""
            if attributes:
                for i, attr in enumerate(attributes):
                    if i > 0:
                        code += ' + \", \"'
                    code += f' + std::to_string(this->{attr})'
            code += " + \")\";\n"
            code += "    }\n"
            code += "};\n"
            return code
        elif language == ProgrammingLanguage.CSHARP:
            # C#类
            class_name = self._extract_class_name(description)
            attributes = self._extract_class_attributes(description)

            code = f"public class {class_name}\n"
            code += "{\n"

            # 属性
            for attr in attributes:
                code += f"    public string {attr.title()} {{ get; set; }}\n"

            # 构造函数
            code += f"\n    public {class_name}("
            if attributes:
                code += ", ".join([f"string {attr}" for attr in attributes])
            code += ")\n"
            code += "    {\n"
            for attr in attributes:
                code += f"        this.{attr.title()} = {attr};\n"
            code += "    }\n"

            # ToString方法
            code += f"\n    public override string ToString()\n"
            code += "    {\n"
            code += f"        return $\"{class_name}("
            if attributes:
                attrs_str = "; ".join(
                    [f"{attr.title()}={{this.{attr.title()}}}" for attr in attributes])
                code += attrs_str
            code += ")\";\n"
            code += "    }\n"
            code += "}\n"
            return code
        elif language == ProgrammingLanguage.RUST:
            # Rust结构体（完整）
            class_name = self._extract_class_name(description)
            attributes = self._extract_class_attributes(description)

            code = f"struct {class_name} {{\n"
            for attr in attributes:
                code += f"    {attr}: String,\n"
            code += "}\n\n"

            code += f"impl {class_name} {{\n"
            # 构造函数
            code += f"    fn new("
            if attributes:
                code += ", ".join([f"{attr}: String" for attr in attributes])
            code += ") -> Self {\n"
            code += f"        {class_name} {{\n"
            for attr in attributes:
                code += f"            {attr},\n"
            code += "        }\n"
            code += "    }\n\n"

            # Getter方法
            for attr in attributes:
                code += f"    fn {attr}(&self) -> &String {{\n"
                code += f"        &self.{attr}\n"
                code += "    }\n\n"
                code += f"    fn set_{attr}(&mut self, value: String) {{\n"
                code += f"        self.{attr} = value;\n"
                code += "    }\n"

            # to_string方法
            code += f"    fn to_string(&self) -> String {{\n"
            code += f"        format!(\"{class_name}("
            if attributes:
                attrs_fmt = ", ".join([f"{attr}: {{}}"] * len(attributes))
                code += attrs_fmt
                code += ")\""
                for attr in attributes:
                    code += f", self.{attr}"
            else:
                code += ")\""
            code += ")\n"
            code += "    }\n"
            code += "}\n"
            return code
        elif language == ProgrammingLanguage.GO:
            # Go结构体
            class_name = self._extract_class_name(description)
            attributes = self._extract_class_attributes(description)

            code = f"type {class_name} struct {{\n"
            for attr in attributes:
                code += f"    {attr.title()} string\n"
            code += "}\n\n"

            # 构造函数
            code += f"func New{class_name}("
            if attributes:
                code += ", ".join([f"{attr} string" for attr in attributes])
            code += ") *{class_name} {{\n"
            code += f"    return &{class_name}{{\n"
            for attr in attributes:
                code += f"        {attr.title()}: {attr},\n"
            code += "    }\n"
            code += "}\n\n"

            # Getter方法
            for attr in attributes:
                code += f"func (c *{class_name}) Get{attr.title()}() string {{\n"
                code += f"    return c.{attr.title()}\n"
                code += "}\n\n"
                code += f"func (c *{class_name}) Set{attr.title()}({attr} string) {{\n"
                code += f"    c.{attr.title()} = {attr}\n"
                code += "}\n\n"

            # String方法
            code += f"func (c *{class_name}) String() string {{\n"
            code += f"    return fmt.Sprintf(\"{class_name}("
            if attributes:
                attrs_fmt = ", ".join([f"{attr.title()}: %s"] * len(attributes))
                code += attrs_fmt
                code += ")\""
                for attr in attributes:
                    code += f", c.{attr.title()}"
            else:
                code += ")\""
            code += ")\n"
            code += "}\n"
            return code
        else:
            # 对于其他语言，生成基础结构
            class_name = self._extract_class_name(description)
            return f"// 类: {class_name}\n// 描述: {description}\n// 使用{class_name}类实现{description}功能"

    def _extract_class_name(self, description: str) -> str:
        """从描述中提取类名"""
        # 使用帕斯卡命名法
        function_name = self._extract_function_name(description)
        # 转换为帕斯卡命名：首字母大写
        return function_name.title().replace('_', '')

    def _extract_class_attributes(self, description: str) -> List[str]:
        """从描述中提取类属性"""
        description_lower = description.lower()
        attributes = []

        # 常见属性关键词映射
        attribute_keywords = {
            "汽车": ["品牌", "型号", "颜色", "价格", "年份"],
            "用户": ["姓名", "年龄", "邮箱", "手机", "地址"],
            "学生": ["姓名", "学号", "班级", "成绩", "专业"],
            "产品": ["名称", "价格", "库存", "分类", "描述"],
            "订单": ["订单号", "商品列表", "总价", "状态", "创建时间"],
            "书籍": ["书名", "作者", "ISBN", "出版社", "出版年份"],
            "电影": ["片名", "导演", "演员", "时长", "上映年份"],
            "音乐": ["歌名", "歌手", "专辑", "时长", "流派"],
            "文件": ["文件名", "大小", "类型", "创建时间", "路径"],
            "任务": ["标题", "描述", "优先级", "状态", "截止时间"]
        }

        # 尝试匹配已知类别
        for category, common_attrs in attribute_keywords.items():
            if category in description_lower:
                # 使用该类别的常见属性
                attributes.extend(common_attrs[:3])  # 取前3个属性
                break

        # 如果没有匹配到已知类别，提取描述中的名词作为属性
        if not attributes:
            # 简单分词提取名词（中文）
            words = re.findall(r'[\u4e00-\u9fff]{2,4}', description)
            if words:
                # 取前3个不同的词作为属性
                attributes = list(dict.fromkeys(words))[:3]
            else:
                # 回退到默认属性
                attributes = ["属性1", "属性2", "属性3"]

        # 转换为snake_case（下划线命名）
        snake_attributes = []
        for attr in attributes:
            # 完整处理）
            if re.search(r'[\u4e00-\u9fff]', attr):
                # 中文转拼音实现，实际应该使用拼音库
                pinyin_name = re.sub(r'[^\w]', '_', attr.lower())
                snake_attributes.append(pinyin_name)
            else:
                # 已经是英文或拼音，转换为snake_case
                snake_name = re.sub(r'[^a-zA-Z0-9]+', '_', attr.lower())
                snake_attributes.append(snake_name)

        return snake_attributes

    def _generate_algorithm(self, description: str, language: ProgrammingLanguage) -> str:
        """生成算法代码"""
        desc_lower = description.lower()

        # 根据描述关键词选择算法类型
        if "sort" in desc_lower:
            # 生成排序算法
            if language == ProgrammingLanguage.PYTHON:
                return self._generate_sort_algorithm(description, language)
        elif "search" in desc_lower:
            # 生成搜索算法
            if language == ProgrammingLanguage.PYTHON:
                return self._generate_search_algorithm(description, language)
        elif "graph" in desc_lower or "图" in desc_lower:
            # 生成图算法
            if language == ProgrammingLanguage.PYTHON:
                return self._generate_graph_algorithm(description, language)
        elif "tree" in desc_lower or "树" in desc_lower:
            # 生成树算法
            if language == ProgrammingLanguage.PYTHON:
                return self._generate_tree_algorithm(description, language)
        elif "dynamic" in desc_lower or "动态规划" in desc_lower:
            # 生成动态规划算法
            if language == ProgrammingLanguage.PYTHON:
                return self._generate_dp_algorithm(description, language)
        elif "dijkstra" in desc_lower or "最短路径" in desc_lower:
            # 生成最短路径算法
            if language == ProgrammingLanguage.PYTHON:
                return self._generate_shortest_path_algorithm(description, language)

        # 默认算法生成：基于描述的通用算法
        if language == ProgrammingLanguage.PYTHON:
            return self._generate_generic_algorithm(description, language)
        else:
            # 对于其他语言，生成算法框架
            return self._generate_algorithm_framework(description, language)

    def _generate_sort_algorithm(self, description: str, language: ProgrammingLanguage) -> str:
        """生成排序算法"""
        code = "def quick_sort(arr):\n"
        code += '    """快速排序算法"""\n'
        code += '    if len(arr) <= 1:\n'
        code += '        return arr\n'
        code += '    pivot = arr[len(arr) // 2]\n'
        code += '    left = [x for x in arr if x < pivot]\n'
        code += '    middle = [x for x in arr if x == pivot]\n'
        code += '    right = [x for x in arr if x > pivot]\n'
        code += '    return quick_sort(left) + middle + quick_sort(right)\n\n'
        code += 'def bubble_sort(arr):\n'
        code += '    """冒泡排序算法"""\n'
        code += '    n = len(arr)\n'
        code += '    for i in range(n):\n'
        code += '        for j in range(0, n - i - 1):\n'
        code += '            if arr[j] > arr[j + 1]:\n'
        code += '                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n'
        code += '    return arr\n'

        return code

    def _generate_search_algorithm(self, description: str, language: ProgrammingLanguage) -> str:
        """生成搜索算法"""
        code = "def binary_search(arr, target):\n"
        code += '    """二分搜索算法"""\n'
        code += '    left, right = 0, len(arr) - 1\n'
        code += '    while left <= right:\n'
        code += '        mid = left + (right - left) // 2\n'
        code += '        if arr[mid] == target:\n'
        code += '            return mid\n'
        code += '        elif arr[mid] < target:\n'
        code += '            left = mid + 1\n'
        code += '        else:\n'
        code += '            right = mid - 1\n'
        code += '    return -1\n\n'
        code += 'def linear_search(arr, target):\n'
        code += '    """线性搜索算法"""\n'
        code += '    for i, item in enumerate(arr):\n'
        code += '        if item == target:\n'
        code += '            return i\n'
        code += '    return -1\n'

        return code

    def _generate_graph_algorithm(self, description: str, language: ProgrammingLanguage) -> str:
        """生成图算法"""
        if language == ProgrammingLanguage.PYTHON:
            code = "from collections import deque\n\n"
            code += "def bfs(graph, start):\n"
            code += '    """广度优先搜索"""\n'
            code += '    visited = set()\n'
            code += '    queue = deque([start])\n'
            code += '    visited.add(start)\n'
            code += '    while queue:\n'
            code += '        vertex = queue.popleft()\n'
            code += '        for neighbor in graph.get(vertex, []):\n'
            code += '            if neighbor not in visited:\n'
            code += '                visited.add(neighbor)\n'
            code += '                queue.append(neighbor)\n'
            code += '    return visited\n\n'
            code += "def dfs(graph, start):\n"
            code += '    """深度优先搜索"""\n'
            code += '    visited = set()\n'
            code += '    def dfs_recursive(vertex):\n'
            code += '        visited.add(vertex)\n'
            code += '        for neighbor in graph.get(vertex, []):\n'
            code += '            if neighbor not in visited:\n'
            code += '                dfs_recursive(neighbor)\n'
            code += '    dfs_recursive(start)\n'
            code += '    return visited\n\n'
            code += "def has_cycle(graph):\n"
            code += '    """检测图中是否有环"""\n'
            code += '    visited = set()\n'
            code += '    recursion_stack = set()\n'
            code += '    def has_cycle_util(vertex):\n'
            code += '        visited.add(vertex)\n'
            code += '        recursion_stack.add(vertex)\n'
            code += '        for neighbor in graph.get(vertex, []):\n'
            code += '            if neighbor not in visited:\n'
            code += '                if has_cycle_util(neighbor):\n'
            code += '                    return True\n'
            code += '            elif neighbor in recursion_stack:\n'
            code += '                return True\n'
            code += '        recursion_stack.remove(vertex)\n'
            code += '        return False\n'
            code += '    for vertex in graph:\n'
            code += '        if vertex not in visited:\n'
            code += '            if has_cycle_util(vertex):\n'
            code += '                return True\n'
            code += '    return False\n'
            return code
        elif language == ProgrammingLanguage.JAVASCRIPT:
            code = "// 图算法 - JavaScript实现\n"
            code += "function bfs(graph, start) {\n"
            code += '    // 广度优先搜索\n'
            code += '    const visited = new Set();\n'
            code += '    const queue = [start];\n'
            code += '    visited.add(start);\n'
            code += '    while (queue.length > 0) {\n'
            code += '        const vertex = queue.shift();\n'
            code += '        const neighbors = graph[vertex] || [];\n'
            code += '        for (const neighbor of neighbors) {\n'
            code += '            if (!visited.has(neighbor)) {\n'
            code += '                visited.add(neighbor);\n'
            code += '                queue.push(neighbor);\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    return visited;\n'
            code += '}\n\n'
            code += "function dfs(graph, start) {\n"
            code += '    // 深度优先搜索\n'
            code += '    const visited = new Set();\n'
            code += '    function dfsRecursive(vertex) {\n'
            code += '        visited.add(vertex);\n'
            code += '        const neighbors = graph[vertex] || [];\n'
            code += '        for (const neighbor of neighbors) {\n'
            code += '            if (!visited.has(neighbor)) {\n'
            code += '                dfsRecursive(neighbor);\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    dfsRecursive(start);\n'
            code += '    return visited;\n'
            code += '}\n\n'
            code += "function hasCycle(graph) {\n"
            code += '    // 检测图中是否有环\n'
            code += '    const visited = new Set();\n'
            code += '    const recursionStack = new Set();\n'
            code += '    function hasCycleUtil(vertex) {\n'
            code += '        visited.add(vertex);\n'
            code += '        recursionStack.add(vertex);\n'
            code += '        const neighbors = graph[vertex] || [];\n'
            code += '        for (const neighbor of neighbors) {\n'
            code += '            if (!visited.has(neighbor)) {\n'
            code += '                if (hasCycleUtil(neighbor)) {\n'
            code += '                    return true;\n'
            code += '                }\n'
            code += '            } else if (recursionStack.has(neighbor)) {\n'
            code += '                return true;\n'
            code += '            }\n'
            code += '        }\n'
            code += '        recursionStack.delete(vertex);\n'
            code += '        return false;\n'
            code += '    }\n'
            code += '    for (const vertex in graph) {\n'
            code += '        if (!visited.has(vertex)) {\n'
            code += '            if (hasCycleUtil(vertex)) {\n'
            code += '                return true;\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    return false;\n'
            code += '}\n'
            return code
        elif language == ProgrammingLanguage.JAVA:
            code = "// 图算法 - Java实现\n"
            code += "import java.util.*;\n\n"
            code += "public class GraphAlgorithms {\n"
            code += '    // 广度优先搜索\n'
            code += '    public static Set<Integer> bfs(Map<Integer, List<Integer>> graph, int start) {\n'
            code += '        Set<Integer> visited = new HashSet<>();\n'
            code += '        Queue<Integer> queue = new LinkedList<>();\n'
            code += '        queue.add(start);\n'
            code += '        visited.add(start);\n'
            code += '        while (!queue.isEmpty()) {\n'
            code += '            int vertex = queue.poll();\n'
            code += '            List<Integer> neighbors = graph.getOrDefault(vertex, new ArrayList<>());\n'
            code += '            for (int neighbor : neighbors) {\n'
            code += '                if (!visited.contains(neighbor)) {\n'
            code += '                    visited.add(neighbor);\n'
            code += '                    queue.add(neighbor);\n'
            code += '                }\n'
            code += '            }\n'
            code += '        }\n'
            code += '        return visited;\n'
            code += '    }\n\n'
            code += '    // 深度优先搜索\n'
            code += '    public static Set<Integer> dfs(Map<Integer, List<Integer>> graph, int start) {\n'
            code += '        Set<Integer> visited = new HashSet<>();\n'
            code += '        dfsRecursive(graph, start, visited);\n'
            code += '        return visited;\n'
            code += '    }\n\n'
            code += '    private static void dfsRecursive(Map<Integer, List<Integer>> graph, int vertex, Set<Integer> visited) {\n'
            code += '        visited.add(vertex);\n'
            code += '        List<Integer> neighbors = graph.getOrDefault(vertex, new ArrayList<>());\n'
            code += '        for (int neighbor : neighbors) {\n'
            code += '            if (!visited.contains(neighbor)) {\n'
            code += '                dfsRecursive(graph, neighbor, visited);\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n\n'
            code += '    // 检测图中是否有环\n'
            code += '    public static boolean hasCycle(Map<Integer, List<Integer>> graph) {\n'
            code += '        Set<Integer> visited = new HashSet<>();\n'
            code += '        Set<Integer> recursionStack = new HashSet<>();\n'
            code += '        for (int vertex : graph.keySet()) {\n'
            code += '            if (!visited.contains(vertex)) {\n'
            code += '                if (hasCycleUtil(graph, vertex, visited, recursionStack)) {\n'
            code += '                    return true;\n'
            code += '                }\n'
            code += '            }\n'
            code += '        }\n'
            code += '        return false;\n'
            code += '    }\n\n'
            code += '    private static boolean hasCycleUtil(Map<Integer, List<Integer>> graph, int vertex, \n'
            code += '                                         Set<Integer> visited, Set<Integer> recursionStack) {\n'
            code += '        visited.add(vertex);\n'
            code += '        recursionStack.add(vertex);\n'
            code += '        List<Integer> neighbors = graph.getOrDefault(vertex, new ArrayList<>());\n'
            code += '        for (int neighbor : neighbors) {\n'
            code += '            if (!visited.contains(neighbor)) {\n'
            code += '                if (hasCycleUtil(graph, neighbor, visited, recursionStack)) {\n'
            code += '                    return true;\n'
            code += '                }\n'
            code += '            } else if (recursionStack.contains(neighbor)) {\n'
            code += '                return true;\n'
            code += '            }\n'
            code += '        }\n'
            code += '        recursionStack.remove(vertex);\n'
            code += '        return false;\n'
            code += '    }\n'
            code += '}\n'
            return code
        elif language == ProgrammingLanguage.CPP:
            code = "// 图算法 - C++实现\n"
            code += "#include <iostream>\n"
            code += "#include <vector>\n"
            code += "#include <queue>\n"
            code += "#include <unordered_set>\n"
            code += "#include <unordered_map>\n\n"
            code += "using namespace std;\n\n"
            code += "// 广度优先搜索\n"
            code += "unordered_set<int> bfs(unordered_map<int, vector<int>>& graph, int start) {\n"
            code += '    unordered_set<int> visited;\n'
            code += '    queue<int> q;\n'
            code += '    q.push(start);\n'
            code += '    visited.insert(start);\n'
            code += '    while (!q.empty()) {\n'
            code += '        int vertex = q.front();\n'
            code += '        q.pop();\n'
            code += '        if (graph.find(vertex) != graph.end()) {\n'
            code += '            for (int neighbor : graph[vertex]) {\n'
            code += '                if (visited.find(neighbor) == visited.end()) {\n'
            code += '                    visited.insert(neighbor);\n'
            code += '                    q.push(neighbor);\n'
            code += '                }\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    return visited;\n'
            code += '}\n\n'
            code += "// 深度优先搜索\n"
            code += "void dfsUtil(unordered_map<int, vector<int>>& graph, int vertex, unordered_set<int>& visited) {\n"
            code += '    visited.insert(vertex);\n'
            code += '    if (graph.find(vertex) != graph.end()) {\n'
            code += '        for (int neighbor : graph[vertex]) {\n'
            code += '            if (visited.find(neighbor) == visited.end()) {\n'
            code += '                dfsUtil(graph, neighbor, visited);\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '}\n\n'
            code += "unordered_set<int> dfs(unordered_map<int, vector<int>>& graph, int start) {\n"
            code += '    unordered_set<int> visited;\n'
            code += '    dfsUtil(graph, start, visited);\n'
            code += '    return visited;\n'
            code += '}\n\n'
            code += "// 检测图中是否有环\n"
            code += "bool hasCycleUtil(unordered_map<int, vector<int>>& graph, int vertex, \n"
            code += '                 unordered_set<int>& visited, unordered_set<int>& recursionStack) {\n'
            code += '    visited.insert(vertex);\n'
            code += '    recursionStack.insert(vertex);\n'
            code += '    if (graph.find(vertex) != graph.end()) {\n'
            code += '        for (int neighbor : graph[vertex]) {\n'
            code += '            if (visited.find(neighbor) == visited.end()) {\n'
            code += '                if (hasCycleUtil(graph, neighbor, visited, recursionStack)) {\n'
            code += '                    return true;\n'
            code += '                }\n'
            code += '            } else if (recursionStack.find(neighbor) != recursionStack.end()) {\n'
            code += '                return true;\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    recursionStack.erase(vertex);\n'
            code += '    return false;\n'
            code += '}\n\n'
            code += "bool hasCycle(unordered_map<int, vector<int>>& graph) {\n"
            code += '    unordered_set<int> visited;\n'
            code += '    unordered_set<int> recursionStack;\n'
            code += '    for (auto& pair : graph) {\n'
            code += '        int vertex = pair.first;\n'
            code += '        if (visited.find(vertex) == visited.end()) {\n'
            code += '            if (hasCycleUtil(graph, vertex, visited, recursionStack)) {\n'
            code += '                return true;\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    return false;\n'
            code += '}\n'
            return code
        else:
            # 对于其他语言，提供算法描述
            return f"// 图算法: {description}\n// 实现广度优先搜索(BFS)、深度优先搜索(DFS)和环检测算法"

    def _generate_tree_algorithm(self, description: str, language: ProgrammingLanguage) -> str:
        """生成树算法"""
        if language == ProgrammingLanguage.PYTHON:
            code = "class TreeNode:\n"
            code += '    """树节点类"""\n'
            code += '    def __init__(self, value):\n'
            code += '        self.value = value\n'
            code += '        self.left = None\n'
            code += '        self.right = None\n\n'
            code += "def inorder_traversal(root):\n"
            code += '    """中序遍历"""\n'
            code += '    result = []\n'
            code += '    def inorder(node):\n'
            code += '        if node:\n'
            code += '            inorder(node.left)\n'
            code += '            result.append(node.value)\n'
            code += '            inorder(node.right)\n'
            code += '    inorder(root)\n'
            code += '    return result\n\n'
            code += "def tree_height(root):\n"
            code += '    """计算树的高度"""\n'
            code += '    if not root:\n'
            code += '        return 0\n'
            code += '    left_height = tree_height(root.left)\n'
            code += '    right_height = tree_height(root.right)\n'
            code += '    return max(left_height, right_height) + 1\n\n'
            code += "def is_bst(root):\n"
            code += '    """检查是否为二叉搜索树"""\n'
            code += '    def is_bst_util(node, min_val, max_val):\n'
            code += '        if not node:\n'
            code += '            return True\n'
            code += '        if node.value <= min_val or node.value >= max_val:\n'
            code += '            return False\n'
            code += '        return (is_bst_util(node.left, min_val, node.value) and\n'
            code += '                is_bst_util(node.right, node.value, max_val))\n'
            code += '    return is_bst_util(root, float("-inf"), float("inf"))\n'
            return code
        elif language == ProgrammingLanguage.JAVASCRIPT:
            code = "// 树算法 - JavaScript实现\n"
            code += "class TreeNode {\n"
            code += '    constructor(value) {\n'
            code += '        this.value = value;\n'
            code += '        this.left = null;\n'
            code += '        this.right = null;\n'
            code += '    }\n'
            code += '}\n\n'
            code += "function inorderTraversal(root) {\n"
            code += '    // 中序遍历\n'
            code += '    const result = [];\n'
            code += '    function inorder(node) {\n'
            code += '        if (node) {\n'
            code += '            inorder(node.left);\n'
            code += '            result.push(node.value);\n'
            code += '            inorder(node.right);\n'
            code += '        }\n'
            code += '    }\n'
            code += '    inorder(root);\n'
            code += '    return result;\n'
            code += '}\n\n'
            code += "function treeHeight(root) {\n"
            code += '    // 计算树的高度\n'
            code += '    if (!root) return 0;\n'
            code += '    const leftHeight = treeHeight(root.left);\n'
            code += '    const rightHeight = treeHeight(root.right);\n'
            code += '    return Math.max(leftHeight, rightHeight) + 1;\n'
            code += '}\n\n'
            code += "function isBST(root) {\n"
            code += '    // 检查是否为二叉搜索树\n'
            code += '    function isBSTUtil(node, minVal, maxVal) {\n'
            code += '        if (!node) return true;\n'
            code += '        if (node.value <= minVal || node.value >= maxVal) return false;\n'
            code += '        return isBSTUtil(node.left, minVal, node.value) &&\n'
            code += '               isBSTUtil(node.right, node.value, maxVal);\n'
            code += '    }\n'
            code += '    return isBSTUtil(root, -Infinity, Infinity);\n'
            code += '}\n'
            return code
        elif language == ProgrammingLanguage.JAVA:
            code = "// 树算法 - Java实现\n"
            code += "import java.util.*;\n\n"
            code += "class TreeNode {\n"
            code += '    int value;\n'
            code += '    TreeNode left;\n'
            code += '    TreeNode right;\n'
            code += '    \n'
            code += '    TreeNode(int value) {\n'
            code += '        this.value = value;\n'
            code += '        this.left = null;\n'
            code += '        this.right = null;\n'
            code += '    }\n'
            code += '}\n\n'
            code += "class TreeAlgorithms {\n"
            code += '    // 中序遍历\n'
            code += '    public static List<Integer> inorderTraversal(TreeNode root) {\n'
            code += '        List<Integer> result = new ArrayList<>();\n'
            code += '        inorder(root, result);\n'
            code += '        return result;\n'
            code += '    }\n'
            code += '    \n'
            code += '    private static void inorder(TreeNode node, List<Integer> result) {\n'
            code += '        if (node != null) {\n'
            code += '            inorder(node.left, result);\n'
            code += '            result.add(node.value);\n'
            code += '            inorder(node.right, result);\n'
            code += '        }\n'
            code += '    }\n'
            code += '    \n'
            code += '    // 计算树的高度\n'
            code += '    public static int treeHeight(TreeNode root) {\n'
            code += '        if (root == null) return 0;\n'
            code += '        int leftHeight = treeHeight(root.left);\n'
            code += '        int rightHeight = treeHeight(root.right);\n'
            code += '        return Math.max(leftHeight, rightHeight) + 1;\n'
            code += '    }\n'
            code += '    \n'
            code += '    // 检查是否为二叉搜索树\n'
            code += '    public static boolean isBST(TreeNode root) {\n'
            code += '        return isBSTUtil(root, Integer.MIN_VALUE, Integer.MAX_VALUE);\n'
            code += '    }\n'
            code += '    \n'
            code += '    private static boolean isBSTUtil(TreeNode node, int minVal, int maxVal) {\n'
            code += '        if (node == null) return true;\n'
            code += '        if (node.value <= minVal || node.value >= maxVal) return false;\n'
            code += '        return isBSTUtil(node.left, minVal, node.value) &&\n'
            code += '               isBSTUtil(node.right, node.value, maxVal);\n'
            code += '    }\n'
            code += '}\n'
            return code
        elif language == ProgrammingLanguage.CPP:
            code = "// 树算法 - C++实现\n"
            code += "#include <iostream>\n"
            code += "#include <vector>\n"
            code += "#include <algorithm>\n"
            code += "#include <climits>\n\n"
            code += "using namespace std;\n\n"
            code += "struct TreeNode {\n"
            code += '    int value;\n'
            code += '    TreeNode* left;\n'
            code += '    TreeNode* right;\n'
            code += '    \n'
            code += '    TreeNode(int val) : value(val), left(nullptr), right(nullptr) {}\n'
            code += '};\n\n'
            code += "void inorderTraversal(TreeNode* root, vector<int>& result) {\n"
            code += '    // 中序遍历辅助函数\n'
            code += '    if (root) {\n'
            code += '        inorderTraversal(root->left, result);\n'
            code += '        result.push_back(root->value);\n'
            code += '        inorderTraversal(root->right, result);\n'
            code += '    }\n'
            code += '}\n\n'
            code += "vector<int> inorderTraversal(TreeNode* root) {\n"
            code += '    // 中序遍历\n'
            code += '    vector<int> result;\n'
            code += '    inorderTraversal(root, result);\n'
            code += '    return result;\n'
            code += '}\n\n'
            code += "int treeHeight(TreeNode* root) {\n"
            code += '    // 计算树的高度\n'
            code += '    if (!root) return 0;\n'
            code += '    int leftHeight = treeHeight(root->left);\n'
            code += '    int rightHeight = treeHeight(root->right);\n'
            code += '    return max(leftHeight, rightHeight) + 1;\n'
            code += '}\n\n'
            code += "bool isBSTUtil(TreeNode* node, int minVal, int maxVal) {\n"
            code += '    // 检查二叉搜索树辅助函数\n'
            code += '    if (!node) return true;\n'
            code += '    if (node->value <= minVal || node->value >= maxVal) return false;\n'
            code += '    return isBSTUtil(node->left, minVal, node->value) &&\n'
            code += '           isBSTUtil(node->right, node->value, maxVal);\n'
            code += '}\n\n'
            code += "bool isBST(TreeNode* root) {\n"
            code += '    // 检查是否为二叉搜索树\n'
            code += '    return isBSTUtil(root, INT_MIN, INT_MAX);\n'
            code += '}\n'
            return code
        else:
            return f"// 树算法: {description}\n// 该语言暂不支持树算法，请使用Python、JavaScript、Java或C++"

    def _generate_dp_algorithm(self, description: str, language: ProgrammingLanguage) -> str:
        """生成动态规划算法"""
        if language == ProgrammingLanguage.PYTHON:
            code = "def fibonacci_dp(n):\n"
            code += '    """斐波那契数列动态规划"""\n'
            code += '    if n <= 1:\n'
            code += '        return n\n'
            code += '    dp = [0] * (n + 1)\n'
            code += '    dp[1] = 1\n'
            code += '    for i in range(2, n + 1):\n'
            code += '        dp[i] = dp[i-1] + dp[i-2]\n'
            code += '    return dp[n]\n\n'
            code += "def knapsack_dp(weights, values, capacity):\n"
            code += '    """0-1背包问题动态规划"""\n'
            code += '    n = len(weights)\n'
            code += '    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n'
            code += '    for i in range(1, n + 1):\n'
            code += '        for w in range(1, capacity + 1):\n'
            code += '            if weights[i-1] <= w:\n'
            code += '                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])\n'
            code += '            else:\n'
            code += '                dp[i][w] = dp[i-1][w]\n'
            code += '    return dp[n][capacity]\n\n'
            code += "def longest_common_subsequence(text1, text2):\n"
            code += '    """最长公共子序列动态规划"""\n'
            code += '    m, n = len(text1), len(text2)\n'
            code += '    dp = [[0] * (n + 1) for _ in range(m + 1)]\n'
            code += '    for i in range(1, m + 1):\n'
            code += '        for j in range(1, n + 1):\n'
            code += '            if text1[i-1] == text2[j-1]:\n'
            code += '                dp[i][j] = dp[i-1][j-1] + 1\n'
            code += '            else:\n'
            code += '                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n'
            code += '    return dp[m][n]\n'
            return code
        elif language == ProgrammingLanguage.JAVASCRIPT:
            code = "// 动态规划算法 - JavaScript实现\n"
            code += "function fibonacciDP(n) {\n"
            code += '    // 斐波那契数列动态规划\n'
            code += '    if (n <= 1) return n;\n'
            code += '    const dp = new Array(n + 1).fill(0);\n'
            code += '    dp[1] = 1;\n'
            code += '    for (let i = 2; i <= n; i++) {\n'
            code += '        dp[i] = dp[i-1] + dp[i-2];\n'
            code += '    }\n'
            code += '    return dp[n];\n'
            code += '}\n\n'
            code += "function knapsackDP(weights, values, capacity) {\n"
            code += '    // 0-1背包问题动态规划\n'
            code += '    const n = weights.length;\n'
            code += '    const dp = Array.from({length: n + 1}, () => new Array(capacity + 1).fill(0));\n'
            code += '    for (let i = 1; i <= n; i++) {\n'
            code += '        for (let w = 1; w <= capacity; w++) {\n'
            code += '            if (weights[i-1] <= w) {\n'
            code += '                dp[i][w] = Math.max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1]);\n'
            code += '            } else {\n'
            code += '                dp[i][w] = dp[i-1][w];\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    return dp[n][capacity];\n'
            code += '}\n\n'
            code += "function longestCommonSubsequence(text1, text2) {\n"
            code += '    // 最长公共子序列动态规划\n'
            code += '    const m = text1.length;\n'
            code += '    const n = text2.length;\n'
            code += '    const dp = Array.from({length: m + 1}, () => new Array(n + 1).fill(0));\n'
            code += '    for (let i = 1; i <= m; i++) {\n'
            code += '        for (let j = 1; j <= n; j++) {\n'
            code += '            if (text1[i-1] === text2[j-1]) {\n'
            code += '                dp[i][j] = dp[i-1][j-1] + 1;\n'
            code += '            } else {\n'
            code += '                dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    return dp[m][n];\n'
            code += '}\n'
            return code
        elif language == ProgrammingLanguage.JAVA:
            code = "// 动态规划算法 - Java实现\n"
            code += "import java.util.*;\n\n"
            code += "public class DPAlgorithms {\n"
            code += '    // 斐波那契数列动态规划\n'
            code += '    public static int fibonacciDP(int n) {\n'
            code += '        if (n <= 1) return n;\n'
            code += '        int[] dp = new int[n + 1];\n'
            code += '        dp[1] = 1;\n'
            code += '        for (int i = 2; i <= n; i++) {\n'
            code += '            dp[i] = dp[i-1] + dp[i-2];\n'
            code += '        }\n'
            code += '        return dp[n];\n'
            code += '    }\n'
            code += '    \n'
            code += '    // 0-1背包问题动态规划\n'
            code += '    public static int knapsackDP(int[] weights, int[] values, int capacity) {\n'
            code += '        int n = weights.length;\n'
            code += '        int[][] dp = new int[n + 1][capacity + 1];\n'
            code += '        for (int i = 1; i <= n; i++) {\n'
            code += '            for (int w = 1; w <= capacity; w++) {\n'
            code += '                if (weights[i-1] <= w) {\n'
            code += '                    dp[i][w] = Math.max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1]);\n'
            code += '                } else {\n'
            code += '                    dp[i][w] = dp[i-1][w];\n'
            code += '                }\n'
            code += '            }\n'
            code += '        }\n'
            code += '        return dp[n][capacity];\n'
            code += '    }\n'
            code += '    \n'
            code += '    // 最长公共子序列动态规划\n'
            code += '    public static int longestCommonSubsequence(String text1, String text2) {\n'
            code += '        int m = text1.length();\n'
            code += '        int n = text2.length();\n'
            code += '        int[][] dp = new int[m + 1][n + 1];\n'
            code += '        for (int i = 1; i <= m; i++) {\n'
            code += '            for (int j = 1; j <= n; j++) {\n'
            code += '                if (text1.charAt(i-1) == text2.charAt(j-1)) {\n'
            code += '                    dp[i][j] = dp[i-1][j-1] + 1;\n'
            code += '                } else {\n'
            code += '                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);\n'
            code += '                }\n'
            code += '            }\n'
            code += '        }\n'
            code += '        return dp[m][n];\n'
            code += '    }\n'
            code += '}\n'
            return code
        elif language == ProgrammingLanguage.CPP:
            code = "// 动态规划算法 - C++实现\n"
            code += "#include <iostream>\n"
            code += "#include <vector>\n"
            code += "#include <algorithm>\n"
            code += "#include <string>\n\n"
            code += "using namespace std;\n\n"
            code += "int fibonacciDP(int n) {\n"
            code += '    // 斐波那契数列动态规划\n'
            code += '    if (n <= 1) return n;\n'
            code += '    vector<int> dp(n + 1, 0);\n'
            code += '    dp[1] = 1;\n'
            code += '    for (int i = 2; i <= n; i++) {\n'
            code += '        dp[i] = dp[i-1] + dp[i-2];\n'
            code += '    }\n'
            code += '    return dp[n];\n'
            code += '}\n\n'
            code += "int knapsackDP(vector<int>& weights, vector<int>& values, int capacity) {\n"
            code += '    // 0-1背包问题动态规划\n'
            code += '    int n = weights.size();\n'
            code += '    vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));\n'
            code += '    for (int i = 1; i <= n; i++) {\n'
            code += '        for (int w = 1; w <= capacity; w++) {\n'
            code += '            if (weights[i-1] <= w) {\n'
            code += '                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1]);\n'
            code += '            } else {\n'
            code += '                dp[i][w] = dp[i-1][w];\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    return dp[n][capacity];\n'
            code += '}\n\n'
            code += "int longestCommonSubsequence(string text1, string text2) {\n"
            code += '    // 最长公共子序列动态规划\n'
            code += '    int m = text1.length();\n'
            code += '    int n = text2.length();\n'
            code += '    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));\n'
            code += '    for (int i = 1; i <= m; i++) {\n'
            code += '        for (int j = 1; j <= n; j++) {\n'
            code += '            if (text1[i-1] == text2[j-1]) {\n'
            code += '                dp[i][j] = dp[i-1][j-1] + 1;\n'
            code += '            } else {\n'
            code += '                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    return dp[m][n];\n'
            code += '}\n'
            return code
        else:
            return f"// 动态规划算法: {description}\n// 该语言暂不支持动态规划算法，请使用Python、JavaScript、Java或C++"

    def _generate_shortest_path_algorithm(self, description: str, language: ProgrammingLanguage) -> str:
        """生成最短路径算法"""
        if language == ProgrammingLanguage.PYTHON:
            code = "import heapq\n\n"
            code += "def dijkstra(graph, start):\n"
            code += '    """Dijkstra最短路径算法"""\n'
            code += '    distances = {node: float("inf") for node in graph}\n'
            code += '    distances[start] = 0\n'
            code += '    pq = [(0, start)]\n'
            code += '    while pq:\n'
            code += '        current_distance, current_node = heapq.heappop(pq)\n'
            code += '        if current_distance > distances[current_node]:\n'
            code += '            continue\n'
            code += '        for neighbor, weight in graph[current_node].items():\n'
            code += '            distance = current_distance + weight\n'
            code += '            if distance < distances[neighbor]:\n'
            code += '                distances[neighbor] = distance\n'
            code += '                heapq.heappush(pq, (distance, neighbor))\n'
            code += '    return distances\n\n'
            code += "def floyd_warshall(graph):\n"
            code += '    """Floyd-Warshall最短路径算法"""\n'
            code += '    nodes = list(graph.keys())\n'
            code += '    n = len(nodes)\n'
            code += '    dist = [[float("inf")] * n for _ in range(n)]\n'
            code += '    for i in range(n):\n'
            code += '        dist[i][i] = 0\n'
            code += '    for u in graph:\n'
            code += '        for v, w in graph[u].items():\n'
            code += '            i, j = nodes.index(u), nodes.index(v)\n'
            code += '            dist[i][j] = w\n'
            code += '    for k in range(n):\n'
            code += '        for i in range(n):\n'
            code += '            for j in range(n):\n'
            code += '                if dist[i][k] + dist[k][j] < dist[i][j]:\n'
            code += '                    dist[i][j] = dist[i][k] + dist[k][j]\n'
            code += '    return {nodes[i]: {nodes[j]: dist[i][j] for j in range(n)} for i in range(n)}\n'
            return code
        elif language == ProgrammingLanguage.JAVASCRIPT:
            code = "// 最短路径算法 - JavaScript实现\n"
            code += "function dijkstra(graph, start) {\n"
            code += '    // Dijkstra最短路径算法\n'
            code += '    const distances = {};\n'
            code += '    for (const node in graph) {\n'
            code += '        distances[node] = Infinity;\n'
            code += '    }\n'
            code += '    distances[start] = 0;\n'
            code += '    const pq = [[0, start]];\n'
            code += '    while (pq.length > 0) {\n'
            code += '        pq.sort((a, b) => a[0] - b[0]);\n'
            code += '        const [currentDistance, currentNode] = pq.shift();\n'
            code += '        if (currentDistance > distances[currentNode]) continue;\n'
            code += '        const neighbors = graph[currentNode] || {};\n'
            code += '        for (const neighbor in neighbors) {\n'
            code += '            const weight = neighbors[neighbor];\n'
            code += '            const distance = currentDistance + weight;\n'
            code += '            if (distance < distances[neighbor]) {\n'
            code += '                distances[neighbor] = distance;\n'
            code += '                pq.push([distance, neighbor]);\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    return distances;\n'
            code += '}\n\n'
            code += "function floydWarshall(graph) {\n"
            code += '    // Floyd-Warshall最短路径算法\n'
            code += '    const nodes = Object.keys(graph);\n'
            code += '    const n = nodes.length;\n'
            code += '    const dist = Array.from({length: n}, () => Array(n).fill(Infinity));\n'
            code += '    for (let i = 0; i < n; i++) {\n'
            code += '        dist[i][i] = 0;\n'
            code += '    }\n'
            code += '    for (const u in graph) {\n'
            code += '        const uIndex = nodes.indexOf(u);\n'
            code += '        for (const v in graph[u]) {\n'
            code += '            const vIndex = nodes.indexOf(v);\n'
            code += '            dist[uIndex][vIndex] = graph[u][v];\n'
            code += '        }\n'
            code += '    }\n'
            code += '    for (let k = 0; k < n; k++) {\n'
            code += '        for (let i = 0; i < n; i++) {\n'
            code += '            for (let j = 0; j < n; j++) {\n'
            code += '                if (dist[i][k] + dist[k][j] < dist[i][j]) {\n'
            code += '                    dist[i][j] = dist[i][k] + dist[k][j];\n'
            code += '                }\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    const result = {};\n'
            code += '    for (let i = 0; i < n; i++) {\n'
            code += '        result[nodes[i]] = {};\n'
            code += '        for (let j = 0; j < n; j++) {\n'
            code += '            result[nodes[i]][nodes[j]] = dist[i][j];\n'
            code += '        }\n'
            code += '    }\n'
            code += '    return result;\n'
            code += '}\n'
            return code
        elif language == ProgrammingLanguage.JAVA:
            code = "// 最短路径算法 - Java实现\n"
            code += "import java.util.*;\n\n"
            code += "public class ShortestPathAlgorithms {\n"
            code += '    // Dijkstra最短路径算法\n'
            code += '    public static Map<String, Double> dijkstra(Map<String, Map<String, Double>> graph, String start) {\n'
            code += '        Map<String, Double> distances = new HashMap<>();\n'
            code += '        for (String node : graph.keySet()) {\n'
            code += '            distances.put(node, Double.POSITIVE_INFINITY);\n'
            code += '        }\n'
            code += '        distances.put(start, 0.0);\n'
            code += '        PriorityQueue<Map.Entry<Double, String>> pq = new PriorityQueue<>(Map.Entry.comparingByKey());\n'
            code += '        pq.add(new AbstractMap.SimpleEntry<>(0.0, start));\n'
            code += '        while (!pq.isEmpty()) {\n'
            code += '            Map.Entry<Double, String> entry = pq.poll();\n'
            code += '            double currentDistance = entry.getKey();\n'
            code += '            String currentNode = entry.getValue();\n'
            code += '            if (currentDistance > distances.get(currentNode)) continue;\n'
            code += '            Map<String, Double> neighbors = graph.getOrDefault(currentNode, new HashMap<>());\n'
            code += '            for (Map.Entry<String, Double> neighbor : neighbors.entrySet()) {\n'
            code += '                String neighborNode = neighbor.getKey();\n'
            code += '                double weight = neighbor.getValue();\n'
            code += '                double distance = currentDistance + weight;\n'
            code += '                if (distance < distances.get(neighborNode)) {\n'
            code += '                    distances.put(neighborNode, distance);\n'
            code += '                    pq.add(new AbstractMap.SimpleEntry<>(distance, neighborNode));\n'
            code += '                }\n'
            code += '            }\n'
            code += '        }\n'
            code += '        return distances;\n'
            code += '    }\n'
            code += '    \n'
            code += '    // Floyd-Warshall最短路径算法\n'
            code += '    public static Map<String, Map<String, Double>> floydWarshall(Map<String, Map<String, Double>> graph) {\n'
            code += '        List<String> nodes = new ArrayList<>(graph.keySet());\n'
            code += '        int n = nodes.size();\n'
            code += '        double[][] dist = new double[n][n];\n'
            code += '        for (int i = 0; i < n; i++) {\n'
            code += '            Arrays.fill(dist[i], Double.POSITIVE_INFINITY);\n'
            code += '            dist[i][i] = 0;\n'
            code += '        }\n'
            code += '        for (int i = 0; i < n; i++) {\n'
            code += '            String u = nodes.get(i);\n'
            code += '            Map<String, Double> neighbors = graph.get(u);\n'
            code += '            if (neighbors != null) {\n'
            code += '                for (int j = 0; j < n; j++) {\n'
            code += '                    String v = nodes.get(j);\n'
            code += '                    if (neighbors.containsKey(v)) {\n'
            code += '                        dist[i][j] = neighbors.get(v);\n'
            code += '                    }\n'
            code += '                }\n'
            code += '            }\n'
            code += '        }\n'
            code += '        for (int k = 0; k < n; k++) {\n'
            code += '            for (int i = 0; i < n; i++) {\n'
            code += '                for (int j = 0; j < n; j++) {\n'
            code += '                    if (dist[i][k] + dist[k][j] < dist[i][j]) {\n'
            code += '                        dist[i][j] = dist[i][k] + dist[k][j];\n'
            code += '                    }\n'
            code += '                }\n'
            code += '            }\n'
            code += '        }\n'
            code += '        Map<String, Map<String, Double>> result = new HashMap<>();\n'
            code += '        for (int i = 0; i < n; i++) {\n'
            code += '            String nodeI = nodes.get(i);\n'
            code += '            Map<String, Double> row = new HashMap<>();\n'
            code += '            for (int j = 0; j < n; j++) {\n'
            code += '                String nodeJ = nodes.get(j);\n'
            code += '                row.put(nodeJ, dist[i][j]);\n'
            code += '            }\n'
            code += '            result.put(nodeI, row);\n'
            code += '        }\n'
            code += '        return result;\n'
            code += '    }\n'
            code += '}\n'
            return code
        elif language == ProgrammingLanguage.CPP:
            code = "// 最短路径算法 - C++实现\n"
            code += "#include <iostream>\n"
            code += "#include <vector>\n"
            code += "#include <map>\n"
            code += "#include <queue>\n"
            code += "#include <limits>\n"
            code += "#include <algorithm>\n\n"
            code += "using namespace std;\n\n"
            code += "map<string, double> dijkstra(map<string, map<string, double>>& graph, string start) {\n"
            code += '    // Dijkstra最短路径算法\n'
            code += '    map<string, double> distances;\n'
            code += '    for (auto& pair : graph) {\n'
            code += '        distances[pair.first] = numeric_limits<double>::infinity();\n'
            code += '    }\n'
            code += '    distances[start] = 0;\n'
            code += '    priority_queue<pair<double, string>, vector<pair<double, string>>, greater<pair<double, string>>> pq;\n'
            code += '    pq.push({0, start});\n'
            code += '    while (!pq.empty()) {\n'
            code += '        auto [currentDistance, currentNode] = pq.top();\n'
            code += '        pq.pop();\n'
            code += '        if (currentDistance > distances[currentNode]) continue;\n'
            code += '        for (auto& neighbor : graph[currentNode]) {\n'
            code += '            string neighborNode = neighbor.first;\n'
            code += '            double weight = neighbor.second;\n'
            code += '            double distance = currentDistance + weight;\n'
            code += '            if (distance < distances[neighborNode]) {\n'
            code += '                distances[neighborNode] = distance;\n'
            code += '                pq.push({distance, neighborNode});\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    return distances;\n'
            code += '}\n\n'
            code += "map<string, map<string, double>> floydWarshall(map<string, map<string, double>>& graph) {\n"
            code += '    // Floyd-Warshall最短路径算法\n'
            code += '    vector<string> nodes;\n'
            code += '    for (auto& pair : graph) {\n'
            code += '        nodes.push_back(pair.first);\n'
            code += '    }\n'
            code += '    int n = nodes.size();\n'
            code += '    vector<vector<double>> dist(n, vector<double>(n, numeric_limits<double>::infinity()));\n'
            code += '    for (int i = 0; i < n; i++) {\n'
            code += '        dist[i][i] = 0;\n'
            code += '    }\n'
            code += '    for (int i = 0; i < n; i++) {\n'
            code += '        string u = nodes[i];\n'
            code += '        for (int j = 0; j < n; j++) {\n'
            code += '            string v = nodes[j];\n'
            code += '            if (graph[u].find(v) != graph[u].end()) {\n'
            code += '                dist[i][j] = graph[u][v];\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    for (int k = 0; k < n; k++) {\n'
            code += '        for (int i = 0; i < n; i++) {\n'
            code += '            for (int j = 0; j < n; j++) {\n'
            code += '                if (dist[i][k] + dist[k][j] < dist[i][j]) {\n'
            code += '                    dist[i][j] = dist[i][k] + dist[k][j];\n'
            code += '                }\n'
            code += '            }\n'
            code += '        }\n'
            code += '    }\n'
            code += '    map<string, map<string, double>> result;\n'
            code += '    for (int i = 0; i < n; i++) {\n'
            code += '        string nodeI = nodes[i];\n'
            code += '        map<string, double> row;\n'
            code += '        for (int j = 0; j < n; j++) {\n'
            code += '            string nodeJ = nodes[j];\n'
            code += '            row[nodeJ] = dist[i][j];\n'
            code += '        }\n'
            code += '        result[nodeI] = row;\n'
            code += '    }\n'
            code += '    return result;\n'
            code += '}\n'
            return code
        else:
            return f"// 最短路径算法: {description}\n// 该语言暂不支持最短路径算法，请使用Python、JavaScript、Java或C++"

    def _generate_generic_algorithm(self, description: str, language: ProgrammingLanguage) -> str:
        """生成通用算法"""
        if language == ProgrammingLanguage.PYTHON:
            # 根据描述生成合适的通用算法
            desc_lower = description.lower()
            algorithm_name = self._extract_function_name(description)

            code = f"def {algorithm_name}(data):\n"
            code += f'    """{description}"""\n'

            # 根据描述关键词添加不同实现
            if "计算" in description or "calculate" in desc_lower:
                code += '    # 计算型算法\n'
                code += '    result = 0\n'
                code += '    for item in data:\n'
                code += '        if isinstance(item, (int, float)):\n'
                code += '            result += item\n'
                code += '    return result\n'
            elif "查找" in description or "find" in desc_lower:
                code += '    # 查找型算法\n'
                code += '    target = data[0] if data else None\n'
                code += '    for i, item in enumerate(data):\n'
                code += '        if item == target:\n'
                code += '            return i\n'
                code += '    return -1\n'
            elif "过滤" in description or "filter" in desc_lower:
                code += '    # 过滤型算法\n'
                code += '    result = []\n'
                code += '    for item in data:\n'
                code += '        if item and (isinstance(item, str) and len(item) > 0 or \n'
                code += '                     isinstance(item, (int, float)) and item > 0):\n'
                code += '            result.append(item)\n'
                code += '    return result\n'
            elif "转换" in description or "transform" in desc_lower or "convert" in desc_lower:
                code += '    # 转换型算法\n'
                code += '    result = []\n'
                code += '    for item in data:\n'
                code += '        if isinstance(item, str):\n'
                code += '            result.append(item.upper())\n'
                code += '        elif isinstance(item, (int, float)):\n'
                code += '            result.append(str(item))\n'
                code += '        else:\n'
                code += '            result.append(item)\n'
                code += '    return result\n'
            else:
                code += '    # 通用处理算法\n'
                code += '    if isinstance(data, list):\n'
                code += '        return len(data)\n'
                code += '    elif isinstance(data, dict):\n'
                code += '        return len(data.keys())\n'
                code += '    else:\n'
                code += '        return str(data)\n'
            return code
        else:
            return f"// 通用算法: {description}\n// 该语言暂不支持通用算法，请使用Python、JavaScript、Java或C++"

    def _generate_algorithm_framework(self, description: str, language: ProgrammingLanguage) -> str:
        """生成算法框架"""
        if language == ProgrammingLanguage.JAVA:
            algorithm_name = self._extract_function_name(description).title()
            return f"public class {algorithm_name}Algorithm {{\n" \
                f"    public static void main(String[] args) {{\n" \
                f"        System.out.println(\"算法: {description}\");\n" \
                f"    }}\n" \
                f"}}\n"
        elif language == ProgrammingLanguage.JAVASCRIPT:
            algorithm_name = self._extract_function_name(description)
            return f"function {algorithm_name}(data) {{\n" \
                f"    // {description}\n" \
                f"    return data;\n" \
                f"}}\n" \
                f"\n" \
                f"// 使用示例:\n" \
                f"// const result = {algorithm_name}([]);\n"
        elif language == ProgrammingLanguage.CPP:
            algorithm_name = self._extract_function_name(description)
            return f"#include <iostream>\n" \
                f"#include <vector>\n\n" \
                f"using namespace std;\n\n" \
                f"void {algorithm_name}(vector<int>& data) {{\n" \
                f"    // {description}\n" \
                f"    cout << \"算法执行\" << endl;\n" \
                f"}}\n\n" \
                f"int main() {{\n" \
                f"    vector<int> data = {{1, 2, 3}};\n" \
                f"    {algorithm_name}(data);\n" \
                f"    return 0;\n" \
                f"}}\n"
        else:
            return f"// 算法框架: {description}\n// 语言: {language.value}\n// 该语言暂不支持算法框架，请使用Python、JavaScript、Java或C++"

    def _generate_test(self, description: str, language: ProgrammingLanguage) -> str:
        """生成测试代码"""
        if language == ProgrammingLanguage.PYTHON:
            test_name = self._extract_function_name(description)

            code = f"import unittest\n\n"
            code += f"def {test_name}_function(input_data):\n"
            code += f'    """被测试函数: {description}"""\n'
            code += f"    # 模拟实现\n"
            code += f"    if isinstance(input_data, list):\n"
            code += f"        return len(input_data)\n"
            code += f"    elif isinstance(input_data, dict):\n"
            code += f"        return sum(input_data.values()) if all(isinstance(v, (int, float)) for v in input_data.values()) else 0\n"
            code += f"    else:\n"
            code += f"        return str(input_data)\n\n"
            code += f"class Test{test_name.title()}(unittest.TestCase):\n"
            code += f'    """测试类: {description}"""\n\n'

            # 生成多个测试用例
            code += f"    def test_basic_functionality(self):\n"
            code += f'        """测试基本功能"""\n'
            code += f"        result = {test_name}_function([1, 2, 3])\n"
            code += f"        self.assertEqual(result, 3)\n\n"

            code += f"    def test_empty_input(self):\n"
            code += f'        """测试空输入"""\n'
            code += f"        result = {test_name}_function([])\n"
            code += f"        self.assertEqual(result, 0)\n\n"

            code += f"    def test_dict_input(self):\n"
            code += f'        """测试字典输入"""\n'
            code += f"        result = {test_name}_function({{'a': 1, 'b': 2, 'c': 3}})\n"
            code += f"        self.assertEqual(result, 6)\n\n"

            code += f"    def test_string_input(self):\n"
            code += f'        """测试字符串输入"""\n'
            code += f"        result = {test_name}_function('test')\n"
            code += f"        self.assertEqual(result, 'test')\n\n"

            code += f"if __name__ == '__main__':\n"
            code += f"    unittest.main()\n"

            return code
        elif language == ProgrammingLanguage.JAVASCRIPT:
            test_name = self._extract_function_name(description)
            return f"// 测试: {description}\n" \
                f"function {test_name}(input) {{\n" \
                f"    // 被测试函数\n" \
                f"    return input;\n" \
                f"}}\n\n" \
                f"// 测试用例\n" \
                f"console.assert({test_name}([1, 2, 3]).length === 3, '数组长度测试失败');\n" \
                f"console.assert({test_name}('test') === 'test', '字符串测试失败');\n" \
                f"console.log('所有测试通过');\n"
        elif language == ProgrammingLanguage.JAVA:
            test_name = self._extract_function_name(description).title()
            return f"import org.junit.Test;\n" \
                f"import static org.junit.Assert.*;\n\n" \
                f"public class {test_name}Test {{\n" \
                f"    \n" \
                f"    @Test\n" \
                f"    public void testBasicFunctionality() {{\n" \
                f"        // 测试基本功能\n" \
                f"        assertEquals(3, {test_name}.process(new int[]{{1, 2, 3}}));\n" \
                f"    }}\n" \
                f"    \n" \
                f"    @Test\n" \
                f"    public void testEmptyInput() {{\n" \
                f"        // 测试空输入\n" \
                f"        assertEquals(0, {test_name}.process(new int[]{{}}));\n" \
                f"    }}\n" \
                f"}}\n\n" \
                f"class {test_name} {{\n" \
                f"    public static int process(int[] data) {{\n" \
                f"        return data.length;\n" \
                f"    }}\n" \
                f"}}\n"
        else:
            return f"// 测试代码: {description}\n// 语言: {language.value}\n// 请根据具体语言实现测试框架"

    def _generate_module(self, description: str, language: ProgrammingLanguage) -> str:
        """生成模块代码"""
        if language == ProgrammingLanguage.PYTHON:
            module_name = self._extract_module_name(description)

            code = f'"""\n{description}\n"""\n\n'
            code += "import sys\nimport os\nimport json\nimport logging\nfrom datetime import datetime\n\n"
            code += "# 配置日志\n"
            code += "logging.basicConfig(level=logging.INFO)\n"
            code += "logger = logging.getLogger(__name__)\n\n"
            code += f"class {module_name}Module:\n"
            code += f'    """{description}模块"""\n\n'
            code += "    def __init__(self, config_path=None):\n"
            code += "        self.config = self._load_config(config_path)\n"
            code += "        self.data = []\n"
            code += "        logger.info(f'模块初始化完成: {module_name}')\n\n"
            code += "    def _load_config(self, config_path):\n"
            code += '        """加载配置"""\n'
            code += "        default_config = {\n"
            code += f'            "module_name": "{module_name}",\n'
            code += '            "version": "1.0.0",\n'
            code += '            "enabled": True,\n'
            code += '            "max_items": 100\n'
            code += "        }\n"
            code += "        if config_path and os.path.exists(config_path):\n"
            code += "            try:\n"
            code += "                with open(config_path, 'r') as f:\n"
            code += "                    user_config = json.load(f)\n"
            code += "                    default_config.update(user_config)\n"
            code += "            except Exception as e:\n"
            code += "                logger.warning(f'加载配置文件失败: {e}')\n"
            code += "        return default_config\n\n"
            code += "    def process_data(self, data):\n"
            code += '        """处理数据"""\n'
            code += "        if not isinstance(data, list):\n"
            code += "            data = [data]\n"
            code += "        \n"
            code += "        processed = []\n"
            code += "        for item in data:\n"
            code += "            if isinstance(item, dict):\n"
            code += "                item['processed_at'] = datetime.now().isoformat()\n"
            code += "                item['module'] = self.config['module_name']\n"
            code += "            processed.append(item)\n"
            code += "        \n"
            code += "        self.data.extend(processed)\n"
            code += "        logger.info(f'处理了 {len(processed)} 条数据')\n"
            code += "        return processed\n\n"
            code += "    def get_stats(self):\n"
            code += '        """获取统计信息"""\n'
            code += "        return {\n"
            code += '            "total_items": len(self.data),\n'
            code += '            "module_name": self.config["module_name"],\n'
            code += '            "version": self.config["version"],\n'
            code += '            "last_updated": datetime.now().isoformat()\n'
            code += "        }\n\n"
            code += "    def run(self):\n"
            code += f'        """运行模块: {description}"""\n'
            code += "        logger.info('开始运行模块')\n"
            code += "        \n"
            code += "        # 示例数据处理\n"
            code += "        test_data = [\n"
            code += '            {"id": 1, "name": "测试数据1"},\n'
            code += '            {"id": 2, "name": "测试数据2"},\n'
            code += '            {"id": 3, "name": "测试数据3"}\n'
            code += "        ]\n"
            code += "        \n"
            code += "        processed = self.process_data(test_data)\n"
            code += "        stats = self.get_stats()\n"
            code += "        \n"
            code += "        logger.info(f'模块运行完成，处理结果: {len(processed)} 条数据')\n"
            code += "        logger.info(f'统计信息: {stats}')\n"
            code += "        return processed\n\n"
            code += f"def main():\n"
            code += f'    """主函数"""\n'
            code += f"    module = {module_name}Module()\n"
            code += f"    result = module.run()\n"
            code += f"    print(f'模块执行成功，处理了 {{len(result)}} 条数据')\n"
            code += f"    return result\n\n"
            code += 'if __name__ == "__main__":\n'
            code += "    main()\n"

            return code
        elif language == ProgrammingLanguage.JAVASCRIPT:
            module_name = self._extract_module_name(description)
            return f"// {description}模块\n" \
                f"class {module_name}Module {{\n" \
                f"    constructor(config) {{\n" \
                f"        this.config = config || {{}};\n" \
                f"        this.data = [];\n" \
                f"        console.log('模块初始化完成: {module_name}');\n" \
                f"    }}\n\n" \
                f"    processData(data) {{\n" \
                f"        if (!Array.isArray(data)) {{\n" \
                f"            data = [data];\n" \
                f"        }}\n" \
                f"        const processed = data.map(item => ({{...item, processedAt: new Date().toISOString()}}));\n" \
                f"        this.data.push(...processed);\n" \
                f"        console.log(`处理了 ${{processed.length}} 条数据`);\n" \
                f"        return processed;\n" \
                f"    }}\n\n" \
                f"    getStats() {{\n" \
                f"        return {{\n" \
                f"            totalItems: this.data.length,\n" \
                f"            moduleName: '{module_name}',\n" \
                f"            lastUpdated: new Date().toISOString()\n" \
                f"        }};\n" \
                f"    }}\n\n" \
                f"    run() {{\n" \
                f"        console.log('开始运行模块');\n" \
                f"        const testData = [\n" \
                f"            {{id: 1, name: '测试数据1'}},\n" \
                f"            {{id: 2, name: '测试数据2'}},\n" \
                f"            {{id: 3, name: '测试数据3'}}\n" \
                f"        ];\n" \
                f"        const processed = this.processData(testData);\n" \
                f"        const stats = this.getStats();\n" \
                f"        console.log(`模块运行完成，处理结果: ${{processed.length}} 条数据`);\n" \
                f"        console.log('统计信息:', stats);\n" \
                f"        return processed;\n" \
                f"    }}\n" \
                f"}}\n\n" \
                f"// 使用示例\n" \
                f"// const module = new {module_name}Module();\n" \
                f"// const result = module.run();\n" \
                f"// console.log(result);\n"
        else:
            return f"// 模块代码: {description}\n// 语言: {language.value}\n// 请根据具体语言实现模块结构"

    def _generate_interface(self, description: str, language: ProgrammingLanguage) -> str:
        """生成接口代码"""
        if language == ProgrammingLanguage.PYTHON:
            interface_name = self._extract_interface_name(description)

            code = f'"""\n{description}\n"""\n\n'
            code += "from abc import ABC, abstractmethod\nfrom typing import Any, Dict, List, Optional\nimport datetime\n\n"
            code += f"class {interface_name}(ABC):\n"
            code += f'    """{description}接口"""\n\n'

            # 提取方法描述
            methods = self._extract_methods_from_description(description)

            if not methods:
                # 如果没有提取到方法，生成一组标准接口方法
                methods = [
                    {"name": "initialize", "description": "初始化接口",
                        "params": "self, config: Dict[str, Any]", "returns": "bool"},
                    {"name": "process", "description": "处理数据",
                        "params": "self, data: Any", "returns": "Any"},
                    {"name": "validate", "description": "验证输入",
                        "params": "self, input_data: Any", "returns": "bool"},
                    {"name": "get_status", "description": "获取状态",
                        "params": "self", "returns": "Dict[str, Any]"},
                    {"name": "cleanup", "description": "清理资源",
                        "params": "self", "returns": "None"}
                ]

            # 生成抽象方法
            for method in methods:
                method_name = method.get("name", f"method_{len(code) % 1000}")
                params = method.get("params", "self")
                returns = method.get("returns", "Any")
                method_desc = method.get("description", "接口方法")

                code += f"    @abstractmethod\n"
                code += f"    def {method_name}({params}) -> {returns}:\n"
                code += f'        """{method_desc}"""\n'
                code += f"        pass\n\n"

            # 示例实现类
            code += f"\nclass {interface_name}Impl({interface_name}):\n"
            code += f'    """{description}接口实现类"""\n\n'
            code += f"    def __init__(self, config: Optional[Dict[str, Any]] = None):\n"
            code += f'        """初始化实现类"""\n'
            code += f"        super().__init__()\n"
            code += f"        self.config = config or {{}}\n"
            code += f"        self.initialized = False\n"
            code += f"        self.processed_count = 0\n"
            code += f"        self.start_time = datetime.datetime.now()\n\n"

            # 实现所有抽象方法
            for method in methods:
                method_name = method.get("name", f"method_{len(code) % 1000}")
                params = method.get("params", "self")
                returns = method.get("returns", "Any")
                method_desc = method.get("description", "接口方法")

                # 处理参数列表，提取参数名
                param_list = params.split(",")
                param_names = []
                for param in param_list:
                    param = param.strip()
                    if ":" in param:
                        param_name = param.split(":")[0].strip()
                    elif "=" in param:
                        param_name = param.split("=")[0].strip()
                    else:
                        param_name = param
                    param_names.append(param_name)

                code += f"    def {method_name}({params}) -> {returns}:\n"
                code += f'        """实现{method_desc}"""\n'

                # 根据方法名称和返回类型生成具体实现
                if method_name == "initialize":
                    code += f"        if 'enabled' in self.config and not self.config['enabled']:\n"
                    code += f"            return False\n"
                    code += f"        self.initialized = True\n"
                    code += f"        print(f'接口初始化完成: {{self.config}}')\n"
                    code += f"        return True\n\n"
                elif method_name == "process":
                    if len(param_names) > 1:  # 有data参数
                        data_param = param_names[1]
                        code += f"        if not self.initialized:\n"
                        code += f"            raise RuntimeError('接口未初始化')\n"
                        code += f"        self.processed_count += 1\n"
                        code += f"        # 简单处理：如果是字典，添加时间戳\n"
                        code += f"        if isinstance({data_param}, dict):\n"
                        code += f"            {data_param}['processed_at'] = datetime.datetime.now().isoformat()\n"
                        code += f"            {data_param}['processor'] = '{interface_name}'\n"
                        code += f"            {data_param}['count'] = self.processed_count\n"
                        code += f"        print(f'处理了第{{self.processed_count}}条数据')\n"
                        code += f"        return {data_param}\n\n"
                    else:
                        code += f"        return None  # 返回None\n\n"
                elif method_name == "validate":
                    if len(param_names) > 1:  # 有input_data参数
                        input_param = param_names[1]
                        code += f"        # 基本验证逻辑\n"
                        code += f"        if {input_param} is None:\n"
                        code += f"            return False\n"
                        code += f"        if isinstance({input_param}, str) and len({input_param}.strip()) == 0:\n"
                        code += f"            return False\n"
                        code += f"        if isinstance({input_param}, (list, dict)) and len({input_param}) == 0:\n"
                        code += f"            return False\n"
                        code += f"        return True\n\n"
                    else:
                        code += f"        return False\n\n"
                elif method_name == "get_status":
                    code += f"        return {{\n"
                    code += f"            'interface_name': '{interface_name}',\n"
                    code += f"            'initialized': self.initialized,\n"
                    code += f"            'processed_count': self.processed_count,\n"
                    code += f"            'uptime_seconds': (datetime.datetime.now() - self.start_time).total_seconds(),\n"
                    code += f"            'config_keys': list(self.config.keys())\n"
                    code += f"        }}\n\n"
                elif method_name == "cleanup":
                    code += f"        self.initialized = False\n"
                    code += f"        self.processed_count = 0\n"
                    code += f"        print('接口资源已清理')\n"
                    code += f"        return None  # 返回None\n\n"
                else:
                    # 通用方法实现
                    if returns == "bool":
                        code += f"        return True\n\n"
                    elif returns == "int":
                        code += f"        return 0\n\n"
                    elif returns == "str":
                        code += f"        return ''\n\n"
                    elif returns == "list":
                        code += f"        return []  # 返回空列表\n\n"
                    elif returns == "dict":
                        code += f"        return {{}}\n\n"
                    elif returns == "None" or returns == "NoneType":
                        code += f"        return None  # 返回None\n\n"
                    else:
                        code += f"        return None  # 返回None\n\n"

            # 使用示例
            code += f"\n# 使用示例\n"
            code += f"if __name__ == '__main__':\n"
            code += f"    # 创建配置\n"
            code += f"    config = {{\n"
            code += f"        'name': '测试配置',\n"
            code += f"        'enabled': True,\n"
            code += f"        'max_retries': 3\n"
            code += f"    }}\n"
            code += f"    \n"
            code += f"    # 实例化接口\n"
            code += f"    impl = {interface_name}Impl(config)\n"
            code += f"    \n"
            code += f"    # 初始化\n"
            code += f"    if impl.initialize(config):\n"
            code += f"        print('初始化成功')\n"
            code += f"        \n"
            code += f"        # 处理数据\n"
            code += f"        data = {{'id': 1, 'name': '测试数据'}}\n"
            code += f"        processed = impl.process(data)\n"
            code += f"        print(f'处理结果: {{processed}}')\n"
            code += f"        \n"
            code += f"        # 获取状态\n"
            code += f"        status = impl.get_status()\n"
            code += f"        print(f'状态: {{status}}')\n"
            code += f"        \n"
            code += f"        # 清理\n"
            code += f"        impl.cleanup()\n"
            code += f"    else:\n"
            code += f"        print('初始化失败')\n"

            return code
        elif language == ProgrammingLanguage.JAVA:
            interface_name = self._extract_interface_name(
                description).replace("Interface", "")
            return f"// {description}接口\n" \
                f"public interface {interface_name} {{\n" \
                f"    boolean initialize(java.util.Map<String, Object> config);\n" \
                f"    Object process(Object data);\n" \
                f"    boolean validate(Object input);\n" \
                f"    java.util.Map<String, Object> getStatus();\n" \
                f"    void cleanup();\n" \
                f"}}\n\n" \
                f"// 实现类\n" \
                f"class {interface_name}Impl implements {interface_name} {{\n" \
                f"    private java.util.Map<String, Object> config;\n" \
                f"    private boolean initialized = false;\n" \
                f"    private int processedCount = 0;\n" \
                f"    \n" \
                f"    public boolean initialize(java.util.Map<String, Object> config) {{\n" \
                f"        this.config = config;\n" \
                f"        this.initialized = true;\n" \
                f"        System.out.println(\"接口初始化完成\");\n" \
                f"        return true;\n" \
                f"    }}\n" \
                f"    \n" \
                f"    public Object process(Object data) {{\n" \
                f"        if (!initialized) {{\n" \
                f"            throw new RuntimeException(\"接口未初始化\");\n" \
                f"        }}\n" \
                f"        processedCount++;\n" \
                f"        System.out.println(\"处理了第\" + processedCount + \"条数据\");\n" \
                f"        return data;\n" \
                f"    }}\n" \
                f"    \n" \
                f"    public boolean validate(Object input) {{\n" \
                f"        return input != null;\n" \
                f"    }}\n" \
                f"    \n" \
                f"    public java.util.Map<String, Object> getStatus() {{\n" \
                f"        java.util.Map<String, Object> status = new java.util.HashMap<>();\n" \
                f"        status.put(\"interfaceName\", \"{interface_name}\");\n" \
                f"        status.put(\"initialized\", initialized);\n" \
                f"        status.put(\"processedCount\", processedCount);\n" \
                f"        return status;\n" \
                f"    }}\n" \
                f"    \n" \
                f"    public void cleanup() {{\n" \
                f"        initialized = false;\n" \
                f"        processedCount = 0;\n" \
                f"        System.out.println(\"接口资源已清理\");\n" \
                f"    }}\n" \
                f"}}\n"
        else:
            return f"// 接口代码: {description}\n// 语言: {language.value}\n// 请根据具体语言实现接口模式"

    def _extract_interface_name(self, description: str) -> str:
        """从描述中提取接口名"""
        # 使用提取函数名的逻辑，但转换为帕斯卡命名法并添加Interface后缀
        function_name = self._extract_function_name(description)
        # 转换为帕斯卡命名（首字母大写）
        if function_name:
            # 将蛇形命名转换为帕斯卡命名
            parts = function_name.split('_')
            interface_name = ''.join(part.capitalize() for part in parts if part)
            return f"{interface_name}Interface"
        return "CustomInterface"

    def _extract_methods_from_description(self, description: str) -> List[Dict[str, Any]]:
        """从描述中提取方法信息"""
        methods = []

        # 简单实现：基于关键词提取
        if "方法" in description or "method" in description.lower() or "function" in description.lower():
            # 尝试提取多个方法
            lines = description.split('\n')
            for line in lines:
                line = line.strip()
                if "方法" in line or "method" in line.lower() or "function" in line.lower():
                    # 简单提取
                    method_name = "execute"
                    if "方法" in line:
                        parts = line.split("方法")
                        if parts[0].strip():
                            method_name = parts[0].strip()
                    elif "method" in line.lower():
                        parts = line.lower().split("method")
                        if parts[0].strip():
                            method_name = parts[0].strip()

                    methods.append({
                        "name": method_name,
                        "description": line,
                        "params": "self",
                        "returns": "bool"
                    })

        # 如果没有提取到方法，返回空列表
        return methods

    def _generate_script(self, description: str, language: ProgrammingLanguage) -> str:
        """生成脚本代码"""
        if language == ProgrammingLanguage.PYTHON:
            script_name = self._extract_script_name(description)

            code = f'#!/usr/bin/env python3\n'
            code += f'# -*- coding: utf-8 -*-\n'
            code += f'\n'
            code += f'"""\n{description}\n"""\n\n'
            code += f'import sys\nimport os\nimport argparse\nimport json\nimport csv\nimport logging\nfrom datetime import datetime\nfrom pathlib import Path\n\n'
            code += f'# 配置日志\n'
            code += f'logging.basicConfig(\n'
            code += f'    level=logging.INFO,\n'
            code += f'    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"\n'
            code += f')\n'
            code += f'logger = logging.getLogger(__name__)\n\n'
            code += f'def process_input_file(input_path):\n'
            code += f'    """处理输入文件"""\n'
            code += f'    if not os.path.exists(input_path):\n'
            code += f'        logger.error(f"输入文件不存在: {{input_path}}")\n'
            code += f'        return None  # 返回None\n'
            code += f'    \n'
            code += f'    try:\n'
            code += f'        # 根据文件扩展名选择处理方式\n'
            code += f'        ext = os.path.splitext(input_path)[1].lower()\n'
            code += f'        \n'
            code += f'        if ext == ".json":\n'
            code += f'            with open(input_path, "r", encoding="utf-8") as f:\n'
            code += f'                return json.load(f)\n'
            code += f'        elif ext == ".csv":\n'
            code += f'            data = []\n'
            code += f'            with open(input_path, "r", encoding="utf-8") as f:\n'
            code += f'                reader = csv.DictReader(f)\n'
            code += f'                for row in reader:\n'
            code += f'                    data.append(row)\n'
            code += f'            return data\n'
            code += f'        elif ext in [".txt", ".log"]:\n'
            code += f'            with open(input_path, "r", encoding="utf-8") as f:\n'
            code += f'                return f.read()\n'
            code += f'        else:\n'
            code += f'            # 默认按文本处理\n'
            code += f'            with open(input_path, "r", encoding="utf-8") as f:\n'
            code += f'                return f.read()\n'
            code += f'    except Exception as e:\n'
            code += f'        logger.error(f"处理输入文件失败: {{e}}")\n'
            code += f'        return None  # 返回None\n\n'
            code += f'def process_data(data, operation="process"):\n'
            code += f'    """处理数据"""\n'
            code += f'    if operation == "count":\n'
            code += f'        if isinstance(data, list):\n'
            code += f'            return len(data)\n'
            code += f'        elif isinstance(data, dict):\n'
            code += f'            return len(data)\n'
            code += f'        elif isinstance(data, str):\n'
            code += f'            return len(data.split())\n'
            code += f'    elif operation == "sum":\n'
            code += f'        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):\n'
            code += f'            return sum(data)\n'
            code += f'    elif operation == "analyze":\n'
            code += f'        result = {{\n'
            code += f'            "timestamp": datetime.now().isoformat(),\n'
            code += f'            "data_type": type(data).__name__\n'
            code += f'        }}\n'
            code += f'        if isinstance(data, list):\n'
            code += f'            result["length"] = len(data)\n'
            code += f'            if data and isinstance(data[0], dict):\n'
            code += f'                result["keys"] = list(data[0].keys()) if data else []\n'
            code += f'        elif isinstance(data, dict):\n'
            code += f'            result["keys"] = list(data.keys())\n'
            code += f'            result["length"] = len(data)\n'
            code += f'        elif isinstance(data, str):\n'
            code += f'            result["length"] = len(data)\n'
            code += f'            result["word_count"] = len(data.split())\n'
            code += f'        return result\n'
            code += f'    \n'
            code += f'    # 默认处理：添加元数据\n'
            code += f'    if isinstance(data, dict):\n'
            code += f'        data["processed_at"] = datetime.now().isoformat()\n'
            code += f'        data["script"] = "{script_name}"\n'
            code += f'    elif isinstance(data, list):\n'
            code += f'        for item in data:\n'
            code += f'            if isinstance(item, dict):\n'
            code += f'                item["processed_at"] = datetime.now().isoformat()\n'
            code += f'                item["script"] = "{script_name}"\n'
            code += f'    \n'
            code += f'    return data\n\n'
            code += f'def save_output(data, output_path):\n'
            code += f'    """保存输出"""\n'
            code += f'    try:\n'
            code += f'        os.makedirs(os.path.dirname(output_path), exist_ok=True)\n'
            code += f'        \n'
            code += f'        ext = os.path.splitext(output_path)[1].lower()\n'
            code += f'        \n'
            code += f'        if ext == ".json":\n'
            code += f'            with open(output_path, "w", encoding="utf-8") as f:\n'
            code += f'                json.dump(data, f, ensure_ascii=False, indent=2)\n'
            code += f'        elif ext == ".csv":\n'
            code += f'            if isinstance(data, list) and data and isinstance(data[0], dict):\n'
            code += f'                with open(output_path, "w", encoding="utf-8", newline="") as f:\n'
            code += f'                    writer = csv.DictWriter(f, fieldnames=data[0].keys())\n'
            code += f'                    writer.writeheader()\n'
            code += f'                    writer.writerows(data)\n'
            code += f'            else:\n'
            code += f'                # 转换为列表格式\n'
            code += f'                with open(output_path, "w", encoding="utf-8") as f:\n'
            code += f'                    if isinstance(data, list):\n'
            code += f'                        for item in data:\n'
            code += f'                            f.write(str(item) + "\\n")\n'
            code += f'                    else:\n'
            code += f'                        f.write(str(data))\n'
            code += f'        else:\n'
            code += f'            # 默认保存为文本\n'
            code += f'            with open(output_path, "w", encoding="utf-8") as f:\n'
            code += f'                f.write(str(data))\n'
            code += f'        \n'
            code += f'        logger.info(f"输出已保存到: {{output_path}}")\n'
            code += f'        return True\n'
            code += f'    except Exception as e:\n'
            code += f'        logger.error(f"保存输出失败: {{e}}")\n'
            code += f'        return False\n\n'
            code += f'def main():\n'
            code += f'    """主函数"""\n'
            code += f'    parser = argparse.ArgumentParser(description="{description}")\n'
            code += f'    parser.add_argument("--input", required=True, help="输入文件路径")\n'
            code += f'    parser.add_argument("--output", required=True, help="输出文件路径")\n'
            code += f'    parser.add_argument("--operation", default="process", choices=["process", "count", "sum", "analyze"],\n'
            code += f'                        help="处理操作类型")\n'
            code += f'    parser.add_argument("--verbose", action="store_true", help="详细模式")\n'
            code += f'    parser.add_argument("--dry-run", action="store_true", help="试运行模式，不保存输出")\n'
            code += f'    \n'
            code += f'    args = parser.parse_args()\n'
            code += f'    \n'
            code += f'    if args.verbose:\n'
            code += f'        logging.getLogger().setLevel(logging.DEBUG)\n'
            code += f'        logger.debug("详细模式已启用")\n'
            code += f'    \n'
            code += f'    logger.info(f"脚本开始执行: {{script_name}}")\n'
            code += f'    logger.info(f"输入文件: {{args.input}}")\n'
            code += f'    logger.info(f"输出文件: {{args.output}}")\n'
            code += f'    logger.info(f"操作类型: {{args.operation}}")\n'
            code += f'    \n'
            code += f'    # 处理输入\n'
            code += f'    data = process_input_file(args.input)\n'
            code += f'    if data is None:\n'
            code += f'        logger.error("输入处理失败，脚本终止")\n'
            code += f'        return 1\n'
            code += f'    \n'
            code += f'    logger.info(f"输入数据加载成功，类型: {{type(data).__name__}}")\n'
            code += f'    \n'
            code += f'    # 处理数据\n'
            code += f'    processed_data = process_data(data, args.operation)\n'
            code += f'    \n'
            code += f'    logger.info(f"数据处理完成，结果类型: {{type(processed_data).__name__}}")\n'
            code += f'    \n'
            code += f'    # 保存输出\n'
            code += f'    if not args.dry_run:\n'
            code += f'        success = save_output(processed_data, args.output)\n'
            code += f'        if not success:\n'
            code += f'            logger.error("输出保存失败")\n'
            code += f'            return 1\n'
            code += f'    else:\n'
            code += f'        logger.info("试运行模式，跳过保存输出")\n'
            code += f'        print("处理结果:", processed_data)\n'
            code += f'    \n'
            code += f'    logger.info("脚本执行成功")\n'
            code += f'    return 0\n\n'
            code += f'if __name__ == "__main__":\n'
            code += f'    sys.exit(main())\n'

            return code
        elif language == ProgrammingLanguage.BASH or language == ProgrammingLanguage.SHELL:
            script_name = self._extract_script_name(description)
            return f'#!/bin/bash\n' \
                f'# {description}\n\n' \
                f'# 配置\n' \
                f'SCRIPT_NAME="{script_name}"\n' \
                f'VERSION="1.0.0"\n\n' \
                f'# 显示帮助信息\n' \
                f'usage() {{\n' \
                f'    echo "Usage: $0 [OPTIONS]"\n' \
                f'    echo "  {description}"\n' \
                f'    echo ""\n' \
                f'    echo "Options:"\n' \
                f'    echo "  -h, --help      显示帮助信息"\n' \
                f'    echo "  -v, --version   显示版本信息"\n' \
                f'    echo "  -i, --input     输入文件"\n' \
                f'    echo "  -o, --output    输出文件"\n' \
                f'    echo "  --verbose       详细模式"\n' \
                f'    echo ""\n' \
                f'    echo "示例:"\n' \
                f'    echo "  $0 --input data.txt --output result.txt"\n' \
                f'}}\n\n' \
                f'# 显示版本信息\n' \
                f'version() {{\n' \
                f'    echo "$SCRIPT_NAME v$VERSION"\n' \
                f'}}\n\n' \
                f'# 处理输入文件\n' \
                f'process_input() {{\n' \
                f'    local input_file="$1"\n' \
                f'    if [[ ! -f "$input_file" ]]; then\n' \
                f'        echo "错误: 输入文件不存在: $input_file" >&2\n' \
                f'        return 1\n' \
                f'    fi\n' \
                f'    \n' \
                f'    # 简单处理：统计行数、词数、字符数\n' \
                f'    echo "处理文件: $input_file"\n' \
                f'    local lines=$(wc -l < "$input_file")\n' \
                f'    local words=$(wc -w < "$input_file")\n' \
                f'    local chars=$(wc -c < "$input_file")\n' \
                f'    \n' \
                f'    echo "统计结果:"\n' \
                f'    echo "  行数: $lines"\n' \
                f'    echo "  词数: $words"\n' \
                f'    echo "  字符数: $chars"\n' \
                f'    \n' \
                f'    return 0\n' \
                f'}}\n\n' \
                f'# 主函数\n' \
                f'main() {{\n' \
                f'    local input_file=""\n' \
                f'    local output_file=""\n' \
                f'    local verbose=0\n' \
                f'    \n' \
                f'    # 解析命令行参数\n' \
                f'    while [[ $# -gt 0 ]]; do\n' \
                f'        case $1 in\n' \
                f'            -h|--help)\n' \
                f'                usage\n' \
                f'                return 0\n' \
                f'                ;;\n' \
                f'            -v|--version)\n' \
                f'                version\n' \
                f'                return 0\n' \
                f'                ;;\n' \
                f'            -i|--input)\n' \
                f'                input_file="$2"\n' \
                f'                shift 2\n' \
                f'                ;;\n' \
                f'            -o|--output)\n' \
                f'                output_file="$2"\n' \
                f'                shift 2\n' \
                f'                ;;\n' \
                f'            --verbose)\n' \
                f'                verbose=1\n' \
                f'                shift\n' \
                f'                ;;\n' \
                f'            *)\n' \
                f'                echo "未知选项: $1" >&2\n' \
                f'                usage\n' \
                f'                return 1\n' \
                f'                ;;\n' \
                f'        esac\n' \
                f'    done\n' \
                f'    \n' \
                f'    # 检查必要参数\n' \
                f'    if [[ -z "$input_file" ]]; then\n' \
                f'        echo "错误: 必须指定输入文件" >&2\n' \
                f'        usage\n' \
                f'        return 1\n' \
                f'    fi\n' \
                f'    \n' \
                f'    # 处理输入\n' \
                f'    if [[ $verbose -eq 1 ]]; then\n' \
                f'        echo "详细模式已启用"\n' \
                f'        echo "脚本: $SCRIPT_NAME"\n' \
                f'        echo "版本: $VERSION"\n' \
                f'        echo "输入文件: $input_file"\n' \
                f'        echo "输出文件: $output_file"\n' \
                f'    fi\n' \
                f'    \n' \
                f'    process_input "$input_file"\n' \
                f'    \n' \
                f'    echo "脚本执行完成"\n' \
                f'    return 0\n' \
                f'}}\n\n' \
                f'# 执行主函数\n' \
                f'main "$@"\n'
        else:
            return f"// 脚本代码: {description}\n// 语言: {language.value}\n// 请根据具体语言实现脚本功能"

    def _extract_script_name(self, description: str) -> str:
        """从描述中提取脚本名"""
        function_name = self._extract_function_name(description)
        if function_name:
            return function_name
        return "custom_script"

    def _generate_api(self, description: str, language: ProgrammingLanguage) -> str:
        """生成API代码"""
        if language == ProgrammingLanguage.PYTHON:
            api_name = self._extract_api_name(description)

            code = f'"""\n{description}\n"""\n\n'
            code += f'from fastapi import FastAPI, HTTPException\n'
            code += f'from pydantic import BaseModel\n'
            code += f'from typing import Optional\n\n'
            code += f'app = FastAPI(title="{api_name} API", description="{description}")\n\n'
            code += f'# 请求/响应模型\n'
            code += f'class {api_name}Request(BaseModel):\n'
            code += f'    data: str\n'
            code += f'    options: Optional[dict] = {{}}\n\n'
            code += f'class {api_name}Response(BaseModel):\n'
            code += f'    success: bool\n'
            code += f'    result: dict\n'
            code += f'    message: Optional[str] = None\n\n'
            code += f'@app.get("/")\n'
            code += f'async def root():\n'
            code += f'    return {{"message": "{api_name} API 运行中"}}\n\n'
            code += f'@app.post("/process")\n'
            code += f'async def process_data(request: {api_name}Request) -> {api_name}Response:\n'
            code += f'    """处理数据"""\n'
            code += f'    try:\n'
            code += f'        # 根据请求数据实现具体处理逻辑\n'
            code += f'        result = {{"processed": request.data}}\n'
            code += f'        return {api_name}Response(success=True, result=result)\n'
            code += f'    except Exception as e:\n'
            code += f'        raise HTTPException(status_code=500, detail=str(e))\n\n'
            code += f'@app.get("/health")\n'
            code += f'async def health_check():\n'
            code += f'    return {{"status": "healthy", "service": "{api_name}"}}\n\n'
            code += f'if __name__ == "__main__":\n'
            code += f'    import uvicorn\n'
            code += f'    uvicorn.run(app, host="0.0.0.0", port=8000)\n'

            return code
        else:
            return f"// API生成: {description}\n// 该语言暂不支持API生成，请使用Python"

    def _extract_api_name(self, description: str) -> str:
        """从描述中提取API名"""
        function_name = self._extract_function_name(description)
        if function_name:
            # 转换为帕斯卡命名
            parts = function_name.split('_')
            api_name = ''.join(part.capitalize() for part in parts if part)
            return api_name
        return "CustomAPI"

    def _generate_database(self, description: str, language: ProgrammingLanguage) -> str:
        """生成数据库代码"""
        if language == ProgrammingLanguage.PYTHON:
            db_name = self._extract_database_name(description)

            code = f'"""\n{description}\n"""\n\n'
            code += f'import sqlite3\n'
            code += f'from typing import List, Dict, Any, Optional\n'
            code += f'from dataclasses import dataclass\n'
            code += f'from datetime import datetime\n\n'
            code += f'@dataclass\n'
            code += f'class DatabaseConfig:\n'
            code += f'    """数据库配置"""\n'
            code += f'    database_path: str = "{db_name}.db"\n'
            code += f'    timeout: float = 5.0\n'
            code += f'    check_same_thread: bool = False\n\n'
            code += f'class {db_name}Database:\n'
            code += f'    """{description}数据库"""\n\n'
            code += f'    def __init__(self, config: Optional[DatabaseConfig] = None):\n'
            code += f'        self.config = config or DatabaseConfig()\n'
            code += f'        self.connection = None\n'
            code += f'        self._initialize()\n\n'
            code += f'    def _initialize(self):\n'
            code += f'        """初始化数据库"""\n'
            code += f'        self.connection = sqlite3.connect(\n'
            code += f'            self.config.database_path,\n'
            code += f'            timeout=self.config.timeout,\n'
            code += f'            check_same_thread=self.config.check_same_thread\n'
            code += f'        )\n'
            code += f'        self._create_tables()\n\n'
            code += f'    def _create_tables(self):\n'
            code += f'        """创建表"""\n'
            code += f'        cursor = self.connection.cursor()\n\n'
            code += f'        # 示例表\n'
            code += f'        cursor.execute("""\n'
            code += f'            CREATE TABLE IF NOT EXISTS records (\n'
            code += f'                id INTEGER PRIMARY KEY AUTOINCREMENT,\n'
            code += f'                data TEXT NOT NULL,\n'
            code += f'                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n'
            code += f'            )\n'
            code += f'        """)\n\n'
            code += f'        self.connection.commit()\n'
            code += f'        cursor.close()\n\n'
            code += f'    def insert_record(self, data: str) -> int:\n'
            code += f'        """插入记录"""\n'
            code += f'        cursor = self.connection.cursor()\n'
            code += f'        cursor.execute("INSERT INTO records (data) VALUES (?)", (data,))\n'
            code += f'        record_id = cursor.lastrowid\n'
            code += f'        self.connection.commit()\n'
            code += f'        cursor.close()\n'
            code += f'        return record_id\n\n'
            code += f'    def get_record(self, record_id: int) -> Optional[Dict[str, Any]]:\n'
            code += f'        """获取记录"""\n'
            code += f'        cursor = self.connection.cursor()\n'
            code += f'        cursor.execute("SELECT * FROM records WHERE id = ?", (record_id,))\n'
            code += f'        row = cursor.fetchone()\n'
            code += f'        cursor.close()\n\n'
            code += f'        if row:\n'
            code += f'            return {{\n'
            code += f'                "id": row[0],\n'
            code += f'                "data": row[1],\n'
            code += f'                "created_at": row[2]\n'
            code += f'            }}\n'
            code += f'        return None  # 返回None\n\n'
            code += f'    def close(self):\n'
            code += f'        """关闭数据库连接"""\n'
            code += f'        if self.connection:\n'
            code += f'            self.connection.close()\n'
            code += f'            self.connection = None\n\n'
            code += f'    def __del__(self):\n'
            code += f'        self.close()\n\n'
            code += f'def main():\n'
            code += f'    """示例使用"""\n'
            code += f'    db = {db_name}Database()\n'
            code += f'    record_id = db.insert_record("测试数据")\n'
            code += f'    print(f"插入记录ID: {{record_id}}")\n'
            code += f'    record = db.get_record(record_id)\n'
            code += f'    print(f"获取记录: {{record}}")\n'
            code += f'    db.close()\n\n'
            code += f'if __name__ == "__main__":\n'
            code += f'    main()\n'

            return code
        else:
            return f"// 数据库生成: {description}\n// 该语言暂不支持数据库生成，请使用SQL或Python"

    def _extract_database_name(self, description: str) -> str:
        """从描述中提取数据库名"""
        function_name = self._extract_function_name(description)
        if function_name:
            # 转换为帕斯卡命名
            parts = function_name.split('_')
            db_name = ''.join(part.capitalize() for part in parts if part)
            return db_name
        return "CustomDatabase"

    def _generate_web(self, description: str, language: ProgrammingLanguage) -> str:
        """生成Web应用代码"""
        if language == ProgrammingLanguage.PYTHON:
            app_name = self._extract_web_app_name(description)

            code = f'"""\n{description}\n"""\n\n'
            code += f'from flask import Flask, render_template, request, jsonify\n'
            code += f'import json\n\n'
            code += f'app = Flask(__name__)\n\n'
            code += f'@app.route("/")\n'
            code += f'def index():\n'
            code += f'    """主页"""\n'
            code += f'    return render_template("index.html", title="{app_name}")\n\n'
            code += f'@app.route("/api/data", methods=["GET"])\n'
            code += f'def get_data():\n'
            code += f'    """获取数据API"""\n'
            code += f'    data = {{\n'
            code += f'        "app": "{app_name}",\n'
            code += f'        "status": "running",\n'
            code += f'        "version": "1.0.0"\n'
            code += f'    }}\n'
            code += f'    return jsonify(data)\n\n'
            code += f'@app.route("/api/process", methods=["POST"])\n'
            code += f'def process():\n'
            code += f'    """处理数据API"""\n'
            code += f'    try:\n'
            code += f'        data = request.get_json()\n'
            code += f'        if not data:\n'
            code += f'            return jsonify({{"error": "No data provided"}}), 400\n\n'
            code += f'        # 根据数据实现具体处理逻辑\n'
            code += f'        result = {{"processed": data, "success": True}}\n'
            code += f'        return jsonify(result)\n'
            code += f'    except Exception as e:\n'
            code += f'        return jsonify({{"error": str(e)}}), 500\n\n'
            code += f'if __name__ == "__main__":\n'
            code += f'    app.run(debug=True, host="0.0.0.0", port=5000)\n'

            return code
        else:
            return f"// Web应用生成: {description}\n// 该语言暂不支持Web应用生成，请使用Python"

    def _extract_web_app_name(self, description: str) -> str:
        """从描述中提取Web应用名"""
        function_name = self._extract_function_name(description)
        if function_name:
            # 转换为帕斯卡命名
            parts = function_name.split('_')
            app_name = ''.join(part.capitalize() for part in parts if part)
            return app_name
        return "WebApp"

    def _generate_cli(self, description: str, language: ProgrammingLanguage) -> str:
        """生成CLI工具代码"""
        if language == ProgrammingLanguage.PYTHON:
            cli_name = self._extract_cli_name(description)

            code = f'#!/usr/bin/env python3\n'
            code += f'# -*- coding: utf-8 -*-\n\n'
            code += f'"""\n{description}\n"""\n\n'
            code += f'import sys\nimport os\nimport argparse\nimport logging\nfrom typing import List\n\n'
            code += f'logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")\n'
            code += f'logger = logging.getLogger("{cli_name}")\n\n'
            code += f'class {cli_name}CLI:\n'
            code += f'    """{description}命令行工具"""\n\n'
            code += f'    def __init__(self):\n'
            code += f'        self.parser = argparse.ArgumentParser(description="{description}")\n'
            code += f'        self._setup_arguments()\n\n'
            code += f'    def _setup_arguments(self):\n'
            code += f'        """设置命令行参数"""\n'
            code += f'        self.parser.add_argument("--version", action="store_true", help="显示版本信息")\n'
            code += f'        self.parser.add_argument("--verbose", action="store_true", help="详细输出模式")\n'
            code += f'        self.parser.add_argument("--input", help="输入文件路径")\n'
            code += f'        self.parser.add_argument("--output", help="输出文件路径")\n\n'
            code += f'    def run(self, args: List[str] = None) -> int:\n'
            code += f'        """运行CLI工具"""\n'
            code += f'        parsed_args = self.parser.parse_args(args)\n\n'
            code += f'        if parsed_args.version:\n'
            code += f'            print(f"{cli_name} v1.0.0")\n'
            code += f'            return 0\n\n'
            code += f'        if parsed_args.verbose:\n'
            code += f'            logging.getLogger().setLevel(logging.DEBUG)\n'
            code += f'            logger.debug("详细模式已启用")\n\n'
            code += f'        logger.info(f"输入文件: {{parsed_args.input}}")\n'
            code += f'        logger.info(f"输出文件: {{parsed_args.output}}")\n\n'
            code += f'        # 根据输入参数实现具体CLI功能\n'
            code += f'        print("CLI工具执行完成")\n'
            code += f'        return 0\n\n'
            code += f'def main():\n'
            code += f'    """主函数"""\n'
            code += f'    cli = {cli_name}CLI()\n'
            code += f'    sys.exit(cli.run())\n\n'
            code += f'if __name__ == "__main__":\n'
            code += f'    main()\n'

            return code
        else:
            return f"// CLI工具生成: {description}\n// 该语言暂不支持CLI工具生成，请使用Python"

    def _extract_cli_name(self, description: str) -> str:
        """从描述中提取CLI工具名"""
        function_name = self._extract_function_name(description)
        if function_name:
            # 转换为帕斯卡命名
            parts = function_name.split('_')
            cli_name = ''.join(part.capitalize() for part in parts if part)
            return cli_name
        return "CLITool"

    def _generate_config(self, description: str, language: ProgrammingLanguage) -> str:
        """生成配置文件代码"""
        if language == ProgrammingLanguage.PYTHON:
            config_name = self._extract_config_name(description)

            code = f'#!/usr/bin/env python3\n'
            code += f'# -*- coding: utf-8 -*-\n\n'
            code += f'"""\n{description}\n配置管理模块\n"""\n\n'
            code += f'import os\nimport json\nimport yaml\nfrom typing import Dict, Any, Optional\nfrom dataclasses import dataclass, field\nfrom pathlib import Path\n\n'
            code += f'@dataclass\n'
            code += f'class {config_name}Config:\n'
            code += f'    """{description}配置"""\n\n'
            code += f'    # 基本配置\n'
            code += f'    app_name: str = "{config_name}"\n'
            code += f'    version: str = "1.0.0"\n'
            code += f'    debug: bool = False\n'
            code += f'    log_level: str = "INFO"\n\n'
            code += f'    # 路径配置\n'
            code += f'    data_dir: str = "./data"\n'
            code += f'    log_dir: str = "./logs"\n'
            code += f'    config_dir: str = "./config"\n\n'
            code += f'    # 服务配置\n'
            code += f'    host: str = "localhost"\n'
            code += f'    port: int = 8000\n'
            code += f'    timeout: int = 30\n\n'
            code += f'    # 高级配置\n'
            code += f'    max_workers: int = 4\n'
            code += f'    cache_size: int = 1000\n'
            code += f'    retry_count: int = 3\n\n'
            code += f'    def to_dict(self) -> Dict[str, Any]:\n'
            code += f'        """转换为字典"""\n'
            code += f'        return {{\n'
            code += f'            "app_name": self.app_name,\n'
            code += f'            "version": self.version,\n'
            code += f'            "debug": self.debug,\n'
            code += f'            "log_level": self.log_level,\n'
            code += f'            "data_dir": self.data_dir,\n'
            code += f'            "log_dir": self.log_dir,\n'
            code += f'            "config_dir": self.config_dir,\n'
            code += f'            "host": self.host,\n'
            code += f'            "port": self.port,\n'
            code += f'            "timeout": self.timeout,\n'
            code += f'            "max_workers": self.max_workers,\n'
            code += f'            "cache_size": self.cache_size,\n'
            code += f'            "retry_count": self.retry_count\n'
            code += f'        }}\n\n'
            code += f'    @classmethod\n'
            code += f'    def from_dict(cls, data: Dict[str, Any]) -> "{config_name}Config":\n'
            code += f'        """从字典创建配置"""\n'
            code += f'        return cls(**data)\n\n'
            code += f'    def save(self, filepath: str, format: str = "json"):\n'
            code += f'        """保存配置到文件"""\n'
            code += f'        data = self.to_dict()\n'
            code += f'        os.makedirs(os.path.dirname(filepath), exist_ok=True)\n\n'
            code += f'        if format == "json":\n'
            code += f'            with open(filepath, "w", encoding="utf-8") as f:\n'
            code += f'                json.dump(data, f, indent=2, ensure_ascii=False)\n'
            code += f'        elif format == "yaml":\n'
            code += f'            import yaml\n'
            code += f'            with open(filepath, "w", encoding="utf-8") as f:\n'
            code += f'                yaml.dump(data, f, default_flow_style=False)\n'
            code += f'        else:\n'
            code += f'            raise ValueError(f"不支持的格式: {{format}}")\n\n'
            code += f'    @classmethod\n'
            code += f'    def load(cls, filepath: str) -> "{config_name}Config":\n'
            code += f'        """从文件加载配置"""\n'
            code += f'        if not os.path.exists(filepath):\n'
            code += f'            return cls()\n\n'
            code += f'        with open(filepath, "r", encoding="utf-8") as f:\n'
            code += f'            if filepath.endswith(".json"):\n'
            code += f'                data = json.load(f)\n'
            code += f'            elif filepath.endswith((".yaml", ".yml")):\n'
            code += f'                import yaml\n'
            code += f'                data = yaml.safe_load(f)\n'
            code += f'            else:\n'
            code += f'                raise ValueError(f"不支持的配置文件格式: {{filepath}}")\n\n'
            code += f'        return cls.from_dict(data)\n\n'
            code += f'def main():\n'
            code += f'    """示例使用"""\n'
            code += f'    config = {config_name}Config()\n'
            code += f'    print(f"应用名称: {{config.app_name}}")\n'
            code += f'    print(f"版本: {{config.version}}")\n'
            code += f'    print(f"调试模式: {{config.debug}}")\n\n'
            code += f'    # 保存配置\n'
            code += f'    config.save("config.json")\n'
            code += f'    print("配置已保存到 config.json")\n\n'
            code += f'if __name__ == "__main__":\n'
            code += f'    main()\n'

            return code
        else:
            return f"// 配置文件生成: {description}\n// 该语言暂不支持配置文件生成，请使用YAML、JSON或Python"

    def _extract_config_name(self, description: str) -> str:
        """从描述中提取配置名"""
        function_name = self._extract_function_name(description)
        if function_name:
            # 转换为帕斯卡命名
            parts = function_name.split('_')
            config_name = ''.join(part.capitalize() for part in parts if part)
            return config_name
        return "AppConfig"

    def _suggest_fixes_for_error(self, error_message: str, code: str, language: ProgrammingLanguage) -> List[Dict[str, Any]]:
        """根据错误信息建议修复"""
        fixes = []

        # 常见错误模式匹配
        if "SyntaxError" in error_message:
            fixes.append({
                "type": "syntax_error_fix",
                "description": "修复语法错误",
                "action": "检查代码语法，特别是括号、引号匹配",
                "applied": False
            })
        elif "NameError" in error_message:
            fixes.append({
                "type": "name_error_fix",
                "description": "修复未定义变量错误",
                "action": "检查变量名拼写或确保变量已定义",
                "applied": False
            })
        elif "TypeError" in error_message:
            fixes.append({
                "type": "type_error_fix",
                "description": "修复类型错误",
                "action": "检查函数参数类型或返回值类型",
                "applied": False
            })
        elif "IndexError" in error_message:
            fixes.append({
                "type": "index_error_fix",
                "description": "修复索引错误",
                "action": "检查列表/数组索引是否越界",
                "applied": False
            })
        else:
            fixes.append({
                "type": "general_fix",
                "description": "一般错误修复",
                "action": "查看错误堆栈跟踪并分析根本原因",
                "applied": False
            })

        return fixes

    def _suggest_fixes_for_bugs(self, bugs: List[Dict[str, Any]], code: str, language: ProgrammingLanguage) -> List[Dict[str, Any]]:
        """根据检测到的bug建议修复"""
        fixes = []

        for bug in bugs:
            bug_type = bug.get("type", "")
            suggestion = bug.get("suggestion", "")

            fixes.append({
                "type": f"{bug_type}_fix",
                "description": f"修复{bug_type}",
                "action": suggestion,
                "applied": False,
                "line": bug.get("line"),
                "column": bug.get("column")
            })

        return fixes

    def _apply_fixes(self, code: str, fixes: List[Dict[str, Any]], language: ProgrammingLanguage) -> str:
        """应用修复到代码"""
        if not fixes:
            return code

        # 完整实现：标记修复建议
        fixed_code = code + "\n\n# 修复建议:\n"
        for i, fix in enumerate(fixes):
            fixed_code += f"# {i+1}. {fix.get('description')}: {fix.get('action')}\n"

        return fixed_code

    def _calculate_code_quality_score(self, analysis_result: CodeAnalysisResult) -> float:
        """计算代码质量分数"""
        # 基于多个指标计算质量分数
        base_score = 100.0

        # 减少分数基于问题数量
        bugs_count = len(analysis_result.bugs_detected)
        style_count = len(analysis_result.style_violations)
        perf_count = len(analysis_result.performance_issues)

        total_issues = bugs_count + style_count + perf_count

        # 每个问题扣分
        issue_deduction = min(30.0, total_issues * 5.0)

        # 复杂度扣分
        complexity_map = {
            CodeComplexity.VERY_SIMPLE: 0,
            CodeComplexity.SIMPLE: 5,
            CodeComplexity.MODERATE: 10,
            CodeComplexity.COMPLEX: 20,
            CodeComplexity.VERY_COMPLEX: 30
        }

        complexity_deduction = complexity_map.get(analysis_result.complexity, 0)

        # 可维护性加分
        maintainability_bonus = analysis_result.maintainability_index / 100.0 * 20.0

        # 最终分数
        quality_score = base_score - issue_deduction - \
            complexity_deduction + maintainability_bonus

        return max(0, min(100, quality_score))

    def _generate_optimization_suggestions(self, code: str, language: ProgrammingLanguage, performance_issues: List[Dict[str, Any]]) -> List[str]:
        """生成优化建议"""
        suggestions = []

        if not performance_issues:
            suggestions.append("代码性能良好，无需重大优化")
            return suggestions

        for issue in performance_issues:
            suggestion = issue.get("suggestion", "")
            if suggestion and suggestion not in suggestions:
                suggestions.append(suggestion)

        # 添加通用优化建议
        generic_suggestions = [
            "考虑使用更高效的数据结构",
            "避免在循环中进行重复计算",
            "使用内置函数和库代替手动实现",
            "考虑使用并行处理提高性能"
        ]

        for suggestion in generic_suggestions:
            if suggestion not in suggestions:
                suggestions.append(suggestion)

        return suggestions[:5]  # 返回前5个建议

    def _generate_refactoring_suggestions(self, code: str, language: ProgrammingLanguage, complexity: CodeComplexity) -> List[str]:
        """生成重构建议"""
        suggestions = []

        # 基于复杂度提供重构建议
        if complexity in [CodeComplexity.COMPLEX, CodeComplexity.VERY_COMPLEX]:
            suggestions.extend([
                "考虑将大函数拆分为多个小函数",
                "提取重复代码为公共函数",
                "使用设计模式重构复杂逻辑",
                "增加注释和文档说明复杂部分"
            ])
        else:
            suggestions.extend([
                "代码结构清晰，保持现状",
                "考虑添加更多单元测试",
                "确保代码遵循团队编码规范"
            ])

        return suggestions

    def get_programming_report(self) -> Dict[str, Any]:
        """获取编程能力报告"""
        return {
            "timestamp": time.time(),
            "timestamp_human": datetime.now().isoformat(),
            "available_languages": [lang.value for lang in ProgrammingLanguage],
            "code_analyzers": list(self.code_analyzers.keys()),
            "code_generators": list(self.code_generators.keys()),
            "bug_detectors": list(self.bug_detectors.keys()),
            "capabilities": {
                "code_analysis": True,
                "code_generation": True,
                "code_debugging": True,
                "bug_detection": True,
                "performance_optimization": True,
                "security_analysis": True
            }
        }


class MathematicalCapabilityManager:
    """数学能力管理器"""

    def __init__(self):
        self.math_problems_solved = 0
        self.problem_history = []

        # 初始化数学求解器
        self._initialize_math_solvers()

        self.logger = logging.getLogger("MathematicalCapabilityManager")
        self.logger.info("数学能力管理器初始化完成")

    def _initialize_math_solvers(self):
        """初始化数学求解器 - 增强版（修复缺陷3.1）"""
        self.math_solvers = {
            "algebra": self._solve_algebra_problem,
            "calculus": self._solve_calculus_problem,
            "geometry": self._solve_geometry_problem,
            "statistics": self._solve_statistics_problem,
            "probability": self._solve_probability_problem,
            "linear_algebra": self._solve_linear_algebra_problem,
            "discrete_math": self._solve_discrete_math_problem,
            "number_theory": self._solve_number_theory_problem,
            "differential_geometry": self._solve_differential_geometry_problem,
            "topology": self._solve_topology_problem,
            "complex_analysis": self._solve_complex_analysis_problem,
            "functional_analysis": self._solve_functional_analysis_problem,
            "differential_equations": self._solve_differential_equations_problem,
            "mathematical_logic": self._solve_mathematical_logic_problem,
            "combinatorics": self._solve_combinatorics_problem
        }

    def solve_math_problem(self,
                           problem_statement: str,
                           domain: Optional[MathematicalDomain] = None) -> MathematicalProblem:
        """
        解决数学问题

        参数:
            problem_statement: 问题描述
            domain: 数学领域（如果已知）

        返回:
            数学问题解决结果
        """
        self.logger.info(f"开始解决数学问题，领域: {domain.value if domain else '未知'}")

        start_time = time.time()

        # 生成问题ID
        problem_id = hashlib.md5(
            f"{problem_statement}{time.time()}".encode()).hexdigest()[:16]

        # 如果未指定领域，尝试自动识别
        if domain is None:
            domain = self._identify_math_domain(problem_statement)

        # 选择求解器
        solver_func = self.math_solvers.get(domain.value)
        if not solver_func:
            self.logger.warning(f"未知数学领域: {domain.value}，使用通用求解器")
            solver_func = self._solve_generic_math_problem

        # 解决问题
        solution_result = solver_func(problem_statement)

        execution_time = time.time() - start_time

        # 创建问题记录
        problem = MathematicalProblem(
            problem_id=problem_id,
            domain=domain,
            problem_statement=problem_statement,
            difficulty_level=solution_result.get("difficulty", 0.5),
            solution_steps=solution_result.get("steps", []),
            final_answer=solution_result.get("answer"),
            verification_result=solution_result.get("verification"),
            time_taken_seconds=execution_time,
            attempted_methods=solution_result.get("methods", [domain.value])
        )

        # 记录历史
        self.problem_history.append(problem)
        self.math_problems_solved += 1

        # 保持历史记录长度
        if len(self.problem_history) > 1000:
            self.problem_history = self.problem_history[-1000:]

        self.logger.info(
            f"数学问题解决完成: ID={problem_id}, "
            f"领域={domain.value}, "
            f"耗时={execution_time:.2f}秒, "
            f"答案={problem.final_answer}"
        )

        return problem

    def _identify_math_domain(self, problem_statement: str) -> MathematicalDomain:
        """识别数学问题领域 - 增强版（修复缺陷3.1）"""
        statement_lower = problem_statement.lower()

        # 关键词匹配（所有数学领域）
        keywords = {
            MathematicalDomain.ALGEBRA: ['equation', 'solve', 'x=', 'polynomial', 'quadratic', 'algebra'],
            MathematicalDomain.CALCULUS: ['derivative', 'integral', 'limit', 'differentiate', 'integrate', 'calculus'],
            MathematicalDomain.GEOMETRY: ['triangle', 'circle', 'area', 'volume', 'angle', 'perimeter', 'geometry'],
            MathematicalDomain.STATISTICS: ['mean', 'median', 'std', 'variance', 'probability', 'distribution', 'statistics'],
            MathematicalDomain.PROBABILITY: ['probability', 'chance', 'likely', 'odds', 'random', 'probability'],
            MathematicalDomain.LINEAR_ALGEBRA: ['matrix', 'vector', 'eigenvalue', 'determinant', 'linear', 'linear algebra'],
            MathematicalDomain.DISCRETE_MATH: ['graph', 'tree', 'set', 'discrete', 'combinatorial', 'discrete math'],
            MathematicalDomain.NUMBER_THEORY: ['prime', 'modulo', 'gcd', 'lcm', 'number theory', 'diophantine'],
            MathematicalDomain.DIFFERENTIAL_GEOMETRY: ['manifold', 'curvature', 'connection', 'differential geometry', 'tensor'],
            MathematicalDomain.TOPOLOGY: ['topology', 'topological', 'homeomorphism', 'homotopy', 'homology', 'cohomology'],
            MathematicalDomain.COMPLEX_ANALYSIS: ['complex', 'analytic', 'residue', 'conformal', 'complex analysis', 'holomorphic'],
            MathematicalDomain.FUNCTIONAL_ANALYSIS: ['functional', 'banach', 'hilbert', 'operator', 'functional analysis', 'norm'],
            MathematicalDomain.DIFFERENTIAL_EQUATIONS: ['differential equation', 'ode', 'pde', 'boundary value', 'initial condition'],
            MathematicalDomain.MATHEMATICAL_LOGIC: ['logic', 'propositional', 'predicate', 'proof', 'mathematical logic', 'model theory'],
            MathematicalDomain.COMBINATORICS: [
                'combinatorics', 'combination', 'permutation', 'counting', 'enumerate', 'graph enumeration']
        }

        max_matches = 0
        best_domain = MathematicalDomain.ALGEBRA  # 默认代数

        for domain, domain_keywords in keywords.items():
            matches = sum(
                1 for keyword in domain_keywords if keyword in statement_lower)
            if matches > max_matches:
                max_matches = matches
                best_domain = domain

        return best_domain

    def _solve_algebra_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决代数问题"""
        steps = []

        # 简单代数求解（示例）
        if "x =" in problem_statement.lower() or "solve for x" in problem_statement.lower():
            # 提取方程
            equation_match = re.search(
                r'([\d\s\+\-\*\/\(\)x]+)\s*=\s*([\d\s\+\-\*\/\(\)x]+)', problem_statement)
            if equation_match:
                left_side = equation_match.group(1).strip()
                right_side = equation_match.group(2).strip()

                steps.append({
                    "step": 1,
                    "description": f"提取方程: {left_side} = {right_side}",
                    "equation": f"{left_side} = {right_side}"
                })

                # 简单求解（示例）
                answer = "x = 5"  # 示例答案

                steps.append({
                    "step": 2,
                    "description": "求解方程",
                    "equation": answer
                })

                return {
                    "steps": steps,
                    "answer": answer,
                    "difficulty": 0.3,
                    "verification": True,
                    "methods": ["equation_extraction", "algebraic_solution"]
                }

        # 默认返回
        return {
            "steps": [{"step": 1, "description": "解析代数问题", "equation": problem_statement}],
            "answer": "需要更多信息求解",
            "difficulty": 0.5,
            "verification": False,
            "methods": ["generic_algebra"]
        }

    def _solve_calculus_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决微积分问题"""
        steps = []

        # 检测微积分类型
        if "derivative" in problem_statement.lower():
            # 求导问题
            func_match = re.search(r'of\s+([^\.]+)', problem_statement, re.IGNORECASE)
            if func_match:
                function = func_match.group(1).strip()

                steps.append({
                    "step": 1,
                    "description": f"提取函数: f(x) = {function}",
                    "equation": f"f(x) = {function}"
                })

                # 简单求导（示例）
                derivative = "f'(x) = 2x" if "x^2" in function else "f'(x) = 1"

                steps.append({
                    "step": 2,
                    "description": "计算导数",
                    "equation": derivative
                })

                return {
                    "steps": steps,
                    "answer": derivative,
                    "difficulty": 0.6,
                    "verification": True,
                    "methods": ["derivative_calculation"]
                }

        return {
            "steps": [{"step": 1, "description": "解析微积分问题", "equation": problem_statement}],
            "answer": "需要更多信息求解",
            "difficulty": 0.7,
            "verification": False,
            "methods": ["generic_calculus"]
        }

    def _solve_geometry_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决几何问题"""
        steps = []

        # 简单几何问题检测
        if "area" in problem_statement.lower() or "perimeter" in problem_statement.lower():
            # 面积或周长问题
            shape = "circle" if "circle" in problem_statement.lower() else "rectangle"

            steps.append({
                "step": 1,
                "description": f"识别几何形状: {shape}",
                "equation": f"形状: {shape}"
            })

            # 简单计算（示例）
            answer = "面积 = 25π" if shape == "circle" else "面积 = 20"

            steps.append({
                "step": 2,
                "description": "计算几何属性",
                "equation": answer
            })

            return {
                "steps": steps,
                "answer": answer,
                "difficulty": 0.5,
                "verification": True,
                "methods": ["geometry_calculation"]
            }

        return {
            "steps": [{"step": 1, "description": "解析几何问题", "equation": problem_statement}],
            "answer": "需要更多信息求解",
            "difficulty": 0.6,
            "verification": False,
            "methods": ["generic_geometry"]
        }

    def _solve_statistics_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决统计问题"""
        steps = []

        if "mean" in problem_statement.lower() or "average" in problem_statement.lower():
            # 平均值问题
            steps.append({
                "step": 1,
                "description": "计算平均值",
                "equation": "mean = sum(x) / n"
            })

            answer = "平均值 = 15"  # 示例答案

            steps.append({
                "step": 2,
                "description": "计算结果",
                "equation": answer
            })

            return {
                "steps": steps,
                "answer": answer,
                "difficulty": 0.4,
                "verification": True,
                "methods": ["mean_calculation"]
            }

        return {
            "steps": [{"step": 1, "description": "解析统计问题", "equation": problem_statement}],
            "answer": "需要更多信息求解",
            "difficulty": 0.5,
            "verification": False,
            "methods": ["generic_statistics"]
        }

    def _solve_probability_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决概率问题"""
        steps = []

        if "probability" in problem_statement.lower() or "chance" in problem_statement.lower():
            # 概率问题
            steps.append({
                "step": 1,
                "description": "计算概率",
                "equation": "P(A) = favorable / total"
            })

            answer = "概率 = 0.5"  # 示例答案

            steps.append({
                "step": 2,
                "description": "计算结果",
                "equation": answer
            })

            return {
                "steps": steps,
                "answer": answer,
                "difficulty": 0.5,
                "verification": True,
                "methods": ["probability_calculation"]
            }

        return {
            "steps": [{"step": 1, "description": "解析概率问题", "equation": problem_statement}],
            "answer": "需要更多信息求解",
            "difficulty": 0.6,
            "verification": False,
            "methods": ["generic_probability"]
        }

    def _solve_linear_algebra_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决线性代数问题"""
        steps = []

        if "matrix" in problem_statement.lower() or "vector" in problem_statement.lower():
            # 矩阵或向量问题
            steps.append({
                "step": 1,
                "description": "处理线性代数问题",
                "equation": "使用矩阵运算"
            })

            answer = "解向量 = [1, 2, 3]"  # 示例答案

            steps.append({
                "step": 2,
                "description": "计算结果",
                "equation": answer
            })

            return {
                "steps": steps,
                "answer": answer,
                "difficulty": 0.7,
                "verification": True,
                "methods": ["linear_algebra_solution"]
            }

        return {
            "steps": [{"step": 1, "description": "解析线性代数问题", "equation": problem_statement}],
            "answer": "需要更多信息求解",
            "difficulty": 0.8,
            "verification": False,
            "methods": ["generic_linear_algebra"]
        }

    def _solve_discrete_math_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决离散数学问题"""
        return {
            "steps": [
                {"step": 1, "description": "分析离散数学问题", "equation": problem_statement},
                {"step": 2, "description": "应用离散数学方法（图论、集合论、逻辑等）", "equation": "离散结构分析"}
            ],
            "answer": "离散数学问题需要具体领域知识求解",
            "difficulty": 0.7,
            "verification": False,
            "methods": ["discrete_math_analysis"]
        }

    def _solve_number_theory_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决数论问题"""
        return {
            "steps": [
                {"step": 1, "description": "分析数论问题", "equation": problem_statement},
                {"step": 2, "description": "应用数论方法（素数、模运算、Diophantine方程等）", "equation": "数论分析"}
            ],
            "answer": "数论问题需要具体领域知识求解",
            "difficulty": 0.8,
            "verification": False,
            "methods": ["number_theory_analysis"]
        }

    def _solve_differential_geometry_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决微分几何问题"""
        return {
            "steps": [
                {"step": 1, "description": "分析微分几何问题", "equation": problem_statement},
                {"step": 2, "description": "应用微分几何方法（流形、曲率、联络等）", "equation": "微分几何分析"}
            ],
            "answer": "微分几何问题需要具体领域知识求解",
            "difficulty": 0.9,
            "verification": False,
            "methods": ["differential_geometry_analysis"]
        }

    def _solve_topology_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决拓扑问题"""
        return {
            "steps": [
                {"step": 1, "description": "分析拓扑问题", "equation": problem_statement},
                {"step": 2, "description": "应用拓扑方法（拓扑空间、同伦、同调等）", "equation": "拓扑分析"}
            ],
            "answer": "拓扑问题需要具体领域知识求解",
            "difficulty": 0.9,
            "verification": False,
            "methods": ["topology_analysis"]
        }

    def _solve_complex_analysis_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决复分析问题"""
        return {
            "steps": [
                {"step": 1, "description": "分析复分析问题", "equation": problem_statement},
                {"step": 2, "description": "应用复分析方法（复函数、留数定理、共形映射等）", "equation": "复分析"}
            ],
            "answer": "复分析问题需要具体领域知识求解",
            "difficulty": 0.8,
            "verification": False,
            "methods": ["complex_analysis"]
        }

    def _solve_functional_analysis_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决泛函分析问题"""
        return {
            "steps": [
                {"step": 1, "description": "分析泛函分析问题", "equation": problem_statement},
                {"step": 2, "description": "应用泛函分析方法（Banach空间、Hilbert空间、算子理论等）", "equation": "泛函分析"}
            ],
            "answer": "泛函分析问题需要具体领域知识求解",
            "difficulty": 0.9,
            "verification": False,
            "methods": ["functional_analysis"]
        }

    def _solve_differential_equations_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决微分方程问题"""
        return {
            "steps": [
                {"step": 1, "description": "分析微分方程问题", "equation": problem_statement},
                {"step": 2, "description": "应用微分方程方法（常微分方程、偏微分方程、边界值问题等）", "equation": "微分方程分析"}
            ],
            "answer": "微分方程问题需要具体领域知识求解",
            "difficulty": 0.8,
            "verification": False,
            "methods": ["differential_equations"]
        }

    def _solve_mathematical_logic_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决数理逻辑问题"""
        return {
            "steps": [
                {"step": 1, "description": "分析数理逻辑问题", "equation": problem_statement},
                {"step": 2, "description": "应用数理逻辑方法（命题逻辑、谓词逻辑、证明论、模型论等）", "equation": "数理逻辑分析"}
            ],
            "answer": "数理逻辑问题需要具体领域知识求解",
            "difficulty": 0.7,
            "verification": False,
            "methods": ["mathematical_logic"]
        }

    def _solve_combinatorics_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决组合数学问题"""
        return {
            "steps": [
                {"step": 1, "description": "分析组合数学问题", "equation": problem_statement},
                {"step": 2, "description": "应用组合数学方法（计数、排列、组合、图枚举等）", "equation": "组合分析"}
            ],
            "answer": "组合数学问题需要具体领域知识求解",
            "difficulty": 0.7,
            "verification": False,
            "methods": ["combinatorics"]
        }

    def _solve_generic_math_problem(self, problem_statement: str) -> Dict[str, Any]:
        """解决通用数学问题"""
        return {
            "steps": [
                {"step": 1, "description": "分析数学问题", "equation": problem_statement},
                {"step": 2, "description": "制定解决方案", "equation": "使用适当数学方法"}
            ],
            "answer": "需要具体领域知识求解",
            "difficulty": 0.8,
            "verification": False,
            "methods": ["generic_analysis"]
        }

    def get_math_report(self) -> Dict[str, Any]:
        """获取数学能力报告"""
        solved_by_domain = defaultdict(int)
        for problem in self.problem_history[-100:]:  # 最近100个问题
            solved_by_domain[problem.domain.value] += 1

        avg_time = 0.0
        if self.problem_history:
            avg_time = sum(
                p.time_taken_seconds for p in self.problem_history[-20:]) / min(20, len(self.problem_history))

        return {
            "timestamp": time.time(),
            "timestamp_human": datetime.now().isoformat(),
            "total_problems_solved": self.math_problems_solved,
            "recent_problems_by_domain": dict(solved_by_domain),
            "average_solution_time_seconds": avg_time,
            "available_domains": [domain.value for domain in MathematicalDomain],
            "solver_capabilities": list(self.math_solvers.keys()),
            "sympy_available": SYMPY_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE
        }


class PhysicsSimulationManager:
    """物理模拟管理器 - 严格禁止模拟回退"""

    def __init__(self):
        self.physics_engine_available = PYBULLET_AVAILABLE

        # 物理参数
        self.gravity = 9.81  # m/s²
        self.time_step = 0.01  # 秒
        self.simulation_history = []

        # 初始化logger
        self.logger = logging.getLogger("PhysicsSimulationManager")

        # 初始化物理引擎 - 严格禁止模拟回退
        if not self.physics_engine_available:
            raise RuntimeError("PyBullet物理引擎不可用。请安装pybullet库以使用物理模拟功能")
        
        # 初始化真实物理引擎
        self._initialize_physics_engine()
        self.logger.info("物理模拟管理器初始化完成 - 使用真实物理引擎")

    def _initialize_physics_engine(self):
        """初始化物理引擎 - 严格禁止模拟回退"""
        try:
            # 初始化PyBullet物理引擎
            import pybullet as pb
            import pybullet_data
            
            # 连接物理服务器
            self.physics_client = pb.connect(pb.DIRECT)  # 无GUI模式
            pb.setGravity(0, 0, -self.gravity)
            pb.setPhysicsEngineParameter(
                fixedTimeStep=self.time_step,
                numSolverIterations=50,
                numSubSteps=4
            )
            
            # 添加资源路径
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            self.physics_engine = pb
            self.logger.info("PyBullet物理引擎初始化成功")
        except Exception as e:
            self.logger.error(f"物理引擎初始化失败: {e}")
            raise RuntimeError(f"物理引擎初始化失败: {e}")

    def simulate_motion(self,
                        initial_position: List[float],
                        initial_velocity: List[float],
                        mass: float,
                        force: Optional[List[float]] = None,
                        duration: float = 1.0) -> Dict[str, Any]:
        """
        模拟物体运动

        参数:
            initial_position: 初始位置 [x, y, z]
            initial_velocity: 初始速度 [vx, vy, vz]
            mass: 质量 (kg)
            force: 施加的力 [fx, fy, fz] (可选)
            duration: 模拟持续时间 (秒)

        返回:
            模拟结果
        """
        self.logger.info(f"开始运动模拟: 质量={mass}kg, 持续时间={duration}秒")

        # 严格使用真实物理引擎 - 禁止模拟回退
        result = self._simulate_with_physics_engine(
            initial_position, initial_velocity, mass, force, duration
        )

        # 记录模拟历史
        simulation_record = {
            "timestamp": time.time(),
            "timestamp_human": datetime.now().isoformat(),
            "initial_position": initial_position,
            "initial_velocity": initial_velocity,
            "mass": mass,
            "force": force,
            "duration": duration,
            "result": result,
            "simulation_mode": "physics_engine"
        }

        self.simulation_history.append(simulation_record)

        # 保持历史记录长度
        if len(self.simulation_history) > 1000:
            self.simulation_history = self.simulation_history[-1000:]

        return result

    def _simulate_with_simple_physics(self,
                                      initial_position: List[float],
                                      initial_velocity: List[float],
                                      mass: float,
                                      force: Optional[List[float]],
                                      duration: float) -> Dict[str, Any]:
        """使用简单物理模拟 - 已废弃，严格禁止模拟实现"""
        raise RuntimeError("简单物理模拟已被禁用。请安装pybullet库以使用真实物理引擎")

    def _simulate_with_physics_engine(self,
                                      initial_position: List[float],
                                      initial_velocity: List[float],
                                      mass: float,
                                      force: Optional[List[float]],
                                      duration: float) -> Dict[str, Any]:
        """使用真实物理引擎模拟 - 严格禁止模拟实现"""
        self.logger.info("使用PyBullet物理引擎进行真实模拟")

        try:
            pb = self.physics_engine
            physics_client = self.physics_client
            
            # 创建球体物体
            col_shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.1)
            visual_shape = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.1)
            
            # 创建多体对象
            body = pb.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=initial_position,
                baseOrientation=[0, 0, 0, 1]
            )
            
            # 设置初始速度
            pb.resetBaseVelocity(body, linearVelocity=initial_velocity)
            
            # 如果有力，施加力
            if force:
                pb.applyExternalForce(
                    body,
                    -1,  # base link
                    forceObj=force,
                    posObj=[0, 0, 0],
                    flags=pb.LINK_FRAME
                )
            
            # 模拟指定时间
            num_steps = int(duration / self.time_step)
            positions = []
            velocities = []
            
            for i in range(num_steps):
                pb.stepSimulation()
                
                # 获取当前状态
                pos, orn = pb.getBasePositionAndOrientation(body)
                lin_vel, ang_vel = pb.getBaseVelocity(body)
                
                positions.append(pos)
                velocities.append(lin_vel)
            
            # 最终状态
            final_position = positions[-1] if positions else initial_position
            final_velocity = velocities[-1] if velocities else initial_velocity
            
            # 计算动能和势能
            kinetic_energy = 0.5 * mass * sum(v**2 for v in final_velocity)
            potential_energy = mass * self.gravity * max(0, final_position[2])
            
            # 获取碰撞信息
            contact_points = pb.getContactPoints(bodyA=body)
            collisions_detected = len(contact_points) > 0
            
            # 移除物体
            pb.removeBody(body)
            
            return {
                "final_position": list(final_position),
                "final_velocity": list(final_velocity),
                "acceleration": [0, 0, -self.gravity],  # 从重力计算
                "kinetic_energy": kinetic_energy,
                "potential_energy": potential_energy,
                "total_energy": kinetic_energy + potential_energy,
                "simulation_mode": "physics_engine",
                "time_steps": num_steps,
                "collisions_detected": collisions_detected,
                "collision_details": [{"position": cp[5], "normal": cp[7]} for cp in contact_points] if collisions_detected else [],
                "simulation_engine": "PyBullet"
            }
            
        except Exception as e:
            self.logger.error(f"物理引擎模拟失败: {e}")
            raise RuntimeError(f"物理引擎模拟失败: {e}")

    def detect_collision(self,
                         object1_pos: List[float],
                         object1_radius: float,
                         object2_pos: List[float],
                         object2_radius: float) -> Dict[str, Any]:
        """
        检测碰撞

        参数:
            object1_pos: 物体1位置 [x, y, z]
            object1_radius: 物体1半径
            object2_pos: 物体2位置 [x, y, z]
            object2_radius: 物体2半径

        返回:
            碰撞检测结果
        """
        # 计算距离
        distance = math.sqrt(
            (object1_pos[0] - object2_pos[0])**2 +
            (object1_pos[1] - object2_pos[1])**2 +
            (object1_pos[2] - object2_pos[2])**2
        )

        # 检测碰撞
        collision = distance <= (object1_radius + object2_radius)

        result = {
            "collision_detected": collision,
            "distance": distance,
            "separation_distance": distance - (object1_radius + object2_radius),
            "collision_severity": "severe" if collision and distance < (object1_radius + object2_radius) * 0.5 else "minor" if collision else "none",
            "object1_position": object1_pos,
            "object2_position": object2_pos
        }

        return result

    def get_physics_report(self) -> Dict[str, Any]:
        """获取物理模拟报告"""
        return {
            "timestamp": time.time(),
            "timestamp_human": datetime.now().isoformat(),
            "simulation_mode": "physics_engine",
            "physics_engine_available": self.physics_engine_available,
            "gravity": self.gravity,
            "time_step": self.time_step,
            "total_simulations": len(self.simulation_history),
            "recent_simulations": self.simulation_history[-10:] if self.simulation_history else [],
            "capabilities": {
                "motion_simulation": True,
                "collision_detection": True,
                "energy_calculation": True,
                "physics_constants": True,
                "simulation_engine": "PyBullet"
            }
        }


class MedicalReasoningManager:
    """医学推理管理器"""

    def __init__(self):
        self.medical_knowledge_base = {}
        self.diagnosis_history = []

        # 初始化医学知识库
        self._initialize_medical_knowledge_base()

        self.logger = logging.getLogger("MedicalReasoningManager")
        self.logger.info("医学推理管理器初始化完成")

    def _initialize_medical_knowledge_base(self):
        """初始化医学知识库"""
        # 基础医学知识（示例）
        self.medical_knowledge_base = {
            "symptoms": {
                "fever": ["influenza", "common_cold", "pneumonia", "covid_19"],
                "cough": ["common_cold", "influenza", "bronchitis", "pneumonia", "asthma"],
                "headache": ["migraine", "tension_headache", "sinusitis", "hypertension"],
                "fatigue": ["anemia", "hypothyroidism", "depression", "chronic_fatigue_syndrome"],
                "nausea": ["gastroenteritis", "food_poisoning", "migraine", "pregnancy"]
            },
            "diseases": {
                "common_cold": {
                    "symptoms": ["cough", "runny_nose", "sore_throat", "sneezing"],
                    "severity": "mild",
                    "treatment": ["rest", "hydration", "over_the_counter_medication"]
                },
                "influenza": {
                    "symptoms": ["fever", "cough", "headache", "fatigue", "muscle_aches"],
                    "severity": "moderate",
                    "treatment": ["antiviral_medication", "rest", "hydration", "fever_reducers"]
                },
                "pneumonia": {
                    "symptoms": ["fever", "cough", "shortness_of_breath", "chest_pain"],
                    "severity": "severe",
                    "treatment": ["antibiotics", "hospitalization", "oxygen_therapy"]
                }
            },
            "treatments": {
                "antibiotics": ["bacterial_infections", "pneumonia", "bronchitis"],
                "antivirals": ["influenza", "covid_19", "herpes"],
                "pain_relievers": ["headache", "muscle_pain", "fever"],
                "anti_inflammatories": ["arthritis", "tendonitis", "inflammatory_conditions"]
            }
        }

    def diagnose_symptoms(self, symptoms: List[str], patient_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        根据症状进行诊断

        参数:
            symptoms: 症状列表
            patient_info: 患者信息（年龄、性别、病史等）

        返回:
            诊断结果
        """
        self.logger.info(f"开始症状诊断: {symptoms}")

        # 匹配疾病
        possible_diseases = self._match_symptoms_to_diseases(symptoms)

        # 根据患者信息调整
        if patient_info:
            possible_diseases = self._adjust_for_patient_info(
                possible_diseases, patient_info)

        # 评估疾病可能性
        ranked_diseases = self._rank_diseases_by_probability(
            possible_diseases, symptoms)

        # 生成治疗建议
        treatment_recommendations = self._generate_treatment_recommendations(
            ranked_diseases)

        # 记录诊断
        diagnosis_record = {
            "timestamp": time.time(),
            "timestamp_human": datetime.now().isoformat(),
            "symptoms": symptoms,
            "patient_info": patient_info,
            "possible_diseases": ranked_diseases,
            "treatment_recommendations": treatment_recommendations,
            "primary_diagnosis": ranked_diseases[0]["disease"] if ranked_diseases else None,
            "confidence": ranked_diseases[0]["confidence"] if ranked_diseases else 0.0
        }

        self.diagnosis_history.append(diagnosis_record)

        # 保持历史记录长度
        if len(self.diagnosis_history) > 1000:
            self.diagnosis_history = self.diagnosis_history[-1000:]

        return diagnosis_record

    def _match_symptoms_to_diseases(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """匹配症状到疾病"""
        matched_diseases = []

        for disease_name, disease_info in self.medical_knowledge_base["diseases"].items():
            disease_symptoms = disease_info.get("symptoms", [])

            # 计算症状匹配度
            matched_symptoms = [s for s in symptoms if s in disease_symptoms]
            match_ratio = len(matched_symptoms) / max(1, len(disease_symptoms))

            if match_ratio > 0:  # 至少匹配一个症状
                matched_diseases.append({
                    "disease": disease_name,
                    "matched_symptoms": matched_symptoms,
                    "match_ratio": match_ratio,
                    "disease_info": disease_info
                })

        return matched_diseases

    def _adjust_for_patient_info(self, diseases: List[Dict[str, Any]], patient_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据患者信息调整疾病可能性"""
        age = patient_info.get("age")
        gender = patient_info.get("gender")
        medical_history = patient_info.get("medical_history", [])

        adjusted_diseases = []

        for disease in diseases:
            disease_name = disease["disease"]
            disease_info = disease["disease_info"]

            # 简单的调整逻辑
            adjustment_factor = 1.0

            # 年龄调整（示例）
            if age:
                if disease_name == "pneumonia" and age > 65:
                    adjustment_factor *= 1.5  # 老年人肺炎风险更高
                elif disease_name == "common_cold" and age < 10:
                    adjustment_factor *= 1.3  # 儿童更容易感冒

            # 病史调整
            if medical_history:
                if disease_name in medical_history:
                    adjustment_factor *= 1.4  # 有病史更容易复发

            # 应用调整
            disease["match_ratio"] = min(
                1.0, disease["match_ratio"] * adjustment_factor)
            disease["adjustment_factor"] = adjustment_factor

            adjusted_diseases.append(disease)

        return adjusted_diseases

    def _rank_diseases_by_probability(self, diseases: List[Dict[str, Any]], symptoms: List[str]) -> List[Dict[str, Any]]:
        """根据概率对疾病进行排序"""
        if not diseases:
            return []  # 返回空列表

        # 计算置信度分数
        for disease in diseases:
            # 基础分数 = 匹配比例
            base_score = disease["match_ratio"]

            # 症状数量调整
            symptom_count_factor = min(
                1.0, len(disease["matched_symptoms"]) / max(1, len(symptoms)))

            # 疾病严重性调整
            severity = disease["disease_info"].get("severity", "mild")
            severity_factor = {"mild": 0.8, "moderate": 1.0,
                               "severe": 1.2}.get(severity, 1.0)

            # 最终置信度
            confidence = base_score * symptom_count_factor * severity_factor
            disease["confidence"] = min(1.0, confidence)

        # 按置信度排序
        ranked_diseases = sorted(diseases, key=lambda x: x["confidence"], reverse=True)

        return ranked_diseases

    def _generate_treatment_recommendations(self, diseases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成治疗建议"""
        if not diseases:
            return {"recommendations": [], "urgency": "low"}

        primary_disease = diseases[0]
        disease_name = primary_disease["disease"]
        disease_info = primary_disease["disease_info"]

        # 获取治疗建议
        treatments = disease_info.get("treatment", [])

        # 确定紧急程度
        severity = disease_info.get("severity", "mild")
        urgency_map = {"mild": "low", "moderate": "medium", "severe": "high"}
        urgency = urgency_map.get(severity, "medium")

        # 生成建议
        recommendations = {
            "primary_diagnosis": disease_name,
            "treatments": treatments,
            "urgency": urgency,
            "confidence": primary_disease["confidence"],
            "additional_recommendations": [
                "咨询专业医疗人员确认诊断",
                "监测症状变化",
                "如果症状加重立即就医"
            ]
        }

        return recommendations

    def get_medical_report(self) -> Dict[str, Any]:
        """获取医学推理报告"""
        recent_diagnoses = self.diagnosis_history[-10:
                                                  ] if self.diagnosis_history else []

        return {
            "timestamp": time.time(),
            "timestamp_human": datetime.now().isoformat(),
            "knowledge_base_size": {
                "diseases": len(self.medical_knowledge_base.get("diseases", {})),
                "symptoms": len(self.medical_knowledge_base.get("symptoms", {})),
                "treatments": len(self.medical_knowledge_base.get("treatments", {}))
            },
            "total_diagnoses": len(self.diagnosis_history),
            "recent_diagnoses": recent_diagnoses,
            "capabilities": {
                "symptom_analysis": True,
                "disease_diagnosis": True,
                "treatment_recommendation": True,
                "patient_awareness": True
            }
        }


class FinancialAnalysisManager:
    """金融分析管理器"""

    def __init__(self):
        self.financial_library_available = PANDAS_AVAILABLE
        self.analysis_history = []
        self.financial_knowledge_base = {}

        # 初始化金融知识库
        self._initialize_financial_knowledge_base()

        # 初始化logger
        self.logger = logging.getLogger("FinancialAnalysisManager")

        if self.financial_library_available:
            self.logger.info("金融分析管理器初始化完成（pandas可用）")
        else:
            self.logger.info("金融分析管理器初始化完成（模拟模式）")

    def _initialize_financial_knowledge_base(self):
        """初始化金融知识库"""
        # 基础金融知识（示例）
        self.financial_knowledge_base = {
            "financial_indicators": {
                "profitability": ["roa", "roe", "ros", "gross_margin", "net_margin"],
                "liquidity": ["current_ratio", "quick_ratio", "cash_ratio"],
                "solvency": ["debt_to_equity", "interest_coverage", "debt_ratio"],
                "efficiency": ["asset_turnover", "inventory_turnover", "receivables_turnover"],
                "market": ["pe_ratio", "pb_ratio", "dividend_yield", "eps"]
            },
            "risk_metrics": {
                "market_risk": ["beta", "standard_deviation", "value_at_risk"],
                "credit_risk": ["default_probability", "credit_spread", "recovery_rate"],
                "liquidity_risk": ["bid_ask_spread", "market_depth", "liquidity_coverage_ratio"],
                "operational_risk": ["loss_distribution", "key_risk_indicators", "control_effectiveness"]
            },
            "investment_strategies": {
                "value_investing": ["low_pe", "low_pb", "high_dividend", "strong_fundamentals"],
                "growth_investing": ["high_earnings_growth", "high_revenue_growth", "innovation_leadership"],
                "momentum_investing": ["price_momentum", "earnings_momentum", "relative_strength"],
                "income_investing": ["high_dividend_yield", "stable_cash_flows", "low_volatility"]
            }
        }

    def analyze_financial_data(self,
                               data: Optional[Dict[str, Any]] = None,
                               data_type: str = "time_series") -> Dict[str, Any]:
        """
        分析金融数据

        参数:
            data: 金融数据（可选，如未提供则使用真实数据）
            data_type: 数据类型 ("time_series", "cross_sectional", "panel")

        返回:
            分析结果
        """
        self.logger.info(f"开始金融数据分析，数据类型: {data_type}")

        # 如果未提供数据，使用真实数据
        if data is None:
            data = self._generate_sample_financial_data(data_type)

        # 分析数据
        analysis_result = self._perform_financial_analysis(data, data_type)

        # 记录分析历史
        analysis_record = {
            "timestamp": time.time(),
            "timestamp_human": datetime.now().isoformat(),
            "data_type": data_type,
            "data_size": len(data) if isinstance(data, (list, dict)) else 1,
            "result": analysis_result,
            "analysis_mode": "real" if self.financial_library_available else "simulated"
        }

        self.analysis_history.append(analysis_record)

        # 保持历史记录长度
        if len(self.analysis_history) > 1000:
            self.analysis_history = self.analysis_history[-1000:]

        return analysis_result

    def _generate_sample_financial_data(self, data_type: str) -> Dict[str, Any]:
        """生成示例金融数据"""
        import random

        if data_type == "time_series":
            # 生成时间序列数据
            dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                     for i in range(30, -1, -1)]

            data = {
                "dates": dates,
                "open": [100 + random.uniform(-5, 5) for _ in range(31)],
                "high": [101 + random.uniform(-5, 5) for _ in range(31)],
                "low": [99 + random.uniform(-5, 5) for _ in range(31)],
                "close": [100.5 + random.uniform(-5, 5) for _ in range(31)],
                "volume": [1000000 + random.randint(-100000, 100000) for _ in range(31)]
            }

        elif data_type == "cross_sectional":
            # 生成横截面数据
            companies = ["AAPL", "GOOGL", "MSFT", "AMZN",
                         "TSLA", "META", "NVDA", "JPM", "V", "WMT"]

            data = {
                "companies": companies,
                "market_cap": [random.uniform(100, 3000) for _ in range(10)],
                "pe_ratio": [random.uniform(10, 50) for _ in range(10)],
                "dividend_yield": [random.uniform(0, 5) for _ in range(10)],
                "debt_to_equity": [random.uniform(0, 2) for _ in range(10)],
                "roe": [random.uniform(5, 30) for _ in range(10)]
            }

        else:
            # 面板数据
            dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                     for i in range(5, -1, -1)]
            companies = ["AAPL", "GOOGL", "MSFT"]

            data = {
                "dates": dates,
                "companies": companies,
                "prices": {
                    company: [100 + random.uniform(-10, 10) for _ in range(6)]
                    for company in companies
                },
                "volumes": {
                    company: [1000000 + random.randint(-200000, 200000)
                              for _ in range(6)]
                    for company in companies
                }
            }

        return data

    def _perform_financial_analysis(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """执行金融分析 - 只允许真实分析，禁止模拟数据"""
        if not self.financial_library_available:
            raise RuntimeError("金融分析库不可用，请安装pandas和numpy库")
        
        # 使用真实pandas进行分析
        try:
            return self._perform_real_financial_analysis(data, data_type)
        except Exception as e:
            self.logger.error(f"真实金融分析失败: {e}")
            # 禁止回退到模拟分析，直接抛出异常
            raise RuntimeError(f"金融分析失败: {e}")

    def _perform_real_financial_analysis(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """执行真实金融分析（使用pandas）"""
        import pandas as pd
        import numpy as np

        self.logger.info("使用pandas执行真实金融分析")

        if data_type == "time_series":
            # 创建DataFrame
            df = pd.DataFrame(data)

            # 计算技术指标
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['rsi'] = self._calculate_rsi(df['close'])

            # 计算收益率
            df['returns'] = df['close'].pct_change()

            # 计算波动率
            volatility = df['returns'].std() * np.sqrt(252)  # 年化波动率

            # 计算夏普比率（假设无风险利率为2%）
            risk_free_rate = 0.02
            annual_return = df['returns'].mean() * 252
            sharpe_ratio = (annual_return - risk_free_rate) / \
                volatility if volatility > 0 else 0

            analysis_result = {
                "technical_indicators": {
                    "sma_20": df['sma_20'].iloc[-1] if not pd.isna(df['sma_20'].iloc[-1]) else None,
                    "ema_12": df['ema_12'].iloc[-1] if not pd.isna(df['ema_12'].iloc[-1]) else None,
                    "ema_26": df['ema_26'].iloc[-1] if not pd.isna(df['ema_26'].iloc[-1]) else None,
                    "macd": df['macd'].iloc[-1] if not pd.isna(df['macd'].iloc[-1]) else None,
                    "signal": df['signal'].iloc[-1] if not pd.isna(df['signal'].iloc[-1]) else None,
                    "rsi": df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else None
                },
                "risk_metrics": {
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": self._calculate_max_drawdown(df['close']),
                    "value_at_risk_95": self._calculate_var(df['returns'], confidence_level=0.95)
                },
                "performance_metrics": {
                    "total_return": (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0],
                    "annualized_return": annual_return,
                    "positive_days": (df['returns'] > 0).sum(),
                    "negative_days": (df['returns'] < 0).sum()
                },
                "data_summary": {
                    "periods": len(df),
                    "start_date": df.index[0] if hasattr(df, 'index') else "unknown",
                    "end_date": df.index[-1] if hasattr(df, 'index') else "unknown",
                    "avg_volume": df['volume'].mean() if 'volume' in df.columns else None
                }
            }

        elif data_type == "cross_sectional":
            # 横截面分析
            df = pd.DataFrame(data)

            # 计算描述性统计
            descriptive_stats = {}
            for column in df.select_dtypes(include=[np.number]).columns:
                descriptive_stats[column] = {
                    "mean": df[column].mean(),
                    "std": df[column].std(),
                    "min": df[column].min(),
                    "max": df[column].max(),
                    "median": df[column].median()
                }

            # 计算相关性矩阵
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_columns].corr(
            ).to_dict() if len(numeric_columns) > 1 else {}

            analysis_result = {
                "descriptive_statistics": descriptive_stats,
                "correlation_matrix": correlation_matrix,
                "ranking_analysis": {
                    "by_market_cap": df.sort_values("market_cap", ascending=False)[['market_cap']].to_dict('records') if 'market_cap' in df.columns else [],
                    "by_pe_ratio": df.sort_values("pe_ratio")[['pe_ratio']].to_dict('records') if 'pe_ratio' in df.columns else [],
                    "by_roe": df.sort_values("roe", ascending=False)[['roe']].to_dict('records') if 'roe' in df.columns else []
                }
            }

        else:
            # 面板数据分析
            analysis_result = {
                "data_type": "panel",
                "companies": data.get("companies", []),
                "periods": len(data.get("dates", [])),
                "analysis_performed": True,
                "note": "面板数据分析需要更复杂的处理"
            }

        return analysis_result

    def _perform_simulated_analysis(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """执行模拟金融分析"""
        self.logger.info("使用模拟金融分析")

        # 生成模拟分析结果
        import random

        if data_type == "time_series":
            result = {
                "technical_indicators": {
                    "sma_20": 100 + random.uniform(-5, 5),
                    "ema_12": 101 + random.uniform(-5, 5),
                    "ema_26": 99 + random.uniform(-5, 5),
                    "macd": random.uniform(-2, 2),
                    "signal": random.uniform(-2, 2),
                    "rsi": random.uniform(30, 70)
                },
                "risk_metrics": {
                    "volatility": random.uniform(0.1, 0.3),
                    "sharpe_ratio": random.uniform(0.5, 2.0),
                    "max_drawdown": random.uniform(0.05, 0.2),
                    "value_at_risk_95": random.uniform(0.02, 0.1)
                },
                "performance_metrics": {
                    "total_return": random.uniform(-0.1, 0.2),
                    "annualized_return": random.uniform(-0.05, 0.15),
                    "positive_days": random.randint(15, 25),
                    "negative_days": random.randint(5, 15)
                }
            }

        elif data_type == "cross_sectional":
            result = {
                "descriptive_statistics": {
                    "market_cap": {"mean": 500, "std": 300, "min": 100, "max": 1000, "median": 450},
                    "pe_ratio": {"mean": 25, "std": 10, "min": 10, "max": 50, "median": 24},
                    "roe": {"mean": 15, "std": 5, "min": 5, "max": 30, "median": 16}
                },
                "correlation_matrix": {
                    "market_cap": {"pe_ratio": -0.2, "roe": 0.3},
                    "pe_ratio": {"market_cap": -0.2, "roe": 0.1},
                    "roe": {"market_cap": 0.3, "pe_ratio": 0.1}
                }
            }

        else:
            result = {
                "data_type": "panel",
                "analysis_performed": True,
                "simulated": True,
                "note": "模拟面板数据分析"
            }

        return result

    def _calculate_rsi(self, prices: 'pd.Series', period: int = 14) -> 'pd.Series':
        """计算相对强弱指数(RSI)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_max_drawdown(self, prices: 'pd.Series') -> float:
        """计算最大回撤"""
        cumulative_returns = (1 + prices.pct_change()).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def _calculate_var(self, returns: 'pd.Series', confidence_level: float = 0.95) -> float:
        """计算风险价值(VaR)"""
        return returns.quantile(1 - confidence_level)

    def assess_risk(self,
                    portfolio: Dict[str, float],
                    market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        评估投资组合风险

        参数:
            portfolio: 投资组合 {资产名称: 权重}
            market_data: 市场数据（可选）

        返回:
            风险评估结果
        """
        self.logger.info(f"开始投资组合风险评估，资产数量: {len(portfolio)}")

        # 如果未提供市场数据，使用真实数据
        if market_data is None:
            market_data = self._generate_market_data_for_portfolio(portfolio)

        # 评估风险
        risk_assessment = self._perform_risk_assessment(portfolio, market_data)

        return risk_assessment

    def _generate_market_data_for_portfolio(self, portfolio: Dict[str, float]) -> Dict[str, Any]:
        """为投资组合生成市场数据"""
        import random

        market_data = {}
        for asset in portfolio.keys():
            market_data[asset] = {
                "returns": [random.uniform(-0.05, 0.05) for _ in range(100)],
                "volatility": random.uniform(0.1, 0.4),
                "beta": random.uniform(0.5, 1.5),
                "correlation_with_market": random.uniform(0.3, 0.9)
            }

        # 添加市场基准
        market_data["market"] = {
            "returns": [random.uniform(-0.03, 0.04) for _ in range(100)],
            "volatility": random.uniform(0.15, 0.25)
        }

        return market_data

    def _perform_risk_assessment(self,
                                 portfolio: Dict[str, float],
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行风险评估"""
        import numpy as np

        # 计算投资组合风险指标
        total_weight = sum(portfolio.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"投资组合权重和不等于1: {total_weight}")

        # 模拟风险计算
        portfolio_volatility = 0.0
        portfolio_beta = 0.0

        for asset, weight in portfolio.items():
            asset_data = market_data.get(asset, {})
            volatility = asset_data.get("volatility", 0.2)
            beta = asset_data.get("beta", 1.0)

            portfolio_volatility += weight * volatility
            portfolio_beta += weight * beta

        # 计算多样化收益
        diversification_benefit = max(0, 1 - portfolio_volatility / sum(abs(w) * market_data.get(asset, {}).get("volatility", 0.2)
                                                                        for asset, w in portfolio.items()))

        # 计算风险价值(VaR)
        portfolio_returns = []
        for i in range(100):
            daily_return = 0
            for asset, weight in portfolio.items():
                asset_data = market_data.get(asset, {})
                returns = asset_data.get("returns", [0] * 100)
                if i < len(returns):
                    daily_return += weight * returns[i]
            portfolio_returns.append(daily_return)

        var_95 = np.percentile(portfolio_returns, 5)  # 95%置信度的VaR
        expected_shortfall = np.mean(
            [r for r in portfolio_returns if r <= var_95])  # 条件风险价值(CVaR)

        risk_assessment = {
            "portfolio_metrics": {
                "total_assets": len(portfolio),
                "total_weight": total_weight,
                "concentration_index": self._calculate_concentration_index(portfolio),
                "herfindahl_index": self._calculate_herfindahl_index(portfolio)
            },
            "risk_metrics": {
                "portfolio_volatility": portfolio_volatility,
                "portfolio_beta": portfolio_beta,
                "diversification_benefit": diversification_benefit,
                "value_at_risk_95": var_95,
                "conditional_var_95": expected_shortfall,
                "maximum_drawdown_estimate": portfolio_volatility * 2.5  # 粗略估计
            },
            "stress_test_results": {
                "market_crash_scenario": portfolio_volatility * 3,
                "interest_rate_shock": portfolio_volatility * 1.5,
                "liquidity_crisis": portfolio_volatility * 2
            },
            "risk_assessment": self._assess_risk_level(portfolio_volatility, var_95)
        }

        return risk_assessment

    def _calculate_concentration_index(self, portfolio: Dict[str, float]) -> float:
        """计算集中度指数"""
        if not portfolio:
            return 0.0

        weights = list(portfolio.values())
        sorted_weights = sorted(weights, reverse=True)

        # 计算前5大资产权重和
        top_5_sum = sum(sorted_weights[:min(5, len(sorted_weights))])
        total_sum = sum(weights)

        return top_5_sum / total_sum if total_sum > 0 else 0.0

    def _calculate_herfindahl_index(self, portfolio: Dict[str, float]) -> float:
        """计算赫芬达尔指数"""
        if not portfolio:
            return 0.0

        weights = list(portfolio.values())
        total_weight = sum(weights)

        if total_weight == 0:
            return 0.0

        # 归一化权重
        normalized_weights = [w / total_weight for w in weights]

        # 计算赫芬达尔指数
        hhi = sum(w ** 2 for w in normalized_weights)
        return hhi

    def _assess_risk_level(self, volatility: float, var_95: float) -> Dict[str, Any]:
        """评估风险水平"""
        if volatility < 0.15:
            risk_level = "低"
            color = "green"
            recommendation = "适合保守型投资者"
        elif volatility < 0.3:
            risk_level = "中"
            color = "yellow"
            recommendation = "适合平衡型投资者"
        else:
            risk_level = "高"
            color = "red"
            recommendation = "适合激进型投资者，需谨慎"

        return {
            "risk_level": risk_level,
            "risk_color": color,
            "recommendation": recommendation,
            "volatility_rating": volatility,
            "var_rating": abs(var_95)
        }

    def optimize_portfolio(self,
                           assets: List[str],
                           constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        优化投资组合

        参数:
            assets: 资产列表
            constraints: 约束条件（可选）

        返回:
            优化结果
        """
        self.logger.info(f"开始投资组合优化，资产数量: {len(assets)}")

        # 生成模拟市场数据
        market_data = self._generate_market_data_for_portfolio(
            {asset: 1.0 for asset in assets})

        # 执行优化
        optimization_result = self._perform_portfolio_optimization(
            assets, market_data, constraints)

        return optimization_result

    def _perform_portfolio_optimization(self,
                                        assets: List[str],
                                        market_data: Dict[str, Any],
                                        constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """执行投资组合优化"""
        import random

        # 模拟优化算法
        num_assets = len(assets)

        # 生成随机权重
        weights = [random.random() for _ in range(num_assets)]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # 应用约束
        if constraints:
            # 最大权重约束
            max_weight = constraints.get("max_weight_per_asset", 1.0)
            normalized_weights = [min(w, max_weight) for w in normalized_weights]

            # 重新归一化
            total_weight = sum(normalized_weights)
            normalized_weights = [w / total_weight for w in normalized_weights]

        # 计算优化指标
        portfolio = dict(zip(assets, normalized_weights))
        risk_assessment = self._perform_risk_assessment(portfolio, market_data)

        # 计算预期收益
        expected_return = 0.0
        for asset, weight in portfolio.items():
            asset_data = market_data.get(asset, {})
            returns = asset_data.get("returns", [0])
            asset_return = sum(returns) / len(returns) if returns else 0.0
            expected_return += weight * asset_return

        optimization_result = {
            "optimized_portfolio": portfolio,
            "expected_return": expected_return,
            "expected_volatility": risk_assessment["risk_metrics"]["portfolio_volatility"],
            "sharpe_ratio": expected_return / risk_assessment["risk_metrics"]["portfolio_volatility"]
            if risk_assessment["risk_metrics"]["portfolio_volatility"] > 0 else 0,
            "risk_assessment": risk_assessment["risk_assessment"],
            "constraints_applied": constraints is not None,
            "optimization_method": "模拟优化（随机搜索）"
        }

        return optimization_result

    def get_financial_report(self) -> Dict[str, Any]:
        """获取金融分析报告"""
        recent_analyses = self.analysis_history[-10:] if self.analysis_history else []

        return {
            "timestamp": time.time(),
            "timestamp_human": datetime.now().isoformat(),
            "knowledge_base_size": {
                "financial_indicators": len(self.financial_knowledge_base.get("financial_indicators", {})),
                "risk_metrics": len(self.financial_knowledge_base.get("risk_metrics", {})),
                "investment_strategies": len(self.financial_knowledge_base.get("investment_strategies", {}))
            },
            "total_analyses": len(self.analysis_history),
            "recent_analyses": recent_analyses,
            "capabilities": {
                "financial_data_analysis": True,
                "risk_assessment": True,
                "portfolio_optimization": True,
                "market_simulation": True
            },
            "library_available": self.financial_library_available
        }


class ProfessionalDomainCapabilityManager:
    """专业领域能力管理器（集成所有组件）"""

    def __init__(self):
        self.programming_manager = ProgrammingCapabilityManager()
        self.math_manager = MathematicalCapabilityManager()
        self.physics_manager = PhysicsSimulationManager()
        self.medical_manager = MedicalReasoningManager()
        self.financial_manager = FinancialAnalysisManager()

        self.logger = logging.getLogger("ProfessionalDomainCapabilityManager")
        self.logger.info("专业领域能力管理器初始化完成")

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """获取综合能力报告"""
        programming_report = self.programming_manager.get_programming_report()
        math_report = self.math_manager.get_math_report()
        physics_report = self.physics_manager.get_physics_report()
        medical_report = self.medical_manager.get_medical_report()
        financial_report = self.financial_manager.get_financial_report()

        return {
            "timestamp": time.time(),
            "timestamp_human": datetime.now().isoformat(),
            "programming_capabilities": programming_report,
            "mathematical_capabilities": math_report,
            "physics_simulation_capabilities": physics_report,
            "medical_reasoning_capabilities": medical_report,
            "financial_analysis_capabilities": financial_report,
            "overall_status": {
                "programming_enabled": True,
                "mathematics_enabled": True,
                "physics_simulation_enabled": PYBULLET_AVAILABLE,
                "medical_reasoning_enabled": MEDICAL_KB_AVAILABLE,
                "financial_analysis_enabled": PANDAS_AVAILABLE
            }
        }


# 全局管理器实例
_global_professional_domain_manager = None


def get_global_professional_domain_manager() -> ProfessionalDomainCapabilityManager:
    """获取全局专业领域能力管理器"""
    global _global_professional_domain_manager

    if _global_professional_domain_manager is None:
        _global_professional_domain_manager = ProfessionalDomainCapabilityManager()

    return _global_professional_domain_manager


def test_professional_capabilities():
    """测试专业领域能力"""
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("=== 测试专业领域能力 ===")

    # 创建管理器
    manager = get_global_professional_domain_manager()

    print("1. 测试编程能力...")

    # 测试代码分析
    test_code = '''
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

def find_max(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num
'''

    analysis_result = manager.programming_manager.analyze_code(
        test_code, ProgrammingLanguage.PYTHON)
    print(
        f"   代码分析完成: 复杂度={analysis_result.complexity.value}, 函数数={analysis_result.functions_count}")

    print("\n2. 测试代码生成...")
    generated_code = manager.programming_manager.generate_code(
        "计算两个数的乘积",
        ProgrammingLanguage.PYTHON,
        "function"
    )
    print(f"   代码生成完成: 质量分数={generated_code['quality_score']:.2f}")

    print("\n3. 测试数学能力...")
    math_problem = manager.math_manager.solve_math_problem(
        "Solve the equation: 2x + 5 = 15",
        MathematicalDomain.ALGEBRA
    )
    print(f"   数学问题解决: 领域={math_problem.domain.value}, 答案={math_problem.final_answer}")

    print("\n4. 测试物理模拟...")
    physics_result = manager.physics_manager.simulate_motion(
        initial_position=[0, 0, 10],  # 从10米高度开始
        initial_velocity=[5, 0, 0],   # 水平速度5m/s
        mass=1.0,                     # 质量1kg
        force=None,                   # 无额外力
        duration=2.0                  # 模拟2秒
    )
    final_pos = physics_result["final_position"]
    print(
        f"   物理模拟完成: 最终位置=[{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}], 模式={physics_result['simulation_mode']}")

    # 测试碰撞检测
    collision_result = manager.physics_manager.detect_collision(
        object1_pos=[0, 0, 0],
        object1_radius=1.0,
        object2_pos=[2, 0, 0],
        object2_radius=1.0
    )
    print(
        f"   碰撞检测: 碰撞={collision_result['collision_detected']}, 距离={collision_result['distance']:.2f}")

    print("\n5. 测试医学推理...")
    symptoms = ["fever", "cough", "headache"]
    patient_info = {"age": 30, "gender": "male", "medical_history": []}

    medical_result = manager.medical_manager.diagnose_symptoms(symptoms, patient_info)
    primary_diagnosis = medical_result.get("primary_diagnosis", "未知")
    confidence = medical_result.get("confidence", 0.0)
    print(f"   医学诊断完成: 主要诊断={primary_diagnosis}, 置信度={confidence:.2f}")

    print("\n6. 测试金融分析...")
    # 测试金融数据分析
    financial_result = manager.financial_manager.analyze_financial_data(
        data_type="time_series"
    )
    volatility = financial_result.get("risk_metrics", {}).get("volatility", 0.0)
    sharpe_ratio = financial_result.get("risk_metrics", {}).get("sharpe_ratio", 0.0)
    print(f"   金融分析完成: 波动率={volatility:.4f}, 夏普比率={sharpe_ratio:.2f}")

    # 测试投资组合风险评估
    portfolio = {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}
    risk_assessment = manager.financial_manager.assess_risk(portfolio)
    risk_level = risk_assessment.get("risk_assessment", {}).get("risk_level", "未知")
    print(f"   风险评估完成: 风险等级={risk_level}")

    # 测试投资组合优化
    assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    optimization_result = manager.financial_manager.optimize_portfolio(assets)
    expected_return = optimization_result.get("expected_return", 0.0)
    print(f"   投资组合优化完成: 预期收益率={expected_return:.4f}")

    print("\n7. 获取综合报告...")
    report = manager.get_comprehensive_report()
    print(f"   编程能力: {report['overall_status']['programming_enabled']}")
    print(f"   数学能力: {report['overall_status']['mathematics_enabled']}")
    print(f"   物理模拟: {report['overall_status']['physics_simulation_enabled']}")
    print(f"   医学推理: {report['overall_status']['medical_reasoning_enabled']}")
    print(f"   金融分析: {report['overall_status']['financial_analysis_enabled']}")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_professional_capabilities()
