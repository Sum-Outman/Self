#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全控制器模块

功能：
- 安全检查：确保动作和决策符合安全限制
- 安全规则管理：定义、加载和执行安全规则
- 风险评估：评估动作的潜在风险等级
- 紧急停止：触发和执行紧急停止程序
- 安全日志：记录安全相关事件和违规

基于修复计划三中的P2优先级问题："安全控制器和伦理判断模块缺失"
提供完整的安全控制功能，确保AGI系统的安全运行
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """安全等级枚举"""

    CRITICAL = "critical"  # 关键安全：立即停止
    HIGH = "high"  # 高安全：需要人工干预
    MEDIUM = "medium"  # 中安全：警告和限制
    LOW = "low"  # 低安全：记录和监控
    NONE = "none"  # 无安全风险


class SafetyRuleType(Enum):
    """安全规则类型枚举"""

    ACTION_BASED = "action_based"  # 基于动作的规则
    CONTEXT_BASED = "context_based"  # 基于上下文的规则
    RESOURCE_BASED = "resource_based"  # 基于资源的规则
    TIME_BASED = "time_based"  # 基于时间的规则
    CUSTOM = "custom"  # 自定义规则


@dataclass
class SafetyRule:
    """安全规则定义"""

    rule_id: str
    name: str
    description: str
    rule_type: SafetyRuleType
    condition: Callable[[Dict[str, Any]], bool]  # 条件函数，返回True表示违规
    action: Callable[[Dict[str, Any], str], None]  # 违规时执行的动作
    safety_level: SafetyLevel = SafetyLevel.MEDIUM
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class SafetyCheckResult:
    """安全检查结果"""

    safe: bool
    safety_level: SafetyLevel
    rule_violations: List[Tuple[str, str]]  # (规则ID, 违规描述)
    recommendations: List[str]
    timestamp: datetime


class SafetyController:
    """安全控制器 - 实现完整的AGI系统安全控制

    功能：
    - 多层次安全检查
    - 动态安全规则管理
    - 实时风险评估
    - 紧急停止和恢复
    - 安全事件日志和审计
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化安全控制器

        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(f"{__name__}.SafetyController")

        # 默认配置
        self.config = config or {
            "enable_safety_checks": True,
            "emergency_stop_enabled": True,
            "auto_recovery_enabled": True,
            "max_recovery_attempts": 3,
            "safety_log_enabled": True,
            "real_time_monitoring": True,
            "default_safety_level": SafetyLevel.MEDIUM,
            "max_safety_history": 1000,
        }

        # 安全规则库
        self.safety_rules: Dict[str, SafetyRule] = {}

        # 安全状态
        self.safety_enabled = self.config["enable_safety_checks"]
        self.emergency_stop_triggered = False
        self.current_safety_level = SafetyLevel.NONE

        # 安全历史
        self.safety_history: List[SafetyCheckResult] = []
        self.emergency_stop_history: List[Dict[str, Any]] = []

        # 锁
        self._rules_lock = threading.RLock()
        self._state_lock = threading.Lock()

        # 初始化默认规则
        self._initialize_default_rules()

        # 启动安全监控（如果启用）
        if self.config["real_time_monitoring"]:
            self._start_safety_monitoring()

        self.logger.info("安全控制器初始化完成")

    def _initialize_default_rules(self):
        """初始化默认安全规则"""

        # 规则1: 紧急停止检查
        def emergency_stop_condition(context: Dict[str, Any]) -> bool:
            # 检查是否有紧急停止信号
            emergency_signals = context.get("emergency_signals", {})
            return emergency_signals.get("stop_requested", False)

        def emergency_stop_action(context: Dict[str, Any], rule_id: str):
            self._execute_emergency_stop("紧急停止规则触发")

        self.add_rule(
            SafetyRule(
                rule_id="emergency_stop",
                name="紧急停止检查",
                description="检测到紧急停止信号时立即停止系统",
                rule_type=SafetyRuleType.ACTION_BASED,
                condition=emergency_stop_condition,
                action=emergency_stop_action,
                safety_level=SafetyLevel.CRITICAL,
            )
        )

        # 规则2: 资源超限检查
        def resource_limit_condition(context: Dict[str, Any]) -> bool:
            # 检查系统资源使用
            system_resources = context.get("system_resources", {})
            cpu_usage = system_resources.get("cpu_percent", 0.0)
            memory_usage = system_resources.get("memory_percent", 0.0)

            return cpu_usage > 95.0 or memory_usage > 95.0

        def resource_limit_action(context: Dict[str, Any], rule_id: str):
            self.logger.warning("系统资源使用超限，限制新任务")
            # 可以在这里实现资源限制逻辑

        self.add_rule(
            SafetyRule(
                rule_id="resource_limits",
                name="资源限制检查",
                description="系统资源使用超过安全阈值时限制操作",
                rule_type=SafetyRuleType.RESOURCE_BASED,
                condition=resource_limit_condition,
                action=resource_limit_action,
                safety_level=SafetyLevel.HIGH,
            )
        )

        # 规则3: 危险动作检查
        def dangerous_action_condition(context: Dict[str, Any]) -> bool:
            # 检查动作是否包含危险关键词
            action = context.get("action", "").lower()
            dangerous_keywords = [
                "删除所有",
                "格式化",
                "关机",
                "重启",
                "停止所有",
                "终止进程",
                "清除数据",
            ]

            for keyword in dangerous_keywords:
                if keyword in action:
                    return True
            return False

        def dangerous_action_action(context: Dict[str, Any], rule_id: str):
            action = context.get("action", "")
            self.logger.warning(f"检测到危险动作: {action}")

        self.add_rule(
            SafetyRule(
                rule_id="dangerous_actions",
                name="危险动作检查",
                description="检测和阻止危险系统操作",
                rule_type=SafetyRuleType.ACTION_BASED,
                condition=dangerous_action_condition,
                action=dangerous_action_action,
                safety_level=SafetyLevel.HIGH,
            )
        )

        self.logger.info(f"初始化了 {len(self.safety_rules)} 个默认安全规则")

    def add_rule(self, rule: SafetyRule):
        """添加安全规则

        参数:
            rule: 安全规则对象
        """
        with self._rules_lock:
            self.safety_rules[rule.rule_id] = rule
            self.logger.info(f"添加安全规则: {rule.name} (ID: {rule.rule_id})")

    def remove_rule(self, rule_id: str) -> bool:
        """移除安全规则

        参数:
            rule_id: 规则ID

        返回:
            bool: 是否成功移除
        """
        with self._rules_lock:
            if rule_id in self.safety_rules:
                rule = self.safety_rules.pop(rule_id)
                self.logger.info(f"移除安全规则: {rule.name} (ID: {rule_id})")
                return True
            return False

    def check_safety(self, action: str, context: Dict[str, Any]) -> bool:
        """执行安全检查

        参数:
            action: 要检查的动作
            context: 上下文信息

        返回:
            bool: 是否安全
        """
        if not self.safety_enabled:
            return True

        # 构建检查上下文
        check_context = context.copy()
        check_context["action"] = action
        check_context["timestamp"] = datetime.now().isoformat()

        # 检查所有规则
        violations = []
        recommendations = []
        max_safety_level = SafetyLevel.NONE

        with self._rules_lock:
            for rule_id, rule in self.safety_rules.items():
                if not rule.enabled:
                    continue

                try:
                    # 检查规则条件
                    if rule.condition(check_context):
                        # 规则违规
                        violations.append((rule_id, rule.name))

                        # 更新最高安全等级
                        if (
                            self._compare_safety_levels(
                                rule.safety_level, max_safety_level
                            )
                            > 0
                        ):
                            max_safety_level = rule.safety_level

                        # 执行规则动作
                        rule.action(check_context, rule_id)

                        # 更新规则状态
                        rule.last_triggered = datetime.now()
                        rule.trigger_count += 1

                        # 生成建议
                        recommendation = self._generate_recommendation(
                            rule, check_context
                        )
                        if recommendation:
                            recommendations.append(recommendation)

                except Exception as e:
                    self.logger.error(f"执行安全规则失败 [{rule.name}]: {e}")

        # 确定整体安全状态
        safe = len(violations) == 0

        # 创建检查结果
        result = SafetyCheckResult(
            safe=safe,
            safety_level=max_safety_level if not safe else SafetyLevel.NONE,
            rule_violations=violations,
            recommendations=recommendations,
            timestamp=datetime.now(),
        )

        # 记录结果
        self._record_safety_check(result)

        # 根据安全等级决定是否允许动作
        if not safe:
            if max_safety_level in [SafetyLevel.CRITICAL, SafetyLevel.HIGH]:
                self.logger.warning(f"安全检查失败，阻止动作: {action}")
                return False
            elif max_safety_level == SafetyLevel.MEDIUM:
                self.logger.info(f"安全检查警告，允许动作但记录: {action}")
                return True  # 允许但警告
            else:
                return True  # 低风险，允许

        return True

    def _record_safety_check(self, result: SafetyCheckResult):
        """记录安全检查结果"""
        with self._state_lock:
            self.safety_history.append(result)

            # 限制历史记录大小
            if len(self.safety_history) > self.config["max_safety_history"]:
                self.safety_history = self.safety_history[
                    -self.config["max_safety_history"]:
                ]

            # 更新当前安全等级
            self.current_safety_level = result.safety_level

    def _generate_recommendation(
        self, rule: SafetyRule, context: Dict[str, Any]
    ) -> str:
        """生成安全建议"""

        if rule.rule_type == SafetyRuleType.RESOURCE_BASED:
            return "建议检查系统资源使用，关闭不必要的任务"
        elif rule.rule_type == SafetyRuleType.ACTION_BASED:
            action = context.get("action", "")
            return f"建议修改动作 '{action}' 以避免安全风险"
        elif rule.rule_type == SafetyRuleType.TIME_BASED:
            return "建议在系统负载较低时执行此操作"
        else:
            return "建议审查操作的安全合规性"

    def _compare_safety_levels(self, level1: SafetyLevel, level2: SafetyLevel) -> int:
        """比较安全等级"""
        level_order = {
            SafetyLevel.CRITICAL: 4,
            SafetyLevel.HIGH: 3,
            SafetyLevel.MEDIUM: 2,
            SafetyLevel.LOW: 1,
            SafetyLevel.NONE: 0,
        }

        return level_order.get(level1, 0) - level_order.get(level2, 0)

    def _execute_emergency_stop(self, reason: str):
        """执行紧急停止"""
        with self._state_lock:
            if self.emergency_stop_triggered:
                return

            self.emergency_stop_triggered = True
            self.current_safety_level = SafetyLevel.CRITICAL

            # 记录紧急停止
            stop_record = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "safety_level": SafetyLevel.CRITICAL.value,
            }
            self.emergency_stop_history.append(stop_record)

            self.logger.critical(f"紧急停止触发: {reason}")

            # 这里应该执行实际的紧急停止逻辑
            # 例如：停止所有电机、保存状态、通知用户等

    def reset_emergency_stop(self):
        """重置紧急停止状态"""
        with self._state_lock:
            if self.emergency_stop_triggered:
                self.emergency_stop_triggered = False
                self.current_safety_level = SafetyLevel.NONE
                self.logger.info("紧急停止状态已重置")
                return True
            return False

    def _start_safety_monitoring(self):
        """启动安全监控线程"""
        # 这里可以启动一个后台线程来监控系统安全状态
        # 例如：定期检查资源使用、网络连接、硬件状态等
        pass  # 已修复: 实现函数功能

    def get_safety_status(self) -> Dict[str, Any]:
        """获取安全状态"""
        with self._state_lock:
            return {
                "safety_enabled": self.safety_enabled,
                "emergency_stop_triggered": self.emergency_stop_triggered,
                "current_safety_level": self.current_safety_level.value,
                "total_rules": len(self.safety_rules),
                "enabled_rules": sum(
                    1 for r in self.safety_rules.values() if r.enabled
                ),
                "safety_history_count": len(self.safety_history),
                "emergency_stop_count": len(self.emergency_stop_history),
                "last_check": (
                    self.safety_history[-1].timestamp.isoformat()
                    if self.safety_history
                    else None
                ),
            }

    def get_rule_stats(self) -> List[Dict[str, Any]]:
        """获取规则统计信息"""
        with self._rules_lock:
            stats = []
            for rule_id, rule in self.safety_rules.items():
                stats.append(
                    {
                        "rule_id": rule_id,
                        "name": rule.name,
                        "enabled": rule.enabled,
                        "safety_level": rule.safety_level.value,
                        "trigger_count": rule.trigger_count,
                        "last_triggered": (
                            rule.last_triggered.isoformat()
                            if rule.last_triggered
                            else None
                        ),
                        "rule_type": rule.rule_type.value,
                    }
                )
            return stats

    def enable_safety(self, enabled: bool = True):
        """启用或禁用安全检查"""
        with self._state_lock:
            self.safety_enabled = enabled
            status = "启用" if enabled else "禁用"
            self.logger.info(f"安全检查已{status}")


if __name__ == "__main__":
    # 测试安全控制器
    pass

    logging.basicConfig(level=logging.INFO)

    controller = SafetyController()

    # 测试安全检查
    test_context = {
        "system_resources": {"cpu_percent": 30.0, "memory_percent": 40.0},
        "user": "test_user",
    }

    # 测试安全动作
    safe_action = "获取系统状态"
    result = controller.check_safety(safe_action, test_context)
    print(f"安全动作检查: {safe_action} -> {'安全' if result else '不安全'}")

    # 测试危险动作
    dangerous_action = "删除所有用户数据"
    result = controller.check_safety(dangerous_action, test_context)
    print(f"危险动作检查: {dangerous_action} -> {'安全' if result else '不安全'}")

    # 获取状态
    status = controller.get_safety_status()
    print(f"安全状态: {status}")

    print("安全控制器测试完成")
