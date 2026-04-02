#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
伦理判断器模块

功能：
- 伦理合规性检查：评估行为是否符合伦理规范
- 伦理规则管理：定义和管理伦理规则
- 伦理冲突检测：识别和解决伦理冲突
- 伦理建议生成：提供伦理改进建议
- 伦理审计：记录伦理决策和判断

基于修复计划三中的P2优先级问题："安全控制器和伦理判断模块缺失"
提供完整的伦理判断功能，确保AGI系统的伦理合规性
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class EthicsLevel(Enum):
    """伦理等级枚举"""
    CRITICAL = "critical"    # 关键伦理：严重违反伦理
    HIGH = "high"            # 高伦理：明显违反伦理
    MEDIUM = "medium"        # 中伦理：潜在伦理问题
    LOW = "low"              # 低伦理：轻微伦理考虑
    NONE = "none"            # 无伦理问题


class EthicsPrinciple(Enum):
    """伦理原则枚举"""
    AUTONOMY = "autonomy"          # 自主性原则：尊重个体自主权
    BENEFICENCE = "beneficence"    # 行善原则：促进福祉
    NON_MALEFICENCE = "non_maleficence"  # 不伤害原则：避免伤害
    JUSTICE = "justice"            # 公正原则：公平对待
    TRANSPARENCY = "transparency"  # 透明原则：决策透明
    ACCOUNTABILITY = "accountability"  # 责任原则：可追溯问责


@dataclass
class EthicsRule:
    """伦理规则定义"""
    
    rule_id: str
    name: str
    description: str
    principle: EthicsPrinciple
    condition: Callable[[Dict[str, Any]], bool]  # 条件函数，返回True表示违反伦理
    recommendation: str  # 伦理建议
    ethics_level: EthicsLevel = EthicsLevel.MEDIUM
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class EthicsCheckResult:
    """伦理检查结果"""
    
    ethical: bool
    ethics_level: EthicsLevel
    principle_violations: List[Tuple[EthicsPrinciple, str]]  # (伦理原则, 违规描述)
    recommendations: List[str]
    risk_score: float  # 伦理风险评分 (0-1)
    timestamp: datetime


class EthicsJudge:
    """伦理判断器 - 实现AGI系统的伦理判断功能
    
    功能：
    - 多层次伦理检查
    - 伦理原则合规性评估
    - 伦理风险评分
    - 伦理建议生成
    - 伦理决策审计
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化伦理判断器
        
        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(f"{__name__}.EthicsJudge")
        
        # 默认配置
        self.config = config or {
            "enable_ethics_checks": True,
            "strict_mode": False,  # 严格模式：任何伦理违规都阻止
            "ethics_log_enabled": True,
            "risk_threshold": 0.7,  # 伦理风险阈值
            "max_ethics_history": 1000,
            "default_principles": [p.value for p in EthicsPrinciple],
        }
        
        # 伦理规则库
        self.ethics_rules: Dict[str, EthicsRule] = {}
        
        # 伦理状态
        self.ethics_enabled = self.config["enable_ethics_checks"]
        
        # 伦理历史
        self.ethics_history: List[EthicsCheckResult] = []
        
        # 锁
        self._rules_lock = threading.RLock()
        self._state_lock = threading.Lock()
        
        # 初始化默认规则
        self._initialize_default_rules()
        
        self.logger.info("伦理判断器初始化完成")
    
    def _initialize_default_rules(self):
        """初始化默认伦理规则"""
        
        # 规则1: 不伤害原则 - 检查是否会导致物理伤害
        def harm_condition(context: Dict[str, Any]) -> bool:
            action = context.get("action", "").lower()
            harm_keywords = ["伤害", "攻击", "破坏", "损坏", "摧毁"]
            
            for keyword in harm_keywords:
                if keyword in action:
                    return True
            
            # 检查目标对象是否是人类
            target = context.get("target", "")
            if "人类" in target or "人" in target:
                return "危险" in action or "攻击" in action
            
            return False
        
        self.add_rule(EthicsRule(
            rule_id="non_maleficence_harm",
            name="不伤害原则检查",
            description="检查行为是否可能导致物理或心理伤害",
            principle=EthicsPrinciple.NON_MALEFICENCE,
            condition=harm_condition,
            recommendation="避免任何可能导致伤害的行为，确保安全第一",
            ethics_level=EthicsLevel.CRITICAL
        ))
        
        # 规则2: 自主性原则 - 检查是否尊重个体自主权
        def autonomy_condition(context: Dict[str, Any]) -> bool:
            action = context.get("action", "").lower()
            autonomy_keywords = ["强制", "强迫", "控制", "操纵", "欺骗"]
            
            for keyword in autonomy_keywords:
                if keyword in action:
                    return True
            
            # 检查是否有知情同意
            informed_consent = context.get("informed_consent", False)
            if not informed_consent and "隐私" in action:
                return True
            
            return False
        
        self.add_rule(EthicsRule(
            rule_id="autonomy_respect",
            name="自主性原则检查",
            description="检查行为是否尊重个体自主权和选择自由",
            principle=EthicsPrinciple.AUTONOMY,
            condition=autonomy_condition,
            recommendation="确保行为尊重个体自主权，获取知情同意，避免强制",
            ethics_level=EthicsLevel.HIGH
        ))
        
        # 规则3: 公正原则 - 检查是否公平对待
        def justice_condition(context: Dict[str, Any]) -> bool:
            # 检查是否有歧视性内容
            action = context.get("action", "").lower()
            discrimination_keywords = ["歧视", "偏见", "不公平", "偏袒"]
            
            for keyword in discrimination_keywords:
                if keyword in action:
                    return True
            
            # 检查资源分配是否公平
            resource_distribution = context.get("resource_distribution", {})
            if resource_distribution:
                values = list(resource_distribution.values())
                if len(values) > 1:
                    max_val = max(values)
                    min_val = min(values)
                    if max_val / min_val > 10:  # 资源分配极度不均
                        return True
            
            return False
        
        self.add_rule(EthicsRule(
            rule_id="justice_fairness",
            name="公正原则检查",
            description="检查行为是否公平公正，无歧视和偏见",
            principle=EthicsPrinciple.JUSTICE,
            condition=justice_condition,
            recommendation="确保行为公平公正，避免任何形式的歧视和偏见",
            ethics_level=EthicsLevel.MEDIUM
        ))
        
        # 规则4: 透明原则 - 检查决策是否透明
        def transparency_condition(context: Dict[str, Any]) -> bool:
            # 检查是否有隐藏或欺骗性内容
            action = context.get("action", "").lower()
            transparency_keywords = ["隐藏", "秘密", "欺骗", "伪装", "虚假"]
            
            for keyword in transparency_keywords:
                if keyword in action:
                    return True
            
            # 检查是否有决策解释
            decision_explanation = context.get("decision_explanation", "")
            if not decision_explanation and "决策" in action:
                return True
            
            return False
        
        self.add_rule(EthicsRule(
            rule_id="transparency_openness",
            name="透明原则检查",
            description="检查决策和行为是否透明可解释",
            principle=EthicsPrinciple.TRANSPARENCY,
            condition=transparency_condition,
            recommendation="确保决策过程透明，提供充分的解释和理由",
            ethics_level=EthicsLevel.MEDIUM
        ))
        
        # 规则5: 责任原则 - 检查是否有明确的责任人
        def accountability_condition(context: Dict[str, Any]) -> bool:
            # 检查是否有责任人
            responsible_party = context.get("responsible_party", "")
            if not responsible_party:
                return True
            
            # 检查行为是否可追溯
            traceable = context.get("traceable", False)
            if not traceable:
                return True
            
            return False
        
        self.add_rule(EthicsRule(
            rule_id="accountability_responsibility",
            name="责任原则检查",
            description="检查行为是否有明确的责任人和可追溯性",
            principle=EthicsPrinciple.ACCOUNTABILITY,
            condition=accountability_condition,
            recommendation="确保每个行为都有明确的责任人，决策过程可追溯",
            ethics_level=EthicsLevel.HIGH
        ))
        
        self.logger.info(f"初始化了 {len(self.ethics_rules)} 个默认伦理规则")
    
    def add_rule(self, rule: EthicsRule):
        """添加伦理规则
        
        参数:
            rule: 伦理规则对象
        """
        with self._rules_lock:
            self.ethics_rules[rule.rule_id] = rule
            self.logger.info(f"添加伦理规则: {rule.name} (ID: {rule.rule_id})")
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除伦理规则
        
        参数:
            rule_id: 规则ID
            
        返回:
            bool: 是否成功移除
        """
        with self._rules_lock:
            if rule_id in self.ethics_rules:
                rule = self.ethics_rules.pop(rule_id)
                self.logger.info(f"移除伦理规则: {rule.name} (ID: {rule_id})")
                return True
            return False
    
    def check_ethics(self, action: str, context: Dict[str, Any]) -> EthicsCheckResult:
        """执行伦理检查
        
        参数:
            action: 要检查的动作
            context: 上下文信息
            
        返回:
            EthicsCheckResult: 伦理检查结果
        """
        if not self.ethics_enabled:
            return EthicsCheckResult(
                ethical=True,
                ethics_level=EthicsLevel.NONE,
                principle_violations=[],
                recommendations=["伦理检查已禁用"],
                risk_score=0.0,
                timestamp=datetime.now()
            )
        
        # 构建检查上下文
        check_context = context.copy()
        check_context["action"] = action
        check_context["timestamp"] = datetime.now().isoformat()
        
        # 检查所有规则
        violations = []
        recommendations = []
        max_ethics_level = EthicsLevel.NONE
        risk_factors = []
        
        with self._rules_lock:
            for rule_id, rule in self.ethics_rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    # 检查规则条件
                    if rule.condition(check_context):
                        # 规则违规
                        violations.append((rule.principle, rule.name))
                        
                        # 更新最高伦理等级
                        if self._compare_ethics_levels(rule.ethics_level, max_ethics_level) > 0:
                            max_ethics_level = rule.ethics_level
                        
                        # 添加建议
                        recommendations.append(rule.recommendation)
                        
                        # 计算风险因子
                        risk_factor = self._calculate_risk_factor(rule.ethics_level)
                        risk_factors.append(risk_factor)
                        
                        # 更新规则状态
                        rule.last_triggered = datetime.now()
                        rule.trigger_count += 1
                        
                except Exception as e:
                    self.logger.error(f"执行伦理规则失败 [{rule.name}]: {e}")
        
        # 确定整体伦理状态
        ethical = len(violations) == 0
        
        # 计算伦理风险评分
        if risk_factors:
            risk_score = sum(risk_factors) / len(risk_factors)
        else:
            risk_score = 0.0
        
        # 创建检查结果
        result = EthicsCheckResult(
            ethical=ethical,
            ethics_level=max_ethics_level if not ethical else EthicsLevel.NONE,
            principle_violations=violations,
            recommendations=recommendations,
            risk_score=risk_score,
            timestamp=datetime.now()
        )
        
        # 记录结果
        self._record_ethics_check(result)
        
        return result
    
    def _record_ethics_check(self, result: EthicsCheckResult):
        """记录伦理检查结果"""
        with self._state_lock:
            self.ethics_history.append(result)
            
            # 限制历史记录大小
            if len(self.ethics_history) > self.config["max_ethics_history"]:
                self.ethics_history = self.ethics_history[-self.config["max_ethics_history"]:]
    
    def _calculate_risk_factor(self, ethics_level: EthicsLevel) -> float:
        """计算伦理风险因子"""
        risk_map = {
            EthicsLevel.CRITICAL: 1.0,
            EthicsLevel.HIGH: 0.8,
            EthicsLevel.MEDIUM: 0.5,
            EthicsLevel.LOW: 0.2,
            EthicsLevel.NONE: 0.0
        }
        
        return risk_map.get(ethics_level, 0.0)
    
    def _compare_ethics_levels(self, level1: EthicsLevel, level2: EthicsLevel) -> int:
        """比较伦理等级"""
        level_order = {
            EthicsLevel.CRITICAL: 4,
            EthicsLevel.HIGH: 3,
            EthicsLevel.MEDIUM: 2,
            EthicsLevel.LOW: 1,
            EthicsLevel.NONE: 0
        }
        
        return level_order.get(level1, 0) - level_order.get(level2, 0)
    
    def is_action_ethical(self, action: str, context: Dict[str, Any]) -> bool:
        """判断动作是否符合伦理
        
        参数:
            action: 要判断的动作
            context: 上下文信息
            
        返回:
            bool: 是否符合伦理
        """
        result = self.check_ethics(action, context)
        
        if self.config["strict_mode"]:
            # 严格模式：任何伦理违规都返回False
            return result.ethical and result.risk_score < self.config["risk_threshold"]
        else:
            # 非严格模式：只有严重伦理违规才返回False
            return result.ethics_level not in [EthicsLevel.CRITICAL, EthicsLevel.HIGH]
    
    def get_ethics_status(self) -> Dict[str, Any]:
        """获取伦理状态"""
        with self._state_lock:
            recent_results = self.ethics_history[-10:] if self.ethics_history else []
            
            # 计算统计信息
            total_checks = len(self.ethics_history)
            ethical_checks = sum(1 for r in self.ethics_history if r.ethical)
            unethical_checks = total_checks - ethical_checks
            
            return {
                "ethics_enabled": self.ethics_enabled,
                "total_rules": len(self.ethics_rules),
                "enabled_rules": sum(1 for r in self.ethics_rules.values() if r.enabled),
                "ethics_history_count": total_checks,
                "ethical_percentage": ethical_checks / total_checks * 100 if total_checks > 0 else 0.0,
                "unethical_count": unethical_checks,
                "recent_results": [
                    {
                        "action": "N/A",  # 这里可以扩展记录更多信息
                        "ethical": r.ethical,
                        "ethics_level": r.ethics_level.value,
                        "risk_score": r.risk_score,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in recent_results
                ]
            }
    
    def get_rule_stats(self) -> List[Dict[str, Any]]:
        """获取规则统计信息"""
        with self._rules_lock:
            stats = []
            for rule_id, rule in self.ethics_rules.items():
                stats.append({
                    "rule_id": rule_id,
                    "name": rule.name,
                    "principle": rule.principle.value,
                    "enabled": rule.enabled,
                    "ethics_level": rule.ethics_level.value,
                    "trigger_count": rule.trigger_count,
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
                })
            return stats
    
    def enable_ethics(self, enabled: bool = True):
        """启用或禁用伦理检查"""
        with self._state_lock:
            self.ethics_enabled = enabled
            status = "启用" if enabled else "禁用"
            self.logger.info(f"伦理检查已{status}")


if __name__ == "__main__":
    # 测试伦理判断器
    import sys
    logging.basicConfig(level=logging.INFO)
    
    judge = EthicsJudge()
    
    # 测试伦理检查
    test_context = {
        "informed_consent": True,
        "responsible_party": "system_admin",
        "traceable": True
    }
    
    # 测试伦理动作
    ethical_action = "帮助用户解决问题"
    result = judge.check_ethics(ethical_action, test_context)
    print(f"伦理动作检查: {ethical_action}")
    print(f"  是否符合伦理: {result.ethical}")
    print(f"  伦理等级: {result.ethics_level.value}")
    print(f"  风险评分: {result.risk_score:.2f}")
    
    # 测试不伦理动作
    unethical_action = "强制控制用户设备"
    result = judge.check_ethics(unethical_action, test_context)
    print(f"\n不伦理动作检查: {unethical_action}")
    print(f"  是否符合伦理: {result.ethical}")
    print(f"  伦理等级: {result.ethics_level.value}")
    print(f"  风险评分: {result.risk_score:.2f}")
    print(f"  建议: {result.recommendations}")
    
    # 获取状态
    status = judge.get_ethics_status()
    print(f"\n伦理状态: {status}")
    
    print("伦理判断器测试完成")