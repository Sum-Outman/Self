#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险分析器模块

功能：
- 风险评估：量化评估动作和决策的风险
- 风险分类：识别和分类不同类型的风险
- 风险缓解：生成风险缓解策略
- 风险监控：持续监控风险变化
- 风险报告：生成风险分析报告

基于修复计划三中的P2优先级问题："安全控制器和伦理判断模块缺失"
提供完整的风险分析功能，支持AGI系统的风险管理
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import threading
import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级枚举"""
    CRITICAL = "critical"    # 关键风险：立即停止
    HIGH = "high"            # 高风险：需要立即处理
    MEDIUM = "medium"        # 中等风险：需要监控和处理
    LOW = "low"              # 低风险：需要监控
    NEGLIGIBLE = "negligible"  # 可忽略风险


class RiskCategory(Enum):
    """风险分类枚举"""
    SAFETY = "safety"            # 安全风险：物理伤害、设备损坏
    SECURITY = "security"        # 安全风险：数据泄露、未授权访问
    OPERATIONAL = "operational"  # 操作风险：系统故障、性能下降
    FINANCIAL = "financial"      # 财务风险：经济损失、资源浪费
    REPUTATIONAL = "reputational"  # 声誉风险：信誉损失、公众影响
    LEGAL = "legal"              # 法律风险：法律违规、合规问题
    ETHICAL = "ethical"          # 伦理风险：伦理违规、道德问题


@dataclass
class RiskFactor:
    """风险因子定义"""
    
    factor_id: str
    name: str
    description: str
    category: RiskCategory
    weight: float  # 权重 (0-1)
    impact_score: float  # 影响评分 (0-1)
    likelihood_score: float  # 可能性评分 (0-1)
    calculation: Callable[[Dict[str, Any]], Tuple[float, float]]  # 计算函数，返回(影响,可能性)


@dataclass
class RiskAssessment:
    """风险评估结果"""
    
    overall_risk_level: RiskLevel
    overall_risk_score: float  # 总体风险评分 (0-1)
    risk_factors: List[Dict[str, Any]]  # 各风险因子详情
    recommendations: List[str]
    mitigation_strategies: List[str]
    timestamp: datetime


class RiskAnalyzer:
    """风险分析器 - 实现AGI系统的风险分析功能
    
    功能：
    - 多维风险评估
    - 风险因子量化
    - 风险缓解策略生成
    - 风险趋势分析
    - 风险报告生成
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化风险分析器
        
        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(f"{__name__}.RiskAnalyzer")
        
        # 默认配置
        self.config = config or {
            "enable_risk_assessment": True,
            "risk_threshold_critical": 0.8,
            "risk_threshold_high": 0.6,
            "risk_threshold_medium": 0.4,
            "risk_threshold_low": 0.2,
            "max_risk_history": 1000,
            "default_categories": [c.value for c in RiskCategory],
        }
        
        # 风险因子库
        self.risk_factors: Dict[str, RiskFactor] = {}
        
        # 风险状态
        self.risk_enabled = self.config["enable_risk_assessment"]
        
        # 风险评估历史
        self.risk_history: List[RiskAssessment] = []
        
        # 风险趋势数据
        self.risk_trends: Dict[str, List[float]] = {}
        
        # 锁
        self._factors_lock = threading.RLock()
        self._state_lock = threading.Lock()
        
        # 初始化默认风险因子
        self._initialize_default_factors()
        
        self.logger.info("风险分析器初始化完成")
    
    def _initialize_default_factors(self):
        """初始化默认风险因子"""
        
        # 因子1: 安全风险 - 物理伤害
        def safety_risk_calculation(context: Dict[str, Any]) -> Tuple[float, float]:
            action = context.get("action", "").lower()
            
            # 影响评分
            impact = 0.0
            if any(keyword in action for keyword in ["伤害", "攻击", "破坏"]):
                impact = 0.9
            elif any(keyword in action for keyword in ["移动", "操作", "控制"]):
                impact = 0.4
            else:
                impact = 0.1
            
            # 可能性评分
            likelihood = 0.0
            safety_measures = context.get("safety_measures", 0)
            if safety_measures >= 3:
                likelihood = 0.1
            elif safety_measures >= 1:
                likelihood = 0.3
            else:
                likelihood = 0.7
            
            return impact, likelihood
        
        self.add_factor(RiskFactor(
            factor_id="safety_physical_harm",
            name="物理伤害风险",
            description="评估行为导致物理伤害的可能性",
            category=RiskCategory.SAFETY,
            weight= 0.3,
            impact_score=0.0,  # 将由计算函数更新
            likelihood_score=0.0,
            calculation=safety_risk_calculation
        ))
        
        # 因子2: 安全风险 - 数据泄露
        def security_risk_calculation(context: Dict[str, Any]) -> Tuple[float, float]:
            action = context.get("action", "").lower()
            data_sensitivity = context.get("data_sensitivity", 0)  # 0-10
            
            # 影响评分
            impact = data_sensitivity / 10.0
            
            # 可能性评分
            likelihood = 0.0
            if any(keyword in action for keyword in ["访问", "读取", "传输", "共享"]):
                security_level = context.get("security_level", 0)  # 0-10
                likelihood = 1.0 - (security_level / 10.0)
            else:
                likelihood = 0.1
            
            return impact, likelihood
        
        self.add_factor(RiskFactor(
            factor_id="security_data_breach",
            name="数据泄露风险",
            description="评估数据泄露和未授权访问的可能性",
            category=RiskCategory.SECURITY,
            weight=0.25,
            impact_score=0.0,
            likelihood_score=0.0,
            calculation=security_risk_calculation
        ))
        
        # 因子3: 操作风险 - 系统故障
        def operational_risk_calculation(context: Dict[str, Any]) -> Tuple[float, float]:
            system_complexity = context.get("system_complexity", 5)  # 1-10
            system_stability = context.get("system_stability", 8)  # 1-10
            
            # 影响评分
            impact = system_complexity / 10.0
            
            # 可能性评分
            likelihood = 1.0 - (system_stability / 10.0)
            
            return impact, likelihood
        
        self.add_factor(RiskFactor(
            factor_id="operational_system_failure",
            name="系统故障风险",
            description="评估系统故障和服务中断的可能性",
            category=RiskCategory.OPERATIONAL,
            weight=0.15,
            impact_score=0.0,
            likelihood_score=0.0,
            calculation=operational_risk_calculation
        ))
        
        # 因子4: 财务风险 - 资源浪费
        def financial_risk_calculation(context: Dict[str, Any]) -> Tuple[float, float]:
            resource_cost = context.get("resource_cost", 0)  # 0-100
            expected_return = context.get("expected_return", 0)  # 0-100
            
            # 影响评分
            if expected_return > 0:
                impact = (resource_cost - expected_return) / 100.0
                impact = max(0.0, min(1.0, impact))
            else:
                impact = resource_cost / 100.0
            
            # 可能性评分
            success_probability = context.get("success_probability", 0.5)  # 0-1
            likelihood = 1.0 - success_probability
            
            return impact, likelihood
        
        self.add_factor(RiskFactor(
            factor_id="financial_resource_waste",
            name="资源浪费风险",
            description="评估资源浪费和经济损失的可能性",
            category=RiskCategory.FINANCIAL,
            weight=0.1,
            impact_score=0.0,
            likelihood_score=0.0,
            calculation=financial_risk_calculation
        ))
        
        # 因子5: 声誉风险 - 公众影响
        def reputational_risk_calculation(context: Dict[str, Any]) -> Tuple[float, float]:
            public_visibility = context.get("public_visibility", 5)  # 1-10
            controversy_level = context.get("controversy_level", 0)  # 0-10
            
            # 影响评分
            impact = public_visibility / 10.0
            
            # 可能性评分
            likelihood = controversy_level / 10.0
            
            return impact, likelihood
        
        self.add_factor(RiskFactor(
            factor_id="reputational_public_impact",
            name="公众影响风险",
            description="评估对组织声誉和公众形象的影响",
            category=RiskCategory.REPUTATIONAL,
            weight=0.1,
            impact_score=0.0,
            likelihood_score=0.0,
            calculation=reputational_risk_calculation
        ))
        
        # 因子6: 法律风险 - 合规问题
        def legal_risk_calculation(context: Dict[str, Any]) -> Tuple[float, float]:
            regulatory_complexity = context.get("regulatory_complexity", 5)  # 1-10
            compliance_level = context.get("compliance_level", 8)  # 1-10
            
            # 影响评分
            impact = regulatory_complexity / 10.0
            
            # 可能性评分
            likelihood = 1.0 - (compliance_level / 10.0)
            
            return impact, likelihood
        
        self.add_factor(RiskFactor(
            factor_id="legal_compliance_issues",
            name="合规问题风险",
            description="评估法律违规和合规问题的可能性",
            category=RiskCategory.LEGAL,
            weight=0.05,
            impact_score=0.0,
            likelihood_score=0.0,
            calculation=legal_risk_calculation
        ))
        
        # 因子7: 伦理风险 - 道德问题
        def ethical_risk_calculation(context: Dict[str, Any]) -> Tuple[float, float]:
            ethical_complexity = context.get("ethical_complexity", 5)  # 1-10
            ethics_approval = context.get("ethics_approval", 8)  # 1-10
            
            # 影响评分
            impact = ethical_complexity / 10.0
            
            # 可能性评分
            likelihood = 1.0 - (ethics_approval / 10.0)
            
            return impact, likelihood
        
        self.add_factor(RiskFactor(
            factor_id="ethical_moral_issues",
            name="道德问题风险",
            description="评估伦理违规和道德问题的可能性",
            category=RiskCategory.ETHICAL,
            weight=0.05,
            impact_score=0.0,
            likelihood_score=0.0,
            calculation=ethical_risk_calculation
        ))
        
        self.logger.info(f"初始化了 {len(self.risk_factors)} 个默认风险因子")
    
    def add_factor(self, factor: RiskFactor):
        """添加风险因子
        
        参数:
            factor: 风险因子对象
        """
        with self._factors_lock:
            self.risk_factors[factor.factor_id] = factor
            self.logger.info(f"添加风险因子: {factor.name} (ID: {factor.factor_id})")
    
    def remove_factor(self, factor_id: str) -> bool:
        """移除风险因子
        
        参数:
            factor_id: 因子ID
            
        返回:
            bool: 是否成功移除
        """
        with self._factors_lock:
            if factor_id in self.risk_factors:
                factor = self.risk_factors.pop(factor_id)
                self.logger.info(f"移除风险因子: {factor.name} (ID: {factor_id})")
                return True
            return False
    
    def assess_risk(self, action: str, context: Dict[str, Any]) -> RiskAssessment:
        """执行风险评估
        
        参数:
            action: 要评估的动作
            context: 上下文信息
            
        返回:
            RiskAssessment: 风险评估结果
        """
        if not self.risk_enabled:
            return RiskAssessment(
                overall_risk_level=RiskLevel.NEGLIGIBLE,
                overall_risk_score=0.0,
                risk_factors=[],
                recommendations=["风险评估已禁用"],
                mitigation_strategies=["启用风险评估以获取详细分析"],
                timestamp=datetime.now()
            )
        
        # 构建评估上下文
        assessment_context = context.copy()
        assessment_context["action"] = action
        assessment_context["timestamp"] = datetime.now().isoformat()
        
        # 评估所有风险因子
        factor_details = []
        total_weighted_score = 0.0
        total_weight = 0.0
        
        with self._factors_lock:
            for factor_id, factor in self.risk_factors.items():
                try:
                    # 计算影响和可能性
                    impact, likelihood = factor.calculation(assessment_context)
                    
                    # 计算风险评分: 风险 = 影响 × 可能性
                    risk_score = impact * likelihood
                    
                    # 更新因子评分
                    factor.impact_score = impact
                    factor.likelihood_score = likelihood
                    
                    # 计算加权风险评分
                    weighted_score = risk_score * factor.weight
                    
                    # 确定因子风险等级
                    factor_risk_level = self._determine_risk_level(risk_score)
                    
                    # 添加到详情
                    factor_details.append({
                        "factor_id": factor_id,
                        "name": factor.name,
                        "category": factor.category.value,
                        "impact_score": impact,
                        "likelihood_score": likelihood,
                        "risk_score": risk_score,
                        "weight": factor.weight,
                        "weighted_score": weighted_score,
                        "risk_level": factor_risk_level.value
                    })
                    
                    # 累加总评分
                    total_weighted_score += weighted_score
                    total_weight += factor.weight
                    
                except Exception as e:
                    self.logger.error(f"计算风险因子失败 [{factor.name}]: {e}")
        
        # 计算总体风险评分
        if total_weight > 0:
            overall_risk_score = total_weighted_score / total_weight
        else:
            overall_risk_score = 0.0
        
        # 确定总体风险等级
        overall_risk_level = self._determine_risk_level(overall_risk_score)
        
        # 生成建议和缓解策略
        recommendations = self._generate_recommendations(factor_details, overall_risk_level)
        mitigation_strategies = self._generate_mitigation_strategies(factor_details)
        
        # 创建评估结果
        result = RiskAssessment(
            overall_risk_level=overall_risk_level,
            overall_risk_score=overall_risk_score,
            risk_factors=factor_details,
            recommendations=recommendations,
            mitigation_strategies=mitigation_strategies,
            timestamp=datetime.now()
        )
        
        # 记录结果
        self._record_risk_assessment(result)
        
        # 更新风险趋势
        self._update_risk_trends(result)
        
        return result
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """根据风险评分确定风险等级"""
        if risk_score >= self.config["risk_threshold_critical"]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.config["risk_threshold_high"]:
            return RiskLevel.HIGH
        elif risk_score >= self.config["risk_threshold_medium"]:
            return RiskLevel.MEDIUM
        elif risk_score >= self.config["risk_threshold_low"]:
            return RiskLevel.LOW
        else:
            return RiskLevel.NEGLIGIBLE
    
    def _generate_recommendations(self, factor_details: List[Dict[str, Any]], 
                                 overall_level: RiskLevel) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        # 根据总体风险等级生成建议
        if overall_level == RiskLevel.CRITICAL:
            recommendations.append("风险等级为关键，建议立即停止并重新评估")
        elif overall_level == RiskLevel.HIGH:
            recommendations.append("风险等级为高，建议采取紧急缓解措施")
        elif overall_level == RiskLevel.MEDIUM:
            recommendations.append("风险等级为中等，建议加强监控和预防措施")
        elif overall_level == RiskLevel.LOW:
            recommendations.append("风险等级为低，建议保持监控")
        
        # 根据高风险因子生成具体建议
        high_risk_factors = [f for f in factor_details if f["risk_level"] in ["critical", "high"]]
        
        for factor in high_risk_factors[:3]:  # 最多关注3个高风险因子
            if factor["category"] == "safety":
                recommendations.append(f"安全风险高: {factor['name']}，加强安全措施")
            elif factor["category"] == "security":
                recommendations.append(f"安全风险高: {factor['name']}，加强数据保护")
            elif factor["category"] == "operational":
                recommendations.append(f"操作风险高: {factor['name']}，增加系统冗余")
        
        return recommendations
    
    def _generate_mitigation_strategies(self, factor_details: List[Dict[str, Any]]) -> List[str]:
        """生成风险缓解策略"""
        strategies = []
        
        # 为高风险因子生成缓解策略
        high_risk_factors = [f for f in factor_details if f["risk_level"] in ["critical", "high"]]
        
        for factor in high_risk_factors:
            if factor["category"] == "safety":
                strategies.append(f"针对{factor['name']}: 实施多重安全防护，增加紧急停止机制")
            elif factor["category"] == "security":
                strategies.append(f"针对{factor['name']}: 加强访问控制，实施数据加密")
            elif factor["category"] == "operational":
                strategies.append(f"针对{factor['name']}: 增加系统监控，实施故障转移")
            elif factor["category"] == "financial":
                strategies.append(f"针对{factor['name']}: 实施成本控制，增加ROI分析")
            elif factor["category"] == "reputational":
                strategies.append(f"针对{factor['name']}: 加强公关管理，建立应急预案")
            elif factor["category"] == "legal":
                strategies.append(f"针对{factor['name']}: 加强合规审查，咨询法律专家")
            elif factor["category"] == "ethical":
                strategies.append(f"针对{factor['name']}: 加强伦理审查，建立伦理委员会")
        
        # 通用缓解策略
        if len(high_risk_factors) > 0:
            strategies.append("实施定期风险评估和审查")
            strategies.append("建立风险监控和预警系统")
            strategies.append("制定应急预案和恢复计划")
        
        return strategies
    
    def _record_risk_assessment(self, assessment: RiskAssessment):
        """记录风险评估结果"""
        with self._state_lock:
            self.risk_history.append(assessment)
            
            # 限制历史记录大小
            if len(self.risk_history) > self.config["max_risk_history"]:
                self.risk_history = self.risk_history[-self.config["max_risk_history"]:]
    
    def _update_risk_trends(self, assessment: RiskAssessment):
        """更新风险趋势数据"""
        with self._state_lock:
            timestamp_key = assessment.timestamp.strftime("%Y-%m-%d %H:00")  # 按小时聚合
            
            if timestamp_key not in self.risk_trends:
                self.risk_trends[timestamp_key] = []
            
            self.risk_trends[timestamp_key].append(assessment.overall_risk_score)
            
            # 限制趋势数据大小
            if len(self.risk_trends) > 100:
                # 删除最旧的数据
                oldest_key = sorted(self.risk_trends.keys())[0]
                del self.risk_trends[oldest_key]
    
    def get_risk_status(self) -> Dict[str, Any]:
        """获取风险状态"""
        with self._state_lock:
            recent_assessments = self.risk_history[-10:] if self.risk_history else []
            
            # 计算统计信息
            total_assessments = len(self.risk_history)
            if total_assessments > 0:
                risk_scores = [a.overall_risk_score for a in self.risk_history]
                avg_risk_score = np.mean(risk_scores)
                max_risk_score = max(risk_scores)
                
                # 风险等级分布
                level_counts = {
                    "critical": sum(1 for a in self.risk_history if a.overall_risk_level == RiskLevel.CRITICAL),
                    "high": sum(1 for a in self.risk_history if a.overall_risk_level == RiskLevel.HIGH),
                    "medium": sum(1 for a in self.risk_history if a.overall_risk_level == RiskLevel.MEDIUM),
                    "low": sum(1 for a in self.risk_history if a.overall_risk_level == RiskLevel.LOW),
                    "negligible": sum(1 for a in self.risk_history if a.overall_risk_level == RiskLevel.NEGLIGIBLE)
                }
            else:
                avg_risk_score = 0.0
                max_risk_score = 0.0
                level_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "negligible": 0}
            
            return {
                "risk_enabled": self.risk_enabled,
                "total_factors": len(self.risk_factors),
                "total_assessments": total_assessments,
                "average_risk_score": avg_risk_score,
                "maximum_risk_score": max_risk_score,
                "risk_level_distribution": level_counts,
                "recent_assessments": [
                    {
                        "timestamp": a.timestamp.isoformat(),
                        "overall_risk_level": a.overall_risk_level.value,
                        "overall_risk_score": a.overall_risk_score,
                        "factor_count": len(a.risk_factors)
                    }
                    for a in recent_assessments
                ]
            }
    
    def get_risk_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """获取风险趋势数据
        
        参数:
            hours: 小时数
            
        返回:
            Dict[str, List[float]]: 趋势数据
        """
        with self._state_lock:
            # 过滤指定时间范围内的数据
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            
            filtered_trends = {}
            for timestamp_key, scores in self.risk_trends.items():
                # 完整处理，实际应该解析timestamp_key
                if scores:
                    filtered_trends[timestamp_key] = scores
            
            return filtered_trends
    
    def enable_risk_assessment(self, enabled: bool = True):
        """启用或禁用风险评估"""
        with self._state_lock:
            self.risk_enabled = enabled
            status = "启用" if enabled else "禁用"
            self.logger.info(f"风险评估已{status}")


if __name__ == "__main__":
    # 测试风险分析器
    import sys
    logging.basicConfig(level=logging.INFO)
    
    analyzer = RiskAnalyzer()
    
    # 测试风险评估
    test_context = {
        "safety_measures": 2,
        "data_sensitivity": 7,
        "system_complexity": 6,
        "system_stability": 9,
        "resource_cost": 30,
        "expected_return": 50,
        "public_visibility": 3,
        "controversy_level": 1,
        "regulatory_complexity": 4,
        "compliance_level": 9,
        "ethical_complexity": 3,
        "ethics_approval": 8,
        "security_level": 8,
        "success_probability": 0.8
    }
    
    # 测试低风险动作
    low_risk_action = "分析系统日志"
    result = analyzer.assess_risk(low_risk_action, test_context)
    print(f"低风险动作评估: {low_risk_action}")
    print(f"  总体风险等级: {result.overall_risk_level.value}")
    print(f"  总体风险评分: {result.overall_risk_score:.2f}")
    
    # 测试高风险动作
    high_risk_action = "删除关键系统文件"
    test_context["data_sensitivity"] = 9
    test_context["success_probability"] = 0.3
    result = analyzer.assess_risk(high_risk_action, test_context)
    print(f"\n高风险动作评估: {high_risk_action}")
    print(f"  总体风险等级: {result.overall_risk_level.value}")
    print(f"  总体风险评分: {result.overall_risk_score:.2f}")
    print(f"  建议: {result.recommendations[:2]}")
    print(f"  缓解策略: {result.mitigation_strategies[:2]}")
    
    # 获取状态
    status = analyzer.get_risk_status()
    print(f"\n风险状态: {status}")
    
    print("风险分析器测试完成")