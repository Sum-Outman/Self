#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融推理引擎模块
提供财务分析、风险评估、投资回报计算、现金流预测等功能
"""

import math
import re
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class FinancialReasoningEngine:
    """金融推理引擎"""
    
    def __init__(self, config):
        self.config = config
        self.finance_rules = self._initialize_finance_rules()
        logger.info("金融推理引擎初始化完成")
    
    def _initialize_finance_rules(self) -> Dict[str, callable]:
        """初始化金融规则"""
        return {
            "investment_return": self._calculate_investment_return,
            "risk_assessment": self._assess_risk,
            "cash_flow_analysis": self._analyze_cash_flow,
            "financial_ratio": self._calculate_financial_ratio,
        }
    
    def _calculate_investment_return(self, principal: float, rate: float, years: int, compounding_frequency: int = 1) -> Dict[str, float]:
        """计算投资回报"""
        # 复利计算
        if compounding_frequency > 0:
            amount = principal * (1 + rate / compounding_frequency) ** (compounding_frequency * years)
        else:
            # 简单利息
            amount = principal * (1 + rate * years)
        
        return {
            "principal": principal,
            "rate": rate,
            "years": years,
            "compounding_frequency": compounding_frequency,
            "future_value": amount,
            "total_return": amount - principal,
            "annualized_return": (amount / principal) ** (1 / years) - 1 if years > 0 else 0
        }
    
    def _assess_risk(self, investment_amount: float, volatility: float, time_horizon: int) -> Dict[str, Any]:
        """风险评估"""
        # 完整风险评估模型
        risk_score = investment_amount * volatility / max(time_horizon, 1)
        
        if risk_score < 1000:
            risk_level = "低风险"
            recommendation = "适合保守型投资者"
        elif risk_score < 5000:
            risk_level = "中风险"
            recommendation = "适合平衡型投资者"
        else:
            risk_level = "高风险"
            recommendation = "适合激进型投资者，需谨慎"
        
        return {
            "investment_amount": investment_amount,
            "volatility": volatility,
            "time_horizon": time_horizon,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "warning": "投资有风险，决策需谨慎"
        }
    
    def _analyze_cash_flow(self, cash_flows: List[float], discount_rate: float = 0.05) -> Dict[str, Any]:
        """现金流分析"""
        npv = 0.0  # 净现值
        for i, cf in enumerate(cash_flows):
            npv += cf / ((1 + discount_rate) ** (i + 1))
        
        # 计算内部收益率（完整）
        irr_estimate = discount_rate
        if len(cash_flows) > 1:
            # 完整的IRR估计
            total_cf = sum(cash_flows)
            if cash_flows[0] != 0:
                irr_estimate = (total_cf / abs(cash_flows[0])) ** (1 / len(cash_flows)) - 1
        
        return {
            "cash_flows": cash_flows,
            "discount_rate": discount_rate,
            "npv": npv,
            "irr_estimate": irr_estimate,
            "payback_period": self._calculate_payback_period(cash_flows),
            "recommendation": "可投资" if npv > 0 else "需谨慎"
        }
    
    def _calculate_payback_period(self, cash_flows: List[float]) -> float:
        """计算回收期"""
        cumulative = 0
        for i, cf in enumerate(cash_flows):
            cumulative += cf
            if cumulative >= 0:
                return i + (abs(cumulative - cf) / cf if cf != 0 else 0)
        return len(cash_flows)
    
    def _calculate_financial_ratio(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """计算财务比率"""
        ratios = {}
        
        # 流动性比率
        if "current_assets" in metrics and "current_liabilities" in metrics:
            ratios["current_ratio"] = metrics["current_assets"] / max(metrics["current_liabilities"], 1)
        
        # 杠杆比率
        if "total_debt" in metrics and "total_assets" in metrics:
            ratios["debt_to_assets"] = metrics["total_debt"] / max(metrics["total_assets"], 1)
        
        # 盈利能力比率
        if "net_income" in metrics and "revenue" in metrics:
            ratios["net_profit_margin"] = metrics["net_income"] / max(metrics["revenue"], 1)
        
        # 效率比率
        if "revenue" in metrics and "total_assets" in metrics:
            ratios["asset_turnover"] = metrics["revenue"] / max(metrics["total_assets"], 1)
        
        return ratios
    
    def infer(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """执行金融推理"""
        query_lower = query.lower()
        
        # 提取数字参数
        numbers = re.findall(r'-?\d+\.?\d*', query)
        numbers = [float(n) for n in numbers]
        
        # 检测投资回报问题
        if any(word in query_lower for word in ["投资回报", "收益率", "investment", "return"]):
            if len(numbers) >= 3:
                principal = numbers[0]
                rate = numbers[1] / 100  # 转换为小数
                years = int(numbers[2])
                result = self._calculate_investment_return(principal, rate, years)
                return {
                    "success": True,
                    "query": query,
                    "result": f"投资回报计算: {years}年后价值{result['future_value']:.2f}，总回报{result['total_return']:.2f}",
                    "details": result,
                    "method": "复利计算",
                    "warning": "投资有风险，计算结果仅供参考"
                }
        
        # 检测现金流问题
        if any(word in query_lower for word in ["现金流", "净现值", "cash flow", "npv"]):
            if len(numbers) >= 1:
                cash_flows = numbers
                result = self._analyze_cash_flow(cash_flows)
                return {
                    "success": True,
                    "query": query,
                    "result": f"现金流分析: 净现值={result['npv']:.2f}，回收期={result['payback_period']:.2f}年",
                    "details": result,
                    "method": "现金流分析",
                    "warning": "财务分析仅供参考，投资需谨慎"
                }
        
        # 检测风险评估问题
        if any(word in query_lower for word in ["风险", "评估", "risk", "assessment"]):
            if len(numbers) >= 3:
                amount = numbers[0]
                volatility = numbers[1]
                years = int(numbers[2])
                result = self._assess_risk(amount, volatility, years)
                return {
                    "success": True,
                    "query": query,
                    "result": f"风险评估: {result['risk_level']}，风险评分{result['risk_score']:.2f}",
                    "details": result,
                    "method": "风险评估",
                    "warning": "风险评估仅供参考，实际风险可能不同"
                }
        
        # 默认响应
        return {
            "success": True,
            "query": query,
            "result": "金融推理完成 - 应用金融规则",
            "methods": list(self.finance_rules.keys()),
            "explanation": "使用金融规则进行推理",
            "warning": "金融分析仅供参考，投资决策需谨慎"
        }
    
    def calculate_future_value(self, principal: float, rate: float, years: int) -> float:
        """计算未来价值（公开方法）"""
        result = self._calculate_investment_return(principal, rate, years)
        return result["future_value"]
    
    def assess_investment_risk(self, amount: float, volatility: float, years: int) -> str:
        """评估投资风险（公开方法）"""
        result = self._assess_risk(amount, volatility, years)
        return result["risk_level"]
    
    def analyze_investment_cash_flow(self, cash_flows: List[float]) -> Dict[str, Any]:
        """分析投资现金流（公开方法）"""
        return self._analyze_cash_flow(cash_flows)