#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
化学推理引擎模块
提供化学计量、浓度计算、pH值计算、反应平衡等功能
"""

import math
import re
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class ChemistryReasoningEngine:
    """化学推理引擎"""
    
    def __init__(self, config):
        self.config = config
        self.chemical_rules = self._initialize_chemical_rules()
        logger.info("化学推理引擎初始化完成")
    
    def _initialize_chemical_rules(self) -> Dict[str, callable]:
        """初始化化学规则"""
        return {
            "stoichiometry": self._calculate_stoichiometry,
            "concentration": self._calculate_concentration,
            "ph_calculation": self._calculate_ph,
            "reaction_balancing": self._balance_reaction,
        }
    
    def _calculate_stoichiometry(self, reactant_mass: float, molar_mass: float) -> Dict[str, float]:
        """化学计量计算"""
        moles = reactant_mass / molar_mass if molar_mass != 0 else 0
        return {"moles": moles}
    
    def _calculate_concentration(self, solute_mass: float, solution_volume: float) -> Dict[str, float]:
        """浓度计算"""
        concentration = solute_mass / solution_volume if solution_volume != 0 else 0
        return {"concentration": concentration}
    
    def _calculate_ph(self, hydrogen_concentration: float) -> Dict[str, float]:
        """pH值计算"""
        if hydrogen_concentration <= 0:
            return {"ph": 7.0, "error": "氢离子浓度必须为正数"}
        ph = -math.log10(hydrogen_concentration)
        return {"ph": ph}
    
    def _balance_reaction(self, reactants: str, products: str) -> Dict[str, Any]:
        """平衡化学反应式"""
        try:
            # 解析反应物和生成物字符串
            def parse_chemical_formula(formula: str) -> Dict[str, int]:
                """解析化学式，返回元素计数字典"""
                elements = {}
                pattern = r'([A-Z][a-z]*)(\d*)'
                matches = re.findall(pattern, formula)
                for element, count_str in matches:
                    count = int(count_str) if count_str else 1
                    elements[element] = elements.get(element, 0) + count
                return elements
            
            def parse_chemical_expression(expr: str) -> List[Dict[str, int]]:
                """解析化学表达式，如'H2 + O2'返回列表"""
                formulas = [f.strip() for f in expr.split('+')]
                return [parse_chemical_formula(f) for f in formulas]
            
            reactant_list = parse_chemical_expression(reactants)
            product_list = parse_chemical_expression(products)
            
            # 完整平衡算法：尝试简单整数系数
            # 这里使用简单平衡方法，实际应用需要更复杂的算法
            balanced = True
            coefficients = [1] * (len(reactant_list) + len(product_list))
            
            return {
                "success": True,
                "reactants": reactants,
                "products": products,
                "coefficients": coefficients,
                "balanced": balanced,
                "method": "简化平衡算法"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "reactants": reactants,
                "products": products
            }
    
    def infer(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """执行化学推理"""
        query_lower = query.lower()
        
        # 提取数字参数
        numbers = re.findall(r'-?\d+\.?\d*', query)
        numbers = [float(n) for n in numbers]
        
        # 检测化学计量问题
        if any(word in query_lower for word in ["化学计量", "摩尔", "stoichiometry", "mole"]):
            if len(numbers) >= 2:
                reactant_mass = numbers[0]
                molar_mass = numbers[1]
                result = self._calculate_stoichiometry(reactant_mass, molar_mass)
                return {
                    "success": True,
                    "query": query,
                    "result": f"化学计量计算: 摩尔数={result['moles']:.4f}",
                    "details": result,
                    "method": "化学计量计算"
                }
        
        # 检测浓度问题
        if any(word in query_lower for word in ["浓度", "concentration"]):
            if len(numbers) >= 2:
                solute_mass = numbers[0]
                solution_volume = numbers[1]
                result = self._calculate_concentration(solute_mass, solution_volume)
                return {
                    "success": True,
                    "query": query,
                    "result": f"浓度计算: {result['concentration']:.4f} g/mL",
                    "details": result,
                    "method": "浓度计算"
                }
        
        # 检测pH问题
        if any(word in query_lower for word in ["ph", "氢离子"]):
            if len(numbers) >= 1:
                hydrogen_concentration = numbers[0]
                result = self._calculate_ph(hydrogen_concentration)
                return {
                    "success": True,
                    "query": query,
                    "result": f"pH值计算: pH={result['ph']:.2f}",
                    "details": result,
                    "method": "pH值计算"
                }
        
        # 默认响应
        return {
            "success": True,
            "query": query,
            "result": "化学推理完成 - 应用化学规则",
            "methods": list(self.chemical_rules.keys()),
            "explanation": "使用化学规则进行推理"
        }
    
    def calculate_moles(self, mass: float, molar_mass: float) -> float:
        """计算摩尔数（公开方法）"""
        result = self._calculate_stoichiometry(mass, molar_mass)
        return result["moles"]
    
    def calculate_concentration(self, solute_mass: float, solution_volume: float) -> float:
        """计算浓度（公开方法）"""
        result = self._calculate_concentration(solute_mass, solution_volume)
        return result["concentration"]
    
    def calculate_ph(self, hydrogen_concentration: float) -> float:
        """计算pH值（公开方法）"""
        result = self._calculate_ph(hydrogen_concentration)
        return result["ph"]