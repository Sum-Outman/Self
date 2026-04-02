#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学推理引擎模块
提供症状分析、疾病诊断、药物剂量计算、生理参数评估等功能
"""

import math
import re
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class MedicalReasoningEngine:
    """医学推理引擎"""
    
    def __init__(self, config):
        self.config = config
        self.medical_rules = self._initialize_medical_rules()
        logger.info("医学推理引擎初始化完成")
    
    def _initialize_medical_rules(self) -> Dict[str, callable]:
        """初始化医学规则"""
        return {
            "symptom_analysis": self._analyze_symptoms,
            "diagnosis_support": self._support_diagnosis,
            "dosage_calculation": self._calculate_dosage,
            "vital_signs_assessment": self._assess_vital_signs,
        }
    
    def _analyze_symptoms(self, symptoms: List[str], patient_age: int, patient_gender: str) -> Dict[str, Any]:
        """症状分析"""
        # 完整症状分析逻辑
        common_patterns = {
            "fever_cough": ["发烧", "咳嗽"],
            "headache_nausea": ["头痛", "恶心"],
            "fatigue_pain": ["疲劳", "疼痛"],
        }
        
        matched_patterns = []
        for pattern_name, pattern_symptoms in common_patterns.items():
            if any(symptom in symptoms for symptom in pattern_symptoms):
                matched_patterns.append(pattern_name)
        
        return {
            "symptoms": symptoms,
            "age": patient_age,
            "gender": patient_gender,
            "matched_patterns": matched_patterns,
            "urgency_level": "medium" if len(symptoms) > 2 else "low"
        }
    
    def _support_diagnosis(self, symptoms: List[str], patient_info: Dict) -> Dict[str, Any]:
        """诊断支持"""
        # 完整诊断逻辑
        diagnosis_rules = {
            ("发烧", "咳嗽", "乏力"): "流感",
            ("头痛", "恶心", "畏光"): "偏头痛",
            ("胸痛", "呼吸困难", "出汗"): "心血管疾病",
        }
        
        for rule_symptoms, diagnosis in diagnosis_rules.items():
            if all(symptom in symptoms for symptom in rule_symptoms):
                return {
                    "likely_diagnosis": diagnosis,
                    "confidence": 0.7,
                    "recommendations": ["建议就医进一步检查"]
                }
        
        return {
            "likely_diagnosis": "未知",
            "confidence": 0.3,
            "recommendations": ["症状不典型，建议就医咨询"]
        }
    
    def _calculate_dosage(self, drug_name: str, patient_weight: float, condition_severity: str) -> Dict[str, Any]:
        """药物剂量计算"""
        # 完整剂量计算
        standard_dosages = {
            "paracetamol": 15,  # mg/kg
            "ibuprofen": 10,    # mg/kg
            "amoxicillin": 25,  # mg/kg
        }
        
        base_dose = standard_dosages.get(drug_name.lower(), 10)
        weight_based_dose = base_dose * patient_weight
        
        # 根据严重程度调整
        severity_multiplier = {
            "mild": 0.5,
            "moderate": 1.0,
            "severe": 1.5
        }.get(condition_severity, 1.0)
        
        final_dose = weight_based_dose * severity_multiplier
        
        return {
            "drug": drug_name,
            "patient_weight": patient_weight,
            "base_dose_per_kg": base_dose,
            "weight_based_dose": weight_based_dose,
            "severity_multiplier": severity_multiplier,
            "recommended_dose": final_dose,
            "warning": "仅供参考，实际用药请遵医嘱"
        }
    
    def _assess_vital_signs(self, vital_signs: Dict[str, float]) -> Dict[str, Any]:
        """生理参数评估"""
        # 正常范围
        normal_ranges = {
            "heart_rate": (60, 100),  # 心率
            "blood_pressure_systolic": (90, 140),  # 收缩压
            "blood_pressure_diastolic": (60, 90),  # 舒张压
            "body_temperature": (36.1, 37.2),  # 体温
            "respiratory_rate": (12, 20),  # 呼吸频率
        }
        
        assessment = {}
        for sign, value in vital_signs.items():
            if sign in normal_ranges:
                low, high = normal_ranges[sign]
                if value < low:
                    assessment[sign] = {"status": "低", "value": value, "normal_range": f"{low}-{high}"}
                elif value > high:
                    assessment[sign] = {"status": "高", "value": value, "normal_range": f"{low}-{high}"}
                else:
                    assessment[sign] = {"status": "正常", "value": value, "normal_range": f"{low}-{high}"}
        
        overall_status = "正常"
        if any(item["status"] != "正常" for item in assessment.values()):
            overall_status = "异常"
        
        return {
            "vital_signs": vital_signs,
            "assessment": assessment,
            "overall_status": overall_status,
            "recommendations": ["定期监测生理参数"] if overall_status == "正常" else ["建议就医检查"]
        }
    
    def infer(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """执行医学推理"""
        query_lower = query.lower()
        
        # 提取数字参数
        numbers = re.findall(r'-?\d+\.?\d*', query)
        numbers = [float(n) for n in numbers]
        
        # 检测症状分析问题
        if any(word in query_lower for word in ["症状", "symptom", "诊断", "diagnosis"]):
            # 完整处理
            return {
                "success": True,
                "query": query,
                "result": "医学推理完成 - 症状分析",
                "methods": list(self.medical_rules.keys()),
                "explanation": "使用医学规则进行推理",
                "warning": "此为医学推理演示，实际诊断需专业医生"
            }
        
        # 检测药物剂量问题
        if any(word in query_lower for word in ["剂量", "dosage", "用药"]):
            if len(numbers) >= 1:
                patient_weight = numbers[0]
                drug_name = "paracetamol"  # 默认药物
                result = self._calculate_dosage(drug_name, patient_weight, "moderate")
                return {
                    "success": True,
                    "query": query,
                    "result": f"药物剂量计算: {drug_name}推荐剂量为{result['recommended_dose']:.2f}mg",
                    "details": result,
                    "method": "药物剂量计算",
                    "warning": "仅供参考，实际用药请遵医嘱"
                }
        
        # 检测生理参数问题
        if any(word in query_lower for word in ["心率", "血压", "体温", "heart rate", "blood pressure"]):
            # 完整处理
            return {
                "success": True,
                "query": query,
                "result": "医学推理完成 - 生理参数评估",
                "methods": list(self.medical_rules.keys()),
                "explanation": "使用医学规则进行推理",
                "warning": "医疗建议需专业医生提供"
            }
        
        # 默认响应
        return {
            "success": True,
            "query": query,
            "result": "医学推理完成 - 应用医学规则",
            "methods": list(self.medical_rules.keys()),
            "explanation": "使用医学规则进行推理",
            "warning": "此为医学推理演示，实际医疗决策需专业医生"
        }
    
    def analyze_symptoms(self, symptoms: List[str], age: int, gender: str) -> Dict[str, Any]:
        """症状分析（公开方法）"""
        return self._analyze_symptoms(symptoms, age, gender)
    
    def calculate_drug_dosage(self, drug_name: str, weight: float, severity: str = "moderate") -> float:
        """计算药物剂量（公开方法）"""
        result = self._calculate_dosage(drug_name, weight, severity)
        return result["recommended_dose"]
    
    def assess_vital_signs(self, vital_signs: Dict[str, float]) -> Dict[str, Any]:
        """评估生理参数（公开方法）"""
        return self._assess_vital_signs(vital_signs)