"""
知识验证器

验证知识的正确性、一致性和完整性。
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime


class KnowledgeValidator:
    """知识验证器

    功能：
    - 验证知识格式和结构
    - 检查知识一致性
    - 检测知识冲突
    - 评估知识质量
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化知识验证器

        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # 验证规则
        self.validation_rules = self._init_validation_rules()

        # 已知冲突模式
        self.conflict_patterns = self._init_conflict_patterns()

        # 验证缓存
        self.validation_cache = {}
        self.cache_size = config.get("validation_cache_size", 1000)

        self.logger.info("知识验证器初始化完成")

    def validate(self, knowledge_item: Dict[str, Any]) -> Dict[str, Any]:
        """验证知识条目

        参数:
            knowledge_item: 知识条目

        返回:
            验证结果
        """
        # 检查缓存
        cache_key = self._generate_cache_key(knowledge_item)
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]

        errors = []
        warnings = []

        # 1. 基本格式验证
        basic_errors = self._validate_basic_format(knowledge_item)
        errors.extend(basic_errors)

        # 如果没有基本格式错误，继续其他验证
        if not basic_errors:
            # 2. 类型特定验证
            type_errors = self._validate_by_type(knowledge_item)
            errors.extend(type_errors)

            # 3. 语义验证
            semantic_errors = self._validate_semantics(knowledge_item)
            errors.extend(semantic_errors)

            # 4. 一致性验证
            consistency_errors = self._validate_consistency(knowledge_item)
            errors.extend(consistency_errors)

            # 5. 质量评估
            quality_warnings = self._assess_quality(knowledge_item)
            warnings.extend(quality_warnings)

        # 准备验证结果
        validation_result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "confidence": self._calculate_confidence(errors, warnings),
        }

        # 缓存结果
        self._cache_validation(cache_key, validation_result)

        return validation_result

    def validate_batch(self, knowledge_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量验证知识条目

        参数:
            knowledge_items: 知识条目列表

        返回:
            批量验证结果
        """
        batch_results = []
        valid_count = 0
        total_errors = 0
        total_warnings = 0

        for item in knowledge_items:
            result = self.validate(item)
            batch_results.append(
                {
                    "id": item.get("id", "unknown"),
                    "type": item.get("type", "unknown"),
                    "valid": result["valid"],
                    "errors": result["errors"],
                    "warnings": result["warnings"],
                    "confidence": result["confidence"],
                }
            )

            if result["valid"]:
                valid_count += 1

            total_errors += len(result["errors"])
            total_warnings += len(result["warnings"])

        return {
            "total": len(knowledge_items),
            "valid": valid_count,
            "invalid": len(knowledge_items) - valid_count,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "average_confidence": (
                sum(r["confidence"] for r in batch_results) / len(batch_results)
                if batch_results
                else 0
            ),
            "details": batch_results,
        }

    def detect_conflicts(
        self,
        knowledge_items: List[Dict[str, Any]],
        existing_knowledge: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """检测知识冲突

        参数:
            knowledge_items: 要检查的知识条目
            existing_knowledge: 现有知识条目（如果为None，则只检查输入列表内的冲突）

        返回:
            冲突列表
        """
        conflicts = []

        # 准备所有知识条目
        all_items = knowledge_items.copy()
        if existing_knowledge:
            all_items.extend(existing_knowledge)

        # 检测冲突
        for i in range(len(all_items)):
            for j in range(i + 1, len(all_items)):
                item1 = all_items[i]
                item2 = all_items[j]

                conflict = self._detect_pairwise_conflict(item1, item2)
                if conflict:
                    conflicts.append(conflict)

        return conflicts

    def suggest_corrections(
        self, knowledge_item: Dict[str, Any], validation_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """建议知识修正

        参数:
            knowledge_item: 知识条目
            validation_result: 验证结果

        返回:
            修正建议列表
        """
        corrections = []

        # 根据错误类型提供修正建议
        for error in validation_result.get("errors", []):
            error_type = error.get("type")
            error_details = error.get("details", {})

            if error_type == "missing_field":
                field_name = error_details.get("field")
                correction = {
                    "type": "add_field",
                    "field": field_name,
                    "suggestion": f"请添加'{field_name}'字段",
                    "confidence": 0.9,
                }
                corrections.append(correction)

            elif error_type == "invalid_format":
                field_name = error_details.get("field")
                expected_format = error_details.get("expected_format")
                correction = {
                    "type": "format_field",
                    "field": field_name,
                    "suggestion": f"字段'{field_name}'格式应为: {expected_format}",
                    "confidence": 0.8,
                }
                corrections.append(correction)

            elif error_type == "inconsistent":
                correction = {
                    "type": "reconcile",
                    "suggestion": "知识存在不一致，请检查并修正",
                    "confidence": 0.7,
                }
                corrections.append(correction)

            elif error_type == "conflict":
                conflicting_id = error_details.get("conflicting_id")
                conflict_type = error_details.get("conflict_type")
                correction = {
                    "type": "resolve_conflict",
                    "suggestion": f"与知识'{conflicting_id}'存在{conflict_type}冲突，请检查",
                    "confidence": 0.6,
                }
                corrections.append(correction)

        # 根据警告提供改进建议
        for warning in validation_result.get("warnings", []):
            warning_type = warning.get("type")

            if warning_type == "low_quality":
                correction = {
                    "type": "improve_quality",
                    "suggestion": "知识质量较低，请提供更多详细信息",
                    "confidence": 0.5,
                }
                corrections.append(correction)

            elif warning_type == "ambiguous":
                correction = {
                    "type": "clarify",
                    "suggestion": "知识表达存在歧义，请明确表达",
                    "confidence": 0.5,
                }
                corrections.append(correction)

        return corrections

    def _init_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """初始化验证规则"""
        rules = {
            "fact": {
                "required_fields": ["statement"],
                "field_formats": {
                    "statement": "string",
                    "evidence": "optional|string|list",
                    "confidence": "optional|float|0-1",
                },
            },
            "rule": {
                "required_fields": ["condition", "conclusion"],
                "field_formats": {
                    "condition": "string",
                    "conclusion": "string",
                    "exceptions": "optional|list",
                    "applicability": "optional|string",
                },
            },
            "procedure": {
                "required_fields": ["name", "steps"],
                "field_formats": {
                    "name": "string",
                    "steps": "list|string",
                    "prerequisites": "optional|list",
                    "expected_outcome": "optional|string",
                },
            },
            "concept": {
                "required_fields": ["name", "definition"],
                "field_formats": {
                    "name": "string",
                    "definition": "string",
                    "examples": "optional|list",
                    "properties": "optional|dict",
                },
            },
            "relationship": {
                "required_fields": ["subject", "relation", "object"],
                "field_formats": {
                    "subject": "string",
                    "relation": "string",
                    "object": "string",
                    "strength": "optional|float|0-1",
                },
            },
            "event": {
                "required_fields": ["description"],
                "field_formats": {
                    "description": "string",
                    "time": "optional|string",
                    "location": "optional|string",
                    "participants": "optional|list",
                },
            },
            "problem_solution": {
                "required_fields": ["problem", "solution"],
                "field_formats": {
                    "problem": "string",
                    "solution": "string|list",
                    "effectiveness": "optional|float|0-1",
                },
            },
            "experience": {
                "required_fields": ["situation", "action", "result"],
                "field_formats": {
                    "situation": "string",
                    "action": "string",
                    "result": "string",
                    "lesson": "optional|string",
                },
            },
        }

        return rules

    def _init_conflict_patterns(self) -> List[Dict[str, Any]]:
        """初始化冲突模式"""
        patterns = [
            {
                "name": "contradiction",
                "description": "两个事实相互矛盾",
                "detector": self._detect_contradiction,
            },
            {
                "name": "circular_reference",
                "description": "知识形成循环引用",
                "detector": self._detect_circular_reference,
            },
            {
                "name": "incompatible_types",
                "description": "知识类型不兼容",
                "detector": self._detect_incompatible_types,
            },
            {
                "name": "temporal_inconsistency",
                "description": "时间顺序不一致",
                "detector": self._detect_temporal_inconsistency,
            },
        ]

        return patterns

    def _validate_basic_format(
        self, knowledge_item: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """验证基本格式"""
        errors = []

        # 检查必需字段
        required_fields = ["id", "type", "content", "created_at"]
        for field in required_fields:
            if field not in knowledge_item:
                errors.append(
                    {
                        "type": "missing_field",
                        "message": f"缺少必需字段: {field}",
                        "details": {"field": field},
                    }
                )

        # 检查类型有效性
        knowledge_type = knowledge_item.get("type")
        if knowledge_type and knowledge_type not in self.validation_rules:
            errors.append(
                {
                    "type": "invalid_type",
                    "message": f"无效的知识类型: {knowledge_type}",
                    "details": {
                        "type": knowledge_type,
                        "valid_types": list(self.validation_rules.keys()),
                    },
                }
            )

        # 检查内容字段
        content = knowledge_item.get("content")
        if not isinstance(content, dict):
            errors.append(
                {
                    "type": "invalid_content",
                    "message": "content字段必须是字典",
                    "details": {"content_type": type(content).__name__},
                }
            )

        return errors

    def _validate_by_type(self, knowledge_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据类型进行验证"""
        errors = []
        knowledge_type = knowledge_item.get("type")
        content = knowledge_item.get("content", {})

        if not knowledge_type or knowledge_type not in self.validation_rules:
            return errors

        rules = self.validation_rules[knowledge_type]

        # 检查必需字段
        for field in rules.get("required_fields", []):
            if field not in content:
                errors.append(
                    {
                        "type": "missing_field",
                        "message": f"类型'{knowledge_type}'需要字段: {field}",
                        "details": {"field": field, "knowledge_type": knowledge_type},
                    }
                )

        # 检查字段格式
        field_formats = rules.get("field_formats", {})
        for field, format_spec in field_formats.items():
            if field in content:
                field_value = content[field]
                format_error = self._validate_field_format(
                    field, field_value, format_spec
                )
                if format_error:
                    errors.append(format_error)

        return errors

    def _validate_field_format(
        self, field_name: str, field_value: Any, format_spec: str
    ) -> Optional[Dict[str, Any]]:
        """验证字段格式"""
        format_parts = format_spec.split("|")

        for format_part in format_parts:
            if format_part == "optional":
                # 可选字段，没有值也可以
                continue

            elif format_part == "string":
                if not isinstance(field_value, str):
                    return {
                        "type": "invalid_format",
                        "message": f"字段'{field_name}'应为字符串",
                        "details": {
                            "field": field_name,
                            "expected_format": "string",
                            "actual_type": type(field_value).__name__,
                        },
                    }

            elif format_part == "list":
                if not isinstance(field_value, list):
                    return {
                        "type": "invalid_format",
                        "message": f"字段'{field_name}'应为列表",
                        "details": {
                            "field": field_name,
                            "expected_format": "list",
                            "actual_type": type(field_value).__name__,
                        },
                    }

            elif format_part == "dict":
                if not isinstance(field_value, dict):
                    return {
                        "type": "invalid_format",
                        "message": f"字段'{field_name}'应为字典",
                        "details": {
                            "field": field_name,
                            "expected_format": "dict",
                            "actual_type": type(field_value).__name__,
                        },
                    }

            elif format_part == "float":
                if not isinstance(field_value, (int, float)):
                    return {
                        "type": "invalid_format",
                        "message": f"字段'{field_name}'应为数值",
                        "details": {
                            "field": field_name,
                            "expected_format": "float",
                            "actual_type": type(field_value).__name__,
                        },
                    }

            elif "-" in format_part:  # 范围检查，如 '0-1'
                try:
                    min_val, max_val = map(float, format_part.split("-"))
                    if isinstance(field_value, (int, float)):
                        if not (min_val <= field_value <= max_val):
                            return {
                                "type": "invalid_range",
                                "message": (
                                    f"字段'{field_name}'应在范围[{min_val}, {max_val}]内"
                                ),
                                "details": {
                                    "field": field_name,
                                    "expected_range": f"{min_val}-{max_val}",
                                    "actual_value": field_value,
                                },
                            }
                except ValueError:
                    pass  # 已实现

        return None  # 返回None

    def _validate_semantics(
        self, knowledge_item: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """语义验证"""
        errors = []
        knowledge_type = knowledge_item.get("type")
        content = knowledge_item.get("content", {})

        # 检查空值或无效值
        for field_name, field_value in content.items():
            if isinstance(field_value, str) and not field_value.strip():
                errors.append(
                    {
                        "type": "empty_value",
                        "message": f"字段'{field_name}'不能为空",
                        "details": {"field": field_name},
                    }
                )

        # 类型特定的语义检查
        if knowledge_type == "fact":
            statement = content.get("statement", "")
            if len(statement.split()) < 3:  # 至少3个词
                errors.append(
                    {
                        "type": "insufficient_detail",
                        "message": "事实描述过于简单",
                        "details": {"field": "statement", "min_words": 3},
                    }
                )

        elif knowledge_type == "rule":
            condition = content.get("condition", "")
            conclusion = content.get("conclusion", "")

            if condition == conclusion:
                errors.append(
                    {
                        "type": "tautology",
                        "message": "规则的条件和结论相同",
                        "details": {"condition": condition, "conclusion": conclusion},
                    }
                )

        elif knowledge_type == "procedure":
            steps = content.get("steps", [])
            if steps and isinstance(steps, list):
                if len(steps) < 2:
                    errors.append(
                        {
                            "type": "insufficient_steps",
                            "message": "过程至少需要2个步骤",
                            "details": {"field": "steps", "min_steps": 2},
                        }
                    )

                # 检查步骤是否明确
                for i, step in enumerate(steps):
                    if isinstance(step, str) and len(step.split()) < 2:
                        errors.append(
                            {
                                "type": "vague_step",
                                "message": f"步骤{i+1}描述不明确",
                                "details": {"step_index": i, "step": step},
                            }
                        )

        elif knowledge_type == "concept":
            definition = content.get("definition", "")
            if len(definition.split()) < 5:  # 至少5个词
                errors.append(
                    {
                        "type": "insufficient_definition",
                        "message": "概念定义过于简单",
                        "details": {"field": "definition", "min_words": 5},
                    }
                )

        return errors

    def _validate_consistency(
        self, knowledge_item: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """一致性验证"""
        errors = []

        # 检查内部一致性
        knowledge_type = knowledge_item.get("type")
        content = knowledge_item.get("content", {})

        if knowledge_type == "relationship":
            subject = content.get("subject", "")
            obj = content.get("object", "")

            if subject and obj and subject == obj:
                errors.append(
                    {
                        "type": "self_relation",
                        "message": "关系的主体和客体相同",
                        "details": {"subject": subject, "object": obj},
                    }
                )

        # 检查时间一致性
        created_at = knowledge_item.get("created_at")
        updated_at = knowledge_item.get("updated_at")

        if created_at and updated_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

                if updated_dt < created_dt:
                    errors.append(
                        {
                            "type": "temporal_inconsistency",
                            "message": "更新时间早于创建时间",
                            "details": {
                                "created_at": created_at,
                                "updated_at": updated_at,
                            },
                        }
                    )
            except ValueError:
                pass  # 已实现

        return errors

    def _assess_quality(self, knowledge_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """评估知识质量"""
        warnings = []
        content = knowledge_item.get("content", {})

        # 检查信息丰富度
        total_info = 0
        for field_value in content.values():
            if isinstance(field_value, str):
                total_info += len(field_value.split())
            elif isinstance(field_value, list):
                total_info += len(field_value)
            elif isinstance(field_value, dict):
                total_info += len(field_value)

        if total_info < 10:  # 信息量不足
            warnings.append(
                {
                    "type": "low_quality",
                    "message": "知识信息量不足",
                    "details": {"total_info": total_info},
                }
            )

        # 检查置信度
        confidence = knowledge_item.get("confidence", 1.0)
        if confidence < 0.7:
            warnings.append(
                {
                    "type": "low_confidence",
                    "message": f"知识置信度较低: {confidence}",
                    "details": {"confidence": confidence},
                }
            )

        # 检查来源
        source = knowledge_item.get("source", "")
        if not source or source == "unknown":
            warnings.append(
                {
                    "type": "unknown_source",
                    "message": "知识来源未知",
                    "details": {"source": source},
                }
            )

        # 检查歧义性
        if self._is_ambiguous(knowledge_item):
            warnings.append(
                {"type": "ambiguous", "message": "知识表达可能存在歧义", "details": {}}
            )

        return warnings

    def _is_ambiguous(self, knowledge_item: Dict[str, Any]) -> bool:
        """检查知识是否存在歧义"""
        content = knowledge_item.get("content", {})

        # 检查模糊词汇
        vague_words = ["可能", "也许", "大概", "或许", "有些", "某种", "某些"]

        for field_value in content.values():
            if isinstance(field_value, str):
                for vague_word in vague_words:
                    if vague_word in field_value:
                        return True

        return False

    def _calculate_confidence(
        self, errors: List[Dict[str, Any]], warnings: List[Dict[str, Any]]
    ) -> float:
        """计算置信度"""
        base_confidence = 1.0

        # 错误降低置信度
        for error in errors:
            error_type = error.get("type")
            if error_type in ["missing_field", "invalid_type", "invalid_content"]:
                base_confidence *= 0.5
            elif error_type in ["invalid_format", "empty_value"]:
                base_confidence *= 0.7
            elif error_type in ["insufficient_detail", "tautology"]:
                base_confidence *= 0.8
            else:
                base_confidence *= 0.9

        # 警告稍微降低置信度
        for warning in warnings:
            warning_type = warning.get("type")
            if warning_type in ["low_confidence", "unknown_source"]:
                base_confidence *= 0.9
            else:
                base_confidence *= 0.95

        return max(0.0, min(1.0, base_confidence))

    def _detect_pairwise_conflict(
        self, item1: Dict[str, Any], item2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """检测两个知识条目之间的冲突"""
        for pattern in self.conflict_patterns:
            detector = pattern["detector"]
            conflict = detector(item1, item2)

            if conflict:
                return {
                    "type": pattern["name"],
                    "description": pattern["description"],
                    "items": [item1.get("id", "unknown"), item2.get("id", "unknown")],
                    "details": conflict,
                }

        return None  # 返回None

    def _detect_contradiction(
        self, item1: Dict[str, Any], item2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """检测矛盾"""
        # 简单实现：检查两个事实是否直接矛盾
        if item1.get("type") == "fact" and item2.get("type") == "fact":
            content1 = item1.get("content", {})
            content2 = item2.get("content", {})

            statement1 = content1.get("statement", "")
            statement2 = content2.get("statement", "")

            # 简单的矛盾检测（实际应用中需要更复杂的逻辑）
            negation_words = ["不", "非", "没", "无", "不是", "不会", "不能"]

            for negation in negation_words:
                if (
                    negation in statement1
                    and negation not in statement2
                    and statement1.replace(negation, "") in statement2
                ):
                    return {
                        "negation_word": negation,
                        "statement1": statement1,
                        "statement2": statement2,
                    }

        return None  # 返回None

    def _detect_circular_reference(
        self, item1: Dict[str, Any], item2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """检测循环引用"""
        # 检查两个规则是否形成循环
        if item1.get("type") == "rule" and item2.get("type") == "rule":
            content1 = item1.get("content", {})
            content2 = item2.get("content", {})

            condition1 = content1.get("condition", "")
            conclusion1 = content1.get("conclusion", "")
            condition2 = content2.get("condition", "")
            conclusion2 = content2.get("conclusion", "")

            # 检查A->B和B->A的情况
            if condition1 == conclusion2 and conclusion1 == condition2:
                return {
                    "rule1": f"{condition1} -> {conclusion1}",
                    "rule2": f"{condition2} -> {conclusion2}",
                }

        return None  # 返回None

    def _detect_incompatible_types(
        self, item1: Dict[str, Any], item2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """检测不兼容的类型"""
        # 检查两个知识条目是否类型不兼容
        type1 = item1.get("type")
        type2 = item2.get("type")

        # 定义不兼容的类型对
        incompatible_pairs = [
            ("fact", "rule"),  # 事实和规则可能冲突
            ("concept", "fact"),  # 概念定义和事实可能冲突
        ]

        for pair in incompatible_pairs:
            if (type1 == pair[0] and type2 == pair[1]) or (
                type1 == pair[1] and type2 == pair[0]
            ):
                return {
                    "type1": type1,
                    "type2": type2,
                    "reason": f"{pair[0]}和{pair[1]}可能不兼容",
                }

        return None  # 返回None

    def _detect_temporal_inconsistency(
        self, item1: Dict[str, Any], item2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """检测时间不一致"""
        # 检查两个事件的时间顺序是否矛盾
        if item1.get("type") == "event" and item2.get("type") == "event":
            content1 = item1.get("content", {})
            content2 = item2.get("content", {})

            time1 = content1.get("time", "")
            time2 = content2.get("time", "")

            # 如果有时间信息且相互矛盾
            if time1 and time2 and time1 == time2:
                # 简单检查：同一时间发生两个不同事件（可能没问题，但标记为潜在冲突）
                description1 = content1.get("description", "")
                description2 = content2.get("description", "")

                if description1 != description2:
                    return {
                        "time": time1,
                        "event1": description1,
                        "event2": description2,
                    }

        return None  # 返回None

    def _generate_cache_key(self, knowledge_item: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 基于知识内容和类型生成键
        import json

        # 对内容进行JSON序列化以生成哈希
        content = knowledge_item.get("content", {})
        content_hash = hash(json.dumps(content, sort_keys=True))

        key_parts = [
            knowledge_item.get("type", ""),
            knowledge_item.get("id", ""),
            str(content_hash),
            knowledge_item.get("updated_at", ""),
        ]

        return "_".join(filter(None, key_parts))

    def _cache_validation(self, cache_key: str, validation_result: Dict[str, Any]):
        """缓存验证结果"""
        self.validation_cache[cache_key] = validation_result

        # 清理旧缓存
        if len(self.validation_cache) > self.cache_size:
            # 移除最旧的条目（简单实现）
            if self.validation_cache:
                oldest_key = next(iter(self.validation_cache))
                del self.validation_cache[oldest_key]

    def clear_cache(self):
        """清空验证缓存"""
        self.validation_cache.clear()
        self.logger.info("验证缓存已清空")
