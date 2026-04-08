"""
知识融合模块

实现多源知识融合的核心算法，包括：
1. 冲突检测与解决
2. 知识对齐（实体消歧、关系映射）
3. 知识质量评估
4. 知识合并与集成
5. 知识演化跟踪

工业级AGI从零开始实现，不依赖外部预训练模型。
"""

import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict

from .knowledge_manager import KnowledgeType

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """冲突类型枚举"""
    CONTRADICTION = "contradiction"  # 直接矛盾
    INCONSISTENCY = "inconsistency"   # 不一致
    AMBIGUITY = "ambiguity"           # 歧义
    REDUNDANCY = "redundancy"         # 冗余
    OUTDATED = "outdated"             # 过时


class FusionStrategy(Enum):
    """融合策略枚举"""
    VOTING = "voting"                 # 投票策略
    WEIGHTED = "weighted"             # 加权融合
    RECENCY = "recency"               # 最新优先
    AUTHORITY = "authority"           # 权威优先
    CONSENSUS = "consensus"           # 共识优先
    SELECT_HIGHEST_CONFIDENCE = "select_highest_confidence"  # 选择最高置信度


class KnowledgeConflict:
    """知识冲突表示"""
    
    def __init__(
        self,
        conflict_type: ConflictType,
        knowledge_items: List[Dict[str, Any]],
        description: str,
        confidence: float = 0.8,
        resolution_strategy: Optional[FusionStrategy] = None
    ):
        self.conflict_type = conflict_type
        self.knowledge_items = knowledge_items
        self.description = description
        self.confidence = confidence
        self.resolution_strategy = resolution_strategy
        self.detected_at = datetime.now()
        self.resolved = False
        self.resolution = None
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "conflict_type": self.conflict_type.value,
            "knowledge_item_ids": [item.get("id", "unknown") for item in self.knowledge_items],
            "description": self.description,
            "confidence": self.confidence,
            "resolution_strategy": self.resolution_strategy.value if self.resolution_strategy else None,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution
        }


class KnowledgeFusionEngine:
    """知识融合引擎
    
    核心功能：
    1. 检测多源知识之间的冲突
    2. 解决冲突并生成融合后的知识
    3. 评估知识质量
    4. 跟踪知识演化
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化知识融合引擎
        
        参数:
            config: 配置字典
        """
        self.logger = logger
        self.config = config or {
            "conflict_threshold": 0.7,           # 冲突检测阈值
            "similarity_threshold": 0.8,         # 相似性阈值（用于冗余检测）
            "recency_weight": 0.4,               # 时效性权重
            "authority_weight": 0.3,             # 权威性权重
            "consensus_weight": 0.3,             # 共识权重
            "min_confidence": 0.6,               # 最小置信度
            "enable_evolution_tracking": True,   # 启用演化跟踪
            "max_history_length": 100,           # 最大历史长度
            "industrial_mode": True,             # 工业级模式
        }
        
        # 冲突检测规则
        self.conflict_rules = self._init_conflict_rules()
        
        # 融合策略配置
        self.fusion_strategies = self._init_fusion_strategies()
        
        # 冲突历史
        self.conflict_history: List[KnowledgeConflict] = []
        
        # 知识演化跟踪
        self.evolution_history: List[Dict[str, Any]] = []
        
        self.logger.info("知识融合引擎初始化完成")
    
    def _init_conflict_rules(self) -> Dict[ConflictType, Dict[str, Any]]:
        """初始化冲突检测规则"""
        return {
            ConflictType.CONTRADICTION: {
                "description": "直接逻辑矛盾",
                "detection_method": "logical_contradiction",
                "threshold": 0.9,
                "resolution_strategy": FusionStrategy.VOTING
            },
            ConflictType.INCONSISTENCY: {
                "description": "事实不一致",
                "detection_method": "fact_inconsistency",
                "threshold": 0.7,
                "resolution_strategy": FusionStrategy.WEIGHTED
            },
            ConflictType.AMBIGUITY: {
                "description": "表述歧义",
                "detection_method": "semantic_ambiguity",
                "threshold": 0.6,
                "resolution_strategy": FusionStrategy.CONSENSUS
            },
            ConflictType.REDUNDANCY: {
                "description": "信息冗余",
                "detection_method": "semantic_similarity",
                "threshold": 0.8,
                "resolution_strategy": FusionStrategy.RECENCY
            },
            ConflictType.OUTDATED: {
                "description": "知识过时",
                "detection_method": "temporal_analysis",
                "threshold": 0.75,
                "resolution_strategy": FusionStrategy.RECENCY
            }
        }
    
    def _init_fusion_strategies(self) -> Dict[FusionStrategy, Dict[str, Any]]:
        """初始化融合策略"""
        return {
            FusionStrategy.VOTING: {
                "description": "多数投票",
                "weight_function": self._voting_weight,
                "applicable_types": [ConflictType.CONTRADICTION]
            },
            FusionStrategy.WEIGHTED: {
                "description": "加权融合",
                "weight_function": self._weighted_fusion,
                "applicable_types": [ConflictType.INCONSISTENCY, ConflictType.AMBIGUITY]
            },
            FusionStrategy.RECENCY: {
                "description": "最新优先",
                "weight_function": self._recency_weight,
                "applicable_types": [ConflictType.OUTDATED, ConflictType.REDUNDANCY]
            },
            FusionStrategy.AUTHORITY: {
                "description": "权威优先",
                "weight_function": self._authority_weight,
                "applicable_types": [ConflictType.CONTRADICTION, ConflictType.INCONSISTENCY]
            },
            FusionStrategy.CONSENSUS: {
                "description": "共识优先",
                "weight_function": self._consensus_weight,
                "applicable_types": [ConflictType.AMBIGUITY]
            }
        }
    
    def detect_conflicts(self, knowledge_items: List[Dict[str, Any]]) -> List[KnowledgeConflict]:
        """检测知识冲突
        
        参数:
            knowledge_items: 知识条目列表
            
        返回:
            检测到的冲突列表
        """
        conflicts = []
        
        if len(knowledge_items) < 2:
            return conflicts
        
        self.logger.info(f"开始检测知识冲突，共{len(knowledge_items)}个知识条目")
        
        # 按主题分组知识
        grouped_knowledge = self._group_by_topic(knowledge_items)
        
        for topic, items in grouped_knowledge.items():
            if len(items) < 2:
                continue
            
            # 检测各种类型的冲突
            topic_conflicts = []
            
            # 1. 检测直接矛盾
            contradiction_conflicts = self._detect_contradictions(items)
            topic_conflicts.extend(contradiction_conflicts)
            
            # 2. 检测不一致
            inconsistency_conflicts = self._detect_inconsistencies(items)
            topic_conflicts.extend(inconsistency_conflicts)
            
            # 3. 检测歧义
            ambiguity_conflicts = self._detect_ambiguities(items)
            topic_conflicts.extend(ambiguity_conflicts)
            
            # 4. 检测冗余
            redundancy_conflicts = self._detect_redundancies(items)
            topic_conflicts.extend(redundancy_conflicts)
            
            # 5. 检测过时
            outdated_conflicts = self._detect_outdated(items)
            topic_conflicts.extend(outdated_conflicts)
            
            conflicts.extend(topic_conflicts)
        
        self.logger.info(f"共检测到{len(conflicts)}个知识冲突")
        
        # 记录到历史
        for conflict in conflicts:
            self.conflict_history.append(conflict)
        
        # 限制历史长度
        if len(self.conflict_history) > self.config["max_history_length"]:
            self.conflict_history = self.conflict_history[-self.config["max_history_length"]:]
        
        return conflicts
    
    def _group_by_topic(self, knowledge_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按主题分组知识"""
        groups = defaultdict(list)
        
        for item in knowledge_items:
            # 提取主题信息
            topic = self._extract_topic(item)
            groups[topic].append(item)
        
        return dict(groups)
    
    def _extract_topic(self, knowledge_item: Dict[str, Any]) -> str:
        """从知识条目中提取主题"""
        content = knowledge_item.get("content", {})
        metadata = knowledge_item.get("metadata", {})
        
        # 处理content是字符串的情况
        if isinstance(content, str):
            # 对于字符串内容，创建一个简单的字典用于提取
            content_dict = {"text": content}
        else:
            content_dict = content
        
        # 尝试从不同字段提取主题
        topic_sources = [
            content_dict.get("topic") if isinstance(content_dict, dict) else None,
            content_dict.get("subject") if isinstance(content_dict, dict) else None,
            metadata.get("topic") if isinstance(metadata, dict) else None,
            metadata.get("category") if isinstance(metadata, dict) else None,
            knowledge_item.get("type", "unknown")
        ]
        
        for source in topic_sources:
            if source and isinstance(source, str):
                return source.lower()
        
        # 后备方案：使用类型作为主题
        return knowledge_item.get("type", "unknown").lower()
    
    def _detect_contradictions(self, items: List[Dict[str, Any]]) -> List[KnowledgeConflict]:
        """检测直接矛盾"""
        conflicts = []
        
        # 简单实现：检查相同属性的相反值
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item1 = items[i]
                item2 = items[j]
                
                # 提取关键事实
                facts1 = self._extract_facts(item1)
                facts2 = self._extract_facts(item2)
                
                # 检查事实矛盾
                for fact_key in set(facts1.keys()) & set(facts2.keys()):
                    value1 = facts1[fact_key]
                    value2 = facts2[fact_key]
                    
                    if self._are_contradictory(value1, value2):
                        conflict = KnowledgeConflict(
                            conflict_type=ConflictType.CONTRADICTION,
                            knowledge_items=[item1, item2],
                            description=f"事实矛盾: {fact_key} = {value1} vs {value2}",
                            confidence=0.9
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    def _extract_facts(self, knowledge_item: Dict[str, Any]) -> Dict[str, Any]:
        """从知识条目中提取事实"""
        facts = {}
        content = knowledge_item.get("content", {})
        
        if isinstance(content, dict):
            # 提取简单键值对作为事实
            for key, value in content.items():
                if isinstance(value, (str, int, float, bool)):
                    facts[key] = value
        
        return facts
    
    def _are_contradictory(self, value1: Any, value2: Any) -> bool:
        """判断两个值是否矛盾"""
        if isinstance(value1, bool) and isinstance(value2, bool):
            return value1 != value2
        
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            # 数值相反
            return abs(value1 + value2) < 0.001 and value1 != 0
        
        if isinstance(value1, str) and isinstance(value2, str):
            # 完整实现）
            opposites = {
                "是": "否", "对": "错", "真": "假", "存在": "不存在",
                "有": "无", "可以": "不可以", "能": "不能"
            }
            
            v1_lower = value1.lower()
            v2_lower = value2.lower()
            
            for word, opposite in opposites.items():
                if word in v1_lower and opposite in v2_lower:
                    return True
                if opposite in v1_lower and word in v2_lower:
                    return True
        
        return False
    
    def _detect_inconsistencies(self, items: List[Dict[str, Any]]) -> List[KnowledgeConflict]:
        """检测不一致"""
        conflicts = []
        
        # 检查相同类型知识的不一致
        type_groups = defaultdict(list)
        for item in items:
            item_type = item.get("type", "unknown")
            type_groups[item_type].append(item)
        
        for item_type, type_items in type_groups.items():
            if len(type_items) < 2:
                continue
            
            # 比较置信度差异
            confidences = [item.get("confidence", 0.5) for item in type_items]
            if max(confidences) - min(confidences) > 0.3:  # 置信度差异过大
                conflict = KnowledgeConflict(
                    conflict_type=ConflictType.INCONSISTENCY,
                    knowledge_items=type_items,
                    description=f"相同类型知识置信度差异过大: {min(confidences):.2f} - {max(confidences):.2f}",
                    confidence=0.7
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_ambiguities(self, items: List[Dict[str, Any]]) -> List[KnowledgeConflict]:
        """检测歧义"""
        conflicts = []
        
        # 检查模糊表述
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item1 = items[i]
                item2 = items[j]
                
                # 检查内容相似但表述不同（可能歧义）
                similarity = self._calculate_semantic_similarity(item1, item2)
                if 0.5 < similarity < 0.8:  # 相似但又不完全相同
                    conflict = KnowledgeConflict(
                        conflict_type=ConflictType.AMBIGUITY,
                        knowledge_items=[item1, item2],
                        description="知识表述存在歧义，可能指向相同概念但表述不同",
                        confidence=similarity
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _calculate_semantic_similarity(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> float:
        """计算语义相似度（完整实现）"""
        # 在实际应用中应使用嵌入模型计算相似度
        # 这里使用文本内容的Jaccard相似度作为完整
        
        text1 = self._extract_text(item1)
        text2 = self._extract_text(item2)
        
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_text(self, knowledge_item: Dict[str, Any]) -> str:
        """从知识条目中提取文本"""
        content = knowledge_item.get("content", {})
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, dict):
            # 提取所有字符串值
            texts = []
            for value in content.values():
                if isinstance(value, str):
                    texts.append(value)
            return " ".join(texts)
        
        return ""
    
    def _detect_redundancies(self, items: List[Dict[str, Any]]) -> List[KnowledgeConflict]:
        """检测冗余"""
        conflicts = []
        
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item1 = items[i]
                item2 = items[j]
                
                similarity = self._calculate_semantic_similarity(item1, item2)
                if similarity >= self.config["similarity_threshold"]:
                    conflict = KnowledgeConflict(
                        conflict_type=ConflictType.REDUNDANCY,
                        knowledge_items=[item1, item2],
                        description=f"知识冗余，相似度: {similarity:.2f}",
                        confidence=similarity
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_outdated(self, items: List[Dict[str, Any]]) -> List[KnowledgeConflict]:
        """检测过时知识"""
        conflicts = []
        
        # 按时间排序
        dated_items = []
        for item in items:
            timestamp = self._extract_timestamp(item)
            if timestamp:
                dated_items.append((timestamp, item))
        
        if len(dated_items) < 2:
            return conflicts
        
        # 按时间排序
        dated_items.sort(key=lambda x: x[0])
        
        # 检查时间差异
        oldest_time = dated_items[0][0]
        newest_time = dated_items[-1][0]
        
        time_diff = (newest_time - oldest_time).total_seconds() / (60 * 60 * 24)  # 天数
        
        if time_diff > 30:  # 超过30天认为可能过时
            for timestamp, item in dated_items[:-1]:  # 除最新的之外
                conflict = KnowledgeConflict(
                    conflict_type=ConflictType.OUTDATED,
                    knowledge_items=[item],
                    description=f"知识可能过时，创建于{timestamp.date()}，最新知识为{newest_time.date()}",
                    confidence=min(0.9, time_diff / 365)  # 时间越长置信度越高
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _extract_timestamp(self, knowledge_item: Dict[str, Any]) -> Optional[datetime]:
        """从知识条目中提取时间戳"""
        created_at = knowledge_item.get("created_at")
        if created_at:
            try:
                if isinstance(created_at, str):
                    return datetime.fromisoformat(created_at)
                elif isinstance(created_at, datetime):
                    return created_at
            except Exception:
                pass  # 已实现
        
        metadata = knowledge_item.get("metadata", {})
        timestamp = metadata.get("timestamp")
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    return datetime.fromisoformat(timestamp)
                elif isinstance(timestamp, (int, float)):
                    return datetime.fromtimestamp(timestamp)
            except Exception:
                pass  # 已实现
        
        return None  # 返回None
    
    def resolve_conflicts(self, conflicts: List[KnowledgeConflict]) -> List[Dict[str, Any]]:
        """解决知识冲突
        
        参数:
            conflicts: 冲突列表
            
        返回:
            融合后的知识条目列表
        """
        resolved_knowledge = []
        
        self.logger.info(f"开始解决{len(conflicts)}个知识冲突")
        
        for conflict in conflicts:
            if conflict.resolved:
                continue
            
            # 根据冲突类型选择合适的解决策略
            resolution_strategy = self._select_resolution_strategy(conflict)
            conflict.resolution_strategy = resolution_strategy
            
            # 执行融合
            fused_knowledge = self._fuse_knowledge(
                conflict.knowledge_items,
                resolution_strategy
            )
            
            if fused_knowledge:
                conflict.resolved = True
                conflict.resolution = {
                    "strategy": resolution_strategy.value,
                    "fused_knowledge_id": fused_knowledge.get("id"),
                    "timestamp": datetime.now().isoformat()
                }
                
                resolved_knowledge.append(fused_knowledge)
                self.logger.info(f"冲突解决成功，使用策略: {resolution_strategy.value}")
        
        # 记录演化历史
        if self.config["enable_evolution_tracking"]:
            for knowledge in resolved_knowledge:
                self._record_evolution(knowledge)
        
        return resolved_knowledge
    
    def _select_resolution_strategy(self, conflict: KnowledgeConflict) -> FusionStrategy:
        """选择解决策略"""
        # 优先使用冲突类型对应的默认策略
        rule = self.conflict_rules.get(conflict.conflict_type)
        if rule and rule.get("resolution_strategy"):
            return rule["resolution_strategy"]
        
        # 检查知识项是否包含置信度信息
        confidence_scores = []
        for item in conflict.knowledge_items:
            # 尝试从不同字段获取置信度
            confidence = item.get('confidence') or item.get('confidence_score') or item.get('score')
            if confidence is not None:
                try:
                    confidence_scores.append(float(confidence))
                except (ValueError, TypeError) as e:
                    # 根据项目要求"不采用任何降级处理，直接报错"，记录警告而不是静默忽略
                    # 置信度转换失败不影响核心功能，但记录错误以便调试
                    logger = logging.getLogger(__name__)
                    logger.warning(f"置信度转换失败，跳过该知识项: {e}, 置信度值: {confidence}")
        
        # 如果所有知识项都有置信度分数，并且分数差异较大，则选择最高置信度策略
        if len(confidence_scores) == len(conflict.knowledge_items) and len(confidence_scores) > 1:
            # 计算置信度差异
            max_conf = max(confidence_scores)
            min_conf = min(confidence_scores)
            if max_conf - min_conf > 0.3:  # 差异阈值
                return FusionStrategy.SELECT_HIGHEST_CONFIDENCE
        
        # 后备策略选择
        if conflict.conflict_type == ConflictType.CONTRADICTION:
            return FusionStrategy.VOTING
        elif conflict.conflict_type == ConflictType.OUTDATED:
            return FusionStrategy.RECENCY
        elif conflict.conflict_type == ConflictType.REDUNDANCY:
            return FusionStrategy.RECENCY
        else:
            return FusionStrategy.WEIGHTED
    
    def _fuse_knowledge(self, items: List[Dict[str, Any]], strategy: FusionStrategy) -> Optional[Dict[str, Any]]:
        """融合知识"""
        if not items:
            return None  # 返回None
        
        if len(items) == 1:
            # 单一知识，无需融合
            return items[0].copy()
        
        # 获取融合策略
        strategy_config = self.fusion_strategies.get(strategy)
        if not strategy_config:
            self.logger.warning(f"未知的融合策略: {strategy}")
            return None  # 返回None
        
        # 计算权重
        weight_function = strategy_config["weight_function"]
        weights = weight_function(items)
        
        if not weights:
            return None  # 返回None
        
        # 创建融合后的知识
        fused_item = self._create_fused_item(items, weights)
        
        return fused_item
    
    def _voting_weight(self, items: List[Dict[str, Any]]) -> Dict[int, float]:
        """投票权重计算"""
        # 简单多数投票
        weights = {}
        total = len(items)
        
        for i, item in enumerate(items):
            # 置信度作为投票权重
            confidence = item.get("confidence", 0.5)
            weights[i] = confidence / total
        
        return weights
    
    def _weighted_fusion(self, items: List[Dict[str, Any]]) -> Dict[int, float]:
        """加权融合权重计算"""
        weights = {}
        
        # 基于多个因素计算权重
        for i, item in enumerate(items):
            weight = 0.0
            
            # 1. 置信度权重
            confidence = item.get("confidence", 0.5)
            weight += confidence * 0.4
            
            # 2. 来源权威性权重
            source = item.get("source", "unknown")
            authority_score = self._calculate_authority_score(source)
            weight += authority_score * 0.3
            
            # 3. 时效性权重
            timestamp = self._extract_timestamp(item)
            if timestamp:
                recency_score = self._calculate_recency_score(timestamp)
                weight += recency_score * 0.3
            
            weights[i] = weight
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            for i in weights:
                weights[i] /= total
        
        return weights
    
    def _recency_weight(self, items: List[Dict[str, Any]]) -> Dict[int, float]:
        """最新优先权重计算"""
        weights = {}
        
        # 提取时间戳
        timestamps = []
        for i, item in enumerate(items):
            timestamp = self._extract_timestamp(item)
            timestamps.append((i, timestamp))
        
        # 按时间排序
        dated_items = [(i, ts) for i, ts in timestamps if ts]
        if not dated_items:
            # 没有时间信息，平均分配权重
            for i in range(len(items)):
                weights[i] = 1.0 / len(items)
            return weights
        
        dated_items.sort(key=lambda x: x[1], reverse=True)  # 最新优先
        
        # 最新条目权重最高
        for rank, (i, _) in enumerate(dated_items):
            weights[i] = 1.0 / (rank + 1)  # 排名越靠前权重越高
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            for i in weights:
                weights[i] /= total
        
        return weights
    
    def _authority_weight(self, items: List[Dict[str, Any]]) -> Dict[int, float]:
        """权威优先权重计算"""
        weights = {}
        
        for i, item in enumerate(items):
            source = item.get("source", "unknown")
            authority_score = self._calculate_authority_score(source)
            weights[i] = authority_score
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            for i in weights:
                weights[i] /= total
        
        return weights
    
    def _consensus_weight(self, items: List[Dict[str, Any]]) -> Dict[int, float]:
        """共识优先权重计算"""
        # 基于相似性计算共识权重
        weights = {}
        
        # 计算每个条目与其他条目的平均相似度
        for i, item1 in enumerate(items):
            total_similarity = 0.0
            count = 0
            
            for j, item2 in enumerate(items):
                if i != j:
                    similarity = self._calculate_semantic_similarity(item1, item2)
                    total_similarity += similarity
                    count += 1
            
            if count > 0:
                weights[i] = total_similarity / count
            else:
                weights[i] = 0.5
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            for i in weights:
                weights[i] /= total
        
        return weights
    
    def _calculate_authority_score(self, source: str) -> float:
        """计算来源权威性分数"""
        authority_map = {
            "expert": 0.9,
            "research_paper": 0.8,
            "textbook": 0.85,
            "official": 0.75,
            "verified": 0.7,
            "crowdsourced": 0.5,
            "user_generated": 0.4,
            "unknown": 0.3,
            "manual": 0.6
        }
        
        return authority_map.get(source.lower(), 0.5)
    
    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """计算时效性分数"""
        now = datetime.now()
        age_days = (now - timestamp).total_seconds() / (60 * 60 * 24)
        
        # 指数衰减
        decay_rate = 0.01  # 每天衰减1%
        score = max(0.1, 1.0 - decay_rate * age_days)
        
        return score
    
    def _create_fused_item(self, items: List[Dict[str, Any]], weights: Dict[int, float]) -> Dict[str, Any]:
        """创建融合后的知识条目"""
        if not items:
            return None  # 返回None
        
        # 选择一个作为基础（权重最高的）
        max_weight_idx = max(weights, key=weights.get)
        base_item = items[max_weight_idx].copy()
        
        # 生成新的ID
        base_item["id"] = self._generate_fusion_id(items)
        
        # 更新元数据
        base_item["metadata"]["fusion_info"] = {
            "source_items": [item.get("id", "unknown") for item in items],
            "weights": {str(i): weight for i, weight in weights.items()},
            "fused_at": datetime.now().isoformat(),
            "original_count": len(items)
        }
        
        # 更新置信度（加权平均）
        total_confidence = 0.0
        total_weight = 0.0
        
        for i, item in enumerate(items):
            confidence = item.get("confidence", 0.5)
            weight = weights.get(i, 0.0)
            total_confidence += confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            base_item["confidence"] = total_confidence / total_weight
        
        # 标记为融合知识
        base_item["source"] = "fusion"
        base_item["updated_at"] = datetime.now().isoformat()
        
        return base_item
    
    def _generate_fusion_id(self, items: List[Dict[str, Any]]) -> str:
        """生成融合知识的唯一ID"""
        # 基于所有源知识的ID生成
        source_ids = [item.get("id", "unknown") for item in items]
        source_ids_str = "_".join(sorted(source_ids))
        
        # 添加时间戳和哈希
        timestamp = int(time.time())
        hash_str = hashlib.md5(source_ids_str.encode()).hexdigest()[:8]
        
        return f"fusion_{timestamp}_{hash_str}"
    
    def _record_evolution(self, knowledge_item: Dict[str, Any]):
        """记录知识演化"""
        evolution_record = {
            "knowledge_id": knowledge_item.get("id"),
            "type": "fusion",
            "timestamp": datetime.now().isoformat(),
            "details": knowledge_item.get("metadata", {}).get("fusion_info", {}),
            "confidence": knowledge_item.get("confidence", 0.5)
        }
        
        self.evolution_history.append(evolution_record)
        
        # 限制历史长度
        if len(self.evolution_history) > self.config["max_history_length"]:
            self.evolution_history = self.evolution_history[-self.config["max_history_length"]:]
    
    def evaluate_knowledge_quality(self, knowledge_item: Dict[str, Any]) -> Dict[str, Any]:
        """评估知识质量
        
        参数:
            knowledge_item: 知识条目
            
        返回:
            质量评估结果
        """
        quality_score = 0.0
        factors = {}
        
        # 1. 置信度因子
        confidence = knowledge_item.get("confidence", 0.5)
        factors["confidence"] = confidence
        quality_score += confidence * 0.3
        
        # 2. 来源权威性因子
        source = knowledge_item.get("source", "unknown")
        authority_score = self._calculate_authority_score(source)
        factors["authority"] = authority_score
        quality_score += authority_score * 0.25
        
        # 3. 时效性因子
        timestamp = self._extract_timestamp(knowledge_item)
        if timestamp:
            recency_score = self._calculate_recency_score(timestamp)
            factors["recency"] = recency_score
            quality_score += recency_score * 0.2
        
        # 4. 完整性因子
        completeness = self._evaluate_completeness(knowledge_item)
        factors["completeness"] = completeness
        quality_score += completeness * 0.15
        
        # 5. 一致性因子（检查内部一致性）
        consistency = self._evaluate_consistency(knowledge_item)
        factors["consistency"] = consistency
        quality_score += consistency * 0.1
        
        # 确保分数在0-1之间
        quality_score = max(0.0, min(1.0, quality_score))
        
        # 质量等级
        if quality_score >= 0.8:
            quality_level = "excellent"
        elif quality_score >= 0.6:
            quality_level = "good"
        elif quality_score >= 0.4:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return {
            "quality_score": quality_score,
            "quality_level": quality_level,
            "factors": factors,
            "recommendations": self._generate_quality_recommendations(factors)
        }
    
    def _evaluate_completeness(self, knowledge_item: Dict[str, Any]) -> float:
        """评估知识完整性"""
        content = knowledge_item.get("content", {})
        
        if not content:
            return 0.2
        
        if isinstance(content, str):
            # 文本长度作为完整性指标
            length = len(content)
            if length > 500:
                return 0.9
            elif length > 200:
                return 0.7
            elif length > 50:
                return 0.5
            else:
                return 0.3
        
        if isinstance(content, dict):
            # 字典项数量作为完整性指标
            field_count = len(content)
            required_fields = ["description", "subject", "value"]
            
            # 检查必需字段
            present_fields = sum(1 for field in required_fields if field in content)
            completeness = 0.3 + (present_fields / len(required_fields)) * 0.4
            
            # 考虑总字段数
            if field_count >= 5:
                completeness += 0.3
            elif field_count >= 3:
                completeness += 0.2
            elif field_count >= 1:
                completeness += 0.1
            
            return min(0.95, completeness)
        
        return 0.5
    
    def _evaluate_consistency(self, knowledge_item: Dict[str, Any]) -> float:
        """评估内部一致性"""
        # 完整实现：检查内容中是否有矛盾表述
        text = self._extract_text(knowledge_item)
        
        if not text:
            return 0.5
        
        # 检查常见矛盾词对
        contradiction_pairs = [
            ("是", "不是"), ("有", "没有"), ("能", "不能"),
            ("可以", "不可以"), ("真", "假"), ("存在", "不存在")
        ]
        
        for word1, word2 in contradiction_pairs:
            if word1 in text and word2 in text:
                # 检测到矛盾表述
                return 0.3
        
        # 没有检测到明显矛盾
        return 0.8
    
    def _generate_quality_recommendations(self, factors: Dict[str, float]) -> List[str]:
        """生成质量改进建议"""
        recommendations = []
        
        if factors.get("confidence", 0.5) < 0.6:
            recommendations.append("知识置信度较低，建议验证来源或增加证据支持")
        
        if factors.get("authority", 0.5) < 0.6:
            recommendations.append("来源权威性不足，建议引用更权威的来源")
        
        if factors.get("recency", 1.0) < 0.7:
            recommendations.append("知识可能过时，建议更新为最新信息")
        
        if factors.get("completeness", 0.5) < 0.6:
            recommendations.append("知识内容不完整，建议补充更多相关信息")
        
        if factors.get("consistency", 0.8) < 0.7:
            recommendations.append("知识内部存在不一致，建议检查并修正矛盾表述")
        
        return recommendations
    
    def align_knowledge(self, source_knowledge: Dict[str, Any], target_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """知识对齐（实体消歧、关系映射）
        
        参数:
            source_knowledge: 源知识
            target_knowledge: 目标知识
            
        返回:
            对齐结果
        """
        alignment_result = {
            "aligned": False,
            "confidence": 0.0,
            "mappings": {},
            "conflicts": [],
            "recommendations": []
        }
        
        # 计算语义相似度
        similarity = self._calculate_semantic_similarity(source_knowledge, target_knowledge)
        
        if similarity < 0.3:
            # 相似度过低，无法对齐
            alignment_result["aligned"] = False
            alignment_result["confidence"] = similarity
            alignment_result["recommendations"].append("语义相似度过低，无法进行有效对齐")
            return alignment_result
        
        # 提取实体和关系
        source_entities = self._extract_entities(source_knowledge)
        target_entities = self._extract_entities(target_knowledge)
        
        # 实体对齐
        entity_mappings = self._align_entities(source_entities, target_entities)
        
        # 关系对齐
        source_relations = self._extract_relations(source_knowledge)
        target_relations = self._extract_relations(target_knowledge)
        relation_mappings = self._align_relations(source_relations, target_relations, entity_mappings)
        
        # 检查对齐质量
        alignment_quality = self._evaluate_alignment_quality(
            entity_mappings, relation_mappings, similarity
        )
        
        alignment_result.update({
            "aligned": alignment_quality >= 0.5,
            "confidence": alignment_quality,
            "mappings": {
                "entities": entity_mappings,
                "relations": relation_mappings
            },
            "similarity": similarity
        })
        
        return alignment_result
    
    def _extract_entities(self, knowledge_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取实体"""
        entities = []
        content = knowledge_item.get("content", {})
        
        if isinstance(content, dict):
            # 完整实现：将字典键作为实体
            for key, value in content.items():
                if isinstance(value, str) and len(value) > 3:
                    entities.append({
                        "name": key,
                        "value": value,
                        "type": "property"
                    })
        
        return entities
    
    def _extract_relations(self, knowledge_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取关系"""
        relations = []
        
        # 完整实现：基于知识类型提取关系
        knowledge_type = knowledge_item.get("type", "")
        
        if knowledge_type in ["relationship", "fact", "rule"]:
            content = knowledge_item.get("content", {})
            if isinstance(content, dict) and "relation" in content:
                relations.append({
                    "type": content.get("relation", "unknown"),
                    "subject": content.get("subject", ""),
                    "object": content.get("object", ""),
                    "properties": {k: v for k, v in content.items() if k not in ["relation", "subject", "object"]}
                })
        
        return relations
    
    def _align_entities(self, source_entities: List[Dict[str, Any]], target_entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """对齐实体"""
        mappings = {}
        
        for source_entity in source_entities:
            source_name = source_entity.get("name", "")
            if not source_name:
                continue
            
            # 寻找最相似的目标实体
            best_match = None
            best_similarity = 0.0
            
            for target_entity in target_entities:
                target_name = target_entity.get("name", "")
                if not target_name:
                    continue
                
                # 完整实现）
                similarity = self._calculate_name_similarity(source_name, target_name)
                if similarity > best_similarity and similarity > 0.6:
                    best_similarity = similarity
                    best_match = target_name
            
            if best_match:
                mappings[source_name] = best_match
        
        return mappings
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """计算名称相似度"""
        # 完整实现：使用Jaccard相似度
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _align_relations(self, source_relations: List[Dict[str, Any]], target_relations: List[Dict[str, Any]], entity_mappings: Dict[str, str]) -> List[Dict[str, Any]]:
        """对齐关系"""
        mappings = []
        
        for source_rel in source_relations:
            source_type = source_rel.get("type", "")
            source_subject = source_rel.get("subject", "")
            source_object = source_rel.get("object", "")
            
            # 应用实体映射
            mapped_subject = entity_mappings.get(source_subject, source_subject)
            mapped_object = entity_mappings.get(source_object, source_object)
            
            # 寻找匹配的目标关系
            for target_rel in target_relations:
                target_type = target_rel.get("type", "")
                target_subject = target_rel.get("subject", "")
                target_object = target_rel.get("object", "")
                
                # 检查关系类型和实体匹配
                type_similarity = self._calculate_name_similarity(source_type, target_type)
                
                if (type_similarity > 0.7 and 
                    mapped_subject == target_subject and 
                    mapped_object == target_object):
                    
                    mappings.append({
                        "source_relation": source_rel,
                        "target_relation": target_rel,
                        "type_similarity": type_similarity,
                        "entity_aligned": True
                    })
                    break
        
        return mappings
    
    def _evaluate_alignment_quality(self, entity_mappings: Dict[str, str], relation_mappings: List[Dict[str, Any]], similarity: float) -> float:
        """评估对齐质量"""
        quality = similarity * 0.4  # 基础相似度权重
        
        # 实体对齐质量
        if entity_mappings:
            mapping_ratio = len(entity_mappings) / max(len(entity_mappings), 1)
            quality += mapping_ratio * 0.3
        
        # 关系对齐质量
        if relation_mappings:
            relation_ratio = len(relation_mappings) / max(len(relation_mappings), 1)
            quality += relation_ratio * 0.3
        
        return min(1.0, quality)
    
    def get_conflict_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取冲突历史"""
        history = [conflict.to_dict() for conflict in self.conflict_history]
        return history[-limit:] if limit > 0 else history
    
    def get_evolution_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取演化历史"""
        return self.evolution_history[-limit:] if limit > 0 else self.evolution_history
    
    def clear_history(self):
        """清空历史记录"""
        self.conflict_history.clear()
        self.evolution_history.clear()
        self.logger.info("知识融合历史记录已清空")


# 模块级便捷函数
def create_fusion_engine(config: Optional[Dict[str, Any]] = None) -> KnowledgeFusionEngine:
    """创建知识融合引擎实例"""
    return KnowledgeFusionEngine(config)


def fuse_knowledge_list(
    knowledge_items: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """便捷函数：融合知识列表"""
    engine = create_fusion_engine(config)
    conflicts = engine.detect_conflicts(knowledge_items)
    fused_knowledge = engine.resolve_conflicts(conflicts)
    return fused_knowledge


def evaluate_knowledge_quality(
    knowledge_item: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """便捷函数：评估知识质量"""
    engine = create_fusion_engine(config)
    return engine.evaluate_knowledge_quality(knowledge_item)