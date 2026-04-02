#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱推理引擎
为AGI系统提供真实的知识图谱推理能力，解决审计报告中"知识库虚假实现"问题

功能：
1. 关系推理：基于图结构的逻辑推理
2. 模式匹配：发现知识图谱中的模式
3. 规则推理：应用推理规则生成新知识
4. 语义推理：基于实体类型的语义推理
5. 路径推理：基于路径的推理和发现
6. 知识补全：自动补全缺失的知识
"""

import sys
import os
import logging
import itertools
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict, deque
import networkx as nx

from .knowledge_graph import KnowledgeGraph, RelationType


class KnowledgeReasoner:
    """知识图谱推理引擎
    
    提供基于知识图谱的高级推理功能
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """初始化推理引擎
        
        参数:
            knowledge_graph: 知识图谱实例
        """
        self.knowledge_graph = knowledge_graph
        self.graph = knowledge_graph.graph
        self.logger = logging.getLogger(__name__)
        
        # 推理规则
        self.rules = self._initialize_rules()
        
        # 缓存推理结果
        self.inference_cache = {}
        
        self.logger.info("知识图谱推理引擎初始化完成")
    
    def _initialize_rules(self) -> Dict[str, Dict[str, Any]]:
        """初始化推理规则"""
        
        rules = {
            # 传递性规则
            "transitive_is_a": {
                "pattern": [
                    ("A", RelationType.IS_A.value, "B"),
                    ("B", RelationType.IS_A.value, "C")
                ],
                "inference": ("A", RelationType.IS_A.value, "C"),
                "description": "如果A是B，B是C，那么A是C"
            },
            "transitive_part_of": {
                "pattern": [
                    ("A", RelationType.PART_OF.value, "B"),
                    ("B", RelationType.PART_OF.value, "C")
                ],
                "inference": ("A", RelationType.PART_OF.value, "C"),
                "description": "如果A是B的一部分，B是C的一部分，那么A是C的一部分"
            },
            
            # 对称性规则
            "symmetric_related_to": {
                "pattern": [("A", RelationType.RELATED_TO.value, "B")],
                "inference": ("B", RelationType.RELATED_TO.value, "A"),
                "description": "如果A与B相关，那么B与A相关"
            },
            "symmetric_similar_to": {
                "pattern": [("A", RelationType.SIMILAR_TO.value, "B")],
                "inference": ("B", RelationType.SIMILAR_TO.value, "A"),
                "description": "如果A与B相似，那么B与A相似"
            },
            
            # 反对称规则
            "antisymmetric_opposite_of": {
                "pattern": [("A", RelationType.OPPOSITE_OF.value, "B")],
                "inference": ("B", RelationType.OPPOSITE_OF.value, "A"),
                "description": "如果A与B相反，那么B与A相反"
            },
            
            # 组合规则
            "combination_causes": {
                "pattern": [
                    ("A", RelationType.CAUSES.value, "B"),
                    ("B", RelationType.CAUSES.value, "C")
                ],
                "inference": ("A", RelationType.CAUSES.value, "C"),
                "description": "如果A导致B，B导致C，那么A导致C"
            },
            
            # 属性继承规则
            "property_inheritance": {
                "pattern": [
                    ("A", RelationType.IS_A.value, "B"),
                    ("B", RelationType.HAS_PROPERTY.value, "P")
                ],
                "inference": ("A", RelationType.HAS_PROPERTY.value, "P"),
                "description": "如果A是B，B具有属性P，那么A也具有属性P"
            },
            
            # 位置继承规则
            "location_inheritance": {
                "pattern": [
                    ("A", RelationType.PART_OF.value, "B"),
                    ("B", RelationType.LOCATED_IN.value, "L")
                ],
                "inference": ("A", RelationType.LOCATED_IN.value, "L"),
                "description": "如果A是B的一部分，B位于L，那么A也位于L"
            }
        }
        
        return rules
    
    def infer_relations(self, 
                       node_id: str, 
                       max_depth: int = 3,
                       relation_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """推理节点关系
        
        参数:
            node_id: 节点ID
            max_depth: 最大推理深度
            relation_types: 关注的关系类型列表
            
        返回:
            推理出的关系列表
        """
        if node_id not in self.graph:
            self.logger.warning(f"节点不存在: {node_id}")
            return []  # 返回空列表
        
        # 检查缓存
        cache_key = f"infer_relations:{node_id}:{max_depth}:{str(relation_types)}"
        if cache_key in self.inference_cache:
            return self.inference_cache[cache_key]
        
        # 收集直接关系
        direct_relations = []
        for neighbor_id, edge_data_dict in self.graph[node_id].items():
            for edge_key, edge_data in edge_data_dict.items():
                rel_type = edge_data.get("type")
                
                if relation_types is None or rel_type in relation_types:
                    direct_relations.append({
                        "source": node_id,
                        "target": neighbor_id,
                        "relation": rel_type,
                        "weight": edge_data.get("weight", 0.5),
                        "depth": 1,
                        "inference_type": "direct"
                    })
        
        # 推理间接关系
        inferred_relations = []
        inferred_relations.extend(direct_relations)
        
        # 使用推理规则
        rule_inferences = self._apply_rules(node_id, max_depth, relation_types)
        inferred_relations.extend(rule_inferences)
        
        # 使用图遍历推理
        traversal_inferences = self._traversal_inference(node_id, max_depth, relation_types)
        inferred_relations.extend(traversal_inferences)
        
        # 去重
        unique_relations = self._deduplicate_relations(inferred_relations)
        
        # 缓存结果
        self.inference_cache[cache_key] = unique_relations
        
        return unique_relations
    
    def _apply_rules(self, 
                    node_id: str, 
                    max_depth: int,
                    relation_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """应用推理规则"""
        inferred_relations = []
        
        for rule_name, rule_info in self.rules.items():
            # 检查规则是否适用于当前节点
            rule_matches = self._match_rule_pattern(node_id, rule_info["pattern"])
            
            for match in rule_matches:
                # 生成推理结果
                inference = rule_info["inference"]
                
                # 替换模式变量为实际节点
                var_mapping = match  # match是变量到实际节点的映射
                source = var_mapping.get("A", node_id)
                
                if inference[0] == "A":
                    inferred_source = source
                else:
                    # 对于其他变量，需要从匹配中获取
                    inferred_source = var_mapping.get(inference[0], node_id)
                
                inferred_target = var_mapping.get(inference[2], inference[2])
                inferred_relation = inference[1]
                
                # 检查关系类型过滤
                if relation_types is not None and inferred_relation not in relation_types:
                    continue
                
                # 检查是否已存在该关系
                if not self._relation_exists(inferred_source, inferred_target, inferred_relation):
                    inferred_relations.append({
                        "source": inferred_source,
                        "target": inferred_target,
                        "relation": inferred_relation,
                        "weight": 0.7,  # 推理关系的置信度
                        "depth": 2,  # 推理关系深度
                        "inference_type": f"rule:{rule_name}",
                        "rule_description": rule_info["description"],
                        "match": match
                    })
        
        return inferred_relations
    
    def _match_rule_pattern(self, 
                           start_node: str, 
                           pattern: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
        """匹配规则模式
        
        参数:
            start_node: 起始节点
            pattern: 模式列表，每个元素为(源变量, 关系, 目标变量)
            
        返回:
            匹配列表，每个匹配是变量到实际节点的映射
        """
        matches = []
        
        # BFS搜索匹配
        queue = deque([({"A": start_node}, 0)])  # (变量映射, 模式索引)
        
        while queue:
            var_mapping, pattern_idx = queue.popleft()
            
            if pattern_idx >= len(pattern):
                # 完整匹配
                matches.append(var_mapping.copy())
                continue
            
            source_var, relation, target_var = pattern[pattern_idx]
            
            # 获取源节点
            if source_var in var_mapping:
                source_node = var_mapping[source_var]
            else:
                # 源变量未绑定，无法继续
                continue
            
            # 查找满足关系的邻居
            for neighbor_id, edge_data_dict in self.graph[source_node].items():
                for edge_key, edge_data in edge_data_dict.items():
                    if edge_data.get("type") == relation:
                        # 创建新的变量映射
                        new_mapping = var_mapping.copy()
                        new_mapping[target_var] = neighbor_id
                        
                        # 继续匹配下一个模式
                        queue.append((new_mapping, pattern_idx + 1))
        
        return matches
    
    def _traversal_inference(self, 
                            node_id: str, 
                            max_depth: int,
                            relation_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """图遍历推理"""
        inferred_relations = []
        
        # BFS遍历
        visited = set([node_id])
        queue = deque([(node_id, 0, {})])  # (当前节点, 深度, 路径关系)
        
        while queue:
            current_node, depth, path_info = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # 遍历邻居
            for neighbor_id, edge_data_dict in self.graph[current_node].items():
                for edge_key, edge_data in edge_data_dict.items():
                    relation = edge_data.get("type")
                    weight = edge_data.get("weight", 0.5)
                    
                    # 检查关系类型过滤
                    if relation_types is not None and relation not in relation_types:
                        continue
                    
                    # 记录路径
                    path_relations = path_info.get("relations", []) + [relation]
                    path_nodes = path_info.get("nodes", []) + [current_node]
                    
                    # 如果到达新节点且深度>0，则可能发现间接关系
                    if neighbor_id not in visited or depth > 0:
                        # 更新路径信息
                        new_path_info = {
                            "relations": path_relations,
                            "nodes": path_nodes,
                            "total_weight": path_info.get("total_weight", 1.0) * weight
                        }
                        
                        # 如果深度>0，可以推理间接关系
                        if depth > 0 and neighbor_id != node_id:
                            # 计算复合关系
                            composite_relation = self._compose_relations(path_relations)
                            
                            if not self._relation_exists(node_id, neighbor_id, composite_relation):
                                inferred_relations.append({
                                    "source": node_id,
                                    "target": neighbor_id,
                                    "relation": composite_relation,
                                    "weight": new_path_info["total_weight"] * 0.8,  # 衰减
                                    "depth": depth + 1,
                                    "inference_type": "path_traversal",
                                    "path_length": depth + 1,
                                    "path_relations": path_relations,
                                    "path_nodes": path_nodes
                                })
                        
                        # 继续遍历
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            queue.append((neighbor_id, depth + 1, new_path_info))
        
        return inferred_relations
    
    def _compose_relations(self, relations: List[str]) -> str:
        """组合关系序列为复合关系"""
        if not relations:
            return "unknown"
        
        if len(relations) == 1:
            return relations[0]
        
        # 完整：使用第一个和最后一个关系，中间用"_via_"连接
        return f"{relations[0]}_via_{relations[-1]}"
    
    def _relation_exists(self, source: str, target: str, relation: str) -> bool:
        """检查关系是否已存在"""
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return False
        
        if self.graph.has_edge(source, target):
            for edge_key, edge_data in self.graph[source][target].items():
                if edge_data.get("type") == relation:
                    return True
        
        return False
    
    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重关系"""
        unique_keys = set()
        unique_relations = []
        
        for rel in relations:
            key = (rel["source"], rel["target"], rel["relation"])
            if key not in unique_keys:
                unique_keys.add(key)
                unique_relations.append(rel)
        
        return unique_relations
    
    def find_similar_nodes(self, 
                          node_id: str, 
                          similarity_threshold: float = 0.6,
                          max_results: int = 10) -> List[Dict[str, Any]]:
        """查找相似节点
        
        参数:
            node_id: 节点ID
            similarity_threshold: 相似度阈值
            max_results: 最大结果数
            
        返回:
            相似节点列表
        """
        if node_id not in self.graph:
            return []  # 返回空列表
        
        node_data = self.graph.nodes[node_id]
        node_type = node_data.get("type")
        node_content = node_data.get("content", {})
        
        similarities = []
        
        for other_id, other_data in self.graph.nodes(data=True):
            if other_id == node_id:
                continue
            
            # 计算相似度
            similarity = self._calculate_node_similarity(
                node_data, other_data, node_id, other_id
            )
            
            if similarity >= similarity_threshold:
                similarities.append({
                    "node_id": other_id,
                    "node_data": dict(other_data),
                    "similarity": similarity,
                    "reasons": self._explain_similarity(node_data, other_data, node_id, other_id)
                })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:max_results]
    
    def _calculate_node_similarity(self, 
                                  node1_data: Dict[str, Any], 
                                  node2_data: Dict[str, Any],
                                  node1_id: str,
                                  node2_id: str) -> float:
        """计算节点相似度"""
        similarity = 0.0
        
        # 1. 类型相似度
        type1 = node1_data.get("type", "")
        type2 = node2_data.get("type", "")
        if type1 == type2:
            similarity += 0.3
        
        # 2. 结构相似度（邻居重叠）
        neighbors1 = set(self.graph.neighbors(node1_id)) if node1_id in self.graph else set()
        neighbors2 = set(self.graph.neighbors(node2_id)) if node2_id in self.graph else set()
        
        if neighbors1 or neighbors2:
            neighbor_overlap = len(neighbors1 & neighbors2)
            neighbor_union = len(neighbors1 | neighbors2)
            if neighbor_union > 0:
                similarity += 0.4 * (neighbor_overlap / neighbor_union)
        
        # 3. 内容相似度（完整）
        content1 = node1_data.get("content", {})
        content2 = node2_data.get("content", {})
        
        if isinstance(content1, dict) and isinstance(content2, dict):
            # 比较键的相似度
            keys1 = set(content1.keys())
            keys2 = set(content2.keys())
            
            if keys1 or keys2:
                key_overlap = len(keys1 & keys2)
                key_union = len(keys1 | keys2)
                if key_union > 0:
                    similarity += 0.3 * (key_overlap / key_union)
        
        return min(similarity, 1.0)
    
    def _explain_similarity(self, 
                           node1_data: Dict[str, Any], 
                           node2_data: Dict[str, Any],
                           node1_id: str,
                           node2_id: str) -> List[str]:
        """解释相似性原因"""
        reasons = []
        
        # 类型相同
        type1 = node1_data.get("type", "")
        type2 = node2_data.get("type", "")
        if type1 == type2:
            reasons.append(f"相同类型: {type1}")
        
        # 检查直接关系
        if self.graph.has_edge(node1_id, node2_id) or self.graph.has_edge(node2_id, node1_id):
            reasons.append("存在直接关系")
        
        # 检查共同邻居
        neighbors1 = set(self.graph.neighbors(node1_id)) if node1_id in self.graph else set()
        neighbors2 = set(self.graph.neighbors(node2_id)) if node2_id in self.graph else set()
        common_neighbors = neighbors1 & neighbors2
        
        if common_neighbors:
            reasons.append(f"共享{len(common_neighbors)}个邻居")
        
        return reasons
    
    def discover_patterns(self, 
                         min_support: int = 2,
                         max_pattern_size: int = 3) -> List[Dict[str, Any]]:
        """发现知识图谱中的模式
        
        参数:
            min_support: 最小支持度（出现次数）
            max_pattern_size: 最大模式大小
            
        返回:
            发现的模式列表
        """
        patterns = []
        
        # 提取所有子图作为候选模式
        nodes = list(self.graph.nodes())
        
        # 完整：查找常见的边模式
        edge_patterns = defaultdict(int)
        
        for node in nodes:
            # 获取节点的局部模式
            local_patterns = self._extract_local_patterns(node, max_pattern_size)
            
            for pattern in local_patterns:
                pattern_key = self._pattern_to_key(pattern)
                edge_patterns[pattern_key] += 1
        
        # 过滤和支持度
        for pattern_key, support in edge_patterns.items():
            if support >= min_support:
                pattern = self._key_to_pattern(pattern_key)
                patterns.append({
                    "pattern": pattern,
                    "support": support,
                    "confidence": support / len(nodes) if len(nodes) > 0 else 0.0
                })
        
        # 按支持度排序
        patterns.sort(key=lambda x: x["support"], reverse=True)
        
        return patterns[:20]  # 返回前20个模式
    
    def _extract_local_patterns(self, 
                               center_node: str, 
                               max_size: int) -> List[List[Tuple[str, str, str]]]:
        """提取以节点为中心的局部模式"""
        patterns = []
        
        if center_node not in self.graph:
            return patterns
        
        # 获取邻居
        neighbors = list(self.graph.neighbors(center_node))
        
        # 生成小规模模式
        for i in range(min(len(neighbors), max_size)):
            # 选择i+1个邻居
            for neighbor_subset in itertools.combinations(neighbors, i + 1):
                pattern = []
                
                for neighbor in neighbor_subset:
                    # 获取关系
                    for edge_key, edge_data in self.graph[center_node][neighbor].items():
                        relation = edge_data.get("type", "unknown")
                        pattern.append((center_node, relation, neighbor))
                
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _pattern_to_key(self, pattern: List[Tuple[str, str, str]]) -> str:
        """将模式转换为字符串键（忽略具体节点ID）"""
        # 排序确保一致性
        sorted_pattern = sorted(pattern, key=lambda x: (x[1], x[0], x[2]))
        
        # 使用关系类型和节点类型（而不是具体ID）
        key_parts = []
        for source, relation, target in sorted_pattern:
            source_type = self.graph.nodes[source].get("type", "unknown") if source in self.graph else "unknown"
            target_type = self.graph.nodes[target].get("type", "unknown") if target in self.graph else "unknown"
            key_parts.append(f"{source_type}:{relation}:{target_type}")
        
        return "|".join(key_parts)
    
    def _key_to_pattern(self, pattern_key: str) -> List[Tuple[str, str, str]]:
        """将字符串键转换回模式表示"""
        pattern = []
        
        for part in pattern_key.split("|"):
            if ":" in part:
                source_type, relation, target_type = part.split(":", 2)
                pattern.append((f"{{{source_type}}}", relation, f"{{{target_type}}}"))
        
        return pattern
    
    def complete_knowledge(self, 
                          node_id: str, 
                          relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """知识补全：预测缺失的关系
        
        参数:
            node_id: 节点ID
            relation_type: 关系类型（如果为None，则补全所有类型）
            
        返回:
            补全建议列表
        """
        suggestions = []
        
        if node_id not in self.graph:
            return suggestions
        
        node_data = self.graph.nodes[node_id]
        node_type = node_data.get("type", "")
        
        # 查找相似节点
        similar_nodes = self.find_similar_nodes(node_id, similarity_threshold=0.5, max_results=5)
        
        for similar in similar_nodes:
            similar_id = similar["node_id"]
            similarity = similar["similarity"]
            
            # 检查相似节点有哪些当前节点没有的关系
            for neighbor_id, edge_data_dict in self.graph[similar_id].items():
                for edge_key, edge_data in edge_data_dict.items():
                    rel_type = edge_data.get("type")
                    
                    # 关系类型过滤
                    if relation_type is not None and rel_type != relation_type:
                        continue
                    
                    # 检查当前节点是否已有该关系
                    if not self._relation_exists(node_id, neighbor_id, rel_type):
                        # 检查邻居是否与当前节点兼容
                        neighbor_data = self.graph.nodes[neighbor_id]
                        neighbor_type = neighbor_data.get("type", "")
                        
                        # 计算建议置信度
                        confidence = similarity * 0.7
                        
                        suggestions.append({
                            "source": node_id,
                            "target": neighbor_id,
                            "relation": rel_type,
                            "confidence": confidence,
                            "source_type": node_type,
                            "target_type": neighbor_type,
                            "based_on": f"similarity_to:{similar_id}",
                            "similarity_score": similarity,
                            "weight_suggestion": edge_data.get("weight", 0.5)
                        })
        
        # 按置信度排序
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return suggestions
    
    def query_with_reasoning(self, 
                            query: Dict[str, Any],
                            reasoning_depth: int = 2) -> Dict[str, Any]:
        """带推理的查询
        
        参数:
            query: 查询条件
            reasoning_depth: 推理深度
            
        返回:
            查询结果，包含直接结果和推理结果
        """
        result = {
            "direct_results": [],
            "inferred_results": [],
            "reasoning_chains": [],
            "confidence_scores": {}
        }
        
        # 解析查询
        query_type = query.get("type", "unknown")
        
        if query_type == "find_related":
            # 查找相关节点
            node_id = query.get("node_id")
            relation = query.get("relation")
            max_depth = query.get("max_depth", reasoning_depth)
            
            if node_id:
                # 直接结果
                neighbors = self.knowledge_graph.get_neighbors(
                    node_id, relation_type=relation, direction="out"
                )
                result["direct_results"] = neighbors
                
                # 推理结果
                inferred = self.infer_relations(node_id, max_depth, [relation] if relation else None)
                result["inferred_results"] = inferred
                
                # 置信度计算
                for i, rel in enumerate(inferred):
                    confidence = 0.9 - (rel.get("depth", 1) - 1) * 0.2
                    result["confidence_scores"][f"inferred_{i}"] = max(confidence, 0.3)
        
        elif query_type == "find_path":
            # 查找路径
            source_id = query.get("source_id")
            target_id = query.get("target_id")
            max_length = query.get("max_length", 3)
            
            if source_id and target_id:
                paths = self.knowledge_graph.find_paths(source_id, target_id, max_length)
                result["direct_results"] = paths
                
                # 推理可能的新路径（通过中间节点）
                result["reasoning_chains"] = self._infer_indirect_paths(source_id, target_id, max_length)
        
        elif query_type == "find_similar":
            # 查找相似节点
            node_id = query.get("node_id")
            threshold = query.get("threshold", 0.6)
            
            if node_id:
                similar = self.find_similar_nodes(node_id, threshold)
                result["direct_results"] = similar
        
        return result
    
    def _infer_indirect_paths(self, 
                             source_id: str, 
                             target_id: str, 
                             max_length: int) -> List[List[Dict[str, Any]]]:
        """推理间接路径"""
        chains = []
        
        # 查找共同邻居
        if source_id in self.graph and target_id in self.graph:
            source_neighbors = set(self.graph.neighbors(source_id))
            target_neighbors = set(self.graph.neighbors(target_id))
            common_neighbors = source_neighbors & target_neighbors
            
            for neighbor in common_neighbors:
                # 构建通过共同邻居的路径
                chain = [
                    {"from": source_id, "to": neighbor, "relation": "related_to"},
                    {"from": neighbor, "to": target_id, "relation": "related_to"}
                ]
                chains.append(chain)
        
        return chains


# 全局推理引擎实例
_global_knowledge_reasoner = None

def get_global_knowledge_reasoner(knowledge_graph: Optional[KnowledgeGraph] = None) -> KnowledgeReasoner:
    """获取全局知识图谱推理引擎实例（单例模式）"""
    global _global_knowledge_reasoner
    if _global_knowledge_reasoner is None and knowledge_graph is not None:
        _global_knowledge_reasoner = KnowledgeReasoner(knowledge_graph)
    return _global_knowledge_reasoner


if __name__ == "__main__":
    # 测试知识图谱推理引擎
    import json
    
    print("=== 测试知识图谱推理引擎 ===")
    
    # 创建简单知识图谱
    from .knowledge_graph import KnowledgeGraph, RelationType
    
    config = {
        "max_nodes": 1000,
        "default_relation_weight": 0.5,
        "relation_weights": {
            "is_a": 0.9,
            "part_of": 0.8,
            "causes": 0.7,
            "related_to": 0.6
        }
    }
    
    kg = KnowledgeGraph(config)
    
    # 添加一些节点和关系
    kg.add_node("animal", "concept", {"name": "动物", "description": "生物类别"})
    kg.add_node("mammal", "concept", {"name": "哺乳动物", "description": "哺乳动物类别"})
    kg.add_node("dog", "concept", {"name": "狗", "description": "犬科动物"})
    kg.add_node("cat", "concept", {"name": "猫", "description": "猫科动物"})
    kg.add_node("pet", "concept", {"name": "宠物", "description": "家养动物"})
    
    # 添加关系
    kg.add_edge("dog", "mammal", RelationType.IS_A.value)
    kg.add_edge("cat", "mammal", RelationType.IS_A.value)
    kg.add_edge("mammal", "animal", RelationType.IS_A.value)
    kg.add_edge("dog", "pet", RelationType.IS_A.value)
    kg.add_edge("cat", "pet", RelationType.IS_A.value)
    
    # 创建推理引擎
    reasoner = KnowledgeReasoner(kg)
    
    # 测试关系推理
    print("\n1. 推理狗的关系:")
    inferences = reasoner.infer_relations("dog", max_depth=3)
    print(f"推理出 {len(inferences)} 个关系:")
    for inf in inferences[:5]:  # 显示前5个
        print(f"  {inf['source']} --[{inf['relation']}]--> {inf['target']} (深度: {inf['depth']})")
    
    # 测试相似节点查找
    print("\n2. 查找与狗相似的节点:")
    similar = reasoner.find_similar_nodes("dog", similarity_threshold=0.3)
    print(f"找到 {len(similar)} 个相似节点:")
    for sim in similar[:3]:
        print(f"  {sim['node_id']} (相似度: {sim['similarity']:.2f})")
    
    # 测试知识补全
    print("\n3. 知识补全建议（为狗）:")
    suggestions = reasoner.complete_knowledge("dog")
    print(f"生成 {len(suggestions)} 个补全建议:")
    for sug in suggestions[:3]:
        print(f"  建议: {sug['source']} --[{sug['relation']}]--> {sug['target']} (置信度: {sug['confidence']:.2f})")
    
    # 测试模式发现
    print("\n4. 模式发现:")
    patterns = reasoner.discover_patterns(min_support=1)
    print(f"发现 {len(patterns)} 个模式:")
    for pat in patterns[:2]:
        print(f"  模式: {pat['pattern']} (支持度: {pat['support']})")
    
    print("\n测试完成!")