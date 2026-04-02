"""
知识管理器

知识库系统的核心管理类，协调知识存储、检索、更新和验证等功能。
"""

import json
import csv
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Any, Optional, Union
from .knowledge_versioning import get_version_controller
from datetime import datetime
from enum import Enum

import torch
from ..embeddings import FromScratchTextEmbedder

from .knowledge_store import KnowledgeStore
from .knowledge_retriever import KnowledgeRetriever
from .knowledge_graph import KnowledgeGraph
from .knowledge_validator import KnowledgeValidator


class KnowledgeType(Enum):
    """知识类型枚举"""

    FACT = "fact"  # 事实
    RULE = "rule"  # 规则
    PROCEDURE = "procedure"  # 过程/方法
    CONCEPT = "concept"  # 概念
    RELATIONSHIP = "relationship"  # 关系
    EVENT = "event"  # 事件
    PROBLEM_SOLUTION = "problem_solution"  # 问题解决方案
    EXPERIENCE = "experience"  # 经验


class KnowledgeManager:
    """知识管理器

    功能：
    - 管理知识库的整体生命周期
    - 协调各个知识组件的工作
    - 提供统一的API接口
    - 处理知识的增删改查
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, memory_system=None):
        """初始化知识管理器

        参数:
            config: 配置字典
            memory_system: 可选的MemorySystem实例，传递给知识图谱
        """
        self.logger = logging.getLogger(__name__)
        self.memory_system = memory_system

        # 默认配置 - 工业级AGI从零开始实现
        default_config = {
            "embedding_model": "from_scratch_text_embedder",  # 从零开始的文本嵌入器（无预训练模型）
            "embedding_dim": 384,
            "vocab_size": 10000,
            "max_length": 512,
            "vector_store_path": "data/knowledge_vectors.pkl",
            "knowledge_db_path": "data/knowledge.db",
            "max_knowledge_items": 10000,
            "similarity_threshold": 0.7,
            "enable_knowledge_graph": True,
            "enable_validation": True,
            "industrial_mode": True,  # 工业级模式
        }
        
        # 合并配置：用户配置覆盖默认配置
        if config is None:
            self.config = default_config
        else:
            self.config = {**default_config, **config}

        # 初始化组件
        self.store = KnowledgeStore(self.config)
        
        # 初始化查询缓存
        self._query_cache = {}  # 缓存查询结果
        self._cache_ttl = 300  # 缓存有效期（秒）
        
        self.retriever = KnowledgeRetriever(self.config)
        self.graph = (
            KnowledgeGraph(self.config, self.memory_system)
            if self.config["enable_knowledge_graph"]
            else None
        )
        self.validator = (
            KnowledgeValidator(self.config)
            if self.config["enable_validation"]
            else None
        )

        # 句子嵌入模型（用于语义搜索）
        self.embedding_model = None
        self._init_embedding_model()

        # 知识统计
        self.stats = {"total_knowledge": 0, "by_type": {}, "last_updated": None}

        self.logger.info("知识管理器初始化完成")

    def _init_embedding_model(self):
        """初始化从零开始的句子嵌入模型 - 工业级AGI实现"""
        try:
            # 使用从零开始的文本嵌入器，不使用任何预训练模型
            self.embedding_model = FromScratchTextEmbedder(
                embedding_dim=self.config.get("embedding_dim", 384),
                vocab_size=self.config.get("vocab_size", 10000),
                max_length=self.config.get("max_length", 512)
            )
            self.logger.info(f"加载从零开始的文本嵌入器: embedding_dim={self.config.get('embedding_dim', 384)}")
        except Exception as e:
            self.logger.warning(
                f"无法加载从零开始的文本嵌入器: {e}"
            )
            self.embedding_model = None

    def add_knowledge(
        self,
        knowledge_type: Union[KnowledgeType, str],
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """添加知识

        参数:
            knowledge_type: 知识类型
            content: 知识内容
            metadata: 元数据
            validate: 是否验证知识

        返回:
            添加结果
        """
        if isinstance(knowledge_type, str):
            knowledge_type = KnowledgeType(knowledge_type)

        # 准备知识条目
        knowledge_item = {
            "id": self._generate_id(),
            "type": knowledge_type.value,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "confidence": 1.0,  # 初始置信度
            "source": "manual",  # 来源
        }

        # 验证知识（如果启用）
        if validate and self.validator:
            validation_result = self.validator.validate(knowledge_item)
            if not validation_result["valid"]:
                self.logger.warning(f"知识验证失败: {validation_result['errors']}")
                knowledge_item["validation_errors"] = validation_result["errors"]
                knowledge_item["confidence"] *= 0.5  # 降低置信度

        # 生成嵌入向量
        embedding = self._generate_embedding(knowledge_item)
        if embedding is not None:
            knowledge_item["embedding"] = embedding

        # 存储知识
        store_result = self.store.add(knowledge_item)

        if store_result["success"]:
            # 更新知识图谱（如果启用）
            if self.graph:
                self.graph.add_node(knowledge_item["id"], knowledge_type.value, content)

            # 更新统计
            self._update_stats(knowledge_type.value, added=True)

            self.logger.info(
                f"添加知识成功: ID={knowledge_item['id']}, 类型={knowledge_type.value}"
            )

        return store_result

    def query_knowledge(
        self,
        query: str,
        knowledge_type: Optional[Union[KnowledgeType, str]] = None,
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """查询知识

        参数:
            query: 查询文本
            knowledge_type: 知识类型过滤
            limit: 返回结果数量
            similarity_threshold: 相似度阈值

        返回:
            知识条目列表
        """
        if isinstance(knowledge_type, str):
            knowledge_type = KnowledgeType(knowledge_type)

        # 生成查询嵌入
        query_embedding = self._generate_query_embedding(query)

        # 检索知识
        results = self.retriever.retrieve(
            query_embedding=query_embedding,
            knowledge_type=knowledge_type.value if knowledge_type else None,
            limit=limit,
            similarity_threshold=similarity_threshold
            or self.config["similarity_threshold"],
        )

        return results

    def search(self, query: str, limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """搜索知识库
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            threshold: 相似度阈值
            
        Returns:
            知识项列表
        """
        try:
            self.logger.info(f"搜索知识库: {query}")
            
            # 生成缓存键
            import time
            cache_key = f"search_{query}_{limit}_{threshold}"
            
            # 检查缓存
            if cache_key in self._query_cache:
                cache_entry = self._query_cache[cache_key]
                if time.time() - cache_entry["timestamp"] < self._cache_ttl:
                    self.logger.info(f"使用缓存结果: {query}")
                    return cache_entry["results"]
                else:
                    # 缓存过期，删除
                    del self._query_cache[cache_key]
            
            # 执行向量搜索
            vector_results = self.store.search_by_vector(query, limit=limit, threshold=threshold)
            
            # 执行关键词搜索
            keyword_results = self.store.search_by_keyword(query, limit=limit)
            
            # 合并结果，去重
            seen_ids = set()
            combined_results = []
            
            # 先添加向量搜索结果（相似度更高）
            for result in vector_results:
                if result["id"] not in seen_ids:
                    combined_results.append(result)
                    seen_ids.add(result["id"])
            
            # 添加关键词搜索结果（补充）
            for result in keyword_results:
                if result["id"] not in seen_ids:
                    combined_results.append(result)
                    seen_ids.add(result["id"])
            
            # 按相关性排序（向量搜索结果优先）
            combined_results.sort(
                key=lambda x: x.get("similarity", 0) if x.get("similarity") else 0,
                reverse=True
            )
            
            # 限制结果数量
            combined_results = combined_results[:limit]
            
            # 缓存结果
            self._query_cache[cache_key] = {
                "timestamp": time.time(),
                "results": combined_results
            }
            
            # 清理过期缓存（每100次搜索清理一次）
            if len(self._query_cache) > 100:
                self._clean_expired_cache()
            
            self.logger.info(f"搜索完成，找到 {len(combined_results)} 个结果")
            return combined_results
            
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            raise

    def get_knowledge_by_id(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取知识

        参数:
            knowledge_id: 知识ID

        返回:
            知识条目或None
        """
        return self.store.get(knowledge_id)

    def update_knowledge(
        self, knowledge_id: str, updates: Dict[str, Any], validate: bool = True
    ) -> Dict[str, Any]:
        """更新知识

        参数:
            knowledge_id: 知识ID
            updates: 更新内容
            validate: 是否验证更新

        返回:
            更新结果
        """
        # 获取现有知识
        existing = self.store.get(knowledge_id)
        if not existing:
            return {"success": False, "error": f"知识不存在: {knowledge_id}"}

        # 准备更新
        updated_item = existing.copy()
        for key, value in updates.items():
            if key in ["content", "metadata"]:
                updated_item[key] = {**updated_item.get(key, {}), **value}
            elif key not in ["id", "created_at"]:
                updated_item[key] = value

        updated_item["updated_at"] = datetime.now().isoformat()

        # 重新验证（如果启用）
        if validate and self.validator:
            validation_result = self.validator.validate(updated_item)
            if not validation_result["valid"]:
                self.logger.warning(f"知识更新验证失败: {validation_result['errors']}")
                updated_item["validation_errors"] = validation_result["errors"]
                updated_item["confidence"] *= 0.8  # 稍微降低置信度

        # 重新生成嵌入
        embedding = self._generate_embedding(updated_item)
        if embedding is not None:
            updated_item["embedding"] = embedding

        # 更新存储
        update_result = self.store.update(knowledge_id, updated_item)

        if update_result["success"] and self.graph:
            # 更新知识图谱
            self.graph.update_node(
                knowledge_id, updated_item["type"], updated_item["content"]
            )

        return update_result

    def delete_knowledge(self, knowledge_id: str) -> Dict[str, Any]:
        """删除知识

        参数:
            knowledge_id: 知识ID

        返回:
            删除结果
        """
        delete_result = self.store.delete(knowledge_id)

        if delete_result["success"]:
            # 从知识图谱中删除
            if self.graph:
                self.graph.remove_node(knowledge_id)

            # 更新统计
            knowledge_type = delete_result.get("knowledge_type")
            if knowledge_type:
                self._update_stats(knowledge_type, added=False)

        return delete_result

    def search_similar(
        self, knowledge_item: Dict[str, Any], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索相似知识

        参数:
            knowledge_item: 参考知识条目
            limit: 返回结果数量

        返回:
            相似知识列表
        """
        embedding = knowledge_item.get("embedding")
        if embedding is None:
            embedding = self._generate_embedding(knowledge_item)

        if embedding is None:
            return []  # 返回空列表

        return self.retriever.retrieve_by_embedding(
            embedding=embedding, limit=limit, exclude_id=knowledge_item.get("id")
        )

    def infer(
        self, premises: List[Dict[str, Any]], inference_type: str = "logical"
    ) -> List[Dict[str, Any]]:
        """知识推理

        参数:
            premises: 前提知识列表
            inference_type: 推理类型 (logical, causal, etc.)

        返回:
            推理结果列表
        """
        if not self.graph:
            return []  # 返回空列表

        # 使用知识图谱进行推理
        inference_results = self.graph.infer(premises, inference_type)

        return inference_results

    def validate_knowledge_base(self) -> Dict[str, Any]:
        """验证整个知识库

        返回:
            验证结果
        """
        if not self.validator:
            return {"valid": True, "warnings": [], "errors": []}

        all_knowledge = self.store.get_all()
        validation_results = []

        for knowledge in all_knowledge:
            result = self.validator.validate(knowledge)
            validation_results.append(
                {
                    "id": knowledge["id"],
                    "type": knowledge["type"],
                    "valid": result["valid"],
                    "errors": result.get("errors", []),
                    "warnings": result.get("warnings", []),
                }
            )

        # 汇总结果
        invalid_count = sum(1 for r in validation_results if not r["valid"])
        warning_count = sum(len(r["warnings"]) for r in validation_results)

        return {
            "total": len(validation_results),
            "valid": len(validation_results) - invalid_count,
            "invalid": invalid_count,
            "warnings": warning_count,
            "details": validation_results,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息

        返回:
            统计信息
        """
        # 从存储中获取最新统计
        all_knowledge = self.store.get_all()
        self.stats["total_knowledge"] = len(all_knowledge)

        # 按类型统计
        type_counts = {}
        for knowledge in all_knowledge:
            k_type = knowledge["type"]
            type_counts[k_type] = type_counts.get(k_type, 0) + 1

        self.stats["by_type"] = type_counts
        self.stats["last_updated"] = datetime.now().isoformat()

        return self.stats

    def export_knowledge(self, filepath: str, format: str = "json") -> bool:
        """导出知识库

        参数:
            filepath: 导出文件路径
            format: 导出格式 (json, csv, tsv, xml)

        返回:
            是否成功
        """
        all_knowledge = self.store.get_all()
        
        if not all_knowledge:
            self.logger.warning("没有知识数据可导出")
            return False

        try:
            if format == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(all_knowledge, f, ensure_ascii=False, indent=2)
                return True
                
            elif format in ["csv", "tsv"]:
                delimiter = "," if format == "csv" else "\t"
                
                # 提取所有可能的字段
                fieldnames = set()
                for item in all_knowledge:
                    fieldnames.update(item.keys())
                    if "metadata" in item and isinstance(item["metadata"], dict):
                        fieldnames.update(f"metadata_{k}" for k in item["metadata"].keys())
                
                fieldnames = sorted(fieldnames)
                
                with open(filepath, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
                    writer.writeheader()
                    
                    for item in all_knowledge:
                        row = item.copy()
                        
                        # 处理metadata字段
                        if "metadata" in row and isinstance(row["metadata"], dict):
                            metadata = row.pop("metadata")
                            for key, value in metadata.items():
                                row[f"metadata_{key}"] = str(value) if value is not None else ""
                        
                        # 确保所有字段都有值
                        for field in fieldnames:
                            if field not in row:
                                row[field] = ""
                            elif not isinstance(row[field], str):
                                row[field] = str(row[field])
                        
                        writer.writerow(row)
                
                return True
                
            elif format == "xml":
                # 创建XML根元素
                root = ET.Element("knowledge_base")
                root.set("export_time", datetime.now().isoformat())
                root.set("item_count", str(len(all_knowledge)))
                
                for item in all_knowledge:
                    item_elem = ET.SubElement(root, "knowledge_item")
                    
                    for key, value in item.items():
                        if key == "metadata" and isinstance(value, dict):
                            metadata_elem = ET.SubElement(item_elem, "metadata")
                            for meta_key, meta_value in value.items():
                                meta_elem = ET.SubElement(metadata_elem, meta_key)
                                meta_elem.text = str(meta_value) if meta_value is not None else ""
                        else:
                            elem = ET.SubElement(item_elem, key)
                            elem.text = str(value) if value is not None else ""
                
                tree = ET.ElementTree(root)
                tree.write(filepath, encoding="utf-8", xml_declaration=True)
                return True
                
            else:
                self.logger.warning(f"不支持的导出格式: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"导出知识库失败: {e}")
            return False

    def import_knowledge(self, filepath: str, format: str = "json") -> Dict[str, Any]:
        """导入知识库

        参数:
            filepath: 导入文件路径
            format: 导入格式 (json, csv, tsv, xml)

        返回:
            导入结果
        """
        try:
            knowledge_items = []
            
            if format == "json":
                with open(filepath, "r", encoding="utf-8") as f:
                    knowledge_items = json.load(f)
                    
            elif format in ["csv", "tsv"]:
                delimiter = "," if format == "csv" else "\t"
                
                with open(filepath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    knowledge_items = list(reader)
                
                # 转换CSV/TSV数据回原来的格式
                for item in knowledge_items:
                    # 分离metadata字段
                    metadata = {}
                    keys_to_remove = []
                    
                    for key, value in item.items():
                        if key.startswith("metadata_"):
                            metadata_key = key[len("metadata_"):]
                            if value:
                                metadata[metadata_key] = value
                            keys_to_remove.append(key)
                    
                    # 移除metadata_前缀的字段
                    for key in keys_to_remove:
                        del item[key]
                    
                    # 添加metadata字段
                    if metadata:
                        item["metadata"] = metadata
                    
                    # 确保必要字段存在
                    if "type" not in item:
                        item["type"] = "fact"
                    if "content" not in item:
                        item["content"] = ""
                    
            elif format == "xml":
                tree = ET.parse(filepath)
                root = tree.getroot()
                
                for item_elem in root.findall("knowledge_item"):
                    item = {}
                    
                    for child in item_elem:
                        if child.tag == "metadata":
                            metadata = {}
                            for meta_elem in child:
                                metadata[meta_elem.tag] = meta_elem.text if meta_elem.text is not None else ""
                            item["metadata"] = metadata
                        else:
                            item[child.tag] = child.text if child.text is not None else ""
                    
                    # 确保必要字段存在
                    if "type" not in item:
                        item["type"] = "fact"
                    if "content" not in item:
                        item["content"] = ""
                    
                    knowledge_items.append(item)
                    
            else:
                return {"success": False, "error": f"不支持的格式: {format}"}
            
            # 导入知识项
            added_count = 0
            for item in knowledge_items:
                result = self.add_knowledge(
                    knowledge_type=item.get("type", "fact"),
                    content=item.get("content", ""),
                    metadata=item.get("metadata", {}),
                    validate=False,  # 导入时不验证，加快速度
                )
                if result["success"]:
                    added_count += 1

            return {
                "success": True,
                "imported": added_count,
                "total": len(knowledge_items),
                "format": format,
            }
            
        except FileNotFoundError:
            return {"success": False, "error": f"文件不存在: {filepath}"}
        except Exception as e:
            self.logger.error(f"导入知识库失败: {e}")
            return {"success": False, "error": str(e)}

    def _generate_id(self) -> str:
        """生成知识ID"""
        import uuid

        return str(uuid.uuid4())

    def _generate_embedding(
        self, knowledge_item: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """生成知识嵌入向量"""
        if not self.embedding_model:
            return None  # 返回None

        try:
            # 从知识内容中提取文本
            text = self._extract_text_from_knowledge(knowledge_item)
            if not text:
                return None  # 返回None

            # 生成嵌入
            embedding = self.embedding_model.encode(text, convert_to_tensor=True)
            return embedding
        except Exception as e:
            self.logger.warning(f"生成嵌入失败: {e}")
            return None  # 返回None

    def _generate_query_embedding(self, query: str) -> Optional[torch.Tensor]:
        """生成查询嵌入向量"""
        if not self.embedding_model:
            return None  # 返回None

        try:
            embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            return embedding
        except Exception as e:
            self.logger.warning(f"生成查询嵌入失败: {e}")
            return None  # 返回None

    def _extract_text_from_knowledge(self, knowledge_item: Dict[str, Any]) -> str:
        """从知识条目中提取文本"""
        content = knowledge_item["content"]
        knowledge_type = knowledge_item["type"]

        if knowledge_type == KnowledgeType.FACT.value:
            return content.get("statement", "")
        elif knowledge_type == KnowledgeType.RULE.value:
            condition = content.get("condition", "")
            conclusion = content.get("conclusion", "")
            return f"如果 {condition} 那么 {conclusion}"
        elif knowledge_type == KnowledgeType.PROCEDURE.value:
            name = content.get("name", "")
            steps = " ".join(content.get("steps", []))
            return f"过程: {name} 步骤: {steps}"
        elif knowledge_type == KnowledgeType.CONCEPT.value:
            return (
                f"概念: {content.get('name', '')} 定义: {content.get('definition', '')}"
            )
        elif knowledge_type == KnowledgeType.RELATIONSHIP.value:
            subject = content.get("subject", "")
            relation = content.get("relation", "")
            obj = content.get("object", "")
            return f"{subject} {relation} {obj}"
        elif knowledge_type == KnowledgeType.EVENT.value:
            description = content.get("description", "")
            return f"事件: {description}"
        elif knowledge_type == KnowledgeType.PROBLEM_SOLUTION.value:
            problem = content.get("problem", "")
            solution = content.get("solution", "")
            return f"问题: {problem} 解决方案: {solution}"
        elif knowledge_type == KnowledgeType.EXPERIENCE.value:
            situation = content.get("situation", "")
            action = content.get("action", "")
            result = content.get("result", "")
            return f"经验: {situation} 行动: {action} 结果: {result}"
        else:
            return str(content)

    def _update_stats(self, knowledge_type: str, added: bool = True):
        """更新统计信息"""
        if added:
            self.stats["total_knowledge"] += 1
            self.stats["by_type"][knowledge_type] = (
                self.stats["by_type"].get(knowledge_type, 0) + 1
            )
        else:
            self.stats["total_knowledge"] = max(0, self.stats["total_knowledge"] - 1)
            if knowledge_type in self.stats["by_type"]:
                self.stats["by_type"][knowledge_type] = max(
                    0, self.stats["by_type"][knowledge_type] - 1
                )

        self.stats["last_updated"] = datetime.now().isoformat()

    def _clean_expired_cache(self) -> None:
        """清理过期缓存"""
        import time
        current_time = time.time()
        expired_keys = []
        for key, entry in self._query_cache.items():
            if current_time - entry["timestamp"] > self._cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._query_cache[key]
        
        if expired_keys:
            self.logger.info(f"清理了 {len(expired_keys)} 个过期缓存")

    def build_knowledge_graph_from_text(self, text: str, domain: str = "general") -> Dict[str, Any]:
        """从文本构建知识图谱
        
        从非结构化文本中提取实体和关系，构建知识图谱
        
        参数:
            text: 输入文本
            domain: 领域（general, medical, finance, technical等）
            
        返回:
            构建结果统计
        """
        try:
            if not text or not text.strip():
                return {"success": False, "error": "输入文本为空"}
            
            self.logger.info(f"开始从文本构建知识图谱（领域: {domain}, 文本长度: {len(text)}）")
            
            # 提取实体
            entities = self._extract_entities_from_text(text, domain)
            self.logger.info(f"提取到 {len(entities)} 个实体")
            
            # 提取关系
            relations = self._extract_relations_from_text(text, entities, domain)
            self.logger.info(f"提取到 {len(relations)} 个关系")
            
            # 构建知识图谱
            graph_results = self._build_graph_from_entities_and_relations(entities, relations)
            
            # 统计信息
            stats = {
                "success": True,
                "text_length": len(text),
                "entities_extracted": len(entities),
                "relations_extracted": len(relations),
                "nodes_added": graph_results.get("nodes_added", 0),
                "edges_added": graph_results.get("edges_added", 0),
                "domain": domain,
                "timestamp": datetime.now().isoformat()
            }
            
            # 如果启用了知识图谱，更新图谱
            if self.graph:
                stats["graph_enabled"] = True
                stats["total_nodes"] = len(self.graph.graph.nodes())
                stats["total_edges"] = len(self.graph.graph.edges())
            else:
                stats["graph_enabled"] = False
                
            return stats
            
        except Exception as e:
            self.logger.error(f"从文本构建知识图谱失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_entities_from_text(self, text: str, domain: str = "general") -> List[Dict[str, Any]]:
        """从文本中提取实体
        
        参数:
            text: 输入文本
            domain: 领域
            
        返回:
            实体列表
        """
        entities = []
        
        # 导入必要的NLP库（不使用预训练模型）
        import re
        from collections import Counter
        
        # 根据领域调整提取规则
        domain_rules = self._get_domain_entity_rules(domain)
        
        # 分割文本为句子
        sentences = re.split(r'[。！？；\.!?;]', text)
        
        entity_id_counter = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 提取名词短语（简单规则）
            # 匹配中文和英文名词短语
            noun_patterns = [
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # 英文专有名词
                r'([A-Z]+(?:\s+[A-Z]+)*)',  # 英文缩写
                r'([\u4e00-\u9fa5]{2,5})',  # 中文词语（2-5字）
                r'(\d+[\u4e00-\u9fa5]+)',  # 数字+中文
                r'([\u4e00-\u9fa5]+\d+)',  # 中文+数字
            ]
            
            for pattern in noun_patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]  # 提取第一个分组
                    
                    if not match or len(match) < 2:
                        continue
                        
                    # 确定实体类型
                    entity_type = self._determine_entity_type(match, sentence, domain_rules)
                    
                    # 创建实体
                    entity = {
                        "id": f"entity_{entity_id_counter}",
                        "text": match,
                        "type": entity_type,
                        "sentence": sentence,
                        "position": sentence.find(match),
                        "domain": domain,
                        "confidence": 0.8,  # 默认置信度
                    }
                    
                    # 检查是否已存在类似实体
                    if not self._is_duplicate_entity(entity, entities):
                        entities.append(entity)
                        entity_id_counter += 1
        
        # 根据频率排序实体
        if entities:
            entity_texts = [e["text"] for e in entities]
            frequency = Counter(entity_texts)
            
            for entity in entities:
                entity["frequency"] = frequency[entity["text"]]
                # 调整置信度基于频率
                freq_score = min(entity["frequency"] / 3.0, 1.0)
                entity["confidence"] = min(entity["confidence"] + (freq_score * 0.2), 1.0)
        
        return entities
    
    def _extract_relations_from_text(self, text: str, entities: List[Dict[str, Any]], 
                                     domain: str = "general") -> List[Dict[str, Any]]:
        """从文本中提取实体间的关系
        
        参数:
            text: 输入文本
            entities: 实体列表
            domain: 领域
            
        返回:
            关系列表
        """
        relations = []
        
        if not entities or len(entities) < 2:
            return relations
        
        import re
        
        # 根据领域调整关系提取规则
        domain_relation_patterns = self._get_domain_relation_patterns(domain)
        
        # 关系动词模式
        relation_verbs = [
            "是", "有", "包含", "属于", "包括", "称为", "叫做", "名为", 
            "导致", "引起", "产生", "造成", "影响", "作用于",
            "使用", "利用", "通过", "借助", "基于",
            "位于", "处在", "存在于", "在...中", "在...上",
            "相似于", "类似于", "不同于", "区别于",
            "大于", "小于", "等于", "高于", "低于",
            "is", "has", "contains", "includes", "belongs to", "called", "named",
            "causes", "leads to", "produces", "results in", "affects",
            "uses", "utilizes", "employs", "through",
            "located in", "situated at", "exists in", "found in",
            "similar to", "different from", "distinct from",
            "greater than", "less than", "equal to"
        ]
        
        # 分割文本为句子
        sentences = re.split(r'[。！？；\.!?;]', text)
        
        relation_id_counter = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 查找句子中的关系动词
            found_relations = []
            
            for verb in relation_verbs:
                if verb in sentence:
                    # 找到包含关系动词的短语
                    verb_index = sentence.find(verb)
                    
                    # 查找动词前后的实体
                    left_text = sentence[:verb_index]
                    right_text = sentence[verb_index + len(verb):]
                    
                    # 在左右文本中查找实体
                    left_entities = []
                    right_entities = []
                    
                    for entity in entities:
                        if entity["sentence"] == sentence:
                            if entity["position"] < verb_index:
                                left_entities.append(entity)
                            elif entity["position"] > verb_index + len(verb):
                                right_entities.append(entity)
                    
                    # 为每对左右实体创建关系
                    for left_entity in left_entities:
                        for right_entity in right_entities:
                            # 创建关系
                            relation = {
                                "id": f"relation_{relation_id_counter}",
                                "source_entity_id": left_entity["id"],
                                "source_entity_text": left_entity["text"],
                                "target_entity_id": right_entity["id"],
                                "target_entity_text": right_entity["text"],
                                "relation_type": verb,
                                "sentence": sentence,
                                "domain": domain,
                                "confidence": 0.7,
                            }
                            
                            # 确定关系类型
                            relation["relation_category"] = self._determine_relation_category(verb, domain)
                            
                            found_relations.append(relation)
                            relation_id_counter += 1
            
            relations.extend(found_relations)
        
        return relations
    
    def _build_graph_from_entities_and_relations(self, entities: List[Dict[str, Any]], 
                                                 relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从实体和关系构建知识图谱
        
        参数:
            entities: 实体列表
            relations: 关系列表
            
        返回:
            构建结果
        """
        nodes_added = 0
        edges_added = 0
        
        if not self.graph:
            return {"nodes_added": 0, "edges_added": 0, "warning": "知识图谱未启用"}
        
        # 添加实体节点
        entity_nodes = {}
        for entity in entities:
            node_id = entity["id"]
            node_type = entity["type"]
            
            content = {
                "text": entity["text"],
                "sentence": entity["sentence"],
                "domain": entity["domain"],
                "confidence": entity["confidence"],
                "frequency": entity.get("frequency", 1),
            }
            
            metadata = {
                "extracted_from": "text_analysis",
                "position": entity["position"],
                "extraction_method": "rule_based",
            }
            
            # 添加节点到知识图谱
            self.graph.add_node(node_id, node_type, content, metadata)
            entity_nodes[node_id] = entity
            nodes_added += 1
        
        # 添加关系边
        for relation in relations:
            source_id = relation["source_entity_id"]
            target_id = relation["target_entity_id"]
            
            if source_id in entity_nodes and target_id in entity_nodes:
                relation_type = relation["relation_type"]
                
                # 映射到标准关系类型
                mapped_relation_type = self._map_relation_to_standard_type(relation_type)
                
                metadata = {
                    "sentence": relation["sentence"],
                    "confidence": relation["confidence"],
                    "domain": relation["domain"],
                    "category": relation.get("relation_category", "general"),
                    "extraction_method": "rule_based",
                }
                
                # 添加边到知识图谱
                self.graph.add_edge(source_id, target_id, mapped_relation_type, weight=relation["confidence"], metadata=metadata)
                edges_added += 1
        
        return {
            "nodes_added": nodes_added,
            "edges_added": edges_added,
            "total_entities": len(entities),
            "total_relations": len(relations),
        }
    
    def _get_domain_entity_rules(self, domain: str) -> Dict[str, Any]:
        """获取领域特定的实体提取规则
        
        参数:
            domain: 领域
            
        返回:
            领域规则
        """
        rules = {
            "general": {
                "entity_types": ["person", "organization", "location", "concept", "event", "object"],
                "patterns": {
                    "person": [r'先生', r'女士', r'博士', r'教授', r'[A-Z][a-z]+ [A-Z][a-z]+'],
                    "organization": [r'公司', r'集团', r'研究所', r'大学', r'医院', r'[A-Z]+\.?$'],
                    "location": [r'市', r'省', r'区', r'街道', r'路', r'[A-Z][a-z]+ City', r'[A-Z][a-z]+ State'],
                    "concept": [r'理论', r'方法', r'技术', r'系统', r'模型'],
                }
            },
            "medical": {
                "entity_types": ["disease", "symptom", "treatment", "drug", "body_part", "test"],
                "patterns": {
                    "disease": [r'病', r'症', r'炎', r'癌', r'综合征'],
                    "drug": [r'素', r'药', r'剂', r'片', r'胶囊'],
                }
            },
            "finance": {
                "entity_types": ["stock", "currency", "company", "index", "economic_indicator"],
                "patterns": {
                    "stock": [r'股', r'票', r'[A-Z]{1,5}'],
                    "currency": [r'元', r'美元', r'欧元', r'日元', r'RMB', r'USD', r'EUR'],
                }
            },
            "technical": {
                "entity_types": ["technology", "tool", "language", "framework", "algorithm"],
                "patterns": {
                    "technology": [r'[A-Z][a-z]+\.js', r'[A-Z][a-z]+\.py', r'[A-Z][a-z]+\.java'],
                    "language": [r'Python', r'Java', r'JavaScript', r'C\+\+', r'Go', r'Rust'],
                }
            }
        }
        
        return rules.get(domain, rules["general"])
    
    def _get_domain_relation_patterns(self, domain: str) -> Dict[str, Any]:
        """获取领域特定的关系提取模式
        
        参数:
            domain: 领域
            
        返回:
            关系模式
        """
        patterns = {
            "general": {
                "is_a": ["是", "是一种", "is a", "is an"],
                "has": ["有", "具有", "包含", "has", "contains"],
                "causes": ["导致", "引起", "造成", "causes", "leads to"],
                "located_in": ["位于", "处在", "在", "located in", "situated at"],
            },
            "medical": {
                "treats": ["治疗", "治愈", "缓解", "treats", "cures"],
                "causes": ["导致", "引起", "诱发", "causes", "induces"],
                "symptom_of": ["症状", "表现", "sign of", "symptom of"],
                "side_effect_of": ["副作用", "不良反应", "side effect of"],
            },
            "finance": {
                "increases": ["增加", "上升", "增长", "increases", "rises"],
                "decreases": ["减少", "下降", "降低", "decreases", "falls"],
                "correlates_with": ["相关", "关联", "correlates with", "associated with"],
                "affects": ["影响", "作用于", "affects", "impacts"],
            }
        }
        
        return patterns.get(domain, patterns["general"])
    
    def _determine_entity_type(self, entity_text: str, context: str, domain_rules: Dict[str, Any]) -> str:
        """确定实体类型
        
        参数:
            entity_text: 实体文本
            context: 上下文
            domain_rules: 领域规则
            
        返回:
            实体类型
        """
        # 检查是否匹配领域特定模式
        for entity_type, patterns in domain_rules.get("patterns", {}).items():
            for pattern in patterns:
                import re
                if re.search(pattern, entity_text):
                    return entity_type
        
        # 基于启发式规则
        if any(char.isdigit() for char in entity_text):
            if any(unit in entity_text for unit in ["元", "美元", "USD", "RMB"]):
                return "currency" if domain_rules.get("entity_types") and "currency" in domain_rules["entity_types"] else "numeric"
            return "numeric"
        
        # 默认类型
        return domain_rules.get("entity_types", ["concept"])[0]
    
    def _determine_relation_category(self, relation_verb: str, domain: str) -> str:
        """确定关系类别
        
        参数:
            relation_verb: 关系动词
            domain: 领域
            
        返回:
            关系类别
        """
        # 通用关系映射
        relation_mapping = {
            "是": "is_a",
            "是一种": "is_a",
            "is a": "is_a",
            "is an": "is_a",
            "有": "has",
            "具有": "has",
            "包含": "has",
            "has": "has",
            "contains": "has",
            "导致": "causes",
            "引起": "causes",
            "造成": "causes",
            "causes": "causes",
            "leads to": "causes",
            "位于": "located_in",
            "处在": "located_in",
            "在": "located_in",
            "located in": "located_in",
            "situated at": "located_in",
        }
        
        # 添加领域特定映射
        if domain == "medical":
            medical_mapping = {
                "治疗": "treats",
                "治愈": "treats",
                "缓解": "treats",
                "treats": "treats",
                "cures": "treats",
                "症状": "symptom_of",
                "表现": "symptom_of",
                "sign of": "symptom_of",
                "symptom of": "symptom_of",
            }
            relation_mapping.update(medical_mapping)
        
        return relation_mapping.get(relation_verb, "related_to")
    
    def _map_relation_to_standard_type(self, relation_type: str) -> str:
        """将关系映射到标准关系类型
        
        参数:
            relation_type: 原始关系类型
            
        返回:
            标准关系类型
        """
        # 导入KnowledgeGraph中的RelationType
        from .knowledge_graph import RelationType
        
        # 映射表
        mapping = {
            "是": RelationType.IS_A.value,
            "is_a": RelationType.IS_A.value,
            "有": RelationType.HAS_PROPERTY.value,
            "has": RelationType.HAS_PROPERTY.value,
            "包含": RelationType.PART_OF.value,
            "contains": RelationType.PART_OF.value,
            "导致": RelationType.CAUSES.value,
            "causes": RelationType.CAUSES.value,
            "位于": RelationType.LOCATED_IN.value,
            "located_in": RelationType.LOCATED_IN.value,
            "使用": RelationType.USES.value,
            "uses": RelationType.USES.value,
            "相似于": RelationType.SIMILAR_TO.value,
            "similar_to": RelationType.SIMILAR_TO.value,
            "不同于": RelationType.OPPOSITE_OF.value,
            "different_from": RelationType.OPPOSITE_OF.value,
            "相关": RelationType.RELATED_TO.value,
            "related_to": RelationType.RELATED_TO.value,
        }
        
        return mapping.get(relation_type, RelationType.RELATED_TO.value)
    
    def _is_duplicate_entity(self, new_entity: Dict[str, Any], existing_entities: List[Dict[str, Any]]) -> bool:
        """检查是否重复实体
        
        参数:
            new_entity: 新实体
            existing_entities: 现有实体列表
            
        返回:
            是否重复
        """
        for entity in existing_entities:
            # 检查文本是否相似（简单实现）
            if entity["text"] == new_entity["text"]:
                return True
            
            # 检查是否在同一句子中的相似位置
            if (entity["sentence"] == new_entity["sentence"] and 
                abs(entity["position"] - new_entity["position"]) < 5):
                return True
        
        return False
