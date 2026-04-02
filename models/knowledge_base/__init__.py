"""
知识库管理系统模块

功能：
- 知识存储：支持多种知识类型（事实、规则、过程、概念等）
- 知识检索：支持语义搜索、向量检索、关键字搜索
- 知识图谱：实体关系图
- 知识推理：基于规则的推理
- 知识更新：学习新知识，更新现有知识
- 知识验证：验证知识的正确性和一致性
"""

from .knowledge_manager import KnowledgeManager, KnowledgeType
from .knowledge_store import KnowledgeStore
from .knowledge_retriever import KnowledgeRetriever
from .knowledge_graph import KnowledgeGraph, RelationType
from .knowledge_validator import KnowledgeValidator
from .knowledge_reasoner import KnowledgeReasoner, get_global_knowledge_reasoner

__all__ = [
    "KnowledgeManager",
    "KnowledgeReasoner",
    "get_global_knowledge_reasoner",
    "KnowledgeType",
    "KnowledgeStore",
    "KnowledgeGraph",
    "RelationType",
    "KnowledgeRetriever",
    "KnowledgeValidator"
]
