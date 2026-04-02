# 知识库管理器 / Knowledge Manager

## 概述 / Overview

### 中文
知识库管理器是整个知识库系统的核心管理类，协调知识存储、检索、更新和验证等功能。支持知识图谱、向量检索、知识验证等完整功能，使用从零开始的文本嵌入器，不依赖任何预训练模型。

### English
The Knowledge Manager is the core management class of the entire knowledge base system, coordinating knowledge storage, retrieval, update, and validation functions. Supports complete features including knowledge graph, vector retrieval, knowledge validation, etc., using a text embedder from scratch without relying on any pre-trained models.

---

## 核心功能 / Core Features

### 中文
1. **知识管理**：知识增删改查完整操作
2. **知识类型**：事实、规则、过程、概念、关系、事件、问题解决方案、经验
3. **知识图谱**：实体关系构建与查询
4. **语义检索**：基于向量的语义搜索
5. **知识验证**：知识质量验证
6. **知识推理**：基于知识图谱的推理
7. **知识持久化**：本地存储与加载

### English
1. **Knowledge Management**: Complete CRUD operations for knowledge
2. **Knowledge Types**: Facts, rules, procedures, concepts, relationships, events, problem solutions, experiences
3. **Knowledge Graph**: Entity relationship construction and querying
4. **Semantic Retrieval**: Vector-based semantic search
5. **Knowledge Validation**: Knowledge quality validation
6. **Knowledge Reasoning**: Knowledge graph-based reasoning
7. **Knowledge Persistence**: Local storage and loading

---

## 核心类 / Core Classes

### KnowledgeType
```python
class KnowledgeType(Enum):
    FACT = "fact"              # 事实 / Fact
    RULE = "rule"              # 规则 / Rule
    PROCEDURE = "procedure"    # 过程/方法 / Procedure
    CONCEPT = "concept"        # 概念 / Concept
    RELATIONSHIP = "relationship"  # 关系 / Relationship
    EVENT = "event"            # 事件 / Event
    PROBLEM_SOLUTION = "problem_solution"  # 问题解决方案 / Problem Solution
    EXPERIENCE = "experience"  # 经验 / Experience
```

### KnowledgeManager
```python
class KnowledgeManager:
    """知识管理器 - 协调知识存储、检索、更新和验证"""
```

### KnowledgeStore
```python
class KnowledgeStore:
    """知识存储 - 负责知识的持久化存储"""
```

### KnowledgeRetriever
```python
class KnowledgeRetriever:
    """知识检索 - 基于向量的语义检索"""
```

### KnowledgeGraph
```python
class KnowledgeGraph:
    """知识图谱 - 实体关系网络"""
```

### KnowledgeValidator
```python
class KnowledgeValidator:
    """知识验证 - 知识质量验证"""
```

### KnowledgeReasoner
```python
class KnowledgeReasoner:
    """知识推理 - 基于知识图谱的推理"""
```

---

## 主要方法 / Main Methods

### 知识管理 / Knowledge Management

#### 添加知识 / Add Knowledge
```python
def add_knowledge(
    knowledge_type: Union[KnowledgeType, str],
    content: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    validate: bool = True
) -> Dict[str, Any]
```
添加新知识，支持验证。

Add new knowledge with validation support.

#### 获取知识 / Get Knowledge
```python
def get_knowledge(knowledge_id: str) -> Optional[Dict[str, Any]]
```
根据ID获取知识。

Get knowledge by ID.

#### 更新知识 / Update Knowledge
```python
def update_knowledge(
    knowledge_id: str,
    content: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool
```
更新已有知识。

Update existing knowledge.

#### 删除知识 / Delete Knowledge
```python
def delete_knowledge(knowledge_id: str) -> bool
```
删除指定知识。

Delete specified knowledge.

### 知识检索 / Knowledge Retrieval

#### 语义搜索 / Semantic Search
```python
def search_knowledge(
    query: str,
    top_k: int = 10,
    knowledge_type: Optional[KnowledgeType] = None
) -> List[Dict[str, Any]]
```
基于语义的知识搜索。

Semantic-based knowledge search.

#### 按类型搜索 / Search by Type
```python
def search_by_type(knowledge_type: KnowledgeType) -> List[Dict[str, Any]]
```
按知识类型搜索。

Search by knowledge type.

### 知识图谱 / Knowledge Graph

#### 添加实体 / Add Entity
```python
def add_entity(
    entity_id: str,
    entity_type: str,
    properties: Dict[str, Any]
) -> bool
```
添加实体到知识图谱。

Add entity to knowledge graph.

#### 添加关系 / Add Relationship
```python
def add_relationship(
    subject_id: str,
    predicate: str,
    object_id: str,
    properties: Optional[Dict[str, Any]] = None
) -> bool
```
添加关系到知识图谱。

Add relationship to knowledge graph.

#### 查询路径 / Query Path
```python
def query_path(
    start_id: str,
    end_id: str,
    max_depth: int = 3
) -> List[List[str]]
```
查询实体间的路径。

Query paths between entities.

### 知识验证 / Knowledge Validation

#### 验证知识 / Validate Knowledge
```python
def validate_knowledge(knowledge_item: Dict[str, Any]) -> Dict[str, Any]
```
验证知识质量。

Validate knowledge quality.

#### 批量验证 / Batch Validate
```python
def batch_validate(knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]
```
批量验证知识。

Batch validate knowledge.

### 知识推理 / Knowledge Reasoning

#### 推理 / Reason
```python
def reason(query: str) -> Dict[str, Any]
```
基于知识图谱的推理。

Reasoning based on knowledge graph.

#### 推导结论 / Derive Conclusions
```python
def derive_conclusions(premises: List[str]) -> List[str]
```
从前提推导结论。

Derive conclusions from premises.

### 持久化 / Persistence

#### 保存知识库 / Save Knowledge Base
```python
def save_knowledge_base(file_path: str) -> bool
```
保存知识库到文件。

Save knowledge base to file.

#### 加载知识库 / Load Knowledge Base
```python
def load_knowledge_base(file_path: str) -> bool
```
从文件加载知识库。

Load knowledge base from file.

---

## 使用示例 / Usage Examples

### 中文
```python
from models.knowledge_base.knowledge_manager import (
    KnowledgeManager,
    KnowledgeType
)

# 创建知识管理器 / Create knowledge manager
manager = KnowledgeManager()

# 添加事实知识 / Add fact knowledge
result = manager.add_knowledge(
    KnowledgeType.FACT,
    {
        "subject": "地球",
        "predicate": "是",
        "object": "行星",
        "description": "地球是太阳系中的一颗行星"
    },
    metadata={"source": "科学知识"}
)
print(f"知识ID: {result['id']}")

# 语义搜索 / Semantic search
results = manager.search_knowledge("太阳系", top_k=5)
for item in results:
    print(f"{item['content']['description']}")

# 添加到知识图谱 / Add to knowledge graph
manager.add_entity("地球", "天体", {"质量": "5.972e24 kg"})
manager.add_entity("太阳", "恒星", {"质量": "1.989e30 kg"})
manager.add_relationship("地球", "围绕", "太阳", {"距离": "1.5亿公里"})

# 查询路径 / Query path
paths = manager.query_path("地球", "太阳")
print(f"路径数量: {len(paths)}")

# 保存知识库 / Save knowledge base
manager.save_knowledge_base("data/knowledge_base.json")
```

### English
```python
from models.knowledge_base.knowledge_manager import (
    KnowledgeManager,
    KnowledgeType
)

# Create knowledge manager
manager = KnowledgeManager()

# Add fact knowledge
result = manager.add_knowledge(
    KnowledgeType.FACT,
    {
        "subject": "Earth",
        "predicate": "is",
        "object": "planet",
        "description": "Earth is a planet in the solar system"
    },
    metadata={"source": "scientific knowledge"}
)
print(f"Knowledge ID: {result['id']}")

# Semantic search
results = manager.search_knowledge("solar system", top_k=5)
for item in results:
    print(f"{item['content']['description']}")

# Add to knowledge graph
manager.add_entity("Earth", "celestial body", {"mass": "5.972e24 kg"})
manager.add_entity("Sun", "star", {"mass": "1.989e30 kg"})
manager.add_relationship("Earth", "orbits", "Sun", {"distance": "150 million km"})

# Query path
paths = manager.query_path("Earth", "Sun")
print(f"Number of paths: {len(paths)}")

# Save knowledge base
manager.save_knowledge_base("data/knowledge_base.json")
```

---

## 相关模块 / Related Modules

- [知识存储](./knowledge-store.md) - 知识存储
- [知识检索](./knowledge-retriever.md) - 知识检索
- [知识图谱](./knowledge-graph.md) - 知识图谱
- [知识验证](./knowledge-validator.md) - 知识验证
