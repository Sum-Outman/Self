"""
记忆系统相关数据模式
定义记忆系统API请求和响应的数据结构
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class MemoryStats(BaseModel):
    """记忆统计信息模型"""

    total_memories: int = Field(..., ge=0, description="总记忆数量")
    short_term_memories: int = Field(..., ge=0, description="短期记忆数量")
    long_term_memories: int = Field(..., ge=0, description="长期记忆数量")
    working_memory_usage: float = Field(..., ge=0, le=100, description="工作内存使用率")
    cache_hit_rate: float = Field(..., ge=0, le=1, description="缓存命中率")
    average_retrieval_time_ms: float = Field(
        ..., ge=0, description="平均检索响应时间(毫秒)"
    )
    memory_usage_mb: float = Field(..., ge=0, description="内存使用量(MB)")
    autonomous_optimizations: int = Field(..., ge=0, description="自主优化次数")
    scene_transitions: int = Field(..., ge=0, description="场景切换次数")
    current_scene: str = Field(..., description="当前场景类型")
    reasoning_operations: int = Field(..., ge=0, description="推理操作次数")
    last_updated: datetime = Field(..., description="最后更新时间")


class MemoryItem(BaseModel):
    """记忆项模型"""

    id: str = Field(..., description="记忆ID")
    content: str = Field(..., description="记忆内容")
    type: str = Field(..., description="记忆类型: short_term, long_term, working")
    created_at: datetime = Field(..., description="创建时间")
    accessed_at: datetime = Field(..., description="最后访问时间")
    importance: float = Field(..., ge=0, le=1, description="重要性分数")
    similarity: float = Field(..., ge=0, le=1, description="相似度分数")
    scene_type: Optional[str] = Field(
        None, description="场景类型: task, learning, problem_solving, social, planning"
    )
    source: str = Field(..., description="来源: user, system, autonomous")


class MemorySearchRequest(BaseModel):
    """记忆搜索请求模型"""

    query: str = Field(..., description="搜索查询")
    memory_type: Optional[str] = Field(None, description="记忆类型过滤")
    scene_type: Optional[str] = Field(None, description="场景类型过滤")
    min_importance: Optional[float] = Field(
        None, ge=0, le=1, description="最小重要性阈值"
    )
    min_similarity: Optional[float] = Field(
        None, ge=0, le=1, description="最小相似度阈值"
    )
    start_date: Optional[datetime] = Field(None, description="开始日期")
    end_date: Optional[datetime] = Field(None, description="结束日期")
    limit: int = Field(10, ge=1, le=100, description="返回结果数量")
    offset: int = Field(0, ge=0, description="偏移量")


class MemorySearchResponse(BaseModel):
    """记忆搜索响应模型"""

    memories: List[MemoryItem] = Field(..., description="记忆列表")
    total: int = Field(..., ge=0, description="总数量")
    query_time_ms: float = Field(..., ge=0, description="查询时间(毫秒)")


class KnowledgeSearchRequest(BaseModel):
    """知识库搜索请求模型"""

    query: str = Field(..., description="搜索查询")
    top_k: int = Field(5, ge=1, le=50, description="返回知识数量")
    similarity_threshold: float = Field(0.7, ge=0, le=1, description="相似度阈值")


class HybridSearchRequest(BaseModel):
    """混合搜索请求模型"""

    query: str = Field(..., description="搜索查询")
    top_k: int = Field(10, ge=1, le=50, description="返回结果总数")
    memory_weight: float = Field(0.6, ge=0, le=1, description="记忆系统结果权重")
    knowledge_weight: float = Field(0.4, ge=0, le=1, description="知识库结果权重")


class HybridSearchResult(BaseModel):
    """混合搜索结果模型"""

    id: str = Field(..., description="结果ID")
    content: str = Field(..., description="内容")
    source: str = Field(..., description="来源: memory, knowledge")
    similarity_score: float = Field(..., ge=0, le=1, description="相似度分数")
    weighted_score: float = Field(..., ge=0, le=1, description="加权分数")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class HybridSearchResponse(BaseModel):
    """混合搜索响应模型"""

    results: List[HybridSearchResult] = Field(..., description="搜索结果列表")
    memory_count: int = Field(..., ge=0, description="记忆结果数量")
    knowledge_count: int = Field(..., ge=0, description="知识结果数量")
    total_time_ms: float = Field(..., ge=0, description="总查询时间(毫秒)")


class SystemConfigUpdate(BaseModel):
    """系统配置更新模型"""

    enable_autonomous_memory: Optional[bool] = Field(
        None, description="启用自主记忆管理"
    )
    enable_scene_classification: Optional[bool] = Field(
        None, description="启用场景分类"
    )
    enable_knowledge_integration: Optional[bool] = Field(
        None, description="启用知识库集成"
    )
    working_memory_capacity: Optional[int] = Field(
        None, ge=10, le=1000, description="工作内存容量"
    )
    cache_size_mb: Optional[int] = Field(
        None, ge=10, le=1024, description="缓存大小(MB)"
    )
    similarity_threshold: Optional[float] = Field(
        None, ge=0.1, le=0.9, description="相似度阈值"
    )


class SystemConfigResponse(BaseModel):
    """系统配置响应模型"""

    config: Dict[str, Any] = Field(..., description="系统配置")
    last_modified: datetime = Field(..., description="最后修改时间")
