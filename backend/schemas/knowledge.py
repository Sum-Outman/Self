"""
知识库相关数据模式
定义知识库API请求和响应的数据结构
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class KnowledgeItemBase(BaseModel):
    """知识项基础模型"""

    title: str = Field(..., max_length=200, description="标题")
    description: Optional[str] = Field(None, description="描述")
    type: str = Field(
        "document",
        description="类型: text, image, video, audio, document, code, dataset",
    )
    content: Optional[str] = Field(None, description="内容或文件路径")
    size: int = Field(0, ge=0, description="文件大小（字节）")
    tags: Optional[List[str]] = Field(None, description="标签列表")


class KnowledgeItemCreate(KnowledgeItemBase):
    """创建知识项请求模型"""

    file_path: Optional[str] = Field(None, description="文件路径")
    meta_data: Optional[Dict[str, Any]] = Field(None, description="元数据")


class KnowledgeItemUpdate(BaseModel):
    """更新知识项请求模型"""

    title: Optional[str] = Field(None, max_length=200, description="标题")
    description: Optional[str] = Field(None, description="描述")
    type: Optional[str] = Field(None, description="类型")
    content: Optional[str] = Field(None, description="内容")
    tags: Optional[List[str]] = Field(None, description="标签列表")
    meta_data: Optional[Dict[str, Any]] = Field(None, description="元数据")


class KnowledgeItemResponse(KnowledgeItemBase):
    """知识项响应模型"""

    id: str
    upload_date: datetime
    uploaded_by: str
    access_count: int
    last_accessed: datetime
    file_url: Optional[str]
    metadata: Optional[Dict[str, Any]]
    embedding: Optional[List[float]]
    similarity: Optional[float]

    class Config:
        from_attributes = True


class KnowledgeSearchRequest(BaseModel):
    """知识搜索请求模型"""

    query: str = Field(..., description="搜索查询")
    type: Optional[str] = Field(None, description="过滤类型")
    tags: Optional[List[str]] = Field(None, description="过滤标签")
    start_date: Optional[str] = Field(None, description="开始日期")
    end_date: Optional[str] = Field(None, description="结束日期")
    limit: int = Field(20, ge=1, le=100, description="每页数量")
    offset: int = Field(0, ge=0, description="偏移量")
    sort_by: str = Field(
        "relevance", description="排序字段: relevance, date, access, size"
    )
    sort_order: str = Field("desc", description="排序顺序: asc, desc")


class KnowledgeSearchResponse(BaseModel):
    """知识搜索响应模型"""

    items: List[KnowledgeItemResponse]
    total: int
    query_time: float
    suggestions: List[str]


class KnowledgeStatsResponse(BaseModel):
    """知识统计响应模型"""

    total_items: int
    total_size: int
    by_type: Dict[str, int]
    by_month: Dict[str, int]
    popular_tags: List[str]
    storage_usage: int
    average_access_count: float


class KnowledgeImportRequest(BaseModel):
    """知识导入请求模型"""

    items: List[KnowledgeItemBase]


class KnowledgeImportResponse(BaseModel):
    """知识导入响应模型"""

    success: int
    failed: int


class KnowledgeExportResponse(BaseModel):
    """知识导出响应模型"""

    download_url: str
    format: str


__all__ = [
    "KnowledgeItemBase",
    "KnowledgeItemCreate",
    "KnowledgeItemUpdate",
    "KnowledgeItemResponse",
    "KnowledgeSearchRequest",
    "KnowledgeSearchResponse",
    "KnowledgeStatsResponse",
    "KnowledgeImportRequest",
    "KnowledgeImportResponse",
    "KnowledgeExportResponse",
]
