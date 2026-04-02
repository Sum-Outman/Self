"""
记忆系统路由模块
处理记忆系统相关的API请求，包括记忆检索、统计、配置等功能

注意：此文件不应直接运行，而是作为FastAPI应用的一部分运行。
直接运行此文件会导致导入错误，因为相对导入需要正确的模块上下文。
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import time
import logging

from backend.dependencies import get_db, get_current_user, get_current_admin
from backend.db_models.user import User
from backend.schemas.memory import (
    MemoryStats,
    MemoryItem,
    MemorySearchRequest,
    MemorySearchResponse,
    KnowledgeSearchRequest,
    HybridSearchRequest,
    HybridSearchResponse,
    SystemConfigUpdate,
    SystemConfigResponse
)
from backend.schemas.response import SuccessResponse, ErrorResponse, PaginatedResponse

# 全局状态管理器（解决app.state不一致问题）
from backend.state_manager import get_memory_system as get_memory_system_global

# 从主应用导入记忆系统依赖函数（延迟导入，避免循环导入）
MEMORY_SYSTEM_AVAILABLE = None  # 将在运行时确定
main_get_memory_system = None   # 将在运行时导入

# 记忆系统实例缓存（避免app.state访问问题）
_memory_system_cache = None

router = APIRouter(prefix="/api/memory", tags=["记忆系统"])


def get_memory_system():
    """获取记忆系统实例依赖（使用全局状态管理器）"""
    global _memory_system_cache
    logger = logging.getLogger(__name__)
    
    # 首先检查缓存
    if _memory_system_cache is not None:
        logger.info(f"从缓存获取记忆系统实例: {_memory_system_cache}")
        return _memory_system_cache
    
    # 从全局状态管理器获取
    memory_system = get_memory_system_global()
    logger.info(f"从全局状态管理器获取memory_system: {memory_system}")
    
    if memory_system is not None:
        # 更新缓存
        _memory_system_cache = memory_system
        logger.info(f"✅ 从全局状态管理器获取到有效实例，更新缓存: {memory_system}")
    else:
        logger.warning("❌ 全局状态管理器中的memory_system为None")
    
    return memory_system


@router.get("/stats", response_model=SuccessResponse[Dict[str, Any]])
async def get_memory_stats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ms = Depends(get_memory_system),
):
    """获取记忆系统统计信息"""
    try:
        # 检查记忆系统是否可用
        if ms is None:
            # 记忆系统未初始化，返回基本统计信息
            memory_stats = MemoryStats(
                total_memories=0,
                short_term_memories=0,
                long_term_memories=0,
                working_memory_usage=0,
                cache_hit_rate=0,
                average_retrieval_time_ms=0,
                memory_usage_mb=0,
                autonomous_optimizations=0,
                scene_transitions=0,
                current_scene="general",
                reasoning_operations=0,
                last_updated=datetime.now(timezone.utc)
            )
        else:
            # 从记忆系统获取统计信息
            stats_data = ms.get_stats(db=db, user_id=user.id)
            
            # 转换为响应格式
            memory_stats = MemoryStats(
                total_memories=stats_data.get("total_memories", 0),
                short_term_memories=stats_data.get("short_term_memories", 0),
                long_term_memories=stats_data.get("long_term_memories", 0),
                working_memory_usage=stats_data.get("working_memory_usage", 0),
                cache_hit_rate=stats_data.get("cache_hit_rate", 0),
                average_retrieval_time_ms=stats_data.get("average_retrieval_time_ms", 0),
                memory_usage_mb=stats_data.get("memory_usage_mb", 0),
                autonomous_optimizations=stats_data.get("autonomous_optimizations", 0),
                scene_transitions=stats_data.get("scene_transitions", 0),
                current_scene=stats_data.get("current_scene", "general"),
                reasoning_operations=stats_data.get("reasoning_operations", 0),
                last_updated=datetime.now(timezone.utc)
            )
        
        return SuccessResponse.create(
            data={
                "stats": memory_stats.dict(),
                "system_initialized": ms is not None
            },
            message="获取记忆系统统计信息成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取记忆统计信息失败: {str(e)}"
        )


@router.post("/search", response_model=SuccessResponse[Dict[str, Any]])
async def search_memories(
    search_request: MemorySearchRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ms = Depends(get_memory_system),
):
    """搜索记忆"""
    try:
        start_time = time.time()
        
        # 调用记忆系统检索功能
        memories = ms.retrieve_relevant_memories(
            query=search_request.query,
            db=db,
            user_id=user.id,
            top_k=search_request.limit,
            similarity_threshold=search_request.min_similarity or ms.similarity_threshold,
            memory_type=search_request.memory_type
        )
        
        # 转换为响应格式
        memory_items = []
        for mem in memories:
            memory_items.append(MemoryItem(
                id=str(mem.get("id", "")),
                content=mem.get("content", ""),
                type=mem.get("memory_type", "long_term"),
                created_at=datetime.fromtimestamp(mem.get("created_at", time.time())),
                accessed_at=datetime.fromtimestamp(mem.get("accessed_at", time.time())),
                importance=mem.get("importance_score", 0.5),
                similarity=mem.get("similarity_score", 0.5),
                scene_type=mem.get("scene_type"),
                source=mem.get("source", "user")
            ))
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return SuccessResponse.create(
            data={
                "memories": memory_items,
                "total": len(memory_items),
                "query_time_ms": query_time_ms,
            },
            message="记忆搜索成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"搜索记忆失败: {str(e)}"
        )


@router.post("/knowledge/search", response_model=SuccessResponse[Dict[str, Any]])
async def search_knowledge(
    search_request: KnowledgeSearchRequest,
    user: User = Depends(get_current_user),
    ms = Depends(get_memory_system),
):
    """从知识库搜索信息"""
    try:
        start_time = time.time()
        
        # 调用知识库检索功能
        knowledge_results = ms.retrieve_from_knowledge_base(
            query=search_request.query,
            top_k=search_request.top_k,
            similarity_threshold=search_request.similarity_threshold
        )
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return SuccessResponse.create(
            data={
                "results": knowledge_results,
                "total": len(knowledge_results),
                "query_time_ms": query_time_ms,
            },
            message="知识库搜索成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"搜索知识库失败: {str(e)}"
        )


@router.post("/hybrid/search", response_model=SuccessResponse[Dict[str, Any]])
async def hybrid_search(
    search_request: HybridSearchRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ms = Depends(get_memory_system),
):
    """混合搜索 - 同时从记忆系统和知识库检索"""
    try:
        start_time = time.time()
        
        # 调用混合检索功能
        hybrid_results = ms.retrieve_hybrid(
            query=search_request.query,
            db=db,
            user_id=user.id,
            top_k=search_request.top_k,
            memory_weight=search_request.memory_weight,
            knowledge_weight=search_request.knowledge_weight
        )
        
        query_time_ms = (time.time() - start_time) * 1000
        
        # 转换为响应格式
        results = []
        memory_count = 0
        knowledge_count = 0
        
        for result in hybrid_results:
            results.append({
                "id": result.get("id", ""),
                "content": result.get("content", ""),
                "source": result.get("source", "unknown"),
                "similarity_score": result.get("similarity_score", result.get("weighted_score", 0.5)),
                "weighted_score": result.get("weighted_score", 0.5),
                "metadata": result.get("metadata", {})
            })
            
            if result.get("source") == "memory":
                memory_count += 1
            elif result.get("source") == "knowledge":
                knowledge_count += 1
        
        return SuccessResponse.create(
            data={
                "results": results,
                "memory_count": memory_count,
                "knowledge_count": knowledge_count,
                "total_time_ms": query_time_ms,
            },
            message="混合搜索成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"混合搜索失败: {str(e)}"
        )


@router.get("/recent", response_model=SuccessResponse[Dict[str, Any]])
async def get_recent_memories(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    ms = Depends(get_memory_system),
):
    """获取最近记忆"""
    try:
        # 搜索最近的记忆（使用空查询获取最近记忆）
        memories = ms.retrieve_relevant_memories(
            query="",  # 空查询获取最近记忆
            db=db,
            user_id=user.id,
            top_k=limit,
            similarity_threshold=0.1  # 低阈值以获取更多结果
        )
        
        # 转换为响应格式
        memory_items = []
        for mem in memories:
            memory_items.append(MemoryItem(
                id=str(mem.get("id", "")),
                content=mem.get("content", ""),
                type=mem.get("memory_type", "long_term"),
                created_at=datetime.fromtimestamp(mem.get("created_at", time.time())),
                accessed_at=datetime.fromtimestamp(mem.get("accessed_at", time.time())),
                importance=mem.get("importance_score", 0.5),
                similarity=mem.get("similarity_score", 0.5),
                scene_type=mem.get("scene_type"),
                source=mem.get("source", "user")
            ))
        
        return SuccessResponse.create(
            data={
                "memories": memory_items,
                "total": len(memory_items),
            },
            message="获取最近记忆成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取最近记忆失败: {str(e)}"
        )


@router.get("/config", response_model=SuccessResponse[Dict[str, Any]])
async def get_system_config(
    user: User = Depends(get_current_admin),
    ms = Depends(get_memory_system),
):
    """获取系统配置（仅管理员）"""
    try:
        # 获取当前配置
        config = ms.config if hasattr(ms, 'config') else {}
        
        return SuccessResponse.create(
            data={
                "config": config,
                "last_modified": datetime.now(timezone.utc).isoformat(),
            },
            message="获取系统配置成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统配置失败: {str(e)}"
        )


@router.put("/config", response_model=SuccessResponse[Dict[str, Any]])
async def update_system_config(
    config_update: SystemConfigUpdate,
    user: User = Depends(get_current_admin),
    ms = Depends(get_memory_system),
):
    """更新系统配置（仅管理员）"""
    try:
        # 这里可以添加配置更新逻辑
        # 由于配置更新涉及系统状态，这里只返回成功消息
        # 实际实现中应该更新记忆系统的配置
        
        return SuccessResponse.create(
            data={
                "config": config_update.dict(),
            },
            message="系统配置更新请求已接收"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新系统配置失败: {str(e)}"
        )


@router.get("/health", response_model=SuccessResponse[Dict[str, Any]])
async def get_memory_system_health(
    ms = Depends(get_memory_system),
):
    """获取记忆系统健康状态"""
    try:
        # 检查记忆系统状态
        if not ms.initialized if hasattr(ms, 'initialized') else False:
            health_status = {
                "status": "not_initialized",
                "message": "记忆系统未初始化",
                "available": False
            }
        else:
            health_status = {
                "status": "healthy",
                "message": "记忆系统运行正常",
                "available": True,
                "features": {
                    "autonomous_memory": ms.enable_autonomous_memory if hasattr(ms, 'enable_autonomous_memory') else False,
                    "scene_classification": ms.enable_scene_classification if hasattr(ms, 'enable_scene_classification') else False,
                    "knowledge_integration": ms.enable_knowledge_base if hasattr(ms, 'enable_knowledge_base') else False,
                    "multimodal_memory": ms.enable_multimodal_memory if hasattr(ms, 'enable_multimodal_memory') else False
                }
            }
        
        return SuccessResponse.create(
            data={
                "health": health_status
            },
            message="获取记忆系统健康状态成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取健康状态失败: {str(e)}"
        )