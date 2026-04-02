"""
知识检索器

基于向量相似度的知识检索，支持语义搜索和相似性匹配。
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import torch


class KnowledgeRetriever:
    """知识检索器

    功能：
    - 基于向量相似度的知识检索
    - 语义搜索
    - 相似性匹配
    - 结果排序和过滤
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化知识检索器

        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # 相似度阈值
        self.similarity_threshold = config.get("similarity_threshold", 0.7)

        # 缓存最近检索结果
        self.cache = {}
        self.cache_size = config.get("cache_size", 100)

        self.logger.info("知识检索器初始化完成")

    def retrieve(
        self,
        query_embedding: Optional[torch.Tensor] = None,
        knowledge_type: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """检索知识

        参数:
            query_embedding: 查询嵌入向量
            knowledge_type: 知识类型过滤
            limit: 返回结果数量
            similarity_threshold: 相似度阈值

        返回:
            知识条目列表（按相似度排序）
        """
        from .knowledge_store import KnowledgeStore

        # 如果没有查询嵌入，返回空列表
        if query_embedding is None:
            return []  # 返回空列表

        # 获取知识存储实例
        store = KnowledgeStore(self.config)

        # 获取所有知识向量
        all_vectors = store.get_all_vectors()

        if not all_vectors:
            return []  # 返回空列表

        # 过滤知识类型
        if knowledge_type:
            filtered_vectors = {}
            for knowledge_id, vector in all_vectors.items():
                knowledge_item = store.get(knowledge_id)
                if knowledge_item and knowledge_item.get("type") == knowledge_type:
                    filtered_vectors[knowledge_id] = vector
            all_vectors = filtered_vectors

        if not all_vectors:
            return []  # 返回空列表

        # 计算相似度
        similarities = self._compute_similarities(query_embedding, all_vectors)

        # 应用阈值
        threshold = similarity_threshold or self.similarity_threshold
        filtered_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]

        if not filtered_indices:
            return []  # 返回空列表

        # 排序（相似度从高到低）
        sorted_indices = sorted(
            filtered_indices, key=lambda i: similarities[i], reverse=True
        )

        # 限制结果数量
        result_indices = sorted_indices[:limit]

        # 获取知识条目
        knowledge_ids = list(all_vectors.keys())
        results = []

        for idx in result_indices:
            knowledge_id = knowledge_ids[idx]
            knowledge_item = store.get(knowledge_id)
            if knowledge_item:
                # 添加相似度分数
                knowledge_item["similarity"] = float(similarities[idx])
                results.append(knowledge_item)

        # 缓存结果
        self._cache_results(query_embedding, knowledge_type, results)

        return results

    def retrieve_by_embedding(
        self, embedding: torch.Tensor, limit: int = 5, exclude_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """根据嵌入向量检索相似知识

        参数:
            embedding: 参考嵌入向量
            limit: 返回结果数量
            exclude_id: 排除的知识ID

        返回:
            相似知识列表
        """
        from .knowledge_store import KnowledgeStore

        store = KnowledgeStore(self.config)
        all_vectors = store.get_all_vectors()

        if not all_vectors:
            return []  # 返回空列表

        # 排除特定ID
        if exclude_id and exclude_id in all_vectors:
            filtered_vectors = {k: v for k, v in all_vectors.items() if k != exclude_id}
        else:
            filtered_vectors = all_vectors

        if not filtered_vectors:
            return []  # 返回空列表

        # 计算相似度
        similarities = self._compute_similarities(embedding, filtered_vectors)

        # 排序（相似度从高到低）
        knowledge_ids = list(filtered_vectors.keys())
        sorted_indices = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )

        # 限制结果数量
        result_indices = sorted_indices[:limit]

        # 获取知识条目
        results = []
        for idx in result_indices:
            knowledge_id = knowledge_ids[idx]
            knowledge_item = store.get(knowledge_id)
            if knowledge_item:
                knowledge_item["similarity"] = float(similarities[idx])
                results.append(knowledge_item)

        return results

    def hybrid_retrieve(
        self,
        query: str,
        query_embedding: Optional[torch.Tensor] = None,
        knowledge_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """混合检索（向量相似度 + 关键字匹配）

        参数:
            query: 查询文本
            query_embedding: 查询嵌入向量
            knowledge_type: 知识类型过滤
            limit: 返回结果数量

        返回:
            知识条目列表
        """
        from .knowledge_store import KnowledgeStore

        store = KnowledgeStore(self.config)

        # 向量检索
        vector_results = []
        if query_embedding is not None:
            vector_results = self.retrieve(
                query_embedding=query_embedding,
                knowledge_type=knowledge_type,
                limit=limit * 2,  # 获取更多结果用于融合
                similarity_threshold=self.similarity_threshold
                * 0.8,  # 降低阈值以获取更多结果
            )

        # 关键字检索
        keyword_results = store.search_by_content(
            query_text=query, knowledge_type=knowledge_type, limit=limit * 2
        )

        # 融合结果
        fused_results = self._fuse_results(vector_results, keyword_results, limit)

        return fused_results

    def retrieve_by_ids(self, knowledge_ids: List[str]) -> List[Dict[str, Any]]:
        """根据ID列表检索知识

        参数:
            knowledge_ids: 知识ID列表

        返回:
            知识条目列表
        """
        from .knowledge_store import KnowledgeStore

        store = KnowledgeStore(self.config)
        results = []

        for knowledge_id in knowledge_ids:
            knowledge_item = store.get(knowledge_id)
            if knowledge_item:
                results.append(knowledge_item)

        return results

    def _compute_similarities(
        self, query_embedding: torch.Tensor, knowledge_vectors: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """计算查询向量与知识向量之间的相似度

        参数:
            query_embedding: 查询嵌入向量
            knowledge_vectors: 知识向量字典

        返回:
            相似度数组
        """
        # 将查询向量转换为numpy数组
        if hasattr(query_embedding, "cpu"):
            query_np = query_embedding.cpu().numpy()
        else:
            query_np = np.array(query_embedding)

        # 确保形状正确
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        # 准备知识向量矩阵
        knowledge_ids = list(knowledge_vectors.keys())
        knowledge_matrix = np.vstack([knowledge_vectors[kid] for kid in knowledge_ids])

        # 计算余弦相似度
        similarities = cosine_similarity(query_np, knowledge_matrix)

        return similarities[0]  # 返回第一行（单个查询）

    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """融合向量检索和关键字检索的结果

        参数:
            vector_results: 向量检索结果
            keyword_results: 关键字检索结果
            limit: 返回结果数量

        返回:
            融合后的结果列表
        """
        # 创建ID到结果的映射
        vector_dict = {item["id"]: item for item in vector_results}

        # 计算融合分数
        fused_scores = {}

        # 向量结果分数（基于相似度）
        for knowledge_id, item in vector_dict.items():
            similarity = item.get("similarity", 0)
            fused_scores[knowledge_id] = {
                "item": item,
                "score": similarity * 0.7,  # 向量相似度权重
            }

        # 关键字结果分数（基于匹配）
        for knowledge_id, item in keyword_results.items():
            # 简单评分：如果在向量结果中已经存在，增加分数；否则，添加新条目
            if knowledge_id in fused_scores:
                fused_scores[knowledge_id]["score"] += 0.3  # 关键字匹配权重
            else:
                fused_scores[knowledge_id] = {
                    "item": item,
                    "score": 0.3,  # 基础关键字匹配分数
                }

        # 排序并返回
        sorted_items = sorted(
            fused_scores.values(), key=lambda x: x["score"], reverse=True
        )

        results = [score_info["item"] for score_info in sorted_items[:limit]]
        return results

    def _cache_results(
        self,
        query_embedding: torch.Tensor,
        knowledge_type: Optional[str],
        results: List[Dict[str, Any]],
    ):
        """缓存检索结果"""
        # 生成缓存键
        cache_key = self._generate_cache_key(query_embedding, knowledge_type)

        # 缓存结果
        self.cache[cache_key] = {
            "results": results,
            "timestamp": self._current_timestamp(),
        }

        # 清理旧缓存
        if len(self.cache) > self.cache_size:
            oldest_key = min(
                self.cache.keys(), key=lambda k: self.cache[k]["timestamp"]
            )
            del self.cache[oldest_key]

    def get_cached_results(
        self, query_embedding: torch.Tensor, knowledge_type: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """获取缓存的检索结果"""
        cache_key = self._generate_cache_key(query_embedding, knowledge_type)

        if cache_key in self.cache:
            # 检查缓存是否过期（例如，1小时）
            cache_age = self._current_timestamp() - self.cache[cache_key]["timestamp"]
            if cache_age < 3600:  # 1小时
                return self.cache[cache_key]["results"]
            else:
                # 移除过期缓存
                del self.cache[cache_key]

        return None  # 返回None

    def _generate_cache_key(
        self, query_embedding: torch.Tensor, knowledge_type: Optional[str]
    ) -> str:
        """生成缓存键"""
        # 将嵌入向量转换为字符串表示（完整）
        embedding_str = ""
        if hasattr(query_embedding, "cpu"):
            embedding_data = query_embedding.cpu().numpy()
        else:
            embedding_data = np.array(query_embedding)

        # 取前几个值和后几个值作为键
        if embedding_data.size > 0:
            flat_data = embedding_data.flatten()
            if len(flat_data) >= 4:
                embedding_str = (
                    f"{flat_data[0]:.4f}_{flat_data[1]:.4f}_"
                    f"{flat_data[-2]:.4f}_{flat_data[-1]:.4f}"
                )

        cache_key = f"{embedding_str}_{knowledge_type or 'all'}"
        return cache_key

    def _current_timestamp(self) -> float:
        """获取当前时间戳"""
        import time

        return time.time()

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.logger.info("检索缓存已清空")
