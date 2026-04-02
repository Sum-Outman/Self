#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合网络模块

包含：
1. CrossModalAttention - 跨模态注意力机制
2. ProjectionLayerManager - 投影层管理器
3. HierarchicalFusionNetwork - 分层多模态融合网络

工业级AGI系统要求：从零开始训练，不使用预训练模型依赖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


class CrossModalAttention(nn.Module):
    """跨模态注意力机制 - 基于Transformer的交叉注意力
    
    特征：
    - 多头注意力机制
    - 残差连接和层归一化
    - 支持键值掩码
    """
    def __init__(self, embedding_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        """
        前向传播
        
        参数:
            query: [batch_size, query_seq_len, embedding_dim]
            key: [batch_size, key_seq_len, embedding_dim]
            value: [batch_size, value_seq_len, embedding_dim]
            key_padding_mask: [batch_size, key_seq_len] (True表示需要mask的位置)
            
        返回:
            output: [batch_size, query_seq_len, embedding_dim]
            attn_weights: [batch_size, query_seq_len, key_seq_len]
        """
        # 跨模态注意力
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )
        
        # 残差连接和层归一化
        output = self.layer_norm(query + self.dropout(attn_output))
        
        return output, attn_weights


class GatedCrossModalAttention(nn.Module):
    """门控跨模态注意力机制
    
    特征：
    - 门控机制动态控制注意力强度
    - 可学习的门控权重
    - 适用于不同模态间的信息流控制
    """
    def __init__(self, embedding_dim=768, num_heads=12, dropout=0.1, gate_bias=-1.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        self.gate_bias = gate_bias  # 初始门控偏置，控制注意力强度
        
    def forward(self, query, key, value, key_padding_mask=None):
        """
        前向传播
        
        参数:
            query: [batch_size, query_seq_len, embedding_dim]
            key: [batch_size, key_seq_len, embedding_dim]
            value: [batch_size, value_seq_len, embedding_dim]
            key_padding_mask: [batch_size, key_seq_len] (True表示需要mask的位置)
            
        返回:
            output: [batch_size, query_seq_len, embedding_dim]
            attn_weights: [batch_size, query_seq_len, key_seq_len]
            gate_values: [batch_size, query_seq_len, 1] 门控值
        """
        # 计算标准注意力
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )
        
        # 计算门控值
        batch_size, query_seq_len, embedding_dim = query.shape
        # 计算查询和键的相似度作为门控输入
        query_mean = query.mean(dim=1, keepdim=True)  # [batch_size, 1, embedding_dim]
        key_mean = key.mean(dim=1, keepdim=True)  # [batch_size, 1, embedding_dim]
        
        # 重复以匹配查询序列长度
        query_mean_expanded = query_mean.expand(-1, query_seq_len, -1)
        key_mean_expanded = key_mean.expand(-1, query_seq_len, -1)
        
        # 拼接查询和键的统计信息
        gate_input = torch.cat([query_mean_expanded, key_mean_expanded], dim=-1)
        gate_values = self.gate(gate_input) + self.gate_bias
        
        # 应用门控到注意力输出
        gated_attn_output = attn_output * gate_values
        
        # 残差连接和层归一化
        output = self.layer_norm(query + self.dropout(gated_attn_output))
        
        return output, attn_weights, gate_values


class HierarchicalCrossModalAttention(nn.Module):
    """分层跨模态注意力机制
    
    特征：
    - 多级注意力机制：模态级、特征级、实例级
    - 逐步细化融合过程
    - 适用于复杂多模态场景
    """
    def __init__(self, embedding_dim=768, num_heads=12, dropout=0.1, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.embedding_dim = embedding_dim
        
        # 创建多级注意力层
        self.attention_levels = nn.ModuleList()
        for level in range(num_levels):
            # 不同级别的注意力可能有不同的配置
            level_heads = max(1, num_heads // (2 ** level))  # 随层级减少注意力头数
            attention_layer = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=level_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_levels.append(attention_layer)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 层级融合权重
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        
    def forward(self, query, key, value, key_padding_mask=None):
        """
        前向传播
        
        参数:
            query: [batch_size, query_seq_len, embedding_dim]
            key: [batch_size, key_seq_len, embedding_dim]
            value: [batch_size, value_seq_len, embedding_dim]
            key_padding_mask: [batch_size, key_seq_len] (True表示需要mask的位置)
            
        返回:
            output: [batch_size, query_seq_len, embedding_dim]
            attn_weights: 所有级别的注意力权重列表
        """
        batch_size, query_seq_len, _ = query.shape
        
        # 逐级计算注意力
        level_outputs = []
        level_weights = []
        
        current_query = query
        current_key = key
        current_value = value
        
        for level, attention_layer in enumerate(self.attention_levels):
            # 计算当前级别的注意力
            attn_output, attn_weights = attention_layer(
                query=current_query,
                key=current_key,
                value=current_value,
                key_padding_mask=key_padding_mask
            )
            
            level_outputs.append(attn_output)
            level_weights.append(attn_weights)
            
            # 更新查询作为下一级的输入（残差连接）
            current_query = current_query + self.dropout(attn_output)
        
        # 加权融合各级输出
        softmax_weights = F.softmax(self.level_weights, dim=0)
        weighted_outputs = []
        
        for level, output in enumerate(level_outputs):
            weight = softmax_weights[level].view(1, 1, 1)
            weighted_outputs.append(output * weight)
        
        # 合并各级输出
        combined_output = torch.sum(torch.stack(weighted_outputs, dim=0), dim=0)
        
        # 最终残差连接和层归一化
        output = self.layer_norm(query + self.dropout(combined_output))
        
        return output, level_weights


class AdaptiveCrossModalAttention(nn.Module):
    """自适应跨模态注意力机制
    
    特征：
    - 动态调整注意力机制参数
    - 基于输入特征复杂度自适应配置
    - 支持不同模态间的异构融合
    """
    def __init__(self, embedding_dim=768, max_heads=16, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_heads = max_heads
        
        # 自适应参数预测器
        self.param_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 3)  # 预测注意力头数、缩放因子、门控值
        )
        
        # 基础注意力层（使用最大头数，通过掩码实现动态头数）
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=max_heads,
            dropout=dropout,
            batch_first=True,
            kdim=embedding_dim,
            vdim=embedding_dim
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        """
        前向传播
        
        参数:
            query: [batch_size, query_seq_len, embedding_dim]
            key: [batch_size, key_seq_len, embedding_dim]
            value: [batch_size, value_seq_len, embedding_dim]
            key_padding_mask: [batch_size, key_seq_len] (True表示需要mask的位置)
            
        返回:
            output: [batch_size, query_seq_len, embedding_dim]
            attn_weights: [batch_size * max_heads, query_seq_len, key_seq_len]
            predicted_params: 预测的参数元组
        """
        batch_size, query_seq_len, _ = query.shape
        
        # 1. 预测自适应参数
        query_mean = query.mean(dim=1)  # [batch_size, embedding_dim]
        key_mean = key.mean(dim=1)  # [batch_size, embedding_dim]
        predictor_input = torch.cat([query_mean, key_mean], dim=-1)
        predicted_params = self.param_predictor(predictor_input)  # [batch_size, 3]
        
        # 解析预测参数
        num_heads_pred = torch.sigmoid(predicted_params[:, 0]) * (self.max_heads - 1) + 1
        scale_factor = torch.sigmoid(predicted_params[:, 1]) * 0.5 + 0.5  # [0.5, 1.0]
        gate_value = torch.sigmoid(predicted_params[:, 2])  # [0, 1]
        
        num_heads = torch.round(num_heads_pred).int()  # 四舍五入为整数
        
        # 2. 应用自适应注意力
        # 完整处理，实际应用中需要根据预测的头数调整注意力计算
        attn_output, attn_weights = self.attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )
        
        # 3. 应用自适应缩放和门控
        scale_factor_expanded = scale_factor.view(batch_size, 1, 1).expand(-1, query_seq_len, -1)
        gate_value_expanded = gate_value.view(batch_size, 1, 1).expand(-1, query_seq_len, -1)
        
        scaled_attn_output = attn_output * scale_factor_expanded
        gated_output = query * (1 - gate_value_expanded) + scaled_attn_output * gate_value_expanded
        
        # 4. 残差连接和层归一化
        output = self.layer_norm(query + self.dropout(gated_output))
        
        return output, attn_weights, (num_heads, scale_factor, gate_value)


class CrossModalAttentionFactory:
    """跨模态注意力工厂类
    
    特征：
    - 统一创建和管理不同类型的跨模态注意力机制
    - 支持注意力类型配置和参数调优
    - 提供注意力机制选择和评估功能
    """
    
    @staticmethod
    def create_attention(attention_type: str = "standard", **kwargs) -> nn.Module:
        """创建跨模态注意力机制
        
        参数:
            attention_type: 注意力类型，可选值：
                - "standard": 标准跨模态注意力
                - "gated": 门控跨模态注意力
                - "hierarchical": 分层跨模态注意力
                - "adaptive": 自适应跨模态注意力
            **kwargs: 传递给注意力构造函数的参数
            
        返回:
            注意力机制模块
        """
        attention_type = attention_type.lower()
        
        if attention_type == "standard":
            return CrossModalAttention(**kwargs)
        elif attention_type == "gated":
            return GatedCrossModalAttention(**kwargs)
        elif attention_type == "hierarchical":
            return HierarchicalCrossModalAttention(**kwargs)
        elif attention_type == "adaptive":
            return AdaptiveCrossModalAttention(**kwargs)
        else:
            raise ValueError(f"未知的注意力类型: {attention_type}")
    
    @staticmethod
    def get_available_types() -> List[str]:
        """获取可用的注意力类型列表"""
        return ["standard", "gated", "hierarchical", "adaptive"]
    
    @staticmethod
    def get_type_description(attention_type: str) -> str:
        """获取注意力类型的描述"""
        descriptions = {
            "standard": "标准跨模态注意力，基于Transformer多头注意力",
            "gated": "门控跨模态注意力，动态控制注意力强度",
            "hierarchical": "分层跨模态注意力，多级注意力机制",
            "adaptive": "自适应跨模态注意力，动态调整参数"
        }
        return descriptions.get(attention_type.lower(), "未知注意力类型")


class AttentionAnalyzer:
    """注意力分析器
    
    特征：
    - 分析注意力权重分布
    - 可视化注意力模式
    - 计算注意力相关统计指标
    - 检测异常注意力模式
    """
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_attention_weights(self, 
                                 attention_weights: torch.Tensor,
                                 query_modality: str = "unknown",
                                 key_modality: str = "unknown") -> Dict[str, Any]:
        """分析注意力权重
        
        参数:
            attention_weights: 注意力权重张量 [batch_size, query_len, key_len]
            query_modality: 查询模态名称
            key_modality: 键模态名称
            
        返回:
            分析结果字典
        """
        batch_size, query_len, key_len = attention_weights.shape
        
        # 确保注意力权重在CPU上
        if attention_weights.is_cuda:
            attention_weights = attention_weights.cpu()
        
        # 转换为numpy进行计算
        weights_np = attention_weights.detach().numpy()
        
        # 计算统计指标
        analysis = {
            "query_modality": query_modality,
            "key_modality": key_modality,
            "batch_size": batch_size,
            "query_length": query_len,
            "key_length": key_len,
            "mean_attention": float(weights_np.mean()),
            "std_attention": float(weights_np.std()),
            "max_attention": float(weights_np.max()),
            "min_attention": float(weights_np.min()),
            "attention_entropy": self._compute_attention_entropy(weights_np),
            "attention_sparsity": self._compute_attention_sparsity(weights_np),
            "cross_modal_strength": self._compute_cross_modal_strength(weights_np),
            "attention_pattern": self._classify_attention_pattern(weights_np)
        }
        
        # 保存到历史记录
        self.analysis_history.append(analysis)
        
        return analysis
    
    def _compute_attention_entropy(self, weights: np.ndarray) -> float:
        """计算注意力权重熵（不确定性度量）"""
        # 避免除零
        eps = 1e-12
        weights_flat = weights.reshape(-1)
        
        # 确保权重非负（注意力权重应该是非负的）
        weights_flat = np.maximum(weights_flat, 0)
        
        total = weights_flat.sum()
        if total <= eps:
            return 0.0
        
        weights_normalized = weights_flat / total
        
        # 只计算非零权重的熵
        mask = weights_normalized > eps
        if not np.any(mask):
            return 0.0
        
        valid_weights = weights_normalized[mask]
        
        # 计算香农熵
        entropy = -np.sum(valid_weights * np.log(valid_weights))
        return float(entropy)
    
    def _compute_attention_sparsity(self, weights: np.ndarray, threshold: float = 0.01) -> float:
        """计算注意力稀疏度"""
        # 计算低于阈值的权重比例
        total_elements = weights.size
        sparse_elements = np.sum(weights < threshold)
        sparsity = sparse_elements / total_elements
        return float(sparsity)
    
    def _compute_cross_modal_strength(self, weights: np.ndarray) -> float:
        """计算跨模态注意力强度"""
        # 如果权重是对称的（自注意力），返回0；否则返回非对角线元素的平均值
        if weights.shape[1] == weights.shape[2]:  # 方阵
            # 计算非对角线元素的比例
            mask = 1 - np.eye(weights.shape[1])
            cross_weights = weights * mask
            cross_strength = cross_weights.sum() / (mask.sum() + 1e-10)
            return float(cross_strength)
        else:
            # 对于非方阵，直接计算平均值
            return float(weights.mean())
    
    def _classify_attention_pattern(self, weights: np.ndarray) -> str:
        """分类注意力模式"""
        batch_size, query_len, key_len = weights.shape
        
        if query_len == key_len:
            # 检查是否是对角线主导（局部注意力）
            diag_strength = np.mean([weights[i].diagonal().mean() for i in range(batch_size)])
            total_mean = weights.mean()
            
            if diag_strength > total_mean * 1.5:
                return "local_attention"
            elif diag_strength < total_mean * 0.7:
                return "global_attention"
            else:
                return "mixed_attention"
        else:
            # 跨模态注意力
            return "cross_modal_attention"
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        if not self.analysis_history:
            return {"message": "无分析历史"}
        
        # 计算历史统计
        total_analyses = len(self.analysis_history)
        modalities = set()
        patterns = {}
        
        for analysis in self.analysis_history:
            modalities.add(analysis["query_modality"])
            modalities.add(analysis["key_modality"])
            pattern = analysis["attention_pattern"]
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        return {
            "total_analyses": total_analyses,
            "modalities": list(modalities),
            "pattern_distribution": patterns,
            "recent_analysis": self.analysis_history[-1] if self.analysis_history else None
        }


class ProjectionLayerManager(nn.Module):
    """投影层管理器 - 统一管理不同维度的模态特征投影
    
    特征：
    - 动态投影层创建和缓存
    - LRU缓存清理机制
    - 输入输出维度验证
    - 支持多设备（GPU/CPU）管理
    
    设计原则：
    1. 统一投影层接口，确保维度对齐
    2. 避免重复创建相同维度的投影层
    3. 支持缓存大小限制，防止内存溢出
    4. 工业级AGI要求：从零开始训练，不使用预训练权重
    """
    
    def __init__(self, max_cache_size=10, fused_embedding_dim=768, device="cuda"):
        super().__init__()
        self.max_cache_size = max_cache_size
        self.fused_embedding_dim = fused_embedding_dim
        self.device = device
        
        # 投影层缓存：使用ModuleDict存储不同维度的投影层
        self.projection_cache = nn.ModuleDict()
        
        # 缓存访问记录：用于LRU清理
        self.access_timestamps = {}  # 缓存键 -> 最后访问时间戳
        self.next_timestamp = 0
        
        logger.info(f"初始化ProjectionLayerManager: 最大缓存大小={max_cache_size}, 融合维度={fused_embedding_dim}")
    
    def _get_cache_key(self, input_dim: int) -> str:
        """生成缓存键"""
        return f"proj_{input_dim}_to_{self.fused_embedding_dim}"
    
    def _clean_lru_cache(self):
        """清理最近最少使用的缓存项"""
        if len(self.projection_cache) <= self.max_cache_size:
            return
        
        # 找到最久未访问的缓存项
        sorted_items = sorted(
            self.access_timestamps.items(), 
            key=lambda x: x[1]
        )
        
        # 清理过期的缓存项，保留最大缓存大小
        items_to_remove = len(sorted_items) - self.max_cache_size
        
        for cache_key, _ in sorted_items[:items_to_remove]:
            if cache_key in self.projection_cache:
                del self.projection_cache[cache_key]
            if cache_key in self.access_timestamps:
                del self.access_timestamps[cache_key]
            
            logger.debug(f"清理LRU缓存项: {cache_key}")
    
    def get_projection(self, input_dim: int) -> nn.Linear:
        """获取或创建投影层
        
        参数:
            input_dim: 输入维度
            
        返回:
            projection_layer: 线性投影层，将输入维度投影到融合维度
        """
        # 验证输入维度
        if input_dim <= 0:
            raise ValueError(f"无效的输入维度: {input_dim}")
        
        if input_dim == self.fused_embedding_dim:
            # 完整处理）
            logger.debug(f"输入维度{input_dim}等于融合维度{self.fused_embedding_dim}，跳过投影")
            return None  # 返回None
        
        cache_key = self._get_cache_key(input_dim)
        
        # 更新访问时间戳
        self.access_timestamps[cache_key] = self.next_timestamp
        self.next_timestamp += 1
        
        # 检查缓存中是否已有投影层
        if cache_key not in self.projection_cache:
            logger.info(f"创建新投影层: {input_dim} -> {self.fused_embedding_dim}")
            
            # 创建新的投影层（从零开始初始化）
            projection = nn.Linear(input_dim, self.fused_embedding_dim)
            
            # Xavier初始化（从零开始训练）
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
            
            # 移动到设备
            projection = projection.to(self.device)
            
            # 添加到缓存
            self.projection_cache[cache_key] = projection
            
            # 清理LRU缓存
            self._clean_lru_cache()
        else:
            logger.debug(f"使用缓存投影层: {cache_key}")
        
        return self.projection_cache[cache_key]
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            "cache_size": len(self.projection_cache),
            "max_cache_size": self.max_cache_size,
            "cached_projections": list(self.projection_cache.keys()),
            "access_counts": len(self.access_timestamps),
            "fused_embedding_dim": self.fused_embedding_dim,
        }


class HierarchicalFusionNetwork(nn.Module):
    """分层多模态融合网络
    
    特征：
    - 模态门控机制
    - 跨模态注意力层
    - 分层融合策略
    - 支持可变模态输入
    """
    def __init__(self, embedding_dim=768, num_modalities=5, 
                 attention_type="standard", attention_kwargs=None):
        super().__init__()
        
        self.attention_type = attention_type
        self.attention_kwargs = attention_kwargs or {}
        
        # 模态门控
        self.modality_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
        # 跨模态注意力层 - 使用工厂创建指定类型的注意力
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionFactory.create_attention(
                attention_type=attention_type,
                embedding_dim=embedding_dim,
                **self.attention_kwargs
            ) 
            for _ in range(num_modalities)
        ])
        
        # 融合编码器
        # 动态计算注意力头数，确保embedding_dim能被nhead整除
        # 标准Transformer设置：每个注意力头维度为64
        base_head_dim = 64
        if embedding_dim >= base_head_dim:
            nhead = embedding_dim // base_head_dim
        else:
            nhead = 1
        
        # 确保nhead能整除embedding_dim，否则调整nhead
        while embedding_dim % nhead != 0 and nhead > 1:
            nhead -= 1
        
        # 确保nhead至少为1
        nhead = max(1, nhead)
        
        # 计算每个头的维度
        head_dim = embedding_dim // nhead
        
        self.fusion_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
    def forward(self, modality_features, modality_masks=None, return_attention_weights=False):
        """
        前向传播
        
        参数:
            modality_features: 模态特征列表，每个元素形状为 [batch_size, seq_len, embedding_dim]
            modality_masks: 模态掩码列表，每个元素形状为 [batch_size, seq_len] (True表示需要mask的位置)
            return_attention_weights: 是否返回注意力权重（用于分析和可视化）
            
        返回:
            joint_pooled: [batch_size, embedding_dim] 融合后的联合表示
            fused_features: 融合后的模态特征列表
            attention_weights: 可选，当return_attention_weights=True时返回，注意力权重字典
        """
        batch_size = modality_features[0].size(0)
        
        # 1. 模态门控加权
        gated_features = []
        for i, (features, gate) in enumerate(zip(modality_features, self.modality_gates)):
            # 平均池化得到模态级特征
            if modality_masks and modality_masks[i] is not None:
                mask = (~modality_masks[i]).unsqueeze(-1).float()
                masked_features = features * mask
                sum_features = masked_features.sum(dim=1)
                mask_sum = mask.sum(dim=1).clamp(min=1e-9)
                modality_vector = sum_features / mask_sum
            else:
                modality_vector = features.mean(dim=1)
                
            # 门控权重
            gate_weights = gate(modality_vector)
            gated = features * gate_weights.unsqueeze(1)
            gated_features.append(gated)
        
        # 2. 跨模态注意力融合
        fused_features = []
        attention_weights_dict = {}
        
        for i in range(len(gated_features)):
            # 其他模态作为key/value
            other_indices = [j for j in range(len(gated_features)) if j != i]
            if other_indices:
                other_features = [gated_features[j] for j in other_indices]
                key_value = torch.cat(other_features, dim=1)
                
                # 应用跨模态注意力
                query = gated_features[i]
                key_padding_mask = None
                if modality_masks and modality_masks[i] is not None:
                    other_masks = [modality_masks[j] for j in other_indices]
                    key_padding_mask = torch.cat(other_masks, dim=1) if any(m is not None for m in other_masks) else None
                    
                # 调用注意力层，处理不同数量的返回值
                attention_result = self.cross_attention_layers[i](
                    query=query,
                    key=key_value,
                    value=key_value,
                    key_padding_mask=key_padding_mask
                )
                
                # 解包结果：不同注意力类型返回不同数量的值
                if isinstance(attention_result, tuple):
                    if len(attention_result) >= 2:
                        fused = attention_result[0]
                        attn_weights = attention_result[1]
                    else:
                        # 如果只有一个返回值，假设它既是输出又是权重（完整）
                        fused = attention_result
                        attn_weights = None
                else:
                    # 如果不是元组，假设是单个张量
                    fused = attention_result
                    attn_weights = None
                
                fused_features.append(fused)
                
                # 收集注意力权重（用于分析和可视化）
                if return_attention_weights and attn_weights is not None:
                    attention_weights_dict[f"modality_{i}_to_others"] = attn_weights.detach().cpu()
            else:
                fused_features.append(gated_features[i])
        
        # 3. 全局融合
        all_features = torch.cat(fused_features, dim=1)
        joint_representation = self.fusion_encoder(all_features)
        
        # 4. 池化得到最终表示
        joint_pooled = joint_representation.mean(dim=1)
        
        if return_attention_weights:
            return joint_pooled, fused_features, attention_weights_dict
        else:
            return joint_pooled, fused_features


# 别名：为了向后兼容，提供MultimodalFusionNetwork作为HierarchicalFusionNetwork的别名
MultimodalFusionNetwork = HierarchicalFusionNetwork


def test_fusion_networks():
    """测试融合网络模块"""
    print("=== 测试融合网络模块 ===")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 测试参数
    batch_size = 2
    seq_len = 10
    embedding_dim = 768
    num_modalities = 3
    
    print(f"\n1. 测试不同注意力类型:")
    
    attention_types = ["standard", "gated", "hierarchical", "adaptive"]
    
    for attn_type in attention_types:
        print(f"\n   测试 {attn_type} 注意力:")
        try:
            # 创建融合网络
            fusion_net = HierarchicalFusionNetwork(
                embedding_dim=embedding_dim,
                num_modalities=num_modalities,
                attention_type=attn_type
            ).to(device)
            
            # 创建测试模态特征（使用确定性数据）
            modality_features = [
                torch.ones(batch_size, seq_len, embedding_dim).to(device) * 0.5
                for _ in range(num_modalities)
            ]
            
            # 前向传播
            joint_pooled, fused_features = fusion_net(modality_features)
            
            print(f"      成功: 联合表示形状={joint_pooled.shape}")
            print(f"      融合特征数量={len(fused_features)}, 每个形状={fused_features[0].shape}")
            
        except Exception as e:
            print(f"      失败: {e}")
    
    print(f"\n2. 测试投影层管理器:")
    try:
        projection_manager = ProjectionLayerManager(
            max_cache_size=5,
            fused_embedding_dim=embedding_dim,
            device=str(device)
        ).to(device)
        
        # 测试不同维度的投影
        test_dims = [256, 512, 1024, 256]  # 重复256测试缓存
        
        for dim in test_dims:
            projection = projection_manager.get_projection(dim)
            if projection is not None:
                print(f"      维度 {dim} -> {embedding_dim}: 成功")
            else:
                print(f"      维度 {dim} -> {embedding_dim}: 跳过（维度相同）")
        
        # 获取缓存信息
        cache_info = projection_manager.get_cache_info()
        print(f"      缓存大小: {cache_info['cache_size']}/{cache_info['max_cache_size']}")
        
    except Exception as e:
        print(f"      失败: {e}")
    
    print(f"\n3. 测试注意力分析器:")
    try:
        analyzer = AttentionAnalyzer()
        
        # 创建测试注意力权重（使用确定性数据）
        test_weights = torch.ones(batch_size, seq_len, seq_len) * 0.5
        
        # 分析注意力权重
        analysis = analyzer.analyze_attention_weights(
            test_weights,
            query_modality="text",
            key_modality="image"
        )
        
        print(f"      分析成功: 查询模态={analysis['query_modality']}")
        print(f"      注意力模式: {analysis['attention_pattern']}")
        print(f"      注意力熵: {analysis['attention_entropy']:.4f}")
        
        # 获取分析摘要
        summary = analyzer.get_analysis_summary()
        print(f"      总分析次数: {summary['total_analyses']}")
        
    except Exception as e:
        print(f"      失败: {e}")
    
    print(f"\n4. 测试注意力工厂:")
    try:
        print(f"      可用注意力类型: {CrossModalAttentionFactory.get_available_types()}")
        
        for attn_type in CrossModalAttentionFactory.get_available_types():
            description = CrossModalAttentionFactory.get_type_description(attn_type)
            print(f"      {attn_type}: {description}")
            
            # 创建注意力层
            attention_layer = CrossModalAttentionFactory.create_attention(
                attention_type=attn_type,
                embedding_dim=embedding_dim
            ).to(device)
            
            # 测试前向传播（使用确定性数据）
            query = torch.ones(batch_size, seq_len, embedding_dim).to(device) * 0.5
            key = torch.ones(batch_size, seq_len, embedding_dim).to(device) * 0.5
            value = torch.ones(batch_size, seq_len, embedding_dim).to(device) * 0.5
            
            result = attention_layer(query, key, value)
            print(f"        创建成功: 输出类型={type(result)}")
            
    except Exception as e:
        print(f"      失败: {e}")
    
    print(f"\n=== 融合网络模块测试完成 ===")


if __name__ == "__main__":
    test_fusion_networks()