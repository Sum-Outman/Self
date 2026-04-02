/**
 * 记忆系统API服务
 * 处理记忆系统的数据获取和操作
 */

import { apiClient } from './client';
import { ApiResponse } from '../../types/api';

// 记忆统计信息接口
export interface MemoryStats {
  total_memories: number;
  short_term_memories: number;
  long_term_memories: number;
  working_memory_usage: number;
  cache_hit_rate: number;
  average_retrieval_time_ms: number;
  memory_usage_mb: number;
  autonomous_optimizations: number;
  scene_transitions: number;
  current_scene: string;
  reasoning_operations: number;
  last_updated: string;
  system_initialized?: boolean;  // 新增：系统初始化状态
}

// 记忆统计API响应接口（包含系统初始化状态）
export interface MemoryStatsResponse {
  success: boolean;
  data: MemoryStats;
  system_initialized: boolean;
  message?: string;
  timestamp?: string;
}

// 记忆项接口
export interface MemoryItem {
  id: string;
  content: string;
  type: 'short_term' | 'long_term' | 'working';
  created_at: string;
  accessed_at: string;
  importance: number;
  similarity: number;
  scene_type?: string;
  source: 'user' | 'system' | 'autonomous';
}

// 记忆搜索请求接口
export interface MemorySearchRequest {
  query: string;
  memory_type?: 'short_term' | 'long_term' | 'working' | string;
  scene_type?: string;
  min_importance?: number;
  min_similarity?: number;
  start_date?: string;
  end_date?: string;
  limit?: number;
  offset?: number;
}

// 记忆搜索响应接口
export interface MemorySearchResponse {
  memories: MemoryItem[];
  total: number;
  query_time_ms: number;
}

// 知识库搜索请求接口
export interface KnowledgeSearchRequest {
  query: string;
  top_k?: number;
  similarity_threshold?: number;
}

// 混合搜索请求接口
export interface HybridSearchRequest {
  query: string;
  top_k?: number;
  memory_weight?: number;
  knowledge_weight?: number;
}

// 混合搜索结果接口
export interface HybridSearchResult {
  id: string;
  content: string;
  source: 'memory' | 'knowledge';
  similarity_score: number;
  weighted_score: number;
  metadata?: Record<string, any>;
}

// 混合搜索响应接口
export interface HybridSearchResponse {
  results: HybridSearchResult[];
  memory_count: number;
  knowledge_count: number;
  total_time_ms: number;
}

// 系统配置更新接口
export interface SystemConfigUpdate {
  enable_autonomous_memory?: boolean;
  enable_scene_classification?: boolean;
  enable_knowledge_integration?: boolean;
  working_memory_capacity?: number;
  cache_size_mb?: number;
  similarity_threshold?: number;
}

// 系统配置响应接口
export interface SystemConfigResponse {
  config: Record<string, any>;
  last_modified: string;
}

// 记忆系统健康状态接口
export interface MemoryHealthStatus {
  status: 'healthy' | 'degraded' | 'not_initialized' | 'unavailable';
  message: string;
  available: boolean;
  features?: {
    autonomous_memory: boolean;
    scene_classification: boolean;
    knowledge_integration: boolean;
    multimodal_memory: boolean;
  };
}

class MemoryApi {
  private apiClient: typeof apiClient;

  constructor() {
    this.apiClient = apiClient;
  }

  // 获取记忆系统统计信息
  async getStats(): Promise<MemoryStatsResponse> {
    try {
      // apiClient.get返回ApiResponse<MemoryStats>，但我们需要MemoryStatsResponse
      const response = await this.apiClient.get('/memory/stats');
      // 将ApiResponse<MemoryStats>转换为MemoryStatsResponse
      // 假设后端返回system_initialized在顶层
      return response as unknown as MemoryStatsResponse;
    } catch (error) {
      console.error('获取记忆统计信息失败:', error);
      throw error;
    }
  }

  // 搜索记忆
  async searchMemories(request: MemorySearchRequest): Promise<ApiResponse<MemorySearchResponse>> {
    try {
      return await this.apiClient.post('/memory/search', request);
    } catch (error) {
      console.error('搜索记忆失败:', error);
      throw error;
    }
  }

  // 从知识库搜索信息
  async searchKnowledge(request: KnowledgeSearchRequest): Promise<ApiResponse<any>> {
    try {
      return await this.apiClient.post('/memory/knowledge/search', request);
    } catch (error) {
      console.error('搜索知识库失败:', error);
      throw error;
    }
  }

  // 混合搜索
  async hybridSearch(request: HybridSearchRequest): Promise<ApiResponse<HybridSearchResponse>> {
    try {
      return await this.apiClient.post('/memory/hybrid/search', request);
    } catch (error) {
      console.error('混合搜索失败:', error);
      throw error;
    }
  }

  // 获取最近记忆
  async getRecentMemories(limit: number = 10): Promise<ApiResponse<MemoryItem[]>> {
    try {
      return await this.apiClient.get(`/memory/recent?limit=${limit}`);
    } catch (error) {
      console.error('获取最近记忆失败:', error);
      throw error;
    }
  }

  // 获取系统配置（仅管理员）
  async getSystemConfig(): Promise<ApiResponse<SystemConfigResponse>> {
    try {
      return await this.apiClient.get('/memory/config');
    } catch (error) {
      console.error('获取系统配置失败:', error);
      throw error;
    }
  }

  // 更新系统配置（仅管理员）
  async updateSystemConfig(config: SystemConfigUpdate): Promise<ApiResponse<any>> {
    try {
      return await this.apiClient.put('/memory/config', config);
    } catch (error) {
      console.error('更新系统配置失败:', error);
      throw error;
    }
  }

  // 获取记忆系统健康状态
  async getHealthStatus(): Promise<ApiResponse<MemoryHealthStatus>> {
    try {
      return await this.apiClient.get('/memory/health');
    } catch (error) {
      console.error('获取记忆系统健康状态失败:', error);
      throw error;
    }
  }
}

// 导出单例实例
export const memoryApi = new MemoryApi();