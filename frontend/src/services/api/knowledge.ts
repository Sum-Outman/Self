import { ApiClient, apiClient } from './client';

// 知识项接口
export interface KnowledgeItem {
  id: string;
  title: string;
  description: string;
  type: 'text' | 'image' | 'video' | 'audio' | 'document' | 'code' | 'dataset';
  content: string;
  size: number;
  upload_date: string;
  tags: string[];
  uploaded_by: string;
  access_count: number;
  last_accessed: string;
  file_url?: string;
  metadata?: Record<string, any>;
  embedding?: number[];
  similarity?: number;
}

// 知识库统计接口
export interface KnowledgeStats {
  total_items: number;
  total_size: number;
  by_type: Record<string, number>;
  by_month: Record<string, number>;
  popular_tags: string[];
  storage_usage: number;
  average_access_count: number;
}

// 搜索请求接口
export interface SearchRequest {
  query: string;
  type?: string;
  tags?: string[];
  start_date?: string;
  end_date?: string;
  limit?: number;
  offset?: number;
  sort_by?: 'relevance' | 'date' | 'access' | 'size';
  sort_order?: 'asc' | 'desc';
}

// 搜索响应接口
export interface SearchResponse {
  items: KnowledgeItem[];
  total: number;
  query_time: number;
  suggestions: string[];
}

// 上传请求接口
export interface UploadRequest {
  file: File;
  title: string;
  description?: string;
  tags?: string[];
  type?: string;
}

// 导入通用API响应接口
import { ApiResponse } from '../../types/api';

class KnowledgeApi {
  private apiClient: ApiClient;

  constructor() {
    this.apiClient = apiClient;
  }

  // 搜索知识项
  async search(request: SearchRequest): Promise<SearchResponse> {
    try {
      const response = (await this.apiClient.post('/knowledge/search', request)) as ApiResponse<SearchResponse>;
      if (!response.data) {
        throw new Error('搜索知识项失败：响应数据为空');
      }
      return response.data;
    } catch (error) {
      console.error('搜索知识项失败:', error);
      throw error;
    }
  }

  // 获取知识项详情
  async getItem(itemId: string): Promise<KnowledgeItem> {
    try {
      const response = (await this.apiClient.get(`/knowledge/items/${itemId}`)) as ApiResponse<KnowledgeItem>;
      if (!response.data) {
        throw new Error('获取知识项详情失败：响应数据为空');
      }
      return response.data;
    } catch (error) {
      console.error('获取知识项详情失败:', error);
      throw error;
    }
  }

  // 获取知识项列表
  async getItems(
    type?: string,
    tags?: string[],
    limit?: number,
    offset?: number
  ): Promise<KnowledgeItem[]> {
    try {
      const params = new URLSearchParams();
      if (type) params.append('type', type);
      if (tags && tags.length > 0) params.append('tags', tags.join(','));
      if (limit) params.append('limit', limit.toString());
      if (offset) params.append('offset', offset.toString());
      
      const response = (await this.apiClient.get(`/knowledge/items?${params.toString()}`)) as ApiResponse<KnowledgeItem[]>;
      if (!response.data) {
        throw new Error('获取知识项列表失败：响应数据为空');
      }
      return response.data;
    } catch (error) {
      console.error('获取知识项列表失败:', error);
      throw error;
    }
  }

  // 上传知识项
  async uploadItem(request: UploadRequest): Promise<ApiResponse<KnowledgeItem>> {
    try {
      const formData = new FormData();
      formData.append('file', request.file);
      formData.append('title', request.title);
      if (request.description) formData.append('description', request.description);
      if (request.tags) formData.append('tags', JSON.stringify(request.tags));
      if (request.type) formData.append('type', request.type);
      
      return await this.apiClient.post('/knowledge/items/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    } catch (error) {
      console.error('上传知识项失败:', error);
      throw error;
    }
  }

  // 更新知识项
  async updateItem(itemId: string, updates: Partial<KnowledgeItem>): Promise<ApiResponse<KnowledgeItem>> {
    try {
      return await this.apiClient.put(`/knowledge/items/${itemId}`, updates);
    } catch (error) {
      console.error('更新知识项失败:', error);
      throw error;
    }
  }

  // 删除知识项
  async deleteItem(itemId: string): Promise<ApiResponse> {
    try {
      return await this.apiClient.delete(`/knowledge/items/${itemId}`);
    } catch (error) {
      console.error('删除知识项失败:', error);
      throw error;
    }
  }

  // 获取知识库统计
  async getStats(): Promise<ApiResponse<KnowledgeStats>> {
    try {
      return await this.apiClient.get('/knowledge/stats');
    } catch (error) {
      console.error('获取知识库统计失败:', error);
      throw error;
    }
  }

  // 获取标签列表
  async getTags(): Promise<ApiResponse<string[]>> {
    try {
      return await this.apiClient.get('/knowledge/tags');
    } catch (error) {
      console.error('获取标签列表失败:', error);
      throw error;
    }
  }

  // 获取类型统计
  async getTypeStats(): Promise<ApiResponse<Record<string, number>>> {
    try {
      return await this.apiClient.get('/knowledge/types/stats');
    } catch (error) {
      console.error('获取类型统计失败:', error);
      throw error;
    }
  }

  // 批量导入知识项
  async importItems(items: Omit<KnowledgeItem, 'id' | 'upload_date' | 'uploaded_by'>[]): Promise<ApiResponse<{success: number, failed: number}>> {
    try {
      return await this.apiClient.post('/knowledge/import', { items });
    } catch (error) {
      console.error('批量导入知识项失败:', error);
      throw error;
    }
  }

  // 导出知识项
  async exportItems(
    type?: string,
    tags?: string[],
    format: 'json' | 'csv' = 'json'
  ): Promise<ApiResponse<{download_url: string, format: string}>> {
    try {
      const params = new URLSearchParams();
      if (type) params.append('type', type);
      if (tags && tags.length > 0) params.append('tags', tags.join(','));
      params.append('format', format);
      
      return await this.apiClient.get(`/knowledge/export?${params.toString()}`);
    } catch (error) {
      console.error('导出知识项失败:', error);
      throw error;
    }
  }

  // 获取相似知识项
  async getSimilarItems(itemId: string, limit: number = 5): Promise<ApiResponse<KnowledgeItem[]>> {
    try {
      return await this.apiClient.get(`/knowledge/items/${itemId}/similar?limit=${limit}`);
    } catch (error) {
      console.error('获取相似知识项失败:', error);
      throw error;
    }
  }

  // 记录知识项访问
  async recordAccess(itemId: string): Promise<ApiResponse> {
    try {
      return await this.apiClient.post(`/knowledge/items/${itemId}/access`);
    } catch (error) {
      console.error('记录知识项访问失败:', error);
      // 不抛出错误，因为这只是分析数据
      return {
        success: false,
        message: '记录访问失败'
      };
    }
  }

  // 获取知识图谱
  async getKnowledgeGraph(
    centerItemId?: string,
    depth: number = 2
  ): Promise<ApiResponse<{nodes: any[], edges: any[]}>> {
    try {
      const params = new URLSearchParams();
      if (centerItemId) params.append('center', centerItemId);
      params.append('depth', depth.toString());
      
      return await this.apiClient.get(`/knowledge/graph?${params.toString()}`);
    } catch (error) {
      console.error('获取知识图谱失败:', error);
      throw error;
    }
  }

  // 文本提取（从上传的文件中提取文本）
  async extractText(file: File): Promise<ApiResponse<{text: string, metadata: any}>> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      return await this.apiClient.post('/knowledge/extract-text', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    } catch (error) {
      console.error('文本提取失败:', error);
      throw error;
    }
  }

  // 生成摘要
  async generateSummary(text: string, length: 'short' | 'medium' | 'long' = 'medium'): Promise<ApiResponse<{summary: string}>> {
    try {
      return await this.apiClient.post('/knowledge/summarize', { text, length });
    } catch (error) {
      console.error('生成摘要失败:', error);
      throw error;
    }
  }
}

// 创建单例实例
export const knowledgeApi = new KnowledgeApi();

// 默认导出
export default knowledgeApi;