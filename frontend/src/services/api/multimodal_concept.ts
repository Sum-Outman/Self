/**
 * 多模态概念认知API服务
 * 提供多模态认知等高级AGI能力
 */

import { apiClient } from './client';
import { 
  ServiceInfo
} from '../../types/multimodal';

/**
 * 多模态概念认知API
 */
export const multimodalConceptApi = {
  /**
   * 获取多模态概念服务信息
   * GET /api/multimodal/concept/service-info
   */
  async getServiceInfo(): Promise<ServiceInfo> {
    try {
      const response = await apiClient.get<ServiceInfo>('/multimodal/concept/service-info');
      return response;
    } catch (error) {
      console.error('获取多模态概念服务信息失败:', error);
      throw error;
    }
  },

  /**
   * 测试多模态概念服务连通性
   * GET /api/multimodal/concept/health
   */
  async checkHealth(): Promise<{ status: string; timestamp: string }> {
    try {
      const response = await apiClient.get<{ status: string; timestamp: string }>('/multimodal/concept/health');
      return response;
    } catch (error) {
      console.error('多模态概念服务健康检查失败:', error);
      throw error;
    }
  },
};