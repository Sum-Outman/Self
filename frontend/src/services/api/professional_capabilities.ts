/**
 * 专业领域能力API服务
 * 提供编程、数学、物理、医学、金融等专业领域能力管理和测试
 */

import { apiClient } from './client';

// 专业领域能力接口
export interface ProfessionalCapability {
  id: string;
  name: string;
  description: string;
  icon: string;
  enabled: boolean;
  status: 'active' | 'inactive' | 'testing' | 'error';
  performance: number;
  last_tested: string;
  test_results: {
    passed: number;
    failed: number;
    total: number;
  };
  capabilities: string[];
}

// 能力测试结果接口
export interface CapabilityTestResult {
  capability_id: string;
  test_name: string;
  status: 'passed' | 'failed' | 'running';
  duration: number;
  result?: any;
  error?: string;
  timestamp: string;
}

// 整体状态接口
export interface OverallStatus {
  total_capabilities: number;
  enabled_capabilities: number;
  average_performance: number;
  last_updated: string;
}

export const professionalCapabilitiesApi = {
  /**
   * 获取专业领域能力列表
   * GET /api/professional/capabilities
   */
  async getCapabilities(): Promise<ProfessionalCapability[]> {
    try {
      const response = await apiClient.get<{ success: boolean; data: ProfessionalCapability[]; message: string }>(
        '/professional/capabilities'
      );
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.message || '获取能力列表失败');
      }
    } catch (error) {
      console.error('获取专业领域能力列表失败:', error);
      throw error;
    }
  },

  /**
   * 获取特定专业领域能力详情
   * GET /api/professional/capabilities/{capability_id}
   */
  async getCapability(capabilityId: string): Promise<ProfessionalCapability> {
    try {
      const response = await apiClient.get<{ success: boolean; data: ProfessionalCapability; message: string }>(
        `/professional/capabilities/${capabilityId}`
      );
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.message || '获取能力详情失败');
      }
    } catch (error) {
      console.error('获取专业领域能力详情失败:', error);
      throw error;
    }
  },

  /**
   * 测试特定专业领域能力
   * POST /api/professional/capabilities/{capability_id}/test
   */
  async testCapability(capabilityId: string): Promise<CapabilityTestResult> {
    try {
      const response = await apiClient.post<{ success: boolean; data: CapabilityTestResult; message: string }>(
        `/professional/capabilities/${capabilityId}/test`
      );
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.message || '能力测试失败');
      }
    } catch (error) {
      console.error('测试专业领域能力失败:', error);
      throw error;
    }
  },

  /**
   * 获取专业领域能力整体状态
   * GET /api/professional/capabilities/overall/status
   */
  async getOverallStatus(): Promise<OverallStatus> {
    try {
      const response = await apiClient.get<{ success: boolean; data: OverallStatus; message: string }>(
        '/professional/capabilities/overall/status'
      );
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.message || '获取整体状态失败');
      }
    } catch (error) {
      console.error('获取专业领域能力整体状态失败:', error);
      throw error;
    }
  },

  /**
   * 保存专业领域能力配置
   * POST /api/professional/capabilities/{capability_id}/configure
   */
  async saveConfig(capabilityId: string, config: any): Promise<{ success: boolean; message: string }> {
    try {
      const response = await apiClient.post<{ success: boolean; message: string }>(
        `/professional/capabilities/${capabilityId}/configure`,
        { config }
      );
      return response;
    } catch (error) {
      console.error('保存专业领域能力配置失败:', error);
      // 如果API端点不存在，返回一个结构化的错误响应
      return {
        success: false,
        message: error instanceof Error ? error.message : '保存配置失败，API端点可能已实现'
      };
    }
  },
};