/**
 * 设备操作学习API服务
 * 提供说明书学习、实体教学等设备操作学习能力
 */

import { apiClient } from './client';
import {
  ManualLearningRequest,
  ManualLearningResponse,
  OperationProceduresResponse,
  ServiceInfo
} from '../../types/multimodal';

/**
 * 设备操作学习API
 * 支持通过说明书和实体教学学习设备操作
 */
export const equipmentLearningApi = {
  /**
   * 从说明书学习设备操作
   * POST /api/equipment-learning/learn-from-manual
   */
  async learnFromManual(request: ManualLearningRequest): Promise<ManualLearningResponse> {
    const formData = new FormData();
    
    // 添加文本和文件
    formData.append('manual_text', request.manual_text);
    formData.append('equipment_name', request.equipment_name);
    formData.append('equipment_type', request.equipment_type);
    formData.append('learning_method', request.learning_method);
    
    if (request.manual_file) {
      formData.append('manual_file', request.manual_file);
    }
    
    try {
      const response = await apiClient.post<ManualLearningResponse>(
        '/equipment-learning/learn-from-manual',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response;
    } catch (error) {
      console.error('说明书学习API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取设备知识
   * GET /api/equipment-learning/equipment-knowledge
   */
  async getEquipmentKnowledge(equipment_id?: string, equipment_name?: string): Promise<{
    success: boolean;
    equipment: Array<{
      equipment_id: string;
      equipment_name: string;
      equipment_type: string;
      learning_method: string;
      learned_at: string;
      confidence_score: number;
      components: Array<{
        component_id: string;
        component_name: string;
        description: string;
        location: string;
        function: string;
      }>;
    }>;
  }> {
    const params: Record<string, string> = {};
    if (equipment_id) {
      params.equipment_id = equipment_id;
    }
    if (equipment_name) {
      params.equipment_name = equipment_name;
    }
    
    try {
      const response = await apiClient.get<{
        success: boolean;
        equipment: Array<{
          equipment_id: string;
          equipment_name: string;
          equipment_type: string;
          learning_method: string;
          learned_at: string;
          confidence_score: number;
          components: Array<{
            component_id: string;
            component_name: string;
            description: string;
            location: string;
            function: string;
          }>;
        }>;
      }>('/equipment-learning/equipment-knowledge', { params });
      return response;
    } catch (error) {
      console.error('获取设备知识API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取操作流程
   * GET /api/equipment-learning/operation-procedures
   */
  async getOperationProcedures(equipment_id: string): Promise<OperationProceduresResponse> {
    try {
      const response = await apiClient.get<OperationProceduresResponse>(
        '/equipment-learning/operation-procedures',
        { params: { equipment_id } }
      );
      return response;
    } catch (error) {
      console.error('获取操作流程API调用失败:', error);
      throw error;
    }
  },

  /**
   * 执行操作步骤
   * POST /api/equipment-learning/execute-operation-step
   * @param simulation_mode 是否使用模拟模式（默认false，使用实际执行；根据项目要求"禁止使用虚拟数据"，应始终设置为false）
   */
  async executeOperationStep(
    equipment_id: string,
    step_number: number,
    simulation_mode: boolean = false
  ): Promise<{
    success: boolean;
    step_number: number;
    equipment_id: string;
    action: string;
    result: Record<string, any>;
    safety_check_passed: boolean;
    execution_time: number;
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        step_number: number;
        equipment_id: string;
        action: string;
        result: Record<string, any>;
        safety_check_passed: boolean;
        execution_time: number;
      }>('/equipment-learning/execute-operation-step', {
        equipment_id,
        step_number,
        simulation_mode,
      });
      return response;
    } catch (error) {
      console.error('执行操作步骤API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取学习会话记录
   * GET /api/equipment-learning/learning-sessions
   */
  async getLearningSessions(equipment_id?: string, limit: number = 20): Promise<{
    success: boolean;
    sessions: Array<{
      session_id: string;
      equipment_id: string;
      equipment_name: string;
      learning_method: string;
      start_time: string;
      end_time?: string;
      steps_completed: number;
      success_rate: number;
      confidence_gain: number;
    }>;
  }> {
    const params: Record<string, string | number> = { limit };
    if (equipment_id) {
      params.equipment_id = equipment_id;
    }
    
    try {
      const response = await apiClient.get<{
        success: boolean;
        sessions: Array<{
          session_id: string;
          equipment_id: string;
          equipment_name: string;
          learning_method: string;
          start_time: string;
          end_time?: string;
          steps_completed: number;
          success_rate: number;
          confidence_gain: number;
        }>;
      }>('/equipment-learning/learning-sessions', { params });
      return response;
    } catch (error) {
      console.error('获取学习会话记录API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取设备类型列表
   * GET /api/equipment-learning/equipment-types
   */
  async getEquipmentTypes(): Promise<{
    success: boolean;
    equipment_types: Array<{
      value: string;
      label: string;
      description: string;
    }>;
  }> {
    try {
      const response = await apiClient.get<{
        success: boolean;
        equipment_types: Array<{
          value: string;
          label: string;
          description: string;
        }>;
      }>('/equipment-learning/equipment-types');
      return response;
    } catch (error) {
      console.error('获取设备类型列表API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取操作类型列表
   * GET /api/equipment-learning/operation-types
   */
  async getOperationTypes(): Promise<{
    success: boolean;
    operation_types: Array<{
      value: string;
      label: string;
      description: string;
    }>;
  }> {
    try {
      const response = await apiClient.get<{
        success: boolean;
        operation_types: Array<{
          value: string;
          label: string;
          description: string;
        }>;
      }>('/equipment-learning/operation-types');
      return response;
    } catch (error) {
      console.error('获取操作类型列表API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取学习方法列表
   * GET /api/equipment-learning/learning-methods
   */
  async getLearningMethods(): Promise<{
    success: boolean;
    learning_methods: Array<{
      value: string;
      label: string;
      description: string;
    }>;
  }> {
    try {
      const response = await apiClient.get<{
        success: boolean;
        learning_methods: Array<{
          value: string;
          label: string;
          description: string;
        }>;
      }>('/equipment-learning/learning-methods');
      return response;
    } catch (error) {
      console.error('获取学习方法列表API调用失败:', error);
      throw error;
    }
  },

  /**
   * 分析说明书文本
   * POST /api/equipment-learning/analyze-manual-text
   */
  async analyzeManualText(manual_text: string): Promise<{
    success: boolean;
    sections: Array<{
      section_type: 'title' | 'description' | 'component' | 'operation_step' | 'safety_warning';
      content: string;
      confidence: number;
      extracted_data?: Record<string, any>;
    }>;
    components: Array<{
      component_name: string;
      description: string;
      location: string;
      function: string;
    }>;
    operation_steps: Array<{
      step_number: number;
      description: string;
      action_type: string;
      target_component?: string;
    }>;
    safety_warnings: string[];
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        sections: Array<{
          section_type: 'title' | 'description' | 'component' | 'operation_step' | 'safety_warning';
          content: string;
          confidence: number;
          extracted_data?: Record<string, any>;
        }>;
        components: Array<{
          component_name: string;
          description: string;
          location: string;
          function: string;
        }>;
        operation_steps: Array<{
          step_number: number;
          description: string;
          action_type: string;
          target_component?: string;
        }>;
        safety_warnings: string[];
      }>('/equipment-learning/analyze-manual-text', { manual_text });
      return response;
    } catch (error) {
      console.error('说明书文本分析API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取设备学习服务信息
   * GET /api/equipment-learning/service-info
   */
  async getServiceInfo(): Promise<ServiceInfo> {
    try {
      const response = await apiClient.get<ServiceInfo>('/equipment-learning/service-info');
      return response;
    } catch (error) {
      console.error('获取设备学习服务信息失败:', error);
      throw error;
    }
  },
};