/**
 * 人形机器人教学API服务
 * 提供儿童式多模态教学能力
 */

import { apiClient } from './client';
import {
  RobotTeachingRequest,
  RobotTeachingResponse,
  ConceptTestRequest,
  ConceptTestResponse,
  ServiceInfo
} from '../../types/multimodal';

/**
 * 人形机器人教学API
 * 支持多模态概念教学、测试、纠正等完整教学流程
 */
export const robotTeachingApi = {
  /**
   * 概念教学
   * POST /api/robot-teaching/teach-concept
   */
  async teachConcept(request: RobotTeachingRequest): Promise<RobotTeachingResponse> {
    const formData = new FormData();
    
    // 添加基本字段
    formData.append('concept_name', request.concept_name);
    if (request.teaching_method) {
      formData.append('teaching_method', request.teaching_method);
    }
    if (request.session_notes) {
      formData.append('session_notes', request.session_notes);
    }
    
    // 添加多模态数据
    if (request.modalities.text) {
      formData.append('text_input', request.modalities.text);
    }
    if (request.modalities.audio_file) {
      formData.append('audio_file', request.modalities.audio_file);
    }
    if (request.modalities.image_file) {
      formData.append('image_file', request.modalities.image_file);
    }
    if (request.modalities.sensor_data) {
      formData.append('sensor_data', JSON.stringify(request.modalities.sensor_data));
    }
    if (request.modalities.spatial_data) {
      formData.append('spatial_data', JSON.stringify(request.modalities.spatial_data));
    }
    if (request.modalities.quantity !== undefined) {
      formData.append('quantity', request.modalities.quantity.toString());
    }
    
    try {
      const response = await apiClient.post<RobotTeachingResponse>(
        '/robot-teaching/teach-concept',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response;
    } catch (error) {
      console.error('机器人概念教学API调用失败:', error);
      throw error;
    }
  },

  /**
   * 概念理解测试
   * POST /api/robot-teaching/test-concept
   */
  async testConcept(request: ConceptTestRequest): Promise<ConceptTestResponse> {
    const formData = new FormData();
    
    formData.append('concept_name', request.concept_name);
    formData.append('test_type', request.test_type);
    
    if (request.test_input.text) {
      formData.append('text_input', request.test_input.text);
    }
    if (request.test_input.image_file) {
      formData.append('image_file', request.test_input.image_file);
    }
    if (request.test_input.audio_file) {
      formData.append('audio_file', request.test_input.audio_file);
    }
    if (request.test_input.question) {
      formData.append('question', request.test_input.question);
    }
    
    try {
      const response = await apiClient.post<ConceptTestResponse>(
        '/robot-teaching/test-concept',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response;
    } catch (error) {
      console.error('机器人概念测试API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取学习进度
   * GET /api/robot-teaching/learning-progress?concept_name=可选
   */
  async getLearningProgress(concept_name?: string): Promise<{
    success: boolean;
    overall_progress: number;
    concept_progress: Record<string, number>;
    recent_sessions: Array<{
      concept_name: string;
      session_id: string;
      timestamp: string;
      teaching_method: string;
      progress_delta: number;
    }>;
  }> {
    const params: Record<string, string> = {};
    if (concept_name) {
      params.concept_name = concept_name;
    }
    
    try {
      const response = await apiClient.get<{
        success: boolean;
        overall_progress: number;
        concept_progress: Record<string, number>;
        recent_sessions: Array<{
          concept_name: string;
          session_id: string;
          timestamp: string;
          teaching_method: string;
          progress_delta: number;
        }>;
      }>('/robot-teaching/learning-progress', { params });
      return response;
    } catch (error) {
      console.error('获取学习进度API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取已学习概念列表
   * GET /api/robot-teaching/learned-concepts
   */
  async getLearnedConcepts(): Promise<{
    success: boolean;
    total_concepts: number;
    concepts: Array<{
      concept_name: string;
      mastery_level: string;
      last_practiced: string;
      modalities: string[];
      teaching_sessions: number;
    }>;
  }> {
    try {
      const response = await apiClient.get<{
        success: boolean;
        total_concepts: number;
        concepts: Array<{
          concept_name: string;
          mastery_level: string;
          last_practiced: string;
          modalities: string[];
          teaching_sessions: number;
        }>;
      }>('/robot-teaching/learned-concepts');
      return response;
    } catch (error) {
      console.error('获取已学习概念列表API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取机器人教学服务信息
   * GET /api/robot-teaching/service-info
   */
  async getServiceInfo(): Promise<ServiceInfo> {
    try {
      const response = await apiClient.get<ServiceInfo>('/robot-teaching/service-info');
      return response;
    } catch (error) {
      console.error('获取机器人教学服务信息失败:', error);
      throw error;
    }
  },

  /**
   * 开始交互式教学会话
   * POST /api/robot-teaching/start-interactive-session
   */
  async startInteractiveSession(concept_name: string): Promise<{
    success: boolean;
    session_id: string;
    concept_name: string;
    websocket_url: string;
    session_start_time: string;
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        session_id: string;
        concept_name: string;
        websocket_url: string;
        session_start_time: string;
      }>('/robot-teaching/start-interactive-session', { concept_name });
      return response;
    } catch (error) {
      console.error('开始交互式教学会话失败:', error);
      throw error;
    }
  },
};