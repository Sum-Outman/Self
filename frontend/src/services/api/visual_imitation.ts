/**
 * 视觉动作模仿API服务
 * 提供姿态估计、动作理解、模仿学习等视觉模仿能力
 */

import { apiClient } from './client';
import {
  ImitationSessionRequest,
  ImitationSessionResponse,
  PoseRecordingRequest,
  PoseRecordingResponse,
  AvailableActionsResponse,
  ServiceInfo
} from '../../types/multimodal';

/**
 * 视觉动作模仿API
 * 支持通过视觉观察人类动作并进行模仿学习
 */
export const visualImitationApi = {
  /**
   * 开始模仿会话
   * POST /api/visual-imitation/start-session
   */
  async startSession(request: ImitationSessionRequest): Promise<ImitationSessionResponse> {
    try {
      const response = await apiClient.post<ImitationSessionResponse>(
        '/visual-imitation/start-session',
        {
          target_action: request.target_action,
          imitation_mode: request.imitation_mode,
          session_notes: request.session_notes,
        }
      );
      return response;
    } catch (error) {
      console.error('开始模仿会话API调用失败:', error);
      throw error;
    }
  },

  /**
   * 录制姿态数据
   * POST /api/visual-imitation/record-pose
   */
  async recordPose(request: PoseRecordingRequest): Promise<PoseRecordingResponse> {
    try {
      const response = await apiClient.post<PoseRecordingResponse>(
        '/visual-imitation/record-pose',
        {
          session_id: request.session_id,
          pose_data: request.pose_data,
        }
      );
      return response;
    } catch (error) {
      console.error('姿态录制API调用失败:', error);
      throw error;
    }
  },

  /**
   * 分析姿态序列
   * POST /api/visual-imitation/analyze-pose-sequence
   */
  async analyzePoseSequence(
    session_id: string,
    start_time: number,
    end_time: number
  ): Promise<{
    success: boolean;
    session_id: string;
    frames_analyzed: number;
    action_type: string;
    action_confidence: number;
    key_frames: Array<{
      timestamp: number;
      joint_positions: Record<string, [number, number, number]>;
      description: string;
    }>;
    motion_characteristics: {
      speed: number;
      smoothness: number;
      symmetry: number;
      complexity: number;
    };
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        session_id: string;
        frames_analyzed: number;
        action_type: string;
        action_confidence: number;
        key_frames: Array<{
          timestamp: number;
          joint_positions: Record<string, [number, number, number]>;
          description: string;
        }>;
        motion_characteristics: {
          speed: number;
          smoothness: number;
          symmetry: number;
          complexity: number;
        };
      }>('/visual-imitation/analyze-pose-sequence', {
        session_id,
        start_time,
        end_time,
      });
      return response;
    } catch (error) {
      console.error('姿态序列分析API调用失败:', error);
      throw error;
    }
  },

  /**
   * 生成机器人轨迹
   * POST /api/visual-imitation/generate-robot-trajectory
   */
  async generateRobotTrajectory(
    session_id: string,
    adaptation_level: number = 0.5
  ): Promise<{
    success: boolean;
    session_id: string;
    trajectory_id: string;
    joint_trajectories: Record<string, Array<[number, number, number]>>;
    duration: number;
    adaptation_level: number;
    safety_check_passed: boolean;
    motion_quality: number;
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        session_id: string;
        trajectory_id: string;
        joint_trajectories: Record<string, Array<[number, number, number]>>;
        duration: number;
        adaptation_level: number;
        safety_check_passed: boolean;
        motion_quality: number;
      }>('/visual-imitation/generate-robot-trajectory', {
        session_id,
        adaptation_level,
      });
      return response;
    } catch (error) {
      console.error('生成机器人轨迹API调用失败:', error);
      throw error;
    }
  },

  /**
   * 评估模仿效果
   * POST /api/visual-imitation/evaluate-imitation
   */
  async evaluateImitation(
    session_id: string,
    reference_session_id: string
  ): Promise<{
    success: boolean;
    session_id: string;
    reference_session_id: string;
    similarity_score: number;
    motion_accuracy: number;
    timing_accuracy: number;
    overall_quality: number;
    feedback: string;
    improvement_suggestions: string[];
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        session_id: string;
        reference_session_id: string;
        similarity_score: number;
        motion_accuracy: number;
        timing_accuracy: number;
        overall_quality: number;
        feedback: string;
        improvement_suggestions: string[];
      }>('/visual-imitation/evaluate-imitation', {
        session_id,
        reference_session_id,
      });
      return response;
    } catch (error) {
      console.error('评估模仿效果API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取可用动作列表
   * GET /api/visual-imitation/available-actions
   */
  async getAvailableActions(category?: string): Promise<AvailableActionsResponse> {
    const params: Record<string, string> = {};
    if (category) {
      params.category = category;
    }
    
    try {
      const response = await apiClient.get<AvailableActionsResponse>(
        '/visual-imitation/available-actions',
        { params }
      );
      return response;
    } catch (error) {
      console.error('获取可用动作列表API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取模仿会话列表
   * GET /api/visual-imitation/imitation-sessions
   */
  async getImitationSessions(limit: number = 20): Promise<{
    success: boolean;
    sessions: Array<{
      session_id: string;
      target_action?: string;
      imitation_mode: string;
      start_time: string;
      end_time?: string;
      frames_recorded: number;
      action_type?: string;
      quality_score?: number;
    }>;
  }> {
    try {
      const response = await apiClient.get<{
        success: boolean;
        sessions: Array<{
          session_id: string;
          target_action?: string;
          imitation_mode: string;
          start_time: string;
          end_time?: string;
          frames_recorded: number;
          action_type?: string;
          quality_score?: number;
        }>;
      }>('/visual-imitation/imitation-sessions', { params: { limit } });
      return response;
    } catch (error) {
      console.error('获取模仿会话列表API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取会话详情
   * GET /api/visual-imitation/session-details
   */
  async getSessionDetails(session_id: string): Promise<{
    success: boolean;
    session: {
      session_id: string;
      target_action?: string;
      imitation_mode: string;
      start_time: string;
      end_time?: string;
      session_notes?: string;
      frames_recorded: number;
      pose_data_count: number;
      analysis_completed: boolean;
      trajectory_generated: boolean;
      evaluation_completed: boolean;
    };
  }> {
    try {
      const response = await apiClient.get<{
        success: boolean;
        session: {
          session_id: string;
          target_action?: string;
          imitation_mode: string;
          start_time: string;
          end_time?: string;
          session_notes?: string;
          frames_recorded: number;
          pose_data_count: number;
          analysis_completed: boolean;
          trajectory_generated: boolean;
          evaluation_completed: boolean;
        };
      }>('/visual-imitation/session-details', { params: { session_id } });
      return response;
    } catch (error) {
      console.error('获取会话详情API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取身体部位列表
   * GET /api/visual-imitation/body-parts
   */
  async getBodyParts(): Promise<{
    success: boolean;
    body_parts: Array<{
      value: string;
      label: string;
      description: string;
    }>;
  }> {
    try {
      const response = await apiClient.get<{
        success: boolean;
        body_parts: Array<{
          value: string;
          label: string;
          description: string;
        }>;
      }>('/visual-imitation/body-parts');
      return response;
    } catch (error) {
      console.error('获取身体部位列表API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取模仿模式列表
   * GET /api/visual-imitation/imitation-modes
   */
  async getImitationModes(): Promise<{
    success: boolean;
    imitation_modes: Array<{
      value: string;
      label: string;
      description: string;
    }>;
  }> {
    try {
      const response = await apiClient.get<{
        success: boolean;
        imitation_modes: Array<{
          value: string;
          label: string;
          description: string;
        }>;
      }>('/visual-imitation/imitation-modes');
      return response;
    } catch (error) {
      console.error('获取模仿模式列表API调用失败:', error);
      throw error;
    }
  },

  /**
   * 批处理姿态数据
   * POST /api/visual-imitation/batch-process-poses
   */
  async batchProcessPoses(
    session_id: string,
    poses: Array<{
      timestamp: number;
      joint_positions: Record<string, [number, number, number]>;
    }>
  ): Promise<{
    success: boolean;
    session_id: string;
    poses_processed: number;
    results: Record<string, {
      success: boolean;
      pose_id?: string;
      error?: string;
    }>;
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        session_id: string;
        poses_processed: number;
        results: Record<string, {
          success: boolean;
          pose_id?: string;
          error?: string;
        }>;
      }>('/visual-imitation/batch-process-poses', {
        session_id,
        poses,
      });
      return response;
    } catch (error) {
      console.error('批处理姿态数据API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取视觉模仿服务信息
   * GET /api/visual-imitation/service-info
   */
  async getServiceInfo(): Promise<ServiceInfo> {
    try {
      const response = await apiClient.get<ServiceInfo>('/visual-imitation/service-info');
      return response;
    } catch (error) {
      console.error('获取视觉模仿服务信息失败:', error);
      throw error;
    }
  },
};