/**
 * 电脑操作API服务
 * 提供屏幕分析、键盘鼠标操作、命令行控制等电脑操作能力
 */

import { apiClient } from './client';
import {
  ScreenAnalysisRequest,
  ScreenAnalysisResponse,
  KeyboardOperationRequest,
  MouseOperationRequest,
  CommandExecutionRequest,
  CommandExecutionResponse,
  ServiceInfo
} from '../../types/multimodal';

/**
 * 电脑操作API
 * 支持机器人实体操作电脑或直接通过API控制电脑
 */
export const computerOperationApi = {
  /**
   * 分析屏幕内容
   * POST /api/computer-operation/analyze-screen
   */
  async analyzeScreen(request?: ScreenAnalysisRequest): Promise<ScreenAnalysisResponse> {
    const formData = new FormData();
    
    if (request?.screenshot_file) {
      formData.append('screenshot_file', request.screenshot_file);
    }
    
    try {
      const response = await apiClient.post<ScreenAnalysisResponse>(
        '/computer-operation/analyze-screen',
        formData,
        {
          headers: request?.screenshot_file ? {
            'Content-Type': 'multipart/form-data',
          } : undefined,
        }
      );
      return response;
    } catch (error) {
      console.error('屏幕分析API调用失败:', error);
      throw error;
    }
  },

  /**
   * 执行键盘操作
   * POST /api/computer-operation/keyboard-operation
   */
  async keyboardOperation(request: KeyboardOperationRequest): Promise<{
    success: boolean;
    operation_id: string;
    keys_pressed: string[];
    text_entered?: string;
    execution_time: number;
    timestamp: string;
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        operation_id: string;
        keys_pressed: string[];
        text_entered?: string;
        execution_time: number;
        timestamp: string;
      }>('/computer-operation/keyboard-operation', {
        keys_json: JSON.stringify(request.keys),
        text: request.text,
        delay_between_keys: request.delay_between_keys,
        press_duration: request.press_duration,
      });
      return response;
    } catch (error) {
      console.error('键盘操作API调用失败:', error);
      throw error;
    }
  },

  /**
   * 执行鼠标操作
   * POST /api/computer-operation/mouse-operation
   */
  async mouseOperation(request: MouseOperationRequest): Promise<{
    success: boolean;
    operation_id: string;
    operation_type: string;
    position?: [number, number];
    execution_time: number;
    timestamp: string;
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        operation_id: string;
        operation_type: string;
        position?: [number, number];
        execution_time: number;
        timestamp: string;
      }>('/computer-operation/mouse-operation', {
        operation_type: request.operation_type,
        position: request.position ? JSON.stringify(request.position) : undefined,
        button: request.button,
        scroll_amount: request.scroll_amount,
        drag_start: request.drag_start ? JSON.stringify(request.drag_start) : undefined,
        drag_end: request.drag_end ? JSON.stringify(request.drag_end) : undefined,
      });
      return response;
    } catch (error) {
      console.error('鼠标操作API调用失败:', error);
      throw error;
    }
  },

  /**
   * 执行命令行命令
   * POST /api/computer-operation/execute-command
   */
  async executeCommand(request: CommandExecutionRequest): Promise<CommandExecutionResponse> {
    try {
      const response = await apiClient.post<CommandExecutionResponse>(
        '/computer-operation/execute-command',
        {
          command: request.command,
          working_directory: request.working_directory,
          timeout: request.timeout,
        }
      );
      return response;
    } catch (error) {
      console.error('命令执行API调用失败:', error);
      throw error;
    }
  },

  /**
   * 自然语言到命令翻译
   * POST /api/computer-operation/translate-natural-language
   */
  async translateNaturalLanguage(natural_language: string): Promise<{
    success: boolean;
    natural_language: string;
    translated_commands: string[];
    confidence: number;
    explanations: string[];
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        natural_language: string;
        translated_commands: string[];
        confidence: number;
        explanations: string[];
      }>('/computer-operation/translate-natural-language', {
        natural_language,
      });
      return response;
    } catch (error) {
      console.error('自然语言翻译API调用失败:', error);
      throw error;
    }
  },

  /**
   * 执行自动化任务
   * POST /api/computer-operation/execute-task
   */
  async executeTask(task_name: string, params: Record<string, any> = {}): Promise<{
    success: boolean;
    task_id: string;
    task_name: string;
    steps_executed: number;
    execution_time: number;
    results: Record<string, any>;
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        task_id: string;
        task_name: string;
        steps_executed: number;
        execution_time: number;
        results: Record<string, any>;
      }>('/computer-operation/execute-task', {
        task_name,
        params: JSON.stringify(params),
      });
      return response;
    } catch (error) {
      console.error('自动化任务执行API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取可用任务列表
   * GET /api/computer-operation/available-tasks
   */
  async getAvailableTasks(): Promise<{
    success: boolean;
    tasks: Array<{
      task_name: string;
      description: string;
      category: string;
      complexity: 'simple' | 'medium' | 'complex';
      estimated_duration: number;
      parameters: Array<{
        name: string;
        type: string;
        required: boolean;
        description: string;
      }>;
    }>;
  }> {
    try {
      const response = await apiClient.get<{
        success: boolean;
        tasks: Array<{
          task_name: string;
          description: string;
          category: string;
          complexity: 'simple' | 'medium' | 'complex';
          estimated_duration: number;
          parameters: Array<{
            name: string;
            type: string;
            required: boolean;
            description: string;
          }>;
        }>;
      }>('/computer-operation/available-tasks');
      return response;
    } catch (error) {
      console.error('获取可用任务列表API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取电脑操作服务信息
   * GET /api/computer-operation/service-info
   */
  async getServiceInfo(): Promise<ServiceInfo> {
    try {
      const response = await apiClient.get<ServiceInfo>('/computer-operation/service-info');
      return response;
    } catch (error) {
      console.error('获取电脑操作服务信息失败:', error);
      throw error;
    }
  },

  /**
   * 设置操作模式
   * POST /api/computer-operation/set-operation-mode
   */
  async setOperationMode(mode: 'vision_based' | 'api_based' | 'command_line'): Promise<{
    success: boolean;
    mode: string;
    message: string;
  }> {
    try {
      const response = await apiClient.post<{
        success: boolean;
        mode: string;
        message: string;
      }>('/computer-operation/set-operation-mode', { mode });
      return response;
    } catch (error) {
      console.error('设置操作模式API调用失败:', error);
      throw error;
    }
  },

  /**
   * 获取操作历史
   * GET /api/computer-operation/operation-history
   */
  async getOperationHistory(limit: number = 50): Promise<{
    success: boolean;
    history: Array<{
      operation_id: string;
      operation_type: string;
      timestamp: string;
      success: boolean;
      details: Record<string, any>;
    }>;
  }> {
    try {
      const response = await apiClient.get<{
        success: boolean;
        history: Array<{
          operation_id: string;
          operation_type: string;
          timestamp: string;
          success: boolean;
          details: Record<string, any>;
        }>;
      }>('/computer-operation/operation-history', {
        params: { limit },
      });
      return response;
    } catch (error) {
      console.error('获取操作历史API调用失败:', error);
      throw error;
    }
  },
};