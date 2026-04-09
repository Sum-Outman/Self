import { ApiClient, apiClient } from './client';
import { ApiResponse } from '../../types/api';

// 训练任务接口
export interface TrainingJob {
  id: string;
  name: string;
  model_type: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  dataset_size: number;
  epochs: number;
  current_epoch: number;
  start_time: string;
  estimated_time: string;
  gpu_usage: number;
  memory_usage: number;
  config: Record<string, any>;
  logs: string[];
  created_at: string;
  updated_at: string;
}

// 训练配置接口
export interface TrainingConfig {
  model_type: 'transformer' | 'multimodal' | 'cognitive' | 'pinn' | 'cnn_enhanced' | 'graph_nn' | 'laplacian' | 'hybrid';
  dataset_path: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  use_gpu: boolean;
  save_checkpoints: boolean;
  early_stopping: boolean;
  validation_split: number;
  model_name?: string;
  description?: string;
}

// 训练请求接口
export interface TrainingRequest {
  name: string;
  description?: string;
  config: TrainingConfig;
}

// 训练统计接口
export interface TrainingStats {
  total_jobs: number;
  running_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  average_training_time: number;
  total_training_hours: number;
  gpu_utilization: number;
}



class TrainingApi {
  private apiClient: ApiClient;

  constructor() {
    this.apiClient = apiClient;
  }

  // 获取训练任务列表
  async getJobs(status?: string): Promise<ApiResponse<TrainingJob[]>> {
    try {
      const params = new URLSearchParams();
      if (status) params.append('status', status);
      
      const response = (await this.apiClient.get(`/training/jobs?${params.toString()}`)) as ApiResponse<TrainingJob[]>;
      return response;
    } catch (error) {
      console.error('获取训练任务列表失败:', error);
      throw error;
    }
  }

  // 获取单个训练任务
  async getJob(jobId: string): Promise<ApiResponse<TrainingJob>> {
    try {
      const response = (await this.apiClient.get(`/training/jobs/${jobId}`)) as ApiResponse<TrainingJob>;
      return response;
    } catch (error) {
      console.error('获取训练任务失败:', error);
      throw error;
    }
  }

  // 创建训练任务
  async createJob(request: TrainingRequest): Promise<ApiResponse<TrainingJob>> {
    try {
      const response = (await this.apiClient.post('/training/jobs', request)) as ApiResponse<TrainingJob>;
      return response;
    } catch (error) {
      console.error('创建训练任务失败:', error);
      throw error;
    }
  }

  // 更新训练任务
  async updateJob(jobId: string, updates: Partial<TrainingJob>): Promise<ApiResponse<TrainingJob>> {
    try {
      const response = (await this.apiClient.put(`/training/jobs/${jobId}`, updates)) as ApiResponse<TrainingJob>;
      return response;
    } catch (error) {
      console.error('更新训练任务失败:', error);
      throw error;
    }
  }

  // 删除训练任务
  async deleteJob(jobId: string): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.delete(`/training/jobs/${jobId}`)) as ApiResponse;
      return response;
    } catch (error) {
      console.error('删除训练任务失败:', error);
      throw error;
    }
  }

  // 暂停训练任务
  async pauseJob(jobId: string): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.post(`/training/jobs/${jobId}/pause`)) as ApiResponse;
      return response;
    } catch (error) {
      console.error('暂停训练任务失败:', error);
      throw error;
    }
  }

  // 恢复训练任务
  async resumeJob(jobId: string): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.post(`/training/jobs/${jobId}/resume`)) as ApiResponse;
      return response;
    } catch (error) {
      console.error('恢复训练任务失败:', error);
      throw error;
    }
  }

  // 停止训练任务
  async stopJob(jobId: string): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.post(`/training/jobs/${jobId}/stop`)) as ApiResponse;
      return response;
    } catch (error) {
      console.error('停止训练任务失败:', error);
      throw error;
    }
  }

  // 获取训练日志
  async getJobLogs(jobId: string, lines?: number): Promise<ApiResponse<string[]>> {
    try {
      const params = new URLSearchParams();
      if (lines) params.append('lines', lines.toString());
      
      const response = (await this.apiClient.get(`/training/jobs/${jobId}/logs?${params.toString()}`)) as ApiResponse<string[]>;
      return response;
    } catch (error) {
      console.error('获取训练日志失败:', error);
      throw error;
    }
  }

  // 获取训练统计
  async getStats(): Promise<ApiResponse<TrainingStats>> {
    try {
      const response = (await this.apiClient.get('/training/stats')) as ApiResponse<TrainingStats>;
      return response;
    } catch (error) {
      console.error('获取训练统计失败:', error);
      throw error;
    }
  }

  // 上传训练数据集
  async uploadDataset(file: File, datasetName: string): Promise<ApiResponse<{path: string, size: number}>> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', datasetName);
      
      const response = (await this.apiClient.post('/training/datasets/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })) as ApiResponse<{path: string, size: number}>;
      return response;
    } catch (error) {
      console.error('上传数据集失败:', error);
      throw error;
    }
  }

  // 获取可用数据集列表
  async getDatasets(): Promise<ApiResponse<{name: string, path: string, size: number, created_at: string}[]>> {
    try {
      const response = (await this.apiClient.get('/training/datasets')) as ApiResponse<{name: string, path: string, size: number, created_at: string}[]>;
      return response;
    } catch (error) {
      console.error('获取数据集列表失败:', error);
      throw error;
    }
  }

  // 获取可用模型类型
  async getModelTypes(): Promise<ApiResponse<{type: string, description: string, parameters: number}[]>> {
    try {
      const response = (await this.apiClient.get('/training/model-types')) as ApiResponse<{type: string, description: string, parameters: number}[]>;
      return response;
    } catch (error) {
      console.error('获取模型类型失败:', error);
      throw error;
    }
  }

  // 获取GPU状态
  async getGpuStatus(): Promise<ApiResponse<{gpu_count: number, memory_total: number, memory_used: number, utilization: number}>> {
    try {
      const response = (await this.apiClient.get('/training/gpu-status')) as ApiResponse<{gpu_count: number, memory_total: number, memory_used: number, utilization: number}>;
      return response;
    } catch (error) {
      console.error('获取GPU状态失败:', error);
      throw error;
    }
  }

  // 下载训练结果
  async downloadResults(jobId: string): Promise<ApiResponse<{download_url: string}>> {
    try {
      const response = (await this.apiClient.get(`/training/jobs/${jobId}/download`)) as ApiResponse<{download_url: string}>;
      return response;
    } catch (error) {
      console.error('下载训练结果失败:', error);
      throw error;
    }
  }

  // 导出训练模型
  async exportModel(jobId: string, format: 'onnx' | 'torchscript' | 'savedmodel'): Promise<ApiResponse<{export_url: string}>> {
    try {
      const response = (await this.apiClient.post(`/training/jobs/${jobId}/export`, { format })) as ApiResponse<{export_url: string}>;
      return response;
    } catch (error) {
      console.error('导出模型失败:', error);
      throw error;
    }
  }

  // 拉普拉斯增强系统 - 获取状态
  async getLaplacianStatus(): Promise<ApiResponse<Record<string, any>>> {
    try {
      const response = (await this.apiClient.get('/training/laplacian/status')) as ApiResponse<Record<string, any>>;
      return response;
    } catch (error) {
      console.error('获取拉普拉斯增强系统状态失败:', error);
      throw error;
    }
  }

  // 拉普拉斯增强系统 - 配置
  async configureLaplacianSystem(configuration: Record<string, any>): Promise<ApiResponse<Record<string, any>>> {
    try {
      const response = (await this.apiClient.post('/training/laplacian/configure', configuration)) as ApiResponse<Record<string, any>>;
      return response;
    } catch (error) {
      console.error('配置拉普拉斯增强系统失败:', error);
      throw error;
    }
  }

  // 拉普拉斯增强系统 - 启用
  async enableLaplacianSystem(enable: boolean = true): Promise<ApiResponse<Record<string, any>>> {
    try {
      const response = (await this.apiClient.post('/training/laplacian/enable', { enable })) as ApiResponse<Record<string, any>>;
      return response;
    } catch (error) {
      console.error('启用/禁用拉普拉斯增强系统失败:', error);
      throw error;
    }
  }

  // 拉普拉斯增强系统 - 应用增强
  async applyLaplacianEnhancement(jobId: string, enhancementParams: Record<string, any> = {}): Promise<ApiResponse<Record<string, any>>> {
    try {
      const payload = { job_id: jobId, ...enhancementParams };
      const response = (await this.apiClient.post('/training/laplacian/apply', payload)) as ApiResponse<Record<string, any>>;
      return response;
    } catch (error) {
      console.error('应用拉普拉斯增强失败:', error);
      throw error;
    }
  }
}

// 创建单例实例
export const trainingApi = new TrainingApi();

// 默认导出
export default trainingApi;