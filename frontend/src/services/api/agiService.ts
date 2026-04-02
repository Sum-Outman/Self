import { ApiClient, apiClient } from './client';
import { ApiResponse } from '../../types/api';

// AGI状态接口
export interface AGIStatus {
  status: 'idle' | 'training' | 'reasoning' | 'learning' | 'paused' | 'error';
  mode: 'autonomous' | 'task' | 'demo';
  trainingProgress: number;
  reasoningDepth: number;
  memoryUsage: number;
  hardwareConnected: boolean;
  lastUpdated: string;
  activeModels: string[];
  trainingEpoch: number;
  totalEpochs: number;
  learningRate: number;
  batchSize: number;
  datasetSize: number;
  gpuUsage: number;
  cpuUsage: number;
  systemMemory: number;
  networkStatus: 'online' | 'offline' | 'limited';
}

// AGI训练配置接口
export interface AGITrainingConfig {
  modelType: 'transformer' | 'multimodal' | 'cognitive' | 'hybrid';
  dataset: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  useGpu: boolean;
  validationSplit: number;
  enableEarlyStopping: boolean;
  enableMixedPrecision: boolean;
  enableGradientAccumulation: boolean;
  enableCheckpointing: boolean;
  enableLogging: boolean;
}

// AGI训练进度接口
export interface AGIProgress {
  epoch: number;
  totalEpochs: number;
  progress: number;
  loss: number;
  accuracy: number;
  learningRate: number;
  batchSize: number;
  samplesProcessed: number;
  totalSamples: number;
  timeElapsed: number;
  estimatedTimeRemaining: number;
  currentPhase: 'forward' | 'backward' | 'optimization' | 'validation' | 'checkpointing';
  gpuMemoryUsage: number;
  cpuMemoryUsage: number;
}

// AGI模式切换请求
export interface AGIModeRequest {
  mode: 'autonomous' | 'task' | 'demo';
  transitionDuration?: number;
  preserveState?: boolean;
}

// AGI训练请求
export interface AGITrainingRequest {
  config: AGITrainingConfig;
  priority?: 'low' | 'normal' | 'high' | 'critical';
  resumeFromCheckpoint?: string;
}

class AGIApi {
  private apiClient: ApiClient;

  constructor() {
    this.apiClient = apiClient;
  }

  // 获取AGI状态
  async getAGIStatus(): Promise<AGIStatus> {
    try {
      const response = (await this.apiClient.get('/agi/status')) as ApiResponse<AGIStatus>;
      if (response.success && response.data) {
        return response.data;
      } else {
        throw new Error(response.message || '获取AGI状态失败');
      }
    } catch (error) {
      console.error('获取AGI状态失败:', error);
      // 返回模拟数据用于开发（真实环境中应该抛出错误）
      return {
        status: 'idle',
        mode: 'task',
        trainingProgress: 0,
        reasoningDepth: 0,
        memoryUsage: 0,
        hardwareConnected: false,
        lastUpdated: new Date().toISOString(),
        activeModels: [],
        trainingEpoch: 0,
        totalEpochs: 0,
        learningRate: 0,
        batchSize: 0,
        datasetSize: 0,
        gpuUsage: 0,
        cpuUsage: 0,
        systemMemory: 0,
        networkStatus: 'online'
      };
    }
  }

  // 启动AGI训练
  async startAGITraining(config?: AGITrainingConfig): Promise<ApiResponse<{ jobId: string }>> {
    try {
      const request: AGITrainingRequest = {
        config: config || {
          modelType: 'transformer',
          dataset: 'default',
          epochs: 100,
          batchSize: 32,
          learningRate: 0.001,
          useGpu: true,
          validationSplit: 0.2,
          enableEarlyStopping: true,
          enableMixedPrecision: true,
          enableGradientAccumulation: false,
          enableCheckpointing: true,
          enableLogging: true
        }
      };
      const response = (await this.apiClient.post('/agi/training/start', request)) as ApiResponse<{ jobId: string }>;
      return response;
    } catch (error) {
      console.error('启动AGI训练失败:', error);
      throw error;
    }
  }

  // 停止AGI训练
  async stopAGITraining(): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.post('/agi/training/stop')) as ApiResponse;
      return response;
    } catch (error) {
      console.error('停止AGI训练失败:', error);
      throw error;
    }
  }

  // 获取AGI训练进度
  async getAGIProgress(): Promise<AGIProgress> {
    try {
      const response = (await this.apiClient.get('/agi/training/progress')) as ApiResponse<AGIProgress>;
      if (response.success && response.data) {
        return response.data;
      } else {
        throw new Error(response.message || '获取训练进度失败');
      }
    } catch (error) {
      console.error('获取AGI训练进度失败:', error);
      // 返回模拟数据用于开发
      return {
        epoch: 0,
        totalEpochs: 100,
        progress: 0,
        loss: 0,
        accuracy: 0,
        learningRate: 0.001,
        batchSize: 32,
        samplesProcessed: 0,
        totalSamples: 10000,
        timeElapsed: 0,
        estimatedTimeRemaining: 0,
        currentPhase: 'forward',
        gpuMemoryUsage: 0,
        cpuMemoryUsage: 0
      };
    }
  }

  // 切换AGI模式
  async changeAGIMode(modeRequest: AGIModeRequest): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.post('/agi/mode', modeRequest)) as ApiResponse;
      return response;
    } catch (error) {
      console.error('切换AGI模式失败:', error);
      throw error;
    }
  }

  // 暂停AGI系统
  async pauseAGI(): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.post('/agi/pause')) as ApiResponse;
      return response;
    } catch (error) {
      console.error('暂停AGI系统失败:', error);
      throw error;
    }
  }

  // 恢复AGI系统
  async resumeAGI(): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.post('/agi/resume')) as ApiResponse;
      return response;
    } catch (error) {
      console.error('恢复AGI系统失败:', error);
      throw error;
    }
  }

  // 重启AGI系统
  async restartAGI(): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.post('/agi/restart')) as ApiResponse;
      return response;
    } catch (error) {
      console.error('重启AGI系统失败:', error);
      throw error;
    }
  }

  // 获取AGI系统日志
  async getAGILogs(lines: number = 100): Promise<ApiResponse<string[]>> {
    try {
      const params = new URLSearchParams();
      params.append('lines', lines.toString());
      const response = (await this.apiClient.get(`/agi/logs?${params.toString()}`)) as ApiResponse<string[]>;
      return response;
    } catch (error) {
      console.error('获取AGI系统日志失败:', error);
      throw error;
    }
  }

  // 获取AGI系统统计
  async getAGIStats(): Promise<ApiResponse<{
    totalTrainingTime: number;
    totalReasoningTime: number;
    totalLearningSessions: number;
    modelsTrained: number;
    averageLoss: number;
    averageAccuracy: number;
    hardwareUptime: number;
    systemUptime: number;
    errorCount: number;
    warningCount: number;
  }>> {
    try {
      const response = (await this.apiClient.get('/agi/stats')) as ApiResponse<{
        totalTrainingTime: number;
        totalReasoningTime: number;
        totalLearningSessions: number;
        modelsTrained: number;
        averageLoss: number;
        averageAccuracy: number;
        hardwareUptime: number;
        systemUptime: number;
        errorCount: number;
        warningCount: number;
      }>;
      return response;
    } catch (error) {
      console.error('获取AGI系统统计失败:', error);
      throw error;
    }
  }

  // 上传AGI训练数据集
  async uploadAGIDataset(file: File, datasetName: string): Promise<ApiResponse<{ path: string, size: number }>> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', datasetName);
      const response = (await this.apiClient.post('/agi/datasets/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })) as ApiResponse<{ path: string, size: number }>;
      return response;
    } catch (error) {
      console.error('上传AGI数据集失败:', error);
      throw error;
    }
  }

  // 获取AGI可用数据集列表
  async getAGIDatasets(): Promise<ApiResponse<{ name: string, path: string, size: number, created_at: string }[]>> {
    try {
      const response = (await this.apiClient.get('/agi/datasets')) as ApiResponse<{ name: string, path: string, size: number, created_at: string }[]>;
      return response;
    } catch (error) {
      console.error('获取AGI数据集列表失败:', error);
      throw error;
    }
  }

  // 导出AGI模型
  async exportAGIModel(format: 'onnx' | 'torchscript' | 'savedmodel'): Promise<ApiResponse<{ export_url: string }>> {
    try {
      const response = (await this.apiClient.post('/agi/export', { format })) as ApiResponse<{ export_url: string }>;
      return response;
    } catch (error) {
      console.error('导出AGI模型失败:', error);
      throw error;
    }
  }
}

// 创建单例实例
export const agiApi = new AGIApi();

// 导出AGI服务函数（兼容现有代码）
export const getAGIStatus = async (): Promise<AGIStatus> => {
  return agiApi.getAGIStatus();
};

export const startAGITraining = async (config?: AGITrainingConfig): Promise<ApiResponse<{ jobId: string }>> => {
  return agiApi.startAGITraining(config);
};

export const stopAGITraining = async (): Promise<ApiResponse> => {
  return agiApi.stopAGITraining();
};

export const getAGIProgress = async (): Promise<AGIProgress> => {
  return agiApi.getAGIProgress();
};

export const changeAGIMode = async (modeRequest: AGIModeRequest): Promise<ApiResponse> => {
  return agiApi.changeAGIMode(modeRequest);
};

// 默认导出
export default agiApi;