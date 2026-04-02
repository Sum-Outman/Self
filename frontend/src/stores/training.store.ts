/**
 * 训练状态管理store
 * 管理AGI模型训练相关的全局状态
 */
import { create } from 'zustand';
import { TrainingStatus, TrainingMode, TrainingProgress } from '../types/training';

// 训练状态类型定义
export interface TrainingState {
  // 训练状态
  trainingStatus: TrainingStatus;
  // 当前训练模式
  currentMode: TrainingMode;
  // 训练进度
  progress: TrainingProgress;
  // 是否正在训练
  isTraining: boolean;
  // 当前训练任务ID
  currentTaskId: string | null;
  // 训练错误信息
  trainingError: string | null;
  // 训练开始时间
  startTime: number | null;
  // 训练结束时间
  endTime: number | null;
  // 训练时长（秒）
  duration: number;
  // 训练指标历史
  metricsHistory: Array<{
    timestamp: number;
    loss: number;
    accuracy: number;
    learningRate: number;
    step: number;
  }>;
  // 训练配置
  trainingConfig: {
    batchSize: number;
    learningRate: number;
    epochs: number;
    checkpointFrequency: number;
    validationFrequency: number;
  };
  // 当前检查点
  currentCheckpoint: string | null;
  // 检查点列表
  checkpoints: Array<{
    id: string;
    timestamp: number;
    step: number;
    loss: number;
    accuracy: number;
    path: string;
  }>;
  
  // 操作方法
  startTraining: (mode: TrainingMode, config?: Partial<TrainingState['trainingConfig']>) => void;
  stopTraining: () => void;
  pauseTraining: () => void;
  resumeTraining: () => void;
  updateProgress: (progress: Partial<TrainingProgress>) => void;
  updateMetrics: (metrics: { loss: number; accuracy: number; learningRate: number; step: number }) => void;
  setTrainingError: (error: string | null) => void;
  addCheckpoint: (checkpoint: Omit<TrainingState['checkpoints'][0], 'id'>) => void;
  loadCheckpoint: (checkpointId: string) => void;
  resetTraining: () => void;
  updateConfig: (config: Partial<TrainingState['trainingConfig']>) => void;
}

// 初始状态
const initialState: Omit<TrainingState, 
  | 'startTraining' 
  | 'stopTraining' 
  | 'pauseTraining' 
  | 'resumeTraining'
  | 'updateProgress'
  | 'updateMetrics'
  | 'setTrainingError'
  | 'addCheckpoint'
  | 'loadCheckpoint'
  | 'resetTraining'
  | 'updateConfig'
> = {
  trainingStatus: 'idle',
  currentMode: 'supervised',
  progress: {
    currentStep: 0,
    totalSteps: 0,
    currentEpoch: 0,
    totalEpochs: 0,
    percentage: 0,
    estimatedTimeRemaining: 0,
  },
  isTraining: false,
  currentTaskId: null,
  trainingError: null,
  startTime: null,
  endTime: null,
  duration: 0,
  metricsHistory: [],
  trainingConfig: {
    batchSize: 32,
    learningRate: 0.001,
    epochs: 10,
    checkpointFrequency: 1000,
    validationFrequency: 100,
  },
  currentCheckpoint: null,
  checkpoints: [],
};

// 创建训练store
export const useTrainingStore = create<TrainingState>((set, get) => ({
  ...initialState,
  
  // 开始训练
  startTraining: (mode: TrainingMode, config?: Partial<TrainingState['trainingConfig']>) => {
    const currentTime = Date.now();
    const taskId = `training_${currentTime}_${Math.random().toString(36).substr(2, 9)}`;
    
    // 更新配置
    const newConfig = {
      ...get().trainingConfig,
      ...config,
    };
    
    set({
      trainingStatus: 'running',
      currentMode: mode,
      isTraining: true,
      currentTaskId: taskId,
      startTime: currentTime,
      endTime: null,
      duration: 0,
      trainingError: null,
      trainingConfig: newConfig,
      progress: {
        currentStep: 0,
        totalSteps: newConfig.epochs * 1000, // 假设每epoch 1000步
        currentEpoch: 0,
        totalEpochs: newConfig.epochs,
        percentage: 0,
        estimatedTimeRemaining: newConfig.epochs * 60, // 假设每epoch 60秒
      },
    });
    
    // 记录训练开始日志
    console.log(`训练开始: 任务ID=${taskId}, 模式=${mode}, 配置=`, newConfig);
  },
  
  // 停止训练
  stopTraining: () => {
    const state = get();
    const currentTime = Date.now();
    const startTime = state.startTime || currentTime;
    const duration = Math.floor((currentTime - startTime) / 1000);
    
    set({
      trainingStatus: 'stopped',
      isTraining: false,
      endTime: currentTime,
      duration,
    });
    
    console.log(`训练停止: 任务ID=${state.currentTaskId}, 时长=${duration}秒`);
  },
  
  // 暂停训练
  pauseTraining: () => {
    set({
      trainingStatus: 'paused',
      isTraining: false,
    });
    
    console.log(`训练暂停: 任务ID=${get().currentTaskId}`);
  },
  
  // 恢复训练
  resumeTraining: () => {
    set({
      trainingStatus: 'running',
      isTraining: true,
    });
    
    console.log(`训练恢复: 任务ID=${get().currentTaskId}`);
  },
  
  // 更新进度
  updateProgress: (progress: Partial<TrainingProgress>) => {
    set((state) => ({
      progress: {
        ...state.progress,
        ...progress,
      },
    }));
  },
  
  // 更新指标
  updateMetrics: (metrics: { loss: number; accuracy: number; learningRate: number; step: number }) => {
    const timestamp = Date.now();
    
    set((state) => ({
      metricsHistory: [
        ...state.metricsHistory,
        {
          timestamp,
          ...metrics,
        },
      ].slice(-1000), // 保留最近1000条记录
    }));
    
    // 同时更新进度
    get().updateProgress({
      currentStep: metrics.step,
    });
  },
  
  // 设置训练错误
  setTrainingError: (error: string | null) => {
    set({
      trainingError: error,
      trainingStatus: error ? 'error' : get().trainingStatus,
    });
    
    if (error) {
      console.error(`训练错误: ${error}`);
    }
  },
  
  // 添加检查点
  addCheckpoint: (checkpoint: Omit<TrainingState['checkpoints'][0], 'id'>) => {
    const id = `checkpoint_${checkpoint.timestamp}_${Math.random().toString(36).substr(2, 6)}`;
    
    set((state) => ({
      checkpoints: [
        ...state.checkpoints,
        {
          id,
          ...checkpoint,
        },
      ],
      currentCheckpoint: id,
    }));
    
    console.log(`检查点添加: ID=${id}, 步数=${checkpoint.step}, 损失=${checkpoint.loss}`);
  },
  
  // 加载检查点
  loadCheckpoint: (checkpointId: string) => {
    const checkpoint = get().checkpoints.find(cp => cp.id === checkpointId);
    
    if (checkpoint) {
      set({
        currentCheckpoint: checkpointId,
      });
      
      console.log(`检查点加载: ID=${checkpointId}`);
      return true;
    }
    
    console.warn(`检查点不存在: ID=${checkpointId}`);
    return false;
  },
  
  // 重置训练状态
  resetTraining: () => {
    set(initialState);
    console.log('训练状态已重置');
  },
  
  // 更新配置
  updateConfig: (config: Partial<TrainingState['trainingConfig']>) => {
    set((state) => ({
      trainingConfig: {
        ...state.trainingConfig,
        ...config,
      },
    }));
    
    console.log('训练配置更新:', config);
  },
}));

// 训练store钩子函数（便捷方法）
export const useTrainingStatus = () => {
  const { trainingStatus, isTraining, currentMode } = useTrainingStore();
  return { trainingStatus, isTraining, currentMode };
};

export const useTrainingProgress = () => {
  const { progress, metricsHistory } = useTrainingStore();
  return { progress, metricsHistory };
};

export const useTrainingConfig = () => {
  const { trainingConfig, updateConfig } = useTrainingStore();
  return { trainingConfig, updateConfig };
};

export const useTrainingControls = () => {
  const { 
    startTraining, 
    stopTraining, 
    pauseTraining, 
    resumeTraining,
    resetTraining,
  } = useTrainingStore();
  
  return {
    startTraining,
    stopTraining,
    pauseTraining,
    resumeTraining,
    resetTraining,
  };
};

export const useTrainingMetrics = () => {
  const { metricsHistory, updateMetrics } = useTrainingStore();
  return { metricsHistory, updateMetrics };
};

export const useCheckpoints = () => {
  const { checkpoints, currentCheckpoint, addCheckpoint, loadCheckpoint } = useTrainingStore();
  return { checkpoints, currentCheckpoint, addCheckpoint, loadCheckpoint };
};