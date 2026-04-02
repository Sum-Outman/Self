/**
 * 训练相关类型定义
 */

// 训练状态类型
export type TrainingStatus = 
  | 'idle'        // 空闲
  | 'running'     // 运行中
  | 'paused'      // 暂停
  | 'stopped'     // 停止
  | 'completed'   // 完成
  | 'error'       // 错误
  | 'initializing' // 初始化中
  | 'validating'  // 验证中
  | 'checkpointing' // 保存检查点中
  | 'restoring';   // 恢复中

// 训练模式类型
export type TrainingMode = 
  | 'supervised'      // 监督学习
  | 'self_supervised' // 自监督学习
  | 'reinforcement'   // 强化学习
  | 'multimodal'      // 多模态学习
  | 'curriculum'      // 课程学习
  | 'imitation'       // 模仿学习
  | 'meta_learning'   // 元学习
  | 'distributed'     // 分布式训练
  | 'federated'       // 联邦学习
  | 'transfer';       // 迁移学习

// 训练进度类型
export interface TrainingProgress {
  // 当前步骤
  currentStep: number;
  // 总步骤
  totalSteps: number;
  // 当前轮次
  currentEpoch: number;
  // 总轮次
  totalEpochs: number;
  // 进度百分比
  percentage: number;
  // 预计剩余时间（秒）
  estimatedTimeRemaining: number;
  // 当前批次
  currentBatch?: number;
  // 总批次
  totalBatches?: number;
  // 当前学习率
  currentLearningRate?: number;
  // 当前损失
  currentLoss?: number;
  // 当前准确率
  currentAccuracy?: number;
}

// 训练指标类型
export interface TrainingMetrics {
  // 损失值
  loss: number;
  // 准确率
  accuracy: number;
  // 精确率
  precision?: number;
  // 召回率
  recall?: number;
  // F1分数
  f1Score?: number;
  // 学习率
  learningRate: number;
  // 梯度范数
  gradientNorm?: number;
  // 训练速度（步/秒）
  stepsPerSecond?: number;
  // 内存使用量
  memoryUsage?: number;
  // GPU使用率
  gpuUtilization?: number;
  // 时间戳
  timestamp: number;
  // 步骤数
  step: number;
  // 轮次数
  epoch: number;
}

// 训练配置类型
export interface TrainingConfig {
  // 批次大小
  batchSize: number;
  // 学习率
  learningRate: number;
  // 轮次数
  epochs: number;
  // 检查点频率（步）
  checkpointFrequency: number;
  // 验证频率（步）
  validationFrequency: number;
  // 优化器类型
  optimizer?: 'adam' | 'sgd' | 'adamw' | 'rmsprop';
  // 损失函数类型
  lossFunction?: string;
  // 是否使用混合精度
  useMixedPrecision?: boolean;
  // 是否使用梯度累积
  useGradientAccumulation?: boolean;
  // 梯度累积步数
  gradientAccumulationSteps?: number;
  // 最大梯度范数
  maxGradNorm?: number;
  // 权重衰减
  weightDecay?: number;
  // 学习率调度器
  scheduler?: 'cosine' | 'linear' | 'step' | 'exponential' | 'plateau';
  // 热身步数
  warmupSteps?: number;
  // 设备类型
  device?: 'cpu' | 'gpu' | 'tpu' | 'auto';
  // GPU ID列表
  gpuIds?: number[];
}

// 训练任务类型
export interface TrainingTask {
  // 任务ID
  id: string;
  // 任务名称
  name: string;
  // 任务描述
  description?: string;
  // 训练状态
  status: TrainingStatus;
  // 训练模式
  mode: TrainingMode;
  // 训练配置
  config: TrainingConfig;
  // 进度信息
  progress: TrainingProgress;
  // 创建时间
  createdAt: number;
  // 开始时间
  startedAt?: number;
  // 结束时间
  endedAt?: number;
  // 持续时间（秒）
  duration?: number;
  // 错误信息
  error?: string;
  // 检查点列表
  checkpoints?: Array<{
    id: string;
    path: string;
    step: number;
    epoch: number;
    loss: number;
    accuracy: number;
    timestamp: number;
  }>;
  // 指标历史
  metricsHistory?: TrainingMetrics[];
}

// 检查点类型
export interface Checkpoint {
  // 检查点ID
  id: string;
  // 检查点路径
  path: string;
  // 步骤数
  step: number;
  // 轮次数
  epoch: number;
  // 损失值
  loss: number;
  // 准确率
  accuracy: number;
  // 创建时间戳
  timestamp: number;
  // 文件大小（字节）
  fileSize?: number;
  // 模型架构
  modelArchitecture?: string;
  // 优化器状态
  hasOptimizerState?: boolean;
  // 训练配置
  trainingConfig?: TrainingConfig;
  // 元数据
  metadata?: Record<string, any>;
}

// 训练结果类型
export interface TrainingResult {
  // 任务ID
  taskId: string;
  // 最终损失
  finalLoss: number;
  // 最终准确率
  finalAccuracy: number;
  // 训练时长（秒）
  trainingTime: number;
  // 最佳检查点
  bestCheckpoint?: Checkpoint;
  // 最终检查点
  finalCheckpoint?: Checkpoint;
  // 评估指标
  evaluationMetrics?: Record<string, number>;
  // 训练报告
  trainingReport?: string;
  // 建议
  recommendations?: string[];
}

// 训练事件类型
export type TrainingEvent = 
  | { type: 'training_started'; taskId: string; config: TrainingConfig }
  | { type: 'training_progress'; taskId: string; progress: TrainingProgress }
  | { type: 'training_metrics'; taskId: string; metrics: TrainingMetrics }
  | { type: 'checkpoint_saved'; taskId: string; checkpoint: Checkpoint }
  | { type: 'validation_started'; taskId: string }
  | { type: 'validation_completed'; taskId: string; metrics: Record<string, number> }
  | { type: 'training_paused'; taskId: string }
  | { type: 'training_resumed'; taskId: string }
  | { type: 'training_stopped'; taskId: string; reason: string }
  | { type: 'training_completed'; taskId: string; result: TrainingResult }
  | { type: 'training_error'; taskId: string; error: string }
  | { type: 'training_warning'; taskId: string; warning: string }
  | { type: 'training_info'; taskId: string; message: string };

// 训练请求类型
export interface TrainingRequest {
  // 训练模式
  mode: TrainingMode;
  // 训练配置
  config: Partial<TrainingConfig>;
  // 模型ID或路径
  modelId?: string;
  // 数据集ID或路径
  datasetId?: string;
  // 验证集ID或路径
  validationId?: string;
  // 任务名称
  taskName?: string;
  // 任务描述
  taskDescription?: string;
  // 回调URL
  callbackUrl?: string;
  // 元数据
  metadata?: Record<string, any>;
}

// 训练响应类型
export interface TrainingResponse {
  // 是否成功
  success: boolean;
  // 任务ID
  taskId?: string;
  // 错误信息
  error?: string;
  // 消息
  message?: string;
  // 预计开始时间
  estimatedStartTime?: number;
  // 预计持续时间
  estimatedDuration?: number;
}

// 分布式训练配置
export interface DistributedTrainingConfig {
  // 分布式策略
  strategy: 'data_parallel' | 'model_parallel' | 'pipeline_parallel' | 'hybrid';
  // 节点数量
  numNodes: number;
  // 每个节点的GPU数量
  gpusPerNode: number;
  // 主节点地址
  masterAddr: string;
  // 主节点端口
  masterPort: number;
  // 后端
  backend: 'nccl' | 'gloo' | 'mpi';
  // 通信优化
  communicationOptimization?: boolean;
  // 梯度同步频率
  gradientSyncFrequency?: number;
  // 模型分片策略
  modelShardingStrategy?: string;
}