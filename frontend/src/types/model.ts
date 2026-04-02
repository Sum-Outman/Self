/**
 * 模型相关类型定义
 * 提供严格的类型安全性，减少any类型使用
 */

// 模型能力
export interface ModelCapabilities {
  text: boolean;
  image: boolean;
  audio: boolean;
  video: boolean;
  multimodal: boolean;
  reasoning: boolean;
  planning: boolean;
  learning: boolean;
  control: boolean;
  speech: boolean;
  vision: boolean;
  sensor: boolean;
  programming: boolean;
  mathematics: boolean;
  physics: boolean;
  chemistry: boolean;
  medicine: boolean;
  finance: boolean;
}

// 模型提供者
export type ModelProvider = 
  | 'Self AGI'
  | 'OpenAI'
  | 'Anthropic'
  | 'Google'
  | 'Meta'
  | 'Microsoft'
  | 'HuggingFace'
  | 'Custom'
  | string;

// 模型格式
export type ModelFormat = 
  | 'pytorch'
  | 'tensorflow'
  | 'onnx'
  | 'safetensors'
  | 'gguf'
  | 'ggml'
  | 'h5'
  | 'pb'
  | string;

// 模型精度
export type ModelPrecision = 
  | 'fp16'
  | 'fp32'
  | 'bf16'
  | 'int8'
  | 'int4'
  | 'mixed'
  | string;

// 模型后端
export interface BackendModel {
  id: string;
  name?: string;
  description?: string;
  provider?: ModelProvider;
  max_tokens?: number;
  supports_multimodal?: boolean;
  capabilities?: Partial<ModelCapabilities>;
  format?: ModelFormat;
  precision?: ModelPrecision;
  size_mb?: number;
  parameters?: number;
  created_at?: string;
  updated_at?: string;
  is_active?: boolean;
  is_available?: boolean;
  is_default?: boolean;
  version?: string;
  license?: string;
  url?: string;
  config?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

// 前端模型
export interface UIModel {
  id: string;
  name: string;
  description: string;
  provider: ModelProvider;
  max_tokens: number;
  supports_multimodal: boolean;
  capabilities: Partial<ModelCapabilities>;
  format?: ModelFormat;
  precision?: ModelPrecision;
  size_mb?: number;
  parameters?: number;
  created_at?: string;
  updated_at?: string;
  is_active?: boolean;
  is_default?: boolean;
  version?: string;
  license?: string;
  url?: string;
  config?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

// 模型列表响应
export interface ModelListResponse {
  models: BackendModel[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

// 模型配置
export interface ModelConfig {
  // 模型标识符
  model_id: string;
  // 温度参数
  temperature?: number;
  // 最大生成长度
  max_tokens?: number;
  // 核采样参数
  top_p?: number;
  // 重复惩罚
  repetition_penalty?: number;
  // 是否流式输出
  stream?: boolean;
  // 停止标记
  stop_sequences?: string[];
  // 是否启用多模态
  enable_multimodal?: boolean;
  // 是否启用推理
  enable_reasoning?: boolean;
  // 是否启用规划
  enable_planning?: boolean;
  // 是否启用学习
  enable_learning?: boolean;
  // 其他配置
  [key: string]: unknown;
}

// 模型性能指标
export interface ModelPerformance {
  // 推理速度（毫秒/标记）
  inference_speed_ms_per_token?: number;
  // 内存使用量（MB）
  memory_usage_mb?: number;
  // GPU使用率（%）
  gpu_utilization_percent?: number;
  // 准确率
  accuracy?: number;
  // 损失值
  loss?: number;
  // 吞吐量（标记/秒）
  throughput_tokens_per_second?: number;
  // 延迟（毫秒）
  latency_ms?: number;
  // 能源消耗（瓦）
  power_consumption_w?: number;
  // 温度（摄氏度）
  temperature_c?: number;
  // 时间戳
  timestamp?: string;
}

// 模型状态
export type ModelStatus = 
  | 'idle'
  | 'loading'
  | 'ready'
  | 'running'
  | 'error'
  | 'unavailable'
  | 'maintenance'
  | 'updating';

// 模型实例
export interface ModelInstance {
  id: string;
  model_id: string;
  status: ModelStatus;
  performance?: ModelPerformance;
  config: ModelConfig;
  started_at?: string;
  last_used_at?: string;
  usage_count?: number;
  error_message?: string;
  host?: string;
  port?: number;
  protocol?: 'http' | 'https' | 'grpc';
  health_check_url?: string;
}

// 模型训练配置
export interface ModelTrainingConfig {
  // 数据集配置
  dataset: {
    name: string;
    path: string;
    format: string;
    size: number;
    split_ratio?: number;
  };
  
  // 超参数
  hyperparameters: {
    batch_size: number;
    learning_rate: number;
    epochs: number;
    optimizer: string;
    scheduler?: string;
    weight_decay?: number;
    gradient_clip?: number;
  };
  
  // 训练策略
  training_strategy: {
    type: string;
    curriculum?: boolean;
    reinforcement?: boolean;
    self_supervised?: boolean;
    multimodal?: boolean;
    distributed?: boolean;
    federated?: boolean;
  };
  
  // 监控配置
  monitoring: {
    enable: boolean;
    checkpoint_frequency: number;
    validation_frequency: number;
    metrics: string[];
    visualization?: boolean;
  };
  
  // 硬件配置
  hardware: {
    device_type: 'cpu' | 'gpu' | 'tpu';
    device_count?: number;
    memory_limit_mb?: number;
    mixed_precision?: boolean;
  };
}

// 模型部署配置
export interface ModelDeploymentConfig {
  // 部署类型
  deployment_type: 'local' | 'cloud' | 'edge' | 'hybrid';
  
  // 扩展配置
  scaling: {
    min_instances: number;
    max_instances: number;
    auto_scaling: boolean;
    target_utilization?: number;
  };
  
  // 网络配置
  network: {
    port: number;
    protocol: 'http' | 'https' | 'grpc';
    cors_enabled: boolean;
    rate_limiting?: boolean;
    max_connections?: number;
  };
  
  // 安全配置
  security: {
    authentication: boolean;
    authorization: boolean;
    encryption: boolean;
    audit_logging: boolean;
  };
  
  // 监控配置
  monitoring: {
    enable: boolean;
    metrics_endpoint: boolean;
    health_endpoint: boolean;
    log_level: string;
  };
}

// 类型转换函数
export function convertBackendModelToUi(backendModel: BackendModel): UIModel {
  return {
    id: backendModel.id,
    name: backendModel.name || backendModel.id,
    description: backendModel.description || 'AGI模型',
    provider: backendModel.provider || 'Self AGI',
    max_tokens: backendModel.max_tokens || 4096,
    supports_multimodal: backendModel.supports_multimodal || false,
    capabilities: backendModel.capabilities || {},
    format: backendModel.format,
    precision: backendModel.precision,
    size_mb: backendModel.size_mb,
    parameters: backendModel.parameters,
    created_at: backendModel.created_at,
    updated_at: backendModel.updated_at,
    is_active: backendModel.is_available !== undefined ? backendModel.is_available : backendModel.is_active,
    is_default: backendModel.is_default,
    version: backendModel.version,
    license: backendModel.license,
    url: backendModel.url,
    config: backendModel.config,
    metadata: backendModel.metadata,
  };
}

// 类型守卫
export function isBackendModel(obj: unknown): obj is BackendModel {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'id' in obj &&
    typeof (obj as any).id === 'string'
  );
}

export function isUIModel(obj: unknown): obj is UIModel {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'id' in obj &&
    typeof (obj as any).id === 'string' &&
    'name' in obj &&
    typeof (obj as any).name === 'string' &&
    'description' in obj &&
    typeof (obj as any).description === 'string' &&
    'provider' in obj &&
    typeof (obj as any).provider === 'string' &&
    'max_tokens' in obj &&
    typeof (obj as any).max_tokens === 'number' &&
    'supports_multimodal' in obj &&
    typeof (obj as any).supports_multimodal === 'boolean'
  );
}

// 模型比较函数
export function compareModels(a: UIModel, b: UIModel): number {
  // 按提供者排序
  if (a.provider !== b.provider) {
    return a.provider.localeCompare(b.provider);
  }
  
  // 按名称排序
  if (a.name !== b.name) {
    return a.name.localeCompare(b.name);
  }
  
  // 按ID排序
  return a.id.localeCompare(b.id);
}

// 模型过滤函数
export function filterModels(models: UIModel[], filter: {
  provider?: ModelProvider;
  supports_multimodal?: boolean;
  capabilities?: Partial<ModelCapabilities>;
  searchText?: string;
}): UIModel[] {
  return models.filter(model => {
    // 按提供者过滤
    if (filter.provider && model.provider !== filter.provider) {
      return false;
    }
    
    // 按多模态支持过滤
    if (filter.supports_multimodal !== undefined && 
        model.supports_multimodal !== filter.supports_multimodal) {
      return false;
    }
    
    // 按能力过滤
    if (filter.capabilities) {
      for (const [capability, required] of Object.entries(filter.capabilities)) {
        const modelCapability = model.capabilities?.[capability as keyof ModelCapabilities];
        if (required === true && modelCapability !== true) {
          return false;
        }
        if (required === false && modelCapability === true) {
          return false;
        }
      }
    }
    
    // 按搜索文本过滤
    if (filter.searchText) {
      const searchLower = filter.searchText.toLowerCase();
      return (
        model.id.toLowerCase().includes(searchLower) ||
        model.name.toLowerCase().includes(searchLower) ||
        model.description.toLowerCase().includes(searchLower) ||
        model.provider.toLowerCase().includes(searchLower)
      );
    }
    
    return true;
  });
}

// 模型分组函数
export function groupModelsByProvider(models: UIModel[]): Record<ModelProvider, UIModel[]> {
  const groups: Record<string, UIModel[]> = {};
  
  models.forEach(model => {
    const provider = model.provider;
    if (!groups[provider]) {
      groups[provider] = [];
    }
    groups[provider].push(model);
  });
  
  return groups;
}

// 模型统计函数
export function getModelStatistics(models: UIModel[]): {
  total: number;
  byProvider: Record<string, number>;
  multimodalCount: number;
  averageTokens: number;
} {
  const statistics = {
    total: models.length,
    byProvider: {} as Record<string, number>,
    multimodalCount: 0,
    averageTokens: 0,
  };
  
  let totalTokens = 0;
  
  models.forEach(model => {
    // 按提供者统计
    if (!statistics.byProvider[model.provider]) {
      statistics.byProvider[model.provider] = 0;
    }
    statistics.byProvider[model.provider]++;
    
    // 统计多模态模型
    if (model.supports_multimodal) {
      statistics.multimodalCount++;
    }
    
    // 累计最大令牌数
    totalTokens += model.max_tokens;
  });
  
  // 计算平均令牌数
  if (models.length > 0) {
    statistics.averageTokens = totalTokens / models.length;
  }
  
  return statistics;
}