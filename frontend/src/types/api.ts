/**
 * API相关类型定义
 * 提供严格的类型安全性，减少any类型使用
 */

import { AxiosRequestConfig, AxiosResponse } from 'axios';

// 基础API响应接口
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  code?: number;
  timestamp?: string;
}

// 分页响应接口
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

// 错误响应接口
export interface ErrorResponse {
  success: false;
  error: string;
  message?: string;
  code: number;
  timestamp?: string;
  details?: Record<string, unknown>;
}

// 成功响应接口
export interface SuccessResponse<T> {
  success: true;
  data: T;
  message?: string;
  code: number;
  timestamp?: string;
}

// HTTP方法类型
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

// API请求配置
export interface ApiRequestConfig<D = unknown> extends Omit<AxiosRequestConfig<D>, 'method'> {
  method?: HttpMethod;
  requireAuth?: boolean;
  retryOnAuthError?: boolean;
}

// API客户端响应处理器
export interface ApiResponseHandler<T = unknown> {
  (response: AxiosResponse<ApiResponse<T>>): T;
}

// 上传文件接口
export interface FileUpload {
  file: File;
  name?: string;
  description?: string;
  tags?: string[];
}

// 上传进度回调
export interface UploadProgressCallback {
  (progress: number, loaded: number, total: number): void;
}

// 批量操作结果
export interface BatchOperationResult<T = unknown> {
  successes: Array<{ id: string; data: T }>;
  failures: Array<{ id: string; error: string }>;
  total: number;
  successCount: number;
  failureCount: number;
}

// WebSocket消息类型
export interface WebSocketMessage<T = unknown> {
  type: string;
  data: T;
  timestamp: number;
  sequence?: number;
}

// 实时数据更新
export interface RealtimeUpdate<T = unknown> {
  operation: 'CREATE' | 'UPDATE' | 'DELETE' | 'BATCH_UPDATE';
  entity: string;
  id: string | string[];
  data?: T;
  timestamp: number;
}

// API端点定义
export interface ApiEndpoint {
  path: string;
  method: HttpMethod;
  description?: string;
  requiresAuth?: boolean;
  requiredPermissions?: string[];
  requestSchema?: Record<string, unknown>;
  responseSchema?: Record<string, unknown>;
}

// API版本信息
export interface ApiVersion {
  version: string;
  deprecated?: boolean;
  endOfLife?: string;
  changelog?: string[];
  endpoints: Record<string, ApiEndpoint>;
}

// API服务状态
export interface ApiServiceStatus {
  service: string;
  status: 'healthy' | 'degraded' | 'unavailable' | 'maintenance';
  responseTime?: number;
  uptime?: number;
  lastCheck?: string;
  errors?: Array<{
    timestamp: string;
    error: string;
    count: number;
  }>;
}

// API速率限制信息
export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset: number;
  used: number;
  window: string;
}

// 缓存控制
export interface CacheControl {
  maxAge?: number;
  staleWhileRevalidate?: number;
  mustRevalidate?: boolean;
  noCache?: boolean;
  noStore?: boolean;
}

// API指标
export interface ApiMetrics {
  requestCount: number;
  errorCount: number;
  averageResponseTime: number;
  p95ResponseTime: number;
  p99ResponseTime: number;
  byEndpoint: Record<string, {
    count: number;
    errorCount: number;
    avgTime: number;
  }>;
  byMethod: Record<HttpMethod, {
    count: number;
    errorCount: number;
    avgTime: number;
  }>;
  byStatusCode: Record<number, number>;
}

// 验证错误
export interface ValidationError {
  field: string;
  message: string;
  code?: string;
  details?: Record<string, unknown>;
}

// 批量验证错误
export interface BatchValidationError {
  index: number;
  errors: ValidationError[];
}

// 条件查询
export interface QueryCondition {
  field: string;
  operator: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'nin' | 'like' | 'ilike' | 'regex' | 'exists';
  value: unknown;
}

// 排序选项
export interface SortOption {
  field: string;
  direction: 'asc' | 'desc';
}

// 分页查询参数
export interface PaginationQuery {
  page?: number;
  size?: number;
  sort?: SortOption[];
  conditions?: QueryCondition[];
  search?: string;
  fields?: string[];
}

// 查询结果
export interface QueryResult<T> {
  data: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
  hasMore: boolean;
}

// 实用类型：确保类型不为any
export type NotAny<T> = unknown extends T ? never : T;

// 实用类型：移除any
export type RemoveAny<T> = T extends any ? unknown : T;

// 注意：移除未使用的isAny函数以通过类型检查

// 类型安全的API响应包装器
export class TypedApiResponse<T> {
  constructor(private response: ApiResponse<T>) {}

  get data(): T | undefined {
    return this.response.data;
  }

  get isSuccess(): boolean {
    return this.response.success === true;
  }

  get error(): string | undefined {
    return this.response.error;
  }

  get message(): string | undefined {
    return this.response.message;
  }

  get code(): number | undefined {
    return this.response.code;
  }

  get timestamp(): string | undefined {
    return this.response.timestamp;
  }

  // 类型安全的数据访问
  getOrThrow(): T {
    if (!this.response.success || this.response.data === undefined) {
      throw new Error(this.response.error || 'API请求失败');
    }
    return this.response.data;
  }

  // 安全的数据访问，提供默认值
  getOrDefault<D>(defaultValue: D): T | D {
    return this.response.success && this.response.data !== undefined 
      ? this.response.data 
      : defaultValue;
  }

  // 链式操作
  map<R>(mapper: (data: T) => R): TypedApiResponse<R> {
    if (!this.response.success || this.response.data === undefined) {
      return new TypedApiResponse<R>({
        ...this.response,
        data: undefined as unknown as R,
      });
    }
    return new TypedApiResponse<R>({
      ...this.response,
      data: mapper(this.response.data),
    });
  }
}