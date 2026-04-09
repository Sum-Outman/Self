/**
 * 全局错误处理服务
 * 统一处理前端错误，提供用户友好的错误反馈
 */

import toast from 'react-hot-toast';
import { errorMonitor } from './errorMonitoring';

export interface ApiError extends Error {
  code?: string | number;
  status?: number;
  statusText?: string;
  response?: {
    status?: number;
    statusText?: string;
    data?: unknown;
    headers?: Record<string, string>;
  };
  config?: unknown;
  request?: unknown;
}

export interface ErrorHandlerConfig {
  /** 是否显示toast通知 */
  showToast: boolean;
  /** 是否记录错误到监控系统 */
  logToMonitor: boolean;
  /** 是否在控制台显示错误 */
  consoleLog: boolean;
  /** 默认错误消息 */
  defaultErrorMessage: string;
  /** 错误类型到消息的映射 */
  errorMessages: Record<string, string>;
}

const defaultConfig: ErrorHandlerConfig = {
  showToast: true,
  logToMonitor: true,
  consoleLog: process.env.NODE_ENV === 'development',
  defaultErrorMessage: '操作失败，请重试',
  errorMessages: {
    // HTTP状态码错误
    '400': '请求格式错误，请检查输入',
    '401': '认证失败，请重新登录',
    '403': '权限不足，无法访问此资源',
    '404': '请求的资源不存在',
    '405': '请求方法不被允许',
    '408': '请求超时，请稍后重试',
    '413': '请求数据过大，请减少数据量',
    '415': '不支持的媒体类型',
    '429': '请求过于频繁，请稍后重试',
    '500': '服务器内部错误，请联系技术支持',
    '502': '网关错误，服务器暂时不可用',
    '503': '服务器暂时不可用，请稍后重试',
    '504': '网关超时，服务器响应时间过长',
    
    // 网络错误
    'NETWORK_ERROR': '网络连接失败，请检查网络连接',
    'CONNECTION_REFUSED': '连接被拒绝，请检查服务器状态',
    'TIMEOUT': '请求超时，请检查网络连接并重试',
    'CORS': '跨域请求被阻止，请检查服务器配置',
    
    // 浏览器错误
    'ABORTED': '请求已被取消',
    'SECURITY': '安全错误，请检查SSL证书',
    'QUOTA_EXCEEDED': '存储空间不足，请清理缓存',
    'NOT_SUPPORTED': '浏览器不支持此功能',
    
    // 业务错误
    'VALIDATION_ERROR': '输入验证失败，请检查输入',
    'RESOURCE_NOT_FOUND': '请求的资源不存在',
    'UNAUTHORIZED': '未授权访问，请登录',
    'FORBIDDEN': '权限不足，无法执行此操作',
    'RATE_LIMITED': '请求过于频繁，请稍后重试',
    'SERVICE_UNAVAILABLE': '服务暂时不可用，请稍后重试',
  },
};

class ErrorHandler {
  private config: ErrorHandlerConfig;

  constructor(config: Partial<ErrorHandlerConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  /**
   * 处理错误
   */
  handleError(error: unknown, context?: string): void {
    const errorObj = this.normalizeError(error);
    const errorType = this.determineErrorType(errorObj);
    const errorMessage = this.getUserFriendlyMessage(errorObj, errorType, context);
    
    // 记录到监控系统
    if (this.config.logToMonitor) {
      errorMonitor.captureError(errorObj, context, {
        errorType,
        context,
        statusCode: errorObj.status,
        url: window.location.href,
      });
    }
    
    // 控制台输出
    if (this.config.consoleLog) {
      console.error(`[错误处理] ${context ? `[${context}] ` : ''}`, errorObj);
    }
    
    // 显示toast通知
    if (this.config.showToast && !this.shouldSuppressToast(errorType, errorObj)) {
      this.showErrorToast(errorMessage, errorType);
    }
    
    // 特殊错误处理（如认证失效）
    this.handleSpecialErrors(errorType, errorObj);
  }

  /**
   * 标准化错误对象
   */
  private normalizeError(error: unknown): ApiError {
    if (error instanceof Error) {
      return error as ApiError;
    }
    
    if (typeof error === 'string') {
      return new Error(error) as ApiError;
    }
    
    if (error && typeof error === 'object') {
      // 尝试从Axios错误中提取信息
      const axiosError = error as any;
      const normalizedError = new Error(axiosError.message || axiosError.toString()) as ApiError;
      
      normalizedError.code = axiosError.code;
      normalizedError.status = axiosError.status;
      normalizedError.statusText = axiosError.statusText;
      normalizedError.response = axiosError.response;
      normalizedError.config = axiosError.config;
      normalizedError.request = axiosError.request;
      
      return normalizedError;
    }
    
    return new Error('未知错误') as ApiError;
  }

  /**
   * 确定错误类型
   */
  private determineErrorType(error: ApiError): string {
    // 1. HTTP状态码
    if (error.response?.status) {
      return error.response.status.toString();
    }
    
    // 2. 网络错误
    if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
      return 'TIMEOUT';
    }
    
    if (error.code === 'ERR_NETWORK' || error.message?.includes('network')) {
      return 'NETWORK_ERROR';
    }
    
    if (error.message?.includes('CORS') || error.message?.includes('cross-origin')) {
      return 'CORS';
    }
    
    // 3. 浏览器错误
    if (error.name === 'AbortError' || error.message?.includes('aborted')) {
      return 'ABORTED';
    }
    
    if (error.name === 'SecurityError' || error.message?.includes('SSL') || error.message?.includes('certificate')) {
      return 'SECURITY';
    }
    
    if (error.name === 'QuotaExceededError') {
      return 'QUOTA_EXCEEDED';
    }
    
    if (error.name === 'NotSupportedError') {
      return 'NOT_SUPPORTED';
    }
    
    // 4. 从错误消息中推断业务错误
    const lowerMessage = error.message.toLowerCase();
    if (lowerMessage.includes('validation') || lowerMessage.includes('invalid')) {
      return 'VALIDATION_ERROR';
    }
    
    if (lowerMessage.includes('not found') || lowerMessage.includes('不存在')) {
      return 'RESOURCE_NOT_FOUND';
    }
    
    if (lowerMessage.includes('unauthorized') || lowerMessage.includes('未授权')) {
      return 'UNAUTHORIZED';
    }
    
    if (lowerMessage.includes('forbidden') || lowerMessage.includes('禁止')) {
      return 'FORBIDDEN';
    }
    
    if (lowerMessage.includes('rate limit') || lowerMessage.includes('频率限制')) {
      return 'RATE_LIMITED';
    }
    
    if (lowerMessage.includes('service unavailable') || lowerMessage.includes('服务不可用')) {
      return 'SERVICE_UNAVAILABLE';
    }
    
    // 5. 默认错误类型
    return 'UNKNOWN_ERROR';
  }

  /**
   * 获取用户友好的错误消息
   */
  private getUserFriendlyMessage(error: ApiError, errorType: string, context?: string): string {
    // 首先检查是否有自定义错误消息
    const errorData = error.response?.data as Record<string, unknown> | undefined;
    
    if (errorData?.detail && typeof errorData.detail === 'string') {
      return errorData.detail;
    }
    
    if (errorData?.message && typeof errorData.message === 'string') {
      return errorData.message;
    }
    
    if (errorData?.error && typeof errorData.error === 'string') {
      return errorData.error;
    }
    
    // 使用错误类型映射的消息
    if (this.config.errorMessages[errorType]) {
      let message = this.config.errorMessages[errorType];
      
      // 添加上下文信息
      if (context) {
        message = `${context}：${message}`;
      }
      
      return message;
    }
    
    // 从错误对象中提取消息
    if (error.message && error.message !== '') {
      return error.message;
    }
    
    // 默认消息
    return this.config.defaultErrorMessage;
  }

  /**
   * 是否应该抑制toast通知
   */
  private shouldSuppressToast(errorType: string, errorObj?: ApiError): boolean {
    // 某些错误类型不需要显示toast
    const suppressTypes = ['ABORTED', 'NOT_SUPPORTED'];
    if (suppressTypes.includes(errorType)) {
      return true;
    }
    
    // 检查错误消息是否包含需要抑制的关键词
    const errorMessage = errorObj?.message?.toLowerCase() || '';
    const suppressKeywords = [
      '真实硬件不可用',
      '硬件不可用',
      '数据为空',
      'empty data',
      'no data',
      '硬件未连接',
      '真实传感器不可用',
      '真实机器人硬件接口不可用',
      'pyautogui不可用',
      'tesseract ocr不可用'
    ];
    
    for (const keyword of suppressKeywords) {
      if (errorMessage.includes(keyword.toLowerCase())) {
        return true;
      }
    }
    
    return false;
  }

  /**
   * 显示错误toast
   */
  private showErrorToast(message: string, errorType: string): void {
    // 根据错误类型设置不同的样式
    let icon = '';
    let duration = 4000;
    
    switch (errorType) {
      case '401':
      case 'UNAUTHORIZED':
        icon = '🔒';
        duration = 5000;
        break;
      case '403':
      case 'FORBIDDEN':
        icon = '🚫';
        duration = 5000;
        break;
      case '404':
      case 'RESOURCE_NOT_FOUND':
        icon = '🔍';
        break;
      case '429':
      case 'RATE_LIMITED':
        icon = '⏰';
        duration = 6000;
        break;
      case '503':
      case 'SERVICE_UNAVAILABLE':
        icon = '🚧';
        duration = 6000;
        break;
      case 'NETWORK_ERROR':
      case 'TIMEOUT':
        icon = '📡';
        duration = 5000;
        break;
    }
    
    toast.error(message, {
      duration,
      icon,
      style: {
        background: '#1f2937', // gray-800
        color: '#f9fafb', // gray-50
      },
    });
  }

  /**
   * 处理特殊错误（如认证失效）
   */
  private handleSpecialErrors(errorType: string, error: ApiError): void {
    // 认证失效（401错误）
    if (errorType === '401' || errorType === 'UNAUTHORIZED') {
      // 触发认证失效事件
      const authExpiredEvent = new CustomEvent('auth:expired');
      window.dispatchEvent(authExpiredEvent);
      
      // 延迟重定向，让用户看到错误消息
      setTimeout(() => {
        if (!window.location.pathname.includes('/login')) {
          window.location.href = '/login';
        }
      }, 3000);
    }
    
    // 权限不足（403错误）
    if (errorType === '403' || errorType === 'FORBIDDEN') {
      // 可以记录权限错误统计
      console.warn('权限不足错误:', error);
    }
    
    // 网络错误
    if (errorType === 'NETWORK_ERROR' || errorType === 'TIMEOUT') {
      // 可以触发网络状态检查
      const networkErrorEvent = new CustomEvent('network:error');
      window.dispatchEvent(networkErrorEvent);
    }
  }

  /**
   * 手动记录成功操作
   */
  logSuccess(message: string, context?: string): void {
    if (this.config.consoleLog) {
      console.log(`[成功] ${context ? `[${context}] ` : ''}${message}`);
    }
    
    // 可以记录到监控系统
    if (this.config.logToMonitor) {
      errorMonitor.captureWarning(`成功: ${message}`, { context, type: 'success' });
    }
  }

  /**
   * 手动记录警告
   */
  logWarning(message: string, context?: string): void {
    if (this.config.consoleLog) {
      console.warn(`[警告] ${context ? `[${context}] ` : ''}${message}`);
    }
    
    if (this.config.logToMonitor) {
      errorMonitor.captureWarning(message, { context });
    }
  }

  /**
   * 更新配置
   */
  updateConfig(config: Partial<ErrorHandlerConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * 获取当前配置
   */
  getConfig(): ErrorHandlerConfig {
    return { ...this.config };
  }
}

// 创建全局实例
export const errorHandler = new ErrorHandler();

// 导出工具函数
export function handleError(error: unknown, context?: string): void {
  errorHandler.handleError(error, context);
}

export function logSuccess(message: string, context?: string): void {
  errorHandler.logSuccess(message, context);
}

export function logWarning(message: string, context?: string): void {
  errorHandler.logWarning(message, context);
}

// React Hook形式的错误处理
export function useErrorHandler() {
  return {
    handleError: (error: unknown, context?: string) => errorHandler.handleError(error, context),
    logSuccess: (message: string, context?: string) => errorHandler.logSuccess(message, context),
    logWarning: (message: string, context?: string) => errorHandler.logWarning(message, context),
  };
}

export default errorHandler;