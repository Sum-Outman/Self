/**
 * 用户体验工具函数
 * 提供改善用户体验的辅助函数和工具
 */

import { toast } from 'react-hot-toast';

/**
 * 用户体验配置接口
 */
export interface UXConfig {
  // 加载状态配置
  loading: {
    // 最小显示时间（毫秒）
    minDisplayTime: number;
    // 最大显示时间（毫秒）
    maxDisplayTime: number;
    // 默认加载文本
    defaultText: string;
  };
  // 错误处理配置
  errorHandling: {
    // 自动重试次数
    autoRetryCount: number;
    // 重试延迟（毫秒）
    retryDelay: number;
    // 显示详细错误信息
    showDetailedErrors: boolean;
  };
  // 动画配置
  animations: {
    // 启用动画
    enabled: boolean;
    // 动画持续时间（毫秒）
    duration: number;
    // 缓动函数
    easing: string;
  };
  // 表单配置
  form: {
    // 自动保存延迟（毫秒）
    autoSaveDelay: number;
    // 实时验证
    realtimeValidation: boolean;
    // 验证延迟（毫秒）
    validationDelay: number;
  };
  // 通知配置
  notifications: {
    // 成功通知持续时间（毫秒）
    successDuration: number;
    // 错误通知持续时间（毫秒）
    errorDuration: number;
    // 警告通知持续时间（毫秒）
    warningDuration: number;
    // 信息通知持续时间（毫秒）
    infoDuration: number;
  };
}

/**
 * 默认用户体验配置
 */
export const defaultUXConfig: UXConfig = {
  loading: {
    minDisplayTime: 300,
    maxDisplayTime: 10000,
    defaultText: '加载中...',
  },
  errorHandling: {
    autoRetryCount: 3,
    retryDelay: 1000,
    showDetailedErrors: process.env.NODE_ENV !== 'production',
  },
  animations: {
    enabled: true,
    duration: 300,
    easing: 'ease-in-out',
  },
  form: {
    autoSaveDelay: 1000,
    realtimeValidation: true,
    validationDelay: 500,
  },
  notifications: {
    successDuration: 3000,
    errorDuration: 5000,
    warningDuration: 4000,
    infoDuration: 3000,
  },
};

/**
 * 加载状态管理器
 */
export class LoadingManager {
  private static instance: LoadingManager;
  private loaders: Map<string, { startTime: number; text: string }> = new Map();
  private config: UXConfig['loading'];
  
  private constructor(config: UXConfig['loading'] = defaultUXConfig.loading) {
    this.config = config;
  }
  
  static getInstance(config?: UXConfig['loading']): LoadingManager {
    if (!LoadingManager.instance) {
      LoadingManager.instance = new LoadingManager(config);
    }
    return LoadingManager.instance;
  }
  
  /**
   * 开始加载
   * @param id 加载ID
   * @param text 加载文本
   */
  start(id: string, text: string = this.config.defaultText): void {
    this.loaders.set(id, {
      startTime: Date.now(),
      text,
    });
  }
  
  /**
   * 结束加载
   * @param id 加载ID
   * @param force 是否强制结束（忽略最小显示时间）
   */
  end(id: string, force: boolean = false): void {
    const loader = this.loaders.get(id);
    if (!loader) return;
    
    const elapsed = Date.now() - loader.startTime;
    const remaining = Math.max(0, this.config.minDisplayTime - elapsed);
    
    if (force || remaining <= 0) {
      this.loaders.delete(id);
    } else {
      // 等待最小显示时间后再结束
      setTimeout(() => {
        this.loaders.delete(id);
      }, remaining);
    }
  }
  
  /**
   * 检查是否正在加载
   * @param id 加载ID
   */
  isLoading(id: string): boolean {
    return this.loaders.has(id);
  }
  
  /**
   * 获取所有正在进行的加载
   */
  getAllLoadings(): Array<{ id: string; text: string; duration: number }> {
    const now = Date.now();
    return Array.from(this.loaders.entries()).map(([id, loader]) => ({
      id,
      text: loader.text,
      duration: now - loader.startTime,
    }));
  }
  
  /**
   * 清除所有加载状态
   */
  clearAll(): void {
    this.loaders.clear();
  }
}

/**
 * 错误处理器
 */
export class ErrorHandler {
  private static instance: ErrorHandler;
  private config: UXConfig['errorHandling'];
  private errorHistory: Array<{
    timestamp: number;
    error: any;
    context: Record<string, any>;
  }> = [];
  
  private constructor(config: UXConfig['errorHandling'] = defaultUXConfig.errorHandling) {
    this.config = config;
  }
  
  static getInstance(config?: UXConfig['errorHandling']): ErrorHandler {
    if (!ErrorHandler.instance) {
      ErrorHandler.instance = new ErrorHandler(config);
    }
    return ErrorHandler.instance;
  }
  
  /**
   * 处理错误
   * @param error 错误对象
   * @param context 错误上下文
   * @param userMessage 显示给用户的消息
   */
  handle(
    error: any,
    context: Record<string, any> = {},
    userMessage?: string
  ): {
    shouldRetry: boolean;
    retryDelay: number;
    userMessage: string;
  } {
    const errorId = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // 记录错误历史
    this.errorHistory.push({
      timestamp: Date.now(),
      error,
      context: { ...context, errorId },
    });
    
    // 限制历史记录大小
    if (this.errorHistory.length > 100) {
      this.errorHistory = this.errorHistory.slice(-100);
    }
    
    // 分析错误类型
    const errorAnalysis = this.analyzeError(error);
    
    // 生成用户友好的消息
    const message = userMessage || this.getUserFriendlyMessage(errorAnalysis);
    
    // 显示错误通知
    this.showErrorNotification(message, errorAnalysis);
    
    return {
      shouldRetry: errorAnalysis.retryable && context.retryCount < this.config.autoRetryCount,
      retryDelay: this.config.retryDelay,
      userMessage: message,
    };
  }
  
  /**
   * 分析错误
   */
  private analyzeError(error: any): {
    type: 'network' | 'server' | 'client' | 'validation' | 'permission' | 'unknown';
    retryable: boolean;
    severity: 'low' | 'medium' | 'high' | 'critical';
    details: string;
  } {
    const errorStr = error?.toString()?.toLowerCase() || '';
    const status = error?.response?.status;
    const code = error?.code || error?.name;
    
    // 网络错误
    if (
      errorStr.includes('network') ||
      errorStr.includes('internet') ||
      code === 'NETWORK_ERROR' ||
      error.name === 'NetworkError'
    ) {
      return {
        type: 'network',
        retryable: true,
        severity: 'medium',
        details: '网络连接问题',
      };
    }
    
    // 服务器错误（5xx）
    if (status >= 500 && status < 600) {
      return {
        type: 'server',
        retryable: true,
        severity: 'high',
        details: `服务器错误 (${status})`,
      };
    }
    
    // 客户端错误（4xx）
    if (status >= 400 && status < 500) {
      if (status === 401 || status === 403) {
        return {
          type: 'permission',
          retryable: false,
          severity: 'high',
          details: `权限错误 (${status})`,
        };
      } else if (status === 422 || status === 400) {
        return {
          type: 'validation',
          retryable: false,
          severity: 'medium',
          details: `验证错误 (${status})`,
        };
      } else {
        return {
          type: 'client',
          retryable: false,
          severity: 'medium',
          details: `客户端错误 (${status})`,
        };
      }
    }
    
    // 超时错误
    if (
      errorStr.includes('timeout') ||
      errorStr.includes('timed out') ||
      code === 'ECONNABORTED' ||
      error.name === 'TimeoutError'
    ) {
      return {
        type: 'network',
        retryable: true,
        severity: 'medium',
        details: '请求超时',
      };
    }
    
    // 默认未知错误
    return {
      type: 'unknown',
      retryable: false,
      severity: 'high',
      details: '未知错误',
    };
  }
  
  /**
   * 获取用户友好的错误消息
   */
  private getUserFriendlyMessage(analysis: ReturnType<ErrorHandler['analyzeError']>): string {
    const messages = {
      network: '网络连接失败，请检查网络设置',
      server: '服务器暂时不可用，请稍后重试',
      client: '请求参数错误，请检查输入',
      validation: '输入验证失败，请检查表单',
      permission: '权限不足，无法执行此操作',
      unknown: '操作失败，请稍后重试',
    };
    
    const detailedMessages = {
      network: '网络连接问题，请检查您的网络连接和代理设置',
      server: '服务器内部错误，我们的技术团队已收到通知',
      client: '客户端请求错误，请检查输入数据格式',
      validation: '表单验证失败，请检查所有必填字段',
      permission: '您的账户没有执行此操作的权限',
      unknown: '发生未知错误，请刷新页面或联系技术支持',
    };
    
    return this.config.showDetailedErrors
      ? detailedMessages[analysis.type]
      : messages[analysis.type];
  }
  
  /**
   * 显示错误通知
   */
  private showErrorNotification(message: string, analysis: ReturnType<ErrorHandler['analyzeError']>): void {
    const duration = analysis.severity === 'critical' ? 10000 :
                    analysis.severity === 'high' ? 7000 :
                    analysis.severity === 'medium' ? 5000 : 3000;
    
    toast.error(message, {
      duration,
      style: {
        background: analysis.severity === 'critical' ? '#7f1d1d' :
                   analysis.severity === 'high' ? '#991b1b' :
                   analysis.severity === 'medium' ? '#dc2626' : '#ef4444',
        color: '#f9fafb',
      },
      iconTheme: {
        primary: '#fca5a5',
        secondary: '#f9fafb',
      },
    });
  }
  
  /**
   * 获取错误历史
   */
  getErrorHistory(): Array<{
    timestamp: number;
    type: string;
    severity: string;
    details: string;
  }> {
    return this.errorHistory.map((entry) => {
      const analysis = this.analyzeError(entry.error);
      return {
        timestamp: entry.timestamp,
        type: analysis.type,
        severity: analysis.severity,
        details: analysis.details,
      };
    });
  }
  
  /**
   * 清除错误历史
   */
  clearErrorHistory(): void {
    this.errorHistory = [];
  }
}

/**
 * 表单验证器
 */
export class FormValidator {
  /**
   * 验证电子邮件
   */
  static validateEmail(email: string): { valid: boolean; message?: string } {
    if (!email) {
      return { valid: false, message: '电子邮件不能为空' };
    }
    
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return { valid: false, message: '电子邮件格式不正确' };
    }
    
    return { valid: true };
  }
  
  /**
   * 验证密码
   */
  static validatePassword(password: string): { valid: boolean; message?: string; strength?: number } {
    if (!password) {
      return { valid: false, message: '密码不能为空' };
    }
    
    if (password.length < 8) {
      return { valid: false, message: '密码至少需要8个字符' };
    }
    
    // 计算密码强度（0-100）
    let strength = 0;
    
    // 长度评分
    strength += Math.min(password.length * 5, 30);
    
    // 多样性评分
    const hasLowercase = /[a-z]/.test(password);
    const hasUppercase = /[A-Z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecial = /[^A-Za-z0-9]/.test(password);
    
    if (hasLowercase) strength += 10;
    if (hasUppercase) strength += 15;
    if (hasNumbers) strength += 15;
    if (hasSpecial) strength += 20;
    
    // 唯一性评分
    const uniqueChars = new Set(password).size;
    strength += Math.min(uniqueChars * 2, 20);
    
    if (strength < 50) {
      return { 
        valid: false, 
        message: '密码强度不足，请使用大小写字母、数字和特殊字符组合',
        strength 
      };
    }
    
    return { valid: true, strength };
  }
  
  /**
   * 验证用户名
   */
  static validateUsername(username: string): { valid: boolean; message?: string } {
    if (!username) {
      return { valid: false, message: '用户名不能为空' };
    }
    
    if (username.length < 3) {
      return { valid: false, message: '用户名至少需要3个字符' };
    }
    
    if (username.length > 30) {
      return { valid: false, message: '用户名不能超过30个字符' };
    }
    
    const usernameRegex = /^[a-zA-Z0-9_\u4e00-\u9fa5]+$/;
    if (!usernameRegex.test(username)) {
      return { valid: false, message: '用户名只能包含字母、数字、下划线和中文字符' };
    }
    
    return { valid: true };
  }
  
  /**
   * 验证手机号码
   */
  static validatePhone(phone: string): { valid: boolean; message?: string } {
    if (!phone) {
      return { valid: false, message: '手机号码不能为空' };
    }
    
    const phoneRegex = /^1[3-9]\d{9}$/;
    if (!phoneRegex.test(phone)) {
      return { valid: false, message: '手机号码格式不正确' };
    }
    
    return { valid: true };
  }
  
  /**
   * 验证URL
   */
  static validateUrl(url: string): { valid: boolean; message?: string } {
    if (!url) {
      return { valid: false, message: 'URL不能为空' };
    }
    
    try {
      new URL(url);
      return { valid: true };
    } catch {
      return { valid: false, message: 'URL格式不正确' };
    }
  }
}

/**
 * 动画工具
 */
export class AnimationHelper {
  /**
   * 创建CSS动画
   */
  static createAnimation(
    element: HTMLElement,
    keyframes: Keyframe[] | PropertyIndexedKeyframes,
    options: KeyframeAnimationOptions
  ): Animation {
    return element.animate(keyframes, options);
  }
  
  /**
   * 淡入动画
   */
  static fadeIn(element: HTMLElement, duration: number = 300): Animation {
    return this.createAnimation(
      element,
      [
        { opacity: 0, transform: 'translateY(10px)' },
        { opacity: 1, transform: 'translateY(0)' },
      ],
      {
        duration,
        easing: 'ease-out',
        fill: 'forwards',
      }
    );
  }
  
  /**
   * 淡出动画
   */
  static fadeOut(element: HTMLElement, duration: number = 300): Animation {
    return this.createAnimation(
      element,
      [
        { opacity: 1, transform: 'translateY(0)' },
        { opacity: 0, transform: 'translateY(10px)' },
      ],
      {
        duration,
        easing: 'ease-in',
        fill: 'forwards',
      }
    );
  }
  
  /**
   * 缩放动画
   */
  static scale(element: HTMLElement, from: number, to: number, duration: number = 300): Animation {
    return this.createAnimation(
      element,
      [
        { transform: `scale(${from})` },
        { transform: `scale(${to})` },
      ],
      {
        duration,
        easing: 'ease-in-out',
        fill: 'forwards',
      }
    );
  }
}

/**
 * 防抖函数
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * 节流函数
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false;
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

/**
 * 复制文本到剪贴板
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      toast.success('已复制到剪贴板');
      return true;
    } else {
      // 降级方案
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      textArea.style.top = '-999999px';
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      const successful = document.execCommand('copy');
      document.body.removeChild(textArea);
      
      if (successful) {
        toast.success('已复制到剪贴板');
      } else {
        toast.error('复制失败');
      }
      
      return successful;
    }
  } catch (error) {
    console.error('复制到剪贴板失败:', error);
    toast.error('复制失败');
    return false;
  }
}

/**
 * 格式化文件大小
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
}

/**
 * 格式化时间
 */
export function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  return `${Math.floor(ms / 3600000)}h ${Math.floor((ms % 3600000) / 60000)}m`;
}

/**
 * 生成唯一ID
 */
export function generateId(prefix: string = 'id'): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// 导出单例实例
export const loadingManager = LoadingManager.getInstance();
export const errorHandler = ErrorHandler.getInstance();