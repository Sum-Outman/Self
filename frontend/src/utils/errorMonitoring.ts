/**
 * 前端错误监控工具
 * 捕获和报告前端错误、性能指标和用户行为
 */

// 错误类型定义
export interface ErrorInfo {
  type: 'error' | 'warning' | 'info' | 'performance';
  message: string;
  stack?: string;
  componentStack?: string;
  timestamp: string;
  url: string;
  userAgent: string;
  userId?: string;
  sessionId?: string;
  additionalData?: Record<string, any>;
}

export interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  timestamp: string;
  tags?: Record<string, string>;
}

export interface MonitoringConfig {
  enabled: boolean;
  apiEndpoint: string;
  sampleRate: number; // 采样率 0-1
  maxErrorsPerMinute: number;
  reportInterval: number; // 报告间隔（毫秒）
  capturePerformance: boolean;
  captureUserActions: boolean;
}

// 默认配置
const defaultConfig: MonitoringConfig = {
  enabled: true,
  apiEndpoint: '/api/monitoring/errors',
  sampleRate: 1.0, // 默认采集所有错误
  maxErrorsPerMinute: 10,
  reportInterval: 10000, // 10秒
  capturePerformance: true,
  captureUserActions: false,
};

class ErrorMonitor {
  private config: MonitoringConfig;
  private errors: ErrorInfo[] = [];
  private performanceMetrics: PerformanceMetric[] = [];
  private errorCount = 0;
  private lastReportTime = 0;
  private sessionId: string;
  private isInitialized = false;

  constructor(config: Partial<MonitoringConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
    this.sessionId = this.generateSessionId();
  }

  /**
   * 初始化错误监控
   */
  initialize(): void {
    if (this.isInitialized || !this.config.enabled) {
      return;
    }

    // 捕获全局错误
    window.addEventListener('error', this.handleGlobalError.bind(this));
    
    // 捕获未处理的Promise拒绝
    window.addEventListener('unhandledrejection', this.handlePromiseRejection.bind(this));
    
    // 捕获控制台错误（可选）
    this.interceptConsoleErrors();
    
    // 性能监控
    if (this.config.capturePerformance && 'performance' in window) {
      this.setupPerformanceMonitoring();
    }
    
    // 定期报告
    this.setupPeriodicReporting();
    
    this.isInitialized = true;
    console.info('前端错误监控已初始化');
  }

  /**
   * 处理全局错误
   */
  private handleGlobalError(event: ErrorEvent): void {
    // 跳过跨域脚本错误（无法获取详细信息）
    if (event.message === 'Script error.') {
      return;
    }

    const errorInfo: ErrorInfo = {
      type: 'error',
      message: event.message,
      stack: event.error?.stack,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      sessionId: this.sessionId,
      additionalData: {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
      },
    };

    this.recordError(errorInfo);
  }

  /**
   * 处理未处理的Promise拒绝
   */
  private handlePromiseRejection(event: PromiseRejectionEvent): void {
    const errorInfo: ErrorInfo = {
      type: 'error',
      message: event.reason?.message || 'Unhandled Promise Rejection',
      stack: event.reason?.stack,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      sessionId: this.sessionId,
      additionalData: {
        reason: String(event.reason),
      },
    };

    this.recordError(errorInfo);
  }

  /**
   * 拦截控制台错误
   */
  private interceptConsoleErrors(): void {
    const originalError = console.error;
    const originalWarn = console.warn;

    console.error = (...args: any[]) => {
      const errorInfo: ErrorInfo = {
        type: 'error',
        message: args.map(arg => String(arg)).join(' '),
        timestamp: new Date().toISOString(),
        url: window.location.href,
        userAgent: navigator.userAgent,
        sessionId: this.sessionId,
      };

      this.recordError(errorInfo);
      originalError.apply(console, args);
    };

    console.warn = (...args: any[]) => {
      const errorInfo: ErrorInfo = {
        type: 'warning',
        message: args.map(arg => String(arg)).join(' '),
        timestamp: new Date().toISOString(),
        url: window.location.href,
        userAgent: navigator.userAgent,
        sessionId: this.sessionId,
      };

      this.recordError(errorInfo);
      originalWarn.apply(console, args);
    };
  }

  /**
   * 设置性能监控
   */
  private setupPerformanceMonitoring(): void {
    // 监控页面加载性能
    window.addEventListener('load', () => {
      setTimeout(() => {
        this.capturePerformanceMetrics();
      }, 0);
    });

    // 监控长任务
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.entryType === 'longtask') {
              this.recordPerformanceMetric({
                name: 'longtask',
                value: entry.duration,
                unit: 'ms',
                timestamp: new Date().toISOString(),
                tags: {
                  entryType: entry.entryType,
                },
              });
            }
          }
        });
        observer.observe({ entryTypes: ['longtask'] });
      } catch (e) {
        console.warn('性能监控初始化失败:', e);
      }
    }
  }

  /**
   * 捕获性能指标
   */
  private capturePerformanceMetrics(): void {
    if (!window.performance || !window.performance.timing) {
      return;
    }

    const timing = window.performance.timing;

    if (timing.loadEventEnd > 0) {
      const metrics = [
        { name: 'dns_lookup', value: timing.domainLookupEnd - timing.domainLookupStart },
        { name: 'tcp_connect', value: timing.connectEnd - timing.connectStart },
        { name: 'request_response', value: timing.responseEnd - timing.requestStart },
        { name: 'dom_parse', value: timing.domComplete - timing.domInteractive },
        { name: 'dom_content_loaded', value: timing.domContentLoadedEventEnd - timing.navigationStart },
        { name: 'page_load', value: timing.loadEventEnd - timing.navigationStart },
      ];

      metrics.forEach(metric => {
        if (metric.value > 0) {
          this.recordPerformanceMetric({
            name: metric.name,
            value: metric.value,
            unit: 'ms',
            timestamp: new Date().toISOString(),
          });
        }
      });
    }

    // 捕获资源加载性能
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.entryType === 'resource') {
              this.recordPerformanceMetric({
                name: 'resource_load',
                value: entry.duration,
                unit: 'ms',
                timestamp: new Date().toISOString(),
                tags: {
                  name: entry.name,
                  initiatorType: (entry as PerformanceResourceTiming).initiatorType,
                },
              });
            }
          }
        });
        observer.observe({ entryTypes: ['resource'] });
      } catch (e) {
        console.warn('资源性能监控初始化失败:', e);
      }
    }
  }

  /**
   * 记录错误
   */
  private recordError(errorInfo: ErrorInfo): void {
    // 采样率控制
    if (Math.random() > this.config.sampleRate) {
      return;
    }

    // 错误频率限制
    this.errorCount++;
    if (this.errorCount > this.config.maxErrorsPerMinute) {
      return;
    }

    // 添加用户ID（如果可用）
    try {
      const userId = localStorage.getItem('user_id');
      if (userId) {
        errorInfo.userId = userId;
      }
    } catch (e) {
      // 忽略localStorage错误
    }

    this.errors.push(errorInfo);
    
    // 立即报告关键错误
    if (errorInfo.type === 'error') {
      this.reportIfNeeded(true);
    }
  }

  /**
   * 记录性能指标
   */
  recordPerformanceMetric(metric: PerformanceMetric): void {
    if (!this.config.capturePerformance) {
      return;
    }

    this.performanceMetrics.push(metric);
  }

  /**
   * 手动记录错误
   */
  captureError(error: Error | string, componentStack?: string, additionalData?: Record<string, any>): void {
    const errorInfo: ErrorInfo = {
      type: 'error',
      message: typeof error === 'string' ? error : error.message,
      stack: typeof error === 'object' ? error.stack : undefined,
      componentStack,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      sessionId: this.sessionId,
      additionalData,
    };

    this.recordError(errorInfo);
  }

  /**
   * 手动记录警告
   */
  captureWarning(message: string, additionalData?: Record<string, any>): void {
    const errorInfo: ErrorInfo = {
      type: 'warning',
      message,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      sessionId: this.sessionId,
      additionalData,
    };

    this.recordError(errorInfo);
  }

  /**
   * 设置定期报告
   */
  private setupPeriodicReporting(): void {
    setInterval(() => {
      this.reportIfNeeded(false);
    }, this.config.reportInterval);
  }

  /**
   * 报告错误（如果需要）
   */
  private reportIfNeeded(force: boolean = false): void {
    const now = Date.now();
    const shouldReport = force || 
      (this.errors.length > 0 && now - this.lastReportTime >= this.config.reportInterval);

    if (!shouldReport) {
      return;
    }

    this.reportErrors();
    this.lastReportTime = now;
  }

  /**
   * 报告错误到后端
   */
  private async reportErrors(): Promise<void> {
    if (this.errors.length === 0) {
      return;
    }

    const errorsToReport = [...this.errors];
    this.errors = [];

    try {
      // 构建请求头，只有在有访问令牌时才包含Authorization头
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      
      // 检查是否有访问令牌（仅在用户已登录时）
      try {
        const accessToken = localStorage.getItem('access_token');
        if (accessToken) {
          headers['Authorization'] = `Bearer ${accessToken}`;
        }
      } catch (e) {
        // 忽略localStorage错误，继续发送无认证请求
        console.debug('无法获取访问令牌，发送无认证错误报告:', e);
      }

      const response = await fetch(this.config.apiEndpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          errors: errorsToReport,
          performanceMetrics: this.performanceMetrics,
          timestamp: new Date().toISOString(),
          sessionId: this.sessionId,
        }),
      });

      if (response.ok) {
        this.performanceMetrics = [];
      } else {
        console.warn('错误报告失败:', response.status);
        // 重新添加错误以便稍后重试
        this.errors.unshift(...errorsToReport);
      }
    } catch (error) {
      console.warn('错误报告请求失败:', error);
      // 重新添加错误以便稍后重试
      this.errors.unshift(...errorsToReport);
    }
  }

  /**
   * 生成会话ID
   */
  private generateSessionId(): string {
    return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now().toString(36);
  }

  /**
   * 获取当前配置
   */
  getConfig(): MonitoringConfig {
    return { ...this.config };
  }

  /**
   * 更新配置
   */
  updateConfig(newConfig: Partial<MonitoringConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * 获取错误统计
   */
  getStats(): { errorCount: number; warningCount: number; performanceMetricsCount: number } {
    const errorCount = this.errors.filter(e => e.type === 'error').length;
    const warningCount = this.errors.filter(e => e.type === 'warning').length;
    
    return {
      errorCount,
      warningCount,
      performanceMetricsCount: this.performanceMetrics.length,
    };
  }

  /**
   * 清空所有记录
   */
  clear(): void {
    this.errors = [];
    this.performanceMetrics = [];
    this.errorCount = 0;
  }
}

// 创建全局实例
export const errorMonitor = new ErrorMonitor();

// 导出工具函数
export function initializeErrorMonitoring(config?: Partial<MonitoringConfig>): void {
  errorMonitor.updateConfig(config || {});
  errorMonitor.initialize();
}

export function captureError(error: Error | string, componentStack?: string, additionalData?: Record<string, any>): void {
  errorMonitor.captureError(error, componentStack, additionalData);
}

export function captureWarning(message: string, additionalData?: Record<string, any>): void {
  errorMonitor.captureWarning(message, additionalData);
}

export function recordPerformanceMetric(metric: PerformanceMetric): void {
  errorMonitor.recordPerformanceMetric(metric);
}

// 默认导出
export default errorMonitor;