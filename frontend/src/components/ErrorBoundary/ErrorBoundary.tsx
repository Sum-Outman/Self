/**
 * React错误边界组件
 * 捕获组件树中的JavaScript错误，并显示降级UI
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { captureError } from '../../utils/errorMonitoring';

export interface ErrorBoundaryProps {
  /** 子组件 */
  children: ReactNode;
  /** 错误时显示的备用UI */
  fallback?: ReactNode;
  /** 错误时显示的消息 */
  errorMessage?: string;
  /** 是否在控制台记录错误 */
  logError?: boolean;
  /** 错误发生时的回调函数 */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** 自定义错误渲染函数 */
  renderError?: (error: Error, errorInfo: ErrorInfo, resetError: () => void) => ReactNode;
}

export interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: undefined,
      errorInfo: undefined,
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    // 更新state使下一次渲染显示降级UI
    return {
      hasError: true,
      error,
      errorInfo: undefined,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // 错误信息
    this.setState({
      error,
      errorInfo,
    });

    // 控制台记录
    if (this.props.logError !== false) {
      console.error('错误边界捕获的错误:', error);
      console.error('组件堆栈:', errorInfo.componentStack);
    }

    // 错误监控
    captureError(error, errorInfo.componentStack || undefined, {
      componentStack: errorInfo.componentStack || undefined,
    });

    // 调用回调函数
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  /**
   * 重置错误状态
   */
  resetError = (): void => {
    this.setState({
      hasError: false,
      error: undefined,
      errorInfo: undefined,
    });
  };

  /**
   * 默认错误UI
   */
  renderDefaultError(): ReactNode {
    const { error, errorInfo } = this.state;
    const { errorMessage } = this.props;

    return (
      <div className="error-boundary min-h-[400px] flex flex-col items-center justify-center p-6 text-center">
        <div className="max-w-md space-y-4">
          {/* 错误图标 */}
          <div className="flex justify-center">
            <div className="rounded-full bg-gray-800 dark:bg-gray-900 p-4">
              <svg
                className="h-12 w-12 text-gray-900 dark:text-gray-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.986-.833-2.756 0L4.34 16.5c-.77.833.192 2.5 1.732 2.5z"
                />
              </svg>
            </div>
          </div>

          {/* 错误标题 */}
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
              {errorMessage || '抱歉，出现了错误'}
            </h2>
            <p className="mt-2 text-gray-600 dark:text-gray-400">
              系统遇到了意外问题，请稍后再试或联系技术支持。
            </p>
          </div>

          {/* 错误详情（开发环境显示） */}
          {process.env.NODE_ENV === 'development' && error && (
            <div className="mt-6 text-left">
              <details className="rounded-lg border border-gray-300 dark:border-gray-600">
                <summary className="cursor-pointer p-3 text-sm font-medium text-gray-700 dark:text-gray-300">
                  查看错误详情
                </summary>
                <div className="max-h-60 overflow-auto p-3 bg-gray-50 dark:bg-gray-800 text-sm">
                  <div className="mb-2">
                    <strong>错误信息:</strong>
                    <pre className="mt-1 whitespace-pre-wrap break-words text-gray-900 dark:text-gray-600">
                      {error.message}
                    </pre>
                  </div>
                  {error.stack && (
                    <div className="mb-2">
                      <strong>错误堆栈:</strong>
                      <pre className="mt-1 whitespace-pre-wrap break-words text-gray-600 dark:text-gray-400">
                        {error.stack}
                      </pre>
                    </div>
                  )}
                  {errorInfo?.componentStack && (
                    <div>
                      <strong>组件堆栈:</strong>
                      <pre className="mt-1 whitespace-pre-wrap break-words text-gray-600 dark:text-gray-400">
                        {errorInfo.componentStack}
                      </pre>
                    </div>
                  )}
                </div>
              </details>
            </div>
          )}

          {/* 操作按钮 */}
          <div className="mt-8 flex flex-col sm:flex-row gap-3 justify-center">
            <button
              type="button"
              onClick={this.resetError}
              className="inline-flex items-center justify-center px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
            >
              <svg
                className="h-5 w-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              重试
            </button>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="inline-flex items-center justify-center px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
            >
              <svg
                className="h-5 w-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
                />
              </svg>
              刷新页面
            </button>
            <button
              type="button"
              onClick={() => window.history.back()}
              className="inline-flex items-center justify-center px-4 py-2 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300 font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
            >
              <svg
                className="h-5 w-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10 19l-7-7m0 0l7-7m-7 7h18"
                />
              </svg>
              返回
            </button>
          </div>

          {/* 技术支持信息 */}
          <div className="mt-8 pt-6 border-t border-gray-300 dark:border-gray-600">
            <p className="text-sm text-gray-500 dark:text-gray-400">
              如果问题持续存在，请联系技术支持。
              <br />
              邮箱: <a href="mailto:support@example.com" className="text-gray-600 dark:text-gray-400 hover:underline">silenceceowtom@qq.com</a>
            </p>
          </div>
        </div>
      </div>
    );
  }

  render(): ReactNode {
    const { hasError, error, errorInfo } = this.state;
    const { children, fallback, renderError } = this.props;

    // 如果发生错误
    if (hasError && error) {
      // 自定义错误渲染
      if (renderError) {
        return renderError(error, errorInfo!, this.resetError);
      }
      
      // 备用UI
      if (fallback) {
        return fallback;
      }
      
      // 默认错误UI
      return this.renderDefaultError();
    }

    // 正常渲染子组件
    return children;
  }
}

/**
 * 高阶组件：包装组件提供错误边界
 */
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ErrorBoundaryProps, 'children'>
): React.FC<P> {
  const WrappedComponent: React.FC<P> = (props) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );
  
  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  return WrappedComponent;
}

/**
 * 全局错误边界（用于应用根组件）
 */
export const GlobalErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <ErrorBoundary
    errorMessage="抱歉，应用出现了严重错误"
    logError={true}
    onError={(error, errorInfo) => {
      console.error('全局错误边界捕获的错误:', error, errorInfo);
    }}
  >
    {children}
  </ErrorBoundary>
);

export default ErrorBoundary;