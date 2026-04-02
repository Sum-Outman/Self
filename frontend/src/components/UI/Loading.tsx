import React from 'react';
import { cn } from '../../utils/cn';
import { Loader2, Brain, Cpu } from 'lucide-react';

export type LoadingType = 'spinner' | 'dots' | 'pulse' | 'progress' | 'skeleton' | 'content';
export type LoadingSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';
export type LoadingVariant = 'primary' | 'secondary' | 'accent' | 'white' | 'gray';

export interface LoadingProps {
  /** 加载类型 */
  type?: LoadingType;
  /** 加载尺寸 */
  size?: LoadingSize;
  /** 加载变体 */
  variant?: LoadingVariant;
  /** 加载文字 */
  text?: string;
  /** 是否全屏 */
  fullscreen?: boolean;
  /** 是否覆盖内容 */
  overlay?: boolean;
  /** 进度值（0-100），仅对progress类型有效 */
  progress?: number;
  /** 是否显示百分比，仅对progress类型有效 */
  showPercentage?: boolean;
  /** 骨架屏行数，仅对skeleton类型有效 */
  skeletonRows?: number;
  /** 自定义类名 */
  className?: string;
  /** 子元素 */
  children?: React.ReactNode;
}

const sizeClasses: Record<LoadingSize, string> = {
  xs: 'h-4 w-4',
  sm: 'h-6 w-6',
  md: 'h-8 w-8',
  lg: 'h-12 w-12',
  xl: 'h-16 w-16',
};

const textSizeClasses: Record<LoadingSize, string> = {
  xs: 'text-xs',
  sm: 'text-sm',
  md: 'text-base',
  lg: 'text-lg',
  xl: 'text-xl',
};

const variantClasses: Record<LoadingVariant, string> = {
  primary: 'text-gray-600 dark:text-gray-400',
  secondary: 'text-gray-600 dark:text-gray-400',
  accent: 'text-gray-600 dark:text-gray-400',
  white: 'text-white',
  gray: 'text-gray-400 dark:text-gray-500',
};

export const Loading: React.FC<LoadingProps> = ({
  type = 'spinner',
  size = 'md',
  variant = 'primary',
  text,
  fullscreen = false,
  overlay = false,
  progress = 0,
  showPercentage = false,
  skeletonRows = 3,
  className,
  children,
}) => {
  const renderLoader = () => {
    switch (type) {
      case 'spinner':
        return (
          <div className="flex flex-col items-center justify-center">
            <Loader2 className={cn(sizeClasses[size], variantClasses[variant], 'animate-spin')} />
            {text && (
              <p className={cn(textSizeClasses[size], 'mt-2', variantClasses[variant])}>
                {text}
              </p>
            )}
          </div>
        );

      case 'dots':
        return (
          <div className="flex flex-col items-center justify-center">
            <div className="flex items-center space-x-1">
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className={cn(
                    sizeClasses[size === 'xs' ? 'xs' : 'sm'],
                    'rounded-full',
                    variantClasses[variant],
                    'animate-bounce'
                  )}
                  style={{
                    animationDelay: `${i * 0.1}s`,
                    animationDuration: '1s',
                  }}
                />
              ))}
            </div>
            {text && (
              <p className={cn(textSizeClasses[size], 'mt-4', variantClasses[variant])}>
                {text}
              </p>
            )}
          </div>
        );

      case 'pulse':
        return (
          <div className="flex flex-col items-center justify-center">
            <div className="relative">
              <div
                className={cn(
                  sizeClasses[size],
                  'rounded-full',
                  variantClasses[variant],
                  'opacity-75 animate-pulse'
                )}
              />
              <div
                className={cn(
                  sizeClasses[size],
                  'rounded-full',
                  variantClasses[variant],
                  'absolute top-0 left-0 animate-ping'
                )}
              />
            </div>
            {text && (
              <p className={cn(textSizeClasses[size], 'mt-4', variantClasses[variant])}>
                {text}
              </p>
            )}
          </div>
        );

      case 'progress':
        const validProgress = Math.min(100, Math.max(0, progress));
        
        return (
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="w-48 md:w-64">
              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={cn(
                    'h-full transition-all duration-300 ease-out',
                    variant === 'primary' && 'bg-gray-600',
                    variant === 'secondary' && 'bg-gray-600',
                    variant === 'accent' && 'bg-gray-600',
                    variant === 'white' && 'bg-white',
                    variant === 'gray' && 'bg-gray-400'
                  )}
                  style={{ width: `${validProgress}%` }}
                />
              </div>
              {showPercentage && (
                <div className="flex justify-between mt-1">
                  <span className={cn(textSizeClasses.sm, variantClasses[variant])}>
                    0%
                  </span>
                  <span className={cn(textSizeClasses.sm, variantClasses[variant])}>
                    {validProgress}%
                  </span>
                  <span className={cn(textSizeClasses.sm, variantClasses[variant])}>
                    100%
                  </span>
                </div>
              )}
            </div>
            {text && (
              <p className={cn(textSizeClasses[size], variantClasses[variant])}>
                {text}
              </p>
            )}
          </div>
        );

      case 'skeleton':
        return (
          <div className="space-y-3">
            {Array.from({ length: skeletonRows }).map((_, i) => (
              <div
                key={i}
                className={cn(
                  'h-4 bg-gray-200 dark:bg-gray-700 rounded animate-pulse',
                  i === skeletonRows - 1 ? 'w-3/4' : 'w-full'
                )}
              />
            ))}
            {text && (
              <p className={cn(textSizeClasses[size], 'mt-2', variantClasses[variant])}>
                {text}
              </p>
            )}
          </div>
        );

      case 'content':
        return (
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="relative">
              <Brain className={cn(sizeClasses[size], variantClasses[variant], 'animate-float')} />
              <Cpu className={cn(
                sizeClasses[size === 'xs' ? 'xs' : 'sm'],
                variantClasses[variant],
                'absolute -top-2 -right-2 animate-pulse-slow'
              )} />
            </div>
            {text && (
              <p className={cn(textSizeClasses[size], variantClasses[variant])}>
                {text}
              </p>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  // 全屏加载
  if (fullscreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white dark:bg-gray-900">
        <div className="text-center">
          {renderLoader()}
          {children}
        </div>
      </div>
    );
  }

  // 覆盖层加载
  if (overlay) {
    return (
      <div className="absolute inset-0 z-40 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
        <div className="text-center">
          {renderLoader()}
          {children}
        </div>
      </div>
    );
  }

  // 内联加载
  return (
    <div className={cn('inline-flex items-center justify-center', className)}>
      {renderLoader()}
      {children}
    </div>
  );
};

Loading.displayName = 'Loading';

// 预定义的加载组件
export const Spinner = (props: Omit<LoadingProps, 'type'>) => (
  <Loading type="spinner" {...props} />
);

export const DotsLoader = (props: Omit<LoadingProps, 'type'>) => (
  <Loading type="dots" {...props} />
);

export const PulseLoader = (props: Omit<LoadingProps, 'type'>) => (
  <Loading type="pulse" {...props} />
);

export const ProgressLoader = (props: Omit<LoadingProps, 'type'>) => (
  <Loading type="progress" {...props} />
);

export const SkeletonLoader = (props: Omit<LoadingProps, 'type'>) => (
  <Loading type="skeleton" {...props} />
);

export const ContentLoader = (props: Omit<LoadingProps, 'type'>) => (
  <Loading type="content" {...props} />
);

// 页面加载组件
export const PageLoader: React.FC<{
  title?: string;
  subtitle?: string;
  variant?: LoadingVariant;
}> = ({ title = '加载中...', subtitle = '请稍候，正在准备您的内容', variant = 'primary' }) => (
  <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-white dark:bg-gray-900">
    <div className="text-center space-y-6">
      <div className="relative">
        <Brain className="h-20 w-20 text-gray-600 dark:text-gray-400 animate-float" />
        <Cpu className="h-8 w-8 text-gray-600 dark:text-gray-400 absolute -top-2 -right-2 animate-pulse-slow" />
      </div>
      <div className="space-y-2">
        <h1 className={cn(
          'text-2xl font-bold',
          variant === 'primary' && 'text-gray-700 dark:text-gray-300',
          variant === 'secondary' && 'text-gray-700 dark:text-gray-300',
          variant === 'accent' && 'text-gray-700 dark:text-gray-300',
          variant === 'white' && 'text-white',
          variant === 'gray' && 'text-gray-500 dark:text-gray-400'
        )}>
          {title}
        </h1>
        <p className={cn(
          'text-lg',
          variant === 'primary' && 'text-gray-600 dark:text-gray-400',
          variant === 'secondary' && 'text-gray-600 dark:text-gray-400',
          variant === 'accent' && 'text-gray-600 dark:text-gray-400',
          variant === 'white' && 'text-white/80',
          variant === 'gray' && 'text-gray-400 dark:text-gray-500'
        )}>
          {subtitle}
        </p>
      </div>
      <div className="w-64 mx-auto">
        <div className="h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div className="h-full bg-gray-600 dark:bg-gray-400 animate-progress" />
        </div>
      </div>
    </div>
  </div>
);

// 骨架屏组件
export const Skeleton: React.FC<{
  type?: 'text' | 'circle' | 'rect' | 'card' | 'list';
  width?: string;
  height?: string;
  className?: string;
}> = ({ type = 'text', width = '100%', height = '1rem', className }) => {
  const baseClasses = 'bg-gray-200 dark:bg-gray-700 animate-pulse rounded';
  
  switch (type) {
    case 'text':
      return (
        <div
          className={cn(baseClasses, className)}
          style={{ width, height }}
        />
      );
    
    case 'circle':
      return (
        <div
          className={cn(baseClasses, 'rounded-full', className)}
          style={{ width, height }}
        />
      );
    
    case 'rect':
      return (
        <div
          className={cn(baseClasses, 'rounded-md', className)}
          style={{ width, height }}
        />
      );
    
    case 'card':
      return (
        <div className={cn(baseClasses, 'rounded-lg p-4 space-y-3', className)}>
          <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-3/4" />
          <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-1/2" />
          <div className="h-20 bg-gray-300 dark:bg-gray-600 rounded w-full" />
        </div>
      );
    
    case 'list':
      return (
        <div className={cn(baseClasses, 'rounded-lg p-4 space-y-2', className)}>
          {[1, 2, 3].map((i) => (
            <div key={i} className="flex items-center space-x-3">
              <div className="h-10 w-10 bg-gray-300 dark:bg-gray-600 rounded-full" />
              <div className="flex-1 space-y-1">
                <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-3/4" />
                <div className="h-2 bg-gray-300 dark:bg-gray-600 rounded w-1/2" />
              </div>
            </div>
          ))}
        </div>
      );
    
    default:
      return null;
  }
};