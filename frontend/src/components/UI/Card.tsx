import React from 'react';
import { cn } from '../../utils/cn';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  /** 是否显示悬停效果 */
  hover?: boolean;
  /** 是否可点击 */
  clickable?: boolean;
  /** 是否显示边框 */
  bordered?: boolean;
  /** 是否显示阴影 */
  shadow?: 'none' | 'sm' | 'md' | 'lg';
  /** 卡片标题 */
  title?: string;
  /** 卡片副标题 */
  subtitle?: string;
  /** 卡片头部内容 */
  header?: React.ReactNode;
  /** 卡片底部内容 */
  footer?: React.ReactNode;
  /** 是否加载中 */
  loading?: boolean;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  (
    {
      className,
      hover = false,
      clickable = false,
      bordered = true,
      shadow = 'md',
      title,
      subtitle,
      header,
      footer,
      loading = false,
      children,
      ...props
    },
    ref
  ) => {
    const shadowClasses = {
      none: '',
      sm: 'shadow-sm',
      md: 'shadow-md',
      lg: 'shadow-lg',
    };
    
    return (
      <div
        ref={ref}
        className={cn(
          'rounded-lg',
          bordered && 'border border-gray-200 dark:border-gray-700',
          'bg-white dark:bg-gray-800',
          shadowClasses[shadow],
          hover && 'hover:shadow-lg transition-shadow duration-200',
          clickable && 'cursor-pointer hover:border-gray-300 dark:hover:border-gray-700',
          loading && 'animate-pulse',
          className
        )}
        role={clickable ? 'button' : undefined}
        tabIndex={clickable ? 0 : undefined}
        {...props}
      >
        {(title || subtitle || header) && (
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
            {header ? (
              <div>{header}</div>
            ) : (
              <>
                {title && (
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                    {title}
                  </h3>
                )}
                {subtitle && (
                  <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                    {subtitle}
                  </p>
                )}
              </>
            )}
          </div>
        )}
        
        <div className="px-6 py-4">
          {loading ? (
            <div className="space-y-3">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded animate-pulse w-5/6"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded animate-pulse w-4/6"></div>
            </div>
          ) : (
            children
          )}
        </div>
        
        {footer && (
          <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 rounded-b-lg">
            {footer}
          </div>
        )}
      </div>
    );
  }
);

Card.displayName = 'Card';

// 导出预定义的卡片组件
export const SimpleCard = (props: Omit<CardProps, 'bordered' | 'shadow'>) => (
  <Card bordered={false} shadow="none" {...props} />
);

export const HoverCard = (props: Omit<CardProps, 'hover'>) => (
  <Card hover {...props} />
);

export const ClickableCard = (props: Omit<CardProps, 'clickable'>) => (
  <Card clickable {...props} />
);