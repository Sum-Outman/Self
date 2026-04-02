import React from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '../../utils/cn';

export type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'outline' | 'ghost';
export type ButtonSize = 'sm' | 'md' | 'lg' | 'icon';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /** 按钮变体 */
  variant?: ButtonVariant;
  /** 按钮尺寸 */
  size?: ButtonSize;
  /** 是否为加载状态 */
  loading?: boolean;
  /** 是否禁用 */
  disabled?: boolean;
  /** 左侧图标 */
  leftIcon?: React.ReactNode;
  /** 右侧图标 */
  rightIcon?: React.ReactNode;
  /** 是否全宽 */
  fullWidth?: boolean;
  /** 点击事件 */
  onClick?: () => void;
  /** 子元素 */
  children?: React.ReactNode;
}

const variantClasses: Record<ButtonVariant, string> = {
  primary: 'bg-gray-600 text-white hover:bg-gray-700 focus:ring-gray-500 dark:focus:ring-offset-gray-900',
  secondary: 'bg-gray-200 text-gray-800 hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 dark:focus:ring-gray-600',
  danger: 'bg-gray-900 text-white hover:bg-gray-900 focus:ring-gray-800 dark:focus:ring-offset-gray-900',
  outline: 'bg-transparent border border-gray-300 text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800',
  ghost: 'bg-transparent text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800',
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-base',
  lg: 'px-6 py-3 text-lg',
  icon: 'p-2',
};

const baseClasses = 'inline-flex items-center justify-center font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed';

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      loading = false,
      disabled = false,
      leftIcon,
      rightIcon,
      fullWidth = false,
      onClick,
      children,
      className,
      ...props
    },
    ref
  ) => {
    const isDisabled = disabled || loading;
    
    return (
      <button
        ref={ref}
        type="button"
        className={cn(
          baseClasses,
          variantClasses[variant],
          sizeClasses[size],
          fullWidth && 'w-full',
          className
        )}
        disabled={isDisabled}
        onClick={onClick}
        aria-busy={loading}
        {...props}
      >
        {loading && (
          <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
        )}
        {!loading && leftIcon && (
          <span className="mr-2" aria-hidden="true">{leftIcon}</span>
        )}
        <span className="truncate">{children}</span>
        {!loading && rightIcon && (
          <span className="ml-2" aria-hidden="true">{rightIcon}</span>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';

// 导出预定义的按钮组件
export const PrimaryButton = (props: Omit<ButtonProps, 'variant'>) => (
  <Button variant="primary" {...props} />
);

export const SecondaryButton = (props: Omit<ButtonProps, 'variant'>) => (
  <Button variant="secondary" {...props} />
);

export const DangerButton = (props: Omit<ButtonProps, 'variant'>) => (
  <Button variant="danger" {...props} />
);

export const OutlineButton = (props: Omit<ButtonProps, 'variant'>) => (
  <Button variant="outline" {...props} />
);

export const GhostButton = (props: Omit<ButtonProps, 'variant'>) => (
  <Button variant="ghost" {...props} />
);