import React, { forwardRef } from 'react';
import { cn } from '../../utils/cn';
import { AlertCircle, Check, Eye, EyeOff, Search } from 'lucide-react';

export type InputVariant = 'default' | 'search' | 'password' | 'textarea';
export type InputSize = 'sm' | 'md' | 'lg';

export interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement | HTMLTextAreaElement>, 'size'> {
  /** 输入框变体 */
  variant?: InputVariant;
  /** 输入框尺寸 */
  size?: InputSize;
  /** 标签文字 */
  label?: string;
  /** 描述文字 */
  description?: string;
  /** 错误信息 */
  error?: string;
  /** 成功状态 */
  success?: boolean;
  /** 左侧图标 */
  leftIcon?: React.ReactNode;
  /** 右侧图标 */
  rightIcon?: React.ReactNode;
  /** 是否全宽 */
  fullWidth?: boolean;
  /** 输入框引用 */
  ref?: React.Ref<HTMLInputElement | HTMLTextAreaElement>;
}

const sizeClasses: Record<InputSize, string> = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-3 py-2 text-base',
  lg: 'px-4 py-3 text-lg',
};

const baseClasses = 'w-full border rounded-lg focus:outline-none focus:ring-2 focus:border-transparent transition-colors disabled:opacity-50 disabled:cursor-not-allowed dark:bg-gray-800 dark:text-gray-100';

export const Input = forwardRef<HTMLInputElement | HTMLTextAreaElement, InputProps>(
  (
    {
      variant = 'default',
      size = 'md',
      label,
      description,
      error,
      success,
      leftIcon,
      rightIcon,
      fullWidth = false,
      className,
      type = 'text',
      ...props
    },
    ref
  ) => {
    const [showPassword, setShowPassword] = React.useState(false);
    const isPassword = variant === 'password';
    const isTextarea = variant === 'textarea';
    const isSearch = variant === 'search';
    
    const inputType = isPassword ? (showPassword ? 'text' : 'password') : type;
    
    const inputClasses = cn(
      baseClasses,
      sizeClasses[size],
      error 
        ? 'border-gray-800 focus:ring-gray-800 dark:border-gray-900' 
        : success
        ? 'border-gray-600 focus:ring-gray-600 dark:border-gray-700'
        : 'border-gray-300 focus:ring-gray-500 dark:border-gray-600 dark:focus:ring-gray-400',
      leftIcon && 'pl-10',
      (rightIcon || isPassword || isSearch) && 'pr-10',
      fullWidth && 'w-full',
      className
    );
    
    // 左侧图标渲染逻辑
    const renderLeftIcon = () => {
      if (isSearch) {
        return <Search className="h-5 w-5" />;
      }
      if (leftIcon) {
        // 如果leftIcon是有效的React元素
        if (React.isValidElement(leftIcon)) {
          return React.cloneElement(leftIcon as React.ReactElement, { className: 'h-5 w-5' });
        }
        // 否则直接渲染
        return <div className="h-5 w-5 flex items-center justify-center">{leftIcon}</div>;
      }
      return null;
    };
    
    const togglePasswordVisibility = () => {
      setShowPassword(!showPassword);
    };
    
    const renderInput = () => {
      if (isTextarea) {
        return (
          <textarea
            ref={ref as React.Ref<HTMLTextAreaElement>}
            className={cn(inputClasses, 'resize-y min-h-[80px]')}
            rows={4}
            {...props as React.TextareaHTMLAttributes<HTMLTextAreaElement>}
          />
        );
      }
      
      return (
        <input
          ref={ref as React.Ref<HTMLInputElement>}
          type={inputType}
          className={inputClasses}
          {...props as React.InputHTMLAttributes<HTMLInputElement>}
        />
      );
    };
    
    return (
      <div className={cn('space-y-1', fullWidth && 'w-full')}>
        {label && (
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
            {props.required && <span className="text-gray-800 ml-1">*</span>}
          </label>
        )}
        
        <div className="relative">
          {(isSearch || leftIcon) && (
            <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500">
              {renderLeftIcon()}
            </div>
          )}
          
          {renderInput()}
          
          {(rightIcon || isPassword || (isSearch && !rightIcon)) && (
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
              {isPassword ? (
                <button
                  type="button"
                  onClick={togglePasswordVisibility}
                  className="text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 focus:outline-none"
                  aria-label={showPassword ? '隐藏密码' : '显示密码'}
                >
                  {showPassword ? (
                    <EyeOff className="h-5 w-5" />
                  ) : (
                    <Eye className="h-5 w-5" />
                  )}
                </button>
              ) : isSearch && !rightIcon ? (
                <Search className="h-5 w-5 text-gray-400 dark:text-gray-500" />
              ) : (
                <div className="text-gray-400 dark:text-gray-500">{rightIcon}</div>
              )}
            </div>
          )}
          
          {success && (
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-600">
              <Check className="h-5 w-5" />
            </div>
          )}
        </div>
        
        {description && !error && (
          <p className="text-sm text-gray-500 dark:text-gray-400">{description}</p>
        )}
        
        {error && (
          <div className="flex items-center text-sm text-gray-900 dark:text-gray-500">
            <AlertCircle className="h-4 w-4 mr-1 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

// 导出预定义的输入框组件
export const TextInput = (props: Omit<InputProps, 'variant'>) => (
  <Input variant="default" {...props} />
);

export const SearchInput = (props: Omit<InputProps, 'variant'>) => (
  <Input variant="search" {...props} />
);

export const PasswordInput = (props: Omit<InputProps, 'variant'>) => (
  <Input variant="password" {...props} />
);

export const Textarea = (props: Omit<InputProps, 'variant'>) => (
  <Input variant="textarea" {...props} />
);