import React from 'react';
import { cn } from '../../utils/cn';
import { Button } from './Button';
import { Loading } from './Loading';
import { AlertCircle, CheckCircle, XCircle, Info } from 'lucide-react';
import { UseFormValidationResult, validationRules, FieldValue } from '../../hooks/useFormValidation';

export interface FormField {
  /** 字段名称 */
  name: string;
  /** 字段标签 */
  label: string;
  /** 字段描述 */
  description?: string;
  /** 字段类型 */
  type?: 'text' | 'password' | 'email' | 'number' | 'textarea' | 'select' | 'checkbox' | 'radio';
  /** 字段组件 */
  component?: React.ReactNode;
  /** 是否必填 */
  required?: boolean;
  /** 实现 */
  实现?: string;
  /** 选项（用于select、radio、checkbox） */
  options?: Array<{ label: string; value: any }>;
  /** 验证规则 */
  rules?: any[];
  /** 是否禁用 */
  disabled?: boolean;
  /** 是否只读 */
  readonly?: boolean;
  /** 是否隐藏 */
  hidden?: boolean;
  /** 列跨度（用于网格布局） */
  colSpan?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
}

export interface FormProps extends React.FormHTMLAttributes<HTMLFormElement> {
  /** 表单标题 */
  title?: string;
  /** 表单描述 */
  description?: string;
  /** 表单字段配置 */
  fields: FormField[];
  /** 表单验证钩子返回的结果 */
  form: UseFormValidationResult;
  /** 提交按钮文字 */
  submitText?: string;
  /** 提交按钮变体 */
  submitVariant?: 'primary' | 'secondary' | 'danger' | 'outline' | 'ghost';
  /** 取消按钮文字 */
  cancelText?: string;
  /** 是否显示取消按钮 */
  showCancel?: boolean;
  /** 取消按钮点击回调 */
  onCancel?: () => void;
  /** 是否显示重置按钮 */
  showReset?: boolean;
  /** 重置按钮文字 */
  resetText?: string;
  /** 是否显示成功状态 */
  showSuccess?: boolean;
  /** 成功消息 */
  successMessage?: string;
  /** 是否显示错误摘要 */
  showErrorSummary?: boolean;
  /** 布局类型 */
  layout?: 'vertical' | 'horizontal' | 'inline' | 'grid';
  /** 网格列数 */
  gridCols?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
  /** 字段间距 */
  fieldSpacing?: 'none' | 'sm' | 'md' | 'lg';
  /** 是否紧凑模式 */
  compact?: boolean;
  /** 自定义提交处理 */
  onSubmit?: (e: React.FormEvent) => void;
  /** 子元素 */
  children?: React.ReactNode;
}

const fieldSpacingClasses = {
  none: 'space-y-0',
  sm: 'space-y-2',
  md: 'space-y-4',
  lg: 'space-y-6',
};

const gridColsClasses = {
  1: 'grid-cols-1',
  2: 'grid-cols-1 sm:grid-cols-2',
  3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
  4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  5: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5',
  6: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6',
  7: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-7',
  8: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8',
  9: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 2xl:grid-cols-9',
  10: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 xl:grid-cols-10',
  11: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 2xl:grid-cols-11',
  12: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-6 3xl:grid-cols-12',
};

export const Form: React.FC<FormProps> = ({
  title,
  description,
  fields,
  form,
  submitText = '提交',
  submitVariant = 'primary',
  cancelText = '取消',
  showCancel = false,
  onCancel,
  showReset = false,
  resetText = '重置',
  showSuccess = false,
  successMessage = '提交成功！',
  showErrorSummary = true,
  layout = 'vertical',
  gridCols = 1,
  fieldSpacing = 'md',
  compact = false,
  onSubmit,
  className,
  children,
  ...props
}) => {
  const {
    values,
    errors,
    touched,
    isSubmitting,
    isValid,
    handleChange,
    handleBlur,
    handleSubmit,
    resetForm,
    getFieldState,
  } = form;

  // 处理表单提交
  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (onSubmit) {
      onSubmit(e);
    } else {
      await handleSubmit(e);
    }
  };

  // 计算表单错误总数
  const errorCount = Object.values(errors).reduce(
    (total, fieldErrors) => total + (fieldErrors?.length || 0),
    0
  );

  // 是否有触摸过的字段
  const hasTouchedFields = Object.values(touched).some(Boolean);

  // 渲染字段
  const renderField = (field: FormField) => {
    if (field.hidden) {
      return null;
    }

    const fieldState = getFieldState(field.name);
    const hasError = fieldState.errors.length > 0;
    const isTouched = fieldState.touched;
    const showError = hasError && isTouched;

    const fieldClasses = cn(
      layout === 'grid' && `col-span-${field.colSpan || 1}`,
      field.colSpan && `sm:col-span-${field.colSpan}`
    );

    return (
      <div key={field.name} className={fieldClasses}>
        <div className="space-y-2">
          {field.label && (
            <label
              htmlFor={field.name}
              className={cn(
                'block text-sm font-medium',
                hasError ? 'text-gray-900 dark:text-gray-500' : 'text-gray-700 dark:text-gray-300'
              )}
            >
              {field.label}
              {field.required && <span className="text-gray-800 ml-1">*</span>}
            </label>
          )}

          {field.description && !showError && (
            <p className="text-sm text-gray-500 dark:text-gray-400">{field.description}</p>
          )}

          {/* 字段组件 */}
          {field.component ? (
            React.cloneElement(field.component as React.ReactElement, {
              id: field.name,
              name: field.name,
              value: values[field.name] || '',
              onChange: (e: any) => {
                const value = e.target?.value ?? e;
                handleChange(field.name, value);
              },
              onBlur: () => handleBlur(field.name),
              disabled: field.disabled || isSubmitting,
              readOnly: field.readonly,
              'aria-invalid': hasError,
              'aria-describedby': showError
                ? `${field.name}-error`
                : field.description
                ? `${field.name}-description`
                : undefined,
            })
          ) : (
            <div>
              {renderFieldByType(field, fieldState, handleChange, handleBlur, isSubmitting)}
            </div>
          )}

          {/* 错误信息 */}
          {showError && (
            <div id={`${field.name}-error`} className="flex items-center text-sm text-gray-900 dark:text-gray-500">
              <AlertCircle className="h-4 w-4 mr-1 flex-shrink-0" />
              <span>{fieldState.errors[0]}</span>
            </div>
          )}

          {/* 成功状态 */}
          {!hasError && isTouched && fieldState.value && showSuccess && (
            <div className="flex items-center text-sm text-gray-700 dark:text-gray-400">
              <CheckCircle className="h-4 w-4 mr-1 flex-shrink-0" />
              <span>验证通过</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  // 根据字段类型渲染输入组件
  const renderFieldByType = (
    field: FormField,
    fieldState: any,
    onChange: (name: string, value: any) => void,
    onBlur: (name: string) => void,
    disabled: boolean
  ) => {
    // 安全获取字符串值
    const getStringValue = (val: FieldValue): string => {
      if (val == null) return '';
      if (typeof val === 'string') return val;
      if (typeof val === 'number') return val.toString();
      if (typeof val === 'boolean') return val ? 'true' : 'false';
      if (Array.isArray(val)) return JSON.stringify(val);
      if (val instanceof File || val instanceof FileList) return '';
      if (typeof val === 'object' && 'length' in val) return '';
      return String(val);
    };

    // 安全获取布尔值
    const getBooleanValue = (val: FieldValue): boolean => {
      if (typeof val === 'boolean') return val;
      if (typeof val === 'string') return val === 'true' || val === '1';
      if (typeof val === 'number') return val !== 0;
      return false;
    };

    const commonProps = {
      id: field.name,
      name: field.name,
      value: getStringValue(values[field.name]),
      onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
        const value = e.target.value;
        onChange(field.name, value);
      },
      onBlur: () => onBlur(field.name),
      disabled: field.disabled || disabled,
      readOnly: field.readonly,
      className: cn(
        'w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:border-transparent transition-colors',
        fieldState.errors.length > 0 && fieldState.touched
          ? 'border-gray-800 focus:ring-gray-800 dark:border-gray-900'
          : 'border-gray-300 focus:ring-gray-500 dark:border-gray-600 dark:focus:ring-gray-400',
        field.disabled || disabled ? 'opacity-50 cursor-not-allowed' : '',
        compact ? 'py-1.5 text-sm' : ''
      ),
      'aria-invalid': fieldState.errors.length > 0,
    };

    switch (field.type) {
      case 'textarea':
        return (
          <textarea
            {...commonProps}
            rows={4}
            placeholder={field.实现}
            className={cn(commonProps.className, 'resize-y')}
          />
        );

      case 'select':
        return (
          <select {...commonProps}>
            <option value="">{field.实现 || '请选择...'}</option>
            {field.options?.map((option) => (
              <option key={option.value} value={String(option.value)}>
                {option.label}
              </option>
            ))}
          </select>
        );

      case 'checkbox':
        return (
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              {...commonProps}
              checked={getBooleanValue(values[field.name])}
              onChange={(e) => onChange(field.name, e.target.checked)}
              className={cn(
                'h-4 w-4 text-gray-600 focus:ring-gray-500 border-gray-300 rounded',
                field.disabled || disabled ? 'opacity-50 cursor-not-allowed' : ''
              )}
            />
            {field.label && (
              <label
                htmlFor={field.name}
                className="text-sm text-gray-700 dark:text-gray-300"
              >
                {field.label}
              </label>
            )}
          </div>
        );

      case 'radio':
        return (
          <div className="space-y-2">
            {field.options?.map((option) => {
              // 安全比较值
              const currentValue = values[field.name];
              const optionValue = option.value;
              const isChecked = (() => {
                if (currentValue == null && optionValue == null) return true;
                if (currentValue == null || optionValue == null) return false;
                return getStringValue(currentValue) === String(optionValue);
              })();
              
              return (
                <div key={option.value} className="flex items-center">
                  <input
                    type="radio"
                    id={`${field.name}-${option.value}`}
                    name={field.name}
                    value={String(optionValue)}
                    checked={isChecked}
                    onChange={(e) => onChange(field.name, e.target.value)}
                    onBlur={() => onBlur(field.name)}
                    disabled={field.disabled || disabled}
                    className="h-4 w-4 text-gray-600 focus:ring-gray-500 border-gray-300"
                  />
                  <label
                    htmlFor={`${field.name}-${option.value}`}
                    className="ml-2 text-sm text-gray-700 dark:text-gray-300"
                  >
                    {option.label}
                  </label>
                </div>
              );
            })}
          </div>
        );

      default:
        return (
          <input
            type={field.type || 'text'}
            {...commonProps}
            placeholder={field.实现}
          />
        );
    }
  };

  return (
    <form
      onSubmit={handleFormSubmit}
      className={cn(
        layout === 'grid' && 'grid gap-4',
        layout !== 'grid' && fieldSpacingClasses[fieldSpacing],
        layout === 'grid' && gridColsClasses[gridCols],
        className
      )}
      noValidate
      {...props}
    >
      {/* 表单标题和描述 */}
      {(title || description) && (
        <div className={layout === 'grid' ? 'col-span-full' : ''}>
          {title && (
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              {title}
            </h2>
          )}
          {description && (
            <p className="text-gray-600 dark:text-gray-400">{description}</p>
          )}
        </div>
      )}

      {/* 错误摘要 */}
      {showErrorSummary && errorCount > 0 && hasTouchedFields && (
        <div className={cn(
          'p-4 rounded-lg border border-gray-700 dark:border-gray-900 bg-gray-900 dark:bg-gray-900/20',
          layout === 'grid' && 'col-span-full'
        )}>
          <div className="flex items-start">
            <XCircle className="h-5 w-5 text-gray-800 mt-0.5 mr-2 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-sm font-medium text-gray-900 dark:text-gray-600">
                表单中存在 {errorCount} 个错误需要修正：
              </h3>
              <ul className="mt-2 text-sm text-gray-900 dark:text-gray-500 space-y-1">
                {Object.entries(errors).map(([fieldName, fieldErrors]) => {
                  if (!fieldErrors?.length) return null;
                  const field = fields.find(f => f.name === fieldName);
                  return (
                    <li key={fieldName}>
                      <span className="font-medium">{field?.label || fieldName}</span>：{fieldErrors[0]}
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* 成功消息 */}
      {showSuccess && isValid && hasTouchedFields && !isSubmitting && (
        <div className={cn(
          'p-4 rounded-lg border border-gray-500 dark:border-gray-800 bg-gray-700 dark:bg-gray-900/20',
          layout === 'grid' && 'col-span-full'
        )}>
          <div className="flex items-center">
            <CheckCircle className="h-5 w-5 text-gray-600 mr-2" />
            <p className="text-gray-800 dark:text-gray-400">{successMessage}</p>
          </div>
        </div>
      )}

      {/* 表单字段 */}
      {fields.map(renderField)}

      {/* 自定义子元素 */}
      {children && (
        <div className={layout === 'grid' ? 'col-span-full' : ''}>
          {children}
        </div>
      )}

      {/* 表单按钮 */}
      <div className={cn(
        'flex items-center space-x-3',
        layout === 'grid' && 'col-span-full',
        layout === 'inline' ? 'inline-flex' : ''
      )}>
        {/* 提交按钮 */}
        <Button
          type="submit"
          variant={submitVariant}
          loading={isSubmitting}
          disabled={isSubmitting}
          className={compact ? 'px-3 py-1.5 text-sm' : ''}
        >
          {submitText}
        </Button>

        {/* 取消按钮 */}
        {showCancel && (
          <Button
            type="button"
            variant="secondary"
            onClick={onCancel}
            disabled={isSubmitting}
            className={compact ? 'px-3 py-1.5 text-sm' : ''}
          >
            {cancelText}
          </Button>
        )}

        {/* 重置按钮 */}
        {showReset && (
          <Button
            type="button"
            variant="ghost"
            onClick={resetForm}
            disabled={isSubmitting}
            className={compact ? 'px-3 py-1.5 text-sm' : ''}
          >
            {resetText}
          </Button>
        )}

        {/* 加载状态 */}
        {isSubmitting && (
          <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
            <Loading type="dots" size="sm" variant="secondary" className="mr-2" />
            <span>正在处理...</span>
          </div>
        )}
      </div>

      {/* 表单状态提示 */}
      {!isValid && hasTouchedFields && (
        <div className={cn(
          'flex items-center text-sm text-amber-600 dark:text-amber-400',
          layout === 'grid' && 'col-span-full'
        )}>
          <Info className="h-4 w-4 mr-1 flex-shrink-0" />
          <span>请修正表单中的错误后重新提交</span>
        </div>
      )}
    </form>
  );
};

Form.displayName = 'Form';

// 预定义的表单组件
export const VerticalForm = (props: Omit<FormProps, 'layout'>) => (
  <Form layout="vertical" {...props} />
);

export const HorizontalForm = (props: Omit<FormProps, 'layout'>) => (
  <Form layout="horizontal" {...props} />
);

export const InlineForm = (props: Omit<FormProps, 'layout'>) => (
  <Form layout="inline" {...props} />
);

export const GridForm = (props: Omit<FormProps, 'layout'>) => (
  <Form layout="grid" {...props} />
);

export const CompactForm = (props: Omit<FormProps, 'compact'>) => (
  <Form compact {...props} />
);

// 快速创建表单字段
export const createField = (
  name: string,
  label: string,
  options?: Partial<FormField>
): FormField => ({
  name,
  label,
  type: 'text',
  required: false,
  ...options,
});

// 表单字段构建器
export const fieldBuilder = {
  text: (name: string, label: string, options?: Partial<FormField>) =>
    createField(name, label, { type: 'text', ...options }),
  
  email: (name: string, label: string, options?: Partial<FormField>) =>
    createField(name, label, { type: 'email', rules: [validationRules.email()], ...options }),
  
  password: (name: string, label: string, options?: Partial<FormField>) =>
    createField(name, label, { type: 'password', rules: [validationRules.password()], ...options }),
  
  textarea: (name: string, label: string, options?: Partial<FormField>) =>
    createField(name, label, { type: 'textarea', ...options }),
  
  select: (name: string, label: string, options: Array<{ label: string; value: any }>, formOptions?: Partial<FormField>) =>
    createField(name, label, { type: 'select', options, ...formOptions }),
  
  checkbox: (name: string, label: string, options?: Partial<FormField>) =>
    createField(name, label, { type: 'checkbox', ...options }),
  
  radio: (name: string, label: string, options: Array<{ label: string; value: any }>, formOptions?: Partial<FormField>) =>
    createField(name, label, { type: 'radio', options, ...formOptions }),
};