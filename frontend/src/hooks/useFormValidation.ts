/**
 * 表单验证钩子
 * 提供表单验证、状态管理和错误处理功能
 */

import { useState, useCallback } from 'react';
import { validateEmail, validatePasswordStrength, ValidationRule } from '../utils/validation';

export type FieldValue = string | number | boolean | File | FileList | Array<unknown> | { length: number } | null | undefined;

export interface FormField {
  name: string;
  value: FieldValue;
  rules?: ValidationRule[];
  required?: boolean;
  label?: string;
  type?: string;
}

export interface FormValidationState {
  [fieldName: string]: {
    value: FieldValue;
    errors: string[];
    touched: boolean;
    dirty: boolean;
    isValid: boolean;
  };
}

export interface UseFormValidationOptions {
  initialValues: Record<string, FieldValue>;
  validationRules?: Record<string, ValidationRule[]>;
  onSubmit: (values: Record<string, FieldValue>, formState: FormValidationState) => Promise<void> | void;
  onError?: (errors: Record<string, string[]>) => void;
  validateOnChange?: boolean;
  validateOnBlur?: boolean;
  validateOnSubmit?: boolean;
}

export interface UseFormValidationResult {
  // 表单状态
  values: Record<string, FieldValue>;
  errors: Record<string, string[]>;
  touched: Record<string, boolean>;
  dirty: Record<string, boolean>;
  isValid: boolean;
  isSubmitting: boolean;
  submitCount: number;
  
  // 表单操作
  handleChange: (name: string, value: FieldValue) => void;
  handleBlur: (name: string) => void;
  handleSubmit: (e?: React.FormEvent) => Promise<void>;
  resetForm: () => void;
  setFieldValue: (name: string, value: FieldValue) => void;
  setFieldError: (name: string, errors: string[]) => void;
  validateField: (name: string) => string[];
  validateForm: () => boolean;
  
  // 字段状态获取
  getFieldState: (name: string) => {
    value: FieldValue;
    errors: string[];
    touched: boolean;
    dirty: boolean;
    isValid: boolean;
  };
}

/**
 * 表单验证钩子
 */
export function useFormValidation({
  initialValues,
  validationRules = {},
  onSubmit,
  onError,
  validateOnChange = true,
  validateOnBlur = true,
  validateOnSubmit = true,
}: UseFormValidationOptions): UseFormValidationResult {
  // 初始化表单状态
  const [values, setValues] = useState<Record<string, FieldValue>>(initialValues);
  const [errors, setErrors] = useState<Record<string, string[]>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [dirty, setDirty] = useState<Record<string, boolean>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitCount, setSubmitCount] = useState(0);

  // 验证单个字段
  const validateField = useCallback((name: string): string[] => {
    const value = values[name];
    const rules = validationRules[name] || [];
    const fieldErrors: string[] = [];

    // 如果没有验证规则，返回空数组
    if (rules.length === 0) {
      return fieldErrors;
    }

    // 类型检查辅助函数
    const isString = (val: FieldValue): val is string => typeof val === 'string';
    const isArray = (val: FieldValue): val is Array<unknown> => Array.isArray(val);
    const hasLength = (val: FieldValue): val is { length: number } => 
      isString(val) || isArray(val) || (val instanceof FileList);

    // 应用验证规则
    for (const rule of rules) {
      // 必填验证
      const isEmpty = () => {
        if (!value) return true;
        if (isString(value) && value.trim() === '') return true;
        if (isArray(value) && value.length === 0) return true;
        if (value instanceof FileList && value.length === 0) return true;
        return false;
      };

      if (rule.required && isEmpty()) {
        fieldErrors.push(`${rule.customMessage || '此字段为必填项'}`);
        continue;
      }

      // 最小长度验证（仅适用于有长度的值）
      if (rule.minLength !== undefined && value && hasLength(value) && value.length < rule.minLength) {
        fieldErrors.push(`${rule.customMessage || `长度不能少于${rule.minLength}个字符`}`);
      }

      // 最大长度验证（仅适用于有长度的值）
      if (rule.maxLength !== undefined && value && hasLength(value) && value.length > rule.maxLength) {
        fieldErrors.push(`${rule.customMessage || `长度不能超过${rule.maxLength}个字符`}`);
      }

      // 正则表达式验证（仅适用于字符串）
      if (rule.pattern && value && isString(value) && !rule.pattern.test(value)) {
        fieldErrors.push(rule.patternMessage || '格式不正确');
      }

      // 类型验证
      if (rule.type) {
        switch (rule.type) {
          case 'email':
            if (value && isString(value) && !validateEmail(value)) {
              fieldErrors.push('请输入有效的邮箱地址');
            }
            break;
          case 'password':
            if (value && isString(value)) {
              const strength = validatePasswordStrength(value);
              if (strength.score < 2) {
                fieldErrors.push('密码强度不足');
              }
            }
            break;
          case 'username':
            if (value && isString(value) && !/^[a-zA-Z0-9_]{3,20}$/.test(value)) {
              fieldErrors.push('用户名只能包含字母、数字和下划线，长度3-20位');
            }
            break;
          case 'phone':
            if (value && isString(value) && !/^1[3-9]\d{9}$/.test(value)) {
              fieldErrors.push('请输入有效的手机号码');
            }
            break;
        }
      }

      // 自定义验证
      if (rule.custom) {
        const customResult = rule.custom(value);
        if (typeof customResult === 'string') {
          fieldErrors.push(customResult);
        } else if (customResult === false) {
          fieldErrors.push(rule.customMessage || '验证失败');
        }
      }
    }

    return fieldErrors;
  }, [values, validationRules]);

  // 验证整个表单
  const validateForm = useCallback((): boolean => {
    const newErrors: Record<string, string[]> = {};
    let isValid = true;

    // 验证所有字段
    Object.keys(validationRules).forEach((name) => {
      const fieldErrors = validateField(name);
      if (fieldErrors.length > 0) {
        newErrors[name] = fieldErrors;
        isValid = false;
      }
    });

    // 验证有值的字段（即使没有规则）
    Object.keys(values).forEach((name) => {
      if (!validationRules[name] && values[name]) {
        // 如果没有规则但有值，确保字段在errors对象中为空数组
        if (!newErrors[name]) {
          newErrors[name] = [];
        }
      }
    });

    setErrors(newErrors);
    return isValid;
  }, [values, validationRules, validateField]);

  // 处理字段变化
  const handleChange = useCallback((name: string, value: FieldValue) => {
    setValues((prev) => ({ ...prev, [name]: value }));
    setDirty((prev) => ({ ...prev, [name]: true }));

    // 实时验证
    if (validateOnChange) {
      const fieldErrors = validateField(name);
      setErrors((prev) => ({ ...prev, [name]: fieldErrors }));
    }
  }, [validateOnChange, validateField]);

  // 处理字段失去焦点
  const handleBlur = useCallback((name: string) => {
    setTouched((prev) => ({ ...prev, [name]: true }));

    // 失焦验证
    if (validateOnBlur) {
      const fieldErrors = validateField(name);
      setErrors((prev) => ({ ...prev, [name]: fieldErrors }));
    }
  }, [validateOnBlur, validateField]);

  // 处理表单提交
  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    if (e) {
      e.preventDefault();
    }

    setIsSubmitting(true);
    setSubmitCount((prev) => prev + 1);

    try {
      // 标记所有字段为已触摸
      const allTouched: Record<string, boolean> = {};
      Object.keys(values).forEach((name) => {
        allTouched[name] = true;
      });
      setTouched(allTouched);

      // 表单验证
      let isValid = true;
      if (validateOnSubmit) {
        isValid = validateForm();
      }

      if (!isValid) {
        if (onError) {
          onError(errors);
        }
        return;
      }

      // 执行提交回调
      await onSubmit(values, getFormState());
    } catch (error) {
      console.error('表单提交失败:', error);
      if (onError) {
        onError(errors);
      }
    } finally {
      setIsSubmitting(false);
    }
  }, [
    values,
    errors,
    validateOnSubmit,
    validateForm,
    onSubmit,
    onError,
  ]);

  // 重置表单
  const resetForm = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
    setDirty({});
    setIsSubmitting(false);
  }, [initialValues]);

  // 设置字段值
  const setFieldValue = useCallback((name: string, value: FieldValue) => {
    handleChange(name, value);
  }, [handleChange]);

  // 设置字段错误
  const setFieldError = useCallback((name: string, errorMessages: string[]) => {
    setErrors((prev) => ({ ...prev, [name]: errorMessages }));
  }, []);

  // 获取表单状态
  const getFormState = useCallback((): FormValidationState => {
    const state: FormValidationState = {};
    
    Object.keys(values).forEach((name) => {
      state[name] = {
        value: values[name],
        errors: errors[name] || [],
        touched: touched[name] || false,
        dirty: dirty[name] || false,
        isValid: !errors[name] || errors[name].length === 0,
      };
    });
    
    return state;
  }, [values, errors, touched, dirty]);

  // 获取字段状态
  const getFieldState = useCallback((name: string) => {
    return {
      value: values[name],
      errors: errors[name] || [],
      touched: touched[name] || false,
      dirty: dirty[name] || false,
      isValid: !errors[name] || errors[name].length === 0,
    };
  }, [values, errors, touched, dirty]);

  // 计算表单整体有效性
  const isValid = Object.keys(errors).every((name) => errors[name].length === 0);

  return {
    // 状态
    values,
    errors,
    touched,
    dirty,
    isValid,
    isSubmitting,
    submitCount,
    
    // 操作
    handleChange,
    handleBlur,
    handleSubmit,
    resetForm,
    setFieldValue,
    setFieldError,
    validateField,
    validateForm,
    
    // 工具
    getFieldState,
  };
}

/**
 * 简化版表单验证钩子
 */
export function useForm<T extends Record<string, FieldValue>>(
  initialValues: T,
  onSubmit: (values: T) => Promise<void> | void
) {
  const {
    values,
    errors,
    touched,
    dirty,
    isValid,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    resetForm,
    setFieldValue,
    getFieldState,
  } = useFormValidation({
    initialValues,
    onSubmit: (values) => onSubmit(values as T),
  });

  return {
    values: values as T,
    errors,
    touched,
    dirty,
    isValid,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    resetForm,
    setFieldValue,
    getFieldState,
  };
}

/**
 * 快速创建表单验证规则
 */
export const validationRules = {
  required: (message?: string): ValidationRule => ({
    required: true,
    customMessage: message || '此字段为必填项',
  }),
  
  minLength: (length: number, message?: string): ValidationRule => ({
    minLength: length,
    customMessage: message || `长度不能少于${length}个字符`,
  }),
  
  maxLength: (length: number, message?: string): ValidationRule => ({
    maxLength: length,
    customMessage: message || `长度不能超过${length}个字符`,
  }),
  
  email: (message?: string): ValidationRule => ({
    type: 'email',
    customMessage: message || '请输入有效的邮箱地址',
  }),
  
  password: (message?: string): ValidationRule => ({
    type: 'password',
    customMessage: message || '密码强度不足',
  }),
  
  username: (message?: string): ValidationRule => ({
    type: 'username',
    customMessage: message || '用户名只能包含字母、数字和下划线，长度3-20位',
  }),
  
  phone: (message?: string): ValidationRule => ({
    type: 'phone',
    customMessage: message || '请输入有效的手机号码',
  }),
  
  pattern: (pattern: RegExp, message?: string): ValidationRule => ({
    pattern,
    patternMessage: message || '格式不正确',
  }),
};