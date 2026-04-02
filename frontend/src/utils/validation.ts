/**
 * 输入验证工具
 * 提供全面的表单和输入验证功能
 */

// 验证规则接口
export interface ValidationRule {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  pattern?: RegExp;
  patternMessage?: string;
  custom?: (value: any) => boolean | string;
  customMessage?: string;
  type?: 'email' | 'url' | 'phone' | 'username' | 'password';
}

// 验证结果接口
export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  fieldErrors: Record<string, string[]>;
}

// 邮箱格式验证
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// 密码强度验证
export const validatePasswordStrength = (password: string): {
  score: number; // 0-4
  strength: 'very-weak' | 'weak' | 'medium' | 'strong' | 'very-strong';
  suggestions: string[];
} => {
  const suggestions: string[] = [];
  let score = 0;

  // 长度检查
  if (password.length >= 8) score += 1;
  else suggestions.push('密码长度至少为8位');

  // 大写字母检查
  if (/[A-Z]/.test(password)) score += 1;
  else suggestions.push('包含至少一个大写字母');

  // 小写字母检查
  if (/[a-z]/.test(password)) score += 1;
  else suggestions.push('包含至少一个小写字母');

  // 数字检查
  if (/\d/.test(password)) score += 1;
  else suggestions.push('包含至少一个数字');

  // 特殊字符检查
  if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) score += 1;
  else suggestions.push('包含至少一个特殊字符 (!@#$%^&*等)');

  // 确定强度等级
  let strength: 'very-weak' | 'weak' | 'medium' | 'strong' | 'very-strong';
  if (score <= 1) strength = 'very-weak';
  else if (score === 2) strength = 'weak';
  else if (score === 3) strength = 'medium';
  else if (score === 4) strength = 'strong';
  else strength = 'very-strong';

  return { score, strength, suggestions };
};

// 用户名验证
export const validateUsername = (username: string): {
  isValid: boolean;
  errors: string[];
} => {
  const errors: string[] = [];

  if (!username.trim()) {
    errors.push('用户名不能为空');
    return { isValid: false, errors };
  }

  if (username.length < 3) {
    errors.push('用户名至少3个字符');
  }

  if (username.length > 50) {
    errors.push('用户名不能超过50个字符');
  }

  // 只允许字母、数字、下划线、连字符
  const usernameRegex = /^[a-zA-Z0-9_-]+$/;
  if (!usernameRegex.test(username)) {
    errors.push('用户名只能包含字母、数字、下划线(_)和连字符(-)');
  }

  // 不能以连字符或下划线开头或结尾
  if (/^[_-]/.test(username) || /[_-]$/.test(username)) {
    errors.push('用户名不能以下划线或连字符开头或结尾');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

// URL验证
export const validateUrl = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

// 电话号码验证（中国大陆）
export const validatePhoneCN = (phone: string): boolean => {
  const phoneRegex = /^1[3-9]\d{9}$/;
  return phoneRegex.test(phone);
};

// 身份证号验证（中国大陆）
export const validateIDCardCN = (idCard: string): boolean => {
  const idCardRegex = /^[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[1-2]\d|3[0-1])\d{3}(\d|X|x)$/;
  return idCardRegex.test(idCard);
};

// 文件验证
export const validateFile = (
  file: File, 
  options: {
    maxSize?: number; // 单位：字节
    allowedTypes?: string[];
  } = {}
): {
  isValid: boolean;
  errors: string[];
} => {
  const errors: string[] = [];
  const { maxSize = 10 * 1024 * 1024, allowedTypes = [] } = options;

  // 文件大小检查
  if (file.size > maxSize) {
    const maxSizeMB = (maxSize / (1024 * 1024)).toFixed(2);
    errors.push(`文件大小不能超过${maxSizeMB}MB`);
  }

  // 文件类型检查
  if (allowedTypes.length > 0) {
    const isAllowed = allowedTypes.some(type => {
      if (type.startsWith('.')) {
        return file.name.toLowerCase().endsWith(type.toLowerCase());
      }
      return file.type.startsWith(type);
    });
    
    if (!isAllowed) {
      errors.push(`不支持的文件类型。允许的类型: ${allowedTypes.join(', ')}`);
    }
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

// 表单字段验证
export const validateField = (
  value: any,
  rules: ValidationRule,
  fieldName: string = ''
): {
  isValid: boolean;
  errors: string[];
} => {
  const errors: string[] = [];

  // 处理空值
  const isEmpty = value === null || value === undefined || value === '' || (Array.isArray(value) && value.length === 0);

  // 必填检查
  if (rules.required && isEmpty) {
    errors.push(`${fieldName || '该字段'}不能为空`);
    return { isValid: false, errors };
  }

  // 如果字段为空且不是必填，直接返回成功
  if (isEmpty && !rules.required) {
    return { isValid: true, errors: [] };
  }

  // 类型检查
  if (rules.type) {
    switch (rules.type) {
      case 'email':
        if (!validateEmail(String(value))) {
          errors.push('请输入有效的邮箱地址');
        }
        break;
      case 'url':
        if (!validateUrl(String(value))) {
          errors.push('请输入有效的URL地址');
        }
        break;
      case 'phone':
        if (!validatePhoneCN(String(value))) {
          errors.push('请输入有效的手机号码');
        }
        break;
      case 'username':
        const usernameResult = validateUsername(String(value));
        if (!usernameResult.isValid) {
          errors.push(...usernameResult.errors);
        }
        break;
      case 'password':
        const passwordResult = validatePasswordStrength(String(value));
        if (passwordResult.score < 3) {
          errors.push('密码强度不足');
          errors.push(...passwordResult.suggestions);
        }
        break;
    }
  }

  // 长度检查
  if (typeof value === 'string') {
    if (rules.minLength !== undefined && value.length < rules.minLength) {
      errors.push(`${fieldName || '该字段'}至少需要${rules.minLength}个字符`);
    }
    if (rules.maxLength !== undefined && value.length > rules.maxLength) {
      errors.push(`${fieldName || '该字段'}不能超过${rules.maxLength}个字符`);
    }
  }

  // 模式检查
  if (rules.pattern && typeof value === 'string') {
    if (!rules.pattern.test(value)) {
      errors.push(rules.patternMessage || `${fieldName || '该字段'}格式不正确`);
    }
  }

  // 自定义验证
  if (rules.custom) {
    const customResult = rules.custom(value);
    if (customResult !== true) {
      errors.push(rules.customMessage || String(customResult) || `${fieldName || '该字段'}验证失败`);
    }
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

// 表单验证
export const validateForm = (
  formData: Record<string, any>,
  validationSchema: Record<string, ValidationRule[]>
): ValidationResult => {
  const errors: string[] = [];
  const fieldErrors: Record<string, string[]> = {};

  Object.entries(validationSchema).forEach(([fieldName, fieldRules]) => {
    const value = formData[fieldName];
    const fieldResult = validateField(value, fieldRules[0], fieldName); // 暂时只支持第一个规则
    
    if (!fieldResult.isValid) {
      errors.push(...fieldResult.errors);
      fieldErrors[fieldName] = fieldResult.errors;
    }
  });

  return {
    isValid: errors.length === 0,
    errors,
    fieldErrors
  };
};

// XSS防护：清理HTML
export const sanitizeHtml = (html: string): string => {
  // 简单的HTML清理，实际项目中应该使用专门的库如DOMPurify
  return html
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/on\w+=/gi, '');
};

// SQL注入防护检查
export const checkSqlInjection = (input: string): boolean => {
  const sqlKeywords = [
    'select', 'insert', 'update', 'delete', 'drop', 'truncate', 'create', 'alter',
    'union', 'join', 'where', 'having', 'group by', 'order by', 'limit', 'offset'
  ];
  
  const sqlPatterns = [
    /'?or'?\s*['"]?\d+['"]?\s*['"]?=[ '"]?\d+/i,
    /--/,
    /\/\*/,
    /\*\//,
    /;.*--/,
    /exec(\s|\()+.*/i,
    /xp_cmdshell/i
  ];

  const lowerInput = input.toLowerCase();
  
  // 检查SQL关键词
  for (const keyword of sqlKeywords) {
    if (lowerInput.includes(keyword) && 
        (lowerInput.includes('=') || lowerInput.includes('(') || lowerInput.includes(')') || lowerInput.includes(';'))) {
      return true;
    }
  }

  // 检查SQL模式
  for (const pattern of sqlPatterns) {
    if (pattern.test(input)) {
      return true;
    }
  }

  return false;
};

// 导出验证器对象
export const validators = {
  email: validateEmail,
  password: validatePasswordStrength,
  username: validateUsername,
  url: validateUrl,
  phone: validatePhoneCN,
  idCard: validateIDCardCN,
  file: validateFile,
  field: validateField,
  form: validateForm,
  sanitizeHtml,
  checkSqlInjection
};

export default validators;