export interface User {
  id: string;
  username: string;
  email: string;
  name?: string;
  full_name?: string;
  avatar?: string;
  role?: 'user' | 'admin' | 'developer';
  is_admin?: boolean;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_login?: string;
  subscription?: Subscription;
  two_factor_enabled?: boolean;
  two_factor_method?: 'email' | 'totp' | null;
  permissions: string[];
}

export interface Subscription {
  id: string;
  plan: string;
  status: 'active' | 'inactive' | 'expired' | 'canceled';
  start_date: string;
  end_date?: string;
  max_api_calls: number;
  api_calls_used: number;
  max_storage: number;
  storage_used: number;
}

export interface ApiKey {
  id: string;
  name: string;
  key: string;
  prefix: string;
  created_at: string;
  last_used?: string;
  is_active: boolean;
  rate_limit?: number;
  expires_at?: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
  full_name?: string;
}

export interface AuthResponse {
  access_token?: string;
  refresh_token?: string;
  token_type?: string;
  expires_in?: number;
  user?: User;
  requires_2fa?: boolean;
  temp_token?: string;
  method?: 'email' | 'totp';
  message?: string;
  timestamp?: string;
}

// 注意：ApiResponse和PaginatedResponse已移至api.ts文件中定义
// 请使用从'../types/api'导入的类型

// 双因素认证相关类型
export interface TwoFactorAuthResponse {
  requires_2fa: boolean;
  temp_token?: string;
  method?: 'email' | 'totp';
  message?: string;
  timestamp?: string;
}

export interface TwoFactorLoginRequest {
  username_or_email: string;
  password: string;
  code: string;
}

export interface TwoFactorSetupRequest {
  method: 'email' | 'totp';
}

export interface TwoFactorVerifyRequest {
  code: string;
}

export interface TwoFactorStatus {
  enabled: boolean;
  method: 'email' | 'totp' | null;
}

export interface TwoFactorBackupCodes {
  codes: string[];
}