import { apiClient } from './client';
import { ApiResponse, PaginatedResponse } from '../../types/api';
import {
  User,
  ApiKey,
  LoginCredentials,
  RegisterData,
  AuthResponse,
  TwoFactorLoginRequest,
  TwoFactorSetupRequest,
  TwoFactorVerifyRequest,
  TwoFactorStatus,
  TwoFactorBackupCodes,
} from '../../types/auth';

export const authService = {
  // 用户认证
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const response = await apiClient.post('/auth/login', credentials);
    
    // 检查是否为SuccessResponse格式（包含success, data, message字段）
    let authData: any = response;
    if (response && typeof response === 'object' && 'data' in response) {
      // 如果是SuccessResponse格式，提取data字段
      authData = response.data;
    }
    
    // 检查是否需要2FA验证
    if (authData.requires_2fa) {
      return {
        ...authData,
        requires_2fa: true,
      };
    }
    
    // 转换用户数据（仅在成功登录时）
    if (!authData.user) {
      // 尝试从响应中提取用户信息
      if (authData.access_token && authData.refresh_token) {
        // 如果有token但没有user对象，尝试获取当前用户
        console.warn('登录响应缺少用户数据，但包含token');
        return {
          ...authData,
          user: {
            id: '',
            username: credentials.username,
            email: '',
            full_name: credentials.username,
            is_admin: false,
            is_active: true,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            role: 'user',
            permissions: [],
          } as User,
        };
      }
      throw new Error('登录响应中缺少用户数据');
    }
    
    return {
      ...authData,
      user: {
        ...authData.user,
        role: authData.user.is_admin ? 'admin' : 'user',
      },
    };
  },

  // 双因素认证
  async loginWith2FA(loginData: TwoFactorLoginRequest): Promise<AuthResponse> {
    const response = await apiClient.post('/auth/2fa/login', loginData);
    
    // 检查是否为SuccessResponse格式（包含success, data, message字段）
    let authData: any = response;
    if (response && typeof response === 'object' && 'data' in response) {
      // 如果是SuccessResponse格式，提取data字段
      authData = response.data;
    }
    
    if (!authData.user) {
      throw new Error('2FA登录响应中缺少用户数据');
    }
    return {
      ...authData,
      user: {
        ...authData.user,
        role: authData.user.is_admin ? 'admin' : 'user',
      },
    };
  },

  async get2FAStatus(): Promise<TwoFactorStatus> {
    return apiClient.get('/auth/2fa/status');
  },

  async setup2FA(setupData: TwoFactorSetupRequest): Promise<any> {
    return apiClient.post('/auth/2fa/setup', setupData);
  },

  async verify2FASetup(verifyData: TwoFactorVerifyRequest): Promise<any> {
    return apiClient.post('/auth/2fa/verify', verifyData);
  },

  async disable2FA(disableData: TwoFactorVerifyRequest): Promise<any> {
    return apiClient.post('/auth/2fa/disable', disableData);
  },

  async get2FABackupCodes(): Promise<TwoFactorBackupCodes> {
    return apiClient.get('/auth/2fa/backup-codes');
  },

  async register(data: RegisterData): Promise<ApiResponse> {
    return apiClient.post<ApiResponse>('/auth/register', data);
  },

  async logout(): Promise<ApiResponse> {
    return apiClient.post<ApiResponse>('/auth/logout');
  },

  async refreshToken(refreshToken: string): Promise<AuthResponse> {
    const response = await apiClient.post('/auth/refresh', { refresh_token: refreshToken });
    
    // 检查是否为SuccessResponse格式（包含success, data, message字段）
    let authData: any = response;
    if (response && typeof response === 'object' && 'data' in response) {
      // 如果是SuccessResponse格式，提取data字段
      authData = response.data;
    }
    
    // 转换用户数据
    if (!authData.user) {
      // 刷新token可能不返回用户数据，只有新的token
      if (authData.access_token && authData.refresh_token) {
        console.warn('刷新令牌响应缺少用户数据，但包含新的token');
        return authData; // 只返回token数据
      }
      throw new Error('刷新令牌响应中缺少用户数据');
    }
    
    return {
      ...authData,
      user: {
        ...authData.user,
        role: authData.user.is_admin ? 'admin' : 'user',
      },
    };
  },

  // 用户管理
  async getCurrentUser(): Promise<User> {
    const userData = (await apiClient.get('/auth/me')) as User;
    // 将后端字段映射到前端User接口
    return {
      ...userData,
      role: userData.is_admin ? 'admin' : 'user',
    };
  },

  async updateUser(data: Partial<User>): Promise<User> {
    const response = (await apiClient.put('/auth/me', data)) as ApiResponse<User>;
    if (!response.data) {
      throw new Error('更新用户数据失败：响应数据为空');
    }
    return response.data;
  },

  async deleteAccount(): Promise<ApiResponse> {
    return apiClient.delete<ApiResponse>('/auth/me');
  },

  async changePassword(oldPassword: string, newPassword: string): Promise<ApiResponse> {
    return apiClient.post('/auth/change-password', {
      old_password: oldPassword,
      new_password: newPassword,
    });
  },

  // 密码重置和邮箱验证
  async forgotPassword(email: string): Promise<ApiResponse> {
    return apiClient.post('/auth/forgot-password', { email });
  },

  async resetPassword(token: string, newPassword: string): Promise<ApiResponse> {
    return apiClient.post('/auth/reset-password', {
      token,
      new_password: newPassword,
    });
  },

  async verifyEmail(token: string): Promise<ApiResponse> {
    return apiClient.post('/auth/verify-email', { token });
  },

  async resendVerificationEmail(email?: string): Promise<ApiResponse> {
    return apiClient.post('/auth/resend-verification', { email });
  },

  // API密钥管理
  async getApiKeys(): Promise<ApiKey[]> {
    const response = (await apiClient.get('/keys')) as ApiKey[];
    return response;
  },

  async createApiKey(name: string, rate_limit?: number): Promise<ApiKey> {
    const response = (await apiClient.post('/keys', { name, rate_limit: rate_limit || 100 })) as ApiResponse<ApiKey>;
    if (!response.data) {
      throw new Error('创建API密钥失败：响应数据为空');
    }
    return response.data;
  },

  async deleteApiKey(id: string): Promise<ApiResponse> {
    const response = (await apiClient.delete(`/keys/${id}`)) as ApiResponse;
    return response;
  },

  async updateApiKey(id: string, data: Partial<ApiKey>): Promise<ApiKey> {
    interface UpdateApiKeyResponse {
      data: {
        api_key: ApiKey;
      };
    }
    const response = (await apiClient.put(`/keys/${id}`, data)) as UpdateApiKeyResponse;
    return response.data.api_key;
  },

  // 订阅管理
  async getSubscription(): Promise<any> {
    return apiClient.get('/users/subscription');
  },

  async updateSubscription(plan: string): Promise<any> {
    return apiClient.post('/users/subscription', { plan });
  },

  async cancelSubscription(): Promise<ApiResponse> {
    return apiClient.delete('/users/subscription');
  },

  // 管理员功能
  async getAllUsers(params?: any): Promise<PaginatedResponse<User>> {
    return apiClient.get('/admin/users', { params });
  },

  async getUserById(id: string): Promise<User> {
    return apiClient.get(`/admin/users/${id}`);
  },

  async updateUserById(id: string, data: Partial<User>): Promise<User> {
    return apiClient.put(`/admin/users/${id}`, data);
  },

  async deleteUserById(id: string): Promise<ApiResponse> {
    return apiClient.delete(`/admin/users/${id}`);
  },

  async toggleUserActive(id: string): Promise<ApiResponse> {
    return apiClient.post(`/admin/users/${id}/toggle-active`);
  },

  async getSystemStats(): Promise<any> {
    return apiClient.get('/auth/admin/dashboard-stats');
  },

  async getApiUsage(): Promise<any> {
    return apiClient.get('/admin/api-usage');
  },
};

export default authService;