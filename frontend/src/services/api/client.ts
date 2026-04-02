import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { ApiResponse } from '../../types/api';
import { errorHandler } from '../../utils/errorHandler';

// 使用环境变量或默认值
let API_BASE_URL = 'http://localhost:8000/api';
if (typeof window !== 'undefined' && (window as any).__API_BASE_URL__) {
  API_BASE_URL = (window as any).__API_BASE_URL__;
} else if (typeof process !== 'undefined' && process.env.VITE_API_URL) {
  API_BASE_URL = process.env.VITE_API_URL;
}

// 验证和规范化API基础URL，确保与后端路由前缀一致
function normalizeApiBaseUrl(url: string): string {
  // 确保URL以斜杠结尾，方便路径拼接
  let normalized = url.trim();
  if (!normalized.endsWith('/')) {
    normalized += '/';
  }
  
  // 检查是否包含/api路径（后端所有路由都有/api前缀）
  if (!normalized.includes('/api')) {
    console.warn(`API基础URL "${url}" 不包含'/api'路径，自动添加。后端所有路由都有'/api'前缀。`);
    // 添加/api到路径末尾
    if (normalized.endsWith('/')) {
      normalized = normalized.slice(0, -1) + '/api/';
    } else {
      normalized += 'api/';
    }
  }
  
  // 移除末尾斜杠，以便axios正确拼接路径
  if (normalized.endsWith('/')) {
    normalized = normalized.slice(0, -1);
  }
  
  return normalized;
}

// 规范化API基础URL
API_BASE_URL = normalizeApiBaseUrl(API_BASE_URL);

// Cookie工具函数
function getCookie(name: string): string | null {
  if (typeof document === 'undefined') return null;
  
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) {
    return parts.pop()?.split(';').shift() || null;
  }
  return null;
}

// CSRF令牌名称（与后端保持一致）
const CSRF_TOKEN_NAME = 'csrftoken';
const CSRF_HEADER_NAME = 'X-CSRFToken';

class ApiClient {
  private client: AxiosInstance;
  private static instance: ApiClient;

  private constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        
        // 添加CSRF令牌保护（对同源请求且非安全方法的请求）
        if (typeof window !== 'undefined') {
          const csrfToken = getCookie(CSRF_TOKEN_NAME);
          const isSameOrigin = config.url?.startsWith('/') || config.baseURL?.includes(window.location.host);
          const isSafeMethod = ['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(config.method?.toUpperCase() || '');
          
          if (csrfToken && isSameOrigin && !isSafeMethod) {
            config.headers[CSRF_HEADER_NAME] = csrfToken;
          }
        }
        
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;
        
        // 如果是401错误且不是刷新token请求，尝试刷新token
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;
          
          try {
            const refreshToken = localStorage.getItem('refresh_token');
            if (!refreshToken) {
              throw new Error('没有刷新令牌');
            }
            
            const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
              refresh_token: refreshToken,
            });
            
            const { access_token, refresh_token } = response.data.data;
            localStorage.setItem('access_token', access_token);
            localStorage.setItem('refresh_token', refresh_token);
            
            // 更新请求头
            originalRequest.headers.Authorization = `Bearer ${access_token}`;
            
            // 重试原始请求
            return this.client(originalRequest);
          } catch (refreshError) {
            // 刷新失败，清除本地存储并重定向到登录页
            localStorage.removeItem('access_token');
            localStorage.removeItem('refresh_token');
            
            // 触发认证失效事件，让AuthContext更新状态
            const authExpiredEvent = new CustomEvent('auth:expired');
            window.dispatchEvent(authExpiredEvent);
            
            // 记录认证错误
            errorHandler.handleError(refreshError, '认证刷新失败');
            
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }
        
        // 使用全局错误处理器处理其他错误
        errorHandler.handleError(error, 'API请求失败');
        
        return Promise.reject(error);
      }
    );
  }

  public static getInstance(): ApiClient {
    if (!ApiClient.instance) {
      ApiClient.instance = new ApiClient();
    }
    return ApiClient.instance;
  }

  public async get<T = unknown>(url: string, config?: AxiosRequestConfig<unknown>): Promise<T> {
    const response = await this.client.get<ApiResponse<T>>(url, config);
    return this.handleResponse(response);
  }

  public async post<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig<unknown>): Promise<T> {
    const response = await this.client.post<ApiResponse<T>>(url, data, config);
    return this.handleResponse(response);
  }

  public async put<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig<unknown>): Promise<T> {
    const response = await this.client.put<ApiResponse<T>>(url, data, config);
    return this.handleResponse(response);
  }

  public async patch<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig<unknown>): Promise<T> {
    const response = await this.client.patch<ApiResponse<T>>(url, data, config);
    return this.handleResponse(response);
  }

  public async delete<T = unknown>(url: string, config?: AxiosRequestConfig<unknown>): Promise<T> {
    const response = await this.client.delete<ApiResponse<T>>(url, config);
    return this.handleResponse(response);
  }

  public getBaseUrl(): string {
    return this.client.defaults.baseURL || API_BASE_URL;
  }

  private handleResponse<T>(response: AxiosResponse<ApiResponse<T>>): T {
    const data = response.data;
    const requestUrl = response.config.url || '未知请求';
    
    // 特殊处理：登录响应可能没有success字段，但有access_token字段
    if ('access_token' in data && data.access_token) {
      // 这是登录响应，直接返回整个响应数据
      return data as unknown as T;
    }
    
    // 特殊处理：响应是数组（如/keys端点）
    if (Array.isArray(data)) {
      return data as unknown as T;
    }
    
    // 特殊处理：响应是对象但没有success字段（如/auth/me端点）
    if (data && typeof data === 'object' && !('success' in data)) {
      // 这是直接返回数据的API响应，直接返回
      return data as unknown as T;
    }
    
    // 标准API响应处理（包含success字段）
    if (!data.success) {
      const errorMessage = data.message || data.error || 'API请求失败';
      // 记录错误
      errorHandler.handleError(new Error(errorMessage), `API响应错误: ${requestUrl}`);
      throw new Error(errorMessage);
    }
    
    if (data.data === undefined) {
      const errorMessage = 'API响应数据为空';
      errorHandler.handleError(new Error(errorMessage), `API数据错误: ${requestUrl}`);
      throw new Error(errorMessage);
    }
    
    return data.data;
  }
}

export { ApiClient };
export const apiClient = ApiClient.getInstance();