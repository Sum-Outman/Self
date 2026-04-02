import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { authService } from '../services/api/auth';
import { User, ApiKey, LoginCredentials, RegisterData } from '../types/auth';

interface AuthContextType {
  user: User | null;
  apiKeys: ApiKey[];
  isLoading: boolean;
  isAuthenticated: boolean;
  isAdmin: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  loginWithToken: (accessToken: string, refreshToken: string, userData?: User) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  refreshApiKeys: () => Promise<void>;
  createApiKey: (name: string, rate_limit?: number) => Promise<ApiKey>;
  deleteApiKey: (id: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

const isDevelopment = process.env.NODE_ENV === 'development';
const log = (...args: any[]) => {
  if (isDevelopment) {
    console.log(...args);
  }
};
const logError = (...args: any[]) => {
  console.error(...args);
};

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();
  
  // 环境检查
  const isDevelopment = process.env.NODE_ENV !== 'production';

  // 初始化时检查认证状态
  useEffect(() => {
    checkAuth();
    
    // 监听localStorage变化
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === 'access_token' || event.key === 'refresh_token') {
        if (isDevelopment) {
          log('检测到token变化，重新检查认证状态');
        }
        checkAuth();
      }
    };
    
    // 监听storage事件
    window.addEventListener('storage', handleStorageChange);
    
    // 监听认证失效事件（来自API客户端）
    const handleAuthExpired = () => {
      if (isDevelopment) {
        log('收到认证失效事件，清除用户状态');
      }
      setUser(null);
      setApiKeys([]);
    };
    
    window.addEventListener('auth:expired', handleAuthExpired);
    
    // 清理函数
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('auth:expired', handleAuthExpired);
    };
  }, []);

  const checkAuth = async () => {
    try {
      const token = localStorage.getItem('access_token');
      if (isDevelopment) {
        log('认证检查 - token存在:', !!token);
      }
      
      if (token) {
        // 验证token有效性
        if (isDevelopment) {
          log('正在验证token有效性...');
        }
        const userData = await authService.getCurrentUser();
        if (isDevelopment) {
          log('获取用户信息成功');
        }
        setUser(userData);
        
        // 加载API密钥
        if (isDevelopment) {
          console.log('正在加载API密钥...');
        }
        const keys = await authService.getApiKeys();
        if (isDevelopment) {
          console.log('API密钥加载成功:', keys?.length || 0, '个密钥');
        }
        setApiKeys(keys || []);
      } else {
        if (isDevelopment) {
          console.log('没有token，用户未登录');
        }
      }
    } catch (error) {
      logError('认证检查失败:', error);
      // 不清除token，让API客户端的拦截器处理刷新和重定向
    } finally {
      if (isDevelopment) {
        console.log('认证检查完成，设置loading为false');
      }
      setIsLoading(false);
    }
  };

  const login = async (credentials: LoginCredentials) => {
    if (isDevelopment) {
      console.log('开始登录流程...');
    }
    try {
      setIsLoading(true);
      if (isDevelopment) {
        console.log('调用authService.login...');
      }
      const { access_token, refresh_token, user: userData } = await authService.login(credentials);
      if (isDevelopment) {
        console.log('登录成功，获取到用户数据');
      }
      
      // 检查token是否存在
      if (!access_token || !refresh_token) {
        throw new Error('登录响应缺少token');
      }
      
      // 保存token
      localStorage.setItem('access_token', access_token);
      localStorage.setItem('refresh_token', refresh_token);
      if (isDevelopment) {
        console.log('token已保存到localStorage');
      }
      
      if (userData) {
        setUser(userData);
        if (isDevelopment) {
          console.log('用户状态已设置');
        }
      } else {
        throw new Error('登录响应缺少用户数据');
      }
      
      // 加载API密钥
      if (isDevelopment) {
        console.log('开始加载API密钥...');
      }
      const keys = await authService.getApiKeys();
      if (isDevelopment) {
        console.log('API密钥加载完成:', keys?.length || 0, '个密钥');
      }
      setApiKeys(keys || []);
      
      if (isDevelopment) {
        console.log('准备导航到/dashboard');
      }
      navigate('/dashboard');
      if (isDevelopment) {
        console.log('导航完成');
      }
    } catch (error) {
      console.error('登录失败:', error);
      throw error;
    } finally {
      if (isDevelopment) {
        console.log('设置isLoading为false');
      }
      setIsLoading(false);
    }
  };

  const loginWithToken = async (accessToken: string, refreshToken: string, userData?: User) => {
    if (isDevelopment) {
      console.log('使用token登录...');
    }
    
    try {
      setIsLoading(true);
      
      // 保存token
      localStorage.setItem('access_token', accessToken);
      localStorage.setItem('refresh_token', refreshToken);
      if (isDevelopment) {
        console.log('token已保存到localStorage');
      }
      
      // 如果有用户数据，直接使用
      if (userData) {
        setUser(userData);
        if (isDevelopment) {
          console.log('用户状态已设置（通过传入的用户数据）');
        }
      } else {
        // 否则获取当前用户信息
        if (isDevelopment) {
          console.log('开始获取用户信息...');
        }
        const currentUser = await authService.getCurrentUser();
        setUser(currentUser);
        if (isDevelopment) {
          console.log('用户信息获取完成');
        }
      }
      
      // 加载API密钥
      if (isDevelopment) {
        console.log('开始加载API密钥...');
      }
      const keys = await authService.getApiKeys();
      if (isDevelopment) {
        console.log('API密钥加载完成:', keys?.length || 0, '个密钥');
      }
      setApiKeys(keys || []);
      
      if (isDevelopment) {
        console.log('准备导航到/dashboard');
      }
      navigate('/dashboard');
      if (isDevelopment) {
        console.log('导航完成');
      }
    } catch (error) {
      console.error('使用token登录失败:', error);
      throw error;
    } finally {
      if (isDevelopment) {
        console.log('设置isLoading为false');
      }
      setIsLoading(false);
    }
  };

  const register = async (data: RegisterData) => {
    try {
      setIsLoading(true);
      await authService.register(data);
      navigate('/login');
    } catch (error) {
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    try {
      await authService.logout();
    } catch (error) {
      console.error('登出失败:', error);
    } finally {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      setUser(null);
      setApiKeys([]);
      navigate('/login');
    }
  };

  const refreshUser = async () => {
    try {
      const userData = await authService.getCurrentUser();
      setUser(userData);
    } catch (error) {
      console.error('刷新用户信息失败:', error);
      throw error;
    }
  };

  const refreshApiKeys = async () => {
    try {
      const keys = await authService.getApiKeys();
      setApiKeys(keys || []);
    } catch (error) {
      console.error('刷新API密钥失败:', error);
      throw error;
    }
  };

  const createApiKey = async (name: string, rate_limit?: number): Promise<ApiKey> => {
    try {
      const apiKey = await authService.createApiKey(name, rate_limit);
      await refreshApiKeys();
      return apiKey;
    } catch (error) {
      console.error('创建API密钥失败:', error);
      throw error;
    }
  };

  const deleteApiKey = async (id: string) => {
    try {
      await authService.deleteApiKey(id);
      await refreshApiKeys();
    } catch (error) {
      console.error('删除API密钥失败:', error);
      throw error;
    }
  };

  const value: AuthContextType = {
    user,
    apiKeys,
    isLoading,
    isAuthenticated: !!user,
    isAdmin: user?.is_admin === true || (user as any)?.role === 'admin',
    login,
    loginWithToken,
    register,
    logout,
    refreshUser,
    refreshApiKeys,
    createApiKey,
    deleteApiKey,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth必须在AuthProvider内使用');
  }
  return context;
}