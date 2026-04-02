import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { LoginCredentials } from '../types/auth';
import { authService } from '../services/api/auth';
import { LogIn, Mail, Lock } from 'lucide-react';
import toast from 'react-hot-toast';
import { Button, Input, Card } from '../components/UI';
import TwoFactorAuthModal from '../components/Auth/TwoFactorAuthModal';

const LoginPage: React.FC = () => {
  const [credentials, setCredentials] = useState<LoginCredentials>({
    username: '',
    password: '',
  });
  const [isLoading, setIsLoading] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  
  // 2FA相关状态
  const [showTwoFactorModal, setShowTwoFactorModal] = useState(false);
  const [twoFactorData, setTwoFactorData] = useState<{
    tempToken?: string;
    method?: 'email' | 'totp';
    message?: string;
    credentials: LoginCredentials;
  } | null>(null);
  
  const { loginWithToken } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!credentials.username.trim() || !credentials.password.trim()) {
      toast.error('请输入用户名和密码');
      return;
    }
    
    try {
      setIsLoading(true);
      const loginCredentials = {
        ...credentials,
        rememberMe,
      };
      
      const response = await authService.login(loginCredentials);
      
      // 检查是否需要2FA验证
      if (response.requires_2fa) {
        setTwoFactorData({
          tempToken: response.temp_token,
          method: response.method,
          message: response.message,
          credentials: loginCredentials,
        });
        setShowTwoFactorModal(true);
      } else {
        // 正常登录流程
        const { access_token, refresh_token, user: userData } = response;
        
        if (access_token && refresh_token) {
          // 使用token登录
          await loginWithToken(access_token, refresh_token, userData);
          toast.success('登录成功！');
        } else {
          throw new Error('登录响应不完整');
        }
      }
    } catch (error: any) {
      const message = error.response?.data?.message || error.message || '登录失败';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTwoFactorVerify = async (code: string) => {
    if (!twoFactorData) {
      toast.error('验证数据无效');
      return;
    }
    
    try {
      setIsLoading(true);
      
      // 调用2FA登录API
      const response = await authService.loginWith2FA({
        username_or_email: twoFactorData.credentials.username,
        password: twoFactorData.credentials.password,
        code,
      });
      
      const { access_token, refresh_token, user: userData } = response;
      
      if (access_token && refresh_token) {
        // 使用token登录
        await loginWithToken(access_token, refresh_token, userData);
        
        // 关闭2FA模态框
        setShowTwoFactorModal(false);
        setTwoFactorData(null);
        
        toast.success('登录成功！');
      } else {
        throw new Error('2FA验证响应不完整');
      }
    } catch (error: any) {
      const message = error.response?.data?.message || error.message || '2FA验证失败';
      toast.error(message);
      throw error; // 重新抛出错误，让模态框处理
    } finally {
      setIsLoading(false);
    }
  };

  const handleTwoFactorModalClose = () => {
    setShowTwoFactorModal(false);
    setTwoFactorData(null);
  };

  const handleDemoLogin = () => {
    setCredentials({
      username: 'demo',
      password: 'demopassword'
    });
    // 自动提交表单
    setTimeout(() => {
      const form = document.querySelector('form');
      if (form) {
        form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      }
    }, 100);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setCredentials(prev => ({
      ...prev,
      [name]: value,
    }));
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-4">
      <div className="w-full max-w-md">
        {/* Logo和标题 */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Self AGI
          </h1>
        </div>

        {/* 登录表单卡片 */}
        <Card 
          title="登录到您的账户"
          shadow="lg"
          className="p-8"
        >
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* 用户名输入 */}
            <Input
              label="用户名"
              leftIcon={<Mail className="h-5 w-5" />}
              name="username"
              type="text"
              value={credentials.username}
              onChange={handleInputChange}
              placeholder="请输入用户名"
              required
              disabled={isLoading}
              fullWidth
              autoComplete="username"
            />

            {/* 密码输入 */}
            <Input
              label="密码"
              leftIcon={<Lock className="h-5 w-5" />}
              variant="password"
              name="password"
              value={credentials.password}
              onChange={handleInputChange}
              placeholder="请输入密码"
              required
              disabled={isLoading}
              fullWidth
              autoComplete="current-password"
            />

            {/* 记住我和忘记密码 */}
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  id="remember-me"
                  name="remember-me"
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="h-4 w-4 text-gray-600 focus:ring-gray-500 border-gray-300 rounded"
                />
                <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                  记住我
                </label>
              </div>
              <div>
                <Link
                  to="/forgot-password"
                  className="text-sm font-medium text-gray-600 hover:text-gray-500 dark:text-gray-400 dark:hover:text-gray-300"
                >
                  忘记密码？
                </Link>
              </div>
            </div>

            {/* 提交按钮 */}
            <Button
              type="submit"
              variant="primary"
              loading={isLoading}
              disabled={isLoading}
              fullWidth
              leftIcon={!isLoading && <LogIn className="w-4 h-4" />}
            >
              {isLoading ? '登录中...' : '登录'}
            </Button>


          </form>

          {/* 演示账户登录按钮 */}
          <div className="mt-4 text-center">
            <Button
              type="button"
              variant="outline"
              onClick={handleDemoLogin}
              disabled={isLoading}
              fullWidth
            >
              使用演示账户登录 (demo/demopassword)
            </Button>
          </div>

          {/* 注册链接 */}
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              还没有账户？{' '}
              <Link
                to="/register"
                className="font-medium text-gray-600 hover:text-gray-500 dark:text-gray-400 dark:hover:text-gray-300"
              >
                立即注册
              </Link>
            </p>
          </div>
        </Card>
      </div>
      
      {/* 2FA验证模态框 */}
      {twoFactorData && (
        <TwoFactorAuthModal
          isOpen={showTwoFactorModal}
          onClose={handleTwoFactorModalClose}
          onVerify={handleTwoFactorVerify}
          method={twoFactorData.method || 'email'}
          tempToken={twoFactorData.tempToken}
          message={twoFactorData.message}
        />
      )}
    </div>
  );
};

export default LoginPage;