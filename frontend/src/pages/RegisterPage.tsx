import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Eye, EyeOff, Mail, Lock, User, UserPlus, ArrowLeft } from 'lucide-react';
import toast from 'react-hot-toast';

interface RegisterData {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
  fullName?: string;
}

const RegisterPage: React.FC = () => {
  const [registerData, setRegisterData] = useState<RegisterData>({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    fullName: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState(0);
  const [emailValid, setEmailValid] = useState<boolean | null>(null);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // 验证输入
    if (!registerData.username.trim()) {
      toast.error('请输入用户名');
      return;
    }
    
    if (!registerData.email.trim()) {
      toast.error('请输入邮箱地址');
      return;
    }
    
    // 邮箱格式验证
    if (!validateEmail(registerData.email)) {
      toast.error('请输入有效的邮箱地址');
      return;
    }
    
    if (!registerData.password.trim()) {
      toast.error('请输入密码');
      return;
    }
    
    if (registerData.password !== registerData.confirmPassword) {
      toast.error('两次输入的密码不一致');
      return;
    }
    
    // 密码强度验证
    if (registerData.password.length < 8) {
      toast.error('密码长度至少为8位');
      return;
    }
    
    const hasUpperCase = /[A-Z]/.test(registerData.password);
    const hasLowerCase = /[a-z]/.test(registerData.password);
    const hasNumbers = /\d/.test(registerData.password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(registerData.password);
    
    if (!hasUpperCase || !hasLowerCase || !hasNumbers || !hasSpecialChar) {
      toast.error('密码必须包含大小写字母、数字和特殊字符');
      return;
    }
    
    try {
      setIsLoading(true);
      const registrationData = {
        username: registerData.username,
        email: registerData.email,
        password: registerData.password,
        full_name: registerData.fullName,
      };
      
      await register(registrationData);
      toast.success('注册成功！请登录');
      navigate('/login');
    } catch (error: any) {
      const message = error.response?.data?.message || error.message || '注册失败';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  // 计算密码强度
  const calculatePasswordStrength = (password: string) => {
    let strength = 0;
    if (password.length >= 8) strength += 1;
    if (/[A-Z]/.test(password)) strength += 1;
    if (/[a-z]/.test(password)) strength += 1;
    if (/\d/.test(password)) strength += 1;
    if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) strength += 1;
    return strength;
  };

  // 验证邮箱格式
  const validateEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  // 获取密码强度标签
  const getPasswordStrengthLabel = (strength: number) => {
    if (strength <= 1) return { label: '极弱', color: 'text-gray-800', bg: 'bg-gray-800' };
    if (strength === 2) return { label: '弱', color: 'text-orange-500', bg: 'bg-orange-500' };
    if (strength === 3) return { label: '中等', color: 'text-gray-600', bg: 'bg-gray-600' };
    if (strength === 4) return { label: '强', color: 'text-gray-600', bg: 'bg-gray-600' };
    return { label: '极强', color: 'text-emerald-500', bg: 'bg-emerald-500' };
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setRegisterData(prev => ({
      ...prev,
      [name]: value,
    }));

    // 更新密码强度
    if (name === 'password') {
      setPasswordStrength(calculatePasswordStrength(value));
    }

    // 验证邮箱格式
    if (name === 'email') {
      setEmailValid(value.trim() === '' ? null : validateEmail(value));
    }
  };

  const handleDemoFill = () => {
    setRegisterData({
      username: 'demo_user',
      email: 'demo@selfagi.com',
      password: 'demopassword',
      confirmPassword: 'demopassword',
      fullName: '演示用户',
    });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-700 to-gray-800 dark:from-gray-900 dark:to-gray-800 px-4">
      <div className="max-w-md w-full">
        <div className="mb-8 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-gray-800 to-gray-700 rounded-2xl mb-4">
            <UserPlus className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            创建账户
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            加入Self AGI，体验下一代人工智能系统
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                用户名 *
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="text"
                  name="username"
                  value={registerData.username}
                  onChange={handleInputChange}
                  className="block w-full pl-10 pr-3 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                  placeholder="请输入用户名"
                  disabled={isLoading}
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                姓名（可选）
              </label>
              <input
                type="text"
                name="fullName"
                value={registerData.fullName}
                onChange={handleInputChange}
                className="block w-full px-3 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                placeholder="请输入真实姓名"
                disabled={isLoading}
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              邮箱地址 *
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Mail className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="email"
                name="email"
                value={registerData.email}
                onChange={handleInputChange}
                className={`block w-full pl-10 pr-3 py-3 border rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-2 focus:border-transparent ${
                  emailValid === null
                    ? 'border-gray-300 dark:border-gray-600 focus:ring-gray-700 focus:border-gray-700'
                    : emailValid
                    ? 'border-gray-400 dark:border-gray-700 focus:ring-gray-600 focus:border-gray-600'
                    : 'border-gray-600 dark:border-gray-900 focus:ring-gray-800 focus:border-gray-800'
                }`}
                placeholder="请输入邮箱地址"
                disabled={isLoading}
              />
            </div>
            {emailValid !== null && (
              <div className={`mt-1 text-sm flex items-center ${
                emailValid ? 'text-gray-700 dark:text-gray-400' : 'text-gray-900 dark:text-gray-500'
              }`}>
                {emailValid ? (
                  <>
                    <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    邮箱格式正确
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                    请输入有效的邮箱地址
                  </>
                )}
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              密码 *
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type={showPassword ? 'text' : 'password'}
                name="password"
                value={registerData.password}
                onChange={handleInputChange}
                className="block w-full pl-10 pr-10 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                placeholder="请输入密码（至少8位，包含大小写字母、数字和特殊字符）"
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center"
                disabled={isLoading}
              >
                {showPassword ? (
                  <EyeOff className="h-5 w-5 text-gray-400 hover:text-gray-500" />
                ) : (
                  <Eye className="h-5 w-5 text-gray-400 hover:text-gray-500" />
                )}
              </button>
            </div>
            {registerData.password && (
              <div className="mt-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    密码强度:
                  </span>
                  <span className={`text-sm font-medium ${
                    getPasswordStrengthLabel(passwordStrength).color
                  }`}>
                    {getPasswordStrengthLabel(passwordStrength).label}
                  </span>
                </div>
                <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${getPasswordStrengthLabel(passwordStrength).bg}`}
                    style={{ width: `${passwordStrength * 20}%` }}
                  ></div>
                </div>
                <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                  密码应至少包含8个字符，包括大小写字母、数字和特殊字符
                </div>
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              确认密码 *
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type={showPassword ? 'text' : 'password'}
                name="confirmPassword"
                value={registerData.confirmPassword}
                onChange={handleInputChange}
                className="block w-full pl-10 pr-3 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                placeholder="确认密码"
                disabled={isLoading}
              />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <input
                id="terms"
                type="checkbox"
                required
                className="h-4 w-4 text-gray-800 focus:ring-gray-700 border-gray-300 rounded"
              />
              <label htmlFor="terms" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                我同意
                <Link to="/terms" className="ml-1 text-gray-800 hover:text-gray-700 dark:text-gray-400">
                  服务条款
                </Link>
              </label>
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-gradient-to-r from-gray-800 to-gray-700 hover:from-gray-900 hover:to-gray-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  注册中...
                </>
              ) : (
                '注册账户'
              )}
            </button>
          </div>

          <div className="text-center space-y-4">
            <button
              type="button"
              onClick={handleDemoFill}
              className="text-sm text-gray-800 hover:text-gray-700 dark:text-gray-400"
            >
              使用演示数据填充
            </button>

            <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                已有账户？
                <Link
                  to="/login"
                  className="ml-1 font-medium text-gray-800 hover:text-gray-700 dark:text-gray-400"
                >
                  立即登录
                </Link>
              </p>
            </div>

            <div>
              <Link
                to="/"
                className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
              >
                <ArrowLeft className="w-4 h-4 mr-1" />
                返回首页
              </Link>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RegisterPage;