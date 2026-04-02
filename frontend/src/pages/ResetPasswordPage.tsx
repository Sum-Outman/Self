import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Lock, Key, Eye, EyeOff, CheckCircle, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import { authService } from '../services/api/auth';

const ResetPasswordPage: React.FC = () => {
  const [passwords, setPasswords] = useState({
    newPassword: '',
    confirmPassword: '',
  });
  const [showPassword, setShowPassword] = useState({
    newPassword: false,
    confirmPassword: false,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  
  const navigate = useNavigate();
  const location = useLocation();
  
  // 从URL获取token
  const queryParams = new URLSearchParams(location.search);
  const token = queryParams.get('token') || '';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // 验证输入
    if (!passwords.newPassword.trim()) {
      toast.error('请输入新密码');
      return;
    }
    
    if (!passwords.confirmPassword.trim()) {
      toast.error('请确认新密码');
      return;
    }
    
    if (passwords.newPassword !== passwords.confirmPassword) {
      toast.error('两次输入的密码不一致');
      return;
    }
    
    // 密码强度验证
    if (passwords.newPassword.length < 8) {
      toast.error('密码长度至少为8位');
      return;
    }
    
    const hasUpperCase = /[A-Z]/.test(passwords.newPassword);
    const hasLowerCase = /[a-z]/.test(passwords.newPassword);
    const hasNumbers = /\d/.test(passwords.newPassword);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(passwords.newPassword);
    
    if (!hasUpperCase || !hasLowerCase || !hasNumbers || !hasSpecialChar) {
      toast.error('密码必须包含大小写字母、数字和特殊字符');
      return;
    }
    
    try {
      setIsLoading(true);
      // 调用后端API重置密码
      await authService.resetPassword(token, passwords.newPassword);
      
      toast.success('密码重置成功');
      setIsCompleted(true);
      
      // 3秒后跳转到登录页面
      setTimeout(() => {
        navigate('/login');
      }, 3000);
    } catch (error: any) {
      const message = error.response?.data?.message || error.message || '密码重置失败';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setPasswords(prev => ({
      ...prev,
      [name]: value,
    }));
  };

  const toggleShowPassword = (field: 'newPassword' | 'confirmPassword') => {
    setShowPassword(prev => ({
      ...prev,
      [field]: !prev[field],
    }));
  };

  // 密码强度计算
  const calculatePasswordStrength = (password: string) => {
    let strength = 0;
    
    if (password.length >= 8) strength += 1;
    if (/[A-Z]/.test(password)) strength += 1;
    if (/[a-z]/.test(password)) strength += 1;
    if (/\d/.test(password)) strength += 1;
    if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) strength += 1;
    
    return strength;
  };

  const passwordStrength = calculatePasswordStrength(passwords.newPassword);
  const strengthLabels = ['非常弱', '弱', '中等', '强', '非常强'];
  const strengthColors = ['bg-gray-800', 'bg-orange-500', 'bg-gray-600', 'bg-gray-700', 'bg-gray-600'];

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-4">
      <div className="w-full max-w-md">
        {/* Logo和标题 */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-gray-600 to-gray-600 rounded-2xl mb-4">
            <Key className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            重置密码
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            设置您的新密码
          </p>
        </div>

        {/* 重置密码表单 */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          {isCompleted ? (
            <div className="text-center space-y-6">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gray-600 dark:bg-gray-900/30 rounded-full mb-4">
                <CheckCircle className="w-10 h-10 text-gray-700 dark:text-gray-400" />
              </div>
              
              <div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                  密码重置成功
                </h2>
                <p className="text-gray-600 dark:text-gray-400">
                  您的密码已成功重置，请使用新密码登录
                </p>
              </div>
              
              <div className="space-y-3">
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  <p>即将跳转到登录页面...</p>
                  <div className="mt-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div className="bg-gray-600 h-2 rounded-full animate-pulse"></div>
                  </div>
                </div>
                
                <Link
                  to="/login"
                  className="block w-full py-3 px-4 text-center text-sm font-medium text-white bg-gradient-to-r from-gray-600 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-700"
                >
                  立即登录
                </Link>
              </div>
            </div>
          ) : (
            <>
              {!token && (
                <div className="mb-6 p-4 bg-gray-800 dark:bg-gray-900/20 border border-gray-600 dark:border-gray-900 rounded-lg">
                  <div className="flex items-start">
                    <AlertCircle className="w-5 h-5 text-gray-700 dark:text-gray-400 mr-2 mt-0.5 flex-shrink-0" />
                    <div className="text-sm text-gray-800 dark:text-gray-500">
                      <p className="font-medium">链接无效或已过期</p>
                      <p className="mt-1">请重新发起密码重置请求</p>
                    </div>
                  </div>
                </div>
              )}
              
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
                设置新密码
              </h2>
              
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* 新密码输入 */}
                <div>
                  <label htmlFor="newPassword" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    新密码
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Lock className="h-5 w-5 text-gray-400" />
                    </div>
                    <input
                      id="newPassword"
                      name="newPassword"
                      type={showPassword.newPassword ? 'text' : 'password'}
                      value={passwords.newPassword}
                      onChange={handlePasswordChange}
                      className="pl-10 pr-10 w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent dark:bg-gray-700 dark:text-gray-100"
                      placeholder="请输入新密码"
                      required
                      disabled={isLoading}
                    />
                    <button
                      type="button"
                      onClick={() => toggleShowPassword('newPassword')}
                      className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    >
                      {showPassword.newPassword ? (
                        <EyeOff className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                      ) : (
                        <Eye className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                      )}
                    </button>
                  </div>
                  
                  {/* 密码强度指示器 */}
                  {passwords.newPassword && (
                    <div className="mt-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-gray-600 dark:text-gray-400">
                          密码强度:
                        </span>
                        <span className={`text-xs font-medium ${
                          passwordStrength <= 1 ? 'text-gray-900 dark:text-gray-500' :
                          passwordStrength <= 2 ? 'text-orange-600 dark:text-orange-400' :
                          passwordStrength <= 3 ? 'text-gray-700 dark:text-gray-400' :
                          passwordStrength <= 4 ? 'text-gray-800 dark:text-gray-400' :
                          'text-gray-700 dark:text-gray-400'
                        }`}>
                          {strengthLabels[passwordStrength - 1] || '非常弱'}
                        </span>
                      </div>
                      <div className="flex space-x-1">
                        {[1, 2, 3, 4, 5].map((level) => (
                          <div
                            key={level}
                            className={`h-1 flex-1 rounded-full ${
                              level <= passwordStrength
                                ? strengthColors[passwordStrength - 1]
                                : 'bg-gray-200 dark:bg-gray-700'
                            }`}
                          />
                        ))}
                      </div>
                      
                      <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                        <p>密码要求:</p>
                        <ul className="mt-1 space-y-1">
                          <li className={`flex items-center ${passwords.newPassword.length >= 8 ? 'text-gray-700 dark:text-gray-400' : ''}`}>
                            <span className="mr-1">•</span> 至少8个字符
                          </li>
                          <li className={`flex items-center ${/[A-Z]/.test(passwords.newPassword) ? 'text-gray-700 dark:text-gray-400' : ''}`}>
                            <span className="mr-1">•</span> 至少1个大写字母
                          </li>
                          <li className={`flex items-center ${/[a-z]/.test(passwords.newPassword) ? 'text-gray-700 dark:text-gray-400' : ''}`}>
                            <span className="mr-1">•</span> 至少1个小写字母
                          </li>
                          <li className={`flex items-center ${/\d/.test(passwords.newPassword) ? 'text-gray-700 dark:text-gray-400' : ''}`}>
                            <span className="mr-1">•</span> 至少1个数字
                          </li>
                          <li className={`flex items-center ${/[!@#$%^&*(),.?":{}|<>]/.test(passwords.newPassword) ? 'text-gray-700 dark:text-gray-400' : ''}`}>
                            <span className="mr-1">•</span> 至少1个特殊字符
                          </li>
                        </ul>
                      </div>
                    </div>
                  )}
                </div>

                {/* 确认密码输入 */}
                <div>
                  <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    确认新密码
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Lock className="h-5 w-5 text-gray-400" />
                    </div>
                    <input
                      id="confirmPassword"
                      name="confirmPassword"
                      type={showPassword.confirmPassword ? 'text' : 'password'}
                      value={passwords.confirmPassword}
                      onChange={handlePasswordChange}
                      className="pl-10 pr-10 w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent dark:bg-gray-700 dark:text-gray-100"
                      placeholder="请再次输入新密码"
                      required
                      disabled={isLoading}
                    />
                    <button
                      type="button"
                      onClick={() => toggleShowPassword('confirmPassword')}
                      className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    >
                      {showPassword.confirmPassword ? (
                        <EyeOff className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                      ) : (
                        <Eye className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                      )}
                    </button>
                  </div>
                  
                  {/* 密码匹配指示器 */}
                  {passwords.confirmPassword && (
                    <div className="mt-2">
                      <div className={`text-xs flex items-center ${
                        passwords.newPassword === passwords.confirmPassword
                          ? 'text-gray-700 dark:text-gray-400'
                          : 'text-gray-900 dark:text-gray-500'
                      }`}>
                        {passwords.newPassword === passwords.confirmPassword ? (
                          <>
                            <CheckCircle className="w-3 h-3 mr-1" />
                            密码匹配
                          </>
                        ) : (
                          <>
                            <AlertCircle className="w-3 h-3 mr-1" />
                            密码不匹配
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* 提交按钮 */}
                <button
                  type="submit"
                  disabled={isLoading || !token || passwords.newPassword !== passwords.confirmPassword || passwordStrength < 3}
                  className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-gradient-to-r from-gray-600 to-gray-600 hover:from-gray-700 hover:to-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      重置中...
                    </>
                  ) : (
                    '重置密码'
                  )}
                </button>

                {/* 返回登录链接 */}
                <div className="text-center">
                  <Link
                    to="/login"
                    className="inline-flex items-center text-sm font-medium text-gray-600 hover:text-gray-500 dark:text-gray-400 dark:hover:text-gray-300"
                  >
                    返回登录
                  </Link>
                </div>
              </form>
              
              {/* 安全提示 */}
              <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  <p className="font-medium mb-2">安全提示：</p>
                  <ul className="space-y-1">
                    <li>• 请勿使用与其他网站相同的密码</li>
                    <li>• 定期更换密码以提高安全性</li>
                    <li>• 不要在公共设备上保存密码</li>
                    <li>• 启用双因素认证增强账户安全</li>
                  </ul>
                </div>
              </div>
            </>
          )}
        </div>

        {/* 版权信息 */}
        <div className="mt-8 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>© {new Date().getFullYear()} Self AGI. 版权所有.</p>
          <p className="mt-1">
            <Link to="/terms" className="hover:text-gray-700 dark:hover:text-gray-300">
              服务条款
            </Link>
            {' · '}
            <Link to="/privacy" className="hover:text-gray-700 dark:hover:text-gray-300">
              隐私政策
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default ResetPasswordPage;