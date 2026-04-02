import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Mail, ArrowLeft, CheckCircle, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import { authService } from '../services/api/auth';

const ForgotPasswordPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!email.trim()) {
      toast.error('请输入邮箱地址');
      return;
    }
    
    // 简单的邮箱格式验证
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      toast.error('请输入有效的邮箱地址');
      return;
    }
    
    try {
      setIsLoading(true);
      // 调用后端API发送密码重置邮件
      await authService.forgotPassword(email);
      
      toast.success('密码重置邮件已发送，请查收您的邮箱');
      setIsSubmitted(true);
    } catch (error: any) {
      const message = error.response?.data?.message || error.message || '发送重置邮件失败';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-4">
      <div className="w-full max-w-md">
        {/* Logo和标题 */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-gray-600 to-gray-600 rounded-2xl mb-4">
            <Mail className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            忘记密码
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            输入您的邮箱地址重置密码
          </p>
        </div>

        {/* 重置密码表单 */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          {isSubmitted ? (
            <div className="text-center space-y-6">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gray-600 dark:bg-gray-900/30 rounded-full mb-4">
                <CheckCircle className="w-10 h-10 text-gray-700 dark:text-gray-400" />
              </div>
              
              <div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                  邮件已发送
                </h2>
                <p className="text-gray-600 dark:text-gray-400">
                  重置密码的链接已发送至您的邮箱
                  <span className="font-medium text-gray-900 dark:text-white ml-1">{email}</span>
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
                  请在24小时内点击邮件中的链接重置密码
                </p>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center justify-center text-sm text-gray-700 dark:text-gray-400 bg-gray-800 dark:bg-gray-900/20 p-3 rounded-lg">
                  <AlertCircle className="w-4 h-4 mr-2" />
                  如果未收到邮件，请检查垃圾邮件文件夹
                </div>
                
                <div className="space-y-3">
                  <button
                    type="button"
                    onClick={() => {
                      setIsSubmitted(false);
                      setEmail('');
                    }}
                    className="w-full py-3 px-4 text-sm font-medium text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/20 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-900/30"
                  >
                    重新发送邮件
                  </button>
                  
                  <Link
                    to="/login"
                    className="block w-full py-3 px-4 text-center text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
                  >
                    返回登录
                  </Link>
                </div>
              </div>
            </div>
          ) : (
            <>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
                重置密码
              </h2>
              
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* 邮箱输入 */}
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    邮箱地址
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Mail className="h-5 w-5 text-gray-400" />
                    </div>
                    <input
                      id="email"
                      name="email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="pl-10 w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent dark:bg-gray-700 dark:text-gray-100"
                      placeholder="请输入注册时使用的邮箱"
                      required
                      disabled={isLoading}
                    />
                  </div>
                  <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                    我们将向该邮箱发送密码重置链接
                  </p>
                </div>

                {/* 提交按钮 */}
                <button
                  type="submit"
                  disabled={isLoading || !email.trim()}
                  className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-gradient-to-r from-gray-600 to-gray-600 hover:from-gray-700 hover:to-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      发送中...
                    </>
                  ) : (
                    '发送重置链接'
                  )}
                </button>

                {/* 返回登录链接 */}
                <div className="text-center">
                  <Link
                    to="/login"
                    className="inline-flex items-center text-sm font-medium text-gray-600 hover:text-gray-500 dark:text-gray-400 dark:hover:text-gray-300"
                  >
                    <ArrowLeft className="w-4 h-4 mr-1" />
                    返回登录
                  </Link>
                </div>
              </form>
            </>
          )}

          {/* 帮助信息 */}
          <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <p className="font-medium mb-1">遇到问题？</p>
              <ul className="space-y-1">
                <li>• 确认邮箱地址是否正确</li>
                <li>• 检查垃圾邮件文件夹</li>
                <li>• 如果仍未收到邮件，请联系管理员</li>
              </ul>
            </div>
          </div>
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

export default ForgotPasswordPage;