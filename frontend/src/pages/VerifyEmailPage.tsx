import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Mail, CheckCircle, XCircle, AlertCircle, RefreshCw } from 'lucide-react';
import toast from 'react-hot-toast';
import { authService } from '../services/api/auth';

// 邮箱验证状态
type VerificationStatus = 'pending' | 'verifying' | 'success' | 'failed' | 'expired' | 'already_verified';

const VerifyEmailPage: React.FC = () => {
  const [status, setStatus] = useState<VerificationStatus>('pending');
  const [isResending, setIsResending] = useState(false);
  
  const navigate = useNavigate();
  const location = useLocation();
  
  // 从URL获取token
  const queryParams = new URLSearchParams(location.search);
  const token = queryParams.get('token') || '';

  useEffect(() => {
    if (token) {
      verifyEmailToken(token);
    } else {
      setStatus('failed');
      toast.error('验证链接无效');
    }
  }, [token]);

  const verifyEmailToken = async (verificationToken: string) => {
    try {
      setStatus('verifying');
      
      // 调用后端API验证邮箱
      const response = await authService.verifyEmail(verificationToken);
      
      // 根据API响应设置状态
      if (response.success) {
        setStatus('success');
        toast.success(response.message || '邮箱验证成功');
        // 3秒后跳转到登录页面
        setTimeout(() => {
          navigate('/login');
        }, 3000);
      } else {
        // 根据错误消息判断状态
        const errorMsg = response.message || '验证失败';
        if (errorMsg.includes('过期')) {
          setStatus('expired');
        } else if (errorMsg.includes('已验证')) {
          setStatus('already_verified');
        } else {
          setStatus('failed');
        }
        toast.error(errorMsg);
      }
    } catch (error: any) {
      console.error('邮箱验证失败:', error);
      setStatus('failed');
      const message = error.response?.data?.message || error.message || '邮箱验证失败';
      toast.error(message);
    }
  };

  const handleResendVerification = async () => {
    try {
      setIsResending(true);
      
      // 调用后端API重新发送验证邮件
      await authService.resendVerificationEmail(token);
      
      toast.success('验证邮件已重新发送，请查收您的邮箱');
    } catch (error: any) {
      const message = error.response?.data?.message || error.message || '发送验证邮件失败';
      toast.error(message);
    } finally {
      setIsResending(false);
    }
  };

  const getStatusContent = () => {
    switch (status) {
      case 'verifying':
        return {
          icon: <RefreshCw className="w-12 h-12 text-gray-700 animate-spin" />,
          title: '验证中...',
          description: '正在验证您的邮箱地址，请稍候',
          color: 'bg-gray-600 dark:bg-gray-900/30 text-gray-800 dark:text-gray-400',
          button: null,
        };
        
      case 'success':
        return {
          icon: <CheckCircle className="w-12 h-12 text-gray-600" />,
          title: '邮箱验证成功',
          description: '您的邮箱地址已验证成功，即将跳转到登录页面',
          color: 'bg-gray-600 dark:bg-gray-900/30 text-gray-700 dark:text-gray-400',
          button: (
            <Link
              to="/login"
              className="inline-flex items-center px-6 py-3 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-emerald-600 rounded-lg hover:from-gray-800 hover:to-emerald-700"
            >
              立即登录
            </Link>
          ),
        };
        
      case 'failed':
        return {
          icon: <XCircle className="w-12 h-12 text-gray-800" />,
          title: '验证失败',
          description: '邮箱验证失败，请检查验证链接是否正确或联系管理员',
          color: 'bg-gray-800 dark:bg-gray-900/30 text-gray-900 dark:text-gray-500',
          button: (
            <button
              onClick={handleResendVerification}
              disabled={isResending}
              className="inline-flex items-center px-6 py-3 text-sm font-medium text-white bg-gradient-to-r from-gray-900 to-gray-700 rounded-lg hover:from-gray-900 hover:to-gray-800 disabled:opacity-50"
            >
              {isResending ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  发送中...
                </>
              ) : (
                '重新发送验证邮件'
              )}
            </button>
          ),
        };
        
      case 'expired':
        return {
          icon: <AlertCircle className="w-12 h-12 text-orange-500" />,
          title: '验证链接已过期',
          description: '验证链接已过期，请重新发送验证邮件',
          color: 'bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400',
          button: (
            <button
              onClick={handleResendVerification}
              disabled={isResending}
              className="inline-flex items-center px-6 py-3 text-sm font-medium text-white bg-gradient-to-r from-orange-600 to-amber-600 rounded-lg hover:from-orange-700 hover:to-amber-700 disabled:opacity-50"
            >
              {isResending ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  发送中...
                </>
              ) : (
                '重新发送验证邮件'
              )}
            </button>
          ),
        };
        
      case 'already_verified':
        return {
          icon: <CheckCircle className="w-12 h-12 text-gray-600" />,
          title: '邮箱已验证',
          description: '您的邮箱地址已经验证过，可以直接登录',
          color: 'bg-gray-600 dark:bg-gray-900/30 text-gray-700 dark:text-gray-400',
          button: (
            <Link
              to="/login"
              className="inline-flex items-center px-6 py-3 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-emerald-600 rounded-lg hover:from-gray-800 hover:to-emerald-700"
            >
              立即登录
            </Link>
          ),
        };
        
      default: // pending
        return {
          icon: <Mail className="w-12 h-12 text-gray-400" />,
          title: '等待验证',
          description: '请检查您的邮箱并点击验证链接',
          color: 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
          button: (
            <button
              onClick={handleResendVerification}
              disabled={isResending}
              className="inline-flex items-center px-6 py-3 text-sm font-medium text-white bg-gradient-to-r from-gray-600 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-700 disabled:opacity-50"
            >
              {isResending ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  发送中...
                </>
              ) : (
                '重新发送验证邮件'
              )}
            </button>
          ),
        };
    }
  };

  const statusContent = getStatusContent();

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-4">
      <div className="w-full max-w-md">
        {/* Logo和标题 */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-gray-600 to-gray-600 rounded-2xl mb-4">
            <Mail className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            邮箱验证
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            验证您的邮箱地址以激活账户
          </p>
        </div>

        {/* 验证状态卡片 */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          <div className="text-center space-y-6">
            {/* 状态图标 */}
            <div className="inline-flex items-center justify-center">
              {statusContent.icon}
            </div>
            
            {/* 状态标题和描述 */}
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                {statusContent.title}
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                {statusContent.description}
              </p>
            </div>
            
            {/* 状态标签 */}
            <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${statusContent.color}`}>
              {status === 'verifying' && (
                <>
                  <RefreshCw className="w-3 h-3 mr-2 animate-spin" />
                  验证中
                </>
              )}
              {status === 'success' && '验证成功'}
              {status === 'failed' && '验证失败'}
              {status === 'expired' && '链接过期'}
              {status === 'already_verified' && '已验证'}
              {status === 'pending' && '等待验证'}
            </div>
            
            {/* 进度条（验证中或跳转中） */}
            {(status === 'verifying' || status === 'success') && (
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-gray-600 h-2 rounded-full animate-pulse" 
                  style={{ 
                    width: status === 'verifying' ? '60%' : '100%',
                    transition: 'width 0.3s ease'
                  }}
                />
              </div>
            )}
            
            {/* 操作按钮 */}
            {statusContent.button && (
              <div className="pt-4">
                {statusContent.button}
              </div>
            )}
            
            {/* 额外信息 */}
            <div className="pt-6 border-t border-gray-200 dark:border-gray-700 space-y-4">
              {status === 'pending' && (
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  <p className="font-medium mb-1">未收到验证邮件？</p>
                  <ul className="space-y-1">
                    <li>• 检查垃圾邮件文件夹</li>
                    <li>• 确认邮箱地址是否正确</li>
                    <li>• 等待几分钟后重试</li>
                    <li>• 联系管理员获取帮助</li>
                  </ul>
                </div>
              )}
              
              {status === 'failed' || status === 'expired' ? (
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  <p className="font-medium mb-1">常见问题：</p>
                  <ul className="space-y-1">
                    <li>• 验证链接有效期为24小时</li>
                    <li>• 每个验证链接只能使用一次</li>
                    <li>• 如果问题持续存在，请联系管理员</li>
                  </ul>
                </div>
              ) : null}
              
              {/* 返回登录链接 */}
              <div className="text-center">
                <Link
                  to="/login"
                  className="inline-flex items-center text-sm font-medium text-gray-600 hover:text-gray-500 dark:text-gray-400 dark:hover:text-gray-300"
                >
                  返回登录
                </Link>
              </div>
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

export default VerifyEmailPage;