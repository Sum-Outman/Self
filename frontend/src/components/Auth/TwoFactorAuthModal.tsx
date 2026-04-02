import React, { useState, useEffect } from 'react';
import { Shield, Mail, AlertCircle, RefreshCw } from 'lucide-react';
import toast from 'react-hot-toast';
import { Button, Input, Modal } from '../UI';

interface TwoFactorAuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onVerify: (code: string) => Promise<void>;
  method: 'email' | 'totp';
  tempToken?: string;
  message?: string;
}

const TwoFactorAuthModal: React.FC<TwoFactorAuthModalProps> = ({
  isOpen,
  onClose,
  onVerify,
  method,
  tempToken,
  message,
}) => {
  const [code, setCode] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [remainingTime, setRemainingTime] = useState(600); // 10分钟，单位秒
  const [isResending, setIsResending] = useState(false);
  const [showBackupCodes, setShowBackupCodes] = useState(false);
  const [backupCode, setBackupCode] = useState('');

  // 倒计时计时器
  useEffect(() => {
    if (!isOpen || remainingTime <= 0) return;

    const timer = setInterval(() => {
      setRemainingTime(prev => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [isOpen, remainingTime]);

  // 格式化剩余时间
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!code.trim()) {
      toast.error('请输入验证码');
      return;
    }
    
    if (code.length !== 6) {
      toast.error('验证码应为6位数字');
      return;
    }
    
    try {
      setIsLoading(true);
      await onVerify(code);
    } finally {
      setIsLoading(false);
    }
  };

  const handleBackupSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!backupCode.trim()) {
      toast.error('请输入备份代码');
      return;
    }
    
    try {
      setIsLoading(true);
      await onVerify(backupCode);
    } finally {
      setIsLoading(false);
    }
  };

  const handleResendCode = async () => {
    try {
      setIsResending(true);
      
      // 重新发送验证码的逻辑
      if (method === 'email') {
        // 对于邮箱2FA，需要调用API重新发送验证码
        // 目前后端没有相关API，显示提示信息
        toast('重新发送验证码功能需要后端API支持');
      } else {
        // 对于TOTP应用，验证码由应用生成，无需重新发送
        toast('TOTP验证码由您的身份验证应用生成，请查看应用获取最新验证码');
      }
    } catch (error) {
      toast.error('发送验证码失败');
    } finally {
      setIsResending(false);
    }
  };

  // 如果是TOTP方法，显示二维码扫描提示
  const renderMethodDescription = () => {
    if (method === 'totp') {
      return (
        <div className="space-y-4">
          <div className="text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              请使用身份验证器应用扫描二维码，或手动输入密钥
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-300 dark:border-gray-700 inline-block">
              <div className="w-48 h-48 bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                {/* 这里应该显示实际的二维码 */}
                <div className="text-center">
                  <Shield className="w-16 h-16 text-gray-400 mx-auto mb-2" />
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {tempToken ? '二维码已加载' : '等待二维码生成...'}
                  </p>
                </div>
              </div>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              扫描后，应用将显示6位验证码
            </p>
          </div>
        </div>
      );
    }
    
    // Email方法
    return (
      <div className="space-y-4">
        <div className="text-center">
          <Mail className="w-12 h-12 text-gray-500 mx-auto mb-4" />
          <p className="text-sm text-gray-600 dark:text-gray-400">
            我们已向您的注册邮箱发送了6位验证码
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
            请在10分钟内输入验证码完成验证
          </p>
        </div>
        
        <div className="bg-gray-800 dark:bg-gray-900/20 border border-gray-600 dark:border-gray-900 rounded-lg p-4">
          <div className="flex items-start">
            <AlertCircle className="w-5 h-5 text-gray-700 dark:text-gray-400 mt-0.5 mr-3 flex-shrink-0" />
            <div className="text-sm text-gray-800 dark:text-gray-500">
              <p className="font-medium">找不到验证邮件？</p>
              <ul className="list-disc pl-4 mt-1 space-y-1">
                <li>请检查垃圾邮件文件夹</li>
                <li>确保您输入的邮箱地址正确</li>
                <li>如果仍然未收到，可尝试重新发送验证码</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="双因素身份验证"
      size="lg"
      disableEscapeClose={true}
      disableOverlayClose={true}
    >
      {message && (
        <div className="mb-6">
          <div className="text-center">
            <p className="text-gray-700 dark:text-gray-300">{message}</p>
          </div>
        </div>
      )}

      {!showBackupCodes ? (
        <>
          {renderMethodDescription()}
          
          <form onSubmit={handleSubmit} className="mt-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                6位验证码
              </label>
              <Input
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                placeholder="请输入6位验证码"
                maxLength={6}
                pattern="\d{6}"
                inputMode="numeric"
                autoFocus
                fullWidth
              />
              <div className="flex justify-between mt-2">
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  剩余时间: {formatTime(remainingTime)}
                </span>
                <button
                  type="button"
                  onClick={() => setShowBackupCodes(true)}
                  className="text-xs text-gray-600 hover:text-gray-500 dark:text-gray-400 dark:hover:text-gray-300"
                >
                  使用备份代码
                </button>
              </div>
            </div>
            
            <div className="flex space-x-3">
              <Button
                type="submit"
                variant="primary"
                loading={isLoading}
                disabled={isLoading || code.length !== 6}
                fullWidth
              >
                验证
              </Button>
              
              {method === 'email' && (
                <Button
                  type="button"
                  variant="secondary"
                  onClick={handleResendCode}
                  loading={isResending}
                  disabled={isResending || remainingTime <= 0}
                  className="flex-1"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  重新发送
                </Button>
              )}
            </div>
          </form>
        </>
      ) : (
        <div className="space-y-6">
          <div className="text-center">
            <Shield className="w-12 h-12 text-gray-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              使用备份代码
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              如果您无法获取验证码，可以使用您的备份代码进行登录
            </p>
            <div className="bg-gray-800 dark:bg-gray-900/20 border border-gray-600 dark:border-gray-900 rounded-lg p-4 mt-4">
              <div className="flex items-start">
                <AlertCircle className="w-5 h-5 text-gray-700 dark:text-gray-400 mt-0.5 mr-3 flex-shrink-0" />
                <div className="text-sm text-gray-800 dark:text-gray-500">
                  <p className="font-medium">重要提示</p>
                  <ul className="list-disc pl-4 mt-1 space-y-1">
                    <li>每个备份代码只能使用一次</li>
                    <li>使用后请立即生成新的备份代码</li>
                    <li>请妥善保管您的备份代码</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
          
          <form onSubmit={handleBackupSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                备份代码
              </label>
              <Input
                value={backupCode}
                onChange={(e) => setBackupCode(e.target.value.trim())}
                placeholder="请输入备份代码"
                fullWidth
              />
            </div>
            
            <div className="flex space-x-3">
              <Button
                type="submit"
                variant="primary"
                loading={isLoading}
                disabled={isLoading || !backupCode.trim()}
                fullWidth
              >
                使用备份代码验证
              </Button>
              
              <Button
                type="button"
                variant="secondary"
                onClick={() => setShowBackupCodes(false)}
                fullWidth
              >
                返回验证码验证
              </Button>
            </div>
          </form>
        </div>
      )}
      
      <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            启用双因素认证可显著提高账户安全性
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            如需帮助，请联系管理员或查看帮助文档
          </p>
        </div>
      </div>
    </Modal>
  );
};

export default TwoFactorAuthModal;