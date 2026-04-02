import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  User,
  Bell,
  Shield,
  Monitor,
  Moon,
  Sun,
  Save,
  RefreshCw,
  Key,
  Eye,
  EyeOff,
} from 'lucide-react';
import toast from 'react-hot-toast';

interface UserSettings {
  username: string;
  email: string;
  fullName: string;
  language: string;
  timezone: string;
  notifications: {
    email: boolean;
    push: boolean;
    sms: boolean;
    trainingComplete: boolean;
    hardwareAlert: boolean;
    billingReminder: boolean;
  };
}

interface SystemSettings {
  theme: 'light' | 'dark' | 'auto';
  fontSize: 'small' | 'medium' | 'large';
  animation: boolean;
  reduceMotion: boolean;
  autoSave: boolean;
  saveInterval: number;
}

interface SecuritySettings {
  twoFactorAuth: boolean;
  sessionTimeout: number;
  loginNotifications: boolean;
  passwordChangeRequired: boolean;
  ipWhitelist: string[];
}

interface DisplaySettings {
  gridView: boolean;
  compactMode: boolean;
  showSidebar: boolean;
  sidebarWidth: number;
  highlightColor: string;
}

const SettingsPage: React.FC = () => {
  const { user } = useAuth();
  const [userSettings, setUserSettings] = useState<UserSettings>({
    username: user?.username || 'user123',
    email: user?.email || 'user@example.com',
    fullName: '张三',
    language: 'zh-CN',
    timezone: 'Asia/Shanghai',
    notifications: {
      email: true,
      push: true,
      sms: false,
      trainingComplete: true,
      hardwareAlert: true,
      billingReminder: true,
    },
  });

  const [systemSettings, setSystemSettings] = useState<SystemSettings>({
    theme: 'auto',
    fontSize: 'medium',
    animation: true,
    reduceMotion: false,
    autoSave: true,
    saveInterval: 5,
  });

  const [securitySettings, setSecuritySettings] = useState<SecuritySettings>({
    twoFactorAuth: false,
    sessionTimeout: 60,
    loginNotifications: true,
    passwordChangeRequired: false,
    ipWhitelist: ['192.168.1.0/24', '10.0.0.0/8'],
  });

  const [displaySettings, setDisplaySettings] = useState<DisplaySettings>({
    gridView: true,
    compactMode: false,
    showSidebar: true,
    sidebarWidth: 240,
    highlightColor: '#3b82f6',
  });

  const [password, setPassword] = useState({
    current: '',
    new: '',
    confirm: '',
  });
  const [showPassword, setShowPassword] = useState({
    current: false,
    new: false,
    confirm: false,
  });
  const [activeTab, setActiveTab] = useState<'profile' | 'system' | 'security' | 'notifications'>('profile');
  const [isSaving, setIsSaving] = useState(false);

  const languages = [
    { value: 'zh-CN', label: '简体中文' },
    { value: 'zh-TW', label: '繁體中文' },
    { value: 'en-US', label: 'English (US)' },
    { value: 'en-GB', label: 'English (UK)' },
    { value: 'ja-JP', label: '日本語' },
    { value: 'ko-KR', label: '한국어' },
  ];

  const timezones = [
    { value: 'Asia/Shanghai', label: '中国标准时间 (UTC+8)' },
    { value: 'Asia/Tokyo', label: '日本标准时间 (UTC+9)' },
    { value: 'America/New_York', label: '美国东部时间 (UTC-5)' },
    { value: 'Europe/London', label: '格林尼治标准时间 (UTC+0)' },
    { value: 'Australia/Sydney', label: '澳大利亚东部时间 (UTC+10)' },
  ];

  // 从localStorage加载设置
  useEffect(() => {
    try {
      const savedSettings = localStorage.getItem('self_agi_settings');
      if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        
        if (parsedSettings.userSettings) {
          setUserSettings(parsedSettings.userSettings);
        }
        if (parsedSettings.systemSettings) {
          setSystemSettings(parsedSettings.systemSettings);
        }
        if (parsedSettings.securitySettings) {
          setSecuritySettings(parsedSettings.securitySettings);
        }
        if (parsedSettings.displaySettings) {
          setDisplaySettings(parsedSettings.displaySettings);
        }
        
        console.log('设置已从localStorage加载');
      }
    } catch (error) {
      console.error('加载设置失败:', error);
    }
  }, []);

  const handleSaveSettings = async () => {
    setIsSaving(true);
    try {
      // 保存设置到localStorage
      const allSettings = {
        userSettings,
        systemSettings,
        securitySettings,
        displaySettings,
      };
      localStorage.setItem('self_agi_settings', JSON.stringify(allSettings));
      
      // 同时尝试保存到后端API（如果可用）
      try {
        // 这里可以添加后端API调用
        // await settingsApi.saveSettings(allSettings);
      } catch (apiError) {
        console.warn('后端设置API不可用，使用localStorage保存', apiError);
      }
      
      toast.success('设置已保存到本地存储');
    } catch (error) {
      toast.error('保存设置失败');
    } finally {
      setIsSaving(false);
    }
  };

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (password.new !== password.confirm) {
      toast.error('新密码和确认密码不匹配');
      return;
    }
    
    if (password.new.length < 8) {
      toast.error('新密码长度至少8位');
      return;
    }
    
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      setPassword({ current: '', new: '', confirm: '' });
      toast.success('密码修改成功');
    } catch (error) {
      toast.error('修改密码失败');
    }
  };

  const handleToggleTwoFactorAuth = () => {
    setSecuritySettings(prev => ({
      ...prev,
      twoFactorAuth: !prev.twoFactorAuth,
    }));
    toast.success(`双重认证已${securitySettings.twoFactorAuth ? '禁用' : '启用'}`);
  };

  const handleAddIP = () => {
    const newIP = prompt('请输入IP地址或网段 (例如: 192.168.1.0/24):');
    if (newIP && !securitySettings.ipWhitelist.includes(newIP)) {
      setSecuritySettings(prev => ({
        ...prev,
        ipWhitelist: [...prev.ipWhitelist, newIP],
      }));
      toast.success('IP地址已添加');
    }
  };

  const handleRemoveIP = (ip: string) => {
    setSecuritySettings(prev => ({
      ...prev,
      ipWhitelist: prev.ipWhitelist.filter(item => item !== ip),
    }));
    toast.success('IP地址已移除');
  };

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            系统设置
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            管理您的个人偏好和系统配置
          </p>
        </div>
        
        <button
          onClick={handleSaveSettings}
          disabled={isSaving}
          className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isSaving ? (
            <>
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              保存中...
            </>
          ) : (
            <>
              <Save className="w-4 h-4 mr-2" />
              保存设置
            </>
          )}
        </button>
      </div>

      {/* 标签页导航 */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('profile')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'profile'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <User className="w-4 h-4 mr-2" />
              个人资料
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('system')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'system'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Monitor className="w-4 h-4 mr-2" />
              系统设置
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('security')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'security'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Shield className="w-4 h-4 mr-2" />
              安全设置
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('notifications')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'notifications'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Bell className="w-4 h-4 mr-2" />
              通知设置
            </div>
          </button>
        </nav>
      </div>

      {/* 个人资料页面 */}
      {activeTab === 'profile' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="mb-6">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              个人资料设置
            </h2>
            <p className="mt-1 text-gray-600 dark:text-gray-400">
              更新您的个人信息和偏好
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                用户名
              </label>
              <input
                type="text"
                value={userSettings.username}
                onChange={(e) => setUserSettings(prev => ({ ...prev, username: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-700 focus:border-gray-700"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                邮箱地址
              </label>
              <input
                type="email"
                value={userSettings.email}
                onChange={(e) => setUserSettings(prev => ({ ...prev, email: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-700 focus:border-gray-700"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                姓名
              </label>
              <input
                type="text"
                value={userSettings.fullName}
                onChange={(e) => setUserSettings(prev => ({ ...prev, fullName: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-700 focus:border-gray-700"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                语言
              </label>
              <select
                value={userSettings.language}
                onChange={(e) => setUserSettings(prev => ({ ...prev, language: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-gray-700 focus:border-gray-700"
              >
                {languages.map((lang) => (
                  <option key={lang.value} value={lang.value}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                时区
              </label>
              <select
                value={userSettings.timezone}
                onChange={(e) => setUserSettings(prev => ({ ...prev, timezone: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-gray-700 focus:border-gray-700"
              >
                {timezones.map((tz) => (
                  <option key={tz.value} value={tz.value}>
                    {tz.label}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                个人简介
              </label>
              <textarea
                rows={4}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-700 focus:border-gray-700"
                placeholder="介绍一下自己..."
              />
            </div>
          </div>
        </div>
      )}

      {/* 系统设置页面 */}
      {activeTab === 'system' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="mb-6">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              系统设置
            </h2>
            <p className="mt-1 text-gray-600 dark:text-gray-400">
              配置系统的外观和行为
            </p>
          </div>
          
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  主题
                </label>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => setSystemSettings(prev => ({ ...prev, theme: 'light' }))}
                    className={`p-3 rounded-lg border flex flex-col items-center ${
                      systemSettings.theme === 'light'
                        ? 'border-gray-700 bg-gray-700 dark:bg-gray-700/20'
                        : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                    }`}
                  >
                    <Sun className="w-6 h-6 text-gray-500 mb-2" />
                    <span className="text-sm font-medium">浅色</span>
                  </button>
                  
                  <button
                    onClick={() => setSystemSettings(prev => ({ ...prev, theme: 'dark' }))}
                    className={`p-3 rounded-lg border flex flex-col items-center ${
                      systemSettings.theme === 'dark'
                        ? 'border-gray-700 bg-gray-700 dark:bg-gray-700/20'
                        : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                    }`}
                  >
                    <Moon className="w-6 h-6 text-gray-600 mb-2" />
                    <span className="text-sm font-medium">深色</span>
                  </button>
                  
                  <button
                    onClick={() => setSystemSettings(prev => ({ ...prev, theme: 'auto' }))}
                    className={`p-3 rounded-lg border flex flex-col items-center ${
                      systemSettings.theme === 'auto'
                        ? 'border-gray-700 bg-gray-700 dark:bg-gray-700/20'
                        : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                    }`}
                  >
                    <Monitor className="w-6 h-6 text-gray-700 mb-2" />
                    <span className="text-sm font-medium">自动</span>
                  </button>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  字体大小
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {(['small', 'medium', 'large'] as const).map((size) => (
                    <button
                      key={size}
                      onClick={() => setSystemSettings(prev => ({ ...prev, fontSize: size }))}
                      className={`p-3 rounded-lg border ${
                        systemSettings.fontSize === size
                          ? 'border-gray-700 bg-gray-700 dark:bg-gray-700/20'
                          : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                      }`}
                    >
                      <span className="text-sm font-medium capitalize">
                        {size === 'small' ? '小' : size === 'medium' ? '中' : '大'}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={systemSettings.animation}
                  onChange={(e) => setSystemSettings(prev => ({ ...prev, animation: e.target.checked }))}
                  className="h-4 w-4 text-gray-700 focus:ring-gray-700 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                  启用动画效果
                </span>
              </label>
              
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={systemSettings.reduceMotion}
                  onChange={(e) => setSystemSettings(prev => ({ ...prev, reduceMotion: e.target.checked }))}
                  className="h-4 w-4 text-gray-700 focus:ring-gray-700 border-gray-300 rounded"
                />
                <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                  减少动画（无障碍模式）
                </span>
              </label>
              
              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                    自动保存间隔
                  </label>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {systemSettings.saveInterval} 分钟
                  </p>
                </div>
                <input
                  type="range"
                  min="1"
                  max="30"
                  value={systemSettings.saveInterval}
                  onChange={(e) => setSystemSettings(prev => ({ ...prev, saveInterval: parseInt(e.target.value) }))}
                  className="w-48 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 安全设置页面 */}
      {activeTab === 'security' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="mb-6">
              <h2 className="text-lg font-medium text-gray-900 dark:text-white">
                修改密码
              </h2>
              <p className="mt-1 text-gray-600 dark:text-gray-400">
                定期更新密码以提高账户安全性
              </p>
            </div>
            
            <form onSubmit={handleChangePassword} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  当前密码
                </label>
                <div className="relative">
                  <input
                    type={showPassword.current ? 'text' : 'password'}
                    value={password.current}
                    onChange={(e) => setPassword(prev => ({ ...prev, current: e.target.value }))}
                    className="w-full px-3 py-2 pr-10 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-700 focus:border-gray-700"
                    placeholder="输入当前密码"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(prev => ({ ...prev, current: !prev.current }))}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                  >
                    {showPassword.current ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  新密码
                </label>
                <div className="relative">
                  <input
                    type={showPassword.new ? 'text' : 'password'}
                    value={password.new}
                    onChange={(e) => setPassword(prev => ({ ...prev, new: e.target.value }))}
                    className="w-full px-3 py-2 pr-10 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-700 focus:border-gray-700"
                    placeholder="输入新密码（至少8位）"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(prev => ({ ...prev, new: !prev.new }))}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                  >
                    {showPassword.new ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  确认新密码
                </label>
                <div className="relative">
                  <input
                    type={showPassword.confirm ? 'text' : 'password'}
                    value={password.confirm}
                    onChange={(e) => setPassword(prev => ({ ...prev, confirm: e.target.value }))}
                    className="w-full px-3 py-2 pr-10 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white 实现-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-700 focus:border-gray-700"
                    placeholder="再次输入新密码"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(prev => ({ ...prev, confirm: !prev.confirm }))}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                  >
                    {showPassword.confirm ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
              
              <div className="pt-4 border-t border-gray-200 dark:border-gray-700 flex justify-end">
                <button
                  type="submit"
                  className="px-6 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-700"
                >
                  <Key className="w-4 h-4 inline mr-2" />
                  修改密码
                </button>
              </div>
            </form>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-medium text-gray-900 dark:text-white flex items-center">
                    <Shield className="w-4 h-4 mr-2" />
                    双重认证
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    为您的账户添加额外的安全层
                  </p>
                </div>
                <button
                  onClick={handleToggleTwoFactorAuth}
                  className={`px-4 py-2 text-sm font-medium rounded-lg ${
                    securitySettings.twoFactorAuth
                      ? 'bg-gray-600 text-gray-600 hover:bg-gray-600 dark:bg-gray-700/30 dark:text-gray-600 dark:hover:bg-gray-600/50'
                      : 'bg-gray-100 text-gray-800 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700'
                  }`}
                >
                  {securitySettings.twoFactorAuth ? '已启用' : '启用'}
                </button>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                    会话超时
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {securitySettings.sessionTimeout} 分钟
                  </p>
                </div>
                <select
                  value={securitySettings.sessionTimeout}
                  onChange={(e) => setSecuritySettings(prev => ({ ...prev, sessionTimeout: parseInt(e.target.value) }))}
                  className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-gray-700 focus:border-gray-700"
                >
                  <option value="15">15分钟</option>
                  <option value="30">30分钟</option>
                  <option value="60">1小时</option>
                  <option value="120">2小时</option>
                  <option value="240">4小时</option>
                </select>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                    登录通知
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    在新设备登录时发送通知
                  </p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={securitySettings.loginNotifications}
                    onChange={(e) => setSecuritySettings(prev => ({ ...prev, loginNotifications: e.target.checked }))}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-gray-700 dark:peer-focus:ring-gray-700 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-gray-700"></div>
                </label>
              </div>
              
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                      IP白名单
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      限制账户只能从特定IP地址访问
                    </p>
                  </div>
                  <button
                    onClick={handleAddIP}
                    className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-700 hover:text-gray-700 dark:hover:text-gray-700"
                  >
                    添加IP
                  </button>
                </div>
                
                <div className="space-y-2">
                  {securitySettings.ipWhitelist.map((ip) => (
                    <div key={ip} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                      <span className="text-sm text-gray-900 dark:text-white font-mono">{ip}</span>
                      <button
                        onClick={() => handleRemoveIP(ip)}
                        className="text-gray-800 hover:text-gray-800 dark:text-gray-800 dark:hover:text-gray-800"
                      >
                        移除
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 通知设置页面 */}
      {activeTab === 'notifications' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="mb-6">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              通知设置
            </h2>
            <p className="mt-1 text-gray-600 dark:text-gray-400">
              管理您接收通知的方式和类型
            </p>
          </div>
          
          <div className="space-y-6">
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                通知渠道
              </h3>
              
              <label className="flex items-center justify-between p-3 border border-gray-300 dark:border-gray-600 rounded-lg hover:border-gray-400 dark:hover:border-gray-500 cursor-pointer">
                <div className="flex items-center">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-gray-700 to-cyan-500 flex items-center justify-center">
                    <Bell className="w-4 h-4 text-white" />
                  </div>
                  <div className="ml-3">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      邮件通知
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      通过电子邮件接收重要通知
                    </div>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={userSettings.notifications.email}
                  onChange={(e) => setUserSettings(prev => ({
                    ...prev,
                    notifications: { ...prev.notifications, email: e.target.checked }
                  }))}
                  className="h-4 w-4 text-gray-700 focus:ring-gray-700 border-gray-300 rounded"
                />
              </label>
              
              <label className="flex items-center justify-between p-3 border border-gray-300 dark:border-gray-600 rounded-lg hover:border-gray-400 dark:hover:border-gray-500 cursor-pointer">
                <div className="flex items-center">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-gray-600 to-emerald-500 flex items-center justify-center">
                    <Bell className="w-4 h-4 text-white" />
                  </div>
                  <div className="ml-3">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      推送通知
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      在浏览器中接收实时推送通知
                    </div>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={userSettings.notifications.push}
                  onChange={(e) => setUserSettings(prev => ({
                    ...prev,
                    notifications: { ...prev.notifications, push: e.target.checked }
                  }))}
                  className="h-4 w-4 text-gray-700 focus:ring-gray-700 border-gray-300 rounded"
                />
              </label>
              
              <label className="flex items-center justify-between p-3 border border-gray-300 dark:border-gray-600 rounded-lg hover:border-gray-400 dark:hover:border-gray-500 cursor-pointer">
                <div className="flex items-center">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-gray-600 to-gray-500 flex items-center justify-center">
                    <Bell className="w-4 h-4 text-white" />
                  </div>
                  <div className="ml-3">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      短信通知
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      通过短信接收紧急通知
                    </div>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={userSettings.notifications.sms}
                  onChange={(e) => setUserSettings(prev => ({
                    ...prev,
                    notifications: { ...prev.notifications, sms: e.target.checked }
                  }))}
                  className="h-4 w-4 text-gray-700 focus:ring-gray-700 border-gray-300 rounded"
                />
              </label>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                通知类型
              </h3>
              
              {([
                { key: 'trainingComplete' as const, label: '训练完成通知', description: '模型训练任务完成时通知' },
                { key: 'hardwareAlert' as const, label: '硬件警报', description: '硬件设备出现异常时通知' },
                { key: 'billingReminder' as const, label: '账单提醒', description: '账单到期前提醒' },
              ] as const).map(({ key, label, description }) => (
                <label key={key} className="flex items-center justify-between p-3 border border-gray-300 dark:border-gray-600 rounded-lg hover:border-gray-400 dark:hover:border-gray-500 cursor-pointer">
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      {label}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      {description}
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={userSettings.notifications[key]}
                    onChange={(e) => setUserSettings(prev => ({
                      ...prev,
                      notifications: { ...prev.notifications, [key]: e.target.checked }
                    }))}
                    className="h-4 w-4 text-gray-700 focus:ring-gray-700 border-gray-300 rounded"
                  />
                </label>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SettingsPage;