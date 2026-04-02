import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { authService } from '../services/api/auth';
import { Link } from 'react-router-dom';
import toast from 'react-hot-toast';
import {
  User,
  Mail,
  Calendar,
  CheckCircle,
  XCircle,
  Edit,
  Save,
  Shield,
  Bell,
  CreditCard,
  Trash2,
  Download,
} from 'lucide-react';

interface UserProfile {
  id: string;
  username: string;
  email: string;
  full_name: string;
  is_active: boolean;
  is_admin: boolean;
  role: string;
  created_at: string;
  updated_at: string;
  last_login?: string;
}

interface ProfileFormData {
  username: string;
  email: string;
  full_name: string;
  current_password: string;
  new_password: string;
  confirm_password: string;
}

const ProfilePage: React.FC = () => {
  const { user: authUser, logout } = useAuth();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [activeTab, setActiveTab] = useState<'profile' | 'security' | 'preferences' | 'subscription'>('profile');
  const [formData, setFormData] = useState<ProfileFormData>({
    username: '',
    email: '',
    full_name: '',
    current_password: '',
    new_password: '',
    confirm_password: '',
  });
  const [passwordStrength, setPasswordStrength] = useState(0);

  // 加载用户资料
  useEffect(() => {
    const loadUserProfile = async () => {
      if (!authUser) {
        setLoading(false);
        return;
      }
      
      try {
        setLoading(true);
        // 从后端API获取完整的用户资料
        const userData = await authService.getCurrentUser();
        
        const userProfile: UserProfile = {
          id: userData.id,
          username: userData.username,
          email: userData.email,
          full_name: userData.full_name || userData.name || userData.username,
          is_active: userData.is_active,
          is_admin: userData.is_admin || false,
          role: userData.role || (userData.is_admin ? 'admin' : 'user'),
          created_at: userData.created_at,
          updated_at: userData.updated_at,
          last_login: userData.last_login,
        };
        setProfile(userProfile);
        setFormData({
          username: userData.username,
          email: userData.email,
          full_name: userData.full_name || userData.name || userData.username,
          current_password: '',
          new_password: '',
          confirm_password: '',
        });
      } catch (error) {
        console.error('加载用户资料失败:', error);
        toast.error('加载用户资料失败，请稍后重试');
        // 降级：使用认证上下文中的用户数据
        const userProfile: UserProfile = {
          id: authUser.id,
          username: authUser.username,
          email: authUser.email,
          full_name: authUser.full_name || authUser.name || authUser.username,
          is_active: authUser.is_active,
          is_admin: authUser.is_admin || false,
          role: authUser.role || (authUser.is_admin ? 'admin' : 'user'),
          created_at: authUser.created_at,
          updated_at: authUser.updated_at,
          last_login: authUser.last_login,
        };
        setProfile(userProfile);
        setFormData({
          username: authUser.username,
          email: authUser.email,
          full_name: authUser.full_name || authUser.name || authUser.username,
          current_password: '',
          new_password: '',
          confirm_password: '',
        });
      } finally {
        setLoading(false);
      }
    };
    
    loadUserProfile();
  }, [authUser]);

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

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));

    if (name === 'new_password') {
      setPasswordStrength(calculatePasswordStrength(value));
    }
  };

  const handleSaveProfile = async () => {
    // 验证表单数据
    if (!formData.email.trim()) {
      toast.error('请输入邮箱地址');
      return;
    }

    // 邮箱格式验证
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      toast.error('请输入有效的邮箱地址');
      return;
    }

    // 如果更改密码，验证密码
    if (formData.new_password) {
      if (!formData.current_password) {
        toast.error('请输入当前密码以更改密码');
        return;
      }

      if (formData.new_password !== formData.confirm_password) {
        toast.error('新密码和确认密码不一致');
        return;
      }

      if (formData.new_password.length < 8) {
        toast.error('密码长度至少为8位');
        return;
      }

      const hasUpperCase = /[A-Z]/.test(formData.new_password);
      const hasLowerCase = /[a-z]/.test(formData.new_password);
      const hasNumbers = /\d/.test(formData.new_password);
      const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(formData.new_password);

      if (!hasUpperCase || !hasLowerCase || !hasNumbers || !hasSpecialChar) {
        toast.error('密码必须包含大小写字母、数字和特殊字符');
        return;
      }
    }

    try {
      setSaving(true);
      
      // 调用后端API更新用户资料
      const updateData: any = {};
      if (formData.email !== profile?.email) {
        updateData.email = formData.email;
      }
      if (formData.full_name !== profile?.full_name) {
        updateData.full_name = formData.full_name;
      }
      
      if (Object.keys(updateData).length > 0) {
        await authService.updateUser(updateData);
      }
      
      // 如果更改了密码，调用密码更新API
      if (formData.new_password) {
        await authService.changePassword(formData.current_password, formData.new_password);
      }

      // 更新本地状态
      if (profile) {
        const updatedProfile = {
          ...profile,
          email: formData.email,
          full_name: formData.full_name,
          updated_at: new Date().toISOString(),
        };
        setProfile(updatedProfile);
      }

      toast.success('个人资料更新成功');
      setEditing(false);
      
      // 清空密码字段
      setFormData(prev => ({
        ...prev,
        current_password: '',
        new_password: '',
        confirm_password: '',
      }));
      setPasswordStrength(0);
    } catch (error: any) {
      console.error('更新个人资料失败:', error);
      const message = error.response?.data?.message || error.message || '更新个人资料失败';
      toast.error(message);
    } finally {
      setSaving(false);
    }
  };

  const handleCancelEdit = () => {
    if (profile) {
      setFormData({
        username: profile.username,
        email: profile.email,
        full_name: profile.full_name,
        current_password: '',
        new_password: '',
        confirm_password: '',
      });
    }
    setEditing(false);
    setPasswordStrength(0);
  };

  const handleDeleteAccount = async () => {
    if (!window.confirm('确定要删除账户吗？此操作不可撤销，所有数据将被永久删除。')) {
      return;
    }

    try {
      setDeleting(true);
      // 调用后端API删除账户
      await authService.deleteAccount();
      
      toast.success('账户已删除');
      // 登出并重定向到首页
      logout();
    } catch (error: any) {
      console.error('删除账户失败:', error);
      const message = error.response?.data?.message || error.message || '删除账户失败';
      toast.error(message);
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-600"></div>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <XCircle className="w-16 h-16 text-gray-800 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            无法加载用户资料
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            请尝试刷新页面或重新登录
          </p>
        </div>
      </div>
    );
  }

  // 格式化日期
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // 密码强度指示器
  const getPasswordStrengthLabel = (strength: number) => {
    if (strength <= 1) return { label: '极弱', color: 'text-gray-800', bg: 'bg-gray-900' };
    if (strength === 2) return { label: '弱', color: 'text-orange-500', bg: 'bg-orange-500' };
    if (strength === 3) return { label: '中等', color: 'text-gray-500', bg: 'bg-gray-800' };
    if (strength === 4) return { label: '强', color: 'text-gray-600', bg: 'bg-gray-700' };
    return { label: '极强', color: 'text-emerald-500', bg: 'bg-emerald-500' };
  };

  const strengthInfo = getPasswordStrengthLabel(passwordStrength);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        {/* 头部 */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                个人资料
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-2">
                管理您的个人信息、安全设置和偏好
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              {!editing ? (
                <button
                  onClick={() => setEditing(true)}
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-600 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-700"
                >
                  <Edit className="w-4 h-4 mr-2" />
                  编辑资料
                </button>
              ) : (
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleCancelEdit}
                    className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
                  >
                    取消
                  </button>
                  <button
                    onClick={handleSaveProfile}
                    disabled={saving}
                    className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-600 to-emerald-600 rounded-lg hover:from-gray-600 hover:to-emerald-700 disabled:opacity-50"
                  >
                    <Save className="w-4 h-4 mr-2" />
                    {saving ? '保存中...' : '保存更改'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* 侧边栏导航 */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4">
              <nav className="space-y-1">
                <button
                  onClick={() => setActiveTab('profile')}
                  className={`w-full flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                    activeTab === 'profile'
                      ? 'bg-gray-50 dark:bg-gray-900/20 text-gray-700 dark:text-gray-300'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                  }`}
                >
                  <User className="w-4 h-4 mr-3" />
                  个人资料
                </button>
                
                <button
                  onClick={() => setActiveTab('security')}
                  className={`w-full flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                    activeTab === 'security'
                      ? 'bg-gray-50 dark:bg-gray-900/20 text-gray-700 dark:text-gray-300'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                  }`}
                >
                  <Shield className="w-4 h-4 mr-3" />
                  账户安全
                </button>
                
                <button
                  onClick={() => setActiveTab('preferences')}
                  className={`w-full flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                    activeTab === 'preferences'
                      ? 'bg-gray-50 dark:bg-gray-900/20 text-gray-700 dark:text-gray-300'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                  }`}
                >
                  <Bell className="w-4 h-4 mr-3" />
                  偏好设置
                </button>
                
                <button
                  onClick={() => setActiveTab('subscription')}
                  className={`w-full flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                    activeTab === 'subscription'
                      ? 'bg-gray-50 dark:bg-gray-900/20 text-gray-700 dark:text-gray-300'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                  }`}
                >
                  <CreditCard className="w-4 h-4 mr-3" />
                  订阅管理
                </button>
              </nav>

              <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
                <div className="px-4">
                  <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                    账户状态
                  </div>
                  <div className="flex items-center">
                    <div className={`w-2 h-2 rounded-full mr-2 ${
                      profile.is_active ? 'bg-gray-700' : 'bg-gray-900'
                    }`}></div>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {profile.is_active ? '账户正常' : '账户已禁用'}
                    </span>
                  </div>
                  <div className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                    角色: {profile.role === 'admin' ? '管理员' : '普通用户'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 主要内容 */}
          <div className="lg:col-span-3">
            {activeTab === 'profile' && (
              <div className="space-y-6">
                {/* 基本信息卡片 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-lg font-medium text-gray-900 dark:text-white">
                      基本信息
                    </h2>
                    <div className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-gray-100 to-gray-100 dark:from-gray-900/30 dark:to-gray-900/30 text-gray-800 dark:text-gray-300">
                      {profile.is_admin ? '管理员账户' : '普通账户'}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        用户名
                      </label>
                      {editing ? (
                        <input
                          type="text"
                          name="username"
                          value={formData.username}
                          onChange={handleInputChange}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          disabled // 用户名通常不可更改
                        />
                      ) : (
                        <div className="flex items-center">
                          <User className="w-5 h-5 text-gray-400 mr-2" />
                          <span className="text-gray-900 dark:text-white font-medium">
                            {profile.username}
                          </span>
                        </div>
                      )}
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        邮箱地址
                      </label>
                      {editing ? (
                        <input
                          type="email"
                          name="email"
                          value={formData.email}
                          onChange={handleInputChange}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                      ) : (
                        <div className="flex items-center">
                          <Mail className="w-5 h-5 text-gray-400 mr-2" />
                          <span className="text-gray-900 dark:text-white font-medium">
                            {profile.email}
                          </span>
                          {profile.email.endsWith('@verified.com') && (
                            <CheckCircle className="w-4 h-4 text-gray-600 ml-2" />
                          )}
                        </div>
                      )}
                    </div>
                    
                    <div className="md:col-span-2">
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        姓名
                      </label>
                      {editing ? (
                        <input
                          type="text"
                          name="full_name"
                          value={formData.full_name}
                          onChange={handleInputChange}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="请输入您的姓名"
                        />
                      ) : (
                        <div className="text-gray-900 dark:text-white font-medium">
                          {profile.full_name || '未设置'}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* 账户信息卡片 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
                    账户信息
                  </h2>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <div className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">
                        账户创建时间
                      </div>
                      <div className="flex items-center text-gray-900 dark:text-white">
                        <Calendar className="w-4 h-4 mr-2 text-gray-400" />
                        {formatDate(profile.created_at)}
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">
                        最后更新时间
                      </div>
                      <div className="flex items-center text-gray-900 dark:text-white">
                        <Calendar className="w-4 h-4 mr-2 text-gray-400" />
                        {formatDate(profile.updated_at)}
                      </div>
                    </div>
                    
                    {profile.last_login && (
                      <div>
                        <div className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">
                          最后登录时间
                        </div>
                        <div className="flex items-center text-gray-900 dark:text-white">
                          <Calendar className="w-4 h-4 mr-2 text-gray-400" />
                          {formatDate(profile.last_login)}
                        </div>
                      </div>
                    )}
                    
                    <div>
                      <div className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">
                        账户状态
                      </div>
                      <div className="flex items-center">
                        <div className={`w-2 h-2 rounded-full mr-2 ${
                          profile.is_active ? 'bg-gray-700' : 'bg-gray-900'
                        }`}></div>
                        <span className={`font-medium ${
                          profile.is_active ? 'text-gray-600 dark:text-gray-600' : 'text-gray-800 dark:text-gray-800'
                        }`}>
                          {profile.is_active ? '正常' : '已禁用'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'security' && (
              <div className="space-y-6">
                {/* 更改密码卡片 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
                    更改密码
                  </h2>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        当前密码
                      </label>
                      <input
                        type="password"
                        name="current_password"
                        value={formData.current_password}
                        onChange={handleInputChange}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        placeholder="请输入当前密码"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        新密码
                      </label>
                      <input
                        type="password"
                        name="new_password"
                        value={formData.new_password}
                        onChange={handleInputChange}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        placeholder="请输入新密码"
                      />
                      {formData.new_password && (
                        <div className="mt-2">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm text-gray-600 dark:text-gray-400">
                              密码强度:
                            </span>
                            <span className={`text-sm font-medium ${strengthInfo.color}`}>
                              {strengthInfo.label}
                            </span>
                          </div>
                          <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full ${strengthInfo.bg}`}
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
                        确认新密码
                      </label>
                      <input
                        type="password"
                        name="confirm_password"
                        value={formData.confirm_password}
                        onChange={handleInputChange}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        placeholder="请再次输入新密码"
                      />
                      {formData.new_password && formData.confirm_password && (
                        <div className="mt-2">
                          {formData.new_password === formData.confirm_password ? (
                            <div className="flex items-center text-gray-600 dark:text-gray-600">
                              <CheckCircle className="w-4 h-4 mr-1" />
                              <span className="text-sm">密码匹配</span>
                            </div>
                          ) : (
                            <div className="flex items-center text-gray-800 dark:text-gray-800">
                              <XCircle className="w-4 h-4 mr-1" />
                              <span className="text-sm">密码不匹配</span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* 双因素认证卡片 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
                    双因素认证 (2FA)
                  </h2>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-gray-900 dark:text-white font-medium mb-1">
                        双因素认证
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        为您的账户添加额外的安全层
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        未启用
                      </span>
                      <button className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-600 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-700">
                        启用
                      </button>
                    </div>
                  </div>
                </div>

                {/* 危险区域卡片 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-800 dark:border-gray-800/30 p-6">
                  <h2 className="text-lg font-medium text-gray-900 dark:text-gray-800 mb-6">
                    危险区域
                  </h2>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-gray-900 dark:text-white font-medium mb-1">
                          删除账户
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          永久删除您的账户和所有数据
                        </div>
                      </div>
                      
                      <button
                        onClick={handleDeleteAccount}
                        disabled={deleting}
                        className={`px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-800 to-rose-600 rounded-lg hover:from-gray-800 hover:to-rose-700 ${deleting ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        {deleting ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white inline mr-2"></div>
                            删除中...
                          </>
                        ) : (
                          <>
                            <Trash2 className="w-4 h-4 inline mr-2" />
                            删除账户
                          </>
                        )}
                      </button>
                    </div>
                    
                    <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        <p className="mb-2">
                          <strong>警告:</strong> 删除账户是不可逆的操作。所有数据，包括：
                        </p>
                        <ul className="list-disc pl-5 space-y-1">
                          <li>个人资料信息</li>
                          <li>聊天历史记录</li>
                          <li>训练数据</li>
                          <li>API密钥</li>
                          <li>支付记录</li>
                        </ul>
                        <p className="mt-2">都将被永久删除且无法恢复。</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'preferences' && (
              <div className="space-y-6">
                {/* 通知设置卡片 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
                    通知设置
                  </h2>
                  
                  <div className="space-y-4">
                    {[
                      { id: 'email_notifications', label: '邮件通知', description: '接收重要账户和安全通知' },
                      { id: 'marketing_emails', label: '营销邮件', description: '接收产品更新和促销信息' },
                      { id: 'push_notifications', label: '推送通知', description: '在设备上接收实时通知' },
                      { id: 'sms_notifications', label: '短信通知', description: '接收短信验证码和重要提醒' },
                    ].map((setting) => (
                      <div key={setting.id} className="flex items-center justify-between">
                        <div>
                          <div className="text-gray-900 dark:text-white font-medium mb-1">
                            {setting.label}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            {setting.description}
                          </div>
                        </div>
                        
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input type="checkbox" className="sr-only peer" />
                          <div className="w-11 h-6 bg-gray-200 dark:bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-gray-700 dark:peer-focus:ring-gray-700 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-gray-700"></div>
                        </label>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 隐私设置卡片 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
                    隐私设置
                  </h2>
                  
                  <div className="space-y-4">
                    {[
                      { id: 'public_profile', label: '公开个人资料', description: '允许其他用户查看您的个人资料' },
                      { id: 'activity_status', label: '在线状态', description: '向其他用户显示您的在线状态' },
                      { id: 'data_collection', label: '数据收集', description: '允许我们收集使用数据以改进服务' },
                      { id: 'data_sharing', label: '数据共享', description: '允许与合作伙伴共享匿名数据' },
                    ].map((setting) => (
                      <div key={setting.id} className="flex items-center justify-between">
                        <div>
                          <div className="text-gray-900 dark:text-white font-medium mb-1">
                            {setting.label}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            {setting.description}
                          </div>
                        </div>
                        
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input type="checkbox" className="sr-only peer" />
                          <div className="w-11 h-6 bg-gray-200 dark:bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-gray-700 dark:peer-focus:ring-gray-700 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-gray-700"></div>
                        </label>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 数据导出卡片 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-1">
                        数据导出
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        导出您的个人数据
                      </p>
                    </div>
                    
                    <button className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-600 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-700">
                      <Download className="w-4 h-4 mr-2" />
                      导出数据
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'subscription' && (
              <div className="space-y-6">
                {/* 当前订阅卡片 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
                    当前订阅
                  </h2>
                  
                  <div className="bg-gradient-to-r from-gray-50 to-gray-50 dark:from-gray-900/20 dark:to-gray-900/20 rounded-xl p-6 mb-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <div className="text-2xl font-bold text-gray-900 dark:text-white">
                          专业版
                        </div>
                        <div className="text-gray-600 dark:text-gray-400 mt-1">
                          每月 ¥199
                        </div>
                      </div>
                      
                      <div className="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium bg-gray-600 dark:bg-gray-700/30 text-gray-600 dark:text-gray-300">
                        <CheckCircle className="w-4 h-4 mr-2" />
                        活跃中
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">API调用</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-white">10,000/月</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">存储空间</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-white">50 GB</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">优先级支持</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-white">是</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">到期时间</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-white">2026-04-11</div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    <button className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-600 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-700">
                      升级套餐
                    </button>
                    <button className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
                      取消订阅
                    </button>
                    <button className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-700 bg-gray-700 dark:bg-gray-700/20 rounded-lg hover:bg-gray-700 dark:hover:bg-gray-700/30">
                      发票历史
                    </button>
                  </div>
                </div>

                {/* 支付方式卡片 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                      支付方式
                    </h3>
                    <button className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-600 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-700">
                      添加支付方式
                    </button>
                  </div>
                  
                  <div className="space-y-4">
                    {[
                      { type: '信用卡', last4: '4242', expiry: '12/26', default: true },
                      { type: 'PayPal', email: 'user@example.com', default: false },
                    ].map((method, index) => (
                      <div key={index} className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                        <div className="flex items-center">
                          <CreditCard className="w-6 h-6 text-gray-400 mr-3" />
                          <div>
                            <div className="text-gray-900 dark:text-white font-medium">
                              {method.type}
                              {method.default && (
                                <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-600 dark:bg-gray-700/30 text-gray-600 dark:text-gray-300">
                                  默认
                                </span>
                              )}
                            </div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">
                              {method.last4 ? `**** **** **** ${method.last4}` : method.email}
                              {method.expiry && ` · 到期 ${method.expiry}`}
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          {!method.default && (
                            <button className="text-sm text-gray-700 dark:text-gray-700 hover:text-gray-700 dark:hover:text-gray-700">
                              设为默认
                            </button>
                          )}
                          <button className="text-sm text-gray-800 dark:text-gray-800 hover:text-gray-800 dark:hover:text-gray-800">
                            移除
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 底部信息 */}
        <div className="mt-8 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>
            需要帮助？查看我们的{' '}
            <Link to="/terms" className="text-gray-600 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300">
              服务条款
            </Link>
            {' '}和{' '}
            <Link to="/privacy" className="text-gray-600 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300">
              隐私政策
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;