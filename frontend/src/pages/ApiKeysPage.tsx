import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  Key,
  Plus,
  Trash2,
  Eye,
  EyeOff,
  Copy,
  RefreshCw,
  Clock,
  AlertCircle,
  CheckCircle,
  XCircle,
  BarChart3,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { ApiKey } from '../types/auth';

// 前端UI接口扩展（包含计算字段）
interface UiApiKey extends ApiKey {
  keyMasked: string;
  createdAt: Date;
  expiresAt: Date | null;
  lastUsed: Date | null; // 后端没有提供，先设为null
  usageCount: number; // 后端没有提供，先设为0
  permissions: string[]; // 后端没有提供，先设为默认权限
  status: 'active' | 'expired' | 'revoked' | 'limited';
}

interface UsageStat {
  date: string;
  requests: number;
  errors: number;
}

const ApiKeysPage: React.FC = () => {
  const { apiKeys: backendKeys, refreshApiKeys, createApiKey, deleteApiKey } = useAuth();
  const [uiApiKeys, setUiApiKeys] = useState<UiApiKey[]>([]);
  const [usageStats, _setUsageStats] = useState<UsageStat[]>([
    { date: '2024-02-18', requests: 1234, errors: 23 },
    { date: '2024-02-19', requests: 1456, errors: 34 },
    { date: '2024-02-20', requests: 1678, errors: 12 },
    { date: '2024-02-21', requests: 1890, errors: 45 },
    { date: '2024-02-22', requests: 1567, errors: 28 },
    { date: '2024-02-23', requests: 1789, errors: 19 },
    { date: '2024-02-24', requests: 1654, errors: 32 },
  ]);

  const [newKeyName, setNewKeyName] = useState('');
  const [selectedPermissions, setSelectedPermissions] = useState<string[]>(['read']);
  const [showKey, setShowKey] = useState<Record<string, boolean>>({});
  const [activeTab, setActiveTab] = useState<'keys' | 'usage' | 'settings'>('keys');
  const [defaultExpirationDays, setDefaultExpirationDays] = useState<number>(365);
  const [isLoading, setIsLoading] = useState({
    keys: false,
    create: false,
  });

  const availablePermissions = [
    { id: 'read', name: '读取数据', description: '允许读取知识库和模型信息' },
    { id: 'write', name: '写入数据', description: '允许上传文档和更新数据' },
    { id: 'train', name: '训练模型', description: '允许启动和监控模型训练' },
    { id: 'query', name: '查询API', description: '允许调用AGI对话和推理接口' },
    { id: 'manage', name: '管理权限', description: '允许管理其他API密钥' },
    { id: 'billing', name: '账单权限', description: '允许查看和支付账单' },
  ];

  // 将后端ApiKey转换为UiApiKey
  const convertToUiKey = (backendKey: ApiKey): UiApiKey => {
    const key = backendKey.key;
    // 如果key已经包含省略号（来自列表响应），则直接使用；否则创建掩码版本
    const keyMasked = key.includes('...') ? key : `${key.substring(0, 10)}••••••••••••••••`;
    
    // 确定状态
    let status: UiApiKey['status'] = 'active';
    if (!backendKey.is_active) {
      status = 'revoked';
    } else if (backendKey.expires_at && new Date(backendKey.expires_at) < new Date()) {
      status = 'expired';
    } else if (backendKey.rate_limit && backendKey.rate_limit < 100) {
      status = 'limited';
    }
    
    // 确定权限（基于速率限制）
    const permissions = [];
    const rateLimit = backendKey.rate_limit || 100;
    if (rateLimit >= 50) permissions.push('read');
    if (rateLimit >= 100) permissions.push('query');
    if (rateLimit >= 200) permissions.push('write');
    if (rateLimit >= 500) permissions.push('train');
    if (rateLimit >= 1000) permissions.push('manage');
    
    return {
      ...backendKey,
      keyMasked,
      createdAt: new Date(backendKey.created_at),
      expiresAt: backendKey.expires_at ? new Date(backendKey.expires_at) : null,
      lastUsed: backendKey.last_used ? new Date(backendKey.last_used) : null,
      usageCount: 0, // 后端没有提供使用计数
      permissions,
      status,
    };
  };

  // 当backendKeys变化时更新UI数据
  useEffect(() => {
    const uiKeys = (backendKeys || []).map(convertToUiKey);
    setUiApiKeys(uiKeys);
  }, [backendKeys]);

  // 初始加载API密钥
  useEffect(() => {
    const loadKeys = async () => {
      try {
        setIsLoading(prev => ({ ...prev, keys: true }));
        await refreshApiKeys();
      } catch (error) {
        console.error('加载API密钥失败:', error);
        toast.error('加载API密钥失败，请检查网络连接并重试');
        // 不设置虚拟数据，保持空数组
        setUiApiKeys([]);
      } finally {
        setIsLoading(prev => ({ ...prev, keys: false }));
      }
    };
    
    loadKeys();
  }, []);

  const handleCreateKey = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!newKeyName.trim()) {
      toast.error('请输入API密钥名称');
      return;
    }
    
    try {
      setIsLoading(prev => ({ ...prev, create: true }));
      
      // 调用后端API创建密钥
      await createApiKey(newKeyName, 100);
      
      // 不需要手动添加到列表，因为refreshApiKeys会更新
      setNewKeyName('');
      setSelectedPermissions(['read']);
      
      toast.success('API密钥创建成功');
    } catch (error) {
      console.error('创建API密钥失败:', error);
      toast.error('创建API密钥失败');
    } finally {
      setIsLoading(prev => ({ ...prev, create: false }));
    }
  };

  const handleRevokeKey = async (id: string) => {
    if (!confirm('确定要撤销此API密钥吗？')) {
      return;
    }
    
    try {
      // 调用后端API删除密钥
      await deleteApiKey(id);
      
      // 不需要手动更新，因为refreshApiKeys会处理
      toast.success('API密钥已撤销');
    } catch (error) {
      console.error('撤销API密钥失败:', error);
      toast.error('撤销API密钥失败');
    }
  };

  const handleCopyKey = (key: string) => {
    navigator.clipboard.writeText(key);
    toast.success('API密钥已复制到剪贴板');
  };

  const handleToggleShowKey = (id: string) => {
    setShowKey(prev => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const handleRefreshKey = async (id: string) => {
    if (!confirm('确定要刷新此API密钥吗？这将生成新的密钥，旧的密钥将立即失效。')) {
      return;
    }
    
    try {
      // 先删除旧密钥
      await deleteApiKey(id);
      
      // 创建新密钥
      const keyName = uiApiKeys.find(k => k.id === id)?.name || 'New Key';
      await createApiKey(`${keyName} (刷新)`, 100);
      
      toast.success('API密钥已刷新');
    } catch (error) {
      console.error('刷新API密钥失败:', error);
      toast.error('刷新API密钥失败');
    }
  };

  const getStatusColor = (status: UiApiKey['status']) => {
    switch (status) {
      case 'active': return 'bg-gray-600 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400';
      case 'expired': return 'bg-gray-700 text-gray-900 dark:bg-gray-900/30 dark:text-gray-400';
      case 'revoked': return 'bg-gray-800 text-gray-900 dark:bg-gray-900/30 dark:text-gray-500';
      case 'limited': return 'bg-gray-600 text-gray-900 dark:bg-gray-900/30 dark:text-gray-400';
    }
  };

  const getStatusIcon = (status: UiApiKey['status']) => {
    switch (status) {
      case 'active': return <CheckCircle className="w-4 h-4" />;
      case 'expired': return <Clock className="w-4 h-4" />;
      case 'revoked': return <XCircle className="w-4 h-4" />;
      case 'limited': return <AlertCircle className="w-4 h-4" />;
    }
  };

  const formatDate = (date: Date | null) => {
    if (!date) return '从未使用';
    return date.toLocaleDateString('zh-CN');
  };

  const getPermissionColor = (permission: string) => {
    const colors: Record<string, string> = {
      read: 'bg-gray-600 text-gray-900 dark:bg-gray-900/30 dark:text-gray-400',
      write: 'bg-gray-600 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400',
      train: 'bg-gray-600 text-gray-900 dark:bg-gray-900/30 dark:text-gray-400',
      query: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400',
      manage: 'bg-gray-800 text-gray-900 dark:bg-gray-900/30 dark:text-gray-500',
      billing: 'bg-gray-700 text-gray-900 dark:bg-gray-900/30 dark:text-gray-400',
    };
    return colors[permission] || 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400';
  };

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            API密钥管理
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            管理和监控Self AGI的API访问密钥
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setActiveTab('keys')}
            className={`px-4 py-2 text-sm font-medium rounded-lg ${
              activeTab === 'keys'
                ? 'bg-gray-600 text-gray-900 dark:bg-gray-900/30 dark:text-gray-400'
                : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <Key className="w-4 h-4 inline mr-2" />
            密钥列表
          </button>
          <button
            onClick={() => setActiveTab('usage')}
            className={`px-4 py-2 text-sm font-medium rounded-lg ${
              activeTab === 'usage'
                ? 'bg-gray-600 text-gray-900 dark:bg-gray-900/30 dark:text-gray-400'
                : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-2" />
            使用统计
          </button>
          <button
            onClick={() => setActiveTab('settings')}
            className={`px-4 py-2 text-sm font-medium rounded-lg ${
              activeTab === 'settings'
                ? 'bg-gray-600 text-gray-900 dark:bg-gray-900/30 dark:text-gray-400'
                : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <Key className="w-4 h-4 inline mr-2" />
            权限设置
          </button>
        </div>
      </div>

      {/* API密钥列表页面 */}
      {activeTab === 'keys' && (
        <>
          {/* 创建新密钥表单 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="mb-6">
              <h2 className="text-lg font-medium text-gray-900 dark:text-white">
                创建新API密钥
              </h2>
              <p className="mt-1 text-gray-600 dark:text-gray-400">
                为应用程序或服务创建新的API访问密钥
              </p>
            </div>
            
            <form onSubmit={handleCreateKey} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    密钥名称 *
                  </label>
                  <input
                    type="text"
                    value={newKeyName}
                    onChange={(e) => setNewKeyName(e.target.value)}
                    placeholder="例如：生产环境API"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                    disabled={isLoading.create}
                  />
                  <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                    建议使用有意义的名称，便于识别和管理
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    权限设置
                  </label>
                  <div className="space-y-2">
                    {availablePermissions.map((permission) => (
                      <div key={permission.id} className="flex items-center">
                        <input
                          type="checkbox"
                          id={`perm-${permission.id}`}
                          checked={selectedPermissions.includes(permission.id)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedPermissions([...selectedPermissions, permission.id]);
                            } else {
                              setSelectedPermissions(
                                selectedPermissions.filter((p) => p !== permission.id)
                              );
                            }
                          }}
                          className="h-4 w-4 text-gray-800 focus:ring-gray-700 border-gray-300 rounded"
                          disabled={isLoading.create}
                        />
                        <label
                          htmlFor={`perm-${permission.id}`}
                          className="ml-2 text-sm text-gray-700 dark:text-gray-300"
                        >
                          {permission.name}
                        </label>
                        <span className="ml-2 text-xs text-gray-500">
                          {permission.description}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-gray-200 dark:border-gray-700 flex justify-end">
                <button
                  type="submit"
                  disabled={isLoading.create}
                  className="inline-flex items-center px-6 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-800 to-gray-700 rounded-lg hover:from-gray-900 hover:to-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading.create ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      创建中...
                    </>
                  ) : (
                    <>
                      <Plus className="w-4 h-4 inline mr-2" />
                      创建API密钥
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* 密钥列表 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-900">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      API密钥
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      状态
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      权限
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      创建时间
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      最后使用
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      操作
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {uiApiKeys.map((key) => (
                    <tr key={key.id} className="hover:bg-gray-50 dark:hover:bg-gray-750">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            {key.name}
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400 font-mono mt-1">
                            {showKey[key.id] ? key.key : key.keyMasked}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(key.status)}`}>
                          {getStatusIcon(key.status)}
                          <span className="ml-1">
                            {key.status === 'active' ? '活跃' : 
                             key.status === 'expired' ? '已过期' : 
                             key.status === 'revoked' ? '已撤销' : '受限制'}
                          </span>
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex flex-wrap gap-1">
                          {key.permissions.slice(0, 3).map((perm) => (
                            <span
                              key={perm}
                              className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${getPermissionColor(perm)}`}
                            >
                              {perm === 'read' ? '读取' :
                               perm === 'write' ? '写入' :
                               perm === 'train' ? '训练' :
                               perm === 'query' ? '查询' :
                               perm === 'manage' ? '管理' :
                               perm === 'billing' ? '账单' : perm}
                            </span>
                          ))}
                          {key.permissions.length > 3 && (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400">
                              +{key.permissions.length - 3}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {formatDate(key.createdAt)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {formatDate(key.lastUsed)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => handleToggleShowKey(key.id)}
                            className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300 p-1"
                            title={showKey[key.id] ? '隐藏密钥' : '显示密钥'}
                          >
                            {showKey[key.id] ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                          </button>
                          <button
                            onClick={() => handleCopyKey(key.key)}
                            className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300 p-1"
                            title="复制密钥"
                          >
                            <Copy className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleRefreshKey(key.id)}
                            className="text-gray-800 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-400 p-1"
                            title="刷新密钥"
                          >
                            <RefreshCw className="w-4 h-4" />
                          </button>
                          {key.status !== 'revoked' && (
                            <button
                              onClick={() => handleRevokeKey(key.id)}
                              className="text-gray-900 hover:text-gray-900 dark:text-gray-500 dark:hover:text-gray-600 p-1"
                              title="撤销密钥"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* 使用统计页面 */}
      {activeTab === 'usage' && (
        <div className="space-y-6">
          {/* 统计数据概览 */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">今日请求</h3>
                  <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">
                    1,654
                  </p>
                </div>
                <BarChart3 className="w-8 h-8 text-gray-700" />
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm text-gray-700 dark:text-gray-400">
                  <span className="mr-1">+12%</span> 比昨日
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">本月总请求</h3>
                  <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">
                    12,456
                  </p>
                </div>
                <BarChart3 className="w-8 h-8 text-gray-600" />
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm text-gray-700 dark:text-gray-400">
                  <span className="mr-1">+24%</span> 比上月
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">错误率</h3>
                  <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">
                    2.1%
                  </p>
                </div>
                <AlertCircle className="w-8 h-8 text-gray-600" />
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm text-gray-900 dark:text-gray-500">
                  <span className="mr-1">+0.3%</span> 比上周
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">活跃密钥</h3>
                  <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">
                    {uiApiKeys.filter(k => k.status === 'active').length}
                  </p>
                </div>
                <Key className="w-8 h-8 text-gray-600" />
              </div>
              <div className="mt-4">
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  总计 {uiApiKeys.length} 个密钥
                </div>
              </div>
            </div>
          </div>

          {/* 请求趋势图 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
              7天请求趋势
            </h2>
            <div className="h-64">
              <div className="flex items-end h-48 space-x-2">
                {usageStats.map((stat, index) => (
                  <div key={index} className="flex-1 flex flex-col items-center">
                    <div className="w-full relative">
                      <div className="absolute bottom-0 left-0 right-0">
                        <div className="relative h-40">
                          <div 
                            className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-gray-700 to-gray-400 rounded-t-lg"
                            style={{ height: `${(stat.requests / 2000) * 100}%` }}
                          />
                          {stat.errors > 0 && (
                            <div 
                              className="absolute top-0 left-0 right-0 bg-gradient-to-t from-gray-800 to-gray-600 rounded-t-lg"
                              style={{ height: `${(stat.errors / 2000) * 100}%` }}
                            />
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                      {stat.date.split('-').slice(1).join('/')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 权限设置页面 */}
      {activeTab === 'settings' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="mb-6">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              API权限设置
            </h2>
            <p className="mt-1 text-gray-600 dark:text-gray-400">
              配置API密钥的默认权限和安全设置
            </p>
          </div>
          
          <div className="space-y-6">
            <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                安全设置
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white">
                      API密钥默认过期时间
                    </h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      新创建的API密钥将在指定时间后自动过期
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <select 
                      value={defaultExpirationDays}
                      onChange={(e) => setDefaultExpirationDays(Number(e.target.value))}
                      className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                    >
                      <option value="30">30天</option>
                      <option value="90">90天</option>
                      <option value="180">180天</option>
                      <option value="365">1年</option>
                      <option value="730">2年</option>
                    </select>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white">
                      默认速率限制
                    </h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      新API密钥的每分钟最大请求数
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="number"
                      min="10"
                      max="10000"
                      defaultValue="100"
                      className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white w-24"
                    />
                    <span className="text-gray-500 dark:text-gray-400">请求/分钟</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white">
                      IP白名单
                    </h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      限制API只能从特定IP地址访问
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button className="px-3 py-1.5 text-sm font-medium text-white bg-gradient-to-r from-gray-800 to-gray-700 rounded-lg">
                      配置
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ApiKeysPage;