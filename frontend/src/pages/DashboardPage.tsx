import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  Activity,
  AlertCircle,
  Brain,
  Cpu,
  Database,
  MessageSquare,
  Smartphone,
  Server,
  Users,
  Zap,
  RefreshCw,
} from 'lucide-react';
import StatCard from '../components/Dashboard/StatCard';
import SystemStatus from '../components/Dashboard/SystemStatus';
import RecentActivity from '../components/Dashboard/RecentActivity';
import QuickActions from '../components/Dashboard/QuickActions';
import ModelTrainingProgress from '../components/Dashboard/ModelTrainingProgress';
import ResourceUsage from '../components/Dashboard/ResourceUsage';
import { PageLoader, Spinner } from '../components/UI';
import { chatApi } from '../services/api/chat';
import authService from '../services/api/auth';
import { monitoringService } from '../services/api/monitoring';
import toast from 'react-hot-toast';
import { handleError, logSuccess } from '../utils/errorHandler';

const DashboardPage: React.FC = () => {
  const { user, isAdmin } = useAuth();
  const [systemStats, setSystemStats] = useState<{
    active_users?: {
      value: number;
      change: string;
      total: number;
    };
    total_api_calls?: {
      value: number;
      change: string;
      period: string;
    };
    model_training_jobs?: {
      value: number;
      change: string;
      status: string;
    };
    active_robots?: {
      value: number;
      change: string;
      status: string;
    };
    system_resources?: {
      system_load: number;
      memory_usage: number;
      storage_usage: number;
      timestamp: string;
    };
    timestamp?: string;
  }>({});

  // 系统监控状态
  const [systemMetrics, setSystemMetrics] = useState<any>(null);
  const [monitoringLoading, setMonitoringLoading] = useState(true);
  const [monitoringError, setMonitoringError] = useState<string | null>(null);

  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [loading, setLoading] = useState(true);
  const [systemMode, setSystemMode] = useState('task');
  const [modeLoading, setModeLoading] = useState(false);
  const [statsError, setStatsError] = useState<string | null>(null);

  const fetchSystemMode = async () => {
    try {
      setModeLoading(true);
      const response = await chatApi.getSystemMode();
      if (response.success && response.data) {
        setSystemMode(response.data.mode);
      }
    } catch (error) {
      handleError(error, '获取系统模式');
    } finally {
      setModeLoading(false);
    }
  };

  const fetchSystemStats = async () => {
    try {
      setStatsError(null);
      const response = await authService.getSystemStats();
      
      if (response.success && response.data) {
        setSystemStats(response.data);
      } else {
        const errorMessage = '获取统计数据失败';
        setStatsError(errorMessage);
        handleError(new Error(errorMessage), '获取系统统计');
      }
    } catch (error: any) {
      handleError(error, '获取系统统计');
      setStatsError(`获取统计数据失败: ${error.message || '未知错误'}`);
    }
  };

  const toggleSystemMode = async () => {
    try {
      setModeLoading(true);
      const newMode = systemMode === 'task' ? 'autonomous' : 'task';
      const response = await chatApi.setSystemMode(newMode);
      if (response.success) {
        setSystemMode(newMode);
        const successMessage = `系统模式已从 ${systemMode === 'task' ? '任务执行模式' : '全自主模式'} 切换到 ${newMode === 'task' ? '任务执行模式' : '全自主模式'}`;
        toast.success(successMessage);
        logSuccess(successMessage, '切换系统模式');
      } else {
        const errorMessage = '切换系统模式失败';
        handleError(new Error(errorMessage), '切换系统模式');
      }
    } catch (error) {
      handleError(error, '切换系统模式');
    } finally {
      setModeLoading(false);
    }
  };

  // 获取系统监控指标
  const fetchSystemMetrics = async () => {
    try {
      setMonitoringError(null);
      const response = await monitoringService.getSystemMetrics();
      
      if (response.success && response.metrics) {
        setSystemMetrics(response.metrics);
      } else {
        const errorMessage = '获取系统监控数据失败';
        setMonitoringError(errorMessage);
        handleError(new Error(errorMessage), '获取系统监控指标');
      }
    } catch (error: any) {
      handleError(error, '获取系统监控指标');
      setMonitoringError(`获取系统监控数据失败: ${error.message || '未知错误'}`);
    } finally {
      setMonitoringLoading(false);
    }
  };

  useEffect(() => {
    // 获取系统模式
    fetchSystemMode();
    
    // 获取系统统计数据
    const loadDashboardData = async () => {
      try {
        await Promise.all([
          fetchSystemStats(),
          fetchSystemMetrics(),
          // 可以添加其他数据获取函数
        ]);
      } catch (error) {
        handleError(error, '加载仪表板数据');
      } finally {
        setLoading(false);
      }
    };
    
    loadDashboardData();

    // 设置定时刷新（每30秒刷新一次）
    refreshIntervalRef.current = setInterval(() => {
      fetchSystemStats();
      fetchSystemMetrics(); // 同时刷新监控数据
    }, 30000);

    // 清理函数
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, []);

  if (loading) {
    return <PageLoader title="加载仪表板..." subtitle="正在获取系统数据和状态" />;
  }

  return (
    <div className="space-y-6">
      {/* 欢迎区域 */}
      <div className="bg-gradient-to-r from-gray-600 to-gray-600 rounded-2xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center mb-2">
              <h1 className="text-3xl font-bold mr-4">
                欢迎回来, {user?.name || user?.username}!
              </h1>
              <div className="flex items-center space-x-2">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${systemMode === 'task' ? 'bg-gray-700' : 'bg-gray-600'}`}>
                  {systemMode === 'task' ? '任务执行模式' : '全自主模式'}
                </span>
                {isAdmin && (
                  <button
                    onClick={toggleSystemMode}
                    disabled={modeLoading}
                    className="flex items-center px-3 py-1 bg-white/20 hover:bg-white/30 rounded-full text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {modeLoading ? (
                      <Spinner size="xs" variant="white" className="mr-1" />
                    ) : (
                      <RefreshCw className="h-3 w-3 mr-1" />
                    )}
                    切换模式
                  </button>
                )}
              </div>
            </div>
            <p className="text-gray-100">
              这是您的 Self AGI 系统控制面板。当前系统运行在 <span className="font-bold">{systemMode === 'task' ? '任务执行模式' : '全自主模式'}</span>。监控系统状态，管理任务，探索人工智能的无限可能。
            </p>
            {systemMode === 'autonomous' && (
              <div className="mt-3 p-2 bg-white/10 rounded-lg inline-block">
                <p className="text-sm">💡 全自主模式已启用：系统将自动学习和执行任务</p>
              </div>
            )}
          </div>
          <div className="hidden md:block">
            <Brain className="h-20 w-20 text-white/20" />
          </div>
        </div>
      </div>

      {/* 快速概览统计 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {systemStats.active_users ? (
          <StatCard
            title="活跃用户"
            value={systemStats.active_users.value.toLocaleString()}
            change={systemStats.active_users.change}
            icon={<Users className="h-5 w-5" />}
            color="gray"
          />
        ) : (
          <div className="card p-4 animate-pulse">
            <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2"></div>
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-1"></div>
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/3"></div>
          </div>
        )}
        {systemStats.total_api_calls ? (
          <StatCard
            title="API调用"
            value={systemStats.total_api_calls.value.toLocaleString()}
            change={systemStats.total_api_calls.change}
            icon={<Zap className="h-5 w-5" />}
            color="gray"
          />
        ) : (
          <div className="card p-4 animate-pulse">
            <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2"></div>
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-1"></div>
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/3"></div>
          </div>
        )}
        {systemStats.model_training_jobs ? (
          <StatCard
            title="训练任务"
            value={systemStats.model_training_jobs.value}
            change={systemStats.model_training_jobs.change}
            icon={<Cpu className="h-5 w-5" />}
            color="gray"
          />
        ) : (
          <div className="card p-4 animate-pulse">
            <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2"></div>
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-1"></div>
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/3"></div>
          </div>
        )}
        {systemStats.active_robots ? (
          <StatCard
            title="活动机器人"
            value={systemStats.active_robots.value}
            change={systemStats.active_robots.change}
            icon={<Smartphone className="h-5 w-5" />}
            color="gray"
          />
        ) : (
          <div className="card p-4 animate-pulse">
            <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2"></div>
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-1"></div>
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/3"></div>
          </div>
        )}
      </div>

      {/* 错误显示 */}
      {statsError && (
        <div className="bg-gray-900 dark:bg-gray-900/20 border border-gray-800 dark:border-gray-800 rounded-lg p-4">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-gray-800 mr-2" />
            <p className="text-gray-900 dark:text-gray-300">{statsError}</p>
          </div>
        </div>
      )}

      {/* 系统状态和快速操作 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* 系统状态 */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                系统状态
              </h2>
              <Activity className="h-5 w-5 text-gray-400" />
            </div>
            <SystemStatus />
          </div>

          {/* 资源使用情况 */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                资源使用情况
              </h2>
              <Server className="h-5 w-5 text-gray-400" />
            </div>
            <ResourceUsage />
          </div>

          {/* 系统实时监控 */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                系统实时监控
              </h2>
              <Activity className="h-5 w-5 text-gray-400" />
            </div>
            
            {monitoringLoading ? (
              <div className="flex justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-gray-500"></div>
              </div>
            ) : monitoringError ? (
              <div className="bg-gray-900 dark:bg-gray-900/20 border border-gray-800 dark:border-gray-800 rounded-lg p-4">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-gray-800 mr-2" />
                  <p className="text-gray-900 dark:text-gray-300">{monitoringError}</p>
                </div>
              </div>
            ) : systemMetrics ? (
              <div className="space-y-4">
                {/* CPU使用率 */}
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">CPU使用率</span>
                    <span className="text-sm font-bold">{systemMetrics.cpu?.percent?.toFixed(1) || 0}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                    <div 
                      className={`h-2.5 rounded-full ${systemMetrics.cpu?.percent > 80 ? 'bg-gray-800' : systemMetrics.cpu?.percent > 60 ? 'bg-gray-600' : 'bg-gray-500'}`}
                      style={{ width: `${systemMetrics.cpu?.percent || 0}%` }}
                    ></div>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    核心: {systemMetrics.cpu?.cores || 0} | 线程: {systemMetrics.cpu?.threads || 0}
                  </div>
                </div>
                
                {/* 内存使用 */}
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">内存使用</span>
                    <span className="text-sm font-bold">{systemMetrics.memory?.percent?.toFixed(1) || 0}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                    <div 
                      className={`h-2.5 rounded-full ${systemMetrics.memory?.percent > 80 ? 'bg-gray-800' : systemMetrics.memory?.percent > 60 ? 'bg-gray-600' : 'bg-gray-500'}`}
                      style={{ width: `${systemMetrics.memory?.percent || 0}%` }}
                    ></div>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    已用: {(systemMetrics.memory?.used || 0).toFixed(1)} MB / 总计: {(systemMetrics.memory?.total || 0).toFixed(1)} MB
                  </div>
                </div>
                
                {/* 磁盘使用 */}
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">磁盘使用</span>
                    <span className="text-sm font-bold">{systemMetrics.disk?.percent?.toFixed(1) || 0}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                    <div 
                      className={`h-2.5 rounded-full ${systemMetrics.disk?.percent > 80 ? 'bg-gray-800' : systemMetrics.disk?.percent > 60 ? 'bg-gray-600' : 'bg-gray-500'}`}
                      style={{ width: `${systemMetrics.disk?.percent || 0}%` }}
                    ></div>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    已用: {(systemMetrics.disk?.used || 0).toFixed(1)} GB / 总计: {(systemMetrics.disk?.total || 0).toFixed(1)} GB
                  </div>
                </div>
                
                {/* 系统信息 */}
                <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">系统运行时间:</span>
                      <span className="ml-2 font-medium">{(systemMetrics.system?.uptime || 0).toFixed(0)}秒</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">活动用户:</span>
                      <span className="ml-2 font-medium">{systemMetrics.system?.users || 0}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">进程内存:</span>
                      <span className="ml-2 font-medium">{(systemMetrics.process?.memory_mb || 0).toFixed(1)} MB</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">网络流量:</span>
                      <span className="ml-2 font-medium">{((systemMetrics.network?.bytes_sent || 0) / 1024 / 1024).toFixed(1)} MB</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                暂无监控数据
              </div>
            )}
          </div>
        </div>

        <div className="space-y-6">
          {/* 快速操作 */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                快速操作
              </h2>
              <Zap className="h-5 w-5 text-gray-400" />
            </div>
            <QuickActions />
          </div>

          {/* 模型训练进度 */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                模型训练
              </h2>
              <Brain className="h-5 w-5 text-gray-400" />
            </div>
            <ModelTrainingProgress />
          </div>
        </div>
      </div>

      {/* 最近活动 */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            最近活动
          </h2>
          <MessageSquare className="h-5 w-5 text-gray-400" />
        </div>
        <RecentActivity />
      </div>

      {/* 管理员面板 */}
      {isAdmin && (
        <div className="card p-6 border-2 border-gray-500">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              管理员面板
            </h2>
            <div className="px-3 py-1 bg-gray-100 dark:bg-gray-900 text-gray-800 dark:text-gray-200 text-sm font-medium rounded-full">
              管理员
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <a
              href="/admin"
              className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="flex items-center">
                <Users className="h-5 w-5 text-gray-400 mr-3" />
                <div>
                  <h3 className="font-medium text-gray-900 dark:text-white">用户管理</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    管理用户账户和权限
                  </p>
                </div>
              </div>
            </a>
            <a
              href="/admin/models"
              className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="flex items-center">
                <Brain className="h-5 w-5 text-gray-400 mr-3" />
                <div>
                  <h3 className="font-medium text-gray-900 dark:text-white">模型管理</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    部署和管理AI模型
                  </p>
                </div>
              </div>
            </a>
            <a
              href="/admin/system"
              className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="flex items-center">
                <Database className="h-5 w-5 text-gray-400 mr-3" />
                <div>
                  <h3 className="font-medium text-gray-900 dark:text-white">系统设置</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    配置系统参数和设置
                  </p>
                </div>
              </div>
            </a>
          </div>
        </div>
      )}

      {/* 使用提示 */}
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <div className="h-8 w-8 rounded-full bg-gray-100 dark:bg-gray-900 flex items-center justify-center">
              <Zap className="h-4 w-4 text-gray-600 dark:text-gray-400" />
            </div>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white">
              使用提示
            </h3>
            <div className="mt-2 text-sm text-gray-600 dark:text-gray-400">
              <ul className="list-disc list-inside space-y-1">
                <li>点击左侧导航栏的"对话"开始与AI对话</li>
                <li>在"训练"页面配置和启动模型训练任务</li>
                <li>使用"硬件"页面连接和管理机器人设备</li>
                <li>在"设置"中自定义您的个人偏好</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;