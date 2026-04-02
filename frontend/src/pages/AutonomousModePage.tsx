/**
 * 自主模式控制页面 (简化版本)
 * 
 * 功能：
 * 1. 实时模式状态显示和监控
 * 2. 自主模式和任务执行模式切换
 * 3. 自主目标创建和管理
 * 4. 决策历史查看和统计
 * 5. 系统参数配置
 */

import React, { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../services/api/client';
import { ApiResponse } from '../types/api';
import toast from 'react-hot-toast';
import {
  Play,
  Pause,
  Square,
  RefreshCw,
  Plus,
  Settings,
  AlertTriangle,
  Brain,
  Target,
  Download,
  Trash2,
  Search,
  X,
} from 'lucide-react';

// 类型定义
type Priority = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'BACKGROUND';

interface NewGoal {
  description: string;
  priority: Priority;
}

interface ModeStatus {
  autonomous_mode_enabled: boolean;
  autonomous_mode_running: boolean;
  current_state: string;
  previous_state: string;
  active_goals_count: number;
  pending_goals_count: number;
  total_goals: number;
  completed_goals: number;
  failed_goals: number;
  avg_goal_duration: number;
  safety_violations: number;
  ethical_violations: number;
  last_state_change: string | null;
  timestamp: string;
}

interface Goal {
  id: string;
  description: string;
  priority: Priority;
  created_at: string;
  deadline: string | null;
  estimated_duration: number | null;
  parameters: Record<string, any>;
  status: string;
  progress: number;
  result: Record<string, any> | null;
  safety_constraints: string[];
  ethical_checks: string[];
  // 兼容字段
  updated_at?: string;
  estimated_completion?: string | null;
  assigned_to?: string | null;
  metadata?: Record<string, any>;
}



interface Config {
  autonomy_level: number;
  risk_tolerance: number;
  learning_rate: number;
  exploration_rate: number;
  max_goals: number;
  safety_threshold: number;
  ethical_threshold: number;
  enable_self_optimization: boolean;
  enable_safety_monitor: boolean;
  enable_ethical_monitor: boolean;
  enable_learning: boolean;
  enable_exploration: boolean;
  log_level: string;
  data_retention_days: number;
}

const AutonomousModePage: React.FC = () => {

  const [modeStatus, setModeStatus] = useState<ModeStatus | null>(null);
  const [goals, setGoals] = useState<Goal[]>([]);
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateGoal, setShowCreateGoal] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [newGoal, setNewGoal] = useState<NewGoal>({
    description: '',
    priority: 'MEDIUM',
  });
  const [editConfig, setEditConfig] = useState<Partial<Config>>({});
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [goalFilter, setGoalFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  // 获取模式状态
  const fetchModeStatus = useCallback(async () => {
    try {
      const response = (await apiClient.get('/api/system/mode/status')) as ApiResponse<ModeStatus>;
      if (response.success && response.data) {
        setModeStatus(response.data);
        setError(null);
      }
    } catch (err: any) {
      console.error('获取模式状态失败:', err);
      setError(err.message || '获取模式状态失败');
    }
  }, []);

  // 获取目标列表
  const fetchGoals = useCallback(async () => {
    try {
      const response = (await apiClient.get('/api/system/mode/goals')) as ApiResponse<Goal[]>;
      if (response.success && response.data) {
        setGoals(response.data);
      }
    } catch (err: any) {
      console.error('获取目标列表失败:', err);
    }
  }, []);

  // 获取决策历史


  // 获取配置
  const fetchConfig = useCallback(async () => {
    try {
      const response = (await apiClient.get('/api/system/mode/config')) as ApiResponse<Config>;
      if (response.success && response.data) {
        setConfig(response.data);
        setEditConfig(response.data);
      }
    } catch (err: any) {
      console.error('获取配置失败:', err);
    }
  }, []);

  // 初始化加载
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        await Promise.all([
          fetchModeStatus(),
          fetchGoals(),
          fetchConfig()
        ]);
      } catch (err: any) {
        console.error('加载数据失败:', err);
        toast.error('加载数据失败');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [fetchModeStatus, fetchGoals, fetchConfig]);

  // 自动刷新
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchModeStatus();
      fetchGoals();
    }, 5000); // 5秒刷新间隔

    return () => clearInterval(interval);
  }, [autoRefresh, fetchModeStatus, fetchGoals]);

  // 切换自主模式
  const toggleAutonomousMode = async () => {
    if (!modeStatus) return;
    
    try {
      const endpoint = modeStatus.autonomous_mode_enabled 
        ? '/api/system/mode/disable' 
        : '/api/system/mode/enable';
      
      const response = (await apiClient.post(endpoint)) as ApiResponse;
      if (response.success) {
        toast.success(`自主模式已${modeStatus.autonomous_mode_enabled ? '禁用' : '启用'}`);
        fetchModeStatus();
      }
    } catch (err: any) {
      console.error('切换自主模式失败:', err);
      toast.error('切换自主模式失败');
    }
  };

  // 启动/停止自主模式
  const toggleAutonomousRunning = async () => {
    if (!modeStatus) return;
    
    try {
      const endpoint = modeStatus.autonomous_mode_running 
        ? '/api/system/mode/stop' 
        : '/api/system/mode/start';
      
      const response = (await apiClient.post(endpoint)) as ApiResponse;
      if (response.success) {
        toast.success(`自主模式已${modeStatus.autonomous_mode_running ? '停止' : '启动'}`);
        fetchModeStatus();
      }
    } catch (err: any) {
      console.error('启动/停止自主模式失败:', err);
      toast.error('启动/停止自主模式失败');
    }
  };

  // 创建新目标
  const handleCreateGoal = async () => {
    if (!newGoal.description.trim()) {
      toast.error('请输入目标描述');
      return;
    }

    try {
      const response = (await apiClient.post('/api/system/mode/goals', newGoal)) as ApiResponse;
      if (response.success) {
        toast.success('目标创建成功');
        setNewGoal({ description: '', priority: 'MEDIUM' });
        setShowCreateGoal(false);
        fetchGoals();
      }
    } catch (err: any) {
      console.error('创建目标失败:', err);
      toast.error('创建目标失败');
    }
  };

  // 更新配置
  const handleUpdateConfig = async () => {
    if (!config) return;

    try {
      const response = (await apiClient.put('/api/system/mode/config', editConfig)) as ApiResponse;
      if (response.success) {
        toast.success('配置更新成功');
        setConfig({ ...config, ...editConfig });
        setShowConfig(false);
      }
    } catch (err: any) {
      console.error('更新配置失败:', err);
      toast.error('更新配置失败');
    }
  };

  // 删除目标
  const handleDeleteGoal = async (goalId: string) => {
    if (!confirm('确定要删除这个目标吗？')) return;

    try {
      const response = (await apiClient.delete(`/api/system/mode/goals/${goalId}`)) as ApiResponse;
      if (response.success) {
        toast.success('目标删除成功');
        fetchGoals();
      }
    } catch (err: any) {
      console.error('删除目标失败:', err);
      toast.error('删除目标失败');
    }
  };

  // 导出数据
  const handleExportData = async () => {
    try {
      const response = (await apiClient.get('/api/system/mode/export')) as ApiResponse<unknown>;
      if (!response.success || !response.data) {
        toast.error('导出数据失败：服务器返回错误');
        return;
      }
      const data = JSON.stringify(response.data, null, 2);
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `autonomous-data-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('数据导出成功');
    } catch (err: any) {
      console.error('导出数据失败:', err);
      toast.error('导出数据失败');
    }
  };

  // 过滤目标
  const filteredGoals = goals.filter(goal => {
    if (goalFilter !== 'all' && goal.status !== goalFilter) return false;
    if (searchTerm && !goal.description.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  // 状态指示器颜色
  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active': return 'bg-gray-600';
      case 'completed': return 'bg-gray-700';
      case 'failed': return 'bg-gray-800';
      case 'pending': return 'bg-gray-600';
      default: return 'bg-gray-500';
    }
  };

  // 优先级颜色
  const getPriorityColor = (priority: string) => {
    switch (priority.toLowerCase()) {
      case 'high': return 'bg-gray-800 text-gray-900';
      case 'medium': return 'bg-gray-700 text-gray-900';
      case 'low': return 'bg-gray-600 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading && !modeStatus) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-12 w-12 text-gray-800 dark:text-gray-400 animate-spin mx-auto" />
          <p className="mt-4 text-gray-600 dark:text-gray-400">加载自主模式数据...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        {/* 标题和操作栏 */}
        <div className="mb-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white">
                自主模式控制
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                管理和监控AGI系统的自主决策功能
              </p>
              <div className="mt-2">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-800 text-white">
                  功能状态：实现中
                </span>
                <span className="ml-2 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-600 text-white">
                  后端服务：已连接
                </span>
                <span className="ml-2 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-700 text-white">
                  模式切换：全自主/任务执行
                </span>
              </div>
            </div>
            
            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleExportData}
                className="px-4 py-2 bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors flex items-center"
              >
                <Download className="h-5 w-5 mr-2" />
                导出数据
              </button>
              
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`px-4 py-2 rounded-lg transition-colors flex items-center ${
                  autoRefresh 
                    ? 'bg-gray-800 text-white hover:bg-gray-900' 
                    : 'bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-700'
                }`}
              >
                <RefreshCw className={`h-5 w-5 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
                {autoRefresh ? '自动刷新中' : '启用自动刷新'}
              </button>
              
              <button
                onClick={() => setShowConfig(true)}
                className="px-4 py-2 bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors flex items-center"
              >
                <Settings className="h-5 w-5 mr-2" />
                系统配置
              </button>
            </div>
          </div>
        </div>

        {/* 错误显示 */}
        {error && (
          <div className="mb-6 p-4 bg-gray-900 dark:bg-gray-900/20 border border-gray-700 dark:border-gray-900 rounded-lg">
            <div className="flex items-center">
              <AlertTriangle className="h-5 w-5 text-gray-900 dark:text-gray-500 mr-2" />
              <p className="text-gray-900 dark:text-gray-600">{error}</p>
              <button
                onClick={() => setError(null)}
                className="ml-auto text-gray-900 dark:text-gray-500 hover:text-gray-900 dark:hover:text-gray-700"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          </div>
        )}

        {/* 主控制面板 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* 状态卡片 */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center">
                  <Brain className="h-8 w-8 text-gray-800 dark:text-gray-400 mr-3" />
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                      自主模式状态
                    </h2>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      最后更新: {modeStatus?.timestamp ? new Date(modeStatus.timestamp).toLocaleString() : '--'}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    modeStatus?.autonomous_mode_enabled
                      ? 'bg-gray-600 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400'
                      : 'bg-gray-800 text-gray-900 dark:bg-gray-900/30 dark:text-gray-500'
                  }`}>
                    {modeStatus?.autonomous_mode_enabled ? '已启用' : '已禁用'}
                  </span>
                  
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    modeStatus?.autonomous_mode_running
                      ? 'bg-gray-600 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400'
                      : 'bg-gray-700 text-gray-900 dark:bg-gray-900/30 dark:text-gray-400'
                  }`}>
                    {modeStatus?.autonomous_mode_running ? '运行中' : '已停止'}
                  </span>
                </div>
              </div>

              {/* 状态指标 */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">活跃目标</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {modeStatus?.active_goals_count || 0}
                  </p>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">完成目标</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {modeStatus?.completed_goals || 0}
                  </p>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">安全违规</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {modeStatus?.safety_violations || 0}
                  </p>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">伦理违规</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {modeStatus?.ethical_violations || 0}
                  </p>
                </div>
              </div>

              {/* 控制按钮 */}
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={toggleAutonomousMode}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center ${
                    modeStatus?.autonomous_mode_enabled
                      ? 'bg-gray-900 hover:bg-gray-900 text-white'
                      : 'bg-gray-700 hover:bg-gray-800 text-white'
                  }`}
                >
                  {modeStatus?.autonomous_mode_enabled ? (
                    <>
                      <Square className="h-5 w-5 mr-2" />
                      禁用自主模式
                    </>
                  ) : (
                    <>
                      <Play className="h-5 w-5 mr-2" />
                      启用自主模式
                    </>
                  )}
                </button>
                
                <button
                  onClick={toggleAutonomousRunning}
                  disabled={!modeStatus?.autonomous_mode_enabled}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center ${
                    !modeStatus?.autonomous_mode_enabled
                      ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                      : modeStatus?.autonomous_mode_running
                      ? 'bg-gray-700 hover:bg-gray-800 text-white'
                      : 'bg-gray-800 hover:bg-gray-900 text-white'
                  }`}
                >
                  {modeStatus?.autonomous_mode_running ? (
                    <>
                      <Pause className="h-5 w-5 mr-2" />
                      停止运行
                    </>
                  ) : (
                    <>
                      <Play className="h-5 w-5 mr-2" />
                      开始运行
                    </>
                  )}
                </button>
                
                <button
                  onClick={fetchModeStatus}
                  className="px-4 py-2 bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors flex items-center"
                >
                  <RefreshCw className="h-5 w-5 mr-2" />
                  刷新状态
                </button>
              </div>
            </div>
          </div>

          {/* 当前状态卡片 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              当前状态
            </h3>
            
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">当前状态</p>
                <p className="font-medium text-gray-900 dark:text-white">
                  {modeStatus?.current_state || '未知'}
                </p>
              </div>
              
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">上一个状态</p>
                <p className="font-medium text-gray-900 dark:text-white">
                  {modeStatus?.previous_state || '未知'}
                </p>
              </div>
              
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">最后状态变更</p>
                <p className="font-medium text-gray-900 dark:text-white">
                  {modeStatus?.last_state_change 
                    ? new Date(modeStatus.last_state_change).toLocaleString() 
                    : '从未变更'}
                </p>
              </div>
              
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">平均目标时长</p>
                <p className="font-medium text-gray-900 dark:text-white">
                  {modeStatus?.avg_goal_duration 
                    ? `${(modeStatus.avg_goal_duration / 1000).toFixed(1)} 秒` 
                    : '--'}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* 目标管理 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                自主目标管理
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                创建、管理和监控自主目标
              </p>
            </div>
            
            <div className="flex items-center space-x-3 mt-4 md:mt-0">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="搜索目标..."
                  value={searchTerm}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchTerm(e.target.value)}
                  className="pl-10 pr-4 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                />
              </div>
              
              <select
                value={goalFilter}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setGoalFilter(e.target.value)}
                className="px-4 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
              >
                <option value="all">所有状态</option>
                <option value="active">活跃</option>
                <option value="pending">等待</option>
                <option value="completed">完成</option>
                <option value="failed">失败</option>
              </select>
              
              <button
                onClick={() => setShowCreateGoal(true)}
                className="px-4 py-2 bg-gray-800 hover:bg-gray-900 text-white rounded-lg transition-colors flex items-center"
              >
                <Plus className="h-5 w-5 mr-2" />
                新建目标
              </button>
            </div>
          </div>

          {/* 目标列表 */}
          {filteredGoals.length === 0 ? (
            <div className="text-center py-12">
              <Target className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600 dark:text-gray-400">暂无目标</p>
              <button
                onClick={() => setShowCreateGoal(true)}
                className="mt-4 px-4 py-2 bg-gray-800 hover:bg-gray-900 text-white rounded-lg transition-colors"
              >
                创建第一个目标
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-600 dark:text-gray-400">目标描述</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-600 dark:text-gray-400">优先级</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-600 dark:text-gray-400">状态</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-600 dark:text-gray-400">进度</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-600 dark:text-gray-400">创建时间</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-600 dark:text-gray-400">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredGoals.map((goal) => (
                    <tr key={goal.id} className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900/50">
                      <td className="py-3 px-4">
                        <div>
                          <p className="text-gray-900 dark:text-white font-medium">{goal.description}</p>
                          {goal.assigned_to && (
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                              分配给: {goal.assigned_to}
                            </p>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(goal.priority)}`}>
                          {goal.priority}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center">
                          <div className={`h-2 w-2 rounded-full mr-2 ${getStatusColor(goal.status)}`} />
                          <span className="text-sm text-gray-900 dark:text-white capitalize">{goal.status}</span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <div className="w-24">
                          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gray-800"
                              style={{ width: `${goal.progress}%` }}
                            />
                          </div>
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 text-center">
                            {goal.progress}%
                          </p>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400">
                        {new Date(goal.created_at).toLocaleDateString()}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex space-x-2">
                          <button
                            onClick={() => handleDeleteGoal(goal.id)}
                            className="p-1 text-gray-900 hover:text-gray-900 dark:text-gray-500 dark:hover:text-gray-600"
                            title="删除"
                          >
                            <Trash2 className="h-5 w-5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* 创建目标模态框 */}
        {showCreateGoal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg max-w-md w-full">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    创建新目标
                  </h3>
                  <button
                    onClick={() => setShowCreateGoal(false)}
                    className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      目标描述
                    </label>
                    <textarea
                      value={newGoal.description}
                      onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => 
                        setNewGoal({ ...newGoal, description: e.target.value })
                      }
                      className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                      rows={3}
                      placeholder="输入目标描述..."
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      优先级
                    </label>
                    <select
                      value={newGoal.priority}
                      onChange={(e: React.ChangeEvent<HTMLSelectElement>) => 
                        setNewGoal({ ...newGoal, priority: e.target.value as Priority })
                      }
                      className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                    >
                      <option value="CRITICAL">关键</option>
                      <option value="HIGH">高</option>
                      <option value="MEDIUM">中</option>
                      <option value="LOW">低</option>
                      <option value="BACKGROUND">后台</option>
                    </select>
                  </div>
                </div>
                
                <div className="flex justify-end space-x-3 mt-6">
                  <button
                    onClick={() => setShowCreateGoal(false)}
                    className="px-4 py-2 bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors"
                  >
                    取消
                  </button>
                  <button
                    onClick={handleCreateGoal}
                    className="px-4 py-2 bg-gray-800 hover:bg-gray-900 text-white rounded-lg transition-colors"
                  >
                    创建目标
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 配置模态框 */}
        {showConfig && config && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    系统配置
                  </h3>
                  <button
                    onClick={() => setShowConfig(false)}
                    className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>
                
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        自主级别
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={editConfig.autonomy_level || config.autonomy_level}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => 
                          setEditConfig({ ...editConfig, autonomy_level: parseInt(e.target.value) })
                        }
                        className="w-full"
                      />
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {editConfig.autonomy_level || config.autonomy_level}%
                      </p>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        风险容忍度
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={editConfig.risk_tolerance || config.risk_tolerance}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => 
                          setEditConfig({ ...editConfig, risk_tolerance: parseInt(e.target.value) })
                        }
                        className="w-full"
                      />
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {editConfig.risk_tolerance || config.risk_tolerance}%
                      </p>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <h4 className="font-medium text-gray-900 dark:text-white">功能开关</h4>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700 dark:text-gray-300">启用自我优化</span>
                      <input
                        type="checkbox"
                        checked={editConfig.enable_self_optimization ?? config.enable_self_optimization}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => 
                          setEditConfig({ ...editConfig, enable_self_optimization: e.target.checked })
                        }
                        className="h-5 w-5 text-gray-800 rounded"
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700 dark:text-gray-300">启用安全监控</span>
                      <input
                        type="checkbox"
                        checked={editConfig.enable_safety_monitor ?? config.enable_safety_monitor}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => 
                          setEditConfig({ ...editConfig, enable_safety_monitor: e.target.checked })
                        }
                        className="h-5 w-5 text-gray-800 rounded"
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700 dark:text-gray-300">启用伦理监控</span>
                      <input
                        type="checkbox"
                        checked={editConfig.enable_ethical_monitor ?? config.enable_ethical_monitor}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => 
                          setEditConfig({ ...editConfig, enable_ethical_monitor: e.target.checked })
                        }
                        className="h-5 w-5 text-gray-800 rounded"
                      />
                    </div>
                  </div>
                </div>
                
                <div className="flex justify-end space-x-3 mt-6">
                  <button
                    onClick={() => {
                      setEditConfig(config);
                      setShowConfig(false);
                    }}
                    className="px-4 py-2 bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors"
                  >
                    取消
                  </button>
                  <button
                    onClick={handleUpdateConfig}
                    className="px-4 py-2 bg-gray-800 hover:bg-gray-900 text-white rounded-lg transition-colors"
                  >
                    保存配置
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AutonomousModePage;