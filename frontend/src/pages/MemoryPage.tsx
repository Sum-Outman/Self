import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  Brain,
  Database,
  BarChart3,
  Cpu,
  Search,
  RefreshCw,
  Layers,
  MemoryStick,
  Activity,
  Zap,
  Shield,
  CheckCircle,
  AlertCircle,
  Trash2,
  PieChart,
  BarChart,
  Eye,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { PageLoader } from '../components/UI';
import { memoryApi, MemoryStats as ApiMemoryStats, MemoryItem as ApiMemoryItem } from '../services/api/memory';

// Chart.js 导入
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
} from 'chart.js';
import { Bar, Pie } from 'react-chartjs-2';

// 记忆可视化组件导入
import MemoryNetworkGraph from '../components/MemoryVisualization/MemoryNetworkGraph';
import MemoryTimeline from '../components/MemoryVisualization/MemoryTimeline';
import ConflictResolutionVisualization from '../components/MemoryVisualization/ConflictResolutionVisualization';

// 注册Chart.js组件
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement
);

// 记忆统计接口（兼容API接口）
interface MemoryStats extends ApiMemoryStats {
  autonomous_optimizations: number;
  scene_transitions: number;
  current_scene: string;
  reasoning_operations: number;
  last_updated: string;
}

// 记忆项接口（兼容API接口）
interface MemoryItem extends ApiMemoryItem {}

const MemoryPage: React.FC = () => {
  const { user: _user, isAdmin: _isAdmin } = useAuth();
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null);
  const [recentMemories, setRecentMemories] = useState<MemoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'memories' | 'visualization' | 'settings'>('overview');
  const [searchQuery, setSearchQuery] = useState('');

  // 加载记忆统计信息
  const loadMemoryStats = async () => {
    try {
      setLoading(true);
      
      // 使用真实API调用获取记忆统计信息
      const statsResponse = await memoryApi.getStats();
      const recentResponse = await memoryApi.getRecentMemories(10);
      
      if (statsResponse.success && statsResponse.data) {
        // statsResponse现在是MemoryStatsResponse类型，包含system_initialized
        // 确保memoryStats数据包含system_initialized
        const memoryStatsData = {
          ...statsResponse.data,
          system_initialized: statsResponse.system_initialized || false
        };
        setMemoryStats(memoryStatsData);
      } else {
        toast.error(`获取记忆统计信息失败: ${statsResponse.message || '未知错误'}`);
        setMemoryStats(null);
      }
      
      if (recentResponse.success && recentResponse.data) {
        setRecentMemories(recentResponse.data);
      } else {
        // API调用失败，显示错误信息，不生成虚拟数据
        toast.error(`获取最近记忆失败: ${recentResponse.message || '未知错误'}`);
        // 保持recentMemories为空数组（不生成虚拟数据）
        setRecentMemories([]);
      }
    } catch (error: any) {
      console.error('加载记忆统计信息失败:', error);
      toast.error(`加载记忆统计信息失败: ${error.message || '未知错误'}`);
      
      // 错误处理：不生成虚拟数据，保持原有状态
      // memoryStats保持为null（已在第93行设置）
      // recentMemories保持为空数组（已在第102行设置）
      // 仅记录错误，不提供虚假数据
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadMemoryStats();
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    loadMemoryStats();
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      toast('请输入搜索关键词');
      return;
    }

    try {
      const response = await memoryApi.searchMemories({
        query: searchQuery,
        limit: 10,
        offset: 0
      });

      if (response.success && response.data) {
        const searchResults = response.data.memories;
        if (searchResults.length === 0) {
          toast('未找到相关记忆');
        } else {
          // 显示搜索结果
          setRecentMemories(searchResults);
          toast.success(`找到 ${searchResults.length} 条相关记忆`);
          
          // 如果当前不在记忆内容标签页，切换到该标签页
          if (activeTab !== 'memories') {
            setActiveTab('memories');
          }
        }
      } else {
        toast.error(`搜索失败: ${response.message || '未知错误'}`);
      }
    } catch (error: any) {
      console.error('搜索记忆失败:', error);
      toast.error(`搜索记忆失败: ${error.message || '未知错误'}`);
    }
  };

  // 格式化百分比
  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  // 格式化文件大小
  const formatFileSize = (sizeMB: number) => {
    if (sizeMB >= 1024) {
      return `${(sizeMB / 1024).toFixed(1)} GB`;
    }
    return `${sizeMB.toFixed(0)} MB`;
  };

  // 格式化日期
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('zh-CN');
  };

  // 获取记忆类型颜色
  const getMemoryTypeColor = (type: string) => {
    switch (type) {
      case 'short_term':
        return 'bg-gray-700 text-gray-700';
      case 'long_term':
        return 'bg-gray-600 text-gray-600';
      case 'working':
        return 'bg-gray-600 text-gray-600';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // 获取场景类型颜色
  const getSceneTypeColor = (sceneType?: string) => {
    switch (sceneType) {
      case 'task':
        return 'bg-gray-700 text-gray-700';
      case 'learning':
        return 'bg-gray-600 text-gray-600';
      case 'problem_solving':
        return 'bg-gray-500 text-gray-500';
      case 'social':
        return 'bg-gray-500 text-gray-500';
      case 'planning':
        return 'bg-gray-800 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // 获取重要性颜色
  const getImportanceColor = (importance: number) => {
    if (importance >= 0.8) return 'text-gray-800';
    if (importance >= 0.6) return 'text-gray-500';
    return 'text-gray-600';
  };

  if (loading && !memoryStats) {
    return <PageLoader />;
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        {/* 页面标题和操作 */}
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-2xl md:text-3xl font-bold text-gray-900 flex items-center gap-2">
                <Brain className="w-8 h-8 text-gray-700" />
                记忆管理系统
              </h1>
              {memoryStats && (
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  memoryStats.system_initialized
                    ? 'bg-gray-600 text-gray-600 border border-gray-600'
                    : 'bg-gray-500 text-gray-500 border border-gray-500'
                }`}>
                  {memoryStats.system_initialized ? '系统已初始化' : '系统未初始化'}
                </span>
              )}
            </div>
            <p className="text-gray-600 mt-2">
              AGI记忆系统的实时监控和管理，包括短期记忆、长期记忆、工作记忆和知识库集成
            </p>
          </div>
          <div className="flex items-center gap-3 mt-4 md:mt-0">
            <div className="relative">
              <input
                type="text"
                placeholder="搜索记忆..."
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-700 focus:border-gray-700 w-full md:w-64"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              />
              <Search className="absolute left-3 top-2.5 w-5 h-5 text-gray-400" />
            </div>
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="flex items-center gap-2 px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
              {refreshing ? '刷新中...' : '刷新'}
            </button>
          </div>
        </div>

        {/* 标签页导航 */}
        <div className="border-b border-gray-200 mb-6">
          <nav className="flex space-x-8">
            {[
              { id: 'overview', label: '概览', icon: BarChart3 },
              { id: 'memories', label: '记忆内容', icon: Database },
              { id: 'visualization', label: '可视化', icon: Eye },
              { id: 'settings', label: '设置', icon: Shield },
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-2 py-3 px-1 border-b-2 text-sm font-medium ${
                    activeTab === tab.id
                      ? 'border-gray-700 text-gray-700'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>

        {activeTab === 'overview' && memoryStats && (
          <>
            {/* 统计卡片 */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
              <div className="bg-white rounded-xl shadow p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">总记忆数量</p>
                    <p className="text-2xl font-bold text-gray-900 mt-1">
                      {memoryStats.total_memories.toLocaleString()}
                    </p>
                  </div>
                  <Database className="w-10 h-10 text-gray-700" />
                </div>
                <div className="mt-4 flex items-center text-sm">
                  <div className="flex-1">
                    <div className="flex justify-between mb-1">
                      <span className="text-gray-600">短期记忆</span>
                      <span className="font-medium">
                        {memoryStats.short_term_memories}
                      </span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gray-700"
                        style={{
                          width: `${(memoryStats.short_term_memories / memoryStats.total_memories) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl shadow p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">缓存命中率</p>
                    <p className="text-2xl font-bold text-gray-900 mt-1">
                      {formatPercent(memoryStats.cache_hit_rate)}
                    </p>
                  </div>
                  <MemoryStick className="w-10 h-10 text-gray-600" />
                </div>
                <div className="mt-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">性能状态</span>
                    <span
                      className={`font-medium ${
                        memoryStats.cache_hit_rate >= 0.7
                          ? 'text-gray-600'
                          : memoryStats.cache_hit_rate >= 0.5
                          ? 'text-gray-500'
                          : 'text-gray-800'
                      }`}
                    >
                      {memoryStats.cache_hit_rate >= 0.7
                        ? '优秀'
                        : memoryStats.cache_hit_rate >= 0.5
                        ? '良好'
                        : '需要优化'}
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden mt-2">
                    <div
                      className="h-full bg-gray-700"
                      style={{ width: `${memoryStats.cache_hit_rate * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl shadow p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">检索响应时间</p>
                    <p className="text-2xl font-bold text-gray-900 mt-1">
                      {memoryStats.average_retrieval_time_ms}ms
                    </p>
                  </div>
                  <Zap className="w-10 h-10 text-gray-500" />
                </div>
                <div className="mt-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">性能状态</span>
                    <span
                      className={`font-medium ${
                        memoryStats.average_retrieval_time_ms <= 100
                          ? 'text-gray-600'
                          : memoryStats.average_retrieval_time_ms <= 200
                          ? 'text-gray-500'
                          : 'text-gray-800'
                      }`}
                    >
                      {memoryStats.average_retrieval_time_ms <= 100
                        ? '极快'
                        : memoryStats.average_retrieval_time_ms <= 200
                        ? '良好'
                        : '较慢'}
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden mt-2">
                    <div
                      className={`h-full ${
                        memoryStats.average_retrieval_time_ms <= 100
                          ? 'bg-gray-700'
                          : memoryStats.average_retrieval_time_ms <= 200
                          ? 'bg-gray-800'
                          : 'bg-gray-900'
                      }`}
                      style={{
                        width: `${Math.min(100, (300 - memoryStats.average_retrieval_time_ms) / 3)}%`,
                      }}
                    />
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl shadow p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">当前场景</p>
                    <p className="text-2xl font-bold text-gray-900 mt-1 capitalize">
                      {memoryStats.current_scene}
                    </p>
                  </div>
                  <Activity className="w-10 h-10 text-gray-600" />
                </div>
                <div className="mt-4">
                  <div className="flex items-center justify-between text-sm mb-2">
                    <span className="text-gray-600">场景切换次数</span>
                    <span className="font-medium">
                      {memoryStats.scene_transitions}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">自主优化次数</span>
                    <span className="font-medium">
                      {memoryStats.autonomous_optimizations}
                    </span>
                  </div>
                </div>
              </div>

              {/* 系统状态卡片 */}
              <div className="bg-white rounded-xl shadow p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">系统初始化状态</p>
                    <p className="text-2xl font-bold text-gray-900 mt-1">
                      {memoryStats.system_initialized ? '正常' : '未初始化'}
                    </p>
                  </div>
                  <Shield className={`w-10 h-10 ${
                    memoryStats.system_initialized ? 'text-gray-600' : 'text-gray-500'
                  }`} />
                </div>
                <div className="mt-4">
                  <div className="flex items-center justify-between text-sm mb-2">
                    <span className="text-gray-600">全局状态管理器</span>
                    <span className={`font-medium ${
                      memoryStats.system_initialized
                        ? 'text-gray-600'
                        : 'text-gray-500'
                    }`}>
                      {memoryStats.system_initialized ? '已连接' : '未连接'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">API版本</span>
                    <span className="font-medium text-gray-900">v1.0.0</span>
                  </div>
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-xs text-gray-500">
                      最后更新: {new Date(memoryStats.last_updated).toLocaleString('zh-CN')}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* 系统状态 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              {/* 系统资源使用 */}
              <div className="bg-white rounded-xl shadow p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <Cpu className="w-5 h-5" />
                  系统资源使用
                </h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-600">工作内存使用率</span>
                      <span className="text-sm font-medium">{memoryStats.working_memory_usage}%</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gray-700"
                        style={{ width: `${memoryStats.working_memory_usage}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-600">总内存使用</span>
                      <span className="text-sm font-medium">{formatFileSize(memoryStats.memory_usage_mb)}</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gray-600"
                        style={{ width: `${Math.min(100, memoryStats.memory_usage_mb / 10)}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-600">推理操作次数</span>
                      <span className="text-sm font-medium">{memoryStats.reasoning_operations}</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gray-700"
                        style={{ width: `${Math.min(100, memoryStats.reasoning_operations / 20)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* 功能状态 */}
              <div className="bg-white rounded-xl shadow p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <Layers className="w-5 h-5" />
                  功能状态
                </h3>
                <div className="space-y-3">
                  {[
                    { label: '自主记忆管理', enabled: true, description: '自我优化的记忆策略' },
                    { label: '情境感知', enabled: true, description: '场景分类和切换检测' },
                    { label: '认知推理集成', enabled: true, description: '记忆与推理深度融合' },
                    { label: '多模态记忆', enabled: true, description: '支持文本、图像、音频' },
                    { label: '知识库集成', enabled: true, description: '与知识库协同工作' },
                  ].map((feature, idx) => (
                    <div key={idx} className="flex items-center justify-between">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-gray-900">{feature.label}</span>
                          {feature.enabled ? (
                            <CheckCircle className="w-4 h-4 text-gray-600" />
                          ) : (
                            <AlertCircle className="w-4 h-4 text-gray-400" />
                          )}
                        </div>
                        <p className="text-sm text-gray-500">{feature.description}</p>
                      </div>
                      <span
                        className={`px-3 py-1 rounded-full text-xs font-medium ${
                          feature.enabled
                            ? 'bg-gray-600 text-gray-600'
                            : 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {feature.enabled ? '已启用' : '已禁用'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* 记忆统计图表 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              {/* 记忆分布饼图 */}
              <div className="bg-white rounded-xl shadow p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <PieChart className="w-5 h-5" />
                  记忆分布
                </h3>
                <div className="h-64">
                  {memoryStats && (
                    <Pie
                      data={{
                        labels: ['短期记忆', '长期记忆', '工作记忆'],
                        datasets: [
                          {
                            data: [
                              memoryStats.short_term_memories,
                              memoryStats.long_term_memories,
                              memoryStats.total_memories - memoryStats.short_term_memories - memoryStats.long_term_memories,
                            ],
                            backgroundColor: [
                              'rgba(59, 130, 246, 0.7)',
                              'rgba(139, 92, 246, 0.7)',
                              'rgba(16, 185, 129, 0.7)',
                            ],
                            borderColor: [
                              'rgb(59, 130, 246)',
                              'rgb(139, 92, 246)',
                              'rgb(16, 185, 129)',
                            ],
                            borderWidth: 1,
                          },
                        ],
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            position: 'bottom',
                          },
                          tooltip: {
                            callbacks: {
                              label: (context) => {
                                const label = context.label || '';
                                const value = context.parsed;
                                const total = context.dataset.data.reduce((a: number, b: number) => a + b, 0);
                                const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
                                return `${label}: ${value} (${percentage}%)`;
                              },
                            },
                          },
                        },
                      }}
                    />
                  )}
                </div>
              </div>

              {/* 性能指标条形图 */}
              <div className="bg-white rounded-xl shadow p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <BarChart className="w-5 h-5" />
                  性能指标
                </h3>
                <div className="h-64">
                  {memoryStats && (
                    <Bar
                      data={{
                        labels: ['缓存命中率', '工作内存使用率', '平均检索时间'],
                        datasets: [
                          {
                            label: '性能指标',
                            data: [
                              memoryStats.cache_hit_rate * 100,
                              memoryStats.working_memory_usage,
                              memoryStats.average_retrieval_time_ms,
                            ],
                            backgroundColor: [
                              'rgba(34, 197, 94, 0.7)',
                              'rgba(59, 130, 246, 0.7)',
                              'rgba(245, 158, 11, 0.7)',
                            ],
                            borderColor: [
                              'rgb(34, 197, 94)',
                              'rgb(59, 130, 246)',
                              'rgb(245, 158, 11)',
                            ],
                            borderWidth: 1,
                          },
                        ],
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                          y: {
                            beginAtZero: true,
                            ticks: {
                              callback: (value) => {
                                if (typeof value === 'number') {
                                  return value.toFixed(1);
                                }
                                return value;
                              },
                            },
                          },
                        },
                        plugins: {
                          tooltip: {
                            callbacks: {
                              label: (context) => {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y;
                                let suffix = '';
                                if (context.dataIndex === 0) suffix = '%';
                                if (context.dataIndex === 1) suffix = '%';
                                if (context.dataIndex === 2) suffix = 'ms';
                                // 安全检查：确保value不是null或undefined
                                if (value === null || value === undefined) {
                                  return `${label}: 0${suffix}`;
                                }
                                return `${label}: ${value.toFixed(2)}${suffix}`;
                              },
                            },
                          },
                        },
                      }}
                    />
                  )}
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'memories' && (
          <div className="bg-white rounded-xl shadow overflow-hidden">
            <div className="p-6 border-b border-gray-200">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">记忆内容</h3>
                  <p className="text-gray-600 mt-1">搜索和管理记忆记录</p>
                </div>
                <div className="flex items-center gap-2">
                  <button className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2">
                    <Search className="w-4 h-4" />
                    高级搜索
                  </button>
                </div>
              </div>
              
              {/* 搜索表单 */}
              <div className="mt-6">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="md:col-span-2">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                      <input
                        type="text"
                        placeholder="搜索记忆内容..."
                        className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-700 focus:border-gray-700"
                      />
                    </div>
                  </div>
                  <div>
                    <select className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-700 focus:border-gray-700">
                      <option value="">所有类型</option>
                      <option value="short_term">短期记忆</option>
                      <option value="long_term">长期记忆</option>
                      <option value="working">工作记忆</option>
                    </select>
                  </div>
                  <div>
                    <button className="w-full px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-700 flex items-center justify-center gap-2">
                      <Search className="w-4 h-4" />
                      搜索
                    </button>
                  </div>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <span className="text-sm text-gray-600">快速筛选:</span>
                  <button className="px-3 py-1 text-sm bg-gray-100 text-gray-800 rounded-full hover:bg-gray-200">最近一天</button>
                  <button className="px-3 py-1 text-sm bg-gray-100 text-gray-800 rounded-full hover:bg-gray-200">高重要性</button>
                  <button className="px-3 py-1 text-sm bg-gray-100 text-gray-800 rounded-full hover:bg-gray-200">系统记忆</button>
                  <button className="px-3 py-1 text-sm bg-gray-100 text-gray-800 rounded-full hover:bg-gray-200">用户记忆</button>
                </div>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      内容
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      类型
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      场景
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      重要性
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      创建时间
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      操作
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {recentMemories.map((memory) => (
                    <tr key={memory.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4">
                        <div className="flex items-center">
                          <div className="ml-4">
                            <div className="text-sm font-medium text-gray-900 line-clamp-2">
                              {memory.content}
                            </div>
                            <div className="text-sm text-gray-500">
                              来源: {memory.source === 'user' ? '用户' : memory.source === 'system' ? '系统' : '自主'}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 text-xs rounded-full ${getMemoryTypeColor(memory.type)}`}>
                          {memory.type === 'short_term' ? '短期记忆' : 
                           memory.type === 'long_term' ? '长期记忆' : '工作记忆'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {memory.scene_type && (
                          <span className={`px-2 py-1 text-xs rounded-full ${getSceneTypeColor(memory.scene_type)}`}>
                            {memory.scene_type === 'task' ? '任务' :
                             memory.scene_type === 'learning' ? '学习' :
                             memory.scene_type === 'problem_solving' ? '问题解决' :
                             memory.scene_type === 'social' ? '社交' :
                             memory.scene_type === 'planning' ? '规划' : memory.scene_type}
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className={`text-sm font-medium ${getImportanceColor(memory.importance)}`}>
                            {(memory.importance * 100).toFixed(0)}%
                          </div>
                          <div className="ml-2 w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gray-700"
                              style={{ width: `${memory.importance * 100}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDate(memory.created_at)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <button className="text-gray-700 hover:text-gray-700 mr-4">
                          查看
                        </button>
                        <button className="text-gray-800 hover:text-gray-800">
                          删除
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
              <div className="text-sm text-gray-700">
                显示 <span className="font-medium">5</span> 条记录，共{' '}
                <span className="font-medium">{recentMemories.length}</span> 条
              </div>
              <div className="flex items-center space-x-2">
                <button className="px-3 py-1 border border-gray-300 rounded text-sm hover:bg-gray-50">
                  上一页
                </button>
                <button className="px-3 py-1 border border-gray-300 rounded text-sm hover:bg-gray-50">
                  下一页
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'visualization' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">记忆可视化</h3>
              <p className="text-gray-600 mb-6">
                使用多种可视化方式探索记忆系统的内部结构和状态。包括记忆关联网络、时间线视图和冲突解决过程。
              </p>
              
              {/* 可视化标签页导航 */}
              <div className="border-b border-gray-200 mb-6">
                <nav className="flex space-x-6">
                  {[
                    { id: 'network', label: '关联网络' },
                    { id: 'timeline', label: '时间线' },
                    { id: 'conflict', label: '冲突解决' },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab('visualization')} // 保持在同一标签页，但可以扩展为子标签页
                      className={`py-2 px-1 border-b-2 text-sm font-medium ${
                        true ? 'border-gray-700 text-gray-700' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </nav>
              </div>

              {/* 记忆网络图 */}
              <div className="mb-8">
                <h4 className="text-md font-medium text-gray-900 mb-4">记忆关联网络</h4>
                <MemoryNetworkGraph height={400} maxNodes={30} />
              </div>

              {/* 记忆时间线 */}
              <div className="mb-8">
                <h4 className="text-md font-medium text-gray-900 mb-4">记忆时间线</h4>
                <MemoryTimeline height={400} days={7} />
              </div>

              {/* 冲突解决可视化 */}
              <div>
                <h4 className="text-md font-medium text-gray-900 mb-4">记忆冲突解决</h4>
                <ConflictResolutionVisualization height={500} />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="bg-white rounded-xl shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">记忆系统设置</h3>
            <div className="space-y-6">
              <div className="border-b border-gray-200 pb-6">
                <h4 className="text-md font-medium text-gray-900 mb-4">性能设置</h4>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">工作内存容量</p>
                      <p className="text-sm text-gray-500">增加容量可处理更复杂的任务，但会消耗更多内存</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <button className="px-3 py-1 border border-gray-300 rounded text-sm hover:bg-gray-50">
                        100条
                      </button>
                      <button className="px-3 py-1 bg-gray-700 text-white rounded text-sm">
                        200条
                      </button>
                      <button className="px-3 py-1 border border-gray-300 rounded text-sm hover:bg-gray-50">
                        500条
                      </button>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">缓存大小</p>
                      <p className="text-sm text-gray-500">增加缓存大小可提高检索速度</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <button className="px-3 py-1 border border-gray-300 rounded text-sm hover:bg-gray-50">
                        100MB
                      </button>
                      <button className="px-3 py-1 bg-gray-700 text-white rounded text-sm">
                        500MB
                      </button>
                      <button className="px-3 py-1 border border-gray-300 rounded text-sm hover:bg-gray-50">
                        1GB
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-b border-gray-200 pb-6">
                <h4 className="text-md font-medium text-gray-900 mb-4">功能开关</h4>
                <div className="space-y-4">
                  {[
                    { label: '启用自主记忆管理', description: '允许系统自动优化记忆策略', enabled: true },
                    { label: '启用情境感知', description: '根据场景自动调整记忆检索', enabled: true },
                    { label: '启用知识库集成', description: '与知识库系统协同工作', enabled: true },
                    { label: '启用多模态记忆', description: '支持图像、音频等非文本记忆', enabled: true },
                  ].map((setting, idx) => (
                    <div key={idx} className="flex items-center justify-between">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{setting.label}</span>
                          {setting.enabled ? (
                            <CheckCircle className="w-4 h-4 text-gray-600" />
                          ) : (
                            <AlertCircle className="w-4 h-4 text-gray-400" />
                          )}
                        </div>
                        <p className="text-sm text-gray-500">{setting.description}</p>
                      </div>
                      <button
                        className={`relative inline-flex h-6 w-11 items-center rounded-full ${
                          setting.enabled ? 'bg-gray-700' : 'bg-gray-200'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
                            setting.enabled ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-md font-medium text-gray-900 mb-4">系统操作</h4>
                <div className="space-y-3">
                  <button className="w-full px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-700 flex items-center justify-center gap-2">
                    <RefreshCw className="w-4 h-4" />
                    清除短期记忆缓存
                  </button>
                  <button className="w-full px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-500 flex items-center justify-center gap-2">
                    <Database className="w-4 h-4" />
                    重新构建记忆索引
                  </button>
                  <button className="w-full px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-800 flex items-center justify-center gap-2">
                    <Trash2 className="w-4 h-4" />
                    删除不重要记忆（高风险）
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 页脚信息 */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>记忆系统最后更新: {memoryStats ? formatDate(memoryStats.last_updated) : '未知'}</p>
          <p className="mt-1">© 2025 Self AGI System. 所有高级AGI记忆功能已启用。</p>
        </div>
      </div>
    </div>
  );
};

export default MemoryPage;