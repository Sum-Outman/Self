import React, { useState, useEffect } from 'react';
import { CheckCircle, XCircle, AlertCircle, Clock, RefreshCw } from 'lucide-react';
import { monitoringService } from '../../services/api/monitoring';

interface ServiceStatus {
  name: string;
  status: 'online' | 'offline' | 'degraded' | 'maintenance';
  responseTime: number;
  lastChecked: string;
  lastUpdate: Date;
}



const SystemStatus: React.FC = () => {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [loading, setLoading] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date>(new Date());
  const [realDataAvailable, setRealDataAvailable] = useState(false);

  // 计算时间差描述
  const getTimeAgo = (date: Date): string => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    
    if (diffSec < 60) return '刚刚';
    if (diffMin < 60) return `${diffMin}分钟前`;
    if (diffHour < 24) return `${diffHour}小时前`;
    return `${Math.floor(diffHour / 24)}天前`;
  };

  // 刷新服务状态
  const refreshServices = async () => {
    setLoading(true);
    try {
      const response = await monitoringService.getServicesStatus();
      if (response.success && response.services) {
        const newServices: ServiceStatus[] = response.services.map(service => {
          // 映射后端服务状态到前端状态
          let status: ServiceStatus['status'] = 'online';
          if (service.status === 'offline') status = 'offline';
          else if (service.status === 'degraded') status = 'degraded';
          else if (service.status === 'maintenance') status = 'maintenance';
          
          return {
            name: service.display_name,
            status,
            responseTime: service.response_time_ms,
            lastChecked: getTimeAgo(new Date(service.last_check)),
            lastUpdate: new Date(service.last_check)
          };
        });
        
        setServices(newServices);
        setRealDataAvailable(true);
        setLastRefreshed(new Date());
      } else {
        // API返回失败或无数据，标记为无真实数据
        setRealDataAvailable(false);
      }
    } catch (error) {
      console.error('刷新服务状态失败:', error);
      setRealDataAvailable(false);
    } finally {
      setLoading(false);
    }
  };

  // 初始加载
  useEffect(() => {
    refreshServices();
    
    // 每30秒自动刷新
    const intervalId = setInterval(refreshServices, 30000);
    
    return () => {
      clearInterval(intervalId);
    };
  }, []);

  const getStatusIcon = (status: ServiceStatus['status']) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="h-5 w-5 text-gray-600" />;
      case 'offline':
        return <XCircle className="h-5 w-5 text-gray-800" />;
      case 'degraded':
        return <AlertCircle className="h-5 w-5 text-gray-600" />;
      case 'maintenance':
        return <Clock className="h-5 w-5 text-gray-700" />;
      default:
        return <AlertCircle className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status: ServiceStatus['status']) => {
    switch (status) {
      case 'online':
        return 'bg-gray-600 text-gray-800 dark:bg-gray-900 dark:text-gray-400';
      case 'offline':
        return 'bg-gray-800 text-gray-900 dark:bg-gray-900 dark:text-gray-600';
      case 'degraded':
        return 'bg-gray-700 text-gray-900 dark:bg-gray-900 dark:text-gray-500';
      case 'maintenance':
        return 'bg-gray-600 text-gray-900 dark:bg-gray-900 dark:text-gray-400';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300';
    }
  };

  const getStatusText = (status: ServiceStatus['status']) => {
    switch (status) {
      case 'online': return '在线';
      case 'offline': return '离线';
      case 'degraded': return '降级';
      case 'maintenance': return '维护中';
      default: return '未知';
    }
  };

  // 计算系统健康度
  const calculateSystemHealth = () => {
    if (services.length === 0) return 94; // 默认值
    
    const statusWeights: Record<ServiceStatus['status'], number> = {
      'online': 100,
      'degraded': 60,
      'maintenance': 40,
      'offline': 0
    };
    
    const totalWeight = services.reduce((sum, service) => {
      return sum + statusWeights[service.status];
    }, 0);
    
    return Math.round(totalWeight / services.length);
  };

  const systemHealth = calculateSystemHealth();
  const getHealthLabel = (score: number) => {
    if (score >= 90) return '优秀';
    if (score >= 70) return '良好';
    if (score >= 50) return '一般';
    return '需关注';
  };

  return (
    <div className="space-y-4">
      {/* 标题栏 */}
      <div className="flex items-center justify-between mb-2">
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">系统服务状态</h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {realDataAvailable ? '实时监控数据' : '无实时数据'} · 最后更新: {getTimeAgo(lastRefreshed)}
            {!realDataAvailable && ' · 后端监控服务不可用'}
          </p>
        </div>
        <button
          onClick={refreshServices}
          disabled={loading}
          className="inline-flex items-center px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          {loading ? '刷新中...' : '刷新'}
        </button>
      </div>

      {/* 服务状态网格 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {services.map((service) => (
          <div
            key={service.name}
            className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-gray-300 dark:hover:border-gray-600 transition-colors"
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium text-gray-900 dark:text-white">
                {service.name}
              </h3>
              {getStatusIcon(service.status)}
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  响应时间
                </p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {service.responseTime}ms
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  状态
                </p>
                <span
                  className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                    service.status
                  )}`}
                >
                  {getStatusText(service.status)}
                </span>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
              最后检查: {service.lastChecked}
            </div>
          </div>
        ))}
      </div>

      {/* 系统健康度 */}
      <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white">
              系统健康度
            </h4>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              基于所有服务的综合评分
            </p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-gray-900 dark:text-white">
              {systemHealth}%
            </div>
            <p className="text-sm text-gray-700 dark:text-gray-400">{getHealthLabel(systemHealth)}</p>
          </div>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${
              systemHealth >= 90 ? 'bg-gray-600' : 
              systemHealth >= 70 ? 'bg-gray-700' : 
              systemHealth >= 50 ? 'bg-gray-800' : 'bg-gray-900'
            }`}
            style={{ width: `${systemHealth}%` }}
          />
        </div>
      </div>

      {/* 状态说明 */}
      <div className="text-sm text-gray-500 dark:text-gray-400">
        <p>
          <span className="inline-flex items-center">
            <CheckCircle className="h-4 w-4 text-gray-600 mr-1" />
            在线
          </span>
          {' '}- 服务正常运行
          {' · '}
          <span className="inline-flex items-center">
            <AlertCircle className="h-4 w-4 text-gray-600 mr-1" />
            降级
          </span>
          {' '}- 服务可用但性能受影响
          {' · '}
          <span className="inline-flex items-center">
            <Clock className="h-4 w-4 text-gray-700 mr-1" />
            维护中
          </span>
          {' '}- 服务处于维护状态
          {' · '}
          <span className="inline-flex items-center">
            <XCircle className="h-4 w-4 text-gray-800 mr-1" />
            离线
          </span>
          {' '}- 服务不可用
        </p>
      </div>
    </div>
  );
};

export default SystemStatus;