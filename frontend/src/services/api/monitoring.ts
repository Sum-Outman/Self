import { apiClient } from './client';
import { ApiResponse } from '../../types/api';

// 监控相关类型定义
export interface SystemMetrics {
  cpu: {
    percent: number;
    cores: number;
    threads: number;
  };
  memory: {
    total: number; // MB
    available: number; // MB
    used: number; // MB
    percent: number;
  };
  disk: {
    total: number; // GB
    used: number; // GB
    free: number; // GB
    percent: number;
  };
  process: {
    memory_mb: number;
    cpu_percent: number;
    threads: number;
    create_time: string;
  };
  network: {
    bytes_sent: number;
    bytes_recv: number;
    packets_sent: number;
    packets_recv: number;
  };
  system: {
    boot_time: string;
    users: number;
    uptime: number;
  };
}

export interface Alert {
  id: string;
  alert_type: string;
  severity: string;
  message: string;
  source: string;
  timestamp: string;
  status: string;
  acknowledged_by?: string;
  acknowledged_at?: string;
  resolved_by?: string;
  resolved_at?: string;
  details?: Record<string, any>;
}

export interface SystemMetricsResponse extends ApiResponse {
  timestamp: string;
  metrics: SystemMetrics;
}

export interface AlertsResponse extends ApiResponse {
  alerts: Alert[];
  total: number;
}

// 服务状态相关类型
export interface ServiceStatus {
  service_name: string;
  display_name: string;
  status: 'online' | 'offline' | 'degraded' | 'maintenance';
  response_time_ms: number;
  last_check: string;
  health_score: number;
  dependencies: string[];
  uptime_percent: number;
  error_rate: number;
  warning_count: number;
}

export interface ServicesStatusResponse extends ApiResponse {
  services: ServiceStatus[];
  overall_health: number;
  timestamp: string;
  check_duration_ms: number;
}

export const monitoringService = {
  // 获取系统监控指标
  async getSystemMetrics(metricType?: string, timeRange: string = '1h'): Promise<SystemMetricsResponse> {
    return apiClient.get('/monitoring/system/metrics', {
      params: { metric_type: metricType, time_range: timeRange }
    });
  },

  // 获取告警列表
  async getAlerts(
    severity?: string,
    startTime?: string,
    endTime?: string,
    limit: number = 100
  ): Promise<AlertsResponse> {
    return apiClient.get('/monitoring/alerts', {
      params: { severity, start_time: startTime, end_time: endTime, limit }
    });
  },

  // 获取系统健康状态
  async getSystemHealth(): Promise<ApiResponse> {
    return apiClient.get('/monitoring/health');
  },

  // 获取API性能统计
  async getApiStats(): Promise<ApiResponse> {
    return apiClient.get('/monitoring/api/stats');
  },

  // 获取错误统计
  async getErrorStats(timeRange: string = '24h'): Promise<ApiResponse> {
    return apiClient.get('/monitoring/errors/stats', {
      params: { time_range: timeRange }
    });
  },

  // 获取所有服务状态
  async getServicesStatus(): Promise<ServicesStatusResponse> {
    try {
      return await apiClient.get('/monitoring/services/status');
    } catch (error) {
      console.error('获取服务状态失败:', error);
      // API失败，返回错误状态而不是虚拟数据
      return {
        success: false,
        message: '监控服务不可用，请检查后端服务状态',
        services: [],
        overall_health: 0,
        timestamp: new Date().toISOString(),
        check_duration_ms: 0,
        error: error instanceof Error ? error.message : '未知错误'
      };
    }
  },

  // 实时监控数据流（轮询方式）
  async subscribeToMetrics(callback: (data: any) => void) {
    // 注意：根据项目要求"禁止使用虚拟数据"，使用真实API轮询
    console.log('实时监控订阅已启动（轮询模式）');
    
    const interval = setInterval(async () => {
      try {
        const metrics = await this.getSystemMetrics();
        callback(metrics);
      } catch (error) {
        console.error('获取实时监控数据失败:', error);
        // 不生成虚拟数据，仅记录错误
      }
    }, 5000); // 每5秒更新一次
    
    return {
      unsubscribe: () => {
        clearInterval(interval);
        console.log('实时监控订阅已停止');
      }
    };
  }
};