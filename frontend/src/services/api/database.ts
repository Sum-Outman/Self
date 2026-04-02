/**
 * 数据库API服务
 * 提供数据库查询优化、性能监控、索引管理等功能
 */

import { ApiClient } from './client';

export interface QueryPerformanceReport {
  timestamp: string;
  uptime_seconds: number;
  total_queries: number;
  queries_per_second: number;
  slow_query_threshold_ms: number;
  total_slow_queries: number;
  total_optimized_queries: number;
  top_slow_queries: Array<{
    fingerprint: string;
    average_time_ms: number;
    count: number;
  }>;
  recent_slow_queries: Array<{
    query: string;
    fingerprint: string;
    execution_time_ms: number;
    params?: any;
    timestamp: string;
    threshold: number;
  }>;
  recent_optimizations: Array<{
    original_query: string;
    optimized_query: string;
    improvement_percent: number;
    original_time_ms: number;
    optimized_time_ms: number;
    timestamp: string;
  }>;
  query_statistics: {
    unique_queries: number;
    most_frequent_query: [string, number] | null;
    average_execution_time_ms: number;
  };
}

export interface TableAnalysis {
  table_name: string;
  column_count: number;
  index_count: number;
  primary_keys: string[];
  columns: Array<{
    name: string;
    type: string;
    nullable: boolean;
    default: any;
    is_primary_key: boolean;
    has_index: boolean;
    optimization_suggestions: string[];
  }>;
  indexes: Array<{
    name: string;
    columns: string[];
    unique: boolean;
    optimization_suggestions: string[];
  }>;
  overall_suggestions: string[];
}

export interface DatabaseMonitorReport {
  timestamp: string;
  running: boolean;
  update_interval: number;
  metrics_history_summary: Record<string, number>;
  active_alerts: Array<{
    type: string;
    message: string;
    severity: string;
    timestamp: string;
    acknowledged?: boolean;
  }>;
  total_alerts: number;
  recent_alerts: Array<{
    type: string;
    message: string;
    severity: string;
    timestamp: string;
  }>;
}

export interface DatabaseHealthCheck {
  service: string;
  status: 'healthy' | 'unhealthy' | 'warning';
  timestamp: string;
  database_type: string;
  connection_status: boolean;
  table_count: number;
  slow_query_count: number;
  optimization_enabled: boolean;
  suggestions: string[];
}

export interface QueryOptimizationRequest {
  query: string;
  language?: 'sql' | 'orm';
  analyze_only?: boolean;
}

export interface QueryOptimizationResult {
  success: boolean;
  original_query: string;
  optimized_query?: string;
  optimization_applied: boolean;
  optimization_rules: Array<{
    name: string;
    description: string;
    details: any;
  }>;
  estimated_improvement_percent?: number;
  execution_plan?: any;
  suggestions: string[];
}

export class DatabaseApi {
  private client: ApiClient;

  constructor(client?: ApiClient) {
    this.client = client || ApiClient.getInstance();
  }

  /**
   * 获取数据库健康状态
   */
  async getHealthCheck(): Promise<DatabaseHealthCheck> {
    return this.client.get<DatabaseHealthCheck>('/api/database/health');
  }

  /**
   * 获取查询性能报告
   */
  async getQueryPerformanceReport(): Promise<QueryPerformanceReport> {
    return this.client.get<QueryPerformanceReport>('/api/database/query-performance');
  }

  /**
   * 分析表结构
   * @param tableName 表名
   */
  async analyzeTable(tableName: string): Promise<TableAnalysis> {
    return this.client.get<TableAnalysis>(`/api/database/tables/${tableName}/analyze`);
  }

  /**
   * 获取所有表名
   */
  async getTables(): Promise<string[]> {
    return this.client.get<string[]>('/api/database/tables');
  }

  /**
   * 获取数据库监控报告
   */
  async getMonitorReport(): Promise<DatabaseMonitorReport> {
    return this.client.get<DatabaseMonitorReport>('/api/database/monitor');
  }

  /**
   * 启动数据库监控
   */
  async startMonitoring(): Promise<{ success: boolean; message: string }> {
    return this.client.post<{ success: boolean; message: string }>('/api/database/monitor/start');
  }

  /**
   * 停止数据库监控
   */
  async stopMonitoring(): Promise<{ success: boolean; message: string }> {
    return this.client.post<{ success: boolean; message: string }>('/api/database/monitor/stop');
  }

  /**
   * 优化查询
   * @param request 查询优化请求
   */
  async optimizeQuery(request: QueryOptimizationRequest): Promise<QueryOptimizationResult> {
    return this.client.post<QueryOptimizationResult>('/api/database/optimize', request);
  }

  /**
   * 执行慢查询分析
   * @param limit 限制返回的慢查询数量
   */
  async analyzeSlowQueries(limit: number = 10): Promise<Array<{
    query: string;
    fingerprint: string;
    execution_time_ms: number;
    count: number;
    average_time_ms: number;
    last_executed: string;
    suggestions: string[];
  }>> {
    return this.client.get(`/api/database/slow-queries?limit=${limit}`);
  }

  /**
   * 获取数据库统计信息
   */
  async getStatistics(): Promise<{
    total_tables: number;
    total_rows: number;
    total_indexes: number;
    database_size_bytes: number;
    average_row_size_bytes: number;
    most_frequent_queries: Array<{ query: string; count: number; avg_time_ms: number }>;
  }> {
    return this.client.get('/api/database/statistics');
  }

  /**
   * 创建索引
   * @param tableName 表名
   * @param columns 列名数组
   * @param indexName 索引名（可选）
   * @param unique 是否唯一索引
   */
  async createIndex(
    tableName: string,
    columns: string[],
    indexName?: string,
    unique: boolean = false
  ): Promise<{ success: boolean; message: string; index_name: string }> {
    return this.client.post(`/api/database/tables/${tableName}/indexes`, {
      columns,
      index_name: indexName,
      unique,
    });
  }

  /**
   * 删除索引
   * @param tableName 表名
   * @param indexName 索引名
   */
  async dropIndex(
    tableName: string,
    indexName: string
  ): Promise<{ success: boolean; message: string }> {
    return this.client.delete(`/api/database/tables/${tableName}/indexes/${indexName}`);
  }

  /**
   * 获取数据库配置
   */
  async getConfiguration(): Promise<{
    database_type: string;
    connection_pool_size: number;
    max_connections: number;
    slow_query_threshold_ms: number;
    query_cache_enabled: boolean;
    query_cache_ttl_seconds: number;
    monitoring_enabled: boolean;
    optimization_enabled: boolean;
  }> {
    return this.client.get('/api/database/configuration');
  }

  /**
   * 更新数据库配置
   * @param config 配置对象
   */
  async updateConfiguration(config: {
    slow_query_threshold_ms?: number;
    query_cache_enabled?: boolean;
    query_cache_ttl_seconds?: number;
    monitoring_enabled?: boolean;
  }): Promise<{ success: boolean; message: string; updated_config: any }> {
    return this.client.put('/api/database/configuration', config);
  }

  /**
   * 执行数据库维护
   * @param maintenanceType 维护类型
   */
  async runMaintenance(maintenanceType: 'vacuum' | 'analyze' | 'reindex' = 'vacuum'): Promise<{
    success: boolean;
    message: string;
    execution_time_ms: number;
    affected_objects?: any[];
  }> {
    return this.client.post(`/api/database/maintenance/${maintenanceType}`);
  }

  /**
   * 获取数据库备份列表
   */
  async getBackups(): Promise<Array<{
    filename: string;
    size_bytes: number;
    created_at: string;
    backup_type: string;
  }>> {
    return this.client.get('/api/database/backups');
  }

  /**
   * 创建数据库备份
   * @param backupType 备份类型
   */
  async createBackup(backupType: 'full' | 'incremental' = 'full'): Promise<{
    success: boolean;
    message: string;
    backup_file: string;
    backup_size_bytes: number;
  }> {
    return this.client.post('/api/database/backup', { backup_type: backupType });
  }

  /**
   * 恢复数据库备份
   * @param backupFile 备份文件名
   */
  async restoreBackup(backupFile: string): Promise<{
    success: boolean;
    message: string;
    restore_time_ms: number;
  }> {
    return this.client.post('/api/database/restore', { backup_file: backupFile });
  }
}

// 导出默认实例
export default new DatabaseApi();