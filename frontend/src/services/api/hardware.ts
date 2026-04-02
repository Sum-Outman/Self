import { ApiClient, apiClient } from './client';
import { ApiResponse } from '../../types/api';

// 硬件设备状态接口
export interface HardwareDevice {
  id: string;
  name: string;
  type: 'gpu' | 'cpu' | 'memory' | 'storage' | 'network' | 'sensor' | 'motor' | 'serial';
  status: 'online' | 'offline' | 'warning' | 'error';
  temperature: number;
  usage: number;
  capacity: number;
  model: string;
  manufacturer: string;
  device_id: string;
  device_type: string;
  connected: boolean;
  last_update: string;
  metadata?: Record<string, any>;
}

// 传感器数据接口
export interface SensorData {
  id: string;
  sensor_id: string;
  sensor_type: string;
  name: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  warning_threshold: number;
  timestamp: string;
  accuracy: number;
  calibrated: boolean;
}

// 系统指标接口
export interface SystemMetric {
  id: string;
  metric_type: string;
  metric_name: string;
  value: number;
  unit: string;
  status: 'normal' | 'warning' | 'error' | 'critical';
  threshold_warning: number;
  threshold_error: number;
  timestamp: string;
  history?: number[];
}

// 串口设备接口
export interface SerialDevice {
  device: string;
  name: string;
  description: string;
  hwid: string;
  vid?: number;
  pid?: number;
  serial_number?: string;
  location?: string;
  manufacturer?: string;
  product?: string;
  interface?: string;
}

// 系统状态接口
export interface SystemStatus {
  status: string;
  uptime: number;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_traffic: {
    bytes_sent: number;
    bytes_received: number;
  };
  active_alerts: number;
  total_metrics: number;
  timestamp: string;
}

// 电机命令接口
export interface MotorCommand {
  motor_id: string;
  command: 'move' | 'stop' | 'reset' | 'calibrate';
  target_position?: number;
  speed_factor?: number;
  blocking?: boolean;
}

// 串口命令接口
export interface SerialCommand {
  command: string;
  port?: string;
  baudrate?: number;
  wait_for_response?: boolean;
  timeout?: number;
}



class HardwareApi {
  private apiClient: ApiClient;

  constructor() {
    this.apiClient = apiClient;
  }

  // 获取硬件设备状态
  async getHardwareStatus(deviceId?: string, deviceType?: string): Promise<ApiResponse<HardwareDevice[]>> {
    try {
      const params = new URLSearchParams();
      if (deviceId) params.append('device_id', deviceId);
      if (deviceType) params.append('device_type', deviceType);
      
      const response = (await this.apiClient.get(`/hardware/status?${params.toString()}`)) as ApiResponse<HardwareDevice[]>;
      return response;
    } catch (error) {
      console.error('获取硬件状态失败:', error);
      throw error;
    }
  }

  // 获取传感器数据
  async getSensorData(
    sensorId?: string,
    sensorType?: string,
    startTime?: Date,
    endTime?: Date
  ): Promise<ApiResponse<SensorData[]>> {
    try {
      const params = new URLSearchParams();
      if (sensorId) params.append('sensor_id', sensorId);
      if (sensorType) params.append('sensor_type', sensorType);
      if (startTime) params.append('start_time', startTime.toISOString());
      if (endTime) params.append('end_time', endTime.toISOString());
      
      const response = (await this.apiClient.get(`/sensors/data?${params.toString()}`)) as ApiResponse<SensorData[]>;
      return response;
    } catch (error) {
      console.error('获取传感器数据失败:', error);
      throw error;
    }
  }

  // 发送电机命令
  async sendMotorCommand(command: MotorCommand): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.post('/motors/command', command)) as ApiResponse;
      return response;
    } catch (error) {
      console.error('发送电机命令失败:', error);
      throw error;
    }
  }

  // 发送串口命令
  async sendSerialCommand(command: SerialCommand): Promise<ApiResponse<{ response?: string }>> {
    try {
      const response = (await this.apiClient.post('/serial/command', command)) as ApiResponse<{ response?: string }>;
      return response;
    } catch (error) {
      console.error('发送串口命令失败:', error);
      throw error;
    }
  }

  // 获取系统指标
  async getSystemMetrics(
    metricType?: string,
    startTime?: Date,
    endTime?: Date,
    limit?: number
  ): Promise<ApiResponse<{
    system_status: SystemStatus;
    active_alerts: unknown[];
    metrics: SystemMetric[];
  }>> {
    try {
      const params = new URLSearchParams();
      if (metricType) params.append('metric_type', metricType);
      if (startTime) params.append('start_time', startTime.toISOString());
      if (endTime) params.append('end_time', endTime.toISOString());
      if (limit) params.append('limit', limit.toString());
      
      const response = (await this.apiClient.get(`/system/metrics?${params.toString()}`)) as ApiResponse<{
        system_status: SystemStatus;
        active_alerts: unknown[];
        metrics: SystemMetric[];
      }>;
      return response;
    } catch (error) {
      console.error('获取系统指标失败:', error);
      throw error;
    }
  }

  // 获取可用串口列表
  async getSerialPorts(): Promise<ApiResponse<SerialDevice[]>> {
    try {
      // 调用真实硬件API获取串口列表
      const response = (await this.apiClient.get('/serial/ports')) as ApiResponse<SerialDevice[]>;
      return response;
    } catch (error) {
      console.error('获取串口列表失败:', error);
      // API失败时抛出错误，不返回虚拟数据
      throw error;
    }
  }

  // 控制硬件设备电源
  async controlDevicePower(deviceId: string, powerOn: boolean): Promise<ApiResponse> {
    try {
      // 通过串口命令控制设备电源
      const command = powerOn 
        ? `POWER_ON ${deviceId}` 
        : `POWER_OFF ${deviceId}`;
      
      const response = (await this.apiClient.post('/serial/command', {
        command,
        wait_for_response: true,
        timeout: 10
      })) as ApiResponse;
      return response;
    } catch (error) {
      console.error('控制设备电源失败:', error);
      throw error;
    }
  }

  // 重启硬件设备
  async rebootDevice(deviceId: string): Promise<ApiResponse> {
    try {
      // 通过串口命令重启设备
      const response = (await this.apiClient.post('/serial/command', {
        command: `REBOOT ${deviceId}`,
        wait_for_response: true,
        timeout: 30
      })) as ApiResponse;
      return response;
    } catch (error) {
      console.error('重启设备失败:', error);
      throw error;
    }
  }
}

// 创建单例实例
export const hardwareApi = new HardwareApi();

// 默认导出
export default hardwareApi;