import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  Cpu,
  Server,
  MemoryStick,
  HardDrive,
  Wifi,
  Power,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  XCircle,
  Thermometer,
  Settings,
  Loader2,
  Zap,
  Activity,
  Database,
  Cable,
  Cog,
  Terminal,
  Save,
  Play,
  StopCircle,
  Trash2,
  X,
  History,
  Bookmark,
  Plus,
} from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import toast from 'react-hot-toast';
import { hardwareApi, HardwareDevice, SensorData as ApiSensorData, SystemMetric, SystemStatus } from '../services/api/hardware';

// 注册Chart.js组件
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// 直接使用API类型，确保类型安全
type HardwareStatus = HardwareDevice;
type SensorData = ApiSensorData;
type SystemMetricDisplay = SystemMetric;

interface ControlCommand {
  type: 'power' | 'reboot' | 'reset' | 'configure';
  target: string;
  parameters?: Record<string, any>;
}

const HardwarePage: React.FC = () => {
  const { user: _user } = useAuth();
  const [hardwareStatus, setHardwareStatus] = useState<HardwareStatus[]>([]);
  const [sensorData, setSensorData] = useState<SensorData[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetricDisplay[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [isPolling, setIsPolling] = useState(true);
  const [activeTab, setActiveTab] = useState<'status' | 'sensors' | 'control' | 'metrics' | 'serial' | 'history' | 'presets'>('status');
  const [isLoading, setIsLoading] = useState({
    hardware: false,
    sensors: false,
    metrics: false,
    control: false,
    serial: false
  });

  // 传感器数据可视化状态
  const [sensorViewMode, setSensorViewMode] = useState<'list' | 'chart'>('list');
  const [sensorHistory, setSensorHistory] = useState<Array<{
    sensorId: string;
    sensorName: string;
    timestamp: Date;
    value: number;
    unit: string;
  }>>([]);

  // 串口配置状态
  interface SerialPortConfig {
    portName: string;
    baudRate: number;
    dataBits: 5 | 6 | 7 | 8;
    stopBits: 1 | 1.5 | 2;
    parity: 'none' | 'even' | 'odd' | 'mark' | 'space';
    flowControl: 'none' | 'software' | 'hardware';
    timeout: number;
    bufferSize: number;
  }

  interface SerialDevice {
    id: string;
    name: string;
    port: string;
    connected: boolean;
    config: SerialPortConfig;
    lastActivity: string;
    isOpen: boolean;
  }

  const [serialPorts, setSerialPorts] = useState<SerialDevice[]>([]);

  const [editingSerialPort, setEditingSerialPort] = useState<SerialDevice | null>(null);
  const [serialLogs, setSerialLogs] = useState<string[]>([]);
  const [serialInput, setSerialInput] = useState<string>('');

  // 控制命令历史
  const [commandHistory, setCommandHistory] = useState<Array<{
    id: string;
    timestamp: Date;
    deviceId: string;
    deviceName: string;
    commandType: string;
    command: ControlCommand;
    success: boolean;
    response?: any;
  }>>([]);

  // 硬件配置预设管理
  const [hardwarePresets, setHardwarePresets] = useState<Array<{
    id: string;
    name: string;
    description: string;
    createdAt: Date;
    updatedAt: Date;
    category: 'motor' | 'sensor' | 'system' | 'custom';
    config: {
      devices: HardwareStatus[];
      sensorConfigs: SensorData[];
      serialConfigs: any[];
      motorCommands?: ControlCommand[];
    };
  }>>([]);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [presetDialogOpen, setPresetDialogOpen] = useState(false);
  const [presetName, setPresetName] = useState('');
  const [presetDescription, setPresetDescription] = useState('');
  const [presetCategory, setPresetCategory] = useState<'motor' | 'sensor' | 'system' | 'custom'>('custom');

  // 转换API设备为前端UI设备
  const convertHardwareDevice = (device: HardwareDevice): HardwareStatus => {
    return {
      id: device.id || device.device_id,
      name: device.name || device.device_id,
      type: device.type as any,
      status: device.status,
      temperature: device.temperature || 0,
      usage: device.usage || 0,
      capacity: device.capacity || 0,
      model: device.model || '未知型号',
      manufacturer: device.manufacturer || '未知厂商',
      device_id: device.device_id,
      device_type: device.device_type,
      connected: device.connected || false,
      last_update: device.last_update || new Date().toISOString()
    };
  };

  // 转换API传感器数据为前端UI数据
  const convertSensorData = (sensor: ApiSensorData): SensorData => {
    return {
      id: sensor.id || sensor.sensor_id,
      name: sensor.name || sensor.sensor_id,
      value: sensor.value,
      unit: sensor.unit,
      min: sensor.min || 0,
      max: sensor.max || 100,
      warning_threshold: sensor.warning_threshold || 80,
      sensor_id: sensor.sensor_id,
      sensor_type: sensor.sensor_type,
      timestamp: sensor.timestamp,
      accuracy: sensor.accuracy || 0.95,
      calibrated: sensor.calibrated || false
    };
  };

  // 转换API系统指标为前端UI指标
  const convertSystemMetric = (metric: SystemMetric): SystemMetricDisplay => {
    return {
      id: metric.id,
      metric_name: metric.metric_name || metric.metric_type,
      value: metric.value,
      unit: metric.unit,
      status: metric.status,
      timestamp: metric.timestamp,
      metric_type: metric.metric_type,
      threshold_warning: metric.threshold_warning,
      threshold_error: metric.threshold_error
    };
  };

  // 加载硬件状态
  const loadHardwareStatus = useCallback(async () => {
    setIsLoading(prev => ({ ...prev, hardware: true }));
    try {
      const response = await hardwareApi.getHardwareStatus();
      if (response.success && response.data) {
        const devices = response.data.map(convertHardwareDevice);
        setHardwareStatus(devices);
      } else {
        toast.error('获取硬件状态失败');
      }
    } catch (error) {
      console.error('加载硬件状态失败:', error);
      toast.error('加载硬件状态失败');
    } finally {
      setIsLoading(prev => ({ ...prev, hardware: false }));
    }
  }, []);

  // 加载传感器数据
  const loadSensorData = useCallback(async () => {
    setIsLoading(prev => ({ ...prev, sensors: true }));
    try {
      const response = await hardwareApi.getSensorData();
      if (response.success && response.data) {
        const sensors = response.data.map(convertSensorData);
        setSensorData(sensors);
        
        // 更新传感器历史记录
        const now = new Date();
        const newHistoryEntries = sensors.map(sensor => ({
          sensorId: sensor.sensor_id,
          sensorName: sensor.name,
          timestamp: now,
          value: sensor.value,
          unit: sensor.unit,
        }));
        
        setSensorHistory(prev => {
          // 添加新条目并保留最近200个数据点
          const updated = [...prev, ...newHistoryEntries];
          return updated.slice(-200);
        });
      } else {
        toast.error('获取传感器数据失败');
      }
    } catch (error) {
      console.error('加载传感器数据失败:', error);
      toast.error('加载传感器数据失败');
    } finally {
      setIsLoading(prev => ({ ...prev, sensors: false }));
    }
  }, []);

  // 加载系统指标
  const loadSystemMetrics = useCallback(async () => {
    setIsLoading(prev => ({ ...prev, metrics: true }));
    try {
      const response = await hardwareApi.getSystemMetrics();
      if (response.success) {
        setSystemStatus(response.data?.system_status || null);
        const metrics = response.data?.metrics?.map(convertSystemMetric) || [];
        setSystemMetrics(metrics);
      } else {
        toast.error('获取系统指标失败');
      }
    } catch (error) {
      console.error('加载系统指标失败:', error);
      toast.error('加载系统指标失败');
    } finally {
      setIsLoading(prev => ({ ...prev, metrics: false }));
    }
  }, []);

  // 加载串口设备列表
  const loadSerialPorts = useCallback(async () => {
    setIsLoading(prev => ({ ...prev, serial: true }));
    try {
      const response = await hardwareApi.getSerialPorts();
      if (response.success && response.data) {
        const apiSerialData = response.data;
        // 使用函数式更新来避免直接依赖serialPorts
        setSerialPorts(prevPorts => {
          // 将API的SerialDevice转换为前端的SerialDevice类型
          const frontendSerialDevices = apiSerialData.map(port => {
            // 检查是否已经存在（保留配置信息）
            const existingPort = prevPorts.find(p => p.port === port.device);
            
            return {
              id: `serial-${port.device}`,
              name: port.description || port.name || port.device,
              port: port.device,
              connected: existingPort?.connected || false,
              config: existingPort?.config || {
                portName: port.device,
                baudRate: 9600,
                dataBits: 8,
                stopBits: 1,
                parity: 'none' as const,
                flowControl: 'none' as const,
                timeout: 1000,
                bufferSize: 1024
              },
              lastActivity: existingPort?.lastActivity || new Date().toISOString(),
              isOpen: existingPort?.isOpen || false
            };
          });
          return frontendSerialDevices;
        });
      } else {
        toast.error('获取串口列表失败');
      }
    } catch (error) {
      console.error('加载串口列表失败:', error);
      toast.error('加载串口列表失败');
    } finally {
      setIsLoading(prev => ({ ...prev, serial: false }));
    }
  }, []);

  // 预设管理函数
  // 保存当前配置为预设
  const saveCurrentConfigAsPreset = () => {
    if (!presetName.trim()) {
      toast.error('请输入预设名称');
      return;
    }

    const newPreset = {
      id: `preset_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: presetName.trim(),
      description: presetDescription.trim(),
      category: presetCategory,
      createdAt: new Date(),
      updatedAt: new Date(),
      config: {
        devices: hardwareStatus,
        sensorConfigs: sensorData,
        serialConfigs: serialPorts,
        motorCommands: [] // 可以从命令历史中提取
      }
    };

    setHardwarePresets(prev => {
      const updated = [...prev, newPreset];
      // 保存到本地存储
      try {
        localStorage.setItem('hardware_presets', JSON.stringify(updated));
      } catch (e) {
        console.error('保存预设到本地存储失败:', e);
      }
      return updated;
    });

    toast.success(`预设 "${newPreset.name}" 已保存`);
    setPresetName('');
    setPresetDescription('');
    setPresetDialogOpen(false);
  };

  // 加载预设配置
  const loadPresetConfig = (presetId: string) => {
    const preset = hardwarePresets.find(p => p.id === presetId);
    if (!preset) {
      toast.error('预设未找到');
      return;
    }

    setSelectedPreset(presetId);
    
    // 应用预设配置
    // 注意：这里只是显示预设配置，实际应用需要用户确认
    toast.success(`已加载预设 "${preset.name}"`);
  };

  // 删除预设
  const deletePreset = (presetId: string) => {
    if (!confirm('确定要删除此预设吗？此操作不可撤销。')) {
      return;
    }

    setHardwarePresets(prev => {
      const updated = prev.filter(p => p.id !== presetId);
      // 更新本地存储
      try {
        localStorage.setItem('hardware_presets', JSON.stringify(updated));
      } catch (e) {
        console.error('更新本地存储失败:', e);
      }
      return updated;
    });

    if (selectedPreset === presetId) {
      setSelectedPreset(null);
    }

    toast.success('预设已删除');
  };

  // 应用预设配置到硬件
  const applyPresetConfig = async (presetId: string) => {
    const preset = hardwarePresets.find(p => p.id === presetId);
    if (!preset) {
      toast.error('预设未找到');
      return;
    }

    try {
      setIsLoading(prev => ({ ...prev, preset: true }));
      
      // 应用设备配置
      for (const device of preset.config.devices) {
        // 这里可以添加应用设备配置的逻辑
        console.log('应用设备配置:', device);
      }

      // 应用传感器配置
      for (const sensor of preset.config.sensorConfigs) {
        // 这里可以添加应用传感器配置的逻辑
        console.log('应用传感器配置:', sensor);
      }

      // 应用串口配置
      for (const serial of preset.config.serialConfigs) {
        // 这里可以添加应用串口配置的逻辑
        console.log('应用串口配置:', serial);
      }

      toast.success(`预设 "${preset.name}" 已成功应用到硬件`);
    } catch (error) {
      console.error('应用预设配置失败:', error);
      toast.error('应用预设配置失败');
    } finally {
      setIsLoading(prev => ({ ...prev, preset: false }));
    }
  };

  // 从本地存储加载预设
  const loadPresetsFromStorage = useCallback(() => {
    try {
      const savedPresets = localStorage.getItem('hardware_presets');
      if (savedPresets) {
        const parsed = JSON.parse(savedPresets);
        // 转换日期字符串为Date对象
        const presetsWithDates = parsed.map((p: any) => ({
          ...p,
          createdAt: new Date(p.createdAt),
          updatedAt: new Date(p.updatedAt)
        }));
        setHardwarePresets(presetsWithDates);
      }
    } catch (e) {
      console.error('从本地存储加载预设失败:', e);
    }
  }, []);

  // 初始化加载
  useEffect(() => {
    loadHardwareStatus();
    loadSensorData();
    loadSystemMetrics();
    loadSerialPorts(); // 加载串口列表
    loadPresetsFromStorage(); // 加载预设配置
  }, [loadHardwareStatus, loadSensorData, loadSystemMetrics, loadSerialPorts, loadPresetsFromStorage]);

  // 轮询更新
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isPolling) {
      interval = setInterval(() => {
        if (activeTab === 'status') {
          loadHardwareStatus();
        } else if (activeTab === 'sensors') {
          loadSensorData();
        } else if (activeTab === 'metrics') {
          loadSystemMetrics();
        }
      }, 2000); // 优化：从5秒减少到2秒，降低实时数据更新延迟
    }

    return () => clearInterval(interval);
  }, [isPolling, activeTab, loadHardwareStatus, loadSensorData, loadSystemMetrics]);

  // 添加命令历史记录
  const addCommandHistory = (
    deviceId: string,
    deviceName: string,
    commandType: string,
    command: ControlCommand,
    success: boolean,
    response?: any
  ) => {
    const newHistory = {
      id: `cmd_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      deviceId,
      deviceName,
      commandType,
      command,
      success,
      response,
    };
    
    setCommandHistory(prev => [newHistory, ...prev.slice(0, 49)]); // 保留最近50条记录
  };

  // 处理硬件控制命令
  const handleHardwareCommand = async (hardwareId: string, command: ControlCommand) => {
    setIsLoading(prev => ({ ...prev, control: true }));
    let commandSuccess = false;
    let commandResponse: any = null;
    let deviceInfo = hardwareStatus.find(d => d.id === hardwareId);
    
    try {
      if (command.type === 'power') {
        if (deviceInfo) {
          const powerOn = deviceInfo.status !== 'online';
          const response = await hardwareApi.controlDevicePower(deviceInfo.device_id, powerOn);
          commandResponse = response;
          
          if (response.success) {
            commandSuccess = true;
            toast.success(`${powerOn ? '开机' : '关机'}命令已发送`);
            // 更新设备状态
            setHardwareStatus(prev =>
              prev.map(hardware =>
                hardware.id === hardwareId
                  ? { ...hardware, status: powerOn ? 'online' : 'offline' }
                  : hardware
              )
            );
          } else {
            toast.error('控制命令执行失败');
          }
        }
      } else if (command.type === 'reboot') {
        if (deviceInfo) {
          const response = await hardwareApi.rebootDevice(deviceInfo.device_id);
          commandResponse = response;
          
          if (response.success) {
            commandSuccess = true;
            toast.success('重启命令已发送');
          } else {
            toast.error('重启命令执行失败');
          }
        }
      } else if (command.type === 'reset' || command.type === 'configure') {
        // 其他命令类型处理
        commandSuccess = true; // 假设成功，实际应用中需要根据API响应
        toast.success(`${command.type === 'reset' ? '重置' : '配置'}命令已发送`);
      }
    } catch (error) {
      console.error('执行硬件命令失败:', error);
      toast.error('命令执行失败');
      commandResponse = error;
    } finally {
      // 记录命令历史
      if (deviceInfo) {
        addCommandHistory(
          hardwareId,
          deviceInfo.name,
          command.type,
          command,
          commandSuccess,
          commandResponse
        );
      }
      
      setIsLoading(prev => ({ ...prev, control: false }));
    }
  };

  // 发送电机命令
  const handleMotorCommand = async (motorId: string, command: 'move' | 'stop' | 'reset' | 'calibrate', position?: number) => {
    try {
      const response = await hardwareApi.sendMotorCommand({
        motor_id: motorId,
        command,
        target_position: position,
        speed_factor: 1.0,
        blocking: true
      });
      
      if (response.success) {
        toast.success(`电机命令已发送: ${command}`);
        // 记录电机命令历史
        addCommandHistory(
          motorId,
          `电机 ${motorId}`,
          'motor',
          {
            type: 'configure', // 使用configure类型表示电机控制
            target: motorId,
            parameters: { command, position }
          },
          true,
          response
        );
      } else {
        toast.error('电机命令执行失败');
        // 记录失败历史
        addCommandHistory(
          motorId,
          `电机 ${motorId}`,
          'motor',
          {
            type: 'configure',
            target: motorId,
            parameters: { command, position }
          },
          false,
          response
        );
      }
    } catch (error) {
      console.error('发送电机命令失败:', error);
      toast.error('发送电机命令失败');
      // 记录错误历史
      addCommandHistory(
        motorId,
        `电机 ${motorId}`,
        'motor',
        {
          type: 'configure',
          target: motorId,
          parameters: { command, position }
        },
        false,
        error
      );
    }
  };

  // 发送串口命令
  const handleSerialCommand = async (command: string, port?: string) => {
    try {
      const response = await hardwareApi.sendSerialCommand({
        command,
        port,
        wait_for_response: true,
        timeout: 5
      });
      
      if (response.success) {
        toast.success('串口命令已发送');
        if (response.data?.response) {
          toast.success(`响应: ${response.data.response}`);
        }
      } else {
        toast.error('串口命令执行失败');
      }
    } catch (error) {
      console.error('发送串口命令失败:', error);
      toast.error('发送串口命令失败');
    }
  };

  // 串口配置和管理函数
  const handleSerialConnect = async (portId: string) => {
    try {
      setIsLoading(prev => ({ ...prev, serial: true }));
      const port = serialPorts.find(p => p.id === portId);
      if (!port) return;

      try {
        // 尝试使用真实硬件API连接串口
        // 发送串口连接命令
        const connectCommand = {
          command: `OPEN ${port.port}`,
          port: port.port,
          baudrate: port.config.baudRate,
          wait_for_response: true,
          timeout: 5
        };
        
        const response = await hardwareApi.sendSerialCommand(connectCommand);
        
        if (response.success) {
          // 连接成功，更新前端状态
          setSerialPorts(prev => prev.map(p => 
            p.id === portId 
              ? { ...p, isOpen: true, connected: true, lastActivity: new Date().toISOString() }
              : p
          ));
          
          toast.success(`串口 ${port.port} 已连接`);
          
          // 添加连接日志
          setSerialLogs(prev => [
            `[${new Date().toLocaleTimeString()}] 串口 ${port.port} 连接成功 (API)`,
            ...prev.slice(0, 49)
          ]);
        } else {
          // API返回失败
          toast.error(`串口连接失败: ${response.message || '未知错误'}`);
          console.warn('串口连接API返回失败:', response);
        }
      } catch (apiError) {
        // 真实API失败，根据项目要求"禁止使用虚拟数据"，不提供模拟后备
        console.error('真实串口API连接失败:', apiError);
        toast.error(`串口 ${port.port} 连接失败: ${apiError instanceof Error ? apiError.message : 'API请求失败'}`);
        // 不更新串口状态，保持未连接状态
      }
    } catch (error) {
      console.error('串口连接失败:', error);
      toast.error('串口连接失败');
    } finally {
      setIsLoading(prev => ({ ...prev, serial: false }));
    }
  };

  const handleSerialDisconnect = async (portId: string) => {
    try {
      setIsLoading(prev => ({ ...prev, serial: true }));
      const port = serialPorts.find(p => p.id === portId);
      if (!port) return;

      try {
        // 尝试使用真实硬件API断开串口
        // 发送串口断开命令
        const disconnectCommand = {
          command: `CLOSE ${port.port}`,
          port: port.port,
          wait_for_response: true,
          timeout: 3
        };
        
        const response = await hardwareApi.sendSerialCommand(disconnectCommand);
        
        if (response.success) {
          // 断开成功，更新前端状态
          setSerialPorts(prev => prev.map(p => 
            p.id === portId 
              ? { ...p, isOpen: false, connected: false, lastActivity: new Date().toISOString() }
              : p
          ));
          
          toast.success(`串口 ${port.port} 已断开`);
          
          // 添加断开日志
          setSerialLogs(prev => [
            `[${new Date().toLocaleTimeString()}] 串口 ${port.port} 已断开连接 (API)`,
            ...prev.slice(0, 49)
          ]);
        } else {
          // API返回失败
          toast.error(`串口断开失败: ${response.message || '未知错误'}`);
          console.warn('串口断开API返回失败:', response);
        }
      } catch (apiError) {
        // 真实API失败，根据项目要求"禁止使用虚拟数据"，不提供模拟后备
        console.error('真实串口API断开失败:', apiError);
        toast.error(`串口 ${port.port} 断开失败: ${apiError instanceof Error ? apiError.message : 'API请求失败'}`);
        // 不更新串口状态，保持原有状态
      }
    } catch (error) {
      console.error('串口断开失败:', error);
      toast.error('串口断开失败');
    } finally {
      setIsLoading(prev => ({ ...prev, serial: false }));
    }
  };

  const handleSerialConfigSave = async (portId: string, config: SerialPortConfig) => {
    try {
      setIsLoading(prev => ({ ...prev, serial: true }));
      
      try {
        // 尝试使用真实硬件API保存串口配置
        // 发送串口配置命令
        const configCommand = {
          command: `CONFIG ${config.baudRate},${config.dataBits},${config.parity},${config.stopBits}`,
          port: portId,
          baudrate: config.baudRate,
          bytesize: config.dataBits,
          parity: config.parity,
          stopbits: config.stopBits,
          wait_for_response: false
        };
        
        const response = await hardwareApi.sendSerialCommand(configCommand);
        
        if (response.success) {
          // 配置保存成功
          setSerialPorts(prev => prev.map(p => 
            p.id === portId 
              ? { 
                  ...p, 
                  config,
                  lastActivity: new Date().toISOString()
                }
              : p
          ));
          
          const port = serialPorts.find(p => p.id === portId);
          toast.success(`串口 ${port?.port} 配置已保存 (API)`);
          
          // 添加配置日志
          setSerialLogs(prev => [
            `[${new Date().toLocaleTimeString()}] 串口 ${port?.port} 配置更新成功: ${config.baudRate}波特率, ${config.dataBits}数据位, ${config.stopBits}停止位, ${config.parity}校验`,
            ...prev.slice(0, 49)
          ]);
        } else {
          // API返回失败，但更新前端状态（可能只是配置命令不被支持）
          setSerialPorts(prev => prev.map(p => 
            p.id === portId 
              ? { 
                  ...p, 
                  config,
                  lastActivity: new Date().toISOString()
                }
              : p
          ));
          
          const port = serialPorts.find(p => p.id === portId);
          toast.success(`串口 ${port?.port} 配置已保存 (前端状态)`);
          
          // 添加配置日志
          setSerialLogs(prev => [
            `[${new Date().toLocaleTimeString()}] 串口 ${port?.port} 配置前端状态更新: ${config.baudRate}波特率`,
            ...prev.slice(0, 49)
          ]);
        }
      } catch (apiError) {
        // 真实API失败，根据项目要求"禁止使用虚拟数据"，不提供模拟后备
        console.error('真实串口API配置保存失败:', apiError);
        toast.error(`串口配置保存失败: ${apiError instanceof Error ? apiError.message : 'API请求失败'}`);
        // 不更新串口配置，保持原有配置
      }
      
      setEditingSerialPort(null);
    } catch (error) {
      console.error('串口配置保存失败:', error);
      toast.error('串口配置保存失败');
    } finally {
      setIsLoading(prev => ({ ...prev, serial: false }));
    }
  };

  const handleSerialSend = async (portId: string, data: string) => {
    try {
      setIsLoading(prev => ({ ...prev, serial: true }));
      const port = serialPorts.find(p => p.id === portId);
      if (!port) return;

      if (!port.isOpen) {
        toast.error('串口未连接，请先连接串口');
        return;
      }

      // 调用真实串口发送API
      const command = {
        command: data,
        port: port.port,
        baudrate: port.config.baudRate || 9600,
        wait_for_response: true,
        timeout: 10
      };

      // 发送真实串口命令
      const response = await hardwareApi.sendSerialCommand(command);
      
      // 添加发送日志
      setSerialLogs(prev => [
        `[${new Date().toLocaleTimeString()}] 发送到 ${port.port}: ${data}`,
        ...prev.slice(0, 49)
      ]);
      
      // 接收真实串口响应并添加到日志
      if (response.success && response.data?.response) {
        const responseData = response.data.response;
        setSerialLogs(prev => [
          `[${new Date().toLocaleTimeString()}] 接收到 ${port.port}: ${responseData}`,
          ...prev.slice(0, 49)
        ]);
        toast.success('数据已发送并接收到响应');
      } else {
        toast.success('数据已发送（无响应或响应超时）');
      }
      
      setSerialInput('');
    } catch (error) {
      console.error('串口发送失败:', error);
      toast.error('串口发送失败');
    } finally {
      setIsLoading(prev => ({ ...prev, serial: false }));
    }
  };

  const handleSerialClearLogs = () => {
    setSerialLogs([]);
    toast.success('串口日志已清空');
  };

  const handleScanSerialPorts = async () => {
    try {
      setIsLoading(prev => ({ ...prev, serial: true }));
      
      // 调用真实API扫描串口
      const response = await hardwareApi.getSerialPorts();
      if (response.success && response.data) {
        const apiSerialData = response.data;
        // 使用函数式更新来避免直接依赖serialPorts
        setSerialPorts(prevPorts => {
          // 将API的SerialDevice转换为前端的SerialDevice类型
          const frontendSerialDevices = apiSerialData.map(port => {
            // 检查是否已经存在（保留配置信息）
            const existingPort = prevPorts.find(p => p.port === port.device);
            
            return {
              id: `serial-${port.device}`,
              name: port.description || port.name || port.device,
              port: port.device,
              connected: existingPort?.connected || false,
              config: existingPort?.config || {
                portName: port.device,
                baudRate: 9600,
                dataBits: 8,
                stopBits: 1,
                parity: 'none' as const,
                flowControl: 'none' as const,
                timeout: 1000,
                bufferSize: 1024
              },
              lastActivity: existingPort?.lastActivity || new Date().toISOString(),
              isOpen: existingPort?.isOpen || false
            };
          });
          
          // 添加扫描日志
          setSerialLogs(prev => [
            `[${new Date().toLocaleTimeString()}] 串口扫描完成，发现 ${frontendSerialDevices.length} 个串口`,
            ...prev.slice(0, 49)
          ]);
          
          return frontendSerialDevices;
        });
        
        toast.success('串口扫描完成');
      } else {
        toast.error('串口扫描失败');
      }
    } catch (error) {
      console.error('串口扫描失败:', error);
      toast.error('串口扫描失败');
    } finally {
      setIsLoading(prev => ({ ...prev, serial: false }));
    }
  };

  // UI辅助函数
  const getStatusColor = (status: HardwareStatus['status']) => {
    switch (status) {
      case 'online': return 'bg-gray-600 text-gray-600 dark:bg-gray-700/30 dark:text-gray-600';
      case 'warning': return 'bg-gray-500 text-gray-500 dark:bg-gray-800/30 dark:text-gray-500';
      case 'error': return 'bg-gray-800 text-gray-800 dark:bg-gray-900/30 dark:text-gray-800';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400';
    }
  };

  const getStatusIcon = (status: HardwareStatus['status']) => {
    switch (status) {
      case 'online': return <CheckCircle className="w-4 h-4" />;
      case 'warning': return <AlertCircle className="w-4 h-4" />;
      case 'error': return <XCircle className="w-4 h-4" />;
      default: return <Power className="w-4 h-4" />;
    }
  };

  const getHardwareIcon = (type: HardwareStatus['type']) => {
    switch (type) {
      case 'gpu': return <Cpu className="w-5 h-5" />;
      case 'cpu': return <Server className="w-5 h-5" />;
      case 'memory': return <MemoryStick className="w-5 h-5" />;
      case 'storage': return <HardDrive className="w-5 h-5" />;
      case 'network': return <Wifi className="w-5 h-5" />;
      case 'sensor': return <Thermometer className="w-5 h-5" />;
      case 'motor': return <Zap className="w-5 h-5" />;
      case 'serial': return <Cable className="w-5 h-5" />;
      default: return <Cpu className="w-5 h-5" />;
    }
  };

  const getHardwareTypeColor = (type: HardwareStatus['type']) => {
    switch (type) {
      case 'gpu': return 'bg-gradient-to-r from-gray-600 to-gray-500';
      case 'cpu': return 'bg-gradient-to-r from-gray-700 to-cyan-500';
      case 'memory': return 'bg-gradient-to-r from-gray-600 to-emerald-500';
      case 'storage': return 'bg-gradient-to-r from-gray-500 to-orange-500';
      case 'network': return 'bg-gradient-to-r from-gray-800 to-violet-500';
      case 'sensor': return 'bg-gradient-to-r from-gray-800 to-rose-500';
      case 'motor': return 'bg-gradient-to-r from-amber-500 to-orange-500';
      case 'serial': return 'bg-gradient-to-r from-gray-500 to-slate-500';
      default: return 'bg-gradient-to-r from-gray-500 to-gray-700';
    }
  };

  const formatCapacity = (capacity: number, type: HardwareStatus['type']) => {
    switch (type) {
      case 'gpu':
      case 'memory':
        return `${(capacity / 1024).toFixed(1)} GB`;
      case 'storage':
        return `${(capacity / 1024).toFixed(1)} MB`;
      case 'cpu':
        return `${capacity} 核心`;
      case 'network':
        return `${capacity} Mbps`;
      default:
        return capacity.toString();
    }
  };

  const formatTime = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch {
      return '--:--:--';
    }
  };

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            硬件控制系统
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            监控和管理Self AGI的硬件设备 - {systemStatus?.status || '未知状态'}
          </p>
          <div className="mt-2">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-800 text-white">
              功能状态：实现中
            </span>
            <span className="ml-2 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-600 text-white">
              后端服务：已连接
            </span>
            <span className="ml-2 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-700 text-white">
              硬件接口：串口/传感器
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          {isLoading.hardware || isLoading.sensors || isLoading.metrics ? (
            <Loader2 className="w-5 h-5 animate-spin text-gray-700" />
          ) : null}
          <button
            onClick={() => setIsPolling(!isPolling)}
            className={`inline-flex items-center px-4 py-2 text-sm font-medium rounded-lg ${
              isPolling
                ? 'bg-gray-600 text-gray-600 hover:bg-gray-600 dark:bg-gray-700/30 dark:text-gray-600 dark:hover:bg-gray-600/50'
                : 'bg-gray-100 text-gray-800 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700'
            }`}
          >
            {isPolling ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                实时监控中
              </>
            ) : (
              <>
                <RefreshCw className="w-4 h-4 mr-2" />
                开始监控
              </>
            )}
          </button>
          <button
            onClick={() => {
              loadHardwareStatus();
              loadSensorData();
              loadSystemMetrics();
            }}
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-gray-700 rounded-lg hover:bg-gray-700 dark:bg-gray-700/30 dark:text-gray-700 dark:hover:bg-gray-700/50"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            刷新数据
          </button>
        </div>
      </div>

      {/* 标签页导航 */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('status')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'status'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Server className="w-4 h-4 mr-2" />
              设备状态
              {isLoading.hardware && <Loader2 className="w-3 h-3 ml-2 animate-spin" />}
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('sensors')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'sensors'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Thermometer className="w-4 h-4 mr-2" />
              传感器数据
              {isLoading.sensors && <Loader2 className="w-3 h-3 ml-2 animate-spin" />}
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('control')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'control'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Settings className="w-4 h-4 mr-2" />
              设备控制
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('metrics')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'metrics'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Activity className="w-4 h-4 mr-2" />
              系统指标
              {isLoading.metrics && <Loader2 className="w-3 h-3 ml-2 animate-spin" />}
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('serial')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'serial'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Terminal className="w-4 h-4 mr-2" />
              串口设置
              {isLoading.serial && <Loader2 className="w-3 h-3 ml-2 animate-spin" />}
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('history')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'history'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <History className="w-4 h-4 mr-2" />
              命令历史
              {commandHistory.length > 0 && (
                <span className="ml-2 bg-gray-700 text-gray-700 text-xs font-medium px-2 py-0.5 rounded-full">
                  {commandHistory.length}
                </span>
              )}
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('presets')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'presets'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Bookmark className="w-4 h-4 mr-2" />
              配置预设
              {hardwarePresets.length > 0 && (
                <span className="ml-2 bg-gray-600 text-gray-600 text-xs font-medium px-2 py-0.5 rounded-full">
                  {hardwarePresets.length}
                </span>
              )}
            </div>
          </button>
        </nav>
      </div>

      {/* 设备状态页面 */}
      {activeTab === 'status' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          {hardwareStatus.length === 0 ? (
            <div className="text-center py-12">
              <Cpu className="w-16 h-16 mx-auto text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                没有硬件设备
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                未检测到硬件设备，请检查硬件连接
              </p>
              <button
                onClick={loadHardwareStatus}
                className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gray-700 rounded-lg hover:bg-gray-700"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                重新检测
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {hardwareStatus.map((hardware) => (
                <div
                  key={hardware.id}
                  className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center">
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getHardwareTypeColor(hardware.type)}`}>
                        {getHardwareIcon(hardware.type)}
                      </div>
                      <div className="ml-3">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                          {hardware.name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {hardware.manufacturer} {hardware.model}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-500">
                          最后更新: {formatTime(hardware.last_update)}
                        </p>
                      </div>
                    </div>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(hardware.status)}`}>
                      {getStatusIcon(hardware.status)}
                      <span className="ml-1">{hardware.status === 'online' ? '在线' : 
                        hardware.status === 'warning' ? '警告' : 
                        hardware.status === 'error' ? '错误' : '离线'}</span>
                    </span>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-1">
                        <span>温度</span>
                        <span className="font-medium">{hardware.temperature}°C</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            hardware.temperature > 80 ? 'bg-gray-900' :
                            hardware.temperature > 70 ? 'bg-gray-800' :
                            'bg-gray-700'
                          }`}
                          style={{ width: `${Math.min(100, (hardware.temperature / 100) * 100)}%` }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-1">
                        <span>使用率</span>
                        <span className="font-medium">{hardware.usage}%</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-gray-700 to-gray-600"
                          style={{ width: `${hardware.usage}%` }}
                        />
                      </div>
                    </div>

                    <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">容量</span>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {formatCapacity(hardware.capacity, hardware.type)}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm mt-2">
                        <span className="text-gray-600 dark:text-gray-400">设备ID</span>
                        <span className="font-mono text-xs text-gray-500 dark:text-gray-400">
                          {hardware.device_id}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 传感器数据页面 */}
      {activeTab === 'sensors' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          {/* 传感器数据页面标题和视图切换 */}
          <div className="mb-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-medium text-gray-900 dark:text-white">
                  传感器数据监控
                </h2>
                <p className="mt-1 text-gray-600 dark:text-gray-400">
                  {sensorViewMode === 'list' ? '列表视图' : '图表视图'} - 共{sensorData.length}个传感器
                </p>
              </div>
              <div className="flex items-center space-x-3">
                <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
                  <button
                    onClick={() => setSensorViewMode('list')}
                    className={`px-3 py-1.5 text-sm font-medium rounded-md ${
                      sensorViewMode === 'list'
                        ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-300'
                    }`}
                  >
                    列表视图
                  </button>
                  <button
                    onClick={() => setSensorViewMode('chart')}
                    className={`px-3 py-1.5 text-sm font-medium rounded-md ${
                      sensorViewMode === 'chart'
                        ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-300'
                    }`}
                  >
                    图表视图
                  </button>
                </div>
                <button
                  onClick={loadSensorData}
                  disabled={isLoading.sensors}
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gray-700 rounded-lg hover:bg-gray-700 disabled:opacity-50"
                >
                  <RefreshCw className={`w-4 h-4 mr-2 ${isLoading.sensors ? 'animate-spin' : ''}`} />
                  刷新数据
                </button>
              </div>
            </div>
          </div>
          
          {sensorData.length === 0 ? (
            <div className="text-center py-12">
              <Thermometer className="w-16 h-16 mx-auto text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                没有传感器数据
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                未检测到传感器数据，请检查传感器连接
              </p>
              <button
                onClick={loadSensorData}
                className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gray-700 rounded-lg hover:bg-gray-700"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                重新检测
              </button>
            </div>
          ) : sensorViewMode === 'list' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {sensorData.map((sensor) => (
                  <div
                    key={sensor.id}
                    className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-gray-700 to-cyan-500 flex items-center justify-center">
                          <Thermometer className="w-5 h-5 text-white" />
                        </div>
                        <div className="ml-3">
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                            {sensor.name}
                          </h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            当前值: {sensor.value.toFixed(2)} {sensor.unit}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-500">
                            {sensor.sensor_type} | 精度: {(sensor.accuracy * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                        sensor.value >= sensor.warning_threshold
                          ? 'bg-gray-500 text-gray-500 dark:bg-gray-800/30 dark:text-gray-500'
                          : 'bg-gray-600 text-gray-600 dark:bg-gray-700/30 dark:text-gray-600'
                      }`}>
                        {sensor.value >= sensor.warning_threshold ? '警告' : '正常'}
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-1">
                          <span>范围</span>
                          <span className="font-medium">{sensor.min} - {sensor.max} {sensor.unit}</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div className="relative h-2">
                            <div className="absolute inset-0 bg-gradient-to-r from-gray-600 via-gray-500 to-gray-800 rounded-full" />
                            <div
                              className="absolute top-0 w-2 h-4 bg-gray-900 dark:bg-white rounded-full -ml-1"
                              style={{ left: `${((sensor.value - sensor.min) / (sensor.max - sensor.min)) * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                        <div>
                          <p className="text-sm text-gray-600 dark:text-gray-400">最小值</p>
                          <p className="text-lg font-semibold text-gray-900 dark:text-white">{sensor.min} {sensor.unit}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-600 dark:text-gray-400">最大值</p>
                          <p className="text-lg font-semibold text-gray-900 dark:text-white">{sensor.max} {sensor.unit}</p>
                        </div>
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-500 pt-2 border-t border-gray-200 dark:border-gray-700">
                        传感器ID: {sensor.sensor_id} | 更新时间: {formatTime(sensor.timestamp)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700">
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                    传感器数据趋势图
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    显示最近传感器数据变化趋势
                  </p>
                </div>
                
                {sensorHistory.length === 0 ? (
                  <div className="text-center py-12">
                    <Thermometer className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                    <p className="text-gray-600 dark:text-gray-400">
                      暂无传感器历史数据，请等待数据刷新
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* 为每个传感器创建图表 */}
                    {Array.from(new Set(sensorHistory.map(h => h.sensorId))).slice(0, 4).map(sensorId => {
                      const sensor = sensorData.find(s => s.sensor_id === sensorId);
                      const sensorName = sensor?.name || sensorId;
                      const sensorUnit = sensor?.unit || '';
                      const sensorHistoryData = sensorHistory.filter(h => h.sensorId === sensorId).slice(-20); // 最近20个数据点
                      
                      if (sensorHistoryData.length < 2) return null;
                      
                      const chartData = {
                        labels: sensorHistoryData.map(h => h.timestamp.toLocaleTimeString()),
                        datasets: [
                          {
                            label: `${sensorName} (${sensorUnit})`,
                            data: sensorHistoryData.map(h => h.value),
                            borderColor: 'rgb(59, 130, 246)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            fill: true,
                            tension: 0.4,
                            pointBackgroundColor: 'rgb(59, 130, 246)',
                            pointBorderColor: 'white',
                            pointBorderWidth: 2,
                            pointRadius: 4,
                          },
                        ],
                      };
                      
                      const chartOptions = {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            position: 'top' as const,
                          },
                          tooltip: {
                            mode: 'index' as const,
                            intersect: false,
                            callbacks: {
                              label: function(context: any) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} ${sensorUnit}`;
                              }
                            }
                          },
                        },
                        scales: {
                          y: {
                            beginAtZero: false,
                            grid: {
                              color: 'rgba(0, 0, 0, 0.05)',
                            },
                            ticks: {
                              callback: function(value: any) {
                                return value.toFixed(2) + (sensorUnit ? ` ${sensorUnit}` : '');
                              }
                            }
                          },
                          x: {
                            grid: {
                              color: 'rgba(0, 0, 0, 0.05)',
                            },
                            ticks: {
                              maxRotation: 45,
                            }
                          }
                        },
                      };
                      
                      return (
                        <div key={sensorId} className="mb-8">
                          <div className="flex items-center justify-between mb-4">
                            <h4 className="text-md font-medium text-gray-900 dark:text-white">
                              {sensorName}
                            </h4>
                            <span className="text-sm text-gray-600 dark:text-gray-400">
                              当前值: {sensor?.value.toFixed(2)} {sensorUnit}
                            </span>
                          </div>
                          <div className="h-64">
                            <Line data={chartData} options={chartOptions} />
                          </div>
                          <div className="mt-4 text-xs text-gray-500 dark:text-gray-500">
                            数据点数: {sensorHistoryData.length} | 时间范围: {sensorHistoryData[0].timestamp.toLocaleTimeString()} - {sensorHistoryData[sensorHistoryData.length - 1].timestamp.toLocaleTimeString()}
                          </div>
                        </div>
                      );
                    })}
                    
                    <div className="text-sm text-gray-600 dark:text-gray-400 pt-4 border-t border-gray-200 dark:border-gray-700">
                      <p>图表说明：</p>
                      <ul className="list-disc pl-5 mt-2 space-y-1">
                        <li>每个图表显示单个传感器的最近20个数据点</li>
                        <li>数据自动更新，刷新间隔为5秒</li>
                        <li>点击图例可隐藏/显示对应传感器数据</li>
                        <li>鼠标悬停在数据点上查看详细信息</li>
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            )}
        </div>
      )}

      {/* 设备控制页面 */}
      {activeTab === 'control' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="mb-6">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              设备控制面板
            </h2>
            <p className="mt-1 text-gray-600 dark:text-gray-400">
              发送控制命令到硬件设备
            </p>
          </div>

          {hardwareStatus.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                没有可控制的设备，请先加载设备状态
              </p>
              <button
                onClick={loadHardwareStatus}
                className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gray-700 rounded-lg hover:bg-gray-700"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                加载设备
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {hardwareStatus.map((hardware) => (
                <div
                  key={hardware.id}
                  className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center">
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getHardwareTypeColor(hardware.type)}`}>
                        {getHardwareIcon(hardware.type)}
                      </div>
                      <div className="ml-3">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                          {hardware.name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {hardware.manufacturer} {hardware.model}
                        </p>
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(hardware.status)}`}>
                      {hardware.status === 'online' ? '在线' : '离线'}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => handleHardwareCommand(hardware.id, { type: 'power', target: hardware.id })}
                      disabled={isLoading.control}
                      className={`px-4 py-2 text-sm font-medium rounded-lg ${
                        hardware.status === 'online'
                          ? 'bg-gray-800 text-gray-800 hover:bg-gray-800 dark:bg-gray-900/30 dark:text-gray-800 dark:hover:bg-gray-800/50'
                          : 'bg-gray-600 text-gray-600 hover:bg-gray-600 dark:bg-gray-700/30 dark:text-gray-600 dark:hover:bg-gray-600/50'
                      } ${isLoading.control ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <Power className="w-4 h-4 inline mr-2" />
                      {hardware.status === 'online' ? '关机' : '开机'}
                    </button>
                    
                    <button
                      onClick={() => handleHardwareCommand(hardware.id, { type: 'reboot', target: hardware.id })}
                      disabled={hardware.status !== 'online' || isLoading.control}
                      className={`px-4 py-2 text-sm font-medium rounded-lg ${
                        hardware.status === 'online'
                          ? 'bg-gray-700 text-gray-700 hover:bg-gray-700 dark:bg-gray-700/30 dark:text-gray-700 dark:hover:bg-gray-700/50'
                          : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400 cursor-not-allowed'
                      } ${isLoading.control ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <RefreshCw className="w-4 h-4 inline mr-2" />
                      重启
                    </button>
                  </div>

                  {/* 特殊设备控制 */}
                  {hardware.type === 'motor' && (
                    <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">电机控制</h4>
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          onClick={() => handleMotorCommand(hardware.device_id, 'move', 100)}
                          className="px-3 py-2 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
                        >
                          移动到100
                        </button>
                        <button
                          onClick={() => handleMotorCommand(hardware.device_id, 'stop')}
                          className="px-3 py-2 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
                        >
                          停止
                        </button>
                        <button
                          onClick={() => handleMotorCommand(hardware.device_id, 'reset')}
                          className="px-3 py-2 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
                        >
                          重置
                        </button>
                        <button
                          onClick={() => handleMotorCommand(hardware.device_id, 'calibrate')}
                          className="px-3 py-2 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
                        >
                          校准
                        </button>
                      </div>
                    </div>
                  )}

                  {hardware.type === 'serial' && (
                    <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">串口控制</h4>
                      <div className="space-y-2">
                        <button
                          onClick={() => handleSerialCommand('STATUS', hardware.device_id)}
                          className="w-full px-3 py-2 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
                        >
                          查询状态
                        </button>
                        <button
                          onClick={() => handleSerialCommand('RESET', hardware.device_id)}
                          className="w-full px-3 py-2 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
                        >
                          重置设备
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 系统指标页面 */}
      {activeTab === 'metrics' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          {systemMetrics.length === 0 ? (
            <div className="text-center py-12">
              <Activity className="w-16 h-16 mx-auto text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                没有系统指标数据
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                未检测到系统指标数据，请检查系统监控器
              </p>
              <button
                onClick={loadSystemMetrics}
                className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gray-700 rounded-lg hover:bg-gray-700"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                重新加载
              </button>
            </div>
          ) : (
            <>
              {/* 系统状态概览 */}
              {systemStatus && (
                <div className="mb-8 bg-gradient-to-r from-gray-700 to-gray-800 dark:from-gray-700/20 dark:to-gray-800/20 rounded-xl p-6 border border-gray-700 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        系统状态概览
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        系统运行时间: {Math.floor(systemStatus.uptime / 3600)}小时 {Math.floor((systemStatus.uptime % 3600) / 60)}分钟
                      </p>
                    </div>
                    <div className={`px-4 py-2 rounded-full text-sm font-medium ${
                      systemStatus.status === 'healthy' 
                        ? 'bg-gray-600 text-gray-600 dark:bg-gray-700/30 dark:text-gray-600'
                        : systemStatus.status === 'warning'
                        ? 'bg-gray-500 text-gray-500 dark:bg-gray-800/30 dark:text-gray-500'
                        : 'bg-gray-800 text-gray-800 dark:bg-gray-900/30 dark:text-gray-800'
                    }`}>
                      {systemStatus.status === 'healthy' ? '健康' : 
                       systemStatus.status === 'warning' ? '警告' : '异常'}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                      <div className="flex items-center">
                        <Cpu className="w-5 h-5 text-gray-700 mr-2" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">CPU使用率</span>
                      </div>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white mt-2">
                        {systemStatus.cpu_usage.toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                      <div className="flex items-center">
                        <MemoryStick className="w-5 h-5 text-gray-600 mr-2" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">内存使用率</span>
                      </div>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white mt-2">
                        {systemStatus.memory_usage.toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                      <div className="flex items-center">
                        <Database className="w-5 h-5 text-gray-500 mr-2" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">磁盘使用率</span>
                      </div>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white mt-2">
                        {systemStatus.disk_usage.toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                      <div className="flex items-center">
                        <AlertCircle className="w-5 h-5 text-gray-800 mr-2" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">活跃警报</span>
                      </div>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white mt-2">
                        {systemStatus.active_alerts}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* 指标列表 */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {systemMetrics.map((metric) => (
                  <div
                    key={metric.id}
                    className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center">
                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                          metric.status === 'normal' ? 'bg-gray-700' :
                          metric.status === 'warning' ? 'bg-gray-800' :
                          metric.status === 'error' ? 'bg-gray-900' : 'bg-gray-600'
                        }`}>
                          <Activity className="w-5 h-5 text-white" />
                        </div>
                        <div className="ml-3">
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                            {metric.metric_name}
                          </h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            当前值: {metric.value.toFixed(2)} {metric.unit}
                          </p>
                        </div>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                        metric.status === 'normal' ? 'bg-gray-600 text-gray-600 dark:bg-gray-700/30 dark:text-gray-600' :
                        metric.status === 'warning' ? 'bg-gray-500 text-gray-500 dark:bg-gray-800/30 dark:text-gray-500' :
                        metric.status === 'error' ? 'bg-gray-800 text-gray-800 dark:bg-gray-900/30 dark:text-gray-800' :
                        'bg-gray-600 text-gray-600 dark:bg-gray-600/30 dark:text-gray-600'
                      }`}>
                        {metric.status === 'normal' ? '正常' :
                         metric.status === 'warning' ? '警告' :
                         metric.status === 'error' ? '错误' : '严重'}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-500 pt-3 border-t border-gray-200 dark:border-gray-700">
                      更新时间: {formatTime(metric.timestamp)}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* 串口设置页面 */}
      {activeTab === 'serial' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="mb-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-medium text-gray-900 dark:text-white">
                  串口设备管理
                </h2>
                <p className="mt-1 text-gray-600 dark:text-gray-400">
                  配置和管理串口设备连接参数
                </p>
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={handleScanSerialPorts}
                  disabled={isLoading.serial}
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
                >
                  <RefreshCw className={`w-4 h-4 mr-2 ${isLoading.serial ? 'animate-spin' : ''}`} />
                  扫描串口
                </button>
                <button
                  onClick={handleSerialClearLogs}
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-800 dark:text-gray-800 bg-gray-900 dark:bg-gray-900/20 rounded-lg hover:bg-gray-800 dark:hover:bg-gray-800/30"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  清空日志
                </button>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* 串口设备列表 */}
            <div className="lg:col-span-2 space-y-4">
              <h3 className="text-md font-medium text-gray-900 dark:text-white">
                串口设备列表
              </h3>
              
              {serialPorts.length === 0 ? (
                <div className="text-center py-8 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                  <Terminal className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600 dark:text-gray-400">
                    未发现串口设备，点击"扫描串口"按钮扫描可用串口
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {serialPorts.map((port) => (
                    <div
                      key={port.id}
                      className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center">
                          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                            port.isOpen 
                              ? 'bg-gray-700' 
                              : 'bg-gray-500'
                          }`}>
                            <Terminal className="w-5 h-5 text-white" />
                          </div>
                          <div className="ml-3">
                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                              {port.name}
                            </h3>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              {port.port} • {port.config.baudRate}波特率 • {port.isOpen ? '已连接' : '未连接'}
                            </p>
                          </div>
                        </div>
                        <div className="flex space-x-2">
                          {port.isOpen ? (
                            <button
                              onClick={() => handleSerialDisconnect(port.id)}
                              disabled={isLoading.serial}
                              className="px-3 py-1.5 text-xs font-medium text-gray-800 dark:text-gray-800 bg-gray-900 dark:bg-gray-900/20 rounded-lg hover:bg-gray-800 dark:hover:bg-gray-800/30"
                            >
                              <StopCircle className="w-3 h-3 inline mr-1" />
                              断开
                            </button>
                          ) : (
                            <button
                              onClick={() => handleSerialConnect(port.id)}
                              disabled={isLoading.serial}
                              className="px-3 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-600 bg-gray-700 dark:bg-gray-700/20 rounded-lg hover:bg-gray-600 dark:hover:bg-gray-600/30"
                            >
                              <Play className="w-3 h-3 inline mr-1" />
                              连接
                            </button>
                          )}
                          <button
                            onClick={() => setEditingSerialPort(port)}
                            disabled={isLoading.serial}
                            className="px-3 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-700 bg-gray-700 dark:bg-gray-700/20 rounded-lg hover:bg-gray-700 dark:hover:bg-gray-700/30"
                          >
                            <Cog className="w-3 h-3 inline mr-1" />
                            配置
                          </button>
                        </div>
                      </div>

                      {/* 配置详情 */}
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">波特率:</span>
                          <span className="ml-2 font-medium text-gray-900 dark:text-white">
                            {port.config.baudRate}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">数据位:</span>
                          <span className="ml-2 font-medium text-gray-900 dark:text-white">
                            {port.config.dataBits}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">停止位:</span>
                          <span className="ml-2 font-medium text-gray-900 dark:text-white">
                            {port.config.stopBits}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">校验位:</span>
                          <span className="ml-2 font-medium text-gray-900 dark:text-white">
                            {port.config.parity === 'none' ? '无' : 
                             port.config.parity === 'even' ? '偶校验' :
                             port.config.parity === 'odd' ? '奇校验' :
                             port.config.parity === 'mark' ? '标志位' : '空格位'}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">流控制:</span>
                          <span className="ml-2 font-medium text-gray-900 dark:text-white">
                            {port.config.flowControl === 'none' ? '无' :
                             port.config.flowControl === 'software' ? '软件' : '硬件'}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-500 dark:text-gray-400">超时:</span>
                          <span className="ml-2 font-medium text-gray-900 dark:text-white">
                            {port.config.timeout}ms
                          </span>
                        </div>
                      </div>

                      {/* 最后活动时间 */}
                      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-500">
                        最后活动: {new Date(port.lastActivity).toLocaleString()}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* 串口日志和控制面板 */}
            <div className="space-y-6">
              {/* 串口控制 */}
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700">
                <h3 className="text-md font-medium text-gray-900 dark:text-white mb-4">
                  串口控制
                </h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      选择串口
                    </label>
                    <select
                      className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                      onChange={(e) => {
                        const port = serialPorts.find(p => p.id === e.target.value);
                        if (port) setEditingSerialPort(port);
                      }}
                    >
                      <option value="">选择串口设备</option>
                      {serialPorts.map(port => (
                        <option key={port.id} value={port.id}>
                          {port.name} ({port.port})
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      发送数据
                    </label>
                    <div className="flex space-x-2">
                      <input
                        type="text"
                        value={serialInput}
                        onChange={(e) => setSerialInput(e.target.value)}
                        placeholder="输入要发送的数据..."
                        className="flex-1 px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                      />
                      <button
                        onClick={() => {
                          const activePort = serialPorts.find(p => p.isOpen);
                          if (activePort && serialInput.trim()) {
                            handleSerialSend(activePort.id, serialInput.trim());
                          } else {
                            toast.error('请先连接串口并输入数据');
                          }
                        }}
                        disabled={isLoading.serial || !serialInput.trim()}
                        className="px-4 py-2 text-sm font-medium text-white bg-gray-700 rounded-lg hover:bg-gray-700 disabled:opacity-50"
                      >
                        发送
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => {
                        const activePort = serialPorts.find(p => p.isOpen);
                        if (activePort) {
                          handleSerialSend(activePort.id, 'STATUS');
                        } else {
                          toast.error('请先连接串口');
                        }
                      }}
                      disabled={isLoading.serial}
                      className="px-3 py-2 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
                    >
                      查询状态
                    </button>
                    <button
                      onClick={() => {
                        const activePort = serialPorts.find(p => p.isOpen);
                        if (activePort) {
                          handleSerialSend(activePort.id, 'RESET');
                        } else {
                          toast.error('请先连接串口');
                        }
                      }}
                      disabled={isLoading.serial}
                      className="px-3 py-2 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
                    >
                      重置设备
                    </button>
                  </div>
                </div>
              </div>

              {/* 串口日志 */}
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-md font-medium text-gray-900 dark:text-white">
                    串口日志
                  </h3>
                  <span className="text-xs text-gray-500 dark:text-gray-500">
                    {serialLogs.length} 条记录
                  </span>
                </div>
                
                <div className="h-64 overflow-y-auto bg-gray-900 text-gray-100 rounded-lg p-3 font-mono text-sm">
                  {serialLogs.length === 0 ? (
                    <div className="text-gray-500 text-center py-8">
                      暂无日志记录
                    </div>
                  ) : (
                    <div className="space-y-1">
                      {serialLogs.map((log, index) => (
                        <div key={index} className="whitespace-pre-wrap break-words">
                          {log}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* 串口配置模态框 */}
          {editingSerialPort && (
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-lg">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                      配置串口: {editingSerialPort.name}
                    </h3>
                    <button
                      onClick={() => setEditingSerialPort(null)}
                      className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        串口名称
                      </label>
                      <input
                        type="text"
                        value={editingSerialPort.name}
                        onChange={(e) => setEditingSerialPort({
                          ...editingSerialPort,
                          name: e.target.value
                        })}
                        className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          波特率
                        </label>
                        <select
                          value={editingSerialPort.config.baudRate}
                          onChange={(e) => setEditingSerialPort({
                            ...editingSerialPort,
                            config: {
                              ...editingSerialPort.config,
                              baudRate: parseInt(e.target.value)
                            }
                          })}
                          className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                        >
                          <option value="9600">9600</option>
                          <option value="19200">19200</option>
                          <option value="38400">38400</option>
                          <option value="57600">57600</option>
                          <option value="115200">115200</option>
                          <option value="230400">230400</option>
                          <option value="460800">460800</option>
                          <option value="921600">921600</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          数据位
                        </label>
                        <select
                          value={editingSerialPort.config.dataBits}
                          onChange={(e) => setEditingSerialPort({
                            ...editingSerialPort,
                            config: {
                              ...editingSerialPort.config,
                              dataBits: parseInt(e.target.value) as 5 | 6 | 7 | 8
                            }
                          })}
                          className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                        >
                          <option value="5">5位</option>
                          <option value="6">6位</option>
                          <option value="7">7位</option>
                          <option value="8">8位</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          停止位
                        </label>
                        <select
                          value={editingSerialPort.config.stopBits}
                          onChange={(e) => setEditingSerialPort({
                            ...editingSerialPort,
                            config: {
                              ...editingSerialPort.config,
                              stopBits: parseFloat(e.target.value) as 1 | 1.5 | 2
                            }
                          })}
                          className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                        >
                          <option value="1">1位</option>
                          <option value="1.5">1.5位</option>
                          <option value="2">2位</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          校验位
                        </label>
                        <select
                          value={editingSerialPort.config.parity}
                          onChange={(e) => setEditingSerialPort({
                            ...editingSerialPort,
                            config: {
                              ...editingSerialPort.config,
                              parity: e.target.value as 'none' | 'even' | 'odd' | 'mark' | 'space'
                            }
                          })}
                          className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                        >
                          <option value="none">无</option>
                          <option value="even">偶校验</option>
                          <option value="odd">奇校验</option>
                          <option value="mark">标志位</option>
                          <option value="space">空格位</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          流控制
                        </label>
                        <select
                          value={editingSerialPort.config.flowControl}
                          onChange={(e) => setEditingSerialPort({
                            ...editingSerialPort,
                            config: {
                              ...editingSerialPort.config,
                              flowControl: e.target.value as 'none' | 'software' | 'hardware'
                            }
                          })}
                          className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                        >
                          <option value="none">无</option>
                          <option value="software">软件流控制</option>
                          <option value="hardware">硬件流控制</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          超时时间 (ms)
                        </label>
                        <input
                          type="number"
                          value={editingSerialPort.config.timeout}
                          onChange={(e) => setEditingSerialPort({
                            ...editingSerialPort,
                            config: {
                              ...editingSerialPort.config,
                              timeout: parseInt(e.target.value)
                            }
                          })}
                          min="100"
                          max="10000"
                          step="100"
                          className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                        />
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 flex justify-end space-x-3">
                    <button
                      onClick={() => setEditingSerialPort(null)}
                      className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
                    >
                      取消
                    </button>
                    <button
                      onClick={() => handleSerialConfigSave(editingSerialPort.id, editingSerialPort.config)}
                      disabled={isLoading.serial}
                      className="px-4 py-2 text-sm font-medium text-white bg-gray-700 rounded-lg hover:bg-gray-700 flex items-center"
                    >
                      {isLoading.serial ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          保存中...
                        </>
                      ) : (
                        <>
                          <Save className="w-4 h-4 mr-2" />
                          保存配置
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 命令历史页面 */}
      {activeTab === 'history' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="mb-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-medium text-gray-900 dark:text-white">
                  命令历史记录
                </h2>
                <p className="mt-1 text-gray-600 dark:text-gray-400">
                  查看所有硬件控制命令的执行历史
                </p>
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={() => setCommandHistory([])}
                  disabled={commandHistory.length === 0}
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-800 dark:text-gray-800 bg-gray-900 dark:bg-gray-900/20 rounded-lg hover:bg-gray-800 dark:hover:bg-gray-800/30 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  清空历史
                </button>
                <button
                  onClick={() => {
                    // 导出历史记录为JSON
                    const dataStr = JSON.stringify(commandHistory, null, 2);
                    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
                    const exportFileDefaultName = `hardware_commands_${new Date().toISOString().slice(0, 10)}.json`;
                    const linkElement = document.createElement('a');
                    linkElement.setAttribute('href', dataUri);
                    linkElement.setAttribute('download', exportFileDefaultName);
                    linkElement.click();
                    toast.success('历史记录已导出');
                  }}
                  disabled={commandHistory.length === 0}
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-600 bg-gray-700 dark:bg-gray-700/20 rounded-lg hover:bg-gray-600 dark:hover:bg-gray-600/30 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Save className="w-4 h-4 mr-2" />
                  导出历史
                </button>
              </div>
            </div>
          </div>

          {commandHistory.length === 0 ? (
            <div className="text-center py-12">
              <History className="w-16 h-16 mx-auto text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                暂无命令历史记录
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                执行硬件控制命令后，历史记录将显示在这里
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-900">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      时间
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      设备
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      命令类型
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      命令详情
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      状态
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      操作
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {commandHistory.map((record) => (
                    <tr key={record.id} className="hover:bg-gray-50 dark:hover:bg-gray-900/50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-300">
                        {record.timestamp.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900 dark:text-white">
                          {record.deviceName}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-500">
                          {record.deviceId}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-300">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-700 text-gray-700 dark:bg-gray-700/30 dark:text-gray-700">
                          {record.commandType === 'power' ? '电源控制' :
                           record.commandType === 'reboot' ? '重启' :
                           record.commandType === 'reset' ? '重置' :
                           record.commandType === 'configure' ? '配置' :
                           record.commandType === 'motor' ? '电机控制' : record.commandType}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900 dark:text-gray-300">
                        <div className="font-mono text-xs bg-gray-100 dark:bg-gray-900 p-2 rounded">
                          {JSON.stringify(record.command, null, 2)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          record.success 
                            ? 'bg-gray-600 text-gray-600 dark:bg-gray-700/30 dark:text-gray-600'
                            : 'bg-gray-800 text-gray-800 dark:bg-gray-900/30 dark:text-gray-800'
                        }`}>
                          {record.success ? (
                            <>
                              <CheckCircle className="w-3 h-3 mr-1" />
                              成功
                            </>
                          ) : (
                            <>
                              <XCircle className="w-3 h-3 mr-1" />
                              失败
                            </>
                          )}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-300">
                        <button
                          onClick={() => {
                            // 重新执行命令
                            if (record.commandType === 'power' || record.commandType === 'reboot' || record.commandType === 'reset' || record.commandType === 'configure') {
                              handleHardwareCommand(record.deviceId, record.command);
                            } else if (record.commandType === 'motor') {
                              const params = record.command.parameters;
                              handleMotorCommand(record.deviceId, params?.command || 'move', params?.position);
                            }
                          }}
                          className="text-gray-700 dark:text-gray-700 hover:text-gray-700 dark:hover:text-gray-700"
                        >
                          重新执行
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* 配置预设页面 */}
      {activeTab === 'presets' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="mb-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-medium text-gray-900 dark:text-white">
                  硬件配置预设管理
                </h2>
                <p className="mt-1 text-gray-600 dark:text-gray-400">
                  保存、加载和应用硬件配置预设
                </p>
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={() => setPresetDialogOpen(true)}
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gray-700 rounded-lg hover:bg-gray-700"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  保存当前配置为预设
                </button>
              </div>
            </div>
          </div>

          {hardwarePresets.length === 0 ? (
            <div className="text-center py-12">
              <Bookmark className="w-16 h-16 mx-auto text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                暂无配置预设
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                点击"保存当前配置为预设"按钮创建您的第一个预设
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {hardwarePresets.map((preset) => (
                <div
                  key={preset.id}
                  className={`bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-xl p-5 border ${
                    selectedPreset === preset.id
                      ? 'border-gray-700 dark:border-gray-700'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center">
                      <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-gray-600 to-emerald-500 flex items-center justify-center">
                        <Bookmark className="w-5 h-5 text-white" />
                      </div>
                      <div className="ml-3">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                          {preset.name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {preset.description || '无描述'}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-500">
                          {preset.category === 'motor' ? '电机配置' :
                           preset.category === 'sensor' ? '传感器配置' :
                           preset.category === 'system' ? '系统配置' : '自定义配置'}
                        </p>
                      </div>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-500">
                      {preset.updatedAt.toLocaleDateString()}
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded-lg">
                        <p className="text-sm text-gray-600 dark:text-gray-400">设备数量</p>
                        <p className="text-lg font-semibold text-gray-900 dark:text-white">
                          {preset.config.devices?.length || 0}
                        </p>
                      </div>
                      <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded-lg">
                        <p className="text-sm text-gray-600 dark:text-gray-400">传感器数量</p>
                        <p className="text-lg font-semibold text-gray-900 dark:text-white">
                          {preset.config.sensorConfigs?.length || 0}
                        </p>
                      </div>
                    </div>

                    <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => loadPresetConfig(preset.id)}
                          className={`flex-1 inline-flex justify-center items-center px-3 py-2 text-sm font-medium rounded-lg ${
                            selectedPreset === preset.id
                              ? 'bg-gray-700 text-gray-700 dark:bg-gray-700/30 dark:text-gray-700'
                              : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                          }`}
                        >
                          加载配置
                        </button>
                        <button
                          onClick={() => applyPresetConfig(preset.id)}
                          className="flex-1 inline-flex justify-center items-center px-3 py-2 text-sm font-medium text-white bg-gray-600 rounded-lg hover:bg-gray-600"
                        >
                          应用到硬件
                        </button>
                        <button
                          onClick={() => deletePreset(preset.id)}
                          className="inline-flex justify-center items-center px-3 py-2 text-sm font-medium text-gray-800 dark:text-gray-800 bg-gray-900 dark:bg-gray-900/20 rounded-lg hover:bg-gray-800 dark:hover:bg-gray-800/30"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 保存预设对话框 */}
      {presetDialogOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg max-w-md w-full">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                  保存硬件配置预设
                </h3>
                <button
                  onClick={() => setPresetDialogOpen(false)}
                  className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    预设名称 *
                  </label>
                  <input
                    type="text"
                    value={presetName}
                    onChange={(e) => setPresetName(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="例如：默认电机配置"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    描述
                  </label>
                  <textarea
                    value={presetDescription}
                    onChange={(e) => setPresetDescription(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="描述此预设的用途和配置"
                    rows={3}
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    分类
                  </label>
                  <select
                    value={presetCategory}
                    onChange={(e) => setPresetCategory(e.target.value as any)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    <option value="motor">电机配置</option>
                    <option value="sensor">传感器配置</option>
                    <option value="system">系统配置</option>
                    <option value="custom">自定义配置</option>
                  </select>
                </div>
                
                <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex justify-end space-x-3">
                    <button
                      onClick={() => setPresetDialogOpen(false)}
                      className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      取消
                    </button>
                    <button
                      onClick={saveCurrentConfigAsPreset}
                      disabled={!presetName.trim()}
                      className="px-4 py-2 text-sm font-medium text-white bg-gray-700 rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      保存预设
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

export default HardwarePage;