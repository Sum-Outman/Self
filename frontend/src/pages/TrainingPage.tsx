import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  Play,
  Pause,
  StopCircle,
  Settings,
  Upload,
  Brain,
  Cpu,
  Zap,
  BarChart3,
  Clock,
  CheckCircle,
  XCircle,
  RefreshCw,
  Plus,
  Trash2,
  HardDrive,
  Loader2,
  X,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { ResponsiveContainer } from '../components/UI';
import { trainingApi, TrainingJob, TrainingConfig, TrainingRequest, TrainingStats } from '../services/api/training';

// 前端UI训练任务接口（兼容现有UI）
interface UiTrainingJob {
  id: string;
  name: string;
  modelType: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  datasetSize: number;
  epochs: number;
  currentEpoch: number;
  startTime: Date;
  estimatedTime: string;
  gpuUsage: number;
  memoryUsage: number;
  config: Record<string, any>;
}

// 系统资源接口
interface SystemResources {
  gpuAvailable: boolean;
  gpuMemory: number;
  gpuUtilization: number;
  systemMemory: number;
  memoryUsed: number;
  diskSpace: number;
  diskUsed: number;
}

const TrainingPage: React.FC = () => {
  const { isAuthenticated, isAdmin, isLoading: authLoading } = useAuth();
  const [jobs, setJobs] = useState<UiTrainingJob[]>([]);
  const [trainingStats, setTrainingStats] = useState<TrainingStats | null>(null);
  const [systemResources, setSystemResources] = useState<SystemResources>({
    gpuAvailable: false,
    gpuMemory: 0,
    gpuUtilization: 0,
    systemMemory: 0,
    memoryUsed: 0,
    diskSpace: 0,
    diskUsed: 0,
  });
  
  const [newConfig, setNewConfig] = useState<TrainingConfig>({
    model_type: 'transformer',
    dataset_path: '',
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
    use_gpu: true,
    save_checkpoints: true,
    early_stopping: true,
    validation_split: 0.2,
    model_name: '',
    description: '',
  });
  
  const [activeTab, setActiveTab] = useState<'jobs' | 'new' | 'config' | 'monitor' | 'laplacian'>('jobs');
  const [isCreatingJob, setIsCreatingJob] = useState(false);
  const [isLoading, setIsLoading] = useState({
    jobs: true,
    stats: true,
    gpu: true,
  });
  const [availableDatasets, setAvailableDatasets] = useState<{name: string, path: string, size: number, created_at: string}[]>([]);
  const [_availableModelTypes, _setAvailableModelTypes] = useState<{type: string, description: string, parameters: number}[]>([]);
  const [datasetUploadFile, setDatasetUploadFile] = useState<File | null>(null);
  const [datasetUploadName, setDatasetUploadName] = useState('');
  const [selectedTraining, setSelectedTraining] = useState<UiTrainingJob | null>(null);
  const [showTrainingDetailModal, setShowTrainingDetailModal] = useState(false);
  
  // 训练监控状态
  const [monitoringData, setMonitoringData] = useState<{
    metrics: Array<{timestamp: string, loss: number, accuracy: number, learning_rate: number}>;
    resourceUsage: Array<{timestamp: string, cpu: number, memory: number, gpu: number}>;
    trainingProgress: Array<{epoch: number, loss: number, accuracy: number}>;
    activeAlerts: Array<{id: string, type: string, message: string, severity: 'info' | 'warning' | 'error'}>;
  }>({
    metrics: [],
    resourceUsage: [],
    trainingProgress: [],
    activeAlerts: [],
  });
  const [monitoringInterval, setMonitoringInterval] = useState<number>(5000); // 默认5秒
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '12h' | '24h'>('1h');
  
  // 拉普拉斯增强系统状态
  const [laplacianStatus, setLaplacianStatus] = useState<{
    available: boolean;
    initialized: boolean;
    enhancement_mode: string;
    enabled_components: string[];
    regularization_lambda: number;
    adaptive_lambda: boolean;
  } | null>(null);
  const [laplacianConfig, setLaplacianConfig] = useState<{
    enhancement_mode: string;
    enabled_components: string[];
    regularization_lambda: number;
    adaptive_lambda: boolean;
  }>({
    enhancement_mode: 'graph_laplacian',
    enabled_components: ['regularization', 'feature_extraction'],
    regularization_lambda: 0.01,
    adaptive_lambda: true,
  });
  const [isLoadingLaplacian, setIsLoadingLaplacian] = useState(false);
  
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // 转换API响应到UI接口
  const convertApiJobToUi = (apiJob: TrainingJob): UiTrainingJob => {
    const startTime = new Date(apiJob.start_time);
    const now = new Date();
    const duration = now.getTime() - startTime.getTime();
    const progressPerHour = apiJob.progress > 0 ? apiJob.progress / (duration / 3600000) : 0;
    const remainingHours = progressPerHour > 0 ? (100 - apiJob.progress) / progressPerHour : 0;
    
    let estimatedTime = '计算中...';
    if (apiJob.status === 'completed') {
      estimatedTime = '已完成';
    } else if (remainingHours > 0) {
      if (remainingHours < 1) {
        estimatedTime = `${Math.round(remainingHours * 60)}分钟`;
      } else if (remainingHours < 24) {
        estimatedTime = `${Math.round(remainingHours)}小时`;
      } else {
        estimatedTime = `${Math.round(remainingHours / 24)}天`;
      }
    }
    
    return {
      id: apiJob.id,
      name: apiJob.name,
      modelType: apiJob.model_type,
      status: apiJob.status,
      progress: apiJob.progress,
      datasetSize: apiJob.dataset_size,
      epochs: apiJob.epochs,
      currentEpoch: apiJob.current_epoch,
      startTime: startTime,
      estimatedTime: estimatedTime,
      gpuUsage: apiJob.gpu_usage,
      memoryUsage: apiJob.memory_usage,
      config: apiJob.config,
    };
  };

  // 处理查看训练详情
  const handleViewTrainingDetail = (job: UiTrainingJob) => {
    setSelectedTraining(job);
    setShowTrainingDetailModal(true);
  };

  // 加载训练任务
  const loadJobs = async () => {
    try {
      setIsLoading(prev => ({ ...prev, jobs: true }));
      const response = await trainingApi.getJobs();
      if (response.success && response.data) {
        const uiJobs = response.data.map(convertApiJobToUi);
        setJobs(uiJobs);
      }
    } catch (error) {
      console.error('加载训练任务失败:', error);
      setJobs([]);
      const errorMessage = error instanceof Error ? error.message : '网络错误';
      toast.error('加载训练任务失败: ' + errorMessage);
    } finally {
      setIsLoading(prev => ({ ...prev, jobs: false }));
    }
  };

  // 加载训练统计
  const loadTrainingStats = async () => {
    try {
      setIsLoading(prev => ({ ...prev, stats: true }));
      const response = await trainingApi.getStats();
      if (response.success && response.data) {
        setTrainingStats(response.data);
      }
    } catch (error) {
      console.error('加载训练统计失败:', error);
      setTrainingStats(null);
    } finally {
      setIsLoading(prev => ({ ...prev, stats: false }));
    }
  };

  // 加载GPU状态
  const loadGpuStatus = async () => {
    try {
      setIsLoading(prev => ({ ...prev, gpu: true }));
      const response = await trainingApi.getGpuStatus();
      if (response.success && response.data) {
        const gpuData = response.data;
        setSystemResources(prev => ({
          ...prev,
          gpuAvailable: gpuData.gpu_count > 0,
          gpuMemory: gpuData.memory_total,
          gpuUtilization: gpuData.utilization,
        }));
      }
    } catch (error) {
      console.error('加载GPU状态失败:', error);
      // API调用失败，保持原有系统资源状态不变
      // 不设置虚拟数据，仅记录错误
    } finally {
      setIsLoading(prev => ({ ...prev, gpu: false }));
    }
  };

  // 加载可用数据集
  const loadDatasets = async () => {
    try {
      const response = await trainingApi.getDatasets();
      if (response.success && response.data) {
        setAvailableDatasets(response.data);
      }
    } catch (error) {
      console.error('加载数据集失败:', error);
      // API调用失败，设置空数据集列表
      setAvailableDatasets([]);
    }
  };

  // 加载可用模型类型
  const loadModelTypes = async () => {
    try {
      const response = await trainingApi.getModelTypes();
      if (response.success && response.data) {
        _setAvailableModelTypes(response.data);
      }
    } catch (error) {
      console.error('加载模型类型失败:', error);
      // API调用失败，设置空模型类型列表
      _setAvailableModelTypes([]);
    }
  };

  // 加载训练监控数据
  const loadMonitoringData = async () => {
    try {
      if (jobs.length === 0) {
        // 如果没有训练任务，设置空监控数据
        setMonitoringData({
          metrics: [],
          resourceUsage: [],
          trainingProgress: [],
          activeAlerts: [],
        });
        return;
      }

      // 获取运行中的训练任务
      const runningJobs = jobs.filter(job => job.status === 'running');
      if (runningJobs.length === 0) {
        // 没有运行中的任务，设置空监控数据
        setMonitoringData({
          metrics: [],
          resourceUsage: [],
          trainingProgress: [],
          activeAlerts: [],
        });
        return;
      }

      // 实际项目中应该调用后端API获取监控数据
      // const response = await trainingApi.getMonitoringData(runningJobs[0].id);
      // if (response.success && response.data) {
      //   setMonitoringData(response.data);
      // } else {
      //   setMonitoringData({
      //     metrics: [],
      //     resourceUsage: [],
      //     trainingProgress: [],
      //   });
      // }
      
      // 目前没有监控数据API，设置空数据
      setMonitoringData({
        metrics: [],
        resourceUsage: [],
        trainingProgress: [],
        activeAlerts: [],
      });
    } catch (error) {
      console.error('加载监控数据失败:', error);
      // 设置空数据作为后备
      setMonitoringData({
        metrics: [],
        resourceUsage: [],
        trainingProgress: [],
        activeAlerts: [],
      });
    }
  };



  // 加载拉普拉斯增强系统状态
  const loadLaplacianStatus = async () => {
    try {
      setIsLoadingLaplacian(true);
      const response = await trainingApi.getLaplacianStatus();
      if (response.success && response.data) {
        setLaplacianStatus(response.data as any);
        // 如果系统已配置，更新本地配置状态
        if (response.data.initialized && response.data.available) {
          setLaplacianConfig({
            enhancement_mode: response.data.enhancement_mode || 'graph_laplacian',
            enabled_components: response.data.enabled_components || ['regularization', 'feature_extraction'],
            regularization_lambda: response.data.regularization_lambda || 0.01,
            adaptive_lambda: response.data.adaptive_lambda || true,
          });
        }
      }
    } catch (error) {
      console.error('加载拉普拉斯增强系统状态失败:', error);
      toast.error('加载拉普拉斯系统状态失败');
    } finally {
      setIsLoadingLaplacian(false);
    }
  };

  // 保存拉普拉斯配置
  const saveLaplacianConfig = async () => {
    try {
      setIsLoadingLaplacian(true);
      const response = await trainingApi.configureLaplacianSystem(laplacianConfig);
      if (response.success && response.data) {
        toast.success('拉普拉斯配置已保存');
        // 重新加载状态
        loadLaplacianStatus();
      } else {
        toast.error('保存配置失败');
      }
    } catch (error) {
      console.error('保存拉普拉斯配置失败:', error);
      toast.error('保存配置失败');
    } finally {
      setIsLoadingLaplacian(false);
    }
  };

  // 启用/禁用拉普拉斯增强系统
  const toggleLaplacianSystem = async (enable: boolean) => {
    try {
      setIsLoadingLaplacian(true);
      const response = await trainingApi.enableLaplacianSystem(enable);
      if (response.success && response.data) {
        toast.success(`拉普拉斯增强系统已${enable ? '启用' : '禁用'}`);
        // 重新加载状态
        loadLaplacianStatus();
      } else {
        toast.error(`${enable ? '启用' : '禁用'}系统失败`);
      }
    } catch (error) {
      console.error(`${enable ? '启用' : '禁用'}拉普拉斯系统失败:`, error);
      toast.error(`${enable ? '启用' : '禁用'}系统失败`);
    } finally {
      setIsLoadingLaplacian(false);
    }
  };

  // 应用拉普拉斯增强到训练任务
  const applyLaplacianEnhancement = async (jobId: string) => {
    try {
      setIsLoadingLaplacian(true);
      const response = await trainingApi.applyLaplacianEnhancement(jobId, {
        enhancement_mode: laplacianConfig.enhancement_mode,
        regularization_lambda: laplacianConfig.regularization_lambda,
      });
      if (response.success && response.data) {
        toast.success('拉普拉斯增强已应用到训练任务');
        // 刷新训练任务列表
        loadJobs();
      } else {
        toast.error('应用增强失败');
      }
    } catch (error) {
      console.error('应用拉普拉斯增强失败:', error);
      toast.error('应用增强失败');
    } finally {
      setIsLoadingLaplacian(false);
    }
  };

  // 清除警报
  const clearAlert = (alertId: string) => {
    setMonitoringData(prev => ({
      ...prev,
      activeAlerts: prev.activeAlerts.filter(alert => alert.id !== alertId)
    }));
    toast.success('警报已清除');
  };

  // 更新监控间隔
  const updateMonitoringInterval = (interval: number) => {
    setMonitoringInterval(interval);
    toast.success(`监控间隔已设置为${interval / 1000}秒`);
  };

  // 初始加载
  useEffect(() => {
    // 只在用户已认证且是管理员时加载数据
    if (!isAuthenticated || !isAdmin || authLoading) {
      return;
    }
    
    loadJobs();
    loadTrainingStats();
    loadGpuStatus();
    loadDatasets();
    loadModelTypes();
    loadMonitoringData();
    loadLaplacianStatus();

    // 设置轮询更新
    pollIntervalRef.current = setInterval(() => {
      if (isAuthenticated && isAdmin) {
        loadJobs();
        loadTrainingStats();
        loadGpuStatus();
        if (autoRefresh) {
          loadMonitoringData();
        }
      }
    }, 5000); // 优化：从10秒减少到5秒，降低实时数据更新延迟

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [isAuthenticated, isAdmin, authLoading, autoRefresh]);

  const handleStartJob = async (jobId: string) => {
    try {
      const response = await trainingApi.resumeJob(jobId);
      if (response.success) {
        toast.success('训练任务已开始');
        loadJobs(); // 刷新任务列表
      } else {
        toast.error('启动任务失败');
      }
    } catch (error) {
      console.error('启动训练任务失败:', error);
      toast.error('启动任务失败');
    }
  };

  const handlePauseJob = async (jobId: string) => {
    try {
      const response = await trainingApi.pauseJob(jobId);
      if (response.success) {
        toast.success('训练任务已暂停');
        loadJobs(); // 刷新任务列表
      } else {
        toast.error('暂停任务失败');
      }
    } catch (error) {
      console.error('暂停训练任务失败:', error);
      toast.error('暂停任务失败');
    }
  };

  const handleStopJob = async (jobId: string) => {
    try {
      const response = await trainingApi.stopJob(jobId);
      if (response.success) {
        toast.success('训练任务已停止');
        loadJobs(); // 刷新任务列表
      } else {
        toast.error('停止任务失败');
      }
    } catch (error) {
      console.error('停止训练任务失败:', error);
      toast.error('停止任务失败');
    }
  };

  const handleDeleteJob = async (jobId: string) => {
    if (!confirm('确定要删除此训练任务吗？')) {
      return;
    }
    
    try {
      const response = await trainingApi.deleteJob(jobId);
      if (response.success) {
        toast.success('训练任务已删除');
        loadJobs(); // 刷新任务列表
      } else {
        toast.error('删除任务失败');
      }
    } catch (error) {
      console.error('删除训练任务失败:', error);
      toast.error('删除任务失败');
    }
  };

  const handleCreateJob = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!newConfig.model_name?.trim()) {
      toast.error('请输入模型名称');
      return;
    }
    
    if (!newConfig.dataset_path.trim()) {
      toast.error('请选择数据集');
      return;
    }
    
    setIsCreatingJob(true);
    
    try {
      const trainingRequest: TrainingRequest = {
        name: newConfig.model_name,
        description: newConfig.description,
        config: newConfig,
      };
      
      const response = await trainingApi.createJob(trainingRequest);
      
      if (response.success && response.data) {
        toast.success('训练任务已创建');
        setNewConfig({
          model_type: 'transformer',
          dataset_path: '',
          epochs: 100,
          batch_size: 32,
          learning_rate: 0.001,
          use_gpu: true,
          save_checkpoints: true,
          early_stopping: true,
          validation_split: 0.2,
          model_name: '',
          description: '',
        });
        setActiveTab('jobs');
        loadJobs(); // 刷新任务列表
      } else {
        toast.error('创建训练任务失败');
      }
    } catch (error) {
      console.error('创建训练任务失败:', error);
      toast.error('创建训练任务失败');
    } finally {
      setIsCreatingJob(false);
    }
  };

  const handleDatasetUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!datasetUploadFile) {
      toast.error('请选择要上传的文件');
      return;
    }
    
    if (!datasetUploadName.trim()) {
      toast.error('请输入数据集名称');
      return;
    }
    
    try {
      toast.loading('上传数据集中...', { id: 'dataset-upload' });
      const response = await trainingApi.uploadDataset(datasetUploadFile, datasetUploadName);
      toast.dismiss('dataset-upload');
      
      if (response.success && response.data) {
        toast.success('数据集上传成功');
        setDatasetUploadFile(null);
        setDatasetUploadName('');
        loadDatasets(); // 刷新数据集列表
        
        // 自动设置新上传的数据集
        setNewConfig(prev => ({
          ...prev,
          dataset_path: response.data!.path,
        }));
      } else {
        toast.error('数据集上传失败');
      }
    } catch (error) {
      console.error('数据集上传失败:', error);
      toast.dismiss('dataset-upload');
      toast.error('数据集上传失败');
    }
  };

  const handleConfigChange = (key: keyof TrainingConfig, value: any) => {
    setNewConfig(prev => ({ ...prev, [key]: value }));
  };

  const getStatusColor = (status: UiTrainingJob['status']) => {
    switch (status) {
      case 'running': return 'bg-gray-600 text-gray-600 dark:bg-gray-700/30 dark:text-gray-600';
      case 'paused': return 'bg-gray-500 text-gray-500 dark:bg-gray-800/30 dark:text-gray-500';
      case 'completed': return 'bg-gray-700 text-gray-700 dark:bg-gray-700/30 dark:text-gray-700';
      case 'failed': return 'bg-gray-800 text-gray-800 dark:bg-gray-900/30 dark:text-gray-800';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400';
    }
  };

  const getStatusIcon = (status: UiTrainingJob['status']) => {
    switch (status) {
      case 'running': return <Play className="w-4 h-4" />;
      case 'paused': return <Pause className="w-4 h-4" />;
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'failed': return <XCircle className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  const getModelTypeColor = (type: string) => {
    switch (type) {
      case 'transformer': return 'bg-gradient-to-r from-gray-700 to-cyan-500';
      case 'multimodal': return 'bg-gradient-to-r from-gray-600 to-gray-500';
      case 'cognitive': return 'bg-gradient-to-r from-gray-600 to-emerald-500';
      case 'pinn': return 'bg-gradient-to-r from-orange-500 to-gray-800';
      case 'cnn_enhanced': return 'bg-gradient-to-r from-gray-800 to-gray-500';
      case 'graph_nn': return 'bg-gradient-to-r from-teal-500 to-cyan-500';
      case 'laplacian': return 'bg-gradient-to-r from-gray-800 to-gray-600';
      case 'hybrid': return 'bg-gradient-to-r from-gray-600 via-gray-500 to-gray-800';
      default: return 'bg-gray-500';
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };



  return (
    <ResponsiveContainer
      type="main"
      maxWidth="screen"
      padding="responsive"
      margin="responsive"
      background="white"
      border="none"
      shadow="none"
      flex={true}
      flexDirection="col"
      alignItems="stretch"
      justifyContent="start"
      className="space-y-6"
    >
      {/* 页面标题和操作 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            模型训练系统
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            训练和管理Self AGI的AI模型
          </p>
          <div className="mt-2">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-800 text-white">
              功能状态：实现中
            </span>
            <span className="ml-2 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-600 text-white">
              后端服务：已连接
            </span>
            <span className="ml-2 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-700 text-white">
              训练模式：GPU/CPU双模式
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`flex items-center px-3 py-1 rounded-full ${
              systemResources.gpuAvailable 
                ? 'bg-gradient-to-r from-gray-600 to-emerald-100 dark:from-gray-600/30 dark:to-emerald-900/30' 
                : 'bg-gradient-to-r from-gray-800 to-gray-500 dark:from-gray-800/30 dark:to-gray-500/30'
            }`}>
              <Cpu className={`w-4 h-4 mr-2 ${
                systemResources.gpuAvailable 
                  ? 'text-gray-600 dark:text-gray-600' 
                  : 'text-gray-800 dark:text-gray-800'
              }`} />
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                GPU: {systemResources.gpuAvailable ? '可用' : '不可用'}
              </span>
            </div>
            <button
              onClick={() => setActiveTab('new')}
              className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-700"
              aria-label="新建训练任务"
            >
              <Plus className="w-4 h-4 mr-2" aria-hidden="true" />
              新建训练
            </button>
          </div>
        </div>
      </div>

      {/* 训练统计卡片 */}
      {trainingStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900 dark:text-white">总训练任务</h3>
              <BarChart3 className="w-5 h-5 text-gray-700" />
            </div>
            <div className="mt-3">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {trainingStats.total_jobs}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {trainingStats.running_jobs} 个运行中
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900 dark:text-white">已完成任务</h3>
              <CheckCircle className="w-5 h-5 text-gray-600" />
            </div>
            <div className="mt-3">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {trainingStats.completed_jobs}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                成功率: {trainingStats.total_jobs > 0 
                  ? `${((trainingStats.completed_jobs / trainingStats.total_jobs) * 100).toFixed(1)}%`
                  : '0%'}
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900 dark:text-white">总训练时间</h3>
              <Clock className="w-5 h-5 text-gray-500" />
            </div>
            <div className="mt-3">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {trainingStats.total_training_hours.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                小时
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900 dark:text-white">GPU利用率</h3>
              <Zap className="w-5 h-5 text-gray-600" />
            </div>
            <div className="mt-3">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {trainingStats.gpu_utilization.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                平均使用率
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 资源状态卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-gray-900 dark:text-white flex items-center">
              <Zap className="w-5 h-5 text-gray-500 mr-2" />
              GPU资源
            </h3>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {systemResources.gpuUtilization.toFixed(1)}% 使用率
            </span>
          </div>
          <div className="space-y-2">
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-gray-700 to-gray-600 h-2 rounded-full"
                style={{ width: `${systemResources.gpuUtilization}%` }}
              />
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {systemResources.gpuAvailable ? '内存可用' : 'GPU不可用'}
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-gray-900 dark:text-white flex items-center">
              <BarChart3 className="w-5 h-5 text-gray-700 mr-2" />
              系统资源
            </h3>
            <button
              onClick={loadGpuStatus}
              className="p-1 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              title="刷新系统资源状态"
              aria-label="刷新系统资源状态"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading.gpu ? 'animate-spin' : ''}`} aria-hidden="true" />
            </button>
          </div>
          <div className="space-y-2">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              加载训练系统状态中...
            </div>
            {isLoading.gpu && (
              <Loader2 className="w-4 h-4 animate-spin text-gray-700" />
            )}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-gray-900 dark:text-white flex items-center">
              <HardDrive className="w-5 h-5 text-gray-600 mr-2" />
              存储状态
            </h3>
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
              数据集: {availableDatasets.length}
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              可用数据集: {availableDatasets.length} 个
            </div>
            <button
              onClick={() => loadDatasets()}
              className="text-sm text-gray-700 dark:text-gray-700 hover:underline"
            >
              刷新数据集列表
            </button>
          </div>
        </div>
      </div>

      {/* 标签页导航 */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('jobs')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'jobs'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Brain className="w-4 h-4 mr-2" />
              训练任务
              <span className="ml-2 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded-full px-2 py-0.5 text-xs">
                {jobs.length}
              </span>
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('new')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'new'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Plus className="w-4 h-4 mr-2" />
              新建训练
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('monitor')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'monitor'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <BarChart3 className="w-4 h-4 mr-2" />
              训练监控
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab('laplacian')}
            className={`py-3 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'laplacian'
                ? 'border-gray-700 text-gray-700 dark:text-gray-700'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <div className="flex items-center">
              <Settings className="w-4 h-4 mr-2" />
              拉普拉斯增强
            </div>
          </button>
        </nav>
      </div>

      {/* 训练任务列表 */}
      {activeTab === 'jobs' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <h2 className="font-semibold text-gray-900 dark:text-white">训练任务</h2>
            <button
              onClick={loadJobs}
              className="inline-flex items-center px-3 py-1 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
              disabled={isLoading.jobs}
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${isLoading.jobs ? 'animate-spin' : ''}`} />
              刷新
            </button>
          </div>
          
          {isLoading.jobs ? (
            <div className="p-8 text-center">
              <Loader2 className="w-8 h-8 animate-spin text-gray-700 mx-auto" />
              <p className="mt-2 text-gray-600 dark:text-gray-400">加载训练任务中...</p>
            </div>
          ) : jobs.length === 0 ? (
            <div className="p-8 text-center">
              <Brain className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <h3 className="font-medium text-gray-900 dark:text-white">暂无训练任务</h3>
              <p className="mt-1 text-gray-600 dark:text-gray-400">点击"新建训练"创建您的第一个训练任务</p>
              <button
                onClick={() => setActiveTab('new')}
                className="mt-4 inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600"
              >
                <Plus className="w-4 h-4 mr-2" />
                新建训练
              </button>
            </div>
          ) : (
            <div className="divide-y divide-gray-200 dark:divide-gray-700">
              {jobs.map(job => (
                <div key={job.id} className="p-4 hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <div className={`w-3 h-3 rounded-full ${getStatusColor(job.status)} flex items-center justify-center`}>
                          {getStatusIcon(job.status)}
                        </div>
                        <h3 className="font-medium text-gray-900 dark:text-white">
                          {job.name}
                        </h3>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(job.status)}`}>
                          {job.status === 'running' ? '运行中' : 
                           job.status === 'paused' ? '已暂停' : 
                           job.status === 'completed' ? '已完成' : 
                           job.status === 'failed' ? '失败' : '等待中'}
                        </span>
                        <span className={`px-2 py-1 text-xs font-medium text-white rounded-full ${getModelTypeColor(job.modelType)}`}>
                          {job.modelType === 'transformer' ? 'Transformer' : 
                           job.modelType === 'multimodal' ? '多模态' : 
                           job.modelType === 'cognitive' ? '认知推理' : 
                           job.modelType === 'pinn' ? 'PINN物理建模' : 
                           job.modelType === 'cnn_enhanced' ? 'CNN增强视觉' : 
                           job.modelType === 'graph_nn' ? '图神经网络' : 
                           job.modelType === 'laplacian' ? '拉普拉斯技术' : 
                           job.modelType === 'hybrid' ? '混合架构' : job.modelType}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-3">
                        <div className="space-y-2">
                          <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                            <Clock className="w-4 h-4 mr-2" />
                            进度: {job.progress.toFixed(1)}%
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-gradient-to-r from-gray-700 to-gray-600 h-2 rounded-full"
                              style={{ width: `${job.progress}%` }}
                            />
                          </div>
                        </div>
                        
                        <div className="space-y-1">
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            轮次: {job.currentEpoch} / {job.epochs}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            数据集大小: {job.datasetSize.toLocaleString()}
                          </div>
                        </div>
                        
                        <div className="space-y-1">
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            预计完成时间: {job.estimatedTime}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            开始时间: {job.startTime.toLocaleDateString('zh-CN')}
                          </div>
                        </div>
                        
                        <div className="space-y-1">
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            GPU使用: {job.gpuUsage.toFixed(1)}%
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            内存使用: {job.memoryUsage.toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="ml-4 flex items-center space-x-2">
                      {job.status === 'paused' || job.status === 'pending' ? (
                        <button
                          onClick={() => handleStartJob(job.id)}
                          className="p-2 text-gray-600 bg-gray-600 dark:bg-gray-700/30 dark:text-gray-600 rounded-lg hover:bg-gray-600 dark:hover:bg-gray-600/50"
                          title="开始训练"
                        >
                          <Play className="w-4 h-4" />
                        </button>
                      ) : job.status === 'running' ? (
                        <button
                          onClick={() => handlePauseJob(job.id)}
                          className="p-2 text-gray-500 bg-gray-500 dark:bg-gray-800/30 dark:text-gray-500 rounded-lg hover:bg-gray-500 dark:hover:bg-gray-500/50"
                          title="暂停训练"
                        >
                          <Pause className="w-4 h-4" />
                        </button>
                      ) : null}
                      
                      {job.status === 'running' && (
                        <button
                          onClick={() => handleStopJob(job.id)}
                          className="p-2 text-gray-800 bg-gray-800 dark:bg-gray-900/30 dark:text-gray-800 rounded-lg hover:bg-gray-800 dark:hover:bg-gray-800/50"
                          title="停止训练"
                        >
                          <StopCircle className="w-4 h-4" />
                        </button>
                      )}
                      
                      {(job.status === 'completed' || job.status === 'failed') && (
                        <button
                          onClick={() => handleDeleteJob(job.id)}
                          className="p-2 text-gray-800 bg-gray-800 dark:bg-gray-900/30 dark:text-gray-800 rounded-lg hover:bg-gray-800 dark:hover:bg-gray-800/50"
                          title="删除任务"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                      
                      <button
                        onClick={() => handleViewTrainingDetail(job)}
                        className="p-2 text-gray-700 bg-gray-700 dark:bg-gray-700/30 dark:text-gray-700 rounded-lg hover:bg-gray-700 dark:hover:bg-gray-700/50"
                        title="查看详情"
                      >
                        <Settings className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 新建训练任务 */}
      {activeTab === 'new' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <h2 className="font-semibold text-gray-900 dark:text-white">新建训练任务</h2>
          </div>
          
          <div className="p-6">
            <form onSubmit={handleCreateJob} className="space-y-6">
              {/* 基础信息 */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label htmlFor="model-name" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    模型名称 *
                  </label>
                  <input
                    id="model-name"
                    type="text"
                    value={newConfig.model_name || ''}
                    onChange={(e) => handleConfigChange('model_name', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                    placeholder="请输入模型名称"
                    required
                    aria-required="true"
                  />
                </div>
                
                <div>
                  <label htmlFor="model-type" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    模型类型
                  </label>
                  <select
                    id="model-type"
                    value={newConfig.model_type}
                    onChange={(e) => handleConfigChange('model_type', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                    aria-label="选择模型类型"
                  >
                    <option value="transformer">Transformer 模型</option>
                    <option value="multimodal">多模态融合模型</option>
                    <option value="cognitive">认知推理模型</option>
                    <option value="pinn">PINN物理建模框架</option>
                    <option value="cnn_enhanced">CNN增强视觉处理</option>
                    <option value="graph_nn">图神经网络</option>
                    <option value="laplacian">拉普拉斯技术</option>
                    <option value="hybrid">混合技术架构</option>
                  </select>
                </div>
              </div>
              
              <div>
                <label htmlFor="training-description" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  训练描述
                </label>
                <textarea
                  id="training-description"
                  value={newConfig.description || ''}
                  onChange={(e) => handleConfigChange('description', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                  rows={2}
                  placeholder="请输入训练任务描述"
                  aria-label="训练任务描述"
                />
              </div>
              
              {/* 数据集配置 */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">数据集配置</h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      选择数据集 *
                    </label>
                    <select
                      value={newConfig.dataset_path}
                      onChange={(e) => handleConfigChange('dataset_path', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                      required
                    >
                      <option value="">请选择数据集</option>
                      {availableDatasets.map(dataset => (
                        <option key={dataset.path} value={dataset.path}>
                          {dataset.name} ({formatFileSize(dataset.size)})
                        </option>
                      ))}
                    </select>
                    {availableDatasets.length === 0 && (
                      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                        暂无可用数据集，请先上传数据集
                      </p>
                    )}
                  </div>
                  
                  <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 dark:text-white mb-3">上传新数据集</h4>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          数据集名称
                        </label>
                        <input
                          type="text"
                          value={datasetUploadName}
                          onChange={(e) => setDatasetUploadName(e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                          placeholder="请输入数据集名称"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          数据集文件
                        </label>
                        <div className="flex items-center space-x-4">
                          <input
                            type="file"
                            onChange={(e) => setDatasetUploadFile(e.target.files?.[0] || null)}
                            className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                          />
                          <button
                            onClick={handleDatasetUpload}
                            disabled={!datasetUploadFile || !datasetUploadName.trim()}
                            className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-600 to-emerald-600 rounded-md hover:from-gray-600 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            <Upload className="w-4 h-4 inline mr-2" />
                            上传
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* 训练参数 */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">训练参数</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      训练轮次
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="1000"
                      value={newConfig.epochs}
                      onChange={(e) => handleConfigChange('epochs', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      批量大小
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="1024"
                      value={newConfig.batch_size}
                      onChange={(e) => handleConfigChange('batch_size', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      学习率
                    </label>
                    <input
                      type="number"
                      step="0.0001"
                      min="0.0001"
                      max="1"
                      value={newConfig.learning_rate}
                      onChange={(e) => handleConfigChange('learning_rate', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                    />
                  </div>
                </div>
                
                <div className="mt-6 space-y-4">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="useGPU"
                      checked={newConfig.use_gpu}
                      onChange={(e) => handleConfigChange('use_gpu', e.target.checked)}
                      className="h-4 w-4 text-gray-700 rounded focus:ring-gray-700"
                    />
                    <label htmlFor="useGPU" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                      使用GPU加速训练 {systemResources.gpuAvailable ? '(可用)' : '(不可用)'}
                    </label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="saveCheckpoints"
                      checked={newConfig.save_checkpoints}
                      onChange={(e) => handleConfigChange('save_checkpoints', e.target.checked)}
                      className="h-4 w-4 text-gray-700 rounded focus:ring-gray-700"
                    />
                    <label htmlFor="saveCheckpoints" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                      保存检查点
                    </label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="earlyStopping"
                      checked={newConfig.early_stopping}
                      onChange={(e) => handleConfigChange('early_stopping', e.target.checked)}
                      className="h-4 w-4 text-gray-700 rounded focus:ring-gray-700"
                    />
                    <label htmlFor="earlyStopping" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                      启用早停
                    </label>
                  </div>
                </div>
              </div>
              
              {/* 提交按钮 */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                <div className="flex items-center justify-between">
                  <button
                    type="button"
                    onClick={() => setActiveTab('jobs')}
                    className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
                  >
                    取消
                  </button>
                  
                  <button
                    type="submit"
                    disabled={isCreatingJob}
                    className="inline-flex items-center px-6 py-3 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isCreatingJob ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        创建中...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        开始训练
                      </>
                    )}
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* 训练监控页面 */}
      {activeTab === 'monitor' && (
        <div className="space-y-6">
          {/* 监控控制面板 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
              <h2 className="font-semibold text-gray-900 dark:text-white">训练监控</h2>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <label className="text-sm text-gray-700 dark:text-gray-300">时间范围:</label>
                  <select
                    value={selectedTimeRange}
                    onChange={(e) => setSelectedTimeRange(e.target.value as any)}
                    className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                  >
                    <option value="1h">最近1小时</option>
                    <option value="6h">最近6小时</option>
                    <option value="12h">最近12小时</option>
                    <option value="24h">最近24小时</option>
                  </select>
                </div>
                
                <div className="flex items-center space-x-2">
                  <label className="text-sm text-gray-700 dark:text-gray-300">监控间隔:</label>
                  <select
                    value={monitoringInterval}
                    onChange={(e) => updateMonitoringInterval(parseInt(e.target.value))}
                    className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
                  >
                    <option value="2000">2秒</option>
                    <option value="5000">5秒</option>
                    <option value="10000">10秒</option>
                    <option value="30000">30秒</option>
                  </select>
                </div>
                
                <button
                  onClick={loadMonitoringData}
                  className="inline-flex items-center px-3 py-1 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  刷新数据
                </button>
                
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="autoRefresh"
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                    className="h-4 w-4 text-gray-700 rounded focus:ring-gray-700"
                  />
                  <label htmlFor="autoRefresh" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                    自动刷新
                  </label>
                </div>
              </div>
            </div>
            
            <div className="p-6">
              {/* 指标卡片 */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-gradient-to-r from-gray-700 to-gray-700 dark:from-gray-700/30 dark:to-gray-700/30 p-4 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-700 dark:text-gray-700">当前损失</p>
                      <p className="text-2xl font-semibold text-gray-700 dark:text-gray-300">
                        {monitoringData.metrics.length > 0 ? monitoringData.metrics[monitoringData.metrics.length - 1].loss.toFixed(4) : '0.0000'}
                      </p>
                    </div>
                    <div className="w-12 h-12 rounded-full bg-gray-700 dark:bg-gray-700/50 flex items-center justify-center">
                      <BarChart3 className="w-6 h-6 text-gray-700 dark:text-gray-700" />
                    </div>
                  </div>
                </div>
                
                <div className="bg-gradient-to-r from-gray-600 to-gray-600 dark:from-gray-600/30 dark:to-gray-600/30 p-4 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-600">当前准确率</p>
                      <p className="text-2xl font-semibold text-gray-600 dark:text-gray-300">
                        {monitoringData.metrics.length > 0 ? (monitoringData.metrics[monitoringData.metrics.length - 1].accuracy * 100).toFixed(2) + '%' : '0.00%'}
                      </p>
                    </div>
                    <div className="w-12 h-12 rounded-full bg-gray-600 dark:bg-gray-600/50 flex items-center justify-center">
                      <CheckCircle className="w-6 h-6 text-gray-600 dark:text-gray-600" />
                    </div>
                  </div>
                </div>
                
                <div className="bg-gradient-to-r from-gray-600 to-gray-600 dark:from-gray-600/30 dark:to-gray-600/30 p-4 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-600">平均CPU使用率</p>
                      <p className="text-2xl font-semibold text-gray-600 dark:text-gray-600">
                        {monitoringData.resourceUsage.length > 0 
                          ? Math.round(monitoringData.resourceUsage.reduce((sum, r) => sum + r.cpu, 0) / monitoringData.resourceUsage.length) + '%' 
                          : '0%'}
                      </p>
                    </div>
                    <div className="w-12 h-12 rounded-full bg-gray-600 dark:bg-gray-600/50 flex items-center justify-center">
                      <Cpu className="w-6 h-6 text-gray-600 dark:text-gray-600" />
                    </div>
                  </div>
                </div>
                
                <div className="bg-gradient-to-r from-orange-50 to-orange-100 dark:from-orange-900/30 dark:to-orange-800/30 p-4 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-orange-600 dark:text-orange-400">活跃警报</p>
                      <p className="text-2xl font-semibold text-orange-800 dark:text-orange-300">
                        {monitoringData.activeAlerts.length}
                      </p>
                    </div>
                    <div className="w-12 h-12 rounded-full bg-orange-100 dark:bg-orange-800/50 flex items-center justify-center">
                      {monitoringData.activeAlerts.length > 0 ? (
                        <XCircle className="w-6 h-6 text-orange-600 dark:text-orange-400" />
                      ) : (
                        <CheckCircle className="w-6 h-6 text-gray-600 dark:text-gray-600" />
                      )}
                    </div>
                  </div>
                </div>
              </div>
              
              {/* 图表区域 */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                {/* 损失和准确率图表 */}
                <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <h3 className="font-medium text-gray-900 dark:text-white mb-4">损失和准确率趋势</h3>
                  <div className="h-64 flex items-center justify-center">
                    <div className="text-center">
                      <BarChart3 className="w-12 h-12 mx-auto text-gray-400 mb-2" />
                      <p className="text-gray-600 dark:text-gray-400">图表组件待集成</p>
                      <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                        {monitoringData.metrics.length} 个数据点
                      </p>
                    </div>
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">最小损失</p>
                      <p className="text-lg font-medium text-gray-900 dark:text-white">
                        {monitoringData.metrics.length > 0 
                          ? Math.min(...monitoringData.metrics.map(m => m.loss)).toFixed(4) 
                          : '0.0000'}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">最大准确率</p>
                      <p className="text-lg font-medium text-gray-900 dark:text-white">
                        {monitoringData.metrics.length > 0 
                          ? (Math.max(...monitoringData.metrics.map(m => m.accuracy)) * 100).toFixed(2) + '%' 
                          : '0.00%'}
                      </p>
                    </div>
                  </div>
                </div>
                
                {/* 资源使用率图表 */}
                <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <h3 className="font-medium text-gray-900 dark:text-white mb-4">资源使用率</h3>
                  <div className="h-64 flex items-center justify-center">
                    <div className="text-center">
                      <Cpu className="w-12 h-12 mx-auto text-gray-400 mb-2" />
                      <p className="text-gray-600 dark:text-gray-400">图表组件待集成</p>
                      <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                        {monitoringData.resourceUsage.length} 个数据点
                      </p>
                    </div>
                  </div>
                  <div className="mt-4 grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">平均CPU</p>
                      <p className="text-lg font-medium text-gray-700 dark:text-gray-700">
                        {monitoringData.resourceUsage.length > 0 
                          ? Math.round(monitoringData.resourceUsage.reduce((sum, r) => sum + r.cpu, 0) / monitoringData.resourceUsage.length) + '%' 
                          : '0%'}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">平均内存</p>
                      <p className="text-lg font-medium text-gray-600 dark:text-gray-600">
                        {monitoringData.resourceUsage.length > 0 
                          ? Math.round(monitoringData.resourceUsage.reduce((sum, r) => sum + r.memory, 0) / monitoringData.resourceUsage.length) + '%' 
                          : '0%'}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">平均GPU</p>
                      <p className="text-lg font-medium text-gray-600 dark:text-gray-600">
                        {monitoringData.resourceUsage.length > 0 
                          ? Math.round(monitoringData.resourceUsage.reduce((sum, r) => sum + r.gpu, 0) / monitoringData.resourceUsage.length) + '%' 
                          : '0%'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* 训练进度图表 */}
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 mb-6">
                <h3 className="font-medium text-gray-900 dark:text-white mb-4">训练进度（按轮次）</h3>
                <div className="h-64 flex items-center justify-center">
                  <div className="text-center">
                    <Brain className="w-12 h-12 mx-auto text-gray-400 mb-2" />
                    <p className="text-gray-600 dark:text-gray-400">图表组件待集成</p>
                    <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                      {monitoringData.trainingProgress.length} 个数据点
                    </p>
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-3 gap-4">
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">总轮次</p>
                    <p className="text-lg font-medium text-gray-900 dark:text-white">
                      {monitoringData.trainingProgress.length > 0 
                        ? monitoringData.trainingProgress[monitoringData.trainingProgress.length - 1].epoch 
                        : 0}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">最终损失</p>
                    <p className="text-lg font-medium text-gray-900 dark:text-white">
                      {monitoringData.trainingProgress.length > 0 
                        ? monitoringData.trainingProgress[monitoringData.trainingProgress.length - 1].loss.toFixed(4) 
                        : '0.0000'}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">最终准确率</p>
                    <p className="text-lg font-medium text-gray-900 dark:text-white">
                      {monitoringData.trainingProgress.length > 0 
                        ? (monitoringData.trainingProgress[monitoringData.trainingProgress.length - 1].accuracy * 100).toFixed(2) + '%' 
                        : '0.00%'}
                    </p>
                  </div>
                </div>
              </div>
              
              {/* 活跃警报 */}
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-medium text-gray-900 dark:text-white">系统警报</h3>
                  {monitoringData.activeAlerts.length > 0 && (
                    <button
                      onClick={() => setMonitoringData(prev => ({ ...prev, activeAlerts: [] }))}
                      className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-300"
                    >
                      清除所有警报
                    </button>
                  )}
                </div>
                
                {monitoringData.activeAlerts.length === 0 ? (
                  <div className="text-center py-8">
                    <CheckCircle className="w-12 h-12 mx-auto text-gray-600 mb-3" />
                    <p className="text-gray-600 dark:text-gray-400">没有活跃警报，系统运行正常</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {monitoringData.activeAlerts.map(alert => (
                      <div
                        key={alert.id}
                        className={`p-3 rounded-lg border ${
                          alert.severity === 'error'
                            ? 'bg-gray-900 dark:bg-gray-900/20 border-gray-800 dark:border-gray-800'
                            : alert.severity === 'warning'
                            ? 'bg-gray-800 dark:bg-gray-800/20 border-gray-500 dark:border-gray-500'
                            : 'bg-gray-700 dark:bg-gray-700/20 border-gray-700 dark:border-gray-700'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center">
                            {alert.severity === 'error' ? (
                              <XCircle className="w-5 h-5 text-gray-800 mr-3" />
                            ) : alert.severity === 'warning' ? (
                              <XCircle className="w-5 h-5 text-gray-500 mr-3" />
                            ) : (
                              <BarChart3 className="w-5 h-5 text-gray-700 mr-3" />
                            )}
                            <div>
                              <p className="font-medium text-gray-900 dark:text-white">{alert.message}</p>
                              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                                {alert.type === 'memory_high' ? '内存警告' : 
                                 alert.type === 'cpu_high' ? 'CPU警告' : 
                                 alert.type === 'gpu_high' ? 'GPU警告' : '系统警告'}
                              </p>
                            </div>
                          </div>
                          <button
                            onClick={() => clearAlert(alert.id)}
                            className="text-sm text-gray-500 hover:text-gray-900 dark:hover:text-gray-300"
                          >
                            清除
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* 运行中的训练任务 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="font-semibold text-gray-900 dark:text-white">运行中的训练任务</h2>
            </div>
            <div className="p-6">
              {jobs.filter(job => job.status === 'running').length === 0 ? (
                <div className="text-center py-8">
                  <Brain className="w-12 h-12 mx-auto text-gray-400 mb-3" />
                  <p className="text-gray-600 dark:text-gray-400">当前没有运行中的训练任务</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {jobs
                    .filter(job => job.status === 'running')
                    .map(job => (
                      <div
                        key={job.id}
                        className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4"
                      >
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <h4 className="font-medium text-gray-900 dark:text-white">{job.name}</h4>
                            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                              {job.modelType} • 已训练 {job.currentEpoch}/{job.epochs} 轮
                            </p>
                          </div>
                          <div className="flex items-center space-x-3">
                            <div className="text-right">
                              <p className="text-sm text-gray-600 dark:text-gray-400">进度</p>
                              <p className="text-lg font-semibold text-gray-700 dark:text-gray-700">
                                {job.progress.toFixed(1)}%
                              </p>
                            </div>
                            <div className="w-16 h-16">
                              <div className="relative w-16 h-16">
                                <svg className="w-16 h-16" viewBox="0 0 36 36">
                                  <path
                                    d="M18 2.0845
                                      a 15.9155 15.9155 0 0 1 0 31.831
                                      a 15.9155 15.9155 0 0 1 0 -31.831"
                                    fill="none"
                                    stroke="#E5E7EB"
                                    strokeWidth="3"
                                  />
                                  <path
                                    d="M18 2.0845
                                      a 15.9155 15.9155 0 0 1 0 31.831
                                      a 15.9155 15.9155 0 0 1 0 -31.831"
                                    fill="none"
                                    stroke="#3B82F6"
                                    strokeWidth="3"
                                    strokeLinecap="round"
                                    strokeDasharray={`${job.progress}, 100`}
                                  />
                                </svg>
                                <div className="absolute inset-0 flex items-center justify-center">
                                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                                    {Math.round(job.progress)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        
                        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">GPU使用率</p>
                            <div className="flex items-center mt-1">
                              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                                <div 
                                  className="bg-gradient-to-r from-gray-600 to-gray-500 h-2 rounded-full"
                                  style={{ width: `${job.gpuUsage}%` }}
                                />
                              </div>
                              <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                                {job.gpuUsage.toFixed(1)}%
                              </span>
                            </div>
                          </div>
                          
                          <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">内存使用率</p>
                            <div className="flex items-center mt-1">
                              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                                <div 
                                  className="bg-gradient-to-r from-gray-700 to-cyan-500 h-2 rounded-full"
                                  style={{ width: `${job.memoryUsage}%` }}
                                />
                              </div>
                              <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                                {job.memoryUsage.toFixed(1)}%
                              </span>
                            </div>
                          </div>
                          
                          <div>
                            <p className="text-sm text-gray-500 dark:text-gray-400">剩余时间</p>
                            <p className="text-lg font-medium text-gray-900 dark:text-white mt-1">
                              {job.estimatedTime}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 拉普拉斯增强系统 */}
      {activeTab === 'laplacian' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="font-semibold text-gray-900 dark:text-white">拉普拉斯增强系统</h2>
            </div>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="font-semibold text-gray-900 dark:text-white">拉普拉斯增强系统</h2>
                <button
                  onClick={loadLaplacianStatus}
                  className="inline-flex items-center px-3 py-1 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
                  disabled={isLoadingLaplacian}
                >
                  <RefreshCw className={`w-4 h-4 mr-2 ${isLoadingLaplacian ? 'animate-spin' : ''}`} />
                  刷新状态
                </button>
              </div>
              
              {isLoadingLaplacian ? (
                <div className="text-center py-8">
                  <Loader2 className="w-8 h-8 animate-spin text-gray-700 mx-auto" />
                  <p className="mt-2 text-gray-600 dark:text-gray-400">加载拉普拉斯系统状态中...</p>
                </div>
              ) : laplacianStatus ? (
                <div className="space-y-6">
                  {/* 状态卡片 */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className={`p-4 rounded-lg ${laplacianStatus.available ? 'bg-gradient-to-r from-gray-700 to-gray-600 dark:from-gray-700/30 dark:to-gray-600/30' : 'bg-gradient-to-r from-gray-800 to-gray-500 dark:from-gray-800/30 dark:to-gray-500/30'}`}>
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-gray-700 dark:text-gray-700">系统可用性</p>
                          <p className="text-2xl font-semibold text-gray-700 dark:text-gray-300">
                            {laplacianStatus.available ? '可用' : '不可用'}
                          </p>
                        </div>
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center ${laplacianStatus.available ? 'bg-gray-700 dark:bg-gray-700/50' : 'bg-gray-800 dark:bg-gray-800/50'}`}>
                          {laplacianStatus.available ? (
                            <CheckCircle className="w-6 h-6 text-gray-700 dark:text-gray-700" />
                          ) : (
                            <XCircle className="w-6 h-6 text-gray-800 dark:text-gray-800" />
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className={`p-4 rounded-lg ${laplacianStatus.initialized ? 'bg-gradient-to-r from-gray-600 to-gray-500 dark:from-gray-600/30 dark:to-gray-500/30' : 'bg-gradient-to-r from-gray-800 to-gray-500 dark:from-gray-800/30 dark:to-gray-500/30'}`}>
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-gray-600 dark:text-gray-600">初始化状态</p>
                          <p className="text-2xl font-semibold text-gray-600 dark:text-gray-300">
                            {laplacianStatus.initialized ? '已初始化' : '未初始化'}
                          </p>
                        </div>
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center ${laplacianStatus.initialized ? 'bg-gray-600 dark:bg-gray-600/50' : 'bg-gray-800 dark:bg-gray-800/50'}`}>
                          {laplacianStatus.initialized ? (
                            <CheckCircle className="w-6 h-6 text-gray-600 dark:text-gray-600" />
                          ) : (
                            <Settings className="w-6 h-6 text-gray-800 dark:text-gray-800" />
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="p-4 rounded-lg bg-gradient-to-r from-gray-600 to-gray-500 dark:from-gray-600/30 dark:to-gray-500/30">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-gray-600 dark:text-gray-600">增强模式</p>
                          <p className="text-2xl font-semibold text-gray-600 dark:text-gray-300">
                            {laplacianStatus.enhancement_mode || '未配置'}
                          </p>
                        </div>
                        <div className="w-12 h-12 rounded-full bg-gray-600 dark:bg-gray-600/50 flex items-center justify-center">
                          <Settings className="w-6 h-6 text-gray-600 dark:text-gray-600" />
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* 配置面板 */}
                  <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-6">
                    <h3 className="font-medium text-gray-900 dark:text-white mb-4">配置参数</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          增强模式
                        </label>
                        <select
                          value={laplacianConfig.enhancement_mode}
                          onChange={(e) => setLaplacianConfig(prev => ({ ...prev, enhancement_mode: e.target.value }))}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                        >
                          <option value="graph_laplacian">图拉普拉斯</option>
                          <option value="manifold_regularization">流形正则化</option>
                          <option value="spectral_clustering">谱聚类增强</option>
                          <option value="multi_scale">多尺度分析</option>
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          启用组件
                        </label>
                        <div className="space-y-2">
                          {['regularization', 'feature_extraction', 'graph_based', 'spectral_analysis'].map(component => (
                            <div key={component} className="flex items-center">
                              <input
                                type="checkbox"
                                id={`component-${component}`}
                                checked={laplacianConfig.enabled_components.includes(component)}
                                onChange={(e) => {
                                  if (e.target.checked) {
                                    setLaplacianConfig(prev => ({
                                      ...prev,
                                      enabled_components: [...prev.enabled_components, component]
                                    }));
                                  } else {
                                    setLaplacianConfig(prev => ({
                                      ...prev,
                                      enabled_components: prev.enabled_components.filter(c => c !== component)
                                    }));
                                  }
                                }}
                                className="h-4 w-4 text-gray-700 rounded focus:ring-gray-700"
                              />
                              <label htmlFor={`component-${component}`} className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                                {component === 'regularization' ? '拉普拉斯正则化' :
                                 component === 'feature_extraction' ? '特征提取增强' :
                                 component === 'graph_based' ? '图结构学习' :
                                 '谱分析优化'}
                              </label>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          正则化强度 (λ)
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="0.1"
                          step="0.001"
                          value={laplacianConfig.regularization_lambda}
                          onChange={(e) => setLaplacianConfig(prev => ({ ...prev, regularization_lambda: parseFloat(e.target.value) }))}
                          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                        <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400 mt-1">
                          <span>0</span>
                          <span>当前: {laplacianConfig.regularization_lambda.toFixed(3)}</span>
                          <span>0.1</span>
                        </div>
                      </div>
                      
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id="adaptive_lambda"
                          checked={laplacianConfig.adaptive_lambda}
                          onChange={(e) => setLaplacianConfig(prev => ({ ...prev, adaptive_lambda: e.target.checked }))}
                          className="h-4 w-4 text-gray-700 rounded focus:ring-gray-700"
                        />
                        <label htmlFor="adaptive_lambda" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                          自适应正则化强度
                        </label>
                      </div>
                    </div>
                    
                    <div className="mt-6 flex space-x-4">
                      <button
                        onClick={saveLaplacianConfig}
                        className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-700"
                        disabled={isLoadingLaplacian}
                      >
                        <Settings className="w-4 h-4 mr-2" />
                        保存配置
                      </button>
                      
                      <button
                        onClick={() => toggleLaplacianSystem(!laplacianStatus?.available)}
                        className={`inline-flex items-center px-4 py-2 text-sm font-medium rounded-lg ${laplacianStatus?.available ? 'text-gray-800 bg-gray-500 dark:bg-gray-800/30 dark:text-gray-500 hover:bg-gray-500 dark:hover:bg-gray-500/50' : 'text-gray-700 bg-gray-600 dark:bg-gray-700/30 dark:text-gray-600 hover:bg-gray-600 dark:hover:bg-gray-600/50'}`}
                        disabled={isLoadingLaplacian}
                      >
                        {laplacianStatus?.available ? (
                          <>
                            <XCircle className="w-4 h-4 mr-2" />
                            禁用系统
                          </>
                        ) : (
                          <>
                            <CheckCircle className="w-4 h-4 mr-2" />
                            启用系统
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                  
                  {/* 应用到训练任务 */}
                  <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                    <h3 className="font-medium text-gray-900 dark:text-white mb-4">应用到训练任务</h3>
                    <div className="space-y-4">
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        选择要应用拉普拉斯增强的训练任务：
                      </p>
                      <div className="space-y-3">
                        {jobs.filter(job => job.status === 'running' || job.status === 'pending').map(job => (
                          <div key={job.id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                            <div>
                              <p className="font-medium text-gray-900 dark:text-white">{job.name}</p>
                              <p className="text-sm text-gray-500 dark:text-gray-400">
                                {job.modelType} • {job.status === 'running' ? '运行中' : '等待中'}
                              </p>
                            </div>
                            <button
                              onClick={() => applyLaplacianEnhancement(job.id)}
                              className="inline-flex items-center px-3 py-1 text-sm font-medium text-gray-700 bg-gray-700 dark:bg-gray-700/30 dark:text-gray-700 rounded-md hover:bg-gray-700 dark:hover:bg-gray-700/50"
                              disabled={isLoadingLaplacian}
                            >
                              <Zap className="w-4 h-4 mr-2" />
                              应用增强
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <Settings className="w-12 h-12 mx-auto text-gray-400 mb-3" />
                  <h3 className="font-medium text-gray-900 dark:text-white">拉普拉斯增强系统未初始化</h3>
                  <p className="mt-1 text-gray-600 dark:text-gray-400">
                    系统可能未启动或配置，请检查后端服务状态。
                  </p>
                  <button
                    onClick={() => toggleLaplacianSystem(true)}
                    className="mt-4 inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600"
                  >
                    <CheckCircle className="w-4 h-4 mr-2" />
                    初始化系统
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 训练详情模态框 */}
      {showTrainingDetailModal && selectedTraining && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  训练任务详情 - {selectedTraining.name}
                </h2>
                <button
                  onClick={() => setShowTrainingDetailModal(false)}
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      基本信息
                    </h3>
                    <dl className="space-y-2">
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">任务ID</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedTraining.id}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">模型类型</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedTraining.modelType}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">状态</dt>
                        <dd className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                          selectedTraining.status === 'running' ? 'bg-gray-600 text-gray-600 dark:bg-gray-700/30 dark:text-gray-600' :
                          selectedTraining.status === 'pending' ? 'bg-gray-500 text-gray-500 dark:bg-gray-800/30 dark:text-gray-500' :
                          selectedTraining.status === 'paused' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400' :
                          selectedTraining.status === 'completed' ? 'bg-gray-700 text-gray-700 dark:bg-gray-700/30 dark:text-gray-700' :
                          'bg-gray-800 text-gray-800 dark:bg-gray-900/30 dark:text-gray-800'
                        }`}>
                          {selectedTraining.status === 'running' ? '运行中' :
                           selectedTraining.status === 'pending' ? '等待中' :
                           selectedTraining.status === 'paused' ? '已暂停' :
                           selectedTraining.status === 'completed' ? '已完成' : '失败'}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">开始时间</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedTraining.startTime.toLocaleString('zh-CN')}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">预计剩余时间</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedTraining.estimatedTime}
                        </dd>
                      </div>
                    </dl>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      训练进度
                    </h3>
                    <div className="space-y-4">
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            总体进度
                          </span>
                          <span className="text-sm text-gray-700 dark:text-gray-300">
                            {selectedTraining.progress.toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                          <div 
                            className="bg-gray-600 h-2.5 rounded-full" 
                            style={{ width: `${selectedTraining.progress}%` }}
                          ></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            GPU使用率
                          </span>
                          <span className="text-sm text-gray-700 dark:text-gray-300">
                            {selectedTraining.gpuUsage.toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                          <div 
                            className="bg-gray-700 h-2.5 rounded-full" 
                            style={{ width: `${selectedTraining.gpuUsage}%` }}
                          ></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            内存使用率
                          </span>
                          <span className="text-sm text-gray-700 dark:text-gray-300">
                            {selectedTraining.memoryUsage.toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                          <div 
                            className="bg-gray-600 h-2.5 rounded-full" 
                            style={{ width: `${selectedTraining.memoryUsage}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      训练参数
                    </h3>
                    <dl className="space-y-2">
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">数据集大小</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedTraining.datasetSize.toLocaleString('zh-CN')} 样本
                        </dd>
                      </div>
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">总轮次</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedTraining.epochs}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">当前轮次</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedTraining.currentEpoch} / {selectedTraining.epochs}
                        </dd>
                      </div>
                    </dl>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      配置信息
                    </h3>
                    <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 max-h-60 overflow-y-auto">
                      <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap break-words">
                        {JSON.stringify(selectedTraining.config, null, 2)}
                      </pre>
                    </div>
                  </div>
                </div>
                
                <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                  <div className="flex items-center justify-between">
                    <button
                      type="button"
                      onClick={() => setShowTrainingDetailModal(false)}
                      className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      关闭
                    </button>
                    
                    <div className="flex space-x-4">
                      {selectedTraining.status === 'running' && (
                        <button
                          type="button"
                          onClick={() => {
                            if (selectedTraining.id) {
                              handlePauseJob(selectedTraining.id);
                            }
                          }}
                          className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-800 bg-gray-500 dark:bg-gray-800/30 dark:text-gray-500 rounded-lg hover:bg-gray-500 dark:hover:bg-gray-500/50"
                        >
                          <Pause className="w-4 h-4 mr-2" />
                          暂停训练
                        </button>
                      )}
                      {selectedTraining.status === 'paused' && (
                        <button
                          type="button"
                          onClick={() => {
                            if (selectedTraining.id) {
                              handleStartJob(selectedTraining.id);
                            }
                          }}
                          className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-gray-600 dark:bg-gray-700/30 dark:text-gray-600 rounded-lg hover:bg-gray-600 dark:hover:bg-gray-600/50"
                        >
                          <Play className="w-4 h-4 mr-2" />
                          继续训练
                        </button>
                      )}
                      {(selectedTraining.status === 'completed' || selectedTraining.status === 'failed') && (
                        <button
                          type="button"
                          onClick={async () => {
                            try {
                              // 调用重新运行训练任务API
                              const response = await fetch(`/api/training/jobs/${selectedTraining.id}/rerun`, {
                                method: 'POST',
                                headers: {
                                  'Content-Type': 'application/json',
                                },
                              });
                              
                              if (response.ok) {
                                toast.success('训练任务已重新运行');
                                // 刷新训练任务列表
                                loadJobs();
                              } else {
                                const errorData = await response.json();
                                toast.error(`重新运行失败: ${errorData.detail || '未知错误'}`);
                              }
                            } catch (error) {
                              console.error('重新运行训练任务失败:', error);
                              toast.error('重新运行失败，请检查网络连接');
                            }
                          }}
                          className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-gray-700 dark:bg-gray-700/30 dark:text-gray-700 rounded-lg hover:bg-gray-700 dark:hover:bg-gray-700/50"
                        >
                          <RefreshCw className="w-4 h-4 mr-2" />
                          重新运行
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </ResponsiveContainer>
  );
};

export default TrainingPage;