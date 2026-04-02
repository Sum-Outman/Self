import React from 'react';
import { TrendingUp, Cpu, Clock, CheckCircle } from 'lucide-react';

interface TrainingJob {
  id: string;
  name: string;
  progress: number;
  status: 'running' | 'completed' | 'failed' | 'pending';
  eta: string;
  accuracy?: number;
}

const ModelTrainingProgress: React.FC = () => {
  const trainingJobs: TrainingJob[] = [
    {
      id: '1',
      name: 'Transformer基础模型',
      progress: 85,
      status: 'running',
      eta: '2小时30分钟',
      accuracy: 92.3,
    },
    {
      id: '2',
      name: '多模态特征提取',
      progress: 45,
      status: 'running',
      eta: '5小时15分钟',
      accuracy: 78.5,
    },
    {
      id: '3',
      name: '视觉语言模型',
      progress: 100,
      status: 'completed',
      eta: '已完成',
      accuracy: 94.7,
    },
    {
      id: '4',
      name: '机器人控制模型',
      progress: 20,
      status: 'running',
      eta: '12小时',
    },
  ];

  const getStatusColor = (status: TrainingJob['status']) => {
    switch (status) {
      case 'running':
        return 'bg-gray-600 text-gray-900 dark:bg-gray-900 dark:text-gray-400';
      case 'completed':
        return 'bg-gray-600 text-gray-800 dark:bg-gray-900 dark:text-gray-400';
      case 'failed':
        return 'bg-gray-800 text-gray-900 dark:bg-gray-900 dark:text-gray-600';
      case 'pending':
        return 'bg-gray-700 text-gray-900 dark:bg-gray-900 dark:text-gray-500';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300';
    }
  };

  const getStatusText = (status: TrainingJob['status']) => {
    switch (status) {
      case 'running': return '训练中';
      case 'completed': return '已完成';
      case 'failed': return '失败';
      case 'pending': return '等待中';
      default: return '未知';
    }
  };

  return (
    <div className="space-y-4">
      {trainingJobs.map((job) => (
        <div key={job.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center">
              <Cpu className="h-5 w-5 text-gray-400 mr-2" />
              <h3 className="font-medium text-gray-900 dark:text-white">
                {job.name}
              </h3>
            </div>
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                job.status
              )}`}
            >
              {job.status === 'completed' && <CheckCircle className="h-3 w-3 mr-1" />}
              {getStatusText(job.status)}
            </span>
          </div>

          {/* 进度条 */}
          <div className="mb-3">
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-gray-600 dark:text-gray-400">进度</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {job.progress}%
              </span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${
                  job.status === 'completed'
                    ? 'bg-gray-600'
                    : job.status === 'failed'
                    ? 'bg-gray-800'
                    : 'bg-gray-700'
                }`}
                style={{ width: `${job.progress}%` }}
              />
            </div>
          </div>

          {/* 详细信息 */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="flex items-center text-gray-500 dark:text-gray-400 mb-1">
                <Clock className="h-4 w-4 mr-1" />
                <span>预计完成</span>
              </div>
              <div className="font-medium text-gray-900 dark:text-white">
                {job.eta}
              </div>
            </div>
            {job.accuracy !== undefined && (
              <div>
                <div className="flex items-center text-gray-500 dark:text-gray-400 mb-1">
                  <TrendingUp className="h-4 w-4 mr-1" />
                  <span>准确率</span>
                </div>
                <div className="font-medium text-gray-900 dark:text-white">
                  {job.accuracy}%
                </div>
              </div>
            )}
          </div>
        </div>
      ))}

      {/* 统计信息 */}
      <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              4
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              总任务数
            </div>
          </div>
          <div>
            <div className="text-2xl font-bold text-gray-700 dark:text-gray-400">
              1
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              已完成
            </div>
          </div>
          <div>
            <div className="text-2xl font-bold text-gray-800 dark:text-gray-400">
              3
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              进行中
            </div>
          </div>
        </div>
      </div>

      {/* 操作按钮 */}
      <div className="flex space-x-2">
        <button
          type="button"
          className="flex-1 py-2 px-4 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
        >
          查看所有任务
        </button>
        <button
          type="button"
          className="flex-1 py-2 px-4 bg-gray-600 border border-transparent rounded-lg text-sm font-medium text-white hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
        >
          新建训练
        </button>
      </div>
    </div>
  );
};

export default ModelTrainingProgress;