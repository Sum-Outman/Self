import React, { useState, useEffect } from 'react';
import { Card, Button, Loading } from '../UI';
import { getAGIStatus, startAGITraining, stopAGITraining, changeAGIMode, AGIStatus } from '../../services/api/agiService';

interface AGIPanelProps {
  className?: string;
}

const AGIPanel: React.FC<AGIPanelProps> = ({ className = '' }) => {
  const [status, setStatus] = useState<AGIStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');

  // 状态颜色映射 - 纯黑白灰
  const statusColors: Record<string, string> = {
    idle: 'bg-gray-200 text-gray-800',
    training: 'bg-gray-300 text-gray-800',
    reasoning: 'bg-gray-400 text-gray-800',
    learning: 'bg-gray-500 text-gray-800',
    paused: 'bg-gray-600 text-gray-100',
    error: 'bg-gray-800 text-gray-100'
  };

  const modeLabels: Record<string, string> = {
    autonomous: '全自主模式',
    task: '任务执行模式',
    demo: '演示模式'
  };

  // 获取AGI状态
  const fetchStatus = async () => {
    try {
      setLoading(true);
      const data = await getAGIStatus();
      setStatus(data);
      setError('');
    } catch (err) {
      setError(`获取AGI状态失败: ${err instanceof Error ? err.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  // 启动训练
  const handleStartTraining = async () => {
    try {
      setLoading(true);
      await startAGITraining();
      setTimeout(fetchStatus, 1000); // 1秒后刷新状态
    } catch (err) {
      setError(`启动训练失败: ${err instanceof Error ? err.message : '未知错误'}`);
      setLoading(false);
    }
  };

  // 停止训练
  const handleStopTraining = async () => {
    try {
      setLoading(true);
      await stopAGITraining();
      setTimeout(fetchStatus, 1000);
    } catch (err) {
      setError(`停止训练失败: ${err instanceof Error ? err.message : '未知错误'}`);
      setLoading(false);
    }
  };

  // 切换模式
  const handleModeChange = async (mode: 'autonomous' | 'task') => {
    try {
      setLoading(true);
      // 调用切换模式的API
      const response = await changeAGIMode({ mode });
      if (response.success) {
        // API调用成功，更新本地状态
        setStatus(prev => prev ? { ...prev, mode } : null);
        setTimeout(fetchStatus, 1000);
      } else {
        throw new Error(response.message || '切换模式失败');
      }
    } catch (err) {
      setError(`切换模式失败: ${err instanceof Error ? err.message : '未知错误'}`);
      setLoading(false);
    }
  };

  // 初始加载和定期刷新
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // 每5秒刷新一次
    return () => clearInterval(interval);
  }, []);

  if (loading && !status) {
    return (
      <div className={`flex items-center justify-center h-64 ${className}`}>
        <Loading text="加载AGI状态..." />
      </div>
    );
  }

  if (error && !status) {
    return (
      <div className={`p-4 ${className}`}>
        <Card className="bg-red-50 border-red-200">
          <div className="text-red-700 font-medium">错误</div>
          <div className="text-red-600 mt-1">{error}</div>
          <Button 
            onClick={fetchStatus} 
            variant="secondary" 
            className="mt-3"
            size="sm"
          >
            重试
          </Button>
        </Card>
      </div>
    );
  }

  if (!status) {
    return null;
  }

  return (
    <div className={`space-y-4 ${className}`}>
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">AGI控制面板</h2>
          <div className="flex items-center space-x-2">
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${statusColors[status.status]}`}>
              {status.status === 'idle' ? '空闲' :
               status.status === 'training' ? '训练中' :
               status.status === 'reasoning' ? '推理中' :
               status.status === 'learning' ? '学习中' :
               status.status === 'paused' ? '已暂停' : '错误'}
            </span>
            <span className="px-3 py-1 bg-gray-100 rounded-full text-sm font-medium text-gray-700">
              {modeLabels[status.mode]}
            </span>
          </div>
        </div>

        {/* 状态信息 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600 mb-1">训练进度</div>
            <div className="flex items-center">
              <div className="w-full bg-gray-200 rounded-full h-2 mr-2">
                <div 
                  className="bg-gray-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${status.trainingProgress}%` }}
                />
              </div>
              <span className="text-sm font-medium">{status.trainingProgress.toFixed(1)}%</span>
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600 mb-1">推理深度</div>
            <div className="text-xl font-bold text-gray-900">{status.reasoningDepth}</div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600 mb-1">内存使用</div>
            <div className="text-xl font-bold text-gray-900">{status.memoryUsage.toFixed(1)} GB</div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600 mb-1">硬件连接</div>
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${status.hardwareConnected ? 'bg-gray-600' : 'bg-gray-400'}`} />
              <span className="text-sm font-medium">
                {status.hardwareConnected ? '已连接' : '未连接'}
              </span>
            </div>
          </div>
        </div>

        {/* 控制按钮 */}
        <div className="flex flex-wrap gap-3 mb-4">
          <Button
            onClick={handleStartTraining}
            disabled={status.status === 'training' || loading}
            variant="primary"
            className="min-w-[120px]"
          >
            {loading && status.status === 'training' ? '启动中...' : '开始训练'}
          </Button>

          <Button
            onClick={handleStopTraining}
            disabled={status.status !== 'training' || loading}
            variant="secondary"
            className="min-w-[120px]"
          >
            停止训练
          </Button>

          <Button
            onClick={() => handleModeChange('autonomous')}
            disabled={status.mode === 'autonomous' || loading}
            variant={status.mode === 'autonomous' ? 'primary' : 'secondary'}
            className="min-w-[120px]"
          >
            全自主模式
          </Button>

          <Button
            onClick={() => handleModeChange('task')}
            disabled={status.mode === 'task' || loading}
            variant={status.mode === 'task' ? 'primary' : 'secondary'}
            className="min-w-[120px]"
          >
            任务模式
          </Button>

          <Button
            onClick={fetchStatus}
            variant="outline"
            className="min-w-[100px]"
          >
            刷新状态
          </Button>
        </div>

        {/* 状态详情 */}
        <div className="border-t pt-4">
          <div className="text-sm text-gray-600 mb-2">状态详情</div>
          <div className="text-sm">
            <div className="flex justify-between py-1">
              <span className="text-gray-700">最后更新时间:</span>
              <span className="font-medium">{new Date(status.lastUpdated).toLocaleString()}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-gray-700">当前模式:</span>
              <span className="font-medium">{modeLabels[status.mode]}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-gray-700">系统状态:</span>
              <span className="font-medium">
                {status.status === 'idle' ? '系统空闲，等待指令' :
                 status.status === 'training' ? '正在进行模型训练' :
                 status.status === 'reasoning' ? '正在进行逻辑推理' :
                 status.status === 'learning' ? '正在进行知识学习' :
                 status.status === 'paused' ? '系统已暂停' : '系统遇到错误'}
              </span>
            </div>
          </div>
        </div>
      </Card>

      {error && (
        <div className="text-red-600 text-sm bg-red-50 p-3 rounded-md">
          <strong>注意:</strong> {error}
        </div>
      )}

      {loading && (
        <div className="text-center py-4">
          <Loading text="处理中..." size="sm" />
        </div>
      )}
    </div>
  );
};

export default AGIPanel;