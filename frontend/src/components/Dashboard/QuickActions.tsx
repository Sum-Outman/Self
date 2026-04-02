import React from 'react';
import { MessageSquare, Play, Settings, Download, Upload, Plus, FileText, Video } from 'lucide-react';

interface QuickAction {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  action: () => void;
  color: string;
}

const QuickActions: React.FC = () => {
  const actions: QuickAction[] = [
    {
      id: '1',
      name: '开始对话',
      description: '与AI进行对话',
      icon: <MessageSquare className="h-5 w-5" />,
      action: () => window.location.href = '/chat',
      color: 'bg-gray-600 text-gray-800 dark:bg-gray-900 dark:text-gray-400',
    },
    {
      id: '2',
      name: '启动训练',
      description: '开始模型训练',
      icon: <Play className="h-5 w-5" />,
      action: () => window.location.href = '/training',
      color: 'bg-gray-600 text-gray-700 dark:bg-gray-900 dark:text-gray-400',
    },
    {
      id: '3',
      name: '硬件连接',
      description: '连接机器人设备',
      icon: <Settings className="h-5 w-5" />,
      action: () => window.location.href = '/hardware',
      color: 'bg-gray-600 text-gray-700 dark:bg-gray-900 dark:text-gray-400',
    },
    {
      id: '4',
      name: '导入知识',
      description: '上传知识库文件',
      icon: <Upload className="h-5 w-5" />,
      action: () => window.location.href = '/knowledge/import',
      color: 'bg-gray-700 text-gray-700 dark:bg-gray-900 dark:text-gray-400',
    },
    {
      id: '5',
      name: '新建任务',
      description: '创建AI任务',
      icon: <Plus className="h-5 w-5" />,
      action: () => window.location.href = '/tasks/new',
      color: 'bg-gray-800 text-gray-900 dark:bg-gray-900 dark:text-gray-500',
    },
    {
      id: '6',
      name: '生成报告',
      description: '创建系统报告',
      icon: <FileText className="h-5 w-5" />,
      action: () => window.location.href = '/reports',
      color: 'bg-gray-700 text-gray-800 dark:bg-gray-900 dark:text-gray-400',
    },
    {
      id: '7',
      name: '视频分析',
      description: '分析视频内容',
      icon: <Video className="h-5 w-5" />,
      action: () => window.location.href = '/video',
      color: 'bg-gray-600 text-gray-700 dark:bg-gray-900 dark:text-gray-400',
    },
    {
      id: '8',
      name: '下载模型',
      description: '下载预训练模型',
      icon: <Download className="h-5 w-5" />,
      action: () => window.location.href = '/models',
      color: 'bg-gray-100 text-gray-600 dark:bg-gray-900 dark:text-gray-400',
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3">
      {actions.map((action) => (
        <button
          key={action.id}
          onClick={action.action}
          className="flex flex-col items-center p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
        >
          <div className={`p-2 rounded-lg ${action.color} mb-2`}>
            {action.icon}
          </div>
          <div className="text-sm font-medium text-gray-900 dark:text-white text-center">
            {action.name}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 text-center mt-1">
            {action.description}
          </div>
        </button>
      ))}
    </div>
  );
};

export default QuickActions;