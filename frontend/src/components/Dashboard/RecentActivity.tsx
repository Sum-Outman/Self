import React from 'react';
import { MessageSquare, Brain, Settings, User, Zap, Clock } from 'lucide-react';

interface ActivityItem {
  id: string;
  type: 'chat' | 'training' | 'system' | 'user' | 'api';
  user: string;
  action: string;
  timestamp: string;
  details?: string;
}

const RecentActivity: React.FC = () => {
  const activities: ActivityItem[] = [
    {
      id: '1',
      type: 'chat',
      user: '张三',
      action: '与AI进行了深度对话',
      timestamp: '5分钟前',
      details: '讨论了人工智能的伦理问题',
    },
    {
      id: '2',
      type: 'training',
      user: '系统',
      action: '完成模型训练任务',
      timestamp: '15分钟前',
      details: 'Transformer模型训练完成，准确率92.3%',
    },
    {
      id: '3',
      type: 'api',
      user: '李四',
      action: '创建了新的API密钥',
      timestamp: '30分钟前',
      details: '用于生产环境API调用',
    },
    {
      id: '4',
      type: 'system',
      user: '管理员',
      action: '更新了系统配置',
      timestamp: '1小时前',
      details: '调整了API速率限制',
    },
    {
      id: '5',
      type: 'user',
      user: '王五',
      action: '注册了新账户',
      timestamp: '2小时前',
    },
    {
      id: '6',
      type: 'chat',
      user: '赵六',
      action: '使用了多模态对话',
      timestamp: '3小时前',
      details: '上传并分析了图像文件',
    },
  ];

  const getActivityIcon = (type: ActivityItem['type']) => {
    switch (type) {
      case 'chat':
        return <MessageSquare className="h-5 w-5 text-gray-700" />;
      case 'training':
        return <Brain className="h-5 w-5 text-gray-600" />;
      case 'system':
        return <Settings className="h-5 w-5 text-gray-500" />;
      case 'user':
        return <User className="h-5 w-5 text-gray-600" />;
      case 'api':
        return <Zap className="h-5 w-5 text-gray-600" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" />;
    }
  };

  const getActivityColor = (type: ActivityItem['type']) => {
    switch (type) {
      case 'chat':
        return 'bg-gray-600 dark:bg-gray-900';
      case 'training':
        return 'bg-gray-600 dark:bg-gray-900';
      case 'system':
        return 'bg-gray-100 dark:bg-gray-900';
      case 'user':
        return 'bg-gray-600 dark:bg-gray-900';
      case 'api':
        return 'bg-gray-700 dark:bg-gray-900';
      default:
        return 'bg-gray-100 dark:bg-gray-900';
    }
  };

  return (
    <div className="flow-root">
      <ul className="-mb-8">
        {activities.map((activity, activityIdx) => (
          <li key={activity.id}>
            <div className="relative pb-8">
              {activityIdx !== activities.length - 1 ? (
                <span
                  className="absolute top-5 left-5 -ml-px h-full w-0.5 bg-gray-200 dark:bg-gray-700"
                  aria-hidden="true"
                />
              ) : null}
              <div className="relative flex items-start space-x-3">
                <div className="relative">
                  <div
                    className={`h-10 w-10 rounded-full flex items-center justify-center ring-8 ring-white dark:ring-gray-800 ${getActivityColor(
                      activity.type
                    )}`}
                  >
                    {getActivityIcon(activity.type)}
                  </div>
                </div>
                <div className="min-w-0 flex-1">
                  <div>
                    <div className="text-sm">
                      <span className="font-medium text-gray-900 dark:text-white">
                        {activity.user}
                      </span>
                      <span className="text-gray-500 dark:text-gray-400">
                        {' '}
                        {activity.action}
                      </span>
                    </div>
                    <div className="mt-0.5 flex items-center text-xs text-gray-500 dark:text-gray-400">
                      <Clock className="flex-shrink-0 mr-1 h-3 w-3" />
                      {activity.timestamp}
                    </div>
                  </div>
                  {activity.details && (
                    <div className="mt-2 text-sm text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-800 p-3 rounded-lg">
                      {activity.details}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </li>
        ))}
      </ul>
      <div className="mt-6 text-center">
        <button
          type="button"
          className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 shadow-sm text-sm font-medium rounded-md text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
        >
          查看全部活动
        </button>
      </div>
    </div>
  );
};

export default RecentActivity;