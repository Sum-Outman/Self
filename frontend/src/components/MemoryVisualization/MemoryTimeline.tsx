import React, { useEffect, useState } from 'react';
import { memoryApi, MemoryItem } from '../../services/api/memory';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

interface TimelineEvent {
  id: string;
  title: string;
  description: string;
  timestamp: Date;
  type: 'short_term' | 'long_term' | 'working';
  importance: number;
  source: string;
  scene_type?: string;
}

interface MemoryTimelineProps {
  days?: number; // 显示最近多少天的记忆
  height?: number;
}

const MemoryTimeline: React.FC<MemoryTimelineProps> = ({
  days = 7,
  height = 500,
}) => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);

  // 获取记忆数据并转换为时间线事件
  useEffect(() => {
    const fetchMemoryData = async () => {
      try {
        setLoading(true);
        setError(null);

        // 获取最近记忆，数量根据天数估算
        const estimatedCount = days * 10; // 每天大约10个记忆
        const response = await memoryApi.getRecentMemories(estimatedCount);
        
        if (response.success && response.data) {
          const memories = response.data;
          
          // 转换为时间线事件
          const timelineEvents: TimelineEvent[] = memories.map((memory: MemoryItem) => {
            const timestamp = new Date(memory.created_at);
            
            // 创建标题和描述
            const title = memory.content.substring(0, 40) + 
                         (memory.content.length > 40 ? '...' : '');
            
            let description = '';
            if (memory.scene_type) {
              const sceneMap: Record<string, string> = {
                'task': '任务',
                'learning': '学习',
                'problem_solving': '问题解决',
                'social': '社交',
                'planning': '规划'
              };
              description += `场景: ${sceneMap[memory.scene_type] || memory.scene_type}`;
            }
            
            description += ` | 重要性: ${(memory.importance * 100).toFixed(0)}%`;
            description += ` | 来源: ${memory.source === 'user' ? '用户' : memory.source === 'system' ? '系统' : '自主'}`;

            return {
              id: memory.id,
              title,
              description,
              timestamp,
              type: memory.type,
              importance: memory.importance,
              source: memory.source,
              scene_type: memory.scene_type,
            };
          });

          // 按时间排序（最新的在前）
          timelineEvents.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

          setEvents(timelineEvents);
        } else {
          throw new Error(response.message || '获取记忆数据失败');
        }
      } catch (err: any) {
        console.error('获取记忆时间线数据失败:', err);
        setError(err.message || '加载记忆时间线失败');
        
        // 提供示例数据用于演示
        const now = new Date();
        const sampleEvents: TimelineEvent[] = [
          {
            id: '1',
            title: '完成AGI系统训练任务',
            description: '场景: 任务 | 重要性: 90% | 来源: 系统',
            timestamp: new Date(now.getTime() - 2 * 60 * 60 * 1000), // 2小时前
            type: 'short_term',
            importance: 0.9,
            source: 'system',
            scene_type: 'task',
          },
          {
            id: '2',
            title: '学习新的神经网络架构',
            description: '场景: 学习 | 重要性: 85% | 来源: 自主',
            timestamp: new Date(now.getTime() - 6 * 60 * 60 * 1000), // 6小时前
            type: 'long_term',
            importance: 0.85,
            source: 'autonomous',
            scene_type: 'learning',
          },
          {
            id: '3',
            title: '解决内存优化问题',
            description: '场景: 问题解决 | 重要性: 80% | 来源: 系统',
            timestamp: new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000), // 1天前
            type: 'working',
            importance: 0.8,
            source: 'system',
            scene_type: 'problem_solving',
          },
          {
            id: '4',
            title: '用户交互会话记录',
            description: '场景: 社交 | 重要性: 70% | 来源: 用户',
            timestamp: new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000), // 2天前
            type: 'short_term',
            importance: 0.7,
            source: 'user',
            scene_type: 'social',
          },
          {
            id: '5',
            title: '长期知识整合',
            description: '场景: 规划 | 重要性: 95% | 来源: 自主',
            timestamp: new Date(now.getTime() - 3 * 24 * 60 * 60 * 1000), // 3天前
            type: 'long_term',
            importance: 0.95,
            source: 'autonomous',
            scene_type: 'planning',
          },
          {
            id: '6',
            title: '系统性能监控',
            description: '场景: 任务 | 重要性: 75% | 来源: 系统',
            timestamp: new Date(now.getTime() - 4 * 24 * 60 * 60 * 1000), // 4天前
            type: 'working',
            importance: 0.75,
            source: 'system',
            scene_type: 'task',
          },
          {
            id: '7',
            title: '新算法实验',
            description: '场景: 学习 | 重要性: 88% | 来源: 自主',
            timestamp: new Date(now.getTime() - 5 * 24 * 60 * 60 * 1000), // 5天前
            type: 'long_term',
            importance: 0.88,
            source: 'autonomous',
            scene_type: 'learning',
          },
        ];

        setEvents(sampleEvents);
      } finally {
        setLoading(false);
      }
    };

    fetchMemoryData();
  }, [days]);

  // 格式化时间显示
  const formatTime = (date: Date): string => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMinutes = Math.floor(diffMs / (1000 * 60));

    if (diffDays > 0) {
      return `${diffDays}天前`;
    } else if (diffHours > 0) {
      return `${diffHours}小时前`;
    } else if (diffMinutes > 0) {
      return `${diffMinutes}分钟前`;
    } else {
      return '刚刚';
    }
  };

  // 获取事件类型颜色
  const getEventTypeColor = (type: string): string => {
    switch (type) {
      case 'short_term':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'long_term':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'working':
        return 'bg-orange-100 text-orange-800 border-orange-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  // 获取事件类型文本
  const getEventTypeText = (type: string): string => {
    switch (type) {
      case 'short_term':
        return '短期';
      case 'long_term':
        return '长期';
      case 'working':
        return '工作';
      default:
        return '未知';
    }
  };

  // 获取重要性颜色
  const getImportanceColor = (importance: number): string => {
    if (importance >= 0.8) return 'bg-red-500';
    if (importance >= 0.6) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  // 处理事件点击
  const handleEventClick = (event: TimelineEvent) => {
    setSelectedEvent(event);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-gray-900"></div>
          <p className="mt-2 text-gray-600">加载记忆时间线...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-red-600">
          <p>加载记忆时间线时出错: {error}</p>
          <p className="text-sm text-gray-600 mt-2">显示示例数据以供演示</p>
        </div>
      </div>
    );
  }

  return (
    <div className="memory-timeline">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900">记忆时间线</h3>
        <p className="text-sm text-gray-600">
          按时间顺序显示记忆事件。最新记忆显示在顶部。
        </p>
        <div className="flex flex-wrap gap-4 mt-2">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-blue-500 mr-2"></div>
            <span className="text-xs">短期记忆</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
            <span className="text-xs">长期记忆</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-orange-500 mr-2"></div>
            <span className="text-xs">工作记忆</span>
          </div>
        </div>
      </div>

      <div 
        style={{ 
          height: `${height}px`, 
          overflowY: 'auto',
          border: '1px solid #e5e7eb',
          borderRadius: '0.5rem',
          padding: '1rem',
        }}
      >
        <div className="relative">
          {/* 时间线轴线 */}
          <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-300"></div>

          {/* 时间线事件 */}
          {events.map((event, index) => (
            <div key={event.id} className="relative mb-6 pl-10">
              {/* 时间点标记 */}
              <div className="absolute left-0 top-0 w-8 h-8 flex items-center justify-center">
                <div 
                  className={`w-4 h-4 rounded-full border-2 ${
                    selectedEvent?.id === event.id 
                      ? 'border-gray-900 bg-gray-900' 
                      : getEventTypeColor(event.type).replace('bg-', 'bg-').split(' ')[0]
                  }`}
                ></div>
              </div>

              {/* 事件卡片 */}
              <div 
                className={`p-4 rounded-lg border cursor-pointer transition-all hover:shadow-md ${
                  selectedEvent?.id === event.id 
                    ? 'border-gray-900 bg-gray-50' 
                    : 'border-gray-200 bg-white'
                }`}
                onClick={() => handleEventClick(event)}
              >
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-medium text-gray-900">{event.title}</h4>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 text-xs rounded-full ${getEventTypeColor(event.type)}`}>
                      {getEventTypeText(event.type)}记忆
                    </span>
                    <div className="flex items-center">
                      <div className="text-xs text-gray-500 mr-1">重要性:</div>
                      <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${getImportanceColor(event.importance)}`}
                          style={{ width: `${event.importance * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>

                <p className="text-sm text-gray-600 mb-2">{event.description}</p>

                <div className="flex justify-between items-center text-xs text-gray-500">
                  <div className="flex items-center">
                    <span className="mr-3">{format(event.timestamp, 'yyyy-MM-dd HH:mm', { locale: zhCN })}</span>
                    <span>({formatTime(event.timestamp)})</span>
                  </div>
                  <div>
                    {event.scene_type && (
                      <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded">
                        {event.scene_type === 'task' ? '任务' :
                         event.scene_type === 'learning' ? '学习' :
                         event.scene_type === 'problem_solving' ? '问题解决' :
                         event.scene_type === 'social' ? '社交' :
                         event.scene_type === 'planning' ? '规划' : event.scene_type}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* 时间线连接线 */}
              {index < events.length - 1 && (
                <div className="absolute left-4 top-8 bottom-0 w-0.5 bg-gray-200"></div>
              )}
            </div>
          ))}

          {events.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <p>暂无记忆数据</p>
              <p className="text-sm mt-1">系统尚未记录任何记忆事件</p>
            </div>
          )}
        </div>
      </div>

      {/* 选中事件详情 */}
      {selectedEvent && (
        <div className="mt-4 p-4 border border-gray-200 rounded-lg bg-gray-50">
          <h4 className="font-medium text-gray-900 mb-2">记忆详情</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-600">ID:</p>
              <p className="font-medium">{selectedEvent.id}</p>
            </div>
            <div>
              <p className="text-gray-600">类型:</p>
              <p className="font-medium">{getEventTypeText(selectedEvent.type)}记忆</p>
            </div>
            <div>
              <p className="text-gray-600">重要性:</p>
              <p className="font-medium">{(selectedEvent.importance * 100).toFixed(0)}%</p>
            </div>
            <div>
              <p className="text-gray-600">来源:</p>
              <p className="font-medium">
                {selectedEvent.source === 'user' ? '用户' : 
                 selectedEvent.source === 'system' ? '系统' : '自主'}
              </p>
            </div>
            <div className="col-span-2">
              <p className="text-gray-600">时间:</p>
              <p className="font-medium">
                {format(selectedEvent.timestamp, 'yyyy年MM月dd日 HH:mm:ss', { locale: zhCN })} 
                <span className="text-gray-500 ml-2">({formatTime(selectedEvent.timestamp)})</span>
              </p>
            </div>
            <div className="col-span-2">
              <p className="text-gray-600">完整内容:</p>
              <p className="mt-1 p-2 bg-white border border-gray-300 rounded">
                {selectedEvent.title}
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="mt-4 text-sm text-gray-600">
        <p>
          共显示 <span className="font-semibold">{events.length}</span> 个记忆事件，
          时间范围: 最近 <span className="font-semibold">{days}</span> 天。
        </p>
        <p className="mt-1">
          提示: 点击事件卡片查看详情，时间线可滚动查看。
        </p>
      </div>
    </div>
  );
};

export default MemoryTimeline;