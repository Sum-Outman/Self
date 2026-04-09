import React, { useEffect, useState } from 'react';

// 冲突类型定义
type ConflictType = 'content' | 'temporal' | 'factual' | 'priority' | 'relational';

// 冲突解决策略定义
type ResolutionStrategy = 'merge' | 'select' | 'dual_version' | 'contextual';

// 冲突项接口
interface ConflictItem {
  id: string;
  type: ConflictType;
  description: string;
  memoryIds: string[]; // 涉及的记忆ID
  confidence: number; // 冲突置信度
  resolutionStrategy?: ResolutionStrategy;
  resolved: boolean;
  timestamp: Date;
}

// 记忆项接口（简化）
interface MemoryItem {
  id: string;
  content: string;
  type: string;
  importance: number;
}

interface ConflictResolutionVisualizationProps {
  height?: number;
}

const ConflictResolutionVisualization: React.FC<ConflictResolutionVisualizationProps> = ({
  height = 600,
}) => {
  const [conflicts, setConflicts] = useState<ConflictItem[]>([]);
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedConflict, setSelectedConflict] = useState<ConflictItem | null>(null);
  const [resolutionInProgress, setResolutionInProgress] = useState(false);

  // 初始化数据
  useEffect(() => {
    const initData = async () => {
      setLoading(true);
      
      // 模拟加载延迟
      await new Promise(resolve => setTimeout(resolve, 1000));

      // 示例记忆数据
      const sampleMemories: MemoryItem[] = [
        { id: 'mem1', content: '用户说喜欢蓝色界面', type: 'short_term', importance: 0.7 },
        { id: 'mem2', content: '用户偏好暗色主题', type: 'long_term', importance: 0.9 },
        { id: 'mem3', content: '系统应使用响应式设计', type: 'working', importance: 0.8 },
        { id: 'mem4', content: '用户要求简化操作流程', type: 'short_term', importance: 0.6 },
        { id: 'mem5', content: '用户希望增加自定义功能', type: 'long_term', importance: 0.85 },
      ];

      // 示例冲突数据
      const sampleConflicts: ConflictItem[] = [
        {
          id: 'conf1',
          type: 'content',
          description: '关于用户界面偏好的冲突：蓝色界面 vs 暗色主题',
          memoryIds: ['mem1', 'mem2'],
          confidence: 0.8,
          resolutionStrategy: 'contextual',
          resolved: true,
          timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2小时前
        },
        {
          id: 'conf2',
          type: 'priority',
          description: '功能优先级冲突：简化操作 vs 增加自定义',
          memoryIds: ['mem4', 'mem5'],
          confidence: 0.6,
          resolutionStrategy: 'merge',
          resolved: false,
          timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000), // 1小时前
        },
        {
          id: 'conf3',
          type: 'factual',
          description: '系统设计原则冲突',
          memoryIds: ['mem3', 'mem5'],
          confidence: 0.7,
          resolutionStrategy: 'select',
          resolved: true,
          timestamp: new Date(Date.now() - 30 * 60 * 1000), // 30分钟前
        },
        {
          id: 'conf4',
          type: 'temporal',
          description: '时间顺序冲突：短期记忆与长期记忆不一致',
          memoryIds: ['mem1', 'mem2'],
          confidence: 0.5,
          resolutionStrategy: 'dual_version',
          resolved: false,
          timestamp: new Date(Date.now() - 15 * 60 * 1000), // 15分钟前
        },
        {
          id: 'conf5',
          type: 'relational',
          description: '记忆关联关系冲突',
          memoryIds: ['mem2', 'mem3', 'mem4'],
          confidence: 0.4,
          resolutionStrategy: undefined,
          resolved: false,
          timestamp: new Date(Date.now() - 5 * 60 * 1000), // 5分钟前
        },
      ];

      setMemories(sampleMemories);
      setConflicts(sampleConflicts);
      setLoading(false);
    };

    initData();
  }, []);

  // 获取冲突类型文本
  const getConflictTypeText = (type: ConflictType): string => {
    const typeMap: Record<ConflictType, string> = {
      content: '内容冲突',
      temporal: '时间冲突',
      factual: '事实冲突',
      priority: '优先级冲突',
      relational: '关联冲突',
    };
    return typeMap[type];
  };

  // 获取冲突类型颜色
  const getConflictTypeColor = (type: ConflictType): string => {
    const colorMap: Record<ConflictType, string> = {
      content: 'bg-red-100 text-red-800 border-red-300',
      temporal: 'bg-blue-100 text-blue-800 border-blue-300',
      factual: 'bg-yellow-100 text-yellow-800 border-yellow-300',
      priority: 'bg-purple-100 text-purple-800 border-purple-300',
      relational: 'bg-green-100 text-green-800 border-green-300',
    };
    return colorMap[type];
  };

  // 获取解决策略文本
  const getStrategyText = (strategy?: ResolutionStrategy): string => {
    if (!strategy) return '未确定';
    
    const strategyMap: Record<ResolutionStrategy, string> = {
      merge: '合并策略',
      select: '选择策略',
      dual_version: '双版本策略',
      contextual: '上下文策略',
    };
    return strategyMap[strategy];
  };

  // 获取解决策略颜色
  const getStrategyColor = (strategy?: ResolutionStrategy): string => {
    if (!strategy) return 'bg-gray-100 text-gray-800';
    
    const colorMap: Record<ResolutionStrategy, string> = {
      merge: 'bg-blue-100 text-blue-800',
      select: 'bg-green-100 text-green-800',
      dual_version: 'bg-purple-100 text-purple-800',
      contextual: 'bg-yellow-100 text-yellow-800',
    };
    return colorMap[strategy];
  };

  // 获取置信度颜色
  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return 'bg-red-500';
    if (confidence >= 0.6) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  // 获取状态文本和颜色
  const getStatusInfo = (resolved: boolean) => {
    return resolved
      ? { text: '已解决', color: 'bg-green-100 text-green-800 border-green-300' }
      : { text: '待解决', color: 'bg-red-100 text-red-800 border-red-300' };
  };

  // 处理解决冲突
  const handleResolveConflict = (conflictId: string) => {
    setResolutionInProgress(true);
    
    // 模拟解决过程
    setTimeout(() => {
      setConflicts(prev => prev.map(conflict => 
        conflict.id === conflictId 
          ? { ...conflict, resolved: true, resolutionStrategy: conflict.resolutionStrategy || 'select' }
          : conflict
      ));
      
      if (selectedConflict?.id === conflictId) {
        setSelectedConflict(prev => prev ? { ...prev, resolved: true } : null);
      }
      
      setResolutionInProgress(false);
    }, 1500);
  };

  // 处理重新分析
  const handleReanalyze = (conflictId: string) => {
    setConflicts(prev => prev.map(conflict => 
      conflict.id === conflictId 
        ? { 
            ...conflict, 
            confidence: Math.min(0.95, conflict.confidence + 0.1),
            resolutionStrategy: ['merge', 'select', 'dual_version', 'contextual'][Math.floor(Math.random() * 4)] as ResolutionStrategy
          }
        : conflict
    ));
  };

  // 获取相关记忆
  const getRelatedMemories = (memoryIds: string[]) => {
    return memories.filter(memory => memoryIds.includes(memory.id));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-gray-900"></div>
          <p className="mt-2 text-gray-600">加载冲突解决可视化...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="conflict-resolution-visualization">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900">记忆冲突解决可视化</h3>
        <p className="text-sm text-gray-600">
          显示记忆系统中的冲突检测和解决过程。基于实时冲突解决算法。
        </p>
        <div className="flex flex-wrap gap-4 mt-2">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
            <span className="text-xs">内容冲突</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-blue-500 mr-2"></div>
            <span className="text-xs">时间冲突</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-yellow-500 mr-2"></div>
            <span className="text-xs">事实冲突</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-purple-500 mr-2"></div>
            <span className="text-xs">优先级冲突</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
            <span className="text-xs">关联冲突</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6" style={{ height: `${height}px` }}>
        {/* 左侧：冲突列表 */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-xl shadow p-4 h-full overflow-y-auto">
            <h4 className="font-medium text-gray-900 mb-4">检测到的冲突</h4>
            
            {conflicts.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <p>未检测到冲突</p>
                <p className="text-sm mt-1">记忆系统状态良好，无冲突需要解决</p>
              </div>
            ) : (
              <div className="space-y-4">
                {conflicts.map(conflict => {
                  const statusInfo = getStatusInfo(conflict.resolved);
                  return (
                    <div
                      key={conflict.id}
                      className={`p-4 border rounded-lg cursor-pointer transition-all hover:shadow-md ${
                        selectedConflict?.id === conflict.id
                          ? 'border-gray-900 bg-gray-50'
                          : 'border-gray-200 bg-white'
                      }`}
                      onClick={() => setSelectedConflict(conflict)}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 text-xs rounded-full ${getConflictTypeColor(conflict.type)}`}>
                            {getConflictTypeText(conflict.type)}
                          </span>
                          <span className={`px-2 py-1 text-xs rounded-full ${statusInfo.color}`}>
                            {statusInfo.text}
                          </span>
                        </div>
                        
                        <div className="flex items-center">
                          <div className="text-xs text-gray-500 mr-2">置信度:</div>
                          <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className={`h-full ${getConfidenceColor(conflict.confidence)}`}
                              style={{ width: `${conflict.confidence * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-xs font-medium ml-2">
                            {(conflict.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>

                      <p className="text-sm text-gray-800 mb-3">{conflict.description}</p>

                      <div className="flex justify-between items-center">
                        <div className="text-xs text-gray-500">
                          涉及记忆: {conflict.memoryIds.length} 个
                          {conflict.resolutionStrategy && (
                            <span className="ml-3">
                              策略: <span className={`px-2 py-1 rounded ${getStrategyColor(conflict.resolutionStrategy)}`}>
                                {getStrategyText(conflict.resolutionStrategy)}
                              </span>
                            </span>
                          )}
                        </div>
                        
                        <div className="flex space-x-2">
                          {!conflict.resolved && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleResolveConflict(conflict.id);
                              }}
                              disabled={resolutionInProgress}
                              className="px-3 py-1 text-xs bg-gray-700 text-white rounded hover:bg-gray-700 disabled:opacity-50"
                            >
                              {resolutionInProgress ? '解决中...' : '立即解决'}
                            </button>
                          )}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleReanalyze(conflict.id);
                            }}
                            className="px-3 py-1 text-xs bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
                          >
                            重新分析
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <div className="mt-6 pt-4 border-t border-gray-200">
              <div className="text-sm text-gray-600">
                <p>
                  共检测到 <span className="font-semibold">{conflicts.length}</span> 个冲突，
                  其中 <span className="font-semibold text-green-600">
                    {conflicts.filter(c => c.resolved).length}
                  </span> 个已解决，
                  <span className="font-semibold text-red-600 ml-2">
                    {conflicts.filter(c => !c.resolved).length}
                  </span> 个待解决。
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* 右侧：冲突详情和解决过程 */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl shadow p-4 h-full overflow-y-auto">
            <h4 className="font-medium text-gray-900 mb-4">
              {selectedConflict ? '冲突详情' : '选择冲突查看详情'}
            </h4>

            {selectedConflict ? (
              <div className="space-y-4">
                <div>
                  <h5 className="text-sm font-medium text-gray-700 mb-2">冲突信息</h5>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">类型:</span>
                      <span className={`px-2 py-1 text-xs rounded-full ${getConflictTypeColor(selectedConflict.type)}`}>
                        {getConflictTypeText(selectedConflict.type)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">状态:</span>
                      <span className={`px-2 py-1 text-xs rounded-full ${getStatusInfo(selectedConflict.resolved).color}`}>
                        {getStatusInfo(selectedConflict.resolved).text}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">置信度:</span>
                      <span className="text-sm font-medium">
                        {(selectedConflict.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">检测时间:</span>
                      <span className="text-sm">
                        {selectedConflict.timestamp.toLocaleTimeString('zh-CN', { 
                          hour: '2-digit', 
                          minute: '2-digit' 
                        })}
                      </span>
                    </div>
                    {selectedConflict.resolutionStrategy && (
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">解决策略:</span>
                        <span className={`px-2 py-1 text-xs rounded ${getStrategyColor(selectedConflict.resolutionStrategy)}`}>
                          {getStrategyText(selectedConflict.resolutionStrategy)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                <div>
                  <h5 className="text-sm font-medium text-gray-700 mb-2">相关记忆</h5>
                  <div className="space-y-2">
                    {getRelatedMemories(selectedConflict.memoryIds).map(memory => (
                      <div key={memory.id} className="p-2 bg-gray-50 rounded border border-gray-200">
                        <div className="flex justify-between mb-1">
                          <span className="text-xs font-medium">{memory.id}</span>
                          <span className="text-xs text-gray-500">
                            重要性: {(memory.importance * 100).toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-xs text-gray-700 truncate">{memory.content}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h5 className="text-sm font-medium text-gray-700 mb-2">解决过程</h5>
                  <div className="text-sm text-gray-600">
                    {selectedConflict.resolved ? (
                      <div className="p-3 bg-green-50 border border-green-200 rounded">
                        <p className="font-medium text-green-800">冲突已解决</p>
                        <p className="mt-1 text-green-700">
                          使用 {getStrategyText(selectedConflict.resolutionStrategy)} 成功解决了此冲突。
                          相关记忆已更新，系统状态一致。
                        </p>
                      </div>
                    ) : (
                      <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
                        <p className="font-medium text-yellow-800">冲突待解决</p>
                        <p className="mt-1 text-yellow-700">
                          建议使用 {getStrategyText(selectedConflict.resolutionStrategy) || '适当的解决策略'}。
                          点击"立即解决"按钮开始解决过程。
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {!selectedConflict.resolved && (
                  <div className="pt-4 border-t border-gray-200">
                    <button
                      onClick={() => handleResolveConflict(selectedConflict.id)}
                      disabled={resolutionInProgress}
                      className="w-full px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 flex items-center justify-center"
                    >
                      {resolutionInProgress ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white mr-2"></div>
                          正在解决冲突...
                        </>
                      ) : (
                        '立即解决此冲突'
                      )}
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                  <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                </div>
                <p>从左侧列表中选择一个冲突</p>
                <p className="text-sm mt-1">查看详细信息和解决过程</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          冲突解决系统基于五种冲突类型检测算法和四种解决策略。
          系统会持续监控记忆一致性，自动检测并解决冲突。
        </p>
        <p className="mt-1">
          提示: 点击冲突查看详情，使用"立即解决"按钮手动解决冲突，或让系统自动处理。
        </p>
      </div>
    </div>
  );
};

export default ConflictResolutionVisualization;