import React, { useEffect, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { memoryApi, MemoryItem } from '../../services/api/memory';

// 记忆节点接口
interface MemoryNode {
  id: string;
  name: string;
  type: 'short_term' | 'long_term' | 'working';
  importance: number;
  similarity?: number;
  scene_type?: string;
  source: string;
  val: number; // 节点大小
  color: string; // 节点颜色
}

// 记忆连接接口
interface MemoryLink {
  source: string;
  target: string;
  strength: number; // 连接强度
  color: string; // 连接颜色
}

// 记忆网络图数据
interface MemoryGraphData {
  nodes: MemoryNode[];
  links: MemoryLink[];
}

interface MemoryNetworkGraphProps {
  maxNodes?: number;
  height?: number;
}

const MemoryNetworkGraph: React.FC<MemoryNetworkGraphProps> = ({
  maxNodes = 50,
  height = 500,
}) => {
  const [graphData, setGraphData] = useState<MemoryGraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const graphRef = useRef<any>(null);

  // 获取记忆数据并构建网络图
  useEffect(() => {
    const fetchMemoryData = async () => {
      try {
        setLoading(true);
        setError(null);

        // 获取最近记忆
        const response = await memoryApi.getRecentMemories(maxNodes);
        
        if (response.success && response.data) {
          const memories = response.data;
          
          // 构建节点
          const nodes: MemoryNode[] = memories.map((memory: MemoryItem, _index: number) => {
            // 根据记忆类型设置颜色
            let color = '#4f46e5'; // 默认紫色
            
            switch (memory.type) {
              case 'short_term':
                color = '#3b82f6'; // 蓝色
                break;
              case 'long_term':
                color = '#10b981'; // 绿色
                break;
              case 'working':
                color = '#f59e0b'; // 橙色
                break;
            }

            // 根据重要性设置节点大小
            const baseSize = 5;
            const importanceFactor = memory.importance * 10;
            const nodeSize = baseSize + importanceFactor;

            return {
              id: memory.id,
              name: memory.content.substring(0, 30) + (memory.content.length > 30 ? '...' : ''),
              type: memory.type,
              importance: memory.importance,
              similarity: memory.similarity,
              scene_type: memory.scene_type,
              source: memory.source,
              val: nodeSize,
              color: color,
            };
          });

          // 构建连接（基于相似性）
          const links: MemoryLink[] = [];
          const similarityThreshold = 0.3; // 相似度阈值

          for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
              // 如果两个记忆有相似性数据，并且相似度超过阈值，创建连接
              if (nodes[i].similarity !== undefined && nodes[j].similarity !== undefined) {
                // 使用模拟相似度（实际项目中应从后端获取记忆间相似度）
                const simulatedSimilarity = Math.random() * 0.5 + 0.3; // 模拟相似度
                
                if (simulatedSimilarity > similarityThreshold) {
                  links.push({
                    source: nodes[i].id,
                    target: nodes[j].id,
                    strength: simulatedSimilarity,
                    color: `rgba(100, 100, 100, ${0.3 + simulatedSimilarity * 0.7})`, // 根据强度设置透明度
                  });
                }
              }
            }

            // 限制连接数量以避免过度拥挤
            if (links.length > nodes.length * 3) break;
          }

          setGraphData({ nodes, links });
        } else {
          throw new Error(response.message || '获取记忆数据失败');
        }
      } catch (err: any) {
        console.error('获取记忆网络数据失败:', err);
        setError(err.message || '加载记忆网络图失败');
        
        // 提供示例数据用于演示
        const sampleNodes: MemoryNode[] = [
          { id: '1', name: '任务规划记忆', type: 'short_term', importance: 0.8, source: 'system', val: 8, color: '#3b82f6' },
          { id: '2', name: '学习记录', type: 'long_term', importance: 0.9, source: 'user', val: 9, color: '#10b981' },
          { id: '3', name: '问题解决', type: 'working', importance: 0.7, source: 'autonomous', val: 7, color: '#f59e0b' },
          { id: '4', name: '场景记忆', type: 'short_term', importance: 0.6, source: 'system', val: 6, color: '#3b82f6' },
          { id: '5', name: '知识整合', type: 'long_term', importance: 0.85, source: 'autonomous', val: 8.5, color: '#10b981' },
        ];

        const sampleLinks: MemoryLink[] = [
          { source: '1', target: '2', strength: 0.7, color: 'rgba(100, 100, 100, 0.7)' },
          { source: '1', target: '3', strength: 0.5, color: 'rgba(100, 100, 100, 0.5)' },
          { source: '2', target: '4', strength: 0.6, color: 'rgba(100, 100, 100, 0.6)' },
          { source: '3', target: '5', strength: 0.8, color: 'rgba(100, 100, 100, 0.8)' },
          { source: '4', target: '5', strength: 0.4, color: 'rgba(100, 100, 100, 0.4)' },
        ];

        setGraphData({ nodes: sampleNodes, links: sampleLinks });
      } finally {
        setLoading(false);
      }
    };

    fetchMemoryData();
  }, [maxNodes]);

  // 处理节点点击
  const handleNodeClick = (node: MemoryNode) => {
    console.log('点击节点:', node);
    // 在实际应用中，这里可以显示节点详情或执行其他操作
  };

  // 处理背景点击
  const handleBackgroundClick = () => {
    console.log('点击背景');
    // 在实际应用中，这里可以清除选择或显示全局信息
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-gray-900"></div>
          <p className="mt-2 text-gray-600">加载记忆网络图...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-red-600">
          <p>加载记忆网络图时出错: {error}</p>
          <p className="text-sm text-gray-600 mt-2">显示示例数据以供演示</p>
        </div>
      </div>
    );
  }

  return (
    <div className="memory-network-graph">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900">记忆关联网络</h3>
        <p className="text-sm text-gray-600">
          显示记忆之间的关联关系。节点大小表示重要性，连线强度表示关联度。
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
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-purple-500 mr-2"></div>
            <span className="text-xs">系统记忆</span>
          </div>
        </div>
      </div>

      <div style={{ height: `${height}px`, border: '1px solid #e5e7eb', borderRadius: '0.5rem' }}>
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          nodeLabel={(node: any) => `
            ${node.name}
            类型: ${node.type === 'short_term' ? '短期记忆' : node.type === 'long_term' ? '长期记忆' : '工作记忆'}
            重要性: ${(node.importance * 100).toFixed(0)}%
            来源: ${node.source === 'user' ? '用户' : node.source === 'system' ? '系统' : '自主'}
          `}
          nodeColor={(node: any) => node.color}
          nodeVal={(node: any) => node.val}
          linkColor={(link: any) => link.color}
          linkWidth={(link: any) => link.strength * 3}
          linkDirectionalParticles={2}
          linkDirectionalParticleSpeed={(link: any) => link.strength * 0.01}
          onNodeClick={handleNodeClick}
          onBackgroundClick={handleBackgroundClick}
          cooldownTime={3000}
          enableZoomInteraction={true}
          enablePanInteraction={true}
          backgroundColor="#ffffff"
        />
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          共显示 <span className="font-semibold">{graphData.nodes.length}</span> 个记忆节点，
          <span className="font-semibold"> {graphData.links.length}</span> 个关联关系。
        </p>
        <p className="mt-1">
          提示: 点击节点查看详情，拖拽节点重新布局，使用滚轮缩放。
        </p>
      </div>
    </div>
  );
};

export default MemoryNetworkGraph;