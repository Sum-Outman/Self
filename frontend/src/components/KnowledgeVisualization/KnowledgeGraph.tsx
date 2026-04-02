import React, { useState, useEffect, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { knowledgeApi } from '../../services/api/knowledge';

// 知识图谱节点接口
interface KnowledgeNode {
  id: string;
  title: string;
  description: string;
  type: string;
  color: string;
  size: number;
  tags: string[];
  access_count: number;
  upload_date: string;
}

// 知识图谱边接口
interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  weight: number;
  type: string;
}

// 知识图谱数据接口
interface KnowledgeGraphData {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  stats: {
    total_nodes: number;
    total_edges: number;
    node_types: Record<string, number>;
    tag_count: number;
  };
  center_item_id?: string;
  depth: number;
}

interface KnowledgeGraphProps {
  centerItemId?: string;
  depth?: number;

  height?: number;
  width?: number;
  onNodeClick?: (node: KnowledgeNode) => void;
  onEdgeClick?: (edge: KnowledgeEdge) => void;
}

const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({
  centerItemId,
  depth = 2,

  height = 500,
  width = 800,
  onNodeClick,
  onEdgeClick,
}) => {
  const [graphData, setGraphData] = useState<KnowledgeGraphData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<KnowledgeNode | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<KnowledgeEdge | null>(null);

  // 获取知识图谱数据
  const fetchGraphData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await knowledgeApi.getKnowledgeGraph(centerItemId, depth);
      
      if (response.success && response.data) {
        setGraphData(response.data as KnowledgeGraphData);
      } else {
        setError('获取知识图谱数据失败');
      }
    } catch (err) {
      console.error('获取知识图谱数据失败:', err);
      setError('加载知识图谱数据时发生错误');
    } finally {
      setLoading(false);
    }
  }, [centerItemId, depth]);

  // 初始加载和参数变化时重新加载
  useEffect(() => {
    fetchGraphData();
  }, [fetchGraphData]);

  // 处理节点点击
  const handleNodeClick = useCallback((node: any) => {
    if (onNodeClick && graphData) {
      const knowledgeNode = graphData.nodes.find(n => n.id === node.id);
      if (knowledgeNode) {
        onNodeClick(knowledgeNode);
      }
    }
  }, [graphData, onNodeClick]);

  // 处理边点击
  const handleEdgeClick = useCallback((edge: any) => {
    if (onEdgeClick && graphData) {
      const knowledgeEdge = graphData.edges.find(e => e.id === edge.id);
      if (knowledgeEdge) {
        onEdgeClick(knowledgeEdge);
      }
    }
  }, [graphData, onEdgeClick]);

  // 处理节点悬停
  const handleNodeHover = useCallback((node: any) => {
    if (node && graphData) {
      const knowledgeNode = graphData.nodes.find(n => n.id === node.id);
      setHoveredNode(knowledgeNode || null);
    } else {
      setHoveredNode(null);
    }
  }, [graphData]);

  // 处理边悬停
  const handleEdgeHover = useCallback((edge: any) => {
    if (edge && graphData) {
      const knowledgeEdge = graphData.edges.find(e => e.id === edge.id);
      setHoveredEdge(knowledgeEdge || null);
    } else {
      setHoveredEdge(null);
    }
  }, [graphData]);

  // 渲染节点标签
  const nodeLabel = useCallback((node: any) => {
    const knowledgeNode = graphData?.nodes.find(n => n.id === node.id);
    if (!knowledgeNode) return node.id;
    
    // 截断长标题
    const title = knowledgeNode.title.length > 20 
      ? `${knowledgeNode.title.substring(0, 20)}...` 
      : knowledgeNode.title;
    
    return `${title} (${knowledgeNode.type})`;
  }, [graphData]);

  // 渲染边标签
  const edgeLabel = useCallback((edge: any) => {
    const knowledgeEdge = graphData?.edges.find(e => e.id === edge.id);
    if (!knowledgeEdge) return '';
    
    return knowledgeEdge.label || knowledgeEdge.type;
  }, [graphData]);

  // 节点颜色
  const nodeColor = useCallback((node: any) => {
    const knowledgeNode = graphData?.nodes.find(n => n.id === node.id);
    return knowledgeNode?.color || '#CCCCCC';
  }, [graphData]);

  // 节点大小
  const nodeVal = useCallback((node: any) => {
    const knowledgeNode = graphData?.nodes.find(n => n.id === node.id);
    return knowledgeNode?.size || 10;
  }, [graphData]);

  // 边颜色
  const edgeColor = useCallback((edge: any) => {
    const knowledgeEdge = graphData?.edges.find(e => e.id === edge.id);
    if (!knowledgeEdge) return '#999999';
    
    // 根据边类型设置颜色
    switch (knowledgeEdge.type) {
      case 'tag_similarity':
        return '#4ECDC4'; // 青色
      case 'type_similarity':
        return '#FECA57'; // 黄色
      default:
        return '#999999';
    }
  }, [graphData]);

  // 边宽度
  const edgeWidth = useCallback((edge: any) => {
    const knowledgeEdge = graphData?.edges.find(e => e.id === edge.id);
    return knowledgeEdge?.weight || 1;
  }, [graphData]);

  // 如果正在加载，显示加载状态
  if (loading) {
    return (
      <div className="flex items-center justify-center" style={{ height, width }}>
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
          <p className="mt-2 text-gray-600">正在加载知识图谱...</p>
        </div>
      </div>
    );
  }

  // 如果发生错误，显示错误信息
  if (error) {
    return (
      <div className="flex items-center justify-center" style={{ height, width }}>
        <div className="text-center text-red-600">
          <p className="mb-2">{error}</p>
          <button
            onClick={fetchGraphData}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            重试
          </button>
        </div>
      </div>
    );
  }

  // 如果没有数据，显示空状态
  if (!graphData || graphData.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center" style={{ height, width }}>
        <div className="text-center text-gray-500">
          <p>暂无知识图谱数据</p>
          <p className="text-sm mt-1">请先添加一些知识项</p>
        </div>
      </div>
    );
  }

  // 准备力导向图数据
  const forceGraphData = {
    nodes: graphData.nodes.map(node => ({
      id: node.id,
      title: node.title,
      type: node.type,
      color: node.color,
      val: node.size,
    })),
    links: graphData.edges.map(edge => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      weight: edge.weight,
      type: edge.type,
    })),
  };

  return (
    <div className="relative" style={{ height, width }}>
      {/* 知识图谱可视化 */}
      <ForceGraph2D
        graphData={forceGraphData}
        nodeLabel={nodeLabel}
        nodeColor={nodeColor}
        nodeVal={nodeVal}
        linkLabel={edgeLabel}
        linkColor={edgeColor}
        linkWidth={edgeWidth}
        linkDirectionalParticles={1}
        linkDirectionalParticleSpeed={0.005}
        linkDirectionalParticleWidth={2}
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        onLinkClick={handleEdgeClick}
        onLinkHover={handleEdgeHover}
        cooldownTime={3000}
        height={height}
        width={width}
      />
      
      {/* 悬停信息面板 */}
      {(hoveredNode || hoveredEdge) && (
        <div className="absolute top-4 left-4 bg-white rounded-lg shadow-lg p-4 max-w-xs z-10">
          {hoveredNode && (
            <div>
              <h4 className="font-semibold text-lg mb-2">{hoveredNode.title}</h4>
              <p className="text-sm text-gray-600 mb-1">
                <span className="font-medium">类型:</span> {hoveredNode.type}
              </p>
              <p className="text-sm text-gray-600 mb-1">
                <span className="font-medium">访问次数:</span> {hoveredNode.access_count}
              </p>
              {hoveredNode.tags && hoveredNode.tags.length > 0 && (
                <div className="mb-1">
                  <span className="font-medium text-sm">标签:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {hoveredNode.tags.slice(0, 3).map((tag, index) => (
                      <span
                        key={index}
                        className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded"
                      >
                        {tag}
                      </span>
                    ))}
                    {hoveredNode.tags.length > 3 && (
                      <span className="px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded">
                        +{hoveredNode.tags.length - 3}
                      </span>
                    )}
                  </div>
                </div>
              )}
              <p className="text-xs text-gray-500 mt-2">
                上传时间: {new Date(hoveredNode.upload_date).toLocaleDateString('zh-CN')}
              </p>
            </div>
          )}
          
          {hoveredEdge && !hoveredNode && (
            <div>
              <h4 className="font-semibold text-lg mb-2">关联关系</h4>
              <p className="text-sm text-gray-600 mb-1">
                <span className="font-medium">类型:</span> {hoveredEdge.type === 'tag_similarity' ? '标签相似性' : '类型相似性'}
              </p>
              <p className="text-sm text-gray-600 mb-1">
                <span className="font-medium">关联:</span> {hoveredEdge.label}
              </p>
              <p className="text-sm text-gray-600">
                <span className="font-medium">强度:</span> {hoveredEdge.weight.toFixed(2)}
              </p>
            </div>
          )}
        </div>
      )}
      
      {/* 图例 */}
      <div className="absolute bottom-4 right-4 bg-white rounded-lg shadow-lg p-4 z-10">
        <h4 className="font-semibold text-sm mb-2">图例</h4>
        <div className="space-y-2">
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full bg-[#4ECDC4] mr-2"></div>
            <span className="text-xs">文本知识</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full bg-[#FF6B6B] mr-2"></div>
            <span className="text-xs">图像知识</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full bg-[#45B7D1] mr-2"></div>
            <span className="text-xs">视频知识</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full bg-[#96CEB4] mr-2"></div>
            <span className="text-xs">音频知识</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full bg-[#4ECDC4] mr-2"></div>
            <span className="text-xs">标签关联</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full bg-[#FECA57] mr-2"></div>
            <span className="text-xs">类型关联</span>
          </div>
        </div>
      </div>
      
      {/* 统计信息 */}
      <div className="absolute top-4 right-4 bg-white rounded-lg shadow-lg p-4 z-10">
        <h4 className="font-semibold text-sm mb-2">统计信息</h4>
        {graphData.stats && (
          <div className="space-y-1">
            <p className="text-xs">
              <span className="font-medium">节点数:</span> {graphData.stats.total_nodes}
            </p>
            <p className="text-xs">
              <span className="font-medium">边数:</span> {graphData.stats.total_edges}
            </p>
            <p className="text-xs">
              <span className="font-medium">标签数:</span> {graphData.stats.tag_count}
            </p>
            {graphData.center_item_id && (
              <p className="text-xs">
                <span className="font-medium">中心节点:</span> 已指定
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default KnowledgeGraph;