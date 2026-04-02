/**
 * 多模态概念系统页面
 * 
 * 功能：
 * 1. 多模态数据输入和融合 (文本、图像、音频、传感器数据)
 * 2. 概念统一和推理
 * 3. 场景识别和处理
 * 4. 跨模态学习和迁移
 * 
 * 纯黑白灰极致简洁实用界面
 */

import React, { useState, useRef } from 'react';
import toast from 'react-hot-toast';
import {
  Image,
  Mic,
  Brain,
  RefreshCw,
  X,
  AlertTriangle,
  Info,
  Play,
  Loader2,
  CheckCircle,
  FileText,
  Box,
  Hash,
  Ear,
  Flame,
} from 'lucide-react';
import { apiClient } from '../services/api/client';

// 多模态概念请求接口
interface MultimodalConceptRequest {
  text: string;
  object_name: string;
  quantity: number;
  teaching_mode: boolean;
  scenario_type: 'concept_understanding' | 'computer_operation' | 'equipment_learning' | 'visual_imitation' | 'teaching';
  context: string;
  audio_file?: File;
  image_file?: File;
  taste_sensor_data?: number[]; // [甜度, 酸度, 苦度, 咸度, 鲜味]
  spatial_3d_data?: number[]; // [x,y,z坐标...]
  sensor_data?: Record<string, any>;
}

// 多模态概念响应接口
interface MultimodalConceptResponse {
  success: boolean;
  concept_understanding: string;
  modality_contributions: Record<string, number>;
  concept_attributes: Record<string, any>;
  unified_representation_length: number;
  quantity: number;
  confidence: number;
  timestamp: string;
  error?: string;
}

// 模态定义
interface Modality {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  enabled: boolean;
}

const MultimodalConceptPage: React.FC = () => {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<MultimodalConceptResponse | null>(null);
  
  // 输入状态
  const [request, setRequest] = useState<MultimodalConceptRequest>({
    text: '两个苹果放在桌子上',
    object_name: '苹果',
    quantity: 2,
    teaching_mode: false,
    scenario_type: 'concept_understanding',
    context: '这是一个多模态概念理解的演示例子',
    taste_sensor_data: [0.8, 0.3, 0.05, 0.01, 0.1], // 甜、微酸、微苦、微咸、微鲜
    spatial_3d_data: [0.1, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.1, -0.1], // 球形点云
  });
  
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  
  // 文件输入引用
  const audioInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  
  // 模态定义
  const modalities: Modality[] = [
    { 
      id: 'text', 
      name: '文本理解', 
      icon: <FileText className="w-5 h-5" />, 
      description: '自然语言处理和语义解析',
      enabled: true
    },
    { 
      id: 'pronunciation', 
      name: '发音识别', 
      icon: <Ear className="w-5 h-5" />, 
      description: '语音识别和音频处理',
      enabled: true
    },
    { 
      id: 'visual', 
      name: '视觉识别', 
      icon: <Image className="w-5 h-5" />, 
      description: '图像和视频分析',
      enabled: true
    },
    { 
      id: 'taste', 
      name: '味觉感知', 
      icon: <Flame className="w-5 h-5" />, 
      description: '味觉传感器数据处理',
      enabled: true
    },
    { 
      id: 'spatial', 
      name: '空间感知', 
      icon: <Box className="w-5 h-5" />, 
      description: '3D空间形状和位置',
      enabled: true
    },
    { 
      id: 'identification', 
      name: '物体识别', 
      icon: <CheckCircle className="w-5 h-5" />, 
      description: '物体分类和识别',
      enabled: true
    },
    { 
      id: 'quantity', 
      name: '数量认知', 
      icon: <Hash className="w-5 h-5" />, 
      description: '数量理解和计数',
      enabled: true
    },
  ];
  
  // 处理文件上传
  const handleFileUpload = (type: 'audio' | 'image', event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    if (type === 'audio') {
      setAudioFile(file);
      toast.success(`音频文件已上传: ${file.name}`);
    } else {
      setImageFile(file);
      toast.success(`图像文件已上传: ${file.name}`);
    }
  };
  
  // 发送多模态概念理解请求
  const handleSubmit = async () => {
    if (!request.text.trim()) {
      toast.error('请输入文本描述');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // 准备表单数据
      const formData = new FormData();
      formData.append('text', request.text);
      formData.append('object_name', request.object_name);
      formData.append('quantity', request.quantity.toString());
      formData.append('teaching_mode', request.teaching_mode.toString());
      formData.append('scenario_type', request.scenario_type);
      formData.append('context', request.context);
      
      // 添加文件
      if (audioFile) {
        formData.append('audio_file', audioFile);
      }
      if (imageFile) {
        formData.append('image_file', imageFile);
      }
      
      // 添加传感器数据
      if (request.taste_sensor_data) {
        formData.append('taste_sensor_data_json', JSON.stringify(request.taste_sensor_data));
      }
      if (request.spatial_3d_data) {
        formData.append('spatial_3d_data_json', JSON.stringify(request.spatial_3d_data));
      }
      
      // 调用真实的多模态概念理解API
      const apiResponse = await apiClient.post<MultimodalConceptResponse>(
        '/multimodal/concept/understand',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      
      setResponse(apiResponse);
      toast.success('多模态概念理解完成！');
      
    } catch (error) {
      console.error('多模态概念理解失败:', error);
      const errorMsg = error instanceof Error ? error.message : '未知错误';
      setError(`概念理解失败: ${errorMsg}`);
      toast.error('多模态概念理解请求失败');
    } finally {
      setLoading(false);
    }
  };
  
  // 重置表单
  const handleReset = () => {
    setRequest({
      text: '',
      object_name: '',
      quantity: 1,
      teaching_mode: false,
      scenario_type: 'concept_understanding',
      context: '',
    });
    setAudioFile(null);
    setImageFile(null);
    setResponse(null);
    setError(null);
    
    if (audioInputRef.current) audioInputRef.current.value = '';
    if (imageInputRef.current) imageInputRef.current.value = '';
    
    toast.success('表单已重置');
  };
  
  // 处理输入变化
  const handleInputChange = (field: keyof MultimodalConceptRequest, value: any) => {
    setRequest(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  // 计算模态贡献图表数据
  const getModalityChartData = () => {
    if (!response?.modality_contributions) return null;
    
    const labels = Object.keys(response.modality_contributions);
    const data = Object.values(response.modality_contributions);
    const colors = [
      'bg-gray-700', 'bg-gray-600', 'bg-gray-600', 
      'bg-gray-600', 'bg-gray-800', 'bg-gray-700', 'bg-gray-600'
    ];
    
    return { labels, data, colors };
  };
  
  const chartData = getModalityChartData();
  
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        {/* 标题区域 */}
        <div className="mb-8">
          <div className="flex items-center mb-4">
            <div className="p-3 bg-gray-800 dark:bg-gray-700 rounded-lg mr-4">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                多模态概念系统
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                集成文本、图像、音频、传感器数据等多模态信息的认知和处理
              </p>
            </div>
          </div>
        </div>

        {/* 错误显示 */}
        {error && (
          <div className="mb-6 p-4 bg-gray-900 dark:bg-gray-900/20 border border-gray-700 dark:border-gray-900 rounded-lg">
            <div className="flex items-center">
              <AlertTriangle className="h-5 w-5 text-gray-900 dark:text-gray-500 mr-2" />
              <p className="text-gray-900 dark:text-gray-600">{error}</p>
              <button
                onClick={() => setError(null)}
                className="ml-auto text-gray-900 dark:text-gray-500 hover:text-gray-900 dark:hover:text-gray-700"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          </div>
        )}

        {/* 主内容区域 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* 左侧：输入控制面板 */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  多模态数据输入
                </h2>
                <div className="flex items-center space-x-2">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={request.teaching_mode}
                      onChange={(e) => handleInputChange('teaching_mode', e.target.checked)}
                      className="h-4 w-4 text-gray-600 dark:text-gray-400 rounded"
                    />
                    <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">教学模式</span>
                  </label>
                </div>
              </div>

              {/* 文本输入 */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  文本描述
                </label>
                <textarea
                  value={request.text}
                  onChange={(e) => handleInputChange('text', e.target.value)}
                  placeholder="例如：两个苹果放在桌子上..."
                  className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                  rows={3}
                />
              </div>

              {/* 对象和数量 */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    对象名称
                  </label>
                  <input
                    type="text"
                    value={request.object_name}
                    onChange={(e) => handleInputChange('object_name', e.target.value)}
                    placeholder="例如：苹果"
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    数量
                  </label>
                  <input
                    type="number"
                    value={request.quantity}
                    onChange={(e) => handleInputChange('quantity', parseInt(e.target.value) || 1)}
                    min="1"
                    max="100"
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                  />
                </div>
              </div>

              {/* 文件上传 */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    音频文件
                  </label>
                  <div className="flex items-center">
                    <input
                      type="file"
                      ref={audioInputRef}
                      onChange={(e) => handleFileUpload('audio', e)}
                      accept="audio/*"
                      className="hidden"
                      id="audio-upload"
                    />
                    <label
                      htmlFor="audio-upload"
                      className="flex-1 px-4 py-2 bg-gray-100 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-800 cursor-pointer flex items-center justify-center"
                    >
                      <Mic className="h-5 w-5 mr-2" />
                      {audioFile ? audioFile.name : '选择音频文件'}
                    </label>
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    图像文件
                  </label>
                  <div className="flex items-center">
                    <input
                      type="file"
                      ref={imageInputRef}
                      onChange={(e) => handleFileUpload('image', e)}
                      accept="image/*"
                      className="hidden"
                      id="image-upload"
                    />
                    <label
                      htmlFor="image-upload"
                      className="flex-1 px-4 py-2 bg-gray-100 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-800 cursor-pointer flex items-center justify-center"
                    >
                      <Image className="h-5 w-5 mr-2" />
                      {imageFile ? imageFile.name : '选择图像文件'}
                    </label>
                  </div>
                </div>
              </div>

              {/* 场景选择 */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  场景类型
                </label>
                <select
                  value={request.scenario_type}
                  onChange={(e) => handleInputChange('scenario_type', e.target.value as any)}
                  className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                >
                  <option value="concept_understanding">概念理解</option>
                  <option value="computer_operation">电脑操作</option>
                  <option value="equipment_learning">设备学习</option>
                  <option value="visual_imitation">视觉模仿</option>
                  <option value="teaching">教学</option>
                </select>
              </div>

              {/* 上下文描述 */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  上下文信息
                </label>
                <textarea
                  value={request.context}
                  onChange={(e) => handleInputChange('context', e.target.value)}
                  placeholder="提供额外的上下文信息..."
                  className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent"
                  rows={2}
                />
              </div>

              {/* 操作按钮 */}
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={handleSubmit}
                  disabled={loading}
                  className="px-6 py-3 bg-gray-800 dark:bg-gray-700 hover:bg-gray-900 dark:hover:bg-gray-600 text-white rounded-lg transition-colors flex items-center disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                      处理中...
                    </>
                  ) : (
                    <>
                      <Play className="h-5 w-5 mr-2" />
                      开始概念理解
                    </>
                  )}
                </button>
                
                <button
                  onClick={handleReset}
                  className="px-6 py-3 bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg transition-colors flex items-center"
                >
                  <RefreshCw className="h-5 w-5 mr-2" />
                  重置输入
                </button>
              </div>
            </div>
          </div>

          {/* 右侧：模态信息面板 */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                多模态支持
              </h2>
              
              <div className="space-y-3">
                {modalities.map((modality) => (
                  <div
                    key={modality.id}
                    className={`p-3 rounded-lg border ${modality.enabled 
                      ? 'bg-gray-50 dark:bg-gray-900 border-gray-300 dark:border-gray-700' 
                      : 'bg-gray-100 dark:bg-gray-800 border-gray-200 dark:border-gray-800 opacity-60'
                    }`}
                  >
                    <div className="flex items-center">
                      <div className={`p-2 rounded-md mr-3 ${modality.enabled ? 'bg-gray-800 dark:bg-gray-700' : 'bg-gray-400 dark:bg-gray-600'}`}>
                        <div className="text-white">
                          {modality.icon}
                        </div>
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h3 className="font-medium text-gray-900 dark:text-white">
                            {modality.name}
                          </h3>
                          {modality.enabled ? (
                            <span className="text-xs px-2 py-1 bg-gray-600 dark:bg-gray-900/30 text-gray-800 dark:text-gray-400 rounded">
                              已启用
                            </span>
                          ) : (
                            <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded">
                              未启用
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                          {modality.description}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              {/* 系统状态 */}
              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                  系统状态
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">启用模态</p>
                    <p className="text-xl font-bold text-gray-900 dark:text-white">
                      {modalities.filter(m => m.enabled).length}
                    </p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">总模态数</p>
                    <p className="text-xl font-bold text-gray-900 dark:text-white">
                      {modalities.length}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 结果展示区域 */}
        {response && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                概念理解结果
              </h2>
              <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                <span>置信度: </span>
                <span className="ml-2 font-bold text-gray-900 dark:text-white">
                  {(response.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* 概念理解结果 */}
            <div className="mb-6">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                概念理解
              </h3>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <p className="text-gray-700 dark:text-gray-300">
                  {response.concept_understanding}
                </p>
              </div>
            </div>

            {/* 模态贡献度 */}
            {chartData && (
              <div className="mb-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                  模态贡献度
                </h3>
                <div className="space-y-2">
                  {chartData.labels.map((label, index) => (
                    <div key={label} className="flex items-center">
                      <div className="w-24 text-sm text-gray-700 dark:text-gray-300">
                        {label === 'text' ? '文本理解' : 
                         label === 'pronunciation' ? '发音识别' :
                         label === 'visual' ? '视觉识别' :
                         label === 'taste' ? '味觉感知' :
                         label === 'spatial' ? '空间感知' :
                         label === 'identification' ? '物体识别' : '数量认知'}
                      </div>
                      <div className="flex-1 ml-4">
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                          <div 
                            className={`h-2.5 rounded-full ${chartData.colors[index]}`}
                            style={{ width: `${chartData.data[index] * 100}%` }}
                          ></div>
                        </div>
                      </div>
                      <div className="w-12 text-right text-sm font-medium text-gray-900 dark:text-white">
                        {(chartData.data[index] * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 概念属性 */}
            {response.concept_attributes && Object.keys(response.concept_attributes).length > 0 && (
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                  概念属性
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {Object.entries(response.concept_attributes).map(([key, value]) => (
                    <div key={key} className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                        {key === 'name' ? '名称' : 
                         key === 'quantity' ? '数量' :
                         key === 'color' ? '颜色' :
                         key === 'shape' ? '形状' :
                         key === 'taste' ? '味道' :
                         key === 'size' ? '大小' :
                         key === 'weight' ? '重量' : '类别'}
                      </p>
                      <p className="text-lg font-bold text-gray-900 dark:text-white">
                        {typeof value === 'number' ? value.toFixed(2) : String(value)}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* 技术信息提示 */}
        <div className="mt-6 p-4 bg-gray-100 dark:bg-gray-900/50 border border-gray-300 dark:border-gray-700 rounded-lg">
          <div className="flex items-center">
            <Info className="h-5 w-5 text-gray-600 dark:text-gray-400 mr-2" />
            <div>
              <p className="text-gray-700 dark:text-gray-300">
                多模态概念系统采用先进的跨模态对齐和融合技术，实现文本、图像、音频、传感器数据的统一认知。
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                技术支持：跨模态注意力机制、统一表示学习、多任务联合训练、动态场景适配。
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MultimodalConceptPage;