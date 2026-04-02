import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  Database,
  Search,
  Upload,
  Download,
  Trash2,
  Edit,
  Eye,
  Filter,
  RefreshCw,
  FileText,
  Image as ImageIcon,
  Video,
  Music,
  Archive,
  Tag,
  Calendar,
  BarChart3,
  Loader2,
  X,
  ChevronDown,
  Star,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { knowledgeApi, KnowledgeItem, KnowledgeStats, SearchRequest, UploadRequest } from '../services/api/knowledge';
import KnowledgeGraph from '../components/KnowledgeVisualization/KnowledgeGraph';

// 前端UI知识项接口（兼容现有UI）
interface UiKnowledgeItem {
  id: string;
  title: string;
  description: string;
  type: 'text' | 'image' | 'video' | 'audio' | 'document' | 'code' | 'dataset';
  content?: string;
  size: number;
  uploadDate: Date;
  tags: string[];
  uploadedBy: string;
  accessCount: number;
  lastAccessed: Date;
  fileUrl?: string;
  metadata?: Record<string, any>;
  similarity?: number;
}

const KnowledgePage: React.FC = () => {
  const { user: _user } = useAuth();
  const [knowledgeItems, setKnowledgeItems] = useState<UiKnowledgeItem[]>([]);
  const [knowledgeStats, setKnowledgeStats] = useState<KnowledgeStats | null>(null);
  const [activeTab, setActiveTab] = useState<'all' | 'text' | 'image' | 'video' | 'audio' | 'document' | 'code' | 'dataset' | 'visualization'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showSearchFilters, setShowSearchFilters] = useState(false);
  const [allTags, setAllTags] = useState<string[]>([]);
  const [typeStats, setTypeStats] = useState<Record<string, number>>({});
  const [selectedItemDetail, setSelectedItemDetail] = useState<UiKnowledgeItem | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  
  // 搜索历史状态
  const [searchHistory, setSearchHistory] = useState<Array<{
    query: string;
    timestamp: Date;
    resultCount: number;
    filters?: any;
  }>>([]);
  
  // 搜索建议状态 - 未来功能
  // const [searchSuggestions, setSearchSuggestions] = useState<string[]>([]);
  
  // 分页状态 - 未来功能
  // const [pagination, setPagination] = useState({
  //   currentPage: 1,
  //   totalPages: 1,
  //   totalItems: 0,
  //   itemsPerPage: 20,
  // });
  
  const [uploadData, setUploadData] = useState<{
    file: File | null;
    title: string;
    description: string;
    tags: string[];
    type: string;
  }>({
    file: null,
    title: '',
    description: '',
    tags: [],
    type: 'document',
  });
  
  const [searchFilters, setSearchFilters] = useState<{
    sortBy: 'relevance' | 'date' | 'access' | 'size';
    sortOrder: 'asc' | 'desc';
    startDate: string;
    endDate: string;
    limit: number;
  }>({
    sortBy: 'relevance',
    sortOrder: 'desc',
    startDate: '',
    endDate: '',
    limit: 20,
  });
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // 编辑相关状态
  const [editingItem, setEditingItem] = useState<UiKnowledgeItem | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editData, setEditData] = useState<{
    title: string;
    description: string;
    tags: string[];
    type: string;
  }>({
    title: '',
    description: '',
    tags: [],
    type: 'document',
  });

  // 转换API响应到UI接口
  const convertApiItemToUi = (apiItem: KnowledgeItem): UiKnowledgeItem => {
    return {
      id: apiItem.id,
      title: apiItem.title,
      description: apiItem.description,
      type: apiItem.type,
      size: apiItem.size,
      uploadDate: new Date(apiItem.upload_date),
      tags: apiItem.tags,
      uploadedBy: apiItem.uploaded_by,
      accessCount: apiItem.access_count,
      lastAccessed: new Date(apiItem.last_accessed),
      fileUrl: apiItem.file_url,
      metadata: apiItem.metadata,
      similarity: apiItem.similarity,
    };
  };

  // 加载知识项列表
  const loadKnowledgeItems = async (page: number = 1) => {
    try {
      setIsSearching(true);
      let response;
      
      // 计算偏移量
      const offset = (page - 1) * searchFilters.limit;
      
      if (searchQuery.trim() || selectedTags.length > 0 || showSearchFilters) {
        // 使用搜索API
        const searchRequest: SearchRequest = {
          query: searchQuery,
          type: activeTab !== 'all' ? activeTab : undefined,
          tags: selectedTags.length > 0 ? selectedTags : undefined,
          start_date: searchFilters.startDate || undefined,
          end_date: searchFilters.endDate || undefined,
          limit: searchFilters.limit,
          offset: offset,
          sort_by: searchFilters.sortBy,
          sort_order: searchFilters.sortOrder,
        };
        
        response = await knowledgeApi.search(searchRequest);
      } else {
        // 使用普通列表API
        response = await knowledgeApi.getItems(
          activeTab !== 'all' ? activeTab : undefined,
          selectedTags.length > 0 ? selectedTags : undefined,
          searchFilters.limit,
          offset
        );
      }
      
      const responseAny = response as any;
      let items: KnowledgeItem[] = [];
      if (responseAny.success && responseAny.data) {
        // ApiResponse 类型
        if ('items' in responseAny.data) {
          // 搜索响应
          items = responseAny.data.items;
        } else {
          // 普通列表响应
          items = responseAny.data;
        }
      } else if (responseAny.items) {
        // 直接 SearchResponse
        items = responseAny.items;
      } else {
        // 直接 KnowledgeItem[]
        items = responseAny;
      }
      const uiItems = items.map(convertApiItemToUi);
      setKnowledgeItems(uiItems);
      
      // 更新分页信息 - 未来功能
      // if (responseAny.success && responseAny.data && 'total' in responseAny.data) {
      //   const total = responseAny.data.total;
      //   const totalPages = Math.ceil(total / searchFilters.limit);
      //   setPagination(prev => ({
      //     ...prev,
      //     currentPage: page,
      //     totalPages: totalPages,
      //     totalItems: total,
      //   }));
        
        // 如果进行了搜索，记录搜索历史
         if (searchQuery.trim() && page === 1) {
           const newSearchHistory = [
             {
               query: searchQuery,
               timestamp: new Date(),
               resultCount: responseAny.success && responseAny.data && 'total' in responseAny.data ? responseAny.data.total : knowledgeItems.length,
               filters: {
                 type: activeTab !== 'all' ? activeTab : undefined,
                 tags: selectedTags,
                 startDate: searchFilters.startDate,
                 endDate: searchFilters.endDate,
               },
             },
             ...searchHistory.slice(0, 9), // 保留最近10条记录
           ];
           setSearchHistory(newSearchHistory);
           
           // 保存到本地存储
           try {
             localStorage.setItem('knowledge_search_history', JSON.stringify(newSearchHistory.map(h => ({
               ...h,
               timestamp: h.timestamp.toISOString(),
             }))));
           } catch (e) {
             console.error('保存搜索历史失败:', e);
}
         }
      }
    catch (error) {
      console.error('加载知识项失败:', error);
      toast.error('加载知识项失败，请检查网络连接并重试');
      // 不设置虚拟数据，保持空数组
      setKnowledgeItems([]);
    } finally {
      setIsSearching(false);
    }
  };

  // 加载知识库统计
  const loadKnowledgeStats = async () => {
    try {
      const response = await knowledgeApi.getStats();
      if (response.success && response.data) {
        setKnowledgeStats(response.data);
      }
    } catch (error) {
      console.error('加载知识库统计失败:', error);
    }
  };

  // 加载标签列表
  const loadTags = async () => {
    try {
      const response = await knowledgeApi.getTags();
      if (response.success && response.data) {
        setAllTags(response.data);
      }
    } catch (error) {
      console.error('加载标签列表失败:', error);
    }
  };

  // 加载类型统计
  const loadTypeStats = async () => {
    try {
      const response = await knowledgeApi.getTypeStats();
      if (response.success && response.data) {
        setTypeStats(response.data);
      }
    } catch (error) {
      console.error('加载类型统计失败:', error);
    }
  };

  // 从本地存储加载搜索历史
  useEffect(() => {
    try {
      const savedHistory = localStorage.getItem('knowledge_search_history');
      if (savedHistory) {
        const parsedHistory = JSON.parse(savedHistory);
        const historyWithDates = parsedHistory.map((h: any) => ({
          ...h,
          timestamp: new Date(h.timestamp),
        }));
        setSearchHistory(historyWithDates.slice(0, 10)); // 最多加载10条
      }
    } catch (e) {
      console.error('加载搜索历史失败:', e);
    }
  }, []);

  // 初始加载
  useEffect(() => {
    loadKnowledgeItems();
    loadKnowledgeStats();
    loadTags();
    loadTypeStats();

    // 设置轮询更新
    pollIntervalRef.current = setInterval(() => {
      loadKnowledgeStats();
    }, 30000); // 每30秒更新一次统计

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [activeTab, selectedTags, searchFilters]);

  // 搜索效果
  useEffect(() => {
    const delayDebounceFn = setTimeout(() => {
      if (searchQuery.trim() || selectedTags.length > 0) {
        loadKnowledgeItems();
      }
    }, 500);

    return () => clearTimeout(delayDebounceFn);
  }, [searchQuery, selectedTags]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // 重置到第一页进行新搜索
    loadKnowledgeItems(1);
  };

  // 处理分页 - 未来功能
  // const handlePageChange = (page: number) => {
  //   loadKnowledgeItems(page);
  // };

  // 清除搜索历史
  const clearSearchHistory = () => {
    setSearchHistory([]);
    try {
      localStorage.removeItem('knowledge_search_history');
    } catch (e) {
      console.error('清除搜索历史失败:', e);
    }
  };

  // 使用搜索历史项
  const useSearchHistoryItem = (historyItem: typeof searchHistory[0]) => {
    setSearchQuery(historyItem.query);
    if (historyItem.filters?.type && historyItem.filters.type !== 'all') {
      setActiveTab(historyItem.filters.type as any);
    }
    if (historyItem.filters?.tags) {
      setSelectedTags(historyItem.filters.tags);
    }
    if (historyItem.filters?.startDate) {
      setSearchFilters(prev => ({ ...prev, startDate: historyItem.filters!.startDate }));
    }
    if (historyItem.filters?.endDate) {
      setSearchFilters(prev => ({ ...prev, endDate: historyItem.filters!.endDate }));
    }
    // 触发搜索
    setTimeout(() => loadKnowledgeItems(1), 100);
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!uploadData.file) {
      toast.error('请选择要上传的文件');
      return;
    }
    
    if (!uploadData.title.trim()) {
      toast.error('请输入文件标题');
      return;
    }
    
    setIsUploading(true);
    
    try {
      const uploadRequest: UploadRequest = {
        file: uploadData.file,
        title: uploadData.title,
        description: uploadData.description,
        tags: uploadData.tags,
        type: uploadData.type,
      };
      
      const response = await knowledgeApi.uploadItem(uploadRequest);
      
      if (response.success && response.data) {
        toast.success('文件上传成功');
        setShowUploadModal(false);
        setUploadData({
          file: null,
          title: '',
          description: '',
          tags: [],
          type: 'document',
        });
        
        // 刷新列表和统计
        loadKnowledgeItems();
        loadKnowledgeStats();
        loadTags();
        loadTypeStats();
      } else {
        toast.error('文件上传失败');
      }
    } catch (error) {
      console.error('文件上传失败:', error);
      toast.error('文件上传失败');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteItem = async (itemId: string) => {
    if (!confirm('确定要删除此知识项吗？')) {
      return;
    }
    
    try {
      const response = await knowledgeApi.deleteItem(itemId);
      if (response.success) {
        toast.success('知识项已删除');
        loadKnowledgeItems();
        loadKnowledgeStats();
        loadTags();
        loadTypeStats();
      } else {
        toast.error('删除失败');
      }
    } catch (error) {
      console.error('删除知识项失败:', error);
      toast.error('删除失败');
    }
  };

  // 编辑知识项相关函数
  const handleEditClick = (item: UiKnowledgeItem) => {
    setEditingItem(item);
    setEditData({
      title: item.title,
      description: item.description,
      tags: [...item.tags],
      type: item.type,
    });
    setShowEditModal(true);
  };

  const handleEditSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingItem) return;
    
    try {
      const response = await knowledgeApi.updateItem(editingItem.id, {
        title: editData.title,
        description: editData.description,
        tags: editData.tags,
        type: editData.type as any,
      });
      
      if (response.success) {
        toast.success('知识项已更新');
        setShowEditModal(false);
        setEditingItem(null);
        loadKnowledgeItems();
        loadKnowledgeStats();
        loadTags();
        loadTypeStats();
      } else {
        toast.error('更新失败: ' + (response.message || '未知错误'));
      }
    } catch (error: any) {
      console.error('更新知识项失败:', error);
      toast.error('更新失败: ' + (error.message || '未知错误'));
    }
  };

  const handleEditInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setEditData(prev => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleEditTagAdd = (tag: string) => {
    if (tag.trim() && !editData.tags.includes(tag.trim())) {
      setEditData(prev => ({
        ...prev,
        tags: [...prev.tags, tag.trim()],
      }));
    }
  };

  const handleEditTagRemove = (tagToRemove: string) => {
    setEditData(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove),
    }));
  };

  const handleRecordAccess = async (itemId: string) => {
    try {
      await knowledgeApi.recordAccess(itemId);
    } catch (error) {
      // 不显示错误，因为这只是分析数据
      console.error('记录访问失败:', error);
    }
  };

  const handleItemClick = async (itemId: string) => {
    // 记录访问
    handleRecordAccess(itemId);
    
    try {
      // 获取知识项详情
      const item = knowledgeItems.find(item => item.id === itemId);
      if (item) {
        setSelectedItemDetail(item);
        setShowDetailModal(true);
      } else {
        // 如果本地没有找到，尝试从API获取
        const response = await knowledgeApi.getItem(itemId);
        const detailItem: UiKnowledgeItem = {
          id: response.id,
          title: response.title,
          description: response.description,
          type: response.type as any,
          content: response.content,
          size: response.size,
          uploadDate: new Date(response.upload_date),
          tags: response.tags,
          uploadedBy: response.uploaded_by,
          accessCount: response.access_count,
          lastAccessed: response.last_accessed ? new Date(response.last_accessed) : new Date(),
          fileUrl: response.file_url,
          metadata: response.metadata,
        };
        setSelectedItemDetail(detailItem);
        setShowDetailModal(true);
      }
    } catch (error) {
      console.error('获取知识项详情失败:', error);
      toast.error('获取知识项详情失败');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadData(prev => ({
        ...prev,
        file,
        title: file.name.replace(/\.[^/.]+$/, ""), // 移除扩展名
        type: getFileType(file.type),
      }));
    }
  };

  const getFileType = (mimeType: string): string => {
    if (mimeType.startsWith('image/')) return 'image';
    if (mimeType.startsWith('video/')) return 'video';
    if (mimeType.startsWith('audio/')) return 'audio';
    if (mimeType.includes('pdf') || mimeType.includes('document') || mimeType.includes('text')) return 'document';
    if (mimeType.includes('code') || mimeType.includes('javascript') || mimeType.includes('python')) return 'code';
    return 'document';
  };

  const handleTagToggle = (tag: string) => {
    setSelectedTags(prev => 
      prev.includes(tag) 
        ? prev.filter(t => t !== tag)
        : [...prev, tag]
    );
  };

  const handleAddTag = (tag: string) => {
    if (tag.trim() && !uploadData.tags.includes(tag.trim())) {
      setUploadData(prev => ({
        ...prev,
        tags: [...prev.tags, tag.trim()],
      }));
    }
  };

  const handleRemoveUploadTag = (tagToRemove: string) => {
    setUploadData(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove),
    }));
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (date: Date): string => {
    return date.toLocaleDateString('zh-CN');
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'text': return <FileText className="w-5 h-5" />;
      case 'image': return <ImageIcon className="w-5 h-5" />;
      case 'video': return <Video className="w-5 h-5" />;
      case 'audio': return <Music className="w-5 h-5" />;
      case 'document': return <FileText className="w-5 h-5" />;
      case 'code': return <Archive className="w-5 h-5" />;
      case 'dataset': return <Database className="w-5 h-5" />;
      default: return <FileText className="w-5 h-5" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'text': return 'bg-gray-700 text-gray-700 dark:bg-gray-700/30 dark:text-gray-700';
      case 'image': return 'bg-gray-600 text-gray-600 dark:bg-gray-700/30 dark:text-gray-600';
      case 'video': return 'bg-gray-600 text-gray-600 dark:bg-gray-600/30 dark:text-gray-600';
      case 'audio': return 'bg-gray-500 text-gray-500 dark:bg-gray-800/30 dark:text-gray-500';
      case 'document': return 'bg-gray-800 text-gray-800 dark:bg-gray-900/30 dark:text-gray-800';
      case 'code': return 'bg-gray-800 text-gray-800 dark:bg-gray-800/30 dark:text-gray-800';
      case 'dataset': return 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-400';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400';
    }
  };

  const getTypeName = (type: string) => {
    switch (type) {
      case 'text': return '文本';
      case 'image': return '图片';
      case 'video': return '视频';
      case 'audio': return '音频';
      case 'document': return '文档';
      case 'code': return '代码';
      case 'dataset': return '数据集';
      default: return '未知';
    }
  };

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            知识库管理系统
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            存储、管理和检索Self AGI的知识资产
          </p>
          <div className="mt-2">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-800 text-white">
              功能状态：实现中
            </span>
            <span className="ml-2 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-600 text-white">
              后端服务：已连接
            </span>
            <span className="ml-2 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-700 text-white">
              知识类型：多模态
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setShowUploadModal(true)}
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-700"
          >
            <Upload className="w-4 h-4 mr-2" />
            上传知识
          </button>
        </div>
      </div>

      {/* 知识库统计卡片 */}
      {knowledgeStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900 dark:text-white">总知识项</h3>
              <Database className="w-5 h-5 text-gray-700" />
            </div>
            <div className="mt-3">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {knowledgeStats.total_items.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                总大小: {formatFileSize(knowledgeStats.total_size)}
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900 dark:text-white">存储使用</h3>
              <BarChart3 className="w-5 h-5 text-gray-600" />
            </div>
            <div className="mt-3">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {knowledgeStats.storage_usage.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                存储空间使用率
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900 dark:text-white">平均访问</h3>
              <Eye className="w-5 h-5 text-gray-600" />
            </div>
            <div className="mt-3">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {knowledgeStats.average_access_count.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                每项平均访问次数
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900 dark:text-white">热门标签</h3>
              <Tag className="w-5 h-5 text-gray-500" />
            </div>
            <div className="mt-3">
              <div className="text-sm text-gray-900 dark:text-white font-medium">
                {knowledgeStats.popular_tags.slice(0, 3).join(', ')}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                共 {knowledgeStats.popular_tags.length} 个热门标签
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 搜索和筛选区域 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="p-4">
          <form onSubmit={handleSearch} className="space-y-4">
            <div className="flex items-center space-x-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="搜索知识项..."
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                  />
                </div>
              </div>
              
              <button
                type="submit"
                disabled={isSearching}
                className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSearching ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  '搜索'
                )}
              </button>
              
              <button
                type="button"
                onClick={() => setShowSearchFilters(!showSearchFilters)}
                className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                <Filter className="w-4 h-4 inline mr-2" />
                筛选
                <ChevronDown className={`w-4 h-4 inline ml-2 transform transition-transform ${showSearchFilters ? 'rotate-180' : ''}`} />
              </button>
              
              <button
                type="button"
                onClick={() => loadKnowledgeItems(1)}
                disabled={isSearching}
                className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                title="刷新"
              >
                <RefreshCw className={`w-5 h-5 ${isSearching ? 'animate-spin' : ''}`} />
              </button>
            </div>

            {/* 高级筛选 */}
            {showSearchFilters && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    排序方式
                  </label>
                  <select
                    value={searchFilters.sortBy}
                    onChange={(e) => setSearchFilters(prev => ({ ...prev, sortBy: e.target.value as any }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                  >
                    <option value="relevance">相关性</option>
                    <option value="date">上传日期</option>
                    <option value="access">访问次数</option>
                    <option value="size">文件大小</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    排序顺序
                  </label>
                  <select
                    value={searchFilters.sortOrder}
                    onChange={(e) => setSearchFilters(prev => ({ ...prev, sortOrder: e.target.value as any }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                  >
                    <option value="desc">降序</option>
                    <option value="asc">升序</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    开始日期
                  </label>
                  <input
                    type="date"
                    value={searchFilters.startDate}
                    onChange={(e) => setSearchFilters(prev => ({ ...prev, startDate: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    结束日期
                  </label>
                  <input
                    type="date"
                    value={searchFilters.endDate}
                    onChange={(e) => setSearchFilters(prev => ({ ...prev, endDate: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                  />
                </div>
              </div>
            )}
          </form>

          {/* 搜索历史记录 */}
          {searchHistory.length > 0 && (
            <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <Calendar className="w-4 h-4 text-gray-400 mr-2" />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">最近搜索</span>
                </div>
                <button
                  onClick={clearSearchHistory}
                  className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                  title="清除所有搜索历史"
                >
                  清除
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {searchHistory.slice(0, 5).map((item, index) => (
                  <button
                    key={index}
                    onClick={() => useSearchHistoryItem(item)}
                    className="inline-flex items-center px-3 py-1.5 text-xs bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-full text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 hover:border-gray-700 dark:hover:border-gray-700 transition-colors"
                    title={`搜索于 ${item.timestamp.toLocaleString()}，找到 ${item.resultCount} 个结果`}
                  >
                    <Search className="w-3 h-3 mr-1.5" />
                    <span className="truncate max-w-[120px]">{item.query}</span>
                    <span className="ml-1.5 text-xs bg-gray-100 dark:bg-gray-700 rounded-full px-1.5 py-0.5">
                      {item.resultCount}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 类型筛选标签 */}
          <div className="mt-4">
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setActiveTab('all')}
                className={`px-3 py-1 text-sm rounded-full ${
                  activeTab === 'all'
                    ? 'bg-gradient-to-r from-gray-700 to-gray-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                全部
              </button>
              
              {Object.entries(typeStats).map(([type, count]) => (
                <button
                  key={type}
                  onClick={() => setActiveTab(type as any)}
                  className={`px-3 py-1 text-sm rounded-full flex items-center ${
                    activeTab === type
                      ? `${getTypeColor(type)} font-medium`
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {getTypeIcon(type)}
                  <span className="ml-1">{getTypeName(type)}</span>
                  <span className="ml-1 bg-gray-200 dark:bg-gray-600 rounded-full px-1.5 py-0.5 text-xs">
                    {count}
                  </span>
                </button>
              ))}
              
              {/* 可视化标签页 */}
              <button
                onClick={() => setActiveTab('visualization')}
                className={`px-3 py-1 text-sm rounded-full flex items-center ${
                  activeTab === 'visualization'
                    ? 'bg-gradient-to-r from-purple-600 to-pink-500 text-white font-medium'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                <Eye className="w-4 h-4" />
                <span className="ml-1">知识图谱</span>
              </button>
            </div>
          </div>

          {/* 标签筛选 */}
          {allTags.length > 0 && (
            <div className="mt-4">
              <div className="flex items-center mb-2">
                <Tag className="w-4 h-4 text-gray-400 mr-2" />
                <span className="text-sm text-gray-600 dark:text-gray-400">筛选标签:</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {allTags.map(tag => (
                  <button
                    key={tag}
                    onClick={() => handleTagToggle(tag)}
                    className={`px-3 py-1 text-sm rounded-full ${
                      selectedTags.includes(tag)
                        ? 'bg-gradient-to-r from-gray-700 to-gray-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 可视化或知识项列表 */}
      {activeTab === 'visualization' ? (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="font-semibold text-gray-900 dark:text-white mb-4">知识图谱可视化</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">展示知识项之间的关系，基于标签和类型相似性。</p>
          <KnowledgeGraph height={600} width={800} />
        </div>
      ) : (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <h2 className="font-semibold text-gray-900 dark:text-white">
            知识项 ({knowledgeItems.length})
          </h2>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {isSearching ? '搜索中...' : '已加载'}
          </div>
        </div>
        
        {isSearching && knowledgeItems.length === 0 ? (
          <div className="p-8 text-center">
            <Loader2 className="w-8 h-8 animate-spin text-gray-700 mx-auto" />
            <p className="mt-2 text-gray-600 dark:text-gray-400">搜索知识项中...</p>
          </div>
        ) : knowledgeItems.length === 0 ? (
          <div className="p-8 text-center">
            <Database className="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <h3 className="font-medium text-gray-900 dark:text-white">暂无知识项</h3>
            <p className="mt-1 text-gray-600 dark:text-gray-400">点击"上传知识"添加您的第一个知识项</p>
            <button
              onClick={() => setShowUploadModal(true)}
              className="mt-4 inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600"
            >
              <Upload className="w-4 h-4 mr-2" />
              上传知识
            </button>
          </div>
        ) : (
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {knowledgeItems.map(item => (
              <div key={item.id} className="p-4 hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <div className={`p-2 rounded-lg ${getTypeColor(item.type)}`}>
                        {getTypeIcon(item.type)}
                      </div>
                      <div>
                        <h3 
                          className="font-medium text-gray-900 dark:text-white cursor-pointer hover:text-gray-700 dark:hover:text-gray-700"
                          onClick={() => handleItemClick(item.id)}
                        >
                          {item.title}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          {item.description}
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex flex-wrap items-center gap-3 mt-3 text-sm text-gray-600 dark:text-gray-400">
                      <div className="flex items-center">
                        <Calendar className="w-4 h-4 mr-1" />
                        {formatDate(item.uploadDate)}
                      </div>
                      
                      <div className="flex items-center">
                        <Eye className="w-4 h-4 mr-1" />
                        访问: {item.accessCount}
                      </div>
                      
                      <div className="flex items-center">
                        <Database className="w-4 h-4 mr-1" />
                        {formatFileSize(item.size)}
                      </div>
                      
                      <div className="flex items-center">
                        上传者: {item.uploadedBy}
                      </div>
                      
                      {item.similarity !== undefined && (
                        <div className="flex items-center">
                          <Star className="w-4 h-4 mr-1 text-gray-500" />
                          相关度: {(item.similarity * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                    
                    <div className="flex flex-wrap gap-2 mt-3">
                      {item.tags.map(tag => (
                        <span
                          key={tag}
                          className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div className="ml-4 flex items-center space-x-2">
                    <button
                      onClick={() => handleItemClick(item.id)}
                      className="p-2 text-gray-700 bg-gray-700 dark:bg-gray-700/30 dark:text-gray-700 rounded-lg hover:bg-gray-700 dark:hover:bg-gray-700/50"
                      title="查看详情"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    
                    {item.fileUrl && (
                      <a
                        href={item.fileUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 text-gray-600 bg-gray-600 dark:bg-gray-700/30 dark:text-gray-600 rounded-lg hover:bg-gray-600 dark:hover:bg-gray-600/50"
                        title="下载文件"
                      >
                        <Download className="w-4 h-4" />
                      </a>
                    )}
                    
                    <button
                      onClick={() => handleEditClick(item)}
                      className="p-2 text-gray-500 bg-gray-500 dark:bg-gray-800/30 dark:text-gray-500 rounded-lg hover:bg-gray-500 dark:hover:bg-gray-500/50"
                      title="编辑"
                    >
                      <Edit className="w-4 h-4" />
                    </button>
                    
                    <button
                      onClick={() => handleDeleteItem(item.id)}
                      className="p-2 text-gray-800 bg-gray-800 dark:bg-gray-900/30 dark:text-gray-800 rounded-lg hover:bg-gray-800 dark:hover:bg-gray-800/50"
                      title="删除"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      )}

      {/* 上传模态框 */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  上传知识项
                </h2>
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <form onSubmit={handleUpload} className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    选择文件 *
                  </label>
                  <div className="flex items-center space-x-4">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="flex-1 px-4 py-3 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg text-center hover:border-gray-700 dark:hover:border-gray-700 transition-colors"
                    >
                      {uploadData.file ? (
                        <div className="text-left">
                          <div className="font-medium text-gray-900 dark:text-white">
                            {uploadData.file.name}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            {formatFileSize(uploadData.file.size)} • {getTypeName(uploadData.type)}
                          </div>
                        </div>
                      ) : (
                        <div className="py-8">
                          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                          <p className="text-gray-600 dark:text-gray-400">
                            点击选择文件，或拖放文件到此区域
                          </p>
                          <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                            支持图片、文档、代码、音频、视频等格式
                          </p>
                        </div>
                      )}
                    </button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    标题 *
                  </label>
                  <input
                    type="text"
                    value={uploadData.title}
                    onChange={(e) => setUploadData(prev => ({ ...prev, title: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                    placeholder="请输入文件标题"
                    required
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    描述
                  </label>
                  <textarea
                    value={uploadData.description}
                    onChange={(e) => setUploadData(prev => ({ ...prev, description: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                    rows={3}
                    placeholder="请输入文件描述"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    类型
                  </label>
                  <select
                    value={uploadData.type}
                    onChange={(e) => setUploadData(prev => ({ ...prev, type: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                  >
                    <option value="document">文档</option>
                    <option value="text">文本</option>
                    <option value="image">图片</option>
                    <option value="video">视频</option>
                    <option value="audio">音频</option>
                    <option value="code">代码</option>
                    <option value="dataset">数据集</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    标签
                  </label>
                  <div className="space-y-3">
                    <div className="flex flex-wrap gap-2 mb-2">
                      {uploadData.tags.map(tag => (
                        <span
                          key={tag}
                          className="inline-flex items-center px-3 py-1 bg-gray-700 dark:bg-gray-700/30 text-gray-700 dark:text-gray-700 rounded-full text-sm"
                        >
                          {tag}
                          <button
                            type="button"
                            onClick={() => handleRemoveUploadTag(tag)}
                            className="ml-2 text-gray-700 dark:text-gray-700 hover:text-gray-700 dark:hover:text-gray-700"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </span>
                      ))}
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      <input
                        type="text"
                        placeholder="输入标签后按Enter添加"
                        className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            e.preventDefault();
                            const input = e.target as HTMLInputElement;
                            handleAddTag(input.value);
                            input.value = '';
                          }
                        }}
                      />
                      <button
                        type="button"
                        onClick={() => {
                          const input = document.querySelector('input[placeholder="输入标签后按Enter添加"]') as HTMLInputElement;
                          if (input.value.trim()) {
                            handleAddTag(input.value.trim());
                            input.value = '';
                          }
                        }}
                        className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
                      >
                        添加
                      </button>
                    </div>
                    
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      热门标签: {allTags.slice(0, 5).join(', ')}
                    </div>
                  </div>
                </div>
                
                <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                  <div className="flex items-center justify-between">
                    <button
                      type="button"
                      onClick={() => setShowUploadModal(false)}
                      className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      取消
                    </button>
                    
                    <button
                      type="submit"
                      disabled={isUploading || !uploadData.file || !uploadData.title.trim()}
                      className="inline-flex items-center px-6 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isUploading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          上传中...
                        </>
                      ) : (
                        <>
                          <Upload className="w-4 h-4 mr-2" />
                          上传知识项
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* 编辑模态框 */}
      {showEditModal && editingItem && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  编辑知识项
                </h2>
                <button
                  onClick={() => setShowEditModal(false)}
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <form onSubmit={handleEditSubmit} className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    标题 *
                  </label>
                  <input
                    type="text"
                    name="title"
                    value={editData.title}
                    onChange={handleEditInputChange}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                    placeholder="请输入文件标题"
                    required
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    描述
                  </label>
                  <textarea
                    name="description"
                    value={editData.description}
                    onChange={handleEditInputChange}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                    rows={3}
                    placeholder="请输入文件描述"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    类型
                  </label>
                  <select
                    name="type"
                    value={editData.type}
                    onChange={handleEditInputChange}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                  >
                    <option value="document">文档</option>
                    <option value="text">文本</option>
                    <option value="image">图片</option>
                    <option value="video">视频</option>
                    <option value="audio">音频</option>
                    <option value="code">代码</option>
                    <option value="dataset">数据集</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    标签
                  </label>
                  <div className="space-y-3">
                    <div className="flex flex-wrap gap-2 mb-2">
                      {editData.tags.map(tag => (
                        <span
                          key={tag}
                          className="inline-flex items-center px-3 py-1 bg-gray-700 dark:bg-gray-700/30 text-gray-700 dark:text-gray-700 rounded-full text-sm"
                        >
                          {tag}
                          <button
                            type="button"
                            onClick={() => handleEditTagRemove(tag)}
                            className="ml-2 text-gray-700 dark:text-gray-700 hover:text-gray-700 dark:hover:text-gray-700"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </span>
                      ))}
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      <input
                        type="text"
                        placeholder="输入标签后按Enter添加"
                        className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-gray-700"
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            e.preventDefault();
                            const input = e.target as HTMLInputElement;
                            handleEditTagAdd(input.value);
                            input.value = '';
                          }
                        }}
                      />
                      <button
                        type="button"
                        onClick={() => {
                          const input = document.querySelector('input[placeholder="输入标签后按Enter添加"]') as HTMLInputElement;
                          if (input.value.trim()) {
                            handleEditTagAdd(input.value.trim());
                            input.value = '';
                          }
                        }}
                        className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
                      >
                        添加
                      </button>
                    </div>
                    
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      热门标签: {allTags.slice(0, 5).join(', ')}
                    </div>
                  </div>
                </div>
                
                <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                  <div className="flex items-center justify-between">
                    <button
                      type="button"
                      onClick={() => setShowEditModal(false)}
                      className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      取消
                    </button>
                    
                    <button
                      type="submit"
                      className="inline-flex items-center px-6 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg hover:from-gray-700 hover:to-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-700"
                    >
                      <Edit className="w-4 h-4 mr-2" />
                      更新知识项
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* 详情模态框 */}
      {showDetailModal && selectedItemDetail && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  知识项详情
                </h2>
                <button
                  onClick={() => setShowDetailModal(false)}
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    {selectedItemDetail.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    {selectedItemDetail.description || '无描述'}
                  </p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      基本信息
                    </h4>
                    <dl className="space-y-2">
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">类型</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedItemDetail.type === 'text' ? '文本' :
                           selectedItemDetail.type === 'image' ? '图片' :
                           selectedItemDetail.type === 'video' ? '视频' :
                           selectedItemDetail.type === 'audio' ? '音频' :
                           selectedItemDetail.type === 'document' ? '文档' :
                           selectedItemDetail.type === 'code' ? '代码' :
                           selectedItemDetail.type === 'dataset' ? '数据集' : selectedItemDetail.type}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">大小</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedItemDetail.size > 1024 * 1024 
                            ? `${(selectedItemDetail.size / (1024 * 1024)).toFixed(2)} MB`
                            : selectedItemDetail.size > 1024
                            ? `${(selectedItemDetail.size / 1024).toFixed(2)} KB`
                            : `${selectedItemDetail.size} 字节`}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">上传日期</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedItemDetail.uploadDate.toLocaleDateString('zh-CN')}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">访问次数</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedItemDetail.accessCount}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-sm text-gray-500 dark:text-gray-400">最后访问</dt>
                        <dd className="text-sm text-gray-900 dark:text-white">
                          {selectedItemDetail.lastAccessed.toLocaleDateString('zh-CN')}
                        </dd>
                      </div>
                    </dl>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      标签
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedItemDetail.tags.map(tag => (
                        <span
                          key={tag}
                          className="inline-flex items-center px-3 py-1 bg-gray-700 dark:bg-gray-700/30 text-gray-700 dark:text-gray-700 rounded-full text-sm"
                        >
                          {tag}
                        </span>
                      ))}
                      {selectedItemDetail.tags.length === 0 && (
                        <span className="text-sm text-gray-500 dark:text-gray-400">无标签</span>
                      )}
                    </div>
                    
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mt-4 mb-2">
                      内容预览
                    </h4>
                    <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 max-h-60 overflow-y-auto">
                      <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap break-words">
                        {selectedItemDetail.content || '无内容'}
                      </pre>
                    </div>
                  </div>
                </div>
                
                <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      上传者: {selectedItemDetail.uploadedBy}
                    </div>
                    
                    <div className="flex space-x-4">
                      <button
                        onClick={() => {
                          setShowDetailModal(false);
                          handleEditClick(selectedItemDetail);
                        }}
                        className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-800 bg-gray-500 dark:bg-gray-800/30 dark:text-gray-500 rounded-lg hover:bg-gray-500 dark:hover:bg-gray-500/50"
                      >
                        <Edit className="w-4 h-4 mr-2" />
                        编辑
                      </button>
                      <button
                        onClick={() => {
                          // 这里可以添加下载功能
                          if (selectedItemDetail.fileUrl) {
                            window.open(selectedItemDetail.fileUrl, '_blank');
                          } else {
                            toast('文件不可用');
                          }
                        }}
                        className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-gray-600 dark:bg-gray-700/30 dark:text-gray-600 rounded-lg hover:bg-gray-600 dark:hover:bg-gray-600/50"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        下载
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default KnowledgePage;