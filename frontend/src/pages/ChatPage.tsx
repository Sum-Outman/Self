import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  Send,
  Mic,
  MicOff,
  Image as ImageIcon,
  Bot,
  User,
  Brain,
  Clock,
  RefreshCw,
  Settings,
  Loader2,
  MessageSquare,
  Plus,
  Trash2,
  Video,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { chatApi } from '../services/api/chat';
import VideoChat from '../components/Video/VideoChat';
import {
  // 转换函数
  convertBackendMessageToUi,
  convertBackendSessionToUi,
  convertBackendModelToUi,
  convertBackendSystemModeToUi,
} from '../types/chat';
// 扩展Window接口以支持录音功能
declare global {
  interface Window {
    currentMediaRecorder?: MediaRecorder;
    currentAudioStream?: MediaStream;
  }
}

import type {
  UiMessage,
  UiChatSession,
  UiModelInfo,
  BackendChatRequest,
  BackendChatSession,
  SystemMode,
  ApiResponse,
  BackendFileUploadResponse,
} from '../types/chat';

// 错误类型守卫
function isErrorWithMessage(error: unknown): error is { 
  message?: string; 
  name?: string; 
  code?: string | number;
  response?: { 
    status?: number; 
    data?: { detail?: string } 
  } 
} {
  return typeof error === 'object' && error !== null;
}

// 聊天模式（兼容现有代码）
type ChatMode = 'text' | 'video';

// 类型安全转换函数


const convertApiSessionToUi = (apiSession: Partial<BackendChatSession>): UiChatSession => {
  // 使用类型安全的转换函数
  const backendSession: BackendChatSession = {
    id: apiSession.id || Date.now().toString(),
    title: apiSession.title || apiSession.last_message?.substring(0, 30) || '新对话',
    model_name: apiSession.model_name || '默认模型',
    created_at: apiSession.created_at || new Date().toISOString(),
    last_message: apiSession.last_message || '',
    message_count: apiSession.message_count || 0,
    metadata: apiSession.metadata,
  };
  return convertBackendSessionToUi(backendSession);
};

const ChatPage: React.FC = () => {
  const { user: _user } = useAuth();
  // 消息按会话存储：{ [sessionId]: UiMessage[] }
  const [messagesBySession, setMessagesBySession] = useState<Record<string, UiMessage[]>>({});
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [sessions, setSessions] = useState<UiChatSession[]>([]);
  const [activeSession, setActiveSession] = useState('');
  
  // 获取当前活动会话的消息
  const currentMessages = messagesBySession[activeSession] || [];
  const [availableModels, setAvailableModels] = useState<UiModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [useMemory, setUseMemory] = useState(true);
  const [systemMode, setSystemMode] = useState<SystemMode>('task');
  
  // 新状态：聊天模式
  const [chatMode, setChatMode] = useState<ChatMode>('text');
  
  // 消息操作状态
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingMessageContent, setEditingMessageContent] = useState('');
  const [messageToDelete, setMessageToDelete] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<{
    messageId: string;
    x: number;
    y: number;
  } | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // WebSocket状态
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [webSocketError, setWebSocketError] = useState<string | null>(null);
  const [pendingMessageIds, setPendingMessageIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    // 滚动到底部
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [currentMessages]);

  useEffect(() => {
    let isMounted = true;
    
    // 加载模型列表
    const loadModels = async () => {
      try {
        const response = await chatApi.getModels();
        if (isMounted && response.success && response.data) {
          // 使用类型安全转换函数映射后端数据到前端格式
          const mappedModels = response.data.map((model: any) => 
            convertBackendModelToUi({
              id: model.id,
              name: model.name || model.id,
              description: model.description || 'AGI模型',
              provider: model.provider || 'Self AGI',
              max_tokens: model.max_tokens || 4096,
              supports_multimodal: model.supports_multimodal || false,
              is_available: model.is_available !== false,
              parameters: model.parameters,
              capabilities: model.capabilities || [],
            })
          );
          setAvailableModels(mappedModels);
          if (mappedModels.length > 0 && !selectedModel) {
            setSelectedModel(mappedModels[0].id);
          }
        }
      } catch (error) {
        if (isMounted) {
          console.error('加载模型列表失败:', error);
        }
      }
    };
    
    // 加载系统模式
    const loadSystemMode = async () => {
      try {
        const response = await chatApi.getSystemMode();
        if (isMounted && response.success && response.data) {
          // 使用类型安全转换函数映射后端系统模式
          const uiSystemMode = convertBackendSystemModeToUi(response.data as any);
          setSystemMode(uiSystemMode.mode);
        }
      } catch (error) {
        if (isMounted) {
          console.error('加载系统模式失败:', error);
        }
      }
    };

    // 加载会话列表
    const loadSessions = async () => {
      try {
        const response = await chatApi.getSessions();
        if (isMounted && response.success && response.data) {
          // 使用类型安全转换函数映射后端会话数据
          const mappedSessions = response.data.map((session: any) =>
            convertBackendSessionToUi({
              id: session.id,
              title: session.title || '新会话',
              model_name: session.model_name || '默认模型',
              created_at: session.created_at || new Date().toISOString(),
              last_message: session.last_message || '开始对话',
              message_count: session.message_count || 0,
            })
          );
          setSessions(mappedSessions);
          
          // 如果有会话，设置第一个为活动会话
          if (mappedSessions.length > 0 && !activeSession) {
            setActiveSession(mappedSessions[0].id);
          } else if (mappedSessions.length === 0) {
            // 如果没有会话，创建一个新会话
            const createResponse = await chatApi.createSession('默认对话', selectedModel || '默认模型');
            if (createResponse.success && createResponse.data) {
              const newSession = convertBackendSessionToUi(createResponse.data);
              setSessions([newSession]);
              setActiveSession(newSession.id);
            }
          }
        }
      } catch (error) {
        if (isMounted) {
          console.error('加载会话列表失败:', error);
        }
      }
    };
    
    loadModels();
    loadSystemMode();
    loadSessions();
    
    // 清理函数
    return () => {
      isMounted = false;
      // 取消任何待处理的请求
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
        console.log('组件卸载：已取消待处理请求');
      }
    };
  }, []);

  // WebSocket连接和事件监听
  useEffect(() => {
    let isMounted = true;
    let reconnectTimeout: NodeJS.Timeout | null = null;

    const connectWebSocket = async () => {
      try {
        if (!isMounted) return;
        
        setWebSocketError(null);
        const connected = await chatApi.connectToChatWebSocket();
        
        if (isMounted && connected) {
          setIsWebSocketConnected(true);
          console.log('WebSocket连接成功');
          
          // 设置事件监听器
          chatApi.addEventListener('connection_established', handleWebSocketConnection);
          chatApi.addEventListener('connection_closed', handleWebSocketDisconnection);
          chatApi.addEventListener('ai_response', handleAIResponse);
          chatApi.addEventListener('typing_status', handleTypingStatus);
          chatApi.addEventListener('error', handleWebSocketError);
          chatApi.addEventListener('message_received', handleMessageReceived);
          
          // 连接特定会话的WebSocket
          if (activeSession && activeSession !== 'default-session') {
            await chatApi.connectToSessionWebSocket(activeSession);
          }
        }
      } catch (error) {
        if (isMounted) {
          console.error('WebSocket连接失败:', error);
          setWebSocketError('WebSocket连接失败，将使用HTTP回退');
          setIsWebSocketConnected(false);
          
          // 5秒后重试
          reconnectTimeout = setTimeout(() => {
            if (isMounted) {
              connectWebSocket();
            }
          }, 5000);
        }
      }
    };

    // WebSocket事件处理函数
    const handleWebSocketConnection = (_event: any) => {
      if (isMounted) {
        setIsWebSocketConnected(true);
        setWebSocketError(null);
        toast.success('实时连接已建立');
      }
    };

    const handleWebSocketDisconnection = (event: any) => {
      if (isMounted) {
        setIsWebSocketConnected(false);
        if (event.code !== 1000) { // 非正常关闭
          toast.error(`连接断开，代码: ${event.code}`);
        }
      }
    };

    const handleAIResponse = (event: any) => {
      if (!isMounted) return;
      
      const { message_id, content, session_id, processing_time } = event;
      
      // 检查是否是我们等待的响应
      if (pendingMessageIds.has(message_id)) {
        // 移除pending状态
        setPendingMessageIds(prev => {
          const newSet = new Set(prev);
          newSet.delete(message_id);
          return newSet;
        });
        
        // 创建AI消息
        const aiMessage: UiMessage = {
          id: `ai_${Date.now()}`,
          content: content || '收到回复',
          sender: 'ai',
          timestamp: new Date(),
          type: 'text',
          status: 'sent',
        };
        
        // 更新消息列表
        setMessagesBySession(prev => ({
          ...prev,
          [session_id || activeSession]: [...(prev[session_id || activeSession] || []), aiMessage]
        }));
        
        // 更新用户消息状态
        setMessagesBySession(prev => ({
          ...prev,
          [activeSession]: (prev[activeSession] || []).map(msg => 
            msg.id === message_id ? { ...msg, status: 'sent' } : msg
          )
        }));
        
        // 更新会话列表
        if (content) {
          setSessions(prev => 
            prev.map(session => 
              session.id === (session_id || activeSession)
                ? { 
                    ...session, 
                    lastMessage: content.length > 30 
                      ? `${content.substring(0, 30)}...` 
                      : content 
                  }
                : session
            )
          );
        }
        
        toast.success(`回复接收成功 (${processing_time ? processing_time.toFixed(2) + '秒' : '实时'})`);
        setIsLoading(false);
      }
    };

    const handleTypingStatus = (event: any) => {
      if (isMounted) {
        setIsTyping(event.is_typing || false);
      }
    };

    const handleWebSocketError = (event: any) => {
      if (isMounted) {
        console.error('WebSocket错误:', event);
        setWebSocketError(event.message || 'WebSocket连接错误');
      }
    };

    const handleMessageReceived = (event: any) => {
      if (isMounted && event.message_id) {
        // 服务器确认收到消息
        setMessagesBySession(prev => ({
          ...prev,
          [activeSession]: (prev[activeSession] || []).map(msg => 
            msg.id === event.message_id ? { ...msg, status: 'sent' } : msg
          )
        }));
      }
    };

    // 初始连接
    connectWebSocket();

    // 清理函数
    return () => {
      isMounted = false;
      
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      
      // 移除事件监听器
      chatApi.removeEventListener('connection_established', handleWebSocketConnection);
      chatApi.removeEventListener('connection_closed', handleWebSocketDisconnection);
      chatApi.removeEventListener('ai_response', handleAIResponse);
      chatApi.removeEventListener('typing_status', handleTypingStatus);
      chatApi.removeEventListener('error', handleWebSocketError);
      chatApi.removeEventListener('message_received', handleMessageReceived);
      
      // 断开WebSocket连接
      chatApi.disconnectChatWebSocket();
      chatApi.disconnectSessionWebSocket();
      
      console.log('WebSocket连接已清理');
    };
  }, [activeSession]); // 当activeSession改变时重新连接会话WebSocket

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // 防重复提交检查
    if (isLoading) {
      toast.error('请等待当前消息发送完成');
      return;
    }
    
    if (!inputMessage.trim()) {
      toast.error('请输入消息内容');
      return;
    }
    
    const newMessage: UiMessage = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date(),
      type: 'text',
      status: 'sending',
    };
    
    // 取消之前的请求（如果有）
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      console.log('已取消之前的请求');
    }
    
    // 创建新的AbortController用于当前请求
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    
    setMessagesBySession(prev => ({
      ...prev,
      [activeSession]: [...(prev[activeSession] || []), newMessage]
    }));
    setInputMessage('');
    setIsLoading(true);
    
    try {
      // WebSocket发送逻辑（如果连接）
      if (isWebSocketConnected) {
        const messageId = newMessage.id;
        
        // 添加到pending消息集合
        setPendingMessageIds(prev => {
          const newSet = new Set(prev);
          newSet.add(messageId);
          return newSet;
        });
        
        try {
          // 通过WebSocket发送消息
          chatApi.sendChatMessageViaWebSocket({
            content: inputMessage,
            session_id: activeSession,
            model_name: selectedModel,
            message_id: messageId
          });
          
          // 更新用户消息状态为发送中（等待服务器确认）
          // 服务器确认将通过message_received事件处理
          
          // 发送打字指示器
          chatApi.sendTypingIndicator(true);
          setTimeout(() => {
            chatApi.sendTypingIndicator(false);
          }, 1000);
          
          // WebSocket发送成功，提前返回
          // AI响应将通过ai_response事件处理
          return;
        } catch (wsError) {
          console.error('WebSocket发送失败，回退到HTTP:', wsError);
          // 继续执行HTTP回退逻辑
          setWebSocketError('WebSocket发送失败，使用HTTP回退');
        }
      }
      
      // HTTP回退逻辑
      // 使用chatApi发送消息
      const chatRequest: BackendChatRequest = {
        message: inputMessage,
        model_name: selectedModel,
        use_memory: useMemory,
        session_id: activeSession,
        temperature: 0.7,
        max_tokens: 1000
      };
      
      // 添加超时处理（60秒超时）
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('请求超时（60秒）')), 60000);
      });
      
      const sendMessagePromise = chatApi.sendMessage(chatRequest, abortController.signal);
      const response = await Promise.race([sendMessagePromise, timeoutPromise]);
      
      // 更新用户消息状态
      setMessagesBySession(prev => ({
        ...prev,
        [activeSession]: (prev[activeSession] || []).map(msg => 
          msg.id === newMessage.id ? { ...msg, status: 'sent' } : msg
        )
      }));
      
      // 添加AI回复（使用类型转换函数）
      const apiMessage = {
        id: (Date.now() + 1).toString(),
        content: response.response,
        sender: 'assistant' as 'assistant', // API使用'assistant'，前端使用'ai'
        timestamp: new Date().toISOString(),
        type: 'text' as 'text',
        status: 'sent' as 'sent',
      };
      const aiMessage = convertBackendMessageToUi(apiMessage);
      
      setMessagesBySession(prev => ({
        ...prev,
        [activeSession]: [...(prev[activeSession] || []), aiMessage]
      }));
      
      // 更新会话列表
      setSessions(prev => 
        prev.map(session => 
          session.id === activeSession 
            ? { 
                ...session, 
                lastMessage: inputMessage.length > 30 
                  ? `${inputMessage.substring(0, 30)}...` 
                  : inputMessage 
              }
            : session
        )
      );
      
      toast.success(`回复接收成功 (${response.processing_time.toFixed(2)}秒)`);
      
    } catch (error: unknown) {
      console.error('发送消息失败:', error);
      
      // 显示更具体的错误信息
      let errorMessage = '发送消息失败，请重试';
      let errorType = 'unknown';
      
      // 1. 超时错误
      if (isErrorWithMessage(error) && (error.message?.includes('超时') || error.code === 'ECONNABORTED' || error.name === 'TimeoutError')) {
        errorMessage = '请求超时，服务器响应时间过长（60秒），请稍后重试';
        errorType = 'timeout';
      }
      // 2. 请求取消错误（AbortController）
      else if (isErrorWithMessage(error) && (error.name === 'AbortError' || error.code === 'ERR_CANCELED')) {
        errorMessage = '请求已被取消';
        errorType = 'aborted';
        console.log('请求被用户取消');
        return; // 取消请求不需要显示错误消息
      }
      // 3. HTTP状态码错误
      else if (isErrorWithMessage(error) && error.response) {
        errorType = 'http';
        if (error.response.status === 400) {
          errorMessage = '请求格式错误，请检查输入';
        } else if (error.response.status === 401) {
          errorMessage = '认证失败，请重新登录';
        } else if (error.response.status === 403) {
          errorMessage = '权限不足，无法访问此资源';
        } else if (error.response.status === 404) {
          errorMessage = '模型不存在或未加载，请检查模型配置';
        } else if (error.response.status === 413) {
          errorMessage = '请求数据过大，请减少消息长度';
        } else if (error.response.status === 415) {
          errorMessage = '不支持的媒体类型';
        } else if (error.response.status === 429) {
          errorMessage = '请求过于频繁，请稍后重试';
        } else if (error.response.status === 503) {
          errorMessage = '服务器暂时不可用，请稍后重试';
        } else if (error.response.status === 504) {
          errorMessage = '网关超时，服务器处理时间过长';
        } else if (error.response.data?.detail) {
          errorMessage = error.response.data.detail;
        } else if (error.response.status && error.response.status >= 500) {
          errorMessage = `服务器内部错误 (${error.response.status})`;
        }
      }
      // 4. 网络错误
      else if (isErrorWithMessage(error) && (error.name === 'NetworkError' || error.message?.includes('network') || error.code === 'NETWORK_ERROR')) {
        errorMessage = '网络连接失败，请检查网络连接';
        errorType = 'network';
      }
      // 5. 解析错误（JSON解析失败）
      else if (isErrorWithMessage(error) && (error.name === 'SyntaxError' || error.message?.includes('JSON') || error.message?.includes('parse'))) {
        errorMessage = '服务器响应格式错误，无法解析';
        errorType = 'parse';
      }
      // 6. CORS错误
      else if (isErrorWithMessage(error) && (error.name === 'SecurityError' || error.message?.includes('CORS') || error.message?.includes('cross-origin'))) {
        errorMessage = '跨域请求被阻止，请检查服务器CORS配置';
        errorType = 'cors';
      }
      // 7. SSL/TLS错误
      else if (isErrorWithMessage(error) && (error.message?.includes('SSL') || error.message?.includes('TLS') || error.message?.includes('certificate'))) {
        errorMessage = '安全连接失败，SSL证书错误';
        errorType = 'ssl';
      }
      // 8. 浏览器兼容性错误
      else if (isErrorWithMessage(error) && (error.name === 'NotSupportedError' || error.message?.includes('不支持') || error.message?.includes('not supported'))) {
        errorMessage = '浏览器不支持此功能，请升级浏览器或使用其他浏览器';
        errorType = 'unsupported';
      }
      // 9. 类型错误（fetch API问题）
      else if (isErrorWithMessage(error) && error.name === 'TypeError' && error.message?.includes('fetch')) {
        errorMessage = '无法连接到服务器，请检查服务器状态';
        errorType = 'connection';
      }
      // 10. 通用错误
      else {
        errorMessage = isErrorWithMessage(error) ? `发送消息失败: ${error.message || '未知错误'}` : '发送消息失败: 未知错误';
        errorType = 'generic';
      }
      
      // 记录错误详情（生产环境可发送到错误监控服务）
      console.error(`聊天错误 [${errorType}]:`, error);
      
      // 显示错误消息（除了取消请求）
      if (errorType !== 'aborted') {
        toast.error(errorMessage);
      }
      
      // 更新消息状态为错误（除了取消请求）
      if (errorType !== 'aborted') {
        setMessagesBySession(prev => ({
          ...prev,
          [activeSession]: (prev[activeSession] || []).map(msg => 
            msg.id === newMessage.id ? { ...msg, status: 'error' } : msg
          )
        }));
      }
      
      // 如果是网络错误，建议用户检查连接
      if (errorType === 'network' || errorType === 'connection') {
        setTimeout(() => {
          toast('建议：请检查网络连接并刷新页面重试', { 
            icon: 'ℹ️',
            duration: 5000 
          });
        }, 1000);
      }
      // 如果是认证错误，建议重新登录
      else if (errorType === 'http' && isErrorWithMessage(error) && error.response?.status === 401) {
        setTimeout(() => {
          toast('认证已过期，请重新登录', { 
            icon: '🔑',
            duration: 5000 
          });
        }, 1000);
      }
    } finally {
      setIsLoading(false);
      
      // 清理AbortController引用（如果当前请求已完成）
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
      }
    }
  };

  const handleEditMessage = (messageId: string, content: string) => {
    setEditingMessageId(messageId);
    setEditingMessageContent(content);
    setContextMenu(null); // 关闭上下文菜单
  };

  const handleSaveEdit = () => {
    if (!editingMessageId || !editingMessageContent.trim()) {
      toast.error('消息内容不能为空');
      return;
    }

    setMessagesBySession(prev => ({
      ...prev,
      [activeSession]: (prev[activeSession] || []).map(msg =>
        msg.id === editingMessageId
          ? { ...msg, content: editingMessageContent, status: 'sent' }
          : msg
      )
    }));

    setEditingMessageId(null);
    setEditingMessageContent('');
    toast.success('消息已更新');
  };

  const handleCancelEdit = () => {
    setEditingMessageId(null);
    setEditingMessageContent('');
  };

  const handleDeleteMessage = (messageId: string) => {
    setMessagesBySession(prev => ({
      ...prev,
      [activeSession]: (prev[activeSession] || []).filter(msg => msg.id !== messageId)
    }));
    setMessageToDelete(null);
    toast.success('消息已删除');
  };

  const handleContextMenu = (e: React.MouseEvent, messageId: string) => {
    e.preventDefault();
    setContextMenu({
      messageId,
      x: e.clientX,
      y: e.clientY
    });
  };

  const handleCopyMessage = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      toast.success('消息已复制到剪贴板');
      setContextMenu(null);
    } catch (error) {
      console.error('复制失败:', error);
      toast.error('复制失败，请手动复制');
    }
  };

  const handleVoiceInput = async () => {
    if (!isRecording) {
      // 开始录音
      try {
        setIsRecording(true);
        toast.loading('正在录音... 请说话', { id: 'recording', duration: 10000 });
        
        // 1. 请求麦克风权限并开始录音
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        const audioChunks: Blob[] = [];
        
        // 设置录音超时（最长30秒）
        const recordingTimeout = setTimeout(() => {
          if (isRecording) {
            mediaRecorder.stop();
            toast.dismiss('recording');
            toast('录音时间过长，已自动停止（最长30秒）', { icon: '⏱️' });
          }
        }, 30000);
        
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunks.push(event.data);
          }
        };
        
        mediaRecorder.onstop = async () => {
          clearTimeout(recordingTimeout);
          
          if (audioChunks.length === 0) {
            setIsRecording(false);
            toast.dismiss('recording');
            toast.error('未检测到音频数据');
            return;
          }
          
          // @ts-ignore - 变量在注释的示例代码中使用
          const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
          setIsRecording(false);
          
          try {
            toast.loading('正在识别语音...', { id: 'transcribing' });
            
            // 2. 将音频发送到后端进行语音识别
            // 在实际应用中，这里应该调用后端的语音识别API
            // const formData = new FormData();
            // formData.append('audio', audioBlob, 'recording.webm');
            // const response = await fetch('/api/speech-to-text', {
            //   method: 'POST',
            //   body: formData,
            // });
            // const result = await response.json();
            // const transcript = result.transcript || '';
            
            // 使用真实的语音识别API
            try {
              const response = await chatApi.speechToText(audioBlob);
              
              if (response.success && response.data?.text) {
                const transcript = response.data.text;
                toast.dismiss('transcribing');
                setInputMessage(transcript);
                toast.success('语音识别完成');
                // 停止所有音轨
                stream.getTracks().forEach(track => track.stop());
                return; // 成功，直接返回
              } else {
                throw new Error(response.message || '语音识别失败');
              }
            } catch (apiError) {
              console.error('语音识别API调用失败:', apiError);
              // API调用失败，提供用户友好的消息
              toast.dismiss('transcribing');
              setInputMessage('');
              toast.error('语音识别服务暂时不可用，请手动输入');
              // 继续执行以停止音轨
            }
            
          } catch (error) {
            console.error('语音识别失败:', error);
            toast.dismiss('transcribing');
            toast.error('语音识别失败，请重试或手动输入');
          } finally {
            // 停止所有音轨
            stream.getTracks().forEach(track => track.stop());
          }
        };
        
        // 开始录音
        mediaRecorder.start();
        
        // 保存引用以便后续停止
        window.currentMediaRecorder = mediaRecorder;
        window.currentAudioStream = stream;
        
      } catch (error: unknown) {
        console.error('启动录音失败:', error);
        setIsRecording(false);
        toast.dismiss('recording');
        
        if (isErrorWithMessage(error)) {
          if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            toast.error('麦克风权限被拒绝，请在浏览器设置中允许麦克风访问');
          } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            toast.error('未找到可用的麦克风设备');
          } else {
            toast.error(`录音失败：${error.message || '未知错误'}`);
          }
        } else {
          toast.error('录音失败：未知错误');
        }
      }
    } else {
      // 停止录音
      try {
        if (window.currentMediaRecorder && 
            window.currentMediaRecorder.state !== 'inactive') {
          window.currentMediaRecorder.stop();
        }
        
        // 停止音频流
        if (window.currentAudioStream) {
          window.currentAudioStream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
          delete window.currentAudioStream;
        }
        
        setIsRecording(false);
        toast.dismiss('recording');
        toast.success('录音已停止，正在识别...');
        
      } catch (error: unknown) {
        console.error('停止录音失败:', error);
        setIsRecording(false);
        toast.dismiss('recording');
        toast.error('停止录音失败');
      }
    }
  };

  const handleNewSession = async () => {
    try {
      const response = await chatApi.createSession(`新会话 ${sessions.length + 1}`);
      if (response.success && response.data) {
        const newSession = convertApiSessionToUi(response.data);
        
        setSessions(prev => [newSession, ...prev]);
        setActiveSession(newSession.id);
        setMessagesBySession(prev => ({ ...prev, [newSession.id]: [] }));
        toast.success('已创建新会话');
      }
    } catch (error) {
      console.error('创建会话失败:', error);
      toast.error('创建会话失败，请检查网络连接并重试');
      // 不创建本地临时会话，保持用户界面不变，等待用户重试
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    if (sessionId === activeSession) {
      toast.error('不能删除当前活跃会话');
      return;
    }
    
    try {
      const response = await chatApi.deleteSession(sessionId);
      if (response.success) {
        setSessions(prev => prev.filter(session => session.id !== sessionId));
        // 同时删除该会话的消息记录
        setMessagesBySession(prev => {
          const newMessages = { ...prev };
          delete newMessages[sessionId];
          return newMessages;
        });
        toast.success('会话已删除');
      }
    } catch (error) {
      console.error('删除会话失败:', error);
      
      // 如果API调用失败，本地删除
      setSessions(prev => prev.filter(session => session.id !== sessionId));
      // 同时删除该会话的消息记录
      setMessagesBySession(prev => {
        const newMessages = { ...prev };
        delete newMessages[sessionId];
        return newMessages;
      });
      toast.success('会话已删除（本地）');
    }
  };

  const handleSelectSession = (sessionId: string) => {
    if (sessionId === activeSession) return; // 已经是当前会话
    
    setActiveSession(sessionId);
    // 消息会自动切换，因为currentMessages依赖于activeSession
    toast.success(`已切换到会话: ${sessions.find(s => s.id === sessionId)?.title || '未命名会话'}`);
  };

  const handleToggleMemory = () => {
    const newUseMemory = !useMemory;
    setUseMemory(newUseMemory);
    toast.success(`记忆系统 ${newUseMemory ? '已启用' : '已禁用'}`);
  };

  const handleToggleSystemMode = async () => {
    const newMode = systemMode === 'task' ? 'autonomous' : 'task';
    
    try {
      // 前端模式映射到后端模式
      const backendMode = newMode === 'task' ? 'assist' : 'autonomous';
      const response = await chatApi.setSystemMode(backendMode);
      if (response.success) {
        setSystemMode(newMode);
        toast.success(`系统模式已切换为：${newMode === 'task' ? '任务执行模式' : '全自主模式'}`);
      } else {
        toast.error('切换系统模式失败');
      }
    } catch (error) {
      console.error('切换系统模式失败:', error);
      toast.error('切换系统模式失败，使用本地模式');
      setSystemMode(newMode);
      toast.success(`系统模式已切换为：${newMode === 'task' ? '任务执行模式' : '全自主模式'}（本地）`);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    // 1. 文件大小限制检查（10MB）
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    if (file.size > MAX_FILE_SIZE) {
      toast.error(`文件大小超过限制（最大10MB），当前文件：${(file.size / (1024 * 1024)).toFixed(2)}MB`);
      event.target.value = ''; // 清除文件选择
      return;
    }
    
    // 2. 文件类型白名单检查
    const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'];
    const ALLOWED_AUDIO_TYPES = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/webm'];
    const ALLOWED_VIDEO_TYPES = ['video/mp4', 'video/webm', 'video/ogg'];
    
    const fileType = file.type.split('/')[0];
    let type: 'image' | 'audio' | 'video' = 'image';
    
    if (fileType === 'audio') {
      if (!ALLOWED_AUDIO_TYPES.includes(file.type)) {
        toast.error(`不支持的音频格式：${file.type}，支持格式：${ALLOWED_AUDIO_TYPES.join(', ')}`);
        event.target.value = '';
        return;
      }
      type = 'audio';
    } else if (fileType === 'video') {
      if (!ALLOWED_VIDEO_TYPES.includes(file.type)) {
        toast.error(`不支持的视频格式：${file.type}，支持格式：${ALLOWED_VIDEO_TYPES.join(', ')}`);
        event.target.value = '';
        return;
      }
      type = 'video';
    } else if (fileType === 'image') {
      if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
        toast.error(`不支持的图片格式：${file.type}，支持格式：${ALLOWED_IMAGE_TYPES.join(', ')}`);
        event.target.value = '';
        return;
      }
      type = 'image';
    } else {
      toast.error(`不支持的文件类型：${file.type}`);
      event.target.value = '';
      return;
    }
    
    // 3. 文件名安全检查（防止路径遍历攻击）
    const fileName = file.name;
    if (fileName.includes('..') || fileName.includes('/') || fileName.includes('\\')) {
      toast.error('文件名包含非法字符');
      event.target.value = '';
      return;
    }
    
    // 4. 文件扩展名检查
    const fileExtension = fileName.split('.').pop()?.toLowerCase();
    const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg', 'mp3', 'wav', 'ogg', 'webm', 'mp4'];
    if (!fileExtension || !ALLOWED_EXTENSIONS.includes(fileExtension)) {
      toast.error(`不支持的文件扩展名：${fileExtension}，支持扩展名：${ALLOWED_EXTENSIONS.join(', ')}`);
      event.target.value = '';
      return;
    }
    
    try {
      toast.loading('上传文件中...', { id: 'upload' });
      
      // 添加超时处理
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('文件上传超时（30秒）')), 30000);
      });
      
      const uploadPromise = chatApi.uploadFile(file, type);
      const response = await Promise.race([uploadPromise, timeoutPromise]) as ApiResponse<BackendFileUploadResponse>;
      
      toast.dismiss('upload');
      
      if (response.success && response.data?.url) {
        // 将文件URL添加到消息中
        const fileMessage = `[${type}: ${file.name}](${response.data.url})`;
        setInputMessage(prev => prev ? `${prev} ${fileMessage}` : fileMessage);
        
        // 添加文件预览消息
        const filePreviewMessage: UiMessage = {
          id: Date.now().toString(),
          content: `已上传${type === 'image' ? '图片' : type === 'audio' ? '音频' : '视频'}文件: ${file.name} (${(file.size / 1024).toFixed(1)}KB)`,
          sender: 'user',
          timestamp: new Date(),
          type,
          status: 'sent',
        };
        
        setMessagesBySession(prev => ({
          ...prev,
          [activeSession]: [...(prev[activeSession] || []), filePreviewMessage]
        }));
        
        toast.success('文件上传成功');
      } else {
        toast.error('文件上传失败：服务器返回错误');
      }
    } catch (error: any) {
      console.error('文件上传失败:', error);
      toast.dismiss('upload');
      
      if (error.message?.includes('超时')) {
        toast.error('文件上传超时，请检查网络连接后重试');
      } else if (error.response?.status === 413) {
        toast.error('文件太大，服务器拒绝接收');
      } else if (error.response?.status === 415) {
        toast.error('不支持的媒体类型');
      } else {
        toast.error(`文件上传失败：${error.message || '未知错误'}`);
      }
    } finally {
      // 清除文件输入
      event.target.value = '';
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('zh-CN');
  };

  // 切换聊天模式
  // HTML转义函数，防止XSS攻击
  const escapeHtml = (text: string): string => {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  };



  const handleToggleChatMode = (mode: ChatMode) => {
    setChatMode(mode);
    toast.success(`已切换到${mode === 'text' ? '文本聊天' : '视频聊天'}模式`);
  };

  // 渲染文本聊天界面
  const renderTextChat = () => (
    <div className="flex-1 flex flex-col">
      {/* 聊天头部 */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
        <div className="flex items-center">
          <div className="w-10 h-10 bg-gradient-to-r from-gray-600 to-gray-800 rounded-full flex items-center justify-center mr-3">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="font-semibold text-gray-900 dark:text-white">
              Self AGI 对话系统
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {systemMode === 'task' ? '任务执行模式' : '全自主模式'} | 记忆系统 {useMemory ? '已启用' : '已禁用'} | 模型: {selectedModel}
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={handleToggleMemory}
            className={`p-2 rounded-lg ${
              useMemory 
                ? 'text-gray-700 bg-gray-700 dark:text-gray-700 dark:bg-gray-700/30' 
                : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
            title="切换记忆系统"
          >
            <Brain className="w-5 h-5" />
          </button>
          <button
            onClick={handleToggleSystemMode}
            className={`p-2 rounded-lg ${
              systemMode === 'autonomous'
                ? 'text-gray-600 bg-gray-600 dark:text-gray-600 dark:bg-gray-600/30' 
                : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
            title="切换系统模式"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
          <button className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800" title="设置">
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* 消息列表 */}
      <div className="flex-1 overflow-auto p-4">
        {currentMessages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <div className="w-20 h-20 bg-gradient-to-r from-gray-200 to-gray-300 dark:from-gray-800/20 dark:to-gray-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <Bot className="w-10 h-10 text-gray-700 dark:text-gray-700" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                开始与Self AGI对话
              </h3>
              <p className="text-gray-600 dark:text-gray-400 max-w-md">
                我是一个自主通用人工智能系统，具备多模态处理、记忆管理、机器人控制等能力。请输入消息开始对话。
              </p>
              <div className="mt-6 grid grid-cols-2 gap-3 max-w-sm mx-auto">
                <button
                  onClick={() => setInputMessage('请介绍一下你的功能')}
                  className="px-4 py-2 text-sm bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                >
                  功能介绍
                </button>
                <button
                  onClick={() => setInputMessage('你能做什么？')}
                  className="px-4 py-2 text-sm bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                >
                  能力询问
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6 max-w-4xl mx-auto" role="list" aria-label="消息列表">
            {currentMessages.map(message => (
              <div
                key={message.id}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                role="listitem"
                aria-label={`${message.sender === 'user' ? '您' : 'AI助手'}的消息，时间：${formatTime(message.timestamp)}，${message.status === 'sending' ? '正在发送' : message.status === 'error' ? '发送失败' : '已发送'}`}
                onContextMenu={(e) => message.sender === 'user' && handleContextMenu(e, message.id)}
              >
                {editingMessageId === message.id ? (
                  // 编辑模式
                  <div className="max-w-xl rounded-2xl px-4 py-3 bg-gradient-to-r from-gray-700 to-gray-900 text-white rounded-br-none">
                    <div className="flex items-center mb-2">
                      <div className="w-6 h-6 rounded-full flex items-center justify-center mr-2 bg-white/20">
                        <User className="w-3 h-3 text-white" />
                      </div>
                      <span className="text-xs opacity-80">编辑消息</span>
                    </div>
                    <textarea
                      value={editingMessageContent}
                      onChange={(e) => setEditingMessageContent(e.target.value)}
                      className="w-full bg-white/20 rounded px-3 py-2 text-white placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-white/50 resize-none"
                      rows={3}
                      placeholder="编辑消息内容..."
                      autoFocus
                    />
                    <div className="flex justify-end space-x-2 mt-3">
                      <button
                        onClick={handleCancelEdit}
                        className="px-3 py-1 text-xs bg-white/20 hover:bg-white/30 rounded transition-colors"
                      >
                        取消
                      </button>
                      <button
                        onClick={handleSaveEdit}
                        className="px-3 py-1 text-xs bg-white hover:bg-white/90 text-gray-700 rounded transition-colors"
                      >
                        保存
                      </button>
                    </div>
                  </div>
                ) : (
                  // 正常显示模式
                  <div
                    className={`max-w-xl rounded-2xl px-4 py-3 ${
                      message.sender === 'user'
                        ? 'bg-gradient-to-r from-gray-700 to-gray-900 text-white rounded-br-none'
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white rounded-bl-none'
                    }`}
                  >
                  <div className="flex items-center mb-2">
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-2 ${
                      message.sender === 'user' ? 'bg-white/20' : 'bg-gradient-to-r from-gray-600 to-gray-800'
                    }`}>
                      {message.sender === 'user' ? (
                        <User className="w-3 h-3 text-white" />
                      ) : (
                        <Bot className="w-3 h-3 text-white" />
                      )}
                    </div>
                    <span className="text-xs opacity-80">
                      {formatTime(message.timestamp)}
                    </span>
                    {message.status === 'sending' && (
                      <span className="ml-2 text-xs opacity-60 flex items-center">
                        <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                        发送中...
                      </span>
                    )}
                    {message.status === 'error' && (
                      <span className="ml-2 text-xs text-gray-800">发送失败</span>
                    )}
                  </div>
                  <div 
                    className="prose prose-sm dark:prose-invert whitespace-pre-wrap break-words"
                    dangerouslySetInnerHTML={{ __html: escapeHtml(message.content).replace(/\n/g, '<br>') }}
                    role="text"
                    aria-live={message.sender === 'ai' ? 'polite' : 'off'}
                    aria-atomic="true"
                  />
                    {/* 消息操作按钮（仅用户消息） */}
                    {message.sender === 'user' && message.status === 'sent' && (
                      <div className="flex justify-end mt-2 space-x-2 opacity-0 hover:opacity-100 transition-opacity">
                        <button
                          onClick={() => handleEditMessage(message.id, message.content)}
                          className="text-xs px-2 py-1 bg-white/20 hover:bg-white/30 rounded transition-colors"
                          title="编辑消息"
                        >
                          编辑
                        </button>
                        <button
                          onClick={() => setMessageToDelete(message.id)}
                          className="text-xs px-2 py-1 bg-gray-900/20 hover:bg-gray-900/30 text-gray-800 rounded transition-colors"
                          title="删除消息"
                        >
                          删除
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
        
        {isLoading && (
          <div className="flex justify-start max-w-4xl mx-auto mt-4">
            <div className="bg-gray-100 dark:bg-gray-800 rounded-2xl px-4 py-3 rounded-bl-none">
              <div className="flex items-center space-x-2">
                <div className="w-6 h-6 rounded-full bg-gradient-to-r from-gray-600 to-gray-800 flex items-center justify-center">
                  <Bot className="w-3 h-3 text-white" />
                </div>
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 输入区域 */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-3">
            <button
              type="button"
              onClick={handleVoiceInput}
              className={`p-3 rounded-full ${
                isRecording
                  ? 'bg-gradient-to-r from-gray-700 to-gray-900 text-white animate-pulse'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-300'
              }`}
              title={isRecording ? '停止录音' : '开始录音'}
            >
              {isRecording ? (
                <MicOff className="w-5 h-5" />
              ) : (
                <Mic className="w-5 h-5" />
              )}
            </button>
            
            <label className="p-3 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-300 cursor-pointer">
              <input
                type="file"
                className="hidden"
                accept="image/*,audio/*,video/*"
                onChange={handleFileUpload}
              />
              <ImageIcon className="w-5 h-5" />
            </label>
            
            <div className="flex-1">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="输入消息...（Shift+Enter换行，Enter发送）"
                className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-full focus:outline-none focus:ring-2 focus:ring-gray-700 focus:border-transparent text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                disabled={isLoading}
                aria-label="消息输入框"
                aria-describedby="input-instructions"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage(e);
                  }
                }}
              />
            </div>
            
            <button
              type="submit"
              disabled={!inputMessage.trim() || isLoading}
              className={`p-3 rounded-full ${
                !inputMessage.trim() || isLoading
                  ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-gray-700 to-gray-900 text-white hover:from-gray-800 hover:to-gray-900'
              }`}
              aria-label={isLoading ? "正在发送消息..." : "发送消息"}
              aria-disabled={!inputMessage.trim() || isLoading}
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" aria-hidden="true" />
              ) : (
                <Send className="w-5 h-5" aria-hidden="true" />
              )}
              <span className="sr-only">{isLoading ? "正在发送消息..." : "发送消息"}</span>
            </button>
          </div>
          
          <div className="mt-3 flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center space-x-4">
              <span className="flex items-center">
                <Clock className="w-3 h-3 mr-1" />
                实时响应: {isWebSocketConnected ? '已连接' : '连接中...'}
              </span>
              <span className="flex items-center">
                <Brain className="w-3 h-3 mr-1" />
                记忆系统: {useMemory ? '启用' : '禁用'}
              </span>
              {isTyping && (
                <span className="flex items-center text-gray-700 dark:text-gray-700">
                  <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  AI正在输入...
                </span>
              )}
              {webSocketError && (
                <span className="flex items-center text-gray-800 dark:text-gray-800">
                  <AlertCircle className="w-3 h-3 mr-1" />
                  连接错误
                </span>
              )}
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-300">
                按 Enter 发送，Shift+Enter 换行
              </span>
            </div>
          </div>
          {/* 屏幕阅读器说明 - 隐藏但可访问 */}
          <div id="input-instructions" className="sr-only">
            消息输入框。输入消息后按Enter键发送，或按Shift+Enter键换行。可使用语音输入、文件上传等功能。
          </div>
        </form>
      </div>
    </div>
  );

  // 上下文菜单
  const renderContextMenu = () => {
    if (!contextMenu) return null;

    return (
      <div
        className="fixed z-50 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg shadow-lg py-2 min-w-[160px]"
        style={{
          left: `${contextMenu.x}px`,
          top: `${contextMenu.y}px`,
        }}
        onClick={() => setContextMenu(null)}
      >
        <button
          onClick={() => {
            const message = currentMessages.find(m => m.id === contextMenu.messageId);
            if (message) {
              handleEditMessage(message.id, message.content);
            }
          }}
          className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          编辑消息
        </button>
        <button
          onClick={() => {
            handleCopyMessage(currentMessages.find(m => m.id === contextMenu.messageId)?.content || '');
          }}
          className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          复制消息
        </button>
        <button
          onClick={() => {
            setMessageToDelete(contextMenu.messageId);
            setContextMenu(null);
          }}
          className="w-full text-left px-4 py-2 text-sm text-gray-800 dark:text-gray-800 hover:bg-gray-900 dark:hover:bg-gray-800/30 transition-colors"
        >
          删除消息
        </button>
      </div>
    );
  };

  // 删除确认对话框
  const renderDeleteConfirm = () => {
    if (!messageToDelete) return null;

    const message = currentMessages.find(m => m.id === messageToDelete);
    const messagePreview = message?.content ? 
      (message.content.length > 50 ? `${message.content.substring(0, 50)}...` : message.content) : 
      '这条消息';

    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            确认删除消息
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            确定要删除消息 "{messagePreview}" 吗？此操作无法撤销。
          </p>
          <div className="flex justify-end space-x-3">
            <button
              onClick={() => setMessageToDelete(null)}
              className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            >
              取消
            </button>
            <button
              onClick={() => handleDeleteMessage(messageToDelete)}
              className="px-4 py-2 text-sm bg-gray-800 hover:bg-gray-800 text-white rounded transition-colors"
            >
              删除
            </button>
          </div>
        </div>
      </div>
    );
  };

  // 渲染视频聊天界面
  const renderVideoChat = () => <VideoChat />;

  return (
    <div className="flex h-[calc(100vh-4rem)] bg-gray-50 dark:bg-gray-900">
      {/* 左侧会话列表 */}
      <div className="w-64 border-r border-gray-200 dark:border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <button
            onClick={handleNewSession}
            className="w-full flex items-center justify-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-700 to-gray-900 rounded-lg hover:from-gray-800 hover:to-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-500"
          >
            <Plus className="w-4 h-4 mr-2" />
            新会话
          </button>
        </div>
        
        <div className="flex-1 overflow-auto p-2">
          {sessions.length === 0 ? (
            <div className="text-center py-8">
              <MessageSquare className="w-12 h-12 mx-auto text-gray-400 mb-3" />
              <p className="text-sm text-gray-600 dark:text-gray-400">
                还没有会话，点击上方按钮创建新会话
              </p>
            </div>
          ) : (
            sessions.map(session => (
              <div
                key={session.id}
                onClick={() => handleSelectSession(session.id)}
                className={`p-3 rounded-lg mb-2 cursor-pointer transition-colors group ${
                  activeSession === session.id
                    ? 'bg-gray-700 dark:bg-gray-700/30 border border-gray-700 dark:border-gray-700'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
              >
                <div className="flex justify-between items-start mb-1">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900 dark:text-white truncate">
                      {session.title}
                    </h4>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 truncate">
                      {session.lastMessage}
                    </p>
                  </div>
                  <div className="flex items-center space-x-1">
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {formatTime(session.createdAt)}
                    </span>
                    {session.id !== activeSession && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteSession(session.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-gray-800 transition-opacity"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-500">
                  {formatDate(session.createdAt)}
                </div>
              </div>
            ))
          )}
        </div>
        
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 space-y-4">
          {/* 聊天模式切换 */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700 dark:text-gray-300">聊天模式</span>
              <div className="flex space-x-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
                <button
                  onClick={() => handleToggleChatMode('text')}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    chatMode === 'text'
                      ? 'bg-gradient-to-r from-gray-600 to-gray-700 text-white' 
                      : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-300'
                  }`}
                >
                  文本
                </button>
                <button
                  onClick={() => handleToggleChatMode('video')}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    chatMode === 'video'
                      ? 'bg-gradient-to-r from-gray-700 to-gray-800 text-white' 
                      : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-300'
                  }`}
                >
                  视频
                </button>
              </div>
            </div>
          </div>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700 dark:text-gray-300">记忆系统</span>
              <button
                onClick={handleToggleMemory}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  useMemory ? 'bg-gray-700' : 'bg-gray-300 dark:bg-gray-700'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    useMemory ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700 dark:text-gray-300">系统模式</span>
              <button
                onClick={handleToggleSystemMode}
                className={`px-3 py-1 text-xs font-medium rounded-full ${
                  systemMode === 'task' 
                    ? 'bg-gradient-to-r from-gray-600 to-gray-700 text-white' 
                    : 'bg-gradient-to-r from-gray-700 to-gray-800 text-white'
                }`}
              >
                {systemMode === 'task' ? '任务模式' : '自主模式'}
              </button>
            </div>
            
            {availableModels.length > 0 && (
              <div>
                <label className="block text-sm text-gray-700 dark:text-gray-300 mb-1">
                  选择模型
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-gray-700"
                >
                  {availableModels.map(model => (
                    <option key={model.name} value={model.name}>
                      {model.displayName || model.name}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 主聊天区域 */}
      <div 
        className="flex-1 flex flex-col" 
        role="main" 
        aria-label="聊天主区域"
      >
        {/* 聊天模式头部 */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div className="flex items-center">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center mr-3 ${
              chatMode === 'text' 
                ? 'bg-gradient-to-r from-gray-600 to-gray-700' 
                : 'bg-gradient-to-r from-gray-700 to-gray-800'
            }`}>
              {chatMode === 'text' ? (
                <MessageSquare className="w-6 h-6 text-white" />
              ) : (
                <Video className="w-6 h-6 text-white" />
              )}
            </div>
            <div>
              <h2 className="font-semibold text-gray-900 dark:text-white">
                {chatMode === 'text' ? 'Self AGI 文本对话' : 'Self AGI 视频聊天'}
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {chatMode === 'text' 
                  ? `${systemMode === 'task' ? '任务执行模式' : '全自主模式'} | 记忆系统 ${useMemory ? '已启用' : '已禁用'}`
                  : '实时视频通信 | WebRTC技术支持'
                }
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* 连接状态指示器 */}
            {chatMode === 'video' && (
              <div className="flex items-center px-3 py-1 rounded-full text-sm bg-gray-600 dark:bg-gray-700/30 text-gray-600 dark:text-gray-600">
                <CheckCircle className="w-4 h-4 mr-1" />
                视频就绪
              </div>
            )}
            
            <button className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800" title="设置">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* 聊天内容区域 */}
        {chatMode === 'text' ? renderTextChat() : renderVideoChat()}
        
        {/* 上下文菜单 */}
        {renderContextMenu()}
        
        {/* 删除确认对话框 */}
        {renderDeleteConfirm()}
        
        {/* 点击其他地方关闭上下文菜单 */}
        {contextMenu && (
          <div 
            className="fixed inset-0 z-40" 
            onClick={() => setContextMenu(null)}
          />
        )}
      </div>
    </div>
  );
};

export default ChatPage;