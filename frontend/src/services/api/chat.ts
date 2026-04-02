import { ApiClient, apiClient } from './client';
import { ApiResponse } from '../../types/api';

// 环境感知的日志函数
const isDevelopment = process.env.NODE_ENV === 'development';
const log = (...args: any[]) => {
  if (isDevelopment) {
    console.log(...args);
  }
};
const logError = (...args: any[]) => {
  console.error(...args);
};

// 聊天消息接口
export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: string;
  type: 'text' | 'image' | 'audio';
  status: 'sending' | 'sent' | 'error';
}

// 聊天会话接口
export interface ChatSession {
  id: string;
  title: string;
  model_name: string;
  created_at: string;
  last_message: string;
  message_count: number;
}

// 聊天请求接口
export interface ChatRequest {
  message: string;
  model_name?: string;
  use_memory?: boolean;
  session_id?: string;
  multimodal_input?: {
    image?: string;
    audio?: string;
    video?: string;
  };
  temperature?: number;
  max_tokens?: number;
}

// 聊天响应接口
export interface ChatResponse {
  success: boolean;
  response: string;
  session_id: string;
  model_name: string;
  processing_time: number;
  tokens_used: number;
  memories_retrieved?: number;
  timestamp: string;
}



// WebSocket消息类型
export type WebSocketMessageType = 
  | 'ping' 
  | 'pong' 
  | 'chat_message' 
  | 'ai_response' 
  | 'message_received' 
  | 'connection_established' 
  | 'connection_closed'
  | 'session_connected' 
  | 'typing_indicator' 
  | 'typing_status' 
  | 'session_update' 
  | 'session_updated' 
  | 'error';

// 多模态处理响应接口
export interface MultimodalProcessResponse {
  success: boolean;
  data?: {
    result?: {
      transcription?: string;
    };
  };
  timestamp: string;
}

// WebSocket消息接口
export interface WebSocketMessage {
  type: WebSocketMessageType;
  message_id?: string;
  original_message_id?: string;
  content?: string;
  session_id?: string;
  model_name?: string;
  processing_time?: number;
  tokens_used?: number;
  memories_retrieved?: number;
  status?: string;
  message?: string;
  timestamp: string;
  action?: string;
  is_typing?: boolean;
  code?: number; // WebSocket关闭代码
  reason?: string; // WebSocket关闭原因
}

// WebSocket事件监听器类型
export type WebSocketEventListener = (event: WebSocketMessage) => void;

// WebSocket连接状态
export enum WebSocketConnectionState {
  CONNECTING = 0,
  OPEN = 1,
  CLOSING = 2,
  CLOSED = 3
}

class ChatApi {
  private apiClient: ApiClient;
  private chatWebSocket: WebSocket | null = null;
  private sessionWebSocket: WebSocket | null = null;
  private eventListeners: Map<string, Set<WebSocketEventListener>> = new Map();
  private pingInterval: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private baseWebSocketUrl: string;

  constructor() {
    this.apiClient = apiClient;
    // 根据当前环境构建WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    this.baseWebSocketUrl = `${protocol}//${host}/api`;
  }

  // 发送聊天消息
  async sendMessage(request: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
    try {
      const modelName = request.model_name || 'model_default';
      const config = signal ? { signal } : undefined;
      const response = (await this.apiClient.post(`/models/${modelName}/chat`, request, config)) as ChatResponse;
      return response;
    } catch (error) {
      logError('发送聊天消息失败:', error);
      throw error;
    }
  }

  // 获取聊天会话列表
  async getSessions(): Promise<ApiResponse<ChatSession[]>> {
    try {
      const response = (await this.apiClient.get('/chat/sessions')) as ApiResponse<ChatSession[]>;
      return response;
    } catch (error) {
      console.error('获取会话列表失败:', error);
      throw error;
    }
  }

  // 创建新会话
  async createSession(title?: string, modelName?: string): Promise<ApiResponse<ChatSession>> {
    try {
      const response = (await this.apiClient.post('/chat/sessions', {
        title: title || `新会话 ${new Date().toLocaleDateString('zh-CN')}`,
        model_name: modelName || '默认模型'
      })) as ApiResponse<ChatSession>;
      return response;
    } catch (error) {
      console.error('创建会话失败:', error);
      throw error;
    }
  }

  // 删除会话
  async deleteSession(sessionId: string): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.delete(`/chat/sessions/${sessionId}`)) as ApiResponse;
      return response;
    } catch (error) {
      console.error('删除会话失败:', error);
      throw error;
    }
  }

  // 获取模型列表
  async getModels(): Promise<ApiResponse<{id: string, name: string, description: string, provider: string, max_tokens: number, supports_multimodal: boolean, is_available: boolean}[]>> {
    try {
      const response = (await this.apiClient.get('/chat/models')) as ApiResponse<{id: string, name: string, description: string, provider: string, max_tokens: number, supports_multimodal: boolean, is_available: boolean}[]>;
      return response;
    } catch (error) {
      console.error('获取模型列表失败:', error);
      throw error;
    }
  }

  // 获取系统模式
  async getSystemMode(): Promise<ApiResponse<{mode: string, description: string, capabilities: string[], restrictions: string[], performance_metrics: unknown}>> {
    try {
      const response = (await this.apiClient.get('/chat/mode')) as ApiResponse<{mode: string, description: string, capabilities: string[], restrictions: string[], performance_metrics: unknown}>;
      return response;
    } catch (error) {
      console.error('获取系统模式失败:', error);
      throw error;
    }
  }

  // 设置系统模式
  async setSystemMode(mode: string): Promise<ApiResponse> {
    try {
      const response = (await this.apiClient.put('/chat/mode', { mode })) as ApiResponse;
      return response;
    } catch (error) {
      console.error('设置系统模式失败:', error);
      throw error;
    }
  }

  // 上传文件用于多模态处理
  async uploadFile(file: File, type: 'image' | 'audio' | 'video'): Promise<ApiResponse<{url: string, filename: string}>> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('type', type);
      
      const response = (await this.apiClient.post('/uploads', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })) as ApiResponse<{url: string, filename: string}>;
      return response;
    } catch (error) {
      console.error('上传文件失败:', error);
      throw error;
    }
  }

  // 语音识别（将音频转换为文本）- 调用多模态处理API
  async speechToText(audioData: Blob): Promise<ApiResponse<{text: string}>> {
    try {
      const formData = new FormData();
      // 创建音频文件对象
      const audioFile = new File([audioData], 'recording.wav', { type: 'audio/wav' });
      formData.append('file', audioFile);
      formData.append('task', 'transcribe');
      
      const response = (await this.apiClient.post('/multimodal/process/audio', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })) as MultimodalProcessResponse;
      
      // 从响应中提取转写文本
      if (response.success && response.data?.result?.transcription) {
        return {
          success: true,
          data: { text: response.data.result.transcription },
          message: '语音识别成功',
          timestamp: response.timestamp
        } as ApiResponse<{text: string}>;
      } else {
        throw new Error('语音识别响应格式不正确');
      }
    } catch (error) {
      console.error('语音识别失败:', error);
      throw error;
    }
  }

  // 文本转语音
  async textToSpeech(text: string, voice?: string): Promise<ApiResponse<{audio_url: string}>> {
    try {
      const response = (await this.apiClient.post('/text-to-speech', { text, voice })) as ApiResponse<{audio_url: string}>;
      return response;
    } catch (error) {
      console.error('文本转语音失败:', error);
      throw error;
    }
  }

  // ==================== WebSocket相关方法 ====================

  /**
   * 连接到聊天WebSocket
   */
  connectToChatWebSocket(): Promise<boolean> {
    return new Promise((resolve, reject) => {
      try {
        // 如果已经连接，先断开
        if (this.chatWebSocket && this.chatWebSocket.readyState === WebSocket.OPEN) {
          this.disconnectChatWebSocket();
        }

        const wsUrl = `${this.baseWebSocketUrl}/ws/chat`;
        this.chatWebSocket = new WebSocket(wsUrl);

        this.chatWebSocket.onopen = () => {
          log('聊天WebSocket连接已建立');
          this.reconnectAttempts = 0;
          
          // 开始发送ping消息保持连接
          this.startPingInterval();
          
          // 触发连接建立事件
          this.emitEvent('connection_established', {
            type: 'connection_established',
            message: 'WebSocket连接已建立',
            timestamp: new Date().toISOString()
          });
          
          resolve(true);
        };

        this.chatWebSocket.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
          } catch (error) {
            console.error('WebSocket消息解析失败:', error, event.data);
          }
        };

        this.chatWebSocket.onerror = (error) => {
          console.error('聊天WebSocket错误:', error);
          this.emitEvent('error', {
            type: 'error',
            message: 'WebSocket连接错误',
            timestamp: new Date().toISOString()
          });
          reject(error);
        };

        this.chatWebSocket.onclose = (event) => {
          log(`聊天WebSocket连接关闭，代码: ${event.code}, 原因: ${event.reason}`);
          
          // 停止ping间隔
          this.stopPingInterval();
          
          // 触发连接关闭事件
          this.emitEvent('connection_closed', {
            type: 'connection_closed',
            code: event.code,
            reason: event.reason,
            timestamp: new Date().toISOString()
          });
          
          // 尝试重连
          if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            log(`尝试重连... (第${this.reconnectAttempts}次)`);
            setTimeout(() => {
              this.connectToChatWebSocket().catch(console.error);
            }, 1000 * this.reconnectAttempts); // 指数退避
          }
        };
      } catch (error) {
        console.error('创建WebSocket连接失败:', error);
        reject(error);
      }
    });
  }

  /**
   * 连接到特定会话的WebSocket
   */
  connectToSessionWebSocket(sessionId: string): Promise<boolean> {
    return new Promise((resolve, reject) => {
      try {
        // 如果已经连接，先断开
        if (this.sessionWebSocket && this.sessionWebSocket.readyState === WebSocket.OPEN) {
          this.disconnectSessionWebSocket();
        }

        const wsUrl = `${this.baseWebSocketUrl}/ws/chat/${sessionId}`;
        this.sessionWebSocket = new WebSocket(wsUrl);

        this.sessionWebSocket.onopen = () => {
          log(`会话 ${sessionId} WebSocket连接已建立`);
          
          this.emitEvent('session_connected', {
            type: 'session_connected',
            session_id: sessionId,
            message: `已连接到聊天会话: ${sessionId}`,
            timestamp: new Date().toISOString()
          });
          
          resolve(true);
        };

        this.sessionWebSocket.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
          } catch (error) {
            console.error('会话WebSocket消息解析失败:', error, event.data);
          }
        };

        this.sessionWebSocket.onerror = (error) => {
          console.error(`会话 ${sessionId} WebSocket错误:`, error);
          reject(error);
        };

        this.sessionWebSocket.onclose = (event) => {
          log(`会话 ${sessionId} WebSocket连接关闭，代码: ${event.code}`);
        };
      } catch (error) {
        console.error('创建会话WebSocket连接失败:', error);
        reject(error);
      }
    });
  }

  /**
   * 断开聊天WebSocket连接
   */
  disconnectChatWebSocket(): void {
    if (this.chatWebSocket) {
      this.stopPingInterval();
      this.chatWebSocket.close(1000, '正常关闭');
      this.chatWebSocket = null;
    }
  }

  /**
   * 断开会话WebSocket连接
   */
  disconnectSessionWebSocket(): void {
    if (this.sessionWebSocket) {
      this.sessionWebSocket.close(1000, '正常关闭');
      this.sessionWebSocket = null;
    }
  }

  /**
   * 发送聊天消息通过WebSocket
   */
  sendChatMessageViaWebSocket(message: {
    content: string;
    session_id?: string;
    model_name?: string;
    message_id?: string;
  }): void {
    if (!this.chatWebSocket || this.chatWebSocket.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket连接未建立');
    }

    const messageId = message.message_id || `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    this.chatWebSocket.send(JSON.stringify({
      type: 'chat_message',
      message_id: messageId,
      content: message.content,
      session_id: message.session_id || 'default-session',
      model_name: message.model_name || '默认模型',
      timestamp: new Date().toISOString()
    }));
  }

  /**
   * 发送打字指示器状态
   */
  sendTypingIndicator(isTyping: boolean): void {
    if (this.chatWebSocket && this.chatWebSocket.readyState === WebSocket.OPEN) {
      this.chatWebSocket.send(JSON.stringify({
        type: 'typing_indicator',
        is_typing: isTyping,
        timestamp: new Date().toISOString()
      }));
    }
  }

  /**
   * 发送会话更新
   */
  sendSessionUpdate(sessionId: string, action: 'create' | 'update' | 'delete'): void {
    if (this.chatWebSocket && this.chatWebSocket.readyState === WebSocket.OPEN) {
      this.chatWebSocket.send(JSON.stringify({
        type: 'session_update',
        session_id: sessionId,
        action: action,
        timestamp: new Date().toISOString()
      }));
    }
  }

  /**
   * 开始发送ping消息保持连接
   */
  private startPingInterval(): void {
    this.stopPingInterval(); // 确保没有现有的间隔
    
    this.pingInterval = setInterval(() => {
      if (this.chatWebSocket && this.chatWebSocket.readyState === WebSocket.OPEN) {
        this.chatWebSocket.send(JSON.stringify({
          type: 'ping',
          timestamp: new Date().toISOString()
        }));
      }
    }, 30000); // 每30秒发送一次ping
  }

  /**
   * 停止ping间隔
   */
  private stopPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * 处理WebSocket消息
   */
  private handleWebSocketMessage(message: WebSocketMessage): void {
    // 触发特定类型的事件
    this.emitEvent(message.type, message);
    
    // 同时触发通用消息事件
    this.emitEvent('message', message);
  }

  /**
   * 添加事件监听器
   */
  addEventListener(eventType: string, listener: WebSocketEventListener): void {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, new Set());
    }
    this.eventListeners.get(eventType)!.add(listener);
  }

  /**
   * 移除事件监听器
   */
  removeEventListener(eventType: string, listener: WebSocketEventListener): void {
    if (this.eventListeners.has(eventType)) {
      this.eventListeners.get(eventType)!.delete(listener);
    }
  }

  /**
   * 触发事件
   */
  private emitEvent(eventType: string, data: WebSocketMessage): void {
    if (this.eventListeners.has(eventType)) {
      this.eventListeners.get(eventType)!.forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error(`事件监听器执行失败 (${eventType}):`, error);
        }
      });
    }
  }

  /**
   * 获取WebSocket连接状态
   */
  getWebSocketConnectionState(): WebSocketConnectionState {
    if (!this.chatWebSocket) {
      return WebSocketConnectionState.CLOSED;
    }
    return this.chatWebSocket.readyState;
  }

  /**
   * 检查WebSocket是否已连接
   */
  isWebSocketConnected(): boolean {
    return this.chatWebSocket?.readyState === WebSocket.OPEN;
  }

  /**
   * 检查会话WebSocket是否已连接
   */
  isSessionWebSocketConnected(): boolean {
    return this.sessionWebSocket?.readyState === WebSocket.OPEN;
  }
}

// 创建单例实例
export const chatApi = new ChatApi();

// 默认导出
export default chatApi;