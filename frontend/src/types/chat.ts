/**
 * Chat 相关类型定义
 * 建立前后端类型映射，确保类型安全
 * 
 * 注意：模型相关类型已移至model.ts文件，这里提供兼容层
 */

import { 
  ModelProvider as ModelProviderFromModel
} from './model';

// ==================== 基础枚举类型 ====================

/** 消息发送者类型 */
export type MessageSender = 'user' | 'assistant' | 'ai' | 'system';

/** 消息类型 */
export type MessageType = 'text' | 'image' | 'audio' | 'video' | 'file' | 'system';

/** 消息状态 */
export type MessageStatus = 'sending' | 'sent' | 'error' | 'read';

/** 聊天模式 */
export type ChatMode = 'text' | 'video' | 'audio';

/** 系统模式 */
export type SystemMode = 'task' | 'autonomous' | 'training' | 'maintenance' | 'assist';

/** AI模型提供商 - 使用model.ts中的定义 */
export type ModelProvider = ModelProviderFromModel;

// ==================== 后端API响应类型 ====================

/** 后端聊天消息响应 */
export interface BackendChatMessage {
  id?: string;
  content: string;
  sender: 'user' | 'assistant' | 'system';
  timestamp: string; // ISO格式字符串
  type: 'text' | 'image' | 'audio' | 'video';
  status?: 'sending' | 'sent' | 'error';
  metadata?: Record<string, any>;
}

/** 后端聊天会话响应 */
export interface BackendChatSession {
  id: string;
  title: string;
  model_name: string;
  created_at: string; // ISO格式字符串
  last_message: string;
  message_count: number;
  metadata?: Record<string, any>;
}

/** 后端聊天请求 */
export interface BackendChatRequest {
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
  system_prompt?: string;
}

/** 后端聊天响应 */
export interface BackendChatResponse {
  success: boolean;
  response: string;
  session_id: string;
  model_name: string;
  processing_time: number;
  tokens_used: number;
  memories_retrieved?: number;
  timestamp: string; // ISO格式字符串
  metadata?: Record<string, any>;
}

/** 后端模型信息 */
export interface BackendModelInfo {
  id: string;
  name: string;
  description: string;
  provider: ModelProvider;
  max_tokens: number;
  supports_multimodal: boolean;
  is_available: boolean;
  parameters?: {
    temperature?: number;
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
  };
  capabilities: string[];
}

/** 后端系统模式响应 */
export interface BackendSystemMode {
  mode: SystemMode;
  description: string;
  capabilities: string[];
  restrictions: string[];
  performance_metrics?: Record<string, any>;
}

/** 后端语音识别响应 */
export interface BackendSpeechToTextResponse {
  success: boolean;
  text: string;
  confidence?: number;
  language?: string;
  duration?: number;
  timestamp: string;
}

/** 后端文件上传响应 */
export interface BackendFileUploadResponse {
  success: boolean;
  url: string;
  filename: string;
  file_type: string;
  size: number;
  upload_date: string;
}

// ==================== 前端UI类型 ====================

/** 前端UI消息接口 */
export interface UiMessage {
  id: string;
  content: string;
  sender: 'user' | 'ai' | 'system';
  timestamp: Date;
  type: MessageType;
  status: MessageStatus;
  metadata?: Record<string, any>;
}

/** 前端UI会话接口 */
export interface UiChatSession {
  id: string;
  title: string;
  createdAt: Date;
  lastMessage: string;
  modelName?: string;
  messageCount?: number;
  metadata?: Record<string, any>;
}

/** 前端模型显示信息 */
export interface UiModelInfo {
  id: string;
  name: string;
  displayName: string;
  description: string;
  provider: ModelProvider;
  maxTokens: number;
  supportsMultimodal: boolean;
  isAvailable: boolean;
  parameters?: {
    temperature?: number;
    topP?: number;
  };
  capabilities: string[];
}

/** 前端系统模式显示信息 */
export interface UiSystemMode {
  mode: SystemMode;
  description: string;
  displayName: string;
  capabilities: string[];
  restrictions: string[];
  icon?: string;
}

// ==================== API响应包装类型 ====================

/** 通用API响应 */
export interface ApiResponse<T = any> {
  success: boolean;
  message?: string;
  data?: T;
  error?: string;
  code?: number;
  timestamp: string;
}

/** 分页响应 */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

// ==================== 类型转换函数 ====================

/** 将后端消息转换为前端UI消息 */
export function convertBackendMessageToUi(backendMessage: BackendChatMessage): UiMessage {
  return {
    id: backendMessage.id || Date.now().toString(),
    content: backendMessage.content,
    sender: backendMessage.sender === 'assistant' ? 'ai' : backendMessage.sender as 'user' | 'ai' | 'system',
    timestamp: new Date(backendMessage.timestamp),
    type: backendMessage.type,
    status: backendMessage.status || 'sent',
    metadata: backendMessage.metadata,
  };
}

/** 将前端UI消息转换为后端消息 */
export function convertUiMessageToBackend(uiMessage: UiMessage): BackendChatMessage {
  // 转换消息类型为后端支持的类型
  const getBackendMessageType = (type: MessageType): 'text' | 'image' | 'audio' | 'video' => {
    if (type === 'file' || type === 'system' || type === 'text') return 'text';
    if (type === 'image' || type === 'audio' || type === 'video') return type;
    return 'text';
  };

  // 转换消息状态为后端支持的状态
  const getBackendMessageStatus = (status: MessageStatus): 'sending' | 'sent' | 'error' | undefined => {
    if (status === 'sending' || status === 'sent' || status === 'error') return status;
    return undefined; // 'read'状态不被后端支持
  };

  return {
    id: uiMessage.id,
    content: uiMessage.content,
    sender: uiMessage.sender === 'ai' ? 'assistant' : uiMessage.sender,
    timestamp: uiMessage.timestamp.toISOString(),
    type: getBackendMessageType(uiMessage.type),
    status: getBackendMessageStatus(uiMessage.status),
    metadata: uiMessage.metadata,
  };
}

/** 将后端会话转换为前端UI会话 */
export function convertBackendSessionToUi(backendSession: BackendChatSession): UiChatSession {
  return {
    id: backendSession.id,
    title: backendSession.title,
    createdAt: new Date(backendSession.created_at),
    lastMessage: backendSession.last_message,
    modelName: backendSession.model_name,
    messageCount: backendSession.message_count,
    metadata: backendSession.metadata,
  };
}

/** 将后端模型信息转换为前端UI模型信息 */
export function convertBackendModelToUi(backendModel: BackendModelInfo): UiModelInfo {
  return {
    id: backendModel.id,
    name: backendModel.id,
    displayName: backendModel.name,
    description: backendModel.description,
    provider: backendModel.provider,
    maxTokens: backendModel.max_tokens,
    supportsMultimodal: backendModel.supports_multimodal,
    isAvailable: backendModel.is_available,
    parameters: backendModel.parameters,
    capabilities: backendModel.capabilities || [],
  };
}

/** 将后端系统模式转换为前端UI系统模式 */
export function convertBackendSystemModeToUi(backendMode: BackendSystemMode): UiSystemMode {
  const modeDisplayNames: Record<SystemMode, string> = {
    'task': '任务执行模式',
    'autonomous': '全自主模式',
    'training': '训练模式',
    'maintenance': '维护模式',
    'assist': '助手模式',
  };
  
  const modeIcons: Record<SystemMode, string> = {
    'task': '🎯',
    'autonomous': '🤖',
    'training': '🧠',
    'maintenance': '🔧',
    'assist': '👤',
  };
  
  return {
    mode: backendMode.mode as SystemMode,
    description: backendMode.description,
    displayName: modeDisplayNames[backendMode.mode as SystemMode] || backendMode.mode,
    capabilities: backendMode.capabilities,
    restrictions: backendMode.restrictions,
    icon: modeIcons[backendMode.mode as SystemMode],
  };
}

// ==================== 安全相关类型 ====================

/** 文件上传验证选项 */
export interface FileUploadValidationOptions {
  maxSize: number; // 字节
  allowedTypes: string[];
  allowedExtensions: string[];
  scanForMalware: boolean;
  validateFileName: boolean;
}

/** 默认文件上传验证选项 */
export const DEFAULT_FILE_UPLOAD_OPTIONS: FileUploadValidationOptions = {
  maxSize: 10 * 1024 * 1024, // 10MB
  allowedTypes: [
    'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml',
    'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/webm',
    'video/mp4', 'video/webm', 'video/ogg',
    'application/pdf', 'text/plain', 'application/json'
  ],
  allowedExtensions: ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg', 'mp3', 'wav', 'ogg', 'webm', 'mp4', 'pdf', 'txt', 'json'],
  scanForMalware: true,
  validateFileName: true,
};

/** 消息内容安全选项 */
export interface MessageSecurityOptions {
  sanitizeHtml: boolean;
  maxLength: number;
  allowExternalLinks: boolean;
  validateUrls: boolean;
  blockScriptTags: boolean;
}

/** 默认消息安全选项 */
export const DEFAULT_MESSAGE_SECURITY_OPTIONS: MessageSecurityOptions = {
  sanitizeHtml: true,
  maxLength: 10000,
  allowExternalLinks: false,
  validateUrls: true,
  blockScriptTags: true,
};

// ==================== 错误类型 ====================

/** 聊天错误类型 */
export interface ChatError {
  code: string;
  message: string;
  details?: any;
  timestamp: Date;
}

/** 错误代码枚举 */
export enum ChatErrorCode {
  NETWORK_ERROR = 'NETWORK_ERROR',
  TIMEOUT_ERROR = 'TIMEOUT_ERROR',
  MODEL_UNAVAILABLE = 'MODEL_UNAVAILABLE',
  INVALID_INPUT = 'INVALID_INPUT',
  UNAUTHORIZED = 'UNAUTHORIZED',
  RATE_LIMITED = 'RATE_LIMITED',
  SERVER_ERROR = 'SERVER_ERROR',
  FILE_UPLOAD_ERROR = 'FILE_UPLOAD_ERROR',
  SPEECH_RECOGNITION_ERROR = 'SPEECH_RECOGNITION_ERROR',
}

// ==================== 常量定义 ====================

/** 默认聊天模型ID */
export const DEFAULT_MODEL_ID = 'model_default';

/** 最大消息长度 */
export const MAX_MESSAGE_LENGTH = 10000;

/** 请求超时时间（毫秒） */
export const REQUEST_TIMEOUT_MS = 60000;

/** 语音录制最大时间（毫秒） */
export const MAX_RECORDING_TIME_MS = 30000;

/** 文件上传最大重试次数 */
export const MAX_FILE_UPLOAD_RETRIES = 3;

/** 支持的音频格式 */
export const SUPPORTED_AUDIO_FORMATS = ['audio/webm', 'audio/wav', 'audio/ogg', 'audio/mpeg'];

/** 支持的图片格式 */
export const SUPPORTED_IMAGE_FORMATS = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'];

/** 支持的视频格式 */
export const SUPPORTED_VIDEO_FORMATS = ['video/mp4', 'video/webm', 'video/ogg'];

// ==================== 实用类型 ====================

/** 消息状态变化回调 */
export type MessageStatusCallback = (messageId: string, status: MessageStatus) => void;

/** 录音状态 */
export interface RecordingState {
  isRecording: boolean;
  duration: number;
  audioChunks: Blob[];
  mediaRecorder?: MediaRecorder;
  stream?: MediaStream;
  startTime?: Date;
}

/** 聊天会话状态 */
export interface ChatSessionState {
  sessionId: string;
  messages: UiMessage[];
  isLoading: boolean;
  lastUpdated: Date;
  modelId: string;
  systemMode: SystemMode;
}

/** 聊天全局状态 */
export interface ChatGlobalState {
  sessions: Record<string, ChatSessionState>;
  activeSessionId: string;
  availableModels: UiModelInfo[];
  selectedModelId: string;
  systemMode: SystemMode;
  isInitialized: boolean;
}
