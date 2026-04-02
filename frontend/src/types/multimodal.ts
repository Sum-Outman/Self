/**
 * 多模态认知和AGI能力类型定义
 */

// 人形机器人教学请求
export interface RobotTeachingRequest {
  concept_name: string;
  teaching_method?: 'demonstration' | 'explanation' | 'interactive' | 'guided_exploration';
  modalities: {
    text?: string;
    audio_file?: File;
    image_file?: File;
    sensor_data?: Record<string, any>;
    spatial_data?: {
      size?: number;
      shape?: string;
      position?: [number, number, number];
    };
    quantity?: number;
  };
  session_notes?: string;
}

// 人形机器人教学响应
export interface RobotTeachingResponse {
  success: boolean;
  session_id: string;
  concept_name: string;
  teaching_method: string;
  modalities_used: string[];
  learning_progress: number;
  robot_feedback: string;
  next_steps: string[];
  timestamp: string;
}

// 概念测试请求
export interface ConceptTestRequest {
  concept_name: string;
  test_type: 'recognition' | 'understanding' | 'application' | 'synthesis';
  test_input: {
    text?: string;
    image_file?: File;
    audio_file?: File;
    question?: string;
  };
}

// 概念测试响应
export interface ConceptTestResponse {
  success: boolean;
  test_score: number;
  correct_answers: string[];
  robot_response: string;
  concept_mastery: 'novice' | 'intermediate' | 'advanced' | 'master';
  feedback: string;
  timestamp: string;
}

// 电脑操作 - 屏幕分析请求
export interface ScreenAnalysisRequest {
  screenshot_file?: File;
}

// 电脑操作 - 屏幕元素
export interface ScreenElement {
  id: string;
  type: 'window' | 'button' | 'text' | 'icon' | 'input_field' | 'menu' | 'taskbar' | 'desktop';
  position: [number, number, number, number]; // x, y, width, height
  text?: string;
  confidence: number;
  attributes?: Record<string, any>;
}

// 电脑操作 - 屏幕分析响应
export interface ScreenAnalysisResponse {
  success: boolean;
  timestamp: string;
  screen_resolution: [number, number];
  elements_detected: number;
  elements: ScreenElement[];
  execution_time: number;
  analysis_mode: 'real' | 'simulation';
}

// 电脑操作 - 键盘操作请求
export interface KeyboardOperationRequest {
  keys: string[];
  text?: string;
  delay_between_keys?: number;
  press_duration?: number;
}

// 电脑操作 - 鼠标操作请求
export interface MouseOperationRequest {
  operation_type: 'click' | 'double_click' | 'right_click' | 'drag' | 'scroll' | 'move';
  position?: [number, number];
  button?: 'left' | 'right' | 'middle';
  scroll_amount?: number;
  drag_start?: [number, number];
  drag_end?: [number, number];
}

// 电脑操作 - 命令执行请求
export interface CommandExecutionRequest {
  command: string;
  working_directory?: string;
  timeout?: number;
}

// 电脑操作 - 命令执行响应
export interface CommandExecutionResponse {
  success: boolean;
  exit_code: number;
  stdout: string;
  stderr: string;
  execution_time: number;
  command: string;
}

// 设备操作学习 - 说明书学习请求
export interface ManualLearningRequest {
  manual_text: string;
  equipment_name: string;
  equipment_type: string;
  learning_method: 'manual_analysis' | 'guided_learning' | 'exploratory_learning';
  manual_file?: File;
}

// 设备操作学习 - 说明书学习响应
export interface ManualLearningResponse {
  success: boolean;
  session_id: string;
  action: string;
  equipment_id: string;
  equipment_name: string;
  components_extracted: number;
  procedures_extracted: number;
  safety_guidelines_extracted: number;
  confidence_score: number;
  learning_summary: string[];
  timestamp: string;
}

// 设备操作 - 操作步骤
export interface OperationStep {
  step_number: number;
  description: string;
  action_type: string;
  target_component?: string;
  parameters?: Record<string, any>;
  safety_warnings?: string[];
  expected_outcome?: string;
}

// 设备操作 - 获取操作流程响应
export interface OperationProceduresResponse {
  success: boolean;
  equipment_id: string;
  equipment_name: string;
  procedures: OperationStep[];
  procedure_count: number;
  safety_guidelines: string[];
  learning_status: 'novice' | 'intermediate' | 'advanced' | 'expert';
  timestamp: string;
}

// 视觉动作模仿 - 开始会话请求
export interface ImitationSessionRequest {
  target_action?: string;
  imitation_mode?: 'record_and_replay' | 'real_time_mirroring' | 'style_transfer';
  session_notes?: string;
}

// 视觉动作模仿 - 开始会话响应
export interface ImitationSessionResponse {
  success: boolean;
  session_id: string;
  start_time: string;
  target_action?: string;
  imitation_mode: string;
  session_notes?: string;
  websocket_url?: string;
}

// 视觉动作模仿 - 姿态录制请求
export interface PoseRecordingRequest {
  session_id: string;
  pose_data: {
    joint_positions: Record<string, [number, number, number]>;
    joint_angles: Record<string, number>;
    timestamp: number;
    confidence: number;
  };
}

// 视觉动作模仿 - 姿态录制响应
export interface PoseRecordingResponse {
  success: boolean;
  pose_id: string;
  session_id: string;
  joints_recorded: number;
  recording_quality: number;
  timestamp: string;
}

// 视觉动作模仿 - 可用动作类型
export interface AvailableAction {
  action_id: string;
  action_name: string;
  category: string;
  description: string;
  complexity: 'simple' | 'medium' | 'complex';
  body_parts_required: string[];
  estimated_duration: number;
  demonstration_count: number;
}

// 视觉动作模仿 - 可用动作响应
export interface AvailableActionsResponse {
  success: boolean;
  total_actions: number;
  actions: AvailableAction[];
  categories: string[];
  timestamp: string;
}

// 通用服务信息
export interface ServiceInfo {
  service_name: string;
  version: string;
  status: 'active' | 'maintenance' | 'offline';
  capabilities: string[];
  supported_modalities: string[];
  performance_metrics: Record<string, number>;
  last_updated: string;
}