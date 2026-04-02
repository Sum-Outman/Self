import { apiClient } from './client';

// 机器人管理相关类型
export interface RobotStatus {
  hardware_available: boolean;
  hardware_status: {
    simulation_available: boolean;
    roslibpy_available: boolean;
    gazebo_running: boolean;
    pybullet_available: boolean;
    pybullet_connected: boolean;
    pybullet_physics_engine: string;
    active_simulation: string;
    robot_model_loaded: boolean;
    physics_engine: string;
    simulation_time: number;
    real_time_factor: number;
  };
  ros_connection: {
    connected: boolean;
    uri: string;
    port: number;
    topics: string[];
    services: string[];
    last_connection: string | null;
    connection_duration: number;
  };
  joint_count: number;
  sensor_count: number;
  control_modes: string[];
  simulation_enabled: boolean;
  real_robot_connected: boolean;
  battery_level: number;
  cpu_temperature: number;
  last_update: string;
  capabilities: Record<string, boolean>;
}

// 多机器人管理相关类型
export enum RobotType {
  HUMANOID = "humanoid",
  MOBILE_ROBOT = "mobile_robot",
  MANIPULATOR = "manipulator",
  AERIAL_DRONE = "aerial_drone",
  UNDERWATER_ROBOT = "underwater_robot",
  WHEELED_ROBOT = "wheeled_robot",
  TRACKED_ROBOT = "tracked_robot",
  CUSTOM = "custom"
}

export enum RobotStatusEnum {
  ONLINE = "online",
  OFFLINE = "offline",
  ERROR = "error",
  BUSY = "busy",
  IDLE = "idle",
  MAINTENANCE = "maintenance",
  SIMULATION = "simulation"
}

export enum ControlMode {
  POSITION = "position",
  VELOCITY = "velocity",
  TORQUE = "torque",
  IMPEDANCE = "impedance",
  FORCE = "force",
  TRAJECTORY = "trajectory"
}

export interface Robot {
  id: number;
  name: string;
  description?: string;
  robot_type: RobotType;
  model?: string;
  manufacturer?: string;
  status: RobotStatusEnum;
  last_seen?: string;
  battery_level: number;
  cpu_temperature?: number;
  connection_type: string;
  connection_params: Record<string, any>;
  ip_address?: string;
  port?: number;
  configuration: Record<string, any>;
  urdf_path?: string;
  simulation_engine: string;
  control_mode: ControlMode;
  capabilities: Record<string, any>;
  joint_count: number;
  sensor_count: number;
  user_id: number;
  is_public: boolean;
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

export interface RobotJoint {
  id: number;
  robot_id: number;
  name: string;
  joint_type: string;
  min_position: number;
  max_position: number;
  max_velocity: number;
  max_torque: number;
  offset: number;
  direction: number;
  current_position: number;
  current_velocity: number;
  current_torque: number;
  temperature: number;
  description?: string;
  parent_link?: string;
  child_link?: string;
  axis: string;
  created_at: string;
  updated_at: string;
}

export interface RobotSensor {
  id: number;
  robot_id: number;
  name: string;
  sensor_type: string;
  model?: string;
  manufacturer?: string;
  sampling_rate?: number;
  accuracy?: number;
  range_min?: number;
  range_max?: number;
  position_x: number;
  position_y: number;
  position_z: number;
  orientation_x: number;
  orientation_y: number;
  orientation_z: number;
  orientation_w: number;
  status: string;
  last_data?: Record<string, any>;
  last_update?: string;
  description?: string;
  calibration_data?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface RobotListResponse {
  robots: Robot[];
  total: number;
  skip: number;
  limit: number;
}

export interface RobotDetailResponse {
  robot: Robot;
  joints: RobotJoint[];
  sensors: RobotSensor[];
}

export interface ROSStatus {
  connected: boolean;
  uri: string;
  port: number;
  topics: string[];
  services: string[];
  last_connection: string | null;
  connection_duration: number;
}

export interface JointState {
  name: string;
  position: number;
  velocity: number;
  torque: number;
  temperature: number;
  voltage: number;
  current: number;
  timestamp: string;
}

export interface SensorData {
  type: string;
  name: string;
  unit: string;
  data: any;
  timestamp: string;
  accuracy: number;
  calibrated: boolean;
}

export interface RobotPose {
  pose_type: string;
  joint_positions: Record<string, number>;
  message: string;
  simulated: boolean;
  execution_time: number;
}

export interface RobotCapabilities {
  movement_control: boolean;
  joint_control: boolean;
  sensor_integration: boolean;
  ros_integration: boolean;
  gazebo_simulation: boolean;
  pybullet_simulation: boolean;
  motion_planning: boolean;
  collision_detection: boolean;
  hardware_abstraction: boolean;
  real_time_control: boolean;
  trajectory_generation: boolean;
  force_control: boolean;
  impedance_control: boolean;
  vision_based_control: boolean;
  multi_robot_coordination: boolean;
  autonomous_navigation: boolean;
  manipulation: boolean;
  locomotion: boolean;
}

export interface RobotApiResponse<T = any> {
  success: boolean;
  timestamp: string;
  data: T;
}

export const robotApi = {
  // 获取机器人状态
  async getStatus(): Promise<RobotApiResponse<RobotStatus>> {
    return apiClient.get('/robot/status');
  },

  // 获取特定机器人状态
  async getRobotStatus(robotId: number): Promise<RobotApiResponse<RobotStatus>> {
    return apiClient.get(`/robots/${robotId}/status`);
  },

  // ROS连接管理
  async connectROS(uri: string = 'http://localhost:11311', port: number = 9090): Promise<RobotApiResponse<{
    connected: boolean;
    uri: string;
    port: number;
    message: string;
  }>> {
    return apiClient.post('/robot/ros/connect', { uri, port });
  },

  async disconnectROS(): Promise<RobotApiResponse<{
    connected: boolean;
    message: string;
  }>> {
    return apiClient.post('/robot/ros/disconnect');
  },

  async getROSStatus(): Promise<RobotApiResponse<ROSStatus>> {
    return apiClient.get('/robot/ros/status');
  },

  // 关节控制
  async getJointStates(jointNames?: string[]): Promise<RobotApiResponse<{
    joints: JointState[];
    count: number;
    unit: Record<string, string>;
  }>> {
    const params = jointNames ? { joint_names: jointNames.join(',') } : {};
    return apiClient.get('/robot/joints', { params });
  },

  async controlJoint(
    jointName: string,
    controlMode: 'position' | 'velocity' | 'torque',
    position?: number,
    velocity?: number,
    torque?: number
  ): Promise<RobotApiResponse<{
    joint_name: string;
    control_mode: string;
    position?: number;
    velocity?: number;
    torque?: number;
    message: string;
    simulated: boolean;
  }>> {
    const params: any = { joint_name: jointName, control_mode: controlMode };
    if (position !== undefined) params.position = position;
    if (velocity !== undefined) params.velocity = velocity;
    if (torque !== undefined) params.torque = torque;
    
    return apiClient.post('/robot/joints/control', {}, { params });
  },

  // 传感器数据
  async getSensorData(sensorTypes?: string[]): Promise<RobotApiResponse<{
    sensors: SensorData[];
    count: number;
  }>> {
    const params = sensorTypes ? { sensor_types: sensorTypes.join(',') } : {};
    return apiClient.get('/robot/sensors', { params });
  },

  // 机器人姿态控制
  async setRobotPose(
    poseType: string,
    jointPositions?: Record<string, number>
  ): Promise<RobotApiResponse<RobotPose>> {
    const data: any = { pose_type: poseType };
    if (jointPositions) data.joint_positions = jointPositions;
    return apiClient.post('/robot/motion/pose', data);
  },

  // Gazebo控制
  async controlGazebo(
    action: 'start' | 'stop' | 'pause' | 'reset' | 'load_world',
    worldName?: string
  ): Promise<RobotApiResponse<{
    action: string;
    world_name?: string;
    message: string;
    simulated: boolean;
    gazebo_running: boolean;
  }>> {
    const params: any = { action };
    if (worldName) params.world_name = worldName;
    return apiClient.post('/robot/gazebo/control', {}, { params });
  },

  // PyBullet控制
  async controlPyBullet(
    action: 'connect' | 'disconnect' | 'step_simulation' | 'reset_simulation' | 'load_urdf',
    urdfPath?: string,
    physicsEngine?: string
  ): Promise<RobotApiResponse<{
    action: string;
    urdf_path?: string;
    physics_engine?: string;
    message: string;
    simulated: boolean;
    pybullet_connected: boolean;
  }>> {
    const params: any = { action };
    if (urdfPath) params.urdf_path = urdfPath;
    if (physicsEngine) params.physics_engine = physicsEngine;
    return apiClient.post('/robot/pybullet/control', {}, { params });
  },

  // 获取机器人能力
  async getCapabilities(): Promise<RobotApiResponse<{
    capabilities: RobotCapabilities;
    supported_count: number;
    total_count: number;
    compatibility: {
      ros_versions: string[];
      gazebo_versions: string[];
      simulation_engines: string[];
      robot_models: string[];
    };
  }>> {
    return apiClient.get('/robot/capabilities');
  },

  // 多机器人管理API
  async getRobots(
    skip: number = 0,
    limit: number = 100,
    robotType?: RobotType,
    statusFilter?: RobotStatusEnum
  ): Promise<RobotApiResponse<RobotListResponse>> {
    const params: any = { skip, limit };
    if (robotType) params.robot_type = robotType;
    if (statusFilter) params.status_filter = statusFilter;
    return apiClient.get('/robots/', { params });
  },

  async getRobot(robotId: number): Promise<RobotApiResponse<RobotDetailResponse>> {
    return apiClient.get(`/robots/${robotId}`);
  },

  async createRobot(
    name: string,
    robotType: RobotType = RobotType.HUMANOID,
    description?: string,
    model?: string,
    manufacturer?: string,
    configuration: Record<string, any> = {},
    urdfPath?: string,
    simulationEngine: string = 'gazebo',
    controlMode: ControlMode = ControlMode.POSITION
  ): Promise<RobotApiResponse<Robot>> {
    const params: any = {
      name,
      robot_type: robotType,
      description,
      model,
      manufacturer,
      configuration: JSON.stringify(configuration),
      urdf_path: urdfPath,
      simulation_engine: simulationEngine,
      control_mode: controlMode
    };
    // 移除undefined参数
    Object.keys(params).forEach(key => params[key] === undefined && delete params[key]);
    if (params.configuration) params.configuration = JSON.parse(params.configuration); // 重新解析为对象
    return apiClient.post('/robots/', {}, { params });
  },

  async connectRobot(robotId: number): Promise<RobotApiResponse<{
    robot_id: number;
    name: string;
    status: RobotStatusEnum;
    last_seen?: string;
  }>> {
    return apiClient.post(`/robots/${robotId}/connect`);
  },

  async disconnectRobot(robotId: number): Promise<RobotApiResponse<{
    robot_id: number;
    name: string;
    status: RobotStatusEnum;
  }>> {
    return apiClient.post(`/robots/${robotId}/disconnect`);
  },

  async setDefaultRobot(robotId: number): Promise<RobotApiResponse<Robot>> {
    return apiClient.post(`/robots/${robotId}/set-default`);
  },

  async getDefaultRobot(): Promise<RobotApiResponse<Robot>> {
    return apiClient.get('/robots/default');
  },

  // WebSocket连接（返回WebSocket URL）
  getWebSocketUrl(): string {
    const baseUrl = apiClient.getBaseUrl().replace('http://', 'ws://').replace('https://', 'wss://');
    return `${baseUrl}/robot/ws`;
  },

  // 运动控制API
  async planPath(
    startPosition: number[],
    goalPosition: number[],
    algorithm: string = 'astar',
    gridSize: number = 0.1,
    maxIterations: number = 1000,
    simulationType: string = 'pybullet'
  ): Promise<RobotApiResponse<{
    path: number[][];
    path_length: number;
    computation_time: number;
    nodes_explored: number;
    algorithm: string;
    simulation_type: string;
  }>> {
    return apiClient.post('/motion-control/plan-path', {
      start_position: startPosition,
      goal_position: goalPosition,
      algorithm,
      grid_size: gridSize,
      max_iterations: maxIterations,
      simulation_type: simulationType
    });
  },

  async executePath(
    path: number[][],
    speed: number = 0.1,
    simulationType: string = 'pybullet'
  ): Promise<RobotApiResponse<{
    points_executed: number;
    speed: number;
    simulation_type: string;
  }>> {
    return apiClient.post('/motion-control/execute-path', {
      path,
      speed,
      simulation_type: simulationType
    });
  },

  async moveToPosition(
    targetPosition: number[],
    speed: number = 0.1,
    simulationType: string = 'pybullet'
  ): Promise<RobotApiResponse<{
    target_position: number[];
    speed: number;
    simulation_type: string;
  }>> {
    return apiClient.post('/motion-control/move-to-position', {
      target_position: targetPosition,
      speed,
      simulation_type: simulationType
    });
  },

  async getSimulationInfo(
    simulationType: string = 'pybullet'
  ): Promise<RobotApiResponse<{
    initialized: boolean;
    connected: boolean;
    simulation_type: string;
    info: any;
  }>> {
    return apiClient.get('/motion-control/simulation-info', {
      params: { simulation_type: simulationType }
    });
  },

  // 示范学习API
  async startDemonstrationRecording(
    robotId: number,
    name: string,
    description: string = '',
    demonstrationType: string = 'joint_control',
    config: Record<string, any> = {}
  ): Promise<RobotApiResponse<{
    demonstration_id: number;
    name: string;
    message: string;
    started_at: string;
  }>> {
    return apiClient.post(`/robot/demonstration/${robotId}/start-recording`, {
      name,
      description,
      demonstration_type: demonstrationType,
      config
    });
  },

  async stopDemonstrationRecording(
    robotId: number
  ): Promise<RobotApiResponse<{
    demonstration_id: number;
    name: string;
    message: string;
    stopped_at: string;
    total_frames: number;
  }>> {
    return apiClient.post(`/robot/demonstration/${robotId}/stop-recording`);
  },

  async pauseDemonstrationRecording(
    robotId: number
  ): Promise<RobotApiResponse<{
    demonstration_id: number;
    name: string;
    message: string;
    paused_at: string;
    frames_recorded: number;
    recording_paused: boolean;
  }>> {
    return apiClient.post(`/robot/demonstration/${robotId}/pause-recording`);
  },

  async resumeDemonstrationRecording(
    robotId: number
  ): Promise<RobotApiResponse<{
    demonstration_id: number;
    name: string;
    message: string;
    resumed_at: string;
    frames_recorded: number;
    recording_resumed: boolean;
  }>> {
    return apiClient.post(`/robot/demonstration/${robotId}/resume-recording`);
  },

  async recordRobotState(
    robotId: number,
    jointPositions: Record<string, number> = {},
    jointVelocities?: Record<string, number>,
    jointTorques?: Record<string, number>,
    sensorData?: Record<string, any>,
    imuData?: Record<string, any>,
    controlCommands?: Record<string, any>,
    environmentState?: Record<string, any>
  ): Promise<RobotApiResponse<{
    frame_id: number;
    timestamp: string;
    message: string;
  }>> {
    return apiClient.post(`/robot/demonstration/${robotId}/record-state`, {
      joint_positions: jointPositions,
      joint_velocities: jointVelocities,
      joint_torques: jointTorques,
      sensor_data: sensorData,
      imu_data: imuData,
      control_commands: controlCommands,
      environment_state: environmentState
    });
  },

  async startDemonstrationPlayback(
    robotId: number,
    demonstrationId: number
  ): Promise<RobotApiResponse<{
    demonstration_id: number;
    robot_id: number;
    message: string;
    started_at: string;
  }>> {
    return apiClient.post(`/robot/demonstration/${robotId}/start-playback/${demonstrationId}`);
  },

  async pauseDemonstrationPlayback(
    robotId: number
  ): Promise<RobotApiResponse<{
    demonstration_id: number;
    robot_id: number;
    message: string;
    paused_at: string;
    playback_paused: boolean;
  }>> {
    return apiClient.post(`/robot/demonstration/${robotId}/pause-playback`);
  },

  async resumeDemonstrationPlayback(
    robotId: number
  ): Promise<RobotApiResponse<{
    demonstration_id: number;
    robot_id: number;
    message: string;
    resumed_at: string;
    playback_resumed: boolean;
  }>> {
    return apiClient.post(`/robot/demonstration/${robotId}/resume-playback`);
  },

  async stopDemonstrationPlayback(
    robotId: number
  ): Promise<RobotApiResponse<{
    demonstration_id: number;
    robot_id: number;
    message: string;
    stopped_at: string;
  }>> {
    return apiClient.post(`/robot/demonstration/${robotId}/stop-playback`);
  }
};