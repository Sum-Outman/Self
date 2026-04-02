import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  Bot,
  Cpu,
  Zap,
  Activity,
  Thermometer,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  Play,
  StopCircle,
  Wifi,
  WifiOff,
  BarChart3,
  Eye,
  Sliders,
  Target,
  Move3d,
  Brain,
  Battery,
  Server,
  Globe,
  Clock,
  Box,
  Film,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { robotApi, RobotStatus, JointState, SensorData, ROSStatus, Robot } from '../services/api/robot';
import RobotScene, { PathData } from '../components/Robot3D/RobotModel';
import { convertJointStatesToPose } from '../components/Robot3D/robotUtils';

const RobotManagementPage: React.FC = () => {
  const { user: _user } = useAuth();
  const [activeTab, setActiveTab] = useState<'overview' | 'control' | 'ros' | 'sensors' | 'gazebo' | 'pybullet' | 'motion-control' | 'visualization' | 'capabilities' | 'demonstration'>('overview');
  const [robotStatus, setRobotStatus] = useState<RobotStatus | null>(null);
  const [rosStatus, setRosStatus] = useState<ROSStatus | null>(null);
  const [jointStates, setJointStates] = useState<JointState[]>([]);
  const [sensorData, setSensorData] = useState<SensorData[]>([]);
  const [isLoading, setIsLoading] = useState({
    status: false,
    ros: false,
    joints: false,
    sensors: false,
    recording: false,
    playback: false,
    demonstration: false,
  });
  const [isPolling, setIsPolling] = useState({
    status: true,
    joints: true,
    sensors: true,
  });

  // 示范学习状态

  // ROS连接状态
  const [rosUri, setRosUri] = useState('http://localhost:11311');
  const [rosPort, setRosPort] = useState(9090);
  const [isConnectingROS, setIsConnectingROS] = useState(false);

  // 关节控制状态
  const [selectedJoint, setSelectedJoint] = useState<string>('head_yaw');
  const [controlMode, setControlMode] = useState<'position' | 'velocity' | 'torque'>('position');
  const [controlValue, setControlValue] = useState<number>(0);
  const [isControllingJoint, setIsControllingJoint] = useState(false);

  // Gazebo控制
  const [gazeboAction, setGazeboAction] = useState<'start' | 'stop' | 'pause' | 'reset' | 'load_world'>('start');
  const [worldName, setWorldName] = useState('empty.world');
  const [isControllingGazebo, setIsControllingGazebo] = useState(false);

  // PyBullet控制
  const [pybulletAction, setPybulletAction] = useState<'connect' | 'disconnect' | 'step_simulation' | 'reset_simulation' | 'load_urdf'>('connect');
  const [urdfPath, setUrdfPath] = useState('humanoid.urdf');
  const [physicsEngine, setPhysicsEngine] = useState('BULLET');
  const [isControllingPyBullet, setIsControllingPyBullet] = useState(false);

  // 姿态控制
  const [poseType, setPoseType] = useState<'stand' | 'sit' | 'walk_ready' | 'custom'>('stand');
  const [customPose, _setCustomPose] = useState<Record<string, number>>({});
  const [isSettingPose, setIsSettingPose] = useState(false);

  // WebSocket实时数据
  const [websocketConnected, setWebsocketConnected] = useState(false);
  const [realTimeData, setRealTimeData] = useState<any>(null);
  const websocketRef = useRef<WebSocket | null>(null);

  // 示范学习状态
  const [demonstrationStatus, setDemonstrationStatus] = useState<any>(null);
  const [isRecordingDemonstration, setIsRecordingDemonstration] = useState(false);
  const [isPlayingDemonstration, setIsPlayingDemonstration] = useState(false);
  const [demonstrationName, setDemonstrationName] = useState('');
  const [demonstrationDescription, setDemonstrationDescription] = useState('');
  const [currentDemonstrationId, setCurrentDemonstrationId] = useState<number | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [loopPlayback, setLoopPlayback] = useState(false);
  const [demonstrationFrames, setDemonstrationFrames] = useState<any[]>([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);

  // 多机器人管理状态
  const [robots, setRobots] = useState<Robot[]>([]);
  const [selectedRobotId, setSelectedRobotId] = useState<number | null>(null);
  const [selectedRobot, setSelectedRobot] = useState<Robot | null>(null);
  const [loadingRobots, setLoadingRobots] = useState(false);

  // 运动控制状态
  const [startPosition, setStartPosition] = useState<number[]>([0, 0, 0.5]);
  const [goalPosition, setGoalPosition] = useState<number[]>([5, 5, 0.5]);
  const [pathPlanningAlgorithm, setPathPlanningAlgorithm] = useState<string>('astar');
  const [plannedPath, setPlannedPath] = useState<number[][]>([]);
  const [isPlanningPath, setIsPlanningPath] = useState<boolean>(false);
  const [pathResult, setPathResult] = useState<any>(null);
  const [isExecutingPath, setIsExecutingPath] = useState<boolean>(false);
  const [moveTargetPosition, setMoveTargetPosition] = useState<number[]>([2, 2, 0.5]);
  const [isMovingToPosition, setIsMovingToPosition] = useState<boolean>(false);
  const [simulationType, setSimulationType] = useState<'pybullet' | 'gazebo'>('pybullet');

  // 转换路径数据用于3D可视化
  const visualizationPath = React.useMemo((): PathData | null => {
    if (!plannedPath || plannedPath.length < 2) return null;
    
    // 转换number[][]到Array<[number, number, number]>
    const points = plannedPath.map(point => {
      // 确保每个点有3个坐标，如果没有则用0填充
      const [x = 0, y = 0, z = 0] = point;
      return [x, y, z] as [number, number, number];
    });
    
    return {
      points,
      color: '#00ff00', // 绿色路径
      lineWidth: 3
    };
  }, [plannedPath]);

  // 加载机器人状态
  const loadRobotStatus = useCallback(async () => {
    if (!isPolling.status) return;
    
    setIsLoading(prev => ({ ...prev, status: true }));
    try {
      const response = await robotApi.getStatus();
      if (response.success) {
        setRobotStatus(response.data);
      }
    } catch (error) {
      console.error('加载机器人状态失败:', error);
      toast.error('加载机器人状态失败');
    } finally {
      setIsLoading(prev => ({ ...prev, status: false }));
    }
  }, [isPolling.status]);

  // 加载机器人列表
  const loadRobots = useCallback(async () => {
    setLoadingRobots(true);
    try {
      // 获取机器人列表
      const robotsResponse = await robotApi.getRobots();
      if (robotsResponse.success) {
        setRobots(robotsResponse.data.robots);
        
        // 如果没有选中的机器人，尝试获取默认机器人
        if (!selectedRobotId && robotsResponse.data.robots.length > 0) {
          // 首先检查是否有默认机器人
          const defaultResponse = await robotApi.getDefaultRobot();
          if (defaultResponse.success) {
            setSelectedRobotId(defaultResponse.data.id);
            setSelectedRobot(defaultResponse.data);
          } else {
            // 如果没有默认机器人，选择第一个机器人
            const firstRobot = robotsResponse.data.robots[0];
            setSelectedRobotId(firstRobot.id);
            setSelectedRobot(firstRobot);
          }
        }
      }
    } catch (error) {
      console.error('加载机器人列表失败:', error);
      toast.error('加载机器人列表失败');
    } finally {
      setLoadingRobots(false);
    }
  }, [selectedRobotId]);

  // 处理机器人切换
  const handleRobotChange = useCallback(async (robotId: number) => {
    setSelectedRobotId(robotId);
    
    // 找到选中的机器人
    const robot = robots.find(r => r.id === robotId);
    if (robot) {
      setSelectedRobot(robot);
      
      // 可以在这里添加加载特定机器人状态的功能
      // 例如：robotApi.connectRobot(robotId) 或加载特定机器人的关节状态
      toast.success(`已切换到机器人: ${robot.name}`);
      
      // 加载选定机器人的关节状态和传感器数据
      if (robot.id) {
        try {
          const statusResponse = await robotApi.getRobotStatus(robot.id);
          if (statusResponse.success) {
            setRobotStatus(statusResponse.data);
          }
        } catch (error) {
          console.error('加载机器人状态失败:', error);
        }
      }
    }
  }, [robots]);

  // 加载ROS状态
  const loadROSStatus = useCallback(async () => {
    setIsLoading(prev => ({ ...prev, ros: true }));
    try {
      const response = await robotApi.getROSStatus();
      if (response.success) {
        setRosStatus(response.data);
      }
    } catch (error) {
      console.error('加载ROS状态失败:', error);
    } finally {
      setIsLoading(prev => ({ ...prev, ros: false }));
    }
  }, []);

  // 加载关节状态
  const loadJointStates = useCallback(async () => {
    if (!isPolling.joints) return;
    
    setIsLoading(prev => ({ ...prev, joints: true }));
    try {
      const response = await robotApi.getJointStates();
      if (response.success) {
        setJointStates(response.data.joints);
      }
    } catch (error) {
      console.error('加载关节状态失败:', error);
    } finally {
      setIsLoading(prev => ({ ...prev, joints: false }));
    }
  }, [isPolling.joints]);

  // 加载传感器数据
  const loadSensorData = useCallback(async () => {
    if (!isPolling.sensors) return;
    
    setIsLoading(prev => ({ ...prev, sensors: true }));
    try {
      const response = await robotApi.getSensorData();
      if (response.success) {
        setSensorData(response.data.sensors);
      }
    } catch (error) {
      console.error('加载传感器数据失败:', error);
    } finally {
      setIsLoading(prev => ({ ...prev, sensors: false }));
    }
  }, [isPolling.sensors]);

  // 连接/断开ROS
  const handleConnectROS = async () => {
    setIsConnectingROS(true);
    try {
      const response = await robotApi.connectROS(rosUri, rosPort);
      if (response.success) {
        toast.success(response.data.message);
        loadROSStatus();
      } else {
        toast.error('ROS连接失败');
      }
    } catch (error) {
      console.error('ROS连接失败:', error);
      toast.error('ROS连接失败');
    } finally {
      setIsConnectingROS(false);
    }
  };

  const handleDisconnectROS = async () => {
    try {
      const response = await robotApi.disconnectROS();
      if (response.success) {
        toast.success(response.data.message);
        loadROSStatus();
      }
    } catch (error) {
      console.error('ROS断开失败:', error);
      toast.error('ROS断开失败');
    }
  };

  // 控制关节
  const handleControlJoint = async () => {
    setIsControllingJoint(true);
    try {
      const params: any = { jointName: selectedJoint, controlMode };
      if (controlMode === 'position') params.position = controlValue;
      if (controlMode === 'velocity') params.velocity = controlValue;
      if (controlMode === 'torque') params.torque = controlValue;

      const response = await robotApi.controlJoint(
        selectedJoint,
        controlMode,
        controlMode === 'position' ? controlValue : undefined,
        controlMode === 'velocity' ? controlValue : undefined,
        controlMode === 'torque' ? controlValue : undefined
      );
      
      if (response.success) {
        toast.success(response.data.message);
        loadJointStates();
      }
    } catch (error) {
      console.error('关节控制失败:', error);
      toast.error('关节控制失败');
    } finally {
      setIsControllingJoint(false);
    }
  };

  // 控制Gazebo
  const handleControlGazebo = async () => {
    setIsControllingGazebo(true);
    try {
      const response = await robotApi.controlGazebo(gazeboAction, worldName);
      if (response.success) {
        toast.success(response.data.message);
      }
    } catch (error) {
      console.error('Gazebo控制失败:', error);
      toast.error('Gazebo控制失败');
    } finally {
      setIsControllingGazebo(false);
    }
  };

  // 控制PyBullet
  const handleControlPyBullet = async () => {
    setIsControllingPyBullet(true);
    try {
      const response = await robotApi.controlPyBullet(pybulletAction, urdfPath, physicsEngine);
      if (response.success) {
        toast.success(response.data.message);
      }
    } catch (error) {
      console.error('PyBullet控制失败:', error);
      toast.error('PyBullet控制失败');
    } finally {
      setIsControllingPyBullet(false);
    }
  };

  // 设置机器人姿态
  const handleSetPose = async () => {
    setIsSettingPose(true);
    try {
      const jointPositions = poseType === 'custom' ? customPose : undefined;
      const response = await robotApi.setRobotPose(poseType, jointPositions);
      if (response.success) {
        toast.success(response.data.message);
        loadJointStates();
      }
    } catch (error) {
      console.error('设置姿态失败:', error);
      toast.error('设置姿态失败');
    } finally {
      setIsSettingPose(false);
    }
  };

  // 运动控制函数
  const handlePlanPath = async () => {
    setIsPlanningPath(true);
    try {
      const response = await robotApi.planPath(
        startPosition,
        goalPosition,
        pathPlanningAlgorithm,
        0.1, // gridSize
        1000, // maxIterations
        simulationType
      );
      if (response.success) {
        setPlannedPath(response.data.path);
        setPathResult(response.data);
        toast.success(`路径规划成功！路径长度: ${response.data.path_length.toFixed(2)}米`);
      } else {
        toast.error('路径规划失败');
      }
    } catch (error) {
      console.error('路径规划失败:', error);
      toast.error('路径规划失败');
    } finally {
      setIsPlanningPath(false);
    }
  };

  const handleExecutePath = async () => {
    if (plannedPath.length === 0) {
      toast.error('没有可执行的路径');
      return;
    }
    
    setIsExecutingPath(true);
    try {
      const response = await robotApi.executePath(
        plannedPath,
        0.1, // speed
        simulationType
      );
      if (response.success) {
        toast.success(`路径执行成功！执行了${response.data.points_executed}个点`);
      } else {
        toast.error('路径执行失败');
      }
    } catch (error) {
      console.error('路径执行失败:', error);
      toast.error('路径执行失败');
    } finally {
      setIsExecutingPath(false);
    }
  };

  const handleMoveToPosition = async () => {
    setIsMovingToPosition(true);
    try {
      const response = await robotApi.moveToPosition(
        moveTargetPosition,
        0.1, // speed
        simulationType
      );
      if (response.success) {
        toast.success(`机器人移动到目标位置成功！`);
      } else {
        toast.error('移动机器人失败');
      }
    } catch (error) {
      console.error('移动机器人失败:', error);
      toast.error('移动机器人失败');
    } finally {
      setIsMovingToPosition(false);
    }
  };

  const handleGetSimulationInfo = async () => {
    try {
      const response = await robotApi.getSimulationInfo(simulationType);
      if (response.success) {
        toast.success(`仿真环境信息: ${response.data.initialized ? '已初始化' : '未初始化'}, ${response.data.connected ? '已连接' : '未连接'}`);
        console.log('仿真环境信息:', response.data);
      }
    } catch (error) {
      console.error('获取仿真信息失败:', error);
      toast.error('获取仿真信息失败');
    }
  };

  // WebSocket连接
  const connectWebSocket = () => {
    try {
      const wsUrl = robotApi.getWebSocketUrl();
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket连接已建立');
        setWebsocketConnected(true);
        toast.success('实时数据连接已建立');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setRealTimeData(data);
        } catch (error) {
          console.error('解析WebSocket数据失败:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket错误:', error);
        toast.error('实时数据连接错误');
      };
      
      ws.onclose = () => {
        console.log('WebSocket连接已关闭');
        setWebsocketConnected(false);
      };
      
      websocketRef.current = ws;
    } catch (error) {
      console.error('WebSocket连接失败:', error);
      toast.error('WebSocket连接失败');
    }
  };

  const disconnectWebSocket = () => {
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
      setWebsocketConnected(false);
      toast.success('实时数据连接已断开');
    }
  };

  // 初始化加载
  useEffect(() => {
    loadRobotStatus();
    loadRobots();
    loadROSStatus();
    loadJointStates();
    loadSensorData();

    // 设置轮询定时器
    const statusInterval = setInterval(loadRobotStatus, 5000);
    const jointsInterval = setInterval(loadJointStates, 1000);
    const sensorsInterval = setInterval(loadSensorData, 2000);

    return () => {
      clearInterval(statusInterval);
      clearInterval(jointsInterval);
      clearInterval(sensorsInterval);
      disconnectWebSocket();
    };
  }, [loadRobotStatus, loadRobots, loadROSStatus, loadJointStates, loadSensorData]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* 页面标题 */}
      <div className="px-6 py-8 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
                <Bot className="w-8 h-8 text-gray-600 dark:text-gray-400" />
                人形机器人管理
              </h1>
              <p className="mt-2 text-gray-600 dark:text-gray-400">
                管理机器人控制、ROS接入、Gazebo仿真和传感器数据
              </p>
            </div>
            <div className="flex items-center space-x-4">
              {/* 机器人选择器 */}
              <div className="relative">
                <select
                  value={selectedRobotId || ''}
                  onChange={(e) => handleRobotChange(parseInt(e.target.value))}
                  disabled={loadingRobots || robots.length === 0}
                  className="px-4 py-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg border border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-600 dark:focus:ring-gray-400 focus:border-transparent appearance-none pr-10"
                >
                  {loadingRobots ? (
                    <option value="">加载机器人列表...</option>
                  ) : robots.length === 0 ? (
                    <option value="">暂无机器人</option>
                  ) : (
                    <>
                      <option value="">选择机器人</option>
                      {robots.map((robot) => (
                        <option key={robot.id} value={robot.id}>
                          {robot.name} {robot.is_default ? '(默认)' : ''} - {robot.status}
                        </option>
                      ))}
                    </>
                  )}
                </select>
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
              </div>

              <button
                onClick={loadRobotStatus}
                disabled={isLoading.status}
                className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 flex items-center gap-2"
              >
                <RefreshCw className={`w-4 h-4 ${isLoading.status ? 'animate-spin' : ''}`} />
                刷新
              </button>
              <button
                onClick={websocketConnected ? disconnectWebSocket : connectWebSocket}
                className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                  websocketConnected
                    ? 'bg-gray-800 hover:bg-gray-700 text-white'
                    : 'bg-gray-600 hover:bg-gray-500 text-white'
                }`}
              >
                {websocketConnected ? (
                  <>
                    <WifiOff className="w-4 h-4" />
                    断开实时数据
                  </>
                ) : (
                  <>
                    <Wifi className="w-4 h-4" />
                    连接实时数据
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 状态指示器 */}
      <div className="px-6 py-4 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              {selectedRobot && (
                <div className="flex items-center space-x-2 bg-white dark:bg-gray-800 px-3 py-1 rounded-lg border border-gray-200 dark:border-gray-700">
                  <Bot className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {selectedRobot.name}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    selectedRobot.status === 'online' ? 'bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-300' :
                    selectedRobot.status === 'offline' ? 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300' :
                    selectedRobot.status === 'error' ? 'bg-gray-800 text-gray-100 dark:bg-gray-900 dark:text-gray-300' :
                    'bg-gray-300 text-gray-800 dark:bg-gray-600 dark:text-gray-300'
                  }`}>
                    {selectedRobot.status === 'online' ? '在线' :
                     selectedRobot.status === 'offline' ? '离线' :
                     selectedRobot.status === 'error' ? '错误' :
                     selectedRobot.status}
                  </span>
                </div>
              )}
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${robotStatus?.hardware_available ? 'bg-gray-600' : 'bg-gray-800'}`} />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  {robotStatus?.hardware_available ? '硬件就绪' : '硬件不可用'}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${rosStatus?.connected ? 'bg-gray-600' : 'bg-gray-800'}`} />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  {rosStatus?.connected ? 'ROS已连接' : 'ROS未连接'}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${robotStatus?.hardware_status?.gazebo_running ? 'bg-gray-600' : 'bg-gray-500'}`} />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  {robotStatus?.hardware_status?.gazebo_running ? 'Gazebo运行中' : 'Gazebo未运行'}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${robotStatus?.hardware_status?.pybullet_connected ? 'bg-gray-600' : 'bg-gray-400'}`} />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  {robotStatus?.hardware_status?.pybullet_connected ? 'PyBullet已连接' : 'PyBullet未连接'}
                </span>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Battery className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {robotStatus?.battery_level?.toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <Thermometer className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {robotStatus?.cpu_temperature?.toFixed(1)}°C
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 导航标签 */}
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="max-w-7xl mx-auto">
          <div className="flex space-x-1 overflow-x-auto">
            {[
              { id: 'overview', label: '概览', icon: Eye },
              { id: 'control', label: '关节控制', icon: Move3d },
              { id: 'ros', label: 'ROS接入', icon: Globe },
              { id: 'sensors', label: '传感器', icon: Activity },
              { id: 'gazebo', label: 'Gazebo仿真', icon: Cpu },
              { id: 'pybullet', label: 'PyBullet仿真', icon: Box },
              { id: 'motion-control', label: '运动控制', icon: Target },
              { id: 'visualization', label: '3D可视化', icon: Bot },
              { id: 'capabilities', label: '能力列表', icon: Brain },
              { id: 'demonstration', label: '示范学习', icon: Film },
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors ${
                    activeTab === tab.id
                      ? 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-300'
                      : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* 主要内容区域 */}
      <div className="px-6 py-8">
        <div className="max-w-7xl mx-auto">
          {/* 概览标签页 */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* 系统状态卡片 */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">关节数量</p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        {robotStatus?.joint_count || 0}
                      </p>
                    </div>
                    <Move3d className="w-8 h-8 text-gray-600 dark:text-gray-400" />
                  </div>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">传感器数量</p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        {robotStatus?.sensor_count || 0}
                      </p>
                    </div>
                    <Activity className="w-8 h-8 text-gray-600" />
                  </div>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">ROS连接</p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        {rosStatus?.connected ? '已连接' : '未连接'}
                      </p>
                    </div>
                    <div className={`p-2 rounded-full ${rosStatus?.connected ? 'bg-gray-600 dark:bg-gray-700/30' : 'bg-gray-800 dark:bg-gray-900/30'}`}>
                      {rosStatus?.connected ? (
                        <Wifi className="w-6 h-6 text-gray-600 dark:text-gray-600" />
                      ) : (
                        <WifiOff className="w-6 h-6 text-gray-800 dark:text-gray-800" />
                      )}
                    </div>
                  </div>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">控制模式</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {robotStatus?.control_modes?.join(', ') || '无'}
                      </p>
                    </div>
                    <Sliders className="w-8 h-8 text-gray-600" />
                  </div>
                </div>
              </div>

              {/* ROS状态卡片 */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                    <Globe className="w-5 h-5" />
                    ROS连接状态
                  </h3>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600 dark:text-gray-400">连接状态</span>
                      <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${rosStatus?.connected ? 'bg-gray-700' : 'bg-gray-900'}`} />
                        <span className={`font-medium ${rosStatus?.connected ? 'text-gray-600 dark:text-gray-600' : 'text-gray-800 dark:text-gray-800'}`}>
                          {rosStatus?.connected ? '已连接' : '未连接'}
                        </span>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600 dark:text-gray-400">URI</span>
                      <span className="font-medium text-gray-900 dark:text-white">{rosStatus?.uri}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600 dark:text-gray-400">端口</span>
                      <span className="font-medium text-gray-900 dark:text-white">{rosStatus?.port}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600 dark:text-gray-400">话题数量</span>
                      <span className="font-medium text-gray-900 dark:text-white">{rosStatus?.topics?.length || 0}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600 dark:text-gray-400">服务数量</span>
                      <span className="font-medium text-gray-900 dark:text-white">{rosStatus?.services?.length || 0}</span>
                    </div>
                    <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex space-x-3">
                        <button
                          onClick={handleConnectROS}
                          disabled={isConnectingROS || rosStatus?.connected}
                          className="flex-1 px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                          {isConnectingROS ? (
                            <>
                              <RefreshCw className="w-4 h-4 animate-spin" />
                              连接中...
                            </>
                          ) : (
                            <>
                              <Wifi className="w-4 h-4" />
                              连接ROS
                            </>
                          )}
                        </button>
                        <button
                          onClick={handleDisconnectROS}
                          disabled={!rosStatus?.connected}
                          className="flex-1 px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          断开连接
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 关节状态卡片 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                      <Move3d className="w-5 h-5" />
                      关节状态
                    </h3>
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={isPolling.joints}
                        onChange={(e) => setIsPolling(prev => ({ ...prev, joints: e.target.checked }))}
                        className="rounded text-gray-700"
                      />
                      实时更新
                    </label>
                  </div>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {jointStates.slice(0, 5).map((joint) => (
                      <div key={joint.name} className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50">
                        <div>
                          <p className="font-medium text-gray-900 dark:text-white">{joint.name}</p>
                          <p className="text-sm text-gray-500 dark:text-gray-400">
                            位置: {joint.position.toFixed(3)} rad
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            {joint.torque.toFixed(2)} Nm
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            {joint.temperature.toFixed(1)}°C
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                  <button
                    onClick={() => setActiveTab('control')}
                    className="w-full mt-4 px-4 py-2 text-gray-700 dark:text-gray-700 hover:bg-gray-700 dark:hover:bg-gray-700/20 rounded-lg border border-gray-700 dark:border-gray-700"
                  >
                    查看更多关节 →
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* 关节控制标签页 */}
          {activeTab === 'control' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* 关节选择和控制面板 */}
                <div className="lg:col-span-2 space-y-6">
                  <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">关节控制面板</h3>
                    <div className="space-y-6">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            选择关节
                          </label>
                          <select
                            value={selectedJoint}
                            onChange={(e) => setSelectedJoint(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          >
                            {jointStates.map((joint) => (
                              <option key={joint.name} value={joint.name}>
                                {joint.name}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            控制模式
                          </label>
                          <select
                            value={controlMode}
                            onChange={(e) => setControlMode(e.target.value as any)}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          >
                            <option value="position">位置控制</option>
                            <option value="velocity">速度控制</option>
                            <option value="torque">扭矩控制</option>
                          </select>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          控制值
                          <span className="ml-2 text-xs text-gray-500">
                            {controlMode === 'position' ? '(弧度)' : 
                             controlMode === 'velocity' ? '(弧度/秒)' : 
                             '(Nm)'}
                          </span>
                        </label>
                        <div className="flex items-center space-x-4">
                          <input
                            type="range"
                            min={controlMode === 'position' ? -3.14 : controlMode === 'velocity' ? -5 : 0}
                            max={controlMode === 'position' ? 3.14 : controlMode === 'velocity' ? 5 : 10}
                            step="0.01"
                            value={controlValue}
                            onChange={(e) => setControlValue(parseFloat(e.target.value))}
                            className="flex-1"
                          />
                          <input
                            type="number"
                            value={controlValue}
                            onChange={(e) => setControlValue(parseFloat(e.target.value) || 0)}
                            className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                            step="0.01"
                          />
                        </div>
                      </div>
                      <button
                        onClick={handleControlJoint}
                        disabled={isControllingJoint}
                        className="w-full px-4 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                      >
                        {isControllingJoint ? (
                          <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            控制中...
                          </>
                        ) : (
                          <>
                            <Target className="w-4 h-4" />
                            应用控制
                          </>
                        )}
                      </button>
                    </div>
                  </div>

                  {/* 姿态控制 */}
                  <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">预定义姿态</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {[
                        { type: 'stand', label: '站立', icon: Activity },
                        { type: 'sit', label: '坐下', icon: StopCircle },
                        { type: 'walk_ready', label: '行走准备', icon: Play },
                      ].map((pose) => {
                        const Icon = pose.icon;
                        return (
                          <button
                            key={pose.type}
                            onClick={() => {
                              setPoseType(pose.type as any);
                              handleSetPose();
                            }}
                            disabled={isSettingPose}
                            className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 flex flex-col items-center justify-center gap-3 disabled:opacity-50"
                          >
                            <Icon className="w-8 h-8 text-gray-700" />
                            <span className="font-medium text-gray-900 dark:text-white">{pose.label}</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* 关节状态列表 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">关节状态</h3>
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                      {jointStates.length} 个关节
                    </span>
                  </div>
                  <div className="space-y-3 max-h-[600px] overflow-y-auto">
                    {jointStates.map((joint) => (
                      <div
                        key={joint.name}
                        className={`p-3 rounded-lg border ${
                          joint.name === selectedJoint
                            ? 'border-gray-700 bg-gray-700 dark:bg-gray-700/20'
                            : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/50'
                        }`}
                        onClick={() => setSelectedJoint(joint.name)}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white">{joint.name}</p>
                            <div className="flex items-center space-x-4 mt-1 text-xs">
                              <span className="text-gray-500 dark:text-gray-400">
                                P: {joint.position.toFixed(3)}
                              </span>
                              <span className="text-gray-500 dark:text-gray-400">
                                V: {joint.velocity.toFixed(3)}
                              </span>
                              <span className="text-gray-500 dark:text-gray-400">
                                T: {joint.torque.toFixed(2)}
                              </span>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-sm text-gray-500 dark:text-gray-400">
                              {joint.temperature.toFixed(1)}°C
                            </p>
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                              {joint.voltage.toFixed(1)}V
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ROS接入标签页 */}
          {activeTab === 'ros' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* ROS连接配置 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">ROS连接配置</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        ROS Master URI
                      </label>
                      <input
                        type="text"
                        value={rosUri}
                        onChange={(e) => setRosUri(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        placeholder="http://localhost:11311"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        ROS Master Port
                      </label>
                      <input
                        type="number"
                        value={rosPort}
                        onChange={(e) => setRosPort(parseInt(e.target.value) || 9090)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        placeholder="9090"
                      />
                    </div>
                    <div className="pt-4">
                      <div className="flex space-x-3">
                        <button
                          onClick={handleConnectROS}
                          disabled={isConnectingROS || rosStatus?.connected}
                          className="flex-1 px-4 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                          {isConnectingROS ? (
                            <>
                              <RefreshCw className="w-4 h-4 animate-spin" />
                              连接中...
                            </>
                          ) : (
                            <>
                              <Wifi className="w-4 h-4" />
                              连接ROS
                            </>
                          )}
                        </button>
                        <button
                          onClick={handleDisconnectROS}
                          disabled={!rosStatus?.connected}
                          className="flex-1 px-4 py-3 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          断开连接
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* ROS话题和服务 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">ROS话题和服务</h3>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-white mb-2">可用话题</h4>
                      <div className="max-h-48 overflow-y-auto">
                        {rosStatus?.topics?.map((topic, index) => (
                          <div key={index} className="py-2 px-3 border-b border-gray-200 dark:border-gray-700 last:border-0">
                            <code className="text-sm text-gray-700 dark:text-gray-700">{topic}</code>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-white mb-2">可用服务</h4>
                      <div className="max-h-48 overflow-y-auto">
                        {rosStatus?.services?.map((service, index) => (
                          <div key={index} className="py-2 px-3 border-b border-gray-200 dark:border-gray-700 last:border-0">
                            <code className="text-sm text-gray-600 dark:text-gray-600">{service}</code>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 传感器标签页 */}
          {activeTab === 'sensors' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* 传感器列表 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">传感器数据</h3>
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={isPolling.sensors}
                        onChange={(e) => setIsPolling(prev => ({ ...prev, sensors: e.target.checked }))}
                        className="rounded text-gray-700"
                      />
                      实时更新
                    </label>
                  </div>
                  <div className="space-y-4 max-h-[600px] overflow-y-auto">
                    {sensorData.map((sensor) => (
                      <div key={sensor.type} className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <div className="p-2 bg-gray-700 dark:bg-gray-700/30 rounded-lg">
                              <Activity className="w-5 h-5 text-gray-700 dark:text-gray-700" />
                            </div>
                            <div>
                              <h4 className="font-medium text-gray-900 dark:text-white">{sensor.name}</h4>
                              <p className="text-sm text-gray-500 dark:text-gray-400">{sensor.type}</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <span className="inline-flex items-center gap-1 text-sm">
                              {sensor.calibrated ? (
                                <CheckCircle className="w-4 h-4 text-gray-600" />
                              ) : (
                                <AlertCircle className="w-4 h-4 text-gray-500" />
                              )}
                              <span className={sensor.calibrated ? 'text-gray-600 dark:text-gray-600' : 'text-gray-500 dark:text-gray-500'}>
                                {sensor.calibrated ? '已校准' : '未校准'}
                              </span>
                            </span>
                            <p className="text-xs text-gray-500 dark:text-gray-400">精度: {(sensor.accuracy * 100).toFixed(1)}%</p>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          {Object.entries(sensor.data).map(([key, value]) => (
                            <div key={key} className="bg-gray-50 dark:bg-gray-700/50 p-3 rounded">
                              <p className="text-xs text-gray-500 dark:text-gray-400 uppercase">{key}</p>
                              <p className="font-medium text-gray-900 dark:text-white">
                                {typeof value === 'object' ? JSON.stringify(value) : value?.toString() || ''}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 传感器类型统计 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">传感器类型统计</h3>
                  <div className="space-y-4">
                    <div className="space-y-3">
                      {['IMU', 'Camera', 'Lidar', 'Depth Camera', 'Force', 'Temperature'].map((type) => {
                        const count = sensorData.filter(s => s.type.toLowerCase().includes(type.toLowerCase())).length;
                        return (
                          <div key={type} className="flex items-center justify-between">
                            <span className="text-gray-600 dark:text-gray-400">{type}</span>
                            <div className="flex items-center gap-3">
                              <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-gray-700"
                                  style={{ width: `${(count / sensorData.length) * 100}%` }}
                                />
                              </div>
                              <span className="font-medium text-gray-900 dark:text-white">{count}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="text-center p-4 bg-gray-700 dark:bg-gray-700/20 rounded-lg">
                          <p className="text-2xl font-bold text-gray-700 dark:text-gray-700">{sensorData.length}</p>
                          <p className="text-sm text-gray-600 dark:text-gray-400">总传感器数</p>
                        </div>
                        <div className="text-center p-4 bg-gray-700 dark:bg-gray-700/20 rounded-lg">
                          <p className="text-2xl font-bold text-gray-600 dark:text-gray-600">
                            {sensorData.filter(s => s.calibrated).length}
                          </p>
                          <p className="text-sm text-gray-600 dark:text-gray-400">已校准传感器</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Gazebo仿真标签页 */}
          {activeTab === 'gazebo' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Gazebo控制 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Gazebo仿真控制</h3>
                  <div className="space-y-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        控制动作
                      </label>
                      <select
                        value={gazeboAction}
                        onChange={(e) => setGazeboAction(e.target.value as any)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      >
                        <option value="start">启动仿真</option>
                        <option value="stop">停止仿真</option>
                        <option value="pause">暂停仿真</option>
                        <option value="reset">重置仿真</option>
                        <option value="load_world">加载世界</option>
                      </select>
                    </div>
                    {gazeboAction === 'load_world' && (
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          世界名称
                        </label>
                        <input
                          type="text"
                          value={worldName}
                          onChange={(e) => setWorldName(e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="empty.world"
                        />
                      </div>
                    )}
                    <button
                      onClick={handleControlGazebo}
                      disabled={isControllingGazebo}
                      className="w-full px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {isControllingGazebo ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          执行中...
                        </>
                      ) : (
                        <>
                          <Cpu className="w-4 h-4" />
                          执行控制
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Gazebo状态 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">仿真状态</h3>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                        <p className="text-sm text-gray-500 dark:text-gray-400">物理引擎</p>
                        <p className="text-lg font-semibold text-gray-900 dark:text-white">
                          {robotStatus?.hardware_status?.physics_engine || '未知'}
                        </p>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                        <p className="text-sm text-gray-500 dark:text-gray-400">仿真时间</p>
                        <p className="text-lg font-semibold text-gray-900 dark:text-white">
                          {robotStatus?.hardware_status?.simulation_time?.toFixed(2) || '0.00'} s
                        </p>
                      </div>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                      <p className="text-sm text-gray-500 dark:text-gray-400">实时因子</p>
                      <div className="mt-2">
                        <div className="w-full h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gray-700"
                            style={{ width: `${(robotStatus?.hardware_status?.real_time_factor || 0) * 100}%` }}
                          />
                        </div>
                        <div className="flex justify-between mt-1">
                          <span className="text-xs text-gray-500 dark:text-gray-400">0.0</span>
                          <span className="text-sm font-medium text-gray-900 dark:text-white">
                            {(robotStatus?.hardware_status?.real_time_factor || 0).toFixed(2)}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400">1.0</span>
                        </div>
                      </div>
                    </div>
                    <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                      <h4 className="font-medium text-gray-900 dark:text-white mb-3">支持的世界</h4>
                      <div className="space-y-2">
                        {['empty.world', 'playground.world', 'warehouse.world', 'office.world'].map((world) => (
                          <button
                            key={world}
                            onClick={() => {
                              setGazeboAction('load_world');
                              setWorldName(world);
                            }}
                            className="w-full px-4 py-2 text-left border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
                          >
                            {world}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* PyBullet仿真标签页 */}
          {activeTab === 'pybullet' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* PyBullet控制 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">PyBullet仿真控制</h3>
                  <div className="space-y-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        控制动作
                      </label>
                      <select
                        value={pybulletAction}
                        onChange={(e) => setPybulletAction(e.target.value as any)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      >
                        <option value="connect">连接仿真</option>
                        <option value="disconnect">断开仿真</option>
                        <option value="step_simulation">单步仿真</option>
                        <option value="reset_simulation">重置仿真</option>
                        <option value="load_urdf">加载URDF</option>
                      </select>
                    </div>
                    {pybulletAction === 'load_urdf' && (
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          URDF文件路径
                        </label>
                        <input
                          type="text"
                          value={urdfPath}
                          onChange={(e) => setUrdfPath(e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="humanoid.urdf"
                        />
                      </div>
                    )}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        物理引擎
                      </label>
                      <select
                        value={physicsEngine}
                        onChange={(e) => setPhysicsEngine(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      >
                        <option value="BULLET">Bullet</option>
                        <option value="DART">DART</option>
                        <option value="TINY">Tiny</option>
                        <option value="NEWTON">Newton</option>
                      </select>
                    </div>
                    <button
                      onClick={handleControlPyBullet}
                      disabled={isControllingPyBullet}
                      className="w-full px-4 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {isControllingPyBullet ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          执行中...
                        </>
                      ) : (
                        <>
                          <Box className="w-4 h-4" />
                          执行控制
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* PyBullet状态 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">仿真状态</h3>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                        <p className="text-sm text-gray-500 dark:text-gray-400">仿真引擎</p>
                        <p className="text-lg font-semibold text-gray-900 dark:text-white">
                          PyBullet
                        </p>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                        <p className="text-sm text-gray-500 dark:text-gray-400">连接状态</p>
                        <p className="text-lg font-semibold text-gray-900 dark:text-white">
                          {robotStatus?.hardware_status?.pybullet_connected ? '已连接' : '未连接'}
                        </p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                        <p className="text-sm text-gray-500 dark:text-gray-400">物理引擎</p>
                        <p className="text-lg font-semibold text-gray-900 dark:text-white">
                          {robotStatus?.hardware_status?.pybullet_physics_engine || physicsEngine}
                        </p>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                        <p className="text-sm text-gray-500 dark:text-gray-400">活跃仿真</p>
                        <p className="text-lg font-semibold text-gray-900 dark:text-white">
                          {robotStatus?.hardware_status?.active_simulation === 'pybullet' ? 'PyBullet' : 'Gazebo'}
                        </p>
                      </div>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                      <p className="text-sm text-gray-500 dark:text-gray-400">仿真能力</p>
                      <div className="mt-2 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-gray-700 dark:text-gray-300">物理仿真</span>
                          <CheckCircle className="w-5 h-5 text-gray-600" />
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-gray-700 dark:text-gray-300">关节控制</span>
                          <CheckCircle className="w-5 h-5 text-gray-600" />
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-gray-700 dark:text-gray-300">传感器系统</span>
                          <CheckCircle className="w-5 h-5 text-gray-600" />
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-gray-700 dark:text-gray-300">碰撞检测</span>
                          <CheckCircle className="w-5 h-5 text-gray-600" />
                        </div>
                      </div>
                    </div>
                    <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                      <h4 className="font-medium text-gray-900 dark:text-white mb-3">支持的URDF模型</h4>
                      <div className="space-y-2">
                        {['humanoid.urdf', 'nao.urdf', 'pepper.urdf', 'custom_robot.urdf'].map((model) => (
                          <button
                            key={model}
                            onClick={() => {
                              setPybulletAction('load_urdf');
                              setUrdfPath(model);
                            }}
                            className="w-full px-4 py-2 text-left border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
                          >
                            {model}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 运动控制标签页 */}
          {activeTab === 'motion-control' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* 路径规划控制 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">路径规划</h3>
                  <div className="space-y-4">
                    {/* 仿真类型选择 */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        仿真环境
                      </label>
                      <div className="flex space-x-2">
                        <button
                          onClick={() => setSimulationType('pybullet')}
                          className={`px-4 py-2 rounded-lg ${simulationType === 'pybullet' ? 'bg-gray-700 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}
                        >
                          PyBullet
                        </button>
                        <button
                          onClick={() => setSimulationType('gazebo')}
                          className={`px-4 py-2 rounded-lg ${simulationType === 'gazebo' ? 'bg-gray-700 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}
                        >
                          Gazebo
                        </button>
                      </div>
                    </div>

                    {/* 起点位置 */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        起点位置 (x, y, z)
                      </label>
                      <div className="flex space-x-2">
                        <input
                          type="number"
                          value={startPosition[0]}
                          onChange={(e) => setStartPosition([parseFloat(e.target.value) || 0, startPosition[1], startPosition[2]])}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="X"
                          step="0.1"
                        />
                        <input
                          type="number"
                          value={startPosition[1]}
                          onChange={(e) => setStartPosition([startPosition[0], parseFloat(e.target.value) || 0, startPosition[2]])}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="Y"
                          step="0.1"
                        />
                        <input
                          type="number"
                          value={startPosition[2]}
                          onChange={(e) => setStartPosition([startPosition[0], startPosition[1], parseFloat(e.target.value) || 0])}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="Z"
                          step="0.1"
                        />
                      </div>
                    </div>

                    {/* 终点位置 */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        终点位置 (x, y, z)
                      </label>
                      <div className="flex space-x-2">
                        <input
                          type="number"
                          value={goalPosition[0]}
                          onChange={(e) => setGoalPosition([parseFloat(e.target.value) || 0, goalPosition[1], goalPosition[2]])}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="X"
                          step="0.1"
                        />
                        <input
                          type="number"
                          value={goalPosition[1]}
                          onChange={(e) => setGoalPosition([goalPosition[0], parseFloat(e.target.value) || 0, goalPosition[2]])}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="Y"
                          step="0.1"
                        />
                        <input
                          type="number"
                          value={goalPosition[2]}
                          onChange={(e) => setGoalPosition([goalPosition[0], goalPosition[1], parseFloat(e.target.value) || 0])}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="Z"
                          step="0.1"
                        />
                      </div>
                    </div>

                    {/* 规划算法选择 */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        路径规划算法
                      </label>
                      <select
                        value={pathPlanningAlgorithm}
                        onChange={(e) => setPathPlanningAlgorithm(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      >
                        <option value="astar">A* 算法</option>
                        <option value="rrt">RRT 算法</option>
                        <option value="rrt_star">RRT* 算法</option>
                        <option value="dijkstra">Dijkstra 算法</option>
                      </select>
                    </div>

                    {/* 规划按钮 */}
                    <button
                      onClick={handlePlanPath}
                      disabled={isPlanningPath}
                      className="w-full px-4 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {isPlanningPath ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          路径规划中...
                        </>
                      ) : (
                        <>
                          <Target className="w-4 h-4" />
                          开始路径规划
                        </>
                      )}
                    </button>

                    {/* 获取仿真信息按钮 */}
                    <button
                      onClick={handleGetSimulationInfo}
                      className="w-full px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
                    >
                      获取仿真环境信息
                    </button>
                  </div>
                </div>

                {/* 路径执行和移动控制 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">路径执行与移动</h3>
                  <div className="space-y-6">
                    {/* 路径信息 */}
                    {pathResult && (
                      <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                        <h4 className="font-medium text-gray-900 dark:text-white mb-2">路径规划结果</h4>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-gray-500 dark:text-gray-400">路径长度:</span>
                            <span className="ml-2 font-medium">{pathResult.path_length?.toFixed(2)} 米</span>
                          </div>
                          <div>
                            <span className="text-gray-500 dark:text-gray-400">计算时间:</span>
                            <span className="ml-2 font-medium">{pathResult.computation_time?.toFixed(3)} 秒</span>
                          </div>
                          <div>
                            <span className="text-gray-500 dark:text-gray-400">探索节点:</span>
                            <span className="ml-2 font-medium">{pathResult.nodes_explored}</span>
                          </div>
                          <div>
                            <span className="text-gray-500 dark:text-gray-400">算法:</span>
                            <span className="ml-2 font-medium">{pathResult.algorithm}</span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* 路径执行按钮 */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        路径执行
                      </label>
                      <div className="flex space-x-2">
                        <button
                          onClick={handleExecutePath}
                          disabled={isExecutingPath || plannedPath.length === 0}
                          className="flex-1 px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                          {isExecutingPath ? (
                            <>
                              <RefreshCw className="w-4 h-4 animate-spin" />
                              执行中...
                            </>
                          ) : (
                            <>
                              <Play className="w-4 h-4" />
                              执行路径
                            </>
                          )}
                        </button>
                        <button
                          onClick={() => setPlannedPath([])}
                          disabled={plannedPath.length === 0}
                          className="px-4 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                          <StopCircle className="w-4 h-4" />
                          清除路径
                        </button>
                      </div>
                      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                        {plannedPath.length > 0 ? `已规划 ${plannedPath.length} 个路径点` : '未规划路径'}
                      </p>
                    </div>

                    {/* 移动到指定位置 */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        移动到指定位置 (x, y, z)
                      </label>
                      <div className="flex space-x-2 mb-3">
                        <input
                          type="number"
                          value={moveTargetPosition[0]}
                          onChange={(e) => setMoveTargetPosition([parseFloat(e.target.value) || 0, moveTargetPosition[1], moveTargetPosition[2]])}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="X"
                          step="0.1"
                        />
                        <input
                          type="number"
                          value={moveTargetPosition[1]}
                          onChange={(e) => setMoveTargetPosition([moveTargetPosition[0], parseFloat(e.target.value) || 0, moveTargetPosition[2]])}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="Y"
                          step="0.1"
                        />
                        <input
                          type="number"
                          value={moveTargetPosition[2]}
                          onChange={(e) => setMoveTargetPosition([moveTargetPosition[0], moveTargetPosition[1], parseFloat(e.target.value) || 0])}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          placeholder="Z"
                          step="0.1"
                        />
                      </div>
                      <button
                        onClick={handleMoveToPosition}
                        disabled={isMovingToPosition}
                        className="w-full px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                      >
                        {isMovingToPosition ? (
                          <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            移动中...
                          </>
                        ) : (
                          <>
                            <Move3d className="w-4 h-4" />
                            移动到目标位置
                          </>
                        )}
                      </button>
                    </div>

                    {/* 路径可视化提示 */}
                    <div className="bg-gray-700 dark:bg-gray-700/20 border border-gray-700 dark:border-gray-700 rounded-lg p-4">
                      <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-1">路径可视化提示</h4>
                      <p className="text-sm text-gray-700 dark:text-gray-700">
                        规划的路径可以在3D可视化标签页中查看。切换到"3D可视化"标签页可以查看路径的3D可视化效果。
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 3D可视化标签页 */}
          {activeTab === 'visualization' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* 3D可视化主视图 */}
                <div className="lg:col-span-2">
                  <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">机器人3D可视化</h3>
                    <div className="w-full h-[600px]">
                      {jointStates.length > 0 ? (
                        <RobotScene
                          pose={convertJointStatesToPose(jointStates)}
                          showControls={true}
                          {...(visualizationPath ? { path: visualizationPath } : {})}
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center bg-gray-100 dark:bg-gray-700 rounded-lg">
                          <div className="text-center">
                            <Bot className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                            <p className="text-gray-600 dark:text-gray-300">正在加载关节数据...</p>
                            <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">请确保机器人已连接并获取关节状态</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* 关节状态面板 */}
                <div>
                  <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 mb-6">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">关节状态</h3>
                    <div className="space-y-4 max-h-[300px] overflow-y-auto">
                      {jointStates.slice(0, 10).map((joint) => (
                        <div key={joint.name} className="flex items-center justify-between py-2 border-b border-gray-100 dark:border-gray-700 last:border-b-0">
                          <div className="flex-1">
                            <p className="text-sm font-medium text-gray-900 dark:text-white">{joint.name}</p>
                            <p className="text-xs text-gray-500 dark:text-gray-400">位置: {joint.position.toFixed(4)} rad</p>
                          </div>
                          <div className="text-right">
                            <div className="text-sm text-gray-700 dark:text-gray-300">
                              {joint.velocity.toFixed(4)} rad/s
                            </div>
                            <div className="text-xs text-gray-500 dark:text-gray-400">
                              {joint.torque.toFixed(4)} N·m
                            </div>
                          </div>
                        </div>
                      ))}
                      {jointStates.length === 0 && (
                        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                          <p>暂无关节数据</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* 可视化控制 */}
                  <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">可视化控制</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          机器人姿态
                        </label>
                        <select
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                          defaultValue="current"
                        >
                          <option value="current">当前姿态</option>
                          <option value="stand">站立姿态</option>
                          <option value="sit">坐姿</option>
                          <option value="walk_ready">行走预备姿态</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          显示选项
                        </label>
                        <div className="space-y-2">
                          <label className="flex items-center">
                            <input type="checkbox" defaultChecked className="rounded border-gray-300 text-gray-700 focus:ring-gray-700" />
                            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">显示网格</span>
                          </label>
                          <label className="flex items-center">
                            <input type="checkbox" defaultChecked className="rounded border-gray-300 text-gray-700 focus:ring-gray-700" />
                            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">显示坐标轴</span>
                          </label>
                          <label className="flex items-center">
                            <input type="checkbox" defaultChecked className="rounded border-gray-300 text-gray-700 focus:ring-gray-700" />
                            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">显示关节标签</span>
                          </label>
                        </div>
                      </div>
                      <button
                        onClick={() => loadJointStates()}
                        className="w-full px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-700 flex items-center justify-center gap-2"
                      >
                        <RefreshCw className="w-4 h-4" />
                        刷新关节数据
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 能力列表标签页 */}
          {activeTab === 'capabilities' && (
            <div className="space-y-6">
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">机器人能力列表</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {[
                    { key: 'movement_control', label: '运动控制', icon: Move3d, description: '基础运动控制能力' },
                    { key: 'joint_control', label: '关节控制', icon: Sliders, description: '精确关节位置控制' },
                    { key: 'sensor_integration', label: '传感器集成', icon: Activity, description: '多传感器数据融合' },
                    { key: 'ros_integration', label: 'ROS集成', icon: Globe, description: 'ROS2接口支持' },
                    { key: 'gazebo_simulation', label: 'Gazebo仿真', icon: Cpu, description: '物理仿真环境' },
                    { key: 'pybullet_simulation', label: 'PyBullet仿真', icon: Box, description: 'Bullet物理仿真环境' },
                    { key: 'motion_planning', label: '运动规划', icon: Target, description: '轨迹规划和避障' },
                    { key: 'collision_detection', label: '碰撞检测', icon: AlertCircle, description: '实时碰撞检测' },
                    { key: 'hardware_abstraction', label: '硬件抽象', icon: Server, description: '统一硬件接口' },
                    { key: 'real_time_control', label: '实时控制', icon: Clock, description: '实时响应能力' },
                    { key: 'trajectory_generation', label: '轨迹生成', icon: BarChart3, description: '平滑轨迹规划' },
                    { key: 'force_control', label: '力控制', icon: Zap, description: '精确力控制' },
                    { key: 'vision_based_control', label: '视觉控制', icon: Eye, description: '视觉反馈控制' },
                    { key: 'autonomous_navigation', label: '自主导航', icon: Brain, description: '自主路径规划' },
                    { key: 'manipulation', label: '操纵能力', icon: Bot, description: '物体抓取和操作' },
                    { key: 'locomotion', label: '移动能力', icon: Activity, description: '动态平衡移动' },
                  ].map((capability) => {
                    const Icon = capability.icon;
                    const supported = robotStatus?.capabilities?.[capability.key as keyof RobotStatus['capabilities']];
                    return (
                      <div
                        key={capability.key}
                        className={`p-4 rounded-lg border ${
                          supported
                            ? 'border-gray-600 dark:border-gray-600 bg-gray-700 dark:bg-gray-700/20'
                            : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/50'
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          <div className={`p-2 rounded-lg ${supported ? 'bg-gray-600 dark:bg-gray-600/30' : 'bg-gray-100 dark:bg-gray-700'}`}>
                            <Icon className={`w-5 h-5 ${supported ? 'text-gray-600 dark:text-gray-600' : 'text-gray-400'}`} />
                          </div>
                          <div>
                            <h4 className="font-medium text-gray-900 dark:text-white">{capability.label}</h4>
                            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{capability.description}</p>
                            <div className="flex items-center gap-2 mt-3">
                              <div className={`w-2 h-2 rounded-full ${supported ? 'bg-gray-700' : 'bg-gray-900'}`} />
                              <span className={`text-xs font-medium ${supported ? 'text-gray-600 dark:text-gray-600' : 'text-gray-800 dark:text-gray-800'}`}>
                                {supported ? '已支持' : '未支持'}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* 示范学习标签页 */}
          {activeTab === 'demonstration' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* 示范录制控制 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">示范录制</h3>
                  
                  <div className="space-y-4">
                    {/* 示范信息 */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        示范名称
                      </label>
                      <input
                        type="text"
                        value={demonstrationName}
                        onChange={(e) => setDemonstrationName(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        placeholder="输入示范名称"
                        disabled={isRecordingDemonstration}
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        示范描述
                      </label>
                      <textarea
                        value={demonstrationDescription}
                        onChange={(e) => setDemonstrationDescription(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        placeholder="输入示范描述"
                        rows={2}
                        disabled={isRecordingDemonstration}
                      />
                    </div>
                    
                    {/* 录制控制按钮 */}
                    <div className="flex space-x-3 pt-2">
                      {!isRecordingDemonstration ? (
                        <button
                          onClick={() => handleStartRecording()}
                          disabled={!demonstrationName || !selectedRobotId}
                          className={`flex-1 px-4 py-2 rounded-lg font-medium ${
                            !demonstrationName || !selectedRobotId
                              ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                              : 'bg-gray-700 hover:bg-gray-700 text-white'
                          }`}
                        >
                          <div className="flex items-center justify-center space-x-2">
                            <Film className="w-4 h-4" />
                            <span>开始录制</span>
                          </div>
                        </button>
                      ) : (
                        <>
                          <button
                            onClick={() => handlePauseRecording()}
                            className="flex-1 px-4 py-2 bg-gray-500 hover:bg-gray-500 text-white rounded-lg font-medium"
                          >
                            暂停录制
                          </button>
                          <button
                            onClick={() => handleStopRecording()}
                            className="flex-1 px-4 py-2 bg-gray-800 hover:bg-gray-800 text-white rounded-lg font-medium"
                          >
                            停止录制
                          </button>
                        </>
                      )}
                    </div>
                    
                    {/* 录制状态显示 */}
                    {isRecordingDemonstration && (
                      <div className="mt-4 p-3 bg-gray-700 dark:bg-gray-700/20 rounded-lg">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-gray-900 rounded-full animate-pulse" />
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              录制中...
                            </span>
                          </div>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            示范ID: {currentDemonstrationId || '--'}
                          </span>
                        </div>
                        <div className="mt-2">
                          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                            <span>帧数: {demonstrationFrames.length}</span>
                            <span>持续时间: {Math.floor(demonstrationFrames.length / 30)}秒</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* 示范回放控制 */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">示范回放</h3>
                  
                  <div className="space-y-4">
                    {/* 回放控制 */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                          播放速度
                        </label>
                        <div className="flex items-center space-x-2">
                          <input
                            type="range"
                            min="0.1"
                            max="3.0"
                            step="0.1"
                            value={playbackSpeed}
                            onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                            className="flex-1"
                          />
                          <span className="text-sm text-gray-700 dark:text-gray-300">{playbackSpeed.toFixed(1)}x</span>
                        </div>
                      </div>
                      
                      <div>
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={loopPlayback}
                            onChange={(e) => setLoopPlayback(e.target.checked)}
                            className="rounded border-gray-300 dark:border-gray-600 text-gray-700 focus:ring-gray-700"
                          />
                          <span className="text-sm text-gray-700 dark:text-gray-300">循环播放</span>
                        </label>
                      </div>
                    </div>
                    
                    {/* 帧控制 */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        当前帧: {currentFrameIndex} / {demonstrationFrames.length}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max={Math.max(0, demonstrationFrames.length - 1)}
                        value={currentFrameIndex}
                        onChange={(e) => setCurrentFrameIndex(parseInt(e.target.value))}
                        className="w-full"
                        disabled={demonstrationFrames.length === 0}
                      />
                    </div>
                    
                    {/* 回放控制按钮 */}
                    <div className="flex space-x-3 pt-2">
                      {!isPlayingDemonstration ? (
                        <button
                          onClick={() => handleStartPlayback()}
                          disabled={demonstrationFrames.length === 0 || !selectedRobotId}
                          className={`flex-1 px-4 py-2 rounded-lg font-medium ${
                            demonstrationFrames.length === 0 || !selectedRobotId
                              ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                              : 'bg-gray-600 hover:bg-gray-600 text-white'
                          }`}
                        >
                          <div className="flex items-center justify-center space-x-2">
                            <Play className="w-4 h-4" />
                            <span>开始播放</span>
                          </div>
                        </button>
                      ) : (
                        <>
                          <button
                            onClick={() => handlePausePlayback()}
                            className="flex-1 px-4 py-2 bg-gray-500 hover:bg-gray-500 text-white rounded-lg font-medium"
                          >
                            暂停播放
                          </button>
                          <button
                            onClick={() => handleStopPlayback()}
                            className="flex-1 px-4 py-2 bg-gray-800 hover:bg-gray-800 text-white rounded-lg font-medium"
                          >
                            停止播放
                          </button>
                        </>
                      )}
                    </div>
                    
                    {/* 示范状态显示 */}
                    {demonstrationStatus && (
                      <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <p className="text-xs text-gray-500 dark:text-gray-400">示范名称</p>
                            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              {demonstrationStatus.name || '--'}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500 dark:text-gray-400">帧数</p>
                            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              {demonstrationStatus.frame_count || 0}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500 dark:text-gray-400">持续时间</p>
                            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              {demonstrationStatus.recording_duration?.toFixed(1) || 0}秒
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500 dark:text-gray-400">状态</p>
                            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              {demonstrationStatus.status || '--'}
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* 示范列表和帧数据 */}
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">示范数据</h3>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      当前示范帧数据 (共{demonstrationFrames.length}帧)
                    </p>
                    <button
                      onClick={() => handleLoadDemonstrations()}
                      className="px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-lg"
                    >
                      刷新示范列表
                    </button>
                  </div>
                  
                  {demonstrationFrames.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {demonstrationFrames.slice(0, 6).map((frame, index) => (
                        <div
                          key={index}
                          className={`p-3 rounded-lg border ${
                            index === currentFrameIndex
                              ? 'border-gray-700 dark:border-gray-700 bg-gray-700 dark:bg-gray-700/20'
                              : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/50'
                          }`}
                        >
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              帧 {index}
                            </span>
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              {frame.timestamp ? new Date(frame.timestamp * 1000).toLocaleTimeString() : '--'}
                            </span>
                          </div>
                          <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span className="text-gray-500 dark:text-gray-400">关节数</span>
                              <span className="text-gray-700 dark:text-gray-300">
                                {frame.joint_positions ? Object.keys(frame.joint_positions).length : 0}
                              </span>
                            </div>
                            <div className="flex justify-between text-xs">
                              <span className="text-gray-500 dark:text-gray-400">传感器数</span>
                              <span className="text-gray-700 dark:text-gray-300">
                                {frame.sensor_data ? Object.keys(frame.sensor_data).length : 0}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <Film className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
                      <p className="text-gray-500 dark:text-gray-400">暂无示范数据</p>
                      <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">
                        开始录制示范以收集机器人运动数据
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 实时数据面板 */}
      {realTimeData && websocketConnected && (
        <div className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 shadow-lg">
          <div className="px-6 py-4">
            <div className="max-w-7xl mx-auto">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium text-gray-900 dark:text-white flex items-center gap-2">
                  <Activity className="w-4 h-4 text-gray-700" />
                  实时数据流
                </h4>
                <button
                  onClick={disconnectWebSocket}
                  className="text-sm text-gray-800 dark:text-gray-800 hover:text-gray-900 dark:hover:text-gray-800"
                >
                  关闭
                </button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-gray-50 dark:bg-gray-700/50 p-3 rounded">
                  <p className="text-xs text-gray-500 dark:text-gray-400">电池电量</p>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {realTimeData.battery_level?.toFixed(1)}%
                  </p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700/50 p-3 rounded">
                  <p className="text-xs text-gray-500 dark:text-gray-400">ROS连接</p>
                  <p className={`font-medium ${realTimeData.ros_connected ? 'text-gray-600 dark:text-gray-600' : 'text-gray-800 dark:text-gray-800'}`}>
                    {realTimeData.ros_connected ? '已连接' : '未连接'}
                  </p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700/50 p-3 rounded">
                  <p className="text-xs text-gray-500 dark:text-gray-400">头部偏航</p>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {realTimeData.joint_states?.[0]?.position?.toFixed(3) || '0.000'} rad
                  </p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700/50 p-3 rounded">
                  <p className="text-xs text-gray-500 dark:text-gray-400">加速度X</p>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {realTimeData.sensor_data?.imu?.acceleration?.x?.toFixed(3) || '0.000'} m/s²
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  // 示范学习处理函数
  const handleStartRecording = async () => {
    if (!selectedRobotId || !demonstrationName) {
      toast.error('请选择机器人并输入示范名称');
      return;
    }

    try {
      setIsRecordingDemonstration(true);
      
      // 调用API开始录制
      const response = await robotApi.startDemonstrationRecording(
        selectedRobotId,
        demonstrationName,
        demonstrationDescription
      );
      
      if (response.success) {
        toast.success('开始录制示范');
        setCurrentDemonstrationId(response.data.demonstration_id);
        setDemonstrationFrames([]);
      } else {
        throw new Error(response.data.message || '开始录制失败');
      }
    } catch (error) {
      console.error('开始录制失败:', error);
      toast.error('开始录制失败');
      setIsRecordingDemonstration(false);
    }
  };

  const handlePauseRecording = async () => {
    if (!selectedRobotId) {
      toast.error('请先选择机器人');
      return;
    }

    try {
      setIsLoading(prev => ({ ...prev, recording: true }));
      const response = await robotApi.pauseDemonstrationRecording(selectedRobotId);
      
      if (response.success) {
        toast.success(`录制已暂停 - 已录制 ${response.data.frames_recorded} 帧`);
        // 更新录制状态 - 已移除setRecordingPaused状态
      } else {
        toast.error(`暂停录制失败: ${response.data?.message || '未知错误'}`);
      }
    } catch (error) {
      console.error('暂停录制失败:', error);
      toast.error('暂停录制失败，API端点可能已实现');
    } finally {
      setIsLoading(prev => ({ ...prev, recording: false }));
    }
  };



  const handleStopRecording = async () => {
    if (!selectedRobotId) {
      toast.error('请先选择机器人');
      return;
    }

    try {
      // 调用API停止录制
      const response = await robotApi.stopDemonstrationRecording(selectedRobotId);
      
      if (response.success) {
        setIsRecordingDemonstration(false);
        toast.success('停止录制');
        
        // 清空临时状态
        setDemonstrationName('');
        setDemonstrationDescription('');
        setCurrentDemonstrationId(null);
      } else {
        throw new Error(response.data.message || '停止录制失败');
      }
    } catch (error) {
      console.error('停止录制失败:', error);
      toast.error('停止录制失败');
    }
  };

  const handleStartPlayback = async () => {
    if (!selectedRobotId) {
      toast.error('请先选择机器人');
      return;
    }
    
    if (!currentDemonstrationId) {
      toast.error('请先录制或加载示范数据');
      return;
    }

    try {
      setIsPlayingDemonstration(true);
      
      // 调用API开始播放
      const response = await robotApi.startDemonstrationPlayback(
        selectedRobotId,
        currentDemonstrationId
      );
      
      if (response.success) {
        toast.success('开始播放示范');
      } else {
        throw new Error(response.data.message || '开始播放失败');
      }
    } catch (error) {
      console.error('开始播放失败:', error);
      toast.error('开始播放失败');
      setIsPlayingDemonstration(false);
    }
  };

  const handlePausePlayback = async () => {
    if (!selectedRobotId) {
      toast.error('请先选择机器人');
      return;
    }

    try {
      setIsLoading(prev => ({ ...prev, playback: true }));
      const response = await robotApi.pauseDemonstrationPlayback(selectedRobotId);
      
      if (response.success) {
        toast.success('播放已暂停');
        // 更新播放状态 - 已移除setPlaybackPaused状态
      } else {
        toast.error(`暂停播放失败: ${response.data?.message || '未知错误'}`);
      }
    } catch (error) {
      console.error('暂停播放失败:', error);
      toast.error('暂停播放失败，API端点可能已实现');
    } finally {
      setIsLoading(prev => ({ ...prev, playback: false }));
    }
  };



  const handleStopPlayback = async () => {
    if (!selectedRobotId) {
      toast.error('请先选择机器人');
      return;
    }

    try {
      // 调用API停止播放
      const response = await robotApi.stopDemonstrationPlayback(selectedRobotId);
      
      if (response.success) {
        setIsPlayingDemonstration(false);
        toast.success('停止播放');
      } else {
        throw new Error(response.data.message || '停止播放失败');
      }
    } catch (error) {
      console.error('停止播放失败:', error);
      toast.error('停止播放失败');
    }
  };

  const handleLoadDemonstrations = async () => {
    if (!selectedRobotId) {
      toast.error('请先选择机器人');
      return;
    }

    try {
      // 实际项目中应调用API加载示范列表，例如：robotApi.getDemonstrations(selectedRobotId)
      // 目前没有实现相关API，设置为空数据
      setDemonstrationFrames([]);
      setDemonstrationStatus({
        name: '',
        frame_count: 0,
        recording_duration: 0,
        status: 'idle'
      });
      toast.success('示范列表已加载（暂无数据）');
    } catch (error) {
      console.error('加载示范列表失败:', error);
      toast.error('加载示范列表失败');
      setDemonstrationFrames([]);
      setDemonstrationStatus({
        name: '',
        frame_count: 0,
        recording_duration: 0,
        status: 'error'
      });
    }
  };
};

export default RobotManagementPage;