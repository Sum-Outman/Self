import { JointState } from '../../services/api/robot';
import { RobotPoseData, JointData } from './RobotModel';

/**
 * 将关节状态转换为3D可视化所需的数据格式
 */
export const convertJointStatesToPose = (
  jointStates: JointState[],
  basePosition: [number, number, number] = [0, 0, 0]
): RobotPoseData => {
  const joints: Record<string, JointData> = {};

  // 处理关节数据
  jointStates.forEach((joint) => {
    // 标准化关节名称映射
    const normalizedName = normalizeJointName(joint.name);
    
    joints[normalizedName] = {
      name: joint.name,
      position: joint.position,  // 弧度
      velocity: joint.velocity,
      torque: joint.torque,
    };
  });

  // 确保所有必需关节都有默认值
  const requiredJoints = [
    'l_shoulder_pitch', 'r_shoulder_pitch',
    'l_elbow_pitch', 'r_elbow_pitch',
    'l_hip_pitch', 'r_hip_pitch',
    'l_knee_pitch', 'r_knee_pitch',
    'l_ankle_pitch', 'r_ankle_pitch',
    'head_yaw', 'head_pitch',
  ];

  requiredJoints.forEach((jointName) => {
    if (!joints[jointName]) {
      joints[jointName] = {
        name: jointName,
        position: 0,
        velocity: 0,
        torque: 0,
      };
    }
  });

  return {
    joints,
    basePosition,
    baseOrientation: [0, 0, 0, 1],  // 单位四元数
  };
};

/**
 * 标准化关节名称
 * 处理不同命名约定的关节名称
 */
const normalizeJointName = (jointName: string): string => {
  // 转换为小写，替换空格和下划线
  const normalized = jointName.toLowerCase().replace(/\s+/g, '_');
  
  // 常见的关节名称映射
  const jointNameMap: Record<string, string> = {
    // 左臂
    'left_shoulder_pitch': 'l_shoulder_pitch',
    'left_shoulder_roll': 'l_shoulder_roll',
    'left_elbow_pitch': 'l_elbow_pitch',
    'left_elbow_yaw': 'l_elbow_pitch',        // 后端使用yaw，映射到pitch
    'left_elbow_roll': 'l_elbow_pitch',       // 后端使用roll，映射到pitch
    'left_wrist_pitch': 'l_wrist_pitch',
    'left_wrist_yaw': 'l_wrist_pitch',        // 后端使用yaw，映射到pitch
    'left_hand': 'l_wrist_pitch',             // 手部关节映射到手腕
    
    // 右臂
    'right_shoulder_pitch': 'r_shoulder_pitch',
    'right_shoulder_roll': 'r_shoulder_roll',
    'right_elbow_pitch': 'r_elbow_pitch',
    'right_elbow_yaw': 'r_elbow_pitch',       // 后端使用yaw，映射到pitch
    'right_elbow_roll': 'r_elbow_pitch',      // 后端使用roll，映射到pitch
    'right_wrist_pitch': 'r_wrist_pitch',
    'right_wrist_yaw': 'r_wrist_pitch',       // 后端使用yaw，映射到pitch
    'right_hand': 'r_wrist_pitch',            // 手部关节映射到手腕
    
    // 左腿
    'left_hip_pitch': 'l_hip_pitch',
    'left_hip_roll': 'l_hip_roll',
    'left_hip_yaw': 'l_hip_pitch',            // 后端使用yaw，映射到pitch
    'left_hip_yaw_pitch': 'l_hip_pitch',      // 复合关节映射到pitch
    'left_knee_pitch': 'l_knee_pitch',
    'left_ankle_pitch': 'l_ankle_pitch',
    'left_ankle_roll': 'l_ankle_roll',
    
    // 右腿
    'right_hip_pitch': 'r_hip_pitch',
    'right_hip_roll': 'r_hip_roll',
    'right_hip_yaw': 'r_hip_pitch',           // 后端使用yaw，映射到pitch
    'right_hip_yaw_pitch': 'r_hip_pitch',     // 复合关节映射到pitch
    'right_knee_pitch': 'r_knee_pitch',
    'right_ankle_pitch': 'r_ankle_pitch',
    'right_ankle_roll': 'r_ankle_roll',
    
    // 头部
    'head_yaw': 'head_yaw',
    'head_pitch': 'head_pitch',
    'head_roll': 'head_roll',
  };

  return jointNameMap[normalized] || normalized;
};

/**
 * 获取关节角度限制
 */
export const getJointLimits = (jointName: string): { min: number; max: number } => {
  const limits: Record<string, { min: number; max: number }> = {
    // 肩膀关节（弧度）
    'l_shoulder_pitch': { min: -2.0, max: 2.0 },
    'r_shoulder_pitch': { min: -2.0, max: 2.0 },
    
    // 肘关节
    'l_elbow_pitch': { min: -2.5, max: 2.5 },
    'r_elbow_pitch': { min: -2.5, max: 2.5 },
    
    // 髋关节
    'l_hip_pitch': { min: -1.5, max: 1.5 },
    'r_hip_pitch': { min: -1.5, max: 1.5 },
    
    // 膝关节
    'l_knee_pitch': { min: 0, max: 2.5 },
    'r_knee_pitch': { min: 0, max: 2.5 },
    
    // 踝关节
    'l_ankle_pitch': { min: -1.0, max: 1.0 },
    'r_ankle_pitch': { min: -1.0, max: 1.0 },
    
    // 头部关节
    'head_yaw': { min: -1.5, max: 1.5 },
    'head_pitch': { min: -0.8, max: 0.8 },
  };

  return limits[jointName] || { min: -Math.PI, max: Math.PI };
};

/**
 * 检查关节角度是否在安全范围内
 */
export const isJointAngleSafe = (jointName: string, angle: number): boolean => {
  const limits = getJointLimits(jointName);
  return angle >= limits.min && angle <= limits.max;
};

/**
 * 生成预定义姿态的关节角度
 */
export const getPredefinedPose = (poseType: string): Record<string, number> => {
  const poses: Record<string, Record<string, number>> = {
    stand: {
      l_shoulder_pitch: 0.0,
      r_shoulder_pitch: 0.0,
      l_elbow_pitch: 0.0,
      r_elbow_pitch: 0.0,
      l_hip_pitch: 0.0,
      r_hip_pitch: 0.0,
      l_knee_pitch: 0.0,
      r_knee_pitch: 0.0,
      l_ankle_pitch: 0.0,
      r_ankle_pitch: 0.0,
      head_yaw: 0.0,
      head_pitch: 0.0,
    },
    sit: {
      l_shoulder_pitch: 0.0,
      r_shoulder_pitch: 0.0,
      l_elbow_pitch: 0.0,
      r_elbow_pitch: 0.0,
      l_hip_pitch: -1.57,
      r_hip_pitch: -1.57,
      l_knee_pitch: 1.57,
      r_knee_pitch: 1.57,
      l_ankle_pitch: -0.78,
      r_ankle_pitch: -0.78,
      head_yaw: 0.0,
      head_pitch: 0.0,
    },
    walk_ready: {
      l_shoulder_pitch: 0.0,
      r_shoulder_pitch: 0.0,
      l_elbow_pitch: 0.0,
      r_elbow_pitch: 0.0,
      l_hip_pitch: 0.2,
      r_hip_pitch: 0.2,
      l_knee_pitch: -0.4,
      r_knee_pitch: -0.4,
      l_ankle_pitch: 0.2,
      r_ankle_pitch: 0.2,
      head_yaw: 0.0,
      head_pitch: 0.0,
    },
  };

  return poses[poseType] || poses.stand;
};