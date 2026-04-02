import React, { useRef, useMemo, memo, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Box, Sphere, Cylinder, Line, OrbitControls as DreiOrbitControls, usePerformanceMonitor } from '@react-three/drei';
import * as THREE from 'three';

// 关节数据类型
export interface JointData {
  name: string;
  position: number;  // 关节位置（弧度）
  velocity: number;
  torque: number;
}

// 机器人姿态数据
export interface RobotPoseData {
  joints: Record<string, JointData>;
  basePosition: [number, number, number];  // [x, y, z]
  baseOrientation: [number, number, number, number];  // [x, y, z, w] 四元数
}

// 单个关节组件 - 增强交互功能
const Joint: React.FC<{
  name: string;
  position: [number, number, number];
  rotation: [number, number, number];
  angle: number;  // 关节角度（弧度）
  type: 'revolute' | 'prismatic' | 'fixed';
  scale?: number;
  color?: string;
  isSelected?: boolean;
  isHovered?: boolean;
  onHover?: (jointName: string, isHovered: boolean) => void;
  onClick?: (jointName: string) => void;
  onDragStart?: (jointName: string) => void;
  onDragEnd?: (jointName: string) => void;
  onDrag?: (jointName: string, deltaAngle: number) => void;
}> = ({ 
  name, 
  position, 
  rotation, 
  angle, 
  type, 
  scale = 1, 
  color = '#3b82f6',
  isSelected = false,
  isHovered = false,
  onHover,
  onClick,
  onDragStart,
  onDragEnd,
  onDrag
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartPosition, setDragStartPosition] = useState<{x: number, y: number} | null>(null);

  useFrame(() => {
    if (meshRef.current) {
      // 根据关节类型更新旋转
      if (type === 'revolute') {
        meshRef.current.rotation.x = angle;
      }
      
      // 根据交互状态调整颜色
      const material = meshRef.current.material as THREE.MeshStandardMaterial;
      if (isSelected) {
        material.color.set('#ffffff'); // 白色表示选中
        material.emissive.set('#444444');
      } else if (isHovered || isDragging) {
        material.color.set('#ffcc00'); // 黄色表示悬停或拖拽
        material.emissive.set('#222222');
      } else {
        material.color.set(color);
        material.emissive.set('#000000');
      }
    }
  });

  // 处理鼠标事件
  const handlePointerEnter = () => {
    if (onHover) onHover(name, true);
  };

  const handlePointerLeave = () => {
    if (onHover) onHover(name, false);
  };

  const handlePointerDown = (_event: THREE.Event) => {
    // Three.js Event没有stopPropagation或nativeEvent属性
    // event.stopPropagation();
    setIsDragging(true);
    // 使用全局鼠标位置作为拖拽起始位置
    setDragStartPosition({
      x: window.innerWidth / 2, // 近似值
      y: window.innerHeight / 2
    });
    if (onDragStart) onDragStart(name);
    if (onClick) onClick(name);
  };

  const handlePointerUp = () => {
    if (isDragging) {
      setIsDragging(false);
      setDragStartPosition(null);
      if (onDragEnd) onDragEnd(name);
    }
  };

  // 监听全局鼠标事件来处理拖拽
  useEffect(() => {
    const handleGlobalMouseMove = (event: MouseEvent) => {
      if (isDragging && dragStartPosition && groupRef.current) {
        // 计算拖拽距离（可以用于调整关节角度）
        const deltaX = event.clientX - dragStartPosition.x;
        // const deltaY = event.clientY - dragStartPosition.y; // 保留以备未来使用（俯仰控制）
        
        // 根据拖拽距离调整关节角度
        // 使用deltaX控制旋转关节的角度变化，deltaY可用于俯仰控制（如果需要）
        const deltaAngle = deltaX * 0.01; // 缩放因子
        
        if (onDrag && Math.abs(deltaAngle) > 0.001) {
          onDrag(name, deltaAngle);
        }
        
        // 更新拖拽起始位置，用于下一次增量计算
        setDragStartPosition({
          x: event.clientX,
          y: event.clientY
        });
      }
    };

    const handleGlobalMouseUp = () => {
    if (isDragging) {
      setIsDragging(false);
      setDragStartPosition(null);
      if (onDragEnd) onDragEnd(name);
    }
  };

    if (isDragging) {
      window.addEventListener('mousemove', handleGlobalMouseMove);
      window.addEventListener('mouseup', handleGlobalMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [isDragging, dragStartPosition, name, onDragEnd]);

  return (
    <group 
      ref={groupRef}
      position={position} 
      rotation={rotation}
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
    >
      {/* 关节主体 */}
      <Sphere args={[0.1 * scale]} position={[0, 0, 0]}>
        <meshStandardMaterial 
          color={isSelected ? '#ffffff' : (isHovered || isDragging) ? '#ffcc00' : color} 
          roughness={0.4} 
          metalness={0.6}
          emissive={isSelected ? '#444444' : (isHovered || isDragging) ? '#222222' : '#000000'}
          emissiveIntensity={0.5}
        />
      </Sphere>
      
      {/* 关节轴指示器 */}
      <Cylinder args={[0.02 * scale, 0.02 * scale, 0.3 * scale, 8]} position={[0, 0.15 * scale, 0]}>
        <meshStandardMaterial color="#ef4444" />
      </Cylinder>
      
      {/* 关节名称标签 - 在选中或悬停时显示 */}
      {(isSelected || isHovered || isDragging) && (
        <mesh position={[0, 0.3 * scale, 0]}>
          {/* 这里可以使用Text组件显示关节名称，简化处理 */}
          <sphereGeometry args={[0.05, 8, 8]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
      )}
    </group>
  );
};

// 连杆组件
const Link: React.FC<{
  start: [number, number, number];
  end: [number, number, number];
  thickness?: number;
  color?: string;
}> = ({ start, end, thickness = 0.05, color = '#6b7280' }) => {
  const points = [new THREE.Vector3(...start), new THREE.Vector3(...end)];
  
  return (
    <Line
      points={points}
      color={color}
      lineWidth={thickness * 10}
      dashed={false}
    />
  );
};

// 简化机器人组件（低细节级别）
const SimplifiedHumanoidRobot: React.FC<{
  pose: RobotPoseData;
}> = memo(({ pose }) => {
  const { basePosition } = pose;

  // 使用几何体和材质的缓存
  const sphereGeometry = useMemo(() => new THREE.SphereGeometry(0.08, 8, 8), []);
  const boxGeometry = useMemo(() => new THREE.BoxGeometry(0.35, 0.55, 0.15), []);
  const cylinderGeometry = useMemo(() => new THREE.CylinderGeometry(0.015, 0.015, 0.2, 6), []);
  
  const blueMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#3b82f6', roughness: 0.5 }), []);
  const redMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#ef4444', roughness: 0.5 }), []);
  const greenMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#10b981', roughness: 0.5 }), []);
  const purpleMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#8b5cf6', roughness: 0.5 }), []);
  const grayMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#4b5563', roughness: 0.6 }), []);
  const yellowMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#fbbf24', roughness: 0.4 }), []);

  return (
    <group position={basePosition}>
      {/* 躯干 */}
      <mesh geometry={boxGeometry} material={grayMaterial} position={[0, 0.75, 0]} />
      
      {/* 头部 */}
      <mesh geometry={sphereGeometry} material={yellowMaterial} position={[0, 1.2, 0]} />
      
      {/* 左臂 */}
      <mesh geometry={cylinderGeometry} material={blueMaterial} position={[-0.45, 0.9, 0]} rotation={[0, 0, Math.PI / 2]} />
      <mesh geometry={sphereGeometry} material={blueMaterial} position={[-0.3, 1.0, 0]} />
      <mesh geometry={cylinderGeometry} material={blueMaterial} position={[-0.75, 0.7, 0]} rotation={[0, 0, Math.PI / 2]} />
      <mesh geometry={sphereGeometry} material={blueMaterial} position={[-0.6, 0.8, 0]} />
      
      {/* 右臂 */}
      <mesh geometry={cylinderGeometry} material={redMaterial} position={[0.45, 0.9, 0]} rotation={[0, 0, Math.PI / 2]} />
      <mesh geometry={sphereGeometry} material={redMaterial} position={[0.3, 1.0, 0]} />
      <mesh geometry={cylinderGeometry} material={redMaterial} position={[0.75, 0.7, 0]} rotation={[0, 0, Math.PI / 2]} />
      <mesh geometry={sphereGeometry} material={redMaterial} position={[0.6, 0.8, 0]} />
      
      {/* 左腿 */}
      <mesh geometry={cylinderGeometry} material={greenMaterial} position={[-0.2, -0.25, 0]} rotation={[0, 0, 0]} />
      <mesh geometry={sphereGeometry} material={greenMaterial} position={[-0.2, 0, 0]} />
      <mesh geometry={cylinderGeometry} material={greenMaterial} position={[-0.2, -0.75, 0]} rotation={[0, 0, 0]} />
      <mesh geometry={sphereGeometry} material={greenMaterial} position={[-0.2, -0.5, 0]} />
      
      {/* 右腿 */}
      <mesh geometry={cylinderGeometry} material={purpleMaterial} position={[0.2, -0.25, 0]} rotation={[0, 0, 0]} />
      <mesh geometry={sphereGeometry} material={purpleMaterial} position={[0.2, 0, 0]} />
      <mesh geometry={cylinderGeometry} material={purpleMaterial} position={[0.2, -0.75, 0]} rotation={[0, 0, 0]} />
      <mesh geometry={sphereGeometry} material={purpleMaterial} position={[0.2, -0.5, 0]} />
    </group>
  );
});

// 性能优化：根据距离自动切换LOD
const AdaptiveHumanoidRobot: React.FC<{
  pose: RobotPoseData;
  interactionState?: {
    selectedJoint: string | null;
    hoveredJoint: string | null;
    jointAngles: Record<string, number>;
  };
  onJointHover?: (jointName: string, isHovered: boolean) => void;
  onJointClick?: (jointName: string) => void;
  onJointDragStart?: (jointName: string) => void;
  onJointDragEnd?: (jointName: string) => void;
  onJointDrag?: (jointName: string, deltaAngle: number) => void;
}> = memo(({ 
  pose,
  interactionState,
  onJointHover,
  onJointClick,
  onJointDragStart,
  onJointDragEnd,
  onJointDrag
}) => {
  const { camera } = useThree();
  const [lodLevel, setLodLevel] = useState<'high' | 'low'>('high');
  
  const robotPosition = useMemo(() => new THREE.Vector3(...pose.basePosition), [pose.basePosition]);

  useFrame(() => {
    const camPosition = camera.position.clone();
    const dist = camPosition.distanceTo(robotPosition);
    
    // 根据距离切换LOD级别
    if (dist > 8) {
      setLodLevel('low');
    } else {
      setLodLevel('high');
    }
  });

  // 使用性能监控来自动调整
  usePerformanceMonitor({
    onIncline: () => {},
    onDecline: () => {},
    onChange: ({ factor }) => {
      // 如果性能下降，切换到低细节
      if (factor > 2 && lodLevel !== 'low') {
        setLodLevel('low');
      } else if (factor < 1 && lodLevel !== 'high') {
        setLodLevel('high');
      }
    }
  });

  return (
    <>
      {lodLevel === 'high' ? (
        <HumanoidRobot 
          pose={pose} 
          interactionState={interactionState}
          onJointHover={onJointHover}
          onJointClick={onJointClick}
          onJointDragStart={onJointDragStart}
          onJointDragEnd={onJointDragEnd}
          onJointDrag={onJointDrag}
        />
      ) : (
        <SimplifiedHumanoidRobot pose={pose} />
      )}
    </>
  );
});

// 机器人主要组件
const HumanoidRobot: React.FC<{
  pose: RobotPoseData;
  interactionState?: {
    selectedJoint: string | null;
    hoveredJoint: string | null;
    jointAngles: Record<string, number>;
  };
  onJointHover?: (jointName: string, isHovered: boolean) => void;
  onJointClick?: (jointName: string) => void;
  onJointDragStart?: (jointName: string) => void;
  onJointDragEnd?: (jointName: string) => void;
  onJointDrag?: (jointName: string, deltaAngle: number) => void;
}> = memo(({ 
  pose,
  interactionState,
  onJointHover,
  onJointClick,
  onJointDragStart,
  onJointDragEnd,
  onJointDrag
}) => {
  const { joints, basePosition } = pose;

  // 计算关节角度：优先使用交互状态中的覆盖角度
  const getJointAngle = (jointName: string): number => {
    if (interactionState?.jointAngles && jointName in interactionState.jointAngles) {
      return interactionState.jointAngles[jointName];
    }
    return joints[jointName]?.position || 0;
  };

  // 检查关节是否被选中或悬停
  const isJointSelected = (jointName: string): boolean => {
    return interactionState?.selectedJoint === jointName;
  };

  const isJointHovered = (jointName: string): boolean => {
    return interactionState?.hoveredJoint === jointName;
  };

  // 简化的机器人骨架定义 - 使用useMemo记忆化
  const skeleton = useMemo(() => ({
    // 躯干
    torso: { start: [0, 0.5, 0] as [number, number, number], end: [0, 1.0, 0] as [number, number, number] },
    
    // 头部
    head: { position: [0, 1.2, 0] as [number, number, number] },
    
    // 左臂
    leftShoulder: { position: [-0.3, 1.0, 0] as [number, number, number] },
    leftElbow: { position: [-0.6, 0.8, 0] as [number, number, number] },
    leftWrist: { position: [-0.9, 0.6, 0] as [number, number, number] },
    
    // 右臂
    rightShoulder: { position: [0.3, 1.0, 0] as [number, number, number] },
    rightElbow: { position: [0.6, 0.8, 0] as [number, number, number] },
    rightWrist: { position: [0.9, 0.6, 0] as [number, number, number] },
    
    // 左腿
    leftHip: { position: [-0.2, 0, 0] as [number, number, number] },
    leftKnee: { position: [-0.2, -0.5, 0] as [number, number, number] },
    leftAnkle: { position: [-0.2, -1.0, 0] as [number, number, number] },
    
    // 右腿
    rightHip: { position: [0.2, 0, 0] as [number, number, number] },
    rightKnee: { position: [0.2, -0.5, 0] as [number, number, number] },
    rightAnkle: { position: [0.2, -1.0, 0] as [number, number, number] },
  }), []);

  return (
    <group position={basePosition}>
      {/* 躯干 */}
      <Box args={[0.4, 0.6, 0.2]} position={[0, 0.75, 0]}>
        <meshStandardMaterial color="#4b5563" roughness={0.5} />
      </Box>
      
      {/* 头部 */}
      <Sphere args={[0.15]} position={skeleton.head.position}>
        <meshStandardMaterial color="#fbbf24" roughness={0.3} />
      </Sphere>
      
      {/* 左臂 */}
      <Link start={[0, 0.9, 0]} end={skeleton.leftShoulder.position} />
      <Joint
        name="left_shoulder"
        position={skeleton.leftShoulder.position}
        rotation={[0, 0, 0]}
        angle={getJointAngle('l_shoulder_pitch')}
        type="revolute"
        color="#3b82f6"
        isSelected={isJointSelected('left_shoulder')}
        isHovered={isJointHovered('left_shoulder')}
        onHover={onJointHover}
        onClick={onJointClick}
        onDragStart={onJointDragStart}
        onDragEnd={onJointDragEnd}
        onDrag={onJointDrag}
      />
      <Link start={skeleton.leftShoulder.position} end={skeleton.leftElbow.position} />
      <Joint
        name="left_elbow"
        position={skeleton.leftElbow.position}
        rotation={[0, 0, 0]}
        angle={getJointAngle('l_elbow_pitch')}
        type="revolute"
        color="#3b82f6"
        isSelected={isJointSelected('left_elbow')}
        isHovered={isJointHovered('left_elbow')}
        onHover={onJointHover}
        onClick={onJointClick}
        onDragStart={onJointDragStart}
        onDragEnd={onJointDragEnd}
      />
      <Link start={skeleton.leftElbow.position} end={skeleton.leftWrist.position} />
      
      {/* 右臂 */}
      <Link start={[0, 0.9, 0]} end={skeleton.rightShoulder.position} />
      <Joint
        name="right_shoulder"
        position={skeleton.rightShoulder.position}
        rotation={[0, 0, 0]}
        angle={getJointAngle('r_shoulder_pitch')}
        type="revolute"
        color="#ef4444"
        isSelected={isJointSelected('right_shoulder')}
        isHovered={isJointHovered('right_shoulder')}
        onHover={onJointHover}
        onClick={onJointClick}
        onDragStart={onJointDragStart}
        onDragEnd={onJointDragEnd}
      />
      <Link start={skeleton.rightShoulder.position} end={skeleton.rightElbow.position} />
      <Joint
        name="right_elbow"
        position={skeleton.rightElbow.position}
        rotation={[0, 0, 0]}
        angle={getJointAngle('r_elbow_pitch')}
        type="revolute"
        color="#ef4444"
        isSelected={isJointSelected('right_elbow')}
        isHovered={isJointHovered('right_elbow')}
        onHover={onJointHover}
        onClick={onJointClick}
        onDragStart={onJointDragStart}
        onDragEnd={onJointDragEnd}
      />
      <Link start={skeleton.rightElbow.position} end={skeleton.rightWrist.position} />
      
      {/* 左腿 */}
      <Link start={[0, 0.3, 0]} end={skeleton.leftHip.position} />
      <Joint
        name="left_hip"
        position={skeleton.leftHip.position}
        rotation={[0, 0, 0]}
        angle={getJointAngle('l_hip_pitch')}
        type="revolute"
        color="#10b981"
        isSelected={isJointSelected('left_hip')}
        isHovered={isJointHovered('left_hip')}
        onHover={onJointHover}
        onClick={onJointClick}
        onDragStart={onJointDragStart}
        onDragEnd={onJointDragEnd}
      />
      <Link start={skeleton.leftHip.position} end={skeleton.leftKnee.position} />
      <Joint
        name="left_knee"
        position={skeleton.leftKnee.position}
        rotation={[0, 0, 0]}
        angle={getJointAngle('l_knee_pitch')}
        type="revolute"
        color="#10b981"
        isSelected={isJointSelected('left_knee')}
        isHovered={isJointHovered('left_knee')}
        onHover={onJointHover}
        onClick={onJointClick}
        onDragStart={onJointDragStart}
        onDragEnd={onJointDragEnd}
      />
      <Link start={skeleton.leftKnee.position} end={skeleton.leftAnkle.position} />
      
      {/* 右腿 */}
      <Link start={[0, 0.3, 0]} end={skeleton.rightHip.position} />
      <Joint
        name="right_hip"
        position={skeleton.rightHip.position}
        rotation={[0, 0, 0]}
        angle={getJointAngle('r_hip_pitch')}
        type="revolute"
        color="#8b5cf6"
        isSelected={isJointSelected('right_hip')}
        isHovered={isJointHovered('right_hip')}
        onHover={onJointHover}
        onClick={onJointClick}
        onDragStart={onJointDragStart}
        onDragEnd={onJointDragEnd}
      />
      <Link start={skeleton.rightHip.position} end={skeleton.rightKnee.position} />
      <Joint
        name="right_knee"
        position={skeleton.rightKnee.position}
        rotation={[0, 0, 0]}
        angle={getJointAngle('r_knee_pitch')}
        type="revolute"
        color="#8b5cf6"
        isSelected={isJointSelected('right_knee')}
        isHovered={isJointHovered('right_knee')}
        onHover={onJointHover}
        onClick={onJointClick}
        onDragStart={onJointDragStart}
        onDragEnd={onJointDragEnd}
      />
      <Link start={skeleton.rightKnee.position} end={skeleton.rightAnkle.position} />
      
      {/* 地面参考网格 */}
      <gridHelper args={[5, 20, '#6b7280', '#9ca3af']} position={[0, -1.2, 0]} />
    </group>
  );
});

// 3D场景组件
// 路径数据接口
export interface PathData {
  points: Array<[number, number, number]>;  // 路径点 [x, y, z]
  color?: string;
  lineWidth?: number;
}

// 交互状态接口
export interface RobotInteractionState {
  selectedJoint: string | null;
  hoveredJoint: string | null;
  isDragging: boolean;
  jointAngles: Record<string, number>; // 关节角度覆盖
  cameraPreset: 'perspective' | 'top' | 'front' | 'side';
  animationPlaying: boolean;
  animationSpeed: number;
}

export const RobotScene: React.FC<{
  pose: RobotPoseData;
  showControls?: boolean;
  path?: PathData | PathData[];  // 单条或多条路径
  enableInteraction?: boolean; // 是否启用交互
  onJointSelect?: (jointName: string | null) => void;
  onJointAngleChange?: (jointName: string, angle: number) => void;
  initialInteractionState?: Partial<RobotInteractionState>;
}> = memo(({ 
  pose, 
  showControls = true, 
  path, 
  enableInteraction = true,
  onJointSelect,
  onJointAngleChange,
  initialInteractionState
}) => {
  // 交互状态管理
  const [interactionState, setInteractionState] = useState<RobotInteractionState>({
    selectedJoint: null,
    hoveredJoint: null,
    isDragging: false,
    jointAngles: {},
    cameraPreset: 'perspective',
    animationPlaying: false,
    animationSpeed: 1.0,
    ...initialInteractionState
  });

  // 处理关节悬停
  const handleJointHover = (jointName: string, isHovered: boolean) => {
    if (!enableInteraction) return;
    
    setInteractionState(prev => ({
      ...prev,
      hoveredJoint: isHovered ? jointName : null
    }));
  };

  // 处理关节点击
  const handleJointClick = (jointName: string) => {
    if (!enableInteraction) return;
    
    const newSelectedJoint = interactionState.selectedJoint === jointName ? null : jointName;
    
    setInteractionState(prev => ({
      ...prev,
      selectedJoint: newSelectedJoint
    }));
    
    if (onJointSelect) {
      onJointSelect(newSelectedJoint);
    }
  };

  // 处理关节拖拽开始
  const handleJointDragStart = (jointName: string) => {
    if (!enableInteraction) return;
    
    setInteractionState(prev => ({
      ...prev,
      isDragging: true,
      selectedJoint: jointName
    }));
  };

  // 处理关节拖拽结束
  const handleJointDragEnd = (jointName: string) => {
    if (!enableInteraction) return;
    
    setInteractionState(prev => ({
      ...prev,
      isDragging: false
    }));
    
    console.log(`关节 ${jointName} 拖拽结束`);
  };

  // 处理关节拖拽角度变化
  const handleJointDrag = (jointName: string, deltaAngle: number) => {
    if (!enableInteraction) return;
    updateJointAngle(jointName, deltaAngle);
  };

  // 更新关节角度
  const updateJointAngle = (jointName: string, deltaAngle: number) => {
    if (!enableInteraction) return;
    
    const currentAngle = interactionState.jointAngles[jointName] || pose.joints[jointName]?.position || 0;
    const newAngle = currentAngle + deltaAngle;
    
    setInteractionState(prev => ({
      ...prev,
      jointAngles: {
        ...prev.jointAngles,
        [jointName]: newAngle
      }
    }));
    
    if (onJointAngleChange) {
      onJointAngleChange(jointName, newAngle);
    }
  };

  // 处理键盘控制
  useEffect(() => {
    if (!enableInteraction) return;
    
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!interactionState.selectedJoint) return;
      
      const jointName = interactionState.selectedJoint;
      let deltaAngle = 0;
      
      switch (event.key) {
        case 'ArrowUp':
          deltaAngle = 0.1; // 增加角度
          break;
        case 'ArrowDown':
          deltaAngle = -0.1; // 减少角度
          break;
        case 'ArrowLeft':
          deltaAngle = -0.05;
          break;
        case 'ArrowRight':
          deltaAngle = 0.05;
          break;
        case 'Home':
          // 重置关节角度
          setInteractionState(prev => ({
            ...prev,
            jointAngles: {
              ...prev.jointAngles,
              [jointName]: 0
            }
          }));
          if (onJointAngleChange) {
            onJointAngleChange(jointName, 0);
          }
          return;
        default:
          return;
      }
      
      if (deltaAngle !== 0) {
        updateJointAngle(jointName, deltaAngle);
        event.preventDefault(); // 防止页面滚动
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [enableInteraction, interactionState.selectedJoint, pose.joints, onJointAngleChange]);

  // 动画循环
  useEffect(() => {
    if (!enableInteraction || !interactionState.animationPlaying) return;
    
    let animationFrameId: number;
    let lastTime = 0;
    const animationSpeed = interactionState.animationSpeed;
    
    const animate = (time: number) => {
      if (lastTime === 0) lastTime = time;
      // const deltaTime = (time - lastTime) / 1000; // 转换为秒（未使用）
      lastTime = time;
      
      // 更新所有关节角度，创建简单的行走动画
      const jointNames = ['l_shoulder_pitch', 'r_shoulder_pitch', 'l_hip_pitch', 'r_hip_pitch', 'l_knee_pitch', 'r_knee_pitch', 'l_elbow_pitch', 'r_elbow_pitch'];
      
      jointNames.forEach(jointName => {
        const baseAngle = Math.sin(time * 0.001 * animationSpeed + jointName.length) * 0.5;
        setInteractionState(prev => ({
          ...prev,
          jointAngles: {
            ...prev.jointAngles,
            [jointName]: baseAngle
          }
        }));
        
        if (onJointAngleChange) {
          onJointAngleChange(jointName, baseAngle);
        }
      });
      
      animationFrameId = requestAnimationFrame(animate);
    };
    
    animationFrameId = requestAnimationFrame(animate);
    
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [enableInteraction, interactionState.animationPlaying, interactionState.animationSpeed, onJointAngleChange]);

  // 使用useMemo缓存相机配置和灯光配置
  const cameraConfig = useMemo(() => {
    // 根据相机预设调整相机位置
    const presetPositions = {
      perspective: [3, 2, 3] as [number, number, number],
      top: [0, 5, 0] as [number, number, number],
      front: [0, 0, 5] as [number, number, number],
      side: [5, 0, 0] as [number, number, number]
    };
    
    return {
      position: presetPositions[interactionState.cameraPreset],
      fov: 50
    };
  }, [interactionState.cameraPreset]);

  const lightConfigs = useMemo(() => [
    { position: [10, 10, 10] as [number, number, number], intensity: 1, castShadow: true },
    { position: [-10, 10, -10] as [number, number, number], intensity: 0.5 }
  ], []);

  // 路径可视化组件
  const PathVisualization = useMemo(() => {
    if (!path) return null;

    const paths = Array.isArray(path) ? path : [path];
    
    return (
      <group>
        {paths.map((pathData, pathIndex) => {
          if (!pathData.points || pathData.points.length < 2) return null;
          
          // 将点转换为Three.js Vector3
          const points = pathData.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
          const color = pathData.color || '#00ff00';
          const lineWidth = pathData.lineWidth || 3;
          
          return (
            <Line
              key={pathIndex}
              points={points}
              color={color}
              lineWidth={lineWidth}
              dashed={false}
            />
          );
        })}
      </group>
    );
  }, [path]);

  // 性能优化状态
  const [performanceMode, setPerformanceMode] = useState<'balanced' | 'performance' | 'quality'>('balanced');
  
  // 使用性能监控自动调整模式
  usePerformanceMonitor({
    onIncline: () => setPerformanceMode('quality'),
    onDecline: () => setPerformanceMode('performance'),
    onChange: ({ factor }) => {
      if (factor > 1.5) {
        setPerformanceMode('performance');
      } else if (factor < 0.5) {
        setPerformanceMode('quality');
      } else {
        setPerformanceMode('balanced');
      }
    }
  });

  // 根据性能模式调整渲染设置
  const renderSettings = useMemo(() => {
    switch (performanceMode) {
      case 'performance':
        return { shadows: false, dpr: 1 };
      case 'quality':
        return { shadows: true, dpr: 2 };
      default:
        return { shadows: true, dpr: 1 };
    }
  }, [performanceMode]);

  return (
    <div className="w-full h-full bg-gray-900 rounded-xl overflow-hidden">
      <Canvas
        camera={cameraConfig}
        shadows={renderSettings.shadows}
        dpr={renderSettings.dpr}
        className="w-full h-full"
      >
        {/* 灯光 */}
        <ambientLight intensity={0.5} />
        {lightConfigs.map((light, index) => (
          <pointLight 
            key={index}
            position={light.position}
            intensity={light.intensity}
            castShadow={light.castShadow || false}
          />
        ))}
        
        {/* 机器人 */}
        <AdaptiveHumanoidRobot 
          pose={pose} 
          interactionState={{
            selectedJoint: interactionState.selectedJoint,
            hoveredJoint: interactionState.hoveredJoint,
            jointAngles: interactionState.jointAngles
          }}
          onJointHover={handleJointHover}
          onJointClick={handleJointClick}
          onJointDragStart={handleJointDragStart}
          onJointDragEnd={handleJointDragEnd}
          onJointDrag={handleJointDrag}
        />
        
        {/* 路径可视化 */}
        {PathVisualization}
        
        {/* 坐标轴指示器 */}
        <axesHelper args={[1]} position={[0, -1, 0]} />
        
        {/* 轨道控制器 */}
        {showControls && <DreiOrbitControls />}
      </Canvas>
      
      {/* 控制面板 */}
      {showControls && (
        <div className="absolute bottom-4 left-4 bg-gray-800/80 text-white p-3 rounded-lg text-sm">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-gray-700" />
              <span>左臂关节</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-gray-800" />
              <span>右臂关节</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-gray-600" />
              <span>左腿关节</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-gray-600" />
              <span>右腿关节</span>
            </div>
          </div>
          
          {/* 交互控制面板 - 仅在启用交互时显示 */}
          {enableInteraction && (
            <div className="mt-3 pt-3 border-t border-gray-700">
              <div className="mb-2">
                <h4 className="text-xs font-medium text-gray-300 mb-1">交互控制</h4>
                <div className="space-y-2">
                  {/* 关节选择状态 */}
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-400">选中关节:</span>
                    <span className="text-xs text-white">
                      {interactionState.selectedJoint || '无'}
                    </span>
                  </div>
                  
                  {/* 关节角度控制 */}
                  {interactionState.selectedJoint && (
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-400">关节角度:</span>
                        <span className="text-xs text-white">
                          {((interactionState.jointAngles[interactionState.selectedJoint] || 
                            pose.joints[interactionState.selectedJoint]?.position || 0) * 180 / Math.PI).toFixed(1)}°
                        </span>
                      </div>
                      <div className="flex space-x-1">
                        <button
                          onClick={() => updateJointAngle(interactionState.selectedJoint!, -0.1)}
                          className="flex-1 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
                          title="减小角度"
                        >
                          -
                        </button>
                        <button
                          onClick={() => updateJointAngle(interactionState.selectedJoint!, 0)}
                          className="flex-1 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
                          title="重置角度"
                        >
                          重置
                        </button>
                        <button
                          onClick={() => updateJointAngle(interactionState.selectedJoint!, 0.1)}
                          className="flex-1 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
                          title="增加角度"
                        >
                          +
                        </button>
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        使用方向键或鼠标拖拽调整关节角度
                      </div>
                    </div>
                  )}
                  
                  {/* 相机预设控制 */}
                  <div className="mt-2">
                    <div className="text-xs text-gray-400 mb-1">相机视角:</div>
                    <div className="grid grid-cols-2 gap-1">
                      {(['perspective', 'top', 'front', 'side'] as const).map((preset) => (
                        <button
                          key={preset}
                          onClick={() => setInteractionState(prev => ({...prev, cameraPreset: preset}))}
                          className={`px-2 py-1 text-xs rounded ${
                            interactionState.cameraPreset === preset
                              ? 'bg-gray-800 text-white'
                              : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                          }`}
                        >
                          {preset === 'perspective' ? '透视' :
                           preset === 'top' ? '俯视' :
                           preset === 'front' ? '前视' : '侧视'}
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  {/* 动画控制 */}
                  <div className="mt-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">动画:</span>
                      <button
                        onClick={() => setInteractionState(prev => ({
                          ...prev,
                          animationPlaying: !prev.animationPlaying
                        }))}
                        className={`px-2 py-1 text-xs rounded ${
                          interactionState.animationPlaying
                            ? 'bg-gray-900 hover:bg-gray-900 text-white'
                            : 'bg-gray-700 hover:bg-gray-800 text-white'
                        }`}
                      >
                        {interactionState.animationPlaying ? '停止' : '播放'}
                      </button>
                    </div>
                    {interactionState.animationPlaying && (
                      <div className="mt-1">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-gray-400">速度:</span>
                          <span className="text-xs text-white">{interactionState.animationSpeed.toFixed(1)}x</span>
                        </div>
                        <input
                          type="range"
                          min="0.1"
                          max="3.0"
                          step="0.1"
                          value={interactionState.animationSpeed}
                          onChange={(e) => setInteractionState(prev => ({
                            ...prev,
                            animationSpeed: parseFloat(e.target.value)
                          }))}
                          className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                        />
                      </div>
                    )}
                  </div>
                  
                  {/* 交互提示 */}
                  <div className="mt-2 text-xs text-gray-500">
                    <div>• 点击关节进行选择</div>
                    <div>• 拖拽关节调整角度</div>
                    <div>• 使用方向键微调角度</div>
                    <div>• Home键重置角度</div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* 性能模式指示器 */}
          <div className="mt-3 pt-3 border-t border-gray-700">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                performanceMode === 'performance' ? 'bg-gray-600' :
                performanceMode === 'quality' ? 'bg-gray-600' :
                'bg-gray-700'
              }`} />
              <span className="text-xs text-gray-300">
                {performanceMode === 'performance' ? '性能模式' :
                 performanceMode === 'quality' ? '画质模式' :
                 '平衡模式'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

export default RobotScene;