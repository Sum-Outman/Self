# Control API | 控制API

This document provides complete API reference for controlling AGI Robot hardware, including robots, sensors, and actuators.

本文档提供控制AGI机器人硬件的完整API参考，包括机器人、传感器和执行器。

## Overview | 概述

The Control API provides a unified interface for controlling all hardware components in the AGI Robot system. The API is designed to be consistent across different hardware platforms and simulation environments.

控制API为控制AGI机器人系统中的所有硬件组件提供统一接口。该API设计为在不同硬件平台和仿真环境中保持一致。

## Core API Classes | 核心API类

### RobotBase Class | RobotBase类
Base class for all robot controllers.

所有机器人控制器的基类。

```python
class RobotBase:
    """Base class for robot control."""
    
    def move_to_position(self, position: List[float]) -> bool:
        """Move robot to target position."""
        pass
    
    def move_to_pose(self, pose: List[float]) -> bool:
        """Move robot to target pose (position + orientation)."""
        pass
    
    def move_joint(self, joint: int, angle: float) -> bool:
        """Move specific joint to target angle."""
        pass
    
    def get_joint_positions(self) -> List[float]:
        """Get current joint positions."""
        pass
    
    def get_pose(self) -> List[float]:
        """Get current end effector pose."""
        pass
    
    def get_force_torque(self) -> List[float]:
        """Get force-torque sensor readings."""
        pass
    
    def stop(self) -> bool:
        """Stop robot movement immediately."""
        pass
    
    def is_moving(self) -> bool:
        """Check if robot is currently moving."""
        pass
```

### SensorBase Class | SensorBase类
Base class for all sensors.

所有传感器的基类。

```python
class SensorBase:
    """Base class for sensor devices."""
    
    def read(self) -> Any:
        """Read sensor data."""
        pass
    
    def start_streaming(self) -> bool:
        """Start continuous data streaming."""
        pass
    
    def stop_streaming(self) -> bool:
        """Stop data streaming."""
        pass
    
    def get_configuration(self) -> Dict:
        """Get sensor configuration."""
        pass
    
    def set_configuration(self, config: Dict) -> bool:
        """Set sensor configuration."""
        pass
    
    def calibrate(self) -> bool:
        """Calibrate sensor."""
        pass
```

### ActuatorBase Class | ActuatorBase类
Base class for all actuators.

所有执行器的基类。

```python
class ActuatorBase:
    """Base class for actuator devices."""
    
    def activate(self) -> bool:
        """Activate actuator."""
        pass
    
    def deactivate(self) -> bool:
        """Deactivate actuator."""
        pass
    
    def set_position(self, position: float) -> bool:
        """Set actuator position."""
        pass
    
    def set_velocity(self, velocity: float) -> bool:
        """Set actuator velocity."""
        pass
    
    def set_torque(self, torque: float) -> bool:
        """Set actuator torque."""
        pass
    
    def get_status(self) -> Dict:
        """Get actuator status."""
        pass
```

## Robot Control API | 机器人控制API

### HumanoidRobot Class | HumanoidRobot类
Control class for humanoid robots.

人形机器人的控制类。

```python
class HumanoidRobot(RobotBase):
    """Control class for humanoid robots."""
    
    def walk(self, direction: str, speed: float = 1.0) -> bool:
        """Walk in specified direction (forward, backward, left, right)."""
        pass
    
    def turn(self, angle: float) -> bool:
        """Turn by specified angle (degrees)."""
        pass
    
    def stand(self) -> bool:
        """Stand up from sitting or lying position."""
        pass
    
    def sit(self) -> bool:
        """Sit down from standing position."""
        pass
    
    def balance(self) -> bool:
        """Activate balancing control."""
        pass
    
    def get_foot_pressure(self) -> List[float]:
        """Get foot pressure sensor readings."""
        pass
    
    def get_imu_data(self) -> Dict:
        """Get IMU data (orientation, acceleration, angular velocity)."""
        pass
```

### RobotArm Class | RobotArm类
Control class for robotic arms.

机械臂的控制类。

```python
class RobotArm(RobotBase):
    """Control class for robotic arms."""
    
    def pick(self, object_position: List[float]) -> bool:
        """Pick object at specified position."""
        pass
    
    def place(self, target_position: List[float]) -> bool:
        """Place object at target position."""
        pass
    
    def move_linear(self, path: List[List[float]]) -> bool:
        """Move along linear path."""
        pass
    
    def move_circular(self, center: List[float], 
                      angle: float, plane: str) -> bool:
        """Move along circular arc."""
        pass
    
    def set_payload(self, mass: float, cog: List[float]) -> bool:
        """Set payload mass and center of gravity."""
        pass
    
    def teach_mode(self, enable: bool) -> bool:
        """Enable/disable teach mode for manual guidance."""
        pass
```

### MobileRobot Class | MobileRobot类
Control class for mobile robots.

移动机器人的控制类。

```python
class MobileRobot(RobotBase):
    """Control class for mobile robots."""
    
    def move(self, linear: float, angular: float) -> bool:
        """Move with linear and angular velocity."""
        pass
    
    def navigate_to(self, goal: List[float]) -> bool:
        """Navigate to goal position."""
        pass
    
    def follow_path(self, path: List[List[float]]) -> bool:
        """Follow specified path."""
        pass
    
    def get_odometry(self) -> Dict:
        """Get odometry data (position, orientation, velocity)."""
        pass
    
    def get_laser_scan(self) -> List[float]:
        """Get laser scan data."""
        pass
    
    def get_map(self) -> Any:
        """Get current map if available."""
        pass
```

## Sensor API | 传感器API

### Camera Class | Camera类
Control class for cameras.

相机的控制类。

```python
class Camera(SensorBase):
    """Control class for cameras."""
    
    def capture_image(self) -> np.ndarray:
        """Capture RGB image."""
        pass
    
    def capture_depth(self) -> np.ndarray:
        """Capture depth image."""
        pass
    
    def capture_point_cloud(self) -> np.ndarray:
        """Capture 3D point cloud."""
        pass
    
    def get_intrinsics(self) -> Dict:
        """Get camera intrinsic parameters."""
        pass
    
    def get_extrinsics(self) -> Dict:
        """Get camera extrinsic parameters."""
        pass
    
    def set_exposure(self, exposure: float) -> bool:
        """Set camera exposure."""
        pass
    
    def set_white_balance(self, white_balance: float) -> bool:
        """Set camera white balance."""
        pass
```

### IMU Class | IMU类
Control class for IMU sensors.

IMU传感器的控制类。

```python
class IMU(SensorBase):
    """Control class for IMU sensors."""
    
    def get_orientation(self) -> List[float]:
        """Get orientation (quaternion or Euler angles)."""
        pass
    
    def get_acceleration(self) -> List[float]:
        """Get linear acceleration."""
        pass
    
    def get_angular_velocity(self) -> List[float]:
        """Get angular velocity."""
        pass
    
    def get_magnetic_field(self) -> List[float]:
        """Get magnetic field strength."""
        pass
    
    def get_temperature(self) -> float:
        """Get sensor temperature."""
        pass
    
    def calibrate_gyro(self) -> bool:
        """Calibrate gyroscope."""
        pass
    
    def calibrate_accelerometer(self) -> bool:
        """Calibrate accelerometer."""
        pass
```

### LiDAR Class | LiDAR类
Control class for LiDAR sensors.

激光雷达传感器的控制类。

```python
class LiDAR(SensorBase):
    """Control class for LiDAR sensors."""
    
    def get_scan(self) -> List[float]:
        """Get LiDAR scan data."""
        pass
    
    def get_point_cloud(self) -> np.ndarray:
        """Get 3D point cloud."""
        pass
    
    def get_intensity(self) -> List[float]:
        """Get intensity values."""
        pass
    
    def set_scan_frequency(self, frequency: float) -> bool:
        """Set scan frequency."""
        pass
    
    def set_angle_range(self, min_angle: float, max_angle: float) -> bool:
        """Set angle range."""
        pass
```

## Actuator API | 执行器API

### Gripper Class | Gripper类
Control class for grippers.

夹爪的控制类。

```python
class Gripper(ActuatorBase):
    """Control class for grippers."""
    
    def open(self) -> bool:
        """Open gripper."""
        pass
    
    def close(self) -> bool:
        """Close gripper."""
        pass
    
    def grasp(self, width: float, force: float) -> bool:
        """Grasp with specified width and force."""
        pass
    
    def release(self) -> bool:
        """Release grasped object."""
        pass
    
    def get_width(self) -> float:
        """Get current gripper width."""
        pass
    
    def get_force(self) -> float:
        """Get current gripping force."""
        pass
```

### Motor Class | Motor类
Control class for motors.

电机的控制类。

```python
class Motor(ActuatorBase):
    """Control class for motors."""
    
    def set_rpm(self, rpm: float) -> bool:
        """Set motor RPM."""
        pass
    
    def set_duty_cycle(self, duty_cycle: float) -> bool:
        """Set motor duty cycle."""
        pass
    
    def set_direction(self, direction: str) -> bool:
        """Set motor direction (forward, reverse)."""
        pass
    
    def get_rpm(self) -> float:
        """Get current RPM."""
        pass
    
    def get_current(self) -> float:
        """Get motor current."""
        pass
    
    def get_temperature(self) -> float:
        """Get motor temperature."""
        pass
```

## High-Level Control API | 高级控制API

### Motion Planning | 运动规划

```python
class MotionPlanner:
    """Motion planning for robots."""
    
    def plan_path(self, start: List[float], 
                  goal: List[float]) -> List[List[float]]:
        """Plan path from start to goal."""
        pass
    
    def plan_trajectory(self, waypoints: List[List[float]]) -> List[List[float]]:
        """Plan trajectory through waypoints."""
        pass
    
    def check_collision(self, pose: List[float]) -> bool:
        """Check if pose is in collision."""
        pass
    
    def optimize_trajectory(self, trajectory: List[List[float]]) -> List[List[float]]:
        """Optimize trajectory for smoothness and efficiency."""
        pass
```

### Force Control | 力控制

```python
class ForceController:
    """Force control for robots."""
    
    def impedance_control(self, desired_force: List[float]) -> bool:
        """Impedance control to achieve desired force."""
        pass
    
    def admittance_control(self, desired_position: List[float]) -> bool:
        """Admittance control to achieve desired position under force constraints."""
        pass
    
    def hybrid_force_position_control(self, force_dof: List[bool], 
                                      position_dof: List[bool]) -> bool:
        """Hybrid force/position control."""
        pass
```

### Vision-Based Control | 基于视觉的控制

```python
class VisualServoing:
    """Visual servoing control."""
    
    def image_based_visual_servoing(self, target_image: np.ndarray) -> bool:
        """Image-based visual servoing."""
        pass
    
    def position_based_visual_servoing(self, target_pose: List[float]) -> bool:
        """Position-based visual servoing."""
        pass
    
    def visual_tracking(self, target: Any) -> bool:
        """Visual tracking of target."""
        pass
```

## Safety API | 安全API

```python
class SafetyMonitor:
    """Safety monitoring and control."""
    
    def enable_safety_limits(self) -> bool:
        """Enable safety limits."""
        pass
    
    def disable_safety_limits(self) -> bool:
        """Disable safety limits (for testing only)."""
        pass
    
    def emergency_stop(self) -> bool:
        """Emergency stop."""
        pass
    
    def check_collision(self) -> bool:
        """Check for collisions."""
        pass
    
    def check_limits(self) -> bool:
        """Check joint limits."""
        pass
    
    def check_temperature(self) -> bool:
        """Check temperature limits."""
        pass
    
    def check_power(self) -> bool:
        """Check power consumption."""
        pass
```

## Usage Examples | 使用示例

### Basic Robot Control | 基本机器人控制

```python
from hardware.unified_interface import RobotArm, Camera

# Initialize hardware
arm = RobotArm(robot_type="ur5e")
camera = Camera(camera_type="realsense_d435")

# Capture image and detect object
image = camera.capture_image()
object_position = detect_object(image)  # Custom object detection

# Move to object and pick
arm.move_to_position(object_position)
arm.pick(object_position)

# Move to target and place
target_position = [0.5, 0.2, 0.3, 0, 0, 0]
arm.move_to_position(target_position)
arm.place(target_position)
```

### Sensor Fusion | 传感器融合

```python
from hardware.unified_interface import IMU, Camera, LiDAR
import numpy as np

# Initialize sensors
imu = IMU(imu_type="bno055")
camera = Camera(camera_type="realsense_d435")
lidar = LiDAR(lidar_type="rplidar_a3")

# Read sensor data
orientation = imu.get_orientation()
image = camera.capture_image()
point_cloud = lidar.get_point_cloud()

# Fuse sensor data
fused_data = fuse_sensors(orientation, image, point_cloud)
```

### Advanced Motion Control | 高级运动控制

```python
from hardware.unified_interface import HumanoidRobot, MotionPlanner

# Initialize robot and planner
robot = HumanoidRobot(robot_type="unitree_h1")
planner = MotionPlanner()

# Plan path to goal
start = robot.get_pose()
goal = [2.0, 1.5, 0, 0, 0, 0]  # x, y, z, roll, pitch, yaw
path = planner.plan_path(start, goal)

# Execute path
for waypoint in path:
    robot.walk_towards(waypoint)
    
    # Check safety
    if robot.check_collision():
        robot.emergency_stop()
        break
```

## Error Handling | 错误处理

All API methods return boolean values indicating success or failure. Detailed error information can be obtained through exception handling or logging.

所有API方法返回布尔值指示成功或失败。详细错误信息可通过异常处理或日志获取。

```python
try:
    success = arm.move_to_position(target_position)
    if not success:
        print(f"Move failed: {arm.get_last_error()}")
except HardwareError as e:
    print(f"Hardware error: {e}")
    # Handle error appropriately
```

## Next Steps | 后续步骤

- **Simulation**: Test control algorithms in simulation first
- **Real Hardware**: Deploy to real hardware after simulation testing
- **Advanced Control**: Implement advanced control algorithms
- **Integration**: Integrate with AGI core intelligence

- **仿真**: 首先在仿真中测试控制算法
- **真实硬件**: 仿真测试后部署到真实硬件
- **高级控制**: 实现高级控制算法
- **集成**: 与AGI核心智能集成

---

*Last Updated: March 30, 2026*  
*最后更新: 2026年3月30日*