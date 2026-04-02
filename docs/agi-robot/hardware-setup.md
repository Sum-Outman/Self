# Hardware Setup | 硬件设置

This guide provides detailed instructions for setting up hardware components for the AGI Robot system, including robot platforms, sensors, actuators, and communication interfaces.

本指南提供 AGI 机器人系统硬件组件设置的详细说明，包括机器人平台、传感器、执行器和通信接口。

## Prerequisites | 先决条件

### Hardware Requirements | 硬件要求
- **Computer**: x86-64 CPU, 16GB+ RAM, USB ports, Ethernet port
- **Robot Platform**: Compatible robot (see Supported Hardware section)
- **Sensors**: Cameras, IMU, LiDAR, etc. as needed for your application
- **Power Supply**: Adequate power supply for all components
- **Cables and Connectors**: Appropriate cables for all connections

- **计算机**: x86-64 CPU，16GB+ RAM，USB端口，以太网端口
- **机器人平台**: 兼容的机器人（见支持的硬件部分）
- **传感器**: 相机、IMU、激光雷达等，根据应用需求
- **电源**: 所有组件的充足电源供应
- **电缆和连接器**: 所有连接的适当电缆

### Software Requirements | 软件要求
- **Operating System**: Windows 10/11, Ubuntu 20.04/22.04, or macOS 12+
- **Python**: 3.9 or higher with pip package manager
- **ROS 2**: Optional, required for Gazebo simulation and ROS integration
- **Drivers**: Specific hardware drivers as required

- **操作系统**: Windows 10/11、Ubuntu 20.04/22.04 或 macOS 12+
- **Python**: 3.9或更高版本，带pip包管理器
- **ROS 2**: 可选，Gazebo仿真和ROS集成所需
- **驱动程序**: 特定硬件驱动程序（如需要）

## Supported Hardware | 支持的硬件

### Robot Platforms | 机器人平台

#### Humanoid Robots | 人形机器人
- **Boston Dynamics Atlas**: Advanced humanoid robot (research only)
- **Unitree H1**: Affordable humanoid robot with good performance
- **Robotis OP3**: Small humanoid robot for education and research
- **Custom Humanoids**: Support for custom humanoid configurations

- **Boston Dynamics Atlas**: 先进人形机器人（仅研究）
- **Unitree H1**: 性能良好的经济型人形机器人
- **Robotis OP3**: 用于教育和研究的小型人形机器人
- **自定义人形机器人**: 支持自定义人形机器人配置

#### Robotic Arms | 机械臂
- **Universal Robots UR Series**: UR3, UR5, UR10, UR16e
- **Franka Emika Panda**: Research-focused collaborative robot
- **KUKA LBR iiwa**: Sensitive collaborative robot for industrial applications
- **DIY Robotic Arms**: Support for custom and open-source robotic arms

- **Universal Robots UR系列**: UR3、UR5、UR10、UR16e
- **Franka Emika Panda**: 研究型协作机器人
- **KUKA LBR iiwa**: 工业应用的敏感协作机器人
- **DIY机械臂**: 支持自定义和开源机械臂

#### Mobile Robots | 移动机器人
- **TurtleBot**: Popular ROS-based mobile robot platform
- **Clearpath Robotics**: Jackal, Husky, Ridgeback mobile robots
- **DJI RoboMaster**: Educational and competition robot platform
- **Custom Mobile Bases**: Support for custom wheeled/tracked platforms

- **TurtleBot**: 流行的基于ROS的移动机器人平台
- **Clearpath Robotics**: Jackal、Husky、Ridgeback移动机器人
- **DJI RoboMaster**: 教育和竞赛机器人平台
- **自定义移动底盘**: 支持自定义轮式/履带式平台

### Sensors | 传感器

#### Vision Sensors | 视觉传感器
- **RGB Cameras**: Logitech, Intel RealSense, USB webcams
- **Depth Cameras**: Intel RealSense D400 series, Microsoft Azure Kinect
- **Stereo Cameras**: ZED camera, Intel RealSense T265
- **Industrial Cameras**: Basler, FLIR, Allied Vision

- **RGB相机**: Logitech、Intel RealSense、USB网络摄像头
- **深度相机**: Intel RealSense D400系列、Microsoft Azure Kinect
- **立体相机**: ZED相机、Intel RealSense T265
- **工业相机**: Basler、FLIR、Allied Vision

#### Inertial Sensors | 惯性传感器
- **IMUs**: Bosch BNO055, InvenSense MPU-9250, Xsens MTi
- **GNSS**: GPS, GLONASS, Galileo receivers (for outdoor navigation)
- **Magnetometers**: Compass sensors for orientation

- **IMU**: Bosch BNO055、InvenSense MPU-9250、Xsens MTi
- **GNSS**: GPS、GLONASS、Galileo接收器（用于室外导航）
- **磁力计**: 方向感知的罗盘传感器

#### Range Sensors | 距离传感器
- **LiDAR**: Velodyne, Ouster, Hesai, Slamtec RPLIDAR
- **Ultrasonic Sensors**: HC-SR04, MaxBotix
- **Infrared Sensors**: Sharp GP2Y0A series, VL53L0X

- **激光雷达**: Velodyne、Ouster、Hesai、Slamtec RPLIDAR
- **超声波传感器**: HC-SR04、MaxBotix
- **红外传感器**: Sharp GP2Y0A系列、VL53L0X

### Actuators | 执行器

#### Motors | 电机
- **Servo Motors**: Dynamixel, Herkulex, Robotis servo series
- **DC Motors**: Brushed and brushless motors with motor controllers
- **Stepper Motors**: NEMA series steppers with drivers
- **Linear Actuators**: Electric linear actuators for linear motion

- **伺服电机**: Dynamixel、Herkulex、Robotis伺服系列
- **直流电机**: 带电机控制器的有刷和无刷电机
- **步进电机**: 带驱动器的NEMA系列步进电机
- **线性执行器**: 线性运动的电动线性执行器

#### Motor Controllers | 电机控制器
- **Servo Controllers**: USB2Dynamixel, U2D2, OpenCM
- **DC Motor Controllers**: RoboClaw, Sabertooth, VESC
- **Stepper Drivers**: DRV8825, A4988, TMC series
- **CAN bus Controllers**: PCAN, SocketCAN compatible devices

- **伺服控制器**: USB2Dynamixel、U2D2、OpenCM
- **直流电机控制器**: RoboClaw、Sabertooth、VESC
- **步进驱动器**: DRV8825、A4988、TMC系列
- **CAN总线控制器**: PCAN、SocketCAN兼容设备

## Connection Setup | 连接设置

### USB Connections | USB连接
Most sensors and controllers connect via USB. Ensure proper drivers are installed.

大多数传感器和控制器通过USB连接。确保安装适当的驱动程序。

```bash
# Check USB devices on Linux
lsusb

# Check COM ports on Windows
python -m serial.tools.list_ports
```

### Serial Connections | 串口连接
For serial devices (RS-232, RS-485, TTL serial):

对于串口设备（RS-232、RS-485、TTL串口）：

```python
import serial

# Configure serial connection
ser = serial.Serial(
    port='/dev/ttyUSB0',  # Linux
    # port='COM3',        # Windows
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)
```

### Ethernet Connections | 以太网连接
For network-connected devices (Ethernet, WiFi):

对于网络连接设备（以太网、WiFi）：

```python
import socket

# Connect to network device
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('192.168.1.100', 5000))
```

### CAN bus Connections | CAN总线连接
For CAN bus devices (common in automotive and industrial robotics):

对于CAN总线设备（常见于汽车和工业机器人）：

```python
import can

# Create CAN bus interface
bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=500000)
```

## Configuration | 配置

### Hardware Configuration File | 硬件配置文件
The system uses a YAML configuration file to define hardware components:

系统使用YAML配置文件定义硬件组件：

```yaml
# hardware_config.yaml
robot:
  type: "ur5e"
  connection:
    type: "tcp"
    host: "192.168.1.50"
    port: 30003
  parameters:
    joint_limits: [-pi, pi, -pi, pi, -pi, pi, -pi, pi]
    max_velocity: [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
    max_acceleration: [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]

sensors:
  camera:
    - name: "front_camera"
      type: "realsense_d435"
      connection: "usb"
      parameters:
        resolution: [1280, 720]
        fps: 30
        depth_enabled: true
  
  imu:
    - name: "body_imu"
      type: "bno055"
      connection: "i2c"
      parameters:
        address: 0x28
        update_rate: 100

actuators:
  gripper:
    - name: "robotiq_2f85"
      type: "robotiq"
      connection: "tcp"
      parameters:
        host: "192.168.1.51"
        port: 63352
```

### Loading Configuration | 加载配置
```python
from hardware.hardware_factory import HardwareFactory

# Load hardware configuration
factory = HardwareFactory(config_file="hardware_config.yaml")

# Initialize robot
robot = factory.create_robot()

# Initialize sensors
sensors = factory.create_sensors()

# Initialize actuators
actuators = factory.create_actuators()
```

## Calibration | 校准

### Robot Calibration | 机器人校准
1. **Home Position Calibration**: Define robot home position
2. **Joint Limit Calibration**: Set joint position limits
3. **Tool Center Point Calibration**: Calibrate end effector position
4. **Kinematic Calibration**: Improve accuracy through kinematic calibration

1. **归位位置校准**: 定义机器人归位位置
2. **关节限位校准**: 设置关节位置限制
3. **工具中心点校准**: 校准末端执行器位置
4. **运动学校准**: 通过运动学校准提高精度

### Sensor Calibration | 传感器校准
1. **Camera Calibration**: Intrinsic and extrinsic camera calibration
2. **IMU Calibration**: Gyroscope and accelerometer bias calibration
3. **LiDAR Calibration**: Laser scanner calibration and alignment
4. **Multi-sensor Calibration**: Calibrate transformations between sensors

1. **相机校准**: 相机内外参校准
2. **IMU校准**: 陀螺仪和加速度计偏置校准
3. **激光雷达校准**: 激光扫描仪校准和对齐
4. **多传感器校准**: 校准传感器间的变换关系

### Calibration Tools | 校准工具
```python
from hardware.calibration import RobotCalibrator, SensorCalibrator

# Robot calibration
robot_calibrator = RobotCalibrator(robot)
robot_calibrator.calibrate_home_position()
robot_calibrator.calibrate_tool_center_point()

# Camera calibration
camera_calibrator = SensorCalibrator(camera)
camera_calibrator.calibrate_intrinsics(checkerboard_size=(9, 6))
camera_calibrator.calibrate_extrinsics(reference_sensor=imu)
```

## Testing | 测试

### Basic Functionality Test | 基本功能测试
```python
# Test robot movement
def test_robot_movement():
    robot.move_to_home()
    
    # Test joint movement
    for joint in range(robot.num_joints):
        target = robot.get_joint_position(joint) + 0.1
        robot.move_joint(joint, target)
    
    # Test Cartesian movement
    current_pose = robot.get_pose()
    target_pose = [current_pose[0] + 0.1, current_pose[1], current_pose[2], 
                   current_pose[3], current_pose[4], current_pose[5]]
    robot.move_to_pose(target_pose)
    
    return True

# Test sensor reading
def test_sensors():
    for sensor_name, sensor in sensors.items():
        data = sensor.read()
        print(f"{sensor_name}: {data}")
        if data is None:
            return False
    return True
```

### Safety Checks | 安全检查
1. **Limit Switch Check**: Verify limit switches are functional
2. **Collision Detection Test**: Test collision detection system
3. **Emergency Stop Test**: Test emergency stop functionality
4. **Power Monitoring**: Monitor power consumption and temperature

1. **限位开关检查**: 验证限位开关功能正常
2. **碰撞检测测试**: 测试碰撞检测系统
3. **紧急停止测试**: 测试紧急停止功能
4. **电源监控**: 监控功耗和温度

## Troubleshooting | 故障排除

### Common Issues | 常见问题

#### Connection Issues | 连接问题
- **Device not detected**: Check cables, power, and drivers
- **Permission denied**: Fix serial port permissions on Linux
- **Timeout errors**: Check baud rate and connection stability

- **设备未检测到**: 检查电缆、电源和驱动程序
- **权限被拒绝**: 修复Linux上的串口权限
- **超时错误**: 检查波特率和连接稳定性

#### Communication Issues | 通信问题
- **Data corruption**: Check cable quality and electromagnetic interference
- **Protocol mismatch**: Verify communication protocol settings
- **Buffer overflow**: Adjust buffer sizes and data rates

- **数据损坏**: 检查电缆质量和电磁干扰
- **协议不匹配**: 验证通信协议设置
- **缓冲区溢出**: 调整缓冲区大小和数据速率

#### Performance Issues | 性能问题
- **Low update rate**: Reduce data size or increase baud rate
- **High latency**: Optimize communication protocol and hardware
- **Jitter**: Use real-time operating system or optimize scheduling

- **更新率低**: 减少数据大小或提高波特率
- **高延迟**: 优化通信协议和硬件
- **抖动**: 使用实时操作系统或优化调度

### Diagnostic Tools | 诊断工具
```python
# Hardware diagnostic tool
python -m hardware.diagnostic --robot --sensors --actuators

# Network diagnostic tool
python -m hardware.network_test --host 192.168.1.50 --port 30003

# Performance test tool
python -m hardware.performance_test --duration 60 --frequency 100
```

## Next Steps | 后续步骤

After hardware setup is complete, proceed to:

硬件设置完成后，继续：

1. **Control API**: Learn how to control the hardware programmatically
2. **Simulation**: Test in simulation before using real hardware
3. **Real Robot**: Integrate with real hardware and test
4. **Applications**: Build applications using the hardware

1. **控制API**: 学习如何通过编程控制硬件
2. **仿真**: 在使用真实硬件前在仿真中测试
3. **真实机器人**: 与真实硬件集成并测试
4. **应用**: 使用硬件构建应用

---

*Last Updated: March 30, 2026*  
*最后更新: 2026年3月30日*