# Robot Simulation | 机器人仿真

This guide covers robot simulation environments in the Self AGI system, including PyBullet, Gazebo, and unified simulation interfaces.

本指南涵盖 Self AGI 系统中的机器人仿真环境，包括 PyBullet、Gazebo 和统一仿真接口。

## Simulation Overview | 仿真概述

### Purpose of Simulation | 仿真目的
- **Development and Testing**: Develop and test algorithms without physical hardware
- **Safety Testing**: Test dangerous scenarios safely
- **Performance Evaluation**: Evaluate algorithms under controlled conditions
- **Training Data Generation**: Generate training data for machine learning

- **开发和测试**: 在没有物理硬件的情况下开发和测试算法
- **安全测试**: 安全地测试危险场景
- **性能评估**: 在受控条件下评估算法
- **训练数据生成**: 为机器学习生成训练数据

### Supported Simulation Environments | 支持的仿真环境

#### PyBullet Simulation | PyBullet仿真
- **Physics Engine**: Bullet Physics for accurate physics simulation
- **Features**: Real-time simulation, accurate collision detection, contact mechanics
- **Robot Support**: URDF robot description format support
- **Sensor Simulation**: Cameras, depth sensors, IMU, force-torque sensors
- **Visualization**: Basic 3D visualization with OpenGL

- **物理引擎**: Bullet Physics 提供精确的物理仿真
- **特性**: 实时仿真、精确碰撞检测、接触力学
- **机器人支持**: 支持 URDF 机器人描述格式
- **传感器仿真**: 相机、深度传感器、IMU、力扭矩传感器
- **可视化**: 使用 OpenGL 的基本 3D 可视化

#### Gazebo Simulation | Gazebo仿真
- **ROS 2 Integration**: Seamless integration with ROS 2 ecosystem
- **High-Quality Rendering**: OGRE-based rendering with advanced graphics
- **Advanced Sensor Models**: Realistic LiDAR, camera, IMU models
- **Plugin System**: Extensible plugin system for custom functionality
- **Multi-Robot Support**: Support for multi-robot simulations

- **ROS 2集成**: 与 ROS 2 生态系统无缝集成
- **高质量渲染**: 基于 OGRE 的渲染，具有高级图形
- **高级传感器模型**: 真实的激光雷达、相机、IMU 模型
- **插件系统**: 用于自定义功能的可扩展插件系统
- **多机器人支持**: 支持多机器人仿真

#### Unified Simulation Interface | 统一仿真接口
- **Common API**: Same API for both PyBullet and Gazebo
- **Abstraction Layer**: Hardware abstraction for simulation and real hardware
- **Easy Switching**: Switch between simulation environments easily
- **Code Reusability**: Same code works for simulation and real hardware

- **通用API**: PyBullet 和 Gazebo 使用相同的 API
- **抽象层**: 仿真和真实硬件的硬件抽象
- **轻松切换**: 轻松切换仿真环境
- **代码可重用性**: 相同的代码适用于仿真和真实硬件

## Setting Up Simulation | 设置仿真

### Prerequisites | 先决条件
```bash
# Install PyBullet
pip install pybullet

# Install Gazebo (requires ROS 2)
# Follow ROS 2 installation instructions for your platform

# Install simulation dependencies
pip install -e ".[simulation]"
```

### PyBullet Setup | PyBullet设置
```python
from hardware.simulation.pybullet_sim import PyBulletSimulation

# Initialize PyBullet simulation
sim = PyBulletSimulation(
    gui=True,           # Enable GUI
    physics_rate=240,   # Physics update rate (Hz)
    gravity=[0, 0, -9.81]  # Gravity vector
)

# Load robot
robot_id = sim.load_robot(
    urdf_path="robots/ur5e.urdf",
    base_position=[0, 0, 0],
    base_orientation=[0, 0, 0, 1]
)

# Load environment
sim.load_environment(
    environment="tabletop",
    objects=["table", "cube", "sphere"]
)
```

### Gazebo Setup | Gazebo设置
```python
from hardware.simulation.gazebo_sim import GazeboSimulation

# Initialize Gazebo simulation
sim = GazeboSimulation(
    world="empty.world",  # Gazebo world file
    physics_engine="ode",  # Physics engine (ode, bullet, dart)
    realtime_factor=1.0    # Real-time factor
)

# Spawn robot
sim.spawn_robot(
    model_name="ur5e",
    namespace="robot1",
    pose=[[0, 0, 0], [0, 0, 0, 1]]
)

# Launch ROS 2 nodes
sim.launch_ros_nodes([
    "robot_state_publisher",
    "joint_state_controller",
    "effort_controller"
])
```

## Simulation Control | 仿真控制

### Robot Control in Simulation | 仿真中的机器人控制
```python
from hardware.unified_interface import RobotArm

# Initialize robot in simulation
arm = RobotArm(
    robot_type="ur5e",
    simulation=sim,      # Simulation instance
    control_mode="position"  # Position control mode
)

# Control robot (same API as real hardware)
arm.move_to_position([0.5, 0.2, 0.3, 0, 0, 0])
arm.pick(object_position=[0.3, 0.1, 0.1])
arm.place(target_position=[0.3, -0.1, 0.1])
```

### Sensor Simulation | 传感器仿真
```python
# Camera simulation
camera = sim.create_camera(
    name="front_camera",
    position=[0.5, 0, 0.3],
    orientation=[0, 0, 0, 1],
    resolution=[640, 480],
    fov=90
)

# Capture image
image = camera.capture_image()
depth = camera.capture_depth()

# LiDAR simulation
lidar = sim.create_lidar(
    name="scan_lidar",
    position=[0, 0, 0.5],
    orientation=[0, 0, 0, 1],
    range=10.0,
    samples=360
)

# Get scan data
scan = lidar.get_scan()
point_cloud = lidar.get_point_cloud()
```

### Physics Simulation | 物理仿真
```python
# Set physics parameters
sim.set_physics_parameters(
    gravity=[0, 0, -9.81],
    timestep=0.001,
    solver_iterations=50
)

# Apply forces
sim.apply_force(
    body_id=robot_id,
    link_index=3,
    force=[10, 0, 0],  # Force in Newtons
    position=[0, 0, 0]  # Position in link frame
)

# Check collisions
collision = sim.check_collision(
    body_a=robot_id,
    body_b=object_id
)

# Get contact information
contacts = sim.get_contact_points(
    body_a=robot_id,
    body_b=object_id
)
```

## Advanced Simulation Features | 高级仿真特性

### Environment Modeling | 环境建模
```python
# Create custom environment
environment = sim.create_environment(
    name="warehouse",
    objects={
        "shelves": {
            "type": "box",
            "position": [0, 2, 0],
            "dimensions": [2, 0.5, 2]
        },
        "conveyor": {
            "type": "plane",
            "position": [0, 0, 0],
            "normal": [0, 0, 1],
            "friction": 0.5
        }
    }
)

# Add dynamic objects
dynamic_object = sim.add_dynamic_object(
    name="moving_box",
    shape="box",
    mass=1.0,
    position=[1, 0, 0.5],
    velocity=[0.5, 0, 0]  # Initial velocity
)
```

### Multi-Robot Simulation | 多机器人仿真
```python
# Create multi-robot simulation
multi_sim = MultiRobotSimulation(
    num_robots=3,
    robot_type="ur5e",
    positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]]
)

# Control multiple robots
robots = multi_sim.get_robots()
robots[0].move_to_position([0.5, 0.2, 0.3])
robots[1].pick(object_position=[1.5, 0.2, 0.1])
robots[2].place(target_position=[2.5, 0.2, 0.1])

# Robot-robot interaction
collision = multi_sim.check_robot_collision(robot_a=0, robot_b=1)
```

### Simulation Recording and Playback | 仿真记录和回放
```python
# Record simulation
recorder = SimulationRecorder(sim)
recorder.start_recording(
    output_file="simulation_recording.bag",
    record_video=True,
    record_sensor_data=True
)

# Run simulation
run_simulation_scenario()

# Stop recording
recorder.stop_recording()

# Playback recording
player = SimulationPlayer()
player.load_recording("simulation_recording.bag")
player.play(
    speed=1.0,  # Playback speed
    loop=False  # Don't loop
)
```

## Simulation for Training | 用于训练的仿真

### Training Data Generation | 训练数据生成
```python
from training.simulation_data import SimulationDataGenerator

# Initialize data generator
data_generator = SimulationDataGenerator(
    simulation=sim,
    robot=robot,
    sensors=[camera, lidar, imu]
)

# Generate training data
dataset = data_generator.generate_dataset(
    num_samples=1000,
    scenarios=["pick_and_place", "navigation", "manipulation"],
    variations={
        "lighting": ["day", "night", "low_light"],
        "object_types": ["box", "sphere", "cylinder"],
        "distractions": ["none", "moving_objects", "noise"]
    }
)

# Save dataset
dataset.save("simulation_dataset.h5")
```

### Reinforcement Learning in Simulation | 仿真中的强化学习
```python
from training.rl_simulation import RLSimulationEnv

# Create RL environment
env = RLSimulationEnv(
    simulation=sim,
    robot=robot,
    task="pick_and_place",
    reward_function=default_reward_function,
    max_steps=1000
)

# Train RL agent
agent = RLAgent(env)
training_results = agent.train(
    num_episodes=10000,
    learning_rate=0.001,
    gamma=0.99
)

# Evaluate agent
evaluation_results = agent.evaluate(
    num_episodes=100,
    render=True
)
```

## Performance Optimization | 性能优化

### Simulation Performance | 仿真性能
```python
# Optimize simulation performance
sim.optimize_performance(
    physics_substeps=1,      # Physics substeps
    contact_solver="direct",  # Contact solver
    parallel_solver=True     # Parallel solver
)

# Measure performance
performance = sim.get_performance_metrics()
print(f"FPS: {performance['fps']}")
print(f"Physics time: {performance['physics_time_ms']}ms")
print(f"Rendering time: {performance['rendering_time_ms']}ms")

# Adjust for real-time
sim.adjust_for_realtime(
    target_fps=60,
    adaptive=True  # Adaptive timestep
)
```

### Headless Simulation | 无头仿真
```python
# Run simulation without GUI (for batch processing)
headless_sim = PyBulletSimulation(
    gui=False,           # No GUI
    physics_rate=1000,   # Higher physics rate
    egl=True            # Use EGL for headless rendering
)

# Batch simulation
batch_simulator = BatchSimulator(
    num_parallel=4,      # 4 parallel simulations
    simulation_class=PyBulletSimulation
)

results = batch_simulator.run_batch(
    scenarios=scenarios,
    num_runs=100,
    save_results=True
)
```

## Troubleshooting | 故障排除

### Common Issues | 常见问题

#### Simulation Stability | 仿真稳定性
- **Physics Instability**: Reduce timestep, increase solver iterations
- **Numerical Issues**: Check unit consistency, avoid extreme parameters
- **Collision Problems**: Adjust collision margins, use convex decomposition

- **物理不稳定性**: 减少时间步长，增加求解器迭代次数
- **数值问题**: 检查单位一致性，避免极端参数
- **碰撞问题**: 调整碰撞边界，使用凸分解

#### Performance Issues | 性能问题
- **Low FPS**: Simplify models, reduce physics complexity
- **High Memory Usage**: Use simplified collision shapes, limit history
- **Slow Rendering**: Reduce texture quality, disable shadows

- **低FPS**: 简化模型，降低物理复杂度
- **高内存使用**: 使用简化的碰撞形状，限制历史记录
- **渲染缓慢**: 降低纹理质量，禁用阴影

#### Integration Issues | 集成问题
- **API Compatibility**: Ensure simulation and control API versions match
- **Network Communication**: Check ROS 2 network configuration
- **Data Synchronization**: Use synchronized timestamps for sensor data

- **API兼容性**: 确保仿真和控制 API 版本匹配
- **网络通信**: 检查 ROS 2 网络配置
- **数据同步**: 对传感器数据使用同步时间戳

### Debugging Tools | 调试工具
```python
# Enable debug visualization
sim.enable_debug_visualization(
    show_aabb=True,      # Show bounding boxes
    show_wireframe=True, # Show wireframe
    show_contact_points=True  # Show contact points
)

# Add debug markers
sim.add_debug_marker(
    position=[0, 0, 0],
    color=[1, 0, 0, 1],  # RGBA
    size=0.1
)

# Log simulation state
sim.log_state(
    variables=["positions", "velocities", "forces"],
    interval=0.1  # Log every 0.1 seconds
)
```

## Best Practices | 最佳实践

### Simulation Design | 仿真设计
1. **Start Simple**: Begin with simple models and scenarios
2. **Incremental Complexity**: Add complexity gradually
3. **Validation**: Validate simulation against real-world data when possible
4. **Documentation**: Document simulation parameters and assumptions

1. **从简单开始**: 从简单模型和场景开始
2. **渐进式复杂性**: 逐步增加复杂性
3. **验证**: 尽可能根据真实世界数据验证仿真
4. **文档化**: 记录仿真参数和假设

### Performance Best Practices | 性能最佳实践
1. **Model Simplification**: Use simplified collision shapes and visuals
2. **Efficient Physics**: Use appropriate physics settings for the task
3. **Batch Processing**: Use headless simulation for batch processing
4. **Resource Management**: Monitor and manage simulation resources

1. **模型简化**: 使用简化的碰撞形状和视觉效果
2. **高效物理**: 为任务使用适当的物理设置
3. **批处理**: 使用无头仿真进行批处理
4. **资源管理**: 监控和管理仿真资源

### Development Best Practices | 开发最佳实践
1. **Code Reusability**: Write code that works for both simulation and real hardware
2. **Testing**: Test algorithms in simulation before real-world deployment
3. **Version Control**: Version control simulation scenarios and parameters
4. **Collaboration**: Share simulation environments and scenarios

1. **代码可重用性**: 编写适用于仿真和真实硬件的代码
2. **测试**: 在真实世界部署前在仿真中测试算法
3. **版本控制**: 对仿真场景和参数进行版本控制
4. **协作**: 共享仿真环境和场景

## Next Steps | 后续步骤

After setting up simulation:

设置仿真后：

1. **Try Examples**: Run example simulation scripts
2. **Create Scenarios**: Create custom simulation scenarios
3. **Integrate with Training**: Use simulation for training data generation
4. **Deploy to Real Hardware**: Transfer algorithms from simulation to real robots

1. **尝试示例**: 运行示例仿真脚本
2. **创建场景**: 创建自定义仿真场景
3. **与训练集成**: 使用仿真生成训练数据
4. **部署到真实硬件**: 将算法从仿真转移到真实机器人

---

*Last Updated: March 30, 2026*  
*最后更新: 2026年3月30日*