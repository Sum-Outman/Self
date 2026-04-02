# Basic Usage | 基本使用

This guide covers basic usage of the Self AGI system, including AGI chat, training management, hardware control, and knowledge base operations.

本指南涵盖 Self AGI 系统的基本使用，包括 AGI 聊天、训练管理、硬件控制和知识库操作。

## AGI Chat | AGI聊天

### Text Chat | 文本聊天
1. **Navigate to Chat**: Go to the chat page from the main navigation
2. **Enter Message**: Type your message in the input box
3. **Send**: Press Enter or click the send button
4. **View Response**: The AGI's response will appear in the chat window

1. **导航到聊天**: 从主导航进入聊天页面
2. **输入消息**: 在输入框中输入您的消息
3. **发送**: 按 Enter 或点击发送按钮
4. **查看响应**: AGI 的响应将出现在聊天窗口中

### Multimodal Chat | 多模态聊天
1. **Upload Files**: Click the upload button to upload images, audio, or video
2. **Combine with Text**: Add text description along with the files
3. **Send**: The AGI will process both the files and text
4. **View Response**: Response may include text, images, or other media

1. **上传文件**: 点击上传按钮上传图像、音频或视频
2. **结合文本**: 添加文本描述与文件一起
3. **发送**: AGI 将同时处理文件和文本
4. **查看响应**: 响应可能包括文本、图像或其他媒体

### Chat Settings | 聊天设置
- **Model Selection**: Choose different AGI models for different tasks
- **Temperature**: Adjust creativity vs. determinism (0.0-1.0)
- **Max Tokens**: Set maximum response length
- **Context Memory**: Enable/disable context memory for conversation history

- **模型选择**: 为不同任务选择不同的 AGI 模型
- **温度**: 调整创造性 vs. 确定性 (0.0-1.0)
- **最大令牌数**: 设置最大响应长度
- **上下文记忆**: 启用/禁用对话历史的上下文记忆

## Training Management | 训练管理

### Start Training | 开始训练
1. **Navigate to Training**: Go to the training management page
2. **Select Dataset**: Choose a dataset for training
3. **Configure Parameters**: Set training parameters (epochs, batch size, learning rate)
4. **Start Training**: Click the start button to begin training
5. **Monitor Progress**: View training progress and metrics in real-time

1. **导航到训练**: 进入训练管理页面
2. **选择数据集**: 选择用于训练的数据集
3. **配置参数**: 设置训练参数（轮数、批次大小、学习率）
4. **开始训练**: 点击开始按钮开始训练
5. **监控进度**: 实时查看训练进度和指标

### Training Monitoring | 训练监控
- **Loss Curves**: View training and validation loss curves
- **Accuracy Metrics**: Monitor accuracy and other performance metrics
- **Resource Usage**: Check GPU/CPU usage and memory consumption
- **Logs**: View training logs and error messages

- **损失曲线**: 查看训练和验证损失曲线
- **准确率指标**: 监控准确率和其他性能指标
- **资源使用**: 检查 GPU/CPU 使用率和内存消耗
- **日志**: 查看训练日志和错误消息

### Model Management | 模型管理
- **Save Checkpoints**: Training checkpoints are saved automatically
- **Load Models**: Load pre-trained models for inference or further training
- **Compare Models**: Compare performance of different models
- **Export Models**: Export trained models for deployment

- **保存检查点**: 训练检查点自动保存
- **加载模型**: 加载预训练模型进行推理或进一步训练
- **比较模型**: 比较不同模型的性能
- **导出模型**: 导出训练好的模型进行部署

## Hardware Control | 硬件控制

### Robot Control | 机器人控制
1. **Navigate to Hardware**: Go to the hardware control page
2. **Select Robot**: Choose a robot from the available options
3. **Control Interface**: Use the control interface to move the robot
4. **Joint Control**: Control individual joints with sliders or direct input
5. **Cartesian Control**: Control end effector position and orientation
6. **Execute Motion**: Execute pre-programmed motions or trajectories

1. **导航到硬件**: 进入硬件控制页面
2. **选择机器人**: 从可用选项中选择机器人
3. **控制界面**: 使用控制界面移动机器人
4. **关节控制**: 使用滑块或直接输入控制单个关节
5. **笛卡尔控制**: 控制末端执行器位置和方向
6. **执行运动**: 执行预编程的运动或轨迹

### Sensor Monitoring | 传感器监控
- **Live Data**: View real-time sensor data (cameras, IMU, LiDAR, etc.)
- **Data Visualization**: Visualize sensor data with charts and graphs
- **Data Recording**: Record sensor data for later analysis
- **Alerts**: Set up alerts for sensor anomalies or thresholds

- **实时数据**: 查看实时传感器数据（相机、IMU、激光雷达等）
- **数据可视化**: 用图表可视化传感器数据
- **数据记录**: 记录传感器数据供后续分析
- **告警**: 为传感器异常或阈值设置告警

### Simulation Control | 仿真控制
1. **Select Simulation**: Choose PyBullet or Gazebo simulation
2. **Load Environment**: Load a simulation environment
3. **Control Robot**: Control the simulated robot same as real hardware
4. **Physics Settings**: Adjust physics parameters (gravity, friction, etc.)
5. **Scenario Testing**: Test different scenarios and conditions

1. **选择仿真**: 选择 PyBullet 或 Gazebo 仿真
2. **加载环境**: 加载仿真环境
3. **控制机器人**: 控制仿真机器人，与真实硬件相同
4. **物理设置**: 调整物理参数（重力、摩擦等）
5. **场景测试**: 测试不同场景和条件

## Knowledge Base | 知识库

### Add Knowledge | 添加知识
1. **Navigate to Knowledge Base**: Go to the knowledge base page
2. **Add Entry**: Click "Add New" to create a new knowledge entry
3. **Enter Information**: Enter title, content, and metadata
4. **Categorize**: Assign categories and tags for organization
5. **Save**: Save the knowledge entry to the database

1. **导航到知识库**: 进入知识库页面
2. **添加条目**: 点击"添加新条目"创建新知识条目
3. **输入信息**: 输入标题、内容和元数据
4. **分类**: 分配类别和标签进行组织
5. **保存**: 将知识条目保存到数据库

### Search Knowledge | 搜索知识
- **Text Search**: Search by keywords in title or content
- **Semantic Search**: Search by meaning using natural language queries
- **Category Filter**: Filter by categories and tags
- **Advanced Search**: Combine multiple search criteria

- **文本搜索**: 按标题或内容中的关键词搜索
- **语义搜索**: 使用自然语言查询按含义搜索
- **类别过滤**: 按类别和标签过滤
- **高级搜索**: 组合多个搜索条件

### Knowledge Graph | 知识图谱
- **Visualization**: View knowledge as an interactive graph
- **Relationships**: Explore relationships between knowledge entries
- **Navigation**: Navigate through connected knowledge
- **Analysis**: Analyze knowledge structure and connections

- **可视化**: 将知识视为交互式图谱查看
- **关系**: 探索知识条目之间的关系
- **导航**: 通过连接的知识导航
- **分析**: 分析知识结构和连接

## System Monitoring | 系统监控

### Dashboard | 仪表板
- **System Health**: View overall system health status
- **Resource Usage**: Monitor CPU, memory, GPU, and disk usage
- **Service Status**: Check status of all system services
- **Performance Metrics**: View key performance indicators

- **系统健康**: 查看整体系统健康状态
- **资源使用**: 监控 CPU、内存、GPU 和磁盘使用率
- **服务状态**: 检查所有系统服务的状态
- **性能指标**: 查看关键性能指标

### Logs | 日志
- **View Logs**: Access system logs and error messages
- **Filter Logs**: Filter logs by level, component, or time
- **Search Logs**: Search for specific log entries
- **Export Logs**: Export logs for analysis or debugging

- **查看日志**: 访问系统日志和错误消息
- **过滤日志**: 按级别、组件或时间过滤日志
- **搜索日志**: 搜索特定日志条目
- **导出日志**: 导出日志进行分析或调试

### Alerts | 告警
- **Alert Configuration**: Configure alert rules and thresholds
- **Alert History**: View history of past alerts
- **Alert Actions**: Define actions for different alerts (email, notification, etc.)
- **Alert Testing**: Test alert configurations

- **告警配置**: 配置告警规则和阈值
- **告警历史**: 查看过去告警的历史
- **告警操作**: 为不同告警定义操作（邮件、通知等）
- **告警测试**: 测试告警配置

## User Management | 用户管理

### User Accounts | 用户账户
- **Create Account**: Create new user accounts
- **Edit Profile**: Edit user profile information
- **Change Password**: Change account password
- **Account Settings**: Configure account preferences and settings

- **创建账户**: 创建新用户账户
- **编辑资料**: 编辑用户资料信息
- **更改密码**: 更改账户密码
- **账户设置**: 配置账户首选项和设置

### Permissions | 权限
- **Role Management**: Manage user roles and permissions
- **Access Control**: Control access to different system features
- **Permission Auditing**: Audit user permissions and access logs
- **Security Settings**: Configure security settings and policies

- **角色管理**: 管理用户角色和权限
- **访问控制**: 控制系统不同功能的访问
- **权限审计**: 审计用户权限和访问日志
- **安全设置**: 配置安全设置和策略

## API Usage | API使用

### Access API | 访问API
1. **Get API Key**: Obtain API key from user settings
2. **API Documentation**: View API documentation at http://localhost:8000/docs
3. **Make Requests**: Use API key to make authenticated requests
4. **Handle Responses**: Process API responses in your application

1. **获取API密钥**: 从用户设置获取 API 密钥
2. **API文档**: 在 http://localhost:8000/docs 查看 API 文档
3. **发起请求**: 使用 API 密钥发起认证请求
4. **处理响应**: 在应用程序中处理 API 响应

### Example API Calls | 示例API调用

#### Chat API | 聊天API
```python
import requests

api_key = "your_api_key"
url = "http://localhost:8000/api/chat/send"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "message": "Hello, how are you?",
    "model_name": "self_agi_v1",
    "use_memory": True
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

#### Training API | 训练API
```python
import requests

api_key = "your_api_key"
url = "http://localhost:8000/api/training/start"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "dataset": "multimodal_dataset",
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

## Tips and Best Practices | 提示和最佳实践

### Performance Tips | 性能提示
- **Batch Processing**: Process multiple items together when possible
- **Caching**: Use caching for frequently accessed data
- **Resource Management**: Monitor and manage system resources
- **Optimization**: Optimize queries and operations for efficiency

- **批处理**: 尽可能一起处理多个项目
- **缓存**: 对频繁访问的数据使用缓存
- **资源管理**: 监控和管理系统资源
- **优化**: 优化查询和操作以提高效率

### Security Tips | 安全提示
- **API Security**: Keep API keys secure and rotate regularly
- **Authentication**: Use strong authentication methods
- **Data Protection**: Protect sensitive data with encryption
- **Access Control**: Implement proper access control mechanisms

- **API安全**: 保持 API 密钥安全并定期轮换
- **认证**: 使用强认证方法
- **数据保护**: 使用加密保护敏感数据
- **访问控制**: 实现适当的访问控制机制

### Usage Tips | 使用提示
- **Start Simple**: Start with basic features before trying advanced ones
- **Read Documentation**: Read documentation for detailed instructions
- **Experiment**: Experiment with different settings and configurations
- **Backup**: Regularly backup important data and configurations

- **从简单开始**: 在尝试高级功能之前从基本功能开始
- **阅读文档**: 阅读文档获取详细说明
- **实验**: 尝试不同的设置和配置
- **备份**: 定期备份重要数据和配置

## Getting Help | 获取帮助

If you need help:

如果需要帮助：

1. **Check Documentation**: Check this documentation and other documentation pages
2. **Search Issues**: Search existing issues on GitHub
3. **Ask Questions**: Ask questions in the GitHub discussions
4. **Contact Support**: Email silencecrowtom@qq.com for support

1. **检查文档**: 检查本文档和其他文档页面
2. **搜索问题**: 在 GitHub 上搜索现有问题
3. **提问**: 在 GitHub 讨论中提问
4. **联系支持**: 发送邮件至 silencecrowtom@qq.com 获取支持

---

*Last Updated: March 30, 2026*  
*最后更新: 2026年3月30日*