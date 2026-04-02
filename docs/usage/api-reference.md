# API Reference | API参考

This document provides complete API reference for the Self AGI system, including authentication, chat, training, hardware control, and system management APIs.

本文档提供 Self AGI 系统的完整 API 参考，包括认证、聊天、训练、硬件控制和系统管理 API。

## API Overview | API概述

### Base URL | 基础URL
```
http://localhost:8000/api
```
For production: `https://api.yourdomain.com/api`

生产环境: `https://api.yourdomain.com/api`

### Authentication | 认证
All API requests require authentication using JWT tokens. Include the token in the Authorization header:

所有 API 请求都需要使用 JWT 令牌进行认证。在 Authorization 头中包含令牌：

```http
Authorization: Bearer <your_jwt_token>
```

### Response Format | 响应格式
All API responses return JSON with the following structure:

所有 API 响应返回 JSON，结构如下：

```json
{
  "status": "success",  // or "error"
  "data": { ... },      // response data (optional)
  "message": "Operation completed successfully",  // status message
  "timestamp": "2026-03-30T10:30:00Z"  // timestamp
}
```

### Error Responses | 错误响应
Error responses include error details:

错误响应包含错误详情：

```json
{
  "status": "error",
  "error": {
    "code": "invalid_request",
    "message": "Invalid request parameters",
    "details": { ... }
  },
  "timestamp": "2026-03-30T10:30:00Z"
}
```

## Authentication API | 认证API

### Login | 登录
```http
POST /api/auth/login
```

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "password123",
  "remember_me": false
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 3600,
    "user": {
      "id": "user_123",
      "email": "user@example.com",
      "name": "John Doe",
      "role": "user",
      "permissions": ["chat", "training", "hardware_read"]
    }
  }
}
```

### Register | 注册
```http
POST /api/auth/register
```

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "password123",
  "name": "John Doe",
  "agree_to_terms": true
}
```

**Response**: Similar to login response.

### Refresh Token | 刷新令牌
```http
POST /api/auth/refresh
```

**Request Body**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response**: New access token.

### Logout | 登出
```http
POST /api/auth/logout
```

**Headers**: `Authorization: Bearer <token>`

**Response**:
```json
{
  "status": "success",
  "message": "Successfully logged out"
}
```

## Chat API | 聊天API

### Send Message | 发送消息
```http
POST /api/chat/send
```

**Request Body**:
```json
{
  "message": "Hello, how are you?",
  "model_name": "self_agi_v1",
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  },
  "context": {
    "use_memory": true,
    "memory_window": 10,
    "knowledge_base_search": true
  },
  "stream": false
}
```

**Response (non-streaming)**:
```json
{
  "status": "success",
  "data": {
    "response": "I'm doing well, thank you for asking! How can I assist you today?",
    "model": "self_agi_v1",
    "tokens_used": 15,
    "processing_time": 0.235,
    "confidence": 0.92
  }
}
```

**Streaming Response**: When `stream: true`, returns Server-Sent Events (SSE).

### Upload Files | 上传文件
```http
POST /api/chat/upload
```

**Content-Type**: `multipart/form-data`

**Form Data**:
- `message`: (optional) Text message
- `image`: (optional) Image file
- `audio`: (optional) Audio file
- `video`: (optional) Video file
- `document`: (optional) Document file

**Response**:
```json
{
  "status": "success",
  "data": {
    "response": "I see you uploaded an image of a cat. It looks very cute!",
    "files_processed": {
      "image": true,
      "audio": false,
      "video": false,
      "document": false
    }
  }
}
```

### Get Chat History | 获取聊天历史
```http
GET /api/chat/history
```

**Query Parameters**:
- `limit`: Number of messages to retrieve (default: 50)
- `offset`: Offset for pagination (default: 0)
- `conversation_id`: Specific conversation ID (optional)
- `start_date`: Start date filter (optional)
- `end_date`: End date filter (optional)

**Response**:
```json
{
  "status": "success",
  "data": {
    "conversations": [
      {
        "id": "conv_123",
        "title": "Discussion about AI",
        "created_at": "2026-03-30T10:30:00Z",
        "updated_at": "2026-03-30T10:45:00Z",
        "message_count": 10,
        "last_message": "What do you think about AGI?"
      }
    ],
    "total": 5,
    "limit": 50,
    "offset": 0
  }
}
```

### Delete Conversation | 删除对话
```http
DELETE /api/chat/conversation/{conversation_id}
```

**Response**:
```json
{
  "status": "success",
  "message": "Conversation deleted successfully"
}
```

## Training API | 训练API

### Start Training | 开始训练
```http
POST /api/training/start
```

**Request Body**:
```json
{
  "task_id": "train_agi_model",
  "dataset": "multimodal_dataset_v2",
  "model_config": {
    "model_name": "self_agi_v2",
    "hidden_size": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 10
  },
  "training_config": {
    "distributed": false,
    "num_gpus": 1,
    "mixed_precision": true,
    "checkpoint_frequency": 1000,
    "validation_frequency": 500
  },
  "notification": {
    "email": "user@example.com",
    "on_complete": true,
    "on_error": true
  }
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "training_id": "train_123456",
    "status": "pending",
    "estimated_duration": "24h",
    "checkpoints_url": "/api/training/checkpoints/train_123456",
    "logs_url": "/api/training/logs/train_123456"
  }
}
```

### Get Training Status | 获取训练状态
```http
GET /api/training/status/{training_id}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "training_id": "train_123456",
    "status": "running",
    "progress": 0.45,
    "current_epoch": 5,
    "total_epochs": 10,
    "metrics": {
      "loss": 0.1234,
      "accuracy": 0.8765,
      "perplexity": 15.32
    },
    "start_time": "2026-03-30T10:30:00Z",
    "estimated_completion": "2026-03-31T10:30:00Z",
    "resources": {
      "gpu_usage": 0.85,
      "memory_usage": 0.72,
      "cpu_usage": 0.45
    }
  }
}
```

### Stop Training | 停止训练
```http
POST /api/training/stop/{training_id}
```

**Response**:
```json
{
  "status": "success",
  "message": "Training stopped successfully",
  "data": {
    "training_id": "train_123456",
    "final_status": "stopped",
    "checkpoint_saved": true,
    "checkpoint_path": "/checkpoints/train_123456_epoch5.pt"
  }
}
```

### List Training Jobs | 列出训练任务
```http
GET /api/training/jobs
```

**Query Parameters**:
- `status`: Filter by status (pending, running, completed, failed)
- `limit`: Number of jobs to retrieve (default: 20)
- `offset`: Offset for pagination (default: 0)

**Response**:
```json
{
  "status": "success",
  "data": {
    "jobs": [
      {
        "id": "train_123456",
        "task": "train_agi_model",
        "status": "completed",
        "progress": 1.0,
        "start_time": "2026-03-30T10:30:00Z",
        "end_time": "2026-03-31T10:30:00Z",
        "metrics": {
          "final_loss": 0.0567,
          "final_accuracy": 0.9345
        }
      }
    ],
    "total": 15,
    "limit": 20,
    "offset": 0
  }
}
```

## Hardware API | 硬件API

### Get Hardware Status | 获取硬件状态
```http
GET /api/hardware/status
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "robots": [
      {
        "id": "robot_1",
        "name": "UR5e Arm",
        "type": "robotic_arm",
        "status": "connected",
        "connection": {
          "type": "tcp",
          "host": "192.168.1.50",
          "port": 30003,
          "latency": 12
        },
        "state": {
          "joint_positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
          "tcp_position": [0.5, 0.2, 0.3, 0, 0, 0],
          "force_torque": [0.1, 0.2, 5.3, 0.01, 0.02, 0.03]
        }
      }
    ],
    "sensors": [
      {
        "id": "camera_1",
        "name": "Front Camera",
        "type": "camera",
        "status": "streaming",
        "resolution": [1280, 720],
        "fps": 30
      }
    ],
    "overall_status": "healthy"
  }
}
```

### Control Robot | 控制机器人
```http
POST /api/hardware/robot/{robot_id}/control
```

**Request Body**:
```json
{
  "command": "move_to_position",
  "parameters": {
    "position": [0.5, 0.2, 0.3, 0, 0, 0],
    "velocity": 0.2,
    "acceleration": 0.1,
    "blend_radius": 0.05
  },
  "safety": {
    "collision_detection": true,
    "force_limit": 50.0,
    "timeout": 30.0
  }
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "command_id": "cmd_123456",
    "status": "executing",
    "estimated_duration": 2.5,
    "progress_url": "/api/hardware/command/cmd_123456"
  }
}
```

### Get Sensor Data | 获取传感器数据
```http
GET /api/hardware/sensor/{sensor_id}/data
```

**Query Parameters**:
- `type`: Data type (image, point_cloud, imu, etc.)
- `format`: Response format (json, binary, base64)
- `latest`: Get latest data only (default: true)

**Response (JSON)**:
```json
{
  "status": "success",
  "data": {
    "sensor_id": "camera_1",
    "timestamp": "2026-03-30T10:30:00Z",
    "data_type": "image",
    "data": {
      "format": "jpeg",
      "width": 1280,
      "height": 720,
      "base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAUABQDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
    }
  }
}
```

### Execute Task | 执行任务
```http
POST /api/hardware/task
```

**Request Body**:
```json
{
  "task_type": "pick_and_place",
  "parameters": {
    "pick_position": [0.3, 0.1, 0.1, 0, 0, 0],
    "place_position": [0.3, -0.1, 0.1, 0, 0, 0],
    "object_size": [0.05, 0.05, 0.05],
    "gripper_force": 20.0
  },
  "robots": ["robot_1"],
  "sensors": ["camera_1", "force_torque_1"],
  "monitoring": {
    "force_threshold": 30.0,
    "collision_detection": true,
    "timeout": 60.0
  }
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "task_id": "task_123456",
    "status": "planned",
    "estimated_duration": 15.0,
    "task_url": "/api/hardware/task/task_123456"
  }
}
```

## Knowledge Base API | 知识库API

### Search Knowledge | 搜索知识
```http
GET /api/knowledge/search
```

**Query Parameters**:
- `query`: Search query (required)
- `type`: Search type (text, semantic, hybrid) (default: hybrid)
- `limit`: Number of results (default: 10)
- `offset`: Offset for pagination (default: 0)
- `categories`: Filter by categories (comma-separated)
- `tags`: Filter by tags (comma-separated)

**Response**:
```json
{
  "status": "success",
  "data": {
    "results": [
      {
        "id": "kb_123",
        "title": "AGI Architecture Overview",
        "content": "The Self AGI system uses a four-layer architecture...",
        "categories": ["architecture", "agi"],
        "tags": ["transformer", "multimodal", "memory"],
        "relevance": 0.92,
        "created_at": "2026-03-30T10:30:00Z",
        "updated_at": "2026-03-30T10:30:00Z"
      }
    ],
    "total": 45,
    "limit": 10,
    "offset": 0
  }
}
```

### Add Knowledge | 添加知识
```http
POST /api/knowledge/add
```

**Request Body**:
```json
{
  "title": "New AGI Concept",
  "content": "This is a new concept about AGI development...",
  "categories": ["concepts", "research"],
  "tags": ["new", "experimental"],
  "metadata": {
    "source": "research_paper",
    "author": "John Doe",
    "confidence": 0.85
  }
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "id": "kb_456",
    "created_at": "2026-03-30T10:30:00Z",
    "vector_id": "vec_789"
  }
}
```

### Update Knowledge | 更新知识
```http
PUT /api/knowledge/{knowledge_id}
```

**Request Body**: Similar to add knowledge.

**Response**: Updated knowledge entry.

### Delete Knowledge | 删除知识
```http
DELETE /api/knowledge/{knowledge_id}
```

**Response**:
```json
{
  "status": "success",
  "message": "Knowledge entry deleted successfully"
}
```

## System API | 系统API

### Health Check | 健康检查
```http
GET /api/health
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "overall": "healthy",
    "services": {
      "database": {
        "status": "healthy",
        "latency": 12
      },
      "redis": {
        "status": "healthy",
        "latency": 2
      },
      "model_server": {
        "status": "healthy",
        "load": 0.45
      },
      "hardware": {
        "status": "healthy",
        "connected_robots": 2
      }
    },
    "resources": {
      "cpu": 0.35,
      "memory": 0.62,
      "disk": 0.28,
      "gpu": 0.15
    },
    "timestamp": "2026-03-30T10:30:00Z"
  }
}
```

### Get System Metrics | 获取系统指标
```http
GET /api/system/metrics
```

**Query Parameters**:
- `period`: Time period (1h, 24h, 7d, 30d) (default: 1h)
- `resolution`: Data resolution (1m, 5m, 1h, 1d) (default: based on period)

**Response**:
```json
{
  "status": "success",
  "data": {
    "cpu_usage": [
      {"timestamp": "2026-03-30T10:00:00Z", "value": 0.35},
      {"timestamp": "2026-03-30T10:01:00Z", "value": 0.38}
    ],
    "memory_usage": [...],
    "api_requests": [...],
    "model_inferences": [...],
    "hardware_operations": [...]
  }
}
```

### Get Logs | 获取日志
```http
GET /api/system/logs
```

**Query Parameters**:
- `level`: Log level (debug, info, warning, error, critical)
- `component`: Component name (api, model, hardware, etc.)
- `start_time`: Start time filter
- `end_time`: End time filter
- `limit`: Number of logs (default: 100)
- `search`: Search text in logs

**Response**:
```json
{
  "status": "success",
  "data": {
    "logs": [
      {
        "timestamp": "2026-03-30T10:30:00Z",
        "level": "info",
        "component": "api",
        "message": "API request processed",
        "details": {
          "endpoint": "/api/chat/send",
          "duration": 0.235,
          "user_id": "user_123"
        }
      }
    ],
    "total": 1250,
    "limit": 100
  }
}
```

## User Management API | 用户管理API

### Get User Profile | 获取用户资料
```http
GET /api/user/profile
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "id": "user_123",
    "email": "user@example.com",
    "name": "John Doe",
    "role": "user",
    "permissions": ["chat", "training", "hardware_read"],
    "created_at": "2026-01-15T10:30:00Z",
    "last_login": "2026-03-30T10:25:00Z",
    "usage_stats": {
      "total_chats": 125,
      "total_training_jobs": 8,
      "total_hardware_operations": 45
    }
  }
}
```

### Update User Profile | 更新用户资料
```http
PUT /api/user/profile
```

**Request Body**:
```json
{
  "name": "John Updated",
  "email": "newemail@example.com"
}
```

**Response**: Updated profile.

### Change Password | 更改密码
```http
POST /api/user/change-password
```

**Request Body**:
```json
{
  "current_password": "oldpassword123",
  "new_password": "newpassword456"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Password changed successfully"
}
```

### Get API Keys | 获取API密钥
```http
GET /api/user/api-keys
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "keys": [
      {
        "id": "key_123",
        "name": "Production Key",
        "key": "sk_*****789",  // masked
        "created_at": "2026-01-15T10:30:00Z",
        "last_used": "2026-03-30T10:25:00Z",
        "permissions": ["chat", "training_read"],
        "rate_limit": 100
      }
    ]
  }
}
```

### Create API Key | 创建API密钥
```http
POST /api/user/api-keys
```

**Request Body**:
```json
{
  "name": "New Integration",
  "permissions": ["chat", "knowledge_read"],
  "rate_limit": 50,
  "expires_at": "2026-12-31T23:59:59Z"  // optional
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "id": "key_456",
    "name": "New Integration",
    "key": "sk_live_abcdef1234567890",  // full key only returned once
    "created_at": "2026-03-30T10:30:00Z",
    "permissions": ["chat", "knowledge_read"],
    "rate_limit": 50
  }
}
```

## Error Codes | 错误码

### Common Error Codes | 常见错误码
- `invalid_request`: Invalid request parameters
- `authentication_required`: Authentication required
- `invalid_credentials`: Invalid email or password
- `insufficient_permissions`: User lacks required permissions
- `resource_not_found`: Requested resource not found
- `rate_limit_exceeded`: Rate limit exceeded
- `service_unavailable`: Service temporarily unavailable
- `internal_error`: Internal server error

- `invalid_request`: 无效的请求参数
- `authentication_required`: 需要认证
- `invalid_credentials`: 无效的邮箱或密码
- `insufficient_permissions`: 用户缺乏所需权限
- `resource_not_found`: 请求的资源未找到
- `rate_limit_exceeded`: 超出速率限制
- `service_unavailable`: 服务暂时不可用
- `internal_error`: 内部服务器错误

### Error Response Example | 错误响应示例
```json
{
  "status": "error",
  "error": {
    "code": "insufficient_permissions",
    "message": "You do not have permission to perform this action",
    "details": {
      "required_permission": "hardware_write",
      "user_permissions": ["chat", "training_read", "hardware_read"]
    }
  },
  "timestamp": "2026-03-30T10:30:00Z"
}
```

## Rate Limiting | 速率限制

### Default Limits | 默认限制
- **Authenticated Users**: 100 requests per minute
- **API Keys**: Varies by key configuration
- **Unauthenticated**: 10 requests per minute

- **认证用户**: 每分钟100个请求
- **API密钥**: 根据密钥配置变化
- **未认证**: 每分钟10个请求

### Rate Limit Headers | 速率限制头
Responses include rate limit headers:

响应包含速率限制头：

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1617032400
```

## WebSocket API | WebSocket API

### Connection | 连接
```
ws://localhost:8000/api/ws
```

**Authentication**: Send JWT token after connection:

连接后发送 JWT 令牌：

```json
{
  "type": "auth",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Events | 事件

#### Chat Stream | 聊天流
```json
{
  "type": "chat_stream",
  "data": {
    "conversation_id": "conv_123",
    "message_id": "msg_456",
    "chunk": "This is a streaming",
    "is_complete": false
  }
}
```

#### Training Updates | 训练更新
```json
{
  "type": "training_update",
  "data": {
    "training_id": "train_123456",
    "epoch": 5,
    "loss": 0.1234,
    "accuracy": 0.8765
  }
}
```

#### Hardware Events | 硬件事件
```json
{
  "type": "hardware_event",
  "data": {
    "robot_id": "robot_1",
    "event": "movement_completed",
    "position": [0.5, 0.2, 0.3, 0, 0, 0]
  }
}
```

#### System Alerts | 系统告警
```json
{
  "type": "system_alert",
  "data": {
    "level": "warning",
    "message": "High CPU usage detected",
    "component": "system",
    "timestamp": "2026-03-30T10:30:00Z"
  }
}
```

## SDKs and Libraries | SDK和库

### Python SDK | Python SDK
```python
from self_agi_sdk import SelfAGIClient

client = SelfAGIClient(
    api_key="your_api_key",
    base_url="http://localhost:8000"
)

# Send chat message
response = client.chat.send(
    message="Hello, how are you?",
    model_name="self_agi_v1"
)

print(response.response)
```

### JavaScript SDK | JavaScript SDK
```javascript
import { SelfAGIClient } from 'self-agi-sdk';

const client = new SelfAGIClient({
  apiKey: 'your_api_key',
  baseUrl: 'http://localhost:8000'
});

// Send chat message
const response = await client.chat.send({
  message: 'Hello, how are you?',
  modelName: 'self_agi_v1'
});

console.log(response.data.response);
```

## Next Steps | 后续步骤

After exploring the API:

探索 API 后：

1. **Try Examples**: Try the API examples in the documentation
2. **Generate API Key**: Generate an API key for your application
3. **Integrate**: Integrate with your application using SDKs
4. **Monitor Usage**: Monitor your API usage and adjust limits as needed

1. **尝试示例**: 尝试文档中的 API 示例
2. **生成API密钥**: 为您的应用程序生成 API 密钥
3. **集成**: 使用 SDK 与您的应用程序集成
4. **监控使用**: 监控您的 API 使用情况并根据需要调整限制

---

*Last Updated: March 30, 2026*  
*最后更新: 2026年3月30日*