# 前端错误处理指南

## 概述

本指南介绍Self AGI系统的前端错误处理机制，包括全局错误处理、API错误处理、用户反馈和错误监控。

## 错误处理架构

### 1. 全局错误处理器 (`errorHandler.ts`)

全局错误处理器提供统一的错误处理机制，包括：
- 错误类型识别和分类
- 用户友好的错误消息
- Toast通知显示
- 错误监控记录
- 特殊错误处理（如认证失效、网络错误等）

### 2. API客户端错误处理 (`client.ts`)

API客户端内置错误处理，自动：
- 处理401错误并尝试刷新token
- 捕获所有API错误并传递给全局错误处理器
- 提供标准化的错误消息

### 3. 错误边界组件 (`ErrorBoundary.tsx`)

React错误边界组件，用于：
- 捕获组件树中的JavaScript错误
- 显示降级UI
- 记录错误信息

### 4. 错误监控系统 (`errorMonitoring.ts`)

错误监控系统提供：
- 前端错误收集和报告
- 性能指标监控
- 用户行为追踪（可选）

## 使用方法

### 基本错误处理

#### 1. 使用全局错误处理器

```typescript
import { handleError, logSuccess, logWarning } from '../utils/errorHandler';

// 处理错误（自动显示toast通知）
try {
  await api.someRequest();
} catch (error) {
  handleError(error, '操作名称');
}

// 记录成功操作
logSuccess('操作成功完成', '操作名称');

// 记录警告
logWarning('需要注意的问题', '操作名称');
```

#### 2. 使用React Hook

```typescript
import { useErrorHandler } from '../utils/errorHandler';

const MyComponent = () => {
  const { handleError, logSuccess, logWarning } = useErrorHandler();
  
  const handleAction = async () => {
    try {
      await api.someRequest();
      logSuccess('操作成功', '组件名称');
    } catch (error) {
      handleError(error, '组件名称');
    }
  };
  
  return <button onClick={handleAction}>执行操作</button>;
};
```

### API错误处理

#### 1. 标准API调用

```typescript
import api from '../services/api/client';

// API客户端会自动处理错误并显示toast通知
const response = await api.get('/some/endpoint');
// 如果发生错误，会自动处理并抛出

// 或者手动处理
try {
  const response = await api.get('/some/endpoint');
} catch (error) {
  // API客户端已经处理了错误，这里可以添加额外的逻辑
  console.log('API调用失败，错误已自动处理');
}
```

#### 2. 自定义API调用

```typescript
import { errorHandler } from '../utils/errorHandler';

try {
  const response = await fetch('/api/endpoint');
  if (!response.ok) {
    // 创建错误对象
    const error = new Error(`HTTP错误: ${response.status}`);
    (error as any).status = response.status;
    (error as any).response = response;
    
    // 使用错误处理器
    errorHandler.handleError(error, '自定义API调用');
    throw error;
  }
  
  const data = await response.json();
  return data;
} catch (error) {
  // 如果errorHandler.handleError没有被调用，手动调用
  if (!(error as any)._handled) {
    errorHandler.handleError(error, '自定义API调用');
  }
  throw error;
}
```

### 错误边界使用

#### 1. 包装页面组件

```typescript
import { ErrorBoundary } from '../components/ErrorBoundary';

const MyPage = () => {
  return (
    <ErrorBoundary 
      errorMessage="页面加载失败"
      onError={(error, errorInfo) => {
        // 自定义错误处理逻辑
        console.error('页面错误:', error);
      }}
    >
      {/* 页面内容 */}
    </ErrorBoundary>
  );
};
```

#### 2. 使用高阶组件

```typescript
import { withErrorBoundary } from '../components/ErrorBoundary';

const MyComponent = () => {
  // 组件逻辑
};

export default withErrorBoundary(MyComponent, {
  errorMessage: '组件加载失败',
  logError: true,
});
```

#### 3. 全局错误边界

```typescript
// 在App.tsx中已经使用了全局错误边界
// 它会捕获整个应用中的未处理错误
```

## 错误类型和消息

### HTTP状态码错误

| 状态码 | 错误类型 | 用户消息 | 处理方式 |
|--------|----------|----------|----------|
| 400 | 请求错误 | 请求格式错误，请检查输入 | 显示toast通知 |
| 401 | 认证失败 | 认证失败，请重新登录 | 显示toast通知，3秒后重定向到登录页 |
| 403 | 权限不足 | 权限不足，无法访问此资源 | 显示toast通知 |
| 404 | 资源不存在 | 请求的资源不存在 | 显示toast通知 |
| 429 | 频率限制 | 请求过于频繁，请稍后重试 | 显示toast通知，持续时间较长 |
| 500 | 服务器错误 | 服务器内部错误，请联系技术支持 | 显示toast通知 |
| 502/503/504 | 服务器不可用 | 服务器暂时不可用，请稍后重试 | 显示toast通知，持续时间较长 |

### 网络错误

| 错误类型 | 用户消息 | 处理方式 |
|----------|----------|----------|
| 网络连接失败 | 网络连接失败，请检查网络连接 | 显示toast通知，触发网络检查事件 |
| 请求超时 | 请求超时，请检查网络连接并重试 | 显示toast通知 |
| CORS错误 | 跨域请求被阻止，请检查服务器配置 | 显示toast通知（开发环境） |
| 连接被拒绝 | 连接被拒绝，请检查服务器状态 | 显示toast通知 |

### 业务错误

| 错误类型 | 用户消息 | 处理方式 |
|----------|----------|----------|
| 验证错误 | 输入验证失败，请检查输入 | 显示toast通知 |
| 资源不存在 | 请求的资源不存在 | 显示toast通知 |
| 未授权访问 | 未授权访问，请登录 | 显示toast通知，3秒后重定向 |
| 权限不足 | 权限不足，无法执行此操作 | 显示toast通知 |
| 频率限制 | 请求过于频繁，请稍后重试 | 显示toast通知，持续时间较长 |
| 服务不可用 | 服务暂时不可用，请稍后重试 | 显示toast通知，持续时间较长 |

## 最佳实践

### 1. 始终使用try-catch处理异步操作

```typescript
// 正确
try {
  const data = await api.getData();
  // 处理数据
} catch (error) {
  handleError(error, '获取数据');
}

// 错误（没有错误处理）
const data = await api.getData(); // 如果出错，用户看不到反馈
```

### 2. 提供有意义的上下文信息

```typescript
// 正确
handleError(error, '保存用户设置');

// 错误
handleError(error); // 没有上下文，难以调试
```

### 3. 在关键操作后记录成功

```typescript
try {
  await api.saveData(data);
  logSuccess('数据保存成功', '保存数据');
} catch (error) {
  handleError(error, '保存数据');
}
```

### 4. 使用适当的错误边界粒度

```typescript
// 页面级别错误边界
<ErrorBoundary errorMessage="页面加载失败">
  <MyPage />
</ErrorBoundary>

// 关键组件级别错误边界
<ErrorBoundary errorMessage="图表加载失败">
  <ComplexChartComponent />
</ErrorBoundary>
```

### 5. 处理特殊错误情况

```typescript
try {
  await api.sensitiveOperation();
} catch (error) {
  const errorType = errorHandler.determineErrorType(error);
  
  if (errorType === '401' || errorType === 'UNAUTHORIZED') {
    // 认证失效的特殊处理
    localStorage.clear();
    window.location.href = '/login';
  } else if (errorType === 'NETWORK_ERROR') {
    // 网络错误的特殊处理
    showNetworkWarning();
  } else {
    handleError(error, '敏感操作');
  }
}
```

## 配置和定制

### 1. 配置错误处理器

```typescript
import { errorHandler } from '../utils/errorHandler';

// 更新配置
errorHandler.updateConfig({
  showToast: process.env.NODE_ENV === 'production', // 生产环境显示toast
  consoleLog: process.env.NODE_ENV === 'development', // 开发环境输出到控制台
  defaultErrorMessage: '操作失败，请联系管理员',
});
```

### 2. 添加自定义错误消息

```typescript
errorHandler.updateConfig({
  errorMessages: {
    ...errorHandler.getConfig().errorMessages,
    'CUSTOM_ERROR': '自定义错误消息',
    'BUSINESS_LOGIC_ERROR': '业务逻辑错误，请检查输入',
  },
});
```

### 3. 禁用特定错误类型的toast通知

```typescript
// 通过覆盖shouldSuppressToast方法
class CustomErrorHandler extends ErrorHandler {
  private shouldSuppressToast(errorType: string): boolean {
    const suppressTypes = ['ABORTED', 'NOT_SUPPORTED', 'CUSTOM_NO_TOAST'];
    return suppressTypes.includes(errorType) || super.shouldSuppressToast(errorType);
  }
}
```

## 测试错误处理

### 1. 测试API错误处理

```typescript
// 模拟API错误
test('应该正确处理API错误', async () => {
  // 模拟网络错误
  jest.spyOn(api, 'get').mockRejectedValue(new Error('网络错误'));
  
  // 执行操作
  await myComponent.performAction();
  
  // 验证错误被处理
  expect(errorHandler.handleError).toHaveBeenCalled();
  expect(toast.error).toHaveBeenCalled();
});
```

### 2. 测试错误边界

```typescript
// 模拟组件错误
test('错误边界应该捕获组件错误', () => {
  const ErrorComponent = () => {
    throw new Error('测试错误');
  };
  
  const { getByText } = render(
    <ErrorBoundary errorMessage="组件错误">
      <ErrorComponent />
    </ErrorBoundary>
  );
  
  expect(getByText('组件错误')).toBeInTheDocument();
});
```

### 3. 测试错误监控

```typescript
// 验证错误被记录到监控系统
test('错误应该被记录到监控系统', () => {
  const error = new Error('测试错误');
  
  errorHandler.handleError(error, '测试');
  
  expect(errorMonitor.captureError).toHaveBeenCalledWith(
    error,
    '测试',
    expect.objectContaining({
      errorType: 'UNKNOWN_ERROR',
    })
  );
});
```

## 故障排除

### 常见问题

#### 1. Toast通知没有显示
- 检查错误处理器配置中的`showToast`设置
- 检查错误类型是否在抑制列表中
- 检查是否在其他地方调用了`toast.dismiss()`

#### 2. 错误没有被记录到监控系统
- 检查错误处理器配置中的`logToMonitor`设置
- 检查监控系统是否已初始化
- 检查网络连接是否正常

#### 3. 认证失效没有重定向
- 检查是否为401错误
- 检查localStorage中是否有有效的refresh_token
- 检查API客户端的token刷新逻辑

#### 4. 错误消息不够友好
- 添加自定义错误消息到配置中
- 检查后端返回的错误消息格式
- 确保错误类型识别正确

### 调试技巧

#### 1. 启用详细日志
```typescript
errorHandler.updateConfig({
  consoleLog: true, // 输出详细错误信息到控制台
});
```

#### 2. 检查错误类型
```typescript
try {
  await api.operation();
} catch (error) {
  const errorType = errorHandler.determineErrorType(error);
  console.log('错误类型:', errorType);
  handleError(error, '操作');
}
```

#### 3. 查看完整错误信息
```typescript
try {
  await api.operation();
} catch (error) {
  console.log('完整错误信息:', error);
  console.log('响应数据:', error.response?.data);
  console.log('状态码:', error.response?.status);
  handleError(error, '操作');
}
```

## 更新日志

### v1.0.0 (2026-03-29)
- 初始版本
- 全局错误处理器
- API客户端错误处理集成
- 错误边界组件
- 错误监控系统
- 完整的使用指南

## 支持

如有问题或建议，请：
1. 查看本文档的故障排除部分
2. 检查代码示例和最佳实践
3. 联系前端开发团队：silencecrowtom@qq.com