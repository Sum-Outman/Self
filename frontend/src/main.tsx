import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/globals.css';

// 导入必要的polyfill
import 'regenerator-runtime/runtime';

// 导入并初始化错误监控
import { initializeErrorMonitoring } from './utils/errorMonitoring';

// 初始化错误监控（仅在生产环境或开发环境启用）
initializeErrorMonitoring({
  enabled: process.env.NODE_ENV === 'production' || true, // 开发环境也启用以便测试
  sampleRate: process.env.NODE_ENV === 'production' ? 0.5 : 1.0,
  capturePerformance: true,
  captureUserActions: false,
});

// 在开发环境显示监控状态
if (process.env.NODE_ENV === 'development') {
  console.log('前端错误监控已初始化');
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);