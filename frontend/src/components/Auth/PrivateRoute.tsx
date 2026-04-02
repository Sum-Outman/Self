import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

interface PrivateRouteProps {
  children: React.ReactNode;
}

const PrivateRoute: React.FC<PrivateRouteProps> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();
  const [hasToken, setHasToken] = React.useState(false);

  React.useEffect(() => {
    // 检查localStorage中是否有token
    const token = localStorage.getItem('access_token');
    setHasToken(!!token);
  }, []);

  // 只在开发环境输出调试日志
  const isDevelopment = process.env.NODE_ENV !== 'production';
  if (isDevelopment) {
    console.log('PrivateRoute - 认证状态:', { isLoading, isAuthenticated, hasToken });
  }

  if (isLoading) {
    if (isDevelopment) {
      console.log('PrivateRoute - 显示加载中状态');
    }
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-gray-700 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">加载中...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    // 如果有token但未认证，可能是验证失败，显示错误页面而不是重定向
    if (hasToken) {
      if (isDevelopment) {
        console.log('PrivateRoute - 有token但未认证，显示验证错误页面');
      }
      return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.502 0L4.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">认证验证失败</h1>
            <p className="text-gray-600 dark:text-gray-400 mb-4">您的登录凭证可能已过期或无效</p>
            <div className="space-x-4">
              <button
                onClick={() => window.location.href = '/login'}
                className="px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-700"
              >
                重新登录
              </button>
              <button
                onClick={() => window.location.reload()}
                className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500"
              >
                重试
              </button>
            </div>
          </div>
        </div>
      );
    }
    
    if (isDevelopment) {
      console.log('PrivateRoute - 用户未认证，重定向到登录页');
    }
    return <Navigate to="/login" replace />;
  }

  if (isDevelopment) {
    console.log('PrivateRoute - 用户已认证，渲染子组件');
  }
  return <>{children}</>;
};

export default PrivateRoute;