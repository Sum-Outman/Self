import React from 'react';
import { useAuth } from '../../contexts/AuthContext';
import PrivateRoute from './PrivateRoute';

interface AdminRouteProps {
  children: React.ReactNode;
}

const AdminRoute: React.FC<AdminRouteProps> = ({ children }) => {
  const { isAdmin, isLoading } = useAuth();

  // 先使用PrivateRoute检查认证状态
  return (
    <PrivateRoute>
      {isLoading ? (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-gray-700 border-t-transparent rounded-full animate-spin mx-auto"></div>
            <p className="mt-4 text-gray-600 dark:text-gray-400">加载中...</p>
          </div>
        </div>
      ) : !isAdmin ? (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-800" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m0 0v2m0-2h2m-2 0H9m3-6a3 3 0 11-6 0 3 3 0 016 0zm4 8a3 3 0 100-6 3 3 0 000 6z" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">无访问权限</h1>
            <p className="text-gray-600 dark:text-gray-400 mb-6">您需要管理员权限才能访问此页面</p>
            <button
              onClick={() => window.location.href = '/dashboard'}
              className="px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-700"
            >
              返回仪表板
            </button>
          </div>
        </div>
      ) : (
        <>{children}</>
      )}
    </PrivateRoute>
  );
};

export default AdminRoute;