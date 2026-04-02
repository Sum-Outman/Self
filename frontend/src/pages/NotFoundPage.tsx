import React from 'react';
import { Link } from 'react-router-dom';
import { Home, AlertTriangle } from 'lucide-react';

const NotFoundPage: React.FC = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <div className="max-w-md w-full text-center">
        <div className="flex justify-center mb-6">
          <div className="w-20 h-20 bg-gradient-to-r from-gray-400 to-orange-500 rounded-full flex items-center justify-center">
            <AlertTriangle className="w-10 h-10 text-white" />
          </div>
        </div>
        
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
          404
        </h1>
        <h2 className="text-2xl font-semibold text-gray-700 dark:text-gray-300 mb-4">
          页面未找到
        </h2>
        
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          抱歉，您访问的页面不存在。请检查URL是否正确，或返回首页继续使用Self AGI系统。
        </p>
        
        <div className="space-y-4">
          <Link
            to="/"
            className="inline-flex items-center justify-center px-6 py-3 text-base font-medium text-white bg-gray-800 rounded-lg hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-700 focus:ring-offset-2 dark:focus:ring-offset-gray-800 transition-colors"
          >
            <Home className="w-5 h-5 mr-2" />
            返回首页
          </Link>
          
          <div className="text-sm text-gray-500 dark:text-gray-400">
            <p>如需帮助，请联系技术支持或查看文档。</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NotFoundPage;