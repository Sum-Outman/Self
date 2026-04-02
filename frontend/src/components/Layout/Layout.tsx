import React from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useTheme } from '../../contexts/ThemeContext';
import { useMobileSidebar } from '../../stores/ui.store';
import { Sun, Moon, Monitor } from 'lucide-react';
import {
  Home,
  MessageSquare,
  Brain,
  Cpu,
  Database,
  Key,
  Settings,
  LogOut,
  Menu,
  X,
  BarChart, Shield,
  Bot,
  MemoryStick,
} from 'lucide-react';

const Layout: React.FC = () => {
  const { user, logout, isAdmin } = useAuth();
  const navigate = useNavigate();
  const { mobileSidebarOpen, toggleMobileSidebar } = useMobileSidebar();

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
    } catch (error) {
      console.error('注销失败:', error);
    }
  };

  const { theme, toggleTheme } = useTheme();

  const allNavigation = [
    { name: '仪表板', href: '/dashboard', icon: Home },
    { name: '聊天对话', href: '/chat', icon: MessageSquare },
    { name: '模型训练', href: '/training', icon: Brain, adminOnly: true },
    { name: '硬件控制', href: '/hardware', icon: Cpu },
    { name: '机器人管理', href: '/robot-management', icon: Bot },
    { name: '知识库', href: '/knowledge', icon: Database, adminOnly: true },
    { name: '记忆系统', href: '/memory', icon: MemoryStick, adminOnly: true },
    { name: 'API密钥', href: '/api-keys', icon: Key },
    { name: '专业领域能力', href: '/professional-capabilities', icon: BarChart, adminOnly: true },
    { name: '管理员面板', href: '/admin', icon: Shield, adminOnly: true },
    { name: '系统设置', href: '/settings', icon: Settings },
  ];

  const navigation = allNavigation.filter(item => !item.adminOnly || isAdmin);

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* 移动端侧边栏遮罩 */}
      {mobileSidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-gray-600 bg-opacity-75 lg:hidden"
          onClick={toggleMobileSidebar}
        />
      )}

      {/* 侧边栏 */}
      <div
        className={`fixed inset-y-0 left-0 z-30 w-64 transform bg-white dark:bg-gray-800 shadow-lg transition-transform duration-200 ease-in-out lg:relative lg:translate-x-0 lg:shadow-none ${
          mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex flex-col h-full">
          {/* 侧边栏头部 */}
          <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-gray-800 to-gray-700 rounded-lg" />
              <span className="text-xl font-bold text-gray-900 dark:text-white">
                Self AGI
              </span>
            </div>
            <button
              className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 lg:hidden"
              onClick={toggleMobileSidebar}
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* 导航菜单 */}
          <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
            {navigation.map((item) => (
              <NavLink
                key={item.name}
                to={item.href}
                className={({ isActive }) =>
                  `flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    isActive
                      ? 'bg-gray-700 text-gray-900 dark:bg-gray-900/30 dark:text-gray-400'
                      : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                  }`
                }
                end
              >
                <item.icon className="flex-shrink-0 w-5 h-5 mr-3" />
                {item.name}
              </NavLink>
            ))}
          </nav>

          {/* 用户信息 */}
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 bg-gradient-to-r from-gray-700 to-gray-600 rounded-full flex items-center justify-center text-white font-bold">
                  {user?.username?.charAt(0).toUpperCase() || 'U'}
                </div>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {user?.username || '用户'}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {isAdmin ? '管理员' : '普通用户'}
                </p>
              </div>
              <button
                onClick={handleLogout}
                className="ml-auto p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                title="注销"
              >
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 主内容区域 */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* 顶部导航栏 */}
        <header className="flex items-center justify-between h-16 px-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
          <button
            className="p-2 text-gray-500 rounded-md hover:text-gray-700 dark:hover:text-gray-300 lg:hidden"
            onClick={toggleMobileSidebar}
          >
            <Menu className="w-6 h-6" />
          </button>

          <div className="flex-1 px-4">
            <div className="max-w-3xl">
              <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
                Self AGI - 自主通用人工智能系统
              </h1>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <button
              onClick={() => {
                console.log('点击主题切换按钮，当前主题:', theme);
                toggleTheme();
              }}
              className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
              title={
                theme === 'light' ? '当前：浅色模式（点击切换）' :
                theme === 'dark' ? '当前：深色模式（点击切换）' :
                '当前：自动模式（点击切换）'
              }
            >
              {theme === 'light' ? (
                <Sun className="w-5 h-5" />
              ) : theme === 'dark' ? (
                <Moon className="w-5 h-5" />
              ) : (
                <Monitor className="w-5 h-5" />
              )}
            </button>
            <button
              className="px-4 py-2 text-sm font-medium text-white bg-gray-800 rounded-md hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-700 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
              onClick={() => navigate('/chat')}
            >
              开始对话
            </button>
          </div>
        </header>

        {/* 页面内容 */}
        <main className="flex-1 overflow-auto bg-gray-50 dark:bg-gray-900">
          <div className="px-4 py-6 mx-auto max-w-7xl sm:px-6 lg:px-8">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;