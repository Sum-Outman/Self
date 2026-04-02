import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  BarChart3,
  Settings,
  Shield,
  TrendingUp,
  Users,
  Wrench,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Brain,
  Database,
  Cpu,
  Play,
  RefreshCw,
  Search,
  Edit,
  Trash2,
  Eye,
  Plus,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { authService } from '../services/api/auth';
import { User } from '../types/auth';
import toast from 'react-hot-toast';

interface UserStats {
  totalUsers: number;
  activeUsers: number;
  newUsersToday: number;
  adminUsers: number;
}

interface AGIStats {
  modelTrainingJobs: number;
  activeModels: number;
  knowledgeBaseSize: number;
  hardwareConnections: number;
}

interface SystemHealth {
  status: 'healthy' | 'warning' | 'critical';
  backendHealth: boolean;
  trainingHealth: boolean;
  databaseHealth: boolean;
  redisHealth: boolean;
  messageQueueHealth: boolean;
}

const AdminPage: React.FC = () => {
  const { user, isAdmin } = useAuth();
  const [activeTab, setActiveTab] = useState<string>('overview');
  
  // 用户管理状态
  const [users, setUsers] = useState<User[]>([]);
  const [usersLoading, setUsersLoading] = useState(false);
  const [totalUsers, setTotalUsers] = useState(0);
  const [userSearch, setUserSearch] = useState('');
  const [userRoleFilter, setUserRoleFilter] = useState<string>('all');
  const [currentUserPage, setCurrentUserPage] = useState(1);
  const [usersPerPage, setUsersPerPage] = useState(10);
  

  
  // 统计数据状态
  const [userStats, _setUserStats] = useState<UserStats>({
    totalUsers: 0,
    activeUsers: 0,
    newUsersToday: 0,
    adminUsers: 0,
  });
  const [agiStats, _setAgiStats] = useState<AGIStats>({
    modelTrainingJobs: 12,
    activeModels: 3,
    knowledgeBaseSize: 1567,
    hardwareConnections: 2,
  });
  const [systemHealth, _setSystemHealth] = useState<SystemHealth>({
    status: 'healthy',
    backendHealth: true,
    trainingHealth: true,
    databaseHealth: true,
    redisHealth: true,
    messageQueueHealth: true,
  });

  useEffect(() => {
    if (!isAdmin) {
      window.location.href = '/dashboard';
    }
  }, [isAdmin]);

  // 加载用户数据
  const fetchUsers = useCallback(async () => {
    if (!isAdmin) return;
    
    try {
      setUsersLoading(true);
      const params: any = {
        page: currentUserPage,
        per_page: usersPerPage,
      };
      
      if (userSearch) {
        params.search = userSearch;
      }
      
      if (userRoleFilter !== 'all') {
        params.role = userRoleFilter;
      }
      
      const response = await authService.getAllUsers(params);
      
      if (response && response.items) {
        setUsers(response.items);
        setTotalUsers(response.total || response.items.length);
        
        // 更新统计数据
        const activeUsers = response.items.filter((user: User) => user.is_active).length;
        const adminUsers = response.items.filter((user: User) => user.is_admin).length;
        _setUserStats(prev => ({
          ...prev,
          totalUsers: response.total || response.items.length,
          activeUsers,
          adminUsers,
        }));
      }
    } catch (error: any) {
      console.error('加载用户数据失败:', error);
      toast.error(`加载用户数据失败: ${error.message || '未知错误'}`);
    } finally {
      setUsersLoading(false);
    }
  }, [isAdmin, currentUserPage, usersPerPage, userSearch, userRoleFilter]);

  // 用户操作处理函数
  const handleUserAction = async (userId: string, action: string) => {
    try {
      switch (action) {
        case 'delete':
          if (window.confirm('确定要删除此用户吗？此操作不可撤销。')) {
            await authService.deleteUserById(userId);
            toast.success('用户删除成功');
            fetchUsers(); // 刷新用户列表
          }
          break;
          
        case 'toggle-active':
          await authService.toggleUserActive(userId);
          toast.success('用户状态已更新');
          fetchUsers(); // 刷新用户列表
          break;
          
        case 'edit':
          // 实现基本的用户编辑功能
          console.log('编辑用户:', userId);
          try {
            // 查找用户数据
            const userToEdit = users.find(u => u.id === userId);
            if (!userToEdit) {
              toast.error('未找到用户数据');
              break;
            }
            
            // 使用简单提示框进行编辑
            const newEmail = prompt('请输入新的邮箱地址：', userToEdit.email || '');
            if (newEmail === null) {
              toast('编辑已取消');
              break;
            }
            
            const newUsername = prompt('请输入新的用户名：', userToEdit.username || '');
            if (newUsername === null) {
              toast('编辑已取消');
              break;
            }
            
            const newFullName = prompt('请输入新的全名：', userToEdit.full_name || '');
            if (newFullName === null) {
              toast('编辑已取消');
              break;
            }
            
            // 更新用户信息
            await authService.updateUserById(userId, {
              email: newEmail,
              username: newUsername,
              full_name: newFullName,
            });
            
            toast.success('用户信息更新成功');
            fetchUsers(); // 刷新用户列表
          } catch (error: any) {
            console.error('编辑用户失败:', error);
            toast.error(`编辑失败: ${error.message || '未知错误'}`);
          }
          break;
          
        default:
          console.log(`操作 ${action} 用户 ${userId}`);
      }
    } catch (error: any) {
      console.error(`用户操作失败 (${action}):`, error);
      toast.error(`操作失败: ${error.message || '未知错误'}`);
    }
  };



  // 当用户管理标签页激活或搜索条件变化时加载用户数据
  useEffect(() => {
    if (activeTab === 'users') {
      fetchUsers();
    }
  }, [activeTab, currentUserPage, usersPerPage, userSearch, userRoleFilter, fetchUsers]);

  // 搜索处理函数
  const handleUserSearch = useCallback(() => {
    setCurrentUserPage(1); // 搜索时重置到第一页
    fetchUsers();
  }, [fetchUsers]);

  // 分页处理函数
  const handleUserPageChange = (page: number) => {
    setCurrentUserPage(page);
  };

  // 每页显示数量变更处理
  const handleUsersPerPageChange = (perPage: number) => {
    setUsersPerPage(perPage);
    setCurrentUserPage(1); // 变更每页数量时重置到第一页
  };

  if (!isAdmin) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Shield className="w-16 h-16 text-gray-800 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-gray-900 mb-2">无访问权限</h1>
          <p className="text-gray-600">您没有权限访问管理后台</p>
        </div>
      </div>
    );
  }

  const renderOverview = () => (
    <div className="space-y-6">
      {/* 系统健康状态 */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2 text-gray-700" />
          系统健康状态
        </h2>
        <div className="flex items-center mb-4">
          <div className={`w-3 h-3 rounded-full mr-2 ${systemHealth.status === 'healthy' ? 'bg-gray-600' : systemHealth.status === 'warning' ? 'bg-gray-600' : 'bg-gray-800'}`}></div>
          <span className="text-lg font-medium">
            系统状态: {systemHealth.status === 'healthy' ? '健康' : systemHealth.status === 'warning' ? '警告' : '严重'}
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
          <div className="flex items-center">
            {systemHealth.backendHealth ? <CheckCircle className="w-5 h-5 text-gray-600 mr-2" /> : <XCircle className="w-5 h-5 text-gray-800 mr-2" />}
            <span>后端服务</span>
          </div>
          <div className="flex items-center">
            {systemHealth.trainingHealth ? <CheckCircle className="w-5 h-5 text-gray-600 mr-2" /> : <XCircle className="w-5 h-5 text-gray-800 mr-2" />}
            <span>训练服务</span>
          </div>
          <div className="flex items-center">
            {systemHealth.databaseHealth ? <CheckCircle className="w-5 h-5 text-gray-600 mr-2" /> : <XCircle className="w-5 h-5 text-gray-800 mr-2" />}
            <span>数据库</span>
          </div>
          <div className="flex items-center">
            {systemHealth.redisHealth ? <CheckCircle className="w-5 h-5 text-gray-600 mr-2" /> : <XCircle className="w-5 h-5 text-gray-800 mr-2" />}
            <span>Redis缓存</span>
          </div>
          <div className="flex items-center">
            {systemHealth.messageQueueHealth ? <CheckCircle className="w-5 h-5 text-gray-600 mr-2" /> : <XCircle className="w-5 h-5 text-gray-800 mr-2" />}
            <span>消息队列</span>
          </div>
        </div>
      </div>

      {/* 统计数据 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* 用户统计 */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <Users className="w-8 h-8 text-gray-700" />
            <TrendingUp className="w-5 h-5 text-gray-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900">用户统计</h3>
          <div className="mt-4 space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600">总用户数</span>
              <span className="font-semibold">{userStats.totalUsers}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">活跃用户</span>
              <span className="font-semibold">{userStats.activeUsers}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">今日新增</span>
              <span className="font-semibold">{userStats.newUsersToday}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">管理员</span>
              <span className="font-semibold">{userStats.adminUsers}</span>
            </div>
          </div>
        </div>



        {/* AGI系统统计 */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <Settings className="w-8 h-8 text-gray-600" />
            <TrendingUp className="w-5 h-5 text-gray-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900">AGI系统统计</h3>
          <div className="mt-4 space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600">训练任务</span>
              <span className="font-semibold">{agiStats.modelTrainingJobs}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">活跃模型</span>
              <span className="font-semibold">{agiStats.activeModels}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">知识库条目</span>
              <span className="font-semibold">{agiStats.knowledgeBaseSize}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">硬件连接</span>
              <span className="font-semibold">{agiStats.hardwareConnections}</span>
            </div>
          </div>
        </div>

        {/* 系统警告 */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <AlertTriangle className="w-8 h-8 text-gray-600" />
            <span className="text-sm font-medium px-3 py-1 bg-gray-700 text-gray-900 rounded-full">
              2个警告
            </span>
          </div>
          <h3 className="text-lg font-semibold text-gray-900">系统警告</h3>
          <div className="mt-4 space-y-3">
            <div className="p-3 bg-gray-800 rounded border border-gray-600">
              <p className="text-sm text-gray-900">数据库连接数接近限制</p>
            </div>
            <div className="p-3 bg-gray-800 rounded border border-gray-600">
              <p className="text-sm text-gray-900">Redis内存使用率85%</p>
            </div>
            <div className="p-3 bg-gray-700 rounded border border-gray-500">
              <p className="text-sm text-gray-800">所有其他系统正常</p>
            </div>
          </div>
        </div>
      </div>

      {/* 最近用户 */}
      <div className="bg-white rounded-lg shadow">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center">
            <Users className="w-5 h-5 mr-2 text-gray-700" />
            最近用户
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">用户ID</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">用户名</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">邮箱</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">角色</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">创建时间</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">操作</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {users.slice(0, 3).map((user) => (
                <tr key={user.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{user.id}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{user.username}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{user.email}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${user.is_admin ? 'bg-gray-600 text-gray-900' : 'bg-gray-100 text-gray-800'}`}>
                      {user.is_admin ? '管理员' : '普通用户'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {new Date(user.created_at).toLocaleDateString('zh-CN')}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <div className="flex space-x-2">
                      <button
                        onClick={() => handleUserAction(user.id, 'edit')}
                        className="text-gray-800 hover:text-gray-900"
                      >
                        编辑
                      </button>
                      <button
                        onClick={() => handleUserAction(user.id, 'delete')}
                        className="text-gray-900 hover:text-gray-900"
                      >
                        删除
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>


    </div>
  );

  const renderUsers = () => {
    // 计算分页信息
    const totalPages = Math.ceil(totalUsers / usersPerPage);
    const startIndex = (currentUserPage - 1) * usersPerPage + 1;
    const endIndex = Math.min(currentUserPage * usersPerPage, totalUsers);

    return (
      <div className="bg-white rounded-lg shadow">
        <div className="p-6 border-b border-gray-200 flex justify-between items-center">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center">
            <Users className="w-5 h-5 mr-2 text-gray-700" />
            用户管理
            {usersLoading && (
              <span className="ml-3 text-sm text-gray-500">加载中...</span>
            )}
          </h2>
          <div className="flex space-x-3">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <input
                  type="text"
                  placeholder="搜索用户名或邮箱..."
                  className="pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gray-700 w-64"
                  value={userSearch}
                  onChange={(e) => setUserSearch(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleUserSearch()}
                />
                <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
              </div>
              <select
                className="px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gray-700"
                value={userRoleFilter}
                onChange={(e) => setUserRoleFilter(e.target.value)}
              >
                <option value="all">所有角色</option>
                <option value="admin">管理员</option>
                <option value="user">普通用户</option>
              </select>
              <button
                onClick={handleUserSearch}
                className="px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-700 flex items-center"
              >
                <Search className="w-4 h-4 mr-2" />
                搜索
              </button>
            </div>
            <button className="px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-600 flex items-center">
              <Plus className="w-4 h-4 mr-2" />
              添加用户
            </button>
          </div>
        </div>
        <div className="p-6">
          {usersLoading ? (
            <div className="flex justify-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-gray-700"></div>
            </div>
          ) : users.length === 0 ? (
            <div className="text-center py-12">
              <Users className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500 text-lg">暂无用户数据</p>
              <p className="text-gray-400 mt-2">尝试修改搜索条件或添加新用户</p>
            </div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead>
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        用户名
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        邮箱
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        角色
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        状态
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        创建时间
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        操作
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {users.map((user) => (
                      <tr key={user.id}>
                        <td className="px-4 py-3 text-sm font-medium text-gray-900">
                          <div className="flex items-center">
                            <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center mr-2">
                              <span className="text-gray-800 font-medium">
                                {user.username?.charAt(0)?.toUpperCase() || 'U'}
                              </span>
                            </div>
                            <div>
                              <div className="font-medium">{user.username}</div>
                              <div className="text-xs text-gray-500">ID: {user.id}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-600">{user.email}</td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            user.is_admin 
                              ? 'bg-gray-600 text-gray-900' 
                              : 'bg-gray-100 text-gray-800'
                          }`}>
                            {user.is_admin ? '管理员' : '普通用户'}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            user.is_active 
                              ? 'bg-gray-600 text-gray-800' 
                              : 'bg-gray-800 text-gray-900'
                          }`}>
                            {user.is_active ? '活跃' : '禁用'}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-600">
                          {new Date(user.created_at).toLocaleDateString('zh-CN')}
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex space-x-2">
                            <button
                              onClick={() => handleUserAction(user.id, 'edit')}
                              className="p-1 text-gray-800 hover:text-gray-900"
                              title="编辑用户"
                            >
                              <Edit className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => handleUserAction(user.id, 'toggle-active')}
                              className={`p-1 ${user.is_active ? 'text-gray-700 hover:text-gray-900' : 'text-gray-700 hover:text-gray-800'}`}
                              title={user.is_active ? '禁用用户' : '启用用户'}
                            >
                              <Eye className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => handleUserAction(user.id, 'delete')}
                              className="p-1 text-gray-900 hover:text-gray-900"
                              title="删除用户"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-6 flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0">
                <div className="text-sm text-gray-600">
                  显示第 {startIndex} 到 {endIndex} 条，共 {totalUsers} 条记录
                </div>
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-600">每页显示:</span>
                    <select
                      className="px-3 py-1 border border-gray-300 rounded text-sm"
                      value={usersPerPage}
                      onChange={(e) => handleUsersPerPageChange(Number(e.target.value))}
                    >
                      <option value={5}>5</option>
                      <option value={10}>10</option>
                      <option value={25}>25</option>
                      <option value={50}>50</option>
                    </select>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleUserPageChange(currentUserPage - 1)}
                      disabled={currentUserPage <= 1}
                      className="px-3 py-1 border border-gray-300 rounded text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                    >
                      <ChevronLeft className="w-4 h-4 mr-1" />
                      上一页
                    </button>
                    <span className="text-sm text-gray-600">
                      第 {currentUserPage} 页，共 {totalPages} 页
                    </span>
                    <button
                      onClick={() => handleUserPageChange(currentUserPage + 1)}
                      disabled={currentUserPage >= totalPages}
                      className="px-3 py-1 border border-gray-300 rounded text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                    >
                      下一页
                      <ChevronRight className="w-4 h-4 ml-1" />
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    );
  };



  const renderAGI = () => (
    <div className="space-y-6">
      {/* AGI系统概览 */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Settings className="w-5 h-5 mr-2 text-gray-600" />
          AGI系统概览
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
            <div className="flex items-center mb-2">
              <Brain className="w-5 h-5 text-gray-700 mr-2" />
              <h3 className="font-medium text-gray-900">训练系统</h3>
            </div>
            <p className="text-gray-800 text-sm">管理模型训练任务和配置</p>
            <div className="mt-3">
              <button 
                onClick={() => window.location.href = '/training'}
                className="text-sm px-3 py-1 bg-gray-600 text-gray-900 hover:bg-gray-500 rounded-md"
              >
                前往训练管理
              </button>
            </div>
          </div>
          
          <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
            <div className="flex items-center mb-2">
              <Database className="w-5 h-5 text-gray-600 mr-2" />
              <h3 className="font-medium text-gray-800">知识库</h3>
            </div>
            <p className="text-gray-700 text-sm">管理知识库条目和搜索</p>
            <div className="mt-3">
              <button 
                onClick={() => window.location.href = '/knowledge'}
                className="text-sm px-3 py-1 bg-gray-600 text-gray-800 hover:bg-gray-500 rounded-md"
              >
                前往知识库管理
              </button>
            </div>
          </div>
          
          <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
            <div className="flex items-center mb-2">
              <Cpu className="w-5 h-5 text-gray-600 mr-2" />
              <h3 className="font-medium text-gray-900">硬件控制</h3>
            </div>
            <p className="text-gray-700 text-sm">管理硬件设备和传感器</p>
            <div className="mt-3">
              <button 
                onClick={() => window.location.href = '/hardware'}
                className="text-sm px-3 py-1 bg-gray-600 text-gray-800 hover:bg-gray-500 rounded-md"
              >
                前往硬件控制
              </button>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* 训练任务概览 */}
          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-3">训练任务概览</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">进行中任务</span>
                <span className="font-semibold">{agiStats.modelTrainingJobs}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">活跃模型</span>
                <span className="font-semibold">{agiStats.activeModels}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">GPU使用率</span>
                <span className="font-semibold">{agiStats.modelTrainingJobs > 0 ? '高' : '低'}</span>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-gray-200">
              <button 
                onClick={() => window.location.href = '/training'}
                className="w-full px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-700"
              >
                管理训练任务
              </button>
            </div>
          </div>
          
          {/* 知识库概览 */}
          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-3">知识库概览</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">知识条目</span>
                <span className="font-semibold">{agiStats.knowledgeBaseSize}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">最近更新</span>
                <span className="font-semibold">今天</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">存储使用</span>
                <span className="font-semibold">{(agiStats.knowledgeBaseSize * 0.1).toFixed(1)} MB</span>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-gray-200">
              <button 
                onClick={() => window.location.href = '/knowledge'}
                className="w-full px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-600"
              >
                管理知识库
              </button>
            </div>
          </div>
        </div>
      </div>
      
      {/* 系统操作 */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">系统操作</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <button className="px-4 py-3 bg-gray-800 text-white rounded-md hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-700 flex flex-col items-center">
            <Play className="w-5 h-5 mb-1" />
            <span className="text-sm">启动训练</span>
          </button>
          <button className="px-4 py-3 bg-gray-700 text-white rounded-md hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-600 flex flex-col items-center">
            <Settings className="w-5 h-5 mb-1" />
            <span className="text-sm">系统诊断</span>
          </button>
          <button className="px-4 py-3 bg-gray-700 text-white rounded-md hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-600 flex flex-col items-center">
            <RefreshCw className="w-5 h-5 mb-1" />
            <span className="text-sm">刷新缓存</span>
          </button>
          <button className="px-4 py-3 bg-gray-900 text-white rounded-md hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-800 flex flex-col items-center">
            <AlertTriangle className="w-5 h-5 mb-1" />
            <span className="text-sm">紧急停止</span>
          </button>
        </div>
      </div>
    </div>
  );

  const renderSettings = () => (
    <div className="bg-white rounded-lg shadow">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center">
          <Wrench className="w-5 h-5 mr-2 text-gray-500" />
          系统设置
        </h2>
      </div>
      <div className="p-6">
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">系统配置</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  系统名称
                </label>
                <input
                  type="text"
                  defaultValue="Self AGI 系统"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gray-700"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  系统版本
                </label>
                <input
                  type="text"
                  defaultValue="v1.0.0"
                  disabled
                  className="w-full px-3 py-2 border border-gray-300 rounded-md bg-gray-100"
                />
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">API设置</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  API基础URL
                </label>
                <input
                  type="text"
                  defaultValue="http://localhost:8002"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gray-700"
                />
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="enable-api-docs"
                  defaultChecked
                  className="h-4 w-4 text-gray-800 focus:ring-gray-700 border-gray-300 rounded"
                />
                <label htmlFor="enable-api-docs" className="ml-2 block text-sm text-gray-700">
                  启用API文档界面
                </label>
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">安全设置</h3>
            <div className="space-y-4">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="require-https"
                  defaultChecked
                  className="h-4 w-4 text-gray-800 focus:ring-gray-700 border-gray-300 rounded"
                />
                <label htmlFor="require-https" className="ml-2 block text-sm text-gray-700">
                  强制HTTPS连接
                </label>
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="enable-cors"
                  defaultChecked
                  className="h-4 w-4 text-gray-800 focus:ring-gray-700 border-gray-300 rounded"
                />
                <label htmlFor="enable-cors" className="ml-2 block text-sm text-gray-700">
                  启用CORS支持
                </label>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  API请求频率限制（每分钟）
                </label>
                <input
                  type="number"
                  defaultValue="1000"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gray-700"
                />
              </div>
            </div>
          </div>
          
          <div className="pt-6 border-t border-gray-200">
            <div className="flex justify-end space-x-3">
              <button className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50">
                取消
              </button>
              <button className="px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-900">
                保存设置
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 顶部导航 */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <Shield className="w-8 h-8 text-gray-700" />
                <span className="ml-2 text-xl font-bold text-gray-900">Self AGI 管理后台</span>
              </div>
            </div>
            <div className="flex items-center">
              <div className="flex items-center space-x-4">
                <span className="text-gray-700">{user?.username || '管理员'}</span>
                <button
                  onClick={() => (window.location.href = '/dashboard')}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
                >
                  返回仪表板
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 标签页导航 */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('overview')}
              className={`px-3 py-2 font-medium text-sm ${activeTab === 'overview' ? 'border-b-2 border-gray-700 text-gray-800' : 'text-gray-500 hover:text-gray-700'}`}
            >
              <BarChart3 className="w-4 h-4 inline mr-2" />
              概览
            </button>
            <button
              onClick={() => setActiveTab('users')}
              className={`px-3 py-2 font-medium text-sm ${activeTab === 'users' ? 'border-b-2 border-gray-700 text-gray-800' : 'text-gray-500 hover:text-gray-700'}`}
            >
              <Users className="w-4 h-4 inline mr-2" />
              用户管理
            </button>

            <button
              onClick={() => setActiveTab('agi')}
              className={`px-3 py-2 font-medium text-sm ${activeTab === 'agi' ? 'border-b-2 border-gray-700 text-gray-800' : 'text-gray-500 hover:text-gray-700'}`}
            >
              <Settings className="w-4 h-4 inline mr-2" />
              AGI管理
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`px-3 py-2 font-medium text-sm ${activeTab === 'settings' ? 'border-b-2 border-gray-700 text-gray-800' : 'text-gray-500 hover:text-gray-700'}`}
            >
              <Wrench className="w-4 h-4 inline mr-2" />
              系统设置
            </button>
          </nav>
        </div>
      </div>

      {/* 主要内容 */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'users' && renderUsers()}
        {activeTab === 'agi' && renderAGI()}
        {activeTab === 'settings' && renderSettings()}
      </main>

      {/* 底部信息 */}
      <footer className="bg-white border-t border-gray-200 mt-8">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            Self AGI 管理后台 v1.0.0 • © 2026 Self AGI 系统 • 最后更新: 2026-03-07
          </p>
        </div>
      </footer>
    </div>
  );
};

export default AdminPage;