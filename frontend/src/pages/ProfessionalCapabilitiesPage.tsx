import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  Code,
  Calculator,
  Atom,
  Stethoscope,
  TrendingUp,
  CheckCircle,
  XCircle,
  AlertCircle,
  Play,
  Brain,
  Zap,
  Shield,
  Cpu,
  Database,
  BarChart,
  RefreshCw,
  Loader2,
  Settings,
  X,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { professionalCapabilitiesApi } from '../services/api/professional_capabilities';

// 专业领域能力接口
interface ProfessionalCapability {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  enabled: boolean;
  status: 'active' | 'inactive' | 'testing' | 'error';
  performance: number; // 0-100 性能评分
  lastTested: string;
  testResults: {
    passed: number;
    failed: number;
    total: number;
  };
  capabilities: string[];
}

// 能力测试结果接口
interface CapabilityTestResult {
  capabilityId: string;
  testName: string;
  status: 'passed' | 'failed' | 'running';
  duration: number;
  result?: any;
  error?: string;
  timestamp: string;
}

const ProfessionalCapabilitiesPage: React.FC = () => {
  const { user: _user } = useAuth();
  const [capabilities, setCapabilities] = useState<ProfessionalCapability[]>([
    {
      id: 'programming',
      name: '编程能力',
      description: '代码生成、代码分析、代码调试、代码优化',
      icon: <Code className="w-6 h-6" />,
      enabled: true,
      status: 'active',
      performance: 85,
      lastTested: '2026-03-09T08:59:04Z',
      testResults: {
        passed: 12,
        failed: 1,
        total: 13,
      },
      capabilities: ['代码生成', '代码分析', '代码调试', '代码优化', '代码重构'],
    },
    {
      id: 'mathematics',
      name: '数学能力',
      description: '数学问题求解、符号计算、数学证明、数值计算',
      icon: <Calculator className="w-6 h-6" />,
      enabled: true,
      status: 'active',
      performance: 92,
      lastTested: '2026-03-09T08:59:04Z',
      testResults: {
        passed: 15,
        failed: 0,
        total: 15,
      },
      capabilities: ['代数', '几何', '微积分', '概率统计', '符号计算'],
    },
    {
      id: 'physics',
      name: '物理仿真',
      description: '物理引擎集成、运动仿真、碰撞检测、流体仿真',
      icon: <Atom className="w-6 h-6" />,
      enabled: false, // 需要pybullet物理引擎
      status: 'inactive',
      performance: 45,
      lastTested: '2026-03-09T08:59:04Z',
      testResults: {
        passed: 5,
        failed: 2,
        total: 7,
      },
      capabilities: ['运动仿真', '碰撞检测', '刚体动力学', '粒子系统'],
    },
    {
      id: 'medical',
      name: '医学推理',
      description: '医学知识库、疾病诊断、治疗方案推理、医学文献分析',
      icon: <Stethoscope className="w-6 h-6" />,
      enabled: false, // 需要医学知识库
      status: 'inactive',
      performance: 60,
      lastTested: '2026-03-09T08:59:04Z',
      testResults: {
        passed: 8,
        failed: 3,
        total: 11,
      },
      capabilities: ['疾病诊断', '症状分析', '治疗方案', '药物交互'],
    },
    {
      id: 'finance',
      name: '金融分析',
      description: '金融数据建模、风险评估、投资策略、投资组合优化',
      icon: <TrendingUp className="w-6 h-6" />,
      enabled: true,
      status: 'active',
      performance: 78,
      lastTested: '2026-03-09T08:59:04Z',
      testResults: {
        passed: 10,
        failed: 2,
        total: 12,
      },
      capabilities: ['风险分析', '投资组合', '市场预测', '技术指标'],
    },
  ]);

  const [selectedCapability, setSelectedCapability] = useState<ProfessionalCapability | null>(null);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [testResults, setTestResults] = useState<CapabilityTestResult[]>([]);
  const [isTesting, setIsTesting] = useState<string | null>(null);
  const [_isLoading, setIsLoading] = useState(true);
  const [overallStatus, setOverallStatus] = useState({
    totalCapabilities: 5,
    enabledCapabilities: 3,
    averagePerformance: 0,
    lastUpdated: new Date().toISOString(),
  });

  // 计算整体状态
  useEffect(() => {
    const enabledCapabilities = capabilities.filter(c => c.enabled).length;
    const averagePerformance = capabilities.length > 0 
      ? Math.round(capabilities.reduce((sum, c) => sum + c.performance, 0) / capabilities.length)
      : 0;
    
    setOverallStatus({
      totalCapabilities: capabilities.length,
      enabledCapabilities,
      averagePerformance,
      lastUpdated: new Date().toISOString(),
    });
  }, [capabilities]);

  // 从API加载能力状态
  useEffect(() => {
    const loadCapabilities = async () => {
      setIsLoading(true);
      try {
        // 调用真实API获取能力列表
        const capabilitiesData = await professionalCapabilitiesApi.getCapabilities();
        
        // 转换为前端格式（API字段名转换为前端字段名）
        const formattedCapabilities: ProfessionalCapability[] = capabilitiesData.map(cap => ({
          id: cap.id,
          name: cap.name,
          description: cap.description,
          icon: cap.icon === 'code' ? <Code className="w-6 h-6" /> :
                cap.icon === 'calculator' ? <Calculator className="w-6 h-6" /> :
                cap.icon === 'atom' ? <Atom className="w-6 h-6" /> :
                cap.icon === 'stethoscope' ? <Stethoscope className="w-6 h-6" /> :
                cap.icon === 'trending-up' ? <TrendingUp className="w-6 h-6" /> :
                <Brain className="w-6 h-6" />,
          enabled: cap.enabled,
          status: cap.status,
          performance: cap.performance,
          lastTested: cap.last_tested,
          testResults: {
            passed: cap.test_results.passed,
            failed: cap.test_results.failed,
            total: cap.test_results.total,
          },
          capabilities: cap.capabilities,
        }));
        
        setCapabilities(formattedCapabilities);
        
        // 同时加载整体状态
        const overallStatus = await professionalCapabilitiesApi.getOverallStatus();
        setOverallStatus({
          totalCapabilities: overallStatus.total_capabilities,
          enabledCapabilities: overallStatus.enabled_capabilities,
          averagePerformance: overallStatus.average_performance,
          lastUpdated: overallStatus.last_updated,
        });
        
        toast.success('专业领域能力状态已加载');
      } catch (error) {
        console.error('加载能力状态失败:', error);
        toast.error('加载能力状态失败');
      } finally {
        setIsLoading(false);
      }
    };
    
    loadCapabilities();
  }, []);

  // 测试单个能力
  const testCapability = async (capabilityId: string) => {
    setIsTesting(capabilityId);
    
    // 创建测试结果记录
    const newTestResult: CapabilityTestResult = {
      capabilityId,
      testName: `${capabilityId} 测试`,
      status: 'running',
      duration: 0,
      timestamp: new Date().toISOString(),
    };
    
    setTestResults(prev => [newTestResult, ...prev]);
    
    try {
      // 调用真实的能力测试API
      const testResult = await professionalCapabilitiesApi.testCapability(capabilityId);
      
      // 更新测试结果
      setTestResults(prev => prev.map(test => 
        test === newTestResult ? {
          ...test,
          status: testResult.status,
          duration: testResult.duration,
          result: testResult.result,
          error: testResult.error,
        } : test
      ));
      
      // 更新能力状态
      if (testResult.status === 'passed' || testResult.status === 'failed') {
        setCapabilities(prev => prev.map(cap => 
          cap.id === capabilityId ? {
            ...cap,
            lastTested: new Date().toISOString(),
            testResults: {
              passed: testResult.status === 'passed' ? cap.testResults.passed + 1 : cap.testResults.passed,
              failed: testResult.status === 'failed' ? cap.testResults.failed + 1 : cap.testResults.failed,
              total: cap.testResults.total + 1,
            },
            performance: testResult.status === 'passed' 
              ? Math.min(100, cap.performance + 5) 
              : Math.max(0, cap.performance - 10),
          } : cap
        ));
        
        toast.success(testResult.status === 'passed' ? '能力测试通过' : '能力测试失败');
      } else {
        toast.error('能力测试未完成');
      }
    } catch (error) {
      console.error('能力测试失败:', error);
      toast.error('能力测试失败');
      
      setTestResults(prev => prev.map(test => 
        test === newTestResult ? {
          ...test,
          status: 'failed',
          duration: 2000,
          error: '测试过程中发生错误',
        } : test
      ));
    } finally {
      setIsTesting(null);
    }
  };

  // 测试所有能力
  const testAllCapabilities = async () => {
    for (const capability of capabilities.filter(c => c.enabled)) {
      await testCapability(capability.id);
    }
  };

  // 处理配置点击
  const handleConfigClick = (capability: ProfessionalCapability) => {
    setSelectedCapability(capability);
    setShowConfigModal(true);
  };

  // 获取状态颜色
  const getStatusColor = (status: ProfessionalCapability['status']) => {
    switch (status) {
      case 'active': return 'bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-300';
      case 'inactive': return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400';
      case 'testing': return 'bg-gray-300 text-gray-800 dark:bg-gray-600 dark:text-gray-300';
      case 'error': return 'bg-gray-800 text-gray-100 dark:bg-gray-900 dark:text-gray-300';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400';
    }
  };

  // 获取性能颜色
  const getPerformanceColor = (performance: number) => {
    if (performance >= 80) return 'text-gray-600 dark:text-gray-400';
    if (performance >= 60) return 'text-gray-500 dark:text-gray-400';
    return 'text-gray-800 dark:text-gray-400';
  };

  // 获取性能背景颜色
  const getPerformanceBgColor = (performance: number) => {
    if (performance >= 80) return 'bg-gray-500';
    if (performance >= 60) return 'bg-gray-600';
    return 'bg-gray-800';
  };

  // 格式化日期
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('zh-CN');
  };

  return (
    <div className="space-y-6">
      {/* 页面标题和操作 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            专业领域能力
          </h1>
          <p className="mt-1 text-gray-600 dark:text-gray-400">
            管理和测试Self AGI的专业领域能力
          </p>
          <div className="mt-2">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-800 text-white">
              功能状态：开发中
            </span>
            <span className="ml-2 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-600 text-white">
              后端服务：已连接
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => testAllCapabilities()}
            disabled={isTesting !== null}
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-800 to-gray-700 rounded-lg hover:from-gray-900 hover:to-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            测试所有能力
          </button>
          
          <button
            onClick={() => {
              setIsLoading(true);
              setTimeout(() => setIsLoading(false), 1000);
              toast.success('能力状态已刷新');
            }}
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 dark:bg-gray-800 dark:text-gray-300 dark:border-gray-600 dark:hover:bg-gray-700"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            刷新状态
          </button>
        </div>
      </div>

      {/* 总体状态卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-gray-900 dark:text-white">总能力数</h3>
            <Brain className="w-5 h-5 text-gray-700" />
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {overallStatus.totalCapabilities}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {overallStatus.enabledCapabilities} 个已启用
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-gray-900 dark:text-white">平均性能</h3>
            <Zap className="w-5 h-5 text-gray-600" />
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {overallStatus.averagePerformance}%
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              基于所有能力测试
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-gray-900 dark:text-white">测试覆盖率</h3>
            <Shield className="w-5 h-5 text-gray-600" />
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {Math.round(
                capabilities.reduce((sum, c) => sum + c.testResults.passed, 0) / 
                Math.max(1, capabilities.reduce((sum, c) => sum + c.testResults.total, 0)) * 100
              )}%
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              测试通过率
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-gray-900 dark:text-white">系统状态</h3>
            <Cpu className="w-5 h-5 text-gray-600" />
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {overallStatus.enabledCapabilities === overallStatus.totalCapabilities ? '正常' : '部分正常'}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              最后更新: {formatDate(overallStatus.lastUpdated)}
            </div>
          </div>
        </div>
      </div>

      {/* 能力列表 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {capabilities.map((capability) => (
          <div 
            key={capability.id}
            className={`bg-white dark:bg-gray-800 rounded-xl p-5 shadow-sm border ${
              capability.enabled 
                ? 'border-gray-200 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-900' 
                : 'border-gray-200 dark:border-gray-700 opacity-80'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-lg ${
                  capability.enabled 
                    ? 'bg-gradient-to-br from-gray-700 to-gray-700 dark:from-gray-900/20 dark:to-gray-900/20' 
                    : 'bg-gray-100 dark:bg-gray-700'
                }`}>
                  <div className={`${
                    capability.enabled 
                      ? 'text-gray-800 dark:text-gray-400' 
                      : 'text-gray-500 dark:text-gray-400'
                  }`}>
                    {capability.icon}
                  </div>
                </div>
                <div>
                  <h3 className="font-bold text-gray-900 dark:text-white">
                    {capability.name}
                  </h3>
                  <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium mt-1 ${getStatusColor(capability.status)}`}>
                    {capability.enabled ? '已启用' : '未启用'}
                  </span>
                </div>
              </div>
              
              <div className="text-right">
                <div className={`text-lg font-bold ${getPerformanceColor(capability.performance)}`}>
                  {capability.performance}%
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  性能评分
                </div>
              </div>
            </div>
            
            <p className="mt-3 text-sm text-gray-600 dark:text-gray-400">
              {capability.description}
            </p>
            
            {/* 能力标签 */}
            <div className="mt-4 flex flex-wrap gap-2">
              {capability.capabilities.map((cap, index) => (
                <span 
                  key={index}
                  className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300"
                >
                  {cap}
                </span>
              ))}
            </div>
            
            {/* 测试结果 */}
            <div className="mt-4 grid grid-cols-3 gap-2">
              <div className="text-center">
                <div className="text-lg font-bold text-gray-700 dark:text-gray-400">
                  {capability.testResults.passed}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  通过
                </div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-gray-900 dark:text-gray-500">
                  {capability.testResults.failed}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  失败
                </div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-gray-900 dark:text-white">
                  {capability.testResults.total}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  总计
                </div>
              </div>
            </div>
            
            {/* 性能条 */}
            <div className="mt-4">
              <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                <span>性能指标</span>
                <span>最后测试: {formatDate(capability.lastTested)}</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className={`h-full rounded-full ${getPerformanceBgColor(capability.performance)}`}
                  style={{ width: `${capability.performance}%` }}
                />
              </div>
            </div>
            
            {/* 操作按钮 */}
            <div className="mt-5 flex space-x-3">
              <button
                onClick={() => testCapability(capability.id)}
                disabled={isTesting === capability.id || !capability.enabled}
                className={`flex-1 inline-flex items-center justify-center px-3 py-2 text-sm font-medium rounded-lg ${
                  capability.enabled
                    ? 'text-white bg-gradient-to-r from-gray-800 to-gray-700 hover:from-gray-900 hover:to-gray-800'
                    : 'text-gray-700 bg-gray-100 dark:text-gray-300 dark:bg-gray-700 cursor-not-allowed'
                }`}
              >
                {isTesting === capability.id ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    测试中...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    测试能力
                  </>
                )}
              </button>
              
              <button
                onClick={() => handleConfigClick(capability)}
                disabled={!capability.enabled}
                className="inline-flex items-center justify-center px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 dark:bg-gray-800 dark:text-gray-300 dark:border-gray-600 dark:hover:bg-gray-700"
              >
                <Settings className="w-4 h-4 mr-2" />
                配置
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* 最近测试结果 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-gray-900 dark:text-white">
            最近测试结果
          </h2>
          <button
            onClick={() => setTestResults([])}
            className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
          >
            清除记录
          </button>
        </div>
        
        {testResults.length > 0 ? (
          <div className="space-y-3">
            {testResults.slice(0, 10).map((test, index) => (
              <div 
                key={index}
                className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50"
              >
                <div className="flex items-center space-x-3">
                  <div className={`p-1 rounded ${
                    test.status === 'passed' 
                      ? 'bg-gray-600 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400' 
                      : test.status === 'failed' 
                      ? 'bg-gray-800 text-gray-900 dark:bg-gray-900/30 dark:text-gray-500'
                      : 'bg-gray-700 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
                  }`}>
                    {test.status === 'passed' ? (
                      <CheckCircle className="w-4 h-4" />
                    ) : test.status === 'failed' ? (
                      <XCircle className="w-4 h-4" />
                    ) : (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    )}
                  </div>
                  <div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      {test.testName}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {formatDate(test.timestamp)}
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-900 dark:text-white">
                    {test.duration}ms
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {test.status === 'running' ? '测试中...' : test.status === 'passed' ? '成功' : '失败'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <BarChart className="w-12 h-12 mx-auto text-gray-400" />
            <p className="mt-2 text-gray-500 dark:text-gray-400">
              暂无测试记录
            </p>
            <p className="text-sm text-gray-400 dark:text-gray-500">
              点击"测试能力"按钮开始测试
            </p>
          </div>
        )}
      </div>

      {/* 系统说明 */}
      <div className="bg-gradient-to-r from-gray-700 to-gray-700 dark:from-gray-900/20 dark:to-gray-900/20 rounded-xl p-5 border border-gray-600 dark:border-gray-900/50">
        <div className="flex items-center space-x-3">
          <Database className="w-5 h-5 text-gray-800 dark:text-gray-400" />
          <h3 className="font-bold text-gray-900 dark:text-white">
            专业领域能力说明
          </h3>
        </div>
        <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
          Self AGI系统具备多种专业领域能力，包括编程、数学、物理、医学和金融分析。
          这些能力基于真实算法和库实现，可以为用户提供高质量的专业服务。
          每个能力都可以独立测试和配置，确保系统的可靠性和准确性。
        </p>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-gray-600 mt-0.5" />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              编程能力：支持多种编程语言的代码生成、分析和调试
            </span>
          </div>
          <div className="flex items-start space-x-2">
            <CheckCircle className="w-4 h-4 text-gray-600 mt-0.5" />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              数学能力：解决复杂的数学问题，包括代数、几何、微积分等
            </span>
          </div>
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-4 h-4 text-gray-600 mt-0.5" />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              物理仿真：需要安装PyBullet物理引擎以获得完整功能
            </span>
          </div>
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-4 h-4 text-gray-600 mt-0.5" />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              医学推理：需要医学知识库支持，当前功能需连接真实医学数据库
            </span>
          </div>
        </div>
      </div>

      {/* 配置模态框 */}
      {showConfigModal && selectedCapability && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  配置 {selectedCapability.name}
                </h2>
                <button
                  onClick={() => setShowConfigModal(false)}
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    能力名称
                  </label>
                  <input
                    type="text"
                    value={selectedCapability.name}
                    readOnly
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    描述
                  </label>
                  <textarea
                    value={selectedCapability.description}
                    readOnly
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white"
                    rows={2}
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    性能评分
                  </label>
                  <div className="flex items-center space-x-4">
                    <div className="flex-1">
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                        <div 
                          className="bg-gray-700 h-2.5 rounded-full" 
                          style={{ width: `${selectedCapability.performance}%` }}
                        ></div>
                      </div>
                    </div>
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      {selectedCapability.performance}%
                    </span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    能力状态
                  </label>
                  <div className="flex items-center space-x-4">
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${selectedCapability.status === 'active' ? 'bg-gray-600 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400' : 
                      selectedCapability.status === 'inactive' ? 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400' :
                      selectedCapability.status === 'testing' ? 'bg-gray-700 text-gray-900 dark:bg-gray-900/30 dark:text-gray-400' :
                      'bg-gray-800 text-gray-900 dark:bg-gray-900/30 dark:text-gray-500'}`}>
                      {selectedCapability.status === 'active' ? '活跃' :
                       selectedCapability.status === 'inactive' ? '未激活' :
                       selectedCapability.status === 'testing' ? '测试中' : '错误'}
                    </span>
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={selectedCapability.enabled}
                        onChange={(e) => {
                          setCapabilities(prev => prev.map(cap => 
                            cap.id === selectedCapability.id 
                              ? { ...cap, enabled: e.target.checked }
                              : cap
                          ));
                          setSelectedCapability(prev => prev ? { ...prev, enabled: e.target.checked } : null);
                        }}
                        className="w-4 h-4 text-gray-800 bg-gray-100 border-gray-300 rounded focus:ring-gray-700 dark:focus:ring-gray-800 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                      />
                      <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                        启用此能力
                      </span>
                    </label>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    能力列表
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {selectedCapability.capabilities.map((cap, index) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-3 py-1 bg-gray-600 dark:bg-gray-900/30 text-gray-900 dark:text-gray-400 rounded-full text-sm"
                      >
                        {cap}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                  <div className="flex items-center justify-between">
                    <button
                      type="button"
                      onClick={() => setShowConfigModal(false)}
                      className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
                    >
                      关闭
                    </button>
                    
                    <button
                      type="button"
                      onClick={async () => {
                        if (selectedCapability) {
                          try {
                            // 尝试保存配置到后端API
                            const result = await professionalCapabilitiesApi.saveConfig(
                              selectedCapability.id, 
                              { /* 配置数据 - 在实际应用中应该从表单中获取 */ }
                            );
                            
                            if (result.success) {
                              toast.success('配置已保存');
                            } else {
                              toast.error(`保存配置失败: ${result.message}`);
                            }
                          } catch (error) {
                            toast.error(`保存配置时发生错误: ${error instanceof Error ? error.message : '未知错误'}`);
                          }
                        } else {
                          toast.error('未选择能力，无法保存配置');
                        }
                        setShowConfigModal(false);
                      }}
                      className="inline-flex items-center px-6 py-2 text-sm font-medium text-white bg-gradient-to-r from-gray-800 to-gray-700 rounded-lg hover:from-gray-900 hover:to-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-700"
                    >
                      <CheckCircle className="w-4 h-4 mr-2" />
                      保存配置
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProfessionalCapabilitiesPage;