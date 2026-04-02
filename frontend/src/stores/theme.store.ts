/**
 * 主题管理系统（Zustand版本）
 * 提供明暗模式切换、主题配置和持久化功能
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Theme, ThemeMode, ThemeConfig, ThemeColorScheme, ThemeStatistics, ThemeHistoryEntry } from '../types/theme';

// 默认主题配置 - 纯黑白灰
const DEFAULT_THEME_CONFIG: ThemeConfig = {
  mode: ThemeMode.AUTO,
  colorScheme: ThemeColorScheme.SYSTEM,
  primaryColor: '#475569', // gray-600
  secondaryColor: '#64748b', // gray-500
  backgroundColor: '#ffffff', // white
  textColor: '#0f172a', // gray-900
  borderRadius: 8,
  spacingUnit: 4,
  transitionDuration: 200,
  enableAnimations: true,
  enableColorContrast: true,
  enableReducedMotion: false,
};

// 暗色主题配置覆盖 - 纯黑白灰
const DARK_THEME_OVERRIDES: Partial<ThemeConfig> = {
  backgroundColor: '#0f172a', // gray-900
  textColor: '#f1f5f9', // gray-100
  primaryColor: '#94a3b8', // gray-400
  secondaryColor: '#94a3b8', // gray-400
};

// 自定义主题配置 - 纯黑白灰主题
const CUSTOM_THEMES: Record<string, Partial<ThemeConfig>> = {
  'light-gray': {
    primaryColor: '#475569', // gray-600
    secondaryColor: '#64748b', // gray-500
    backgroundColor: '#ffffff', // white
    textColor: '#0f172a', // gray-900
  },
  'dark-gray': {
    primaryColor: '#94a3b8', // gray-400
    secondaryColor: '#94a3b8', // gray-400
    backgroundColor: '#0f172a', // gray-900
    textColor: '#f1f5f9', // gray-100
  },
  'light-minimal': {
    primaryColor: '#64748b', // gray-500
    secondaryColor: '#94a3b8', // gray-400
    backgroundColor: '#f8fafc', // gray-50
    textColor: '#334155', // gray-700
  },
  'dark-minimal': {
    primaryColor: '#cbd5e1', // gray-300
    secondaryColor: '#94a3b8', // gray-400
    backgroundColor: '#1e293b', // gray-800
    textColor: '#e2e8f0', // gray-200
  },
};

interface ThemeState {
  // 状态
  config: ThemeConfig;
  currentTheme: Theme;
  initialized: boolean;
  loading: boolean;
  error: string | null;
  
  // 历史记录
  themeHistory: ThemeHistoryEntry[];
  
  // 统计信息
  stats: ThemeStatistics;
  
  // 可用主题列表
  availableThemes: string[];
  
  // 方法
  initialize: () => Promise<void>;
  switchTheme: (theme: Theme, source?: string) => void;
  switchMode: (mode: ThemeMode) => void;
  applyThemeConfig: (config: ThemeConfig) => void;
  applyCustomTheme: (themeName: string) => void;
  resetToDefault: () => void;
  exportConfig: () => string;
  importConfig: (configString: string) => boolean;
  clearHistory: () => void;
  getThemeState: () => any;
  getThemeStats: () => ThemeStatistics;
}

// 计算当前主题
const calculateCurrentTheme = (
  mode: ThemeMode,
  systemPrefersDark: boolean,
  savedTheme: Theme | null
): Theme => {
  switch (mode) {
    case ThemeMode.LIGHT:
      return 'light';
    case ThemeMode.DARK:
      return 'dark';
    case ThemeMode.AUTO:
      return systemPrefersDark ? 'dark' : 'light';
    case ThemeMode.CUSTOM:
      return savedTheme || 'light';
    default:
      return 'light';
  }
};

// 应用主题到DOM
const applyThemeToDOM = (theme: Theme, config: ThemeConfig) => {
  const root = document.documentElement;
  
  // 移除旧的theme类
  root.classList.remove('light-theme', 'dark-theme');
  
  // 添加新的theme类（保持向后兼容）
  root.classList.add(`${theme}-theme`);
  
  // 更新data-theme属性
  root.setAttribute('data-theme', theme);
  
  // 同时添加/移除'dark'类以匹配ThemeContext
  if (theme === 'dark') {
    root.classList.add('dark');
  } else {
    root.classList.remove('dark');
  }
  
  // 应用CSS变量（保持向后兼容）
  applyCSSVariables(theme === 'dark' ? { ...config, ...DARK_THEME_OVERRIDES } : config);
  
  // 触发自定义事件
  const event = new CustomEvent('theme-change', {
    detail: { theme, config }
  });
  window.dispatchEvent(event);
};

// 应用CSS变量
const applyCSSVariables = (config: ThemeConfig) => {
  const root = document.documentElement;
  
  // 基础颜色变量
  root.style.setProperty('--primary-color', config.primaryColor);
  root.style.setProperty('--secondary-color', config.secondaryColor);
  root.style.setProperty('--background-color', config.backgroundColor);
  root.style.setProperty('--text-color', config.textColor);
  
  // 布局变量
  root.style.setProperty('--border-radius', `${config.borderRadius}px`);
  root.style.setProperty('--spacing-unit', `${config.spacingUnit}px`);
  
  // 过渡变量
  root.style.setProperty('--transition-duration', `${config.transitionDuration}ms`);
  
  // 功能标志
  root.style.setProperty('--enable-animations', config.enableAnimations ? '1' : '0');
  root.style.setProperty('--enable-color-contrast', config.enableColorContrast ? '1' : '0');
  root.style.setProperty('--enable-reduced-motion', config.enableReducedMotion ? '1' : '0');
};

const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      // 初始状态
      config: DEFAULT_THEME_CONFIG,
      currentTheme: 'light',
      initialized: false,
      loading: false,
      error: null,
      themeHistory: [],
      availableThemes: Object.keys(CUSTOM_THEMES),
      stats: {
        totalSwitches: 0,
        manualSwitches: 0,
        autoSwitches: 0,
        lastSwitchTime: 0,
        averageSwitchDuration: 0,
        preferenceCounts: {
          light: 0,
          dark: 0,
          auto: 0,
          manual: 0,
        },
      },

      // 初始化主题系统
      initialize: async () => {
        set({ loading: true });
        
        try {
          // 从localStorage加载保存的配置（由persist中间件处理）
          const state = get();
          
          // 从系统获取当前主题偏好
          const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
          const savedTheme = localStorage.getItem('theme') as Theme | null;
          
          // 计算当前主题
          const currentTheme = calculateCurrentTheme(
            state.config.mode,
            systemPrefersDark,
            savedTheme
          );
          
          // 应用主题到DOM
          applyThemeToDOM(currentTheme, state.config);
          
          // 设置系统主题监听器
          const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
          
          const handleSystemThemeChange = (e: MediaQueryListEvent) => {
            const { config } = get();
            if (config.mode === ThemeMode.AUTO) {
              const newTheme = e.matches ? 'dark' : 'light';
              get().switchTheme(newTheme, 'system');
            }
          };
          
          darkModeMediaQuery.addEventListener('change', handleSystemThemeChange);
          
          // 监听页面可见性变化
          document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
              const { config } = get();
              if (config.mode === ThemeMode.AUTO) {
                const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                const newTheme = systemPrefersDark ? 'dark' : 'light';
                get().switchTheme(newTheme, 'visibility-change');
              }
            }
          });
          
          set({ 
            currentTheme, 
            initialized: true, 
            loading: false, 
            error: null 
          });
          
        } catch (err) {
          console.error('主题系统初始化失败:', err);
          set({ 
            error: err instanceof Error ? err.message : '未知错误',
            loading: false,
            currentTheme: 'light',
          });
          
          // 应用默认亮色主题
          applyThemeToDOM('light', DEFAULT_THEME_CONFIG);
        }
      },

      // 切换主题
      switchTheme: (theme: Theme, source: string = 'user') => {
        const state = get();
        if (state.currentTheme === theme) return;
        
        const startTime = performance.now();
        const oldTheme = state.currentTheme;
        
        try {
          // 更新当前主题
          set({ currentTheme: theme });
          
          // 应用主题到DOM
          applyThemeToDOM(theme, state.config);
          
          // 更新统计信息
          const newStats = { ...state.stats };
          newStats.totalSwitches++;
          newStats.lastSwitchTime = Date.now();
          
          if (source === 'user') {
            newStats.manualSwitches++;
          } else {
            newStats.autoSwitches++;
          }
          
          // 更新偏好统计
          newStats.preferenceCounts[theme]++;
          if (source === 'user') {
            newStats.preferenceCounts.manual++;
          } else {
            newStats.preferenceCounts.auto++;
          }
          
          // 计算平均切换时长
          const switchDuration = performance.now() - startTime;
          const oldAverage = newStats.averageSwitchDuration;
          const switchCount = newStats.totalSwitches;
          
          newStats.averageSwitchDuration = 
            (oldAverage * (switchCount - 1) + switchDuration) / switchCount;
          
          // 记录切换历史
          const newHistoryEntry: ThemeHistoryEntry = {
            timestamp: Date.now(),
            fromTheme: oldTheme,
            toTheme: theme,
            mode: state.config.mode,
            config: { ...state.config },
          };
          
          let newHistory = [...state.themeHistory, newHistoryEntry];
          if (newHistory.length > 100) {
            newHistory = newHistory.slice(-50);
          }
          
          set({ 
            stats: newStats,
            themeHistory: newHistory,
          });
          
          console.log(`主题切换: ${oldTheme} -> ${theme}, 来源: ${source}, 耗时: ${switchDuration.toFixed(1)}ms`);
          
        } catch (err) {
          console.error('切换主题失败:', err);
          set({ 
            error: err instanceof Error ? err.message : '未知错误',
            currentTheme: oldTheme,
          });
          
          // 回滚
          applyThemeToDOM(oldTheme, state.config);
        }
      },

      // 切换主题模式
      switchMode: (mode: ThemeMode) => {
        const state = get();
        const oldMode = state.config.mode;
        
        const newConfig = { ...state.config, mode };
        set({ config: newConfig });
        
        // 如果切换到自动模式，根据系统偏好设置主题
        if (mode === ThemeMode.AUTO) {
          const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
          const newTheme = systemPrefersDark ? 'dark' : 'light';
          get().switchTheme(newTheme, 'auto');
        }
        
        console.log(`主题模式切换: ${oldMode} -> ${mode}`);
      },

      // 应用主题配置
      applyThemeConfig: (config: ThemeConfig) => {
        // 根据模式计算主题
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const theme = calculateCurrentTheme(config.mode, systemPrefersDark, null);
        
        // 更新配置
        set({ config });
        
        // 切换主题
        get().switchTheme(theme, 'config-change');
        
        // 应用CSS变量
        applyCSSVariables(config);
      },

      // 应用自定义主题
      applyCustomTheme: (themeName: string) => {
        const customConfig = CUSTOM_THEMES[themeName];
        if (!customConfig) {
          console.warn(`自定义主题不存在: ${themeName}`);
          return;
        }
        
        const state = get();
        const newConfig = {
          ...state.config,
          ...customConfig,
          mode: ThemeMode.CUSTOM,
        };
        
        get().applyThemeConfig(newConfig);
        console.log(`应用自定义主题: ${themeName}`);
      },

      // 重置为默认主题
      resetToDefault: () => {
        set({ config: DEFAULT_THEME_CONFIG });
        
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const theme = systemPrefersDark ? 'dark' : 'light';
        get().switchTheme(theme, 'reset');
        
        console.log('主题已重置为默认');
      },

      // 导出主题配置
      exportConfig: () => {
        const state = get();
        return JSON.stringify({
          config: state.config,
          currentTheme: state.currentTheme,
          exportTime: new Date().toISOString(),
          version: '1.0.0',
        }, null, 2);
      },

      // 导入主题配置
      importConfig: (configString: string) => {
        try {
          const data = JSON.parse(configString);
          
          if (data.config && data.currentTheme) {
            const newConfig = { ...DEFAULT_THEME_CONFIG, ...data.config };
            
            set({ config: newConfig });
            get().switchTheme(data.currentTheme, 'import');
            
            console.log('主题配置导入成功');
            return true;
          }
          
          return false;
          
        } catch (err) {
          console.error('导入主题配置失败:', err);
          set({ error: '导入失败: 配置文件格式错误' });
          return false;
        }
      },

      // 清除主题历史
      clearHistory: () => {
        set({ themeHistory: [] });
        console.log('主题历史已清除');
      },

      // 获取当前主题状态
      getThemeState: () => {
        const state = get();
        return {
          currentTheme: state.currentTheme,
          config: state.config,
          initialized: state.initialized,
          loading: state.loading,
          error: state.error,
          stats: state.stats,
          themeHistory: state.themeHistory.slice(-10),
          availableThemes: state.availableThemes,
        };
      },

      // 获取主题统计信息
      getThemeStats: () => {
        const state = get();
        return {
          ...state.stats,
          totalHistoryEntries: state.themeHistory.length,
          currentTheme: state.currentTheme,
          currentMode: state.config.mode,
          preferenceCounts: {
            light: state.themeHistory.filter(h => h.toTheme === 'light').length,
            dark: state.themeHistory.filter(h => h.toTheme === 'dark').length,
            auto: state.themeHistory.filter(h => h.mode === ThemeMode.AUTO).length,
            manual: state.themeHistory.filter(h => h.mode !== ThemeMode.AUTO).length,
          },
        };
      },
    }),
    {
      name: 'theme-storage', // localStorage中的key
      partialize: (state) => ({
        config: state.config,
        currentTheme: state.currentTheme,
        themeHistory: state.themeHistory,
        stats: state.stats,
      }),
    }
  )
);

export default useThemeStore;