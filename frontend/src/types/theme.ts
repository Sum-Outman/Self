/**
 * 主题类型定义
 */

// 基础主题类型
export type Theme = 'light' | 'dark';

// 主题模式
export enum ThemeMode {
  LIGHT = 'light',
  DARK = 'dark',
  AUTO = 'auto',
  CUSTOM = 'custom',
}

// 颜色方案
export enum ThemeColorScheme {
  SYSTEM = 'system',
  LIGHT = 'light',
  DARK = 'dark',
  BLUE = 'blue',
  GREEN = 'green',
  PURPLE = 'purple',
  ORANGE = 'orange',
}

// 主题配置接口
export interface ThemeConfig {
  // 主题模式
  mode: ThemeMode;
  
  // 颜色方案
  colorScheme: ThemeColorScheme;
  
  // 颜色配置
  primaryColor: string;
  secondaryColor: string;
  backgroundColor: string;
  textColor: string;
  
  // 布局配置
  borderRadius: number;
  spacingUnit: number;
  
  // 动画配置
  transitionDuration: number;
  enableAnimations: boolean;
  enableColorContrast: boolean;
  enableReducedMotion: boolean;
  
  // 自定义配置（可选）
  customColors?: Record<string, string>;
  customSpacing?: Record<string, number>;
  customFonts?: Record<string, string>;
}

// 主题状态接口
export interface ThemeState {
  currentTheme: Theme;
  config: ThemeConfig;
  initialized: boolean;
  loading: boolean;
  error: string | null;
}

// 主题切换事件
export interface ThemeChangeEvent {
  theme: Theme;
  config: ThemeConfig;
  timestamp: number;
  source: 'user' | 'system' | 'auto' | 'config-change';
}

// 主题统计信息
export interface ThemeStatistics {
  totalSwitches: number;
  manualSwitches: number;
  autoSwitches: number;
  lastSwitchTime: number;
  averageSwitchDuration: number;
  preferenceCounts: {
    light: number;
    dark: number;
    auto: number;
    manual: number;
  };
}

// 主题历史记录
export interface ThemeHistoryEntry {
  timestamp: number;
  fromTheme: Theme;
  toTheme: Theme;
  mode: ThemeMode;
  config: ThemeConfig;
}

// CSS变量映射
export interface CSSVariableMap {
  [key: string]: string;
}

// 主题导出数据
export interface ThemeExportData {
  config: ThemeConfig;
  currentTheme: Theme;
  exportTime: string;
  version: string;
}

// 主题预设
export interface ThemePreset {
  id: string;
  name: string;
  description: string;
  config: Partial<ThemeConfig>;
  previewColors: {
    primary: string;
    secondary: string;
    background: string;
    text: string;
  };
}

// 可访问性配置
export interface AccessibilityConfig {
  contrastRatio: number;
  fontSizeMultiplier: number;
  lineHeightMultiplier: number;
  letterSpacingMultiplier: number;
  enableHighContrast: boolean;
  enableDyslexiaFont: boolean;
  enableColorBlindMode: boolean;
}

// 主题管理器选项
export interface ThemeManagerOptions {
  persistToLocalStorage: boolean;
  syncAcrossTabs: boolean;
  enableSystemListeners: boolean;
  defaultMode: ThemeMode;
  defaultColorScheme: ThemeColorScheme;
  fallbackTheme: Theme;
  debugMode: boolean;
}