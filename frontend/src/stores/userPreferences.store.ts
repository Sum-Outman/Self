/**
 * 用户偏好设置store
 * 管理用户个性化设置和偏好
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// 用户偏好类型定义
export interface UserPreferences {
  // 主题偏好
  theme: 'light' | 'dark' | 'auto';
  // 语言偏好
  language: 'zh' | 'en' | 'auto';
  // 布局偏好
  layout: {
    // 侧边栏位置
    sidebarPosition: 'left' | 'right';
    // 侧边栏默认状态
    sidebarDefaultOpen: boolean;
    // 侧边栏宽度
    sidebarWidth: number;
    // 内容区域最大宽度
    contentMaxWidth: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  };
  // 通知偏好
  notifications: {
    // 启用通知
    enabled: boolean;
    // 声音通知
    soundEnabled: boolean;
    // 桌面通知
    desktopNotifications: boolean;
    // 邮件通知
    emailNotifications: boolean;
    // 通知类型
    notificationTypes: {
      trainingComplete: boolean;
      trainingError: boolean;
      systemAlert: boolean;
      newMessage: boolean;
    };
  };
  // 可访问性偏好
  accessibility: {
    // 字体大小
    fontSize: 'small' | 'medium' | 'large' | 'xlarge';
    // 高对比度模式
    highContrast: boolean;
    // 减少动画
    reducedMotion: boolean;
    // 键盘导航增强
    enhancedKeyboardNav: boolean;
  };
  // 数据偏好
  dataPreferences: {
    // 数据保存期限
    dataRetentionDays: number;
    // 自动保存频率
    autoSaveFrequency: number; // 分钟
    // 备份频率
    backupFrequency: 'daily' | 'weekly' | 'monthly' | 'never';
    // 导出格式
    exportFormat: 'json' | 'csv' | 'excel';
  };
  // 隐私偏好
  privacy: {
    // 数据收集
    dataCollection: boolean;
    // 个性化广告
    personalizedAds: boolean;
    // 分析数据共享
    analyticsSharing: boolean;
    // 位置数据
    locationData: boolean;
  };
  // 快捷方式偏好
  shortcuts: Record<string, string>;
  // 最近使用的功能
  recentFeatures: string[];
  // 收藏的功能
  favoriteFeatures: string[];
  // 隐藏的功能
  hiddenFeatures: string[];
  // 自定义设置
  customSettings: Record<string, any>;
}

// 初始状态
const initialState: UserPreferences = {
  theme: 'auto',
  language: 'zh',
  layout: {
    sidebarPosition: 'left',
    sidebarDefaultOpen: true,
    sidebarWidth: 256,
    contentMaxWidth: 'xl',
  },
  notifications: {
    enabled: true,
    soundEnabled: true,
    desktopNotifications: false,
    emailNotifications: false,
    notificationTypes: {
      trainingComplete: true,
      trainingError: true,
      systemAlert: true,
      newMessage: true,
    },
  },
  accessibility: {
    fontSize: 'medium',
    highContrast: false,
    reducedMotion: false,
    enhancedKeyboardNav: false,
  },
  dataPreferences: {
    dataRetentionDays: 30,
    autoSaveFrequency: 5,
    backupFrequency: 'weekly',
    exportFormat: 'json',
  },
  privacy: {
    dataCollection: true,
    personalizedAds: false,
    analyticsSharing: false,
    locationData: false,
  },
  shortcuts: {
    'toggle_sidebar': 'Ctrl+B',
    'toggle_dark_mode': 'Ctrl+D',
    'search': 'Ctrl+K',
    'new_chat': 'Ctrl+N',
    'save': 'Ctrl+S',
    'refresh': 'F5',
  },
  recentFeatures: ['dashboard', 'chat', 'training', 'hardware'],
  favoriteFeatures: ['chat', 'dashboard'],
  hiddenFeatures: [],
  customSettings: {},
};

// 创建用户偏好store
export const useUserPreferencesStore = create<UserPreferences>()(
  persist(() => initialState, {
    name: 'user-preferences',
  })
);

// 用户偏好store钩子函数（便捷方法）
export const useThemePreference = () => {
  const theme = useUserPreferencesStore((state) => state.theme);
  const setTheme = (newTheme: UserPreferences['theme']) => {
    useUserPreferencesStore.setState({ theme: newTheme });
  };
  return { theme, setTheme };
};

export const useLayoutPreferences = () => {
  const layout = useUserPreferencesStore((state) => state.layout);
  const setLayout = (newLayout: Partial<UserPreferences['layout']>) => {
    useUserPreferencesStore.setState((state) => ({
      layout: { ...state.layout, ...newLayout },
    }));
  };
  return { layout, setLayout };
};

export const useNotificationPreferences = () => {
  const notifications = useUserPreferencesStore((state) => state.notifications);
  const setNotifications = (newNotifications: Partial<UserPreferences['notifications']>) => {
    useUserPreferencesStore.setState((state) => ({
      notifications: { ...state.notifications, ...newNotifications },
    }));
  };
  return { notifications, setNotifications };
};

export const useAccessibilityPreferences = () => {
  const accessibility = useUserPreferencesStore((state) => state.accessibility);
  const setAccessibility = (newAccessibility: Partial<UserPreferences['accessibility']>) => {
    useUserPreferencesStore.setState((state) => ({
      accessibility: { ...state.accessibility, ...newAccessibility },
    }));
  };
  return { accessibility, setAccessibility };
};

export const useDataPreferences = () => {
  const dataPreferences = useUserPreferencesStore((state) => state.dataPreferences);
  const setDataPreferences = (newDataPreferences: Partial<UserPreferences['dataPreferences']>) => {
    useUserPreferencesStore.setState((state) => ({
      dataPreferences: { ...state.dataPreferences, ...newDataPreferences },
    }));
  };
  return { dataPreferences, setDataPreferences };
};

export const usePrivacyPreferences = () => {
  const privacy = useUserPreferencesStore((state) => state.privacy);
  const setPrivacy = (newPrivacy: Partial<UserPreferences['privacy']>) => {
    useUserPreferencesStore.setState((state) => ({
      privacy: { ...state.privacy, ...newPrivacy },
    }));
  };
  return { privacy, setPrivacy };
};

export const useShortcuts = () => {
  const shortcuts = useUserPreferencesStore((state) => state.shortcuts);
  const setShortcuts = (newShortcuts: UserPreferences['shortcuts']) => {
    useUserPreferencesStore.setState({ shortcuts: newShortcuts });
  };
  const updateShortcut = (action: string, shortcut: string) => {
    useUserPreferencesStore.setState((state) => ({
      shortcuts: { ...state.shortcuts, [action]: shortcut },
    }));
  };
  return { shortcuts, setShortcuts, updateShortcut };
};

export const useFeaturePreferences = () => {
  const recentFeatures = useUserPreferencesStore((state) => state.recentFeatures);
  const favoriteFeatures = useUserPreferencesStore((state) => state.favoriteFeatures);
  const hiddenFeatures = useUserPreferencesStore((state) => state.hiddenFeatures);
  
  const addRecentFeature = (feature: string) => {
    useUserPreferencesStore.setState((state) => ({
      recentFeatures: [
        feature,
        ...state.recentFeatures.filter((f) => f !== feature),
      ].slice(0, 10), // 保留最近10个
    }));
  };
  
  const toggleFavoriteFeature = (feature: string) => {
    useUserPreferencesStore.setState((state) => {
      const isFavorite = state.favoriteFeatures.includes(feature);
      return {
        favoriteFeatures: isFavorite
          ? state.favoriteFeatures.filter((f) => f !== feature)
          : [...state.favoriteFeatures, feature],
      };
    });
  };
  
  const toggleHiddenFeature = (feature: string) => {
    useUserPreferencesStore.setState((state) => {
      const isHidden = state.hiddenFeatures.includes(feature);
      return {
        hiddenFeatures: isHidden
          ? state.hiddenFeatures.filter((f) => f !== feature)
          : [...state.hiddenFeatures, feature],
      };
    });
  };
  
  return {
    recentFeatures,
    favoriteFeatures,
    hiddenFeatures,
    addRecentFeature,
    toggleFavoriteFeature,
    toggleHiddenFeature,
  };
};

export const useCustomSettings = () => {
  const customSettings = useUserPreferencesStore((state) => state.customSettings);
  const setCustomSettings = (newCustomSettings: UserPreferences['customSettings']) => {
    useUserPreferencesStore.setState({ customSettings: newCustomSettings });
  };
  const updateCustomSetting = (key: string, value: any) => {
    useUserPreferencesStore.setState((state) => ({
      customSettings: { ...state.customSettings, [key]: value },
    }));
  };
  return { customSettings, setCustomSettings, updateCustomSetting };
};

// 重置用户偏好到默认值
export const resetUserPreferences = () => {
  useUserPreferencesStore.setState(initialState);
};

// 导出用户偏好到JSON
export const exportUserPreferences = (): string => {
  const state = useUserPreferencesStore.getState();
  return JSON.stringify(state, null, 2);
};

// 从JSON导入用户偏好
export const importUserPreferences = (json: string): boolean => {
  try {
    const data = JSON.parse(json);
    useUserPreferencesStore.setState(data);
    return true;
  } catch (error) {
    console.error('导入用户偏好失败:', error);
    return false;
  }
};